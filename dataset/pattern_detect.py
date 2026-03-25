from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_uti import PVDataSet

TOTAL_FEATURE_NAMES = [
    "weekly_total_power",
    "weekly_peak_power",
    "svd_entropy",
    "mean_daily_entropy",
    "mean_hourly_entropy",
    "daily_total_cv",
    "avg_sunlight_hours",
]

VECTOR_FEATURE_META = [
    ("interday_max_change_rate", "interday_max_change_rate_day"),
    ("intraday_diff_mean", "intraday_diff_mean_day"),
    ("intraday_diff_var", "intraday_diff_var_day"),
]


def load_pv_multiscale_features(npy_path: Path) -> np.ndarray:
    dataset = PVDataSet(npy_file_path=str(npy_path), normalize=False, max_power=None)
    features = dataset.feature_matrix.detach().cpu().numpy().astype(np.float64)
    if features.ndim != 3 or features.shape[1:] != (4, 7):
        raise ValueError(f"Expected feature shape (N, 4, 7), but got {features.shape}.")
    return features


def build_distance_matrix(features: np.ndarray, chunk_size: int = 128) -> np.ndarray:
    """
    Mixed distance required by user:
    1) scalar features (features[:, 0, :]) -> sum(abs diff)
    2) vector features (features[:, 1:, :]) -> Euclidean distance on each vector, then sum
    """
    n_samples = features.shape[0]
    scalar_features = features[:, 0, :]  # (N, 7)
    vector_features = features[:, 1:, :]  # (N, 3, 7)

    distance_matrix = np.zeros((n_samples, n_samples), dtype=np.float64)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)

        scalar_chunk = scalar_features[start:end]  # (B, 7)
        scalar_dist = np.sum(
            np.abs(scalar_chunk[:, np.newaxis, :] - scalar_features[np.newaxis, :, :]),
            axis=2,
        )

        vector_dist = np.zeros((end - start, n_samples), dtype=np.float64)
        for block_idx in range(vector_features.shape[1]):
            block_chunk = vector_features[start:end, block_idx, :]  # (B, 7)
            full_block = vector_features[:, block_idx, :]  # (N, 7)
            diff = block_chunk[:, np.newaxis, :] - full_block[np.newaxis, :, :]
            vector_dist += np.linalg.norm(diff, axis=2)

        distance_matrix[start:end] = scalar_dist + vector_dist

    np.fill_diagonal(distance_matrix, 0.0)
    return distance_matrix


def initialize_medoids(distance_matrix: np.ndarray, n_clusters: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n_samples = distance_matrix.shape[0]

    medoids = [int(rng.integers(n_samples))]
    all_indices = np.arange(n_samples)

    while len(medoids) < n_clusters:
        min_dist = np.min(distance_matrix[:, medoids], axis=1)
        min_dist[medoids] = 0.0
        total = float(np.sum(min_dist))

        if total <= 0:
            remaining = np.setdiff1d(all_indices, np.array(medoids), assume_unique=False)
            new_medoid = int(rng.choice(remaining))
        else:
            probs = min_dist / total
            new_medoid = int(rng.choice(n_samples, p=probs))
            while new_medoid in medoids:
                new_medoid = int(rng.choice(n_samples, p=probs))

        medoids.append(new_medoid)

    return np.array(medoids, dtype=int)


def kmedoids_precomputed(
    distance_matrix: np.ndarray,
    n_clusters: int,
    max_iter: int = 200,
    random_state: int = 42,
) -> dict[str, Any]:
    n_samples = distance_matrix.shape[0]
    if n_clusters <= 0 or n_clusters > n_samples:
        raise ValueError("n_clusters must be in [1, n_samples].")

    medoids = initialize_medoids(distance_matrix, n_clusters, random_state)

    for _ in range(max_iter):
        distances_to_medoids = distance_matrix[:, medoids]
        labels = np.argmin(distances_to_medoids, axis=1)

        new_medoids = medoids.copy()
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            if cluster_indices.size == 0:
                continue

            intra_dist = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
            costs = np.sum(intra_dist, axis=1)
            best_local_idx = int(np.argmin(costs))
            new_medoids[cluster_id] = int(cluster_indices[best_local_idx])

        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids

    final_distances = distance_matrix[:, medoids]
    final_labels = np.argmin(final_distances, axis=1)
    cluster_sizes = [int(np.sum(final_labels == cid)) for cid in range(n_clusters)]
    cluster_weights = [size / n_samples for size in cluster_sizes]

    return {
        "medoid_indices": medoids,
        "labels": final_labels,
        "cluster_sizes": cluster_sizes,
        "cluster_weights": cluster_weights,
    }


def build_feature_ranges(features: np.ndarray) -> dict[str, Any]:
    scalar_features = features[:, 0, :]
    vector_features = features[:, 1:, :]

    scalar_min = np.min(scalar_features, axis=0)
    scalar_max = np.max(scalar_features, axis=0)

    ranges: dict[str, Any] = {"scalar_features": {}}
    for i, name in enumerate(TOTAL_FEATURE_NAMES):
        ranges["scalar_features"][name] = {
            "min": float(scalar_min[i]),
            "max": float(scalar_max[i]),
        }

    ranges["vector_features"] = {}
    for block_idx, (block_name, dim_prefix) in enumerate(VECTOR_FEATURE_META):
        block_data = vector_features[:, block_idx, :]  # (N, 7)
        block_min = np.min(block_data, axis=0)
        block_max = np.max(block_data, axis=0)
        ranges["vector_features"][block_name] = {
            "min": [float(v) for v in block_min],
            "max": [float(v) for v in block_max],
            "dimension_names": [f"{dim_prefix}{d + 1}" for d in range(7)],
        }

    return ranges


def build_center_output(features: np.ndarray, clustering_result: dict[str, Any]) -> list[dict[str, Any]]:
    centers: list[dict[str, Any]] = []
    medoid_indices = clustering_result["medoid_indices"]
    cluster_sizes = clustering_result["cluster_sizes"]
    cluster_weights = clustering_result["cluster_weights"]

    for cluster_id, medoid_idx in enumerate(medoid_indices.tolist()):
        center_feature = features[medoid_idx]  # (4, 7)
        scalar_part = center_feature[0]

        center_info: dict[str, Any] = {
            "cluster_id": int(cluster_id),
            "medoid_sample_index": int(medoid_idx),
            "cluster_size": int(cluster_sizes[cluster_id]),
            "cluster_weight": float(cluster_weights[cluster_id]),
            "scalar_features": {
                name: float(scalar_part[idx]) for idx, name in enumerate(TOTAL_FEATURE_NAMES)
            },
            "vector_features": {
                VECTOR_FEATURE_META[0][0]: [float(v) for v in center_feature[1]],
                VECTOR_FEATURE_META[1][0]: [float(v) for v in center_feature[2]],
                VECTOR_FEATURE_META[2][0]: [float(v) for v in center_feature[3]],
            },
        }
        centers.append(center_info)

    return centers


def save_medoids_pattern_excel(features: np.ndarray, clustering_result: dict[str, Any], output_path: Path) -> None:
    """
    保存聚类中心模式到Excel表格
    
    Args:
        features: 特征矩阵，形状为 (N, 4, 7)
        clustering_result: 聚类结果，包含medoid_indices
        output_path: 输出Excel文件路径
    """
    medoid_indices = clustering_result["medoid_indices"]
    n_clusters = len(medoid_indices)
    
    # 准备数据
    data = []
    
    # 第一行：节点编号
    node_row = ["节点编号"] + [f"Node {i}" for i in range(n_clusters)]
    data.append(node_row)
    
    # 第二行：7个标量特征名称
    feature_row = ["特征名称"] + TOTAL_FEATURE_NAMES
    data.append(feature_row)
    
    # 第三~六行：特征矩阵
    feature_types = ["标量特征", "向量特征1", "向量特征2", "向量特征3"]
    for i in range(4):
        row = [feature_types[i]]
        for medoid_idx in medoid_indices:
            # 提取该medoid的第i层特征
            medoid_feature = features[medoid_idx][i]
            row.extend([f"{v:.4f}" for v in medoid_feature])
        data.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存到Excel
    df.to_excel(output_path, index=False, header=False)
    print(f"✅ 聚类中心模式已保存到：{output_path}")


def run(
    npy_path: Path,
    n_clusters: int,
    random_state: int,
    max_iter: int,
    chunk_size: int,
    output_json: Path,
) -> dict[str, Any]:
    features = load_pv_multiscale_features(npy_path)
    distance_matrix = build_distance_matrix(features, chunk_size=chunk_size)

    clustering_result = kmedoids_precomputed(
        distance_matrix=distance_matrix,
        n_clusters=n_clusters,
        max_iter=max_iter,
        random_state=random_state,
    )

    output = {
        "dataset": str(npy_path),
        "n_samples": int(features.shape[0]),
        "n_clusters": int(n_clusters),
        "distance_rule": {
            "scalar": "sum(abs(x_i - y_i))",
            "vector": "sum(||v_i - v_j||_2) for 3 vector blocks",
        },
        "cluster_centers": build_center_output(features, clustering_result),
        "pv_feature_global_range": build_feature_ranges(features),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    # 保存Excel表格
    excel_output = output_json.parent / "mediods_pattern.xlsx"
    save_medoids_pattern_excel(features, clustering_result, excel_output)

    return output


def main() -> None:
    default_npy = PROJECT_ROOT / "dataset" / "PVdata.npy"
    default_output = PROJECT_ROOT / "dataset" / "pv_multiscale_cluster_result.json"

    parser = argparse.ArgumentParser(description="Cluster PV multiscale features with mixed distance.")
    parser.add_argument("--npy-path", type=Path, default=default_npy, help="Path of PV .npy data.")
    parser.add_argument("--n-clusters", type=int, default=12, help="Number of clusters (default: 12).")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--output-json", type=Path, default=default_output)
    args = parser.parse_args()

    result = run(
        npy_path=args.npy_path,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        max_iter=args.max_iter,
        chunk_size=args.chunk_size,
        output_json=args.output_json,
    )

    print(f"Samples: {result['n_samples']}")
    print(f"Clusters: {result['n_clusters']}")
    print(f"Result saved to: {args.output_json}")

    print("\n12 cluster centers (all features):")
    for center in result["cluster_centers"]:
        print(json.dumps(center, ensure_ascii=False, indent=2))

    print("\nPV global feature min/max range:")
    print(json.dumps(result["pv_feature_global_range"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
