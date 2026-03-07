import time
import numpy as np
import pandas as pd


def minmax_normalize(data, max_val, min_val):
    # 计算每列的最小值和范围
    max_val = np.asarray(max_val).flatten()
    min_val = np.asarray(min_val).flatten()
    ranges = max_val - min_val

    # 避免除零错误（处理常数列）
    ranges[ranges == 0] = 1.0  # 将范围0替换为1，避免除零

    # 执行归一化
    normalized_data = (data - min_val) / ranges

    return normalized_data, ranges, min_val


def denormalize_features(normalized_data, feature_min, feature_range):
    feature_min = np.asarray(feature_min).flatten()
    feature_range = np.asarray(feature_range).flatten()
    return normalized_data * feature_range + feature_min


def kmedoids_initial(data, n_clusters):
    n_samples = data.shape[0]
    # 随机选择第一个中心
    medoid_indices = [np.random.randint(n_samples)]

    for _ in range(1, n_clusters):
        # 计算每个点到最近中心的距离
        min_dists = np.full(n_samples, np.inf)
        for idx in medoid_indices:
            dists = np.sqrt(np.sum((data - data[idx]) ** 2, axis=1))
            min_dists = np.minimum(min_dists, dists)

        # 选择距离平方最大的点作为新中心
        probabilities = min_dists ** 2 / np.sum(min_dists ** 2)
        new_idx = np.random.choice(n_samples, p=probabilities)
        medoid_indices.append(new_idx)

    return np.array(medoid_indices)


def kmedoids(data, n_clusters, max_iter=300, random_state=None, verbose=False):

    np.random.seed(random_state)
    n_samples, n_features = data.shape
    # 1. 初始化 - 随机选择 medoids
    # medoid_indices = np.random.choice(n_samples, n_clusters, replace=False)
    # medoids = data[medoid_indices]
    medoid_indices = kmedoids_initial(data, n_clusters)
    medoids = data[medoid_indices]

    # 2. 迭代优化
    for iteration in range(max_iter):
        if verbose:
            start_time = time.time()
            print(f"Iteration {iteration + 1}/{max_iter}: ", end="")
        # 计算所有点到 medoids 的距离
        # 使用广播机制避免显式循环
        distances = np.zeros((n_samples, n_clusters))
        for i in range(n_clusters):
            # 使用欧氏距离
            diff = data - medoids[i]
            distances[:, i] = np.sqrt(np.sum(diff ** 2, axis=1))
        # 分配每个点到最近的 medoid
        labels = np.argmin(distances, axis=1)
        # 更新 medoids
        new_medoid_indices = []
        for cluster_idx in range(n_clusters):
            # 获取当前簇的所有点
            cluster_points = data[labels == cluster_idx]
            if len(cluster_points) == 0:
                # 如果簇为空，随机选择一个点
                new_medoid_indices.append(np.random.choice(n_samples))
                continue
            # 计算簇内所有点之间的距离
            # 使用广播计算点与点之间的距离
            diff = cluster_points[:, np.newaxis, :] - cluster_points
            point_distances = np.sqrt(np.sum(diff ** 2, axis=2))
            # 计算每个点到簇内所有其他点的总距离
            total_distances = np.sum(point_distances, axis=1)
            # 选择总距离最小的点作为新的 medoid
            best_idx = np.argmin(total_distances)
            new_medoid_indices.append(np.where(labels == cluster_idx)[0][best_idx])
        # 更新 medoids
        new_medoid_indices = np.array(new_medoid_indices)
        new_medoids = data[new_medoid_indices]
        # 检查收敛：medoids 是否变化
        if np.array_equal(medoid_indices, new_medoid_indices):
            if verbose:
                print(f"Converged at iteration {iteration + 1}")
            break
        medoid_indices = new_medoid_indices
        medoids = new_medoids
        if verbose:
            elapsed = time.time()
            print(f"Completed in {elapsed:.4f} seconds")

    # 最终分配
    distances = np.zeros((n_samples, n_clusters))
    for i in range(n_clusters):
        diff = data - medoids[i]
        distances[:, i] = np.sqrt(np.sum(diff ** 2, axis=1))
    labels = np.argmin(distances, axis=1)
    # 计算簇内平均距离
    cluster_avg_distances = []
    cluster_sizes = []
    for cluster_idx in range(n_clusters):
        cluster_points = data[labels == cluster_idx]
        cluster_size = len(cluster_points)
        cluster_sizes.append(cluster_size)
        if len(cluster_points) > 0:
            diff = cluster_points - medoids[cluster_idx]
            cluster_dists = np.sqrt(np.sum(diff ** 2, axis=1))
            cluster_avg_distances.append(np.mean(cluster_dists))
        else:
            cluster_avg_distances.append(0)
    # 计算簇权重（簇大小占总样本数的比例）
    cluster_weights = np.array(cluster_sizes) / n_samples

    # 将归一化的聚类中心还原为原始特征值
    result = {
        "medoid_indices": medoid_indices,
        "medoids": medoids,
        "labels": labels,
        "cluster_sizes": cluster_sizes,  # 每个簇的大小
        "cluster_weights": cluster_weights,  # 每个簇的权重
        "cluster_avg_distances": cluster_avg_distances,
        "iterations": iteration + 1
    }

    return result


def typical_scenarios_pattern(total_patterns, pattern_max, pattern_min):
    total_patterns_array = np.array(total_patterns)
    total_patterns_array, feature_range, feature_min = minmax_normalize(total_patterns_array, pattern_max, pattern_min)
    kmedoids_result = kmedoids(
        total_patterns_array,
        n_clusters=8,
        max_iter=400,
        random_state=42,
        verbose=True
    )
    print("\n聚类结果:")
    print(f"聚类中心索引: {kmedoids_result['medoid_indices']}")
    print(f"簇大小: {kmedoids_result['cluster_sizes']}")
    print(f"簇权重: {kmedoids_result['cluster_weights']}")
    # 打印每个聚类中心的特征值
    print("\n聚类中心特征值 (原始尺度):")
    for i, medoid in enumerate(kmedoids_result['medoids']):
        origin_medoid = denormalize_features(medoid, feature_min, feature_range)
        np.set_printoptions(precision=4, suppress=True)
        print(f"  簇 {i}: {origin_medoid}")

    return kmedoids_result
