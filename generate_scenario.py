import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from model_classify import ContrastivePretrainModel, PGDM


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


def compute_entropy(values: np.ndarray) -> float:
    values = np.abs(values).astype(np.float64) + 1e-10
    prob = values / (np.sum(values) + 1e-10)
    return float(-np.sum(prob * np.log(prob + 1e-10)))


def compute_pv_feature_matrix_from_week(week_data: np.ndarray) -> np.ndarray:
    if week_data.shape != (7, 24):
        raise ValueError(f"week_data shape must be (7, 24), got {week_data.shape}")

    total_features = np.zeros(7, dtype=np.float64)
    interday_features = np.zeros(7, dtype=np.float64)
    intraday_diff_mean = np.zeros(7, dtype=np.float64)
    intraday_diff_var = np.zeros(7, dtype=np.float64)

    total_features[0] = np.sum(week_data)
    total_features[1] = np.max(week_data)
    singular_values = np.linalg.svd(week_data, full_matrices=False, compute_uv=False)
    total_features[2] = compute_entropy(singular_values)

    row_entropy = 0.0
    for day in range(7):
        row_entropy += compute_entropy(week_data[day])
    total_features[3] = row_entropy / 7.0

    col_entropy = 0.0
    for hour in range(24):
        col_entropy += compute_entropy(week_data[:, hour])
    total_features[4] = col_entropy / 24.0

    daily_totals = np.sum(week_data, axis=1)
    total_features[5] = np.std(daily_totals, ddof=1) / (np.mean(daily_totals) + 1e-10)

    daily_max = np.max(week_data, axis=1)
    for day in range(7):
        prev_day = (day - 1) % 7
        interday_features[day] = (daily_max[day] - daily_max[prev_day]) / (daily_max[prev_day] + 1e-10)

    sunlight_hours_list = []
    for day in range(7):
        day_data = week_data[day]
        diff = np.diff(day_data)
        intraday_diff_mean[day] = np.mean(diff)
        intraday_diff_var[day] = np.var(diff, ddof=1)
        sunlight_hours_list.append(np.sum(day_data > 0.01))
    total_features[6] = np.mean(sunlight_hours_list)

    return np.stack([total_features, interday_features, intraday_diff_mean, intraday_diff_var], axis=0)


def recompute_pv_features_from_generated(y_gen_real: np.ndarray, normalization_max: float) -> np.ndarray:
    if y_gen_real.ndim != 2 or y_gen_real.shape[1] != 168:
        raise ValueError(f"Generated sequences must be (N, 168), got {y_gen_real.shape}")
    if normalization_max <= 0:
        raise ValueError(f"normalization_max must be positive, got {normalization_max}")

    y_norm = np.clip(y_gen_real / normalization_max, 0.0, 1.0)
    features = []
    for sample in y_norm:
        week_data = sample.reshape(7, 24)
        features.append(compute_pv_feature_matrix_from_week(week_data))
    return np.array(features, dtype=np.float64)


def feature_matrix_to_named_dict(feature_matrix: np.ndarray) -> dict:
    return {
        "scalar_features": {
            TOTAL_FEATURE_NAMES[i]: float(feature_matrix[0, i])
            for i in range(7)
        },
        "vector_features": {
            VECTOR_FEATURE_META[0][0]: [float(v) for v in feature_matrix[1]],
            VECTOR_FEATURE_META[1][0]: [float(v) for v in feature_matrix[2]],
            VECTOR_FEATURE_META[2][0]: [float(v) for v in feature_matrix[3]],
        },
        "raw_matrix_4x7": feature_matrix.tolist(),
    }


def make_date_folder(output_dir: str) -> str:
    date_name = datetime.now().strftime("%Y-%m-%d")
    base_folder = os.path.join(output_dir, date_name)
    if not os.path.exists(base_folder):
        os.makedirs(base_folder, exist_ok=True)
        return base_folder

    idx = 1
    while True:
        candidate = os.path.join(output_dir, f"{date_name}_{idx:02d}")
        if not os.path.exists(candidate):
            os.makedirs(candidate, exist_ok=True)
            return candidate
        idx += 1


class ModelLoader:
    def __init__(self, model_path: str = "pgdm_pv_model.pth", config_path: str = "training_config.json"):
        self.model_path = model_path
        self.config_path = config_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = {}
        self.normalization_max = 1.0
        self.pattern_features_max = np.ones((4, 7), dtype=np.float32)
        self.dataset_type = "PVDataSet"
        self.load_config()
        self.load_model()

    def load_config(self) -> None:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"找不到配置文件: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

    @staticmethod
    def _ensure_4x7(array_like, name: str) -> np.ndarray:
        arr = np.array(array_like, dtype=np.float32)
        if arr.shape == (4, 7):
            return arr
        raise ValueError(f"{name} 需要是形状 (4,7)，当前是 {arr.shape}")

    def load_model(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"找不到模型文件: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        contrastive_model = ContrastivePretrainModel(
            pattern_input_dim=7,
            scenario_seq_len=168,
            embed_dim=512,
        )
        contrastive_pretrain_path = "contrastive_pretrain_model.pth"
        if os.path.exists(contrastive_pretrain_path):
            contrastive_model.load_state_dict(torch.load(contrastive_pretrain_path, map_location=self.device))
            print("对比学习预训练模型加载成功")
        else:
            print("未找到 contrastive_pretrain_model.pth，使用随机初始化的模式编码器")

        self.model = PGDM(
            seq_len=self.config.get("seq_len", 168),
            latent_dim=self.config.get("latent_dim", 128),
            pattern_embed_dim=self.config.get("pattern_embed_dim", 512),
            diffusion_T=self.config.get("diffusion_T", 100),
            pretrained_pattern_encoder=contrastive_model.pattern_encoder,
            freeze_pattern_encoder=True,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.model.eval()

        self.normalization_max = float(
            checkpoint.get("normalization_max", self.config.get("normalization_max", 1.0))
        )

        ckpt_pf_max = checkpoint.get("pattern_features_max", None)
        cfg_pf_max = self.config.get("pattern_features_max", None)
        if ckpt_pf_max is not None:
            self.pattern_features_max = self._ensure_4x7(ckpt_pf_max, "checkpoint.pattern_features_max")
        elif cfg_pf_max is not None:
            self.pattern_features_max = self._ensure_4x7(cfg_pf_max, "config.pattern_features_max")
        else:
            print("未找到 pattern_features_max，默认使用全1矩阵(4x7)")
            self.pattern_features_max = np.ones((4, 7), dtype=np.float32)

        self.dataset_type = checkpoint.get("dataset_type", self.config.get("dataset_type", "PVDataSet"))
        print(f"模型加载完成，设备: {self.device}")
        print(f"数据集类型: {self.dataset_type}")
        print(f"normalization_max: {self.normalization_max}")

    def normalize_target_features(self, x_target) -> torch.Tensor:
        x_target = np.array(x_target, dtype=np.float32)
        if x_target.shape == (4, 7):
            x_target = np.expand_dims(x_target, axis=0)
        if x_target.shape != (1, 4, 7):
            raise ValueError(f"PV目标特征必须是 (1,4,7) 或 (4,7)，当前是 {x_target.shape}")

        x_target_normalized = x_target / (self.pattern_features_max[np.newaxis, :, :] + 1e-10)
        return torch.tensor(x_target_normalized, dtype=torch.float32, device=self.device)


def generate_samples(model_loader: ModelLoader, x_target, num_samples: int = 10):
    x_target_normalized = model_loader.normalize_target_features(x_target)
    x_target_tensor = x_target_normalized.repeat(num_samples, 1, 1)

    with torch.no_grad():
        y_gen_normalized = model_loader.model.generate(x_target_tensor, num_samples=num_samples)

    y_gen_real = y_gen_normalized.cpu().numpy() * model_loader.normalization_max
    return y_gen_real, x_target_normalized.cpu().numpy()


def save_results(
    y_gen_real: np.ndarray,
    x_target_original: np.ndarray,
    x_target_normalized: np.ndarray,
    model_loader: ModelLoader,
    output_dir: str = "generated_pv_sequences",
) -> str:
    save_folder = make_date_folder(output_dir)

    np.save(os.path.join(save_folder, "generated_pv_real.npy"), y_gen_real)

    df = pd.DataFrame(y_gen_real)
    columns = []
    for day in range(1, 8):
        for hour in range(24):
            columns.append(f"Day{day}_Hour{hour:02d}")
    df.columns = columns
    df.to_excel(os.path.join(save_folder, "generated_pv_sequences.xlsx"), index=False)

    for i, sample in enumerate(y_gen_real, start=1):
        plt.figure(figsize=(16, 6))
        hours = list(range(168))
        plt.plot(hours, sample, "b-", linewidth=2, alpha=0.8, marker="o", markersize=3)
        day_ticks = list(range(0, 169, 24))
        plt.xticks(day_ticks, [f"Day {d // 24 + 1}" if d % 24 == 0 else "" for d in day_ticks], fontsize=10)
        for day_start in range(0, 168, 24):
            plt.axvline(x=day_start, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        plt.minorticks_on()
        plt.grid(True, which="major", alpha=0.3)
        plt.grid(True, which="minor", alpha=0.1, linestyle=":")
        plt.xlabel("Time (7-Day Period)")
        plt.ylabel("PV Power (kW)")
        plt.ylim(bottom=0)
        plt.title(f"Generated Weekly PV Profile #{i:02d}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"{i:02d}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    sample_features = recompute_pv_features_from_generated(y_gen_real, model_loader.normalization_max)
    recomputed_json = {
        "dataset_type": model_loader.dataset_type,
        "feature_definition": {
            "shape": "4x7",
            "row_0_scalar_names": TOTAL_FEATURE_NAMES,
            "row_1": "interday_max_change_rate(day1~day7)",
            "row_2": "intraday_diff_mean(day1~day7)",
            "row_3": "intraday_diff_var(day1~day7)",
        },
        "target_features": {
            "input_original_4x7": np.array(x_target_original[0], dtype=float).tolist(),
            "input_normalized_4x7": np.array(x_target_normalized[0], dtype=float).tolist(),
        },
        "generated_samples_recomputed_features": [
            {
                "sample_index": idx + 1,
                **feature_matrix_to_named_dict(sample_features[idx]),
            }
            for idx in range(sample_features.shape[0])
        ],
    }
    with open(os.path.join(save_folder, "target_and_generated_features.json"), "w", encoding="utf-8") as f:
        json.dump(recomputed_json, f, ensure_ascii=False, indent=2)

    generation_info = {
        "dataset_type": model_loader.dataset_type,
        "normalization_max": model_loader.normalization_max,
        "pattern_features_max_4x7": model_loader.pattern_features_max.tolist(),
        "x_target_original_4x7": np.array(x_target_original, dtype=float).tolist(),
        "x_target_normalized_4x7": np.array(x_target_normalized, dtype=float).tolist(),
        "num_samples": int(len(y_gen_real)),
        "saved_png_files": [f"{i:02d}.png" for i in range(1, len(y_gen_real) + 1)],
    }
    with open(os.path.join(save_folder, "generation_info.json"), "w", encoding="utf-8") as f:
        json.dump(generation_info, f, ensure_ascii=False, indent=2)

    print(f"生成结果已保存到: {save_folder}")
    return save_folder


def main():
    print("=== 加载PGDM模型并生成光伏场景 ===")
    try:
        model_loader = ModelLoader(
            model_path="pgdm_pv_model.pth",
            config_path="training_config.json",
        )
        if model_loader.dataset_type != "PVDataSet":
            print(f"当前模型数据集类型为 {model_loader.dataset_type}，本脚本按PV特征定义执行。")

        # 按 data_uti.py 的 PV 特征定义输入: 
        # 行含义:
        # row0: [weekly_total_power, weekly_peak_power, svd_entropy, mean_daily_entropy,
        #        mean_hourly_entropy, daily_total_cv, avg_sunlight_hours]
        # row1: 7天 interday_max_change_rate
        # row2: 7天 intraday_diff_mean
        # row3: 7天 intraday_diff_var
        x_target = np.array(
            [
                [
                    [17.8, 0.66, 0.70, 2.04, 1.89, 0.34, 9.14],
                    [0.08,-0.08,0.5120192170143127,0.04292529448866844,-0.4588415026664734,0.14929579198360443,0.02941175177693367],
                    [0.00, 0.0, 0, 0.00, 0, 0, 0.00],
                    [0.004354909062385559,0.0038848184049129486,0.00843390915542841,0.009192636236548424,0.003215000033378601,0.0034749996848404408,0.0037990000564604998],
                ]
            ],
            dtype=np.float32,
        )

        num_generate = 10
        print("开始生成样本...")
        print(f"目标特征形状: {x_target.shape}")
        y_gen_real, x_target_normalized = generate_samples(
            model_loader=model_loader,
            x_target=x_target,
            num_samples=num_generate,
        )

        save_folder = save_results(
            y_gen_real=y_gen_real,
            x_target_original=x_target,
            x_target_normalized=x_target_normalized,
            model_loader=model_loader,
            output_dir="generated_pv_sequences",
        )

        print("生成完成")
        print(f"样本数: {len(y_gen_real)}")
        print(f"序列形状: {y_gen_real.shape}")
        print(f"功率范围: {y_gen_real.min():.4f} ~ {y_gen_real.max():.4f}")
        print(f"保存位置: {save_folder}")

    except Exception as e:
        print(f"发生错误: {e}")
        print("请确认模型文件、配置文件和预训练编码器文件存在。")


if __name__ == "__main__":
    main()
