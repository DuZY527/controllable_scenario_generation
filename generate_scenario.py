import torch
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from model_classify import ContrastivePretrainModel, PGDM
from data_uti import PVDataSet, WDDataSet


class ModelLoader:
    """加载训练好的模型"""
    def __init__(self, model_path="pgdm_pv_model.pth", config_path="training_config.json"):
        self.model_path = model_path
        self.config_path = config_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_config()
        self.load_model()

    def load_config(self):
        """加载训练配置"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"找不到配置文件：{self.config_path}")

        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

        print("✅ 配置加载成功！")
        print(f"数据集类型：{self.config.get('dataset_type', 'PVDataSet')}")
        print(f"归一化最大值：{self.config.get('normalization_max', 'N/A')}")
        print(f"特征最大值：{self.config.get('pattern_features_max', 'N/A')}")

    def load_model(self):
        """加载训练好的模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"找不到模型文件：{self.model_path}")
        # 加载模型检查点
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        # 重建对比学习模型
        contrastive_model = ContrastivePretrainModel(
            pattern_input_dim=4,
            scenario_seq_len=24,
            embed_dim=512
        )
        # 加载对比学习模型的预训练权重
        contrastive_pretrain_path = "contrastive_pretrain_model.pth"
        if os.path.exists(contrastive_pretrain_path):
            contrastive_model.load_state_dict(torch.load(contrastive_pretrain_path, map_location=self.device))
            print("✅ 对比学习模型加载成功！")
        else:
            print("⚠️  警告：未找到对比学习预训练模型，使用随机初始化")
        # 创建PGDM模型
        self.model = PGDM(
            seq_len=self.config.get("seq_len", 24),
            latent_dim=self.config.get("latent_dim", 64),
            pattern_embed_dim=self.config.get("pattern_embed_dim", 512),
            diffusion_T=self.config.get("diffusion_T", 100),
            pretrained_pattern_encoder=contrastive_model.pattern_encoder,
            freeze_pattern_encoder=True
        ).to(self.device)
        # 加载PGDM模型权重
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        # 获取归一化信息
        self.normalization_max = checkpoint.get("normalization_max", 1.0)
        self.pattern_features_max = checkpoint.get("pattern_features_max", np.ones(4))
        self.dataset_type = checkpoint.get("dataset_type", "PVDataSet")
        print(f"✅ 模型加载成功！设备：{self.device}")
        print(f"✅ 数据集类型：{self.dataset_type}")

    def normalize_target_features(self, x_target):
        """归一化目标特征"""
        if isinstance(x_target, list):
            x_target = np.array(x_target)
        if isinstance(x_target, np.ndarray):
            x_target = torch.tensor(x_target, dtype=torch.float32)
        # 确保形状正确
        if x_target.dim() == 1:
            x_target = x_target.unsqueeze(0)
        # 归一化
        x_target_normalized = x_target / torch.tensor(self.pattern_features_max, dtype=torch.float32)
        return x_target_normalized.to(self.device)


def generate_samples(model_loader, x_target, num_samples=10, output_dir="generated_samples"):
    """生成样本"""
    os.makedirs(output_dir, exist_ok=True)
    # 归一化目标特征
    x_target_normalized = model_loader.normalize_target_features(x_target)
    # 扩展为多个样本
    x_target_tensor = x_target_normalized.repeat(num_samples, 1)
    # 生成序列
    with torch.no_grad():
        y_gen_normalized = model_loader.model.generate(x_target_tensor, num_samples=num_samples)
    # 反归一化到真实值
    y_gen_real = y_gen_normalized.cpu().numpy() * model_loader.normalization_max

    return y_gen_real, x_target_normalized.cpu().numpy()


def save_results(y_gen_real, x_target_original, x_target_normalized,
                 model_loader, output_dir="generated_samples"):
    """保存生成结果"""
    # 创建特征命名的文件夹
    if model_loader.dataset_type == "PVDataSet":
        feature_str = f"avg{x_target_original[0, 0]:.2f}_max{x_target_original[0, 1]:.2f}_sun{x_target_original[0, 2]:.1f}_fluc{x_target_original[0, 3]:.2f}"
        y_label = "PV Power (kW)"
        title_prefix = "PV"
    else:  # WDDataSet
        feature_str = f"avg{x_target_original[0, 0]:.2f}_max{x_target_original[0, 1]:.2f}_min{x_target_original[0, 2]:.2f}_fluc{x_target_original[0, 3]:.2f}"
        y_label = "WD Power (kW)"
        title_prefix = "WD"
    save_folder = os.path.join(output_dir, feature_str)
    os.makedirs(save_folder, exist_ok=True)
    # 保存npy文件
    np.save(os.path.join(save_folder, f"generated_{model_loader.dataset_type.lower()}_real.npy"), y_gen_real)
    # 保存Excel文件
    df = pd.DataFrame(y_gen_real)
    df.columns = [f'Hour_{i + 1:02d}' for i in range(24)]
    excel_path = os.path.join(save_folder, f"generated_{model_loader.dataset_type.lower()}_sequences.xlsx")
    df.to_excel(excel_path, index=False)
    # 绘制曲线图
    for i, sample in enumerate(y_gen_real):
        plt.figure(figsize=(10, 5))
        hours = list(range(24))
        plt.plot(hours, sample, 'b-', linewidth=2, alpha=0.8, marker='o', markersize=4)
        plt.xticks(hours, [f'{h:02d}:00' for h in hours], rotation=45)
        plt.grid(True, alpha=0.3)
        plt.xlabel('Time (Hour of Day)')
        plt.ylabel(y_label)
        plt.ylim(bottom=0)

       # 根据数据集类型设置标题
        if model_loader.dataset_type == "PVDataSet":
            feature_info = f'Avg={x_target_original[0, 0]:.1f}kW, Max={x_target_original[0, 1]:.1f}kW, Sun={x_target_original[0, 2]:.1f}h, Fluc={x_target_original[0, 3]:.1f}kW'
            plt.title(f'Generated Daily PV Profile {i + 1}\n{feature_info}')
        else:
            feature_info = f'Avg={x_target_original[0, 0]:.1f}kW, Max={x_target_original[0, 1]:.1f}kW, Min={x_target_original[0, 2]:.1f}kW, Fluc={x_target_original[0, 3]:.1f}kW'
            plt.title(f'Generated Daily WD Profile {i + 1}\n{feature_info}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"generated_sample_{i + 1:02d}.png"), dpi=150, bbox_inches='tight')
        plt.close()
    # 保存特征信息
    feature_info = {
        'dataset_type': model_loader.dataset_type,
        'x_target_original': x_target_original.tolist(),
        'x_target_normalized': x_target_normalized.tolist(),
        'pattern_features_max': model_loader.pattern_features_max.tolist(),
        'normalization_max': float(model_loader.normalization_max),
        'num_samples': len(y_gen_real),
        'features': ['average_power(kW)', 'max_power(kW)',
                     ('sunshine_duration(hours)' if model_loader.dataset_type == "PVDataSet" else 'min_power(kW)'),
                     'fluctuation_power(kW)']
    }

    with open(os.path.join(save_folder, "generation_info.json"), 'w') as f:
        json.dump(feature_info, f, indent=2)

    print(f"✅ 所有结果已保存到：{save_folder}")
    return save_folder


def main():
    """主函数：加载模型并生成样本"""
    print("=== 加载训练好的PGDM模型 ===")
    try:
        # 加载模型
        model_loader = ModelLoader(
            model_path="pgdm_pv_model.pth",
            config_path="training_config.json"
        )
        # 设置目标特征（根据你的数据集类型选择）
        if model_loader.dataset_type == "PVDataSet":
            # 光伏数据特征：[平均功率, 最大功率, 日照时长, 波动功率]
            x_target = np.array([[0.1, 0.4, 9, 0.98]])  # 示例特征
        else:
            # 风电数据特征：[平均功率, 最大功率, 最小功率, 波动功率]
            x_target = np.array([[0.2, 0.6, 0.05, 1.2]])  # 示例特征
        # 生成样本
        print(f"\n正在生成符合目标特征的样本...")
        print(f"目标特征：{x_target[0]}")
        print(f"数据集类型：{model_loader.dataset_type}")
        num_generate = 10
        y_gen_real, x_target_normalized = generate_samples(
            model_loader=model_loader,
            x_target=x_target,
            num_samples=num_generate,
            output_dir="generated_pv_sequences"
        )
        # 保存结果
        save_folder = save_results(
            y_gen_real=y_gen_real,
            x_target_original=x_target,
            x_target_normalized=x_target_normalized,
            model_loader=model_loader,
            output_dir="generated_pv_sequences"
        )
        # 打印统计信息
        print(f"\n✅ 生成完成！")
        print(f"生成样本数：{len(y_gen_real)}")
        print(f"生成序列形状：{y_gen_real.shape}")
        print(f"功率范围：{y_gen_real.min():.2f} ~ {y_gen_real.max():.2f}")
        print(f"保存位置：{save_folder}")

    except Exception as e:
        print(f"❌ 错误：{e}")
        print("\n请确保：")
        print("1. 已经运行 train.py 训练并保存了模型")
        print("2. pgdm_trained_model.pth 和 training_config.json 文件存在")
        print("3. contrastive_pretrain_model.pth 文件存在（如果需要的话）")


if __name__ == '__main__':
    main()
