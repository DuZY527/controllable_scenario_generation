import sys
print("当前Python解释器路径:", sys.executable)
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import json
from model_classify import ContrastivePretrainModel, train_pgdm, PGDM
from data_uti import excel8760_to_daily24_npy, PVDataSet, WDDataSet, excel8760_to_weekly168_npy
from PatternEncoderPretrain import train_contrastive_model


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 验证PyTorch+CUDA
print("PyTorch版本：", torch.__version__)
print("CUDA是否可用：", torch.cuda.is_available())  # 输出True表示GPU可用
print("GPU数量：", torch.cuda.device_count())  # 输出GPU数量（≥1即正常）
# 1. 加载本地数据集
excel_file_path = \
        r"D:\Controllable_weekly_scenario_generation\controllable_scenario_generation\dataset\PVdata.xlsx"
result = excel8760_to_weekly168_npy(
        excel_path=excel_file_path,
        sheet_name=None,  # 读取所有sheet
        skip_header=0
)
# 加载数据
NPY_FILE_PATH = \
        r"D:\Controllable_weekly_scenario_generation\controllable_scenario_generation\dataset\PVdata.npy"
train_dataset = PVDataSet(npy_file_path=NPY_FILE_PATH, normalize=True, max_power=None)
# train_dataset = WDDataSet(npy_file_path=NPY_FILE_PATH, normalize=True, max_power=None)
pretrain_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        drop_last=True
)
train_loader = DataLoader(
        train_dataset,
        batch_size=10,
        shuffle=True,
        drop_last=True
)
# ---------------------- 3. 初始化模型 ----------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"✅ 使用设备：{device}")
# ---------------------- 第一步：预训练对比学习模型 ----------------------
# print("\n=== 第一步：预训练对比学习模型 ===")
contrastive_model = ContrastivePretrainModel(
        pattern_input_dim=7,
        scenario_seq_len=168,
        embed_dim=512
)
# 训练对比学习模型
contrastive_model = train_contrastive_model(
        model=contrastive_model,
        train_loader=pretrain_loader,
        epochs=3000,
        lr=1e-3,
        device=device,
        save_every=50,
        checkpoint_path="contrastive_pretrain_checkpoint.pth",
        best_model_path="contrastive_pretrain_model_best.pth",
        resume=True
)
# 保存预训练模型
torch.save(contrastive_model.state_dict(), "contrastive_pretrain_model.pth")
print("✅ 对比学习预训练模型保存成功！")
# ---------------------- 第二步：训练PGDM模型 ----------------------
print("\n=== 第二步：训练PGDM模型 ===")
# 使用预训练的模式编码器
pretrained_pattern_encoder = contrastive_model.pattern_encoder
pgdm_model = PGDM(
        seq_len=168,
        latent_dim=128,
        pattern_embed_dim=512,
        diffusion_T=500,
        pretrained_pattern_encoder=pretrained_pattern_encoder,
        freeze_pattern_encoder=True
).to(device)
# 训练PGDM模型
print("\n开始训练PGDM模型...")
pgdm_model = train_pgdm(
        model=pgdm_model,
        train_loader=train_loader,
        epochs=1200,
        lr=1e-3,
        device=device,
        save_every=50,
        checkpoint_path="pgdm_pv_checkpoint.pth",
        best_model_path="pgdm_pv_model_best.pth",
        resume=False
)
# 保存模型（含归一化信息，方便后续生成时反归一化）
save_dict = {
        "model_state_dict": pgdm_model.state_dict(),
        "normalization_max": train_dataset.get_normalization_max()  # 保存归一化最大值
}
torch.save(save_dict, "pgdm_pv_model.pth")
print("✅ 模型保存成功！文件：pgdm_pv_model.pth")
torch.save(save_dict, "pgdm_wd_model.pth")
print("✅ 模型保存成功！文件：pgdm_wd_model.pth")

# 保存训练配置信息
config_info = {
        "seq_len": 168,
        "latent_dim": 128,
        "pattern_embed_dim": 512,
        "diffusion_T": 100,
        "training_epochs": 1000,
        "batch_size": 10,
        "learning_rate": 1e-3,
        "normalization_max": float(train_dataset.get_normalization_max()),
        "pattern_features_max": train_dataset.get_pattern_features_max().tolist(),
        "num_samples": len(train_dataset)
}

with open("training_config.json", "w") as f:
        json.dump(config_info, f, indent=2)

print("✅ 训练配置已保存到：training_config.json")
