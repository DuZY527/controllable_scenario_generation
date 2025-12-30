# This is a sample Python script.
# 打开Python终端（在ddpm-env环境中输入python）

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import math
import PIL
import matplotlib.pyplot as plt
from typing import Tuple, List

from PatternEncoderPretrain import ContrastivePretrainModel, train_contrastive_model

import json
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def excel8760_to_daily24_npy(excel_path, output_npy_path=None, sheet_name=None, skip_header=1):
    """
    将Excel中的8760长度数据分割为1*24的周数据，并保存为npy文件
    参数：
    excel_path: Excel文件路径（str或Path）
    output_npy_path: 输出npy文件路径，默认与Excel同目录同文件名.npy
    sheet_name: 要读取的sheet名称，None表示读取所有sheet
    skip_header: 跳过Excel开头的行数（表头），默认1（跳过第一行表头）
    """
    # 处理输出路径
    if output_npy_path is None:
        excel_path = Path(excel_path)
        output_npy_path = excel_path.parent / f"{excel_path.stem}.npy"
    # 存储所有1*24的数据
    all_weekly_data = []
    # 读取Excel文件
    if sheet_name is None:
        # 读取所有sheet
        excel_file = pd.ExcelFile(excel_path)
        sheet_names = excel_file.sheet_names
    else:
        sheet_names = [sheet_name]
    for sheet in sheet_names:
        print(f"正在处理sheet: {sheet}")
        # 读取当前sheet数据（跳过表头）
        df = pd.read_excel(excel_path, sheet_name=sheet, skiprows=skip_header)
        # 遍历每一列（每列应为8760长度的数据）
        for col_name in df.columns:
            # 获取列数据，去除NaN值，转换为numpy数组
            col_data = df[col_name].dropna().values
            # 验证数据长度是否为8760
            if len(col_data) != 8760:
                print(f"警告：列 '{col_name}' 的长度为 {len(col_data)}，不是8760，已跳过该列")
                continue
            print(f"正在处理列: {col_name} (8760个数据点)")
            # reshape为(-,24)，每个行向量是1*24的日数据
            weekly_data = col_data[:8761].reshape(-1, 24)
            # 验证分割后的数据形状
            if weekly_data.shape != (365, 24):
                print(f"警告：列 '{col_name}' 分割后形状为 {weekly_data.shape}，不是(365,24)，已跳过该列")
                continue
            # 将当前列的日数据添加到总列表
            all_weekly_data.append(weekly_data)
    # 合并所有数据（形状：(总样本数, 24)）
    if all_weekly_data:
        final_data = np.concatenate(all_weekly_data, axis=0)
        # final_data = final_data.reshape(-1, 1, 24)
        # 保存为npy文件
        np.save(output_npy_path, final_data)
        print(f"\n数据处理完成！")
        print(f"原始数据：{len(all_weekly_data)} 列 × 8760 个数据点")
        print(f"分割后：{final_data.shape[0]} 个1×24的日数据样本")
        print(f"保存路径：{output_npy_path}")
        return final_data
    else:
        print("错误：没有找到有效的8760长度数据列！")
        return None


class PVDataSet(Dataset):
    def __init__(self, npy_file_path: str, normalize: bool = True, max_power: float = None):
        # 1. 检查文件是否存在
        if not os.path.exists(npy_file_path):
            raise FileNotFoundError(f"找不到npy文件：{npy_file_path}")
        # 2. 加载npy数据
        self.raw_pv_data = np.load(npy_file_path, allow_pickle=False)  # shape=(N, 24)
        # 3. 数据有效性检查
        self._validate_data()
        # 4. 归一化处理（光伏出力非负，用最大值归一化）
        self.pv_data = self._normalize_data(self.raw_pv_data, normalize, max_power)
        self.pattern_features = self._compute_pattern_features()  # (N, 4)：x_avg, x_max, x_sun, x_fluc
        # 计算原始数据的特征统计量（用于后续归一化真实值输入）
        self.raw_pattern_features = self._compute_raw_pattern_features()
        self.pattern_features_max = self._compute_pattern_features_max()

    def _validate_data(self):
        """检查数据格式和有效性"""
        # 检查维度
        if self.raw_pv_data.ndim != 2:
            raise ValueError(f"npy文件数据必须是2维（样本数×24），当前维度：{self.raw_pv_data.ndim}")
        # 检查序列长度
        if self.raw_pv_data.shape[1] != 24:
            raise ValueError(f"每条数据必须是1×168维，当前序列长度：{self.raw_pv_data.shape[1]}")
        # 检查异常值（NaN/无穷大）
        if np.any(np.isnan(self.raw_pv_data)) or np.any(np.isinf(self.raw_pv_data)):
            raise ValueError("npy文件中包含NaN或无穷大值，请先清理数据")
        # 检查非负性（光伏出力不能为负）
        if np.any(self.raw_pv_data < 0):
            print("⚠️  警告：npy文件中存在负值，已自动裁剪为0")
            self.raw_pv_data = np.clip(self.raw_pv_data, 0, None)

    def _normalize_data(self, data: np.ndarray, normalize: bool, max_power: float) -> torch.Tensor:
        """归一化到 [0,1]"""
        if not normalize:
            print("⚠️  警告：未开启归一化，可能导致模型训练不稳定")
            return torch.tensor(data, dtype=torch.float32)
        # 确定归一化最大值
        if max_power is not None:
            if max_power <= 0:
                raise ValueError("max_power必须为正数")
            norm_max = max_power
        else:
            norm_max = np.max(data)
            if norm_max == 0:
                raise ValueError("所有光伏出力数据均为0，无法归一化")
        # 归一化并转为tensor
        normalized_data = data / norm_max
        # 确保归一化后在 [0,1]（避免浮点误差）
        normalized_data = np.clip(normalized_data, 0.0, 1.0)
        return torch.tensor(normalized_data, dtype=torch.float32)  # (N, 24)

    def _compute_pattern_features(self) -> torch.Tensor:
        n_data = self.pv_data.shape[0]
        features = []
        for i in range(n_data):
            seq = self.pv_data[i]  # (24,)
            # 1. 平均功率 x_avg
            x_avg = torch.mean(seq)
            # 2. 最大功率 x_max
            x_max = torch.max(seq)
            # 3. 日照时长 x_sun（出力>0的时刻数）
            x_sun = torch.sum(seq > 0).float()
            # 4. 波动功率 x_fluc（相邻差值绝对值之和）
            x_fluc = torch.sum(torch.abs(seq - torch.roll(seq, shifts=1)))
            features.append(torch.stack([x_avg, x_max, x_sun, x_fluc]))
        return torch.stack(features)

    def _compute_raw_pattern_features(self) -> np.ndarray:
        """计算原始数据的模式特征（未归一化）"""
        n_data = self.raw_pv_data.shape[0]
        features = []
        for i in range(n_data):
            seq = self.raw_pv_data[i]  # (24,)
            # 1. 平均功率 x_avg
            x_avg = np.mean(seq)
            # 2. 最大功率 x_max
            x_max = np.max(seq)
            # 3. 日照时长 x_sun（出力>0的时刻数）
            x_sun = np.sum(seq > 0)
            # 4. 波动功率 x_fluc（相邻差值绝对值之和）
            x_fluc = np.sum(np.abs(seq - np.roll(seq, shift=1)))
            features.append([x_avg, x_max, x_sun, x_fluc])
        return np.array(features)

    def _compute_pattern_features_max(self) -> np.ndarray:
        """计算模式特征的最大值，用于归一化真实值输入"""
        return np.max(self.raw_pattern_features, axis=0)

    def __len__(self) -> int:
        return self.pv_data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回：(光伏出力序列, 对应的模式特征)"""
        return self.pv_data[idx], self.pattern_features[idx]

    def get_normalization_max(self) -> float:
        """返回归一化时使用的最大功率（用于后续反归一化真实值）"""
        return np.max(self.raw_pv_data) if self.raw_pv_data.size > 0 else 1.0

    def get_pattern_features_max(self) -> np.ndarray:
        """返回模式特征的最大值"""
        return self.pattern_features_max


class WDDataSet(Dataset):
    def __init__(self, npy_file_path: str, normalize: bool = True, max_power: float = None):
        # 1. 检查文件是否存在
        if not os.path.exists(npy_file_path):
            raise FileNotFoundError(f"找不到npy文件：{npy_file_path}")
        # 2. 加载npy数据
        self.raw_wd_data = np.load(npy_file_path, allow_pickle=False)  # shape=(N, 24)
        # 3. 数据有效性检查
        self._validate_data()
        # 4. 归一化处理（光伏出力非负，用最大值归一化）
        self.wd_data = self._normalize_data(self.raw_wd_data, normalize, max_power)
        self.pattern_features = self._compute_pattern_features()  # (N, 4)：x_avg, x_max, x_min, x_fluc
        # 计算原始数据的特征统计量（用于后续归一化真实值输入）
        self.raw_pattern_features = self._compute_raw_pattern_features()
        self.pattern_features_max = self._compute_pattern_features_max()

    def _validate_data(self):
        """检查数据格式和有效性"""
        # 检查维度
        if self.raw_wd_data.ndim != 2:
            raise ValueError(f"npy文件数据必须是2维（样本数×24），当前维度：{self.raw_wd_data.ndim}")
        # 检查序列长度
        if self.raw_wd_data.shape[1] != 24:
            raise ValueError(f"每条数据必须是1×168维，当前序列长度：{self.raw_wd_data.shape[1]}")
        # 检查异常值（NaN/无穷大）
        if np.any(np.isnan(self.raw_wd_data)) or np.any(np.isinf(self.raw_wd_data)):
            raise ValueError("npy文件中包含NaN或无穷大值，请先清理数据")
        # 检查非负性（风机出力不能为负）
        if np.any(self.raw_wd_data < 0):
            print("⚠️  警告：npy文件中存在负值，已自动裁剪为0")
            self.raw_pv_data = np.clip(self.raw_wd_data, 0, None)

    def _normalize_data(self, data: np.ndarray, normalize: bool, max_power: float) -> torch.Tensor:
        """归一化到 [0,1]"""
        if not normalize:
            print("⚠️  警告：未开启归一化，可能导致模型训练不稳定")
            return torch.tensor(data, dtype=torch.float32)
        # 确定归一化最大值
        if max_power is not None:
            if max_power <= 0:
                raise ValueError("max_power必须为正数")
            norm_max = max_power
        else:
            norm_max = np.max(data)
            if norm_max == 0:
                raise ValueError("所有光伏出力数据均为0，无法归一化")
        # 归一化并转为tensor
        normalized_data = data / norm_max
        # 确保归一化后在 [0,1]（避免浮点误差）
        normalized_data = np.clip(normalized_data, 0.0, 1.0)
        return torch.tensor(normalized_data, dtype=torch.float32)  # (N, 24)

    def _compute_pattern_features(self) -> torch.Tensor:
        n_data = self.wd_data.shape[0]
        features = []
        for i in range(n_data):
            seq = self.wd_data[i]  # (24,)
            # 1. 平均功率 x_avg
            x_avg = torch.mean(seq)
            # 2. 最大功率 x_max
            x_max = torch.max(seq)
            # 3. 日照时长 x_sun（出力>0的时刻数）
            x_sun = torch.sum(seq > 0).float()
            # 4. 波动功率 x_fluc（相邻差值绝对值之和）
            x_fluc = torch.sum(torch.abs(seq - torch.roll(seq, shifts=1)))
            features.append(torch.stack([x_avg, x_max, x_sun, x_fluc]))
        return torch.stack(features)

    def _compute_raw_pattern_features(self) -> np.ndarray:
        """计算原始数据的模式特征（未归一化）"""
        n_data = self.raw_wd_data.shape[0]
        features = []
        for i in range(n_data):
            seq = self.raw_wd_data[i]  # (24,)
            # 1. 平均功率 x_avg
            x_avg = np.mean(seq)
            # 2. 最大功率 x_max
            x_max = np.max(seq)
            # 3. 最小功率
            x_min = np.min(seq)
            # 4. 波动功率 x_fluc（相邻差值绝对值之和）
            x_fluc = np.sum(np.abs(seq - np.roll(seq, shift=1)))
            features.append([x_avg, x_max, x_min, x_fluc])
        return np.array(features)

    def _compute_pattern_features_max(self) -> np.ndarray:
        """计算模式特征的最大值，用于归一化真实值输入"""
        return np.max(self.raw_pattern_features, axis=0)

    def __len__(self) -> int:
        return self.wd_data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回：(光伏出力序列, 对应的模式特征)"""
        return self.wd_data[idx], self.pattern_features[idx]

    def get_normalization_max(self) -> float:
        """返回归一化时使用的最大功率（用于后续反归一化真实值）"""
        return np.max(self.raw_wd_data) if self.raw_wd_data.size > 0 else 1.0

    def get_pattern_features_max(self) -> np.ndarray:
        """返回模式特征的最大值"""
        return self.pattern_features_max


class PerceptualVAE(nn.Module):
    def __init__(self, seq_len: int = 24, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        # 1D-CNN编码器配置
        encoder_channels = [1, 32, 64, 128]  # 通道数逐步增加
        decoder_channels = [128, 64, 32, 1]
        self.encoder_dims = encoder_channels
        self.decoder_dims = decoder_channels
        encoder_layers = []
        # 第一层卷积
        encoder_layers.extend([
            nn.Conv1d(encoder_channels[0], encoder_channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(encoder_channels[1])
        ])
        # 中间层卷积（下采样）
        encoder_layers.extend([
            nn.Conv1d(encoder_channels[1], encoder_channels[2], kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(encoder_channels[2])
        ])
        encoder_layers.extend([
            nn.Conv1d(encoder_channels[2], encoder_channels[3], kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(encoder_channels[3])
        ])
        # 全局平均池化
        encoder_layers.append(nn.AdaptiveAvgPool1d(1))
        encoder_layers.append(nn.Flatten())
        # 输出层到潜在空间
        self.encoder_conv = nn.Sequential(*encoder_layers)
        self.encoder_fc = nn.Linear(encoder_channels[3], 2 * latent_dim)  # 输出均值和对数方差
        # 解码器 - 使用转置卷积重建时序数据
        # 初始全连接层，将潜在向量扩展到适合转置卷积的尺寸
        self.decoder_fc = nn.Linear(latent_dim, decoder_channels[0] * 6)  # 6是经过下采样后的长度
        decoder_layers = []
        # 转置卷积层（上采样）
        decoder_layers.extend([
            nn.ConvTranspose1d(decoder_channels[0], decoder_channels[1],
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(decoder_channels[1])
        ])
        decoder_layers.extend([
            nn.ConvTranspose1d(decoder_channels[1], decoder_channels[2],
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(decoder_channels[2])
        ])
        # 最后一层卷积，不改变尺寸，只调整通道数
        decoder_layers.extend([
            nn.Conv1d(decoder_channels[2], decoder_channels[3], kernel_size=3, padding=1),
            nn.Sigmoid()  # 确保输出在[0,1]范围内
        ])
        self.decoder_conv = nn.Sequential(*decoder_layers)
        # 用于感知损失的特征提取器（编码器的中间层）
        self.encoder_feature_indices = [0, 3, 6]  # 编码器特征层索引
        self.decoder_feature_indices = [0, 3]  # 解码器特征层索引
        # 通道调整层，用于处理感知损失中的通道不匹配
        self.channel_adjust = nn.ModuleDict({
            '32to64': nn.Conv1d(32, 64, 1),  # 1x1卷积调整通道数
            '64to32': nn.Conv1d(64, 32, 1),
            '128to64': nn.Conv1d(128, 64, 1),
            '64to128': nn.Conv1d(64, 128, 1)
        })

    def encode(self, x):
        """编码过程：输入序列 -> 潜在空间参数"""
        # 调整输入形状: (B, 24) -> (B, 1, 24)
        x = x.unsqueeze(1)
        # 通过卷积编码器
        conv_features = self.encoder_conv(x)
        # 全连接层得到潜在参数
        mu_log_var = self.encoder_fc(conv_features)
        mu, log_var = torch.chunk(mu_log_var, 2, dim=1)
        return mu, log_var

    def decode(self, z):
        """解码过程：潜在向量 -> 重建序列"""
        # 通过全连接层扩展
        h = self.decoder_fc(z)
        # 调整形状: (B, 128*6) -> (B, 128, 6)
        h = h.view(h.size(0), 128, 6)
        # 通过转置卷积重建序列
        reconstructed = self.decoder_conv(h)  # (B, 1, 24)
        return reconstructed.squeeze(1)  # (B, 24)

    def get_encoder_features(self, x):
        """获取编码器中间层特征，用于感知损失"""
        features = []
        x = x.unsqueeze(1)  # (B, 1, 24)
        # 手动遍历编码器层以收集中间特征
        for i, layer in enumerate(self.encoder_conv):
            x = layer(x)
            if i in self.encoder_feature_indices:
                features.append(x)
            # 在Flatten层之后停止收集特征
            if isinstance(layer, nn.Flatten):
                break
        return features

    def get_decoder_features(self, z):
        """获取解码器中间层特征，用于感知损失"""
        features = []
        # 通过全连接层
        h = self.decoder_fc(z)
        h = h.view(h.size(0), 128, 6)
        # 手动遍历解码器层以收集中间特征
        for i, layer in enumerate(self.decoder_conv):
            h = layer(h)
            if i in [0, 3]:  # 对应转置卷积层的输出
                features.append(h)
        return features

    def adjust_feature_channels(self, feat, target_channels):
        """调整特征图通道数以匹配目标通道数"""
        if feat.size(1) == target_channels:
            return feat

        # 根据通道数差异选择合适的调整层
        if feat.size(1) == 32 and target_channels == 64:
            return self.channel_adjust['32to64'](feat)
        elif feat.size(1) == 64 and target_channels == 32:
            return self.channel_adjust['64to32'](feat)
        elif feat.size(1) == 128 and target_channels == 64:
            return self.channel_adjust['128to64'](feat)
        elif feat.size(1) == 64 and target_channels == 128:
            return self.channel_adjust['64to128'](feat)
        else:
            # 如果无法匹配，使用自适应池化调整到最小公共尺寸
            min_channels = min(feat.size(1), target_channels)
            return feat[:, :min_channels, :]

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """重参数化技巧：从 N(μ, σ²) 采样潜在向量 z"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            y: 光伏出力序列，(B, 24)
        Returns:
            y_recon: 重构序列，(B, 24)
            mu: 潜在向量均值，(B, latent_dim)
            log_var: 潜在向量对数方差，(B, latent_dim)
            perceptual_feats: 感知特征（编码器中间层输出），用于计算感知损失
        """
        # 1. 编码：获取 mu, log_var
        mu, log_var = self.encode(y)
        # 2. 获取编码器中间特征（用于感知损失）
        perceptual_feats = self.get_encoder_features(y)
        # 3. 重参数化采样
        z = self.reparameterize(mu, log_var)  # (B, latent_dim)
        # 4. 解码：重构序列
        y_recon = self.decode(z)  # (B, 24)
        return y_recon, mu, log_var, perceptual_feats

    def compute_vae_loss(
            self, y: torch.Tensor, y_recon: torch.Tensor,
            mu: torch.Tensor, log_var: torch.Tensor,
            perceptual_feats: List[torch.Tensor], recon_feats: List[torch.Tensor],
            lambda_kl: float = 1e-2,  # KL损失权重
            lambda_perceptual: float = 0.1  # 感知损失权重
    ) -> torch.Tensor:
        """
        感知VAE损失：L1重构损失 + 感知损失 + KL散度
        Args:
            recon_feats: 重构序列的感知特征（解码器中间层输出）
        """
        # 1. L1重构损失（比L2更能保留细节波动）
        l1_loss = F.l1_loss(y_recon, y, reduction="mean")
        # 2. 感知损失（编码器/解码器对应层特征的L1损失）
        perceptual_loss = 0.0
        if len(perceptual_feats) > 0 and len(recon_feats) > 0:
            min_layers = min(len(perceptual_feats), len(recon_feats))
            for i in range(min_layers):
                feat = perceptual_feats[i]
                recon_feat = recon_feats[i]
                # 确保特征形状一致
                if feat.shape != recon_feat.shape:
                    # 调整通道数
                    if feat.size(1) != recon_feat.size(1):
                        # 将解码器特征通道数调整到与编码器匹配
                        recon_feat_adjusted = self.adjust_feature_channels(recon_feat, feat.size(1))
                    else:
                        recon_feat_adjusted = recon_feat
                    # 调整空间维度
                    if feat.dim() == 3 and recon_feat_adjusted.dim() == 3:
                        target_length = min(feat.shape[2], recon_feat_adjusted.shape[2])
                        feat_pooled = F.adaptive_avg_pool1d(feat, target_length)
                        recon_feat_pooled = F.adaptive_avg_pool1d(recon_feat_adjusted, target_length)

                        # 最终形状检查
                        if feat_pooled.shape == recon_feat_pooled.shape:
                            perceptual_loss += F.l1_loss(recon_feat_pooled, feat_pooled, reduction="mean")
                        else:
                            # 如果形状仍然不匹配，跳过该层
                            continue
                    else:
                        # 对于非3维特征，跳过
                        continue
                else:
                    perceptual_loss += F.l1_loss(recon_feat, feat, reduction="mean")
            if min_layers > 0:
                perceptual_loss /= min_layers
        # 3. KL散度（正则化潜在空间）
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        # 总损失
        total_loss = l1_loss + lambda_perceptual * perceptual_loss + lambda_kl * kl_loss
        return total_loss


class TimeEmbedding(nn.Module):  # 复制之前的TimeEmbedding类
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, 4 * dim)
        self.linear2 = nn.Linear(4 * dim, 4 * dim)
        # 预计算位置编码参数
        self.register_buffer('inv_freq', None)

    def forward(self, t):
        if t.dim() == 0:
            t = t.unsqueeze(0)
        # 确保t在合理范围内（0到diffusion_T）
        t = torch.clamp(t, 0, 1000)  # 限制t的范围
        batch_size = t.shape[0]
        device = t.device
        # 创建正弦余弦嵌入
        half_dim = self.dim // 2
        # 预计算频率参数（如果还没有的话）
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device).float() / half_dim))
        # 计算位置编码
        pos_enc = t.float().unsqueeze(1) * inv_freq.unsqueeze(0)
        # 分别计算正弦和余弦
        sin_emb = torch.sin(pos_enc)
        cos_emb = torch.cos(pos_enc)
        emb = torch.cat([sin_emb, cos_emb], dim=-1)
        # 如果dim是奇数，填充零
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(batch_size, 1, device=t.device)], dim=1)
        # 通过线性层变换
        emb = self.linear1(emb)
        emb = nn.SiLU()(emb)
        emb = self.linear2(emb)
        return emb


class DenoiseNet(nn.Module):
    """去噪网络：输入(潜在向量z + 时间步嵌入t_emb + 模式嵌入p_emb) → 输出预测噪声"""
    def __init__(self, latent_dim: int = 64, pattern_embed_dim: int = 512, time_emb_base_dim: int = 32):
        super().__init__()
        # 时间步嵌入
        self.time_emb = TimeEmbedding(dim=time_emb_base_dim)
        self.time_emb_out_dim = 4 * time_emb_base_dim
        # 模式嵌入映射
        self.pattern_fc = nn.Linear(pattern_embed_dim, latent_dim)
        # 去噪主干网络（全连接）
        total_input_dim = latent_dim + self.time_emb_out_dim + latent_dim   # z + t_emb + p_emb
        self.denoise_fc = nn.Sequential(
            nn.Linear(total_input_dim, 256),  # z + t_emb + p_emb
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, latent_dim)  # 输出与z同维度的噪声
        )
        # print(f"DenoiseNet配置:")
        # print(f"  - 潜在向量维度: {latent_dim}")
        # print(f"  - 时间嵌入维度: {self.time_emb_out_dim}")
        # print(f"  - 模式嵌入维度: {latent_dim} (映射后)")
        # print(f"  - 总输入维度: {total_input_dim}")

    def forward(self, z: torch.Tensor, t: torch.Tensor, p_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: 加噪后的潜在向量，(B, latent_dim)
            t: 时间步，(B,)
            p_emb: 模式嵌入，(B, 512)
        Returns:
            eps_pred: 预测的噪声，(B, latent_dim)
        """
        # 时间步嵌入
        t_emb = self.time_emb(t)  # (B, time_emb_dim)
        # 模式嵌入映射
        p_emb = self.pattern_fc(p_emb)  # (B, latent_dim)
        # 拼接输入
        x = torch.cat([z, t_emb, p_emb], dim=1)  # (B, 3*latent_dim)
        # 预测噪声
        return self.denoise_fc(x)


class LatentDiffusionModel(nn.Module):
    def __init__(self, latent_dim: int = 64, pattern_embed_dim: int = 512, T: int = 100):
        super().__init__()
        self.T = T  # 扩散步数，参考原文表2
        if T <= 0:
            raise ValueError("T must be positive")
        self.denoise_net = DenoiseNet(latent_dim, pattern_embed_dim)
        # 预计算扩散系数 β_t, α_t, \bar{α}_t（参考原文 Eq.8-10）
        self.register_buffer("beta", torch.linspace(1e-4, 0.05, T))  # β从1e-4增至0.05
        self.register_buffer("alpha", 1.0 - self.beta)
        self.register_buffer("alpha_bar", torch.cumprod(self.alpha, dim=0))  # \bar{α}_t = product(α_1..α_t)

    def forward_diffusion(self, z_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向扩散：z_0 → z_t（加噪过程，参考原文 Eq.10）
        Args:
            z_0: 初始潜在向量（VAE输出），(B, latent_dim)
            t: 时间步，(B,)
        Returns:
            z_t: 加噪后的潜在向量，(B, latent_dim)
            eps: 加入的噪声，(B, latent_dim)
        """
        B, D = z_0.shape
        # 采样噪声
        eps = torch.randn_like(z_0)  # (B, D)
        # 获取 \bar{α}_t（按时间步t索引）
        alpha_bar_t = torch.zeros(B, 1, device=z_0.device)
        for i in range(B):
            idx = min(t[i].long().item(), self.T - 1)
            alpha_bar_t[i] = self.alpha_bar[idx]
        # 计算 z_t = sqrt(α_bar_t) * z_0 + sqrt(1 - α_bar_t) * eps
        z_t = torch.sqrt(alpha_bar_t) * z_0 + torch.sqrt(1 - alpha_bar_t) * eps
        return z_t, eps

    def compute_diffusion_loss(self, z_0: torch.Tensor, p_emb: torch.Tensor) -> torch.Tensor:
        """
        扩散损失（参考原文 Eq.14）：MSE(预测噪声, 真实噪声)
        Args:
            z_0: 初始潜在向量，(B, latent_dim)
            p_emb: 模式嵌入，(B, 512)
        """
        B = z_0.shape[0]
        # 1. 随机采样时间步 t（1~T）
        t = torch.randint(1, self.T, (B,), device=z_0.device)  # (B,)
        # 2. 前向扩散：生成 z_t 和真实噪声 eps
        z_t, eps_true = self.forward_diffusion(z_0, t)
        # 3. 去噪网络预测噪声
        eps_pred = self.denoise_net(z_t, t, p_emb)
        # 4. MSE损失
        return F.mse_loss(eps_pred, eps_true, reduction="mean")

    def reverse_diffusion(self, p_emb: torch.Tensor, latent_dim: int = 64, num_samples: int = 1) -> torch.Tensor:
        """
        反向去噪：从纯噪声 z_T ~ N(0,1) 生成 z_0（参考原文 Eq.15）
        Args:
            p_emb: 目标模式嵌入，(num_samples, 512)
            num_samples: 生成样本数
        Returns:
            z_0: 生成的潜在向量，(num_samples, latent_dim)
        """
        device = p_emb.device
        # 初始化 z_T ~ N(0,1)
        z_t = torch.randn(num_samples, latent_dim, device=device)
        # 逐步去噪（从 T 到 1）
        for t in range(self.T-1, -1, -1):
            # 时间步向量（批量为num_samples）
            t_vec = torch.tensor([t]*num_samples, device=device)  # (num_samples,)
            # 预测噪声
            eps_pred = self.denoise_net(z_t, t_vec, p_emb)
            # 计算系数
            alpha_t = self.alpha[t]  # alpha从0开始索引，t从1开始
            alpha_bar_t = self.alpha_bar[t]
            if t > 0:
                alpha_bar_t_prev = self.alpha_bar[t - 1]
                beta_t = self.beta[t]
                # 去噪更新
                z_t = (1 / torch.sqrt(alpha_t)) * (z_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_pred)
                # 添加噪声
                sigma_t = torch.sqrt(beta_t * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t))
                z_t += sigma_t * torch.randn_like(z_t)
            else:
                # t=0时的最终步骤
                z_t = (1 / torch.sqrt(alpha_t)) * (z_t - (self.beta[t] / torch.sqrt(1 - alpha_bar_t)) * eps_pred)
        return z_t  # z_0


class PGDM(nn.Module):
    def __init__(self, seq_len: int = 24, latent_dim: int = 64, pattern_embed_dim: int = 512, diffusion_T: int = 100,
                 pretrained_pattern_encoder=None, freeze_pattern_encoder: bool = True):
        super().__init__()
        # 1. 模式编码器 + 场景编码器（对比学习用）
        if pretrained_pattern_encoder is None:
            raise ValueError("⚠️  警告：未提供预训练的模式编码器")
        else:
            self.pattern_encoder = pretrained_pattern_encoder
            if freeze_pattern_encoder:
                # 冻结模式编码器参数
                for param in self.pattern_encoder.parameters():
                    param.requires_grad = False
                print("✅ 模式编码器参数已冻结")
        # 2. 感知VAE
        self.perceptual_vae = PerceptualVAE(seq_len=seq_len, latent_dim=latent_dim)
        # 3. 潜在扩散模型
        self.latent_diffusion = LatentDiffusionModel(latent_dim=latent_dim, pattern_embed_dim=pattern_embed_dim,
                                                     T=diffusion_T)

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：计算总损失（对比损失 + VAE损失 + 扩散损失）
        Args:
            y: 光伏出力序列，(B, 24)
            x: 模式特征，(B, 4)
        Returns:
            total_loss: 总损失
            contrast_loss: 对比损失
            vae_loss: VAE损失
            diffusion_loss: 扩散损失
        """
        B = y.shape[0]
        # 1. 对比损失：模式编码器 + 场景编码器
        p_emb = self.pattern_encoder(x)  # (B, 512)
        # 2. 感知VAE损失：重构 + 感知 + KL
        y_recon, mu, log_var, perceptual_feats = self.perceptual_vae(y)
        # 计算重构序列的感知特征（解码器中间层）
        z_recon = self.perceptual_vae.reparameterize(mu, log_var)
        recon_feats = self.perceptual_vae.get_decoder_features(z_recon)
        vae_loss = self.perceptual_vae.compute_vae_loss(
            y, y_recon, mu, log_var, perceptual_feats, recon_feats
        )
        # 3. 扩散损失：基于VAE的潜在向量z_0
        z_0 = self.perceptual_vae.reparameterize(mu, log_var)  # (B, latent_dim)
        diffusion_loss = self.latent_diffusion.compute_diffusion_loss(z_0, p_emb)
        # 总损失（权重可根据实验调整，此处均设为1）
        total_loss = vae_loss + diffusion_loss
        return total_loss, vae_loss, diffusion_loss

    def generate(self, x_target: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        生成光伏出力序列：给定目标模式特征 x_target，生成 num_samples 个序列
        Args:
            x_target: 目标模式特征，(num_samples, 4)
            num_samples: 生成样本数
        Returns:
            y_gen: 生成的光伏出力序列，(num_samples, 24)
        """
        self.eval()
        with torch.no_grad():
            # 1. 目标模式嵌入
            p_emb_target = self.pattern_encoder(x_target)  # (num_samples, 512)
            # 2. 反向扩散生成潜在向量 z_0
            z_0_gen = self.latent_diffusion.reverse_diffusion(p_emb_target, latent_dim=self.perceptual_vae.latent_dim,
                                                              num_samples=num_samples)
            # 3. VAE解码生成光伏序列
            y_gen = self.perceptual_vae.decode(z_0_gen)  # (num_samples, 24)
            # 裁剪到 [0,1]（光伏出力非负）
            y_gen = torch.clamp(y_gen, 0.0, 1.0)
        self.train()
        return y_gen


# ---------------------- 训练函数 ----------------------
def train_pgdm(
        model: PGDM, train_loader: DataLoader,
        epochs: int = 800, lr: float = 1e-3,
        device: torch.device = torch.device("cpu")
):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss_epoch = 0.0
        total_vae_loss_epoch = 0.0
        total_diffusion_loss_epoch = 0.0
        for batch_idx, (y_batch, x_batch) in enumerate(train_loader):
            # 数据移至设备
            y_batch = y_batch.to(device)  # (B, 24)
            x_batch = x_batch.to(device)  # (B, 4)
            # 前向传播计算损失
            total_loss, vae_loss, diffusion_loss = model(y_batch, x_batch)
            # 反向传播优化
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # 累加损失
            total_loss_epoch += total_loss.item() * y_batch.shape[0]
            total_vae_loss_epoch += vae_loss.item() * y_batch.shape[0]
            total_diffusion_loss_epoch += diffusion_loss.item() * y_batch.shape[0]
        # 计算epoch平均损失
        avg_total_loss = total_loss_epoch / len(train_loader.dataset)
        avg_vae_loss = total_vae_loss_epoch / len(train_loader.dataset)
        avg_diffusion_loss = total_diffusion_loss_epoch / len(train_loader.dataset)
        scheduler.step()

        # 打印日志（每50个epoch）
        if (epoch + 1) % 50 == 0 or epoch < 50:
            print(f"Epoch [{epoch + 1}/{epochs}], "
                  f"Avg Total Loss: {avg_total_loss:.4f}, "
                  f"VAE Loss: {avg_vae_loss:.4f}, "
                  f"Diffusion Loss: {avg_diffusion_loss:.4f}")
    return model


# ---------------------- 生成函数（调用示例） ----------------------
def generate_samples(
        model: PGDM, x_target: np.ndarray, num_samples: int = 10,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), output_dir: str = "generated_samples"
) -> Tuple[np.ndarray, str]:
    """
    Args:
    x_target: 目标模式特征，shape=(1,4) → [x_avg, x_max, x_sun, x_fluc]
        num_samples: 生成样本数
        output_dir: 输出目录基础路径
    Returns:
        y_gen_np: 生成的光伏序列，shape=(num_samples, 24)
        save_folder: 保存的文件夹路径
    """
    # 创建特征命名的文件夹
    feature_str = f"avg{x_target[0, 0]:.2f}_max{x_target[0, 1]:.2f}_sun{x_target[0, 2]:.2f}_fluc{x_target[0, 3]:.2f}"
    save_folder = os.path.join(output_dir, feature_str)
    os.makedirs(save_folder, exist_ok=True)
    # 扩展为(num_samples,4)（多个样本共享同一目标模式）
    x_target_tensor = torch.tensor(x_target, dtype=torch.float32).repeat(num_samples, 1).to(device)
    # 生成序列
    y_gen = model.generate(x_target_tensor, num_samples=num_samples)
    # 转为numpy数组
    y_gen_np = y_gen.cpu().numpy()
    return y_gen_np, save_folder


if __name__ == '__main__':
    # 验证PyTorch+CUDA
    print("PyTorch版本：", torch.__version__)
    print("CUDA是否可用：", torch.cuda.is_available())  # 输出True表示GPU可用
    print("GPU数量：", torch.cuda.device_count())  # 输出GPU数量（≥1即正常）
    # 1. 加载本地数据集
    excel_file_path = \
        r"D:\科研\可控场景生成\Diffusion model\diffusion model for controllable scenario generation\dataset\PVdata.xlsx"
    result = excel8760_to_daily24_npy(
        excel_path=excel_file_path,
        sheet_name=None,  # 读取所有sheet
        skip_header=0  
    )
    # 加载数据
    NPY_FILE_PATH = \
        r"D:\科研\可控场景生成\Diffusion model\diffusion model for controllable scenario generation\dataset\PVdata.npy"
    train_dataset = PVDataSet(npy_file_path=NPY_FILE_PATH, normalize=True, max_power=None)
    # train_dataset = WDDataSet(npy_file_path=NPY_FILE_PATH, normalize=True, max_power=None)
    train_loader = DataLoader(
        train_dataset,
        batch_size=10,
        shuffle=True,
        drop_last=True
    )
    # for batch in train_loader:
    #     print(f"批次形状：{batch.shape}，数值范围：{batch.min():.4f}~{batch.max():.4f}")
    #     break
    # ---------------------- 3. 初始化模型 ----------------------
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"✅ 使用设备：{device}")
    # ---------------------- 第一步：预训练对比学习模型 ----------------------
    print("\n=== 第一步：预训练对比学习模型 ===")
    contrastive_model = ContrastivePretrainModel(
        pattern_input_dim=4,
        scenario_seq_len=24,
        embed_dim=512
    )
    # 训练对比学习模型
    contrastive_model = train_contrastive_model(
        model=contrastive_model,
        train_loader=train_loader,
        epochs=500,
        lr=1e-3,
        device=device
    )
    # 保存预训练模型
    torch.save(contrastive_model.state_dict(), "contrastive_pretrain_model.pth")
    print("✅ 对比学习预训练模型保存成功！")
    # ---------------------- 第二步：训练PGDM模型 ----------------------
    print("\n=== 第二步：训练PGDM模型 ===")
    # 使用预训练的模式编码器
    pretrained_pattern_encoder = contrastive_model.pattern_encoder
    pgdm_model = PGDM(
        seq_len=24,
        latent_dim=64,
        pattern_embed_dim=512,
        diffusion_T=100,
        pretrained_pattern_encoder=pretrained_pattern_encoder,
        freeze_pattern_encoder=True
    ).to(device)
    # 训练PGDM模型
    print("\n开始训练PGDM模型...")
    pgdm_model = train_pgdm(
        model=pgdm_model,
        train_loader=train_loader,
        epochs=800,
        lr=1e-3,
        device=device
    )
    # 保存模型（含归一化信息，方便后续生成时反归一化）
    save_dict = {
        "model_state_dict": pgdm_model.state_dict(),
        "normalization_max": train_dataset.get_normalization_max()  # 保存归一化最大值
    }
    torch.save(save_dict, "pgdm_pv_model_with_norm.pth")
    print("✅ 模型保存成功！文件：pgdm_pv_model_with_norm.pth")
    # ---------------------- 第三步：生成光伏序列 ----------------------
    print("\n=== 第三步：生成光伏序列 ===")
    # 获取训练数据的特征最大值用于归一化
    pattern_features_max = train_dataset.get_pattern_features_max()
    print(f"训练数据特征最大值: {pattern_features_max}")
    x_target = np.array([[0.1, 0.4, 9, 0.98]])
    x_target_normalized = x_target / pattern_features_max
    num_generate = 10  # 生成10条符合目标模式的序列

    # 生成序列（归一化后的值）
    y_gen_normalized, save_folder = generate_samples(
        model=pgdm_model,
        x_target=x_target_normalized,
        num_samples=num_generate,
        device=device,
        output_dir="generated_pv_sequences"
    )
    # 反归一化到真实功率值（可选，根据需求启用）
    norm_max = train_dataset.get_normalization_max()
    y_gen_real = y_gen_normalized * norm_max  # 真实功率值
    # 打印生成结果信息
    print(f"\n✅ 生成完成！生成序列形状：{y_gen_real.shape}（样本数×24）")
    print(f"✅ 生成序列真实功率范围：{y_gen_real.min():.2f} ~ {y_gen_real.max():.2f}")
    print(f"✅ 保存文件夹：{save_folder}")
    # 保存生成的真实功率序列到npy文件
    np.save(os.path.join(save_folder, "generated_pv_real.npy"), y_gen_real)
    # np.save(os.path.join(save_folder, "generated_wd_real.npy"), y_gen_real)
    # 保存生成的真实功率序列到Excel文件
    df = pd.DataFrame(y_gen_real)
    df.columns = [f'Hour_{i + 1:02d}' for i in range(24)]
    excel_path = os.path.join(save_folder, "generated_pv_sequences.xlsx")
    # excel_path = os.path.join(save_folder, "generated_wd_sequences.xlsx")
    df.to_excel(excel_path, index=False)
    # 绘制曲线图并保存
    for i, sample in enumerate(y_gen_real):
        sample_flat = sample.flatten()
        plt.figure(figsize=(10, 5))
        # 绘制24小时曲线
        hours = list(range(24))
        plt.plot(hours, sample_flat, 'b-', linewidth=2, alpha=0.8, marker='o', markersize=4)
        # 设置横坐标刻度
        plt.xticks(hours, [f'{h:02d}:00' for h in hours], rotation=45)
        # 添加网格和标签
        plt.grid(True, alpha=0.3)
        plt.xlabel('Time (Hour of Day)')
        plt.ylabel('PV Power (kW)')
        # plt.ylabel('WD Power (kW)')
        # 设置y轴从0开始
        plt.ylim(bottom=0)
        # 标题包含特征信息
        # feature_info = f'Avg={x_target[0, 0]:.1f}kW, Max={x_target[0, 1]:.1f}kW, Min={x_target[0, 2]:.1f}h, Fluc={x_target[0, 3]:.1f}kW'
        # plt.title(f'Generated Daily WD Profile {i + 1}\n{feature_info}')
        feature_info = f'Avg={x_target[0, 0]:.1f}kW, Max={x_target[0, 1]:.1f}kW, Sun={x_target[0, 2]:.1f}h, Fluc={x_target[0, 3]:.1f}kW'
        plt.title(f'Generated Daily PV Profile {i + 1}\n{feature_info}')
        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"generated_sample_{i + 1:02d}.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(
            f"✅ 样本 {i + 1} 已保存，形状: {sample.shape}, 数值范围: [{sample_flat.min():.2f}, {sample_flat.max():.2f}]")
        # 保存特征信息文件
    feature_info = {
        'x_target_real': x_target.tolist(),
        'x_target_normalized': x_target_normalized.tolist(),
        'pattern_features_max': pattern_features_max.tolist(),
        'features': ['average_power(kW)', 'max_power(kW)', 'sunshine_duration(hours)', 'fluctuation_power(kW)'],
        'num_samples': num_generate,
        'normalization_max': norm_max
    }
    # feature_info = {
    #     'x_target_real': x_target.tolist(),
    #     'x_target_normalized': x_target_normalized.tolist(),
    #     'pattern_features_max': pattern_features_max.tolist(),
    #     'features': ['average_power(kW)', 'max_power(kW)', 'min_power(kW)', 'fluctuation_power(kW)'],
    #     'num_samples': num_generate,
    #     'normalization_max': norm_max
    # }
    with open(os.path.join(save_folder, "generation_info.json"), 'w') as f:
        json.dump(feature_info, f, indent=2)

    print(f"\n✅ 所有生成序列和图表已保存到：{save_folder}")
    print(f"✅ 特征信息已保存到：generation_info.json")

