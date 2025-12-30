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
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class PatternEncoder(nn.Module):
    """模式编码器：4维模式特征 → 512维嵌入"""
    def __init__(self, input_dim: int = 4, embed_dim: int = 512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch_size, 4) → output: (batch_size, 512)"""
        return self.fc(x)


class ScenarioEncoder(nn.Module):
    """场景编码器：168维光伏序列 → 512维嵌入（含位置注意力）"""
    def __init__(self, seq_len: int = 24, embed_dim: int = 512):
        super().__init__()
        # 序列嵌入：168维 → 256维
        self.seq_embed = nn.Linear(seq_len, 256)
        # 位置注意力（参考原文 3.2 节位置注意力机制）
        self.attention = nn.Linear(256, 256)
        # 最终映射到512维
        self.fc = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """y: (batch_size, 24) → output: (batch_size, 512)"""
        # 1. 序列嵌入
        y_embed = self.seq_embed(y)  # (B, 256)
        # 2. 位置注意力：Softmax(σ(Linear(y_embed))) ⊙ y_embed
        attn_weight = F.softmax(torch.sigmoid(self.attention(y_embed)), dim=1)  # (B, 256)
        y_attended = y_embed * attn_weight  # (B, 256)
        # 3. 最终嵌入
        return self.fc(y_attended)


def contrastive_loss(pe_embed: torch.Tensor, se_embed: torch.Tensor) -> torch.Tensor:
    """
    Args:
        pe_embed: 模式编码器输出，(B, 512)
        se_embed: 场景编码器输出，(B, 512)
    Returns:
        交叉熵对比损失
    """
    B = pe_embed.shape[0]
    # 1. 计算余弦相似度矩阵 e_ij = (se_i · pe_j) / (||se_i||·||pe_j||)
    pe_norm = F.normalize(pe_embed, dim=1)  # (B, 512)
    se_norm = F.normalize(se_embed, dim=1)  # (B, 512)
    sim_matrix = torch.matmul(se_norm, pe_norm.T)  # (B, B)：e_ij

    # 2. 交叉熵损失（行方向 + 列方向，参考原文 Eq.7）
    # 行方向：每个 se_i 对应 pe_i 为正样本
    row_labels = torch.arange(B, device=sim_matrix.device)  # (B,)
    row_loss = F.cross_entropy(sim_matrix, row_labels)
    # 列方向：每个 pe_j 对应 se_j 为正样本
    col_loss = F.cross_entropy(sim_matrix.T, row_labels)
    return (row_loss + col_loss) / 2


class ContrastivePretrainModel(nn.Module):
    """对比学习预训练模型：包含模式编码器和场景编码器"""

    def __init__(self, pattern_input_dim: int = 4, scenario_seq_len: int = 24, embed_dim: int = 512):
        super().__init__()
        self.pattern_encoder = PatternEncoder(pattern_input_dim, embed_dim)
        self.scenario_encoder = ScenarioEncoder(scenario_seq_len, embed_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 模式特征，(B, 4)
            y: 光伏出力序列，(B, 24)
        Returns:
            p_emb: 模式嵌入，(B, 512)
            s_emb: 场景嵌入，(B, 512)
        """
        p_emb = self.pattern_encoder(x)
        s_emb = self.scenario_encoder(y)
        return p_emb, s_emb

    def compute_loss(self, p_emb: torch.Tensor, s_emb: torch.Tensor) -> torch.Tensor:
        """计算对比损失"""
        return contrastive_loss(p_emb, s_emb)


def train_contrastive_model(
        model: ContrastivePretrainModel,
        train_loader: DataLoader,
        epochs: int = 100,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu")
):
    """训练对比学习预训练模型"""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("开始训练对比学习预训练模型...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (y_batch, x_batch) in enumerate(train_loader):
            y_batch = y_batch.to(device)
            x_batch = x_batch.to(device)

            # 前向传播
            p_emb, s_emb = model(x_batch, y_batch)
            loss = model.compute_loss(p_emb, s_emb)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        if (epoch + 1) % 20 == 0 or epoch < 20:
            print(f"预训练 Epoch [{epoch + 1}/{epochs}], 平均对比损失: {avg_loss:.4f}")

    return model

