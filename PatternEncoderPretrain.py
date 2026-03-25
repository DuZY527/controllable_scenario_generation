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
from typing import Tuple, List, Optional
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class PatternEncoder(nn.Module):
    """模式编码器：4*7维模式特征 → 512维嵌入"""
    def __init__(self, input_dim: int = 7, embed_dim: int = 512):
        super().__init__()
        # 基于3尺度特征组分解对应编码
        # 整体尺度特征编码
        self.daily_encoder = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        # 日际尺度特征编码
        self.interdaily_encoder = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        # 日内小尺度特征编码
        self.intradaily_encoder = nn.Sequential(
            nn.Linear(15, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        # 多尺度特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(384, 512),  # 128*3 = 384
            nn.ReLU(),
            nn.LayerNorm(512)
        )
        # 用于多尺度对比学习的投影头
        self.daily_projection = nn.Linear(128, embed_dim)
        self.interdaily_projection = nn.Linear(128, embed_dim)
        self.intradaily_projection = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # 分解7维特征
        daily_feat = x[:, 0, [0, 1, 2, 5, 6]]   #
        interdaily_feat = torch.cat([x[:, 0, 3:4], x[:, 1, :]], dim=1)  #
        intradaily_feat = torch.cat([x[:, 0, 4:5], x[:, 2, :], x[:, 3, :]], dim=1)  #
        # 各尺度编码
        daily_encoded = self.daily_encoder(daily_feat)
        interdaily_encoded = self.interdaily_encoder(interdaily_feat)
        intradaily_encoded = self.intradaily_encoder(intradaily_feat)
        # 多尺度特征融合
        fused = torch.cat([daily_encoded, interdaily_encoded, intradaily_encoded], dim=1)
        overall_embedding = self.fusion(fused)
        # 为多尺度对比学习准备的特征
        multi_scale_features = [
            self.daily_projection(daily_encoded),
            self.interdaily_projection(interdaily_encoded),
            self.intradaily_projection(intradaily_encoded)
        ]

        return overall_embedding, multi_scale_features


class ScenarioEncoder(nn.Module):
    """场景编码器：168维光伏序列 → 512维嵌入（并行多尺度分支+多头注意力）"""
    def __init__(self, seq_len: int = 168, embed_dim: int = 512):
        super().__init__()
        # 1. 并行多尺度分支
        # 日尺度分支：处理24小时窗口
        self.daily_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1),  # 保持长度24
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1),  # 聚合为 (B,128,1)
            nn.Flatten()  # (B,128)
        )
        self.daily_attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        # 半周尺度分支：处理84小时窗口
        self.halfweek_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5, padding=2),  # 保持长度84
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.halfweek_attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        # 周尺度分支：处理168小时全局序列
        self.weekly_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=7, padding=3),  # 保持长度168
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.weekly_attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        # 2. 多尺度融合的注意力机制
        self.scale_attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        # 3. 最终映射层
        self.fc = nn.Sequential(
            nn.Linear(384, 512),  # 128*3 = 384
            nn.ReLU(),
            nn.LayerNorm(512)
        )
        # 用于多尺度对比学习的投影头
        self.daily_projection = nn.Linear(128, embed_dim)
        self.halfweek_projection = nn.Linear(128, embed_dim)
        self.weekly_projection = nn.Linear(128, embed_dim)

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch_size = y.shape[0]
        # 1. 日尺度处理：将168小时分为7个24小时日序列
        daily_sequences = y.reshape(batch_size, 7, 24)  # (B, 7, 24)
        daily_features = []
        for day_idx in range(7):
            daily_seq = daily_sequences[:, day_idx, :]  # (B, 24)
            daily_feat = self.daily_branch(daily_seq.unsqueeze(1)).unsqueeze(1)  # (B, 1, 128)
            daily_features.append(daily_feat)
        daily_features = torch.cat(daily_features, dim=1)  # (B, 7, 128)
        daily_attended, _ = self.daily_attention(daily_features, daily_features, daily_features)
        daily_pooled = torch.mean(daily_attended, dim=1)  # (B, 128)
        # 2. 半周尺度处理：将168小时分为2个84小时半周序列
        halfweek_seq1 = y[:, :84]  # 前半周
        halfweek_seq2 = y[:, 84:168]  # 后半周
        halfweek_feat1 = self.halfweek_branch(halfweek_seq1.unsqueeze(1)).unsqueeze(1)  # (B, 1, 128)
        halfweek_feat2 = self.halfweek_branch(halfweek_seq2.unsqueeze(1)).unsqueeze(1)  # (B, 1, 128)
        halfweek_features = torch.cat([halfweek_feat1, halfweek_feat2], dim=1)  # (B, 2, 128)
        halfweek_attended, _ = self.halfweek_attention(halfweek_features, halfweek_features, halfweek_features)
        halfweek_pooled = torch.mean(halfweek_attended, dim=1)  # (B, 128)
        # 3. 周尺度处理：全局168小时序列
        weekly_feat = self.weekly_branch(y.unsqueeze(1)).unsqueeze(1)  # (B, 1, 128)
        weekly_attended, _ = self.weekly_attention(weekly_feat, weekly_feat, weekly_feat)
        weekly_pooled = weekly_attended.squeeze(1)  # (B, 128)
        # 4. 多尺度融合
        scale_features = torch.stack([daily_pooled, halfweek_pooled, weekly_pooled], dim=1)  # (B, 3, 128)
        scale_attended, _ = self.scale_attention(scale_features, scale_features, scale_features)
        scale_pooled = torch.mean(scale_attended, dim=1)  # (B, 128)
        # 5. 最终嵌入
        fused_features = torch.cat([daily_pooled, halfweek_pooled, weekly_pooled], dim=1)  # (B, 384)
        overall_embedding = self.fc(fused_features)
        # 为多尺度对比学习准备的特征
        multi_scale_features = [
            self.daily_projection(daily_pooled),
            self.halfweek_projection(halfweek_pooled),
            self.weekly_projection(weekly_pooled)
        ]

        return overall_embedding, multi_scale_features


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


def hierarchical_contrastive_loss(
        pe_embed: torch.Tensor,
        se_embed: torch.Tensor,
        pe_multi_scale: List[torch.Tensor],
        se_multi_scale: List[torch.Tensor],
        weights: Optional[List[float]] = None
) -> torch.Tensor:
    if weights is None:
        weights = [0.4, 0.2, 0.2, 0.2]  # 整体, 日尺度, 周尺度, 跨尺度一致性
    # 1. 整体对比损失
    overall_loss = contrastive_loss(pe_embed, se_embed)
    # 2. 尺度对应损失（注意：模式编码器的尺度与场景编码器的尺度对应关系）
    # 日尺度：模式日尺度 vs 场景日尺度
    daily_loss = contrastive_loss(pe_multi_scale[0], se_multi_scale[0])
    # 周尺度：模式日际/日内尺度 vs 场景半周/周尺度
    weekly_loss = contrastive_loss(
        torch.cat([pe_multi_scale[1], pe_multi_scale[2]], dim=1),
        torch.cat([se_multi_scale[1], se_multi_scale[2]], dim=1)
    )
    # 3. 跨尺度一致性损失（鼓励同一数据的不同尺度表示一致）
    consistency_loss = 0.0
    # 模式编码器内部尺度一致性
    for i in range(len(pe_multi_scale)):
        for j in range(i + 1, len(pe_multi_scale)):
            consistency_loss += F.mse_loss(
                F.normalize(pe_multi_scale[i], dim=1),
                F.normalize(pe_multi_scale[j], dim=1)
            )
    # 场景编码器内部尺度一致性
    for i in range(len(se_multi_scale)):
        for j in range(i + 1, len(se_multi_scale)):
            consistency_loss += F.mse_loss(
                F.normalize(se_multi_scale[i], dim=1),
                F.normalize(se_multi_scale[j], dim=1)
            )
    consistency_loss = consistency_loss / (len(pe_multi_scale) * (len(pe_multi_scale) - 1) / 2 +
                                           len(se_multi_scale) * (len(se_multi_scale) - 1) / 2)
    # 4. 加权总损失
    total_loss = (
            weights[0] * overall_loss +
            weights[1] * daily_loss +
            weights[2] * weekly_loss +
            weights[3] * consistency_loss
    )

    return total_loss


class ContrastivePretrainModel(nn.Module):
    """对比学习预训练模型：包含模式编码器和场景编码器"""

    def __init__(self, pattern_input_dim: int = 7, scenario_seq_len: int = 168, embed_dim: int = 512):
        super().__init__()
        self.pattern_encoder = PatternEncoder(pattern_input_dim, embed_dim)
        self.scenario_encoder = ScenarioEncoder(scenario_seq_len, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            x: 模式特征，(B, 4)
            y: 光伏出力序列，(B, 24)
        Returns:
            p_emb: 模式嵌入，(B, 512)
            s_emb: 场景嵌入，(B, 512)
        """
        p_emb, p_multi_scale = self.pattern_encoder(x)
        s_emb, s_multi_scale = self.scenario_encoder(y)
        return p_emb, s_emb, p_multi_scale, s_multi_scale

    def compute_loss(self, p_emb: torch.Tensor, s_emb: torch.Tensor, p_multi_scale: List[torch.Tensor], s_multi_scale: List[torch.Tensor]) -> torch.Tensor:
        """计算加权多尺度对比损失"""
        return hierarchical_contrastive_loss(p_emb, s_emb, p_multi_scale, s_multi_scale)


def train_contrastive_model(
        model: ContrastivePretrainModel,
        train_loader: DataLoader,
        epochs: int = 10000,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu")
):
    """训练对比学习预训练模型"""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("开始训练对比学习预训练模型...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (y_batch, x_batch) in enumerate(train_loader):
            y_batch = y_batch.to(device)
            x_batch = x_batch.to(device)
            # 前向传播
            p_emb, s_emb, p_multi_scale, s_multi_scale = model(x_batch, y_batch)
            loss = model.compute_loss(p_emb, s_emb, p_multi_scale, s_multi_scale)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / num_batches

        if (epoch + 1) % 100 == 0 or epoch < 20:
            print(f"预训练 Epoch [{epoch + 1}/{epochs}], 平均对比损失: {avg_loss:.4f}")

    return model


# Override with robust checkpoint version.
def train_contrastive_model(
        model: ContrastivePretrainModel,
        train_loader: DataLoader,
        epochs: int = 10000,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
        save_every: int = 50,
        checkpoint_path: str = "contrastive_pretrain_checkpoint.pth",
        best_model_path: str = "contrastive_pretrain_model_best.pth",
        resume: bool = True
):
    """Train contrastive model with periodic checkpointing and auto-resume support."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    save_every = max(1, int(save_every))
    start_epoch = 0
    best_loss = float("inf")

    def _save_training_checkpoint(epoch_idx: int, current_loss: float) -> None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            {
                "epoch": epoch_idx,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": float(best_loss),
                "current_loss": float(current_loss),
            },
            checkpoint_path
        )

    if resume and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            best_loss = float(checkpoint.get("best_loss", float("inf")))
            start_epoch = int(checkpoint.get("epoch", -1)) + 1
            if start_epoch < epochs:
                print(f"Resume contrastive training from epoch {start_epoch + 1}/{epochs}")
            else:
                print("Contrastive checkpoint already reached target epochs, training will stop.")
        except Exception as e:
            print(f"Warning: failed to load contrastive checkpoint, start from scratch. Error: {e}")
            start_epoch = 0
            best_loss = float("inf")

    print("Start contrastive pretraining...")
    epoch = start_epoch - 1
    total_loss = 0.0
    num_batches = 0

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            total_loss = 0.0
            num_batches = 0

            for y_batch, x_batch in train_loader:
                y_batch = y_batch.to(device)
                x_batch = x_batch.to(device)

                p_emb, s_emb, p_multi_scale, s_multi_scale = model(x_batch, y_batch)
                loss = model.compute_loss(p_emb, s_emb, p_multi_scale, s_multi_scale)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(num_batches, 1)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_dir = os.path.dirname(best_model_path)
                if best_model_dir:
                    os.makedirs(best_model_dir, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "best_loss": float(best_loss),
                        "model_state_dict": model.state_dict(),
                    },
                    best_model_path
                )

            if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
                _save_training_checkpoint(epoch, avg_loss)
                print(
                    f"[Checkpoint] contrastive epoch {epoch + 1}/{epochs}, "
                    f"current_loss={avg_loss:.4f}, best_loss={best_loss:.4f}"
                )

            if (epoch + 1) % 100 == 0 or epoch < 20:
                print(f"Epoch [{epoch + 1}/{epochs}], Avg Contrastive Loss: {avg_loss:.4f}")
    except KeyboardInterrupt:
        interrupted_loss = total_loss / max(num_batches, 1) if num_batches > 0 else best_loss
        _save_training_checkpoint(epoch, interrupted_loss)
        print(
            f"\nTraining interrupted. Checkpoint saved at epoch {max(epoch + 1, 0)} "
            f"to {checkpoint_path}"
        )

    return model

