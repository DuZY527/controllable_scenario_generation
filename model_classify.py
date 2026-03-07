import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import os
from typing import Tuple, List

from PatternEncoderPretrain import ContrastivePretrainModel


class PerceptualVAE(nn.Module):
    def __init__(self, seq_len: int = 168, latent_dim: int = 128):  # 注意：输入改为168小时
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        # 多尺度卷积配置：3h, 24h, 48h, 72h, 120h
        encoder_channels = [1, 64, 128, 256, 512, 1024]  # 增加一层以匹配5个尺度
        decoder_channels = [1024, 512, 256, 128, 64, 1]  # 对称解码器
        self.encoder_dims = encoder_channels
        self.decoder_dims = decoder_channels
        # 1. 多尺度编码器设计
        encoder_layers = []
        # 第一层：3小时卷积（捕捉局部细节和小时级波动）
        encoder_layers.extend([
            nn.Conv1d(encoder_channels[0], encoder_channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(encoder_channels[1])
        ])
        # 第二层：24小时卷积（捕捉日周期模式）
        # 计算下采样参数：168 -> 84 (stride=2)
        encoder_layers.extend([
            nn.Conv1d(encoder_channels[1], encoder_channels[2], kernel_size=24, stride=2, padding=11),  # 168->84
            nn.ReLU(),
            nn.BatchNorm1d(encoder_channels[2])
        ])
        # 第三层：48小时卷积（捕捉2日模式）
        # 计算下采样参数：84 -> 42 (stride=2)
        encoder_layers.extend([
            nn.Conv1d(encoder_channels[2], encoder_channels[3], kernel_size=48, stride=2, padding=23),  # 84->42
            nn.ReLU(),
            nn.BatchNorm1d(encoder_channels[3])
        ])
        # 第四层：72小时卷积（捕捉3日模式）
        # 计算下采样参数：42 -> 21 (stride=2)
        encoder_layers.extend([
            nn.Conv1d(encoder_channels[3], encoder_channels[4], kernel_size=72, stride=2, padding=35),  # 42->21
            nn.ReLU(),
            nn.BatchNorm1d(encoder_channels[4])
        ])
        # 第五层：120小时卷积（捕捉周中模式）
        # 计算下采样参数：21 -> 10 (stride=2)
        encoder_layers.extend([
            nn.Conv1d(encoder_channels[4], encoder_channels[5], kernel_size=120, stride=2, padding=59),  # 21->10
            nn.ReLU(),
            nn.BatchNorm1d(encoder_channels[5])
        ])
        # 全局平均池化
        encoder_layers.append(nn.AdaptiveAvgPool1d(1))
        encoder_layers.append(nn.Flatten())
        # 编码器序列
        self.encoder_conv = nn.Sequential(*encoder_layers)
        # 输出层到潜在空间
        self.encoder_fc = nn.Linear(encoder_channels[5], 2 * latent_dim)  # 输出均值和对数方差

        self.decoder_input_length = 10
        # 2. 多尺度解码器设计（与编码器对称）
        # 初始全连接层，将潜在向量扩展到适合转置卷积的尺寸
        self.decoder_fc = nn.Linear(latent_dim, decoder_channels[0] * 10)  # 10是下采样后的长度
        decoder_layers = []
        # 第一层上采样：10 -> 21
        decoder_layers.extend([
            nn.ConvTranspose1d(decoder_channels[0], decoder_channels[1],
                               kernel_size=120, stride=2, padding=59, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(decoder_channels[1])
        ])
        # 第二层上采样：21 -> 42
        decoder_layers.extend([
            nn.ConvTranspose1d(decoder_channels[1], decoder_channels[2],
                               kernel_size=72, stride=2, padding=35, output_padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(decoder_channels[2])
        ])
        # 第三层上采样：42 -> 84
        decoder_layers.extend([
            nn.ConvTranspose1d(decoder_channels[2], decoder_channels[3],
                               kernel_size=48, stride=2, padding=23, output_padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(decoder_channels[3])
        ])
        # 第四层上采样：84 -> 168
        decoder_layers.extend([
            nn.ConvTranspose1d(decoder_channels[3], decoder_channels[4],
                               kernel_size=24, stride=2, padding=11, output_padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(decoder_channels[4])
        ])
        # 最后一层：调整到最终输出（不改变尺寸，只调整通道数）
        decoder_layers.extend([
            nn.Conv1d(decoder_channels[4], decoder_channels[5], kernel_size=3, padding=1),
            nn.Sigmoid()  # 确保输出在[0,1]范围内
        ])
        self.decoder_conv = nn.Sequential(*decoder_layers)
        # 3. 感知损失配置
        # 编码器特征层索引（每层卷积后）
        self.encoder_feature_indices = [0, 2, 5, 8, 11]  # 对应5个卷积层的输出
        # 解码器特征层索引（每个转置卷积后）
        self.decoder_feature_indices = [0, 3, 6, 9]  # 对应4个转置卷积层的输出（最后一层卷积前）
        # 4. 通道调整层，用于处理感知损失中的通道不匹配
        self.channel_adjust = nn.ModuleDict({
            '64to128': nn.Conv1d(64, 128, 1),
            '128to64': nn.Conv1d(128, 64, 1),
            '128to256': nn.Conv1d(128, 256, 1),
            '256to128': nn.Conv1d(256, 128, 1),
            '256to512': nn.Conv1d(256, 512, 1),
            '512to256': nn.Conv1d(512, 256, 1),
            '512to1024': nn.Conv1d(512, 1024, 1),
            '1024to512': nn.Conv1d(1024, 512, 1)
        })
        # 打印网络配置信息
        print(f"✅ 多尺度PerceptualVAE初始化完成:")
        print(f"   - 输入序列长度: {seq_len}小时")
        print(f"   - 潜在空间维度: {latent_dim}")
        print(f"   - 编码器通道: {encoder_channels}")
        print(f"   - 解码器通道: {decoder_channels}")
        print(f"   - 卷积尺度: 3h-24h-48h-72h-120h")

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
        h = h.view(h.size(0), self.decoder_dims[0], self.decoder_input_length)
        # 通过转置卷积重建序列
        reconstructed = self.decoder_conv(h)  # (B, 1, 24)
        return reconstructed.squeeze(1)  # (B, 24)

    def get_encoder_features(self, x):
        """获取编码器中间层特征，用于感知损失"""
        features = []
        x = x.unsqueeze(1)  # (B, 1, 168)
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
        # 修改点1：使用动态维度而不是硬编码的 128, 6
        h = h.view(h.size(0), self.decoder_dims[0], self.decoder_input_length)
        # 手动遍历解码器层以收集中间特征
        for i, layer in enumerate(self.decoder_conv):
            h = layer(h)
            if i in self.decoder_feature_indices:
                features.append(h)
        return features

    def adjust_feature_channels(self, feat, target_channels):
        """调整特征图通道数以匹配目标通道数"""
        if feat.size(1) == target_channels:
            return feat
        # 根据通道数差异选择合适的调整层
        channel_pair = f"{feat.size(1)}to{target_channels}"
        if channel_pair in self.channel_adjust:
            return self.channel_adjust[channel_pair](feat)
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
    def __init__(self, latent_dim: int = 128, pattern_embed_dim: int = 512, T: int = 100):
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

    def reverse_diffusion(self, p_emb: torch.Tensor, latent_dim: int = 128, num_samples: int = 1) -> torch.Tensor:
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
    def __init__(self, seq_len: int = 168, latent_dim: int = 128, pattern_embed_dim: int = 512, diffusion_T: int = 100,
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
        p_emb, _ = self.pattern_encoder(x)  # (B, 512)
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
            p_emb_target, _ = self.pattern_encoder(x_target)  # (num_samples, 512)
            # 2. 反向扩散生成潜在向量 z_0
            z_0_gen = self.latent_diffusion.reverse_diffusion(p_emb_target, latent_dim=self.perceptual_vae.latent_dim,
                                                              num_samples=num_samples)
            # 3. VAE解码生成光伏序列
            y_gen = self.perceptual_vae.decode(z_0_gen)  # (num_samples, 24)
            # 裁剪到 [0,1]（光伏出力非负）
            y_gen = torch.clamp(y_gen, 0.0, 1.0)
        self.train()
        return y_gen


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
