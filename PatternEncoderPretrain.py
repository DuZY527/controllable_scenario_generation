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
    """Pattern encoder: map 4x7 features to a 512-d embedding."""
    def __init__(self, input_dim: int = 7, embed_dim: int = 512):
        super().__init__()
        # Encode three scale-specific feature groups.
        # Daily-scale feature encoder.
        self.daily_encoder = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        # Interdaily-scale feature encoder.
        self.interdaily_encoder = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        # Intradaily-scale feature encoder.
        self.intradaily_encoder = nn.Sequential(
            nn.Linear(15, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        # 
        self.fusion = nn.Sequential(
            nn.Linear(384, 512),  # 128*3 = 384
            nn.ReLU(),
            nn.LayerNorm(512)
        )
        # Projection heads for multi-scale contrastive learning.
        self.daily_projection = nn.Linear(128, embed_dim)
        self.interdaily_projection = nn.Linear(128, embed_dim)
        self.intradaily_projection = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Decompose 7-dimensional pattern features by scale.
        daily_feat = x[:, 0, [0, 1, 2, 5, 6]]   #
        interdaily_feat = torch.cat([x[:, 0, 3:4], x[:, 1, :]], dim=1)  #
        intradaily_feat = torch.cat([x[:, 0, 4:5], x[:, 2, :], x[:, 3, :]], dim=1)  #
        # Encode each scale branch.
        daily_encoded = self.daily_encoder(daily_feat)
        interdaily_encoded = self.interdaily_encoder(interdaily_feat)
        intradaily_encoded = self.intradaily_encoder(intradaily_feat)
        # Fuse scale features into the overall embedding.
        fused = torch.cat([daily_encoded, interdaily_encoded, intradaily_encoded], dim=1)
        overall_embedding = self.fusion(fused)
        # Multi-scale features used by hierarchical contrastive loss.
        multi_scale_features = [
            self.daily_projection(daily_encoded),
            self.interdaily_projection(interdaily_encoded),
            self.intradaily_projection(intradaily_encoded)
        ]

        return overall_embedding, multi_scale_features


class ScenarioEncoder(nn.Module):
    """Scenario encoder: map a 168-length sequence to a 512-d embedding."""
    def __init__(self, seq_len: int = 168, embed_dim: int = 512):
        super().__init__()
        # 1. Parallel multi-scale branches.
        # Daily branch (24-hour window).
        self.daily_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1),  # 24
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1),  # Pool to (B, 128, 1)
            nn.Flatten()  # (B,128)
        )
        self.daily_attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        # Half-week branch (84-hour window).
        self.halfweek_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5, padding=2),  # 84
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.halfweek_attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        # 168
        self.weekly_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=7, padding=3),  # 168
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.weekly_attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        # 2. Cross-scale attention for multi-scale interaction.
        self.scale_attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        # 3. 
        self.fc = nn.Sequential(
            nn.Linear(384, 512),  # 128*3 = 384
            nn.ReLU(),
            nn.LayerNorm(512)
        )
        # Projection heads for multi-scale contrastive learning.
        self.daily_projection = nn.Linear(128, embed_dim)
        self.halfweek_projection = nn.Linear(128, embed_dim)
        self.weekly_projection = nn.Linear(128, embed_dim)

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch_size = y.shape[0]
        # 1. Daily scale: split 168 hours into 7 daily windows of 24.
        daily_sequences = y.reshape(batch_size, 7, 24)  # (B, 7, 24)
        daily_features = []
        for day_idx in range(7):
            daily_seq = daily_sequences[:, day_idx, :]  # (B, 24)
            daily_feat = self.daily_branch(daily_seq.unsqueeze(1)).unsqueeze(1)  # (B, 1, 128)
            daily_features.append(daily_feat)
        daily_features = torch.cat(daily_features, dim=1)  # (B, 7, 128)
        daily_attended, _ = self.daily_attention(daily_features, daily_features, daily_features)
        daily_pooled = torch.mean(daily_attended, dim=1)  # (B, 128)
        # 2. Half-week scale: split into two 84-hour windows.
        halfweek_seq1 = y[:, :84]  # First half-week
        halfweek_seq2 = y[:, 84:168]  # Second half-week
        halfweek_feat1 = self.halfweek_branch(halfweek_seq1.unsqueeze(1)).unsqueeze(1)  # (B, 1, 128)
        halfweek_feat2 = self.halfweek_branch(halfweek_seq2.unsqueeze(1)).unsqueeze(1)  # (B, 1, 128)
        halfweek_features = torch.cat([halfweek_feat1, halfweek_feat2], dim=1)  # (B, 2, 128)
        halfweek_attended, _ = self.halfweek_attention(halfweek_features, halfweek_features, halfweek_features)
        halfweek_pooled = torch.mean(halfweek_attended, dim=1)  # (B, 128)
        # 3. 168
        weekly_feat = self.weekly_branch(y.unsqueeze(1)).unsqueeze(1)  # (B, 1, 128)
        weekly_attended, _ = self.weekly_attention(weekly_feat, weekly_feat, weekly_feat)
        weekly_pooled = weekly_attended.squeeze(1)  # (B, 128)
        # 4. Multi-scale fusion via scale attention.
        scale_features = torch.stack([daily_pooled, halfweek_pooled, weekly_pooled], dim=1)  # (B, 3, 128)
        scale_attended, _ = self.scale_attention(scale_features, scale_features, scale_features)
        scale_pooled = torch.mean(scale_attended, dim=1)  # (B, 128)
        # 5. Final fused embedding.
        fused_features = torch.cat([daily_pooled, halfweek_pooled, weekly_pooled], dim=1)  # (B, 384)
        overall_embedding = self.fc(fused_features)
        # Multi-scale features used by hierarchical contrastive loss.
        multi_scale_features = [
            self.daily_projection(daily_pooled),
            self.halfweek_projection(halfweek_pooled),
            self.weekly_projection(weekly_pooled)
        ]

        return overall_embedding, multi_scale_features


def contrastive_loss(pe_embed: torch.Tensor, se_embed: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Args:
        pe_embed: Pattern encoder embedding, shape (B, 512).
        se_embed: Scenario encoder embedding, shape (B, 512).
        temperature: Temperature coefficient for similarity scaling.
    Returns:
        Symmetric contrastive loss.
    """
    B = pe_embed.shape[0]
    # 1. Cosine similarity matrix: e_ij = <se_i, pe_j> / (||se_i|| * ||pe_j||).
    pe_norm = F.normalize(pe_embed, dim=1)  # (B, 512)
    se_norm = F.normalize(se_embed, dim=1)  # (B, 512)
    sim_matrix = torch.matmul(se_norm, pe_norm.T)  # (B, B)
    sim_matrix = sim_matrix / temperature

    # 2. Bidirectional contrastive loss (row and column cross-entropy).
    # Row direction: match se_i to pe_i as positives.
    row_labels = torch.arange(B, device=sim_matrix.device)  # (B,)
    row_loss = F.cross_entropy(sim_matrix, row_labels)
    # Column direction: match pe_j to se_j as positives.
    col_loss = F.cross_entropy(sim_matrix.T, row_labels)
    return (row_loss + col_loss) / 2


def hierarchical_contrastive_loss(
        pe_embed: torch.Tensor,
        se_embed: torch.Tensor,
        pe_multi_scale: List[torch.Tensor],
        se_multi_scale: List[torch.Tensor],
        weights: Optional[List[float]] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if weights is None:
        weights = [0.4, 0.2, 0.2, 0.2]  # overall, daily, weekly, consistency
    # 1. 
    overall_loss = contrastive_loss(pe_embed, se_embed)
    # 2. 
    # Daily-scale alignment: pattern daily vs scenario daily.
    daily_loss = contrastive_loss(pe_multi_scale[0], se_multi_scale[0])
    # Weekly-scale alignment: pattern inter/intradaily vs scenario halfweek/weekly.
    weekly_loss = contrastive_loss(
        torch.cat([pe_multi_scale[1], pe_multi_scale[2]], dim=1),
        torch.cat([se_multi_scale[1], se_multi_scale[2]], dim=1)
    )
    # 3. 
    consistency_loss = 0.0
    # Consistency across pattern encoder scales.
    for i in range(len(pe_multi_scale)):
        for j in range(i + 1, len(pe_multi_scale)):
            consistency_loss += F.mse_loss(
                F.normalize(pe_multi_scale[i], dim=1),
                F.normalize(pe_multi_scale[j], dim=1)
            )
    # Consistency across scenario encoder scales.
    for i in range(len(se_multi_scale)):
        for j in range(i + 1, len(se_multi_scale)):
            consistency_loss += F.mse_loss(
                F.normalize(se_multi_scale[i], dim=1),
                F.normalize(se_multi_scale[j], dim=1)
            )
    consistency_loss = consistency_loss / (len(pe_multi_scale) * (len(pe_multi_scale) - 1) / 2 +
                                           len(se_multi_scale) * (len(se_multi_scale) - 1) / 2)
    # 4. Weighted total loss.
    total_loss = (
            weights[0] * overall_loss +
            weights[1] * daily_loss +
            weights[2] * weekly_loss +
            weights[3] * consistency_loss
    )

    return total_loss, overall_loss, daily_loss, weekly_loss, consistency_loss


class ContrastivePretrainModel(nn.Module):
    """Contrastive pretraining model wrapping pattern and scenario encoders."""

    def __init__(self, pattern_input_dim: int = 7, scenario_seq_len: int = 168, embed_dim: int = 512):
        super().__init__()
        self.pattern_encoder = PatternEncoder(pattern_input_dim, embed_dim)
        self.scenario_encoder = ScenarioEncoder(scenario_seq_len, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            x: Pattern input tensor, shape (B, 4, 7).
            y: Scenario sequence tensor, shape (B, 168).
        Returns:
            p_emb: Pattern embedding, shape (B, 512).
            s_emb: Scenario embedding, shape (B, 512).
            p_multi_scale: Pattern multi-scale embeddings (3 tensors).
            s_multi_scale: Scenario multi-scale embeddings (3 tensors).
        """
        p_emb, p_multi_scale = self.pattern_encoder(x)
        s_emb, s_multi_scale = self.scenario_encoder(y)
        return p_emb, s_emb, p_multi_scale, s_multi_scale

    def compute_loss(self, p_emb: torch.Tensor, s_emb: torch.Tensor, p_multi_scale: List[torch.Tensor], s_multi_scale: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute hierarchical contrastive loss."""
        return hierarchical_contrastive_loss(p_emb, s_emb, p_multi_scale, s_multi_scale)



# Override with robust checkpoint version.
def train_contrastive_model(
        model: ContrastivePretrainModel,
        train_loader: DataLoader,
        epochs: int = 3000,
        lr: float = 1e-3,
        device: torch.device = None,
        save_every: int = 50,
        checkpoint_path: str = "contrastive_pretrain_checkpoint.pth",
        best_model_path: str = "contrastive_pretrain_model_best.pth",
        resume: bool = True,
        visualize: bool = True,
        visualize_interval: int = 1
):
    """Train contrastive model with periodic checkpointing and auto-resume support."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    save_every = max(1, int(save_every))
    start_epoch = 0
    best_loss = float("inf")
    
    # loss
    loss_history = []
    
    # 
    if visualize:
        plt.ion()  # Enable interactive plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot([], [], 'b-', label='Contractive Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Contrastive PreTraining Loss')
        ax.legend()
        ax.grid(True)

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
                "loss_history": loss_history,
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
            # loss
            if "loss_history" in checkpoint:
                loss_history = checkpoint["loss_history"]
            if start_epoch < epochs:
                print(f"Resume contrastive training from epoch {start_epoch + 1}/{epochs}")
            else:
                print("Contrastive checkpoint already reached target epochs, training will stop.")
        except Exception as e:
            print(f"Warning: failed to load contrastive checkpoint, start from scratch. Error: {e}")
            start_epoch = 0
            best_loss = float("inf")
            loss_history = []

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
                loss, overall_loss, daily_loss, weekly_loss, consistency_loss = model.compute_loss(p_emb, s_emb, p_multi_scale, s_multi_scale)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(num_batches, 1)
            loss_history.append(avg_loss)

            # Get current batch losses (last batch)
            current_overall = overall_loss.item()
            current_daily = daily_loss.item()
            current_weekly = weekly_loss.item()
            current_consistency = consistency_loss.item()

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

            if (epoch + 1) % save_every == 0 or (epoch + 1) < 20:
                print(f"Epoch [{epoch + 1}/{epochs}], Avg Contrastive Loss: {avg_loss:.4f}")
                print(f"  - Overall Loss: {current_overall:.4f}")
                print(f"  - Daily Loss: {current_daily:.4f}")
                print(f"  - Weekly Loss: {current_weekly:.4f}")
                print(f"  - Consistency Loss: {current_consistency:.4f}")
            
            # Update live visualization.
            if visualize and (epoch + 1) % visualize_interval == 0:
                line.set_data(range(1, len(loss_history) + 1), loss_history)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()
                
                # loss
                if (epoch + 1) % (save_every * 2) == 0:
                    loss_plot_path = os.path.join(os.path.dirname(checkpoint_path), "loss_visualization.png") if os.path.dirname(checkpoint_path) else "loss_visualization.png"
                    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
                    print(f"Loss visualization saved to: {loss_plot_path}")
    except KeyboardInterrupt:
        interrupted_loss = total_loss / max(num_batches, 1) if num_batches > 0 else best_loss
        _save_training_checkpoint(epoch, interrupted_loss)
        print(
            f"\nTraining interrupted. Checkpoint saved at epoch {max(epoch + 1, 0)} "
            f"to {checkpoint_path}"
        )
    finally:
        # loss
        if visualize:
            plt.ioff()  # Disable interactive plotting
            loss_plot_path = os.path.join(os.path.dirname(checkpoint_path), "loss_visualization_final.png") if os.path.dirname(checkpoint_path) else "loss_visualization_final.png"
            plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Final loss visualization saved to: {loss_plot_path}")

    return model



