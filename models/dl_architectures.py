"""
Phase 7 / Stage 7-8: Optimized Comparative Deep Learning Architectures for Edge-IIoTset Intrusion Detection.
Standardized PyTorch implementations with Apple Silicon MPS & CPU compatibility,
stabilized with LayerNorm and Tabular Skip Projections to prevent BatchNorm mode collapse:
- Model A: Standard CNN-BiGRU
- Model B: CNN + BiGRU + Multi-Head Attention
- Model C: Depthwise Separable CNN + BiGRU
- Model D: CNN + Transformer Encoder
- Model E: CNN + BiGRU + Transformer
- Model F: Temporal Convolutional Network (TCN)
- Model G: Lightweight Transformer (Tabular Transformer)
- Model H: Residual CNN + BiGRU + Attention
- Model I: CNN + BiGRU + SE + Attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cnn import DepthwiseSeparableConv1d
from models.se_attention import SEAttention1d
from models.bigru import TemporalBiGRU
from models.temporal_attention import MultiHeadTemporalAttention


def safe_pool1d(x: torch.Tensor, target_len: int = 11) -> torch.Tensor:
    """Safe 1D adaptive pooling handling Apple Silicon MPS & ONNX non-divisible size limitation."""
    if x.device.type == "mps":
        return F.adaptive_avg_pool1d(x.cpu(), target_len).to(x.device)
    return F.adaptive_avg_pool1d(x, target_len)


# Model A: Standard CNN-BiGRU
class ModelA_CNN_BiGRU(nn.Module):
    def __init__(self, input_dim: int = 22, num_classes: int = 15):
        super().__init__()
        self.tab = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU()
        )
        self.bigru = nn.GRU(32, 32, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(64 + 64, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = safe_pool1d(self.conv(x.unsqueeze(1)), 11).transpose(1, 2)
        out, _ = self.bigru(h)
        seq_feat = out.mean(dim=1)
        tab_feat = self.tab(x)
        return self.fc(torch.cat([seq_feat, tab_feat], dim=-1))


# Model B: CNN + BiGRU + Multi-Head Attention
class ModelB_CNN_BiGRU_MHA(nn.Module):
    def __init__(self, input_dim: int = 22, num_classes: int = 15):
        super().__init__()
        self.tab = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU()
        )
        self.bigru = nn.GRU(32, 32, batch_first=True, bidirectional=True)
        self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm(64)
        self.fc = nn.Sequential(
            nn.Linear(64 + 64, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = safe_pool1d(self.conv(x.unsqueeze(1)), 11).transpose(1, 2)
        gru_out, _ = self.bigru(h)
        attn_out, _ = self.mha(gru_out, gru_out, gru_out)
        seq_feat = self.ln(gru_out + attn_out).mean(dim=1)
        tab_feat = self.tab(x)
        return self.fc(torch.cat([seq_feat, tab_feat], dim=-1))


# Model C: Depthwise Separable CNN + BiGRU
class ModelC_DW_CNN_BiGRU(nn.Module):
    def __init__(self, input_dim: int = 22, num_classes: int = 15):
        super().__init__()
        self.tab = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.dw_cnn = DepthwiseSeparableConv1d(1, 32, kernel_size=3, padding=1)
        self.bigru = nn.GRU(32, 32, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(64 + 64, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = safe_pool1d(self.dw_cnn(x.unsqueeze(1)), 11).transpose(1, 2)
        out, _ = self.bigru(h)
        seq_feat = out.mean(dim=1)
        tab_feat = self.tab(x)
        return self.fc(torch.cat([seq_feat, tab_feat], dim=-1))


# Model D: CNN + Transformer Encoder
class ModelD_CNN_Transformer(nn.Module):
    def __init__(self, input_dim: int = 22, num_classes: int = 15):
        super().__init__()
        self.tab = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU()
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Sequential(
            nn.Linear(32 + 64, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = safe_pool1d(self.conv(x.unsqueeze(1)), 11).transpose(1, 2)
        out = self.transformer(h)
        seq_feat = out.mean(dim=1)
        tab_feat = self.tab(x)
        return self.fc(torch.cat([seq_feat, tab_feat], dim=-1))


# Model E: CNN + BiGRU + Transformer
class ModelE_CNN_BiGRU_Transformer(nn.Module):
    def __init__(self, input_dim: int = 22, num_classes: int = 15):
        super().__init__()
        self.tab = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU()
        )
        self.bigru = nn.GRU(32, 32, batch_first=True, bidirectional=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Sequential(
            nn.Linear(64 + 64, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = safe_pool1d(self.conv(x.unsqueeze(1)), 11).transpose(1, 2)
        gru_out, _ = self.bigru(h)
        out = self.transformer(gru_out)
        seq_feat = out.mean(dim=1)
        tab_feat = self.tab(x)
        return self.fc(torch.cat([seq_feat, tab_feat], dim=-1))


# Model F: Temporal Convolutional Network (TCN)
class ModelF_TCN(nn.Module):
    def __init__(self, input_dim: int = 22, num_classes: int = 15):
        super().__init__()
        self.tab = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.in_proj = nn.Conv1d(1, 32, kernel_size=1)
        self.conv1 = nn.Conv1d(32, 32, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=4, dilation=4)
        self.fc = nn.Sequential(
            nn.Linear(32 + 64, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = F.gelu(self.in_proj(x.unsqueeze(1)))
        h1 = F.gelu(self.conv1(h0)) + h0
        h2 = F.gelu(self.conv2(h1)) + h1
        h3 = F.gelu(self.conv3(h2)) + h2
        seq_feat = safe_pool1d(h3, 1).squeeze(-1)
        tab_feat = self.tab(x)
        return self.fc(torch.cat([seq_feat, tab_feat], dim=-1))


# Model G: Lightweight Tabular Transformer
class ModelG_LightweightTransformer(nn.Module):
    def __init__(self, input_dim: int = 22, num_classes: int = 15):
        super().__init__()
        self.embed = nn.Linear(input_dim, 64)
        self.pos_emb = nn.Parameter(torch.randn(1, 11, 64) * 0.02)
        self.proj = nn.Linear(64, 11 * 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Sequential(
            nn.Linear(64 + 64, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        emb = F.gelu(self.embed(x))
        tokens = self.proj(emb).view(b, 11, 64) + self.pos_emb
        out = self.transformer(tokens)
        seq_feat = out.mean(dim=1)
        return self.fc(torch.cat([seq_feat, emb], dim=-1))


# Model H: Residual CNN + BiGRU + Attention
class ModelH_ResCNN_BiGRU_Attn(nn.Module):
    def __init__(self, input_dim: int = 22, num_classes: int = 15):
        super().__init__()
        self.tab = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.bigru = nn.GRU(32, 32, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm(64)
        self.fc = nn.Sequential(
            nn.Linear(64 + 64, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x.unsqueeze(1)
        h = F.gelu(self.conv1(x_in))
        h = F.gelu(self.conv2(h)) + h
        seq = safe_pool1d(h, 11).transpose(1, 2)
        gru_out, _ = self.bigru(seq)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        seq_feat = self.ln(gru_out + attn_out).mean(dim=1)
        tab_feat = self.tab(x)
        return self.fc(torch.cat([seq_feat, tab_feat], dim=-1))


# Model I: CNN + BiGRU + SE + Attention
class ModelI_CNN_BiGRU_SE_Attn(nn.Module):
    def __init__(self, input_dim: int = 22, num_classes: int = 15):
        super().__init__()
        self.tab = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU()
        )
        self.se = SEAttention1d(channels=32, reduction=8)
        self.bigru = nn.GRU(32, 32, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm(64)
        self.fc = nn.Sequential(
            nn.Linear(64 + 64, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.unsqueeze(1))
        h_se, _ = self.se(h)
        seq = safe_pool1d(h_se, 11).transpose(1, 2)
        gru_out, _ = self.bigru(seq)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        seq_feat = self.ln(gru_out + attn_out).mean(dim=1)
        tab_feat = self.tab(x)
        return self.fc(torch.cat([seq_feat, tab_feat], dim=-1))


def get_deep_learning_models(input_dim: int = 22, num_classes: int = 15):
    return {
        "CNN-BiGRU": ModelA_CNN_BiGRU(input_dim, num_classes),
        "CNN-BiGRU-MHA": ModelB_CNN_BiGRU_MHA(input_dim, num_classes),
        "DW-CNN-BiGRU": ModelC_DW_CNN_BiGRU(input_dim, num_classes),
        "CNN-Transformer": ModelD_CNN_Transformer(input_dim, num_classes),
        "CNN-BiGRU-Transformer": ModelE_CNN_BiGRU_Transformer(input_dim, num_classes),
        "TCN": ModelF_TCN(input_dim, num_classes),
        "Lightweight-Transformer": ModelG_LightweightTransformer(input_dim, num_classes),
        "ResCNN-BiGRU-Attn": ModelH_ResCNN_BiGRU_Attn(input_dim, num_classes),
        "CNN-BiGRU-SE-Attn": ModelI_CNN_BiGRU_SE_Attn(input_dim, num_classes)
    }
