"""
Proposed Model: Mutual Information–Driven Hybrid Multi-Scale Depthwise CNN–BiGRU–Attention Network
with Squeeze-and-Excitation, Low-Rank Residual Fusion, and Entropy-Gated Early Exit Routing.
Engineered for Real-Time High-Throughput Edge-IIoT Multiclass Intrusion Detection.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cnn import DepthwiseSeparableConv1d
from models.ghost_module import GhostModule1d
from models.se_attention import SEAttention1d
from models.bigru import TemporalBiGRU
from models.temporal_attention import MultiHeadTemporalAttention
from models.low_rank import LowRankLinear


class MultiScaleTemporalConv1d(nn.Module):
    """Multi-Scale Temporal Convolution block with parallel kernel branches (k=3, k=5)."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_ch = out_channels // 2
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_ch),
            nn.GELU()
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels, mid_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(mid_ch),
            nn.GELU()
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(mid_ch * 2, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        cat = torch.cat([b3, b5], dim=1)
        return self.fuse(cat)


class ProposedHybridEdgeIIoTModel(nn.Module):
    """
    Research-Grade Proposed Hybrid IDS Architecture:
    Tabular Embedding -> Residual Depthwise-Separable 1D-CNN -> Ghost Module ->
    Squeeze-and-Excitation -> Multi-Scale Temporal Conv -> Fast Early Exit Head ->
    BiGRU -> Multi-Head Self-Attention -> Low-Rank Projection -> Residual Tabular Fusion ->
    Deep Multiclass Classification Head with Entropy-Based Dynamic Routing.
    """

    def __init__(
        self,
        input_dim: int = 43,
        num_classes: int = 15,
        conv_channels: int = 64,
        ghost_ratio: int = 2,
        se_reduction: int = 8,
        gru_hidden_dim: int = 64,
        num_attention_heads: int = 4,
        low_rank: int = 16,
        dropout: float = 0.20,
        entropy_threshold: float = 0.35
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.conv_channels = conv_channels
        self.gru_hidden_dim = gru_hidden_dim
        self.entropy_threshold = entropy_threshold
        self.temperature = nn.Parameter(torch.ones(1) * 1.0, requires_grad=False)

        # 1. Feature Embedding Projection & Tabular Skip Connection
        self.tabular_embed = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU()
        )

        # 2. Residual Depthwise Separable 1D-CNN
        self.dw_cnn = DepthwiseSeparableConv1d(
            in_channels=1,
            out_channels=conv_channels // 2,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # 3. Ghost Module 1D
        self.ghost = GhostModule1d(
            in_channels=conv_channels // 2,
            out_channels=conv_channels,
            kernel_size=3,
            ratio=ghost_ratio
        )

        # 4. Multi-Scale Temporal Convolutions & Adaptive Pooling
        self.multi_scale = MultiScaleTemporalConv1d(
            in_channels=conv_channels,
            out_channels=conv_channels
        )
        self.seq_len = 16
        self.pool = nn.AdaptiveAvgPool1d(self.seq_len)

        # 5. Squeeze-and-Excitation Channel Attention
        self.se_attention = SEAttention1d(channels=conv_channels, reduction=se_reduction)

        # 6. Fast Path: Intermediate Early-Exit Classifier Head
        fast_feat_dim = conv_channels * self.seq_len
        self.fast_head = nn.Sequential(
            nn.Linear(fast_feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        # 7. Deep Recurrent Path: Shared Bi-GRU
        self.bigru = TemporalBiGRU(
            input_dim=conv_channels,
            hidden_dim=gru_hidden_dim,
            num_layers=1,
            dropout=dropout
        )

        # 8. Multi-Head Temporal Self-Attention
        gru_out_dim = gru_hidden_dim * 2  # 128
        self.mha = MultiHeadTemporalAttention(
            embed_dim=gru_out_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        self.ln_attn = nn.LayerNorm(gru_out_dim)

        # 9. Pointwise Low-Rank Projection Head
        deep_latent_dim = gru_out_dim * self.seq_len  # 128 * 16 = 2048
        self.low_rank_proj = LowRankLinear(
            in_features=deep_latent_dim,
            out_features=128,
            rank=low_rank
        )

        # 10. Deep Multiclass Classifier Head & Latent Representation Fusion
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def calculate_entropy(self, probabilities: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Normalized Shannon Predictive Entropy: H(p) = - 1 / log(C) * sum(p * log(p + eps))."""
        log_probs = torch.log(probabilities + eps)
        raw_entropy = -torch.sum(probabilities * log_probs, dim=-1)
        max_entropy = math.log(self.num_classes)
        return raw_entropy / max_entropy

    def set_temperature(self, temp: float):
        """Sets post-hoc calibration temperature."""
        self.temperature.data.fill_(max(0.01, float(temp)))

    def forward(
        self,
        x: torch.Tensor,
        routing_mode: str = "dynamic",
        custom_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Forward pass with entropy-guided dynamic early-exit routing and uncertainty estimation.
        """
        t0 = time.perf_counter()
        threshold = custom_threshold if custom_threshold is not None else self.entropy_threshold
        batch_size = x.size(0)

        # Tabular representation and spatial sequence preparation
        x_tab = x if x.dim() == 2 else x.squeeze(1)
        tab_feat = self.tabular_embed(x_tab)
        x_1d = x_tab.unsqueeze(1)

        # Spatial Feature Extraction: DW-CNN -> Ghost -> MultiScale -> Pool -> SE Attention
        h = self.dw_cnn(x_1d)
        h = self.ghost(h)
        h = self.multi_scale(h)
        if h.device.type == "mps":
            h = self.pool(h.cpu()).to(h.device)
        else:
            h = self.pool(h)
        h_se, se_weights = self.se_attention(h)

        # Fast Head Logits & Predictive Entropy
        fast_feat = h_se.view(batch_size, -1)
        fast_logits = self.fast_head(fast_feat) / self.temperature
        fast_probs = F.softmax(fast_logits, dim=-1)
        entropy = self.calculate_entropy(fast_probs)

        # Training Mode: compute both branches for compound loss
        if routing_mode == "train":
            seq_in = h_se.transpose(1, 2)
            gru_out, _ = self.bigru(seq_in)
            attn_out, attn_map = self.mha(gru_out)
            gru_attn_fused = self.ln_attn(gru_out + attn_out)
            latent_features = gru_attn_fused.mean(dim=1)
            fused_latent = torch.cat([latent_features, tab_feat], dim=1)
            deep_logits = self.classifier(fused_latent) / self.temperature
            deep_probs = F.softmax(deep_logits, dim=-1)

            t_elapsed = (time.perf_counter() - t0) * 1000.0
            return {
                "fast_logits": fast_logits,
                "fast_probs": fast_probs,
                "deep_logits": deep_logits,
                "deep_probs": deep_probs,
                "latent_features": latent_features,
                "entropy": entropy,
                "se_weights": se_weights,
                "attn_map": attn_map,
                "latency_ms": t_elapsed
            }

        # Dynamic Inference Mode
        if routing_mode == "always_fast":
            exit_early = torch.ones(batch_size, dtype=torch.bool, device=x.device)
        elif routing_mode == "always_deep":
            exit_early = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        else:
            max_conf, _ = torch.max(fast_probs, dim=-1)
            exit_early = (max_conf > 0.95) & (entropy < threshold)

        if exit_early.all():
            t_elapsed = (time.perf_counter() - t0) * 1000.0
            max_conf, preds = torch.max(fast_probs, dim=-1)
            return {
                "logits": fast_logits,
                "probabilities": fast_probs,
                "predictions": preds,
                "confidence": max_conf,
                "entropy": entropy,
                "path": ["FAST"] * batch_size,
                "se_weights": se_weights,
                "attn_map": None,
                "latency_ms": t_elapsed,
                "early_exit_ratio": 1.0
            }

        # Deep Branch Execution
        seq_in = h_se.transpose(1, 2)
        gru_out, _ = self.bigru(seq_in)
        attn_out, attn_map = self.mha(gru_out)
        gru_attn_fused = self.ln_attn(gru_out + attn_out)
        latent_features = gru_attn_fused.mean(dim=1)
        fused_latent = torch.cat([latent_features, tab_feat], dim=1)
        deep_logits = self.classifier(fused_latent) / self.temperature
        deep_probs = F.softmax(deep_logits, dim=-1)

        final_logits = torch.where(exit_early.unsqueeze(1), fast_logits, deep_logits)
        final_probs = torch.where(exit_early.unsqueeze(1), fast_probs, deep_probs)
        max_conf, preds = torch.max(final_probs, dim=-1)
        paths = ["FAST" if e.item() else "DEEP" for e in exit_early]

        t_elapsed = (time.perf_counter() - t0) * 1000.0
        early_ratio = float(exit_early.float().mean().item())

        return {
            "logits": final_logits,
            "probabilities": final_probs,
            "predictions": preds,
            "confidence": max_conf,
            "entropy": entropy,
            "path": paths,
            "se_weights": se_weights,
            "attn_map": attn_map,
            "latency_ms": t_elapsed,
            "early_exit_ratio": early_ratio
        }

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        cnn_params = sum(p.numel() for p in self.dw_cnn.parameters()) + sum(p.numel() for p in self.multi_scale.parameters())
        ghost_params = sum(p.numel() for p in self.ghost.parameters())
        se_params = sum(p.numel() for p in self.se_attention.parameters())
        fast_head_params = sum(p.numel() for p in self.fast_head.parameters())
        bigru_params = sum(p.numel() for p in self.bigru.parameters())
        mha_params = sum(p.numel() for p in self.mha.parameters())
        low_rank_params = sum(p.numel() for p in self.low_rank_proj.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        tab_params = sum(p.numel() for p in self.tabular_embed.parameters())

        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "cnn_parameters": cnn_params,
            "ghost_parameters": ghost_params,
            "se_parameters": se_params,
            "fast_head_parameters": fast_head_params,
            "bigru_parameters": bigru_params,
            "temporal_attention_parameters": mha_params,
            "low_rank_parameters": low_rank_params,
            "classifier_parameters": classifier_params,
            "tabular_parameters": tab_params
        }
