"""
Proposed Model: Mutual Information–Driven Hybrid CNN–Ghost–BiGRU–Attention Network
with Entropy-Based Early Exit Routing for Edge-IIoT Multiclass Intrusion Detection.
"""

from typing import Dict, Optional, Tuple, Union, Any
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


class ProposedHybridEdgeIIoTModel(nn.Module):
    """
    Proposed PhD Hybrid IDS Architecture:
    Depthwise Separable 1D-CNN + Ghost Module + SE Attention + Entropy Early Exit +
    Shared Bi-GRU + Multi-Head Temporal Self-Attention + Pointwise Low-Rank Projection.
    """

    def __init__(
        self,
        input_dim: int = 22,
        num_classes: int = 15,
        conv_channels: int = 64,
        ghost_ratio: int = 2,
        se_reduction: int = 8,
        gru_hidden_dim: int = 64,
        num_attention_heads: int = 4,
        low_rank: int = 16,
        dropout: float = 0.25,
        entropy_threshold: float = 0.35
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.conv_channels = conv_channels
        self.gru_hidden_dim = gru_hidden_dim
        self.entropy_threshold = entropy_threshold

        # 1. Feature Expansion Layer & Tabular Skip Embedding
        self.tabular_embed = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU()
        )

        # 2. Depthwise Separable 1D-CNN
        self.pad_len = (16 - (input_dim % 16)) if (input_dim % 16) != 0 else 0
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

        # 4. Adaptive Pooling & Squeeze-and-Excitation Attention
        # Length 16 guarantees Apple Silicon MPS and CUDA divisibility
        self.seq_len = 16
        self.pool = nn.AdaptiveAvgPool1d(self.seq_len)
        self.se_attention = SEAttention1d(channels=conv_channels, reduction=se_reduction)

        # 5. Fast Path: Early Exit Classifier
        fast_feat_dim = conv_channels * self.seq_len
        self.fast_head = nn.Sequential(
            nn.Linear(fast_feat_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        # 6. Deep Temporal Path: Shared-weight Bi-GRU
        # Treats channel activations as temporal steps: (Batch, seq_len, conv_channels)
        self.bigru = TemporalBiGRU(
            input_dim=conv_channels,
            hidden_dim=gru_hidden_dim,
            num_layers=1,
            dropout=dropout
        )

        # 7. Multi-Head Temporal Self-Attention
        gru_out_dim = gru_hidden_dim * 2  # 128
        self.mha = MultiHeadTemporalAttention(
            embed_dim=gru_out_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )

        # 8. Pointwise Low-Rank Projection Head
        deep_latent_dim = gru_out_dim * self.seq_len  # 128 * 16 = 2048
        self.low_rank_proj = LowRankLinear(
            in_features=deep_latent_dim,
            out_features=128,
            rank=low_rank
        )

        # 9. Deep Multiclass Classifier & Latent Feature Extractor (for Center Loss)
        # Combines Deep Spatial-Temporal representation (128) + Tabular Embed representation (128) = 256
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def calculate_entropy(self, probabilities: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Normalized Shannon Predictive Entropy:
        H(p) = - 1 / log(C) * sum(p_i * log(p_i + eps))
        Returns: Tensor in range [0, 1]
        """
        log_probs = torch.log(probabilities + eps)
        raw_entropy = -torch.sum(probabilities * log_probs, dim=-1)
        max_entropy = math.log(self.num_classes)
        return raw_entropy / max_entropy

    def forward(
        self,
        x: torch.Tensor,
        routing_mode: str = "dynamic",  # dynamic, always_fast, always_deep, or train
        custom_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Forward pass with dynamic early exit routing.
        Args:
            x: Input tensor of shape (Batch, 22)
            routing_mode: 'dynamic' (inference), 'train' (returns both), 'always_fast', 'always_deep'
            custom_threshold: Overrides default self.entropy_threshold if provided
        """
        t0 = time.perf_counter()
        threshold = custom_threshold if custom_threshold is not None else self.entropy_threshold
        batch_size = x.size(0)

        # Tabular representation and spatial preparation
        x_tab = x if x.dim() == 2 else x.squeeze(1)
        tab_feat = self.tabular_embed(x_tab)

        x_pad = F.pad(x_tab, (0, self.pad_len)) if self.pad_len > 0 else x_tab
        x_1d = x_pad.unsqueeze(1)

        # CNN + Ghost feature extraction
        h = self.dw_cnn(x_1d)            # (B, 32, Seq)
        h = self.ghost(h)                 # (B, 64, Seq)
        h = self.pool(h)                  # (B, 64, 16)
        h_se, se_weights = self.se_attention(h)  # (B, 64, 16), (B, 64)

        # Flatten for fast exit head
        fast_feat = h_se.view(batch_size, -1)
        fast_logits = self.fast_head(fast_feat)
        fast_probs = F.softmax(fast_logits, dim=-1)
        entropy = self.calculate_entropy(fast_probs)

        # In training mode, always compute deep path for compound loss calculation
        if routing_mode == "train":
            # Deep Temporal Path
            # Transpose to (B, Seq_Len=16, Feat_Dim=64) for RNN/Attention
            seq_in = h_se.transpose(1, 2)
            gru_out, _ = self.bigru(seq_in)              # (B, 16, 128)
            attn_out, attn_map = self.mha(gru_out)       # (B, 16, 128), (B, 16, 16)
            deep_flatten = attn_out.view(batch_size, -1) # (B, 2048)
            latent_features = self.low_rank_proj(deep_flatten) # (B, 128)
            fused_latent = torch.cat([latent_features, tab_feat], dim=1) # (B, 256)
            deep_logits = self.classifier(fused_latent)
            deep_probs = F.softmax(deep_logits, dim=-1)

            t_elapsed = (time.perf_counter() - t0) * 1000.0  # ms
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
            # Dynamic: Exit fast if entropy is below threshold
            exit_early = entropy < threshold

        # If all samples exit fast
        if exit_early.all():
            t_elapsed = (time.perf_counter() - t0) * 1000.0
            return {
                "logits": fast_logits,
                "probabilities": fast_probs,
                "predictions": torch.argmax(fast_probs, dim=-1),
                "entropy": entropy,
                "path": ["FAST"] * batch_size,
                "se_weights": se_weights,
                "attn_map": None,
                "latency_ms": t_elapsed,
                "early_exit_ratio": 1.0
            }

        # If any samples require Deep Temporal Path
        seq_in = h_se.transpose(1, 2)
        gru_out, _ = self.bigru(seq_in)
        attn_out, attn_map = self.mha(gru_out)
        deep_flatten = attn_out.view(batch_size, -1)
        latent_features = self.low_rank_proj(deep_flatten)
        fused_latent = torch.cat([latent_features, tab_feat], dim=1)
        deep_logits = self.classifier(fused_latent)
        deep_probs = F.softmax(deep_logits, dim=-1)

        # Combine predictions based on routing decision
        final_logits = torch.where(exit_early.unsqueeze(1), fast_logits, deep_logits)
        final_probs = torch.where(exit_early.unsqueeze(1), fast_probs, deep_probs)
        predictions = torch.argmax(final_probs, dim=-1)
        paths = ["FAST" if e.item() else "DEEP" for e in exit_early]

        t_elapsed = (time.perf_counter() - t0) * 1000.0
        early_ratio = float(exit_early.float().mean().item())

        return {
            "logits": final_logits,
            "probabilities": final_probs,
            "predictions": predictions,
            "entropy": entropy,
            "path": paths,
            "se_weights": se_weights,
            "attn_map": attn_map,
            "latency_ms": t_elapsed,
            "early_exit_ratio": early_ratio
        }

    def count_parameters(self) -> Dict[str, int]:
        """Detailed parameter count breakdown."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        cnn_params = sum(p.numel() for p in self.dw_cnn.parameters())
        ghost_params = sum(p.numel() for p in self.ghost.parameters())
        se_params = sum(p.numel() for p in self.se_attention.parameters())
        fast_head_params = sum(p.numel() for p in self.fast_head.parameters())
        bigru_params = sum(p.numel() for p in self.bigru.parameters())
        mha_params = sum(p.numel() for p in self.mha.parameters())
        low_rank_params = sum(p.numel() for p in self.low_rank_proj.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())

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
            "classifier_parameters": classifier_params
        }
