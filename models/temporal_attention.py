"""
Multi-Head Temporal Self-Attention Module for Edge-IIoT IDS.
Enables cross-timestep temporal interaction and outputs attention heatmaps for explainability.
"""

from typing import Tuple
import math
import torch
import torch.nn as nn


class MultiHeadTemporalAttention(nn.Module):
    """
    Multi-Head Temporal Self-Attention:
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, Embed_Dim)
        Returns:
            Tuple of:
                - Attended output (Batch, Seq_Len, Embed_Dim)
                - Mean attention weight matrix across heads (Batch, Seq_Len, Seq_Len)
        """
        batch_size, seq_len, _ = x.size()
        residual = x

        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        context = torch.matmul(attn_weights_dropped, v)
        # Reshape context back to (Batch, Seq_Len, Embed_Dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(context)

        # Residual connection + LayerNorm
        output = self.layer_norm(output + residual)

        # Average attention weights across all heads for explainability heatmap
        mean_attn_map = attn_weights.mean(dim=1)  # (Batch, Seq_Len, Seq_Len)

        return output, mean_attn_map
