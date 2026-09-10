"""
Bidirectional GRU Module for Edge-IIoT Temporal Pattern Extraction.
Captures sequential dependencies across packet representations.
"""

from typing import Tuple
import torch
import torch.nn as nn


class TemporalBiGRU(nn.Module):
    """
    Bidirectional GRU with residual projection option.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        batch_first: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim * 2  # Bidirectional

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.layer_norm = nn.LayerNorm(self.output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, Features)
        Returns:
            out: (Batch, Seq_Len, hidden_dim * 2)
            h_n: (num_layers * 2, Batch, hidden_dim)
        """
        out, h_n = self.gru(x)
        out = self.layer_norm(out)
        return out, h_n
