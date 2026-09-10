"""
Squeeze-and-Excitation (SE) 1D Attention Module.
Dynamically recalibrates channel-wise feature responses and yields explainability weights.
"""

from typing import Tuple
import torch
import torch.nn as nn


class SEAttention1d(nn.Module):
    """
    Squeeze-and-Excitation block for 1D representations (Hu et al., CVPR 2018).
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channels = channels
        reduced_channels = max(1, channels // reduction)

        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Tensor of shape (Batch, Channels, Length)
        Returns:
            Tuple of:
                - Recalibrated tensor (Batch, Channels, Length)
                - Channel weights (Batch, Channels)
        """
        b, c, _ = x.size()
        # Squeeze: (B, C, 1) -> (B, C)
        squeezed = self.squeeze(x).view(b, c)
        # Excitation: (B, C)
        weights = self.excitation(squeezed)
        # Scale: (B, C, 1) * (B, C, L)
        scaled = x * weights.view(b, c, 1)
        return scaled, weights
