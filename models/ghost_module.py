"""
Ghost Module 1D for Edge-IIoT IDS.
Generates intrinsic feature maps + cheap linear ghost maps to reduce FLOPs.
"""

import math
import torch
import torch.nn as nn


class GhostModule1d(nn.Module):
    """
    1D Ghost Module (Han et al., CVPR 2020 adapted to 1D signals).
    Splits output channels into primary features and cheap ghost features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        ratio: int = 2,
        dw_kernel_size: int = 3,
        relu: bool = True
    ):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        # Primary convolution
        self.primary_conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                init_channels,
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm1d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )

        # Cheap depthwise linear operations for ghost features
        self.cheap_operation = nn.Sequential(
            nn.Conv1d(
                init_channels,
                new_channels,
                dw_kernel_size,
                stride=1,
                padding=dw_kernel_size // 2,
                groups=init_channels,
                bias=False
            ),
            nn.BatchNorm1d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        primary = self.primary_conv(x)
        ghost = self.cheap_operation(primary)
        out = torch.cat([primary, ghost], dim=1)
        return out[:, :self.out_channels, :]
