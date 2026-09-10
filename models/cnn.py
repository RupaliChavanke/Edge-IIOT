"""
Depthwise Separable 1D-CNN Module for Edge-IIoT IDS.
Drastically reduces parameters and FLOPs compared to standard Conv1D.
"""

import torch
import torch.nn as nn
from typing import Tuple


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise Separable 1D Convolution:
    1. Depthwise Conv1d (groups = in_channels)
    2. Pointwise Conv1d (kernel_size = 1)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False
    ):
        super().__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias
        )
        # Pointwise convolution
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        return self.act(out)
