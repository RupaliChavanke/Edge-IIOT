"""
Pointwise Low-Rank Linear Projection Layer.
Decomposes high-dimensional dense representations into low-rank factorized projections.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    """
    Factorized Low-Rank Linear Projection:
    W ~ U * V^T, where U in R^{in x rank}, V in R^{rank x out}.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 16, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)

        self.u_proj = nn.Linear(in_features, self.rank, bias=False)
        self.v_proj = nn.Linear(self.rank, out_features, bias=bias)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.u_proj(x)
        h = self.act(h)
        return self.v_proj(h)

    def get_parameter_comparison(self) -> Dict[str, float]:
        """Calculates parameters saved vs full rank projection."""
        full_params = self.in_features * self.out_features + self.out_features
        low_params = (self.in_features * self.rank) + (self.rank * self.out_features) + self.out_features
        reduction_pct = ((full_params - low_params) / full_params) * 100.0
        return {
            "full_rank_parameters": full_params,
            "low_rank_parameters": low_params,
            "reduction_percentage": reduction_pct,
            "rank": self.rank
        }
