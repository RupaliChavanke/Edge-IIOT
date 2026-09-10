"""
Unit Tests for Proposed Deep Learning Model Components.
"""

import pytest
import torch
from models.cnn import DepthwiseSeparableConv1d
from models.ghost_module import GhostModule1d
from models.se_attention import SEAttention1d
from models.bigru import TemporalBiGRU
from models.temporal_attention import MultiHeadTemporalAttention
from models.low_rank import LowRankLinear
from models.proposed_model import ProposedHybridEdgeIIoTModel


def test_depthwise_separable_conv1d():
    conv = DepthwiseSeparableConv1d(in_channels=1, out_channels=32, kernel_size=3)
    x = torch.randn(4, 1, 22)
    out = conv(x)
    assert out.shape == (4, 32, 22)


def test_ghost_module_1d():
    ghost = GhostModule1d(in_channels=32, out_channels=64, ratio=2)
    x = torch.randn(4, 32, 22)
    out = ghost(x)
    assert out.shape == (4, 64, 22)


def test_se_attention_1d():
    se = SEAttention1d(channels=64, reduction=8)
    x = torch.randn(4, 64, 11)
    scaled, weights = se(x)
    assert scaled.shape == (4, 64, 11)
    assert weights.shape == (4, 64)
    assert (weights >= 0.0).all() and (weights <= 1.0).all()


def test_temporal_bigru():
    bigru = TemporalBiGRU(input_dim=64, hidden_dim=64)
    x = torch.randn(4, 11, 64)
    out, h_n = bigru(x)
    assert out.shape == (4, 11, 128)  # 64 * 2 directions


def test_multihead_temporal_attention():
    mha = MultiHeadTemporalAttention(embed_dim=128, num_heads=4)
    x = torch.randn(4, 11, 128)
    out, attn_map = mha(x)
    assert out.shape == (4, 11, 128)
    assert attn_map.shape == (4, 11, 11)


def test_low_rank_projection():
    lr_layer = LowRankLinear(in_features=1408, out_features=128, rank=16)
    x = torch.randn(4, 1408)
    out = lr_layer(x)
    assert out.shape == (4, 128)
    comp = lr_layer.get_parameter_comparison()
    assert comp["reduction_percentage"] > 80.0


def test_proposed_hybrid_model_modes():
    model = ProposedHybridEdgeIIoTModel(input_dim=22, num_classes=15)
    x = torch.randn(6, 22)

    # Train mode
    train_out = model(x, routing_mode="train")
    assert "fast_logits" in train_out and "deep_logits" in train_out
    assert train_out["fast_logits"].shape == (6, 15)
    assert train_out["deep_logits"].shape == (6, 15)

    # Dynamic inference mode
    inf_out = model(x, routing_mode="dynamic")
    assert "predictions" in inf_out and "entropy" in inf_out
    assert len(inf_out["predictions"]) == 6
    assert len(inf_out["path"]) == 6
