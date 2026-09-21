"""
Unit Tests for Losses and Scientific Evaluation Metrics.
"""

import pytest
import torch
import numpy as np

from training.losses import FocalLoss, CenterLoss, ProposedCompoundLoss
from training.metrics import calculate_comprehensive_metrics


def test_focal_loss():
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.randn(8, 15)
    targets = torch.randint(0, 15, (8,))
    val = loss_fn(logits, targets)
    assert val.item() > 0.0
    assert not torch.isnan(val).any()


def test_center_loss():
    loss_fn = CenterLoss(num_classes=15, feat_dim=128)
    features = torch.randn(8, 128)
    targets = torch.randint(0, 15, (8,))
    val = loss_fn(features, targets)
    assert val.item() >= 0.0


def test_compound_loss():
    compound = ProposedCompoundLoss(num_classes=15, feat_dim=128)
    fast_l = torch.randn(8, 15)
    deep_l = torch.randn(8, 15)
    latent = torch.randn(8, 128)
    targets = torch.randint(0, 15, (8,))
    tot, focal_val, center_val, supcon_val = compound(fast_l, deep_l, latent, targets)
    assert tot.item() > 0.0


def test_comprehensive_metrics_calculation():
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 2, 0, 2, 2, 0, 1])
    class_names = ["ClassA", "ClassB", "ClassC"]

    m = calculate_comprehensive_metrics(y_true, y_pred, class_names=class_names)
    assert "Accuracy" in m
    assert "Macro-F1" in m or "F1_Macro" in m
    assert m["Accuracy"] > 0.5
    assert len(m["Per_Class"]) == 3
    assert "Confusion_Matrix" in m
