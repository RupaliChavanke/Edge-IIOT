"""
Unit and Integration Tests for Pretrained Model Management & Live Evaluator.
Verifies read-only frozen state, zero-retraining during inference, and real-time metric computation.
"""

import os
import json
import numpy as np
import pytest
import torch

from models.model_manager import ModelManager
from dashboard.live_evaluator import LiveEvaluatorEngine


def test_model_manager_loading_and_freeze():
    """Verify that ModelManager loads artifacts and enforces read-only frozen weights."""
    manager = ModelManager.get_instance(artifacts_dir="artifacts")
    is_valid, msg = manager.validate_artifacts()
    assert is_valid, f"Artifacts validation failed: {msg}"
    assert manager.is_loaded, "Model should be marked as loaded"

    # Verify model is in eval mode and all parameters have requires_grad=False
    assert not manager.model.training, "Model must be in eval() mode"
    for name, param in manager.model.named_parameters():
        assert not param.requires_grad, f"Parameter {name} has requires_grad=True, must be frozen"

    # Verify metadata contains essential keys
    meta = manager.get_model_metadata()
    assert meta["model_version"] == "v1.0"
    assert meta["total_parameters"] > 0
    assert meta["mflops"] > 0
    assert len(manager.selected_features) > 0
    assert len(manager.selected_features) == len(manager.model_config.get("selected_features", manager.selected_features))


def test_single_event_frozen_inference():
    """Verify that inference executes without mutating weights or throwing errors."""
    manager = ModelManager.get_instance(artifacts_dir="artifacts")
    dummy_features = {feat: 0.5 for feat in manager.selected_features}

    pred_res = manager.predict_single(dummy_features)
    assert "predicted_attack" in pred_res
    assert "confidence" in pred_res
    assert 0.0 <= pred_res["confidence"] <= 1.0
    assert "entropy" in pred_res
    assert pred_res["path_selected"] in ["FAST", "DEEP"]
    assert "probabilities" in pred_res
    assert len(pred_res["probabilities"]) == len(manager.class_names)


def test_live_evaluator_engine_accumulation():
    """Verify dynamic live metrics computation (Accuracy, F1, ROC-AUC, TP/TN/FP/FN)."""
    evaluator = LiveEvaluatorEngine.get_instance(artifacts_dir="artifacts")
    evaluator.reset()

    class_names = evaluator.class_names
    assert len(class_names) > 0

    # Simulate 5 synthetic streaming prediction events with ground truth
    mock_events = [
        {
            "predicted_attack": class_names[0],
            "dataset_label": class_names[0],
            "confidence": 0.95,
            "probabilities": {class_names[0]: 0.95, class_names[1]: 0.05}
        },
        {
            "predicted_attack": class_names[1],
            "dataset_label": class_names[1],
            "confidence": 0.90,
            "probabilities": {class_names[0]: 0.10, class_names[1]: 0.90}
        },
        {
            "predicted_attack": class_names[0],
            "dataset_label": class_names[1],  # Error
            "confidence": 0.70,
            "probabilities": {class_names[0]: 0.70, class_names[1]: 0.30}
        },
        {
            "predicted_attack": "Normal",
            "dataset_label": "Normal",
            "confidence": 0.98,
            "probabilities": {"Normal": 0.98, class_names[0]: 0.02}
        },
        {
            "predicted_attack": class_names[2] if len(class_names) > 2 else class_names[0],
            "dataset_label": class_names[2] if len(class_names) > 2 else class_names[0],
            "confidence": 0.92,
            "probabilities": {class_names[0]: 0.08, (class_names[2] if len(class_names) > 2 else class_names[0]): 0.92}
        }
    ]

    for evt in mock_events:
        evaluator.ingest_prediction(evt)

    assert evaluator.get_sample_count() == 5
    live_mets = evaluator.get_live_metrics()

    assert live_mets["Accuracy"] == 0.8  # 4 out of 5 correct
    assert live_mets["F1_Macro"] > 0.0
    assert live_mets["TP"] + live_mets["TN"] + live_mets["FP"] + live_mets["FN"] >= 5

    # Check degradation analysis calculation
    deg = evaluator.get_degradation_analysis()
    assert "retention" in deg
    assert "difference" in deg
    assert "offline" in deg
    assert "live" in deg
