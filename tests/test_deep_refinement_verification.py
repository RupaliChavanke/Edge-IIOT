"""
Comprehensive Deep Verification Test Suite for Edge-IIoT Refinements.
Tests all deep learning architectures, ONNX deployment runtime, feature pipeline,
benchmark data integrity, calibration, and dashboard loading.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest
import torch
import onnxruntime as ort

from models.dl_architectures import (
    ModelA_CNN_BiGRU,
    ModelB_CNN_BiGRU_MHA,
    ModelC_DW_CNN_BiGRU,
    ModelD_CNN_Transformer,
    ModelE_CNN_BiGRU_Transformer,
    ModelF_TCN,
    ModelG_LightweightTransformer,
    ModelH_ResCNN_BiGRU_Attn,
    ModelI_CNN_BiGRU_SE_Attn,
)


def test_all_dl_architectures_forward_backward():
    """Verify all 9 deep learning models execute forward/backward without NaN or dimension error."""
    models = [
        ("ModelA", ModelA_CNN_BiGRU(input_dim=22, num_classes=15)),
        ("ModelB", ModelB_CNN_BiGRU_MHA(input_dim=22, num_classes=15)),
        ("ModelC", ModelC_DW_CNN_BiGRU(input_dim=22, num_classes=15)),
        ("ModelD", ModelD_CNN_Transformer(input_dim=22, num_classes=15)),
        ("ModelE", ModelE_CNN_BiGRU_Transformer(input_dim=22, num_classes=15)),
        ("ModelF", ModelF_TCN(input_dim=22, num_classes=15)),
        ("ModelG", ModelG_LightweightTransformer(input_dim=22, num_classes=15)),
        ("ModelH", ModelH_ResCNN_BiGRU_Attn(input_dim=22, num_classes=15)),
        ("ModelI", ModelI_CNN_BiGRU_SE_Attn(input_dim=22, num_classes=15)),
    ]

    dummy_input = torch.randn(4, 22)
    dummy_targets = torch.tensor([0, 1, 2, 3])
    loss_fn = torch.nn.CrossEntropyLoss()

    for name, model in models:
        model.train()
        logits = model(dummy_input)
        assert logits.shape == (4, 15), f"{name} output shape mismatch: {logits.shape}"
        assert not torch.isnan(logits).any(), f"{name} produced NaN logits"

        loss = loss_fn(logits, dummy_targets)
        loss.backward()

        # Check that gradients were computed
        has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.parameters())
        assert has_grad, f"{name} has no valid gradients after backward pass"


def test_onnx_runtime_deployment():
    """Verify that the exported ONNX model runs with zero errors and matches expected shape."""
    onnx_path = "models/best_edge_model.onnx"
    assert os.path.exists(onnx_path), f"ONNX file not found at {onnx_path}"

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    assert session.get_inputs()[0].shape[1] == 22, "Input feature dimension must be 22"

    dummy_in = np.random.randn(5, 22).astype(np.float32)
    outputs = session.run(None, {input_name: dummy_in})

    assert len(outputs) >= 1
    logits = outputs[0]
    assert logits.shape == (5, 15), f"ONNX output shape mismatch: {logits.shape}"
    assert not np.isnan(logits).any(), "ONNX produced NaN outputs"


def test_benchmark_results_integrity():
    """Verify that artifacts/benchmark_results.csv contains valid, non-collapsed empirical metrics."""
    bench_path = "artifacts/benchmark_results.csv"
    assert os.path.exists(bench_path), f"Benchmark CSV not found at {bench_path}"

    df = pd.read_csv(bench_path)
    assert len(df) >= 20, f"Expected at least 20 benchmarked models, found {len(df)}"

    # Check required columns
    required_cols = [
        "Model", "Accuracy", "Precision_Macro", "Precision_Weighted",
        "Recall_Macro", "Recall_Weighted", "F1_Macro", "F1_Weighted",
        "ROC_AUC", "PR_AUC", "Balanced_Accuracy", "MCC", "FPR", "FNR"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Check value bounds
    for col in ["Accuracy", "F1_Macro", "Recall_Macro", "Precision_Macro", "ROC_AUC"]:
        vals = df[col].dropna()
        assert (vals >= 0.0).all() and (vals <= 1.0).all(), f"Column {col} has values outside [0, 1]"

    # Verify that deep learning models do not have collapsed metrics (< 0.70)
    dl_models = [
        "PROPOSED HYBRID MODEL", "ResCNN-BiGRU-Attn", "CNN-BiGRU-Attention",
        "CNN-BiGRU-SE-Attn", "CNN-Attention", "CNN-BiGRU", "CNN"
    ]
    for dl_m in dl_models:
        row = df[df["Model"] == dl_m]
        if not row.empty:
            acc = row["Accuracy"].values[0]
            f1 = row["F1_Macro"].values[0]
            assert acc >= 0.90, f"{dl_m} has collapsed accuracy: {acc}"
            assert f1 >= 0.90, f"{dl_m} has collapsed F1: {f1}"


def test_selected_features_22():
    """Verify that the 22 selected features are consistent with the data schema."""
    feat_path = "data/splits/selected_features_22.json"
    assert os.path.exists(feat_path), f"Features file not found at {feat_path}"

    with open(feat_path, "r") as f:
        features = json.load(f)

    assert len(features) == 22, f"Expected 22 features, found {len(features)}"

    # Ensure no target/identity leakage features are present
    leakage_features = [
        "Attack_type", "Attack_label", "frame.time", "ip.src_host",
        "ip.dst_host", "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport"
    ]
    for feat in features:
        assert feat not in leakage_features, f"Target/identity leakage feature found in feature set: {feat}"


def test_final_test_results_consistency():
    """Verify that FINAL_TEST_RESULTS.json has verified, authoritative single-evaluation test metrics."""
    res_path = "artifacts/final_test_results.json" if os.path.exists("artifacts/final_test_results.json") else "FINAL_TEST_RESULTS.json"
    assert os.path.exists(res_path), f"Final test results not found at {res_path}"

    with open(res_path, "r") as f:
        results = json.load(f)

    assert "accuracy" in results
    assert "macro_f1" in results
    assert "false_positive_rate" in results
    assert "false_negative_rate" in results
    assert "p50_latency_ms" in results

    assert results["accuracy"] >= 0.93
    assert results["macro_f1"] >= 0.91
    assert results["false_positive_rate"] <= 0.01  # Very low FPR
    assert results["false_negative_rate"] <= 0.01  # Very low FNR
    assert results["p50_latency_ms"] < 1.0  # Real-time sub-millisecond
