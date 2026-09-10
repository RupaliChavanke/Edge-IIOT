"""
Unit Tests for Preprocessing and Feature Selection Modules.
"""

import pytest
import numpy as np
import pandas as pd

from preprocessing.cleaner import EdgeIIoTCleaner
from preprocessing.encoder import EdgeIIoTEncoder
from preprocessing.scaler import EdgeIIoTScaler
from preprocessing.mrmr_jmi import MRMRJMISelector


def test_cleaner_imputation():
    df = pd.DataFrame({
        "frame.time": ["1.0", "2.0"],
        "tcp.dstport": [80.0, np.nan],
        "tcp.srcport": ["443", "inf"],
        "http.request.method": ["GET", None],
        "Attack_type": ["Normal", "DDoS"]
    })
    cleaner = EdgeIIoTCleaner(drop_metadata=True)
    cleaner.fit(df)
    transformed = cleaner.transform(df)

    assert "frame.time" not in transformed.columns
    assert "tcp.dstport" in transformed.columns
    assert not transformed["tcp.dstport"].isnull().any()
    assert not np.isinf(transformed["tcp.srcport"]).any()


def test_encoder_and_target_discovery():
    df = pd.DataFrame({
        "proto": ["tcp", "udp", "tcp"],
        "Attack_type": ["Normal", "DDoS_UDP", "Normal"]
    })
    encoder = EdgeIIoTEncoder()
    encoder.fit(df, categorical_cols=["proto"], target_col="Attack_type")

    assert set(encoder.classes_) == {"DDoS_UDP", "Normal"}
    y_enc = encoder.transform_target(df["Attack_type"])
    assert len(y_enc) == 3
    assert encoder.class_weights is not None


def test_scaler_and_outlier_clamping():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [1000.0, -500.0]], dtype=np.float32)
    scaler = EdgeIIoTScaler(method="robust", clip_outliers=True)
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    assert X_scaled.shape == (3, 2)
    assert not np.isnan(X_scaled).any()
    assert not np.isinf(X_scaled).any()


def test_mrmr_jmi_selection():
    np.random.seed(42)
    X = np.random.randn(100, 10)
    # Make feature 2 highly predictive
    y = (X[:, 2] > 0).astype(int)

    selector = MRMRJMISelector(k_features=4, random_state=42)
    selector.fit(X, y)

    assert len(selector.selected_features_) == 4
    assert selector.ranking_df_ is not None
    assert len(selector.ranking_df_) == 10
    X_proj = selector.transform(X)
    assert X_proj.shape == (100, 4)
