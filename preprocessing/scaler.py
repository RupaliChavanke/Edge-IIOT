"""
Robust and Quantile Rescaling Module for Edge-IIoTset.
Fitted strictly on training data to avoid data leakage.
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, QuantileTransformer
import logging

logger = logging.getLogger(__name__)


class EdgeIIoTScaler:
    """Robust or Quantile scaler with outlier clipping."""

    def __init__(self, method: str = "robust", clip_outliers: bool = True, quantile_range: tuple = (5.0, 95.0)):
        self.method = method
        self.clip_outliers = clip_outliers
        self.quantile_range = quantile_range
        self.feature_names: List[str] = []
        self.clip_mins: np.ndarray = np.array([])
        self.clip_maxs: np.ndarray = np.array([])
        self.is_fitted = False

        if method == "robust":
            self.scaler = RobustScaler(quantile_range=quantile_range, unit_variance=True)
        elif method == "quantile":
            self.scaler = QuantileTransformer(output_distribution="normal", random_state=42)
        else:
            raise ValueError(f"Unknown scaling method: {method}")

    def fit(self, X: Union[pd.DataFrame, np.ndarray], feature_names: Optional[List[str]] = None) -> "EdgeIIoTScaler":
        """Fit scaler on training features."""
        if self.method == "robust":
            self.scaler = RobustScaler(quantile_range=self.quantile_range, unit_variance=True)
        elif self.method == "quantile":
            self.scaler = QuantileTransformer(output_distribution="normal", random_state=42)

        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_arr = X.values.astype(np.float32)
        else:
            self.feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]
            X_arr = np.asarray(X, dtype=np.float32)

        # Replace any infs with finite bounds before fitting
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=1e6, neginf=-1e6)

        self.scaler.fit(X_arr)
        scaled_train = self.scaler.transform(X_arr)

        if self.clip_outliers:
            # Set clip bounds at 0.1th and 99.9th percentiles of scaled train data
            self.clip_mins = np.percentile(scaled_train, 0.1, axis=0) - 5.0
            self.clip_maxs = np.percentile(scaled_train, 99.9, axis=0) + 5.0

        self.is_fitted = True
        logger.info(f"EdgeIIoTScaler ({self.method}) fitted on {len(self.feature_names)} features.")
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform features using fitted parameters."""
        if not self.is_fitted:
            raise ValueError("EdgeIIoTScaler must be fitted before transform().")

        if isinstance(X, pd.DataFrame):
            # Align columns if necessary
            X_arr = X[self.feature_names].values.astype(np.float32) if set(self.feature_names).issubset(X.columns) else X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=1e6, neginf=-1e6)
        scaled = self.scaler.transform(X_arr)

        if self.clip_outliers and len(self.clip_mins) > 0:
            scaled = np.clip(scaled, self.clip_mins, self.clip_maxs)

        # Numerical safety clamp for neural network stability
        scaled = np.clip(scaled, -10.0, 10.0)
        return scaled.astype(np.float32)

    def transform_single(self, feature_vector: np.ndarray) -> np.ndarray:
        """Fast single-vector transform for online streaming inference."""
        arr = feature_vector.reshape(1, -1)
        return self.transform(arr).flatten()
