"""
Feature & Label Encoders for Edge-IIoTset.
Dynamically discovers classes and encodes categoricals.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


class EdgeIIoTEncoder:
    """Encodes categorical features and dynamic multiclass targets."""

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.feature_encoders: Dict[str, Dict[str, int]] = {}
        self.classes_: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.class_weights: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, categorical_cols: List[str], target_col: str = "Attack_type") -> "EdgeIIoTEncoder":
        """Fit encoders dynamically on training dataframe."""
        # 1. Fit Target Label Encoder
        if target_col in df.columns:
            targets = df[target_col].astype(str).values
            self.label_encoder.fit(targets)
            self.classes_ = list(self.label_encoder.classes_)
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes_)}
            self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.classes_)}

            # Compute balanced class weights: n_samples / (n_classes * count_c)
            counts = pd.Series(targets).value_counts()
            total_samples = len(targets)
            num_classes = len(self.classes_)
            weights = []
            for cls_name in self.classes_:
                cnt = counts.get(cls_name, 1)
                w = total_samples / (num_classes * cnt)
                weights.append(w)
            # Normalize so mean weight is 1.0
            weights_arr = np.array(weights, dtype=np.float32)
            self.class_weights = weights_arr / np.mean(weights_arr)
            logger.info(f"Discovered {num_classes} attack classes: {self.classes_}")

        # 2. Fit Categorical Feature Encoders (Frequency / Ordinal mapping)
        for col in categorical_cols:
            if col in df.columns:
                unique_vals = df[col].astype(str).unique()
                mapping = {val: idx + 1 for idx, val in enumerate(unique_vals)}
                mapping["unknown"] = 0
                self.feature_encoders[col] = mapping

        self.is_fitted = True
        return self

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical feature columns into numeric integers."""
        if not self.is_fitted:
            raise ValueError("EdgeIIoTEncoder must be fitted before transform_features().")

        out = df.copy()
        for col, mapping in self.feature_encoders.items():
            if col in out.columns:
                out[col] = out[col].astype(str).map(mapping).fillna(0).astype(np.float32)
        return out

    def transform_target(self, target_series: pd.Series) -> np.ndarray:
        """Encode string attack types to integer labels."""
        if not self.is_fitted:
            raise ValueError("EdgeIIoTEncoder must be fitted before transform_target().")
        return self.label_encoder.transform(target_series.astype(str).values)

    def inverse_transform_target(self, label_indices: np.ndarray) -> List[str]:
        """Convert integer predictions back to attack type strings."""
        if not self.is_fitted:
            raise ValueError("EdgeIIoTEncoder must be fitted before inverse_transform_target().")
        return list(self.label_encoder.inverse_transform(label_indices))

    def encode_single_feature_dict(self, feature_dict: Dict[str, any]) -> Dict[str, float]:
        """Encode categoricals in a single streaming event dictionary."""
        res = dict(feature_dict)
        for col, mapping in self.feature_encoders.items():
            if col in res:
                str_val = str(res[col])
                res[col] = float(mapping.get(str_val, 0))
        return res
