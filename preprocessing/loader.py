"""
Dataset Loader, Splitter, and Preprocessing Orchestrator for Edge-IIoTset.
Supports chunked streaming and stratified train/val/test splits.
"""

from typing import Dict, List, Optional, Tuple
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

from preprocessing.cleaner import EdgeIIoTCleaner
from preprocessing.encoder import EdgeIIoTEncoder
from preprocessing.scaler import EdgeIIoTScaler
from preprocessing.mrmr_jmi import MRMRJMISelector

logger = logging.getLogger(__name__)


class EdgeIIoTDataLoader:
    """Orchestrates loading, cleaning, encoding, scaling, and feature selection."""

    def __init__(
        self,
        raw_path: str = "data/raw/ML-EdgeIIoT-dataset.csv",
        sample_path: str = "data/samples/edge_iiot_sample.csv",
        k_features: int = 22,
        scale_method: str = "robust",
        random_state: int = 42
    ):
        self.raw_path = raw_path
        self.sample_path = sample_path
        self.k_features = k_features
        self.scale_method = scale_method
        self.random_state = random_state

        self.cleaner = EdgeIIoTCleaner(drop_metadata=True)
        self.encoder = EdgeIIoTEncoder()
        self.scaler = EdgeIIoTScaler(method=scale_method)
        self.selector = MRMRJMISelector(k_features=k_features, random_state=random_state)
        self.is_fitted = False

    def load_raw_data(self, use_sample: bool = False, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Load dataset from disk in chunks if large."""
        if (use_sample or not os.path.exists(self.raw_path)) and os.path.exists(self.sample_path):
            path = self.sample_path
        else:
            path = self.raw_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset path not found: {path}")

        logger.info(f"Loading Edge-IIoTset from {path} (max_rows={max_rows})...")
        if max_rows is not None:
            df = pd.read_csv(path, nrows=max_rows, low_memory=False)
        else:
            # Chunked read to minimize memory spikes
            chunks = []
            chunk_size = 50000
            for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)

        logger.info(f"Loaded {len(df)} records, {len(df.columns)} columns.")
        return df

    def fit_transform_pipeline(
        self,
        df: Optional[pd.DataFrame] = None,
        use_sample: bool = False,
        max_rows: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes scientific preprocessing strictly on Train split:
        1. Split into Train (70%), Val (15%), Test (15%)
        2. Fit cleaner, encoder, scaler, and mRMR-JMI ONLY on Train
        3. Transform Val and Test without target leakage
        """
        if df is None:
            df = self.load_raw_data(use_sample=use_sample, max_rows=max_rows)

        # Drop rows missing target
        target_col = "Attack_type"
        if target_col not in df.columns:
            raise ValueError(f"Required target column '{target_col}' not found in dataframe.")

        df = df.dropna(subset=[target_col]).reset_index(drop=True)

        # Exact duplicate detection & removal
        duplicates_before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        duplicates_after = len(df)
        duplicates_removed = duplicates_before - duplicates_after
        logger.info(f"Deduplication: {duplicates_before} raw -> {duplicates_removed} duplicates pruned -> {duplicates_after} clean.")

        # 1. Stratified & Group-Aware Split: Train (70%), Val (15%), Test (15%)
        y_all = df[target_col].astype(str)
        train_df, temp_df = train_test_split(
            df, test_size=0.30, stratify=y_all, random_state=self.random_state
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, stratify=temp_df[target_col].astype(str), random_state=self.random_state
        )

        logger.info(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        # 2. Fit Cleaner on Train
        self.cleaner.fit(train_df)
        train_clean = self.cleaner.transform(train_df)
        val_clean = self.cleaner.transform(val_df)
        test_clean = self.cleaner.transform(test_df)

        # 3. Fit Encoders on Train
        self.encoder.fit(train_clean, categorical_cols=self.cleaner.fitted_categorical_columns, target_col=target_col)
        train_enc = self.encoder.transform_features(train_clean)
        val_enc = self.encoder.transform_features(val_clean)
        test_enc = self.encoder.transform_features(test_clean)

        y_train = self.encoder.transform_target(train_df[target_col])
        y_val = self.encoder.transform_target(val_df[target_col])
        y_test = self.encoder.transform_target(test_df[target_col])

        feature_cols = self.cleaner.fitted_numeric_columns + self.cleaner.fitted_categorical_columns
        X_train_df = train_enc[feature_cols]
        X_val_df = val_enc[feature_cols]
        X_test_df = test_enc[feature_cols]

        # 4. Fit Scaler on Train
        self.scaler.fit(X_train_df)
        X_train_scaled = self.scaler.transform(X_train_df)
        X_val_scaled = self.scaler.transform(X_val_df)
        X_test_scaled = self.scaler.transform(X_test_df)

        # 5. Fit mRMR-JMI Feature Selector on Train
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
        self.selector.fit(X_train_scaled_df, y_train)

        X_train_mrmr = self.selector.transform(X_train_scaled_df)
        X_val_mrmr = self.selector.transform(pd.DataFrame(X_val_scaled, columns=feature_cols))
        X_test_mrmr = self.selector.transform(pd.DataFrame(X_test_scaled, columns=feature_cols))

        self.is_fitted = True

        return {
            "X_train": X_train_mrmr,
            "y_train": y_train,
            "X_val": X_val_mrmr,
            "y_val": y_val,
            "X_test": X_test_mrmr,
            "y_test": y_test,
            "feature_names": self.selector.selected_features_,
            "class_names": self.encoder.classes_,
            "class_weights": self.encoder.class_weights,
            "ranking_df": self.selector.ranking_df_,
            "test_raw_df": test_df,
            "duplicates_stats": {
                "before": duplicates_before,
                "removed": duplicates_removed,
                "after": duplicates_after
            }
        }

    def save_pipeline(self, output_dir: str = "checkpoints") -> None:
        """Save fitted transformers to disk."""
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "cleaner.pkl"), "wb") as f:
            pickle.dump(self.cleaner, f)
        with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
            pickle.dump(self.encoder, f)
        with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)
        with open(os.path.join(output_dir, "feature_selector.pkl"), "wb") as f:
            pickle.dump(self.selector, f)
        logger.info(f"Saved preprocessing pipeline artifacts to {output_dir}")

    def load_pipeline(self, output_dir: str = "checkpoints") -> "EdgeIIoTDataLoader":
        """Load fitted pipeline artifacts from disk."""
        with open(os.path.join(output_dir, "cleaner.pkl"), "rb") as f:
            self.cleaner = pickle.load(f)
        with open(os.path.join(output_dir, "label_encoder.pkl"), "rb") as f:
            self.encoder = pickle.load(f)
        with open(os.path.join(output_dir, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(output_dir, "feature_selector.pkl"), "rb") as f:
            self.selector = pickle.load(f)
        self.is_fitted = True
        logger.info(f"Loaded preprocessing pipeline from {output_dir}")
        return self

    def process_single_streaming_event(self, raw_features: Dict[str, Any]) -> np.ndarray:
        """
        End-to-end online transform for a single incoming event from Redpanda.
        Returns: 1D array of shape (k_features,)
        """
        if not self.is_fitted:
            self.load_pipeline()

        cleaned_dict = self.cleaner.clean_single_event(raw_features)
        encoded_dict = self.encoder.encode_single_feature_dict(cleaned_dict)

        feature_cols = self.cleaner.fitted_numeric_columns + self.cleaner.fitted_categorical_columns
        vec = np.array([float(encoded_dict.get(c, 0.0)) for c in feature_cols], dtype=np.float32)

        scaled_vec = self.scaler.transform_single(vec)
        scaled_dict = {col: scaled_vec[i] for i, col in enumerate(feature_cols)}

        return self.selector.transform_single_dict(scaled_dict)
