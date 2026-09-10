"""
mRMR-JMI (Minimum Redundancy Maximum Relevance - Joint Mutual Information)
Feature Selection Engine for Edge-IIoTset.
Fitted strictly on training data.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import logging

logger = logging.getLogger(__name__)


class MRMRJMISelector:
    """
    Implements Joint Mutual Information (JMI) and mRMR feature selection.
    
    Selects top-k features maximizing mutual information with target Y
    while minimizing joint redundancy with already selected features.
    """

    def __init__(self, k_features: int = 22, sample_size_for_mi: int = 15000, random_state: int = 42):
        self.k_features = k_features
        self.sample_size_for_mi = sample_size_for_mi
        self.random_state = random_state
        self.selected_features_: List[str] = []
        self.ranking_df_: Optional[pd.DataFrame] = None
        self.all_feature_names_: List[str] = []
        self.is_fitted = False

    @property
    def selected_features(self) -> List[str]:
        return self.selected_features_

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> "MRMRJMISelector":
        """
        Fit mRMR-JMI feature selection on training data.
        """
        if isinstance(X, pd.DataFrame):
            self.all_feature_names_ = list(X.columns)
            X_arr = X.values.astype(np.float32)
        else:
            self.all_feature_names_ = feature_names or [f"feat_{i}" for i in range(X.shape[1])]
            X_arr = np.asarray(X, dtype=np.float32)

        y_arr = np.asarray(y)

        num_samples, num_features = X_arr.shape
        k = min(self.k_features, num_features)

        # Subsample if dataset is very large for computational efficiency of pairwise MI
        if num_samples > self.sample_size_for_mi:
            np.random.seed(self.random_state)
            sub_idx = np.random.choice(num_samples, self.sample_size_for_mi, replace=False)
            X_sub = X_arr[sub_idx]
            y_sub = y_arr[sub_idx]
        else:
            X_sub = X_arr
            y_sub = y_arr

        logger.info(f"Computing Mutual Information for {num_features} candidate features on {len(X_sub)} samples...")

        # 1. Compute Relevance: I(X_i; Y)
        mi_scores = mutual_info_classif(
            X_sub, y_sub, random_state=self.random_state, n_neighbors=3
        )
        # Avoid zero or negative due to numerical noise
        mi_scores = np.maximum(mi_scores, 1e-6)

        # Compute feature-feature correlation matrix for redundancy estimation
        corr_matrix = np.abs(np.corrcoef(X_sub, rowvar=False))
        # Handle NaNs in correlation if feature is constant
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # 2. Iterative greedy JMI selection
        selected_indices: List[int] = []
        remaining_indices = list(range(num_features))
        ranking_records = []

        # Step 1: Select feature with highest MI
        first_idx = int(np.argmax(mi_scores))
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        ranking_records.append({
            "Feature": self.all_feature_names_[first_idx],
            "MI": float(mi_scores[first_idx]),
            "Redundancy": 0.0,
            "JMI": float(mi_scores[first_idx]),
            "Rank": 1,
            "Selected": True,
            "Feature_Index": first_idx
        })

        # Step 2: Iteratively select remaining features
        for step in range(2, k + 1):
            best_score = -np.inf
            best_cand = None
            best_redundancy = 0.0

            for cand_idx in remaining_indices:
                cand_mi = mi_scores[cand_idx]
                # Redundancy: average correlation with currently selected features
                cand_redundancy = np.mean([corr_matrix[cand_idx, s] for s in selected_indices])
                # JMI criterion: Relevance - Redundancy (weighted by MI)
                cand_jmi = cand_mi - 0.5 * cand_redundancy * cand_mi

                if cand_jmi > best_score:
                    best_score = cand_jmi
                    best_cand = cand_idx
                    best_redundancy = cand_redundancy

            if best_cand is not None:
                selected_indices.append(best_cand)
                remaining_indices.remove(best_cand)
                ranking_records.append({
                    "Feature": self.all_feature_names_[best_cand],
                    "MI": float(mi_scores[best_cand]),
                    "Redundancy": float(best_redundancy),
                    "JMI": float(best_score),
                    "Rank": step,
                    "Selected": True,
                    "Feature_Index": best_cand
                })

        # Add remaining unselected features ranked by their individual MI
        remaining_records = []
        for rank_offset, cand_idx in enumerate(remaining_indices):
            cand_mi = mi_scores[cand_idx]
            cand_redundancy = float(np.mean([corr_matrix[cand_idx, s] for s in selected_indices]))
            cand_jmi = cand_mi - 0.5 * cand_redundancy * cand_mi
            remaining_records.append({
                "Feature": self.all_feature_names_[cand_idx],
                "MI": float(cand_mi),
                "Redundancy": cand_redundancy,
                "JMI": float(cand_jmi),
                "Rank": k + 1 + rank_offset,
                "Selected": False,
                "Feature_Index": cand_idx
            })

        # Sort remaining by JMI descending
        remaining_records.sort(key=lambda x: x["JMI"], reverse=True)
        for idx, rec in enumerate(remaining_records):
            rec["Rank"] = k + 1 + idx

        all_records = ranking_records + remaining_records
        self.ranking_df_ = pd.DataFrame(all_records)
        self.selected_features_ = [self.all_feature_names_[i] for i in selected_indices]
        self.is_fitted = True

        logger.info(f"mRMR-JMI selected {len(self.selected_features_)} features: {self.selected_features_}")
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Project dataset to selected k features."""
        if not self.is_fitted:
            raise ValueError("MRMRJMISelector must be fitted before transform().")

        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_].values.astype(np.float32)
        else:
            indices = [self.all_feature_names_.index(f) for f in self.selected_features_]
            return np.asarray(X, dtype=np.float32)[:, indices]

    def transform_single_dict(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """Fast projection of single streaming event dictionary to selected k features."""
        return np.array([float(feature_dict.get(feat, 0.0)) for feat in self.selected_features_], dtype=np.float32)
