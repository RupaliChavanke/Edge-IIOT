"""
Confusion Matrix Analysis Module.
Generates raw, row-normalized, column-normalized, and per-class One-vs-Rest confusion metrics.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import logging

logger = logging.getLogger(__name__)


class ConfusionMatrixAnalyzer:
    """Computes multidimensional confusion matrices and exportable tabular representations."""

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]):
        self.y_true = np.asarray(y_true).astype(int)
        self.y_pred = np.asarray(y_pred).astype(int)
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.cm_raw = confusion_matrix(self.y_true, self.y_pred, labels=np.arange(self.num_classes))

    def get_raw_matrix(self) -> np.ndarray:
        """Returns raw count confusion matrix."""
        return self.cm_raw

    def get_normalized_matrix(self, mode: str = "recall") -> np.ndarray:
        """
        Normalized confusion matrix.
        mode: 'recall' (rows sum to 1), 'precision' (cols sum to 1), or 'total'
        """
        cm = self.cm_raw.astype(np.float64)
        if mode == "recall":
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            return np.round(cm / row_sums, 4)
        elif mode == "precision":
            col_sums = cm.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1.0
            return np.round(cm / col_sums, 4)
        else:
            total = cm.sum()
            return np.round(cm / (total if total > 0 else 1.0), 4)

    def get_per_class_ovr_df(self) -> pd.DataFrame:
        """Computes One-vs-Rest breakdown for each class."""
        records = []
        total_samples = int(np.sum(self.cm_raw))

        for i, cls_name in enumerate(self.class_names):
            tp = int(self.cm_raw[i, i])
            fn = int(np.sum(self.cm_raw[i, :]) - tp)
            fp = int(np.sum(self.cm_raw[:, i]) - tp)
            tn = int(total_samples - (tp + fn + fp))

            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            records.append({
                "Class": cls_name,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "Specificity": round(spec, 4),
                "F1_Score": round(f1, 4),
                "FPR": round(fpr, 4),
                "FNR": round(fnr, 4),
                "Support": tp + fn
            })

        return pd.DataFrame(records)

    def export_csv(self, file_path: str, mode: str = "raw") -> None:
        """Exports selected matrix to CSV."""
        if mode == "raw":
            data = self.get_raw_matrix()
        else:
            data = self.get_normalized_matrix(mode)
        df = pd.DataFrame(data, index=self.class_names, columns=self.class_names)
        df.to_csv(file_path)
