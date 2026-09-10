"""
Precision-Recall (PR) Curve and Average Precision (AP) Computation Module.
Computes per-class and macro-average PR curves.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import logging

logger = logging.getLogger(__name__)


def compute_multiclass_pr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """Computes Precision-Recall curves and Average Precision scores."""
    y_true = np.asarray(y_true).astype(int)
    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=np.arange(num_classes))

    if y_bin.shape[1] != y_prob.shape[1]:
        return {}

    precision_dict = {}
    recall_dict = {}
    ap_dict = {}

    for i in range(num_classes):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = float(average_precision_score(y_bin[:, i], y_prob[:, i]))
        precision_dict[class_names[i]] = prec.tolist()
        recall_dict[class_names[i]] = rec.tolist()
        ap_dict[class_names[i]] = ap

    # Macro average AP
    ap_dict["macro"] = float(np.mean([ap_dict[cls_name] for cls_name in class_names]))

    return {
        "precision": precision_dict,
        "recall": recall_dict,
        "average_precision": ap_dict,
        "class_names": class_names
    }
