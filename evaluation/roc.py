"""
ROC Curve and Area Under Curve (AUC) Computation Module.
Computes per-class One-vs-Rest ROC curves, micro-average, and macro-average curves.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import logging

logger = logging.getLogger(__name__)


def compute_multiclass_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Computes per-class and aggregate ROC curves and AUC scores.
    """
    y_true = np.asarray(y_true).astype(int)
    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=np.arange(num_classes))

    # In case binary or missing columns
    if y_bin.shape[1] != y_prob.shape[1]:
        return {}

    fpr_dict = {}
    tpr_dict = {}
    auc_dict = {}

    # Per-class ROC
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        cls_auc = auc(fpr, tpr)
        fpr_dict[class_names[i]] = fpr.tolist()
        tpr_dict[class_names[i]] = tpr.tolist()
        auc_dict[class_names[i]] = float(cls_auc)

    # Micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    auc_micro = float(auc(fpr_micro, tpr_micro))
    fpr_dict["micro"] = fpr_micro.tolist()
    tpr_dict["micro"] = tpr_micro.tolist()
    auc_dict["micro"] = auc_micro

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([np.array(fpr_dict[class_names[i]]) for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[class_names[i]], tpr_dict[class_names[i]])
    mean_tpr /= num_classes
    auc_macro = float(auc(all_fpr, mean_tpr))
    fpr_dict["macro"] = all_fpr.tolist()
    tpr_dict["macro"] = mean_tpr.tolist()
    auc_dict["macro"] = auc_macro

    return {
        "fpr": fpr_dict,
        "tpr": tpr_dict,
        "auc": auc_dict,
        "class_names": class_names
    }
