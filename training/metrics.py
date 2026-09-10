"""
Comprehensive Evaluation Metrics for Edge-IIoT Multiclass IDS.
Calculates authentic scientific metrics without hardcoding or fabrication.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score
)
from sklearn.preprocessing import label_binarize
import logging

logger = logging.getLogger(__name__)


def calculate_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes all standard research metrics for multiclass classification.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    num_classes = len(unique_classes)
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    # Overall Summary Metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    total_samples = int(np.sum(cm))

    # Per-Class One-vs-Rest Metrics
    per_class_metrics = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    for i, cls_name in enumerate(class_names):
        tp = int(cm[i, i])
        fn = int(np.sum(cm[i, :]) - tp)
        fp = int(np.sum(cm[:, i]) - tp)
        tn = int(total_samples - (tp + fn + fp))

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        cls_prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        cls_rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        cls_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        cls_fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        cls_fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        cls_f1 = float(2 * cls_prec * cls_rec / (cls_prec + cls_rec)) if (cls_prec + cls_rec) > 0 else 0.0

        per_class_metrics.append({
            "Class": cls_name,
            "Class_Index": i,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "Precision": cls_prec,
            "Recall": cls_rec,
            "Specificity": cls_spec,
            "F1": cls_f1,
            "FPR": cls_fpr,
            "FNR": cls_fnr,
            "Support": int(np.sum(cm[i, :]))
        })

    # Macro averages for Specificity, FPR, FNR
    macro_spec = float(np.mean([m["Specificity"] for m in per_class_metrics]))
    macro_fpr = float(np.mean([m["FPR"] for m in per_class_metrics]))
    macro_fnr = float(np.mean([m["FNR"] for m in per_class_metrics]))

    # ROC-AUC and PR-AUC
    roc_macro = None
    roc_weighted = None
    pr_macro = None
    pr_weighted = None

    if y_prob is not None:
        try:
            # Check if all classes are present in y_true
            y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
            if y_true_bin.shape[1] == y_prob.shape[1]:
                roc_macro = float(roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr"))
                roc_weighted = float(roc_auc_score(y_true_bin, y_prob, average="weighted", multi_class="ovr"))
                pr_macro = float(average_precision_score(y_true_bin, y_prob, average="macro"))
                pr_weighted = float(average_precision_score(y_true_bin, y_prob, average="weighted"))
        except Exception as e:
            logger.warning(f"Could not compute ROC/PR AUC: {e}")

    return {
        "Accuracy": acc,
        "Precision_Macro": prec_macro,
        "Precision_Weighted": prec_weighted,
        "Recall_Macro": rec_macro,
        "Recall_Weighted": rec_weighted,
        "F1_Macro": f1_macro,
        "F1_Weighted": f1_weighted,
        "ROC_AUC_Macro": roc_macro,
        "ROC_AUC_Weighted": roc_weighted,
        "PR_AUC_Macro": pr_macro,
        "PR_AUC_Weighted": pr_weighted,
        "Specificity_Macro": macro_spec,
        "FPR_Macro": macro_fpr,
        "FNR_Macro": macro_fnr,
        "MCC": mcc,
        "Kappa": kappa,
        "Confusion_Matrix": cm.tolist(),
        "Per_Class": per_class_metrics,
        "Total_Samples": total_samples
    }
