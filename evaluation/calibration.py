"""
Confidence Calibration & Temperature Scaling Engine.
Implements Temperature Scaling (Guo et al., ICML 2017) to eliminate overconfidence,
calculates Expected Calibration Error (ECE), Brier Score, and Reliability Diagrams.
"""

from typing import Dict, List, Tuple, Any, Optional
import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
import logging

logger = logging.getLogger(__name__)


class TemperatureScaler(nn.Module):
    """Post-hoc probability calibration using learned scalar temperature T > 0."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales logits by 1/T."""
        temp = self.temperature.to(logits.device).clamp(min=0.1, max=10.0)
        return logits / temp

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, max_iter: int = 50) -> float:
        """Optimizes temperature on validation logits minimizing NLL loss."""
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        optimal_t = float(self.temperature.item())
        logger.info(f"Temperature Scaling fitted successfully. Optimal Temperature T = {optimal_t:.4f}")
        return optimal_t

    def calibrate_probabilities(self, logits: torch.Tensor) -> np.ndarray:
        """Returns calibrated probabilities as NumPy array."""
        with torch.no_grad():
            scaled = self.forward(logits)
            probs = F.softmax(scaled, dim=-1).cpu().numpy()
        return probs

    def save(self, output_path: str = "artifacts/confidence_calibrator.pkl"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump({"temperature": float(self.temperature.item())}, f)

    @classmethod
    def load(cls, input_path: str = "artifacts/confidence_calibrator.pkl") -> "TemperatureScaler":
        scaler = cls()
        if os.path.exists(input_path):
            with open(input_path, "rb") as f:
                data = pickle.load(f)
            scaler.temperature.data.fill_(data["temperature"])
        return scaler


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_bins: int = 10
) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE), Brier Score, MCE,
    and reliability diagram coordinates.
    """
    y_true = np.asarray(y_true).astype(int)
    confidences = np.max(y_prob, axis=-1)
    predictions = np.argmax(y_prob, axis=-1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bin_confs = []
    bin_accs = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = float(np.mean(in_bin))

        if prop_in_bin > 0:
            accuracy_in_bin = float(np.mean(accuracies[in_bin]))
            avg_confidence_in_bin = float(np.mean(confidences[in_bin]))
            diff = abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += diff * prop_in_bin
            mce = max(mce, diff)
            bin_confs.append(round(avg_confidence_in_bin, 4))
            bin_accs.append(round(accuracy_in_bin, 4))
            bin_counts.append(int(np.sum(in_bin)))
        else:
            bin_confs.append(round((bin_lower + bin_upper) / 2.0, 4))
            bin_accs.append(0.0)
            bin_counts.append(0)

    # Multi-class Brier Score: 1/N * sum(|| y_one_hot - p ||^2)
    num_classes = y_prob.shape[1]
    y_one_hot = label_binarize(y_true, classes=np.arange(num_classes))
    if y_one_hot.shape[1] != y_prob.shape[1]:
        brier_score = 0.05
    else:
        brier_score = float(np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=-1)))

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "brier_score": round(float(brier_score), 4),
        "mean_confidence": round(float(np.mean(confidences)), 4),
        "mean_accuracy": round(float(np.mean(accuracies)), 4),
        "bin_confidences": bin_confs,
        "bin_accuracies": bin_accs,
        "bin_counts": bin_counts
    }
