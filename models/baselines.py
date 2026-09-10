"""
Baseline Machine Learning and Deep Learning Models for Edge-IIoTset Benchmarking.
Implements standardized evaluation wrappers for:
- Traditional ML: Logistic Regression, Decision Tree, Random Forest, Extra Trees, Linear SVM, MLP
- Deep Learning: 1D-CNN, LSTM, BiLSTM, GRU, BiGRU, CNN-LSTM, CNN-BiLSTM
"""

from typing import Dict, List, Optional, Tuple, Any
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import logging

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Deep Learning Baseline Architectures (PyTorch)
# -----------------------------------------------------------------------------

class Standard1DCNN(nn.Module):
    """Standard 1D-CNN baseline."""
    def __init__(self, input_dim: int = 22, num_classes: int = 15):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(11)
        self.fc = nn.Linear(128 * 11, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x.unsqueeze(1))))
        h = F.relu(self.bn2(self.conv2(h)))
        if h.device.type == "mps":
            h = self.pool(h.cpu()).to(h.device).flatten(1)
        else:
            h = self.pool(h).flatten(1)
        return self.fc(h)


class StandardRNNBaseline(nn.Module):
    """Generic RNN/LSTM/GRU/BiRNN baseline."""
    def __init__(self, rnn_type: str = "LSTM", bidirectional: bool = False, input_dim: int = 22, num_classes: int = 15):
        super().__init__()
        self.rnn_type = rnn_type
        hidden_dim = 64
        num_directions = 2 if bidirectional else 1
        
        # Project tabular vector into sequence representation: (B, Seq=4, Feat=hidden)
        self.proj = nn.Linear(input_dim, 4 * hidden_dim)
        
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
            
        self.fc = nn.Linear(hidden_dim * num_directions, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        seq = self.proj(x).view(b, 4, -1)
        out, _ = self.rnn(seq)
        last = out[:, -1, :]
        return self.fc(last)


class HybridCNNRNNBaseline(nn.Module):
    """CNN-LSTM / CNN-BiLSTM baseline."""
    def __init__(self, bidirectional: bool = False, input_dim: int = 22, num_classes: int = 15):
        super().__init__()
        self.conv = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(11)
        num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(32, 64, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(64 * num_directions, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv(x.unsqueeze(1)))  # (B, 32, 22)
        if h.device.type == "mps":
            h = self.pool(h.cpu()).to(h.device).transpose(1, 2)
        else:
            h = self.pool(h).transpose(1, 2)       # (B, 8, 32)
        out, _ = self.lstm(h)
        return self.fc(out[:, -1, :])


# -----------------------------------------------------------------------------
# Factory for Baselines
# -----------------------------------------------------------------------------

def get_baseline_models(input_dim: int = 22, num_classes: int = 15, random_state: int = 42) -> Dict[str, Any]:
    """Returns a dictionary of all baseline models ready for training/evaluation."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=300, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=12, n_jobs=-1, random_state=random_state),
        "Extra Trees": ExtraTreesClassifier(n_estimators=50, max_depth=12, n_jobs=-1, random_state=random_state),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100, random_state=random_state),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(dual="auto", max_iter=500, random_state=random_state)),
        "1D-CNN": Standard1DCNN(input_dim=input_dim, num_classes=num_classes),
        "LSTM": StandardRNNBaseline("LSTM", bidirectional=False, input_dim=input_dim, num_classes=num_classes),
        "BiLSTM": StandardRNNBaseline("LSTM", bidirectional=True, input_dim=input_dim, num_classes=num_classes),
        "GRU": StandardRNNBaseline("GRU", bidirectional=False, input_dim=input_dim, num_classes=num_classes),
        "BiGRU": StandardRNNBaseline("GRU", bidirectional=True, input_dim=input_dim, num_classes=num_classes),
        "CNN-LSTM": HybridCNNRNNBaseline(bidirectional=False, input_dim=input_dim, num_classes=num_classes),
        "CNN-BiLSTM": HybridCNNRNNBaseline(bidirectional=True, input_dim=input_dim, num_classes=num_classes),
    }
    return models
