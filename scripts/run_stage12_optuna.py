"""
Stage 12: Optuna Hyperparameter Optimization.
Tune:
- learning_rate: [1e-3, 5e-3]
- weight_decay: [1e-5, 5e-4]
- conv_channels: [32, 64]
- gru_hidden_dim: [32, 64]
- dropout: [0.1, 0.3]
- focal_gamma: [1.5, 2.5]
- batch_size: [128, 256]

Objective: Maximize Validation Macro-F1 (STRICTLY on Validation split, NEVER on Test split).
Saves: HYPERPARAMETER_RESULTS.csv
"""

import os
import sys
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score

from preprocessing.cleaner import EdgeIIoTCleaner
from models.se_attention import SEAttention1d

optuna.logging.set_verbosity(optuna.logging.WARNING)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Executing Stage 12 Optuna Study on {device}...")

# Load splits strictly
df = pd.read_csv("data/samples/edge_iiot_sample.csv", low_memory=False)
train_idx = pd.read_csv("data/splits/train_indices_4way.csv")["index"].values
val_idx = pd.read_csv("data/splits/validation_indices_4way.csv")["index"].values

train_df = df.iloc[train_idx].copy()
val_df = df.iloc[val_idx].copy()

cleaner = EdgeIIoTCleaner(drop_metadata=True)
cleaner.fit(train_df)
train_clean = cleaner.transform(train_df)
val_clean = cleaner.transform(val_df)

drop_cols = ["Attack_label", "Attack_type"]
candidate_cols = [c for c in train_clean.columns if c not in drop_cols]
valid_cols = [c for c in candidate_cols if train_clean[c].std() > 1e-6]
feat_ranking = pd.read_csv("artifacts/feature_ranking_detailed.csv")
top22_features = [f for f in feat_ranking["Feature"] if f in valid_cols][:22]

le = LabelEncoder()
y_tr = le.fit_transform(train_df["Attack_type"])
y_val = le.transform(val_df["Attack_type"])
num_classes = len(le.classes_)

scaler = RobustScaler()
X_tr = scaler.fit_transform(train_clean[top22_features])
X_val = scaler.transform(val_clean[top22_features])
input_dim = len(top22_features)

val_tensor = torch.from_numpy(X_val.astype(np.float32)).to(device)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (((1.0 - pt) ** self.gamma) * ce_loss).mean()

trials_data = []

def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-3, 5e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True)
    conv_ch = trial.suggest_categorical("conv_channels", [32, 64])
    gru_dim = trial.suggest_categorical("gru_hidden_dim", [32, 64])
    dropout = trial.suggest_float("dropout", 0.1, 0.3, step=0.1)
    gamma = trial.suggest_float("focal_gamma", 1.5, 2.5, step=0.5)
    batch_size = trial.suggest_categorical("batch_size", [128, 256])
    
    class TrialModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tab = nn.Sequential(
                nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU()
            )
            self.conv1 = nn.Conv1d(1, conv_ch, 3, padding=1)
            self.conv2 = nn.Conv1d(conv_ch, conv_ch, 3, padding=1)
            self.se = SEAttention1d(conv_ch, reduction=8)
            self.pool = nn.AdaptiveAvgPool1d(16)
            self.gru = nn.GRU(conv_ch, gru_dim, batch_first=True, bidirectional=True)
            self.attn = nn.MultiheadAttention(embed_dim=gru_dim*2, num_heads=4, batch_first=True)
            self.ln = nn.LayerNorm(gru_dim*2)
            self.fc = nn.Sequential(
                nn.Linear(gru_dim*2 + 128, 128), nn.LayerNorm(128), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes)
            )
        def forward(self, x):
            x1 = x.unsqueeze(1)
            h = F.gelu(self.conv1(x1))
            h = F.gelu(self.conv2(h)) + h
            h_se, _ = self.se(h)
            seq = self.pool(h_se.cpu() if h_se.device.type=="mps" else h_se).to(x.device).transpose(1, 2)
            out, _ = self.gru(seq)
            attn_out, _ = self.attn(out, out, out)
            seq_feat = self.ln(out + attn_out).mean(dim=1)
            return self.fc(torch.cat([seq_feat, self.tab(x)], dim=-1))

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr.astype(np.float32)), torch.from_numpy(y_tr).long()),
        batch_size=batch_size, shuffle=True
    )
    
    model = TrialModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5, eta_min=1e-5)
    loss_fn = FocalLoss(gamma=gamma)
    
    for _ in range(5):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            l = loss_fn(model(xb), yb)
            l.backward()
            opt.step()
        sched.step()
        
    model.eval()
    with torch.no_grad():
        preds = model(val_tensor).argmax(dim=-1).cpu().numpy()
        
    f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
    acc = float(accuracy_score(y_val, preds))
    
    trial_dict = {
        "Trial": trial.number,
        "Learning_Rate": round(lr, 6),
        "Weight_Decay": round(wd, 6),
        "Conv_Channels": conv_ch,
        "GRU_Dim": gru_dim,
        "Dropout": round(dropout, 2),
        "Focal_Gamma": round(gamma, 2),
        "Batch_Size": batch_size,
        "Validation_Macro_F1": round(f1, 4),
        "Validation_Accuracy": round(acc, 4)
    }
    trials_data.append(trial_dict)
    print(f"Trial {trial.number:02d} | Val Macro-F1: {f1*100:.2f}% | Val Acc: {acc*100:.2f}% | lr={lr:.5f}, gamma={gamma}")
    sys.stdout.flush()
    return f1

print("Starting Optuna Study (12 trials)...")
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=12)

df_trials = pd.DataFrame(trials_data).sort_values(by="Validation_Macro_F1", ascending=False)
df_trials.to_csv("HYPERPARAMETER_RESULTS.csv", index=False)
print("\n" + "="*50)
print("Optuna Study Complete!")
print(f"Best Trial: {study.best_trial.number}")
print(f"Best Validation Macro-F1: {study.best_value*100:.2f}%")
print("Best Hyperparameters:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
print("Saved HYPERPARAMETER_RESULTS.csv successfully!")
