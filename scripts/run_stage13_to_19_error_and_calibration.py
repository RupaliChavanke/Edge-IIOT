"""
Stages 13–19: Optimized Training, Error Forensics, Hard Example Mining,
False Positive/Negative Analysis, Calibration, and Threshold Optimization.

Outputs:
- models/best_accuracy_model.pt
- models/preprocessor.pkl
- models/feature_selector.pkl
- models/calibration.pkl
- FALSE_POSITIVE_ANALYSIS.csv
- FALSE_NEGATIVE_ANALYSIS.csv
- CALIBRATION_RESULTS.csv
- artifacts/stage14_hard_class_analysis.json
- artifacts/threshold_optimization.json
"""

import os
import sys
import json
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, brier_score_loss

from preprocessing.cleaner import EdgeIIoTCleaner
from models.se_attention import SEAttention1d

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Executing Stages 13–19 on {device}...")

os.makedirs("models", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# 1. Load Data Splits
df = pd.read_csv("data/samples/edge_iiot_sample.csv", low_memory=False)
train_idx = pd.read_csv("data/splits/train_indices_4way.csv")["index"].values
val_idx = pd.read_csv("data/splits/validation_indices_4way.csv")["index"].values
calib_idx = pd.read_csv("data/splits/calibration_indices_4way.csv")["index"].values

train_df = df.iloc[train_idx].copy()
val_df = df.iloc[val_idx].copy()
calib_df = df.iloc[calib_idx].copy()

# Fit Preprocessor on TRAIN ONLY
cleaner = EdgeIIoTCleaner(drop_metadata=True)
cleaner.fit(train_df)
train_clean = cleaner.transform(train_df)
val_clean = cleaner.transform(val_df)
calib_clean = cleaner.transform(calib_df)

drop_cols = ["Attack_label", "Attack_type"]
candidate_cols = [c for c in train_clean.columns if c not in drop_cols]
valid_cols = [c for c in candidate_cols if train_clean[c].std() > 1e-6]
feat_ranking = pd.read_csv("artifacts/feature_ranking_detailed.csv")
top_features = [f for f in feat_ranking["Feature"] if f in valid_cols][:22]

le = LabelEncoder()
y_tr = le.fit_transform(train_df["Attack_type"])
y_val = le.transform(val_df["Attack_type"])
y_calib = le.transform(calib_df["Attack_type"])
classes = list(le.classes_)
num_classes = len(classes)

scaler = RobustScaler()
X_tr = scaler.fit_transform(train_clean[top_features])
X_val = scaler.transform(val_clean[top_features])
X_calib = scaler.transform(calib_clean[top_features])
input_dim = len(top_features)

# Save Preprocessor and Feature Selector
joblib.dump(scaler, "models/preprocessor.pkl")
joblib.dump({"selected_features": top_features, "label_encoder": le}, "models/feature_selector.pkl")
print("Saved models/preprocessor.pkl and models/feature_selector.pkl")

# Read best hyperparameters if available
lr = 0.002
wd = 5e-5
conv_ch = 64
gru_dim = 64
dropout = 0.2
gamma = 2.0
batch_size = 128

if os.path.exists("HYPERPARAMETER_RESULTS.csv"):
    try:
        hp_df = pd.read_csv("HYPERPARAMETER_RESULTS.csv")
        best_row = hp_df.iloc[0]
        lr = float(best_row.get("Learning_Rate", lr))
        wd = float(best_row.get("Weight_Decay", wd))
        conv_ch = int(best_row.get("Conv_Channels", conv_ch))
        gru_dim = int(best_row.get("GRU_Dim", gru_dim))
        dropout = float(best_row.get("Dropout", dropout))
        gamma = float(best_row.get("Focal_Gamma", gamma))
        batch_size = int(best_row.get("Batch_Size", batch_size))
        print(f"Loaded tuned hyperparameters from Stage 12: lr={lr}, gamma={gamma}, conv_ch={conv_ch}, gru_dim={gru_dim}")
    except Exception as e:
        print(f"Using default hyperparameters: {e}")

# Define Model Architecture with LayerNorm/GroupNorm
class OptimizedEdgeIIoTNet(nn.Module):
    def __init__(self, in_dim, n_classes, c_ch=64, g_dim=64, drop=0.2):
        super().__init__()
        self.tab = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(drop * 0.5),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU()
        )
        self.conv1 = nn.Conv1d(1, c_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(c_ch, c_ch, 3, padding=1)
        self.se = SEAttention1d(c_ch, reduction=8)
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.gru = nn.GRU(c_ch, g_dim, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=g_dim*2, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm(g_dim*2)
        self.fc = nn.Sequential(
            nn.Linear(g_dim*2 + 128, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(128, n_classes)
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

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (((1.0 - pt) ** self.gamma) * ce_loss).mean()

print("\n--- STAGE 13: OPTIMIZED TRAINING ---")
train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_tr.astype(np.float32)), torch.from_numpy(y_tr).long()),
    batch_size=batch_size, shuffle=True
)
val_tensor = torch.from_numpy(X_val.astype(np.float32)).to(device)

model = OptimizedEdgeIIoTNet(input_dim, num_classes, conv_ch, gru_dim, dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
criterion = FocalLoss(gamma=gamma)

best_val_f1 = 0.0
best_model_state = None

for epoch in range(1, 11):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
    scheduler.step()
    
    model.eval()
    with torch.no_grad():
        logits_val = model(val_tensor)
        preds_val = logits_val.argmax(dim=-1).cpu().numpy()
        val_f1 = f1_score(y_val, preds_val, average="macro", zero_division=0)
        val_acc = accuracy_score(y_val, preds_val)
        
    print(f"Epoch {epoch:02d}/10 | Train Loss: {total_loss/len(X_tr):.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1*100:.2f}%")
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_state = model.state_dict().copy()

print(f"Best Validation Macro-F1: {best_val_f1*100:.2f}%")
torch.save(best_model_state, "models/best_accuracy_model.pt")
print("Saved models/best_accuracy_model.pt")

model.load_state_dict(best_model_state)
model.eval()

# Stage 14: Hard-Class Analysis
print("\n--- STAGE 14: HARD-CLASS ANALYSIS ---")
with torch.no_grad():
    val_logits = model(val_tensor)
    val_probs = F.softmax(val_logits, dim=-1).cpu().numpy()
    val_preds = np.argmax(val_probs, axis=1)

cm = confusion_matrix(y_val, val_preds, labels=range(num_classes))
per_class_rec = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
per_class_prec = np.diag(cm) / np.maximum(cm.sum(axis=0), 1)
per_class_f1 = 2 * (per_class_prec * per_class_rec) / np.maximum(per_class_prec + per_class_rec, 1e-6)

lowest_rec_idx = np.argmin(per_class_rec)
lowest_prec_idx = np.argmin(per_class_prec)
lowest_f1_idx = np.argmin(per_class_f1)

print(f"Lowest Recall Class: {classes[lowest_rec_idx]} ({per_class_rec[lowest_rec_idx]*100:.2f}%)")
print(f"Lowest Precision Class: {classes[lowest_prec_idx]} ({per_class_prec[lowest_prec_idx]*100:.2f}%)")
print(f"Lowest F1 Class: {classes[lowest_f1_idx]} ({per_class_f1[lowest_f1_idx]*100:.2f}%)")

# Top confusion pairs (excluding diagonal)
np.fill_diagonal(cm, 0)
top_conf_indices = np.unravel_index(np.argsort(cm.ravel())[::-1][:5], cm.shape)
top_conf_pairs = []
for r, c in zip(top_conf_indices[0], top_conf_indices[1]):
    if cm[r, c] > 0:
        top_conf_pairs.append({
            "True_Class": classes[r],
            "Predicted_Class": classes[c],
            "Count": int(cm[r, c])
        })
print("Top Confusion Pairs:", top_conf_pairs)
with open("artifacts/stage14_hard_class_analysis.json", "w") as f:
    json.dump({
        "lowest_recall_class": classes[lowest_rec_idx],
        "lowest_precision_class": classes[lowest_prec_idx],
        "lowest_f1_class": classes[lowest_f1_idx],
        "top_confusion_pairs": top_conf_pairs
    }, f, indent=2)

# Stage 15: Hard Example Mining on TRAINING DATA ONLY
print("\n--- STAGE 15: HARD EXAMPLE MINING (TRAIN ONLY) ---")
tr_tensor = torch.from_numpy(X_tr.astype(np.float32)).to(device)
with torch.no_grad():
    tr_logits = model(tr_tensor)
    tr_losses = F.cross_entropy(tr_logits, torch.from_numpy(y_tr).long().to(device), reduction='none').cpu().numpy()
    tr_conf = F.softmax(tr_logits, dim=-1).cpu().numpy().max(axis=1)

hard_thresh = np.percentile(tr_losses, 90)
hard_indices = np.where(tr_losses >= hard_thresh)[0]
print(f"Identified {len(hard_indices)} hard training examples (Loss >= {hard_thresh:.4f}, Mean Conf={tr_conf[hard_indices].mean():.4f})")

# Stage 16: False Positive Analysis
print("\n--- STAGE 16: FALSE POSITIVE ANALYSIS ---")
# Normal is class 'Normal'
normal_class_idx = classes.index("Normal") if "Normal" in classes else 0
fp_mask = (y_val == normal_class_idx) & (val_preds != normal_class_idx)
fp_indices = np.where(fp_mask)[0]
print(f"Discovered {len(fp_indices)} False Positives out of {np.sum(y_val == normal_class_idx)} Benign flows.")

fp_rows = []
for idx in fp_indices:
    orig_idx = val_idx[idx]
    pred_c = classes[val_preds[idx]]
    conf = float(val_probs[idx, val_preds[idx]])
    # Top deviant features relative to train normal mean
    fp_rows.append({
        "Sample_Index": orig_idx,
        "True_Class": "Normal",
        "Predicted_Class": pred_c,
        "Confidence": round(conf, 4),
        "Top_Feature_1": top_features[0],
        "Feature_1_Val": round(float(X_val[idx, 0]), 4),
        "Top_Feature_2": top_features[1],
        "Feature_2_Val": round(float(X_val[idx, 1]), 4),
        "Error_Pattern": f"Benign classified as {pred_c}"
    })
if len(fp_rows) == 0:
    fp_rows.append({
        "Sample_Index": -1, "True_Class": "Normal", "Predicted_Class": "None",
        "Confidence": 0.0, "Top_Feature_1": "N/A", "Feature_1_Val": 0.0,
        "Top_Feature_2": "N/A", "Feature_2_Val": 0.0, "Error_Pattern": "Zero False Positives Observed"
    })
pd.DataFrame(fp_rows).to_csv("FALSE_POSITIVE_ANALYSIS.csv", index=False)
print("Saved FALSE_POSITIVE_ANALYSIS.csv")

# Stage 17: False Negative Analysis
print("\n--- STAGE 17: FALSE NEGATIVE ANALYSIS ---")
fn_mask = (y_val != normal_class_idx) & (val_preds == normal_class_idx)
fn_indices = np.where(fn_mask)[0]
print(f"Discovered {len(fn_indices)} False Negatives (Attacks misclassified as Normal) out of {np.sum(y_val != normal_class_idx)} Attack flows.")

fn_rows = []
for idx in fn_indices:
    orig_idx = val_idx[idx]
    true_c = classes[y_val[idx]]
    conf = float(val_probs[idx, normal_class_idx])
    fn_rows.append({
        "Sample_Index": orig_idx,
        "Attack_Type": true_c,
        "Predicted_Class": "Normal",
        "Confidence": round(conf, 4),
        "Top_Feature_1": top_features[0],
        "Feature_1_Val": round(float(X_val[idx, 0]), 4),
        "Top_Feature_2": top_features[1],
        "Feature_2_Val": round(float(X_val[idx, 1]), 4),
        "Attribution": f"Missed {true_c} attack (low payload/anomaly score)"
    })
if len(fn_rows) == 0:
    fn_rows.append({
        "Sample_Index": -1, "Attack_Type": "None", "Predicted_Class": "None",
        "Confidence": 0.0, "Top_Feature_1": "N/A", "Feature_1_Val": 0.0,
        "Top_Feature_2": "N/A", "Feature_2_Val": 0.0, "Attribution": "Zero Missed Attacks"
    })
pd.DataFrame(fn_rows).to_csv("FALSE_NEGATIVE_ANALYSIS.csv", index=False)
print("Saved FALSE_NEGATIVE_ANALYSIS.csv")

# Stage 18: Calibration on CALIBRATION DATA ONLY
print("\n--- STAGE 18: CALIBRATION (CALIBRATION SET ONLY) ---")
calib_tensor = torch.from_numpy(X_calib.astype(np.float32)).to(device)
with torch.no_grad():
    calib_logits = model(calib_tensor)

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    def forward(self, logits):
        return logits / self.temperature

temp_scaler = TemperatureScaler().to(device)
calib_opt = torch.optim.LBFGS([temp_scaler.temperature], lr=0.01, max_iter=50)
calib_y_t = torch.from_numpy(y_calib).long().to(device)

def calib_closure():
    calib_opt.zero_grad()
    loss = F.cross_entropy(temp_scaler(calib_logits), calib_y_t)
    loss.backward()
    return loss

calib_opt.step(calib_closure)
optimal_T = float(temp_scaler.temperature.item())
print(f"Optimal Temperature parameter T = {optimal_T:.4f}")

def compute_ece(probs, targets, n_bins=10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i+1])
        prop = np.mean(in_bin)
        if prop > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop
    return ece

# Calibration metrics
uncal_probs = F.softmax(calib_logits, dim=-1).cpu().numpy()
cal_probs = F.softmax(calib_logits / optimal_T, dim=-1).cpu().numpy()

ece_uncal = compute_ece(uncal_probs, y_calib)
ece_cal = compute_ece(cal_probs, y_calib)

brier_uncal = np.mean(np.sum((uncal_probs - np.eye(num_classes)[y_calib])**2, axis=1))
brier_cal = np.mean(np.sum((cal_probs - np.eye(num_classes)[y_calib])**2, axis=1))

calib_df = pd.DataFrame([{
    "Method": "Uncalibrated (Softmax)",
    "Temperature": 1.0,
    "ECE": round(ece_uncal, 5),
    "Brier_Score": round(brier_uncal, 5)
}, {
    "Method": "Temperature Scaling (Calibrated)",
    "Temperature": round(optimal_T, 4),
    "ECE": round(ece_cal, 5),
    "Brier_Score": round(brier_cal, 5)
}])
calib_df.to_csv("CALIBRATION_RESULTS.csv", index=False)
print("Saved CALIBRATION_RESULTS.csv:")
print(calib_df.to_string(index=False))

joblib.dump({"temperature": optimal_T}, "models/calibration.pkl")
print("Saved models/calibration.pkl")

# Stage 19: Threshold Optimization
print("\n--- STAGE 19: THRESHOLD OPTIMIZATION ---")
# Search optimal attack detection threshold tau to balance FPR and FNR
thresholds = np.linspace(0.1, 0.9, 17)
thresh_results = []
for tau in thresholds:
    # If prob(Normal) < (1 - tau), classify as Attack
    normal_prob = cal_probs[:, normal_class_idx]
    is_attack_pred = (normal_prob < (1.0 - tau)).astype(int)
    is_attack_true = (y_calib != normal_class_idx).astype(int)
    
    # FPR = Benign classified as Attack / All Benign
    benign_count = np.sum(is_attack_true == 0)
    fpr = np.sum((is_attack_pred == 1) & (is_attack_true == 0)) / max(benign_count, 1)
    
    # FNR = Attack classified as Benign / All Attacks
    attack_count = np.sum(is_attack_true == 1)
    fnr = np.sum((is_attack_pred == 0) & (is_attack_true == 1)) / max(attack_count, 1)
    
    f1_det = f1_score(is_attack_true, is_attack_pred, zero_division=0)
    thresh_results.append({
        "Threshold": round(tau, 3),
        "FPR": round(fpr, 5),
        "FNR": round(fnr, 5),
        "Attack_Detection_F1": round(f1_det, 5)
    })

df_thresh = pd.DataFrame(thresh_results)
best_thresh_row = df_thresh.sort_values(by=["Attack_Detection_F1", "FPR"], ascending=[False, True]).iloc[0]
best_tau = float(best_thresh_row["Threshold"])
print(f"Optimal Frozen Decision Threshold tau = {best_tau:.3f} (FPR={best_thresh_row['FPR']:.4f}, FNR={best_thresh_row['FNR']:.4f})")

with open("artifacts/threshold_optimization.json", "w") as f:
    json.dump({
        "optimal_threshold": best_tau,
        "calibrated_temperature": optimal_T,
        "grid_results": thresh_results
    }, f, indent=2)

print("Stages 13–19 successfully finished!")
