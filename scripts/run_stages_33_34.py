"""
Stages 33 & 34: Multi-Seed & 5-Fold Stratified Cross-Validation (Isolated Process).
Outputs:
- artifacts/multiseed_and_cv_results.json
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

print("Running Stages 33 & 34 in isolated process...")

df = pd.read_csv("data/samples/edge_iiot_sample.csv", low_memory=False)
train_idx = pd.read_csv("data/splits/train_indices_4way.csv")["index"].values
val_idx = pd.read_csv("data/splits/validation_indices_4way.csv")["index"].values

train_df = df.iloc[train_idx].copy()
val_df = df.iloc[val_idx].copy()

from preprocessing.cleaner import EdgeIIoTCleaner
cleaner = EdgeIIoTCleaner(drop_metadata=True)
cleaner.fit(train_df)
train_clean = cleaner.transform(train_df)
val_clean = cleaner.transform(val_df)

feat_info = joblib.load("models/feature_selector.pkl")
top_features = feat_info["selected_features"]
le = feat_info["label_encoder"]
y_tr = le.transform(train_df["Attack_type"])
y_val = le.transform(val_df["Attack_type"])

scaler = joblib.load("models/preprocessor.pkl")
X_tr = scaler.transform(train_clean[top_features])
X_val = scaler.transform(val_clean[top_features])

seeds = [42, 52, 62, 72, 82]
seed_f1s = []
seed_accs = []

for s in seeds:
    clf = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=s, n_jobs=2)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_val)
    f1_s = float(f1_score(y_val, preds, average="macro", zero_division=0))
    acc_s = float(accuracy_score(y_val, preds))
    seed_f1s.append(f1_s)
    seed_accs.append(acc_s)
    print(f"Seed {s} | Val Macro-F1: {f1_s*100:.2f}% | Val Acc: {acc_s*100:.2f}%")

mean_f1 = float(np.mean(seed_f1s))
std_f1 = float(np.std(seed_f1s))
mean_acc = float(np.mean(seed_accs))
std_acc = float(np.std(seed_accs))

print(f"\nMulti-Seed (5 seeds): Mean Acc = {mean_acc*100:.2f}% ± {std_acc*100:.2f}% | Mean Macro-F1 = {mean_f1*100:.2f}% ± {std_f1*100:.2f}%")

# 5-fold Stratified Cross-Validation on combined train+val
X_all = np.vstack([X_tr, X_val])
y_all = np.concatenate([y_tr, y_val])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1s = []
cv_accs = []

for fold, (tr_i, te_i) in enumerate(skf.split(X_all, y_all), 1):
    clf = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=2)
    clf.fit(X_all[tr_i], y_all[tr_i])
    preds = clf.predict(X_all[te_i])
    f1_f = float(f1_score(y_all[te_i], preds, average="macro", zero_division=0))
    acc_f = float(accuracy_score(y_all[te_i], preds))
    cv_f1s.append(f1_f)
    cv_accs.append(acc_f)
    print(f"Fold {fold}/5 | Macro-F1: {f1_f*100:.2f}% | Acc: {acc_f*100:.2f}%")

mean_cv_f1 = float(np.mean(cv_f1s))
std_cv_f1 = float(np.std(cv_f1s))
mean_cv_acc = float(np.mean(cv_accs))
std_cv_acc = float(np.std(cv_accs))

print(f"\n5-Fold Stratified CV: Mean Acc = {mean_cv_acc*100:.2f}% ± {std_cv_acc*100:.2f}% | Mean Macro-F1 = {mean_cv_f1*100:.2f}% ± {std_cv_f1*100:.2f}%")

with open("artifacts/multiseed_and_cv_results.json", "w") as f:
    json.dump({
        "multi_seed_results": {
            "seeds": seeds,
            "f1_scores": [round(x, 4) for x in seed_f1s],
            "accuracies": [round(x, 4) for x in seed_accs],
            "mean_macro_f1": round(mean_f1, 4),
            "std_macro_f1": round(std_f1, 4),
            "mean_accuracy": round(mean_acc, 4),
            "std_accuracy": round(std_acc, 4)
        },
        "stratified_5fold_cv": {
            "fold_f1_scores": [round(x, 4) for x in cv_f1s],
            "fold_accuracies": [round(x, 4) for x in cv_accs],
            "cv_mean_macro_f1": round(mean_cv_f1, 4),
            "cv_std_macro_f1": round(std_cv_f1, 4),
            "cv_mean_accuracy": round(mean_cv_acc, 4),
            "cv_std_accuracy": round(std_cv_acc, 4)
        }
    }, f, indent=2)

print("Saved artifacts/multiseed_and_cv_results.json successfully!")
