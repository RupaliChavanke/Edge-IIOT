"""
Phase 10: Bayesian Hyperparameter Optimization with Optuna.
Tunes:
- learning_rate: [0.0005, 0.005]
- weight_decay: [1e-5, 1e-3]
- dropout: [0.10, 0.35]
- conv_channels: [32, 64]
- gru_hidden_dim: [32, 64]
- focal_gamma: [1.5, 3.0]
- lambda_center: [0.001, 0.05]
Objective: Maximize Validation Macro-F1 (strictly on validation split, never on test split).
Stores results in: experiments/optuna/
"""

import os
import json
import optuna
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler, LabelEncoder

from preprocessing.cleaner import EdgeIIoTCleaner
from models.proposed_model import ProposedHybridEdgeIIoTModel
from training.losses import ProposedCompoundLoss

def run_optuna_study(n_trials=10, epochs_per_trial=4, output_dir="experiments/optuna"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Optuna Bayesian Optimization on {device} (n_trials={n_trials})...")

    # Load splits
    df = pd.read_csv("data/samples/edge_iiot_sample.csv", low_memory=False)
    train_idx = pd.read_csv("data/splits/train_indices.csv")["index"].values
    val_idx = pd.read_csv("data/splits/validation_indices.csv")["index"].values

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    cleaner = EdgeIIoTCleaner(drop_metadata=True)
    cleaner.fit(train_df)
    train_clean = cleaner.transform(train_df)
    val_clean = cleaner.transform(val_df)

    drop_cols = ["Attack_label", "Attack_type"]
    candidate_cols = [c for c in train_clean.columns if c not in drop_cols]
    X_tr = train_clean[candidate_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_val = val_clean[candidate_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    non_zero_cols = [c for c in candidate_cols if X_tr[c].std() > 1e-6]
    X_tr_np = X_tr[non_zero_cols].values
    X_val_np = X_val[non_zero_cols].values

    le = LabelEncoder()
    y_tr = le.fit_transform(train_df["Attack_type"])
    y_val = le.transform(val_df["Attack_type"])
    num_classes = len(le.classes_)

    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr_np)
    X_val_s = scaler.transform(X_val_np)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr_s.astype(np.float32)), torch.from_numpy(y_tr).long()),
        batch_size=128,
        shuffle=True
    )

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.15, 0.35)
        conv_ch = trial.suggest_categorical("conv_channels", [32, 64])
        gru_dim = trial.suggest_categorical("gru_hidden_dim", [32, 64])
        gamma = trial.suggest_float("focal_gamma", 1.5, 2.5)
        l_center = trial.suggest_float("lambda_center", 0.005, 0.03, log=True)

        model = ProposedHybridEdgeIIoTModel(
            input_dim=len(non_zero_cols),
            num_classes=num_classes,
            conv_channels=conv_ch,
            gru_hidden_dim=gru_dim,
            dropout=dropout
        ).to(device)

        criterion = ProposedCompoundLoss(
            num_classes=num_classes,
            feat_dim=128,
            gamma=gamma,
            lambda_center=l_center
        ).to(device)

        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(criterion.center_loss.parameters()),
            lr=lr,
            weight_decay=wd
        )

        val_tensor = torch.from_numpy(X_val_s.astype(np.float32)).to(device)

        for ep in range(epochs_per_trial):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb, routing_mode="train")
                loss, _, _, _ = criterion(out["fast_logits"], out["deep_logits"], out["latent_features"], yb)
                loss.backward()
                optimizer.step()

            # Eval on Val
            model.eval()
            with torch.no_grad():
                v_out = model(val_tensor, routing_mode="dynamic")
                preds = v_out["predictions"].cpu().numpy()
                val_f1 = f1_score(y_val, preds, average="macro", zero_division=0)

            trial.report(val_f1, ep)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return val_f1

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)

    print(f"\nOptuna study complete! Best Val Macro-F1: {study.best_value*100:.2f}%")
    print("Best Hyperparameters:", study.best_params)

    # Save results
    study.trials_dataframe().to_csv(os.path.join(output_dir, "optuna_study.csv"), index=False)
    with open(os.path.join(output_dir, "best_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Saved results to {output_dir}")

if __name__ == "__main__":
    run_optuna_study()
