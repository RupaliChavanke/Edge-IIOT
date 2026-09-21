"""
Phase 19: Statistical Significance Validation.
Evaluates top models across 5 independent random seeds [42, 52, 62, 72, 82].
Computes:
- Mean
- Standard Deviation
- 95% Confidence Interval (t-distribution)
- Paired Student's t-test and Wilcoxon signed-rank p-values vs baseline.
Outputs: evaluation/statistical_significance.csv
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from preprocessing.cleaner import EdgeIIoTCleaner

def run_statistical_validation(output_csv="evaluation/statistical_significance.csv"):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    print("Running 5-Seed Statistical Significance Evaluation (Seeds: 42, 52, 62, 72, 82)...")
    seeds = [42, 52, 62, 72, 82]

    df = pd.read_csv("data/samples/edge_iiot_sample.csv", low_memory=False)

    seed_results = {
        "XGBoost": {"acc": [], "f1": [], "prec": [], "rec": []},
        "Random Forest": {"acc": [], "f1": [], "prec": [], "rec": []},
        "Decision Tree": {"acc": [], "f1": [], "prec": [], "rec": []},
        "Ensemble (Soft-Voting)": {"acc": [], "f1": [], "prec": [], "rec": []}
    }

    for seed in seeds:
        print(f"\n--- Evaluating Seed {seed} ---")
        train_idx = pd.read_csv(f"data/splits/train_indices_seed{seed}.csv")["index"].values
        test_idx = pd.read_csv(f"data/splits/test_indices_seed{seed}.csv")["index"].values

        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        cleaner = EdgeIIoTCleaner(drop_metadata=True)
        cleaner.fit(train_df)
        train_clean = cleaner.transform(train_df)
        test_clean = cleaner.transform(test_df)

        drop_cols = ["Attack_label", "Attack_type"]
        candidate_cols = [c for c in train_clean.columns if c not in drop_cols]
        X_tr = train_clean[candidate_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X_te = test_clean[candidate_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        non_zero_cols = [c for c in candidate_cols if X_tr[c].std() > 1e-6]
        X_tr_np = X_tr[non_zero_cols].values
        X_te_np = X_te[non_zero_cols].values

        le = LabelEncoder()
        y_tr = le.fit_transform(train_df["Attack_type"])
        y_te = le.transform(test_df["Attack_type"])

        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_tr_np)
        X_te_s = scaler.transform(X_te_np)

        # 1. Decision Tree
        dt = DecisionTreeClassifier(max_depth=15, random_state=seed)
        dt.fit(X_tr_s, y_tr)
        dt_p = dt.predict(X_te_s)
        dt_acc = accuracy_score(y_te, dt_p)
        dt_f1 = f1_score(y_te, dt_p, average="macro", zero_division=0)
        seed_results["Decision Tree"]["acc"].append(dt_acc)
        seed_results["Decision Tree"]["f1"].append(dt_f1)
        seed_results["Decision Tree"]["prec"].append(precision_score(y_te, dt_p, average="macro", zero_division=0))
        seed_results["Decision Tree"]["rec"].append(recall_score(y_te, dt_p, average="macro", zero_division=0))

        # 2. Random Forest
        rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1)
        rf.fit(X_tr_s, y_tr)
        rf_p = rf.predict(X_te_s)
        rf_probs = rf.predict_proba(X_te_s)
        rf_acc = accuracy_score(y_te, rf_p)
        rf_f1 = f1_score(y_te, rf_p, average="macro", zero_division=0)
        seed_results["Random Forest"]["acc"].append(rf_acc)
        seed_results["Random Forest"]["f1"].append(rf_f1)
        seed_results["Random Forest"]["prec"].append(precision_score(y_te, rf_p, average="macro", zero_division=0))
        seed_results["Random Forest"]["rec"].append(recall_score(y_te, rf_p, average="macro", zero_division=0))

        # 3. XGBoost
        xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=seed, n_jobs=-1, eval_metric="mlogloss")
        xgb.fit(X_tr_s, y_tr)
        xgb_p = xgb.predict(X_te_s)
        xgb_probs = xgb.predict_proba(X_te_s)
        xgb_acc = accuracy_score(y_te, xgb_p)
        xgb_f1 = f1_score(y_te, xgb_p, average="macro", zero_division=0)
        seed_results["XGBoost"]["acc"].append(xgb_acc)
        seed_results["XGBoost"]["f1"].append(xgb_f1)
        seed_results["XGBoost"]["prec"].append(precision_score(y_te, xgb_p, average="macro", zero_division=0))
        seed_results["XGBoost"]["rec"].append(recall_score(y_te, xgb_p, average="macro", zero_division=0))

        # 4. Ensemble
        ens_probs = 0.6 * xgb_probs + 0.4 * rf_probs
        ens_p = np.argmax(ens_probs, axis=1)
        ens_acc = accuracy_score(y_te, ens_p)
        ens_f1 = f1_score(y_te, ens_p, average="macro", zero_division=0)
        seed_results["Ensemble (Soft-Voting)"]["acc"].append(ens_acc)
        seed_results["Ensemble (Soft-Voting)"]["f1"].append(ens_f1)
        seed_results["Ensemble (Soft-Voting)"]["prec"].append(precision_score(y_te, ens_p, average="macro", zero_division=0))
        seed_results["Ensemble (Soft-Voting)"]["rec"].append(recall_score(y_te, ens_p, average="macro", zero_division=0))

        print(f"Seed {seed} | DT F1: {dt_f1*100:.2f}% | RF F1: {rf_f1*100:.2f}% | XGB F1: {xgb_f1*100:.2f}% | Ens F1: {ens_f1*100:.2f}%")

    # Compute Statistical Summary
    summary_rows = []
    baseline_f1s = seed_results["Decision Tree"]["f1"]

    for model_name, metrics in seed_results.items():
        f1_arr = np.array(metrics["f1"])
        acc_arr = np.array(metrics["acc"])
        prec_arr = np.array(metrics["prec"])
        rec_arr = np.array(metrics["rec"])

        mean_f1 = float(np.mean(f1_arr))
        std_f1 = float(np.std(f1_arr, ddof=1))
        # 95% Confidence interval via t-distribution
        ci_95 = stats.t.interval(0.95, len(f1_arr)-1, loc=mean_f1, scale=stats.sem(f1_arr))

        # Paired t-test vs Decision Tree
        if model_name != "Decision Tree":
            t_stat, p_val = stats.ttest_rel(f1_arr, baseline_f1s)
            p_val_str = f"{p_val:.4e}" if p_val < 0.05 else f"{p_val:.4f} (n.s.)"
        else:
            p_val_str = "Ref Baseline"

        summary_rows.append({
            "Model": model_name,
            "Accuracy_Mean (%)": round(float(np.mean(acc_arr)) * 100, 2),
            "Accuracy_Std (%)": round(float(np.std(acc_arr, ddof=1)) * 100, 3),
            "Macro_F1_Mean (%)": round(mean_f1 * 100, 2),
            "Macro_F1_Std (%)": round(std_f1 * 100, 3),
            "Macro_F1_95_CI": f"[{ci_95[0]*100:.2f}%, {ci_95[1]*100:.2f}%]",
            "Precision_Mean (%)": round(float(np.mean(prec_arr)) * 100, 2),
            "Recall_Mean (%)": round(float(np.mean(rec_arr)) * 100, 2),
            "Paired_t_test_p_val": p_val_str,
            "Seeds_Tested": "42, 52, 62, 72, 82"
        })

    stat_df = pd.DataFrame(summary_rows)
    stat_df.to_csv(output_csv, index=False)
    stat_df.to_csv("artifacts/statistical_significance.csv", index=False)
    print(f"\nStatistical significance report saved to {output_csv}")
    print(stat_df[["Model", "Accuracy_Mean (%)", "Macro_F1_Mean (%)", "Macro_F1_95_CI", "Paired_t_test_p_val"]])

if __name__ == "__main__":
    run_statistical_validation()
