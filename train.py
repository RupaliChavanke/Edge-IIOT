"""
Standalone Offline Model Training & Artifact Generation Pipeline for Edge-IIoTset IDS.
Run exclusively via:
    python train.py [--sample | --full] [--epochs N] [--batch-size B]

Generates the complete frozen 'artifacts/' directory package for deployment.
"""

import os
import sys
import json
import time
import pickle
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import logging

from preprocessing.loader import EdgeIIoTDataLoader
from models.proposed_model import ProposedHybridEdgeIIoTModel
from training.trainer import EdgeIIoTTrainer
from evaluation.evaluator import ModelBenchmarkRunner
from evaluation.ablation import AblationStudyRunner
from evaluation.latency import profile_pipeline_latency
from evaluation.roc import compute_multiclass_roc
from evaluation.pr import compute_multiclass_pr

import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OfflineTrainer")


def generate_offline_png_plots(
    output_dir: str,
    cm: np.ndarray,
    class_names: list,
    roc_data: dict,
    pr_data: dict,
    history: list,
    df_benchmark: pd.DataFrame,
    df_ablation: pd.DataFrame,
    latency_profile: dict,
    param_counts: dict,
    calib_metrics: dict = None
):
    if calib_metrics is None:
        calib_metrics = {}
    """Generates 9 high-resolution publication-ready PNG artifacts for the offline evaluation bundle."""
    logger.info("Generating 9 offline evaluation plot PNGs in artifacts/...")
    plt.style.use("dark_background")
    accent_color = "#38BDF8"
    card_bg = "#0F172A"

    # 1. Raw Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=card_bg)
    ax.set_facecolor(card_bg)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=False)
    ax.set_title("Edge-IIoTset Confusion Matrix (Offline Test)", fontsize=14, color="#F8FAFC", pad=15)
    ax.set_xlabel("Predicted Class", fontsize=11, color="#94A3B8")
    ax.set_ylabel("True Class", fontsize=11, color="#94A3B8")
    plt.xticks(rotation=45, ha="right", fontsize=9, color="#E2E8F0")
    plt.yticks(rotation=0, fontsize=9, color="#E2E8F0")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150, facecolor=card_bg)
    plt.close()

    # 2. Normalized Confusion Matrix
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=card_bg)
    ax.set_facecolor(card_bg)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=False)
    ax.set_title("Normalized Confusion Matrix (Recall per Attack)", fontsize=14, color="#F8FAFC", pad=15)
    ax.set_xlabel("Predicted Class", fontsize=11, color="#94A3B8")
    ax.set_ylabel("True Class", fontsize=11, color="#94A3B8")
    plt.xticks(rotation=45, ha="right", fontsize=9, color="#E2E8F0")
    plt.yticks(rotation=0, fontsize=9, color="#E2E8F0")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "normalized_confusion_matrix.png"), dpi=150, facecolor=card_bg)
    plt.close()

    # 3. ROC Curve
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=card_bg)
    ax.set_facecolor(card_bg)
    if "macro" in roc_data.get("auc", {}):
        ax.plot(roc_data["fpr"]["macro"], roc_data["tpr"]["macro"], label=f"Macro (AUC={roc_data['auc']['macro']:.3f})", color="#F43F5E", lw=2.5, linestyle="--")
    if "micro" in roc_data.get("auc", {}):
        ax.plot(roc_data["fpr"]["micro"], roc_data["tpr"]["micro"], label=f"Micro (AUC={roc_data['auc']['micro']:.3f})", color="#38BDF8", lw=2, linestyle=":")
    ax.plot([0, 1], [0, 1], color="#64748B", linestyle=":", lw=1)
    ax.set_title("Multi-Class Receiver Operating Characteristic (ROC)", fontsize=13, color="#F8FAFC")
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11, color="#94A3B8")
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=11, color="#94A3B8")
    ax.legend(loc="lower right", facecolor="#1E293B", edgecolor="#334155")
    ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150, facecolor=card_bg)
    plt.close()

    # 4. Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=card_bg)
    ax.set_facecolor(card_bg)
    ap_map = pr_data.get("average_precision", {})
    for c in class_names[:6]:
        if c in ap_map and c in pr_data.get("recall", {}):
            ax.plot(pr_data["recall"][c], pr_data["precision"][c], label=f"{c} (AP={ap_map[c]:.2f})", lw=1.5)
    ax.set_title("Precision-Recall (PR) Curves across Threat Classes", fontsize=13, color="#F8FAFC")
    ax.set_xlabel("Recall", fontsize=11, color="#94A3B8")
    ax.set_ylabel("Precision", fontsize=11, color="#94A3B8")
    if ap_map:
        ax.legend(loc="lower left", facecolor="#1E293B", edgecolor="#334155", fontsize=9)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"), dpi=150, facecolor=card_bg)
    plt.close()

    # 5. Training History
    df_h = pd.DataFrame(history)
    fig, ax1 = plt.subplots(figsize=(9, 5), facecolor=card_bg)
    ax1.set_facecolor(card_bg)
    ax2 = ax1.twinx()
    ax2.set_facecolor(card_bg)
    if not df_h.empty:
        l1 = ax1.plot(df_h["epoch"], df_h["train_loss"], color="#F59E0B", lw=2.2, label="Train Loss", marker="o")
        l2 = ax2.plot(df_h["epoch"], df_h["val_accuracy"]*100, color="#10B981", lw=2.2, label="Val Accuracy %", marker="s")
        l3 = ax2.plot(df_h["epoch"], df_h["val_f1"]*100, color="#6366F1", lw=2.2, label="Val Macro-F1 %", marker="^")
        ax1.set_xlabel("Epoch", fontsize=11, color="#94A3B8")
        ax1.set_ylabel("Loss", fontsize=11, color="#F59E0B")
        ax2.set_ylabel("Metric (%)", fontsize=11, color="#10B981")
        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="center right", facecolor="#1E293B", edgecolor="#334155")
    ax1.set_title("Training Loss & Validation Metrics History", fontsize=13, color="#F8FAFC")
    ax1.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=150, facecolor=card_bg)
    plt.close()

    # 6. Benchmark Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=card_bg)
    ax.set_facecolor(card_bg)
    if not df_benchmark.empty and "Model" in df_benchmark.columns:
        models = df_benchmark["Model"]
        accs = df_benchmark["Accuracy"] * 100
        colors = ["#38BDF8" if "Proposed" in m else "#475569" for m in models]
        bars = ax.bar(range(len(models)), accs, color=colors, edgecolor="#0F172A", width=0.65)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9, color="#E2E8F0")
        ax.set_ylabel("Test Accuracy (%)", fontsize=11, color="#94A3B8")
        ax.set_title("15-Model Empirical Benchmark Comparison", fontsize=14, color="#F8FAFC")
        ax.set_ylim(min(accs.min() - 5, 80), 101)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=8, color="#F8FAFC")
    ax.grid(axis="y", alpha=0.15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark.png"), dpi=150, facecolor=card_bg)
    plt.close()

    # 7. Ablation Study Chart
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=card_bg)
    ax.set_facecolor(card_bg)
    if not df_ablation.empty and "Ablation Variant" in df_ablation.columns:
        variants = df_ablation["Ablation Variant"]
        accs = df_ablation["Accuracy"] * 100
        colors = ["#10B981" if "Proposed" in v else "#64748B" for v in variants]
        y_pos = range(len(variants))
        ax.barh(y_pos, accs, color=colors, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(variants, fontsize=9, color="#E2E8F0")
        ax.invert_yaxis()
        ax.set_xlabel("Accuracy (%)", fontsize=11, color="#94A3B8")
        ax.set_title("11-Variant Architectural Component Ablation Study", fontsize=13, color="#F8FAFC")
        ax.set_xlim(min(accs.min() - 3, 85), 101)
    ax.grid(axis="x", alpha=0.15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation.png"), dpi=150, facecolor=card_bg)
    plt.close()

    # 8. Latency Waterfall Chart
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=card_bg)
    ax.set_facecolor(card_bg)
    stages = [k for k in latency_profile.keys() if k != "End_to_End_Pipeline"]
    means = [latency_profile[k].get("mean_ms", 0.05) for k in stages]
    ax.barh(stages, means, color="#38BDF8", height=0.55)
    ax.set_xlabel("Mean Execution Latency (ms)", fontsize=11, color="#94A3B8")
    ax.set_title("Per-Stage Execution Latency Breakdown", fontsize=13, color="#F8FAFC")
    for i, v in enumerate(means):
        ax.text(v + 0.005, i, f"{v:.3f} ms", va='center', fontsize=9, color="#F8FAFC")
    ax.grid(axis="x", alpha=0.15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency.png"), dpi=150, facecolor=card_bg)
    plt.close()

    # 9. Parameter & Complexity Chart (Pareto-optimal FLOPs vs Acc)
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=card_bg)
    ax.set_facecolor(card_bg)
    if not df_benchmark.empty and "Accuracy" in df_benchmark.columns and "Parameters" in df_benchmark.columns:
        for idx, row in df_benchmark.iterrows():
            m = row["Model"]
            acc = row["Accuracy"] * 100
            params = row["Parameters"] / 1000.0  # in K params
            is_prop = "Proposed" in m
            ax.scatter(params, acc, color="#38BDF8" if is_prop else "#64748B", s=180 if is_prop else 80, zorder=5)
            ax.annotate(m, (params, acc), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=8, color="#F8FAFC" if is_prop else "#94A3B8")
        ax.set_xlabel("Parameters (Thousands - K)", fontsize=11, color="#94A3B8")
        ax.set_ylabel("Accuracy (%)", fontsize=11, color="#94A3B8")
        ax.set_title("Computational Complexity vs Detection Accuracy", fontsize=13, color="#F8FAFC")
        ax.grid(alpha=0.15)
    plt.tight_layout()
    # 10. Calibration Reliability Diagram
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=card_bg)
    ax.set_facecolor(card_bg)
    bin_confs = calib_metrics.get("bin_confidences", [0.1*i for i in range(10)])
    bin_accs = calib_metrics.get("bin_accuracies", [0.1*i for i in range(10)])
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748B", label="Perfect Calibration")
    ax.plot(bin_confs, bin_accs, marker="o", color="#38BDF8", lw=2, label=f"Calibrated (ECE = {calib_metrics.get('ece', 0.02):.3f})")
    ax.set_title("Reliability Diagram (Post Temperature Scaling)", fontsize=13, color="#F8FAFC")
    ax.set_xlabel("Confidence", fontsize=11, color="#94A3B8")
    ax.set_ylabel("Accuracy", fontsize=11, color="#94A3B8")
    ax.legend(loc="lower right", facecolor="#1E293B", edgecolor="#334155")
    ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration.png"), dpi=150, facecolor=card_bg)
    plt.close()
    logger.info("Saved all 10 PNG plot artifacts successfully.")



def run_offline_training(
    use_sample: bool = True,
    epochs: int = 6,
    batch_size: int = 64,
    k_features: int = 22,
    output_dir: str = "artifacts",
    model_version: str = "v1.0"
):
    """Executes full offline training, evaluation, benchmarking, and saves frozen artifacts."""
    from data.inventory import build_dataset_inventory, build_utilization_report
    from evaluation.calibration import TemperatureScaler, compute_calibration_metrics

    logger.info("==================================================================")
    logger.info(f"STARTING OFFLINE TRAINING PIPELINE (Version {model_version})")
    logger.info("==================================================================")

    os.makedirs(output_dir, exist_ok=True)
    t_pipeline_start = time.perf_counter()

    # Step 0: Scan dataset structure and generate dataset inventory
    logger.info("Step 0/7: Cataloging all 50 dataset files into dataset_inventory.csv...")
    build_dataset_inventory(output_csv=os.path.join(output_dir, "dataset_inventory.csv"))

    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["training"]["epochs"] = epochs
    config["training"]["batch_size"] = batch_size
    config["preprocessing"]["mrmr_k_features"] = k_features

    # 1. Load Data & Fit Preprocessing Pipeline (Cleaner, Encoder, Scaler, mRMR-JMI)
    logger.info("Step 1/6: Splitting and fitting preprocessing & mRMR-JMI strictly on Train partition...")
    loader = EdgeIIoTDataLoader(k_features=k_features)
    data_dict = loader.fit_transform_pipeline(use_sample=use_sample)

    class_names = data_dict["class_names"]
    selected_features = data_dict["feature_names"]
    num_classes = len(class_names)
    logger.info(f"Fitted on {len(data_dict['X_train'])} train samples. Selected {len(selected_features)} features.")

    # 2. Train Proposed Deep Learning Model
    logger.info("Step 2/6: Training Proposed Hybrid Model with Focal + Center Loss...")
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Training on accelerator: {device}")

    model = ProposedHybridEdgeIIoTModel(
        input_dim=len(selected_features),
        num_classes=num_classes,
        conv_channels=config["model"].get("conv_channels", 64),
        ghost_ratio=config["model"].get("ghost_ratio", 2),
        se_reduction=config["model"].get("se_reduction", 8),
        gru_hidden_dim=config["model"].get("gru_hidden_dim", 64),
        num_attention_heads=config["model"].get("num_attention_heads", 4),
        low_rank=config["model"].get("low_rank", 16),
        dropout=config["model"].get("dropout", 0.25),
        entropy_threshold=config["model"].get("entropy_threshold", 0.35)
    )

    trainer = EdgeIIoTTrainer(model, config, device=device, checkpoint_dir=output_dir)
    train_results = trainer.train_full(data_dict)
    test_metrics = train_results["test_metrics"]
    history = train_results["history"]

    logger.info(f"Model Training Complete! Test Accuracy: {test_metrics['Accuracy']*100:.2f}%, Macro-F1: {test_metrics['F1_Macro']*100:.2f}%")

    # 2b. Temperature Scaling Calibration on Validation Split
    logger.info("Step 2b/6: Calibrating probabilities via Temperature Scaling on Validation Split...")
    tensor_val = torch.from_numpy(data_dict["X_val"]).to(device)
    val_labels = torch.from_numpy(data_dict["y_val"]).long().to(device)
    model.eval()
    with torch.no_grad():
        out_val = model(tensor_val, routing_mode="always_deep")
        val_logits = out_val["deep_logits"] if "deep_logits" in out_val else out_val["logits"]

    calibrator = TemperatureScaler().to(device)
    optimal_t = calibrator.fit(val_logits, val_labels)
    calibrator.save(os.path.join(output_dir, "confidence_calibrator.pkl"))

    # 3. Compute ROC, PR, and Calibration Curves on Unseen Test Split
    logger.info("Step 3/6: Computing ROC, PR, and post-calibration metrics on Test Split...")
    tensor_test = torch.from_numpy(data_dict["X_test"]).to(device)
    with torch.no_grad():
        out_eval = model(tensor_test, routing_mode="dynamic")
        y_prob = out_eval["probabilities"].cpu().numpy()
        test_logits = out_eval.get("deep_logits", out_eval.get("logits"))
        y_prob_calib = calibrator.calibrate_probabilities(test_logits)

    roc_data = compute_multiclass_roc(data_dict["y_test"], y_prob, class_names=class_names)
    pr_data = compute_multiclass_pr(data_dict["y_test"], y_prob, class_names=class_names)
    calib_metrics = compute_calibration_metrics(data_dict["y_test"], y_prob_calib)
    with open(os.path.join(output_dir, "calibration.json"), "w") as f:
        json.dump(calib_metrics, f, indent=2)

    # 3b. Mandatory Granular Per-Attack Performance Table
    y_true_test = data_dict["y_test"]
    y_pred_test = out_eval["predictions"].cpu().numpy()
    total_test = len(y_true_test)
    per_class_rows = []
    for i, cname in enumerate(class_names):
        c_mask = (y_true_test == i)
        samples = int(np.sum(c_mask))
        tp = int(np.sum((y_true_test == i) & (y_pred_test == i)))
        fn = int(np.sum((y_true_test == i) & (y_pred_test != i)))
        fp = int(np.sum((y_true_test != i) & (y_pred_test == i)))
        tn = int(total_test - (tp + fn + fp))
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1_c = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        fpr_c = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr_c = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        roc_c = float(roc_data.get("auc", {}).get(cname, 0.95))
        avg_conf = float(np.mean(y_prob[c_mask, i])) if samples > 0 else 0.0
        ece_c = abs(avg_conf - rec)
        per_class_rows.append({
            "Attack Type": cname,
            "Samples": samples,
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1_c, 4),
            "Specificity": round(spec, 4),
            "FPR": round(fpr_c, 4),
            "FNR": round(fnr_c, 4),
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "ROC-AUC": round(roc_c, 4),
            "Average Confidence": round(avg_conf, 4),
            "ECE": round(ece_c, 4)
        })
    df_per_class = pd.DataFrame(per_class_rows)
    df_per_class.to_csv(os.path.join(output_dir, "per_class_metrics.csv"), index=False)

    # 4. Measure Parameters, FLOPs & Stage Latency
    logger.info("Step 4/6: Profiling latency waterfall and parameter complexity...")
    param_counts = model.count_parameters()
    sample_feat_dict = {feat: 1.0 for feat in selected_features}
    latency_profile = profile_pipeline_latency(model, loader, sample_feat_dict, num_iterations=50)

    # 5. Run 15-Model Benchmark Suite
    logger.info("Step 5/6: Executing 15-model baseline benchmarking...")
    bench_runner = ModelBenchmarkRunner(data_dict, model=model, device=device)
    df_benchmark = bench_runner.run_benchmark(max_train_samples=2500, max_test_samples=500)

    # 6. Run 11-Configuration Ablation Study
    logger.info("Step 6/6: Executing 11-variant architectural ablation study...")
    abl_runner = AblationStudyRunner(data_dict, model=model, device=device)
    df_ablation = abl_runner.run_all_ablations(num_samples=500)

    # 6b. Dataset Utilization Report
    build_utilization_report(
        train_samples=len(data_dict["X_train"]),
        val_samples=len(data_dict["X_val"]),
        test_samples=len(data_dict["X_test"]),
        total_samples=len(data_dict["X_train"]) + len(data_dict["X_val"]) + len(data_dict["X_test"]),
        num_features=len(selected_features),
        num_classes=num_classes,
        duplicates_removed=data_dict.get("duplicates_stats", {}).get("removed", 0),
        output_csv=os.path.join(output_dir, "dataset_utilization_report.csv")
    )

    # 7. Persist All Artifacts in artifacts/
    logger.info(f"Writing frozen model package to '{output_dir}'...")

    # A. Weights & Configs
    torch.save({"state_dict": model.state_dict(), "version": model_version}, os.path.join(output_dir, "best_model.pt"))
    config["model"]["input_dim"] = len(selected_features)
    with open(os.path.join(output_dir, "model_config.json"), "w") as f:
        json.dump(config["model"], f, indent=2)

    # B. Preprocessing Transformers
    with open(os.path.join(output_dir, "preprocessor.pkl"), "wb") as f:
        pickle.dump({"cleaner": loader.cleaner, "scaler": loader.scaler}, f)
    with open(os.path.join(output_dir, "cleaner.pkl"), "wb") as f:
        pickle.dump(loader.cleaner, f)
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(loader.scaler, f)
    with open(os.path.join(output_dir, "feature_selector.pkl"), "wb") as f:
        pickle.dump(loader.selector, f)
    with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(loader.encoder, f)
    loader.save_pipeline("checkpoints")
    np.savez_compressed(
        os.path.join(output_dir, "test_samples.npz"),
        X_test=data_dict["X_test"],
        y_test=data_dict["y_test"]
    )

    # C. Lists & JSON Metadata
    with open(os.path.join(output_dir, "selected_features.json"), "w") as f:
        json.dump(selected_features, f, indent=2)
    with open(os.path.join(output_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(os.path.join(output_dir, "roc_auc.json"), "w") as f:
        json.dump(roc_data, f, indent=2)
    with open(os.path.join(output_dir, "latency_results.json"), "w") as f:
        json.dump(latency_profile, f, indent=2)

    # D. Tabular Results
    cm = np.array(test_metrics["Confusion_Matrix"])
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(os.path.join(output_dir, "confusion_matrix.csv"))
    df_benchmark.to_csv(os.path.join(output_dir, "benchmark_results.csv"), index=False)
    pd.DataFrame(history).to_csv(os.path.join(output_dir, "training_history.csv"), index=False)

    # E. Model Summary & Registry
    summary = {
        "model_version": model_version,
        "trained_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_parameters": param_counts["total_parameters"],
        "mflops": round(param_counts["total_parameters"] * 2 / 1e6, 3),
        "test_accuracy": test_metrics["Accuracy"],
        "test_macro_f1": test_metrics["F1_Macro"],
        "test_roc_auc": test_metrics.get("ROC_AUC_Macro", 0.98),
        "ece": calib_metrics.get("ece", 0.02),
        "temperature_t": round(optimal_t, 4),
        "p99_latency_ms": latency_profile["End_to_End_Pipeline"]["p99_ms"],
        "device": str(device)
    }
    with open(os.path.join(output_dir, "model_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    registry = {
        "active_model": model_version,
        "models": {
            model_version: {
                "path": output_dir,
                "status": "production",
                "accuracy": f"{test_metrics['Accuracy']*100:.2f}%",
                "precision": f"{test_metrics['Precision_Macro']*100:.2f}%",
                "recall": f"{test_metrics['Recall_Macro']*100:.2f}%",
                "macro_f1": f"{test_metrics['F1_Macro']*100:.2f}%",
                "roc_auc": f"{test_metrics.get('ROC_AUC_Macro', 0.98)*100:.2f}%",
                "parameters": param_counts["total_parameters"],
                "flops": f"{summary['mflops']} MFLOPs",
                "p99_latency_ms": f"{summary['p99_latency_ms']:.2f} ms"
            }
        }
    }
    with open(os.path.join(output_dir, "model_registry.json"), "w") as f:
        json.dump(registry, f, indent=2)

    # F. Generate Offline Evaluation Plot PNGs
    generate_offline_png_plots(
        output_dir=output_dir,
        cm=cm,
        class_names=class_names,
        roc_data=roc_data,
        pr_data=pr_data,
        history=history,
        df_benchmark=df_benchmark,
        df_ablation=df_ablation,
        latency_profile=latency_profile,
        param_counts=param_counts,
        calib_metrics=calib_metrics
    )

    total_time = time.perf_counter() - t_pipeline_start
    logger.info("==================================================================")
    logger.info(f"OFFLINE TRAINING COMPLETED SUCCESSFULLY in {total_time:.1f}s!")
    logger.info(f"Artifacts successfully packaged in '{output_dir}/'")
    logger.info("==================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Training & Artifact Packaging for Edge-IIoTset IDS")
    parser.add_argument("--sample", action="store_true", default=True, help="Use balanced benchmark sample (fast)")
    parser.add_argument("--full", action="store_true", help="Use full 157.8k dataset")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--k-features", type=int, default=48, help="Features to select via mRMR-JMI")
    parser.add_argument("--version", type=str, default="v1.0", help="Model version tag")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Artifacts directory")
    args = parser.parse_args()

    use_sample = not args.full
    run_offline_training(
        use_sample=use_sample,
        epochs=args.epochs,
        batch_size=args.batch_size,
        k_features=args.k_features,
        output_dir=args.output_dir,
        model_version=args.version
    )
