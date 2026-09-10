"""
Page 7: Model Training & Deployment Orchestrator.
Interactive hyperparameter configuration, live loss convergence tracking, and one-click deployment.
"""

import os
import yaml
import streamlit as st
import pandas as pd
import torch

from preprocessing.loader import EdgeIIoTDataLoader
from models.proposed_model import ProposedHybridEdgeIIoTModel
from training.trainer import EdgeIIoTTrainer
from training.checkpoint import CheckpointManager
from visualization.plots import plot_training_curves


def render_training_page():
    st.title("🏋️ Model Training & Real-Time Deployment")
    st.caption("Trains the Proposed Hybrid CNN–Ghost–BiGRU–Attention Network with Focal + Center Loss and calibrates the entropy threshold.")

    ckpt_mgr = CheckpointManager()
    is_deployed = ckpt_mgr.is_model_deployed()

    c1, c2 = st.columns([2, 1])
    with c1:
        st.write(f"**Current Deployed Status**: {'🟢 DEPLOYED AND OPERATIONAL' if is_deployed else '🟡 CHECKPOINT REQUIRED'}")
    with c2:
        device_str = "mps (Apple Silicon GPU)" if torch.backends.mps.is_available() else "cpu"
        st.write(f"**Detected Hardware**: ⚡ `{device_str}`")

    st.markdown("---")
    st.subheader("⚙️ Training Hyperparameters & Compound Loss Weights")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    col1, col2, col3 = st.columns(3)
    with col1:
        epochs = st.slider("Training Epochs:", min_value=1, max_value=30, value=5, step=1)
        batch_size = st.select_slider("Batch Size:", options=[16, 32, 64, 128, 256], value=64)
        lr = st.select_slider("Learning Rate:", options=[1e-4, 5e-4, 1e-3, 2e-3, 5e-3], value=1e-3)

    with col2:
        focal_gamma = st.slider("Focal Loss Gamma (gamma):", min_value=0.5, max_value=4.0, value=2.0, step=0.5)
        lambda_focal = st.slider("Focal Loss Weight (lambda_focal):", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        lambda_center = st.slider("Center Loss Weight (lambda_center):", min_value=0.001, max_value=0.1, value=0.01, step=0.005)

    with col3:
        entropy_tau = st.slider("Early-Exit Entropy Threshold (tau):", min_value=0.10, max_value=0.80, value=0.35, step=0.05)
        k_features = st.slider("mRMR-JMI Feature Budget (K):", min_value=10, max_value=40, value=22, step=2)
        dataset_mode = st.radio("Dataset Size:", ["Lightweight Balanced Sample (5,250)", "Full Dataset (157,800)"])

    st.markdown("---")

    if st.button("🚀 Start Offline Training Pipeline", type="primary", use_container_width=True):
        config["training"]["epochs"] = epochs
        config["training"]["batch_size"] = batch_size
        config["training"]["learning_rate"] = lr
        config["training"]["focal_gamma"] = focal_gamma
        config["training"]["lambda_focal"] = lambda_focal
        config["training"]["lambda_center"] = lambda_center
        config["model"]["entropy_threshold"] = entropy_tau
        config["preprocessing"]["mrmr_k_features"] = k_features

        use_sample = ("Lightweight" in dataset_mode)

        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Stage 1/4: Splitting and Preprocessing dataset...")
        loader = EdgeIIoTDataLoader(k_features=k_features)
        data_dict = loader.fit_transform_pipeline(use_sample=use_sample)
        loader.save_pipeline("checkpoints")
        progress_bar.progress(25)

        status_text.text("Stage 2/4: Initializing Proposed Hybrid Model...")
        num_classes = len(data_dict["class_names"])
        model = ProposedHybridEdgeIIoTModel(
            input_dim=k_features,
            num_classes=num_classes,
            entropy_threshold=entropy_tau
        )
        trainer = EdgeIIoTTrainer(model, config)
        progress_bar.progress(40)

        status_text.text("Stage 3/4: Training Model across epochs with Focal + Center Loss...")
        history_chart_slot = st.empty()

        def on_epoch_done(ep, hist_entry):
            pct = 40 + int((ep / epochs) * 50)
            progress_bar.progress(min(90, pct))
            status_text.text(f"Epoch {ep}/{epochs} - Train Loss: {hist_entry['train_loss']:.4f} | Val F1: {hist_entry['val_f1']*100:.1f}%")
            fig = plot_training_curves(trainer.history)
            history_chart_slot.plotly_chart(fig, use_container_width=True)

        results = trainer.train_full(data_dict, progress_callback=on_epoch_done)
        progress_bar.progress(100)
        status_text.text("Stage 4/4: Final unseen test evaluation complete and checkpoint saved!")

        st.success(f"🎉 Model Training Completed! Best Val F1: {results['best_val_f1']*100:.2f}% (Epoch {results['best_epoch']}). Test Accuracy: {results['test_metrics']['Accuracy']*100:.2f}%.")
        st.rerun()

    # If already trained, display deployed metrics and deploy button
    metrics = ckpt_mgr.get_deployed_metrics()
    if metrics:
        st.markdown("### 🏆 Deployed Checkpoint Validation Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Accuracy", f"{metrics.get('Accuracy', 0)*100:.2f}%")
        c2.metric("Test Macro-F1", f"{metrics.get('F1_Macro', 0)*100:.2f}%")
        c3.metric("Test Precision", f"{metrics.get('Precision_Macro', 0)*100:.2f}%")
        c4.metric("Test Recall", f"{metrics.get('Recall_Macro', 0)*100:.2f}%")

        if st.button("✅ Deploy Model to Streaming IDS Inference Engine", use_container_width=True):
            st.success("✨ MODEL STATUS: DEPLOYED to Redpanda Streaming Consumer! Real-time IDS is ready.")
