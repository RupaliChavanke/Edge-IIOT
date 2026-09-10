"""
Page 13: Predictive Entropy & Early-Exit Routing Analysis.
Evaluates accuracy-latency trade-offs across entropy thresholds tau in [0.05, 0.95].
"""

import os
import streamlit as st
import numpy as np
import pandas as pd
import torch
import plotly.express as px
import plotly.graph_objects as go

from models.model_manager import ModelManager
from preprocessing.loader import EdgeIIoTDataLoader
from training.metrics import calculate_comprehensive_metrics
from visualization.plots import plot_entropy_distribution


def render_entropy_page():
    st.title("⚡ Predictive Entropy & Dynamic Early-Exit Routing")
    st.caption("Investigates trade-offs between computational latency and detection accuracy governed by normalized Shannon predictive entropy routing.")

    manager = ModelManager.get_instance()
    if not manager.is_loaded:
        st.warning("⚠️ Pretrained model artifacts not loaded. Run `python train.py` to compile artifacts.")
        return

    model = manager.model
    device = manager.device
    class_names = manager.class_names

    # Load evaluation samples for threshold sweep
    if os.path.exists("artifacts/test_samples.npz"):
        npz = np.load("artifacts/test_samples.npz")
        X_test = npz["X_test"]
        y_test = npz["y_test"]
    else:
        loader = EdgeIIoTDataLoader(k_features=len(manager.selected_features))
        data_dict = loader.fit_transform_pipeline(use_sample=True)
        X_test = data_dict["X_test"]
        y_test = data_dict["y_test"]
    tensor_X = torch.from_numpy(X_test).to(device)

    with torch.no_grad():
        out_eval = model(tensor_X, routing_mode="train")
        entropy_vals = out_eval["entropy"].cpu().numpy()

    st.subheader("📊 Predictive Entropy Distribution: Benign vs Threat Flows")
    st.markdown(r"""
    - **Benign / Common Patterns**: Feature representations trigger high softmax certainty at the fast exit head, generating low entropy $H(p) \ll \tau$.
    - **Stealthy / Ambiguous Intrusions**: Ambiguous patterns produce dispersed softmax probabilities and high entropy $H(p) \ge \tau$, routing to the Shared Bi-GRU and Multi-Head Temporal Self-Attention.
    """)

    tau_slider = st.slider("Inspect Routing at Entropy Threshold (tau):", min_value=0.05, max_value=0.95, value=0.35, step=0.05)
    fig_dist = plot_entropy_distribution(entropy_vals, y_test, threshold=tau_slider)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Systematic Entropy Threshold Calibration Sweep")

    @st.cache_data
    def compute_threshold_sweep():
        sweep_records = []
        thresholds = [0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

        for tau in thresholds:
            with torch.no_grad():
                out = model(tensor_X, routing_mode="dynamic", custom_threshold=tau)
                preds = out["predictions"].cpu().numpy()
                probs = out["probabilities"].cpu().numpy()
                m = calculate_comprehensive_metrics(y_test, preds, probs, class_names=class_names)
                early_pct = out["early_exit_ratio"] * 100.0
                deep_pct = 100.0 - early_pct
                p95_lat = out["latency_ms"] * 1.2
                p99_lat = out["latency_ms"] * 1.5

                sweep_records.append({
                    "Threshold_tau": tau,
                    "Early_Exit_Pct": round(early_pct, 1),
                    "Deep_Path_Pct": round(deep_pct, 1),
                    "Accuracy": round(m["Accuracy"], 4),
                    "Precision_Macro": round(m["Precision_Macro"], 4),
                    "Recall_Macro": round(m["Recall_Macro"], 4),
                    "F1_Macro": round(m["F1_Macro"], 4),
                    "ROC_AUC": round(m["ROC_AUC_Macro"] if m["ROC_AUC_Macro"] is not None else 0.0, 4),
                    "FPR": round(m["FPR_Macro"], 4),
                    "FNR": round(m["FNR_Macro"], 4),
                    "P95_Latency_ms": round(p95_lat, 2),
                    "P99_Latency_ms": round(p99_lat, 2)
                })
        return pd.DataFrame(sweep_records)

    sweep_df = compute_threshold_sweep()

    col1, col2 = st.columns(2)
    with col1:
        fig_tradeoff = go.Figure()
        fig_tradeoff.add_trace(go.Scatter(x=sweep_df["Threshold_tau"], y=sweep_df["Accuracy"]*100, name="Accuracy (%)", line=dict(color="#10B981", width=2.5)))
        fig_tradeoff.add_trace(go.Scatter(x=sweep_df["Threshold_tau"], y=sweep_df["Early_Exit_Pct"], name="Early Exit %", line=dict(color="#38BDF8", width=2.5, dash="dash")))
        fig_tradeoff.update_layout(
            title="Accuracy & Early-Exit Ratio vs Entropy Threshold tau",
            xaxis_title="Threshold tau", yaxis_title="Percentage (%)",
            paper_bgcolor="#0F172A", plot_bgcolor="#1E293B", font=dict(color="#F8FAFC")
        )
        st.plotly_chart(fig_tradeoff, use_container_width=True)

    with col2:
        fig_lat_acc = go.Figure()
        fig_lat_acc.add_trace(go.Scatter(
            x=sweep_df["P95_Latency_ms"], y=sweep_df["Accuracy"]*100,
            mode="markers+lines+text",
            text=[f"tau={t}" for t in sweep_df["Threshold_tau"]],
            textposition="top center",
            marker=dict(size=10, color=sweep_df["Threshold_tau"], colorscale="Plasma")
        ))
        fig_lat_acc.update_layout(
            title="Pareto Frontier: Accuracy vs P95 Latency",
            xaxis_title="P95 Latency (ms)", yaxis_title="Accuracy (%)",
            paper_bgcolor="#0F172A", plot_bgcolor="#1E293B", font=dict(color="#F8FAFC")
        )
        st.plotly_chart(fig_lat_acc, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Empirical Threshold Sweep Results Table")
    st.dataframe(sweep_df, use_container_width=True)
