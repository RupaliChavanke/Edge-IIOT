"""
Page 10: Multidimensional Confusion Matrix Analysis.
Interactive raw, row-normalized, and column-normalized heatmaps with One-vs-Rest metrics and export options.
"""

import streamlit as st
import numpy as np
import pandas as pd

from training.checkpoint import CheckpointManager
from visualization.confusion import plot_confusion_matrix_heatmap
from evaluation.confusion import ConfusionMatrixAnalyzer


def render_confusion_page():
    st.title("🧮 Multiclass Confusion Matrix & Error Heatmaps")
    st.caption("Detailed breakdown of true positives, false alarms, and inter-class confusion patterns.")

    ckpt_mgr = CheckpointManager()
    metrics = ckpt_mgr.get_deployed_metrics()

    if not metrics or "Confusion_Matrix" not in metrics:
        st.warning("⚠️ No confusion matrix data found. Please evaluate or train a model checkpoint first.")
        return

    cm_raw = np.array(metrics["Confusion_Matrix"])
    class_names = [p["Class"] for p in metrics.get("Per_Class", [])]
    if not class_names:
        class_names = [f"Class_{i}" for i in range(len(cm_raw))]

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        mode = st.radio("Display Representation:", ["Raw Sample Counts", "Normalized (Recall % per Class)"], horizontal=True)
    with col_ctrl2:
        export_format = st.selectbox("Export Format:", ["CSV", "JSON"])

    st.markdown("---")
    cm_mode = "normalized" if "Normalized" in mode else "raw"
    fig_cm = plot_confusion_matrix_heatmap(cm_raw, class_names=class_names, mode=cm_mode)
    st.plotly_chart(fig_cm, use_container_width=True)

    # Downloads
    df_cm = pd.DataFrame(cm_raw, index=class_names, columns=class_names)
    if export_format == "CSV":
        st.download_button("📥 Download Confusion Matrix CSV", df_cm.to_csv(), "confusion_matrix.csv", "text/csv")
    else:
        st.download_button("📥 Download Confusion Matrix JSON", df_cm.to_json(), "confusion_matrix.json", "application/json")

    st.markdown("---")
    st.subheader("🔍 Class-Specific One-vs-Rest (OvR) Confusion Breakdown")

    selected_class = st.selectbox("Select Attack Category to Inspect:", class_names)
    pc_data = metrics.get("Per_Class", [])
    class_stat = next((item for item in pc_data if item["Class"] == selected_class), None)

    if class_stat:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("True Positives (TP)", f"{class_stat['TP']:,}")
        c2.metric("True Negatives (TN)", f"{class_stat['TN']:,}")
        c3.metric("False Positives (FP)", f"{class_stat['FP']:,}")
        c4.metric("False Negatives (FN)", f"{class_stat['FN']:,}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Precision", f"{class_stat['Precision']*100:.2f}%")
        c6.metric("Recall / Detection Rate", f"{class_stat['Recall']*100:.2f}%")
        c7.metric("Specificity", f"{class_stat['Specificity']*100:.2f}%")
        c8.metric("F1-Score", f"{class_stat['F1']*100:.2f}%")
