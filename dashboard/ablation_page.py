"""
Page 23: Architectural Ablation Study.
Rigorous empirical evaluation isolating the individual performance contribution of each design component.
Reads directly from compiled offline artifacts in artifacts/.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px

from evaluation.ablation import AblationStudyRunner


def render_ablation_page():
    st.title("🔬 23. Architectural Ablation Experiments")
    st.caption("Systematic component ablation validating the theoretical necessity of mRMR-JMI, Ghost, Depthwise CNN, SE, Early-Exit, Bi-GRU, Attention, and Low-Rank heads.")

    ablation_df = AblationStudyRunner.load_saved_results()

    if ablation_df is None or ablation_df.empty:
        # Check if saved in experiments/ or artifacts/
        if os.path.exists("artifacts/ablation_results.csv"):
            ablation_df = pd.read_csv("artifacts/ablation_results.csv")
        elif os.path.exists("experiments/ablation_results.json"):
            ablation_df = pd.read_json("experiments/ablation_results.json")

    c_btn, c_txt = st.columns([1, 2])
    with c_btn:
        png_ablation = "artifacts/ablation.png"
        if os.path.exists(png_ablation):
            st.success("✅ Offline Ablation Artifacts Active")

    with c_txt:
        if ablation_df is not None:
            st.success(f"Ablation results loaded with {len(ablation_df)} empirical configurations.")

    if ablation_df is None or ablation_df.empty:
        st.info("No precomputed ablation results found in `artifacts/ablation_results.csv`. Run `python train.py` to compile all study results.")
        return

    st.markdown("---")
    st.subheader("📊 F1-Score vs Latency vs Parameter Impact")

    col1, col2 = st.columns(2)
    var_col = "Ablation_Variant" if "Ablation_Variant" in ablation_df.columns else ("Ablation Step" if "Ablation Step" in ablation_df.columns else ablation_df.columns[0])
    f1_col = "F1_Macro" if "F1_Macro" in ablation_df.columns else ("Macro F1" if "Macro F1" in ablation_df.columns else None)
    lat_col = "Latency_ms" if "Latency_ms" in ablation_df.columns else ("Latency (ms)" if "Latency (ms)" in ablation_df.columns else None)

    with col1:
        if f1_col:
            fig_f1 = px.bar(
                ablation_df.sort_values(by=f1_col, ascending=True),
                x=f1_col, y=var_col,
                orientation="h",
                color=f1_col,
                color_continuous_scale="Tealgrn",
                title="Macro-F1 Score Degradation under Component Removal"
            )
            fig_f1.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#1E293B", font=dict(color="#F8FAFC"))
            st.plotly_chart(fig_f1, use_container_width=True)

    with col2:
        if "Parameters" in ablation_df.columns and lat_col:
            fig_params = px.bar(
                ablation_df.sort_values(by="Parameters", ascending=True),
                x="Parameters", y=var_col,
                orientation="h",
                color=lat_col,
                color_continuous_scale="Plasma",
                title="Model Parameter Footprint & Latency (ms)"
            )
            fig_params.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#1E293B", font=dict(color="#F8FAFC"))
            st.plotly_chart(fig_params, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Comprehensive Empirical Ablation Table")

    max_cols = [c for c in ["Accuracy", "F1_Macro", "Macro F1", "ROC_AUC"] if c in ablation_df.columns]
    min_cols = [c for c in ["Latency_ms", "Latency (ms)", "Parameters", "MFLOPs", "FLOPs_M", "FPR"] if c in ablation_df.columns]

    styler = ablation_df.style
    if max_cols:
        styler = styler.highlight_max(subset=max_cols, color="#065F46")
    if min_cols:
        styler = styler.highlight_min(subset=min_cols, color="#1E3A8A")

    st.dataframe(styler, use_container_width=True)

    png_path = "artifacts/ablation.png"
    if os.path.exists(png_path):
        st.markdown("---")
        st.subheader("Publication-Ready Ablation Breakdown Artifact")
        st.image(png_path, caption="Publication Ablation Summary (artifacts/ablation.png)", use_container_width=True)

    st.download_button(
        "📥 Download Ablation Study CSV",
        data=ablation_df.to_csv(index=False),
        file_name="edge_iiot_ablation_results.csv",
        mime="text/csv"
    )
