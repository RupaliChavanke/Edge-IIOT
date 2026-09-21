"""
Page 9: Scientific Model Benchmarking.
Comparative evaluation of Proposed Hybrid Architecture against 15 Machine Learning and Deep Learning Baselines.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px

from evaluation.evaluator import ModelBenchmarkRunner, BENCHMARK_RESULTS_FILE
from preprocessing.loader import EdgeIIoTDataLoader
from visualization.plots import plot_model_radar_comparison


def render_benchmark_page():
    st.title("⚖️ Comprehensive Algorithm Benchmarking")
    st.caption("Empirical benchmark comparing Proposed Hybrid Model against 15 baseline algorithms on identical unseen test splits.")

    benchmark_df = ModelBenchmarkRunner.load_saved_benchmark()

    c_run, c_info = st.columns([1, 2])
    with c_run:
        if st.button("🚀 Run Live Benchmark Suite (15 Models)", type="primary", use_container_width=True):
            with st.spinner("Executing training and latency profiling for all 15 algorithms..."):
                loader = EdgeIIoTDataLoader()
                if os.path.exists("checkpoints/cleaner.pkl"):
                    loader.load_pipeline("checkpoints")
                    data_dict = loader.fit_transform_pipeline(use_sample=True)
                else:
                    data_dict = loader.fit_transform_pipeline(use_sample=True)
                
                runner = ModelBenchmarkRunner(data_dict)
                benchmark_df = runner.run_benchmark()
                st.success("Benchmark completed and logged!")
                st.rerun()

    with c_info:
        if benchmark_df is not None:
            st.success(f"Loaded benchmark dataset with {len(benchmark_df)} algorithms evaluated across 21 scientific metrics.")

    if benchmark_df is None:
        st.info("Click 'Run Live Benchmark Suite' above to benchmark all 15 models.")
        return

    st.markdown("---")
    st.subheader("📊 Primary Accuracy vs F1 vs Latency Comparison")

    col1, col2 = st.columns(2)
    with col1:
        fig_bar = px.bar(
            benchmark_df.sort_values(by="F1_Macro", ascending=True),
            x="F1_Macro", y="Model",
            orientation="h",
            color="F1_Macro",
            color_continuous_scale="Viridis",
            title="Macro-F1 Score Comparison across Models"
        )
        fig_bar.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#1E293B", font=dict(color="#F8FAFC"))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_lat = px.scatter(
            benchmark_df,
            x="P95_Latency_ms", y="Accuracy",
            size="Parameters", color="Model",
            hover_name="Model",
            title="Accuracy vs P95 Inference Latency Trade-Off (Bubble Size = Parameters)"
        )
        fig_lat.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#1E293B", font=dict(color="#F8FAFC"))
        st.plotly_chart(fig_lat, use_container_width=True)

    st.markdown("---")
    st.subheader("🕸️ Multi-Criteria Efficiency Radar Chart")
    st.plotly_chart(plot_model_radar_comparison(benchmark_df), use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Complete Scientific Benchmark Comparison Table")
    max_cols = [c for c in ["Accuracy", "F1_Macro", "ROC_AUC", "MCC"] if c in benchmark_df.columns]
    min_cols = [c for c in ["P95_Latency_ms", "P95 Latency (ms)", "FPR", "Parameters"] if c in benchmark_df.columns]

    styler = benchmark_df.style
    if max_cols:
        styler = styler.highlight_max(subset=max_cols, color="#065F46")
    if min_cols:
        styler = styler.highlight_min(subset=min_cols, color="#1E3A8A")

    st.dataframe(styler, use_container_width=True)

    st.download_button(
        "📥 Download Full Benchmark Results CSV",
        data=benchmark_df.to_csv(index=False),
        file_name="edge_iiot_benchmark_results.csv",
        mime="text/csv"
    )
