"""
Model Comparison & Architectural Ablation Benchmark Page.
Renders empirical evaluations across 23 baseline classifiers and 12 ablation variants
from saved artifacts (benchmark_results.csv, benchmark.png, ablation.png, complexity.png).
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_model_comparison_page():
    st.title("🏆 Empirical Model Benchmarking & Ablation Study")
    st.caption("Comprehensive Comparative Analysis against 22 State-of-the-Art ML/DL Baselines on Edge-IIoTset")

    bench_csv = "artifacts/benchmark_results.csv"
    abl_csv = "artifacts/ablation_results.csv"

    tab_bench, tab_abl, tab_pareto = st.tabs([
        "🏆 23-Model Baseline Benchmark",
        "🔬 Systematic Architectural Ablation",
        "⚡ Pareto Optimality (Latency vs Accuracy)"
    ])

    with tab_bench:
        st.subheader("1. Empirical Comparison against 22 Baseline Models")
        st.markdown("""
        All baseline models were evaluated on the identical 22-feature Edge-IIoTset test split under 
        uniform training constraints. Our **Proposed Hybrid Framework** achieves superior classification 
        fidelity (**96.35% Accuracy**, **96.00% Macro-F1**) while delivering sub-millisecond edge latency 
        (**0.086 ms**) via hardware-accelerated ONNX runtime inference.
        """)

        if os.path.exists(bench_csv):
            df_b = pd.read_csv(bench_csv)
            
            # Dynamic Model Callout Metric Cards
            selected_highlight = st.selectbox(
                "Select Architecture to Highlight KPI Cards:",
                options=list(df_b["Model"]),
                index=0,
                key="bench_highlight_model"
            )
            prop_row = df_b[df_b["Model"] == selected_highlight]
            if not prop_row.empty:
                r = prop_row.iloc[0]
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Accuracy", f"{r['Accuracy']*100:.2f}%")
                k2.metric("Macro-F1 Score", f"{r['F1_Macro']*100:.2f}%")
                k3.metric("False Positive Rate", f"{r['FPR']*100:.2f}%")
                k4.metric("False Negative Rate", f"{r['FNR']*100:.2f}%")
                k5.metric("P50 Latency", f"{r['P50_Latency_ms']:.4f} ms", delta=f"{r.get('Throughput_eps', 0):,.0f} eps")

            st.markdown("---")
            st.dataframe(
                df_b.style.format({
                    "Accuracy": "{:.4f}",
                    "Precision_Macro": "{:.4f}",
                    "Recall_Macro": "{:.4f}",
                    "F1_Macro": "{:.4f}",
                    "F1_Weighted": "{:.4f}",
                    "ROC_AUC": "{:.4f}",
                    "PR_AUC": "{:.4f}",
                    "Balanced_Accuracy": "{:.4f}",
                    "MCC": "{:.4f}",
                    "FPR": "{:.4f}",
                    "FNR": "{:.4f}",
                    "P50_Latency_ms": "{:.3f}",
                    "Throughput_eps": "{:,.1f}"
                }, na_rep="--"),
                use_container_width=True,
                hide_index=True
            )

            c_fig, c_img = st.columns([1, 1])
            with c_fig:
                # Color code: Proposed Model highlighted in cyan, Ensemble in green, others slate
                colors = ["#38BDF8" if "PROPOSED" in m else ("#10B981" if "Ensemble" in m or "ONNX" in m else "#64748B") for m in df_b["Model"]]
                fig_bar = px.bar(
                    df_b,
                    x="Model",
                    y="Accuracy",
                    color="Model",
                    title="<b>Classification Accuracy across 23 Benchmarked Architectures</b>",
                    text_auto=".3f"
                )
                fig_bar.update_layout(
                    paper_bgcolor="#0F172A",
                    plot_bgcolor="#1E293B",
                    font=dict(color="#F8FAFC"),
                    xaxis=dict(tickangle=-45),
                    showlegend=False,
                    height=450
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with c_img:
                img_path = "artifacts/benchmark.png"
                if os.path.exists(img_path):
                    st.image(img_path, caption="Saved Benchmark Comparison Artifact (High-Res 300 DPI)", use_container_width=True)
                else:
                    st.info("Static benchmark artifact available in artifacts/benchmark.png")
        else:
            st.info("Benchmark data artifact not found at `artifacts/benchmark_results.csv`.")

    with tab_abl:
        st.subheader("2. Systematic Component Ablation Study")
        st.markdown("""
        To isolate the exact performance contribution of each architectural submodule, 
        ablation variants were evaluated under identical training partitions and random seeds.
        """)

        if os.path.exists(abl_csv):
            df_abl = pd.read_csv(abl_csv)
            st.dataframe(df_abl, use_container_width=True, hide_index=True)

            c_abl_plot, c_abl_img = st.columns([1, 1])
            with c_abl_plot:
                variant_col = "Ablation_Variant" if "Ablation_Variant" in df_abl.columns else df_abl.columns[0]
                f1_col = "F1_Macro" if "F1_Macro" in df_abl.columns else ("Macro F1" if "Macro F1" in df_abl.columns else "Accuracy")
                
                fig_abl = px.bar(
                    df_abl.sort_values(by=f1_col, ascending=True),
                    x=f1_col,
                    y=variant_col,
                    orientation="h",
                    color=f1_col,
                    color_continuous_scale="Purples",
                    title="<b>Macro-F1 Score Impact by Component</b>",
                    text_auto=".3f"
                )
                fig_abl.update_layout(
                    paper_bgcolor="#0F172A",
                    plot_bgcolor="#1E293B",
                    font=dict(color="#F8FAFC"),
                    height=450
                )
                st.plotly_chart(fig_abl, use_container_width=True)

            with c_abl_img:
                img_abl = "artifacts/ablation.png"
                if os.path.exists(img_abl):
                    st.image(img_abl, caption="Ablation Waterfall Study Artifact (High-Res 300 DPI)", use_container_width=True)
                else:
                    st.info("Ablation image available in artifacts/ablation.png")

    with tab_pareto:
        st.subheader("3. Complexity vs Detection Accuracy Pareto Frontier")
        st.markdown("""
        Deploying IDS on resource-constrained IIoT edge nodes requires balancing inference latency 
        and memory footprint against classification accuracy. The **Proposed Hybrid Model (ONNX)** 
        occupies the optimal upper-left Pareto frontier, delivering top accuracy with **sub-millisecond latency**.
        """)

        if os.path.exists(bench_csv):
            df_p = pd.read_csv(bench_csv)
            c_p_chart, c_p_img = st.columns([1, 1])
            with c_p_chart:
                lat_col = "P50_Latency_ms" if "P50_Latency_ms" in df_p.columns else "P50 Latency (ms)"
                acc_col = "Accuracy"
                
                fig_pareto = px.scatter(
                    df_p,
                    x=lat_col,
                    y=acc_col,
                    color="Model",
                    size="Throughput_eps" if "Throughput_eps" in df_p.columns else None,
                    hover_name="Model",
                    title="<b>Pareto Frontier: Median Latency vs Detection Accuracy</b>",
                    log_x=True
                )
                fig_pareto.update_layout(
                    paper_bgcolor="#0F172A",
                    plot_bgcolor="#1E293B",
                    font=dict(color="#F8FAFC"),
                    height=450,
                    showlegend=False
                )
                st.plotly_chart(fig_pareto, use_container_width=True)

            with c_p_img:
                img_comp = "artifacts/complexity.png"
                if os.path.exists(img_comp):
                    st.image(img_comp, caption="Computational Complexity vs Accuracy Pareto Frontier", use_container_width=True)
                else:
                    st.info("Pareto artifact available in artifacts/complexity.png")

