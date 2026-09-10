"""
Model Comparison & Architectural Ablation Benchmark Page.
Renders empirical evaluations across 15 baseline classifiers and 11 ablation variants
from saved artifacts (benchmark_results.csv, benchmark.png, ablation.png, complexity.png).
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_model_comparison_page():
    st.title("🏆 Empirical Model Benchmarking & 11-Variant Ablation Study")
    st.caption("Comprehensive Comparative Analysis against 15 State-of-the-Art ML/DL Classifiers")

    bench_csv = "artifacts/benchmark_results.csv"

    tab_bench, tab_abl, tab_pareto = st.tabs([
        "15-Model Baseline Benchmark",
        "11-Variant Architectural Ablation",
        "Pareto Optimality (FLOPs vs Accuracy)"
    ])

    with tab_bench:
        st.subheader("1. Empirical Comparison against 15 Baseline Models")
        st.markdown("""
        All baseline models were evaluated on the identical 22-feature Edge-IIoTset test split under 
        uniform training constraints. Our **Proposed Hybrid Model** achieves superior classification 
        fidelity while maintaining lower computational complexity than standard transformers.
        """)

        if os.path.exists(bench_csv):
            df_b = pd.read_csv(bench_csv)
            st.dataframe(df_b, use_container_width=True, hide_index=True)

            c_fig, c_img = st.columns([1, 1])
            with c_fig:
                fig_bar = px.bar(
                    df_b,
                    x="Model",
                    y="Accuracy",
                    color="Model",
                    title="<b>Classification Accuracy across 15 Benchmarked Architectures</b>",
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
                    st.image(img_path, caption="Saved Benchmark Comparison Artifact")
                else:
                    st.info("Static benchmark artifact available in artifacts/benchmark.png")
        else:
            st.info("Run `python train.py` to compile `artifacts/benchmark_results.csv`.")

    with tab_abl:
        st.subheader("2. 11-Configuration Architectural Component Ablation")
        st.markdown("""
        To isolate the exact contribution of each architectural submodule, 11 ablation variants were evaluated:
        1. **Proposed Full Architecture** (CNN + Ghost + SE + BiGRU + MHA + LowRank + Focal/Center)
        2. **w/o Ghost Module** (Standard 1D-CNN)
        3. **w/o SE Attention** (No channel weighting)
        4. **w/o Bi-GRU** (Feedforward only)
        5. **w/o Multi-Head Attention** (Direct pooling)
        6. **w/o Low-Rank Projection** (Dense $2048 \\to 15$ linear projection)
        7. **w/o Center Loss** (Focal loss only)
        8. **w/o Focal Loss** (Standard Cross-Entropy)
        9. **Always Fast Path** (100% early exit)
        10. **Always Deep Path** (100% BiGRU+Attention)
        11. **Random Routing Baseline**
        """)

        img_abl = "artifacts/ablation.png"
        if os.path.exists(img_abl):
            st.image(img_abl, caption="11-Variant Ablation Study Results (Saved Artifact)", use_container_width=True)
        else:
            st.info("Run `python train.py` to compile `artifacts/ablation.png`.")

    with tab_pareto:
        st.subheader("3. Complexity vs Accuracy Pareto Frontier")
        st.markdown("""
        Deploying IDS on resource-constrained IIoT edge nodes requires balancing parameter count and FLOPs 
        against detection accuracy. The **Proposed Model** occupies the upper-left Pareto frontier, delivering 
        competitive detection accuracy with an order of magnitude fewer parameters than standard transformers.
        """)

        img_comp = "artifacts/complexity.png"
        if os.path.exists(img_comp):
            st.image(img_comp, caption="Computational Complexity vs Accuracy Pareto Frontier", use_container_width=True)
        else:
            st.info("Run `python train.py` to compile `artifacts/complexity.png`.")
