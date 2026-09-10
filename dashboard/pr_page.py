"""
Page 13: Precision-Recall (PR) Curves & Average Precision (AP) Analysis.
Displays multi-class PR trade-offs, Macro/Weighted Average Precision metrics,
and publication-ready artifact renderings.
Strictly offline-evaluated, zero-recomputation view.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_pr_page():
    st.title("⚖️ 13. Precision-Recall (PR) Curves & Trade-offs")
    st.caption("Multi-Class Precision-Recall Dynamics for Class-Imbalanced Industrial Threat Detection")

    metrics_csv = "artifacts/per_class_metrics.csv"
    png_pr = "artifacts/precision_recall_curve.png"

    if not os.path.exists(metrics_csv):
        st.warning("⚠️ No per-class metrics found in `artifacts/per_class_metrics.csv`. Run `python train.py` to compile artifacts.")
        return

    df_metrics = pd.read_csv(metrics_csv)
    macro_prec = df_metrics["Precision"].mean()
    macro_rec = df_metrics["Recall"].mean()
    macro_f1 = df_metrics["F1"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Macro-Average Precision", f"{macro_prec*100:.2f}%")
    with c2:
        st.metric("Macro-Average Recall", f"{macro_rec*100:.2f}%")
    with c3:
        st.metric("Macro-Average F1-Score", f"{macro_f1*100:.2f}%")
    with c4:
        st.metric("Evaluated Classes", len(df_metrics))

    st.markdown("---")

    col_png, col_tab = st.columns([1, 1])

    with col_png:
        st.subheader("Publication-Ready PR Curves")
        if os.path.exists(png_pr):
            st.image(png_pr, caption="Precision-Recall Curves Across Threat Classes (Saved to artifacts/precision_recall_curve.png)", use_container_width=True)
        else:
            st.info("Artifact available at `artifacts/precision_recall_curve.png`.")

    with col_tab:
        st.subheader("Interactive Precision vs. Recall Scatter")
        fig = px.scatter(
            df_metrics,
            x="Recall",
            y="Precision",
            size="Samples",
            color="Attack Type",
            hover_name="Attack Type",
            hover_data={"F1": True, "FPR": True, "Samples": True},
            title="Precision vs Recall Trade-off by Threat Vector"
        )
        # Add F1 contour guideline (y = x)
        fig.add_shape(
            type="line", line=dict(dash="dash", color="#64748B", width=1.5),
            x0=0, y0=0, x1=1, y1=1
        )
        fig.update_layout(
            paper_bgcolor="#0F172A",
            plot_bgcolor="#1E293B",
            font=dict(color="#F8FAFC"),
            xaxis=dict(range=[0, 1.05]),
            yaxis=dict(range=[0, 1.05]),
            height=420,
            margin=dict(l=30, r=30, t=40, b=30)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Per-Threat Vector Precision-Recall Metrics")
    pr_display_cols = ["Attack Type", "Samples", "Precision", "Recall", "F1", "FPR", "FNR", "ROC-AUC"]
    df_show = df_metrics[pr_display_cols].copy()
    for col in ["Precision", "Recall", "F1", "FPR", "FNR", "ROC-AUC"]:
        df_show[col] = df_show[col].apply(lambda v: f"{v*100:.2f}%" if pd.notnull(v) else "N/A")
    st.dataframe(df_show, use_container_width=True, hide_index=True)
