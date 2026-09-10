"""
Page 6: mRMR-JMI Feature Selection Engine.
Explores Joint Mutual Information ranking, feature relevance vs redundancy, and correlation heatmaps.
"""

import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from preprocessing.loader import EdgeIIoTDataLoader
from visualization.plots import plot_mrmr_feature_importance


def render_mrmr_page():
    st.title("🎯 mRMR-JMI Joint Mutual Information Feature Selection")
    st.caption("Information-theoretic dimensionality reduction maximizing relevance to attacks while penalizing multi-feature redundancy.")

    selector_path = "artifacts/feature_selector.pkl" if os.path.exists("artifacts/feature_selector.pkl") else "checkpoints/feature_selector.pkl"

    if not os.path.exists(selector_path):
        st.info("mRMR-JMI selector not yet trained. Run offline training (`python train.py`) to generate artifacts.")
        if st.button("🚀 Fit mRMR-JMI Feature Selector (61 to 22 Features)"):
            with st.spinner("Computing pairwise mutual information and greedy JMI ranking..."):
                loader = EdgeIIoTDataLoader(k_features=22)
                loader.fit_transform_pipeline(use_sample=True)
                loader.save_pipeline("checkpoints")
                st.success("mRMR-JMI selector successfully fitted and saved!")
                st.rerun()
        return

    with open(selector_path, "rb") as f:
        selector = pickle.load(f)

    ranking_df = selector.ranking_df_
    selected_features = selector.selected_features_

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Original Candidate Attributes", f"{len(ranking_df)}")
    m2.metric("Selected mRMR Features", f"{len(selected_features)}")
    m3.metric("Dimensionality Reduction", f"{((len(ranking_df) - len(selected_features)) / len(ranking_df))*100:.1f}%")
    top_jmi = ranking_df["JMI"].max()
    m4.metric("Peak Feature JMI Score", f"{top_jmi:.3f} nats")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs([
        "📊 mRMR-JMI Feature Ranking Chart",
        "📋 Complete Information Table",
        "🔥 Pairwise Correlation Heatmap"
    ])

    with tab1:
        st.subheader("Relevance, Redundancy, and Net JMI Scores")
        top_k = st.slider("Select number of top features to plot:", min_value=10, max_value=len(ranking_df), value=22)
        fig_mrmr = plot_mrmr_feature_importance(ranking_df, top_k=top_k)
        st.plotly_chart(fig_mrmr, use_container_width=True)

    with tab2:
        st.subheader("Feature Ranking Table (Relevance vs Redundancy)")
        st.dataframe(
            ranking_df.style.highlight_max(subset=["MI", "JMI"], color="#065F46"),
            use_container_width=True
        )

    with tab3:
        st.subheader("Correlation Heatmap of Top Selected 22 Features")
        sample_path = "data/samples/edge_iiot_sample.csv"
        if os.path.exists(sample_path):
            df_samp = pd.read_csv(sample_path, nrows=1000, low_memory=False)
            avail_cols = [c for c in selected_features if c in df_samp.columns]
            if avail_cols:
                # Convert to numeric
                num_df = df_samp[avail_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
                corr = num_df.corr()
                fig_corr = px.imshow(
                    corr,
                    x=avail_cols, y=avail_cols,
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    title="Correlation Matrix of Selected 22 Features"
                )
                fig_corr.update_layout(paper_bgcolor="#0F172A", font=dict(color="#F8FAFC"), height=600)
                st.plotly_chart(fig_corr, use_container_width=True)
