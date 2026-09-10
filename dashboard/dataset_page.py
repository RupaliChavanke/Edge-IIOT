"""
Page 3: Dataset Explorer.
In-depth exploratory data analysis of Edge-IIoTset: class balance, protocols, distributions, and sample browser.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


def render_dataset_page():
    st.title("📂 Edge-IIoTset Cybersecurity Dataset Explorer")
    st.caption("Comprehensive analysis of raw network traffic attributes, IoT/IIoT attack taxonomy, and statistical distributions.")

    sample_path = "data/samples/edge_iiot_sample.csv"
    raw_path = "data/raw/ML-EdgeIIoT-dataset.csv"

    dataset_source = st.radio(
        "Select Dataset Partition to Inspect:",
        ["Lightweight Benchmark Sample (5,250 rows, 15 balanced classes)", "Full Dataset (157,800 rows)"],
        horizontal=True
    )

    path_to_load = sample_path if "Lightweight" in dataset_source else raw_path

    if not os.path.exists(path_to_load):
        if "Full" in dataset_source and os.path.exists(sample_path):
            st.info("ℹ️ The 78MB raw dataset file is preserved in offline storage. Auto-loading the comprehensive benchmark sample (5,250 rows across all 15 classes).")
            path_to_load = sample_path
        else:
            st.warning(f"File not found at `{path_to_load}`.")
            return

    @st.cache_data
    def load_cached_df(path, nrows):
        return pd.read_csv(path, nrows=nrows, low_memory=False)

    nrows = 5250 if "Lightweight" in dataset_source else 15000
    df = load_cached_df(path_to_load, nrows=nrows)

    # High-level metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Loaded Records", f"{len(df):,}")
    m2.metric("Total Attributes", f"{len(df.columns)}")
    m3.metric("Attack Classes", f"{df['Attack_type'].nunique()}")
    m4.metric("Benign vs Attack Split", f"{(df['Attack_type'] == 'Normal').mean()*100:.1f}% Normal")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Multiclass Attack Taxonomy Distribution")
        counts = df["Attack_type"].value_counts().reset_index()
        counts.columns = ["Attack_Type", "Sample_Count"]
        fig_bar = px.bar(
            counts, x="Attack_Type", y="Sample_Count",
            color="Attack_Type",
            title="Class Frequency Distribution in Loaded Partition",
            template="plotly_dark"
        )
        fig_bar.update_layout(xaxis_tickangle=-45, showlegend=False, paper_bgcolor="#0F172A", plot_bgcolor="#1E293B")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("🎯 Binary Classification Split")
        bin_counts = df["Attack_label"].value_counts().reset_index()
        bin_counts["Label_Name"] = bin_counts["Attack_label"].map({0: "Normal (0)", 1: "Attack (1)"})
        fig_pie = px.pie(
            bin_counts, names="Label_Name", values="count",
            hole=0.4, title="Normal vs Attack Proportions",
            template="plotly_dark",
            color_discrete_sequence=["#10B981", "#EF4444"]
        )
        fig_pie.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#0F172A")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Tabular Record Inspector & Feature Types")

    filter_class = st.selectbox("Filter records by attack type:", ["All Classes"] + list(df["Attack_type"].unique()))
    display_df = df if filter_class == "All Classes" else df[df["Attack_type"] == filter_class]

    st.dataframe(display_df.head(100), use_container_width=True)

    # Column Summary Table
    with st.expander("📋 View Complete Feature Schema & Data Types"):
        schema_df = pd.DataFrame({
            "Feature Name": df.columns,
            "Data Type": [str(t) for t in df.dtypes],
            "Non-Null Count": df.notnull().sum().values,
            "Unique Values": df.nunique().values
        })
        st.dataframe(schema_df, use_container_width=True)
