"""
Page 10: Per-Attack Detailed Metrics & Safety-Critical Analysis.
Displays the mandatory granular per-attack-type evaluation table:
Samples, Precision, Recall, F1, Specificity, FPR, FNR, TP, TN, FP, FN, ROC-AUC, Average Confidence, ECE.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px


def render_per_attack_page():
    st.title("🎯 Per-Attack Granular Performance & Safety Diagnostics")
    st.caption("Detailed Breakdown of Detection Rates across all 15 Threat Types | Unseen Test Split")

    csv_path = "artifacts/per_class_metrics.csv"

    st.markdown("""
    <div style="background-color: #1E293B; border-left: 4px solid #38BDF8; padding: 12px 18px; border-radius: 4px; margin-bottom: 20px;">
        <span style="font-weight: 700; color: #38BDF8; font-size: 14px;">NO COMPROMISE ON MINORITY ATTACKS</span>
        <div style="font-size: 12px; color: #94A3B8; margin-top: 2px;">
            In industrial automation, high aggregate accuracy is meaningless if a single stealthy attack (e.g. Backdoor or SQL Injection) 
            evades detection. This mandatory table reveals exact per-class metrics, false-negative rates, and calibration errors.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists(csv_path):
        df_p = pd.read_csv(csv_path)

        # Highlight Worst-Performing Class & Highest Risk Class
        worst_f1_row = df_p.sort_values("F1").iloc[0]
        highest_fnr_row = df_p[df_p["Attack Type"] != "Normal"].sort_values("FNR", ascending=False).iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Evaluated Attacks", len(df_p[df_p["Attack Type"] != "Normal"]))
        with c2:
            st.metric("Macro Average F1", f"{df_p['F1'].mean()*100:.2f}%")
        with c3:
            st.metric("Lowest Attack F1", f"{worst_f1_row['Attack Type']} ({worst_f1_row['F1']*100:.1f}%)")
        with c4:
            st.metric("Highest FNR (Miss Rate)", f"{highest_fnr_row['Attack Type']} ({highest_fnr_row['FNR']*100:.1f}%)")

        st.markdown("---")

        # Section 1: Mandatory Per-Attack Table
        st.subheader("1. Comprehensive Per-Attack Evaluation Table")
        st.dataframe(df_p, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Section 2: Sensitivity (Recall) vs False Negative Rate (FNR)
        st.subheader("2. Detection Sensitivity vs Evaded Threat Rate")
        c_bar1, c_bar2 = st.columns(2)

        with c_bar1:
            fig_rec = px.bar(
                df_p,
                x="Attack Type",
                y="Recall",
                title="<b>Attack Detection Rate (Recall / Sensitivity)</b>",
                color="Recall",
                color_continuous_scale="Viridis",
                text_auto=".2f"
            )
            fig_rec.update_layout(
                paper_bgcolor="#0F172A",
                plot_bgcolor="#1E293B",
                font=dict(color="#F8FAFC"),
                xaxis=dict(tickangle=-45),
                height=400
            )
            st.plotly_chart(fig_rec, use_container_width=True)

        with c_bar2:
            fig_fnr = px.bar(
                df_p[df_p["Attack Type"] != "Normal"],
                x="Attack Type",
                y="FNR",
                title="<b>False Negative Rate (Threats Bypassing IDS)</b>",
                color="FNR",
                color_continuous_scale="Reds",
                text_auto=".2f"
            )
            fig_fnr.update_layout(
                paper_bgcolor="#0F172A",
                plot_bgcolor="#1E293B",
                font=dict(color="#F8FAFC"),
                xaxis=dict(tickangle=-45),
                height=400
            )
            st.plotly_chart(fig_fnr, use_container_width=True)

        st.markdown("---")

        # Section 3: Safety-Critical Missed Attack Forensics
        st.subheader("3. Safety-Critical Missed Attack Forensics (FNR Audit)")
        st.markdown("""
        False negatives represent undetected penetrations. Every missed incident allows adversaries to persist inside the SCADA/ICS network.
        """)
        critical_missed = df_p[(df_p["Attack Type"] != "Normal") & (df_p["FN"] > 0)][["Attack Type", "Samples", "TP", "FN", "FNR", "Average Confidence", "ECE"]]
        st.dataframe(critical_missed, use_container_width=True, hide_index=True)

    else:
        st.info("Run `python train.py` to compile `artifacts/per_class_metrics.csv`.")
