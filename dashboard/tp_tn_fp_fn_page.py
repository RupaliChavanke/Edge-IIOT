"""
TP/TN/FP/FN Detection Matrix & Class-Level Diagnostic Page.
Presents granular per-attack-class detection metrics, sensitivity, specificity, and false alarms.
"""

import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from dashboard.live_evaluator import LiveEvaluatorEngine
from dashboard.state import get_producer, get_consumer
from models.model_manager import ModelManager


def render_tp_tn_fp_fn_page():
    st.title("🎯 Threat Detection Matrix: TP / TN / FP / FN Diagnostics")
    st.caption("Granular Per-Class Operational Reliability Breakdown | Live Streaming IDS")

    prod = get_producer()
    cons = get_consumer()
    is_streaming = bool((prod and prod.is_running and not getattr(prod, "is_paused", False)) or (cons and cons.is_running))

    evaluator = LiveEvaluatorEngine.get_instance()
    manager = ModelManager.get_instance()
    metrics = evaluator.get_live_metrics()
    sample_count = evaluator.get_sample_count()
    class_names = manager.class_names or evaluator.class_names

    st.markdown("""
    <div style="background-color: #1E293B; border-left: 4px solid #38BDF8; padding: 12px 18px; border-radius: 4px; margin-bottom: 20px;">
        <span style="font-weight: 700; color: #38BDF8; font-size: 14px;">SOC OPERATIONAL TELEMETRY</span>
        <div style="font-size: 12px; color: #94A3B8; margin-top: 2px;">
            In high-assurance IIoT environments, aggregate accuracy hides critical single-attack vulnerabilities. 
            This view decomposes system performance into <b>True Positives (Attacks Neutralized)</b>, 
            <b>True Negatives (Legitimate Operations Maintained)</b>, <b>False Positives (Wasted Incident Response)</b>, 
            and <b>False Negatives (Undetected Intrusions)</b> across every threat vector.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top Global Counts
    tp = metrics.get("TP", 0)
    tn = metrics.get("TN", 0)
    fp = metrics.get("FP", 0)
    fn = metrics.get("FN", 0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total True Positives (TP)", f"{tp:,}", help="Malicious events correctly flagged as attacks")
    with c2:
        st.metric("Total True Negatives (TN)", f"{tn:,}", help="Normal telemetry correctly passed as benign")
    with c3:
        st.metric("Total False Positives (FP)", f"{fp:,}", help="Benign traffic misclassified as attack (false alarm)")
    with c4:
        st.metric("Total False Negatives (FN)", f"{fn:,}", help="Cyberattacks that bypassed detection")

    st.markdown("---")

    # Section 1: Granular Per-Class Breakdown
    st.subheader("1. Per-Attack-Class Diagnostic Matrix")

    cm_list = metrics.get("Confusion_Matrix") or metrics.get("confusion_matrix") or []
    cm = np.array(cm_list)
    if len(cm) > 0 and len(class_names) == len(cm):
        total_samples = np.sum(cm)
        class_stats = []

        for i, cname in enumerate(class_names):
            tp = int(cm[i, i])
            fn = int(np.sum(cm[i, :]) - tp)
            fp = int(np.sum(cm[:, i]) - tp)
            tn = int(total_samples - (tp + fn + fp))
            
            support = tp + fn
            sensitivity = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
            specificity = (tn / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0
            f1 = (2 * tp / (2 * tp + fp + fn)) * 100.0 if (2 * tp + fp + fn) > 0 else 0.0

            class_stats.append({
                "Threat Vector": cname,
                "Support": support,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Detection Rate (Sensitivity)": f"{sensitivity:.1f}%",
                "Specificity": f"{specificity:.1f}%",
                "Class F1-Score": f"{f1:.1f}%"
            })

        df_stats = pd.DataFrame(class_stats)
        st.dataframe(df_stats, use_container_width=True, hide_index=True)

        # Section 2: Visual Comparison of Detection vs Miss Rates
        st.markdown("---")
        st.subheader("2. Threat Neutralization vs Miss Rate Visualization")
        
        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            fig_bar = px.bar(
                df_stats,
                x="Threat Vector",
                y=["TP", "FN"],
                title="<b>Threat Neutralized (TP) vs Evaded (FN) by Class</b>",
                barmode="stack",
                color_discrete_map={"TP": "#10B981", "FN": "#EF4444"}
            )
            fig_bar.update_layout(
                paper_bgcolor="#0F172A",
                plot_bgcolor="#1E293B",
                font=dict(color="#F8FAFC"),
                xaxis=dict(tickangle=-45),
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with c_chart2:
            fig_fp = px.bar(
                df_stats,
                x="Threat Vector",
                y="FP",
                title="<b>False Alarms (FP) Triggered per Threat Type</b>",
                color="FP",
                color_continuous_scale="Reds"
            )
            fig_fp.update_layout(
                paper_bgcolor="#0F172A",
                plot_bgcolor="#1E293B",
                font=dict(color="#F8FAFC"),
                xaxis=dict(tickangle=-45),
                height=400
            )
            st.plotly_chart(fig_fp, use_container_width=True)

    else:
        st.info("Start dataset replay or stream packets through Redpanda to populate per-class diagnostic tables.")

    # Live auto-refresh when streaming is active
    if is_streaming:
        time.sleep(1.0)
        st.rerun()
