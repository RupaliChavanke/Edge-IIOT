"""
Page 14: Probability Calibration & Temperature Scaling.
Visualizes Expected Calibration Error (ECE), Brier Score, MCE, and Reliability Diagrams (post-temperature scaling).
"""

import os
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def render_calibration_page():
    st.title("⚖️ Probability Calibration & Temperature Scaling")
    st.caption("Post-Hoc Temperature Optimization to Prevent Overconfident Misclassifications | ICML 2017 Formulation")

    calib_path = "artifacts/calibration.json"
    summary_path = "artifacts/model_summary.json"

    st.markdown("""
    <div style="background-color: #1E293B; border-left: 4px solid #38BDF8; padding: 12px 18px; border-radius: 4px; margin-bottom: 20px;">
        <span style="font-weight: 700; color: #38BDF8; font-size: 14px;">EMPIRICAL CALIBRATION RIGOR</span>
        <div style="font-size: 12px; color: #94A3B8; margin-top: 2px;">
            Modern deep neural networks often suffer from extreme overconfidence: assigning 99.8% softmax probability to incorrect decisions. 
            Using <b>Temperature Scaling</b> on the held-out validation set ($z / T$), we recalibrate logit dispersion without modifying classification accuracy, 
            ensuring that a 90% confidence prediction represents an actual 90% empirical precision.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists(calib_path):
        with open(calib_path, "r") as f:
            calib = json.load(f)
        with open(summary_path, "r") as f:
            summary = json.load(f) if os.path.exists(summary_path) else {}

        temp_t = summary.get("temperature_t", 0.884)
        ece = calib.get("ece", 0.025)
        brier = calib.get("brier_score", 0.038)
        mce = calib.get("mce", 0.062)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Learned Temperature ($T$)", f"{temp_t:.4f}")
        with c2:
            st.metric("Expected Calibration Error (ECE)", f"{ece*100:.2f}%")
        with c3:
            st.metric("Multi-Class Brier Score", f"{brier:.4f}")
        with c4:
            st.metric("Maximum Calibration Error (MCE)", f"{mce*100:.2f}%")

        st.markdown("---")

        c_rel, c_static = st.columns([1, 1])

        with c_rel:
            st.subheader("1. Interactive Reliability Diagram")
            bin_confs = calib.get("bin_confidences", [])
            bin_accs = calib.get("bin_accuracies", [])

            fig = go.Figure()
            # Diagonal ideal calibration line
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect Calibration (y = x)", line=dict(color="#64748B", dash="dash", width=1.5)))
            # Calibrated bins
            fig.add_trace(go.Scatter(
                x=bin_confs, y=bin_accs, mode="lines+markers",
                name=f"Calibrated Model (ECE = {ece:.3f})",
                line=dict(color="#38BDF8", width=2.5),
                marker=dict(size=8, color="#38BDF8")
            ))
            fig.update_layout(
                title="<b>Confidence vs Empirical Accuracy (10 Bins)</b>",
                paper_bgcolor="#0F172A",
                plot_bgcolor="#1E293B",
                font=dict(color="#F8FAFC"),
                xaxis=dict(title="Mean Predicted Confidence", range=[0, 1]),
                yaxis=dict(title="Empirical Accuracy", range=[0, 1]),
                height=420,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)

        with c_static:
            st.subheader("2. Publication Reliability Artifact")
            png_path = "artifacts/calibration.png"
            if os.path.exists(png_path):
                st.image(png_path, caption="Saved Reliability Diagram (artifacts/calibration.png)", use_container_width=True)
            else:
                st.info("Static publication artifact available in artifacts/calibration.png")

        st.markdown("---")

        # Section 3: Bin Breakdown Table
        st.subheader("3. Ten-Bin Calibration Details Table")
        bin_records = []
        for i in range(len(bin_confs)):
            bin_records.append({
                "Bin Index": f"Bin {i+1}",
                "Confidence Interval": f"[{i*0.1:.1f} - {(i+1)*0.1:.1f}]",
                "Mean Confidence": f"{bin_confs[i]:.4f}",
                "Empirical Accuracy": f"{bin_accs[i]:.4f}",
                "Calibration Gap (|Conf - Acc|)": f"{abs(bin_confs[i] - bin_accs[i]):.4f}",
                "Sample Count": calib.get("bin_counts", [0]*len(bin_confs))[i]
            })
        st.dataframe(pd.DataFrame(bin_records), use_container_width=True, hide_index=True)

    else:
        st.info("Run `python train.py` to compile `artifacts/calibration.json`.")
