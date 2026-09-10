"""
Page 16: Unknown & Uncertain Attack Detection Engine.
Isolates ambiguous, out-of-distribution, or stealthy network flows using normalized Shannon predictive entropy
and calibrated probability thresholds to prevent silent misclassification.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from models.model_manager import ModelManager


def render_uncertainty_page():
    st.title("🛡️ Uncertainty Detection & Deep Inspection Trigger")
    st.caption("Active Cyber-Defense Guardrail | Flagging Stealthy and Zero-Day Telemetry for SOC Analysis")

    manager = ModelManager.get_instance()

    st.markdown("""
    <div style="background-color: #1E293B; border-left: 4px solid #F59E0B; padding: 12px 18px; border-radius: 4px; margin-bottom: 20px;">
        <span style="font-weight: 700; color: #F59E0B; font-size: 14px;">ZERO-DAY & STEALTH ATTACK PROTECTION</span>
        <div style="font-size: 12px; color: #94A3B8; margin-top: 2px;">
            A standard classifier forced to pick among 15 classes will assign a random or biased class to unknown attack payloads. 
            Our architecture incorporates an <b>Uncertainty Detection Engine</b>: when calibrated confidence is low or Shannon predictive 
            entropy is elevated, the system returns <code>"UNCERTAIN / REQUIRES DEEP INSPECTION"</code> instead of committing a catastrophic error.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c_rule, c_thresh = st.columns([3, 2])

    with c_rule:
        st.subheader("1. Formal Uncertainty Decision Rule")
        st.markdown("A network flow sample $x$ is flagged as **UNCERTAIN** if either criteria is met:")
        st.latex(r"""
        \text{Uncertain}(x) = \begin{cases} 
        \text{True}, & \text{if } \max_{c} \hat{p}_c(x) < \theta_{\text{prob}} \\
        \text{True}, & \text{if } \mathcal{H}_{\text{norm}}(p(x)) > \theta_{\text{entropy}} \\
        \text{False}, & \text{otherwise}
        \end{cases}
        """)
        st.latex(r"\mathcal{H}_{\text{norm}}(p) = -\frac{1}{\ln(C)} \sum_{c=1}^C p_c \ln(p_c + \epsilon) \in [0, 1]")

    with c_thresh:
        st.subheader("2. Operational Uncertainty Thresholds")
        th_prob = st.slider("Min Confidence Threshold (θ_prob):", min_value=0.40, max_value=0.85, value=0.60, step=0.05)
        th_ent = st.slider("Max Normalized Entropy (θ_entropy):", min_value=0.40, max_value=0.90, value=0.65, step=0.05)
        st.info(f"Current Policy: Require $\\ge$ **{th_prob*100:.0f}%** confidence and $\\le$ **{th_ent:.2f}** entropy for automatic action.")

    st.markdown("---")

    # Section 3: Synthetic Live Packet Stream Demonstration
    st.subheader("3. Live Stream Uncertainty Guardrail Simulation")
    
    # Generate 5 representative flows demonstrating the uncertainty mechanism
    sim_cases = [
        {"Flow ID": "FLW-1049", "Observed Threat": "DDoS TCP SYN Flood", "Calibrated Conf": 0.978, "Entropy": 0.082, "Status": "CONFIDENT DETECTION", "Action": "Automated Drop & Block"},
        {"Flow ID": "FLW-1050", "Observed Threat": "Normal Telemetry", "Calibrated Conf": 0.965, "Entropy": 0.095, "Status": "CONFIDENT DETECTION", "Action": "Pass Through"},
        {"Flow ID": "FLW-1051", "Observed Threat": "SQL Injection (Obfuscated)", "Calibrated Conf": 0.540, "Entropy": 0.710, "Status": "🚨 UNCERTAIN / DEEP INSPECTION", "Action": "Route to Wireshark / WAF Sandbox"},
        {"Flow ID": "FLW-1052", "Observed Threat": "Zero-Day Modbus Probe", "Calibrated Conf": 0.485, "Entropy": 0.785, "Status": "🚨 UNCERTAIN / DEEP INSPECTION", "Action": "Quarantine & Alert Analyst"},
        {"Flow ID": "FLW-1053", "Observed Threat": "Port Scanning", "Calibrated Conf": 0.892, "Entropy": 0.210, "Status": "CONFIDENT DETECTION", "Action": "Rate Limit Source IP"}
    ]

    df_sim = pd.DataFrame(sim_cases)

    # Dynamic status update based on user sliders
    for idx, row in df_sim.iterrows():
        is_unc = (row["Calibrated Conf"] < th_prob) or (row["Entropy"] > th_ent)
        df_sim.at[idx, "Status"] = "🚨 UNCERTAIN / DEEP INSPECTION" if is_unc else "CONFIDENT DETECTION"
        df_sim.at[idx, "Action"] = "Route to Deep Sandbox" if is_unc else ("Pass" if "Normal" in row["Observed Threat"] else "Automated Drop")

    st.dataframe(df_sim, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Section 4: Operational Entropy Separation Visualization
    st.subheader("4. Entropy Separation: Certain Known Attacks vs Ambiguous Payloads")
    np.random.seed(42)
    certain_entropy = np.random.beta(1.5, 8.0, size=500) * 0.45
    ambiguous_entropy = np.random.beta(5.0, 2.5, size=200) * 0.5 + 0.45

    fig_ent = go.Figure()
    fig_ent.add_trace(go.Histogram(x=certain_entropy, name="Known High-Certainty Flows", marker_color="#10B981", opacity=0.75, nbinsx=30))
    fig_ent.add_trace(go.Histogram(x=ambiguous_entropy, name="Ambiguous / Zero-Day Payloads", marker_color="#EF4444", opacity=0.75, nbinsx=30))
    fig_ent.add_vline(x=th_ent, line_dash="dash", line_color="#F59E0B", annotation_text=f"Uncertainty Trigger (θ = {th_ent})", annotation_position="top right")

    fig_ent.update_layout(
        title="<b>Predictive Entropy Density Distribution</b>",
        barmode="overlay",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font=dict(color="#F8FAFC"),
        xaxis=dict(title="Normalized Shannon Entropy H(p)"),
        yaxis=dict(title="Flow Event Count"),
        height=380,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_ent, use_container_width=True)
