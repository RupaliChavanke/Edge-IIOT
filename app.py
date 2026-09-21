"""
Edge-IIoTset Real-Time Intelligent Intrusion Detection System.
Redpanda-Based Mutual Information–Driven Hybrid CNN–BiGRU Framework.
Central Streamlit Application Entry Point with Pretrained Model Verification and Frozen Deployment.
"""

import os
import streamlit as st
import torch

# Set page configuration at the absolute top
st.set_page_config(
    page_title="Edge-IIoT Streaming IDS | PhD Research",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Modern SOC CSS Styling with Animations & Glassmorphism
st.markdown("""
<style>
    /* Dark Theme Core */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #111C35 0%, #080D1A 60%, #050811 100%);
        color: #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Smooth Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0B0F19;
    }
    ::-webkit-scrollbar-thumb {
        background: #1E293B;
        border-radius: 4px;
        border: 1px solid #334155;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #0284C7;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0B1120;
        border-right: 1px solid rgba(56, 189, 248, 0.15);
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4);
    }
    
    /* Glassmorphic Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.75) 0%, rgba(30, 41, 59, 0.5) 100%);
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 10px;
        padding: 12px 16px;
        backdrop-filter: blur(12px);
        transition: all 0.28s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(56, 189, 248, 0.6);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(14, 165, 233, 0.2);
    }
    div[data-testid="stMetricValue"] {
        font-size: 26px;
        font-weight: 800;
        color: #38BDF8;
        letter-spacing: -0.5px;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 11px;
        color: #94A3B8;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* Live Animated Pulsing Indicators */
    .pulse-dot {
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        margin-right: 6px;
        position: relative;
        vertical-align: middle;
    }
    .pulse-green {
        background-color: #10B981;
        box-shadow: 0 0 10px #10B981;
        animation: pulse-ring-green 2s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
    }
    .pulse-blue {
        background-color: #38BDF8;
        box-shadow: 0 0 10px #38BDF8;
        animation: pulse-ring-blue 2s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
    }
    .pulse-amber {
        background-color: #F59E0B;
        box-shadow: 0 0 10px #F59E0B;
        animation: pulse-ring-amber 2s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
    }
    .pulse-red {
        background-color: #EF4444;
        box-shadow: 0 0 10px #EF4444;
        animation: pulse-ring-red 2s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
    }

    @keyframes pulse-ring-green {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.8); }
        70% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    @keyframes pulse-ring-blue {
        0% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.8); }
        70% { box-shadow: 0 0 0 8px rgba(56, 189, 248, 0); }
        100% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
    }
    @keyframes pulse-ring-amber {
        0% { box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.8); }
        70% { box-shadow: 0 0 0 8px rgba(245, 158, 11, 0); }
        100% { box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
    }
    @keyframes pulse-ring-red {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.8); }
        70% { box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    /* Buttons Modern Elevation */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        color: #FFFFFF;
        border: 1px solid rgba(56, 189, 248, 0.4);
        border-radius: 8px;
        font-weight: 600;
        padding: 6px 18px;
        transition: all 0.25s ease-in-out;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%);
        border-color: #38BDF8;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.4);
    }
    
    /* Table Headers */
    thead tr th {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #0284C7 !important;
    }
    
    /* Sidebar Section Divider */
    .nav-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.35), transparent);
        margin: 12px 0;
    }

    /* Highlight Badges */
    .badge-ok {
        background-color: rgba(16, 185, 129, 0.18);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 3px 8px;
        border-radius: 5px;
        font-weight: 700;
        font-size: 11px;
    }
    .badge-warn {
        background-color: rgba(245, 158, 11, 0.18);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 3px 8px;
        border-radius: 5px;
        font-weight: 700;
        font-size: 11px;
    }
    .badge-err {
        background-color: rgba(239, 68, 68, 0.18);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 3px 8px;
        border-radius: 5px;
        font-weight: 700;
        font-size: 11px;
    }

    /* =========================================================
       HIGH-END CYBER SOC SIDEBAR NAVIGATION & RADIO CARD EFFECTS
       ========================================================= */
    /* Sidebar Radio Group */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] {
        gap: 6px !important;
        padding: 2px 0 !important;
    }

    /* Individual Nav Item Card */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.45) 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.14) !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        margin-bottom: 2px !important;
        cursor: pointer !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3) !important;
        position: relative !important;
        display: flex !important;
        align-items: center !important;
        backdrop-filter: blur(8px) !important;
        width: 100% !important;
    }

    /* Card Hover: Slide-Right + Cyan Glow */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background: linear-gradient(135deg, rgba(2, 132, 199, 0.22) 0%, rgba(30, 41, 59, 0.85) 100%) !important;
        border-color: rgba(56, 189, 248, 0.55) !important;
        transform: translateX(4px) !important;
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.25) !important;
    }

    /* Hide standard circular radio input */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] input[type="radio"] {
        position: absolute !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        pointer-events: none !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[data-testid="stRadioButtonCustom"],
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[class*="StyledRadio"] {
        display: none !important;
    }

    /* Nav Item Typography */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label p,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p {
        font-size: 12px !important;
        font-weight: 500 !important;
        color: #CBD5E1 !important;
        margin: 0 !important;
        letter-spacing: 0.2px !important;
        line-height: 1.35 !important;
        transition: color 0.2s ease, font-weight 0.2s ease !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover p {
        color: #F8FAFC !important;
    }

    /* Active / Selected Card State */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked),
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] {
        background: linear-gradient(90deg, rgba(2, 132, 199, 0.35) 0%, rgba(15, 23, 42, 0.95) 100%) !important;
        border-left: 4px solid #38BDF8 !important;
        border-top: 1px solid rgba(56, 189, 248, 0.5) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.25) !important;
        border-bottom: 1px solid rgba(56, 189, 248, 0.25) !important;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.22), inset 0 0 12px rgba(56, 189, 248, 0.08) !important;
        transform: translateX(3px) !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) p,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] p,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
        color: #38BDF8 !important;
        font-weight: 700 !important;
    }

    /* Active Pulsing Indicator Dot */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked)::after {
        content: "";
        position: absolute;
        right: 12px;
        top: 50%;
        transform: translateY(-50%);
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #38BDF8;
        box-shadow: 0 0 10px #38BDF8;
        animation: pulse-ring-blue 2s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
    }

    /* Horizontal Segmented Switcher (Categorized vs All 27) */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"][aria-orientation="horizontal"] {
        display: flex !important;
        flex-direction: row !important;
        gap: 6px !important;
        background: rgba(11, 17, 32, 0.9) !important;
        padding: 4px !important;
        border-radius: 8px !important;
        border: 1px solid rgba(56, 189, 248, 0.18) !important;
        margin-bottom: 6px !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"][aria-orientation="horizontal"] label {
        flex: 1 !important;
        justify-content: center !important;
        text-align: center !important;
        padding: 6px 8px !important;
        margin: 0 !important;
        border-radius: 6px !important;
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"][aria-orientation="horizontal"] label:hover {
        background: rgba(30, 41, 59, 0.6) !important;
        transform: none !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"][aria-orientation="horizontal"] label:has(input:checked) {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        border: 1px solid #38BDF8 !important;
        box-shadow: 0 2px 10px rgba(14, 165, 233, 0.4) !important;
        transform: none !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"][aria-orientation="horizontal"] label:has(input:checked)::after {
        display: none !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"][aria-orientation="horizontal"] label:has(input:checked) p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Quick Jump Buttons Styling */
    section[data-testid="stSidebar"] .qjump-col div.stButton > button {
        background: rgba(15, 23, 42, 0.75) !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        color: #CBD5E1 !important;
        font-size: 11px !important;
        padding: 4px 6px !important;
        border-radius: 6px !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }
    section[data-testid="stSidebar"] .qjump-col div.stButton > button:hover {
        background: rgba(2, 132, 199, 0.3) !important;
        border-color: #38BDF8 !important;
        color: #38BDF8 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 3px 10px rgba(14, 165, 233, 0.25) !important;
    }

    /* Search and Selectbox in Sidebar */
    section[data-testid="stSidebar"] div[data-testid="stTextInput"] input {
        background-color: rgba(15, 23, 42, 0.85) !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        border-radius: 8px !important;
        color: #F8FAFC !important;
        font-size: 12px !important;
        padding: 7px 12px !important;
        transition: all 0.2s ease !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stTextInput"] input:focus {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.35) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div {
        background-color: rgba(15, 23, 42, 0.85) !important;
        border: 1px solid rgba(56, 189, 248, 0.22) !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

from dashboard.state import init_session_state
from streaming.health import RedpandaHealthChecker
from models.model_manager import ModelManager

# Initialize session state for background streaming services
init_session_state()

# -------------------------------------------------------------
# 1. MANDATORY STARTUP VERIFICATION: PRETRAINED MODEL ARTIFACTS
# -------------------------------------------------------------
manager = ModelManager.get_instance(artifacts_dir="artifacts")
is_valid, validation_msg = manager.validate_artifacts()
metadata = manager.get_model_metadata()

# Query Redpanda status dynamically
health_checker = RedpandaHealthChecker()
health_info = health_checker.check_health()
rp_status = health_info.get("status", "DISCONNECTED")

# Sidebar Header with Glowing System Brand
st.sidebar.markdown("""
<div style="padding: 6px 0 12px 0;">
    <div style="font-size: 20px; font-weight: 900; color: #F8FAFC; letter-spacing: -0.5px;">
        🛡️ Edge-IIoT IDS
    </div>
    <div style="font-size: 11px; color: #38BDF8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
        Streaming Intelligent Defense System
    </div>
</div>
""", unsafe_allow_html=True)

# Pretrained Model Status Badge with Animated Pulse
if is_valid and manager.is_loaded:
    st.sidebar.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(16, 185, 129, 0.35); margin-bottom: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
        <div style="font-size: 10px; color: #94A3B8; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px;">Neural Edge Model</div>
        <div style="font-size: 13px; font-weight: bold; margin-top: 3px; color: #10B981;">
            <span class="pulse-dot pulse-green"></span>LOADED (FROZEN v1.0)
        </div>
        <div style="font-size: 11px; color: #CBD5E1; margin-top: 4px;">
            Deployed Accuracy: <b style="color: #38BDF8;">96.35%</b> (ONNX)
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.9); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.35); margin-bottom: 10px;">
        <div style="font-size: 10px; color: #94A3B8; text-transform: uppercase; font-weight: 800;">Neural Edge Model</div>
        <div style="font-size: 13px; font-weight: bold; margin-top: 3px; color: #EF4444;">
            <span class="pulse-dot pulse-red"></span>MODEL NOT FOUND
        </div>
        <div style="font-size: 11px; color: #EF4444; margin-top: 4px;">
            Run <code>python train.py</code> offline
        </div>
    </div>
    """, unsafe_allow_html=True)

# Streaming Platform Health Badge (Redpanda Cluster or Cloud In-Memory Bus)
if rp_status == 'CONNECTED':
    rp_badge = '<span class="pulse-dot pulse-green"></span>CONNECTED (Port 19092)'
    subtext_html = health_info.get('broker', 'localhost:19092')
    border_col = "rgba(16, 185, 129, 0.35)"
    text_col = "#10B981"
elif rp_status == 'CLOUD_MODE':
    rp_badge = '<span class="pulse-dot pulse-blue"></span>CLOUD STREAM BUS'
    subtext_html = "InMemoryStreamingBus (Zero-Broker Cloud)"
    border_col = "rgba(56, 189, 248, 0.35)"
    text_col = "#38BDF8"
else:
    rp_badge = '<span class="pulse-dot pulse-red"></span>DISCONNECTED'
    subtext_html = "Broker Offline • Run docker compose"
    border_col = "rgba(239, 68, 68, 0.35)"
    text_col = "#EF4444"

st.sidebar.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%); padding: 10px 14px; border-radius: 8px; border: 1px solid {border_col}; margin-bottom: 14px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
    <div style="font-size: 10px; color: #94A3B8; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px;">Streaming Backbone</div>
    <div style="font-size: 13px; font-weight: bold; margin-top: 3px; color: {text_col};">
        {rp_badge}
    </div>
    <div style="font-size: 10px; color: #64748B; margin-top: 3px;">{subtext_html}</div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. 27 MANDATORY RESEARCH NAVIGATION PAGES (MODERN CYBER SOC PANE)
# -------------------------------------------------------------
import re

RESEARCH_PAGES_META = [
    {"num": 1, "label": "🌟 1. Executive Summary", "category": "Executive Command", "keywords": "summary overview kpi blueprint system"},
    {"num": 2, "label": "📊 2. Dataset Explorer", "category": "Dataset & Features", "keywords": "dataset explorer edge iiot telemetry features"},
    {"num": 3, "label": "📦 3. Full Dataset Inventory", "category": "Dataset & Features", "keywords": "inventory table files records distribution balance"},
    {"num": 4, "label": "🏷️ 4. Attack Taxonomy", "category": "Dataset & Features", "keywords": "attack taxonomy categories mitre 14 threats benign"},
    {"num": 5, "label": "⚙️ 5. Feature Engineering", "category": "Dataset & Features", "keywords": "feature engineering preprocessing imputation scaling clamping"},
    {"num": 6, "label": "🎯 6. mRMR-JMI Selection", "category": "Dataset & Features", "keywords": "mrmr jmi mutual information 22 features selection inductive"},
    {"num": 7, "label": "🧠 7. Model Architecture", "category": "Architecture & Ablation", "keywords": "model architecture 1d-cnn ghost se bigru attention 3d"},
    {"num": 8, "label": "💎 8. Pretrained Model", "category": "Executive Command", "keywords": "pretrained frozen onnx 96.35 checkpoint weights verified"},
    {"num": 9, "label": "📉 9. Offline Performance", "category": "Performance & Proof", "keywords": "offline performance test accuracy f1 precision recall holdout"},
    {"num": 10, "label": "🎯 10. Per-Attack Analysis", "category": "Performance & Proof", "keywords": "per-attack breakdown classwise ddos backdoors mitm fingerprinting"},
    {"num": 11, "label": "🔲 11. Confusion Matrix", "category": "Performance & Proof", "keywords": "confusion matrix 15-class heatmap normalized count matrix"},
    {"num": 12, "label": "📈 12. ROC-AUC", "category": "Performance & Proof", "keywords": "roc auc curves false positive rate tpr macro micro multiclass"},
    {"num": 13, "label": "📊 13. Precision-Recall", "category": "Performance & Proof", "keywords": "precision recall curve pr-auc trade-off class curves"},
    {"num": 14, "label": "🌡️ 14. Calibration", "category": "Performance & Proof", "keywords": "temperature calibration scaling ece reliability diagram confidence"},
    {"num": 15, "label": "🔍 15. Confidence Analysis", "category": "Performance & Proof", "keywords": "confidence distribution correct incorrect certainty histograms"},
    {"num": 16, "label": "❓ 16. Uncertainty Detection", "category": "Performance & Proof", "keywords": "uncertainty entropy ood shannon router early exit threshold"},
    {"num": 17, "label": "🐼 17. Redpanda Streaming", "category": "Streaming Telemetry", "keywords": "redpanda streaming kafka topics broker latency partition"},
    {"num": 18, "label": "🛡️ 18. Live IDS", "category": "Streaming Telemetry", "keywords": "live ids pipeline inference events predictions alerts firewall"},
    {"num": 19, "label": "⚡ 19. Live Metrics", "category": "Streaming Telemetry", "keywords": "live metrics streaming gauges accuracy throughput rolling"},
    {"num": 20, "label": "⚖️ 20. TP/TN/FP/FN", "category": "Performance & Proof", "keywords": "tp tn fp fn rates confusion counts detection rate matrix"},
    {"num": 21, "label": "⏱️ 21. Latency", "category": "Streaming Telemetry", "keywords": "latency p50 p95 p99 microsecond execution time breakdown"},
    {"num": 22, "label": "🚀 22. Throughput", "category": "Streaming Telemetry", "keywords": "throughput messages events per second benchmark eps load"},
    {"num": 23, "label": "🔬 23. Ablation", "category": "Architecture & Ablation", "keywords": "ablation study ghost se bigru attention temperature component"},
    {"num": 24, "label": "🏆 24. Model Comparison", "category": "Architecture & Ablation", "keywords": "model comparison baselines rf xgb mlp lstm proposed 96.35"},
    {"num": 25, "label": "💡 25. Explainability", "category": "Architecture & Ablation", "keywords": "explainability xai shap attention weights feature importances saliency"},
    {"num": 26, "label": "🔎 26. Error Analysis", "category": "Performance & Proof", "keywords": "error analysis misclassifications false alarms edge cases root-cause"},
    {"num": 27, "label": "🎓 27. PhD Demonstration Mode", "category": "Executive Command", "keywords": "phd defense demonstration presentation walkthrough oral evaluation"}
]

# Track active page integer state
if "active_page_num" not in st.session_state:
    old_str = st.session_state.get("active_page", "1. Executive Summary")
    m = re.search(r"(\d+)\.", str(old_str))
    st.session_state["active_page_num"] = int(m.group(1)) if m else 1

# Quick Command Jumps (Fast Pins)
st.sidebar.markdown("""
<div style="font-size: 10px; color: #94A3B8; text-transform: uppercase; font-weight: 800; letter-spacing: 0.6px; margin-bottom: 6px;">
    ⚡ Quick Command Jumps
</div>
""", unsafe_allow_html=True)
q1, q2 = st.sidebar.columns(2)
with q1:
    if st.button("🌟 1. Overview", key="qj_overview", use_container_width=True):
        st.session_state["active_page_num"] = 1
        st.rerun()
    if st.button("💎 8. Model 96%", key="qj_model", use_container_width=True):
        st.session_state["active_page_num"] = 8
        st.rerun()
with q2:
    if st.button("🧠 7. Architecture", key="qj_arch", use_container_width=True):
        st.session_state["active_page_num"] = 7
        st.rerun()
    if st.button("🎓 27. Defense", key="qj_demo", use_container_width=True):
        st.session_state["active_page_num"] = 27
        st.rerun()

st.sidebar.markdown("<div class='nav-divider'></div>", unsafe_allow_html=True)

# Navigation Layout Mode: Categorized Modules vs Full Flat List
nav_view = st.sidebar.radio(
    "Navigation View:",
    options=["📂 Categorized", "📑 All 27 Pages"],
    horizontal=True,
    label_visibility="collapsed",
    key="nav_view_segmented"
)

# Real-Time Search / Filter Input
search_term = st.sidebar.text_input(
    "🔍 Filter 27 pages...",
    placeholder="Search pages (e.g. latency, roc)...",
    label_visibility="collapsed",
    key="nav_search_input"
).strip().lower()

if search_term:
    matching_pages = [
        p for p in RESEARCH_PAGES_META
        if search_term in p["label"].lower() or search_term in p["keywords"].lower() or search_term in p["category"].lower()
    ]
    st.sidebar.markdown(f"""
    <div style="font-size: 10px; color: #38BDF8; font-weight: 700; margin: 4px 0 6px 0;">
        🔍 Found {len(matching_pages)} of 27 pages matching "{search_term}":
    </div>
    """, unsafe_allow_html=True)
    if matching_pages:
        available_options = [p["label"] for p in matching_pages]
    else:
        st.sidebar.warning("No pages match your filter.")
        available_options = [p["label"] for p in RESEARCH_PAGES_META]
else:
    if nav_view == "📂 Categorized":
        # Cluster definitions with page counts
        CATEGORIES_MAP = {
            "🌟 Executive Command (3 Pages)": "Executive Command",
            "📊 Dataset & Features (5 Pages)": "Dataset & Features",
            "🧠 Architecture & Ablation (4 Pages)": "Architecture & Ablation",
            "📈 Performance & Proof (10 Pages)": "Performance & Proof",
            "⚡ Streaming Telemetry (5 Pages)": "Streaming Telemetry"
        }
        
        # Determine default category based on currently active page
        current_cat = next(
            (p["category"] for p in RESEARCH_PAGES_META if p["num"] == st.session_state["active_page_num"]),
            "Executive Command"
        )
        cat_keys = list(CATEGORIES_MAP.keys())
        default_cat_idx = 0
        for i, k in enumerate(cat_keys):
            if CATEGORIES_MAP[k] == current_cat:
                default_cat_idx = i
                break
        
        chosen_cat_key = st.sidebar.selectbox(
            "📂 Functional Area:",
            options=cat_keys,
            index=default_cat_idx,
            key="cat_selectbox"
        )
        selected_cat_name = CATEGORIES_MAP[chosen_cat_key]
        available_options = [p["label"] for p in RESEARCH_PAGES_META if p["category"] == selected_cat_name]
    else:
        st.sidebar.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin: 4px 0 6px 0;">
            <span style="font-size: 11px; font-weight: 800; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.6px;">All 27 Research Pages</span>
            <span style="font-size: 10px; color: #10B981; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); padding: 1px 6px; border-radius: 4px; font-weight: 700;">27 Active</span>
        </div>
        """, unsafe_allow_html=True)
        available_options = [p["label"] for p in RESEARCH_PAGES_META]

# Compute active radio index safely
selected_idx = 0
for idx, opt in enumerate(available_options):
    m_opt = re.search(r"(\d+)\.", opt)
    if m_opt and int(m_opt.group(1)) == st.session_state["active_page_num"]:
        selected_idx = idx
        break

selected_page_label = st.sidebar.radio(
    "Navigation Options:",
    options=available_options,
    index=selected_idx,
    label_visibility="collapsed",
    key="soc_nav_radio"
)

# Update active page in session state
m_sel = re.search(r"(\d+)\.", selected_page_label)
if m_sel:
    st.session_state["active_page_num"] = int(m_sel.group(1))
    st.session_state["active_page"] = selected_page_label

# Sidebar System Controls
st.sidebar.markdown("<div class='nav-divider'></div>", unsafe_allow_html=True)
from dashboard.state import get_producer, get_consumer
prod = get_producer()
cons = get_consumer()
prod_status = '🟢 RUNNING' if (prod and prod.is_running and not getattr(prod, 'is_paused', False)) else ('⏸️ PAUSED' if (prod and getattr(prod, 'is_paused', False)) else '⚪ STOPPED')
cons_status = '🟢 RUNNING' if (cons and cons.is_running) else '⚪ STOPPED'

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    st.markdown(f"<div style='font-size:11px; color:#94A3B8;'>Producer<br><b style='color:#F8FAFC;'>{prod_status}</b></div>", unsafe_allow_html=True)
with col_s2:
    st.markdown(f"<div style='font-size:11px; color:#94A3B8;'>Consumer<br><b style='color:#F8FAFC;'>{cons_status}</b></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='font-size:10px; color:#475569; text-align:center; margin-top:10px;'>Edge-IIoT Intelligent IDS • PhD Platform v2.0</div>", unsafe_allow_html=True)

# Lazy-loaded page dispatching
page_to_render = st.session_state.get("active_page_num", 1)

if page_to_render == 1:
    from dashboard.overview import render_overview_page
    render_overview_page()
elif page_to_render == 2:
    from dashboard.dataset_page import render_dataset_page
    render_dataset_page()
elif page_to_render == 3:
    from dashboard.inventory_page import render_inventory_page
    render_inventory_page()
elif page_to_render == 4:
    from dashboard.taxonomy_page import render_taxonomy_page
    render_taxonomy_page()
elif page_to_render == 5:
    from dashboard.preprocessing_page import render_preprocessing_page
    render_preprocessing_page()
elif page_to_render == 6:
    from dashboard.mrmr_page import render_mrmr_page
    render_mrmr_page()
elif page_to_render == 7:
    from dashboard.architecture_page import render_architecture_page
    render_architecture_page()
elif page_to_render == 8:
    from dashboard.pretrained_page import render_pretrained_page
    render_pretrained_page()
elif page_to_render == 9:
    from dashboard.model_perf_page import render_model_perf_page
    render_model_perf_page()
elif page_to_render == 10:
    from dashboard.per_attack_page import render_per_attack_page
    render_per_attack_page()
elif page_to_render == 11:
    from dashboard.confusion_page import render_confusion_page
    render_confusion_page()
elif page_to_render == 12:
    from dashboard.roc_page import render_roc_page
    render_roc_page()
elif page_to_render == 13:
    from dashboard.pr_page import render_pr_page
    render_pr_page()
elif page_to_render == 14:
    from dashboard.calibration_page import render_calibration_page
    render_calibration_page()
elif page_to_render == 15:
    from dashboard.confidence_page import render_confidence_page
    render_confidence_page()
elif page_to_render == 16:
    from dashboard.uncertainty_page import render_uncertainty_page
    render_uncertainty_page()
elif page_to_render == 17:
    from dashboard.streaming_page import render_streaming_page
    render_streaming_page()
elif page_to_render == 18:
    from dashboard.live_ids_page import render_live_ids_page
    render_live_ids_page()
elif page_to_render == 19:
    from dashboard.live_metrics_page import render_live_metrics_page
    render_live_metrics_page()
elif page_to_render == 20:
    from dashboard.tp_tn_fp_fn_page import render_tp_tn_fp_fn_page
    render_tp_tn_fp_fn_page()
elif page_to_render == 21:
    from dashboard.latency_page import render_latency_page
    render_latency_page()
elif page_to_render == 22:
    from dashboard.throughput_page import render_throughput_page
    render_throughput_page()
elif page_to_render == 23:
    from dashboard.ablation_page import render_ablation_page
    render_ablation_page()
elif page_to_render == 24:
    from dashboard.model_comparison_page import render_model_comparison_page
    render_model_comparison_page()
elif page_to_render == 25:
    from dashboard.explainability_page import render_explainability_page
    render_explainability_page()
elif page_to_render == 26:
    from dashboard.error_page import render_error_page
    render_error_page()
elif page_to_render == 27:
    from dashboard.phd_demo_page import render_phd_demo_page
    render_phd_demo_page()


