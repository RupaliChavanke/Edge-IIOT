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

# Custom High-End Modern SOC CSS Styling
st.markdown("""
<style>
    /* Dark Theme Core */
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    
    /* Metrics Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 700;
        color: #38BDF8;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 13px;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Buttons */
    div.stButton > button:first-child {
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    
    /* Table Headers */
    thead tr th {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
    }
    
    /* Highlight Badges */
    .badge-ok {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10B981;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-warn {
        background-color: rgba(245, 158, 11, 0.2);
        color: #F59E0B;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-err {
        background-color: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
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

# Sidebar Header & System State Badges
st.sidebar.title("🛡️ Edge-IIoT IDS")
st.sidebar.caption("Redpanda Streaming IDS | PhD Viva Demonstration")

# Pretrained Model Status Badge
if is_valid and manager.is_loaded:
    st.sidebar.markdown(f"""
    <div style="background-color: #1E293B; padding: 10px 14px; border-radius: 6px; border: 1px solid #334155; margin-bottom: 10px;">
        <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase; font-weight: bold;">Pretrained Model</div>
        <div style="font-size: 14px; font-weight: bold; margin-top: 2px;">
            <span class="badge-ok">● PRETRAINED MODEL LOADED</span>
        </div>
        <div style="font-size: 11px; color: #38BDF8; margin-top: 4px;">
            Version: <b>{metadata.get('model_version', 'v1.0')}</b> (Frozen)
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown(f"""
    <div style="background-color: #1E293B; padding: 10px 14px; border-radius: 6px; border: 1px solid #334155; margin-bottom: 10px;">
        <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase; font-weight: bold;">Pretrained Model</div>
        <div style="font-size: 14px; font-weight: bold; margin-top: 2px;">
            <span class="badge-err">● MODEL NOT FOUND</span>
        </div>
        <div style="font-size: 11px; color: #EF4444; margin-top: 4px;">
            Run <code>python train.py</code> offline
        </div>
    </div>
    """, unsafe_allow_html=True)

# Streaming Platform Health Badge (Redpanda Cluster or Cloud In-Memory Bus)
if rp_status == 'CONNECTED':
    badge_html = '<span class="badge-ok">● REDPANDA CONNECTED</span>'
    subtext_html = health_info.get('broker', 'localhost:19092')
elif rp_status == 'CLOUD_MODE':
    badge_html = '<span class="badge-ok" style="background-color: rgba(56, 189, 248, 0.2); color: #38BDF8; border: 1px solid #0284C7;">● CLOUD STREAMING ACTIVE</span>'
    subtext_html = "InMemoryStreamingBus (Zero-Broker Cloud)"
else:
    badge_html = '<span class="badge-err">● DISCONNECTED</span>'
    subtext_html = health_info.get('broker', 'localhost:19092')

st.sidebar.markdown(f"""
<div style="background-color: #1E293B; padding: 10px 14px; border-radius: 6px; border: 1px solid #334155; margin-bottom: 15px;">
    <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase; font-weight: bold;">Streaming Backbone</div>
    <div style="font-size: 13px; font-weight: bold; margin-top: 2px;">
        {badge_html}
    </div>
    <div style="font-size: 11px; color: #64748B; margin-top: 4px;">{subtext_html}</div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. 27 MANDATORY RESEARCH NAVIGATION PAGES (PROMPT SPECIFICATION)
# -------------------------------------------------------------
pages = [
    "1. Executive Summary",
    "2. Dataset Explorer",
    "3. Full Dataset Inventory",
    "4. Attack Taxonomy",
    "5. Feature Engineering",
    "6. mRMR-JMI",
    "7. Model Architecture",
    "8. Pretrained Model",
    "9. Offline Performance",
    "10. Per-Attack Analysis",
    "11. Confusion Matrix",
    "12. ROC-AUC",
    "13. Precision-Recall",
    "14. Calibration",
    "15. Confidence Analysis",
    "16. Uncertainty Detection",
    "17. Redpanda Streaming",
    "18. Live IDS",
    "19. Live Metrics",
    "20. TP/TN/FP/FN",
    "21. Latency",
    "22. Throughput",
    "23. Ablation",
    "24. Model Comparison",
    "25. Explainability",
    "26. Error Analysis",
    "27. PhD Demonstration Mode"
]

selected_page = st.sidebar.radio("Navigation Menu:", pages, index=0)

# Sidebar System Controls
st.sidebar.markdown("---")
from dashboard.state import get_producer, get_consumer
prod = get_producer()
cons = get_consumer()
prod_status = '🟢 RUNNING' if (prod and prod.is_running and not getattr(prod, 'is_paused', False)) else ('⏸️ PAUSED' if (prod and getattr(prod, 'is_paused', False)) else '⚪ STOPPED')
cons_status = '🟢 RUNNING' if (cons and cons.is_running) else '⚪ STOPPED'
st.sidebar.write(f"**Producer**: {prod_status}")
st.sidebar.write(f"**Inference**: {cons_status}")

# Lazy-loaded page dispatching
if selected_page == "1. Executive Summary":
    from dashboard.overview import render_overview_page
    render_overview_page()
elif selected_page == "2. Dataset Explorer":
    from dashboard.dataset_page import render_dataset_page
    render_dataset_page()
elif selected_page == "3. Full Dataset Inventory":
    from dashboard.inventory_page import render_inventory_page
    render_inventory_page()
elif selected_page == "4. Attack Taxonomy":
    from dashboard.taxonomy_page import render_taxonomy_page
    render_taxonomy_page()
elif selected_page == "5. Feature Engineering":
    from dashboard.preprocessing_page import render_preprocessing_page
    render_preprocessing_page()
elif selected_page == "6. mRMR-JMI":
    from dashboard.mrmr_page import render_mrmr_page
    render_mrmr_page()
elif selected_page == "7. Model Architecture":
    from dashboard.architecture_page import render_architecture_page
    render_architecture_page()
elif selected_page == "8. Pretrained Model":
    from dashboard.pretrained_page import render_pretrained_page
    render_pretrained_page()
elif selected_page == "9. Offline Performance":
    from dashboard.model_perf_page import render_model_perf_page
    render_model_perf_page()
elif selected_page == "10. Per-Attack Analysis":
    from dashboard.per_attack_page import render_per_attack_page
    render_per_attack_page()
elif selected_page == "11. Confusion Matrix":
    from dashboard.confusion_page import render_confusion_page
    render_confusion_page()
elif selected_page == "12. ROC-AUC":
    from dashboard.roc_page import render_roc_page
    render_roc_page()
elif selected_page == "13. Precision-Recall":
    from dashboard.pr_page import render_pr_page
    render_pr_page()
elif selected_page == "14. Calibration":
    from dashboard.calibration_page import render_calibration_page
    render_calibration_page()
elif selected_page == "15. Confidence Analysis":
    from dashboard.confidence_page import render_confidence_page
    render_confidence_page()
elif selected_page == "16. Uncertainty Detection":
    from dashboard.uncertainty_page import render_uncertainty_page
    render_uncertainty_page()
elif selected_page == "17. Redpanda Streaming":
    from dashboard.streaming_page import render_streaming_page
    render_streaming_page()
elif selected_page == "18. Live IDS":
    from dashboard.live_ids_page import render_live_ids_page
    render_live_ids_page()
elif selected_page == "19. Live Metrics":
    from dashboard.live_metrics_page import render_live_metrics_page
    render_live_metrics_page()
elif selected_page == "20. TP/TN/FP/FN":
    from dashboard.tp_tn_fp_fn_page import render_tp_tn_fp_fn_page
    render_tp_tn_fp_fn_page()
elif selected_page == "21. Latency":
    from dashboard.latency_page import render_latency_page
    render_latency_page()
elif selected_page == "22. Throughput":
    from dashboard.throughput_page import render_throughput_page
    render_throughput_page()
elif selected_page == "23. Ablation":
    from dashboard.ablation_page import render_ablation_page
    render_ablation_page()
elif selected_page == "24. Model Comparison":
    from dashboard.model_comparison_page import render_model_comparison_page
    render_model_comparison_page()
elif selected_page == "25. Explainability":
    from dashboard.explainability_page import render_explainability_page
    render_explainability_page()
elif selected_page == "26. Error Analysis":
    from dashboard.error_page import render_error_page
    render_error_page()
elif selected_page == "27. PhD Demonstration Mode":
    from dashboard.phd_demo_page import render_phd_demo_page
    render_phd_demo_page()

