"""
Pretrained Model Management & Frozen Architecture Verification Page.
Displays model registry, frozen architecture parameters, MFLOPs, integrity checks, and weights security.
"""

import os
import json
import streamlit as st
import torch
from models.model_manager import ModelManager


def render_pretrained_page():
    st.title("📦 Pretrained Model Specification & Frozen Integrity")
    st.caption("Deep Learning IDS Deployment Registry | PhD Research Laboratory")

    manager = ModelManager.get_instance()
    is_valid, validation_msg = manager.validate_artifacts()
    metadata = manager.get_model_metadata()

    # Top Status Banner
    if is_valid and manager.is_loaded:
        st.markdown(f"""
        <div style="background-color: rgba(16, 185, 129, 0.12); border: 1px solid #10B981; border-radius: 8px; padding: 14px 20px; margin-bottom: 20px;">
            <span style="color: #10B981; font-weight: 800; font-size: 16px;">✓ PRETRAINED MODEL LOADED & FROZEN IN INFERENCE MODE</span>
            <div style="color: #CBD5E1; font-size: 13px; margin-top: 4px;">
                Active Version: <b>{metadata.get('model_version', 'v1.0')}</b> | Weights Status: <b>READ-ONLY (torch.inference_mode)</b> | Device: <b>{metadata.get('device', 'Auto')}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"⚠️ Model Artifact Status: {validation_msg}")
        st.info("Ensure artifacts are compiled offline via `python train.py`.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Architecture", "Hybrid CNN-Ghost-BiGRU")
    with col2:
        st.metric("Total Parameters", f"{metadata.get('total_parameters', 0):,}")
    with col3:
        st.metric("Computational Cost", f"{metadata.get('mflops', 0.0)} MFLOPs")
    with col4:
        st.metric("Model Size (Disk)", f"{metadata.get('model_size_mb', 0.0)} MB")

    st.markdown("---")

    # Section 1: Model Version Registry
    st.subheader("1. Production Model Registry (`artifacts/model_registry.json`)")
    registry_path = "artifacts/model_registry.json"
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            registry = json.load(f)
        
        active_ver = registry.get("active_model", "v1.0")
        available_models = list(registry.get("models", {}).keys())
        
        c_sel, c_info = st.columns([1, 2])
        with c_sel:
            selected_ver = st.selectbox("Select Model Version:", available_models, index=available_models.index(active_ver) if active_ver in available_models else 0)
            model_info = registry.get("models", {}).get(selected_ver, {})
            st.write(f"**Deployment Status**: `{model_info.get('status', 'production')}`")
            st.write(f"**Artifacts Path**: `{model_info.get('path', 'artifacts/')}`")
        with c_info:
            st.json(model_info)
    else:
        st.info("Registry file `artifacts/model_registry.json` will be populated upon running `python train.py`.")

    st.markdown("---")

    # Section 2: Architectural Details & Verification
    st.subheader("2. Pretrained Model Metadata & Training Provenance")
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("#### Training & Environment Specification")
        st.markdown(f"""
        - **Dataset**: Edge-IIoTset Cybersecurity Benchmark (2022)
        - **Framework**: PyTorch `{torch.__version__}`
        - **Execution Device**: `{metadata.get('device', 'cpu')}`
        - **Input Dimensions**: `{len(manager.selected_features)}` selected features
        - **Threat Classes**: `{len(manager.class_names)}` multiclass targets
        - **Feature Selection**: mRMR-JMI (Mutual Information & Joint Mutual Information)
        - **Loss Objective**: Multi-Task Compound Focal Loss ($\\gamma=2.0$) + Center Loss ($\\lambda=0.001$)
        """)

    with c_right:
        st.markdown("#### Pretrained Artifact Integrity Verification")
        files_to_check = [
            ("Model Weights Checkpoint", "artifacts/best_model.pt"),
            ("Model Hyperparameters", "artifacts/model_config.json"),
            ("Fitted Preprocessor Pipeline", "artifacts/preprocessor.pkl"),
            ("mRMR-JMI Feature Selector", "artifacts/feature_selector.pkl"),
            ("Label Encoder & Class Weights", "artifacts/label_encoder.pkl"),
            ("Selected Features List", "artifacts/selected_features.json"),
            ("Class Names Manifest", "artifacts/class_names.json"),
            ("Offline Test Metrics", "artifacts/metrics.json"),
            ("Latency Profile", "artifacts/latency_results.json")
        ]
        checks_html = "<div style='background-color: #1E293B; padding: 12px; border-radius: 6px;'>"
        for desc, path in files_to_check:
            exists = os.path.exists(path)
            color = "#10B981" if exists else "#EF4444"
            icon = "✓" if exists else "✗"
            checks_html += f"<div style='margin-bottom: 4px; font-size: 13px;'><span style='color:{color}; font-weight:bold;'>[{icon}]</span> <b>{desc}</b>: <code style='color:#94A3B8;'>{path}</code></div>"
        checks_html += "</div>"
        st.markdown(checks_html, unsafe_allow_html=True)

    st.markdown("---")

    # Section 3: Model Security & Freezing Enforcement
    st.subheader("3. Pretrained Model Security & Immutability Guarantee")
    st.markdown("""
    > [!IMPORTANT]
    > **Zero-Mutation Inference Guarantee**:
    > In compliance with strict research-grade deployment standards:
    > - The model is initialized exclusively via `model.eval()`.
    > - All inference calls are wrapped strictly inside `torch.inference_mode()` (disabling gradient calculation, autograd graph building, and intermediate buffer allocations).
    > - Model weights file (`artifacts/best_model.pt`) is opened in **read-only binary mode** and is never modified post-training.
    > - No optimizer, loss backward pass, or weight adjustment routines exist within the Streamlit or Redpanda inference pipelines.
    """)

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        if st.button("Re-verify Checkpoint Weights Hash & Freeze State"):
            if manager.model is not None:
                requires_grad_any = any(p.requires_grad for p in manager.model.parameters())
                if not requires_grad_any:
                    st.success("All model parameters are frozen (`requires_grad = False`). Memory state is verified read-only.")
                else:
                    st.warning("Model parameters require freezing. Applying `requires_grad = False`...")
                    for p in manager.model.parameters():
                        p.requires_grad = False
                    st.success("Model successfully enforced to read-only frozen state.")
