"""
Page 25: Model Explainability & Attention Visualization.
Interactive Squeeze-and-Excitation channel weight inspector and Multi-Head Temporal Attention heatmaps.
Strictly offline-compatible: uses frozen resident ModelManager without dataset re-fitting.
"""

import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F

from models.model_manager import ModelManager
from visualization.attention import plot_se_channel_weights, plot_temporal_attention_heatmap


def render_explainability_page():
    st.title("💡 25. Model Explainability & Attention Maps")
    st.caption("Interpreting neural feature recalibration and sequence attention patterns across distinct network attacks.")

    manager = ModelManager.get_instance()
    if not manager.is_loaded:
        manager.load_artifacts()

    if not manager.is_loaded or manager.model is None:
        st.warning("⚠️ Pretrained model not loaded. Please run offline training first.")
        return

    class_names = manager.class_names
    selected_features = manager.selected_features
    device = manager.device
    model = manager.model

    st.subheader("🔍 Select Threat Vector for Explainability Inspection")
    c_sel, c_tweak = st.columns([1, 1])

    with c_sel:
        target_class = st.selectbox("Inspect Architecture Behaviors for Class:", class_names, index=0)

    # Deterministic representative activation vector for the selected class
    cls_idx = class_names.index(target_class)
    np.random.seed(42 + cls_idx * 7)
    base_vector = np.random.uniform(0.1, 0.9, size=(len(selected_features),))
    # Give distinctive feature spikes depending on class nature
    if "DDoS" in target_class:
        base_vector[0] = 3.5  # Port
        base_vector[5] = 4.2  # Packet length / flags
    elif "SQL" in target_class or "XSS" in target_class:
        base_vector[12] = 2.8 # Content length
        base_vector[18] = 3.1 # HTTP response
    elif "Normal" in target_class:
        base_vector = np.clip(base_vector * 0.4, 0.05, 0.5)

    with c_tweak:
        st.info(f"Target Threat: **{target_class}** (Class #{cls_idx}) | Feature Dimension: **{len(selected_features)}**")

    # Run zero-grad forward pass in train mode to extract SE and Attention internal states
    sample_tensor = torch.from_numpy(base_vector).float().unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(sample_tensor, routing_mode="train")
        se_weights = out["se_weights"].cpu().numpy()[0]
        attn_map = out["attn_map"].cpu().numpy()[0]
        fast_prob = out["fast_probs"].cpu().numpy()[0]
        deep_prob = out["deep_probs"].cpu().numpy()[0]
        entropy_val = float(out["entropy"][0].item())

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    pred_class_fast = class_names[int(np.argmax(fast_prob))]
    pred_class_deep = class_names[int(np.argmax(deep_prob))]
    m1.metric("Inspected Attack Type", target_class)
    m2.metric("Fast Path Prediction", pred_class_fast, delta=f"{np.max(fast_prob)*100:.1f}% conf")
    m3.metric("Deep Path Prediction", pred_class_deep, delta=f"{np.max(deep_prob)*100:.1f}% conf")
    m4.metric("Predictive Entropy H(p)", f"{entropy_val:.4f}", delta="Fast Exit" if entropy_val < 0.35 else "Deep Path Required")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Squeeze-and-Excitation (SE) Channel Recalibration")
        st.write(r"Learned channel importance weights $s_c \in [0, 1]$ adaptively gating 1D-CNN and Ghost feature maps:")
        fig_se = plot_se_channel_weights(se_weights, top_k=20)
        st.plotly_chart(fig_se, use_container_width=True)

    with col2:
        st.subheader("2. Multi-Head Temporal Self-Attention Matrix")
        st.write(r"Inter-timestep attention dependency matrix $A_{i,j} = \text{softmax}(Q K^T / \sqrt{d_k})$:")
        fig_attn = plot_temporal_attention_heatmap(attn_map)
        st.plotly_chart(fig_attn, use_container_width=True)
