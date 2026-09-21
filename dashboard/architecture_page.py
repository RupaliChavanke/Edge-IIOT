"""
Page 2: System Architecture & Theoretical Foundations.
Provides deep PhD-level mathematical descriptions, layer-by-layer formulas, and interactive diagrams.
"""

import os
import streamlit as st
import pandas as pd
from visualization.architecture import render_research_architecture_figure
from models.proposed_model import ProposedHybridEdgeIIoTModel


def render_architecture_page():
    st.title("🏛️ Proposed Research Architecture & Theoretical Model")
    st.caption("Mathematical formulation and modular structural breakdown of the hybrid streaming intrusion detection system.")

    from visualization.architecture import get_pipeline_stage_details
    stage_details = get_pipeline_stage_details()

    tab_vis1, tab_vis2, tab_vis3 = st.tabs([
        "🌐 3D Real-World Physical Architecture (White Studio)",
        "📐 Formal Publication Neural Pipeline Blueprint",
        "🔍 Interactive Stage-by-Stage Architecture Inspector"
    ])

    with tab_vis1:
        arch_3d_path = "artifacts/architecture_3d_realworld.png"
        if os.path.exists(arch_3d_path):
            st.markdown("""
            <div style="background:#FFFFFF; border-radius:12px; padding:16px; border:1px solid #E2E8F0; box-shadow:0 8px 24px rgba(0,0,0,0.15); margin-bottom:14px;">
            """, unsafe_allow_html=True)
            st.image(
                arch_3d_path, 
                caption="Figure 1: Authentic 3D Isometric Physical Architecture for Edge-IIoT Real-Time Intrusion Detection System (Clean White Background)",
                use_container_width=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

            with open(arch_3d_path, "rb") as f_img:
                st.download_button(
                    label="📥 Download High-Resolution 3D Architecture Diagram (PNG)",
                    data=f_img.read(),
                    file_name="edge_iiot_3d_realworld_architecture.png",
                    mime="image/png"
                )

        st.markdown("""
        <style>
        .component-pill {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.65) 100%);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 10px;
            padding: 12px 14px;
            margin-bottom: 10px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
            backdrop-filter: blur(8px);
            transition: all 0.25s ease;
            box-sizing: border-box;
        }
        .component-pill:hover {
            border-color: rgba(56, 189, 248, 0.6);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(56, 189, 248, 0.18);
        }
        .component-pill h5 {
            margin: 0 0 6px 0;
            font-size: 13px;
            font-weight: 700;
            color: #38BDF8;
            display: flex;
            align-items: center;
            gap: 6px;
            letter-spacing: 0.2px;
        }
        .component-pill p {
            margin: 0;
            font-size: 11.5px;
            color: #CBD5E1;
            line-height: 1.5;
        }
        .component-pill code {
            background: rgba(16, 185, 129, 0.2);
            color: #34D399;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', 'Courier New', monospace;
            font-size: 10.5px;
            border: 1px solid rgba(16, 185, 129, 0.35);
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("#### 🛠️ Real-World Physical System Breakdown")
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            st.markdown("""
            <div class="component-pill">
                <h5>🏭 1. Industrial IoT Field Sensing (Edge-IIoTset)</h5>
                <p>Siemens S7-1200 / Modbus-TCP PLCs and smart factory environmental sensors transmitting 61 raw network flow features under heavy industrial noise.</p>
            </div>
            <div class="component-pill">
                <h5>⚡ 2. Ruggedized DIN-Rail Edge Gateway</h5>
                <p>Low-power industrial IoT edge computing router performing wire-speed packet capture, preliminary framing, and socket serialization.</p>
            </div>
            <div class="component-pill">
                <h5>🚀 3. Distributed Redpanda Ingestion Broker</h5>
                <p>Enterprise C++ Kafka-compatible engine receiving raw event streams on topic <code>edge-iiot-raw</code> with zero memory copies and P99 latency under 1.2 ms.</p>
            </div>
            """, unsafe_allow_html=True)

        with c_p2:
            st.markdown("""
            <div class="component-pill">
                <h5>🧠 4. Edge AI Neural Inference Accelerator</h5>
                <p>Hardware-accelerated ONNX runtime executing fused Depthwise Conv1D, Ghost Modules, SE Attention, and BiGRU in <b>0.089 ms</b> with 9,896 eps throughput.</p>
            </div>
            <div class="component-pill">
                <h5>🛡️ 5. Dual-Monitor SOC Cyber Defense Workstation</h5>
                <p>Automated inline firewall mitigation, predictive entropy routing diagnostics, and real-time 15-class threat classification with human-in-the-loop overrides.</p>
            </div>
            <div class="component-pill">
                <h5>🔒 6. Automated Inline Threat Mitigation Engine</h5>
                <p>Autonomous TCP reset, dynamic firewall rule synthesis, and sub-second zero-trust edge isolation via <code>ids-alerts</code> topic.</p>
            </div>
            """, unsafe_allow_html=True)

    with tab_vis2:
        theme_sel = st.radio("Blueprint Theme:", ["Clean White (Publication)", "Cyber Dark"], horizontal=True, key="arch_page_theme")
        st.plotly_chart(
            render_research_architecture_figure(theme="white" if "White" in theme_sel else "dark"),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True}
        )

    with tab_vis3:
        st.markdown("### 🔍 Interactive Deep Architecture & Tensor Inspector")
        st.caption("Click through any pipeline stage to inspect its mathematical transformations, tensor shapes, parameters, and latency budgets.")

        selected_stage_key = st.selectbox(
            "Select Pipeline Stage to Inspect:",
            options=list(stage_details.keys()),
            index=4,  # Default to Stage 5 (Spatial 1D-CNN + Ghost + SE)
            key="arch_stage_inspector_select"
        )
        st_info = stage_details[selected_stage_key]

        c_box1, c_box2, c_box3 = st.columns(3)
        with c_box1:
            st.metric("Input Tensor", st_info["input_shape"])
        with c_box2:
            st.metric("Output Tensor", st_info["output_shape"])
        with c_box3:
            st.metric("Stage Latency", st_info["latency"])

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 10px; padding: 18px; margin: 14px 0;">
            <div style="font-size: 16px; font-weight: 800; color: #38BDF8;">{st_info['name']}</div>
            <div style="font-size: 12px; color: #94A3B8; margin-top: 4px;"><b>Active Components:</b> {st_info['components']}</div>
            <div style="font-size: 12px; color: #10B981; margin-top: 4px;"><b>Parameter Footprint:</b> {st_info['params']}</div>
            <div style="font-size: 13px; color: #E2E8F0; margin-top: 10px; line-height: 1.6;">{st_info['mechanism']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Mathematical Formulation:**")
        st.latex(st_info["formula"])

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Mathematical Formulations",
        "📐 Layer-by-Layer Architecture",
        "⚖️ Compound Loss Design",
        "📊 Computational Complexity (FLOPs/Params)"
    ])

    with tab1:
        st.markdown(r"""
        ### Theoretical Formulations of Core Components

        #### 1. mRMR-JMI Feature Selection
        Given candidate feature set $\mathcal{F} = \{X_1, \dots, X_M\}$ and multiclass intrusion target $Y$:
        $$\text{Relevance}(X_i) = I(X_i; Y) = \iint p(x_i, y) \log \frac{p(x_i, y)}{p(x_i)p(y)} \, dx_i \, dy$$
        The Joint Mutual Information (JMI) criterion for selecting feature $X_i \in \mathcal{F} \setminus \mathcal{S}$ into currently selected subset $\mathcal{S}$ is:
        $$JMI(X_i) = I(X_i; Y) - \frac{1}{|\mathcal{S}|} \sum_{X_j \in \mathcal{S}} \left( I(X_i; X_j) - I(X_i; X_j \mid Y) \right)$$

        #### 2. Depthwise Separable 1D-CNN
        Standard 1D convolution with kernel size $K$, input channels $C_{in}$, and output channels $C_{out}$ requires $K \times C_{in} \times C_{out}$ parameters.
        Our depthwise separable design decomposes this into:
        1. **Depthwise Conv**: $K \times C_{in} \times 1$
        2. **Pointwise Conv**: $1 \times C_{in} \times C_{out}$
        $$\text{Parameter Reduction Ratio} = \frac{K \cdot C_{in} + C_{in} \cdot C_{out}}{K \cdot C_{in} \cdot C_{out}} = \frac{1}{C_{out}} + \frac{1}{K}$$

        #### 3. Ghost Module 1D
        Instead of generating all $C_{out}$ channels via heavy convolutions, we generate $m = \lceil C_{out} / s \rceil$ intrinsic feature maps via primary convolution, then apply cheap linear depthwise transforms $\Phi_{i,j}$:
        $$y_{i,j} = \Phi_{i,j}(y_i'), \quad \forall i=1,\dots,m, \quad j=1,\dots,s-1$$

        #### 4. Squeeze-and-Excitation (SE) Channel Recalibration
        - **Squeeze**: Global spatial pooling $z_c = \frac{1}{L} \sum_{l=1}^L u_c(l)$
        - **Excitation**: $s = \sigma(W_2 \cdot \text{GELU}(W_1 \cdot z))$, where $W_1 \in \mathbb{R}^{\frac{C}{r} \times C}, W_2 \in \mathbb{R}^{C \times \frac{C}{r}}$ with reduction ratio $r=8$.
        - **Recalibration**: $\tilde{X}_c = s_c \cdot U_c$

        #### 5. Normalized Shannon Predictive Entropy Router
        For candidate distribution $p \in \Delta^{C-1}$:
        $$H(p) = -\frac{1}{\ln(C)} \sum_{k=1}^C p_k \ln(p_k + \epsilon), \quad H(p) \in [0, 1]$$
        $$\text{Routing Decision}(x) = \begin{cases} \text{Fast Path Head (Exit)}, & \text{if } H(p) < \tau \\ \text{Deep Temporal Path (Bi-GRU + MHA + Low-Rank)}, & \text{if } H(p) \ge \tau \end{cases}$$
        """)

    with tab2:
        st.markdown(r"""
        ### Deep Temporal Path Specifications

        #### 1. Bidirectional Gated Recurrent Unit (Bi-GRU)
        Models forward and reverse packet context:
        $$\overrightarrow{h_t} = \text{GRU}(x_t, \overrightarrow{h_{t-1}}), \quad \overleftarrow{h_t} = \text{GRU}(x_t, \overleftarrow{h_{t+1}})$$
        $$H_t = [\overrightarrow{h_t} \,\|\, \overleftarrow{h_t}] \in \mathbb{R}^{2 \cdot d_{hidden}}$$

        #### 2. Multi-Head Temporal Self-Attention (MHA)
        With $h=4$ attention heads, queries, keys, and values are linearly projected:
        $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
        $$\text{MHA}(H) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

        #### 3. Pointwise Low-Rank Factorization
        Dense output matrix $W \in \mathbb{R}^{D_{in} \times D_{out}}$ is factorized into $U \in \mathbb{R}^{D_{in} \times r}$ and $V \in \mathbb{R}^{r \times D_{out}}$:
        $$z = x \cdot U \cdot V^T, \quad \text{with rank } r=16 \ll \min(D_{in}, D_{out})$$
        Reduces parameters from $2048 \times 128 = 262,144$ down to $16 \times (2048 + 128) = 34,816$ (an **86.7% reduction**).
        """)

    with tab3:
        st.markdown(r"""
        ### Multi-Task Compound Objective Function
        To address severe class imbalance in Edge-IIoTset while enforcing high intra-class compactness and inter-class separability:
        $$\mathcal{L}_{total} = \lambda_{focal} \mathcal{L}_{focal} + \lambda_{center} \mathcal{L}_{center}$$

        #### Focal Loss Term
        $$\mathcal{L}_{focal} = -\sum_{k=1}^C \alpha_k (1 - p_k)^\gamma y_k \log(p_k)$$
        - Downweights well-classified easy samples ($(1 - p_k)^\gamma \to 0$), forcing gradients to focus on rare attack categories (e.g., Fingerprinting, MITM).
        - Setting $\gamma = 2.0$, $\alpha_k$ set to inverse class frequency.

        #### Center Loss Term
        $$\mathcal{L}_{center} = \frac{1}{2B} \sum_{i=1}^B \| f(x_i) - c_{y_i} \|_2^2$$
        - Pulls deep latent embeddings $f(x_i)$ towards dynamically updated class centroids $c_{y_i}$, minimizing false positive confusion between benign background traffic and stealthy attacks.
        """)

    with tab4:
        model = ProposedHybridEdgeIIoTModel()
        param_dict = model.count_parameters()

        st.subheader("Model Parameter Budget Breakdown")
        df_p = pd.DataFrame([
            {"Submodule": k.replace("_parameters", "").title(), "Parameters": v}
            for k, v in param_dict.items() if k != "total_parameters"
        ])
        st.dataframe(df_p, use_container_width=True)
        st.metric("Total Model Parameters", f"{param_dict['total_parameters']:,}")
