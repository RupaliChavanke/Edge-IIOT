"""
Page 2: System Architecture & Theoretical Foundations.
Provides deep PhD-level mathematical descriptions, layer-by-layer formulas, and interactive diagrams.
"""

import streamlit as st
import pandas as pd
from visualization.architecture import render_research_architecture_figure
from models.proposed_model import ProposedHybridEdgeIIoTModel


def render_architecture_page():
    st.title("🏛️ Proposed Research Architecture & Theoretical Model")
    st.caption("Mathematical formulation and modular structural breakdown of the hybrid streaming intrusion detection system.")

    st.plotly_chart(render_research_architecture_figure(), use_container_width=True)

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
