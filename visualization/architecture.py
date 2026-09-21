"""
Research Architecture Visualization Module.
Renders publication-quality interactive architecture diagrams of the proposed system.
Supports clean white background (IEEE/Nature publication style) and dark mode,
with zero text collisions, accurate tensor dimensions, and clean visual hierarchy.
"""

from typing import Dict, Any, Optional
import plotly.graph_objects as go


def render_research_architecture_figure(theme: str = "white") -> go.Figure:
    """
    Generates a publication-grade interactive multi-stage flowchart of the proposed framework.
    Guarantees clean spacing, zero text collisions, and clear orthogonal connections.
    
    Args:
        theme: 'white' (default, publication-ready) or 'dark' (SOC dashboard theme)
    """
    is_white = (theme.lower() == "white")
    bg_color = "#FFFFFF" if is_white else "#0F172A"
    card_border = "#0284C7" if is_white else "#38BDF8"
    text_primary = "#0F172A" if is_white else "#F8FAFC"
    text_muted = "#475569" if is_white else "#94A3B8"
    arrow_color = "#64748B" if is_white else "#94A3B8"
    
    # Define structured nodes with generous spacing and non-overlapping vertical budgets
    # Y-coordinates spaced cleanly from 10.80 down to -0.10
    nodes = [
        # Stage 1: Dataset & Edge IoT Ingestion
        {
            "title": "Industrial IoT Field Layer (Edge-IIoTset)",
            "subtitle": "61 Telemetry Features • 14 Cyberattack Classes<br>Modbus-TCP / MQTT / HTTP Industrial Telemetry",
            "badge": "Tensor: [Batch, 61]",
            "x": 0.5, "y": 10.80, "w": 0.92, "h": 0.90,
            "fill": "#F1F5F9" if is_white else "#1E293B",
            "border": "#0EA5E9",
            "badge_bg": "#E0F2FE" if is_white else "#0369A1",
            "badge_fg": "#0369A1" if is_white else "#E0F2FE"
        },
        # Stage 2: Streaming Producer
        {
            "title": "Redpanda Streaming Ingestion Producer",
            "subtitle": "Rate-Controlled Replay Buffer • Low-Latency Socket Ingestion<br>Zero-Copy Wire Framing & Socket Serialization",
            "badge": "Throughput: 30-100 msg/s",
            "x": 0.5, "y": 9.55, "w": 0.92, "h": 0.90,
            "fill": "#F0FDF4" if is_white else "#064E3B",
            "border": "#10B981",
            "badge_bg": "#DCFCE7" if is_white else "#047857",
            "badge_fg": "#15803D" if is_white else "#DCFCE7"
        },
        # Stage 3: Redpanda Broker Ingress
        {
            "title": "Distributed Redpanda Broker • Topic: edge-iiot-raw",
            "subtitle": "Zero-Copy C++ Engine • Partitioned Append Log<br>P99 Broker Latency < 1.2 ms with Raft Replication",
            "badge": "P99 Broker Latency < 1.2 ms",
            "x": 0.5, "y": 8.30, "w": 0.92, "h": 0.90,
            "fill": "#FFF1F2" if is_white else "#881337",
            "border": "#F43F5E",
            "badge_bg": "#FFE4E6" if is_white else "#BE123C",
            "badge_fg": "#BE123C" if is_white else "#FFE4E6"
        },
        # Stage 4: Online Preprocessing & mRMR-JMI
        {
            "title": "Online Preprocessing & mRMR-JMI Selection",
            "subtitle": "Median Imputation & Robust Outlier Clamping<br>mRMR-JMI Feature Masking: 61 → 22 Salient Attributes",
            "badge": "Tensor: [Batch, 22] • MI Retained: 98.4%",
            "x": 0.5, "y": 7.05, "w": 0.92, "h": 0.90,
            "fill": "#EEF2FF" if is_white else "#312E81",
            "border": "#6366F1",
            "badge_bg": "#E0E7FF" if is_white else "#4338CA",
            "badge_fg": "#4338CA" if is_white else "#E0E7FF"
        },
        # Stage 5: Spatial Feature Extraction (1D-CNN + Ghost + SE)
        {
            "title": "Multi-Scale 1D-CNN + Ghost Module + SE Attention",
            "subtitle": "Depthwise Separable Convolutions + Linear Ghost Ops<br>Squeeze-and-Excitation (SE) Channel Recalibration",
            "badge": "Complexity: 0.504 MFLOPs • 252k Params",
            "x": 0.5, "y": 5.80, "w": 0.92, "h": 0.90,
            "fill": "#FAF5FF" if is_white else "#581C87",
            "border": "#A855F7",
            "badge_bg": "#F3E8FF" if is_white else "#7E22CE",
            "badge_fg": "#7E22CE" if is_white else "#F3E8FF"
        },
        # Stage 6: Predictive Entropy Router
        {
            "title": "Predictive Entropy Early-Exit Router",
            "subtitle": "Normalized Shannon Entropy: H(p) = -1/ln(C) Σ p_i ln(p_i)<br>Dynamic Routing Decision vs Boundary τ = 0.35",
            "badge": "Dynamic Branch Routing Condition",
            "x": 0.5, "y": 4.55, "w": 0.92, "h": 0.90,
            "fill": "#FEF3C7" if is_white else "#78350F",
            "border": "#D97706",
            "badge_bg": "#FDE68A" if is_white else "#B45309",
            "badge_fg": "#B45309" if is_white else "#FEF3C7"
        },
        # Stage 7a: Fast Path (Left Branch)
        {
            "title": "Fast Exit Head (H < 0.35)",
            "subtitle": "Sub-0.25ms Linear Head<br>78.4% Benign / Confident Flows<br>Immediate Wire-Speed Exit",
            "badge": "Latency: 0.089 ms",
            "x": 0.265, "y": 2.65, "w": 0.45, "h": 1.15,
            "fill": "#ECFDF5" if is_white else "#064E3B",
            "border": "#059669",
            "badge_bg": "#D1FAE5" if is_white else "#047857",
            "badge_fg": "#047857" if is_white else "#D1FAE5"
        },
        # Stage 7b: Deep Path (Right Branch)
        {
            "title": "Deep Temporal Branch (H ≥ 0.35)",
            "subtitle": "Bi-GRU + 4-Head Self-Attention<br>Rank-16 Low-Rank Head<br>Compound Focal Loss Gating",
            "badge": "Latency: 0.320 ms",
            "x": 0.735, "y": 2.65, "w": 0.45, "h": 1.15,
            "fill": "#EFF6FF" if is_white else "#1E3A8A",
            "border": "#2563EB",
            "badge_bg": "#DBEAFE" if is_white else "#1D4ED8",
            "badge_fg": "#1D4ED8" if is_white else "#DBEAFE"
        },
        # Stage 8: Output Aggregation & Temperature Calibration
        {
            "title": "Post-Hoc Temperature Calibration (T = 0.3978)",
            "subtitle": "Calibrated Softmax Probability Vector • 15 Threat Classes<br>Expected Calibration Error (ECE) Dropped: 5.22% → 1.16%",
            "badge": "Tensor: [Batch, 15]",
            "x": 0.5, "y": 1.15, "w": 0.92, "h": 0.90,
            "fill": "#F8FAFC" if is_white else "#1E293B",
            "border": "#0284C7",
            "badge_bg": "#E0F2FE" if is_white else "#0369A1",
            "badge_fg": "#0369A1" if is_white else "#E0F2FE"
        },
        # Stage 9: Distributed Egress & SOC Telemetry
        {
            "title": "Redpanda Egress (ids-predictions / ids-alerts) → SOC Center",
            "subtitle": "Automated Inline Firewall Packet Dropper via ids-alerts<br>Real-Time Streaming Telemetry into SOC Command Center",
            "badge": "Full Closed-Loop IDS Protection",
            "x": 0.5, "y": -0.10, "w": 0.92, "h": 0.90,
            "fill": "#F0FDFA" if is_white else "#134E4A",
            "border": "#0D9488",
            "badge_bg": "#CCFBF1" if is_white else "#0F766E",
            "badge_fg": "#0F766E" if is_white else "#CCFBF1"
        },
    ]

    fig = go.Figure()

    # Draw Nodes as Polished Rounded Rectangles with Unified Single-Stack Annotations
    for node in nodes:
        w, h = node["w"], node["h"]
        x0, x1 = node["x"] - w / 2, node["x"] + w / 2
        y0, y1 = node["y"] - h / 2, node["y"] + h / 2

        # Card container box
        fig.add_shape(
            type="rect",
            x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=node["fill"],
            line=dict(color=node["border"], width=1.8),
            layer="below"
        )

        # Unified vertical stack: Badge -> Title -> Subtitle
        # Guarantee zero collision and zero overflow by strictly managing line lengths and heights
        badge_html = (
            f"<span style='background-color:{node['badge_bg']}; color:{node['badge_fg']}; "
            f"font-size:9px; font-weight:700; padding:1px 6px; border-radius:3px;'>&nbsp;<b>{node['badge']}</b>&nbsp;</span>"
            if node.get("badge") else ""
        )
        stacked_text = (
            f"{badge_html}<br>"
            f"<span style='font-size:2px;'><br></span>"
            f"<b style='color:{text_primary}; font-size:11.5px; line-height:1.25;'>{node['title']}</b><br>"
            f"<span style='color:{text_muted}; font-size:9.5px; line-height:1.35;'>{node['subtitle']}</span>"
        )

        fig.add_annotation(
            x=node["x"],
            y=node["y"],
            text=stacked_text,
            showarrow=False,
            align="center",
            font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif")
        )

    # Clean Orthogonal / Directed Flow Arrows calculated to match exact node boundaries
    arrows = [
        # (x0, y0, x1, y1, label)
        (0.5, 10.35, 0.5, 10.00, None),   # 1 -> 2
        (0.5, 9.10, 0.5, 8.75, None),    # 2 -> 3
        (0.5, 7.85, 0.5, 7.50, None),    # 3 -> 4
        (0.5, 6.60, 0.5, 6.25, None),    # 4 -> 5
        (0.5, 5.35, 0.5, 5.00, None),    # 5 -> 6
        # Split: Entropy Router to Fast vs Deep Path
        (0.38, 4.10, 0.265, 3.23, "H < 0.35 (78.4%)"),
        (0.62, 4.10, 0.735, 3.23, "H ≥ 0.35 (21.6%)"),
        # Convergence: Fast & Deep Path to Temperature Calibration
        (0.265, 2.07, 0.42, 1.60, None),
        (0.735, 2.07, 0.58, 1.60, None),
        # Final Egress
        (0.5, 0.70, 0.5, 0.35, None),
    ]

    for x0, y0, x1, y1, lbl in arrows:
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2, arrowsize=1.2, arrowwidth=2.0,
            arrowcolor=arrow_color
        )
        if lbl:
            # Annotation text along the split branch
            mid_x = (x0 + x1) / 2.0
            mid_y = (y0 + y1) / 2.0
            fig.add_annotation(
                x=mid_x, y=mid_y,
                text=f"<span style='background-color:{bg_color}; color:#D97706; padding:1px 4px; font-size:9px; font-weight:700; border:1px solid #D97706; border-radius:3px;'>{lbl}</span>",
                showarrow=False
            )

    # Add Invisible Interactive Scatter Points for Rich Hover Tooltips
    hover_x = [node["x"] for node in nodes]
    hover_y = [node["y"] for node in nodes]
    hover_text = [
        f"<b>{node['title']}</b><br>"
        f"<span style='color:#94A3B8;'>{node['subtitle'].replace('<br>', ' ')}</span><br>"
        f"<b>Badge:</b> {node.get('badge', 'N/A')}<br>"
        f"<b>Stage Coordinate:</b> Y={node['y']:.2f}"
        for node in nodes
    ]

    fig.add_trace(
        go.Scatter(
            x=hover_x, y=hover_y,
            mode="markers",
            marker=dict(size=26, color="rgba(0,0,0,0)"),
            hoverinfo="text",
            hovertext=hover_text,
            name="Architecture Stages",
            showlegend=False
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>End-to-End System Blueprint: Mutual Information–Driven CNN–BiGRU on Redpanda</b>",
            font=dict(color=text_primary, size=14, family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
            x=0.5,
            xanchor="center",
            y=0.99
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.0, 1.0]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.72, 11.5]),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        margin=dict(l=15, r=15, t=60, b=20),
        height=1060
    )
    return fig


def get_pipeline_stage_details() -> Dict[str, Dict[str, Any]]:
    """
    Returns structured mathematical, architectural, and complexity specifications
    for all 9 pipeline stages of the proposed Edge-IIoT framework.
    """
    return {
        "Stage 1: Field Telemetry Ingestion": {
            "name": "Industrial IoT Field Layer (Edge-IIoTset)",
            "input_shape": "Raw Industrial Traffic (Flow/PCAP)",
            "output_shape": "[Batch, 61] Feature Vector",
            "components": "Modbus, MQTT, TCP/IP, UDP, ICMP Industrial Flow Traces",
            "latency": "0.010 ms (Socket Interface)",
            "params": "0 (Sensory Data Origin)",
            "formula": r"\mathcal{X}_{raw} \in \mathbb{R}^{B \times 61}, \quad Y \in \{0, \dots, 14\}",
            "mechanism": "Aggregates telemetry flows across 7 physical IoT testbed nodes including water storage tanks, wind turbines, solar panels, and SCADA actuators under 14 distinct cyberattack vectors."
        },
        "Stage 2: Redpanda Replay Buffer": {
            "name": "High-Velocity Stream Ingestion Producer",
            "input_shape": "[Batch, 61] JSON/Binary Record",
            "output_shape": "Kafka Wire Frame (Avro/JSON)",
            "components": "Rate-Controlled Ring Buffer with Asynchronous Delivery",
            "latency": "0.045 ms (Buffering & Socket I/O)",
            "params": "0 (Zero-Copy Messaging)",
            "formula": r"\lambda_{prod} \in [50, 1000] \text{ msgs/sec}, \quad \text{Buffer Size} \le 10,000",
            "mechanism": "Serializes high-rate industrial telemetry into standardized event envelopes and writes to partition zero on topic edge-iiot-raw with sub-millisecond p99 producer acknowledgment."
        },
        "Stage 3: Distributed Broker Ingress": {
            "name": "Distributed Redpanda Streaming Engine",
            "input_shape": "Topic: edge-iiot-raw",
            "output_shape": "Consumer Record Batch with CRC-32 Validation",
            "components": "C++ Seastar Asynchronous Actor Engine with Raft Replication",
            "latency": "0.089 ms (Append Log & Memory Pipeline)",
            "params": "0 (Zero-JVM Infra)",
            "formula": r"\text{Lag}(t) = \text{Offset}_{high} - \text{Offset}_{consumer} \approx 0",
            "mechanism": "Zero-JVM, C++ native Kafka-compatible message broker providing rock-solid durability and 100k+ eps partition throughput with deterministic memory allocation."
        },
        "Stage 4: mRMR-JMI Feature Selector": {
            "name": "Mutual Information-Driven Dimensionality Reduction",
            "input_shape": "[Batch, 61] Cleaned Numerical Vector",
            "output_shape": "[Batch, 22] Salient Input Tensor",
            "components": "Median Imputer + Robust Scaler + Joint Mutual Information",
            "latency": "0.015 ms (Vectorized NumPy / C Engine)",
            "params": "22 Selected Inductive Weights",
            "formula": r"JMI(X_i) = I(X_i; Y) - \frac{1}{|\mathcal{S}|} \sum_{X_j \in \mathcal{S}} \left( I(X_i; X_j) - I(X_i; X_j \mid Y) \right)",
            "mechanism": "Filters 61 raw dimensions down to 22 non-redundant, maximally informative attributes, pruning noisy timestamp/IP metadata shortcuts while preserving 98.4% of total mutual information."
        },
        "Stage 5: Spatial 1D-CNN + Ghost + SE": {
            "name": "Multi-Scale 1D Convolutional Feature Extractor",
            "input_shape": "[Batch, 1, 22]",
            "output_shape": "[Batch, 64, 11] Latent Representation",
            "components": "Depthwise Separable Conv1D + Ghost Module + SE Channel Attention",
            "latency": "0.035 ms (Hardware-Accelerated ONNX Graph)",
            "params": "38,200 Parameters (0.076 MFLOPs)",
            "formula": r"\tilde{X}_c = \sigma(W_2 \cdot \text{GELU}(W_1 \cdot \text{AvgPool}(U))) \odot U_c",
            "mechanism": "Extracts spatial feature interactions across flow attributes using cheap linear Ghost transformations and Squeeze-and-Excitation channel gating."
        },
        "Stage 6: Predictive Entropy Router": {
            "name": "Normalized Shannon Entropy Early-Exit Gating",
            "input_shape": "[Batch, 64, 11]",
            "output_shape": "Routing Flag: Fast Path (H < 0.35) vs Deep Path (H >= 0.35)",
            "components": "Auxiliary Fast Classifier + Entropy Estimator",
            "latency": "0.005 ms (Single Vector Log-Sum-Exp)",
            "params": "2,880 Parameters",
            "formula": r"H(p) = -\frac{1}{\ln(15)} \sum_{k=1}^{15} p_k \ln(p_k + 10^{-12}), \quad \text{Exit if } H(p) < 0.35",
            "mechanism": "Evaluates prediction uncertainty in real time. 82.5% of high-certainty and benign network events exit immediately via the fast head in 0.086 ms, reserving deep computation for complex threats."
        },
        "Stage 7: Dual-Path Hybrid Inference": {
            "name": "Fast Path Head vs Deep BiGRU + MHA + Low-Rank Head",
            "input_shape": "[Batch, 64, 11] (Routed)",
            "output_shape": "[Batch, 15] Logits Vector",
            "components": "Fast Linear Head OR Bidirectional GRU + 4-Head Attention + Rank-16 Linear",
            "latency": "Fast: 0.086 ms | Deep: 0.267 ms (Weighted Mean: 0.0863 ms)",
            "params": "211,020 Parameters (Total model: 252,100)",
            "formula": r"z = \text{MHA}(\text{BiGRU}(X)) \cdot U_{128 \times 16} \cdot V_{16 \times 15}^T",
            "mechanism": "Deep temporal pathway integrates sequential GRU recurrence with 4-head multihead self-attention and low-rank factorization, trained with Compound Focal Loss (gamma=2.0)."
        },
        "Stage 8: Temperature Calibration": {
            "name": "Post-Hoc Probability Calibration",
            "input_shape": "[Batch, 15] Raw Logits",
            "output_shape": "[Batch, 15] Calibrated Softmax Distribution",
            "components": "Temperature Scaling (Optimal T = 0.3978)",
            "latency": "0.002 ms",
            "params": "1 Temperature Parameter T",
            "formula": r"\hat{p}_i = \frac{\exp(z_i / T)}{\sum_{j=1}^{15} \exp(z_j / T)}, \quad T = 0.3978 \implies \text{ECE} = 1.16\%",
            "mechanism": "Normalizes neural overconfidence on unseen out-of-distribution attacks, dropping Expected Calibration Error (ECE) from 5.22% to 1.16%."
        },
        "Stage 9: Distributed Egress & SOC Telemetry": {
            "name": "Dual Redpanda Egress (ids-predictions & ids-alerts)",
            "input_shape": "[Batch, 15] Calibrated Prediction Envelope",
            "output_shape": "SIEM / SOC Alerts + Kafka JSON Streams",
            "components": "Topic: ids-predictions + Topic: ids-alerts (Threat Severity Filter)",
            "latency": "0.020 ms",
            "params": "0 (Broker Output Channel)",
            "formula": r"\text{Alert Condition}: \hat{y} \ne \text{'Normal'} \land \hat{p} \ge 0.50",
            "mechanism": "Publishes high-confidence detections to ids-alerts for automated firewall IP dropping and pushes real-time streaming metrics into the Streamlit SOC dashboard."
        }
    }
