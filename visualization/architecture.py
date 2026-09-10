"""
Research Architecture Visualization Module.
Renders publication-quality interactive architecture diagrams of the proposed system.
"""

import plotly.graph_objects as go


def render_research_architecture_figure() -> go.Figure:
    """Generates an interactive multi-stage flowchart of the proposed framework."""
    stages = [
        # Data & Streaming Layer
        {"name": "Edge-IIoTset Dataset\n(61 Network Features)", "x": 0.5, "y": 9.5, "color": "#1E293B", "width": 0.6, "height": 0.6},
        {"name": "Redpanda Streaming Producer\n(Rate Controlled)", "x": 0.5, "y": 8.5, "color": "#0F766E", "width": 0.6, "height": 0.5},
        {"name": "REDPANDA BROKER\nTopic: edge-iiot-raw", "x": 0.5, "y": 7.5, "color": "#E11D48", "width": 0.6, "height": 0.6},
        # Online Preprocessing
        {"name": "Online Preprocessing & Rescaling\n(Imputation + Robust Scaling)", "x": 0.5, "y": 6.4, "color": "#4338CA", "width": 0.65, "height": 0.6},
        {"name": "mRMR-JMI Feature Selection\n(61 Features -> 22 Features)", "x": 0.5, "y": 5.4, "color": "#0369A1", "width": 0.65, "height": 0.5},
        # Spatial Feature Extraction
        {"name": "Depthwise Separable 1D-CNN\n+ Ghost Module (Low FLOPs)", "x": 0.5, "y": 4.4, "color": "#7C3AED", "width": 0.65, "height": 0.5},
        {"name": "Adaptive Pooling + SE Attention\n(Dynamic Channel Calibration)", "x": 0.5, "y": 3.4, "color": "#9333EA", "width": 0.65, "height": 0.5},
        # Entropy Router
        {"name": "Predictive Entropy Router\nH(p) < tau", "x": 0.5, "y": 2.3, "color": "#D97706", "width": 0.5, "height": 0.5},
        # Fast Path
        {"name": "Fast Path Head\n(Early Exit)", "x": 0.18, "y": 1.1, "color": "#059669", "width": 0.28, "height": 0.6},
        # Deep Temporal Path
        {"name": "Deep Temporal Path:\nShared Bi-GRU + 4-Head Attention\n+ Low-Rank Head (Focal+Center Loss)", "x": 0.75, "y": 1.1, "color": "#2563EB", "width": 0.42, "height": 0.6},
        # Streaming Output & Dashboard
        {"name": "REDPANDA Topics:\nids-predictions & ids-alerts", "x": 0.5, "y": -0.1, "color": "#E11D48", "width": 0.6, "height": 0.5},
        {"name": "Streamlit Real-Time Dashboard\n(SOC Analytics & Telemetry)", "x": 0.5, "y": -1.0, "color": "#0D9488", "width": 0.65, "height": 0.5},
    ]

    fig = go.Figure()

    # Draw nodes
    for node in stages:
        w, h = node["width"], node["height"]
        x0, x1 = node["x"] - w / 2, node["x"] + w / 2
        y0, y1 = node["y"] - h / 2, node["y"] + h / 2

        fig.add_shape(
            type="rect",
            x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=node["color"],
            line=dict(color="#FFFFFF", width=1.5),
            layer="below"
        )
        fig.add_annotation(
            x=node["x"], y=node["y"],
            text=f"<b>{node['name']}</b>",
            showarrow=False,
            font=dict(color="#FFFFFF", size=11, family="Arial")
        )

    # Add connection arrows
    arrows = [
        (0.5, 9.2, 0.5, 8.75),
        (0.5, 8.25, 0.5, 7.8),
        (0.5, 7.2, 0.5, 6.7),
        (0.5, 6.1, 0.5, 5.65),
        (0.5, 5.15, 0.5, 4.65),
        (0.5, 4.15, 0.5, 3.65),
        (0.5, 3.15, 0.5, 2.55),
        # Routing arrows
        (0.4, 2.1, 0.22, 1.4),   # To Fast Path
        (0.6, 2.1, 0.72, 1.4),   # To Deep Path
        # Convergence arrows
        (0.22, 0.8, 0.42, 0.15),
        (0.72, 0.8, 0.58, 0.15),
        (0.5, -0.35, 0.5, -0.75)
    ]

    for x0, y0, x1, y1 in arrows:
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2, arrowsize=1.2, arrowwidth=2,
            arrowcolor="#94A3B8"
        )

    fig.update_layout(
        title="<b>Proposed System: Redpanda-Based Mutual Information–Driven Hybrid CNN–BiGRU Framework</b>",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.4, 10.2]),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        margin=dict(l=20, r=20, t=50, b=20),
        height=750
    )
    return fig
