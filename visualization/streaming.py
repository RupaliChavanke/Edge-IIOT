"""
Real-Time SOC Streaming Visualizations and Alert Widgets.
"""

from typing import Dict, List, Any
import numpy as np
import plotly.graph_objects as go


def plot_soc_threat_gauge(threat_score: float) -> go.Figure:
    """Renders real-time SOC threat level gauge."""
    threat_pct = min(100.0, max(0.0, threat_score * 100.0))

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=threat_pct,
        title={"text": "<b>CURRENT SOC THREAT LEVEL</b>", "font": {"size": 16, "color": "#F8FAFC"}},
        number={"suffix": "%", "font": {"size": 28, "color": "#F8FAFC"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
            "bar": {"color": "#EF4444" if threat_pct > 70 else ("#F59E0B" if threat_pct > 40 else "#10B981")},
            "bgcolor": "#1E293B",
            "borderwidth": 2,
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, 40], "color": "rgba(16, 185, 129, 0.2)"},
                {"range": [40, 70], "color": "rgba(245, 158, 11, 0.2)"},
                {"range": [70, 100], "color": "rgba(239, 68, 68, 0.2)"}
            ]
        }
    ))

    fig.update_layout(
        paper_bgcolor="#0F172A",
        font=dict(color="#F8FAFC"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=280
    )
    return fig


def plot_live_attack_distribution(attack_counts: Dict[str, int]) -> go.Figure:
    """Donut chart showing real-time proportions of detected attack categories."""
    labels = list(attack_counts.keys())
    values = list(attack_counts.values())

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        textinfo="label+percent",
        marker=dict(colors=["#10B981", "#EF4444", "#F59E0B", "#3B82F6", "#8B5CF6", "#EC4899", "#14B8A6"])
    )])

    fig.update_layout(
        title="<b>Live Stream Traffic Composition</b>",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font=dict(color="#F8FAFC"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        showlegend=False
    )
    return fig
