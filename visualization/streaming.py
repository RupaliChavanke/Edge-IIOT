"""
Real-Time SOC Streaming Visualizations and Alert Widgets.
"""

from typing import Dict, List, Any
import numpy as np
import plotly.graph_objects as go


def plot_soc_threat_gauge(threat_score: float) -> go.Figure:
    """Renders real-time high-tech SOC threat level gauge."""
    threat_pct = min(100.0, max(0.0, threat_score * 100.0))

    if threat_pct > 70:
        bar_color = "#EF4444"
        status_text = "🚨 CRITICAL: ACTIVE INTRUSION"
        status_color = "#EF4444"
    elif threat_pct > 35:
        bar_color = "#F59E0B"
        status_text = "⚠️ ELEVATED: ANOMALOUS FLOWS"
        status_color = "#F59E0B"
    else:
        bar_color = "#10B981"
        status_text = "🛡️ DEFENSE CONDITION: STABLE"
        status_color = "#10B981"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=threat_pct,
        title={"text": f"<b>CURRENT SOC THREAT LEVEL</b><br><span style='font-size:12px; color:{status_color};'>{status_text}</span>", "font": {"size": 14, "color": "#F8FAFC"}},
        number={"suffix": "%", "font": {"size": 32, "color": "#FFFFFF", "family": "Inter, sans-serif"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1.5, "tickcolor": "#64748B", "nticks": 6},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "rgba(30, 41, 59, 0.5)",
            "borderwidth": 1,
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, 35], "color": "rgba(16, 185, 129, 0.15)"},
                {"range": [35, 70], "color": "rgba(245, 158, 11, 0.15)"},
                {"range": [70, 100], "color": "rgba(239, 68, 68, 0.20)"}
            ],
            "threshold": {
                "line": {"color": "#EF4444", "width": 3},
                "thickness": 0.75,
                "value": 85.0
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font=dict(color="#F8FAFC", family="Inter, sans-serif"),
        margin=dict(l=40, r=40, t=60, b=20),
        height=260
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
