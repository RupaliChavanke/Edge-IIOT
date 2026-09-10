"""
Attention and Explainability Visualization Module.
Visualizes Squeeze-and-Excitation channel weights and Multi-Head Temporal Attention matrices.
"""

from typing import List, Optional
import numpy as np
import plotly.graph_objects as go


def plot_se_channel_weights(weights: np.ndarray, top_k: int = 24) -> go.Figure:
    """Renders bar chart of Squeeze-and-Excitation channel recalibration weights."""
    weights = np.asarray(weights).flatten()
    indices = np.argsort(weights)[::-1][:top_k]
    top_w = weights[indices]
    channel_labels = [f"Ch_{idx}" for idx in indices]

    fig = go.Figure(go.Bar(
        x=channel_labels,
        y=top_w,
        marker=dict(color=top_w, colorscale="Plasma", showscale=True),
        text=[f"{w:.3f}" for w in top_w],
        textposition="outside"
    ))

    fig.update_layout(
        title=f"<b>Squeeze-and-Excitation (SE) Attention: Top {top_k} Active Channels</b>",
        xaxis_title="Channel Index",
        yaxis_title="Recalibration Scale Factor s_c in [0, 1]",
        yaxis=dict(range=[0, 1.15]),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font=dict(color="#F8FAFC"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=420
    )
    return fig


def plot_temporal_attention_heatmap(attn_matrix: np.ndarray) -> go.Figure:
    """Renders heatmap of cross-timestep attention weights from Multi-Head Self-Attention."""
    attn_matrix = np.asarray(attn_matrix)
    seq_len = attn_matrix.shape[0]
    timesteps = [f"t_{i+1}" for i in range(seq_len)]

    fig = go.Figure(data=go.Heatmap(
        z=attn_matrix,
        x=timesteps,
        y=timesteps,
        colorscale="Inferno",
        text=np.round(attn_matrix, 3),
        texttemplate="%{text}",
        textfont={"size": 10, "color": "white"}
    ))

    fig.update_layout(
        title="<b>Multi-Head Temporal Self-Attention Matrix (Sequence Inter-Packet Dependency)</b>",
        xaxis_title="Key / Value Timestep",
        yaxis_title="Query Timestep",
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font=dict(color="#F8FAFC"),
        margin=dict(l=60, r=40, t=50, b=60),
        height=480
    )
    return fig
