"""
Confusion Matrix Visualization Module.
Renders publication-grade interactive heatmaps for Raw, Recall-normalized, and Precision-normalized matrices.
"""

from typing import List, Optional
import numpy as np
import plotly.graph_objects as go


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: List[str],
    mode: str = "raw"
) -> go.Figure:
    """
    Renders an interactive heatmap for the confusion matrix.
    mode: 'raw' (counts) or 'normalized' (percentages 0-100%)
    """
    cm = np.asarray(cm)
    if mode == "normalized":
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_display = np.round((cm / row_sums) * 100.0, 1)
        z_text = [[f"{val:.1f}%" for val in row] for row in cm_display]
        colorscale = "Viridis"
        title = "<b>Normalized Confusion Matrix (Recall % per Class)</b>"
    else:
        cm_display = cm
        z_text = [[str(int(val)) for val in row] for row in cm_display]
        colorscale = "Blues"
        title = "<b>Raw Confusion Matrix (Sample Counts)</b>"

    fig = go.Figure(data=go.Heatmap(
        z=cm_display,
        x=class_names,
        y=class_names,
        text=z_text,
        texttemplate="%{text}",
        textfont={"size": 10, "color": "white"},
        colorscale=colorscale,
        showscale=True
    ))

    fig.update_layout(
        title=title,
        xaxis_title="<b>Predicted Attack Type</b>",
        yaxis_title="<b>Actual Ground Truth Class</b>",
        xaxis=dict(tickangle=45, color="#F8FAFC"),
        yaxis=dict(autorange="reversed", color="#F8FAFC"),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font=dict(color="#F8FAFC"),
        margin=dict(l=100, r=40, t=50, b=100),
        height=650
    )
    return fig
