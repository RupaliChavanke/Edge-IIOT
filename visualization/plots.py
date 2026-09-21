"""
Scientific Visualization Plots Module.
Generates publication-ready Plotly figures for metrics, trade-offs, distributions, and benchmarks.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def plot_multiclass_roc(roc_data: Dict[str, Any]) -> go.Figure:
    """Renders interactive multi-class ROC curves with AUC annotations."""
    fig = go.Figure()

    if not roc_data or "fpr" not in roc_data:
        fig.add_annotation(text="No ROC data available", showarrow=False)
        return fig

    # Micro & Macro
    if "macro" in roc_data["auc"]:
        fig.add_trace(go.Scatter(
            x=roc_data["fpr"]["macro"],
            y=roc_data["tpr"]["macro"],
            name=f"Macro-average (AUC = {roc_data['auc']['macro']:.3f})",
            line=dict(color="#F43F5E", width=3, dash="dash")
        ))

    if "micro" in roc_data["auc"]:
        fig.add_trace(go.Scatter(
            x=roc_data["fpr"]["micro"],
            y=roc_data["tpr"]["micro"],
            name=f"Micro-average (AUC = {roc_data['auc']['micro']:.3f})",
            line=dict(color="#38BDF8", width=2.5, dash="dot")
        ))

    # Top individual classes (show up to 8 for clarity)
    class_names = roc_data.get("class_names", [])
    colors = px.colors.qualitative.Dark24
    for i, cls in enumerate(class_names[:8]):
        if cls in roc_data["auc"]:
            fig.add_trace(go.Scatter(
                x=roc_data["fpr"][cls],
                y=roc_data["tpr"][cls],
                name=f"{cls} (AUC = {roc_data['auc'][cls]:.3f})",
                line=dict(color=colors[i % len(colors)], width=1.5)
            ))

    # Diagonal random guess line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name="Random Classifier",
        line=dict(color="#64748B", dash="dash", width=1.2),
        showlegend=False
    ))

    fig.update_layout(
        title="<b>Receiver Operating Characteristic (ROC) Curves</b>",
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR / Recall)",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font=dict(color="#F8FAFC"),
        legend=dict(x=0.6, y=0.1, bgcolor="rgba(15, 23, 42, 0.7)"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=520
    )
    return fig


def plot_training_curves(history: List[Dict[str, Any]]) -> go.Figure:
    """Plots training loss, focal loss, and validation macro-F1 over epochs."""
    df = pd.DataFrame(history)
    fig = go.Figure()

    if df.empty:
        fig.add_annotation(text="No training history available", showarrow=False)
        return fig

    fig.add_trace(go.Scatter(
        x=df["epoch"], y=df["train_loss"],
        mode="lines+markers", name="Total Train Loss",
        line=dict(color="#F59E0B", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=df["epoch"], y=df["focal_loss"],
        mode="lines", name="Focal Loss Component",
        line=dict(color="#EC4899", width=1.8, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=df["epoch"], y=df["val_accuracy"] * 100,
        mode="lines+markers", name="Val Accuracy (%)",
        line=dict(color="#10B981", width=2.5), yaxis="y2"
    ))
    fig.add_trace(go.Scatter(
        x=df["epoch"], y=df["val_f1"] * 100,
        mode="lines+markers", name="Val Macro-F1 (%)",
        line=dict(color="#6366F1", width=2.5), yaxis="y2"
    ))

    fig.update_layout(
        title="<b>Training & Validation Convergence (Compound Loss vs F1)</b>",
        xaxis_title="Epoch",
        yaxis=dict(title="Loss Value", color="#F59E0B"),
        yaxis2=dict(title="Accuracy / Macro-F1 (%)", color="#10B981", overlaying="y", side="right", range=[0, 105]),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font=dict(color="#F8FAFC"),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(15, 23, 42, 0.7)"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=480
    )
    return fig


def plot_mrmr_feature_importance(ranking_df: pd.DataFrame, top_k: int = 22) -> go.Figure:
    """Plots top selected features by mRMR-JMI score with MI and Redundancy components."""
    sub = ranking_df.head(top_k).copy().iloc[::-1]  # Invert so rank 1 is on top
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=sub["Feature"],
        x=sub["MI"],
        name="Relevance (Mutual Info I(X; Y))",
        orientation="h",
        marker=dict(color="#3B82F6")
    ))
    fig.add_trace(go.Bar(
        y=sub["Feature"],
        x=-sub["Redundancy"],
        name="Redundancy Penalty",
        orientation="h",
        marker=dict(color="#EF4444")
    ))
    fig.add_trace(go.Scatter(
        y=sub["Feature"],
        x=sub["JMI"],
        name="Net JMI Score",
        mode="markers+lines",
        marker=dict(color="#10B981", size=10, symbol="diamond"),
        line=dict(color="#10B981", width=2)
    ))

    fig.update_layout(
        title=f"<b>mRMR-JMI Feature Ranking: Top {top_k} Selected Features</b>",
        xaxis_title="Score / Mutual Information (nats)",
        yaxis_title="Feature Name",
        barmode="relative",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font=dict(color="#F8FAFC"),
        legend=dict(x=0.6, y=0.1, bgcolor="rgba(15, 23, 42, 0.7)"),
        margin=dict(l=150, r=40, t=50, b=40),
        height=620
    )
    return fig


def plot_entropy_distribution(entropy_vals: np.ndarray, labels: np.ndarray, threshold: float = 0.35) -> go.Figure:
    """Plots predictive entropy histograms separating Normal vs Attack events."""
    fig = go.Figure()

    normal_mask = (labels == 0)
    normal_entropy = entropy_vals[normal_mask]
    attack_entropy = entropy_vals[~normal_mask]

    fig.add_trace(go.Histogram(
        x=normal_entropy,
        name="Normal Traffic (High Certainty)",
        marker=dict(color="#10B981", opacity=0.7),
        nbinsx=30
    ))
    fig.add_trace(go.Histogram(
        x=attack_entropy,
        name="Attack Traffic (Varied Uncertainty)",
        marker=dict(color="#EF4444", opacity=0.7),
        nbinsx=30
    ))

    fig.add_vline(
        x=threshold,
        line_width=3, line_dash="dash", line_color="#F59E0B",
        annotation_text=f"Exit Threshold tau={threshold}",
        annotation_position="top left"
    )

    fig.update_layout(
        title="<b>Predictive Entropy Distribution: Normal vs Attack Traffic</b>",
        xaxis_title="Normalized Predictive Entropy H(p) in [0, 1]",
        yaxis_title="Event Count",
        barmode="overlay",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font=dict(color="#F8FAFC"),
        legend=dict(x=0.6, y=0.9, bgcolor="rgba(15, 23, 42, 0.7)"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=450
    )
    return fig


def plot_latency_waterfall(latency_profile: Dict[str, Any]) -> go.Figure:
    """Waterfall chart showing latency contribution of each pipeline component."""
    stages = [
        "Ingestion",
        "Preprocessing",
        "mRMR_Selection",
        "CNN_Ghost_SE",
        "BiGRU",
        "Attention_LowRank",
        "Serialization_Publish"
    ]
    labels = ["Ingestion", "Clean/Scale", "mRMR-JMI", "CNN+Ghost", "Bi-GRU", "MHA+LowRank", "Publish"]
    values = [latency_profile[s]["mean_ms"] for s in stages]
    measures = ["relative"] * len(stages)

    fig = go.Figure(go.Waterfall(
        name="Streaming Latency",
        orientation="v",
        measure=measures,
        x=labels,
        textposition="outside",
        text=[f"{v:.2f}ms" for v in values],
        y=values,
        connector=dict(line=dict(color="#64748B")),
        decreasing=dict(marker=dict(color="#10B981")),
        increasing=dict(marker=dict(color="#38BDF8")),
        totals=dict(marker=dict(color="#F43F5E"))
    ))

    fig.update_layout(
        title="<b>End-to-End Streaming Latency Waterfall Breakdown</b>",
        yaxis_title="Stage Latency (milliseconds)",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font=dict(color="#F8FAFC"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=480
    )
    return fig


def plot_model_radar_comparison(benchmark_df: pd.DataFrame) -> go.Figure:
    """Radar chart comparing Proposed Model against top baselines."""
    fig = go.Figure()
    categories = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC", "Speed_Rank", "Efficiency_Rank"]

    models_to_plot = ["PROPOSED HYBRID MODEL", "XGBoost", "Random Forest", "CNN-BiGRU-Attention", "Original Baseline Model"]
    colors = ["#10B981", "#38BDF8", "#F59E0B", "#8B5CF6", "#EF4444"]

    for i, m_name in enumerate(models_to_plot):
        row = benchmark_df[benchmark_df["Model"] == m_name]
        if not row.empty:
            r = row.iloc[0]
            # Speed rank: inverse latency normalized to 0-1
            speed_score = max(0.2, min(1.0, 5.0 / max(0.5, r.get("P95_Latency_ms", 5.0))))
            param_score = max(0.2, min(1.0, 300000.0 / max(10000, r.get("Parameters", 200000))))

            values = [
                r.get("Accuracy", 0.9),
                r.get("Precision_Macro", 0.9),
                r.get("Recall_Macro", 0.9),
                r.get("F1_Macro", 0.9),
                r.get("ROC_AUC", 0.95),
                speed_score,
                param_score
            ]
            # Close the radar polygon
            values.append(values[0])

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                name=m_name,
                line=dict(color=colors[i % len(colors)], width=2)
            ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.05], color="#94A3B8"),
            bgcolor="#1E293B"
        ),
        title="<b>Multi-Dimensional Model Efficiency Radar Chart</b>",
        paper_bgcolor="#0F172A",
        font=dict(color="#F8FAFC"),
        legend=dict(x=0.8, y=0.95, bgcolor="rgba(15, 23, 42, 0.7)"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=520
    )
    return fig
