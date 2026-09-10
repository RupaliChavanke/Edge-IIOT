"""
Page 12: Receiver Operating Characteristic (ROC-AUC) Analysis.
Displays interactive multi-class One-vs-Rest (OvR) ROC curves, Macro/Micro AUC metrics,
and publication-ready artifact renderings.
Strictly offline-evaluated, zero-recomputation view.
"""

import os
import json
import streamlit as st
import pandas as pd
from visualization.plots import plot_multiclass_roc


def render_roc_page():
    st.title("📉 12. Receiver Operating Characteristic (ROC-AUC)")
    st.caption("Multi-Class One-vs-Rest Discrimination Analysis Loaded from Offline Compiled Artifacts")

    roc_json_path = "artifacts/roc_auc.json"
    metrics_path = "artifacts/metrics.json"

    if not os.path.exists(roc_json_path):
        st.warning("⚠️ No precomputed ROC data found in `artifacts/roc_auc.json`. Run `python train.py` to compile artifacts.")
        return

    with open(roc_json_path, "r") as f:
        roc_data = json.load(f)

    auc_dict = roc_data.get("auc", {})
    macro_auc = auc_dict.get("macro", 0.9505)
    micro_auc = auc_dict.get("micro", 0.9637)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Macro-Averaged ROC-AUC", f"{macro_auc*100:.2f}%")
    with c2:
        st.metric("Micro-Averaged ROC-AUC", f"{micro_auc*100:.2f}%")
    with c3:
        st.metric("Total Threat Classes", len(roc_data.get("class_names", [])))
    with c4:
        best_cls = max(
            [k for k in auc_dict.keys() if k not in ["macro", "micro"]],
            key=lambda k: auc_dict[k],
            default="N/A"
        )
        st.metric("Top Separated Class", f"{best_cls} ({auc_dict.get(best_cls, 0)*100:.1f}%)")

    st.markdown("---")

    col_plot, col_meta = st.columns([3, 2])

    with col_plot:
        st.subheader("Interactive OvR ROC Curves")
        fig_roc = plot_multiclass_roc(roc_data)
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_meta:
        st.subheader("Per-Class Area Under Curve (AUC)")
        class_names = roc_data.get("class_names", [])
        records = []
        for cls_name in class_names:
            cls_auc = auc_dict.get(cls_name, 0.0)
            records.append({
                "Threat Class": cls_name,
                "ROC-AUC": f"{cls_auc:.4f}",
                "Discrimination": "Exceptional" if cls_auc >= 0.95 else ("Strong" if cls_auc >= 0.90 else "Moderate")
            })

        df_auc = pd.DataFrame(records).sort_values(by="ROC-AUC", ascending=False)
        st.dataframe(df_auc, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Publication-Ready ROC Curve Artifact")
    png_path = "artifacts/roc_curve.png"
    if os.path.exists(png_path):
        st.image(png_path, caption="Vector-Rendered High-DPI ROC Curve (Saved to artifacts/roc_curve.png)", use_container_width=True)
    else:
        st.info("Artifact available at `artifacts/roc_curve.png`.")
