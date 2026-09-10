"""
Page 5: Stream Preprocessing & Rescaling.
Details online imputation, categorical encoding, robust scaling, and data hygiene.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

from preprocessing.loader import EdgeIIoTDataLoader


def render_preprocessing_page():
    st.title("⚙️ Online Stream Preprocessing & Rescaling")
    st.caption("Scientific feature engineering pipeline fitted strictly on training split to prevent data leakage.")

    from models.model_manager import ModelManager
    manager = ModelManager.get_instance()
    loader = EdgeIIoTDataLoader()
    
    pipeline_exists = os.path.exists("checkpoints/cleaner.pkl") and os.path.exists("checkpoints/scaler.pkl")
    if pipeline_exists:
        try:
            loader.load_pipeline("checkpoints")
            st.success("✅ Preprocessing pipeline loaded from `checkpoints/` (fitted on training data).")
        except Exception as e:
            st.warning(f"Could not load checkpoints: {e}")
    elif os.path.exists("artifacts/preprocessor.pkl"):
        manager.load_artifacts()
        st.success("✅ Preprocessing pipeline loaded from `artifacts/` (fitted on offline training split).")
    else:
        st.info("ℹ️ Pipeline not yet fitted. Run offline training (`python train.py`) to generate artifacts.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🧹 Cleaning & Imputation",
        "🔤 Categorical Encoding",
        "⚖️ Robust Scaling & Outliers",
        "🧪 Live Single-Event Transform"
    ])

    with tab1:
        st.subheader("Data Sanitization & Missing Value Strategy")
        st.markdown(r"""
        - **Metadata Drop**: Features containing explicit source/destination IP addresses (`ip.src_host`, `ip.dst_host`) and raw payloads (`tcp.payload`) are purged to prevent memorization shortcuts.
        - **Imputation**: Numeric columns undergo median imputation derived strictly from the training split. Categorical columns undergo mode imputation.
        - **Infinite Values**: Network tools frequently produce `inf` or `NaN` in latency/rate fields; these are converted to finite limits and median imputed.
        """)
        if loader.is_fitted:
            st.write(f"**Identified Numeric Attributes**: {len(loader.cleaner.fitted_numeric_columns)}")
            st.write(f"**Identified Categorical Attributes**: {len(loader.cleaner.fitted_categorical_columns)}")
            with st.expander("View Imputation Values per Feature"):
                st.json(loader.cleaner.impute_values)

    with tab2:
        st.subheader("Categorical Feature Encoding & Target Discovery")
        st.markdown(r"""
        - Target labels (`Attack_type`) are dynamically discovered from the dataset without hardcoding classes.
        - Network protocol fields (`http.request.method`, `mqtt.protoname`, etc.) are mapped to ordinal integer representations with an `unknown` fallback index for zero-day protocol variants.
        """)
        if loader.is_fitted:
            st.write("**Discovered Multiclass Attack Targets (15 Classes)**:")
            st.write(loader.encoder.classes_)
            if loader.encoder.class_weights is not None:
                st.write("**Calculated Focal Loss Class Weights**:")
                df_w = pd.DataFrame({
                    "Class": loader.encoder.classes_,
                    "Balanced_Weight": np.round(loader.encoder.class_weights, 3)
                })
                st.dataframe(df_w, use_container_width=True)

    with tab3:
        st.subheader("Robust Scaling & Percentile Outlier Clamping")
        st.markdown(r"""
        - Utilizes **RobustScaler** with interquartile range (IQR 5th to 95th percentile) rather than standard Z-score:
          $$x_{scaled} = \frac{x - \text{median}(x)}{\text{IQR}(x)}$$
        - This prevents large volume volumetric DDoS spikes from corrupting the scaling of low-rate attacks like MITM or Port Scanning.
        - Extreme outliers beyond the 0.1th and 99.9th percentiles are clamped.
        """)

    with tab4:
        st.subheader("Interactive Online Event Preprocessing Simulator")
        st.write("Simulate raw incoming JSON dictionary and observe transformed 22-dimensional feature vector.")

        sample_event = {
            "tcp.dstport": 80.0,
            "tcp.srcport": 49152.0,
            "tcp.ack": 1205.0,
            "tcp.seq": 3410.0,
            "tcp.flags": 2.0,
            "tcp.len": 128.0,
            "http.request.method": "GET"
        }
        st.json(sample_event)

        if st.button("Transform Sample Event"):
            if loader.is_fitted:
                out_vec = loader.process_single_streaming_event(sample_event)
                st.write(f"**Transformed Feature Vector (Shape: {out_vec.shape})**:")
                st.dataframe(pd.DataFrame([out_vec], columns=loader.selector.selected_features_))
            else:
                st.warning("Please fit the pipeline first.")
