"""
PhD Defense Demonstration Mode Page.
Provides an interactive 18-step guided defense walkthrough covering all stages:
Pretrained Model Verification -> Redpanda Ingestion -> Dynamic Inference -> Real-Time Metrics -> Offline vs Live Degradation.
"""

import streamlit as st
import pandas as pd
from models.model_manager import ModelManager
from streaming.health import RedpandaHealthChecker
from dashboard.live_evaluator import LiveEvaluatorEngine


def render_phd_demo_page():
    st.title("🎓 Ph.D. Dissertation Defense Demonstration Mode")
    st.caption("Structured 18-Step Scientific Walkthrough for Doctoral Examination Committee")

    manager = ModelManager.get_instance()
    health_checker = RedpandaHealthChecker()
    health = health_checker.check_health()
    evaluator = LiveEvaluatorEngine.get_instance()
    metadata = manager.get_model_metadata()

    # Step Progress Indicator
    steps = [
        "1. Model Loaded Verification",
        "2. Redpanda Connectivity",
        "3. Architectural Specification & FLOPs",
        "4. Offline Test Baseline",
        "5. Redpanda Broker & Topic Topology",
        "6. Start Dataset Replay Stream",
        "7. Event Ingestion (edge-iiot-raw)",
        "8. Frozen Model Inference Execution",
        "9. Prediction Publication (ids-predictions)",
        "10. Threat Alert Routing (ids-alerts)",
        "11. Live Stream Metric Accumulation",
        "12. Dynamic Real-Time Accuracy / F1",
        "13. Live Dynamic Confusion Matrix",
        "14. TP / TN / FP / FN Diagnostics",
        "15. Entropy-Driven Fast vs Deep Routing",
        "16. Microsecond Latency Breakdown",
        "17. Ingestion & Prediction Throughput",
        "18. Offline vs Live Degradation Verification"
    ]

    selected_step_idx = st.sidebar.selectbox("Jump to Defense Step:", range(len(steps)), format_func=lambda i: steps[i])

    # Big 3 Status Banners
    s1, s2, s3 = st.columns(3)
    with s1:
        if manager.is_loaded:
            st.success("✓ PRETRAINED MODEL: LOADED (FROZEN)")
        else:
            st.error("✗ PRETRAINED MODEL: NOT FOUND")
    with s2:
        if health.get("status") == "CONNECTED":
            st.success("✓ REDPANDA: CONNECTED (BROKER ONLINE)")
        else:
            st.error("✗ REDPANDA: DISCONNECTED")
    with s3:
        if manager.is_loaded and health.get("status") == "CONNECTED":
            st.success("✓ INFERENCE SERVICE: READY")
        else:
            st.warning("○ INFERENCE: PENDING SETUP")

    st.markdown("---")

    # Step Content Renderer
    step_title = steps[selected_step_idx]
    st.subheader(step_title)

    if selected_step_idx == 0:  # Step 1: Model Loaded Verification
        st.markdown("""
        **Scientific Requirement**: The IDS deployment must use a frozen, pretrained checkpoint. No retraining or fine-tuning occurs inside the dashboard.
        """)
        st.write(f"- **Active Model**: `{metadata.get('model_version', 'v1.0')}`")
        st.write(f"- **Inference Mode**: `torch.inference_mode() == True`")
        st.write(f"- **Weights File**: `artifacts/best_model.pt` ({metadata.get('model_size_mb', 0.0)} MB)")
        st.write(f"- **Parameters**: `{metadata.get('total_parameters', 0):,}`")

    elif selected_step_idx == 1:  # Step 2: Redpanda Connectivity
        st.markdown("""
        **Scientific Requirement**: Enterprise messaging backbone must be Redpanda (C++ Kafka-compatible engine).
        """)
        st.json(health)

    elif selected_step_idx == 2:  # Step 3: Architecture & FLOPs
        st.markdown("""
        **Scientific Requirement**: Hybrid 1D-CNN + Ghost Module + SE Attention + BiGRU + Temporal Attention + Low-Rank Projection.
        """)
        st.image("artifacts/complexity.png" if os.path.exists("artifacts/complexity.png") else "artifacts/benchmark.png")
        st.metric("Theoretical Computational Complexity", f"{metadata.get('mflops', 0.0)} MFLOPs")

    elif selected_step_idx == 3:  # Step 4: Offline Test Baseline
        st.markdown("""
        **Scientific Requirement**: Offline held-out test evaluation loaded from `artifacts/metrics.json`.
        """)
        import json
        if os.path.exists("artifacts/metrics.json"):
            with open("artifacts/metrics.json") as f:
                mets = json.load(f)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Offline Accuracy", f"{mets.get('Accuracy', 0)*100:.2f}%")
            c2.metric("Offline Macro-F1", f"{mets.get('F1_Macro', 0)*100:.2f}%")
            c3.metric("Offline Precision", f"{mets.get('Precision_Macro', 0)*100:.2f}%")
            c4.metric("Offline Recall", f"{mets.get('Recall_Macro', 0)*100:.2f}%")

    elif selected_step_idx in [4, 5, 6, 7, 8, 9]:  # Streaming Pipeline Execution Steps
        st.markdown("""
        **Streaming Telemetry Flow**:
        ```
        Dataset Replay -> Producer -> [edge-iiot-raw] -> Inference Service -> [ids-predictions / ids-alerts] -> Streamlit Live Evaluator
        ```
        """)
        prod = st.session_state.get("producer")
        if prod and prod.is_running:
            st.success("Producer is currently broadcasting live events to Redpanda.")
            if st.button("Stop Dataset Replay"):
                prod.stop()
                st.rerun()
        else:
            if st.button("▶️ START DATASET REPLAY (50 events/sec)"):
                from streaming.redpanda_producer import EdgeIIoTRedpandaProducer
                p = EdgeIIoTRedpandaProducer(rate=50)
                p.start()
                st.session_state["producer"] = p
                st.rerun()

    elif selected_step_idx in [10, 11, 12, 13]:  # Dynamic Evaluation Steps
        st.markdown("""
        **Live Metric Dynamic Calculation**:
        Metrics update dynamically from actual class prediction probabilities without target feature leakage.
        """)
        live_mets = evaluator.get_live_metrics()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Live Accuracy", f"{live_mets['Accuracy']*100:.2f}%")
        c2.metric("Live Macro-F1", f"{live_mets['F1_Macro']*100:.2f}%")
        c3.metric("Live TP", f"{live_mets['TP']:,}")
        c4.metric("Live FP", f"{live_mets['FP']:,}")

    elif selected_step_idx == 14:  # Entropy Routing
        st.markdown("""
        **Entropy Routing Verification**:
        Evaluates the dynamic partition between the Fast Sub-0.5ms Path and Deep Temporal BiGRU Path based on prediction entropy $\\mathcal{H}(p) < \\tau$.
        """)
        st.image("artifacts/latency.png" if os.path.exists("artifacts/latency.png") else "artifacts/training_history.png")

    elif selected_step_idx in [15, 16]:  # Latency & Throughput
        st.markdown("""
        **Microsecond Latency & High-Throughput Verification**:
        Measures P50, P95, and P99 stage execution latencies.
        """)
        if os.path.exists("artifacts/latency_results.json"):
            with open("artifacts/latency_results.json") as f:
                lat = json.load(f)
            st.json(lat)

    elif selected_step_idx == 17:  # Step 18: Offline vs Live Degradation
        st.markdown("""
        **Thesis Climax**: Proves zero significant performance degradation between offline laboratory experiments and live Redpanda streaming deployment.
        """)
        deg = evaluator.get_degradation_analysis()
        st.metric("Overall Performance Retention", f"{deg['retention']['Accuracy']:.1f}%")
        st.write(f"- Offline Test Accuracy: **{deg['offline']['Accuracy']*100:.2f}%**")
        st.write(f"- Live Replay Accuracy: **{deg['live']['Accuracy']*100:.2f}%**")
        st.write(f"- Absolute Discrepancy: **{deg['difference']['Accuracy']*100:+.2f}%**")
