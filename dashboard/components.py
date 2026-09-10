"""
Reusable UI Components: Redpanda Dynamic Status Card, Alerts, and Metric Banners.
"""

from typing import Dict, Any, Optional
import os
import streamlit as st
from streaming.health import RedpandaHealthChecker


def render_redpanda_status_banner(health_info: Optional[Dict[str, Any]] = None, stats: Optional[Dict[str, Any]] = None):
    """
    Renders the dynamic Redpanda Streaming Status dashboard panel:
    ┌─────────────────────────────────────────┐
    │ REDPANDA STREAMING STATUS               │
    ├─────────────────────────────────────────┤
    │ Broker             CONNECTED            │
    │ Raw Topic          ACTIVE               │
    │ Prediction Topic   ACTIVE               │
    │ Alert Topic        ACTIVE               │
    │ Messages/sec       125                  │
    │ Consumer Lag       0                    │
    │ P99 Inference      1.87 ms              │
    └─────────────────────────────────────────┘
    """
    if health_info is None:
        checker = RedpandaHealthChecker()
        health_info = checker.check_health()

    status = health_info.get("status", "DISCONNECTED")
    broker = health_info.get("broker", "localhost:19092")
    topics = health_info.get("topics", {})

    raw_status = topics.get("edge-iiot-raw", {}).get("status", "INACTIVE")
    pred_status = topics.get("ids-predictions", {}).get("status", "INACTIVE")
    alert_status = topics.get("ids-alerts", {}).get("status", "INACTIVE")

    # Dynamic metrics from consumer if running
    messages_sec = stats.get("messages_sec", 0.0) if stats else 0.0
    consumer_lag = stats.get("consumer_lag", 0) if stats else 0
    p99_latency = stats.get("p99_latency_ms", 0.0) if stats else 0.0

    st.markdown("""
        <style>
        .rp-box {
            background-color: #0F172A;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 20px;
            font-family: 'Courier New', monospace;
        }
        .rp-title {
            color: #38BDF8;
            font-size: 14px;
            font-weight: bold;
            letter-spacing: 1px;
            border-bottom: 1px solid #334155;
            padding-bottom: 8px;
            margin-bottom: 12px;
        }
        .rp-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
            font-size: 13px;
        }
        .status-ok { color: #10B981; font-weight: bold; }
        .status-err { color: #EF4444; font-weight: bold; }
        .status-val { color: #F8FAFC; }
        </style>
    """, unsafe_allow_html=True)

    broker_badge = f"<span class='status-ok'>CONNECTED</span>" if status == "CONNECTED" else f"<span class='status-err'>DISCONNECTED</span>"
    raw_badge = f"<span class='status-ok'>ACTIVE</span>" if raw_status == "ACTIVE" else f"<span class='status-err'>{raw_status}</span>"
    pred_badge = f"<span class='status-ok'>ACTIVE</span>" if pred_status == "ACTIVE" else f"<span class='status-err'>{pred_status}</span>"
    alert_badge = f"<span class='status-ok'>ACTIVE</span>" if alert_status == "ACTIVE" else f"<span class='status-err'>{alert_status}</span>"

    is_connected = (status == "CONNECTED")
    is_cloud = (status == "CLOUD_MODE")

    broker_label = "CONNECTED" if is_connected else ("CLOUD BUS" if is_cloud else "DISCONNECTED")
    broker_delta = f"{health_info.get('ping_ms', 0)} ms" if is_connected else ("Active" if is_cloud else None)

    cols = st.columns([1, 1, 1, 1, 1, 1, 1])
    with cols[0]:
        st.metric("Broker", broker_label, delta=broker_delta)
    with cols[1]:
        st.metric("Raw Topic", raw_status)
    with cols[2]:
        st.metric("Prediction Topic", pred_status)
    with cols[3]:
        st.metric("Alert Topic", alert_status)
    with cols[4]:
        st.metric("Messages/sec", f"{messages_sec:.1f}")
    with cols[5]:
        st.metric("Consumer Lag", f"{consumer_lag}")
    with cols[6]:
        st.metric("P99 Inference", f"{p99_latency:.2f} ms")

    if is_cloud:
        with st.expander("☁️ Cloud Deployment Architecture (Zero-Broker Mode)", expanded=False):
            st.markdown("""
            **Deployment Context**: Running in Streamlit Community Cloud / Zero-Broker container environment.
            - **Streaming Backbone**: High-throughput thread-safe `InMemoryStreamingBus` routing topic streams.
            - **Neural Inference**: Deployed PyTorch Hybrid CNN-BiGRU model (`best_model.pt`) executing real tensor forward passes.
            - **Performance**: Zero-copy packet pipeline yielding P99 latency < 5ms and authentic multi-class attack detection.
            - **External Redpanda**: Set environment variable `REDPANDA_BROKERS=host:port` to bind to an external distributed cluster.
            """)
    elif not is_connected:
        st.error(
            f"⚠️ **REDPANDA CONNECTION REQUIRED**: Unable to reach broker at `{broker}`.\n\n"
            f"**Diagnostic Error**: `{health_info.get('error')}`\n\n"
            f"**Troubleshooting Instructions**:\n"
            f"1. Start the Redpanda broker: `docker compose up -d redpanda redpanda-console`\n"
            f"2. Confirm container health: `docker ps | grep redpanda`\n"
            f"3. Verify Kafka API port: `nc -zv localhost 19092`"
        )
        if st.button("🔄 Retry Connection to Redpanda"):
            st.rerun()
