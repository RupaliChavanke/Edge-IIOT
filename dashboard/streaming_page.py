"""
Page 4: Redpanda Streaming Infrastructure.
Real-time monitoring of Redpanda topics, partitions, consumer groups, lag, and producer/consumer controls.
"""

import time
import streamlit as st
import pandas as pd
from confluent_kafka import Consumer, KafkaError

from dashboard.components import render_redpanda_status_banner
from dashboard.state import (
    get_producer,
    get_consumer,
    start_producer,
    stop_producer,
    start_consumer,
    stop_consumer,
    inject_single_sample_to_redpanda
)
from streaming.health import RedpandaHealthChecker


def render_streaming_page():
    st.title("⚡ Redpanda Event-Streaming Backbone")
    st.caption("Industrial-grade distributed streaming platform: topic partitions, consumer offsets, and real-time ingestion pipelines.")

    checker = RedpandaHealthChecker()
    health = checker.check_health()
    render_redpanda_status_banner(health)

    prod = get_producer()
    cons = get_consumer()

    st.markdown("---")
    c_ctrl1, c_ctrl2 = st.columns([1, 1])

    with c_ctrl1:
        st.subheader("📤 Streaming Producer Controls")
        rate = st.slider("Target Ingestion Rate (events/sec):", min_value=5, max_value=500, value=30, step=5)
        
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            if st.button("▶️ START PRODUCER", type="primary", use_container_width=True):
                p = start_producer(rate=rate)
                st.success("Producer started producing to 'edge-iiot-raw'!")
                st.rerun()
        with col_p2:
            if st.button("⏸️ PAUSE", use_container_width=True):
                if prod:
                    prod.pause()
                    st.info("Producer paused.")
                    st.rerun()
        with col_p3:
            if st.button("⏹️ STOP PRODUCER", use_container_width=True):
                stop_producer()
                st.warning("Producer stopped.")
                st.rerun()

        if prod:
            st.json(prod.get_stats())

    with c_ctrl2:
        st.subheader("📥 Streaming Consumer & Inference Controls")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("▶️ START INFERENCE", type="primary", use_container_width=True):
                c = start_consumer()
                st.success("Inference consumer active on 'edge-iiot-raw'!")
                st.rerun()
        with col_c2:
            if st.button("⏹️ STOP INFERENCE", use_container_width=True):
                stop_consumer()
                st.warning("Consumer stopped.")
                st.rerun()

        if cons:
            st.json(cons.get_stats())

    st.markdown("---")
    st.subheader("🎯 Single-Sample Live Injection Through Redpanda")
    st.info("💡 Notice: The sample is published to `edge-iiot-raw` in Redpanda. The background consumer consumes it, processes it through mRMR-JMI, runs neural inference, and publishes the prediction to `ids-predictions`.")

    with st.expander("Inject Test Network Packet into Redpanda"):
        attack_options = [
            "Normal", "DDoS_UDP", "DDoS_ICMP", "SQL_injection", "Ransomware", "Port_Scanning", "MITM"
        ]
        chosen_sim = st.selectbox("Simulated Traffic Profile:", attack_options)

        if st.button("🚀 Publish Event to Redpanda 'edge-iiot-raw'"):
            # Construct realistic features
            sample_feats = {
                "tcp.dstport": 80.0 if "HTTP" in chosen_sim or "SQL" in chosen_sim else (53.0 if "DNS" in chosen_sim else 443.0),
                "tcp.srcport": 54321.0,
                "tcp.ack": 1024.0,
                "tcp.seq": 5000.0,
                "tcp.flags": 2.0 if "SYN" in chosen_sim else 16.0,
                "tcp.len": 1400.0 if "DDoS" in chosen_sim else 64.0,
                "http.request.method": "POST" if "SQL" in chosen_sim else "GET",
                "udp.port": 53.0 if "UDP" in chosen_sim else 0.0,
                "icmp.checksum": 1234.0 if "ICMP" in chosen_sim else 0.0
            }
            ok = inject_single_sample_to_redpanda(sample_feats, ground_truth=chosen_sim)
            if ok:
                st.success(f"Successfully dispatched {chosen_sim} event to Redpanda topic `edge-iiot-raw`! Watch consumer output.")
            else:
                st.error("Failed to publish to Redpanda broker.")

    # Live Event Feed from ids-predictions
    st.markdown("---")
    st.subheader("📡 Live Redpanda Ingestion & Prediction Stream ('ids-predictions')")

    if st.button("🔄 Poll Recent Messages from Topic"):
        # Read last 5 messages from ids-predictions
        try:
            consumer = Consumer({
                "bootstrap.servers": health.get("broker", "localhost:19092"),
                "group.id": "streamlit-ui-monitor-group",
                "auto.offset.reset": "latest",
                "enable.auto.commit": False
            })
            consumer.subscribe(["ids-predictions"])
            events = []
            for _ in range(5):
                msg = consumer.poll(timeout=0.3)
                if msg and not msg.error():
                    import json
                    events.append(json.loads(msg.value().decode("utf-8")))
            consumer.close()

            if events:
                df_ev = pd.DataFrame(events)
                cols_to_show = ["timestamp", "event_id", "predicted_attack", "confidence", "entropy", "path_selected", "risk_level"]
                existing_cols = [c for c in cols_to_show if c in df_ev.columns]
                st.dataframe(df_ev[existing_cols], use_container_width=True)
            else:
                st.info("No new events currently in partition buffer. Launch producer & inference to observe live stream.")
        except Exception as e:
            st.warning(f"Unable to read messages: {e}")
