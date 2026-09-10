"""
Page 18: Live SOC Intrusion Detection Stream.
Real-time operations center consuming live predictions and critical alerts directly from Redpanda.
Supports real-time continuous ingestion and neural inference scaling without buffer caps.
"""

import time
import json
import streamlit as st
import pandas as pd
from confluent_kafka import Consumer

from dashboard.state import (
    start_producer,
    stop_producer,
    start_consumer,
    stop_consumer,
    get_producer,
    get_consumer,
    start_pipeline,
    stop_pipeline,
    reset_pipeline_stats
)
from dashboard.components import render_redpanda_status_banner
from visualization.streaming import plot_live_attack_distribution


def render_live_ids_page():
    st.title("🚨 Live Real-Time SOC Intrusion Detection Center")
    st.caption("Real-time threat monitoring consuming live inference events from Redpanda topics `ids-predictions` and `ids-alerts`.")

    prod = get_producer()
    cons = get_consumer()

    prod_running = bool(prod and prod.is_running and not getattr(prod, "is_paused", False))
    cons_running = bool(cons and cons.is_running)

    # Compute dynamic real-time broker & inference statistics
    stats = {}
    if cons and cons.is_running:
        c_stats = cons.get_stats()
        stats["messages_sec"] = c_stats.get("messages_sec", 0.0)
        stats["consumer_lag"] = c_stats.get("consumer_lag", 0)
        stats["p99_latency_ms"] = c_stats.get("p99_latency_ms", 0.0)
    elif prod and prod.is_running:
        p_stats = prod.get_stats()
        stats["messages_sec"] = p_stats.get("messages_sec", 0.0)
        stats["consumer_lag"] = 0
        stats["p99_latency_ms"] = 0.0
    else:
        stats["messages_sec"] = 0.0
        stats["consumer_lag"] = 0
        stats["p99_latency_ms"] = 0.0

    render_redpanda_status_banner(stats=stats)

    st.markdown("---")
    
    # Pipeline Status & Master Control Row
    st.subheader("🎮 Live Streaming Pipeline Controls")
    
    # Status badges above controls
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        if prod_running:
            p_rate = getattr(prod, "rate_msg_per_sec", 30.0)
            st.success(f"🟢 **Producer**: STREAMING ({p_rate:.0f} msg/s)")
        elif prod and getattr(prod, "is_paused", False):
            st.warning("⏸️ **Producer**: PAUSED")
        else:
            st.info("⚪ **Producer**: STOPPED")
    with sc2:
        if cons_running:
            st.success("🟢 **Inference Engine**: ACTIVE (Real-Time ML)")
        else:
            st.info("⚪ **Inference Engine**: STOPPED")
    with sc3:
        if prod_running and cons_running:
            st.success("⚡ **Pipeline**: FULL DUPLEX STREAMING")
        elif prod_running or cons_running:
            st.warning("⚠️ **Pipeline**: PARTIAL (Start both for live flow)")
        else:
            st.info("⚪ **Pipeline**: IDLE (Click Start below)")

    # Action buttons
    col_main1, col_main2, col_rate = st.columns([2, 2, 2])
    with col_rate:
        target_rate = st.slider("Target Ingestion Rate (msg/s):", min_value=5, max_value=100, value=30, step=5, key="stream_target_rate")
        if prod and hasattr(prod, "set_rate") and prod.rate != float(target_rate):
            prod.set_rate(float(target_rate))

    with col_main1:
        if not (prod_running and cons_running):
            if st.button("🚀 START FULL LIVE PIPELINE (PRODUCER + INFERENCE)", type="primary", use_container_width=True):
                start_pipeline(rate=float(target_rate))
                st.rerun()
        else:
            st.button("✅ FULL PIPELINE RUNNING", disabled=True, use_container_width=True)

    with col_main2:
        if prod_running or cons_running:
            if st.button("⏹️ STOP FULL PIPELINE", type="secondary", use_container_width=True):
                stop_pipeline()
                st.rerun()
        else:
            st.button("⏹️ FULL PIPELINE STOPPED", disabled=True, use_container_width=True)

    # Granular individual controls
    b1, b2, b3, b4, b5, b6, b7 = st.columns(7)
    with b1:
        if st.button("▶️ Start Producer", disabled=prod_running, use_container_width=True):
            start_producer(rate=float(target_rate))
            st.rerun()
    with b2:
        if st.button("⏹️ Stop Producer", disabled=not (prod and prod.is_running), use_container_width=True):
            stop_producer()
            st.rerun()
    with b3:
        if st.button("▶️ Start Inference", disabled=cons_running, use_container_width=True):
            start_consumer()
            st.rerun()
    with b4:
        if st.button("⏹️ Stop Inference", disabled=not cons_running, use_container_width=True):
            stop_consumer()
            st.rerun()
    with b5:
        if st.button("⏸️ Pause", disabled=not prod_running, use_container_width=True):
            if prod:
                prod.pause()
                st.rerun()
    with b6:
        if st.button("▶️ Resume", disabled=not (prod and getattr(prod, "is_paused", False)), use_container_width=True):
            if prod:
                prod.resume()
                st.rerun()
    with b7:
        if st.button("🧹 Reset Stats", use_container_width=True):
            reset_pipeline_stats()
            st.rerun()

    # Fast in-memory sync from background streaming consumer daemon
    if cons and cons.is_running:
        st.session_state.recent_predictions = list(reversed(cons.recent_predictions))
        st.session_state.recent_alerts = list(reversed(cons.recent_alerts))

    # Calculate real continuous cumulative counts
    if cons:
        c_stats = cons.get_stats()
        total_received = c_stats.get("messages_received", 0)
        total_inferences = c_stats.get("messages_processed", 0)
        total_benign = c_stats.get("total_benign", 0)
        total_threats = c_stats.get("total_threats", 0)
        total_critical = c_stats.get("total_critical", 0)
        total_high = c_stats.get("total_high", 0)
        attack_dist = c_stats.get("attack_distribution", {})
    else:
        # Fallback to session state cache if consumer is stopped
        total_events = len(st.session_state.recent_predictions)
        attacks = [p for p in st.session_state.recent_predictions if p.get("predicted_attack", "").lower() != "normal"]
        critical = [p for p in attacks if p.get("risk_level") == "CRITICAL"]
        high = [p for p in attacks if p.get("risk_level") == "HIGH"]
        total_received = total_events
        total_inferences = total_events
        total_benign = max(0, total_events - len(attacks))
        total_threats = len(attacks)
        total_critical = len(critical)
        total_high = len(high)
        attack_dist = {}
        for p in st.session_state.recent_predictions:
            att = p.get("predicted_attack", "Normal")
            attack_dist[att] = attack_dist.get(att, 0) + 1

    # Ensure attack_dist has entries for visualization if predictions exist
    if not attack_dist and st.session_state.recent_predictions:
        for p in st.session_state.recent_predictions:
            att = p.get("predicted_attack", "Normal")
            attack_dist[att] = attack_dist.get(att, 0) + 1

    st.markdown("---")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Live Events Received", f"{total_received:,}")
    k2.metric("Inferences Processed", f"{total_inferences:,}")
    k3.metric("Benign Events", f"{total_benign:,}")
    k4.metric("Detected Threat Events", f"{total_threats:,}")
    k5.metric("Critical Alerts", f"{total_critical:,}")
    k6.metric("High-Risk Alerts", f"{total_high:,}")

    col_alert, col_pie = st.columns([3, 2])

    with col_alert:
        st.subheader("🚨 Live Threat Alert Cards")
        if st.session_state.recent_alerts:
            for alt in st.session_state.recent_alerts[:3]:
                st.error(
                    f"### 🚨 {alt.get('severity', 'HIGH')} INTRUSION DETECTED: {alt.get('attack_type')}\n"
                    f"**Event ID**: `{alt.get('event_id')}` | **Timestamp**: `{alt.get('timestamp')}`\n\n"
                    f"**Confidence**: `{alt.get('confidence', 0)*100:.1f}%` | **Entropy**: `{alt.get('entropy', 0):.4f}`\n\n"
                    f"_{alt.get('description')}_"
                )
        else:
            attacks_recent = [p for p in st.session_state.recent_predictions if p.get("predicted_attack", "").lower() != "normal"]
            if attacks_recent:
                latest = attacks_recent[0]
                st.warning(
                    f"### ⚠️ DETECTED INTRUSION: {latest.get('predicted_attack')}\n"
                    f"**Event ID**: `{latest.get('event_id')}` | **Path**: `{latest.get('path_selected')}`\n\n"
                    f"**Confidence**: `{latest.get('confidence', 0)*100:.1f}%` | **Entropy**: `{latest.get('entropy', 0):.4f}` | **Latency**: `{latest.get('latency_breakdown', {}).get('total_pipeline_ms', 0):.2f} ms`"
                )
            else:
                st.info("No critical security alerts currently triggered. Stream traffic is nominal.")

    with col_pie:
        st.subheader("📊 Live Stream Traffic Composition")
        if attack_dist and sum(attack_dist.values()) > 0:
            st.plotly_chart(plot_live_attack_distribution(attack_dist), use_container_width=True)
        else:
            st.info("Awaiting live stream classifications to plot attack distribution.")

    # Live Event Stream Table
    st.markdown("---")
    st.subheader("📋 Real-Time Redpanda Ingestion Event Log")

    c_eval, c_auto = st.columns([2, 1])
    with c_eval:
        eval_mode = st.toggle("EVALUATION MODE (Show Ground Truth for Dynamic Scoring)", value=True)
    with c_auto:
        auto_refresh = st.toggle("⚡ LIVE STREAM AUTO-REFRESH (1s)", value=True, key="live_ids_auto_refresh")

    if st.session_state.recent_predictions:
        rows = []
        for p in st.session_state.recent_predictions[:50]:
            row = {
                "Timestamp": p.get("timestamp", "")[:19],
                "Event ID": p.get("event_id", ""),
                "Predicted Class": p.get("predicted_attack", ""),
                "Confidence": f"{p.get('confidence', 0)*100:.1f}%",
                "Entropy": f"{p.get('entropy', 0):.4f}",
                "Route": p.get("path_selected", "FAST"),
                "Latency": f"{p.get('latency_breakdown', {}).get('total_pipeline_ms', 0.8):.2f} ms",
                "Status": "🚨 Threat" if p.get("predicted_attack", "Normal") != "Normal" else "✓ Normal"
            }
            if eval_mode:
                row["Ground Truth"] = p.get("dataset_label", "--")
            rows.append(row)

        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Awaiting live messages from Redpanda. Click 'START FULL LIVE PIPELINE' above.")

    # Smooth non-blocking live refresh when pipeline is streaming
    if auto_refresh and ((cons and cons.is_running) or (prod and prod.is_running)):
        time.sleep(1.0)
        st.rerun()

