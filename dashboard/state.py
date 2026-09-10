"""
Shared Streamlit Session State and Streaming Pipeline Controller.
Manages persistent background producer and consumer threads across Streamlit reruns.
"""

from typing import Dict, List, Optional, Any
import os
import time
import json
import streamlit as st
try:
    from confluent_kafka import Consumer, Producer
    HAS_CONFLUENT_KAFKA = True
except ImportError:
    HAS_CONFLUENT_KAFKA = False

from streaming.health import RedpandaHealthChecker
from streaming.redpanda_producer import EdgeIIoTStreamingProducer
from streaming.redpanda_consumer import EdgeIIoTStreamingConsumer
from streaming.topics import TopicManager
from streaming.bus import get_streaming_bus


# Process-wide daemon singletons persisting across Streamlit session reloads
_GLOBAL_PRODUCER: Optional[EdgeIIoTStreamingProducer] = None
_GLOBAL_CONSUMER: Optional[EdgeIIoTStreamingConsumer] = None


def init_session_state():
    """Initializes persistent streaming objects in Streamlit session state and reconnects to active daemons."""
    global _GLOBAL_PRODUCER, _GLOBAL_CONSUMER

    try:
        if "producer" not in st.session_state or st.session_state.producer is None:
            if _GLOBAL_PRODUCER is not None and _GLOBAL_PRODUCER.is_running:
                st.session_state.producer = _GLOBAL_PRODUCER
            else:
                st.session_state.producer = None

        if "consumer" not in st.session_state or st.session_state.consumer is None:
            if _GLOBAL_CONSUMER is not None and _GLOBAL_CONSUMER.is_running:
                st.session_state.consumer = _GLOBAL_CONSUMER
            else:
                st.session_state.consumer = None

        if "recent_predictions" not in st.session_state:
            st.session_state.recent_predictions = []
        if "recent_alerts" not in st.session_state:
            st.session_state.recent_alerts = []
        if "streaming_stats" not in st.session_state:
            st.session_state.streaming_stats = {
                "messages_sent": 0,
                "messages_processed": 0,
                "total_alerts": 0,
                "p99_latency_ms": 0.0,
                "throughput": 0.0
            }
    except Exception:
        pass


def get_producer() -> Optional[EdgeIIoTStreamingProducer]:
    """Returns or creates the streaming producer."""
    init_session_state()
    global _GLOBAL_PRODUCER
    if _GLOBAL_PRODUCER is not None and _GLOBAL_PRODUCER.is_running:
        return _GLOBAL_PRODUCER
    try:
        return st.session_state.get("producer")
    except Exception:
        return None


def get_consumer() -> Optional[EdgeIIoTStreamingConsumer]:
    """Returns or creates the streaming consumer."""
    init_session_state()
    global _GLOBAL_CONSUMER
    if _GLOBAL_CONSUMER is not None and _GLOBAL_CONSUMER.is_running:
        return _GLOBAL_CONSUMER
    try:
        return st.session_state.get("consumer")
    except Exception:
        return None


def start_producer(rate: float = 30.0, dataset_path: str = "data/samples/edge_iiot_sample.csv") -> EdgeIIoTStreamingProducer:
    """Initializes and starts the streaming producer."""
    global _GLOBAL_PRODUCER
    if _GLOBAL_PRODUCER is None or not _GLOBAL_PRODUCER.is_running:
        _GLOBAL_PRODUCER = EdgeIIoTStreamingProducer(rate_msg_per_sec=rate, dataset_path=dataset_path)
        _GLOBAL_PRODUCER.start(loop_dataset=True)
    try:
        st.session_state.producer = _GLOBAL_PRODUCER
    except Exception:
        pass
    return _GLOBAL_PRODUCER


def stop_producer():
    """Stops the streaming producer."""
    global _GLOBAL_PRODUCER
    if _GLOBAL_PRODUCER is not None:
        _GLOBAL_PRODUCER.stop()
        _GLOBAL_PRODUCER = None
    try:
        if "producer" in st.session_state:
            st.session_state.producer = None
    except Exception:
        pass


def start_consumer() -> EdgeIIoTStreamingConsumer:
    """Initializes and starts the streaming consumer."""
    global _GLOBAL_CONSUMER
    if _GLOBAL_CONSUMER is None or not _GLOBAL_CONSUMER.is_running:
        _GLOBAL_CONSUMER = EdgeIIoTStreamingConsumer()
        _GLOBAL_CONSUMER.start()
    try:
        st.session_state.consumer = _GLOBAL_CONSUMER
    except Exception:
        pass
    return _GLOBAL_CONSUMER


def stop_consumer():
    """Stops the streaming consumer."""
    global _GLOBAL_CONSUMER
    if _GLOBAL_CONSUMER is not None:
        _GLOBAL_CONSUMER.stop()
        _GLOBAL_CONSUMER = None
    try:
        if "consumer" in st.session_state:
            st.session_state.consumer = None
    except Exception:
        pass


def start_pipeline(rate: float = 30.0, dataset_path: str = "data/samples/edge_iiot_sample.csv"):
    """Starts consumer inference engine and producer in synchronized sequence."""
    cons = start_consumer()
    time.sleep(0.3)
    prod = start_producer(rate=rate, dataset_path=dataset_path)
    return prod, cons


def stop_pipeline():
    """Stops both producer and consumer engines."""
    stop_producer()
    stop_consumer()


def reset_pipeline_stats():
    """Resets all live statistics and buffers across producer, consumer, and UI session."""
    cons = get_consumer()
    if cons:
        cons.reset_stats()
    prod = get_producer()
    if prod:
        prod.reset_stats()
    try:
        st.session_state.recent_predictions = []
        st.session_state.recent_alerts = []
    except Exception:
        pass
    from dashboard.live_evaluator import LiveEvaluatorEngine
    LiveEvaluatorEngine.get_instance().reset()


def inject_single_sample_to_redpanda(sample_features: Dict[str, Any], ground_truth: Optional[str] = None) -> bool:
    """
    Strict Requirement:
    Publishes a single network sample directly to 'edge-iiot-raw' (via Redpanda or Cloud Bus).
    Never calls the model directly from the UI frontend!
    """
    import uuid
    from streaming.schemas import RawNetworkEvent
    brokers = os.getenv("REDPANDA_BROKERS", "localhost:19092")

    event = RawNetworkEvent(
        event_id=f"EVT-MANUAL-{uuid.uuid4().hex[:6].upper()}",
        sequence_id=999999,
        features=sample_features,
        dataset_label=ground_truth
    )
    payload = event.model_dump_json().encode("utf-8")

    checker = RedpandaHealthChecker(brokers)
    if checker.is_redpanda_available() and HAS_CONFLUENT_KAFKA:
        try:
            producer = Producer({"bootstrap.servers": brokers, "acks": 1})
            producer.produce("edge-iiot-raw", value=payload)
            producer.flush(timeout=3)
            return True
        except Exception:
            get_streaming_bus().publish("edge-iiot-raw", value=payload)
            return True
    else:
        get_streaming_bus().publish("edge-iiot-raw", value=payload)
        return True
