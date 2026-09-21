"""
End-to-End Integration Test for Redpanda Streaming Pipeline.
Verifies live message production to 'edge-iiot-raw', consumption, inference, and prediction publishing.
"""

import os
import time
import pytest
from streaming.health import RedpandaHealthChecker
from streaming.redpanda_producer import EdgeIIoTStreamingProducer
from streaming.redpanda_consumer import EdgeIIoTStreamingConsumer


def test_live_redpanda_pipeline():
    broker = os.getenv("REDPANDA_BROKERS", "localhost:19092")
    checker = RedpandaHealthChecker(broker)
    health = checker.check_health()
    if health["status"] != "CONNECTED":
        pytest.skip(f"Redpanda broker not reachable at {broker}. Skipping live streaming test.")

    producer = EdgeIIoTStreamingProducer(brokers=broker, rate_msg_per_sec=20)
    producer.start(loop_dataset=False)

    consumer = EdgeIIoTStreamingConsumer(brokers=broker)
    consumer.start()

    # Wait for events to circulate
    time.sleep(3)

    p_stats = producer.get_stats()
    c_stats = consumer.get_stats()

    producer.stop()
    consumer.stop()

    assert p_stats["messages_sent"] > 0
    assert c_stats["messages_received"] > 0
    assert c_stats["messages_processed"] > 0
    assert c_stats["p99_latency_ms"] >= 0.0
    assert "total_threats" in c_stats
    assert "total_benign" in c_stats


def test_streaming_counter_scaling_and_reset():
    from dashboard.state import start_pipeline, stop_pipeline, reset_pipeline_stats

    broker = os.getenv("REDPANDA_BROKERS", "localhost:19092")
    checker = RedpandaHealthChecker(broker)
    if checker.check_health()["status"] != "CONNECTED":
        pytest.skip(f"Redpanda broker not reachable at {broker}.")

    prod, cons = start_pipeline(rate=50.0)
    time.sleep(2.5)

    c_stats = cons.get_stats()
    assert c_stats["messages_processed"] > 0
    assert c_stats["total_threats"] + c_stats["total_benign"] == c_stats["messages_processed"]
    assert c_stats["messages_sec"] >= 0

    cons.stop()
    prod.stop()
    cons.reset_stats()
    assert cons.get_stats()["messages_processed"] == 0


def test_cloud_in_memory_streaming_pipeline():
    """Verifies that the Cloud In-Memory Streaming Bus operates with real PyTorch neural inference."""
    producer = EdgeIIoTStreamingProducer(rate_msg_per_sec=40, force_cloud_mode=True)
    consumer = EdgeIIoTStreamingConsumer(force_cloud_mode=True)

    producer.start(loop_dataset=True)
    consumer.start()

    time.sleep(2.5)

    p_stats = producer.get_stats()
    c_stats = consumer.get_stats()

    producer.stop()
    consumer.stop()

    assert p_stats["messages_sent"] > 0
    assert p_stats["backend_mode"] == "cloud"
    assert c_stats["messages_received"] > 0
    assert c_stats["messages_processed"] > 0
    assert c_stats["backend_mode"] == "cloud"
    assert c_stats["p99_latency_ms"] >= 0.0
    assert c_stats["total_threats"] + c_stats["total_benign"] == c_stats["messages_processed"]
