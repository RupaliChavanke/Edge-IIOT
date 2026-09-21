"""
Live Integration Test Script for Containerized Redpanda Streaming Pipeline.
Verifies production to 'edge-iiot-raw', neural inference by consumer,
and dispatch to 'ids-predictions' and 'ids-alerts'.
"""

import time
import os
from streaming.redpanda_producer import EdgeIIoTStreamingProducer
from streaming.redpanda_consumer import EdgeIIoTStreamingConsumer

def main():
    broker = os.getenv("REDPANDA_BROKERS", "redpanda:9092")
    print(f"--- STARTING LIVE DOCKER REDPANDA PIPELINE TEST (Broker: {broker}) ---")

    producer = EdgeIIoTStreamingProducer(brokers=broker, rate_msg_per_sec=30)
    producer.start(loop_dataset=True)

    consumer = EdgeIIoTStreamingConsumer(brokers=broker)
    consumer.start()

    time.sleep(4.5)

    p_stats = producer.get_stats()
    c_stats = consumer.get_stats()

    producer.stop()
    consumer.stop()

    sent = p_stats['messages_sent']
    recv = c_stats['messages_received']
    proc = c_stats['messages_processed']
    threats = c_stats['total_threats']
    benign = c_stats['total_benign']
    crit = c_stats['total_critical']
    p50 = c_stats['p50_latency_ms']
    p99 = c_stats['p99_latency_ms']
    early = c_stats['early_exit_percentage']
    last_pred = c_stats.get('last_prediction') or {}
    pred_attack = last_pred.get('predicted_attack', 'N/A')
    conf = last_pred.get('confidence', 0.0)

    print("\n--- LIVE STREAMING EXECUTION RESULTS ---")
    print(f"Messages Sent by Producer:      {sent}")
    print(f"Messages Received by Consumer:  {recv}")
    print(f"Messages Processed by Consumer: {proc}")
    print(f"Detected Threats:               {threats}")
    print(f"Detected Benign Packets:        {benign}")
    print(f"Critical Alerts Generated:      {crit}")
    print(f"P50 Pipeline Latency:           {p50} ms")
    print(f"P99 Pipeline Latency:           {p99} ms")
    print(f"Early Exit Percentage:          {early} %")
    print(f"Last Predicted Attack Type:     {pred_attack}")
    print(f"Last Prediction Confidence:     {conf}")

    assert sent > 0, "Producer failed to send messages"
    assert recv > 0, "Consumer failed to receive messages"
    assert proc > 0, "Consumer failed to process messages"
    assert (threats + benign) == proc, "Threat and benign count mismatch"
    print("\n--- TEST PASSED: ALL RESULTS PRODUCED PROPERLY ---")

if __name__ == "__main__":
    main()
