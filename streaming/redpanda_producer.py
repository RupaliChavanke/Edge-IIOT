"""
High-Performance Rate-Controlled Redpanda Streaming Producer for Edge-IIoTset.
Publishes raw network flow events to 'edge-iiot-raw' without target leakage.
Seamlessly falls back to InMemoryStreamingBus when operating in Cloud / Zero-Broker environments.
"""

from typing import Dict, List, Optional, Any
import os
import sys
import time
import json
import uuid
import argparse
import threading
from collections import deque
import pandas as pd
import numpy as np
import logging

try:
    from confluent_kafka import Producer
    HAS_CONFLUENT_KAFKA = True
except ImportError:
    HAS_CONFLUENT_KAFKA = False

from streaming.schemas import RawNetworkEvent
from streaming.topics import TopicManager
from streaming.health import RedpandaHealthChecker
from streaming.bus import get_streaming_bus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RedpandaProducer")


class EdgeIIoTStreamingProducer:
    """Streams Edge-IIoTset events into Redpanda with rate limiting and pause/resume control."""

    def __init__(
        self,
        brokers: Optional[str] = None,
        topic: str = "edge-iiot-raw",
        dataset_path: str = "data/samples/edge_iiot_sample.csv",
        rate_msg_per_sec: float = 50.0,
        batch_size: int = 10,
        force_cloud_mode: bool = False
    ):
        self.brokers = brokers or os.getenv("REDPANDA_BROKERS", "localhost:19092")
        self.topic = topic
        self.dataset_path = dataset_path
        self.rate_msg_per_sec = float(rate_msg_per_sec)
        self.batch_size = int(batch_size)

        # Determine streaming backend (Redpanda Cluster vs Cloud In-Memory Bus)
        self.use_cloud_bus = force_cloud_mode or not HAS_CONFLUENT_KAFKA or not RedpandaHealthChecker(self.brokers).is_redpanda_available()
        self.producer = None

        if not self.use_cloud_bus and HAS_CONFLUENT_KAFKA:
            try:
                TopicManager(self.brokers).create_topics()
                self.conf = {
                    "bootstrap.servers": self.brokers,
                    "client.id": "edge-iiot-producer",
                    "linger.ms": 5,
                    "compression.type": "snappy",
                    "acks": 1,
                    "queue.buffering.max.messages": 100000
                }
                self.producer = Producer(self.conf)
                logger.info(f"RedpandaProducer connected to cluster at {self.brokers}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redpanda at {self.brokers} ({e}). Falling back to InMemoryStreamingBus.")
                self.use_cloud_bus = True
                self.producer = None
        else:
            logger.info("RedpandaProducer active in Cloud In-Memory Streaming Bus mode.")

        # Threading controls
        self.is_running = False
        self.is_paused = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # Unpaused initially
        self._stop_event = threading.Event()
        self.producer_thread: Optional[threading.Thread] = None

        # Statistics
        self.total_messages_sent = 0
        self.total_delivery_errors = 0
        self.current_throughput = 0.0
        self.last_event_timestamp = ""
        self.recent_timestamps: deque = deque(maxlen=200)

    def delivery_callback(self, err, msg):
        """Callback invoked on delivery confirmation or error."""
        if err is not None:
            self.total_delivery_errors += 1
            logger.warning(f"Message delivery failed: {err}")
        else:
            self.total_messages_sent += 1
            self.recent_timestamps.append(time.perf_counter())

    def stream_worker(self, loop_dataset: bool = True):
        """Worker thread producing messages with exact rate control."""
        mode_str = "Cloud In-Memory Bus" if self.use_cloud_bus else f"Redpanda ({self.brokers})"
        logger.info(f"Streaming thread started [{mode_str}]. Rate: {self.rate_msg_per_sec} msg/s, Target: {self.topic}")
        seq_id = 0

        while not self._stop_event.is_set():
            if not os.path.exists(self.dataset_path):
                logger.error(f"Dataset path does not exist: {self.dataset_path}")
                break

            # Read dataset in chunks preserving exact text/numeric types
            chunk_size = 2000
            for chunk in pd.read_csv(self.dataset_path, chunksize=chunk_size, dtype=str, low_memory=False):
                if self._stop_event.is_set():
                    break

                for _, row in chunk.iterrows():
                    if self._stop_event.is_set():
                        break

                    # Handle pause state
                    self._pause_event.wait()

                    t_start = time.perf_counter()

                    # Extract ground truth label separately (no leakage into feature payload)
                    dataset_label = str(row.get("Attack_type", "Unknown"))
                    features = {col: (val if pd.notnull(val) else None) for col, val in row.items() if col not in ["Attack_type", "Attack_label"]}

                    seq_id += 1
                    event = RawNetworkEvent(
                        event_id=f"EVT-{uuid.uuid4().hex[:8].upper()}",
                        sequence_id=seq_id,
                        features=features,
                        dataset_label=dataset_label
                    )

                    payload = event.model_dump_json().encode("utf-8")

                    if self.use_cloud_bus:
                        get_streaming_bus().publish(
                            topic=self.topic,
                            value=payload,
                            key=str(seq_id),
                            callback=self.delivery_callback
                        )
                    else:
                        try:
                            self.producer.produce(
                                topic=self.topic,
                                value=payload,
                                key=str(seq_id),
                                callback=self.delivery_callback
                            )
                        except BufferError:
                            self.producer.poll(0.05)
                            self.producer.produce(
                                topic=self.topic,
                                value=payload,
                                key=str(seq_id),
                                callback=self.delivery_callback
                            )

                    self.last_event_timestamp = event.timestamp

                    # Poll for delivery callbacks periodically
                    if not self.use_cloud_bus and self.producer and (seq_id % self.batch_size == 0):
                        self.producer.poll(0)

                    # Precise rate pacing
                    delay = 1.0 / max(1.0, self.rate_msg_per_sec)
                    elapsed = time.perf_counter() - t_start
                    sleep_time = delay - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            if not loop_dataset:
                logger.info("Finished one pass over dataset. Stopping.")
                break

        if not self.use_cloud_bus and self.producer:
            self.producer.flush(timeout=5)
        self.is_running = False
        logger.info(f"Streaming worker stopped. Total sent: {self.total_messages_sent}")

    def start(self, loop_dataset: bool = True):
        """Starts background streaming thread."""
        if self.is_running:
            return
        self.is_running = True
        self.is_paused = False
        self._stop_event.clear()
        self._pause_event.set()
        self.producer_thread = threading.Thread(target=self.stream_worker, args=(loop_dataset,), daemon=True)
        self.producer_thread.start()

    def pause(self):
        """Pauses streaming without stopping thread."""
        self.is_paused = True
        self._pause_event.clear()

    def resume(self):
        """Resumes paused streaming."""
        self.is_paused = False
        self._pause_event.set()

    def stop(self):
        """Terminates streaming thread."""
        self.is_running = False
        self._stop_event.set()
        self._pause_event.set()  # Unblock if paused
        if self.producer_thread and self.producer_thread.is_alive():
            self.producer_thread.join(timeout=3)

    def set_rate(self, new_rate: float):
        """Dynamically adjust message rate."""
        self.rate_msg_per_sec = max(1.0, float(new_rate))

    @property
    def rate(self) -> float:
        return self.rate_msg_per_sec

    @rate.setter
    def rate(self, new_rate: float):
        self.set_rate(new_rate)

    def get_throughput(self) -> float:
        """Calculates instantaneous messages/sec produced over recent time window."""
        if not self.is_running or self.is_paused or len(self.recent_timestamps) < 2:
            return 0.0
        now = time.perf_counter()
        while self.recent_timestamps and (now - self.recent_timestamps[0]) > 4.0:
            self.recent_timestamps.popleft()
        if len(self.recent_timestamps) < 2:
            return 0.0
        dt = now - self.recent_timestamps[0]
        return round(float(len(self.recent_timestamps) / max(0.05, dt)), 1)

    def reset_stats(self):
        """Resets all producer statistics."""
        self.total_messages_sent = 0
        self.total_delivery_errors = 0
        self.recent_timestamps.clear()
        self.last_event_timestamp = ""

    def get_stats(self) -> Dict[str, Any]:
        """Returns live statistics."""
        mode_label = "In-Memory Bus (Cloud)" if self.use_cloud_bus else f"Redpanda ({self.brokers})"
        return {
            "status": "RUNNING" if self.is_running and not self.is_paused else ("PAUSED" if self.is_paused else "STOPPED"),
            "messages_sent": self.total_messages_sent,
            "errors": self.total_delivery_errors,
            "target_rate_msg_sec": self.rate_msg_per_sec,
            "messages_sec": self.get_throughput(),
            "topic": self.topic,
            "brokers": self.brokers,
            "backend_mode": "cloud" if self.use_cloud_bus else "redpanda",
            "backend_label": mode_label,
            "last_event_time": self.last_event_timestamp
        }


# Alias for backward compatibility
EdgeIIoTTestSetProducer = EdgeIIoTStreamingProducer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge-IIoTset Redpanda Producer")
    parser.add_argument("--rate", type=float, default=25.0, help="Messages per second")
    parser.add_argument("--dataset", type=str, default="data/samples/edge_iiot_sample.csv", help="CSV dataset path")
    parser.add_argument("--loop", action="store_true", help="Loop dataset indefinitely")
    parser.add_argument("--cloud", action="store_true", help="Force cloud in-memory bus mode")
    args = parser.parse_args()

    producer = EdgeIIoTStreamingProducer(dataset_path=args.dataset, rate_msg_per_sec=args.rate, force_cloud_mode=args.cloud)
    producer.start(loop_dataset=args.loop)

    try:
        while True:
            time.sleep(2)
            stats = producer.get_stats()
            print(f"Producer [{stats['backend_label']}]: {stats['status']} | Sent: {stats['messages_sent']} | Rate: {stats['target_rate_msg_sec']} msg/s")
    except KeyboardInterrupt:
        producer.stop()
