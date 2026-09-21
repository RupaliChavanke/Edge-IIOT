"""
Redpanda Cluster Health & Diagnostic Probing Module.
Dynamically checks cluster connection, topic partitions, and consumer lag.
Seamlessly detects and supports Cloud Mode (InMemoryStreamingBus) when Docker/Redpanda is not running.
"""

from typing import Dict, List, Optional, Any
import os
import socket
import time
import logging

try:
    from confluent_kafka.admin import AdminClient
    from confluent_kafka import Consumer, KafkaError, TopicPartition
    HAS_CONFLUENT_KAFKA = True
except ImportError:
    HAS_CONFLUENT_KAFKA = False

from streaming.bus import get_streaming_bus

logger = logging.getLogger(__name__)


class RedpandaHealthChecker:
    """Probes Redpanda cluster health, topics status, and consumer lag dynamically."""

    def __init__(self, brokers: Optional[str] = None):
        self.brokers = brokers or os.getenv("REDPANDA_BROKERS", "localhost:19092")
        self.explicit_cloud_mode = os.getenv("STREAMING_MODE", "").lower() in ["cloud", "memory", "in_memory"]

    def _broker_socket_available(self, timeout: float) -> bool:
        """Avoid creating Kafka clients when the configured broker is offline."""
        try:
            broker = self.brokers.split(",", 1)[0].strip()
            host, port = broker.rsplit(":", 1)
            with socket.create_connection((host.strip("[]"), int(port)), timeout=timeout):
                return True
        except (OSError, ValueError):
            return False

    def is_redpanda_available(self, timeout: float = 1.5) -> bool:
        """Fast check if Redpanda broker is responsive."""
        if self.explicit_cloud_mode or not HAS_CONFLUENT_KAFKA:
            return False
        if not self._broker_socket_available(timeout):
            return False
        try:
            admin = AdminClient({
                "bootstrap.servers": self.brokers,
                "socket.timeout.ms": int(timeout * 1000)
            })
            metadata = admin.list_topics(timeout=timeout)
            return bool(metadata.brokers)
        except Exception:
            return False

    def check_health(self) -> Dict[str, Any]:
        """
        Dynamically verifies Redpanda cluster connectivity or activates Cloud Fallback.
        """
        t0 = time.perf_counter()

        mandatory_topics = [
            "edge-iiot-raw",
            "edge-iiot-preprocessed",
            "ids-predictions",
            "ids-alerts",
            "ids-metrics"
        ]

        if not self.explicit_cloud_mode and HAS_CONFLUENT_KAFKA:
            if not self._broker_socket_available(0.25):
                error_str = f"Broker {self.brokers} is not accepting TCP connections."
            else:
                try:
                    admin = AdminClient({
                        "bootstrap.servers": self.brokers,
                        "socket.timeout.ms": 2000
                    })
                    # Query cluster metadata
                    metadata = admin.list_topics(timeout=2)
                    ping_ms = (time.perf_counter() - t0) * 1000.0

                    broker_list = list(metadata.brokers.values())
                    topics_meta = metadata.topics

                    topic_status = {}
                    for t_name in mandatory_topics:
                        if t_name in topics_meta:
                            t_info = topics_meta[t_name]
                            num_partitions = len(t_info.partitions)
                            topic_status[t_name] = {
                                "status": "ACTIVE" if not t_info.error else f"ERROR: {t_info.error}",
                                "partitions": num_partitions
                            }
                        else:
                            topic_status[t_name] = {
                                "status": "NOT_CREATED",
                                "partitions": 0
                            }

                    return {
                        "status": "CONNECTED",
                        "mode": "redpanda",
                        "broker": self.brokers,
                        "ping_ms": round(ping_ms, 2),
                        "broker_count": len(broker_list),
                        "brokers": [f"{b.host}:{b.port} (node {b.id})" for b in broker_list],
                        "topics": topic_status,
                        "error": None,
                        "cloud_fallback": False,
                        "troubleshooting": None
                    }
                except Exception as e:
                    logger.debug(f"Redpanda cluster check at {self.brokers} failed: {e}. Activating Cloud Mode.")
                    error_str = str(e)
        else:
            error_str = "Explicit cloud mode requested or confluent_kafka unavailable."

        # Fallback to Cloud In-Memory Streaming Bus
        topic_status = {
            t_name: {
                "status": "ACTIVE",
                "partitions": 1
            }
            for t_name in mandatory_topics
        }

        return {
            "status": "CLOUD_MODE",
            "mode": "cloud",
            "broker": "InMemoryStreamingBus (Cloud)",
            "ping_ms": 0.1,
            "broker_count": 1,
            "brokers": ["InMemoryStreamingBus (Thread-Safe Virtual Queue)"],
            "topics": topic_status,
            "error": None,
            "cloud_fallback": True,
            "underlying_error": error_str,
            "troubleshooting": (
                "Running in Zero-Dependency Cloud Mode.\n"
                "To connect to an external Redpanda/Kafka cluster:\n"
                "1. Set REDPANDA_BROKERS=host:port in environment variables.\n"
                "2. For local Docker: docker compose up -d redpanda redpanda-console"
            )
        }

    def get_consumer_lag(self, group_id: str = "iiot-ids-inference", topic: str = "edge-iiot-raw") -> int:
        """Computes current consumer lag for the inference consumer group or in-memory queue."""
        # Check in-memory bus lag first if cloud mode
        if self.explicit_cloud_mode or not HAS_CONFLUENT_KAFKA:
            return get_streaming_bus().get_lag(topic)

        try:
            consumer = Consumer({
                "bootstrap.servers": self.brokers,
                "group.id": group_id,
                "enable.auto.commit": False,
                "session.timeout.ms": 2000
            })
            metadata = consumer.list_topics(topic, timeout=1)
            if topic not in metadata.topics:
                consumer.close()
                return get_streaming_bus().get_lag(topic)

            partitions = metadata.topics[topic].partitions
            total_lag = 0

            for p_id in partitions.keys():
                tp = TopicPartition(topic, p_id)
                low, high = consumer.get_watermark_offsets(tp, timeout=1)
                committed = consumer.committed([tp], timeout=1)
                if committed and committed[0].offset >= 0:
                    comm_offset = committed[0].offset
                else:
                    comm_offset = high
                lag = max(0, high - comm_offset)
                total_lag += lag

            consumer.close()
            return total_lag
        except Exception:
            return get_streaming_bus().get_lag(topic)
