"""
Redpanda Topic Management and Initialization.
Configures partitions, replication factors, and ensures mandatory topics exist.
"""

from typing import List, Dict, Optional, Any
import os
from confluent_kafka.admin import AdminClient, NewTopic
import logging

logger = logging.getLogger(__name__)

MANDATORY_TOPICS = [
    "edge-iiot-raw",
    "edge-iiot-preprocessed",
    "ids-predictions",
    "ids-alerts",
    "ids-metrics",
    "ids-dead-letter"
]


class TopicManager:
    """Manages Kafka/Redpanda topic creation and verification."""

    def __init__(self, brokers: Optional[str] = None):
        self.brokers = brokers or os.getenv("REDPANDA_BROKERS", "localhost:19092")
        self.admin = AdminClient({"bootstrap.servers": self.brokers})

    def create_topics(self, num_partitions: int = 3, replication_factor: int = 1) -> Dict[str, str]:
        """Creates all required topics if they don't already exist."""
        results = {}
        try:
            # Query existing topics
            metadata = self.admin.list_topics(timeout=5)
            existing = set(metadata.topics.keys())

            new_topics = []
            for topic_name in MANDATORY_TOPICS:
                if topic_name not in existing:
                    new_topics.append(
                        NewTopic(
                            topic_name,
                            num_partitions=num_partitions,
                            replication_factor=replication_factor
                        )
                    )
                else:
                    results[topic_name] = "EXISTS"

            if new_topics:
                futures = self.admin.create_topics(new_topics)
                for topic_name, future in futures.items():
                    try:
                        future.result()  # Blocks until topic is created
                        results[topic_name] = "CREATED"
                        logger.info(f"Topic '{topic_name}' successfully created.")
                    except Exception as e:
                        results[topic_name] = f"ERROR: {e}"
                        logger.warning(f"Error creating topic '{topic_name}': {e}")
        except Exception as e:
            logger.error(f"Failed to connect to broker {self.brokers}: {e}")
            results["cluster_error"] = str(e)

        return results

    def list_topics_info(self) -> Dict[str, Dict[str, Any]]:
        """Returns details on partitions and brokers for all topics."""
        info = {}
        try:
            metadata = self.admin.list_topics(timeout=3)
            for topic_name, topic_meta in metadata.topics.items():
                if topic_name in MANDATORY_TOPICS or not topic_name.startswith("_"):
                    info[topic_name] = {
                        "partitions": len(topic_meta.partitions),
                        "partition_ids": list(topic_meta.partitions.keys()),
                        "error": str(topic_meta.error) if topic_meta.error else None
                    }
        except Exception as e:
            logger.warning(f"Failed to list topics: {e}")
        return info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = TopicManager()
    res = manager.create_topics()
    print("Topic Creation Results:", res)
