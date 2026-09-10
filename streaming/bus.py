"""
High-Performance In-Memory Streaming Bus.
Provides zero-dependency, thread-safe Kafka-compatible pub/sub streaming
for cloud environments (e.g. Streamlit Community Cloud) where external Docker/Redpanda
brokers are not present.
"""

from typing import Dict, Optional, Callable, Tuple, Any
import queue
import time
import threading
import logging

logger = logging.getLogger(__name__)


class StreamingMessage:
    """Kafka Message interface adapter for in-memory streaming events."""

    def __init__(self, topic: str, value: bytes, key: Optional[str] = None, offset: int = 0):
        self._topic = topic
        self._value = value
        self._key = key.encode("utf-8") if isinstance(key, str) else key
        self._offset = offset
        self._timestamp = int(time.time() * 1000)

    def value(self) -> bytes:
        return self._value

    def key(self) -> Optional[bytes]:
        return self._key

    def topic(self) -> str:
        return self._topic

    def offset(self) -> int:
        return self._offset

    def timestamp(self) -> Tuple[int, int]:
        return (1, self._timestamp)

    def error(self) -> Optional[Any]:
        return None


class InMemoryStreamingBus:
    """
    Thread-safe in-memory message bus routing events across streaming topics.
    Mimics Kafka/Redpanda partition queues with rate control and consumer lag tracking.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(InMemoryStreamingBus, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, maxsize_per_topic: int = 10000):
        if getattr(self, "_initialized", False):
            return
        self.maxsize_per_topic = maxsize_per_topic
        self._topics: Dict[str, queue.Queue] = {
            "edge-iiot-raw": queue.Queue(maxsize=maxsize_per_topic),
            "edge-iiot-preprocessed": queue.Queue(maxsize=maxsize_per_topic),
            "ids-predictions": queue.Queue(maxsize=maxsize_per_topic),
            "ids-alerts": queue.Queue(maxsize=maxsize_per_topic),
            "ids-metrics": queue.Queue(maxsize=maxsize_per_topic)
        }
        self._offsets: Dict[str, int] = {t: 0 for t in self._topics}
        self._offsets_lock = threading.Lock()
        self._initialized = True
        logger.info("InMemoryStreamingBus initialized for Cloud & Zero-Broker mode.")

    def _get_queue(self, topic: str) -> queue.Queue:
        if topic not in self._topics:
            with self._lock:
                if topic not in self._topics:
                    self._topics[topic] = queue.Queue(maxsize=self.maxsize_per_topic)
                    self._offsets[topic] = 0
        return self._topics[topic]

    def publish(
        self,
        topic: str,
        value: bytes,
        key: Optional[str] = None,
        callback: Optional[Callable[[Optional[Exception], StreamingMessage], None]] = None
    ) -> StreamingMessage:
        """Publishes a byte payload to a named topic with offset tracking."""
        q = self._get_queue(topic)
        with self._offsets_lock:
            self._offsets[topic] += 1
            current_offset = self._offsets[topic]

        msg = StreamingMessage(topic=topic, value=value, key=key, offset=current_offset)

        try:
            # If queue is full, drop oldest item to maintain real-time low latency
            if q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            q.put_nowait(msg)
            if callback:
                callback(None, msg)
        except Exception as e:
            if callback:
                callback(e, msg)
            raise e

        return msg

    def poll(self, topic: str, timeout: float = 0.5) -> Optional[StreamingMessage]:
        """Polls for the next message on the specified topic."""
        q = self._get_queue(topic)
        try:
            return q.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def get_lag(self, topic: str) -> int:
        """Returns the current queue depth (consumer lag) for the topic."""
        if topic in self._topics:
            return self._topics[topic].qsize()
        return 0

    def clear(self, topic: Optional[str] = None):
        """Clears buffers for one or all topics."""
        topics = [topic] if topic else list(self._topics.keys())
        for t in topics:
            if t in self._topics:
                q = self._topics[t]
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break


def get_streaming_bus() -> InMemoryStreamingBus:
    """Accessor for the global singleton in-memory streaming bus."""
    return InMemoryStreamingBus()
