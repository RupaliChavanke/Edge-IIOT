"""
Redpanda Real-Time Streaming Consumer for Edge-IIoTset Intrusion Detection.
Consumes raw network flow events, performs neural inference with calibrated confidence,
routes through early exit or deep path, and publishes predictions and alerts.
Seamlessly falls back to InMemoryStreamingBus when operating in Cloud / Zero-Broker environments.
"""

from typing import Dict, List, Optional, Any
import os
import sys
import time
import json
import uuid
import threading
from collections import deque
import numpy as np
import pandas as pd
import torch
import logging

try:
    from confluent_kafka import Consumer, Producer, KafkaError, TopicPartition
    HAS_CONFLUENT_KAFKA = True
except ImportError:
    HAS_CONFLUENT_KAFKA = False

from streaming.schemas import (
    RawNetworkEvent,
    PredictionResult,
    SecurityAlert,
    LatencyBreakdown
)
from streaming.topics import TopicManager
from streaming.health import RedpandaHealthChecker
from streaming.bus import get_streaming_bus
from preprocessing.loader import EdgeIIoTDataLoader
from models.proposed_model import ProposedHybridEdgeIIoTModel
from training.checkpoint import CheckpointManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RedpandaConsumer")


class EdgeIIoTStreamingConsumer:
    """Consumes raw network traffic from Redpanda and generates real-time predictions and alerts."""

    def __init__(
        self,
        brokers: Optional[str] = None,
        raw_topic: str = "edge-iiot-raw",
        prediction_topic: str = "ids-predictions",
        alert_topic: str = "ids-alerts",
        metrics_topic: str = "ids-metrics",
        group_id: Optional[str] = None,
        checkpoint_dir: str = "checkpoints",
        force_cloud_mode: bool = False
    ):
        self.brokers = brokers or os.getenv("REDPANDA_BROKERS", "localhost:19092")
        self.raw_topic = raw_topic
        self.prediction_topic = prediction_topic
        self.alert_topic = alert_topic
        self.metrics_topic = metrics_topic
        self.group_id = group_id or f"iiot-ids-inference-{int(time.time())}"
        self.checkpoint_dir = checkpoint_dir

        # Determine streaming backend
        self.use_cloud_bus = force_cloud_mode or not HAS_CONFLUENT_KAFKA or not RedpandaHealthChecker(self.brokers).is_redpanda_available()
        self.consumer = None
        self.producer = None

        if not self.use_cloud_bus and HAS_CONFLUENT_KAFKA:
            try:
                TopicManager(self.brokers).create_topics()
                self.consumer_conf = {
                    "bootstrap.servers": self.brokers,
                    "group.id": self.group_id,
                    "auto.offset.reset": "latest",
                    "enable.auto.commit": False,
                    "session.timeout.ms": 6000
                }
                self.consumer = Consumer(self.consumer_conf)
                self.consumer.subscribe([self.raw_topic])

                self.producer = Producer({
                    "bootstrap.servers": self.brokers,
                    "client.id": "edge-iiot-consumer-publisher",
                    "linger.ms": 2,
                    "acks": 1
                })
                logger.info(f"RedpandaConsumer connected to cluster at {self.brokers}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redpanda at {self.brokers} ({e}). Falling back to InMemoryStreamingBus.")
                self.use_cloud_bus = True
                self.consumer = None
                self.producer = None
        else:
            logger.info("RedpandaConsumer active in Cloud In-Memory Streaming Bus mode.")

        # Load preprocessing pipeline
        self.loader = EdgeIIoTDataLoader()
        self.loader.load_pipeline(checkpoint_dir)

        # Load deployed model
        self.class_names = self.loader.encoder.classes_
        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        input_dim = len(self.loader.selector.selected_features) if self.loader.selector else 48
        self.model = ProposedHybridEdgeIIoTModel(input_dim=input_dim, num_classes=len(self.class_names))
        self.checkpoint_mgr = CheckpointManager(checkpoint_dir)
        self.model, _ = self.checkpoint_mgr.load_checkpoint(self.model, device=str(self.device))
        self.model.eval()

        # Threading controls
        self.is_running = False
        self._stop_event = threading.Event()
        self.consumer_thread: Optional[threading.Thread] = None

        # Live Metrics & Memory Buffers for Instantaneous UI Serving
        self.messages_received = 0
        self.messages_processed = 0
        self.dropped_messages = 0
        self.total_alerts = 0
        self.total_benign = 0
        self.total_threats = 0
        self.total_critical = 0
        self.total_high = 0
        self.total_medium = 0
        self.total_low = 0
        self.attack_distribution: Dict[str, int] = {}
        self.latency_records: List[float] = []
        self.infer_latency_records: List[float] = []
        self.early_exit_count = 0
        self.deep_path_count = 0
        self.last_prediction: Optional[Dict[str, Any]] = None
        self.recent_predictions: deque = deque(maxlen=500)
        self.recent_alerts: deque = deque(maxlen=100)
        self.recent_timestamps: deque = deque(maxlen=300)

        # Warmup forward pass to eliminate first-packet cold-start
        try:
            with torch.no_grad():
                dummy = torch.zeros((1, input_dim), dtype=torch.float32, device=self.device)
                _ = self.model(dummy)
        except Exception:
            pass

    def process_event(self, raw_event_json: str) -> Optional[PredictionResult]:
        """Preprocesses a single event, performs inference, routes via entropy, and creates prediction."""
        t_ingest_start = time.perf_counter()

        try:
            event_dict = json.loads(raw_event_json)
            event = RawNetworkEvent(**event_dict)
        except Exception as e:
            logger.error(f"Failed to parse raw event JSON: {e}")
            self.dropped_messages += 1
            return None

        t_ingest = (time.perf_counter() - t_ingest_start) * 1000.0

        # Preprocess features
        t_prep_start = time.perf_counter()
        df_single = pd.DataFrame([event.features])

        try:
            df_clean = self.loader.cleaner.transform(df_single)
            df_scaled = self.loader.scaler.transform(df_clean)
            X_tensor = self.loader.selector.transform(df_scaled)
        except Exception as e:
            logger.error(f"Preprocessing failed for event {event.event_id}: {e}")
            self.dropped_messages += 1
            return None

        t_prep = (time.perf_counter() - t_prep_start) * 1000.0

        # PyTorch Neural Inference
        t_infer_start = time.perf_counter()
        inp = torch.tensor(X_tensor, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            out = self.model(inp, custom_threshold=0.35)

        t_infer = (time.perf_counter() - t_infer_start) * 1000.0

        probs_np = out["probabilities"][0].cpu().numpy()
        pred_idx = int(out["logits"].argmax(dim=-1)[0].item())
        confidence = float(np.max(probs_np))
        entropy_val = float(out["entropy"][0].item())
        path_selected = out["path"][0]  # 'FAST' or 'DEEP'
        predicted_attack = self.class_names[pred_idx]

        probabilities_dict = {c: float(probs_np[i]) for i, c in enumerate(self.class_names)}

        from preprocessing.taxonomy import decode_hierarchical_prediction
        hier = decode_hierarchical_prediction(
            class_probs=probs_np,
            class_names=self.class_names,
            entropy_val=entropy_val,
            route=path_selected,
            calibrated=True,
            model_version="v1.0"
        )

        if path_selected == "FAST":
            self.early_exit_count += 1
        else:
            self.deep_path_count += 1

        # Risk scoring
        is_attack = (predicted_attack.lower() != "normal")
        risk_score = float(confidence) if is_attack else float(1.0 - confidence)

        if not is_attack:
            risk_level = "NORMAL"
        elif risk_score >= 0.90:
            risk_level = "CRITICAL"
        elif risk_score >= 0.70:
            risk_level = "HIGH"
        elif risk_score >= 0.40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        total_pipe = t_ingest + t_prep + t_infer
        self.latency_records.append(total_pipe)
        self.infer_latency_records.append(t_infer)
        if len(self.latency_records) > 2000:
            self.latency_records.pop(0)
        if len(self.infer_latency_records) > 2000:
            self.infer_latency_records.pop(0)

        # Build PredictionResult
        prediction = PredictionResult(
            event_id=event.event_id,
            timestamp=event.timestamp,
            sequence_id=event.sequence_id,
            predicted_attack=predicted_attack,
            confidence=round(confidence, 4),
            entropy=round(entropy_val, 4),
            path_selected=path_selected,
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            latency_breakdown=LatencyBreakdown(
                ingestion_ms=round(t_ingest, 3),
                preprocessing_ms=round(t_prep, 3),
                inference_ms=round(t_infer, 3),
                total_pipeline_ms=round(total_pipe, 3)
            ),
            dataset_label=event.dataset_label,
            probabilities=probabilities_dict,
            binary_prediction=hier.get("binary_prediction", "Attack" if is_attack else "Normal"),
            binary_confidence=round(hier.get("binary_confidence", confidence), 4),
            threat_category=hier.get("threat_category", "Normal"),
            threat_confidence=round(hier.get("threat_confidence", confidence), 4),
            calibrated=True,
            model_version="v1.0",
            uncertain=hier.get("uncertain", False),
            uncertainty_flag=hier.get("uncertainty_flag", "CONFIDENT DETECTION")
        )

        return prediction

    def consume_worker(self):
        """Worker loop polling stream bus and dispatching predictions."""
        mode_str = "Cloud In-Memory Bus" if self.use_cloud_bus else f"Redpanda ({self.brokers})"
        logger.info(f"Consumer worker loop started [{mode_str}]. Group: {self.group_id}")

        while not self._stop_event.is_set():
            if self.use_cloud_bus:
                msg = get_streaming_bus().poll(self.raw_topic, timeout=0.2)
                if msg is None:
                    continue
            else:
                try:
                    msg = self.consumer.poll(timeout=0.5)
                    if msg is None:
                        continue
                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            continue
                        else:
                            logger.warning(f"Consumer error: {msg.error()}")
                            continue
                except Exception as e:
                    time.sleep(0.1)
                    continue

            self.messages_received += 1
            raw_payload = msg.value().decode("utf-8")
            prediction = self.process_event(raw_payload)

            if prediction is not None:
                self.messages_processed += 1
                pred_dict = prediction.model_dump()
                self.last_prediction = pred_dict
                self.recent_predictions.append(pred_dict)
                self.recent_timestamps.append(time.perf_counter())

                # Cumulative classification counters
                is_attack = (prediction.predicted_attack.lower() != "normal")
                if is_attack:
                    self.total_threats += 1
                else:
                    self.total_benign += 1

                if prediction.risk_level == "CRITICAL":
                    self.total_critical += 1
                elif prediction.risk_level == "HIGH":
                    self.total_high += 1
                elif prediction.risk_level == "MEDIUM":
                    self.total_medium += 1
                elif prediction.risk_level == "LOW":
                    self.total_low += 1

                att = prediction.predicted_attack
                self.attack_distribution[att] = self.attack_distribution.get(att, 0) + 1

                try:
                    from dashboard.live_evaluator import LiveEvaluatorEngine
                    LiveEvaluatorEngine.get_instance().ingest_prediction(pred_dict)
                except Exception:
                    pass

                pred_json = prediction.model_dump_json()

                # Publish to ids-predictions
                if self.use_cloud_bus:
                    get_streaming_bus().publish(
                        topic=self.prediction_topic,
                        value=pred_json.encode("utf-8"),
                        key=prediction.event_id
                    )
                else:
                    if self.producer:
                        try:
                            self.producer.produce(
                                topic=self.prediction_topic,
                                value=pred_json.encode("utf-8"),
                                key=prediction.event_id
                            )
                        except BufferError:
                            self.producer.poll(0.05)
                            self.producer.produce(
                                topic=self.prediction_topic,
                                value=pred_json.encode("utf-8"),
                                key=prediction.event_id
                            )

                # If attack detected with high risk, publish to ids-alerts
                if prediction.risk_level in ["HIGH", "CRITICAL"]:
                    self.total_alerts += 1
                    alert = SecurityAlert(
                        event_id=prediction.event_id,
                        timestamp=prediction.timestamp,
                        severity=prediction.risk_level,
                        attack_type=prediction.predicted_attack,
                        confidence=prediction.confidence,
                        entropy=prediction.entropy,
                        description=(
                            f"Block IP & quarantine port immediately. Threat detected as {prediction.predicted_attack} "
                            f"with {prediction.confidence*100:.1f}% confidence (Entropy: {prediction.entropy:.3f})."
                        )
                    )
                    alert_dict = alert.model_dump()
                    self.recent_alerts.append(alert_dict)

                    if self.use_cloud_bus:
                        get_streaming_bus().publish(
                            topic=self.alert_topic,
                            value=alert.model_dump_json().encode("utf-8"),
                            key=alert.alert_id
                        )
                    else:
                        if self.producer:
                            try:
                                self.producer.produce(
                                    topic=self.alert_topic,
                                    value=alert.model_dump_json().encode("utf-8"),
                                    key=alert.alert_id
                                )
                            except BufferError:
                                self.producer.poll(0.05)
                                self.producer.produce(
                                    topic=self.alert_topic,
                                    value=alert.model_dump_json().encode("utf-8"),
                                    key=alert.alert_id
                                )

        logger.info(f"Consumer worker loop stopped. Processed: {self.messages_processed}")

    def start(self):
        """Starts background consumption thread."""
        if self.is_running:
            return
        self.is_running = True
        self._stop_event.clear()
        self.consumer_thread = threading.Thread(target=self.consume_worker, daemon=True)
        self.consumer_thread.start()

    def stop(self):
        """Stops background consumption thread and cleans up handles."""
        self.is_running = False
        self._stop_event.set()
        if not self.use_cloud_bus and self.consumer:
            try:
                self.consumer.close()
            except Exception:
                pass
        if self.consumer_thread and self.consumer_thread.is_alive():
            self.consumer_thread.join(timeout=3)

    def get_throughput(self) -> float:
        """Calculates instantaneous messages/sec processed over recent time window."""
        if not self.is_running or len(self.recent_timestamps) < 2:
            return 0.0
        now = time.perf_counter()
        while self.recent_timestamps and (now - self.recent_timestamps[0]) > 4.0:
            self.recent_timestamps.popleft()
        if len(self.recent_timestamps) < 2:
            return 0.0
        dt = now - self.recent_timestamps[0]
        return round(float(len(self.recent_timestamps) / max(0.05, dt)), 1)

    def get_consumer_lag(self) -> int:
        """Calculates real-time consumer lag on the raw topic partition."""
        if self.use_cloud_bus:
            return get_streaming_bus().get_lag(self.raw_topic)

        if not HAS_CONFLUENT_KAFKA or not self.consumer:
            return 0

        try:
            tp = TopicPartition(self.raw_topic, 0)
            low, high = self.consumer.get_watermark_offsets(tp, timeout=0.1)
            pos = self.consumer.position([tp])
            if pos and pos[0].offset >= 0 and high >= 0:
                return max(0, high - pos[0].offset)
            elif high >= 0 and self.messages_processed > 0:
                return max(0, high - self.messages_processed)
        except Exception:
            pass
        return 0

    def reset_stats(self):
        """Resets all live statistics and recent event buffers."""
        self.messages_received = 0
        self.messages_processed = 0
        self.dropped_messages = 0
        self.total_alerts = 0
        self.total_benign = 0
        self.total_threats = 0
        self.total_critical = 0
        self.total_high = 0
        self.total_medium = 0
        self.total_low = 0
        self.attack_distribution.clear()
        self.latency_records.clear()
        self.infer_latency_records.clear()
        self.recent_predictions.clear()
        self.recent_alerts.clear()
        self.recent_timestamps.clear()
        self.early_exit_count = 0
        self.deep_path_count = 0
        self.last_prediction = None
        if self.use_cloud_bus:
            get_streaming_bus().clear()

    def get_stats(self) -> Dict[str, Any]:
        """Calculates real-time P50, P95, P99 latencies, throughput, and detection counts."""
        pipe_pool = self.latency_records[3:] if len(self.latency_records) > 8 else self.latency_records
        infer_pool = self.infer_latency_records[3:] if len(self.infer_latency_records) > 8 else self.infer_latency_records

        lats = pipe_pool if pipe_pool else [0.0]
        inf_lats = infer_pool if infer_pool else [0.0]

        p50 = float(np.percentile(lats, 50))
        p95 = float(np.percentile(lats, 95))
        p99 = float(np.percentile(lats, 99))
        p50_infer = float(np.percentile(inf_lats, 50))
        p99_infer = float(np.percentile(inf_lats, 99))
        avg_lat = float(np.mean(lats))

        total_decisions = max(1, self.early_exit_count + self.deep_path_count)
        early_pct = (self.early_exit_count / total_decisions) * 100.0

        throughput = self.get_throughput()
        lag = self.get_consumer_lag()
        mode_label = "In-Memory Bus (Cloud)" if self.use_cloud_bus else f"Redpanda ({self.brokers})"

        return {
            "status": "RUNNING" if self.is_running else "STOPPED",
            "messages_received": self.messages_received,
            "messages_processed": self.messages_processed,
            "dropped_messages": self.dropped_messages,
            "total_alerts": self.total_alerts,
            "total_benign": self.total_benign,
            "total_threats": self.total_threats,
            "total_critical": self.total_critical,
            "total_high": self.total_high,
            "total_medium": self.total_medium,
            "total_low": self.total_low,
            "attack_distribution": dict(self.attack_distribution),
            "messages_sec": throughput,
            "consumer_lag": lag,
            "p50_latency_ms": round(p50, 2),
            "p95_latency_ms": round(p95, 2),
            "p99_latency_ms": round(p99, 2),
            "p50_infer_ms": round(p50_infer, 3),
            "p99_infer_ms": round(p99_infer, 3),
            "avg_latency_ms": round(avg_lat, 2),
            "early_exit_percentage": round(early_pct, 1),
            "deep_path_percentage": round(100.0 - early_pct, 1),
            "backend_mode": "cloud" if self.use_cloud_bus else "redpanda",
            "backend_label": mode_label,
            "last_prediction": self.last_prediction
        }


if __name__ == "__main__":
    consumer = EdgeIIoTStreamingConsumer()
    consumer.start()

    try:
        while True:
            time.sleep(2)
            stats = consumer.get_stats()
            print(
                f"Consumer [{stats['backend_label']}]: Recv={stats['messages_received']} | Proc={stats['messages_processed']} | "
                f"P99={stats['p99_latency_ms']}ms | EarlyExit={stats['early_exit_percentage']}%"
            )
    except KeyboardInterrupt:
        consumer.stop()
