"""
Resident Pretrained Model Streaming Inference Service for Edge-IIoTset IDS.
Loads frozen model artifacts once into memory and continuously consumes from 'edge-iiot-raw',
generating predictions on 'ids-predictions' and high-risk alerts on 'ids-alerts'.
"""

import os
import sys
import time
import json
import uuid
import argparse
import threading
from typing import Optional
from confluent_kafka import Consumer, Producer, KafkaError
import logging

from models.model_manager import ModelManager
from streaming.schemas import (
    RawNetworkEvent,
    PredictionResult,
    SecurityAlert,
    LatencyBreakdown,
    StreamTelemetry
)
from streaming.topics import TopicManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("InferenceService")


class PretrainedInferenceService:
    """Resident inference service consuming from Redpanda using a frozen pretrained model."""

    def __init__(
        self,
        artifacts_dir: str = "artifacts",
        brokers: Optional[str] = None,
        raw_topic: str = "edge-iiot-raw",
        pred_topic: str = "ids-predictions",
        alert_topic: str = "ids-alerts",
        metrics_topic: str = "ids-metrics",
        group_id: str = "iiot-ids-inference"
    ):
        self.artifacts_dir = artifacts_dir
        self.brokers = brokers or os.getenv("REDPANDA_BROKERS", "localhost:19092")
        self.raw_topic = raw_topic
        self.pred_topic = pred_topic
        self.alert_topic = alert_topic
        self.metrics_topic = metrics_topic
        self.group_id = group_id

        # 1. Initialize topics in Redpanda
        TopicManager(self.brokers).create_topics()

        # 2. Load Pretrained Model Once into Memory via ModelManager
        logger.info("Initializing resident ModelManager...")
        self.manager = ModelManager.get_instance(artifacts_dir=artifacts_dir)
        loaded = self.manager.load_artifacts()
        if not loaded:
            raise RuntimeError(f"MODEL ARTIFACT COMPATIBILITY ERROR: Missing artifacts in '{artifacts_dir}'! Run 'python train.py' first.")

        logger.info(f"Pretrained model successfully resident in memory. Device: {self.manager.device}")

        # 3. Setup Redpanda Consumer & Producer
        self.consumer = Consumer({
            "bootstrap.servers": self.brokers,
            "group.id": self.group_id,
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
            "session.timeout.ms": 6000
        })
        self.consumer.subscribe([self.raw_topic])

        self.producer = Producer({
            "bootstrap.servers": self.brokers,
            "client.id": "edge-iiot-inference-daemon",
            "linger.ms": 2,
            "acks": 1
        })

        self.is_running = False
        self._stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None

        # Statistics
        self.events_processed = 0
        self.alerts_generated = 0
        self.dropped_events = 0
        self.latencies = []

    def process_raw_message(self, raw_bytes: bytes) -> Optional[PredictionResult]:
        """Preprocesses incoming event, executes inference, and packages prediction."""
        t_ingest_start = time.perf_counter()
        try:
            event_dict = json.loads(raw_bytes.decode("utf-8"))
            event = RawNetworkEvent(**event_dict)
        except Exception:
            self.dropped_events += 1
            return None

        t_ingest = (time.perf_counter() - t_ingest_start) * 1000.0

        # Execute single prediction through frozen ModelManager (no gradients, no retraining)
        pred_dict = self.manager.predict_single(event.features)

        pred_result = PredictionResult(
            event_id=event.event_id,
            timestamp=event.timestamp,
            sequence_id=event.sequence_id,
            predicted_attack=pred_dict["predicted_attack"],
            confidence=pred_dict["confidence"],
            entropy=pred_dict["entropy"],
            path_selected=pred_dict["path_selected"],
            risk_score=pred_dict["risk_score"],
            risk_level=pred_dict["risk_level"],
            latency_breakdown=LatencyBreakdown(
                ingestion_ms=round(t_ingest, 3),
                preprocessing_ms=pred_dict["latency_breakdown"]["preprocessing_ms"],
                inference_ms=pred_dict["latency_breakdown"]["inference_ms"],
                total_pipeline_ms=round(t_ingest + pred_dict["latency_breakdown"]["total_pipeline_ms"], 3)
            ),
            dataset_label=event.dataset_label,  # Ground truth retained strictly for live evaluation
            probabilities={c: p for c, p in zip(self.manager.class_names, pred_dict.get("probabilities", []))},
            binary_prediction=pred_dict.get("binary_prediction", "Normal"),
            binary_confidence=pred_dict.get("binary_confidence", 1.0),
            threat_category=pred_dict.get("threat_category", "Normal"),
            threat_confidence=pred_dict.get("threat_confidence", 1.0),
            attack_type=pred_dict.get("attack_type", pred_dict["predicted_attack"]),
            attack_confidence=pred_dict.get("attack_confidence", pred_dict["confidence"]),
            route=pred_dict.get("route", pred_dict["path_selected"]),
            calibrated=pred_dict.get("calibrated", True),
            model_version=pred_dict.get("model_version", "v1.0"),
            uncertain=pred_dict.get("uncertain", False),
            uncertainty_flag=pred_dict.get("uncertainty_flag", "NORMAL")
        )
        return pred_result

    def run_loop(self):
        """Continuous event consumption loop."""
        logger.info(f"Inference Service active and polling '{self.raw_topic}' from broker {self.brokers}...")

        while not self._stop_event.is_set():
            msg = self.consumer.poll(timeout=0.5)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    logger.warning(f"Consumer error: {msg.error()}")
                    continue

            pred = self.process_raw_message(msg.value())
            if pred is not None:
                self.events_processed += 1
                self.latencies.append(pred.latency_breakdown.total_pipeline_ms)
                if len(self.latencies) > 1000:
                    self.latencies.pop(0)

                payload = pred.model_dump_json().encode("utf-8")
                # 1. Publish prediction
                self.producer.produce(topic=self.pred_topic, value=payload, key=pred.event_id)

                # 2. Publish high-risk alert if applicable
                if pred.risk_level in ["HIGH", "CRITICAL"]:
                    self.alerts_generated += 1
                    alert = SecurityAlert(
                        event_id=pred.event_id,
                        timestamp=pred.timestamp,
                        severity=pred.risk_level,
                        attack_type=pred.predicted_attack,
                        confidence=pred.confidence,
                        entropy=pred.entropy,
                        description=f"High-confidence threat ({pred.predicted_attack}) routed through {pred.path_selected} path.",
                    )
                    self.producer.produce(topic=self.alert_topic, value=alert.model_dump_json().encode("utf-8"), key=alert.alert_id)

                self.producer.poll(0)

            self.consumer.commit(msg, asynchronous=True)

        self.consumer.close()
        self.producer.flush(timeout=3)
        self.is_running = False
        logger.info("Inference Service stopped.")

    def start(self):
        """Starts background worker thread."""
        if self.is_running:
            return
        self.is_running = True
        self._stop_event.clear()
        self.worker_thread = threading.Thread(target=self.run_loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        """Stops inference service."""
        self.is_running = False
        self._stop_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3)


# Alias for backward compatibility
InferenceService = PretrainedInferenceService


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge-IIoTset Pretrained Model Inference Daemon")
    parser.add_argument("--artifacts", type=str, default="artifacts", help="Artifacts directory")
    args = parser.parse_args()

    service = PretrainedInferenceService(artifacts_dir=args.artifacts)
    service.start()

    try:
        while True:
            time.sleep(3)
            logger.info(f"Status: RUNNING | Processed Events: {service.events_processed} | Alerts: {service.alerts_generated}")
    except KeyboardInterrupt:
        service.stop()
