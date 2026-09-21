"""
Unit Tests for Redpanda Schemas and Health Probing.
"""

import pytest
from streaming.schemas import (
    RawNetworkEvent,
    PredictionResult,
    SecurityAlert,
    LatencyBreakdown
)
from streaming.health import RedpandaHealthChecker


def test_raw_event_schema_serialization():
    event = RawNetworkEvent(
        sequence_id=42,
        features={"tcp.dstport": 80.0, "tcp.flags": 2.0},
        dataset_label="DDoS_HTTP"
    )
    json_str = event.model_dump_json()
    assert "EVT-" in event.event_id
    assert "DDoS_HTTP" in json_str
    assert "features" in json_str


def test_prediction_result_schema():
    pred = PredictionResult(
        event_id="EVT-1234",
        timestamp="2026-09-10T10:00:00Z",
        sequence_id=1,
        predicted_attack="Normal",
        confidence=0.985,
        entropy=0.042,
        path_selected="FAST",
        risk_score=0.015,
        risk_level="NORMAL",
        latency_breakdown=LatencyBreakdown(
            ingestion_ms=0.1,
            preprocessing_ms=0.5,
            inference_ms=1.2,
            total_pipeline_ms=1.8
        )
    )
    assert pred.path_selected == "FAST"
    assert pred.risk_level == "NORMAL"


def test_redpanda_health_checker():
    checker = RedpandaHealthChecker("localhost:19092")
    health = checker.check_health()
    assert "status" in health
    assert health["status"] in ["CONNECTED", "DISCONNECTED", "CLOUD_MODE"]
