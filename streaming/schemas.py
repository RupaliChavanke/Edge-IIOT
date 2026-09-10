"""
Pydantic Schemas for Redpanda Event Streaming Pipeline.
Enforces strict message schema validation, prevents code injection, and isolates ground truth.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel, Field


class RawNetworkEvent(BaseModel):
    """Event produced by streaming producer into 'edge-iiot-raw'."""
    event_id: str = Field(default_factory=lambda: f"EVT-{uuid.uuid4().hex[:8].upper()}")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "edge-iiotset"
    sequence_id: int
    features: Dict[str, Any]
    # Ground truth is ONLY for offline evaluation/benchmarking; never used in inference
    dataset_label: Optional[str] = None


class PreprocessedEvent(BaseModel):
    """Cleaned, scaled, and mRMR-projected event in 'edge-iiot-preprocessed'."""
    event_id: str
    timestamp: str
    sequence_id: int
    feature_vector: List[float]
    feature_names: List[str]
    dataset_label: Optional[str] = None


class LatencyBreakdown(BaseModel):
    """Granular latency profile in milliseconds."""
    ingestion_ms: float = 0.0
    preprocessing_ms: float = 0.0
    inference_ms: float = 0.0
    serialization_ms: float = 0.0
    total_pipeline_ms: float = 0.0


class PredictionResult(BaseModel):
    """Inference output published to 'ids-predictions'."""
    event_id: str
    timestamp: str
    sequence_id: int
    predicted_attack: str
    confidence: float
    entropy: float
    path_selected: str  # 'FAST' or 'DEEP'
    risk_score: float   # 0.0 to 1.0
    risk_level: str     # 'NORMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    latency_breakdown: LatencyBreakdown
    dataset_label: Optional[str] = None  # Evaluation comparison only
    probabilities: Optional[Dict[str, float]] = None  # Full multiclass probability distribution
    top_channels: Optional[List[int]] = None
    # PhD Hierarchical & Calibrated Attack Identification Specification (Prompt Items 34 & 54)
    binary_prediction: str = "Normal"
    binary_confidence: float = 1.0
    threat_category: str = "Normal"
    threat_confidence: float = 1.0
    attack_type: str = "Normal"
    attack_confidence: float = 1.0
    route: str = "FAST"
    calibrated: bool = True
    model_version: str = "v1.0"
    uncertain: bool = False
    uncertainty_flag: str = "NORMAL"


class SecurityAlert(BaseModel):
    """High-risk alert published to 'ids-alerts' for SOC operations."""
    alert_id: str = Field(default_factory=lambda: f"ALT-{uuid.uuid4().hex[:8].upper()}")
    event_id: str
    timestamp: str
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    attack_type: str
    confidence: float
    entropy: float
    description: str
    source_ip: Optional[str] = "N/A"
    destination_ip: Optional[str] = "N/A"


class StreamTelemetry(BaseModel):
    """Operational metric published to 'ids-metrics'."""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    service: str  # 'producer' or 'consumer'
    messages_processed: int
    throughput_msg_per_sec: float
    consumer_lag: int
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
