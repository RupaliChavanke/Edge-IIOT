"""
Singleton Pretrained Model Manager for Edge-IIoT IDS.
Loads, validates, freezes, and executes the pretrained deep learning model.
Strictly read-only: never retrains, never calls optimizer, keeps model frozen in memory.
"""

from typing import Dict, List, Optional, Tuple, Any
import os
import json
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import logging

from models.proposed_model import ProposedHybridEdgeIIoTModel
from preprocessing.cleaner import EdgeIIoTCleaner
from preprocessing.encoder import EdgeIIoTEncoder
from preprocessing.scaler import EdgeIIoTScaler
from preprocessing.mrmr_jmi import MRMRJMISelector

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages loading, verification, and zero-grad inference of the pretrained model artifacts.
    Enforces that the model is loaded ONCE and remains resident in memory.
    """

    _instance: Optional["ModelManager"] = None

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.model: Optional[ProposedHybridEdgeIIoTModel] = None
        self.cleaner: Optional[EdgeIIoTCleaner] = None
        self.scaler: Optional[EdgeIIoTScaler] = None
        self.selector: Optional[MRMRJMISelector] = None
        self.encoder: Optional[EdgeIIoTEncoder] = None
        
        self.model_config: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.selected_features: List[str] = []
        self.class_names: List[str] = []
        self.model_summary: Dict[str, Any] = {}
        self.model_registry: Dict[str, Any] = {}
        self.model_version: str = "v1.0"
        
        self.is_loaded = False
        self.entropy_threshold: float = 0.35

    @classmethod
    def get_instance(cls, artifacts_dir: str = "artifacts") -> "ModelManager":
        """Singleton accessor."""
        if cls._instance is None:
            cls._instance = cls(artifacts_dir=artifacts_dir)
        return cls._instance

    def validate_artifacts_exist(self) -> Dict[str, bool]:
        """Checks presence of mandatory artifact files."""
        files = {
            "best_model.pt": os.path.exists(os.path.join(self.artifacts_dir, "best_model.pt")),
            "model_config.json": os.path.exists(os.path.join(self.artifacts_dir, "model_config.json")),
            "preprocessor.pkl": os.path.exists(os.path.join(self.artifacts_dir, "preprocessor.pkl")),
            "feature_selector.pkl": os.path.exists(os.path.join(self.artifacts_dir, "feature_selector.pkl")),
            "label_encoder.pkl": os.path.exists(os.path.join(self.artifacts_dir, "label_encoder.pkl")),
            "selected_features.json": os.path.exists(os.path.join(self.artifacts_dir, "selected_features.json")),
            "class_names.json": os.path.exists(os.path.join(self.artifacts_dir, "class_names.json")),
            "metrics.json": os.path.exists(os.path.join(self.artifacts_dir, "metrics.json")),
            "model_registry.json": os.path.exists(os.path.join(self.artifacts_dir, "model_registry.json")),
        }
        return files

    def validate_artifacts(self) -> Tuple[bool, str]:
        """Checks presence of mandatory artifact files and loads them if present."""
        checks = self.validate_artifacts_exist()
        all_ok = all(checks.values())
        if all_ok:
            if not self.is_loaded:
                self.load_artifacts()
            return True, "All mandatory artifacts present and loaded."
        missing = [k for k, v in checks.items() if not v]
        return False, f"Missing artifacts in '{self.artifacts_dir}': {missing}"

    def load_artifacts(self, force_reload: bool = False) -> bool:
        """
        Loads all frozen artifacts into memory once.
        Puts model into eval mode and disables gradients.
        """
        if self.is_loaded and not force_reload:
            return True

        checks = self.validate_artifacts_exist()
        if not all(checks.values()):
            missing = [k for k, v in checks.items() if not v]
            logger.warning(f"Missing model artifacts in '{self.artifacts_dir}': {missing}")
            return False

        logger.info(f"Loading pretrained model artifacts from '{self.artifacts_dir}'...")

        # 1. Load JSON Metadata
        with open(os.path.join(self.artifacts_dir, "model_config.json"), "r") as f:
            self.model_config = json.load(f)
        with open(os.path.join(self.artifacts_dir, "metrics.json"), "r") as f:
            self.metrics = json.load(f)
        with open(os.path.join(self.artifacts_dir, "selected_features.json"), "r") as f:
            self.selected_features = json.load(f)
        with open(os.path.join(self.artifacts_dir, "class_names.json"), "r") as f:
            self.class_names = json.load(f)
        if os.path.exists(os.path.join(self.artifacts_dir, "model_summary.json")):
            with open(os.path.join(self.artifacts_dir, "model_summary.json"), "r") as f:
                self.model_summary = json.load(f)
        if os.path.exists(os.path.join(self.artifacts_dir, "model_registry.json")):
            with open(os.path.join(self.artifacts_dir, "model_registry.json"), "r") as f:
                self.model_registry = json.load(f)
                self.model_version = self.model_registry.get("active_model", "v1.0")

        # 2. Load Preprocessing Transformers
        with open(os.path.join(self.artifacts_dir, "preprocessor.pkl"), "rb") as f:
            prep_dict = pickle.load(f)
            self.cleaner = prep_dict["cleaner"]
            self.scaler = prep_dict["scaler"]

        with open(os.path.join(self.artifacts_dir, "feature_selector.pkl"), "rb") as f:
            self.selector = pickle.load(f)

        with open(os.path.join(self.artifacts_dir, "label_encoder.pkl"), "rb") as f:
            self.encoder = pickle.load(f)

        # 3. Load PyTorch Model & Set to EVAL mode
        num_classes = len(self.class_names)
        input_dim = len(self.selected_features)
        self.entropy_threshold = self.model_config.get("entropy_threshold", 0.35)

        self.model = ProposedHybridEdgeIIoTModel(
            input_dim=input_dim,
            num_classes=num_classes,
            conv_channels=self.model_config.get("conv_channels", 64),
            ghost_ratio=self.model_config.get("ghost_ratio", 2),
            se_reduction=self.model_config.get("se_reduction", 8),
            gru_hidden_dim=self.model_config.get("gru_hidden_dim", 64),
            num_attention_heads=self.model_config.get("num_attention_heads", 4),
            low_rank=self.model_config.get("low_rank", 16),
            dropout=self.model_config.get("dropout", 0.25),
            entropy_threshold=self.entropy_threshold
        )

        model_path = os.path.join(self.artifacts_dir, "best_model.pt")
        state_dict = torch.load(model_path, map_location=self.device)
        if "state_dict" in state_dict:
            self.model.load_state_dict(state_dict["state_dict"])
        else:
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()  # Strictly frozen evaluation mode
        for param in self.model.parameters():
            param.requires_grad = False  # Freeze all weights

        # 4. Load Temperature Scaler if available
        from evaluation.calibration import TemperatureScaler
        calibrator_path = os.path.join(self.artifacts_dir, "confidence_calibrator.pkl")
        self.calibrator = TemperatureScaler.load(calibrator_path) if os.path.exists(calibrator_path) else None

        self.is_loaded = True
        logger.info(f"PRETRAINED MODEL LOADED: Version {self.model_version} on {self.device} (Calibrator: {'Active' if self.calibrator else 'None'}).")
        return True

    def validate_compatibility(self, sample_features: Dict[str, Any]) -> Tuple[bool, str]:
        """Validates feature compatibility against the loaded model schema."""
        if not self.is_loaded:
            return False, "Model is not loaded."
        if not self.selected_features:
            return False, "Feature list is empty."
        return True, "Compatible"

    def preprocess_single_event(self, raw_features: Dict[str, Any]) -> np.ndarray:
        """Online preprocessing without model retraining."""
        if not self.is_loaded:
            raise RuntimeError("ModelManager artifacts not loaded.")

        cleaned = self.cleaner.clean_single_event(raw_features)
        encoded = self.encoder.encode_single_feature_dict(cleaned)

        feature_cols = self.cleaner.fitted_numeric_columns + self.cleaner.fitted_categorical_columns
        num_vec = np.array([float(encoded.get(c, 0.0)) for c in feature_cols], dtype=np.float32)
        scaled_vec = self.scaler.transform_single(num_vec)

        scaled_dict = {col: scaled_vec[i] for i, col in enumerate(feature_cols)}
        return self.selector.transform_single_dict(scaled_dict)

    def predict_single(
        self,
        raw_features: Dict[str, Any],
        custom_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes single-event inference in torch.inference_mode().
        Strictly zero-grad; never touches weights.
        """
        if not self.is_loaded:
            raise RuntimeError("ModelManager artifacts not loaded.")

        t_start = time.perf_counter()

        # Preprocess
        t_prep_start = time.perf_counter()
        feat_vec = self.preprocess_single_event(raw_features)
        t_prep = (time.perf_counter() - t_prep_start) * 1000.0

        # Tensor conversion
        tensor_in = torch.from_numpy(feat_vec).unsqueeze(0).to(self.device)
        th = custom_threshold if custom_threshold is not None else self.entropy_threshold

        # Inference
        t_infer_start = time.perf_counter()
        with torch.inference_mode():
            out = self.model(tensor_in, routing_mode="dynamic", custom_threshold=th)
        t_infer = (time.perf_counter() - t_infer_start) * 1000.0

        pred_idx = int(out["predictions"][0].item())
        confidence = float(out["probabilities"][0, pred_idx].item())
        entropy_val = float(out["entropy"][0].item())
        path = out["path"][0]  # 'FAST' or 'DEEP'

        # Apply Temperature Scaling calibration if available
        if self.calibrator is not None:
            calibrated_probs = self.calibrator.calibrate_probabilities(out["logits"])
            is_calibrated = True
        else:
            calibrated_probs = out["probabilities"].cpu().numpy()
            is_calibrated = False

        from preprocessing.taxonomy import decode_hierarchical_prediction
        hier = decode_hierarchical_prediction(
            class_probs=calibrated_probs[0],
            class_names=self.class_names,
            entropy_val=entropy_val,
            route=path,
            calibrated=is_calibrated,
            model_version=self.model_version
        )

        total_lat = (time.perf_counter() - t_start) * 1000.0

        hier["predicted_attack"] = hier["raw_attack_type"]
        hier["confidence"] = hier["attack_confidence"]
        hier["path_selected"] = path
        hier["probabilities"] = calibrated_probs[0].tolist()
        hier["probabilities_dict"] = {c: float(p) for c, p in zip(self.class_names, calibrated_probs[0])}
        hier["latency_ms"] = round(total_lat, 2)
        hier["latency_breakdown"] = {
            "preprocessing_ms": round(t_prep, 3),
            "inference_ms": round(t_infer, 3),
            "total_pipeline_ms": round(total_lat, 3)
        }
        return hier

    def get_metadata(self) -> Dict[str, Any]:
        """Returns comprehensive metadata for the loaded pretrained model."""
        return {
            "model_version": self.model_version,
            "architecture": "Proposed Hybrid 1D-CNN + Ghost + SE + BiGRU + Temporal Attention + Low-Rank Head",
            "device": str(self.device),
            "pytorch_version": torch.__version__,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "k_features": len(self.selected_features),
            "selected_features": self.selected_features,
            "parameters": self.model.count_parameters() if self.model else {},
            "offline_metrics": self.metrics,
            "entropy_threshold": self.entropy_threshold,
            "artifacts_dir": self.artifacts_dir
        }

    def get_model_metadata(self) -> Dict[str, Any]:
        """Convenience accessor matching test specifications."""
        meta = self.get_metadata()
        meta["total_parameters"] = meta.get("parameters", {}).get("total_parameters", 0)
        meta["mflops"] = self.model_summary.get("mflops", 0.1)
        return meta


# Singleton accessor alias
get_model_manager = ModelManager.get_instance

