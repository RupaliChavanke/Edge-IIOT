"""
Model Checkpointing and Deployment State Manager.
"""

from typing import Dict, Optional, Any
import os
import json
import torch
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Saves, loads, and manages model weights and evaluation metadata."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model_path = os.path.join(checkpoint_dir, "best_model.pt")
        self.metrics_path = os.path.join(checkpoint_dir, "metrics.json")
        self.config_path = os.path.join(checkpoint_dir, "config.json")

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        metrics: Dict[str, Any],
        config: Dict[str, Any],
        epoch: int
    ) -> None:
        """Save PyTorch model state dict, metrics, and metadata."""
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "metrics": metrics,
            "config": config
        }
        torch.save(state, self.model_path)

        # Save readable metrics JSON
        with open(self.metrics_path, "w") as f:
            # Filter out non-serializable elements if any
            clean_metrics = {k: v for k, v in metrics.items() if k != "Confusion_Matrix_Array"}
            json.dump(clean_metrics, f, indent=2)

        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Checkpoint successfully saved at {self.model_path}")

    def load_checkpoint(self, model: torch.nn.Module, device: str = "cpu") -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Load state dict into model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No checkpoint found at {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        logger.info(f"Loaded checkpoint from {self.model_path} (epoch {checkpoint.get('epoch', 0)})")
        return model, checkpoint.get("metrics", {})

    def is_model_deployed(self) -> bool:
        """Returns True if a valid checkpoint exists and is ready for streaming inference."""
        return os.path.exists(self.model_path) and os.path.exists(self.metrics_path)

    def get_deployed_metrics(self) -> Optional[Dict[str, Any]]:
        """Load deployed model test metrics if available, checking artifacts/ first."""
        artifact_metrics = "artifacts/metrics.json"
        if os.path.exists(artifact_metrics):
            with open(artifact_metrics, "r") as f:
                return json.load(f)
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, "r") as f:
                return json.load(f)
        return None
