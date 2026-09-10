"""
PyTorch Training Orchestrator for Proposed Hybrid Model.
Includes validation, entropy threshold calibration, and test set evaluation.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import logging

from models.proposed_model import ProposedHybridEdgeIIoTModel
from training.losses import ProposedCompoundLoss
from training.metrics import calculate_comprehensive_metrics
from training.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


def get_optimal_device() -> torch.device:
    """Selects MPS (Apple Silicon GPU), CUDA, or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class EdgeIIoTTrainer:
    """Trains the proposed hybrid model with Focal + Center Loss and early-exit routing."""

    def __init__(
        self,
        model: ProposedHybridEdgeIIoTModel,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints"
    ):
        self.config = config
        self.device = device or get_optimal_device()
        self.model = model.to(self.device)
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

        # Hyperparameters
        train_cfg = config.get("training", {})
        self.batch_size = train_cfg.get("batch_size", 64)
        self.epochs = train_cfg.get("epochs", 15)
        self.lr = train_cfg.get("learning_rate", 0.001)
        self.weight_decay = train_cfg.get("weight_decay", 1e-4)
        self.gamma = train_cfg.get("focal_gamma", 2.0)
        self.lambda_focal = train_cfg.get("lambda_focal", 1.0)
        self.lambda_center = train_cfg.get("lambda_center", 0.01)
        self.patience = train_cfg.get("early_stopping_patience", 5)

        # Loss and Optimizer
        self.criterion = ProposedCompoundLoss(
            num_classes=model.num_classes,
            feat_dim=128,
            gamma=self.gamma,
            lambda_focal=self.lambda_focal,
            lambda_center=self.lambda_center
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.criterion.center_loss.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2
        )

        self.history: List[Dict[str, Any]] = []

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Runs one epoch of training."""
        self.model.train()
        total_loss = 0.0
        total_focal = 0.0
        total_center = 0.0
        num_batches = 0

        for X_b, y_b in dataloader:
            X_b = X_b.to(self.device)
            y_b = y_b.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(X_b, routing_mode="train")

            loss, focal_val, center_val = self.criterion(
                fast_logits=out["fast_logits"],
                deep_logits=out["deep_logits"],
                latent_features=out["latent_features"],
                targets=y_b
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_focal += focal_val.item()
            total_center += center_val.item()
            num_batches += 1

        return {
            "loss": total_loss / max(1, num_batches),
            "focal_loss": total_focal / max(1, num_batches),
            "center_loss": total_center / max(1, num_batches)
        }

    def evaluate(
        self,
        dataloader: DataLoader,
        class_names: List[str],
        routing_mode: str = "dynamic",
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Evaluates model performance on validation or test set."""
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        all_paths = []
        latencies = []

        with torch.no_grad():
            for X_b, y_b in dataloader:
                X_b = X_b.to(self.device)
                out = self.model(X_b, routing_mode=routing_mode, custom_threshold=threshold)

                all_preds.append(out["predictions"].cpu().numpy())
                all_targets.append(y_b.numpy())
                all_probs.append(out["probabilities"].cpu().numpy())
                all_paths.extend(out["path"])
                latencies.append(out["latency_ms"])

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        y_prob = np.concatenate(all_probs, axis=0)

        metrics = calculate_comprehensive_metrics(y_true, y_pred, y_prob, class_names=class_names)
        early_exit_pct = (sum(1 for p in all_paths if p == "FAST") / max(1, len(all_paths))) * 100.0

        metrics["Early_Exit_Percentage"] = early_exit_pct
        metrics["Deep_Path_Percentage"] = 100.0 - early_exit_pct
        metrics["Average_Batch_Latency_ms"] = float(np.mean(latencies))
        metrics["P50_Latency_ms"] = float(np.percentile(latencies, 50))
        metrics["P95_Latency_ms"] = float(np.percentile(latencies, 95))
        metrics["P99_Latency_ms"] = float(np.percentile(latencies, 99))
        return metrics

    def train_full(
        self,
        data_dict: Dict[str, Any],
        progress_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """Full training procedure across epochs with early stopping."""
        train_ds = TensorDataset(torch.from_numpy(data_dict["X_train"]).float(), torch.from_numpy(data_dict["y_train"]).long())
        val_ds = TensorDataset(torch.from_numpy(data_dict["X_val"]).float(), torch.from_numpy(data_dict["y_val"]).long())
        test_ds = TensorDataset(torch.from_numpy(data_dict["X_test"]).float(), torch.from_numpy(data_dict["y_test"]).long())

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        class_names = data_dict["class_names"]
        best_f1 = -1.0
        best_epoch = 0
        patience_counter = 0

        logger.info(f"Starting training for {self.epochs} epochs on device: {self.device}...")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_stats = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader, class_names=class_names, routing_mode="dynamic")

            epoch_time = time.time() - t0
            val_f1 = val_metrics["F1_Macro"]
            self.scheduler.step(val_f1)

            history_entry = {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "focal_loss": train_stats["focal_loss"],
                "center_loss": train_stats["center_loss"],
                "val_accuracy": val_metrics["Accuracy"],
                "val_f1": val_f1,
                "early_exit_pct": val_metrics["Early_Exit_Percentage"],
                "time_sec": epoch_time
            }
            self.history.append(history_entry)

            if progress_callback:
                progress_callback(epoch, history_entry)

            logger.info(
                f"Epoch {epoch}/{self.epochs} | Loss: {train_stats['loss']:.4f} | "
                f"Val Acc: {val_metrics['Accuracy']*100:.2f}% | Val F1: {val_f1*100:.2f}% | "
                f"Early Exit: {val_metrics['Early_Exit_Percentage']:.1f}% | Time: {epoch_time:.1f}s"
            )

            # Checkpoint on best validation macro-F1
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    metrics=val_metrics,
                    config=self.config,
                    epoch=epoch
                )
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with F1={best_f1:.4f}.")
                    break

        # Load best model for final unseen test set evaluation
        logger.info("Evaluating best model on unseen test set...")
        self.model, _ = self.checkpoint_manager.load_checkpoint(self.model, device=str(self.device))
        test_metrics = self.evaluate(test_loader, class_names=class_names, routing_mode="dynamic")

        # Save final test evaluation results
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            metrics=test_metrics,
            config=self.config,
            epoch=best_epoch
        )

        return {
            "test_metrics": test_metrics,
            "best_epoch": best_epoch,
            "best_val_f1": best_f1,
            "history": self.history
        }
