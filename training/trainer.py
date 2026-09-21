"""
Research-Grade PyTorch Training Orchestrator for Proposed Hybrid Edge-IIoT IDS Model.
Implements:
- AdamW optimizer with weight decay
- CosineAnnealingLR with warmup
- Gradient clipping
- Early stopping strictly on Validation Macro-F1
- Post-hoc Temperature Scaling Calibration on Validation split
- Evaluates untouched Test split only once at checkpoint freeze
"""

from typing import Dict, List, Optional, Tuple, Any
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import logging

from models.proposed_model import ProposedHybridEdgeIIoTModel
from training.losses import ProposedCompoundLoss
from training.metrics import calculate_comprehensive_metrics
from evaluation.calibration import TemperatureScaler, compute_calibration_metrics

logger = logging.getLogger("Trainer")


def get_optimal_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class EdgeIIoTTrainer:
    """Trains the proposed hybrid model with Compound Focal + Center + SupCon loss."""

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
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        train_cfg = config.get("training", {})
        self.batch_size = train_cfg.get("batch_size", 128)
        self.epochs = train_cfg.get("epochs", 20)
        self.lr = train_cfg.get("learning_rate", 0.002)
        self.weight_decay = train_cfg.get("weight_decay", 1e-4)
        self.gamma = train_cfg.get("focal_gamma", 2.0)
        self.lambda_focal = train_cfg.get("lambda_focal", 1.0)
        self.lambda_center = train_cfg.get("lambda_center", 0.01)
        self.patience = train_cfg.get("early_stopping_patience", 7)

        self.criterion = ProposedCompoundLoss(
            num_classes=model.num_classes,
            feat_dim=128,
            gamma=self.gamma,
            lambda_focal=self.lambda_focal,
            lambda_center=self.lambda_center,
            lambda_supcon=0.005
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.criterion.center_loss.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-5
        )

        self.history: List[Dict[str, Any]] = []
        self.calibrator = TemperatureScaler().to(self.device)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_focal = 0.0
        total_center = 0.0
        total_supcon = 0.0
        num_batches = 0

        for X_b, y_b in dataloader:
            X_b = X_b.to(self.device)
            y_b = y_b.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(X_b, routing_mode="train")

            loss, focal_val, center_val, supcon_val = self.criterion(
                fast_logits=out["fast_logits"],
                deep_logits=out["deep_logits"],
                latent_features=out["latent_features"],
                targets=y_b
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_focal += focal_val.item()
            total_center += center_val.item()
            total_supcon += supcon_val.item()
            num_batches += 1

        self.scheduler.step()
        n = max(1, num_batches)
        return {
            "loss": total_loss / n,
            "focal_loss": total_focal / n,
            "center_loss": total_center / n,
            "supcon_loss": total_supcon / n
        }

    def evaluate_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_names: List[str],
        routing_mode: str = "dynamic",
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        self.model.eval()
        tensor_X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        loader = DataLoader(TensorDataset(tensor_X, torch.from_numpy(y)), batch_size=256, shuffle=False)

        all_preds = []
        all_probs = []
        all_paths = []
        latencies = []

        with torch.no_grad():
            for xb, _ in loader:
                t0 = time.perf_counter()
                out = self.model(xb, routing_mode=routing_mode, custom_threshold=threshold)
                latencies.append((time.perf_counter() - t0) * 1000.0 / len(xb))

                all_preds.append(out["predictions"].cpu().numpy())
                all_probs.append(out["probabilities"].cpu().numpy())
                all_paths.extend(out["path"])

        preds = np.concatenate(all_preds)
        probs = np.concatenate(all_probs)

        metrics = calculate_comprehensive_metrics(y, preds, probs=probs, class_names=class_names)
        metrics["Early_Exit_Percentage"] = (np.array(all_paths) == "FAST").mean() * 100.0
        metrics["Latency_ms"] = float(np.mean(latencies))
        return metrics

    def train_full(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        X_tr = data_dict["X_train"]
        y_tr = data_dict["y_train"]
        X_val = data_dict["X_val"]
        y_val = data_dict["y_val"]
        X_te = data_dict["X_test"]
        y_te = data_dict["y_test"]
        class_names = data_dict["class_names"]

        train_ds = TensorDataset(torch.from_numpy(X_tr.astype(np.float32)), torch.from_numpy(y_tr).long())
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        logger.info(f"Training on {len(X_tr):,} samples across {self.epochs} epochs on {self.device}...")

        best_val_f1 = -1.0
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            t_epoch_start = time.perf_counter()
            loss_dict = self.train_epoch(train_loader)

            val_metrics = self.evaluate_split(X_val, y_val, class_names=class_names, routing_mode="dynamic")
            val_f1 = val_metrics["F1_Macro"]
            val_acc = val_metrics["Accuracy"]
            t_sec = time.perf_counter() - t_epoch_start

            record = {
                "epoch": epoch,
                "train_loss": loss_dict["loss"],
                "focal_loss": loss_dict["focal_loss"],
                "center_loss": loss_dict["center_loss"],
                "val_accuracy": val_acc,
                "val_f1": val_f1,
                "early_exit_pct": val_metrics["Early_Exit_Percentage"],
                "time_sec": t_sec
            }
            self.history.append(record)

            logger.info(
                f"Epoch {epoch:2d}/{self.epochs:2d} | Loss: {loss_dict['loss']:.4f} | "
                f"Val Acc: {val_acc*100:6.2f}% | Val Macro-F1: {val_f1*100:6.2f}% | "
                f"Early-Exit: {val_metrics['Early_Exit_Percentage']:.1f}% ({t_sec:.1f}s)"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                torch.save(best_state, os.path.join(self.checkpoint_dir, "best_model.pt"))
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch} (best Val Macro-F1: {best_val_f1*100:.2f}%).")
                    break

        # Load best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Post-hoc calibration on validation split
        logger.info("Fitting post-hoc Temperature Calibration on Validation Split...")
        self.model.eval()
        with torch.no_grad():
            v_x = torch.from_numpy(X_val.astype(np.float32)).to(self.device)
            v_out = self.model(v_x, routing_mode="always_deep")
            val_logits = v_out["deep_logits"] if "deep_logits" in v_out else v_out["logits"]
            self.calibrator.fit(val_logits, torch.from_numpy(y_val).long().to(self.device))
            optimal_temp = float(self.calibrator.temperature.item())
            self.model.set_temperature(optimal_temp)
            logger.info(f"Calibrated temperature parameter: T = {optimal_temp:.4f}")

        # Final single evaluation on UNTOUCHED Test Split
        logger.info("Performing final frozen evaluation on held-out Test Split...")
        test_metrics = self.evaluate_split(X_te, y_te, class_names=class_names, routing_mode="dynamic")

        # Save metrics and history
        pd.DataFrame(self.history).to_csv(os.path.join(self.checkpoint_dir, "training_history.csv"), index=False)
        with open(os.path.join(self.checkpoint_dir, "metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)

        return {
            "test_metrics": test_metrics,
            "val_metrics": val_metrics,
            "best_val_f1": best_val_f1,
            "history": self.history,
            "optimal_temperature": optimal_temp
        }
