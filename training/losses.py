"""
Compound and Class-Balanced Loss Suite for Multiclass IIoT Intrusion Detection.
Implements:
1. Focal Loss (Lin et al., ICCV 2017)
2. Center Loss (Wen et al., ECCV 2016)
3. Class-Balanced Loss (Cui et al., CVPR 2019)
4. Label Smoothing Cross Entropy
5. Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)
6. Proposed Compound Focal + Center + SupCon Loss
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multiclass Focal Loss for hard-example mining under severe class imbalance."""
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        pt = torch.sum(probs * targets_one_hot, dim=-1)
        pt = torch.clamp(pt, min=1e-8, max=1.0 - 1e-8)
        focal_weight = torch.pow(1.0 - pt, self.gamma)
        log_pt = torch.log(pt)
        loss = -focal_weight * log_pt

        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            loss = loss * self.alpha[targets]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CenterLoss(nn.Module):
    """Center Loss minimizing intra-class cluster dispersion in deep latent space."""
    def __init__(self, num_classes: int = 15, feat_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim) * 0.05)

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size = features.size(0)
        centers_batch = self.centers[targets]
        diff = features - centers_batch
        loss = 0.5 * torch.sum(torch.pow(diff, 2)) / (batch_size * self.feat_dim)
        return loss


class ClassBalancedLoss(nn.Module):
    """Class-Balanced Loss based on effective number of samples (Cui et al., CVPR 2019)."""
    def __init__(self, samples_per_class: torch.Tensor, beta: float = 0.999, gamma: float = 2.0):
        super().__init__()
        effective_num = 1.0 - torch.pow(beta, samples_per_class.float())
        weights = (1.0 - beta) / (effective_num + 1e-8)
        weights = weights / weights.sum() * len(samples_per_class)
        self.register_buffer("weights", weights)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        pt = torch.sum(probs * targets_one_hot, dim=-1).clamp(1e-8, 1.0 - 1e-8)
        focal_weight = torch.pow(1.0 - pt, self.gamma)
        loss = -focal_weight * torch.log(pt) * self.weights[targets]
        return loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with uniform label smoothing."""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_preds = F.log_softmax(logits, dim=-1)
        loss = -log_preds.sum(dim=-1)
        nll = F.nll_loss(log_preds, targets)
        return (1 - self.smoothing) * nll + (self.smoothing / num_classes) * loss.mean()


class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Representation Learning Loss (SupCon)."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        features: normalized embeddings of shape (Batch, Dim)
        targets: ground truth class indices (Batch,)
        """
        norm_feat = F.normalize(features, dim=1)
        similarity = torch.matmul(norm_feat, norm_feat.T) / self.temperature

        batch_size = targets.size(0)
        mask = torch.eq(targets.unsqueeze(1), targets.unsqueeze(0)).float().to(features.device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask

        exp_sim = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        return loss


class ProposedCompoundLoss(nn.Module):
    """
    Proposed Compound Research Loss:
    L_total = 0.5 * FL(fast_logits, y) + 0.5 * FL(deep_logits, y) +
              lambda_center * CenterLoss(latent, y) +
              lambda_supcon * SupConLoss(latent, y)
    """
    def __init__(
        self,
        num_classes: int = 15,
        feat_dim: int = 128,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        lambda_focal: float = 1.0,
        lambda_center: float = 0.01,
        lambda_supcon: float = 0.005
    ):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        self.center_loss = CenterLoss(num_classes=num_classes, feat_dim=feat_dim)
        self.supcon_loss = SupervisedContrastiveLoss(temperature=0.07)
        self.lambda_focal = lambda_focal
        self.lambda_center = lambda_center
        self.lambda_supcon = lambda_supcon

    def forward(
        self,
        fast_logits: torch.Tensor,
        deep_logits: torch.Tensor,
        latent_features: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_fast = self.focal_loss(fast_logits, targets)
        loss_deep = self.focal_loss(deep_logits, targets)
        loss_center = self.center_loss(latent_features, targets)
        loss_supcon = self.supcon_loss(latent_features, targets)

        total_loss = (
            0.5 * loss_fast +
            0.5 * loss_deep +
            self.lambda_center * loss_center +
            self.lambda_supcon * loss_supcon
        )
        return total_loss, 0.5 * (loss_fast + loss_deep), loss_center, loss_supcon
