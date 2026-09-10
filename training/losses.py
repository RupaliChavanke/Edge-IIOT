"""
Compound Loss Functions: Focal Loss + Center Loss for Edge-IIoTset Multiclass IDS.
Tackles extreme class imbalance and minimizes intra-class variance.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multiclass Focal Loss (Lin et al., ICCV 2017):
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (Batch, Num_Classes)
            targets: (Batch,) with integer class labels
        """
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=-1)
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        # Compute pt
        pt = torch.sum(probs * targets_one_hot, dim=-1)
        pt = torch.clamp(pt, min=1e-8, max=1.0 - 1e-8)

        # Compute focal weight
        focal_weight = torch.pow(1.0 - pt, self.gamma)

        # Cross entropy term
        log_pt = torch.log(pt)
        loss = -focal_weight * log_pt

        # Apply class balancing alpha if provided
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha[targets]
            loss = loss * alpha_t

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CenterLoss(nn.Module):
    """
    Center Loss (Wen et al., ECCV 2016):
    L_center = 0.5 * sum(|| f_i - c_{y_i} ||^2)
    Minimizes intra-class variation in deep latent feature space.
    """

    def __init__(self, num_classes: int = 15, feat_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        # Learned class centers
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Latent embeddings of shape (Batch, Feat_Dim)
            targets: Ground truth class indices of shape (Batch,)
        """
        batch_size = features.size(0)
        # Gather centers for target labels: (Batch, Feat_Dim)
        centers_batch = self.centers[targets]
        # Compute mean squared Euclidean distance per dimension
        diff = features - centers_batch
        loss = 0.5 * torch.sum(torch.pow(diff, 2)) / (batch_size * self.feat_dim)
        return loss


class ProposedCompoundLoss(nn.Module):
    """
    Compound Loss:
    L_total = lambda_fast * FL(fast_logits, y) +
              lambda_deep * FL(deep_logits, y) +
              lambda_center * CenterLoss(latent_features, y)
    """

    def __init__(
        self,
        num_classes: int = 15,
        feat_dim: int = 128,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        lambda_focal: float = 1.0,
        lambda_center: float = 0.01
    ):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        self.center_loss = CenterLoss(num_classes=num_classes, feat_dim=feat_dim)
        self.lambda_focal = lambda_focal
        self.lambda_center = lambda_center

    def forward(
        self,
        fast_logits: torch.Tensor,
        deep_logits: torch.Tensor,
        latent_features: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_fast = self.focal_loss(fast_logits, targets)
        loss_deep = self.focal_loss(deep_logits, targets)
        loss_center = self.center_loss(latent_features, targets)

        total_loss = self.lambda_focal * (0.5 * loss_fast + 0.5 * loss_deep) + self.lambda_center * loss_center
        return total_loss, (0.5 * loss_fast + 0.5 * loss_deep), loss_center
