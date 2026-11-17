"""Metric Learning Loss Functions for Few-Shot Adaptation.

Implements Supervised Contrastive Loss (SupCon) and Hard-Negative Triplet Loss
specifically designed for few-shot adaptation on personal objects.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss (SupCon).

    Reference:
        Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.

    Pulls all views of the same object together in spherical embedding space
    while pushing embeddings from distinct objects apart.
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculate SupCon loss.

        Args:
            features: L2-normalized embeddings of shape (batch_size, embed_dim).
            labels: 1D tensor of class integer IDs of shape (batch_size,).

        Returns:
            Scalar loss tensor.
        """
        device = features.device
        batch_size = features.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Dot product similarity matrix (features are already L2-normalized)
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # Subtract max for numerical stability
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - sim_max.detach()

        # Mask out self-contrast (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        # Compute log-probabilities: log(exp(sim) / sum_k(exp(sim_k)))
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Mean log-probability over all positive views for each anchor
        pos_per_anchor = mask.sum(1)
        # Avoid division by zero for anchors with no other positive samples
        valid_anchors = pos_per_anchor > 0

        if not valid_anchors.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_anchors] / pos_per_anchor[valid_anchors]

        # Loss scaled by temperature ratio
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos.mean()
        return loss


class HardNegativeTripletLoss(nn.Module):
    """Triplet Margin Loss with online hard-negative mining.

    Especially useful when the gallery contains visually confusable items
    (e.g., two similar books or multiple sets of keys), guaranteeing an
    explicit cosine margin between the true object and its closest competitor.
    """

    def __init__(self, margin: float = 0.15) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss with hard positive and hard negative mining.

        Args:
            features: L2-normalized features (N, D).
            labels: Object class IDs (N,).
        """
        device = features.device
        N = features.size(0)
        if N < 3:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Cosine distance = 1 - dot product (for unit vectors)
        cos_sim = torch.matmul(features, features.T)
        cos_dist = 1.0 - cos_sim

        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_diff = ~labels_eq

        # Hardest positive for each anchor (largest distance among positives, excluding self)
        pos_dist = cos_dist.clone()
        pos_dist[~labels_eq] = -1e9
        pos_dist.fill_diagonal_(-1e9)
        hardest_pos, _ = pos_dist.max(dim=1)

        # Hardest negative for each anchor (smallest distance among negatives)
        neg_dist = cos_dist.clone()
        neg_dist[~labels_diff] = 1e9
        hardest_neg, _ = neg_dist.min(dim=1)

        # Only count anchors that have at least one valid positive and one valid negative
        valid_pos = (labels_eq.sum(dim=1) > 1)
        valid_neg = (labels_diff.sum(dim=1) > 0)
        valid_mask = valid_pos & valid_neg

        if not valid_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = F.relu(hardest_pos[valid_mask] - hardest_neg[valid_mask] + self.margin)
        return loss.mean()


class MetricAdaptationLoss(nn.Module):
    """Combined SupCon + Hard-Negative Triplet loss for fine-grained separation."""

    def __init__(self, temperature: float = 0.07, triplet_margin: float = 0.15, triplet_weight: float = 0.5) -> None:
        super().__init__()
        self.supcon = SupervisedContrastiveLoss(temperature=temperature)
        self.triplet = HardNegativeTripletLoss(margin=triplet_margin)
        self.triplet_weight = triplet_weight

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss_sc = self.supcon(features, labels)
        if self.triplet_weight > 0:
            loss_tr = self.triplet(features, labels)
            return loss_sc + self.triplet_weight * loss_tr
        return loss_sc
