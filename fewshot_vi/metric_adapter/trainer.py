"""Trainer for Few-Shot Metric Adaptation.

Adapts the projection head using Supervised Contrastive Loss and Triplet Loss
over the registered object embeddings in ~1 second on consumer CPU hardware.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.optim as optim

from .adapter import FewShotMetricAdapter
from .loss import MetricAdaptationLoss, SupervisedContrastiveLoss

log = logging.getLogger(__name__)


@dataclass
class AdaptationMetrics:
    """Metrics quantifying the few-shot metric adaptation gain."""
    num_classes: int
    num_samples: int
    initial_loss: float
    final_loss: float
    intra_class_sim_before: float
    intra_class_sim_after: float
    inter_class_sim_before: float
    inter_class_sim_after: float
    margin_before: float
    margin_after: float
    margin_gain: float
    training_time_s: float

    def summary(self) -> str:
        return (
            f"Few-Shot Adaptation ({self.num_classes} classes, {self.num_samples} samples in {self.training_time_s:.2f}s): "
            f"Loss: {self.initial_loss:.4f} -> {self.final_loss:.4f} | "
            f"Intra-Sim: {self.intra_class_sim_before:.3f} -> {self.intra_class_sim_after:.3f} | "
            f"Inter-Sim: {self.inter_class_sim_before:.3f} -> {self.inter_class_sim_after:.3f} | "
            f"Margin: {self.margin_before:.3f} -> {self.margin_after:.3f} (+{self.margin_gain:.3f})"
        )


def _compute_cluster_metrics(
    features: np.ndarray, labels: np.ndarray
) -> Tuple[float, float, float]:
    """Compute average intra-class similarity, inter-class similarity, and margin."""
    if len(features) < 2 or len(np.unique(labels)) < 2:
        # Single class case: compute intra-class similarity only
        if len(features) >= 2:
            sims = features @ features.T
            np.fill_diagonal(sims, np.nan)
            intra = float(np.nanmean(sims))
            return intra, 0.0, intra
        return 1.0, 0.0, 1.0

    # L2 normalize
    norms = np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    normed = features / norms
    sim_matrix = normed @ normed.T

    intra_sims: List[float] = []
    inter_sims: List[float] = []

    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        mask = (labels == lbl)
        pos_sims = sim_matrix[mask][:, mask]
        np.fill_diagonal(pos_sims, np.nan)
        valid_pos = pos_sims[~np.isnan(pos_sims)]
        if len(valid_pos) > 0:
            intra_sims.extend(valid_pos.tolist())

        neg_sims = sim_matrix[mask][:, ~mask].flatten()
        if len(neg_sims) > 0:
            inter_sims.extend(neg_sims.tolist())

    intra = float(np.mean(intra_sims)) if intra_sims else 1.0
    inter = float(np.mean(inter_sims)) if inter_sims else 0.0
    margin = intra - inter
    return intra, inter, margin


def _augment_feature_views(
    embeddings: np.ndarray, target_count: int, seed: int = 42
) -> np.ndarray:
    """Augment few-shot feature vectors using mild manifold jitter and interpolation."""
    n, d = embeddings.shape
    if n >= target_count:
        return embeddings

    rng = np.random.default_rng(seed)
    augmented = [embeddings]
    needed = target_count - n

    for _ in range(needed):
        idx1 = rng.integers(0, n)
        idx2 = rng.integers(0, n)
        # Interpolation / mixup between two views of the same object
        alpha = rng.uniform(0.6, 1.0)
        v = alpha * embeddings[idx1] + (1.0 - alpha) * embeddings[idx2]
        # Mild spherical Gaussian noise
        noise = rng.normal(0, 0.015, size=d).astype(np.float32)
        v = v + noise
        v = v / max(float(np.linalg.norm(v)), 1e-8)
        augmented.append(v.reshape(1, d))

    return np.vstack(augmented)


class MetricAdapterTrainer:
    """Trains or adapts a FewShotMetricAdapter on registered personal object embeddings."""

    def __init__(
        self,
        adapter: Optional[FewShotMetricAdapter] = None,
        lr: float = 1.5e-3,
        weight_decay: float = 1e-4,
        temperature: float = 0.07,
        epochs: int = 50,
        device: str = "cpu",
    ) -> None:
        self.adapter = adapter or FewShotMetricAdapter()
        self.lr = lr
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.epochs = epochs
        self.device = device
        self.criterion = MetricAdaptationLoss(temperature=temperature, triplet_margin=0.12)

    def train_on_gallery(
        self,
        E: np.ndarray,
        owner: np.ndarray,
        min_views_per_class: int = 8,
    ) -> AdaptationMetrics:
        """Adapt metric projection head using all active gallery embeddings.

        Args:
            E: Matrix of shape (M, in_dim) containing un-adapted vision embeddings.
            owner: Array of shape (M,) containing class/object IDs.
            min_views_per_class: Target sample count per class via manifold augmentation.

        Returns:
            AdaptationMetrics reporting loss and separation gains.
        """
        start_time = time.time()
        if E.shape[0] == 0:
            return AdaptationMetrics(0, 0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)

        unique_classes = np.unique(owner)
        num_classes = len(unique_classes)
        if num_classes < 2:
            raise ValueError("few-shot metric learning needs at least two registered objects")

        # Baseline metrics before adaptation
        intra_before, inter_before, margin_before = _compute_cluster_metrics(E, owner)

        # Prepare balanced training batch with feature-space augmentations
        all_feats: List[np.ndarray] = []
        all_labels: List[int] = []

        for cid in unique_classes:
            class_feats = E[owner == cid]
            aug_feats = _augment_feature_views(class_feats, min_views_per_class, seed=int(cid))
            all_feats.append(aug_feats)
            all_labels.extend([int(cid)] * len(aug_feats))

        train_X = np.vstack(all_feats).astype(np.float32)
        train_y = np.array(all_labels, dtype=np.int64)

        # Map labels to 0..C-1 for PyTorch loss
        _, indexed_y = np.unique(train_y, return_inverse=True)

        tensor_X = torch.from_numpy(train_X).to(self.device)
        tensor_y = torch.from_numpy(indexed_y).to(self.device)

        # Setup optimizer and scheduler
        self.adapter.train()
        self.adapter.to(self.device)
        optimizer = optim.AdamW(
            self.adapter.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)

        initial_loss = 0.0
        final_loss = 0.0

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            projected = self.adapter(tensor_X)
            loss = self.criterion(projected, tensor_y)

            if epoch == 0:
                initial_loss = float(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            final_loss = float(loss.item())

        self.adapter.eval()
        training_time = time.time() - start_time

        # Compute metrics after adaptation
        adapted_E = self.adapter.transform_numpy(E)
        intra_after, inter_after, margin_after = _compute_cluster_metrics(adapted_E, owner)
        margin_gain = margin_after - margin_before

        metrics = AdaptationMetrics(
            num_classes=num_classes,
            num_samples=int(E.shape[0]),
            initial_loss=initial_loss,
            final_loss=final_loss,
            intra_class_sim_before=intra_before,
            intra_class_sim_after=intra_after,
            inter_class_sim_before=inter_before,
            inter_class_sim_after=inter_after,
            margin_before=margin_before,
            margin_after=margin_after,
            margin_gain=margin_gain,
            training_time_s=training_time,
        )
        log.info(metrics.summary())
        return metrics
