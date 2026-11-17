"""Lightweight Few-Shot Metric Learning Projection Head / Adapter.

Adapts frozen vision backbone embeddings (e.g. DINOv2 384-d vectors) into a metric
subspace where multiple views of the same personal object cluster tightly and distinct
personal objects (including fine-grained confusers) are separated by an angular margin.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class FewShotMetricAdapter(nn.Module):
    """Parametric projection head for few-shot metric adaptation.

    Maps frozen vision embeddings (D_in) -> adapted metric space (D_out)
    with L2 normalization output for cosine distance matching.
    """

    def __init__(
        self,
        in_dim: int = 384,
        hidden_dim: int = 384,
        out_dim: int = 256,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Projection block with normalization and non-linear activation
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)

        # Residual shortcut: if in_dim == out_dim, identity; else linear projection
        if in_dim == out_dim:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Linear(in_dim, out_dim, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize with small weights for gentle residual adaptation."""
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="linear")
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)
        if isinstance(self.shortcut, nn.Linear):
            nn.init.orthogonal_(self.shortcut.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass projecting and L2-normalizing the adapted vectors.

        Args:
            x: Tensor of shape (N, in_dim) or (in_dim,).

        Returns:
            L2-normalized tensor of shape (N, out_dim) or (out_dim,).
        """
        orig_dim = x.dim()
        if orig_dim == 1:
            x = x.unsqueeze(0)

        # Residual projection
        residual = self.shortcut(x)
        h = self.fc1(x)
        h = self.ln1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.ln2(h)

        out = residual + h
        # Project onto unit hypersphere for cosine similarity
        out = F.normalize(out, p=2, dim=-1, eps=1e-8)

        if orig_dim == 1:
            out = out.squeeze(0)
        return out

    def transform_numpy(self, arr: np.ndarray) -> np.ndarray:
        """Convenience method to adapt NumPy embeddings directly."""
        if arr is None or arr.size == 0:
            return np.zeros((0, self.out_dim), dtype=np.float32)

        is_1d = arr.ndim == 1
        tensor = torch.from_numpy(np.atleast_2d(arr).astype(np.float32))
        self.eval()
        with torch.inference_mode():
            adapted = self.forward(tensor).cpu().numpy()
        return adapted[0] if is_1d else adapted

    def save(self, path: Union[str, Path]) -> None:
        """Save adapter weights to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        try:
            torch.save(
                {
                    "state_dict": self.state_dict(),
                    "in_dim": self.in_dim,
                    "hidden_dim": self.hidden_dim,
                    "out_dim": self.out_dim,
                },
                str(temp_path),
            )
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)
        log.info("Saved FewShotMetricAdapter checkpoint to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cpu") -> FewShotMetricAdapter:
        """Load adapter from checkpoint."""
        checkpoint = torch.load(str(path), map_location=device, weights_only=True)
        model = cls(
            in_dim=checkpoint.get("in_dim", 384),
            hidden_dim=checkpoint.get("hidden_dim", 384),
            out_dim=checkpoint.get("out_dim", 256),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
