"""A tiny NON-LEARNED colour-histogram embedder.

This is NOT DINOv2 and is far weaker. It exists only so the evaluation and
benchmark plumbing can be exercised on machines without PyTorch/model weights
(and in tests). Every report produced with it is labelled with its name.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

BINS = (8, 4, 4)
DIM = BINS[0] * BINS[1] * BINS[2]
GRID = 3


def _hist(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, list(BINS), [0, 180, 0, 256, 0, 256]).ravel().astype(np.float32)
    n = float(np.linalg.norm(h))
    return h / n if n > 1e-8 else h


class ColorHistEmbedder:
    name = "color-hist (non-learned baseline, NOT DINOv2)"
    dim = DIM

    def embed(self, crops: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """CLS = global HSV histogram; patch tokens = HSV histograms of a GRIDxGRID tiling."""
        cls, pat = [], []
        for c in crops:
            if c is None or c.size == 0:
                raise ValueError("empty crop passed to embedder")
            img = cv2.resize(c, (48, 48), interpolation=cv2.INTER_AREA)
            cls.append(_hist(img))
            step = 48 // GRID
            pat.append(np.stack([_hist(img[i * step:(i + 1) * step, j * step:(j + 1) * step])
                                 for i in range(GRID) for j in range(GRID)]))
        return np.stack(cls).astype(np.float32), np.stack(pat).astype(np.float32)

    def warmup(self) -> None:
        self.embed([np.zeros((48, 48, 3), np.uint8)])

    def close(self) -> None:
        return None
