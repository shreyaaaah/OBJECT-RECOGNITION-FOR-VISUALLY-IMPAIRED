"""Per-track image-quality signals used for active re-view and safe updates."""
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from ..types import Quality


def laplacian_blur(crop: np.ndarray) -> float:
    """Variance of the Laplacian; low values mean a blurry crop."""
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def edge_cuts(bbox: tuple[float, float, float, float], frame_w: int, frame_h: int,
              margin_px: int) -> frozenset:
    """Sides of ``bbox`` that touch the frame border (within ``margin_px``)."""
    x1, y1, x2, y2 = bbox
    cuts = set()
    if x1 <= margin_px:
        cuts.add("left")
    if x2 >= frame_w - margin_px:
        cuts.add("right")
    if y1 <= margin_px:
        cuts.add("top")
    if y2 >= frame_h - margin_px:
        cuts.add("bottom")
    return frozenset(cuts)


def assess(crop: np.ndarray, bbox: tuple[float, float, float, float], frame_shape: tuple[int, ...],
           blur_var_min: float, edge_margin_px: int, min_box_area_frac: float,
           margin: Optional[float] = None, min_margin: float = 0.0) -> Quality:
    """Compute a ``Quality`` record.

    ``margin`` is the matcher's top-1/top-2 margin (``None`` or ``inf`` means
    there is no runner-up, which is not a low-margin condition).
    """
    fh, fw = frame_shape[:2]
    blur = laplacian_blur(crop)
    blurry = blur < blur_var_min
    cuts = edge_cuts(bbox, fw, fh, edge_margin_px)
    area_frac = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]) / float(max(fw * fh, 1))
    too_small = area_frac < min_box_area_frac
    low_margin = margin is not None and math.isfinite(margin) and margin < min_margin
    parts = [
        min(blur / (2.0 * blur_var_min), 1.0) if blur_var_min > 0 else 1.0,
        0.0 if cuts else 1.0,
        0.0 if too_small else 1.0,
        0.0 if low_margin else 1.0,
    ]
    return Quality(blur=blur, blurry=blurry, edge_cut=cuts, too_small=too_small,
                   low_margin=low_margin, score=float(np.mean(parts)))
