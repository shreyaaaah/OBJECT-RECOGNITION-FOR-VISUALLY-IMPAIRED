"""Direction zones and box-size distance estimation.

Offsets are measured as a fraction of the FULL frame size from the frame centre
(range -0.5..+0.5). ``|offset| < near_frac`` is "centre"; ``< slight_frac`` is
"slightly left/right (up/down)"; beyond that it is plain "left/right (up/down)".
``pan`` is the same offset rescaled to -1..+1 for stereo panning.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Direction:
    horizontal: str          # "centre" | "slightly left" | "left" | "slightly right" | "right"
    vertical: str            # "centre" | "slightly up" | "up" | "slightly down" | "down"
    h_offset: float          # fraction of frame width, left negative
    v_offset: float          # fraction of frame height, up negative (image coords) -> see vertical word
    pan: float               # -1 (hard left) .. +1 (hard right)


def _zone(offset: float, near: float, slight: float, neg: str, pos: str) -> str:
    a = abs(offset)
    if a < near:
        return "centre"
    word = neg if offset < 0 else pos
    return f"slightly {word}" if a < slight else word


def direction(bbox: tuple[float, float, float, float], frame_w: int, frame_h: int,
              near_frac: float, slight_frac: float) -> Direction:
    cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    h_off = (cx - frame_w / 2.0) / frame_w
    v_off = (cy - frame_h / 2.0) / frame_h   # image y grows downward
    return Direction(
        horizontal=_zone(h_off, near_frac, slight_frac, "left", "right"),
        vertical=_zone(v_off, near_frac, slight_frac, "up", "down"),
        h_offset=h_off, v_offset=v_off,
        pan=max(-1.0, min(1.0, 2.0 * h_off)),
    )


def focal_px(frame_w: int, hfov_deg: float) -> float:
    return (frame_w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def distance_m(bbox_width_px: float, frame_w: int, hfov_deg: float, known_width_cm: float) -> float:
    """Pinhole estimate ``d = known_width * focal_px / bbox_width_px`` in metres."""
    if bbox_width_px <= 0:
        return math.inf
    return (known_width_cm / 100.0) * focal_px(frame_w, hfov_deg) / bbox_width_px


def distance_phrase(d_m: float) -> str:
    """Speech-friendly rounding: very close / half a metre / one metre / N metres."""
    if not math.isfinite(d_m):
        return "distance unknown"
    r = round(d_m * 2) / 2.0
    if r < 0.5:
        return "very close"
    if r == 0.5:
        return "about half a metre"
    if r == 1.0:
        return "about one metre"
    if r == 1.5:
        return "about one and a half metres"
    n = int(round(r))
    return f"about {n} metres"
