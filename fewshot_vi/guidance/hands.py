"""Optional hand-guided reach (config flag ``features.hand_guidance``; OFF by default).

``reach_instruction`` is pure geometry (testable without MediaPipe).
``HandTracker`` wraps MediaPipe Hands lazily; if MediaPipe is missing it reports
itself unavailable instead of inventing hand positions.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def _overlaps(point: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def reach_instruction(hand_xy: tuple[float, float], target_box: tuple[float, float, float, float],
                      near_px: float = 40.0) -> str:
    """Spoken instruction moving the hand centroid toward the target box centre."""
    if _overlaps(hand_xy, target_box):
        return "Stop, it's right there."
    cx, cy = (target_box[0] + target_box[2]) / 2.0, (target_box[1] + target_box[3]) / 2.0
    dx, dy = cx - hand_xy[0], cy - hand_xy[1]
    parts = []
    if abs(dx) > near_px:
        parts.append(("slightly " if abs(dx) < 3 * near_px else "") + ("right" if dx > 0 else "left"))
    if abs(dy) > near_px:
        parts.append(("slightly " if abs(dy) < 3 * near_px else "") + ("down" if dy > 0 else "up"))
    return f"Move your hand {' and '.join(parts)}." if parts else "You are very close."


class HandTracker:
    def __init__(self) -> None:
        self.available = False
        self._hands = None
        try:
            import mediapipe as mp
            self._hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
            self.available = True
        except Exception as exc:  # noqa: BLE001
            log.warning("hand guidance unavailable (%s: %s)", type(exc).__name__, exc)

    def centroid(self, frame_bgr: np.ndarray) -> Optional[tuple[float, float]]:
        if not self.available:
            return None
        import cv2
        res = self._hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        pts = np.array([(lm.x * w, lm.y * h) for lm in res.multi_hand_landmarks[0].landmark])
        c = pts.mean(axis=0)
        return float(c[0]), float(c[1])

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
