"""Self-contained IoU tracker with constant-velocity prediction.

Chosen over Ultralytics ByteTrack so the tracker is fully unit-testable without
model weights. It keeps stable track IDs, predicts boxes on frames where the
detector is skipped, and flags tracks that need a fresh embedding.
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

import numpy as np

from ..types import Detection, Track

Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class IoUTracker:
    def __init__(self, match_iou: float = 0.3, max_missed: int = 10, reembed_iou: float = 0.6,
                 reembed_every: int = 30, history: int = 7) -> None:
        self.match_iou, self.max_missed = match_iou, max_missed
        self.reembed_iou, self.reembed_every, self.history = reembed_iou, reembed_every, history
        self.tracks: dict[int, Track] = {}
        self._vel: dict[int, np.ndarray] = {}
        self._next_id = 1
        self.frame_idx = 0

    def _new_track(self, det: Detection) -> Track:
        t = Track(track_id=self._next_id, bbox=det.box, score=det.score,
                  class_name=det.class_name, class_id=det.class_id,
                  label_history=deque(maxlen=self.history), conf_history=deque(maxlen=self.history))
        t.category_history = deque(maxlen=self.history)
        self._vel[t.track_id] = np.zeros(4)
        self.tracks[t.track_id] = t
        self._next_id += 1
        return t

    def predict(self) -> None:
        """Advance every track one frame using its velocity (for skipped-detection frames)."""
        self.frame_idx += 1
        for tid, t in list(self.tracks.items()):
            v = self._vel[tid]
            t.bbox = tuple(float(x) for x in np.asarray(t.bbox) + v)  # type: ignore[assignment]
            t.age += 1
            t.missed += 1
            if t.missed > self.max_missed:
                del self.tracks[tid]
                del self._vel[tid]
        self._flag_reembed()

    def update(self, detections: Sequence[Detection]) -> list[Track]:
        """Associate detections (greedy by IoU) with tracks; spawn tracks for the rest."""
        self.predict()
        pairs = []
        for tid, t in self.tracks.items():
            for di, d in enumerate(detections):
                v = iou(t.bbox, d.box)
                if v >= self.match_iou:
                    pairs.append((v, tid, di))
        pairs.sort(reverse=True)
        used_t: set[int] = set()
        used_d: set[int] = set()
        for _, tid, di in pairs:
            if tid in used_t or di in used_d:
                continue
            used_t.add(tid)
            used_d.add(di)
            t, d = self.tracks[tid], detections[di]
            new = np.asarray(d.box, dtype=float)
            self._vel[tid] = 0.5 * self._vel[tid] + 0.5 * (new - np.asarray(t.bbox))
            t.bbox, t.score, t.missed = d.box, d.score, 0
            t.class_name, t.class_id = d.class_name, d.class_id
        for di, d in enumerate(detections):
            if di not in used_d:
                self._new_track(d)
        self._flag_reembed()
        return list(self.tracks.values())

    def _flag_reembed(self) -> None:
        for t in self.tracks.values():
            if t.embedding is None or t.last_embed_bbox is None:
                t.needs_embed = True
            elif iou(t.bbox, t.last_embed_bbox) < self.reembed_iou:
                t.needs_embed = True
            elif self.frame_idx - t.last_embed_frame >= self.reembed_every:
                t.needs_embed = True

    def mark_embedded(self, track: Track, embedding: np.ndarray,
                      patches: Optional[np.ndarray] = None) -> None:
        track.embedding, track.patches = embedding, patches
        track.last_embed_frame = self.frame_idx
        track.last_embed_bbox = track.bbox
        track.needs_embed = False

    def active(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.missed <= self.max_missed]
