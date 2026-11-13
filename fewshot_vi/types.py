"""Shared dataclasses and enums used across the pipeline."""
from __future__ import annotations

import enum
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


class Decision(str, enum.Enum):
    ACCEPT = "ACCEPT"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"


class Mode(str, enum.Enum):
    IDENTIFY = "IDENTIFY"
    FIND = "FIND"
    ENROL = "ENROL"


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class Match:
    """Result of matching one query embedding against the gallery."""
    object_id: Optional[int]
    name: Optional[str]
    score: float
    margin: float
    decision: Decision
    runner_up_id: Optional[int] = None
    ranking: list[tuple[int, float]] = field(default_factory=list)


@dataclass
class Quality:
    blur: float
    blurry: bool
    edge_cut: frozenset
    too_small: bool
    low_margin: bool
    score: float

    @property
    def ok(self) -> bool:
        return not (self.blurry or self.edge_cut or self.too_small or self.low_margin)


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    age: int = 0
    missed: int = 0
    last_embed_frame: int = -10**9
    last_embed_bbox: Optional[tuple[float, float, float, float]] = None
    embedding: Optional[np.ndarray] = None
    patches: Optional[np.ndarray] = None
    needs_embed: bool = True
    score: float = 0.0
    label_history: deque = field(default_factory=lambda: deque(maxlen=7))
    conf_history: deque = field(default_factory=lambda: deque(maxlen=7))
    match: Optional[Match] = None
    smoothed_name: Optional[str] = None
    smoothed_id: Optional[int] = None
    smoothed_conf: float = 0.0
    quality: Optional[Quality] = None
    stable_accepts: int = 0
    rerank_pending: bool = False
    rerank_agrees: Optional[bool] = None   # did patch re-rank agree with the global-embedding winner?
    update_tried: bool = False


@dataclass
class GuidanceEvent:
    kind: str           # "announce" | "hint" | "find" | "memory" | "answer" | "beep" | "system"
    text: str
    priority: int       # higher = more urgent; see guidance.engine PRIORITY_*
    pan: float = 0.0    # -1 (left) .. +1 (right)
    urgency: float = 0.0  # 0 (far) .. 1 (touching); drives beep rate/pitch
    ts: float = 0.0


@dataclass
class Sighting:
    object_id: int
    name: str
    ts: float
    scene_label: Optional[str]
    bbox: tuple[float, float, float, float]
    conf: float


class AppState:
    """Thread-safe application state: mode, find target, mute, shutdown."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._mode = Mode.IDENTIFY
        self._target: Optional[str] = None
        self._muted = False
        self._shutdown = threading.Event()
        self._find_started: Optional[float] = None
        self._pending: list[tuple[str, Any]] = []

    @property
    def mode(self) -> Mode:
        with self._lock:
            return self._mode

    @property
    def target(self) -> Optional[str]:
        with self._lock:
            return self._target

    @property
    def muted(self) -> bool:
        with self._lock:
            return self._muted

    @property
    def find_started(self) -> Optional[float]:
        with self._lock:
            return self._find_started

    def set_mode(self, mode: Mode, target: Optional[str] = None, now: Optional[float] = None) -> None:
        with self._lock:
            self._mode = mode
            self._target = target if mode == Mode.FIND else None
            self._find_started = now if mode == Mode.FIND else None

    def set_muted(self, muted: bool) -> None:
        with self._lock:
            self._muted = muted

    def request_shutdown(self) -> None:
        self._shutdown.set()

    @property
    def shutting_down(self) -> bool:
        return self._shutdown.is_set()

    def post(self, kind: str, payload: Any = None) -> None:
        """Queue a request (e.g. 'enrol', 'delete') for the vision loop."""
        with self._lock:
            self._pending.append((kind, payload))

    def drain(self) -> list[tuple[str, Any]]:
        with self._lock:
            out, self._pending = self._pending, []
            return out
