"""Accuracy, open-set error rates, latency statistics and rolling FPS."""
from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


class LatencyStats:
    """Thread-safe per-stage latency recorder (keeps the most recent ``maxlen`` samples)."""

    def __init__(self, maxlen: int = 500) -> None:
        self._lock = threading.Lock()
        self._d: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))

    def record(self, stage: str, seconds: float) -> None:
        with self._lock:
            self._d[stage].append(float(seconds))

    def count(self, stage: str) -> int:
        with self._lock:
            return len(self._d.get(stage, ()))

    def mean_ms(self, stage: str) -> Optional[float]:
        with self._lock:
            v = list(self._d.get(stage, ()))
        return float(np.mean(v) * 1000.0) if v else None

    def p95_ms(self, stage: str) -> Optional[float]:
        with self._lock:
            v = list(self._d.get(stage, ()))
        return float(np.percentile(v, 95) * 1000.0) if v else None

    def summary(self) -> dict[str, dict[str, float]]:
        with self._lock:
            stages = list(self._d)
        out = {}
        for s in stages:
            n = self.count(s)
            if n:
                out[s] = {"n": n, "mean_ms": self.mean_ms(s), "p95_ms": self.p95_ms(s)}
        return out


class FpsMeter:
    def __init__(self, window: int = 30) -> None:
        self._t: deque = deque(maxlen=window)

    def tick(self, t: float) -> None:
        self._t.append(t)

    @property
    def fps(self) -> float:
        if len(self._t) < 2 or self._t[-1] <= self._t[0]:
            return 0.0
        return (len(self._t) - 1) / (self._t[-1] - self._t[0])


def top1_accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not len(y_true):
        return float("nan")
    return float(np.mean([a == b for a, b in zip(y_true, y_pred)]))


@dataclass
class OpenSetCounts:
    known_total: int = 0
    known_correct: int = 0       # accepted as the right class
    known_wrong: int = 0         # accepted as a WRONG known class (worst case for a blind user)
    known_rejected: int = 0      # reported unknown/not-sure for a known object
    unknown_total: int = 0
    unknown_accepted: int = 0    # an unknown object accepted as some known class

    def add(self, true_label: Optional[str], pred_label: Optional[str]) -> None:
        """``true_label=None`` means the query is from an UNKNOWN class; ``pred_label=None`` means rejected."""
        if true_label is None:
            self.unknown_total += 1
            self.unknown_accepted += pred_label is not None
        else:
            self.known_total += 1
            if pred_label is None:
                self.known_rejected += 1
            elif pred_label == true_label:
                self.known_correct += 1
            else:
                self.known_wrong += 1

    @property
    def top1_accuracy(self) -> float:
        return self.known_correct / self.known_total if self.known_total else float("nan")

    @property
    def far(self) -> float:
        """False accept rate: unknown objects accepted as a known class."""
        return self.unknown_accepted / self.unknown_total if self.unknown_total else float("nan")

    @property
    def frr(self) -> float:
        """False reject rate: known objects reported as unknown / not sure."""
        return self.known_rejected / self.known_total if self.known_total else float("nan")

    @property
    def misid_rate(self) -> float:
        return self.known_wrong / self.known_total if self.known_total else float("nan")

    def as_dict(self) -> dict[str, float]:
        return {"top1_acc": self.top1_accuracy, "FAR": self.far, "FRR": self.frr,
                "misid": self.misid_rate, "known_n": self.known_total, "unknown_n": self.unknown_total}


def open_set_metrics(pairs: Iterable[tuple[Optional[str], Optional[str]]]) -> OpenSetCounts:
    c = OpenSetCounts()
    for t, p in pairs:
        c.add(t, p)
    return c
