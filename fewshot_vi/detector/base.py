"""Swappable detector interface and a scripted fake for tests."""
from __future__ import annotations

import abc
from typing import Sequence

import numpy as np

from ..types import Detection


class Detector(abc.ABC):
    """Class-agnostic box proposer. Implementations must NOT filter by class."""

    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return detections for ``frame`` (BGR uint8, HxWx3)."""

    def close(self) -> None:  # optional resource release
        return None


class FakeDetector(Detector):
    """Returns scripted detections; cycles through ``script`` one entry per call."""

    def __init__(self, script: Sequence[Sequence[Detection]]) -> None:
        self.script = [list(s) for s in script]
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        out = self.script[min(self.calls, len(self.script) - 1)] if self.script else []
        self.calls += 1
        return list(out)
