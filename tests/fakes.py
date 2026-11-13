"""Hardware-free fakes: camera, embedder, speaker backend and synthetic frames."""
from __future__ import annotations

import threading

import numpy as np

from fewshot_vi.types import Detection
from fewshot_vi.voice import tts

H, W = 480, 640


def textured_frame(boxes=(), seed=0):
    """Dark noisy background; each ``(x1,y1,x2,y2,channel)`` box is a sharp, tinted noise patch.

    The tint channel (0=B,1=G,2=R) is what ``FakeEmbedder`` uses as the object's identity.
    """
    rng = np.random.default_rng(seed)
    f = rng.integers(0, 40, (H, W, 3), dtype=np.uint8)
    for (x1, y1, x2, y2, ch) in boxes:
        patch = rng.integers(0, 60, (y2 - y1, x2 - x1, 3), dtype=np.uint8)
        patch[..., ch] = rng.integers(140, 255, (y2 - y1, x2 - x1), dtype=np.uint8)
        f[y1:y2, x1:x2] = patch
    return f


def det(x1, y1, x2, y2, score=0.9):
    return Detection(float(x1), float(y1), float(x2), float(y2), score)


class FakeEmbedder:
    """Maps a crop's dominant colour channel to a fixed (cls, patches) pair."""

    def __init__(self, cls_by_channel, patches_by_channel=None, dim=32):
        self.cls_by_channel = {k: np.asarray(v, np.float32) for k, v in cls_by_channel.items()}
        self.patches_by_channel = patches_by_channel or {}
        self.dim = dim
        self.calls = 0
        self.batch_sizes = []
        self.closed = False

    def embed(self, crops):
        self.calls += 1
        self.batch_sizes.append(len(crops))
        cls, pat = [], []
        for c in crops:
            ch = int(np.argmax(c.reshape(-1, 3).mean(axis=0)))
            cls.append(self.cls_by_channel.get(ch, np.eye(self.dim, dtype=np.float32)[self.dim - 1]))
            pat.append(self.patches_by_channel.get(ch, np.ones((4, self.dim), np.float32)))
        return np.stack(cls), np.stack(pat)

    def close(self):
        self.closed = True


class FakeCamera:
    """Yields the given frames then signals ``finished``. Mimics ``Camera.read/stop``."""

    def __init__(self, frames, error=None):
        self._frames = list(frames)
        self._i = 0
        self.finished = threading.Event()
        self.error = error
        self.stopped = False

    def read(self, timeout=1.0):
        if self._i >= len(self._frames):
            self.finished.set()
            return None
        f = self._frames[self._i]
        self._i += 1
        return f, 0.0

    def stop(self):
        self.stopped = True


class ScriptedDetector:
    """Returns ``script[call_index]`` (list of Detection); an Exception entry is raised."""

    def __init__(self, script):
        self.script, self.calls, self.closed = list(script), 0, False

    def detect(self, frame):
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return list(item)

    def close(self):
        self.closed = True


class RecordingBackend(tts.TTSBackend):
    name = "recording"

    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)
