"""Shared helpers: synthetic embeddings, fake clocks, temp stores. No hardware/model/network."""
from __future__ import annotations

import numpy as np
import pytest

from fewshot_vi.config import config_from_dict
from fewshot_vi.gallery.store import GalleryStore

DIM = 32


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def basis(i: int, dim: int = DIM) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


def near(base: np.ndarray, sim: float, axis: int = DIM - 1) -> np.ndarray:
    """Unit vector with cosine similarity ``sim`` to unit vector ``base`` (uses a spare axis)."""
    other = np.zeros_like(base)
    other[axis] = 1.0
    other = unit(other - (other @ base) * base)
    return unit(sim * base + np.sqrt(max(0.0, 1 - sim * sim)) * other)


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture
def cfg():
    return config_from_dict()


@pytest.fixture
def store():
    s = GalleryStore(":memory:", dim=DIM)
    yield s
    s.close()


@pytest.fixture
def clock():
    return FakeClock()
