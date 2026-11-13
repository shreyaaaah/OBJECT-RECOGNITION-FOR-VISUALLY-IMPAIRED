"""Dataset scanning and reproducible N-way K-shot episode sampling.

Dataset layout (ORBIT/CORe50-style exports work after flattening per object)::

    dataset/<class_name>/*.jpg        (.jpeg/.png/.bmp are accepted too)

Each image is treated as ONE object crop (the whole image is embedded). Episodes
contain ``n_way`` KNOWN classes (K support images enrolled, ``n_query`` queries
each, disjoint from the support) plus ``n_unknown`` UNKNOWN classes whose
``n_query`` images are only used as open-set queries.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class DatasetError(ValueError):
    """The dataset is missing, malformed, or too small for the requested episodes."""


@dataclass
class Episode:
    index: int
    known: list[str]
    unknown: list[str]
    support: dict[str, list[Path]]
    query_known: dict[str, list[Path]]
    query_unknown: dict[str, list[Path]] = field(default_factory=dict)


def scan_dataset(root: str | Path, max_per_class: Optional[int] = None) -> dict[str, list[Path]]:
    """``{class_name: sorted image paths}`` for every class folder that has images."""
    r = Path(root)
    if not r.is_dir():
        raise DatasetError(
            f"dataset folder not found: {r}\n"
            "Expected layout: <dataset>/<class_name>/*.jpg (one sub-folder per object class).")
    index: dict[str, list[Path]] = {}
    for d in sorted(p for p in r.iterdir() if p.is_dir()):
        imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
        if max_per_class is not None:
            imgs = imgs[:max_per_class]
        if imgs:
            index[d.name] = imgs
    if not index:
        raise DatasetError(f"no class folders with images found under {r}; expected <dataset>/<class>/*.jpg")
    return index


def check_capacity(index: dict[str, list[Path]], n_way: int, k_shot: int, n_query: int, n_unknown: int) -> None:
    """Raise a helpful ``DatasetError`` if the dataset cannot supply the requested episodes."""
    need_known = k_shot + n_query
    ok_known = [c for c, v in index.items() if len(v) >= need_known]
    ok_unknown_pool = [c for c, v in index.items() if len(v) >= n_query]
    problems = []
    if len(ok_known) < n_way:
        problems.append(f"need {n_way} known classes with >= {need_known} images (k_shot {k_shot} + n_query {n_query}); "
                        f"only {len(ok_known)} qualify")
    if n_unknown < 1:
        problems.append("n_unknown must be >= 1 so false-accept rate is defined")
    elif len(ok_unknown_pool) < n_way + n_unknown:
        problems.append(f"need {n_way + n_unknown} classes in total (n_way {n_way} + n_unknown {n_unknown}) with "
                        f">= {n_query} images; only {len(ok_unknown_pool)} qualify")
    if problems:
        sizes = ", ".join(f"{c}:{len(v)}" for c, v in sorted(index.items())[:12])
        raise DatasetError("dataset too small for the requested episodes:\n  - " + "\n  - ".join(problems) +
                           f"\n  classes found ({len(index)}): {sizes}{' ...' if len(index) > 12 else ''}\n"
                           "  Add classes/images, or lower --n-way/--k-shot/--n-query/--n-unknown.")


def sample_episodes(index: dict[str, list[Path]], n_way: int, k_shot: int, n_query: int,
                    n_unknown: int, n_episodes: int, seed: int) -> list[Episode]:
    """Reproducible: the same ``(index, parameters, seed)`` always yields identical episodes."""
    check_capacity(index, n_way, k_shot, n_query, n_unknown)
    eligible_known = sorted(c for c, v in index.items() if len(v) >= k_shot + n_query)
    episodes = []
    for e in range(n_episodes):
        rng = random.Random(f"{seed}:{k_shot}:{e}")
        known = sorted(rng.sample(eligible_known, n_way))
        pool = sorted(c for c, v in index.items() if c not in known and len(v) >= n_query)
        unknown = sorted(rng.sample(pool, n_unknown))
        support, qk, qu = {}, {}, {}
        for c in known:
            imgs = list(index[c])
            rng.shuffle(imgs)
            support[c] = sorted(imgs[:k_shot])
            qk[c] = sorted(imgs[k_shot:k_shot + n_query])
        for c in unknown:
            imgs = list(index[c])
            rng.shuffle(imgs)
            qu[c] = sorted(imgs[:n_query])
        episodes.append(Episode(e, known, unknown, support, qk, qu))
    return episodes


def load_image(path: Path, max_side: int = 512) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise DatasetError(f"cannot read image: {path}")
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / float(max(h, w))
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return img
