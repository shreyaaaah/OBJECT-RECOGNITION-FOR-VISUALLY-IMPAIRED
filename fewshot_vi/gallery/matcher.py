"""Open-set cosine matcher with margin rules and confuser-aware ambiguity.

Per-object score = max(prototype similarity, mean of the top-k embedding
similarities). Decision rules (thresholds come from config):

* ``ACCEPT``    : s1 >= accept_sim AND margin >= min_margin AND top-2 are not
                  confusers of each other.
* ``AMBIGUOUS`` : ambiguous_sim <= s1 (and not accepted), OR the top two
                  candidates are confusers of each other while s1 is at least
                  ambiguous_sim (they need patch-level re-ranking).
* ``UNKNOWN``   : otherwise (including an empty gallery).

Single-object rule (explicit): when only ONE object exists there is no runner-up,
so the margin is ``+inf`` and the margin test passes; safety then rests entirely
on ``accept_sim``. A missing second candidate NEVER turns an ACCEPT into
AMBIGUOUS, and never turns a below-threshold score into an ACCEPT.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..types import Decision, Match
from .store import GalleryStore


class Matcher:
    def __init__(self, store: GalleryStore, accept_sim: float = 0.70, ambiguous_sim: float = 0.55,
                 min_margin: float = 0.08, knn_k: int = 3) -> None:
        self.store = store
        self.accept_sim, self.ambiguous_sim = accept_sim, ambiguous_sim
        self.min_margin, self.knn_k = min_margin, knn_k

    def scores(self, q: np.ndarray) -> list[tuple[int, float]]:
        """Per-object scores for query ``q``, sorted best-first."""
        store = self.store
        E, owner = store.snapshot()
        if E.shape[0] == 0:
            return []
        q = np.asarray(q, dtype=np.float32).ravel()
        q = q / max(float(np.linalg.norm(q)), 1e-8)
        sims = E @ q  # (M,)  -- the single matrix multiply
        out: list[tuple[int, float]] = []
        for oid in np.unique(owner):
            mask = owner == oid
            s = sims[mask]
            proto = E[mask].mean(axis=0)
            proto = proto / max(float(np.linalg.norm(proto)), 1e-8)
            proto_sim = float(proto @ q)
            k = min(self.knn_k, s.shape[0])
            topk = float(np.sort(s)[-k:].mean())
            out.append((int(oid), max(proto_sim, topk)))
        out.sort(key=lambda t: t[1], reverse=True)
        return out

    def match(self, q: np.ndarray) -> Match:
        ranking = self.scores(q)
        if not ranking:
            return Match(None, None, 0.0, 0.0, Decision.UNKNOWN, None, [])
        oid1, s1 = ranking[0]
        if len(ranking) > 1:
            oid2, s2 = ranking[1]
            margin = s1 - s2
        else:
            oid2, margin = None, math.inf  # explicit single-object rule (see module docstring)
        name = self.store.get_name(oid1)
        confusable = oid2 is not None and self.store.are_confusers(oid1, oid2)
        if s1 >= self.accept_sim and margin >= self.min_margin and not confusable:
            decision = Decision.ACCEPT
        elif s1 >= self.ambiguous_sim:
            # either in the ambiguous band, or too close to a runner-up, or confusable
            decision = Decision.AMBIGUOUS
        else:
            decision = Decision.UNKNOWN
        return Match(oid1, name, s1, margin, decision, oid2, ranking)

    def confuser_candidates(self, m: Match) -> list[int]:
        """Objects that should be patch re-ranked for an AMBIGUOUS match: the top
        object plus every ranked object in its confuser group that is itself at
        least ``ambiguous_sim``."""
        if m.object_id is None:
            return []
        group = self.store.confuser_group(m.object_id)
        if group is None:
            return []
        cands = [m.object_id]
        for oid, s in m.ranking:
            if oid != m.object_id and s >= self.ambiguous_sim and self.store.confuser_group(oid) == group:
                cands.append(oid)
        return cands if len(cands) > 1 else []
