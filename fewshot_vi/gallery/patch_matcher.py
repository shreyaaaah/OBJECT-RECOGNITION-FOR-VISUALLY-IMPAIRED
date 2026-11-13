"""Patch-token (symmetric chamfer-style) matching for look-alike objects.

Used only for AMBIGUOUS matches whose candidates are in the same confuser
group. ``PatchReranker`` runs the work in a daemon worker thread with a
deadline measured from submission (queue wait + compute). A result that misses
the deadline is still returned by ``poll()`` but flagged ``met_deadline=False``,
and ``apply_rerank`` ignores it, so the provisional decision is preserved.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..types import Decision, Match

log = logging.getLogger(__name__)


def _norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def symmetric_patch_score(query: np.ndarray, stored: np.ndarray) -> float:
    """Normalise tokens, build the token similarity matrix, and average the
    mean row-max (query->stored) and mean column-max (stored->query)."""
    if query.size == 0 or stored.size == 0:
        return -1.0
    q, s = _norm(np.asarray(query, np.float32)), _norm(np.asarray(stored, np.float32))
    sim = q @ s.T
    return 0.5 * (float(sim.max(axis=1).mean()) + float(sim.max(axis=0).mean()))


def score_candidate(query_patches: np.ndarray, views: list[np.ndarray]) -> float:
    """Candidate score = best score over its stored enrolment views."""
    if not views:
        return -1.0
    return max(symmetric_patch_score(query_patches, v) for v in views)


def rerank(query_patches: np.ndarray, candidates: dict[int, list[np.ndarray]]) -> list[tuple[int, float]]:
    """Return ``[(object_id, patch_score)]`` sorted best-first."""
    out = [(oid, score_candidate(query_patches, views)) for oid, views in candidates.items()]
    out.sort(key=lambda t: t[1], reverse=True)
    return out


@dataclass
class RerankResult:
    track_id: int
    ranking: list[tuple[int, float]]
    elapsed_s: float
    met_deadline: bool


@dataclass
class _Job:
    track_id: int
    patches: np.ndarray
    candidates: dict[int, list[np.ndarray]]
    t_submit: float


class PatchReranker:
    """Asynchronous re-ranker with a bounded job queue and a result queue."""

    def __init__(self, deadline_s: float = 0.03, max_pending: int = 8) -> None:
        self.deadline_s = deadline_s
        self._jobs: "queue.Queue[Optional[_Job]]" = queue.Queue(maxsize=max_pending)
        self._results: "queue.Queue[RerankResult]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rerank", daemon=True)
        self._thread.start()

    def submit(self, track_id: int, patches: np.ndarray, candidates: dict[int, list[np.ndarray]]) -> bool:
        """Non-blocking submit. Returns False (job dropped) if the queue is full."""
        try:
            self._jobs.put_nowait(_Job(track_id, patches, candidates, time.perf_counter()))
            return True
        except queue.Full:
            return False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            if job is None:
                break
            try:
                t0 = time.perf_counter()
                ranking = rerank(job.patches, job.candidates)
                t1 = time.perf_counter()
                self._results.put(RerankResult(job.track_id, ranking, t1 - t0,
                                               (t1 - job.t_submit) <= self.deadline_s))
            except Exception:  # noqa: BLE001 - never kill the worker
                log.exception("patch rerank failed for track %s", job.track_id)

    def poll(self) -> list[RerankResult]:
        out = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        self._stop.set()
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=1.0)


def apply_rerank(provisional: Match, result: RerankResult, store_names: Callable[[int], Optional[str]],
                 accept_patch_sim: float, min_gain: float = 0.0, min_patch_margin: float = 0.0) -> Match:
    """Fold a finished re-rank into a provisional match.

    The re-ranked winner replaces the provisional one only if its patch score
    reaches ``accept_patch_sim``, it beats the runner-up by ``min_patch_margin``,
    and (when it differs from the provisional top object) it beats the
    provisional object's patch score by more than ``min_gain``. Otherwise the
    provisional result is returned unchanged.
    """
    if not result.met_deadline or not result.ranking:
        return provisional
    best_id, best = result.ranking[0]
    second = result.ranking[1][1] if len(result.ranking) > 1 else -1.0
    if best < accept_patch_sim or (best - second) < min_patch_margin:
        return provisional
    scores = dict(result.ranking)
    if best_id != provisional.object_id and provisional.object_id in scores:
        if best - scores[provisional.object_id] <= min_gain:
            return provisional
    global_score = next((s for oid, s in provisional.ranking if oid == best_id), provisional.score)
    margin = best - second if second > -1.0 else provisional.margin
    return Match(best_id, store_names(best_id), global_score, margin, Decision.ACCEPT,
                 provisional.runner_up_id, provisional.ranking)
