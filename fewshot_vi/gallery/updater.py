"""Safe self-improving gallery.

``GalleryUpdater.maybe_add`` appends a new ``source='auto'`` embedding only when
EVERY guard passes. ``'enrol'`` embeddings are never removed: when an object is
at its cap, the most redundant ``'auto'`` embedding is replaced; if there is no
auto embedding to replace, the update is skipped.
"""
from __future__ import annotations

import logging
import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..types import Decision, Match, Quality
from .store import GalleryStore

log = logging.getLogger(__name__)


@dataclass
class UpdateResult:
    added: bool
    reason: str
    replaced_id: Optional[int] = None


class GalleryUpdater:
    def __init__(self, store: GalleryStore, window: int, max_per_object: int, min_sim: float,
                 min_margin: float, cooldown_s: float, duplicate_sim: float, drift_min_sim: float,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.store, self.window, self.max_per_object = store, window, max_per_object
        self.min_sim, self.min_margin, self.cooldown_s = min_sim, min_margin, cooldown_s
        self.duplicate_sim, self.drift_min_sim = duplicate_sim, drift_min_sim
        self.clock = clock
        self._last_add: dict[int, float] = {}
        self._frozen: set[int] = set()

    def is_frozen(self, object_id: int) -> bool:
        return object_id in self._frozen

    def check_drift(self, object_id: int) -> bool:
        """True if the object's auto embeddings have drifted from its enrolment
        prototype (mean auto/prototype similarity below ``drift_min_sim``)."""
        proto = self.store.enrol_prototype(object_id)
        auto = self.store.embeddings_of(object_id, "auto")
        if proto is None or auto.shape[0] == 0:
            return False
        return float((auto @ proto).mean()) < self.drift_min_sim

    def _redundancy_victim(self, object_id: int) -> Optional[int]:
        """Auto embedding with the highest mean similarity to the object's others."""
        rows = self.store.embedding_rows(object_id)
        if len(rows) < 2:
            return None
        vecs = np.stack([r[2] for r in rows])
        sim = vecs @ vecs.T
        np.fill_diagonal(sim, 0.0)
        mean_sim = sim.sum(axis=1) / (len(rows) - 1)
        best, best_s = None, -math.inf
        for (eid, src, _), s in zip(rows, mean_sim):
            if src == "auto" and s > best_s:
                best, best_s = eid, float(s)
        return best

    def maybe_add(self, object_id: int, embedding: np.ndarray, match: Match, quality: Optional[Quality],
                  stable_accepts: int, confuser_agrees: bool = True) -> UpdateResult:
        if self.is_frozen(object_id):
            return UpdateResult(False, "frozen: drift guard tripped")
        if match.decision != Decision.ACCEPT or match.object_id != object_id:
            return UpdateResult(False, "not an accepted match for this object")
        if stable_accepts < self.window:
            return UpdateResult(False, "not stable for the full temporal window")
        if match.score < self.min_sim:
            return UpdateResult(False, "score below update threshold")
        if math.isfinite(match.margin) and match.margin < self.min_margin:
            return UpdateResult(False, "margin below update threshold")
        if quality is None or not quality.ok:
            return UpdateResult(False, "quality check failed")
        if self.store.confuser_group(object_id) is not None and not confuser_agrees:
            return UpdateResult(False, "confuser re-rank did not agree")
        now = self.clock()
        last = self._last_add.get(object_id)
        if last is not None and now - last < self.cooldown_s:
            return UpdateResult(False, "cooldown active")
        emb = np.asarray(embedding, dtype=np.float32).ravel()
        emb = emb / max(float(np.linalg.norm(emb)), 1e-8)
        existing = self.store.embeddings_of(object_id)
        if existing.shape[0] and float((existing @ emb).max()) >= self.duplicate_sim:
            return UpdateResult(False, "near-duplicate of an existing embedding")
        replaced = None
        if existing.shape[0] >= self.max_per_object:
            replaced = self._redundancy_victim(object_id)
            if replaced is None:
                return UpdateResult(False, "gallery full and no auto embedding to replace")
            self.store.delete_embedding(replaced, allow_enrol=False)
        self.store.add_embedding(object_id, emb, "auto")
        self._last_add[object_id] = now
        if self.check_drift(object_id):
            self._frozen.add(object_id)
            log.warning("auto-embeddings for %s drifted below %.2f similarity to the enrolment "
                        "prototype; automatic updates stopped for this object",
                        self.store.get_name(object_id), self.drift_min_sim)
        return UpdateResult(True, "added", replaced)


@dataclass
class _UpdateJob:
    object_id: int
    embedding: np.ndarray
    match: Match
    quality: Optional[Quality]
    stable_accepts: int
    confuser_agrees: bool


class GalleryUpdateWorker:
    """Run safe gallery checks and SQLite mutations off the vision thread.

    ``submit`` only copies a small immutable snapshot into a bounded queue. If
    the queue is full, the caller can retry on a later frame without blocking.
    Shutdown drains all accepted jobs before returning.
    """

    def __init__(self, updater: GalleryUpdater, max_pending: int = 16) -> None:
        self.updater = updater
        self._jobs: "queue.Queue[Optional[_UpdateJob]]" = queue.Queue(maxsize=max_pending)
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="gallery-writer", daemon=True)
        self._thread.start()

    def submit(self, object_id: int, embedding: np.ndarray, match: Match,
               quality: Optional[Quality], stable_accepts: int,
               confuser_agrees: bool = True) -> bool:
        """Queue a snapshot without waiting for database work."""
        if self._closed:
            return False
        match_copy = Match(match.object_id, match.name, float(match.score), float(match.margin),
                           match.decision, match.runner_up_id, list(match.ranking))
        quality_copy = None if quality is None else Quality(
            quality.blur, quality.blurry, frozenset(quality.edge_cut), quality.too_small,
            quality.low_margin, quality.score)
        job = _UpdateJob(object_id, np.asarray(embedding, dtype=np.float32).copy(), match_copy,
                         quality_copy, stable_accepts, confuser_agrees)
        try:
            self._jobs.put_nowait(job)
            return True
        except queue.Full:
            return False

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return
                try:
                    result = self.updater.maybe_add(job.object_id, job.embedding, job.match,
                                                    job.quality, job.stable_accepts,
                                                    confuser_agrees=job.confuser_agrees)
                    if result.added:
                        log.info("gallery writer: added auto embedding for %s",
                                 self.updater.store.get_name(job.object_id))
                    else:
                        log.debug("gallery writer skipped object %s: %s", job.object_id, result.reason)
                except Exception:  # noqa: BLE001 - writer failures must not kill its worker
                    log.exception("gallery update failed for object %s", job.object_id)
            finally:
                self._jobs.task_done()

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Wait for queued work to finish; primarily useful for tests/shutdown."""
        deadline = time.monotonic() + timeout
        while self._jobs.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.005)
        return self._jobs.unfinished_tasks == 0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._jobs.put(None)  # blocks only during shutdown until queue drains
        self._thread.join()
