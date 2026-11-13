"""SQLite-backed gallery with synchronised in-memory matrices.

Tables: ``objects``, ``embeddings`` (``source IN ('enrol','auto')``) and
``sightings``. All CLS embeddings are mirrored in one contiguous matrix ``E``
(M, D) plus an ``owner`` array so matching is a single matrix multiply. The
mirror is rebuilt incrementally after every insert/delete.

Patch tokens (at most ``top_patches`` salient tokens) are stored ONLY for
enrolment embeddings; automatic embeddings never carry patches.

A single connection is shared across threads behind an ``RLock`` (SQLite
connections are opened with ``check_same_thread=False``).
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..types import Sighting

SCHEMA = """
CREATE TABLE IF NOT EXISTS objects(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    created_at REAL NOT NULL,
    confuser_group INTEGER
);
CREATE TABLE IF NOT EXISTS embeddings(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_id INTEGER NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
    cls_blob BLOB NOT NULL,
    patches_blob BLOB,
    source TEXT NOT NULL CHECK(source IN ('enrol','auto')),
    created_at REAL NOT NULL,
    CHECK(source = 'enrol' OR patches_blob IS NULL)
);
CREATE TABLE IF NOT EXISTS sightings(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_id INTEGER NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
    ts REAL NOT NULL,
    scene_label TEXT,
    scene_blob BLOB,
    bbox TEXT NOT NULL,
    conf REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_emb_obj ON embeddings(object_id);
CREATE INDEX IF NOT EXISTS idx_sight_obj_ts ON sightings(object_id, ts);
"""


class GalleryError(RuntimeError):
    pass


def _l2(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-8)


class GalleryStore:
    def __init__(self, db_path: str | Path = ":memory:", dim: int = 384, top_patches: int = 32) -> None:
        self.dim, self.top_patches = dim, top_patches
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)
        self._db.commit()
        self.E = np.zeros((0, dim), dtype=np.float32)
        self.owner = np.zeros((0,), dtype=np.int64)
        self.emb_ids = np.zeros((0,), dtype=np.int64)
        self.sources: list[str] = []
        self._reload()

    # ------------------------------------------------------------------ mirror
    def _reload(self) -> None:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, object_id, cls_blob, source FROM embeddings ORDER BY id").fetchall()
            if rows:
                self.E = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows]).astype(np.float32)
                self.dim = self.E.shape[1]
            else:
                self.E = np.zeros((0, self.dim), dtype=np.float32)
            self.emb_ids = np.array([r[0] for r in rows], dtype=np.int64)
            self.owner = np.array([r[1] for r in rows], dtype=np.int64)
            self.sources = [r[3] for r in rows]

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """Consistent ``(E, owner)`` pair (arrays are replaced, never mutated in place)."""
        with self._lock:
            return self.E, self.owner

    # ----------------------------------------------------------------- objects
    def add_object(self, name: str) -> int:
        name = name.strip()
        if not name:
            raise GalleryError("object name must not be empty")
        with self._lock:
            try:
                cur = self._db.execute("INSERT INTO objects(name, created_at) VALUES(?, ?)",
                                       (name, time.time()))
            except sqlite3.IntegrityError as exc:
                raise GalleryError(f"object {name!r} already exists") from exc
            self._db.commit()
            return int(cur.lastrowid)

    def get_id(self, name: str) -> Optional[int]:
        with self._lock:
            r = self._db.execute("SELECT id FROM objects WHERE name = ?", (name.strip(),)).fetchone()
            return int(r[0]) if r else None

    def get_name(self, object_id: int) -> Optional[str]:
        with self._lock:
            r = self._db.execute("SELECT name FROM objects WHERE id = ?", (object_id,)).fetchone()
            return r[0] if r else None

    def list_objects(self) -> list[tuple[int, str]]:
        with self._lock:
            return [(int(i), n) for i, n in
                    self._db.execute("SELECT id, name FROM objects ORDER BY id").fetchall()]

    def names(self) -> list[str]:
        return [n for _, n in self.list_objects()]

    def delete_object(self, name: str) -> bool:
        with self._lock:
            oid = self.get_id(name)
            if oid is None:
                return False
            self._db.execute("DELETE FROM embeddings WHERE object_id = ?", (oid,))
            self._db.execute("DELETE FROM sightings WHERE object_id = ?", (oid,))
            self._db.execute("DELETE FROM objects WHERE id = ?", (oid,))
            self._db.commit()
            self._reload()
            return True

    # ------------------------------------------------------------- confusers
    def set_confusers(self, name_a: str, name_b: str) -> int:
        """Place two objects in the same confuser group (merging existing groups)."""
        with self._lock:
            a, b = self.get_id(name_a), self.get_id(name_b)
            if a is None or b is None:
                raise GalleryError("both objects must exist to be marked as confusers")
            if a == b:
                raise GalleryError("an object cannot be a confuser of itself")
            ga, gb = self.confuser_group(a), self.confuser_group(b)
            if ga is None and gb is None:
                g = int(self._db.execute("SELECT COALESCE(MAX(confuser_group),0)+1 FROM objects").fetchone()[0])
            else:
                g = ga if ga is not None else gb
            for gx in {ga, gb} - {None}:
                self._db.execute("UPDATE objects SET confuser_group=? WHERE confuser_group=?", (g, gx))
            self._db.execute("UPDATE objects SET confuser_group=? WHERE id IN (?,?)", (g, a, b))
            self._db.commit()
            return g

    def confuser_group(self, object_id: int) -> Optional[int]:
        with self._lock:
            r = self._db.execute("SELECT confuser_group FROM objects WHERE id=?", (object_id,)).fetchone()
            return r[0] if r and r[0] is not None else None

    def are_confusers(self, a: int, b: int) -> bool:
        ga, gb = self.confuser_group(a), self.confuser_group(b)
        return ga is not None and ga == gb and a != b

    # ------------------------------------------------------------ embeddings
    def add_embedding(self, object_id: int, cls: np.ndarray, source: str,
                      patches: Optional[np.ndarray] = None) -> int:
        if source not in ("enrol", "auto"):
            raise GalleryError("source must be 'enrol' or 'auto'")
        if source == "auto" and patches is not None:
            raise GalleryError("auto embeddings must not store patch tokens")
        cls = _l2(np.asarray(cls, dtype=np.float32).ravel())
        if self.E.shape[0] and cls.shape[0] != self.E.shape[1]:
            raise GalleryError(f"embedding dim {cls.shape[0]} != gallery dim {self.E.shape[1]}")
        blob_p = None
        if patches is not None:
            p = np.asarray(patches, dtype=np.float32)
            if p.shape[0] > self.top_patches:
                idx = np.sort(np.argsort(-np.linalg.norm(p, axis=1))[: self.top_patches])
                p = p[idx]
            blob_p = np.ascontiguousarray(p).tobytes()
        with self._lock:
            if self.get_name(object_id) is None:
                raise GalleryError(f"unknown object id {object_id}")
            cur = self._db.execute(
                "INSERT INTO embeddings(object_id, cls_blob, patches_blob, source, created_at) "
                "VALUES(?,?,?,?,?)", (object_id, cls.tobytes(), blob_p, source, time.time()))
            self._db.commit()
            if self.E.shape[0] == 0:
                self.E = cls[None, :].copy()
            else:
                self.E = np.vstack([self.E, cls[None, :]])
            self.owner = np.append(self.owner, object_id)
            self.emb_ids = np.append(self.emb_ids, cur.lastrowid)
            self.sources.append(source)
            self.dim = self.E.shape[1]
            return int(cur.lastrowid)

    def delete_embedding(self, emb_id: int, allow_enrol: bool = False) -> bool:
        """Delete one embedding. ``'enrol'`` rows are protected unless ``allow_enrol``."""
        with self._lock:
            r = self._db.execute("SELECT source FROM embeddings WHERE id=?", (emb_id,)).fetchone()
            if r is None:
                return False
            if r[0] == "enrol" and not allow_enrol:
                raise GalleryError("enrolment embeddings are never removed automatically")
            self._db.execute("DELETE FROM embeddings WHERE id=?", (emb_id,))
            self._db.commit()
            self._reload()
            return True

    def embeddings_of(self, object_id: int, source: Optional[str] = None) -> np.ndarray:
        with self._lock:
            mask = self.owner == object_id
            if source is not None:
                mask &= np.array([s == source for s in self.sources], dtype=bool) if self.sources else mask
            return self.E[mask]

    def embedding_rows(self, object_id: int, source: Optional[str] = None) -> list[tuple[int, str, np.ndarray]]:
        """``(embedding_id, source, vector)`` rows for an object."""
        with self._lock:
            out = []
            for i in np.nonzero(self.owner == object_id)[0]:
                if source is None or self.sources[i] == source:
                    out.append((int(self.emb_ids[i]), self.sources[i], self.E[i]))
            return out

    def count(self, object_id: int, source: Optional[str] = None) -> int:
        return len(self.embedding_rows(object_id, source))

    def patches_of(self, object_id: int) -> list[np.ndarray]:
        """Stored patch-token sets (enrolment views only) for an object."""
        with self._lock:
            rows = self._db.execute(
                "SELECT patches_blob FROM embeddings WHERE object_id=? AND source='enrol' "
                "AND patches_blob IS NOT NULL ORDER BY id", (object_id,)).fetchall()
        return [np.frombuffer(r[0], dtype=np.float32).reshape(-1, self.dim) for r in rows]

    def prototype(self, object_id: int) -> Optional[np.ndarray]:
        """Mean embedding of an object, re-normalised; ``None`` if it has none."""
        e = self.embeddings_of(object_id)
        return None if e.shape[0] == 0 else _l2(e.mean(axis=0))

    def enrol_prototype(self, object_id: int) -> Optional[np.ndarray]:
        e = self.embeddings_of(object_id, "enrol")
        return None if e.shape[0] == 0 else _l2(e.mean(axis=0))

    def prototypes(self) -> tuple[np.ndarray, np.ndarray]:
        """``(object_ids (K,), prototype matrix (K,D))`` for objects with embeddings."""
        ids, vecs = [], []
        for oid, _ in self.list_objects():
            p = self.prototype(oid)
            if p is not None:
                ids.append(oid)
                vecs.append(p)
        if not ids:
            return np.zeros((0,), np.int64), np.zeros((0, self.dim), np.float32)
        return np.array(ids, dtype=np.int64), np.stack(vecs).astype(np.float32)

    # -------------------------------------------------------------- sightings
    def add_sighting(self, s: Sighting, scene_blob: Optional[bytes] = None) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO sightings(object_id, ts, scene_label, scene_blob, bbox, conf) VALUES(?,?,?,?,?,?)",
                (s.object_id, s.ts, s.scene_label, scene_blob, ",".join(f"{v:.1f}" for v in s.bbox), s.conf))
            self._db.commit()
            return int(cur.lastrowid)

    def last_sighting(self, object_id: int) -> Optional[Sighting]:
        with self._lock:
            r = self._db.execute(
                "SELECT s.object_id, o.name, s.ts, s.scene_label, s.bbox, s.conf FROM sightings s "
                "JOIN objects o ON o.id = s.object_id WHERE s.object_id=? ORDER BY s.ts DESC, s.id DESC LIMIT 1",
                (object_id,)).fetchone()
        if r is None:
            return None
        bbox = tuple(float(x) for x in r[4].split(","))
        return Sighting(int(r[0]), r[1], float(r[2]), r[3], bbox, float(r[5]))  # type: ignore[arg-type]

    def sighting_count(self, object_id: Optional[int] = None) -> int:
        with self._lock:
            if object_id is None:
                return int(self._db.execute("SELECT COUNT(*) FROM sightings").fetchone()[0])
            return int(self._db.execute("SELECT COUNT(*) FROM sightings WHERE object_id=?",
                                        (object_id,)).fetchone()[0])

    def close(self) -> None:
        with self._lock:
            try:
                self._db.close()
            except sqlite3.Error:
                pass
