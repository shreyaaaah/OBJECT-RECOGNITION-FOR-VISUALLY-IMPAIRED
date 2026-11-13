"""Object memory: confident sightings written through a writer queue, plus
structured and natural-language last-seen answers."""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from ..gallery.store import GalleryStore
from ..types import Sighting

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ formatting
def humanise_delta(seconds: float) -> str:
    """'just now' / 'N minutes ago' / 'N hours ago' / 'yesterday' / 'N days ago'."""
    s = max(0.0, seconds)
    if s < 60:
        return "just now"
    mins = int(s // 60)
    if mins < 60:
        return f"{mins} minute{'s' if mins != 1 else ''} ago"
    hours = int(s // 3600)
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = int(s // 86400)
    if days == 1:
        return "yesterday"
    return f"{days} days ago"


def format_last_seen(name: str, sighting: Optional[Sighting], now: float) -> str:
    """Natural-language answer to "where did I leave my <name>?"."""
    if sighting is None:
        return f"I haven't seen your {name} yet."
    when = humanise_delta(now - sighting.ts)
    where = f", {_with_article(sighting.scene_label)}" if sighting.scene_label else ""
    if when == "just now":
        return f"I just saw your {name}{where}."
    verb = "were" if name.lower().endswith("s") and not name.lower().endswith("ss") else "was"
    return f"Your {name} {verb} last seen {when}{where}."


def _with_article(scene: str) -> str:
    s = scene.strip()
    if s.lower().startswith(("on ", "in ", "at ", "near ")):
        return s
    return f"on the {s}" if s.lower() in {"desk", "table", "dining table", "sofa", "shelf", "bed", "floor"} \
        else f"in the {s}"


# ---------------------------------------------------------------------- memory
@dataclass
class _Entry:
    sighting: Sighting
    scene_blob: Optional[bytes]


class ObjectMemory:
    """Logs sightings above ``min_conf`` with a per-object cooldown. DB writes go
    through a bounded queue drained by a writer thread so SQLite never stalls the
    vision loop (``offer`` never blocks; if the queue is full the sighting is dropped)."""

    def __init__(self, store: GalleryStore, min_conf: float, cooldown_s: float,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
                 max_pending: int = 64) -> None:
        self.store, self.min_conf, self.cooldown_s = store, min_conf, cooldown_s
        self.clock, self.wall = clock, wall
        self._last_log: dict[int, float] = {}
        self._q: "queue.Queue[Optional[_Entry]]" = queue.Queue(maxsize=max_pending)
        self._thread = threading.Thread(target=self._run, name="db-writer", daemon=True)
        self._thread.start()
        self.dropped = 0

    def offer(self, object_id: int, name: str, conf: float, bbox: tuple[float, float, float, float],
              scene_label: Optional[str] = None, scene_blob: Optional[bytes] = None) -> bool:
        """Queue a sighting if it passes the confidence and cooldown checks."""
        if conf < self.min_conf:
            return False
        now = self.clock()
        last = self._last_log.get(object_id)
        if last is not None and now - last < self.cooldown_s:
            return False
        entry = _Entry(Sighting(object_id, name, self.wall(), scene_label, bbox, conf), scene_blob)
        try:
            self._q.put_nowait(entry)
        except queue.Full:
            self.dropped += 1
            return False
        self._last_log[object_id] = now
        return True

    def _run(self) -> None:
        while True:
            e = self._q.get()
            if e is None:
                self._q.task_done()
                break
            try:
                self.store.add_sighting(e.sighting, e.scene_blob)
            except Exception:  # noqa: BLE001
                log.exception("failed to write sighting")
            finally:
                self._q.task_done()

    def flush(self, timeout: float = 5.0) -> None:
        """Block until queued sightings are written (used by tests and at shutdown)."""
        end = time.time() + timeout
        while self._q.unfinished_tasks and time.time() < end:
            time.sleep(0.005)

    def last_seen(self, name: str) -> Optional[Sighting]:
        """Structured last sighting for ``name`` (``None`` if unknown or never seen)."""
        oid = self.store.get_id(name)
        return None if oid is None else self.store.last_sighting(oid)

    def answer(self, name: str, now: Optional[float] = None) -> str:
        now = self.wall() if now is None else now
        if self.store.get_id(name) is None:
            return f"I don't have an object called {name}."
        return format_last_seen(name, self.last_seen(name), now)

    def close(self) -> None:
        self.flush()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)
