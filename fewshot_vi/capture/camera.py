"""Threaded camera capture with a drop-oldest bounded queue."""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

log = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or stops delivering frames."""


class Camera:
    """Grab frames on a background thread.

    A ``Queue(maxsize=2)`` holds the newest frames; when it is full the OLDEST
    frame is discarded so latency never accumulates. ``capture_factory`` lets
    tests inject a fake capture object exposing ``isOpened/read/release/set``.
    """

    def __init__(self, index: int | str = 0, width: int = 640, height: int = 480,
                 target_fps: int = 30, capture_factory: Optional[Callable[[Any], Any]] = None) -> None:
        self.index, self.width, self.height, self.target_fps = index, width, height, target_fps
        self._factory = capture_factory
        self._cap: Any = None
        self._q: "queue.Queue[tuple[np.ndarray, float]]" = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.dropped = 0
        self.error: Optional[str] = None
        self.finished = threading.Event()  # set when a finite source (video file / fake) ends

    def start(self) -> "Camera":
        if self._factory is not None:
            self._cap = self._factory(self.index)
        else:
            import cv2
            self._cap = cv2.VideoCapture(self.index)
        if self._cap is None or not self._cap.isOpened():
            raise CameraError(f"cannot open camera {self.index!r}; check that it is connected, "
                              "not in use by another app, and that you have permission to use it")
        try:
            import cv2
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        except Exception:  # fake captures / unsupported props are fine
            pass
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()
        return self

    def _put_drop_oldest(self, item: tuple[np.ndarray, float]) -> None:
        while True:
            try:
                self._q.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    def _run(self) -> None:
        fails = 0
        while not self._stop.is_set():
            try:
                ok, frame = self._cap.read()
            except Exception as exc:  # noqa: BLE001 - device errors vary
                log.warning("camera read raised: %s", exc)
                ok, frame = False, None
            if not ok or frame is None:
                fails += 1
                if fails >= 30:
                    self.error = "camera stopped delivering frames"
                    log.error(self.error)
                    break
                time.sleep(0.01)
                continue
            fails = 0
            self._put_drop_oldest((frame, time.perf_counter()))
        self.finished.set()

    def read(self, timeout: float = 1.0) -> Optional[tuple[np.ndarray, float]]:
        """Return the next ``(frame, timestamp)`` or ``None`` on timeout."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001
                pass
            self._cap = None
