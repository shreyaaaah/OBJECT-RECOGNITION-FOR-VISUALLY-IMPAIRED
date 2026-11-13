"""Text-to-speech with a nonblocking, interruptible, priority-aware worker.

``Speaker.say`` only enqueues and returns immediately, so vision never blocks.
Rules enforced by the worker:

* a queued item is played highest priority first (FIFO within a priority);
* low-priority items older than ``stale_s`` are dropped without being spoken;
* an item with a HIGHER priority than the one currently being spoken interrupts
  it (when the backend supports ``stop()``);
* if the engine fails (or is missing) the text goes to the console fallback and
  the failure is logged -- speech is never silently reported as successful.
"""
from __future__ import annotations

import abc
import heapq
import itertools
import logging
import threading
import time
from typing import Callable, Optional

log = logging.getLogger(__name__)


class TTSBackend(abc.ABC):
    name = "backend"

    @abc.abstractmethod
    def speak(self, text: str) -> None:
        """Speak ``text`` (may block until finished)."""

    def stop(self) -> None:
        """Interrupt the current utterance (best effort)."""

    def close(self) -> None:
        return None


class ConsoleBackend(TTSBackend):
    """Explicit fallback: prints instead of speaking."""
    name = "console"

    def __init__(self, printer: Callable[[str], None] = print) -> None:
        self._print = printer

    def speak(self, text: str) -> None:
        self._print(f"[speech] {text}")


class Pyttsx3Backend(TTSBackend):
    name = "pyttsx3"

    def __init__(self) -> None:
        import pyttsx3  # raises ImportError if unavailable -> caller falls back
        self._engine = pyttsx3.init()

    def speak(self, text: str) -> None:
        self._engine.say(text)
        self._engine.runAndWait()

    def stop(self) -> None:
        try:
            self._engine.stop()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        self.stop()


def make_backend() -> TTSBackend:
    """pyttsx3 if it initialises, otherwise the console fallback (with a warning)."""
    try:
        return Pyttsx3Backend()
    except Exception as exc:  # noqa: BLE001 - ImportError, missing espeak, no audio device...
        log.warning("offline TTS unavailable (%s: %s); using console fallback -- nothing will be spoken aloud",
                    type(exc).__name__, exc)
        return ConsoleBackend()


class Speaker:
    def __init__(self, backend: Optional[TTSBackend] = None, stale_s: float = 2.0,
                 stale_below_priority: int = 20, clock: Callable[[], float] = time.monotonic,
                 max_queue: int = 32) -> None:
        self.backend = backend or ConsoleBackend()
        self.stale_s, self.stale_below, self.clock, self.max_queue = stale_s, stale_below_priority, clock, max_queue
        self._heap: list[tuple[int, int, float, str]] = []
        self._counter = itertools.count()
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._current_priority: Optional[int] = None
        self.spoken: list[str] = []
        self.dropped_stale = 0
        self.interrupted = 0
        self.errors = 0
        self._thread = threading.Thread(target=self._run, name="tts", daemon=True)
        self._thread.start()

    def say(self, text: str, priority: int = 10) -> bool:
        """Enqueue ``text``. Never blocks. Returns False if it was dropped (queue full)."""
        if not text or not text.strip():
            return False
        with self._cv:
            if len(self._heap) >= self.max_queue:
                # drop the lowest-priority queued item to make room, if the new one outranks it
                worst = max(self._heap, key=lambda e: (-e[0], e[1]))
                if -worst[0] >= priority:
                    return False
                self._heap.remove(worst)
                heapq.heapify(self._heap)
            heapq.heappush(self._heap, (-priority, next(self._counter), self.clock(), text))
            interrupt = self._current_priority is not None and priority > self._current_priority
            self._cv.notify()
        if interrupt:
            self.interrupted += 1
            try:
                self.backend.stop()
            except Exception:  # noqa: BLE001
                log.exception("TTS stop failed")
        return True

    def clear(self, below_priority: Optional[int] = None) -> None:
        with self._cv:
            self._heap = [e for e in self._heap if below_priority is not None and -e[0] >= below_priority]
            heapq.heapify(self._heap)

    def pending(self) -> int:
        with self._cv:
            return len(self._heap)

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                while not self._heap and not self._stop.is_set():
                    self._cv.wait(timeout=0.1)
                if self._stop.is_set():
                    return
                neg_p, _, t_enq, text = heapq.heappop(self._heap)
                prio = -neg_p
                if prio < self.stale_below and self.clock() - t_enq > self.stale_s:
                    self.dropped_stale += 1
                    continue
                self._current_priority = prio
            try:
                self.backend.speak(text)
                self.spoken.append(text)
            except Exception:  # noqa: BLE001
                self.errors += 1
                log.exception("TTS backend failed; falling back to console for this utterance")
                print(f"[speech] {text}")
            finally:
                with self._cv:
                    self._current_priority = None

    def wait_idle(self, timeout: float = 5.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            with self._cv:
                if not self._heap and self._current_priority is None:
                    return True
            time.sleep(0.01)
        return False

    def close(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        self._thread.join(timeout=2.0)
        try:
            self.backend.close()
        except Exception:  # noqa: BLE001
            pass
