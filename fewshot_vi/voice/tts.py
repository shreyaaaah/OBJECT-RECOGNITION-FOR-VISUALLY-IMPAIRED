"""Text-to-speech with a nonblocking, interruptible, priority-aware worker.

``Speaker.say`` only enqueues and returns immediately, so vision never blocks.
Rules enforced by the worker:

* a queued item is played highest priority first (FIFO within a priority);
* low-priority items older than ``stale_s`` are dropped without being spoken;
* an item with a HIGHER priority than the one currently being spoken interrupts
  it when the backend supports thread-safe ``stop()``; thread-affine backends
  finish the current utterance before playing queued urgent speech;
* if the engine fails (or is missing) the text goes to the console fallback and
  the failure is logged -- speech is never silently reported as successful.
"""
from __future__ import annotations

import abc
import base64
import heapq
import itertools
import logging
import queue
import shutil
import sys
import subprocess
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
    thread_affine = True

    def __init__(self) -> None:
        # SAPI5/COM engines must be created and driven from the same apartment.
        # Speaker owns one worker thread, so defer initialization to that thread.
        import pyttsx3  # raises ImportError if unavailable -> caller falls back
        self._pyttsx3 = pyttsx3
        self._engine = None
        self._owner_thread: Optional[int] = None
        self._pythoncom = None

    def start_worker(self) -> None:
        if self._engine is not None:
            return
        if sys.platform == "win32":
            import pythoncom
            pythoncom.CoInitialize()
            self._pythoncom = pythoncom
        self._owner_thread = threading.get_ident()
        try:
            self._engine = self._pyttsx3.init()
        except Exception:
            if self._pythoncom is not None:
                self._pythoncom.CoUninitialize()
                self._pythoncom = None
            self._owner_thread = None
            raise

    def speak(self, text: str) -> None:
        if self._engine is None or self._owner_thread != threading.get_ident():
            raise RuntimeError("pyttsx3 must be started and used on the speaker worker thread")
        self._engine.say(text)
        self._engine.runAndWait()

    def stop(self) -> None:
        # pyttsx3's SAPI COM object is thread-affine; never call it from the
        # vision/voice thread. The queued urgent utterance plays after this one.
        if self._owner_thread == threading.get_ident() and self._engine is not None:
            self._engine.stop()

    def close_worker(self) -> None:
        if self._owner_thread != threading.get_ident():
            return
        engine, self._engine = self._engine, None
        pythoncom, self._pythoncom = self._pythoncom, None
        self._owner_thread = None
        try:
            if engine is not None:
                engine.stop()
        finally:
            if pythoncom is not None:
                pythoncom.CoUninitialize()


class WindowsSystemSpeechBackend(TTSBackend):
    """Use Windows' built-in .NET System.Speech through a persistent PowerShell worker.

    This avoids pyttsx3's SAPI voice-enumeration failure on some current Windows
    installs while keeping PowerShell and the speech synthesizer off the vision
    thread. A line protocol transports base64-encoded UTF-8 text safely.
    """
    name = "windows-system-speech"
    thread_affine = True

    _SCRIPT = r"""
$ErrorActionPreference = 'Stop'
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    Add-Type -AssemblyName System.Speech
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    [Console]::WriteLine('@READY')
    while ($true) {
        $line = [Console]::ReadLine()
        if ($null -eq $line -or $line -eq '__QUIT__') { break }
        try {
            $bytes = [Convert]::FromBase64String($line)
            $text = [System.Text.Encoding]::UTF8.GetString($bytes)
            $synth.Speak($text)
            [Console]::WriteLine('@OK')
        } catch {
            [Console]::WriteLine('@ERR:' + $_.Exception.Message)
        }
    }
    $synth.Dispose()
} catch {
    [Console]::WriteLine('@INITERR:' + $_.Exception.Message)
    exit 1
}
"""

    def __init__(self) -> None:
        self._powershell = shutil.which("powershell.exe") or shutil.which("powershell")
        if self._powershell is None:
            raise RuntimeError("Windows PowerShell was not found")
        self._process: Optional[subprocess.Popen] = None
        self._output: queue.Queue[Optional[str]] = queue.Queue()
        self._reader: Optional[threading.Thread] = None

    def _read_output(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            self._output.put(None)
            return
        for line in process.stdout:
            self._output.put(line.rstrip("\r\n"))
        self._output.put(None)

    def _receive(self, expected: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Windows speech worker did not return {expected}")
            try:
                line = self._output.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError(f"Windows speech worker did not return {expected}") from exc
            if line is None:
                raise RuntimeError("Windows speech worker exited unexpectedly")
            if line == expected:
                return
            if line.startswith("@ERR:") or line.startswith("@INITERR:"):
                raise RuntimeError(line.partition(":")[2])
            if line.startswith("@"):
                raise RuntimeError(f"Unexpected Windows speech worker response: {line}")
            log.debug("Windows speech worker: %s", line)

    def start_worker(self) -> None:
        if self._process is not None:
            return
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._process = subprocess.Popen(
                [self._powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden",
                 "-ExecutionPolicy", "Bypass", "-Command", self._SCRIPT],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=flags)
            self._reader = threading.Thread(target=self._read_output, name="windows-speech-output", daemon=True)
            self._reader.start()
            self._receive("@READY", timeout=10.0)
        except Exception:
            self._terminate_process()
            raise

    def speak(self, text: str) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("Windows speech worker is not running")
        payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
        process.stdin.write(payload + "\n")
        process.stdin.flush()
        self._receive("@OK", timeout=max(30.0, len(text) / 8.0))

    def stop(self) -> None:
        # The synchronous System.Speech call is completed before the worker reads
        # its next command, so higher-priority messages play immediately after it.
        return None

    def _terminate_process(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        if self._reader is not None:
            self._reader.join(timeout=1.0)
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
        self._reader = None

    def close_worker(self) -> None:
        process = self._process
        if process is not None and process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write("__QUIT__\n")
                process.stdin.flush()
                process.wait(timeout=3.0)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self._terminate_process()


def make_backend() -> TTSBackend:
    """Choose the platform's offline speech backend, otherwise console fallback."""
    try:
        if sys.platform == "win32":
            return WindowsSystemSpeechBackend()
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
        if interrupt and not getattr(self.backend, "thread_affine", False):
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
        try:
            start_worker = getattr(self.backend, "start_worker", None)
            if start_worker is not None:
                try:
                    start_worker()
                except Exception as exc:  # noqa: BLE001 - audio/COM init can fail on the target PC
                    log.warning("offline TTS could not initialise on its worker (%s: %s); "
                                "using console output -- nothing will be spoken aloud",
                                type(exc).__name__, exc)
                    self.backend = ConsoleBackend()
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
        finally:
            close_worker = getattr(self.backend, "close_worker", None)
            if close_worker is not None:
                try:
                    close_worker()
                except Exception:  # noqa: BLE001
                    log.exception("TTS worker cleanup failed")

    def wait_idle(self, timeout: float = 5.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            with self._cv:
                if not self._heap and self._current_priority is None:
                    return True
            time.sleep(0.01)
        return False

    def close(self) -> None:
        # Give queued announcements a chance to play before process shutdown.
        self.wait_idle(3.0)
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        self._thread.join(timeout=2.0)
        close = getattr(self.backend, "close_worker", None)
        if close is None:
            try:
                self.backend.close()
            except Exception:  # noqa: BLE001
                pass
