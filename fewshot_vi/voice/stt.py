"""Speech input behind an interface.

* ``KeyboardSTT``  - always available; reads a typed line (the documented fallback).
* ``MicrophoneSTT``- real microphone + ``faster-whisper`` when the optional
  dependencies and an input device exist. Records until silence (energy VAD) or
  a maximum duration after ``listen`` is called (push-to-talk friendly).
* ``make_stt``     - picks the microphone implementation if usable, otherwise the
  keyboard, and logs which one is active (never pretends a mic worked).

``VoiceListener`` runs ``listen`` on its own thread and delivers recognised text
to a callback, so voice input never blocks the vision loop.
"""
from __future__ import annotations

import abc
import logging
import queue
import sys
import threading
import time
from typing import Callable, Optional, TextIO

import numpy as np

log = logging.getLogger(__name__)


class STT(abc.ABC):
    name = "stt"

    @abc.abstractmethod
    def listen(self, prompt: Optional[str] = None, timeout: Optional[float] = None) -> Optional[str]:
        """Return the next utterance as text, or ``None`` on timeout / end of input."""

    def close(self) -> None:
        return None


class KeyboardSTT(STT):
    """Typed commands. Blocks in ``listen`` (so run it on a VoiceListener thread)."""
    name = "keyboard"

    def __init__(self, stream: Optional[TextIO] = None, printer: Callable[[str], None] = print) -> None:
        self._stream = stream
        self._print = printer
        self.eof = False

    def listen(self, prompt: Optional[str] = None, timeout: Optional[float] = None) -> Optional[str]:
        if prompt:
            self._print(f"[ask] {prompt}")
        try:
            if self._stream is not None:
                # Injected streams (used by tests and embedders) retain normal
                # readline semantics.  The real console path polls so the
                # listener can honour its timeout and stop request.
                line = self._stream.readline()
            elif timeout is None:
                line = input("> ")
            else:
                line = self._read_console(timeout)
        except EOFError:
            self.eof = True
            return None
        if line is None:
            return None
        if line == "":
            self.eof = True
            return None
        return line.strip() or None

    def _read_console(self, timeout: float) -> Optional[str]:
        """Read one console line while polling, returning None on timeout.

        Windows' ``select`` does not support console handles, so use msvcrt
        there.  POSIX terminals use ``select`` and still fall back to input
        for unusual non-file stdin objects.
        """
        sys.stdout.write("> ")
        sys.stdout.flush()
        deadline = time.monotonic() + max(0.0, timeout)
        chars: list[str] = []
        if sys.platform == "win32":
            import msvcrt
            while time.monotonic() < deadline:
                if not msvcrt.kbhit():
                    time.sleep(0.02)
                    continue
                ch = msvcrt.getwch()
                if ch in ("\r", "\n"):
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    return "".join(chars)
                if ch == "\x03":  # Ctrl+C
                    raise KeyboardInterrupt
                if ch == "\b":
                    if chars:
                        chars.pop()
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                elif ch and ch >= " ":
                    chars.append(ch)
                    sys.stdout.write(ch)
                    sys.stdout.flush()
            return None
        try:
            import select
            while time.monotonic() < deadline:
                ready, _, _ = select.select([sys.stdin], [], [], min(0.05, deadline - time.monotonic()))
                if ready:
                    return sys.stdin.readline()
            return None
        except (OSError, ValueError):
            # A redirected or non-selectable console cannot be polled portably.
            return input("> ")

class MicrophoneSTT(STT):
    name = "microphone"

    def __init__(self, model_size: str = "base.en", sample_rate: int = 16000, max_seconds: float = 6.0,
                 silence_s: float = 0.8, energy_threshold: float = 0.01) -> None:
        try:
            import sounddevice as sd  # noqa: F401
            from faster_whisper import WhisperModel
        except Exception as exc:  # ImportError or PortAudio OSError
            raise RuntimeError(f"microphone STT unavailable: {exc}") from exc
        import sounddevice as sd
        try:
            sd.check_input_settings(samplerate=sample_rate, channels=1)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"no usable audio input device: {exc}") from exc
        self.sample_rate, self.max_seconds = sample_rate, max_seconds
        self.silence_s, self.energy = silence_s, energy_threshold
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def _record(self, timeout: Optional[float]) -> Optional[np.ndarray]:
        import sounddevice as sd
        block = int(self.sample_rate * 0.1)
        chunks: list[np.ndarray] = []
        started, quiet, total = False, 0.0, 0.0
        limit = min(self.max_seconds, timeout) if timeout else self.max_seconds
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32", blocksize=block) as s:
            while total < limit:
                data, _ = s.read(block)
                total += 0.1
                loud = float(np.sqrt(np.mean(data ** 2))) > self.energy
                if loud:
                    started, quiet = True, 0.0
                elif started:
                    quiet += 0.1
                if started:
                    chunks.append(data[:, 0].copy())
                if started and quiet >= self.silence_s:
                    break
        return np.concatenate(chunks) if chunks else None

    def listen(self, prompt: Optional[str] = None, timeout: Optional[float] = None) -> Optional[str]:
        audio = self._record(timeout)
        if audio is None:
            return None
        segments, _ = self._model.transcribe(audio, language="en", beam_size=1)
        text = " ".join(s.text.strip() for s in segments).strip()
        return text or None


def make_stt(prefer_microphone: bool = True) -> STT:
    if prefer_microphone:
        try:
            stt = MicrophoneSTT()
            log.info("speech input: microphone (faster-whisper)")
            return stt
        except Exception as exc:  # noqa: BLE001
            log.warning("%s; falling back to keyboard input (type commands, e.g. 'find keys')", exc)
    return KeyboardSTT()


class VoiceListener:
    """Runs ``stt.listen`` on a daemon thread; each utterance goes to ``on_text``."""

    def __init__(self, stt: STT, on_text: Callable[[str], None]) -> None:
        self.stt, self.on_text = stt, on_text
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="voice", daemon=True)

    def start(self) -> "VoiceListener":
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                text = self.stt.listen(timeout=2.0)
            except Exception:  # noqa: BLE001
                log.exception("speech input failed")
                self._stop.wait(0.5)
                continue
            if getattr(self.stt, "eof", False):
                break
            if text:
                try:
                    self.on_text(text)
                except Exception:  # noqa: BLE001
                    log.exception("voice command handler failed")

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            # Console input polls at short intervals; microphone reads are
            # bounded by the listener's two-second listen timeout.
            self._thread.join(timeout=3.0)


class ReplyQueue:
    """Thread-safe single-slot mailbox so an in-progress dialogue (enrolment,
    delete confirmation) can ask a question and wait for the next utterance."""

    def __init__(self) -> None:
        self._q: "queue.Queue[str]" = queue.Queue()
        self.waiting = threading.Event()

    def put(self, text: str) -> None:
        self._q.put(text)

    def get(self, timeout: float) -> Optional[str]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return
