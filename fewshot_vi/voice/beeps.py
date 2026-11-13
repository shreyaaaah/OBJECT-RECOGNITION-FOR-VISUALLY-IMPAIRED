"""Stereo-panned proximity beeps with graceful fallback when audio output is unavailable."""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)


def make_beep(pan: float, urgency: float, sample_rate: int = 22050, duration_s: float = 0.08,
              base_hz: float = 880.0, near_boost_hz: float = 400.0) -> np.ndarray:
    """Stereo (N,2) float32 sine burst. Pitch rises with ``urgency`` (0 far .. 1 very close);
    equal-power panning places it left (-1) .. right (+1)."""
    urgency = float(np.clip(urgency, 0.0, 1.0))
    pan = float(np.clip(pan, -1.0, 1.0))
    hz = base_hz + near_boost_hz * urgency
    n = max(1, int(sample_rate * duration_s))
    t = np.arange(n) / sample_rate
    env = np.minimum(1.0, np.minimum(t, duration_s - t) / 0.01)  # 10 ms fade in/out avoids clicks
    tone = (0.4 * np.sin(2 * np.pi * hz * t) * env).astype(np.float32)
    theta = (pan + 1.0) * np.pi / 4.0
    return np.stack([tone * np.cos(theta), tone * np.sin(theta)], axis=1).astype(np.float32)


class Beeper:
    def __init__(self, sample_rate: int = 22050, player: Optional[Callable[[np.ndarray, int], None]] = None) -> None:
        self.sample_rate = sample_rate
        self.available = True
        self.played = 0
        if player is not None:
            self._player = player
        else:
            self._player = self._sd_player()

    def _sd_player(self) -> Optional[Callable[[np.ndarray, int], None]]:
        try:
            import sounddevice as sd
            sd.check_output_settings(samplerate=self.sample_rate, channels=2)

            def play(data: np.ndarray, sr: int) -> None:
                sd.play(data, sr, blocking=False)
            return play
        except Exception as exc:  # noqa: BLE001 - ImportError, PortAudio OSError, no device
            log.warning("beeps disabled: audio output unavailable (%s)", exc)
            self.available = False
            return None

    def beep(self, pan: float, urgency: float) -> bool:
        """Play one beep; returns False (and does nothing) when audio output is unavailable."""
        if self._player is None:
            return False
        try:
            self._player(make_beep(pan, urgency, self.sample_rate), self.sample_rate)
            self.played += 1
            return True
        except Exception:  # noqa: BLE001
            log.exception("beep playback failed; disabling beeps")
            self._player, self.available = None, False
            return False
