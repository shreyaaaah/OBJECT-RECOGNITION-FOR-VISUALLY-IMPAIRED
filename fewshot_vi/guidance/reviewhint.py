"""Active re-view hints: at most ONE concise instruction, rate-limited, never
repeated back-to-back."""
from __future__ import annotations

import time
from typing import Callable, Optional

from ..types import Decision, Quality

EDGE_HINTS = {
    "left": "Move the camera left.",
    "right": "Move the camera right.",
    "top": "Tilt the camera up.",
    "bottom": "Tilt the camera down.",
}
EDGE_ORDER = ("left", "right", "top", "bottom")
HINT_BLUR = "Hold the camera steady."
HINT_SMALL = "Move closer."
HINT_UNSURE = "Not sure. Try a different angle."


def choose_hint(q: Optional[Quality], decision: Optional[Decision]) -> Optional[str]:
    """Pick the single most useful hint for a quality record, or ``None``.

    Priority: edge cut > blur > too small > low margin / ambiguous.
    """
    if q is None:
        return None
    for side in EDGE_ORDER:
        if side in q.edge_cut:
            return EDGE_HINTS[side]
    if q.blurry:
        return HINT_BLUR
    if q.too_small:
        return HINT_SMALL
    if q.low_margin or decision == Decision.AMBIGUOUS:
        return HINT_UNSURE
    return None


class ReviewHinter:
    def __init__(self, cooldown_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.cooldown_s, self.clock = cooldown_s, clock
        self._last_text: Optional[str] = None
        self._last_t: Optional[float] = None

    def next_hint(self, q: Optional[Quality], decision: Optional[Decision]) -> Optional[str]:
        text = choose_hint(q, decision)
        if text is None:
            return None
        now = self.clock()
        if self._last_t is not None and now - self._last_t < self.cooldown_s:
            return None
        if text == self._last_text:
            return None  # never the same hint twice in a row
        self._last_text, self._last_t = text, now
        return text

    def note_success(self) -> None:
        """A confirmed recognition resets the repeat guard, so a later, genuinely
        new occurrence of the same problem can be announced again."""
        self._last_text = None
