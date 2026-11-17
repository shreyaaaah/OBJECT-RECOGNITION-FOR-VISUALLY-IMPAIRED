"""Guidance engine: turns smoothed tracks + mode into prioritised ``GuidanceEvent``s.

Safety principle: only SMOOTHED, confirmed ACCEPT results (``track.smoothed_name``)
are ever named. Anything else yields at most an active re-view hint such as
"Not sure. Try a different angle." -- a false positive is worse than a miss.

Priorities (higher wins; see ``voice.tts``): SYSTEM 40 > ANSWER 30 > FIND 20 >
ANNOUNCE 10 > HINT 5 > BEEP 1. ``pan`` is -1 (left) .. +1 (right) everywhere.
"""
from __future__ import annotations

import time
from typing import Callable, Optional, Sequence

from ..types import Decision, GuidanceEvent, Mode, Track
from . import geometry
from .reviewhint import ReviewHinter

PRIORITY_SYSTEM = 40
PRIORITY_ANSWER = 30
PRIORITY_FIND = 20
PRIORITY_ANNOUNCE = 10
PRIORITY_HINT = 5
PRIORITY_BEEP = 1


class GuidanceEngine:
    def __init__(self, cfg, memory_answer: Optional[Callable[[str], str]] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg, self.memory_answer, self.clock = cfg, memory_answer, clock
        self.hinter = ReviewHinter(cfg.guidance.hint_cooldown_s, clock)
        self._announced: dict[str, float] = {}
        self._last_find_say: float = -1e9
        self._last_beep: float = -1e9
        self._find_key: Optional[tuple] = None
        self._last_visible: Optional[float] = None
        self._memory_done = False

    # ------------------------------------------------------------------ helpers
    def _describe(self, t: Track, frame_shape: tuple[int, ...]):
        g, gp = self.cfg.guidance, self.cfg.guidance_policy
        fh, fw = frame_shape[:2]
        d = geometry.direction(t.bbox, fw, fh, gp.near_zone_frac, gp.slight_zone_frac)
        dist = geometry.distance_m(t.bbox[2] - t.bbox[0], fw, g.camera_hfov_deg,
                                   g.known_object_width_cm_default)
        return d, dist

    @staticmethod
    def _text(name: str, d: geometry.Direction, dist: float) -> str:
        parts = [name[:1].upper() + name[1:], d.horizontal]
        if d.vertical != "centre":
            parts.append(d.vertical)
        parts.append(geometry.distance_phrase(dist))
        return ", ".join(parts) + "."

    def _best_hint_track(self, tracks: Sequence[Track], frame_shape, only_name: Optional[str] = None):
        """The largest unconfirmed track with a plausible (ACCEPT/AMBIGUOUS) candidate.
        UNKNOWN tracks are ignored so random background objects never trigger hints."""
        best, best_area = None, -1.0
        for t in tracks:
            if t.class_name is not None:
                continue  # a general category is already available; no gallery re-view hint needed
            if t.smoothed_name is not None:
                continue
            if t.rerank_pending:  # the look-alike re-rank may still resolve this; don't ask for a re-view yet
                continue
            if t.match is None or t.match.object_id is None:
                continue
            if t.match.decision == Decision.UNKNOWN:  # not a plausible candidate: stay silent
                continue
            if only_name is not None and (t.match.name or "").lower() != only_name.lower():
                continue
            area = (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1])
            if area > best_area:
                best, best_area = t, area
        return best

    # --------------------------------------------------------------------- main
    def process(self, tracks: Sequence[Track], mode: Mode, target: Optional[str],
                frame_shape: tuple[int, ...], find_started: Optional[float] = None,
                muted: bool = False) -> list[GuidanceEvent]:
        if muted or mode == Mode.ENROL:
            return []
        now = self.clock()
        gcfg, pol = self.cfg.guidance, self.cfg.guidance_policy
        events: list[GuidanceEvent] = []

        if mode == Mode.FIND and target:
            key = (target.lower(), find_started)
            if key != self._find_key:  # new FIND session
                self._find_key, self._memory_done = key, False
                self._last_visible = find_started if find_started is not None else now
                self._last_find_say, self._last_beep = -1e9, -1e9
            hits = [t for t in tracks if t.smoothed_name and t.smoothed_name.lower() == target.lower()]
            if hits:
                t = max(hits, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                self._last_visible, self._memory_done = now, False
                self.hinter.note_success()
                d, dist = self._describe(t, frame_shape)
                if now - self._last_find_say >= gcfg.announce_cooldown_s:
                    self._last_find_say = now
                    events.append(GuidanceEvent("find", self._text(t.smoothed_name, d, dist),
                                                PRIORITY_FIND, d.pan, 0.0, now))
                span = max(pol.beep_far_m - pol.beep_near_m, 1e-6)
                urgency = min(1.0, max(0.0, (pol.beep_far_m - dist) / span))
                interval = pol.beep_max_interval_s - urgency * (pol.beep_max_interval_s - pol.beep_min_interval_s)
                if now - self._last_beep >= interval:
                    self._last_beep = now
                    events.append(GuidanceEvent("beep", "", PRIORITY_BEEP, d.pan, urgency, now))
                return events
            # target not visible: active hints, then memory after the wait
            if self.cfg.features.active_hints:
                t = self._best_hint_track(tracks, frame_shape, only_name=target)
                if t is not None:
                    text = self.hinter.next_hint(t.quality, t.match.decision if t.match else None)
                    if text:
                        events.append(GuidanceEvent("hint", text, PRIORITY_HINT, 0.0, 0.0, now))
            waited = now - (self._last_visible if self._last_visible is not None else now)
            if (self.cfg.features.memory and self.memory_answer and not self._memory_done
                    and waited >= pol.find_memory_wait_s):
                self._memory_done = True
                events.append(GuidanceEvent("memory", self.memory_answer(target), PRIORITY_FIND, 0.0, 0.0, now))
            return events

        # ---- IDENTIFY -------------------------------------------------------
        for t in tracks:
            if not t.smoothed_name:
                continue
            key = t.smoothed_name.lower()
            last = self._announced.get(key)
            if last is not None and now - last < gcfg.announce_cooldown_s:
                continue
            self._announced[key] = now
            self.hinter.note_success()
            d, dist = self._describe(t, frame_shape)
            events.append(GuidanceEvent("announce", self._text(t.smoothed_name, d, dist),
                                        PRIORITY_ANNOUNCE, d.pan, 0.0, now))
        if not events and self.cfg.features.active_hints:
            t = self._best_hint_track(tracks, frame_shape)
            if t is not None:
                text = self.hinter.next_hint(t.quality, t.match.decision if t.match else None)
                if text:
                    events.append(GuidanceEvent("hint", text, PRIORITY_HINT, 0.0, 0.0, now))
        return events
