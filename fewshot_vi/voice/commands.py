"""Rule-based intent parser with fuzzy matching of verbs and gallery names."""
from __future__ import annotations

import difflib
import enum
import re
from dataclasses import dataclass
from typing import Optional, Sequence


class Intent(str, enum.Enum):
    IDENTIFY = "IDENTIFY"
    FIND = "FIND"
    MEMORY_QUERY = "MEMORY_QUERY"
    ENROL = "ENROL"
    DELETE = "DELETE"
    LIST = "LIST"
    MUTE = "MUTE"
    RESUME = "RESUME"
    CONFUSER = "CONFUSER"
    YES = "YES"
    NO = "NO"
    UNKNOWN = "UNKNOWN"


@dataclass
class Command:
    intent: Intent
    name: Optional[str] = None       # resolved gallery name (None if it could not be resolved)
    raw_name: Optional[str] = None   # what the user actually said
    text: str = ""

    @property
    def needs_name(self) -> bool:
        return self.intent in (Intent.FIND, Intent.MEMORY_QUERY, Intent.DELETE, Intent.CONFUSER)


FILLERS = {"please", "hey", "um", "uh", "ok", "okay", "claude", "my", "the", "a", "an", "me",
           "can", "could", "you", "would", "now", "just"}
VERBS = ["find", "locate", "delete", "forget", "remove", "erase", "register", "enrol", "enroll",
         "identify", "list", "resume", "unmute", "mute", "quiet", "stop", "remember", "learn"]
YES = {"yes", "yeah", "yep", "yup", "sure", "correct", "right", "confirm", "ok", "okay"}
NO = {"no", "nope", "nah", "cancel", "negative", "don't", "dont"}

_PATTERNS: list[tuple[Intent, re.Pattern[str]]] = [
    (Intent.MEMORY_QUERY, re.compile(
        r"^(?:where did i (?:leave|put|see|place|keep|last see)|when did i (?:last )?(?:see|leave|use)|"
        r"last seen|where (?:was|were)) (?P<n>.+)$")),
    (Intent.CONFUSER, re.compile(
        r"^(?:this is |it is |it's |its )?(?:similar to|like|a look ?alike (?:of|for)|confused with) (?P<n>.+)$")),
    (Intent.DELETE, re.compile(r"^(?:delete|forget|remove|erase) (?P<n>.+)$")),
    (Intent.FIND, re.compile(
        r"^(?:find|locate|look for|search for|where is|where's|wheres|where are|show me) (?P<n>.+)$")),
    (Intent.ENROL, re.compile(
        r"^(?:register|add|enrol|enroll|remember|learn)(?: new)?(?: this)?(?: object| item| thing)?$")),
    (Intent.IDENTIFY, re.compile(
        r"^(?:what is this|what is it|whats this|what's this|identify(?: this| it)?|what am i holding|"
        r"what do you see)$")),
    (Intent.LIST, re.compile(r"^(?:list(?: objects| items)?|what objects do i have|which objects do i have)$")),
    (Intent.MUTE, re.compile(r"^(?:stop|quiet|mute|be quiet|silence)$")),
    (Intent.RESUME, re.compile(r"^(?:resume|unmute|continue|start again|speak)$")),
]


def normalise(text: str) -> str:
    t = re.sub(r"[^\w\s']", " ", text.lower())
    words = [w for w in t.split() if w not in FILLERS]
    return " ".join(words)


def resolve_name(spoken: str, names: Sequence[str], cutoff: float = 0.6) -> Optional[str]:
    """Match what was said to a gallery name: exact, then containment, then fuzzy."""
    s = normalise(spoken)
    if not s or not names:
        return None
    lowered = {n.lower(): n for n in names}
    if s in lowered:
        return lowered[s]
    for low, orig in lowered.items():  # "blue mug please" / "mug"
        if low in s.split() or s in low.split():
            return orig
    close = difflib.get_close_matches(s, list(lowered), n=1, cutoff=cutoff)
    return lowered[close[0]] if close else None


def _fix_first_verb(s: str) -> str:
    parts = s.split(" ", 1)
    if parts[0] in VERBS or parts[0] in {"where", "what", "whats", "this", "it", "its"}:
        return s
    close = difflib.get_close_matches(parts[0], VERBS, n=1, cutoff=0.75)
    if close:
        return close[0] + (" " + parts[1] if len(parts) > 1 else "")
    return s


def _match(s: str) -> tuple[Optional[Intent], Optional[str]]:
    for intent, pat in _PATTERNS:
        m = pat.match(s)
        if m:
            return intent, (m.groupdict().get("n") if "n" in pat.groupindex else None)
    return None, None


def parse_command(text: str, gallery_names: Sequence[str] = (), cutoff: float = 0.6) -> Command:
    s = normalise(text)
    if not s:
        return Command(Intent.UNKNOWN, text=text)
    words = set(s.split())
    if s in YES or (len(words) <= 2 and words & YES):
        return Command(Intent.YES, text=text)
    if s in NO or (len(words) <= 2 and words & NO):
        return Command(Intent.NO, text=text)
    intent, raw = _match(s)
    if intent is None:
        intent, raw = _match(_fix_first_verb(s))
    if intent is None:
        return Command(Intent.UNKNOWN, text=text)
    if raw is None:
        return Command(intent, text=text)
    raw = raw.strip()
    return Command(intent, name=resolve_name(raw, gallery_names, cutoff), raw_name=raw, text=text)
