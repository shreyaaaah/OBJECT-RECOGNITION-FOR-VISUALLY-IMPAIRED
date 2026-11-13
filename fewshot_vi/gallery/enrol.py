"""Guided few-shot enrolment: no gradient updates, just embeddings saved to the gallery.

Flow (``run_enrolment``):
1. ask the user to hold the object steadily; wait for a stable, large, central detection;
2. capture ``min_views``..``max_views`` quality-approved views within ``timeout_s``,
   prompting the user to rotate between views (blurred, edge-cut, small and
   near-duplicate views are rejected with a spoken hint);
3. embed each view plus ``augmentations_per_view`` deterministic augmentations;
4. ask for the object's name and confirm it; save everything as ``source='enrol'``;
5. offer to mark an existing object as a confuser.

All I/O is injected (``get_frame``, ``speak``, ``ask``, ``clock``) so the flow is
testable without hardware. With no microphone, ``ask`` is backed by keyboard input.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from ..embedder.dino_embedder import select_salient
from ..quality.quality import assess
from ..tracker.tracker import iou
from ..types import Detection
from .store import GalleryError, GalleryStore

log = logging.getLogger(__name__)

YES_WORDS = {"yes", "yeah", "yep", "correct", "right", "sure", "ok", "okay", "yup", "confirm"}


@dataclass
class EnrolResult:
    success: bool
    message: str
    name: Optional[str] = None
    views: int = 0
    embeddings: int = 0
    confuser_of: Optional[str] = None
    log: list[str] = field(default_factory=list)


def is_yes(text: Optional[str]) -> bool:
    return bool(text) and any(w in YES_WORDS for w in text.lower().replace(",", " ").split())


def augment(crop: np.ndarray, n: int, seed: int) -> list[np.ndarray]:
    """Deterministic augmentations (same ``seed`` -> identical output): horizontal
    flip, brightness/contrast, per-channel colour gain, mild crop+resize, mild blur,
    and combinations of those."""
    rng = np.random.default_rng(seed)
    h, w = crop.shape[:2]
    out: list[np.ndarray] = []
    for i in range(n):
        img = crop.copy()
        kind = i % 5
        if kind == 0:
            img = cv2.flip(img, 1)
        elif kind == 1:
            alpha = float(rng.uniform(0.8, 1.2))   # contrast
            beta = float(rng.uniform(-25, 25))      # brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        elif kind == 2:
            gains = rng.uniform(0.9, 1.1, size=3).astype(np.float32)  # colour jitter
            img = np.clip(img.astype(np.float32) * gains, 0, 255).astype(np.uint8)
        elif kind == 3:
            frac = float(rng.uniform(0.85, 1.0))     # mild random-resized crop
            ch, cw = max(2, int(h * frac)), max(2, int(w * frac))
            y0 = int(rng.integers(0, h - ch + 1))
            x0 = int(rng.integers(0, w - cw + 1))
            img = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            k = int(rng.choice([3, 5]))               # mild blur
            img = cv2.GaussianBlur(img, (k, k), 0)
        if i >= 5:  # second pass: combine with another change so it differs from the singles above
            if kind == 0:  # already flipped; flipping again would reproduce the original image
                img = cv2.convertScaleAbs(img, alpha=float(rng.uniform(0.85, 1.15)), beta=float(rng.uniform(-20, 20)))
            else:
                img = cv2.flip(img, 1)
        out.append(img)
    return out


def pick_central_box(dets: list[Detection], frame_shape: tuple[int, ...],
                     min_area_frac: float) -> Optional[Detection]:
    """Largest-and-most-central detection with area >= ``min_area_frac`` of the frame."""
    fh, fw = frame_shape[:2]
    cx, cy = fw / 2.0, fh / 2.0
    best, best_key = None, None
    for d in dets:
        area = (d.x2 - d.x1) * (d.y2 - d.y1)
        if area / float(fw * fh) < min_area_frac:
            continue
        dist = np.hypot((d.x1 + d.x2) / 2 - cx, (d.y1 + d.y2) / 2 - cy) / np.hypot(cx, cy)
        key = dist - 0.5 * area / float(fw * fh)  # nearer centre and larger is better
        if best_key is None or key < best_key:
            best, best_key = d, key
    return best


def crop_box(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, int(round(box[2]))), min(h, int(round(box[3])))
    return frame[y1:y2, x1:x2]


def hint_for(q) -> str:
    if q.blurry:
        return "Hold the object steady."
    if q.edge_cut:
        return "Keep the whole object inside the camera view."
    if q.too_small:
        return "Move the object closer."
    return "Hold it steady."


def run_enrolment(*, cfg, store: GalleryStore, detector, embedder,
                  get_frame: Callable[[], Optional[np.ndarray]],
                  speak: Callable[[str], None], ask: Callable[[str], Optional[str]],
                  clock: Callable[[], float] = time.monotonic,
                  resolve_name: Optional[Callable[[str, list[str]], Optional[str]]] = None) -> EnrolResult:
    ec = cfg.enrol
    res = EnrolResult(False, "")
    say_log: list[str] = []

    def say(t: str) -> None:
        say_log.append(t)
        log.info("enrol: %s", t)
        speak(t)

    say("Hold the object steadily in front of the camera.")
    start = clock()
    views: list[dict] = []
    prev_box: Optional[tuple] = None
    stable = 0
    last_hint_t = -1e9
    frames_seen = 0
    proposals_seen = 0
    large_proposals_seen = 0
    stable_candidates_seen = 0
    blurry_rejections = 0
    edge_rejections = 0
    small_rejections = 0

    while len(views) < ec.max_views:
        if clock() - start > ec.timeout_s:
            break
        frame = get_frame()
        if frame is None:
            continue
        frames_seen += 1
        try:
            proposals = detector.detect(frame)
            proposals_seen += len(proposals)
            large_proposals_seen += sum(
                ((d.x2 - d.x1) * (d.y2 - d.y1)) / max(frame.shape[0] * frame.shape[1], 1)
                >= ec.min_box_area_frac for d in proposals)
            det = pick_central_box(proposals, frame.shape, ec.min_box_area_frac)
        except Exception:  # noqa: BLE001 - a bad frame must not abort enrolment
            log.exception("detector failed during enrolment")
            continue
        if det is None:
            stable, prev_box = 0, None
            if clock() - last_hint_t >= ec.no_detection_hint_s:
                if proposals_seen == 0:
                    say("I can't see an object yet. Hold it centered, move it closer, and use good lighting.")
                else:
                    say("I see something, but it is too small. Move it closer and keep it centered.")
                last_hint_t = clock()
            continue
        if prev_box is not None and iou(prev_box, det.box) >= ec.stable_iou:
            stable += 1
        else:
            stable = 1
        prev_box = det.box
        if stable < ec.stable_frames:
            continue
        stable_candidates_seen += 1
        crop = crop_box(frame, det.box)
        if crop.size == 0:
            stable = 0
            continue
        q = assess(crop, det.box, frame.shape, cfg.quality.blur_var_min,
                   cfg.quality.edge_margin_px, cfg.quality.min_box_area_frac)
        if not (q.ok):
            blurry_rejections += int(q.blurry)
            edge_rejections += int(bool(q.edge_cut))
            small_rejections += int(q.too_small)
            if clock() - last_hint_t > 2.0:
                say(hint_for(q))
                last_hint_t = clock()
            stable = 0
            continue
        crops = [crop] + augment(crop, ec.augmentations_per_view, ec.seed + len(views))
        cls, patches = embedder.embed(crops)
        if views and max(float(v["cls"][0] @ cls[0]) for v in views) >= cfg.gallery_policy.duplicate_sim:
            if clock() - last_hint_t > 2.0:
                say("Rotate it a little more.")
                last_hint_t = clock()
            stable = 0
            continue
        views.append({"cls": cls, "patches": patches})
        say(f"Got view {len(views)}." + (" Rotate it slowly." if len(views) < ec.max_views else ""))
        stable, prev_box = 0, None
        if len(views) >= ec.min_views and len(views) >= ec.max_views:
            break

    res.views = len(views)
    if len(views) < ec.min_views:
        if frames_seen and proposals_seen == 0:
            detail = ("The detector did not propose any boxes. It may not detect this object type; "
                      "try a larger, clearly visible item such as a cup or bottle.")
        elif large_proposals_seen == 0:
            detail = "The detector's boxes were too small. Move the object closer so it fills more of the frame."
        elif stable_candidates_seen == 0:
            detail = "The boxes did not stay still long enough. Hold the camera and object steady."
        elif edge_rejections or blurry_rejections or small_rejections:
            issues = []
            if blurry_rejections:
                issues.append("blurred")
            if edge_rejections:
                issues.append("cut off at the frame edge")
            if small_rejections:
                issues.append("too small")
            detail = "Views were rejected as " + ", ".join(issues) + ". Improve the view and try again."
        else:
            detail = "Rotate the object more between views, then hold each view steady."
        res.message = (f"Enrolment timed out with {len(views)} good view(s); need at least "
                       f"{ec.min_views}. {detail} Nothing was saved. "
                       f"Diagnostics: frames={frames_seen}, boxes={proposals_seen}, "
                       f"large_boxes={large_proposals_seen}, stable_views={stable_candidates_seen}, "
                       f"blur_rejections={blurry_rejections}, edge_rejections={edge_rejections}, "
                       f"small_rejections={small_rejections}.")
        say(res.message)
        res.log = say_log
        return res

    # ---- name + confirmation ------------------------------------------------
    name: Optional[str] = None
    existing = store.names()
    for _ in range(3):
        heard = ask("What should I call this object?")
        if not heard or not heard.strip():
            continue
        heard = heard.strip()
        if any(heard.lower() == n.lower() for n in existing):
            say(f"You already have an object called {heard}. Please choose another name.")
            continue
        if is_yes(ask(f"I heard {heard}. Is that right?")):
            name = heard
            break
    if name is None:
        res.message = "No confirmed name was given. Nothing was saved."
        say(res.message)
        res.log = say_log
        return res

    # ---- persist ------------------------------------------------------------
    try:
        oid = store.add_object(name)
    except GalleryError as exc:
        res.message = str(exc)
        res.log = say_log
        return res
    n_emb = 0
    for v in views:
        for i in range(v["cls"].shape[0]):
            store.add_embedding(oid, v["cls"][i], "enrol",
                                patches=select_salient(v["patches"][i], cfg.rerank.top_patches))
            n_emb += 1
    res.success, res.name, res.embeddings = True, name, n_emb
    res.message = f"Saved {name} from {len(views)} views."
    say(res.message)

    # ---- confuser offer -----------------------------------------------------
    others = [n for n in existing if n.lower() != name.lower()]
    if others and is_yes(ask("Is it similar to another object you already have?")):
        picked = ask("Which one?")
        target = None
        if picked:
            if resolve_name is not None:
                target = resolve_name(picked, others)
            else:
                target = next((n for n in others if n.lower() == picked.strip().lower()), None)
        if target:
            store.set_confusers(name, target)
            res.confuser_of = target
            say(f"Okay, I will take extra care telling {name} and {target} apart.")
        else:
            say("I did not recognise that object, so no look-alike link was made.")
    res.log = say_log
    return res
