"""Real-time pipeline: threads, bounded queues, per-frame processing and shutdown.

Threads
-------
* capture    - ``Camera`` thread, ``Queue(maxsize=2)`` drop-oldest
* vision     - the thread that calls ``Pipeline.run`` (detect/track/embed/match/guidance)
* rerank     - ``PatchReranker`` worker (look-alike patch matching, deadline-bound)
* db-writer  - ``ObjectMemory`` writer queue (sightings)
* tts        - ``Speaker`` worker (priority queue, interruptible)
* voice      - ``VoiceListener`` (speech/keyboard input -> ``handle_text``)
* scene      - ``SceneWorker`` (CLIP scene labels, off the critical path)

Everything touching hardware or models is injected, so ``Pipeline`` can be
exercised with fakes (see tests/test_pipeline_smoke.py).

Temporal smoothing note: embeddings are computed once per track and refreshed
only when the box changes a lot or after ``tracker.reembed_every_frames``; each
frame the track's current match is added to a bounded vote history. A label is
spoken only once ``min_votes`` of the last ``window`` votes agree.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from typing import Any, Callable, Optional

import cv2
import numpy as np

from .config import Config
from .eval.metrics import FpsMeter, LatencyStats
from .gallery.enrol import EnrolResult, crop_box, run_enrolment
from .gallery.matcher import Matcher
from .gallery.patch_matcher import PatchReranker, apply_rerank
from .gallery.store import GalleryStore
from .gallery.updater import GalleryUpdateWorker, GalleryUpdater
from .guidance import engine as eng
from .guidance.engine import GuidanceEngine
from .guidance.hands import reach_instruction
from .memory.object_memory import ObjectMemory
from .memory.scene import NullSceneClassifier, SceneClassifier
from .quality.quality import assess
from .tracker.tracker import IoUTracker
from .types import AppState, Decision, GuidanceEvent, Match, Mode, Track
from .voice.commands import Intent, parse_command
from .voice.stt import ReplyQueue, VoiceListener

log = logging.getLogger(__name__)


class SceneWorker:
    """Classifies the latest frame every ``interval_s`` seconds on its own thread."""

    def __init__(self, classifier: SceneClassifier, interval_s: float) -> None:
        self.classifier, self.interval_s = classifier, interval_s
        self.label: Optional[str] = None
        self.embedding: Optional[np.ndarray] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="scene", daemon=True)

    def start(self) -> "SceneWorker":
        if self.classifier.available or hasattr(self.classifier, "_load"):
            self._thread.start()
        return self

    def submit(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def _run(self) -> None:
        last = -1e9
        while not self._stop.wait(0.1):
            now = time.monotonic()
            if now - last < self.interval_s:
                continue
            with self._lock:
                frame, self._frame = self._frame, None
            if frame is None:
                continue
            last = now
            try:
                res = self.classifier.classify(frame)
            except Exception:  # noqa: BLE001
                log.exception("scene classification failed")
                continue
            if res is not None:
                self.label, self.embedding = res
            if not self.classifier.available and getattr(self.classifier, "_loaded", False):
                break  # classifier disabled itself; stop trying

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


def draw_overlay(frame: np.ndarray, tracks: list[Track], fps: float, stats: LatencyStats) -> np.ndarray:
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t.bbox)
        name = t.smoothed_name
        colour = (0, 200, 0) if name else ((0, 200, 255) if t.match and t.match.decision == Decision.AMBIGUOUS else (0, 0, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        score = (f" {t.match.score:.2f}" if t.match and t.match.object_id is not None
                 else f" {t.score:.2f}" if t.class_name is not None else "")
        proposal = f" [{t.class_name}]" if t.class_name and name is None else ""
        label = f"#{t.track_id} {name or '?'}{proposal}{score}"
        cv2.putText(out, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    lines = [f"FPS {fps:.1f}"]
    for stage in ("detect", "embed", "match", "rerank", "guidance", "total"):
        m = stats.mean_ms(stage)
        if m is not None:
            lines.append(f"{stage} {m:.1f} ms")
    for i, text in enumerate(lines):
        cv2.putText(out, text, (8, 18 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class Pipeline:
    def __init__(self, cfg: Config, store: GalleryStore, detector, embedder, camera, speaker,
                 beeper=None, stt=None, scene: Optional[SceneClassifier] = None,
                 reranker: Optional[PatchReranker] = None, hand_tracker=None,
                 state: Optional[AppState] = None, clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg, self.store, self.detector, self.embedder = cfg, store, detector, embedder
        # If no personal references exist, detector categories are sufficient;
        # skip the substantially slower DINO embedding pass in live mode.
        self._personal_enabled = bool(store.names())
        self.camera, self.speaker, self.beeper, self.stt = camera, speaker, beeper, stt
        self.clock = clock
        self.state = state or AppState()
        self.matcher = Matcher(store, cfg.matching.accept_sim, cfg.matching.ambiguous_sim,
                               cfg.matching.min_margin, cfg.matching.knn_k)
        tr = cfg.tracker
        self.tracker = IoUTracker(tr.match_iou, tr.max_missed_frames, tr.reembed_iou,
                                  tr.reembed_every_frames, history=cfg.temporal.window)
        self.reranker = reranker if reranker is not None else (
            PatchReranker(cfg.rerank.deadline_s) if cfg.features.patch_rerank else None)
        self.memory = ObjectMemory(store, cfg.memory.min_conf_to_log, cfg.memory.log_cooldown_s, clock=clock)
        self.updater = GalleryUpdater(store, cfg.temporal.window, cfg.gallery.max_embeddings_per_object,
                                      cfg.gallery.update_min_sim, cfg.gallery.update_min_margin,
                                      cfg.gallery.update_cooldown_s, cfg.gallery_policy.duplicate_sim,
                                      cfg.gallery_policy.drift_min_sim, clock=clock)
        self.gallery_writer = GalleryUpdateWorker(self.updater) if cfg.features.self_improve else None
        self.guidance = GuidanceEngine(cfg, memory_answer=self.memory.answer, clock=clock)
        self.scene = SceneWorker(scene or NullSceneClassifier(), cfg.memory.scene_every_n_s)
        self.hands = hand_tracker if cfg.features.hand_guidance else None
        self.stats = LatencyStats()
        self.fps = FpsMeter()
        self.frame_idx = 0
        self.frames_ok = 0
        self.frame_errors = 0
        self.events_emitted: list[GuidanceEvent] = []
        self.last_enrolled: Optional[str] = None
        self._pending_delete: Optional[tuple[str, float]] = None
        self._replies = ReplyQueue()
        self._dialog_active = threading.Event()
        self._listener: Optional[VoiceListener] = None
        self._last_hand_say = -1e9
        self._closed = False
        self.last_frame_shape: tuple[int, ...] = (cfg.camera.height, cfg.camera.width, 3)

    # ===================================================================== speech
    def say(self, text: str, priority: int = eng.PRIORITY_ANSWER) -> None:
        self.speaker.say(text, priority)

    def ask(self, prompt: str) -> Optional[str]:
        """Ask a question and wait for the next utterance (voice thread if running, else STT directly)."""
        self.say(prompt, eng.PRIORITY_SYSTEM)
        timeout = self.cfg.voice.confirm_timeout_s
        if self._listener is not None:
            self._replies.drain()
            self._dialog_active.set()
            try:
                return self._replies.get(timeout)
            finally:
                self._dialog_active.clear()
        if self.stt is None:
            return None
        return self.stt.listen(prompt=prompt, timeout=timeout)

    # ================================================================== commands
    def handle_text(self, text: str) -> None:
        """Voice/keyboard entry point (runs on the voice thread; never touches vision state directly)."""
        if self._dialog_active.is_set():  # answer to an in-progress question
            self._replies.put(text)
            return
        names = self.store.names()
        categories = (getattr(self.detector, "class_names", [])
                      if self.cfg.features.category_fallback else [])
        cmd = parse_command(text, list(dict.fromkeys(names + categories)), self.cfg.voice.fuzzy_cutoff)
        I = Intent
        if cmd.intent in (I.DELETE, I.CONFUSER) and cmd.name not in names:
            cmd.name = None  # categories can be found, but are not gallery entries to delete/link
        if self._pending_delete is not None:
            name, t0 = self._pending_delete
            self._pending_delete = None
            if cmd.intent == I.YES and self.clock() - t0 <= self.cfg.voice.confirm_timeout_s * 4:
                self.state.post("delete", name)
                self.say(f"Deleting {name}.")
            else:
                self.say("Okay, I will keep it.")
            return
        if cmd.needs_name and cmd.name is None and cmd.intent != I.CONFUSER:
            self.say(f"I don't have an object called {cmd.raw_name}.")
            return
        if cmd.intent == I.IDENTIFY:
            self.state.set_mode(Mode.IDENTIFY)
            self.say("Identifying.")
        elif cmd.intent == I.FIND:
            self.state.set_mode(Mode.FIND, cmd.name, now=self.clock())
            self.say(f"Looking for {cmd.name}.")
        elif cmd.intent == I.MEMORY_QUERY:
            if cmd.name not in self.store.names():
                self.say(f"I have not learned the {cmd.name} class yet. Add a few examples before asking me to find it.")
            else:
                self.say(self.memory.answer(cmd.name))
        elif cmd.intent == I.ENROL:
            self.state.post("enrol")
        elif cmd.intent == I.DELETE:
            self._pending_delete = (cmd.name, self.clock())
            self.say(f"Delete {cmd.name}? Say yes to confirm.", eng.PRIORITY_SYSTEM)
        elif cmd.intent == I.LIST:
            names = self.store.names()
            self.say("You have " + (", ".join(names) + "." if names else
                     "no few-shot classes enrolled yet. Add a few examples for each class."))
        elif cmd.intent == I.MUTE:
            self.state.set_muted(True)
            self.speaker.clear(below_priority=eng.PRIORITY_ANSWER)
        elif cmd.intent == I.RESUME:
            self.state.set_muted(False)
            self.say("Resuming.")
        elif cmd.intent == I.CONFUSER:
            if not self.last_enrolled:
                self.say("Register an object first, then tell me which one it is similar to.")
            elif cmd.name is None:
                self.say(f"I don't have an object called {cmd.raw_name}.")
            else:
                try:
                    self.store.set_confusers(self.last_enrolled, cmd.name)
                    self.say(f"Okay, {self.last_enrolled} and {cmd.name} are marked as look-alikes.")
                except Exception as exc:  # noqa: BLE001
                    self.say(str(exc))
        elif cmd.intent in (I.YES, I.NO):
            pass  # nothing pending
        else:
            self.say("Sorry, I did not understand.", eng.PRIORITY_HINT)

    def _service_requests(self) -> None:
        for kind, payload in self.state.drain():
            try:
                if kind == "enrol":
                    self.run_enrolment()
                elif kind == "delete":
                    ok = self.store.delete_object(payload)
                    self._personal_enabled = bool(self.store.names())
                    self.tracker.tracks.clear()  # drop cached matches that may name the deleted object
                    self.say(f"{payload} deleted." if ok else f"I could not find {payload}.")
            except Exception:  # noqa: BLE001
                log.exception("request %s failed", kind)
                self.say("Sorry, that did not work.", eng.PRIORITY_ANSWER)

    # =================================================================== enrol
    def run_enrolment(self) -> EnrolResult:
        prev_mode, prev_target = self.state.mode, self.state.target
        self.state.set_mode(Mode.ENROL)

        def get_frame():
            item = self.camera.read(timeout=0.5)
            return None if item is None else item[0]

        from .voice.commands import resolve_name
        try:
            res = run_enrolment(cfg=self.cfg, store=self.store, detector=self.detector, embedder=self.embedder,
                                get_frame=get_frame, speak=lambda t: self.say(t, eng.PRIORITY_SYSTEM),
                                ask=self.ask, clock=self.clock,
                                resolve_name=lambda s, names: resolve_name(s, names, self.cfg.voice.fuzzy_cutoff))
        finally:
            self.state.set_mode(prev_mode, prev_target, now=self.clock())
        if res.success:
            self.last_enrolled = res.name
            self._personal_enabled = True
            self.tracker.tracks.clear()
            if self.cfg.features.memory and isinstance(self.scene.classifier, NullSceneClassifier):
                self.scene.stop()
                from .memory.scene import ClipSceneClassifier
                self.scene = SceneWorker(ClipSceneClassifier(), self.cfg.memory.scene_every_n_s).start()
        return res

    # ============================================================== frame logic
    def _crop(self, frame: np.ndarray, t: Track) -> np.ndarray:
        return crop_box(frame, t.bbox)

    def _vote(self, t: Track) -> None:
        """Push this frame's vote, then derive the smoothed label."""
        m = t.match
        vote = m.object_id if (m is not None and m.decision == Decision.ACCEPT) else None
        t.label_history.append(vote)
        votes = [v for v in t.label_history if v is not None]
        t.smoothed_id = t.smoothed_name = None
        t.stable_accepts = 0
        if votes:
            oid, n = Counter(votes).most_common(1)[0]
            if n >= self.cfg.temporal.min_votes:
                t.smoothed_id, t.smoothed_name = oid, self.store.get_name(oid)
                run = 0
                for v in reversed(t.label_history):
                    if v == oid:
                        run += 1
                    else:
                        break
                t.stable_accepts = run
        if m is not None and t.smoothed_id is not None:
            t.conf_history.append(m.score)
            t.smoothed_conf = float(np.mean(t.conf_history))
        elif t.smoothed_id is None:
            t.smoothed_conf = 0.0
        # The detector label is a proposal, not the few-shot answer. By default
        # only an accepted gallery match may be named. The optional fallback
        # exposes ordinary pretrained YOLO categories when explicitly enabled.
        if t.smoothed_name is not None or not self.cfg.features.category_fallback:
            t.category_history.clear()
            return
        if t.missed == 0:
            if (t.class_name is not None
                    and t.score >= self.cfg.detector.category_min_conf):
                if t.category_history and t.category_history[-1][0] != t.class_name:
                    t.category_history.clear()
                t.category_history.append((t.class_name, t.score))
            else:
                t.category_history.append((None, t.score))
        streak = []
        for name, score in reversed(t.category_history):
            if name is None:
                break
            streak.append((name, score))
        if len(streak) >= self.cfg.temporal.min_votes:
            category = streak[0][0]
            mean_conf = float(np.mean([score for _, score in streak]))
            if mean_conf >= self.cfg.detector.category_min_conf:
                t.smoothed_id = None
                t.smoothed_name = category
                t.smoothed_conf = mean_conf
                t.stable_accepts = len(streak)

    def process_frame(self, frame: np.ndarray, ts: float) -> list[GuidanceEvent]:
        """All per-frame work. Raises on error; ``step`` wraps it."""
        cfg, rec = self.cfg, self.stats.record
        t_start = time.perf_counter()
        self.frame_idx += 1
        self.last_frame_shape = frame.shape
        self.scene.submit(frame)

        # 1. detect (every N frames) + track ----------------------------------
        t0 = time.perf_counter()
        if (self.frame_idx - 1) % cfg.detector.detect_every_n_frames == 0:
            self.tracker.update(self.detector.detect(frame))
            rec("detect", time.perf_counter() - t0)
        else:
            self.tracker.predict()
        tracks = self.tracker.active()

        # 2. fold in finished patch re-ranks ---------------------------------
        if self.reranker is not None:
            by_id = {t.track_id: t for t in tracks}
            for r in self.reranker.poll():
                rec("rerank", r.elapsed_s)
                t = by_id.get(r.track_id)
                if t is None or t.match is None:
                    continue
                t.rerank_pending = False
                before = t.match.object_id
                t.match = apply_rerank(t.match, r, self.store.get_name, cfg.rerank.accept_patch_sim,
                                       cfg.rerank.min_gain, cfg.rerank.min_patch_margin)
                t.rerank_agrees = (r.met_deadline and bool(r.ranking) and r.ranking[0][0] == before
                                   and t.match.decision == Decision.ACCEPT)

        # 3. embed tracks that need it (ONE batched forward pass) -------------
        need = []
        for t in tracks:
            if not t.needs_embed or t.missed > 0:
                continue
            if t.class_name is not None and not self._personal_enabled:
                continue
            crop = self._crop(frame, t)
            if crop.size == 0:
                continue
            need.append((t, crop))
        if need:
            t0 = time.perf_counter()
            cls, patches = self.embedder.embed([c for _, c in need])
            rec("embed", time.perf_counter() - t0)
            t0 = time.perf_counter()
            for i, (t, _) in enumerate(need):
                self.tracker.mark_embedded(t, cls[i], patches[i])
                t.match = self.matcher.match(cls[i])
                t.update_tried, t.rerank_agrees = False, None
                if (t.match.decision == Decision.AMBIGUOUS and self.reranker is not None
                        and not t.rerank_pending):
                    cands = self.matcher.confuser_candidates(t.match)
                    if cands:
                        views = {oid: self.store.patches_of(oid) for oid in cands}
                        if self.reranker.submit(t.track_id, patches[i], views):
                            t.rerank_pending = True
            rec("match", time.perf_counter() - t0)

        # 4. smoothing, quality, safe updates, memory -------------------------
        for t in tracks:
            self._vote(t)
            if t.match is not None and t.embedding is not None:
                t.quality = assess(self._crop(frame, t), t.bbox, frame.shape, cfg.quality.blur_var_min,
                                   cfg.quality.edge_margin_px, cfg.quality.min_box_area_frac,
                                   margin=t.match.margin, min_margin=cfg.matching.min_margin)
            if (cfg.features.self_improve and t.smoothed_id is not None and not t.update_tried
                    and t.stable_accepts >= cfg.temporal.window and t.match is not None
                    and t.match.object_id == t.smoothed_id and t.embedding is not None):
                group = self.store.confuser_group(t.smoothed_id)
                agrees = True if group is None else bool(t.rerank_agrees)
                if self.gallery_writer is not None:
                    t.update_tried = self.gallery_writer.submit(
                        t.smoothed_id, t.embedding, t.match, t.quality,
                        t.stable_accepts, confuser_agrees=agrees)
            if cfg.features.memory and t.smoothed_id is not None:
                emb = self.scene.embedding
                self.memory.offer(t.smoothed_id, t.smoothed_name or "", t.smoothed_conf, t.bbox,
                                  self.scene.label,
                                  None if emb is None else np.asarray(emb, np.float32).tobytes())

        # 5. guidance -> audio -------------------------------------------------
        t0 = time.perf_counter()
        events = self.guidance.process(tracks, self.state.mode, self.state.target, frame.shape,
                                       self.state.find_started, self.state.muted)
        events += self._hand_events(frame, tracks)
        for e in events:
            self._emit(e, ts)
        rec("guidance", time.perf_counter() - t0)

        now = time.perf_counter()
        rec("total", now - t_start)
        self.fps.tick(now)
        self.frames_ok += 1
        self._tracks_view = tracks
        return events

    def _hand_events(self, frame: np.ndarray, tracks: list[Track]) -> list[GuidanceEvent]:
        if self.hands is None or self.state.mode != Mode.FIND or not self.state.target or self.state.muted:
            return []
        tgt = [t for t in tracks if t.smoothed_name and t.smoothed_name.lower() == self.state.target.lower()]
        if not tgt:
            return []
        now = self.clock()
        if now - self._last_hand_say < self.cfg.guidance.announce_cooldown_s:
            return []
        hand = self.hands.centroid(frame)
        if hand is None:
            return []
        self._last_hand_say = now
        return [GuidanceEvent("hand", reach_instruction(hand, tgt[0].bbox), eng.PRIORITY_FIND, 0.0, 0.0, now)]

    def _emit(self, e: GuidanceEvent, frame_ts: float) -> None:
        self.events_emitted.append(e)
        if len(self.events_emitted) > 1000:
            del self.events_emitted[:500]
        if e.kind == "beep":
            if self.beeper is not None:
                self.beeper.beep(e.pan, e.urgency)
            return
        self.speaker.say(e.text, e.priority)
        self.stats.record("recognition_to_speech_enqueue", time.perf_counter() - frame_ts)

    def step(self, frame: np.ndarray, ts: float) -> list[GuidanceEvent]:
        """One frame; a failure is logged and counted but never propagates."""
        try:
            return self.process_frame(frame, ts)
        except Exception:  # noqa: BLE001 - a single bad frame must not end the live loop
            self.frame_errors += 1
            log.exception("frame %d failed; continuing", self.frame_idx)
            return []

    # ================================================================== run loop
    def start_background(self) -> None:
        self.scene.start()
        if self.stt is not None and self._listener is None:
            self._listener = VoiceListener(self.stt, self.handle_text).start()

    def run(self, show: bool = False, max_frames: Optional[int] = None) -> None:
        self.start_background()
        last_report = time.monotonic()
        try:
            while not self.state.shutting_down:
                if time.monotonic() - last_report >= 5.0:
                    last_report = time.monotonic()
                    log.info("rolling fps=%.1f  total=%s ms  frame_errors=%d", self.fps.fps,
                             "n/a" if self.stats.mean_ms("total") is None else f"{self.stats.mean_ms('total'):.1f}",
                             self.frame_errors)
                item = self.camera.read(timeout=1.0)
                if item is None:
                    if getattr(self.camera, "error", None):
                        log.error("camera error: %s", self.camera.error)
                        break
                    if self.camera.finished.is_set():
                        break
                    continue
                frame, ts = item
                self.step(frame, ts)
                self._service_requests()
                if show:
                    try:
                        cv2.imshow("fewshot_vi", draw_overlay(frame, getattr(self, "_tracks_view", []),
                                                              self.fps.fps, self.stats))
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            self.state.request_shutdown()
                    except cv2.error as exc:
                        log.warning("cannot show window (%s); continuing without display", exc)
                        show = False
                if max_frames is not None and self.frames_ok + self.frame_errors >= max_frames:
                    break
        except KeyboardInterrupt:
            log.info("interrupted; shutting down")
        finally:
            self.close()

    def timing_report(self) -> str:
        lines = [f"frames ok={self.frames_ok} errors={self.frame_errors} rolling_fps={self.fps.fps:.1f}"]
        for stage, s in self.stats.summary().items():
            lines.append(f"  {stage:<32} n={s['n']:<5} mean={s['mean_ms']:.2f} ms  p95={s['p95_ms']:.2f} ms")
        return "\n".join(lines)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.state.request_shutdown()
        steps: list[tuple[str, Callable[[], Any]]] = [
            ("voice", lambda: self._listener and self._listener.stop()),
            ("stt", lambda: self.stt and self.stt.close()),
            ("scene", self.scene.stop),
            ("rerank", lambda: self.reranker and self.reranker.close()),
            ("gallery-writer", lambda: self.gallery_writer and self.gallery_writer.close()),
            ("memory", self.memory.close),
            ("speaker", self.speaker.close),
            ("camera", self.camera.stop),
            ("detector", self.detector.close),
            ("embedder", lambda: getattr(self.embedder, "close", lambda: None)()),
            ("hands", lambda: self.hands and self.hands.close()),
            ("gallery", self.store.close),
            ("windows", cv2.destroyAllWindows),
        ]
        for name, fn in steps:
            try:
                fn()
            except Exception:  # noqa: BLE001 - keep releasing the rest
                log.debug("error while closing %s", name, exc_info=True)


def build_pipeline(cfg: Config, db_path: str, camera_index: Optional[Any] = None, keyboard_only: bool = False,
                   need_camera: bool = True) -> Pipeline:
    """Construct the real (hardware/model-backed) pipeline.

    The camera is opened FIRST so a missing camera fails fast (``CameraError``)
    before any model is loaded or any resource is created. Models are still
    loaded lazily; call ``preload_models`` to surface download/dependency
    problems before the live loop starts.
    """
    from .capture.camera import Camera
    from .detector.yolo_detector import YoloDetector
    from .embedder.dino_embedder import DinoEmbedder
    from .guidance.hands import HandTracker
    from .memory.scene import ClipSceneClassifier
    from .voice import tts
    from .voice.beeps import Beeper
    from .voice.stt import make_stt

    cam = Camera(cfg.camera.index if camera_index is None else camera_index, cfg.camera.width,
                 cfg.camera.height, cfg.camera.target_fps)
    if need_camera:
        cam.start()  # raises CameraError with a helpful message
    try:
        store = GalleryStore(db_path, top_patches=cfg.rerank.top_patches)
        det = YoloDetector(cfg.detector.weights, cfg.detector.conf, cfg.detector.max_objects,
                           imgsz=cfg.detector.imgsz)
        emb = DinoEmbedder(cfg.embedder.model, cfg.embedder.input_size, cfg.embedder.fp16)
        speaker = tts.Speaker(tts.make_backend(), stale_s=cfg.guidance_policy.stale_speech_s)
        scene = (ClipSceneClassifier() if cfg.features.memory and store.names()
                 else NullSceneClassifier("no personal objects enrolled"))
        hands = HandTracker() if cfg.features.hand_guidance else None
        return Pipeline(cfg, store, det, emb, cam, speaker, Beeper(), make_stt(not keyboard_only), scene,
                        hand_tracker=hands)
    except Exception:
        cam.stop()
        raise


def preload_models(pipe: Pipeline, warm_embedder: bool = True) -> None:
    """Load/warm the detector and embedder now so download or dependency errors
    surface before the live loop (raises the underlying exception)."""
    warm_det = getattr(pipe.detector, "warmup", None)
    if warm_det is not None:
        warm_det()
    warm_emb = getattr(pipe.embedder, "warmup", None)
    if warm_embedder and warm_emb is not None:
        warm_emb()
