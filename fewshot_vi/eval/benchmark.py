"""Per-stage and end-to-end latency benchmark (mean / p95), with honest reporting.

* A stage that cannot run (missing dependency, no weights, no network) is reported
  as ``NOT MEASURED`` with the reason -- never as a number.
* ``match`` and ``rerank`` use a SYNTHETIC random gallery (their cost does not
  depend on image content); this is stated in the table.
* ``detect`` / ``embed`` / end-to-end use the real models when they can be
  loaded. The end-to-end loop runs the real ``Pipeline`` on a real still image
  (``--image`` or Ultralytics' bundled ``bus.jpg``), in two variants: a static
  scene (embeddings cached per track) and a stress case that re-embeds every
  track every frame.
* ``--synthetic`` replaces the detector/embedder with trivial stubs to measure
  PIPELINE OVERHEAD ONLY; it is labelled as such and says nothing about model speed.
* "recognition_to_speech_enqueue" is the time from frame hand-off to the speech
  request being queued. The audible start of speech additionally depends on the
  TTS engine and audio device, which this benchmark cannot measure.
"""
from __future__ import annotations

import logging
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ..config import Config, config_from_dict
from ..embedder.dino_embedder import DinoEmbedder
from ..gallery.matcher import Matcher
from ..gallery.patch_matcher import rerank
from ..gallery.store import GalleryStore
from ..guidance.engine import GuidanceEngine
from ..types import Decision, Match, Mode, Quality, Track
from ..voice import tts

log = logging.getLogger(__name__)


@dataclass
class Row:
    stage: str
    status: str                 # "measured" | "NOT MEASURED"
    n: int = 0
    mean_ms: float = float("nan")
    p95_ms: float = float("nan")
    note: str = ""


def _timeit(fn: Callable[[], Any], n: int, warmup: int = 3) -> list[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def _row(stage: str, samples: list[float], note: str = "") -> Row:
    return Row(stage, "measured", len(samples), float(np.mean(samples) * 1e3),
               float(np.percentile(samples, 95) * 1e3), note)


def _missing(stage: str, reason: Exception | str) -> Row:
    r = reason if isinstance(reason, str) else f"{type(reason).__name__}: {reason}"
    return Row(stage, "NOT MEASURED", note=r.splitlines()[0][:160] if r else "")


def hardware_summary() -> list[str]:
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    lines = [f"platform : {platform.platform()}", f"cpu      : {cpu} ({os.cpu_count()} logical cores)",
             f"python   : {platform.python_version()}"]
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None)
                                                         and torch.backends.mps.is_available() else "cpu")
        gpu = f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""
        lines.append(f"torch    : {torch.__version__}, selected device = {dev}{gpu}, threads = {torch.get_num_threads()}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"torch    : unavailable ({type(exc).__name__})")
    return lines


# ----------------------------------------------------------------- stand-ins
class _SilentBackend(tts.TTSBackend):
    name = "silent"

    def speak(self, text: str) -> None:
        return None


class _NullCamera:
    finished = type("E", (), {"is_set": staticmethod(lambda: True)})()
    error = None

    def read(self, timeout: float = 0.0):
        return None

    def stop(self) -> None:
        return None


class _Shared:
    """Forwards to a model but ignores ``close`` so one model can serve several Pipelines."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        if name == "close":
            return lambda: None
        return getattr(self._inner, name)


class OverheadDetector:
    """Trivial detector for --synthetic runs: three fixed boxes. Measures nothing about YOLO."""

    def detect(self, frame):
        from ..types import Detection
        h, w = frame.shape[:2]
        return [Detection(w * 0.1, h * 0.2, w * 0.3, h * 0.5, 0.9), Detection(w * 0.4, h * 0.3, w * 0.6, h * 0.6, 0.9),
                Detection(w * 0.7, h * 0.2, w * 0.9, h * 0.5, 0.9)]

    def close(self) -> None:
        return None


class OverheadEmbedder:
    """Trivial embedder for --synthetic runs. Measures nothing about DINOv2."""
    dim = 384

    def embed(self, crops):
        rng = np.random.default_rng(0)
        cls = rng.normal(size=(len(crops), self.dim)).astype(np.float32)
        pat = rng.normal(size=(len(crops), 64, self.dim)).astype(np.float32)
        return cls / np.linalg.norm(cls, axis=1, keepdims=True), pat / np.linalg.norm(pat, axis=2, keepdims=True)

    def warmup(self) -> None:
        return None

    def close(self) -> None:
        return None


RANDOM_INIT_LABEL = "vit-s14-random-init (timm architecture, NO pretrained weights; latency only)"


def random_init_embedder(input_size: int = 224) -> DinoEmbedder:
    """ViT-S/14 with RANDOM weights: same compute as DINOv2-S, meaningless features."""
    import timm
    import torch
    net = timm.create_model("vit_small_patch14_dinov2", pretrained=False, num_classes=0, img_size=input_size)
    net.eval()

    def run(batch: np.ndarray):
        with torch.inference_mode():
            tok = net.forward_features(torch.from_numpy(batch))
        tok = tok.float().numpy()
        return tok[:, 0], tok[:, 1:]
    emb = DinoEmbedder(input_size=input_size, backbone=run)
    emb.name = RANDOM_INIT_LABEL  # type: ignore[attr-defined]
    return emb


def default_frame(path: Optional[str] = None) -> tuple[np.ndarray, str]:
    """A real still image if one is available, else a synthetic texture (labelled)."""
    import cv2
    cands = [path] if path else []
    try:
        import ultralytics
        cands.append(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    except Exception:  # noqa: BLE001
        pass
    for c in cands:
        if c and Path(c).is_file():
            img = cv2.imread(c)
            if img is not None:
                return cv2.resize(img, (640, 480)), f"real image {Path(c).name} resized to 640x480"
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (480, 640, 3), dtype=np.uint8), "SYNTHETIC noise frame (no real image found)"


# ------------------------------------------------------------------ stages
def _bench_detect(det, frame, n) -> Row:
    try:
        return _row("detect (1 frame)", _timeit(lambda: det.detect(frame), n))
    except Exception as exc:  # noqa: BLE001
        return _missing("detect (1 frame)", exc)


def _bench_embed(emb, frame, n) -> list[Row]:
    rows = []
    crop = frame[100:300, 150:350]
    for b in (1, 4):
        name = f"embed (batch of {b} crops)"
        try:
            rows.append(_row(name, _timeit(lambda b=b: emb.embed([crop] * b), max(5, n // 2), warmup=2)))
        except Exception as exc:  # noqa: BLE001
            rows.append(_missing(name, exc))
    return rows


def _bench_match(dim: int, n: int, objects: int, per_object: int) -> Row:
    rng = np.random.default_rng(0)
    store = GalleryStore(":memory:", dim=dim)
    for i in range(objects):
        oid = store.add_object(f"obj{i}")
        for _ in range(per_object):
            v = rng.normal(size=dim).astype(np.float32)
            store.add_embedding(oid, v, "enrol")
    m = Matcher(store)
    q = rng.normal(size=dim).astype(np.float32)
    row = _row("match (global embedding)", _timeit(lambda: m.match(q), n * 4),
               f"SYNTHETIC gallery: {objects} objects x {per_object} embeddings, dim {dim}")
    store.close()
    return row


def _bench_rerank(dim: int, n: int) -> Row:
    rng = np.random.default_rng(1)
    q = rng.normal(size=(32, dim)).astype(np.float32)
    cands = {i: [rng.normal(size=(32, dim)).astype(np.float32) for _ in range(5)] for i in (1, 2)}
    return _row("rerank (patch, 2 candidates)", _timeit(lambda: rerank(q, cands), n * 2),
                f"SYNTHETIC tokens: 32 query x 2 candidates x 5 views x 32 tokens, dim {dim}")


def _bench_guidance(cfg: Config, n: int) -> Row:
    eng = GuidanceEngine(cfg, clock=lambda: 0.0)
    tracks = []
    for i in range(5):
        t = Track(i, (60.0 * i, 100.0, 60.0 * i + 50, 160.0))
        t.smoothed_name, t.smoothed_id = f"obj{i}", i
        t.match = Match(i, f"obj{i}", 0.9, 0.2, Decision.ACCEPT)
        t.quality = Quality(100.0, False, frozenset(), False, False, 1.0)
        tracks.append(t)
    return _row("guidance (5 tracks)", _timeit(lambda: eng.process(tracks, Mode.IDENTIFY, None, (480, 640, 3)), n * 4),
                "announce cooldown suppresses repeats after the first call")


def _bench_pipeline(cfg: Config, detector, embedder, frame, frames: int, label: str, stress: bool) -> list[Row]:
    from ..pipeline import Pipeline
    d = cfg.to_dict()
    d["detector"]["detect_every_n_frames"] = cfg.detector.detect_every_n_frames
    if stress:
        d["tracker"]["reembed_every_frames"] = 1
    c = config_from_dict(d)
    store = GalleryStore(":memory:", dim=getattr(embedder, "dim", None) or 384, top_patches=c.rerank.top_patches)
    rng = np.random.default_rng(2)
    for i in range(5):
        oid = store.add_object(f"obj{i}")
        for _ in range(5):
            store.add_embedding(oid, rng.normal(size=store.dim).astype(np.float32), "enrol")
    speaker = tts.Speaker(_SilentBackend())
    pipe = Pipeline(c, store, _Shared(detector), _Shared(embedder), _NullCamera(), speaker)
    try:
        for _ in range(3):  # warm-up frames (model init, JIT, caches)
            pipe.step(frame, time.perf_counter())
        pipe.stats.__init__()
        pipe.frames_ok = 0
        t0 = time.perf_counter()
        for _ in range(frames):
            pipe.step(frame, time.perf_counter())
        wall = time.perf_counter() - t0
    finally:
        pipe.close()
    tag = "stress: re-embed every track every frame" if stress else "static scene, embeddings cached per track"
    rows = []
    for stage, nice in (("total", "end-to-end frame"), ("detect", "  detect (only on detect frames)"),
                        ("embed", "  embed (only when needed)"), ("match", "  match"),
                        ("guidance", "  guidance")):
        s = pipe.stats.summary().get(stage)
        if s:
            rows.append(Row(f"[{label}; {tag}] {nice}", "measured", s["n"], s["mean_ms"], s["p95_ms"]))
    fps = pipe.frames_ok / wall if wall > 0 else float("nan")
    rows.append(Row(f"[{label}; {tag}] throughput", "measured", pipe.frames_ok, 1000.0 / fps if fps else float("nan"),
                    float("nan"), f"{fps:.1f} frames/s over {pipe.frames_ok} frames ({pipe.frame_errors} errors)"))
    s = pipe.stats.summary().get("recognition_to_speech_enqueue")
    if s:
        rows.append(Row(f"[{label}; {tag}] recognition -> speech enqueued", "measured", s["n"], s["mean_ms"],
                        s["p95_ms"], "excludes TTS engine start / audio device"))
    else:
        rows.append(Row(f"[{label}; {tag}] recognition -> speech enqueued", "NOT MEASURED", 0, note="no object was "
                        "confirmed in this run, so no speech request was produced"))
    return rows


# ------------------------------------------------------------------- driver
def run_benchmark(cfg: Config, detector: Any = None, embedder: Any = None, *, iterations: int = 30,
                  frames: int = 60, image: Optional[str] = None, gallery_objects: int = 20,
                  per_object: int = 20, detector_label: str = "none", embedder_label: str = "none",
                  detector_error: Optional[str] = None, embedder_error: Optional[str] = None) -> tuple[list[Row], str]:
    frame, frame_desc = default_frame(image)
    rows: list[Row] = []
    dim = getattr(embedder, "dim", None) or 384
    if detector is not None:
        rows.append(_bench_detect(detector, frame, iterations))
    else:
        rows.append(_missing("detect (1 frame)", detector_error or "no detector available"))
    if embedder is not None:
        rows += _bench_embed(embedder, frame, iterations)
        dim = getattr(embedder, "dim", None) or dim
    else:
        for b in (1, 4):
            rows.append(_missing(f"embed (batch of {b} crops)", embedder_error or "no embedder available"))
    rows.append(_bench_match(dim, iterations, gallery_objects, per_object))
    rows.append(_bench_rerank(dim, iterations))
    rows.append(_bench_guidance(cfg, iterations))
    if detector is not None and embedder is not None:
        try:
            lab = "synthetic overhead stubs" if (isinstance(detector, OverheadDetector)) else "real detector + embedder"
            for stress in (False, True):
                rows += _bench_pipeline(cfg, detector, embedder, frame, frames, lab, stress)
        except Exception as exc:  # noqa: BLE001
            log.exception("end-to-end benchmark failed")
            rows.append(_missing("end-to-end loop", exc))
    else:
        rows.append(_missing("end-to-end loop", "needs both a detector and an embedder (see rows above)"))
    return rows, format_report(rows, hardware_summary(), frame_desc, detector_label, embedder_label, cfg)


def format_report(rows: list[Row], hw: list[str], frame_desc: str, detector_label: str, embedder_label: str,
                  cfg: Config) -> str:
    out = ["Latency benchmark", *hw, f"detector : {detector_label}", f"embedder : {embedder_label}",
           f"frame    : {frame_desc}", f"detect_every_n_frames = {cfg.detector.detect_every_n_frames}", ""]
    w = max(len(r.stage) for r in rows)
    out.append(f"{'stage'.ljust(w)}  {'status':<12} {'n':>5} {'mean ms':>9} {'p95 ms':>9}  note")
    out.append("-" * (w + 52))
    for r in rows:
        if r.status == "measured":
            p95 = "" if r.p95_ms != r.p95_ms else f"{r.p95_ms:9.2f}"
            out.append(f"{r.stage.ljust(w)}  {r.status:<12} {r.n:>5} {r.mean_ms:9.2f} {p95:>9}  {r.note}")
        else:
            out.append(f"{r.stage.ljust(w)}  {r.status:<12} {'':>5} {'':>9} {'':>9}  {r.note}")
    return "\n".join(out)
