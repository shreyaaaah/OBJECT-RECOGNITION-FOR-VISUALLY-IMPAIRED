#!/usr/bin/env python3
"""Entry point.

    python run.py --mode live  [--show] [--camera 0|video.mp4] [--keyboard-only]
    python run.py --mode enrol [--camera 0]
    python run.py --mode eval  --dataset DIR [--ablation] [--embedder dino|color-hist]
    python run.py --mode benchmark [--embedder dino|random-init] [--synthetic]

Exit codes: 0 ok, 1 runtime failure (camera, enrolment), 2 usage/config/dataset error,
3 missing dependency or model that cannot be loaded.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
log = logging.getLogger("run")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Few-shot personal object recogniser for the visually impaired")
    p.add_argument("--mode", choices=["live", "enrol", "eval", "benchmark"], required=True)
    p.add_argument("--config", default=str(HERE / "config.yaml"))
    p.add_argument("--db", default=str(HERE / "data" / "gallery.db"), help="SQLite gallery path")
    p.add_argument("--camera", default=None, help="camera index or video file (default: config camera.index)")
    p.add_argument("--show", action="store_true", help="show the debug overlay window")
    p.add_argument("--keyboard-only", action="store_true", help="never try the microphone; type commands")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--dataset", default=None, help="eval: dataset/<class>/*.jpg")
    p.add_argument("--ablation", action="store_true", help="eval: run all five ablations")
    p.add_argument("--embedder", choices=["dino", "color-hist", "random-init"], default="dino",
                   help="dino = DINOv2 ViT-S/14 (default); color-hist = non-learned baseline (plumbing checks); "
                        "random-init = ViT-S/14 with RANDOM weights (benchmark latency only)")
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--k-shot", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--n-query", type=int, default=5)
    p.add_argument("--n-unknown", type=int, default=3)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sessions", type=int, default=3)
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--out", default=None, help="eval: CSV output path")
    p.add_argument("--iterations", type=int, default=30, help="benchmark iterations per stage")
    p.add_argument("--bench-frames", type=int, default=60)
    p.add_argument("--image", default=None, help="benchmark: still image to use")
    p.add_argument("--synthetic", action="store_true",
                   help="benchmark: stub detector/embedder (measures pipeline overhead ONLY)")
    return p


def _camera_arg(v):
    if v is None:
        return None
    return int(v) if v.isdigit() else v


def make_embedder(name: str, cfg):
    if name == "color-hist":
        from fewshot_vi.embedder.color_hist import ColorHistEmbedder
        e = ColorHistEmbedder()
        return e, e.name
    if name == "random-init":
        from fewshot_vi.eval.benchmark import RANDOM_INIT_LABEL, random_init_embedder
        return random_init_embedder(cfg.embedder.input_size), RANDOM_INIT_LABEL
    from fewshot_vi.embedder.dino_embedder import DinoEmbedder
    return (DinoEmbedder(cfg.embedder.model, cfg.embedder.input_size, cfg.embedder.fp16),
            f"{cfg.embedder.model} (DINOv2 via torch.hub)")


def _run_live_or_enrol(args, cfg) -> int:
    from fewshot_vi.capture.camera import CameraError
    from fewshot_vi.pipeline import build_pipeline, preload_models
    try:
        pipe = build_pipeline(cfg, args.db, _camera_arg(args.camera), args.keyboard_only)
    except CameraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        print("Loading models (first run downloads weights) ...")
        preload_models(pipe)
    except Exception as exc:  # noqa: BLE001 - missing torch/ultralytics/weights/network
        print(f"error: cannot load models: {type(exc).__name__}: {exc}\n"
              "Check `pip install -r requirements.txt`, network access for the first-run model download, "
              "and the README 'First run' section.", file=sys.stderr)
        pipe.close()
        return 3
    try:
        if args.mode == "enrol":
            print("Guided enrolment. Type answers at the '>' prompt if no microphone is available.")
            res = pipe.run_enrolment()
            pipe.speaker.wait_idle(3.0)
            print(("OK: " if res.success else "FAILED: ") + res.message)
            return 0 if res.success else 1
        names = pipe.store.names()
        pipe.say("Ready. " + ("You have " + ", ".join(names) + "." if names else
                              "No objects yet. Say register object."), 40)
        print("Live. Commands: what is this | find <obj> | where did I leave <obj> | register object | "
              "delete <obj> | list objects | stop | resume.  Press q in the window or Ctrl+C to quit.")
        pipe.run(show=args.show, max_frames=args.max_frames)
        print(pipe.timing_report())
        return 0
    finally:
        pipe.close()


def _run_eval(args, cfg) -> int:
    from fewshot_vi.eval.episodes import DatasetError
    from fewshot_vi.eval.run_eval import run_ablation
    if not args.dataset:
        print("error: --dataset DIR is required for --mode eval (layout: DIR/<class>/*.jpg)", file=sys.stderr)
        return 2
    try:
        embedder, label = make_embedder(args.embedder, cfg)
        _, report = run_ablation(cfg, embedder, label, args.dataset, ks=args.k_shot, n_way=args.n_way,
                                 n_query=args.n_query, n_unknown=args.n_unknown, episodes=args.episodes,
                                 seed=args.seed, sessions=args.sessions, max_per_class=args.max_per_class,
                                 full=args.ablation, out_csv=args.out, progress=print)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"error: evaluation could not run: {type(exc).__name__}: {exc}\n"
              "If this is a model-download problem, see the README 'First run' section; "
              "`--embedder color-hist` runs the plumbing without PyTorch (NOT a DINOv2 result).", file=sys.stderr)
        return 3
    print(report)
    return 0


def _run_benchmark(args, cfg) -> int:
    from fewshot_vi.eval import benchmark as bm
    det = emb = None
    det_err = emb_err = None
    det_label = emb_label = "none"
    if args.synthetic:
        det, emb = bm.OverheadDetector(), bm.OverheadEmbedder()
        det_label, emb_label = "SYNTHETIC stub (overhead only)", "SYNTHETIC stub (overhead only)"
    else:
        try:
            from fewshot_vi.detector.yolo_detector import YoloDetector
            det = YoloDetector(cfg.detector.weights, cfg.detector.conf, cfg.detector.max_objects,
                               imgsz=cfg.detector.imgsz)
            det.warmup()
            det_label = f"{cfg.detector.weights} (Ultralytics YOLO)"
        except Exception as exc:  # noqa: BLE001
            det, det_err = None, f"{type(exc).__name__}: {exc}"
        try:
            emb, emb_label = make_embedder("dino" if args.embedder == "dino" else args.embedder, cfg)
            emb.warmup()
        except Exception as exc:  # noqa: BLE001
            emb, emb_err = None, f"{type(exc).__name__}: {exc}"
    _, report = bm.run_benchmark(cfg, det, emb, iterations=args.iterations, frames=args.bench_frames,
                                 image=args.image, detector_label=det_label, embedder_label=emb_label,
                                 detector_error=det_err, embedder_error=emb_err)
    print(report)
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    sys.path.insert(0, str(HERE))
    from fewshot_vi.config import ConfigError, load_config
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"error: invalid configuration: {exc}", file=sys.stderr)
        return 2
    logging.basicConfig(level=getattr(logging, str(cfg.logging.level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.mode in ("live", "enrol"):
        return _run_live_or_enrol(args, cfg)
    if args.mode == "eval":
        return _run_eval(args, cfg)
    return _run_benchmark(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
