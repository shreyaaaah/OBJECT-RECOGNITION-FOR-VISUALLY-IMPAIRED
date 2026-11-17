"""FastAPI backend server for the Few-Shot Personal Object Recognizer.

Provides REST endpoints for live camera recognition, video-based few-shot registration,
batch registration, object memory queries, and few-shot metric adaptation status.
"""
from __future__ import annotations

import base64
import logging
import time
from contextlib import asynccontextmanager
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import Config, load_config
from ..detector.base import Detector
from ..detector.yolo_detector import YoloDetector
from ..embedder.dino_embedder import DinoEmbedder
from ..gallery.matcher import Matcher
from ..gallery.store import GalleryStore
from ..guidance import geometry
from ..guidance.reviewhint import ReviewHinter
from ..memory.object_memory import ObjectMemory, format_last_seen, humanise_delta
from ..metric_adapter.adapter import FewShotMetricAdapter
from ..metric_adapter.trainer import MetricAdapterTrainer, _compute_cluster_metrics
from ..quality.quality import assess, laplacian_blur
from ..tracker.tracker import IoUTracker
from ..types import Decision, Detection, Match, Sighting, Track
from ..video_registration.video_processor import VideoRegistrationEngine, frame_to_base64_jpeg
from .schemas import (
    AdapterStatusResponse,
    BeepSchema,
    DetectionItemSchema,
    DirectionSchema,
    FindStatusSchema,
    MemoryItemSchema,
    ObjectItemSchema,
    RecognizeResponse,
    RenameObjectRequest,
)

log = logging.getLogger("fewshot_api")
logging.basicConfig(level=logging.INFO)

ROOT_DIR = Path(__file__).resolve().parents[2]
MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_BATCH_BYTES = 300 * 1024 * 1024


class AppContext:
    """Singleton context maintaining models and gallery state."""

    def __init__(self, cfg: Optional[Config] = None, db_path: Optional[str] = None) -> None:
        config_path = ROOT_DIR / "config.yaml"
        if not config_path.exists():
            config_path = None
        self.cfg = cfg or load_config(config_path)
        db_file = db_path or str(ROOT_DIR / "data" / "gallery.db")
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)
        self.store = GalleryStore(db_file, dim=384)

        # Models
        self.detector: Optional[Detector] = None
        self.embedder: Optional[DinoEmbedder] = None
        self.metric_adapter: Optional[FewShotMetricAdapter] = None
        self.matcher: Optional[Matcher] = None
        self.memory: Optional[ObjectMemory] = None
        self.video_engine = VideoRegistrationEngine(
            min_views=self.cfg.enrol.min_views,
            max_views=self.cfg.enrol.max_views,
            augmentations_per_view=self.cfg.enrol.augmentations_per_view,
        )

        # Recognition tracking and temporal smoothing state
        self.tracker = IoUTracker(
            reembed_iou=self.cfg.tracker.reembed_iou,
            reembed_every=self.cfg.tracker.reembed_every_frames,
            match_iou=self.cfg.tracker.match_iou,
            max_missed=self.cfg.tracker.max_missed_frames,
        )
        self.hinter = ReviewHinter(self.cfg.guidance.hint_cooldown_s)
        self.last_announced: Dict[str, float] = {}
        self.frame_count = 0
        self.adapter_checkpoint_path = Path(db_file).with_name("metric_adapter.pt")

    def initialize(self) -> None:
        """Load detector, embedder, and metric adapter."""
        if self.detector is not None and self.embedder is not None and self.matcher is not None:
            return
        # 1. Detector
        weights_path = ROOT_DIR / self.cfg.detector.weights
        if not weights_path.exists():
            weights_path = ROOT_DIR.parent / self.cfg.detector.weights
        self.detector = YoloDetector(
            weights=str(weights_path),
            conf=self.cfg.detector.conf,
            imgsz=self.cfg.detector.imgsz,
        )

        # 2. Embedder (DINOv2)
        self.embedder = DinoEmbedder(
            model=self.cfg.embedder.model,
            input_size=self.cfg.embedder.input_size,
            fp16=self.cfg.embedder.fp16,
        )

        # 3. Metric Adapter
        if self.adapter_checkpoint_path.exists():
            try:
                self.metric_adapter = FewShotMetricAdapter.load(self.adapter_checkpoint_path)
                log.info("Loaded adapted metric projection head from %s", self.adapter_checkpoint_path)
            except Exception as e:
                log.warning("Could not load adapter checkpoint: %s; starting fresh adapter", e)
                self.metric_adapter = FewShotMetricAdapter(in_dim=384, hidden_dim=384, out_dim=256)
        else:
            self.metric_adapter = FewShotMetricAdapter(in_dim=384, hidden_dim=384, out_dim=256)
        # Keep an untrained head inactive. It is fitted only after real examples
        # from at least two enrolled objects exist.
        self.store.adapter = self.metric_adapter
        self.store.use_adapter = False

        # If gallery already has objects, adapt metric head
        if len(np.unique(self.store.owner)) >= 2:
            try:
                metrics = self.store.adapt_metric(epochs=35)
                if metrics:
                    log.info("Adapted metric head on startup: %s", metrics.summary())
                    self.metric_adapter.save(self.adapter_checkpoint_path)
            except Exception as e:
                log.warning("Initial metric adaptation skipped: %s", e)

        # 4. Matcher
        self.matcher = Matcher(
            store=self.store,
            accept_sim=self.cfg.matching.accept_sim,
            ambiguous_sim=self.cfg.matching.ambiguous_sim,
            min_margin=self.cfg.matching.min_margin,
            knn_k=self.cfg.matching.knn_k,
        )

        # 5. Memory
        self.memory = ObjectMemory(
            store=self.store,
            min_conf=self.cfg.memory.min_conf_to_log,
            cooldown_s=self.cfg.memory.log_cooldown_s,
        )


context = AppContext()


def create_app(app_context: Optional[AppContext] = None, load_models: bool = True) -> FastAPI:
    """Create the local web app; tests can inject a context without loading weights."""
    ctx = app_context or context

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            if load_models:
                ctx.initialize()
            yield
        finally:
            if ctx.memory is not None:
                ctx.memory.close()
                ctx.memory = None
            for model in (ctx.detector, ctx.embedder):
                close = getattr(model, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception:
                        log.exception("Error closing a model during web shutdown")
            ctx.store.close()

    app = FastAPI(
        title="Few-Shot Personal Object Recognizer API",
        description="Metric-learning based personal object recognition and spatial audio guidance for visually impaired users.",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.context = ctx

    # ------------------------------------------------------------- Health Check
    @app.get("/api/health")
    def health():
        return {
            "status": "healthy",
            "models_loaded": {
                "detector": ctx.detector is not None,
                "embedder": ctx.embedder is not None,
                "metric_adapter": ctx.metric_adapter is not None,
            },
            "gallery": {
                "objects_count": len(ctx.store.list_objects()),
                "total_embeddings": int(ctx.store.E.shape[0]),
                "use_adapted_metric": ctx.store.use_adapter,
            },
        }

    # ------------------------------------------------------- Live Recognition
    @app.post("/api/recognize", response_model=RecognizeResponse)
    async def recognize(
        frame: UploadFile = File(None),
        frame_base64: Optional[str] = Form(None),
        mode: str = Form("IDENTIFY"),
        target: Optional[str] = Form(None),
    ):
        """Analyze a camera frame: detection -> DINOv2 -> metric adaptation -> spatial audio guidance."""
        t_start = time.perf_counter()
        img_bytes = None
        if frame is not None:
            img_bytes = await frame.read()
        elif frame_base64:
            if "," in frame_base64:
                frame_base64 = frame_base64.split(",", 1)[1]
            img_bytes = base64.b64decode(frame_base64)

        if not img_bytes:
            raise HTTPException(status_code=400, detail="No image provided")

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise HTTPException(status_code=400, detail="Invalid image data")

        fh, fw = img.shape[:2]
        now = time.monotonic()
        if ctx.detector is None or ctx.embedder is None or ctx.matcher is None:
            raise HTTPException(status_code=503, detail="Recognition models are not loaded")
        ctx.frame_count += 1

        # 1. Object Detection (YOLOv8), with tracker prediction between detector
        # passes just like the desktop real-time pipeline.
        detect_every = max(1, ctx.cfg.detector.detect_every_n_frames)
        run_detector = not ctx.tracker.active() or ctx.frame_count % detect_every == 0
        detections = ctx.detector.detect(img) if run_detector else []
        # Update tracker
        ctx.tracker.update(detections)
        active_tracks = ctx.tracker.active()

        # 2. Extract crops for tracks needing re-embedding
        crops_to_embed: List[np.ndarray] = []
        tracks_needing_embed: List[Track] = []

        for t in active_tracks:
            x1, y1, x2, y2 = [int(v) for v in t.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)
            if x2 > x1 and y2 > y1:
                crop = img[y1:y2, x1:x2]
                if t.needs_embed or t.embedding is None:
                    crops_to_embed.append(crop)
                    tracks_needing_embed.append(t)
                t.quality = assess(
                    crop,
                    t.bbox,
                    img.shape,
                    ctx.cfg.quality.blur_var_min,
                    ctx.cfg.quality.edge_margin_px,
                    ctx.cfg.quality.min_box_area_frac,
                )

        if crops_to_embed:
            cls_vecs, patch_vecs = ctx.embedder.embed(crops_to_embed)
            for i, t in enumerate(tracks_needing_embed):
                t.embedding = cls_vecs[i]
                t.patches = patch_vecs[i]
                t.needs_embed = False
                t.last_embed_frame = ctx.frame_count
                t.last_embed_bbox = t.bbox

        # 3. Match each track against personal gallery using Metric Adapter
        matched_items: List[DetectionItemSchema] = []
        spoken_announcements: List[str] = []
        guidance_hint: Optional[str] = None

        target_found = False
        target_dist = None
        target_dir = None
        target_phrase = None
        beep_info = BeepSchema(enabled=False)

        gp = ctx.cfg.guidance_policy
        g = ctx.cfg.guidance

        for t in active_tracks:
            if t.embedding is not None:
                match = ctx.matcher.match(t.embedding)
                t.match = match

                # Temporal vote smoothing
                vote = match.object_id if match.decision == Decision.ACCEPT else None
                t.label_history.append(vote)
                votes = [v for v in t.label_history if v is not None]
                if votes:
                    oid, n = Counter(votes).most_common(1)[0]
                    if n >= ctx.cfg.temporal.min_votes:
                        t.smoothed_id = oid
                        t.smoothed_name = ctx.store.get_name(oid)
                    else:
                        t.smoothed_id = t.smoothed_name = None
                else:
                    t.smoothed_id = t.smoothed_name = None

            # Spatial geometry
            d_dir = geometry.direction(t.bbox, fw, fh, gp.near_zone_frac, gp.slight_zone_frac)
            d_m = geometry.distance_m(
                t.bbox[2] - t.bbox[0], fw, g.camera_hfov_deg, g.known_object_width_cm_default
            )
            dist_phrase = geometry.distance_phrase(d_m)

            # Speak a personal identity only after the configured temporal vote
            # threshold; a single noisy frame must not be announced as certain.
            item_name = t.smoothed_name

            spoken_desc = ""
            if item_name:
                spoken_desc = f"Your {item_name} is {d_dir.horizontal}, {dist_phrase}."

                # Log sighting to object memory
                if (ctx.memory and t.match and t.match.decision == Decision.ACCEPT
                        and t.match.object_id == t.smoothed_id and t.match.score >= 0.75):
                    ctx.memory.offer(
                        t.match.object_id,
                        item_name,
                        t.match.score,
                        t.bbox,
                        scene_label=None,
                    )

                # Announce cooldown check
                last_say = ctx.last_announced.get(item_name, 0.0)
                if now - last_say >= ctx.cfg.guidance.announce_cooldown_s:
                    ctx.last_announced[item_name] = now
                    spoken_announcements.append(spoken_desc)

                # Find mode check
                if mode.upper() == "FIND" and target and target.lower() in item_name.lower():
                    target_found = True
                    target_dist = d_m
                    target_dir = DirectionSchema(
                        horizontal=d_dir.horizontal,
                        vertical=d_dir.vertical,
                        pan=round(d_dir.pan, 2),
                    )
                    target_phrase = f"Found your {item_name}! {d_dir.horizontal}, {dist_phrase}."

                    # Calculate progressive audio beeping rate and pitch
                    # Closer object -> faster beeps (lower interval_ms) and higher pitch (freq_hz)
                    clamped_dist = max(0.2, min(d_m, 3.0))
                    urgency = 1.0 - (clamped_dist - 0.2) / 2.8  # 0 (far) .. 1 (touching)
                    interval_ms = int(100 + (1.0 - urgency) * 700)
                    freq_hz = int(440 + urgency * 500)
                    beep_info = BeepSchema(
                        enabled=True,
                        interval_ms=interval_ms,
                        freq_hz=freq_hz,
                        pan=round(d_dir.pan, 2),
                    )

            det_schema = DetectionItemSchema(
                box=[round(v, 1) for v in t.bbox],
                name=item_name,
                object_id=t.smoothed_id,
                score=round(t.match.score if t.match else 0.0, 3),
                margin=round(t.match.margin if (t.match and np.isfinite(t.match.margin)) else 0.0, 3),
                decision=str(t.match.decision.value if t.match else "UNKNOWN"),
                direction=DirectionSchema(
                    horizontal=d_dir.horizontal,
                    vertical=d_dir.vertical,
                    pan=round(d_dir.pan, 2),
                ),
                distance_m=round(d_m, 2),
                distance_phrase=dist_phrase,
                spoken_phrase=spoken_desc,
            )
            matched_items.append(det_schema)

        # 4. Smart Visual Guidance Hints when recognition is tricky
        for t in active_tracks:
            if t.smoothed_name is None and t.match is not None:
                if t.match.decision == Decision.AMBIGUOUS:
                    guidance_hint = ctx.hinter.hint_for(t.quality, ambiguous=True)
                    if guidance_hint:
                        break
                elif t.quality and not t.quality.ok:
                    guidance_hint = ctx.hinter.hint_for(t.quality, ambiguous=False)
                    if guidance_hint:
                        break

        # In find mode: if target not found, check memory
        find_status = None
        if mode.upper() == "FIND" and target:
            if not target_found and ctx.memory:
                mem_answer = ctx.memory.answer(target)
                target_phrase = mem_answer
            find_status = FindStatusSchema(
                target=target,
                found=target_found,
                distance_m=round(target_dist, 2) if target_dist else None,
                direction=target_dir,
                spoken_phrase=target_phrase,
                beep=beep_info,
            )

        elapsed = time.perf_counter() - t_start
        fps = round(1.0 / max(elapsed, 1e-4), 1)

        return RecognizeResponse(
            detections=matched_items,
            spoken_announcements=spoken_announcements,
            guidance_hint=guidance_hint,
            find_status=find_status,
            fps=fps,
        )

    # ------------------------------------------------ Video Registration
    @app.post("/api/register/video")
    async def register_video(
        video: UploadFile = File(...),
        name: str = Form(...),
        confuser_of: Optional[str] = Form(None),
    ):
        """Register a personal object using a short 5-10 second video."""
        content = await video.read(MAX_VIDEO_BYTES + 1)
        if len(content) > MAX_VIDEO_BYTES:
            raise HTTPException(status_code=413, detail="Video is larger than the 100 MB upload limit")
        if not content:
            raise HTTPException(status_code=400, detail="Empty video file")
        if ctx.embedder is None:
            raise HTTPException(status_code=503, detail="Recognition models are not loaded")
        if ctx.store.get_id(name) is None and len(ctx.store.list_objects()) >= 10:
            raise HTTPException(status_code=400, detail="This project gallery supports at most 10 personal objects")

        try:
            result = ctx.video_engine.register_from_video(
                store=ctx.store,
                embedder=ctx.embedder,
                name=name,
                video_source=content,
                detector=ctx.detector,
                confuser_of=confuser_of,
                adapt_metric=True,
            )
            ctx.metric_adapter = ctx.store.adapter
            # Save adapted adapter weights
            if ctx.metric_adapter and ctx.store.use_adapter:
                try:
                    ctx.metric_adapter.save(ctx.adapter_checkpoint_path)
                except Exception as exc:
                    log.exception("Could not persist the adapted metric checkpoint")
                    result["checkpoint_warning"] = str(exc)
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            log.exception("Video registration failed")
            raise HTTPException(status_code=500, detail="Video registration failed; see server log") from e

    # ------------------------------------------------ Batch Registration
    @app.post("/api/register/batch")
    async def register_batch(
        videos: List[UploadFile] = File(...),
        names: List[str] = Form(...),
        confusers: Optional[List[str]] = Form(None),
    ):
        """Register multiple personal objects in one seamless session."""
        if not 1 <= len(videos) <= 10:
            raise HTTPException(status_code=400, detail="Submit between 1 and 10 object videos per batch")
        if len(videos) != len(names):
            raise HTTPException(status_code=400, detail="Number of videos and names must match")
        if ctx.embedder is None:
            raise HTTPException(status_code=503, detail="Recognition models are not loaded")
        if any(not name.strip() for name in names):
            raise HTTPException(status_code=400, detail="Every video needs a non-empty object name")
        if len({name.strip().casefold() for name in names}) != len(names):
            raise HTTPException(status_code=400, detail="Object names must be unique within a batch")
        current_names = {name.casefold() for _, name in ctx.store.list_objects()}
        new_names = {name.strip().casefold() for name in names} - current_names
        if len(current_names) + len(new_names) > 10:
            raise HTTPException(status_code=400, detail="This project gallery supports at most 10 personal objects")

        results = []
        failures = []
        total_bytes = 0
        confusers_list = confusers or [None] * len(names)

        for i, vid in enumerate(videos):
            name = names[i].strip()
            conf_of = confusers_list[i] if i < len(confusers_list) else None
            content = await vid.read(MAX_VIDEO_BYTES + 1)
            try:
                if not content:
                    raise ValueError("Empty video file")
                if len(content) > MAX_VIDEO_BYTES:
                    raise ValueError("Video is larger than the 100 MB per-video upload limit")
                if total_bytes + len(content) > MAX_BATCH_BYTES:
                    failures.append({"name": name, "detail": "Batch exceeds the 300 MB total upload limit"})
                    continue
                total_bytes += len(content)
                res = ctx.video_engine.register_from_video(
                    store=ctx.store,
                    embedder=ctx.embedder,
                    name=name,
                    video_source=content,
                    detector=ctx.detector,
                    confuser_of=conf_of,
                    adapt_metric=False,  # adapt jointly at the end
                )
                results.append(res)
            except Exception as exc:  # continue; report per-item failures without hiding successful saves
                if isinstance(exc, HTTPException):
                    raise
                log.warning("Batch registration failed for %r: %s", name, exc)
                failures.append({"name": name, "detail": str(exc)})

        # Joint metric adaptation over all registered classes
        adaptation_error = None
        try:
            metrics = ctx.store.adapt_metric(epochs=60)
        except Exception as exc:
            log.exception("Batch metric adaptation failed")
            metrics = None
            adaptation_error = str(exc)
        ctx.metric_adapter = ctx.store.adapter
        if metrics and ctx.metric_adapter:
            try:
                ctx.metric_adapter.save(ctx.adapter_checkpoint_path)
            except Exception as exc:
                log.exception("Could not persist the adapted batch metric checkpoint")
                adaptation_error = str(exc)

        return {
            "success": bool(results),
            "registered_count": len(results),
            "items": results,
            "failures": failures,
            "joint_adaptation": metrics.summary() if metrics else "At least two real object classes are needed for metric learning.",
            "adaptation_error": adaptation_error,
        }

    # ------------------------------------------------ Object Management
    @app.get("/api/objects")
    def list_objects() -> List[ObjectItemSchema]:
        """List all enrolled personal objects."""
        items: List[ObjectItemSchema] = []
        now = time.time()
        for oid, name in ctx.store.list_objects():
            e_count = ctx.store.count(oid)
            conf_grp = ctx.store.confuser_group(oid)
            sighting = ctx.store.last_sighting(oid)
            last_ts = sighting.ts if sighting else None
            last_phrase = format_last_seen(name, sighting, now) if sighting else None
            loc = sighting.scene_label if sighting else None

            # Get created_at from first embedding or now
            items.append(
                ObjectItemSchema(
                    id=oid,
                    name=name,
                    created_at=last_ts or now,
                    embedding_count=e_count,
                    confuser_group=conf_grp,
                    last_seen_ts=last_ts,
                    last_seen_phrase=last_phrase,
                    location=loc,
                )
            )
        return items

    @app.put("/api/objects/{object_id}")
    def rename_object(object_id: int, req: RenameObjectRequest):
        """Rename an enrolled object."""
        try:
            ok = ctx.store.rename_object(object_id, req.name)
            if not ok:
                raise HTTPException(status_code=404, detail="Object not found")
            return {"success": True, "object_id": object_id, "name": req.name}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.delete("/api/objects/{object_id}")
    def delete_object(object_id: int):
        """Delete an enrolled object and retrain metric adapter."""
        name = ctx.store.get_name(object_id)
        if not name:
            raise HTTPException(status_code=404, detail="Object not found")
        ctx.store.delete_object(name)
        # Retrain metric adapter on remaining objects
        # Deleting an enrolled class invalidates the old metric parameters;
        # retrain from scratch on what remains to avoid retaining its influence.
        ctx.store.reset_adapter()
        adaptation_error = None
        try:
            metrics = ctx.store.adapt_metric(epochs=40) if ctx.store.E.shape[0] > 0 else None
        except Exception as exc:
            log.exception("Metric reset after deleting %r failed", name)
            metrics = None
            adaptation_error = str(exc)
        ctx.metric_adapter = ctx.store.adapter
        if metrics and ctx.metric_adapter:
            try:
                ctx.metric_adapter.save(ctx.adapter_checkpoint_path)
            except Exception as exc:
                log.exception("Could not persist metric checkpoint after deletion")
                adaptation_error = str(exc)
                ctx.adapter_checkpoint_path.unlink(missing_ok=True)
        elif ctx.adapter_checkpoint_path.exists() and not ctx.store.use_adapter:
            ctx.adapter_checkpoint_path.unlink(missing_ok=True)
        return {"success": True, "deleted_object": name, "adaptation_error": adaptation_error}

    @app.delete("/api/objects")
    def clear_all_objects():
        """Privacy guarantee: wipe all personal objects, embeddings, and sightings."""
        ctx.store.clear_all()
        ctx.metric_adapter = None
        if ctx.adapter_checkpoint_path.exists():
            ctx.adapter_checkpoint_path.unlink(missing_ok=True)
        return {"success": True, "message": "All personal object data wiped permanently."}

    # ------------------------------------------------ Memory & Find
    @app.get("/api/memory")
    def get_memory() -> List[MemoryItemSchema]:
        """Query last known location and timestamp of all objects."""
        if not ctx.memory:
            return []
        records = ctx.memory.all_last_seen()
        return [MemoryItemSchema(**r) for r in records]

    @app.get("/api/memory/{name}")
    def query_object_location(name: str):
        """Natural language answer: 'Where is my wallet?'."""
        if not ctx.memory:
            return {"answer": "Memory system not initialized."}
        answer = ctx.memory.answer(name)
        return {"name": name, "spoken_answer": answer}

    @app.get("/api/find/{name}")
    def find_target_status(name: str):
        """Query target item status for Find Mode."""
        oid = ctx.store.get_id(name)
        if oid is None:
            raise HTTPException(status_code=404, detail=f"No object registered with name '{name}'")
        sighting = ctx.store.last_sighting(oid)
        return {
            "name": name,
            "object_id": oid,
            "last_sighting": {
                "time_ago": humanise_delta(time.time() - sighting.ts) if sighting else None,
                "location": sighting.scene_label if sighting else None,
                "confidence": sighting.conf if sighting else None,
            }
            if sighting
            else None,
        }

    # ---------------------------------------- Few-Shot Metric Adaptation
    @app.post("/api/adapter/train")
    def trigger_adaptation(epochs: int = Query(50, ge=10, le=200)):
        """Explicitly run metric adaptation training and return margin metrics."""
        if ctx.store.E.shape[0] == 0:
            raise HTTPException(status_code=400, detail="Cannot adapt: gallery is empty")
        metrics = ctx.store.adapt_metric(epochs=epochs)
        ctx.metric_adapter = ctx.store.adapter
        if metrics and ctx.metric_adapter:
            ctx.metric_adapter.save(ctx.adapter_checkpoint_path)
            return {
                "success": True,
                "summary": metrics.summary(),
                "num_classes": metrics.num_classes,
                "num_samples": metrics.num_samples,
                "initial_loss": round(metrics.initial_loss, 4),
                "final_loss": round(metrics.final_loss, 4),
                "intra_sim_before": round(metrics.intra_class_sim_before, 3),
                "intra_sim_after": round(metrics.intra_class_sim_after, 3),
                "inter_sim_before": round(metrics.inter_class_sim_before, 3),
                "inter_sim_after": round(metrics.inter_class_sim_after, 3),
                "margin_gain": round(metrics.margin_gain, 3),
                "training_time_s": round(metrics.training_time_s, 2),
            }
        return {"success": False, "message": "No adaptation performed"}

    @app.get("/api/adapter/status", response_model=AdapterStatusResponse)
    def adapter_status():
        """Retrieve metric adaptation statistics."""
        E, owner = ctx.store.E, ctx.store.owner
        intra, inter, margin = _compute_cluster_metrics(
            ctx.store.E_adapted if ctx.store.use_adapter else E, owner
        )
        classes = len(np.unique(owner)) if owner.size else 0
        return AdapterStatusResponse(
            active=ctx.store.use_adapter and (ctx.metric_adapter is not None),
            num_classes=classes,
            total_embeddings=int(E.shape[0]),
            in_dim=384,
            out_dim=ctx.metric_adapter.out_dim if ctx.metric_adapter else 384,
            margin=round(margin, 3),
            intra_class_sim=round(intra, 3),
            inter_class_sim=round(inter, 3),
        )

    # ------------------------------------------------ Frontend Static Files
    frontend_dist = ROOT_DIR / "frontend" / "dist"
    if not frontend_dist.exists():
        frontend_dist = ROOT_DIR.parent / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

    return app


app = create_app()
