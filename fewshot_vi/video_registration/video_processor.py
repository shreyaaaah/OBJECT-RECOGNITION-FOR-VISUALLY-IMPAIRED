"""Video-based few-shot personal object registration engine.

Extracts useful frames from 5-10 second videos, rejects blurry frames via Laplacian
variance, eliminates near-duplicate frames via embedding cosine distance, crops the object,
extracts DINOv2 embeddings, and triggers few-shot metric adaptation.
"""
from __future__ import annotations

import base64
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from ..gallery.enrol import augment
from ..gallery.store import GalleryError, GalleryStore
from ..quality.quality import laplacian_blur

log = logging.getLogger(__name__)


@dataclass
class ProcessedVideoResult:
    """Outcome of processing an enrollment video."""
    crops: List[np.ndarray]
    embeddings: np.ndarray
    patches: np.ndarray
    blur_scores: List[float]
    selected_indices: List[int]
    stats: dict


def frame_to_base64_jpeg(img: np.ndarray, quality: int = 80) -> str:
    """Encode an image as base64 JPEG string for preview/web display."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


class VideoRegistrationEngine:
    def __init__(
        self,
        min_views: int = 3,
        max_views: int = 8,
        default_blur_thresh: float = 45.0,
        duplicate_sim_thresh: float = 0.94,
        augmentations_per_view: int = 6,
        min_duration_s: float = 5.0,
        max_duration_s: float = 10.0,
    ) -> None:
        self.min_views = min_views
        self.max_views = max_views
        self.default_blur_thresh = default_blur_thresh
        self.duplicate_sim_thresh = duplicate_sim_thresh
        self.augmentations_per_view = augmentations_per_view
        self.min_duration_s = min_duration_s
        self.max_duration_s = max_duration_s

    def extract_sharp_diverse_crops(
        self,
        video_source: Union[str, Path, bytes],
        detector: Any = None,
        embedder: Any = None,
    ) -> ProcessedVideoResult:
        """Process a video file or byte buffer into diverse, sharp object crops.

        Args:
            video_source: File path or raw video bytes (webm/mp4/avi).
            detector: Optional object detector (YoloDetector or similar).
            embedder: DinoEmbedder to extract feature vectors and evaluate diversity.
        """
        temp_file: Optional[Path] = None
        if isinstance(video_source, bytes):
            # Keep the actual browser container extension so OpenCV/FFmpeg can
            # choose the right demuxer (Chromium normally records WebM).
            if video_source.startswith(b"\x1aE\xdf\xa3"):
                suffix = ".webm"
            elif len(video_source) > 12 and video_source[4:8] == b"ftyp":
                suffix = ".mp4"
            elif video_source.startswith(b"RIFF"):
                suffix = ".avi"
            else:
                suffix = ".mp4"
            tf = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tf.write(video_source)
            tf.flush()
            tf.close()
            temp_file = Path(tf.name)
            cap_path = str(temp_file)
        else:
            cap_path = str(video_source)

        cap = cv2.VideoCapture(cap_path)
        if not cap.isOpened():
            if temp_file and temp_file.exists():
                temp_file.unlink(missing_ok=True)
            raise ValueError(f"Cannot open video from {cap_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_s = frame_count / fps if fps > 0 and frame_count > 0 else None
        if duration_s is not None and not self.min_duration_s <= duration_s <= self.max_duration_s:
            cap.release()
            if temp_file and temp_file.exists():
                temp_file.unlink(missing_ok=True)
            raise ValueError(f"Registration video must be {self.min_duration_s:.0f}-{self.max_duration_s:.0f} "
                             f"seconds; received {duration_s:.1f} seconds")

        # Sample at ~3-4 frames per second to cover the clip evenly
        sample_step = max(1, int(fps / 3.5))

        raw_frames: List[Tuple[int, np.ndarray, float]] = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if idx % sample_step == 0:
                blur_score = laplacian_blur(frame)
                raw_frames.append((idx, frame, blur_score))
            idx += 1
        cap.release()

        # Some camera containers omit frame-count metadata. Validate duration
        # from decoded frames in that case instead of silently accepting clips.
        # Use the number of decoded frames as the final duration check. Some
        # browser-generated containers report a stale or zero frame count.
        duration_s = idx / fps if fps > 0 else (duration_s or 0.0)
        if not self.min_duration_s <= duration_s <= self.max_duration_s:
            if temp_file and temp_file.exists():
                temp_file.unlink(missing_ok=True)
            raise ValueError(f"Registration video must be {self.min_duration_s:.0f}-{self.max_duration_s:.0f} "
                             f"seconds; received {duration_s:.1f} seconds")

        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)

        if not raw_frames:
            raise ValueError("No frames could be read from the video")

        total_sampled = len(raw_frames)

        # 1. Blur Filtering
        # Filter frames by Laplacian variance
        sharp_frames = [f for f in raw_frames if f[2] >= self.default_blur_thresh]
        # Fallback if user's camera has lower blur variance overall
        blur_thresh_used = self.default_blur_thresh
        if len(sharp_frames) < self.min_views:
            # Sort by sharpness and pick the sharpest
            raw_frames.sort(key=lambda x: x[2], reverse=True)
            sharp_frames = raw_frames[: max(self.min_views, 8)]
            blur_thresh_used = float(sharp_frames[-1][2]) if sharp_frames else 0.0

        # 2. Object Localization / Cropping
        candidate_crops: List[Tuple[int, np.ndarray, float]] = []
        for f_idx, frame, b_score in sharp_frames:
            crop = self._crop_object(frame, detector)
            if crop is not None and crop.size > 0:
                candidate_crops.append((f_idx, crop, b_score))

        if not candidate_crops:
            # Saliency central fallback
            for f_idx, frame, b_score in sharp_frames:
                h, w = frame.shape[:2]
                crop = frame[int(h * 0.15) : int(h * 0.85), int(w * 0.15) : int(w * 0.85)]
                candidate_crops.append((f_idx, crop, b_score))

        # 3. Embedding and Deduplication Filtering
        selected_crops: List[np.ndarray] = []
        selected_indices: List[int] = []
        selected_blur: List[float] = []
        selected_embeds: List[np.ndarray] = []
        selected_patches: List[np.ndarray] = []

        if embedder is not None:
            # Embed all candidate crops
            crops_only = [c[1] for c in candidate_crops]
            all_cls, all_patch = embedder.embed(crops_only)

            for i, (f_idx, crop, b_score) in enumerate(candidate_crops):
                emb = all_cls[i]
                if len(selected_embeds) == 0:
                    selected_crops.append(crop)
                    selected_indices.append(f_idx)
                    selected_blur.append(b_score)
                    selected_embeds.append(emb)
                    selected_patches.append(all_patch[i])
                    continue

                # Cosine similarity with already accepted views
                existing_matrix = np.stack(selected_embeds)
                max_sim = float((existing_matrix @ emb).max())

                # If sufficiently distinct from existing views (not duplicate angle)
                if max_sim < self.duplicate_sim_thresh:
                    selected_crops.append(crop)
                    selected_indices.append(f_idx)
                    selected_blur.append(b_score)
                    selected_embeds.append(emb)
                    selected_patches.append(all_patch[i])

                if len(selected_crops) >= self.max_views:
                    break

            # If deduplication was too aggressive, fill up to min_views from candidates
            if len(selected_crops) < self.min_views:
                for i, (f_idx, crop, b_score) in enumerate(candidate_crops):
                    if f_idx not in selected_indices:
                        selected_crops.append(crop)
                        selected_indices.append(f_idx)
                        selected_blur.append(b_score)
                        selected_embeds.append(all_cls[i])
                        selected_patches.append(all_patch[i])
                    if len(selected_crops) >= self.min_views:
                        break

            final_cls = np.stack(selected_embeds) if selected_embeds else np.zeros((0, 384), dtype=np.float32)
            final_patches = np.stack(selected_patches) if selected_patches else np.zeros((0, 0, 384), dtype=np.float32)
        else:
            # Stub/test fallback without embedder
            for f_idx, crop, b_score in candidate_crops[: self.max_views]:
                selected_crops.append(crop)
                selected_indices.append(f_idx)
                selected_blur.append(b_score)
            final_cls = np.zeros((len(selected_crops), 384), dtype=np.float32)
            final_patches = np.zeros((len(selected_crops), 1, 384), dtype=np.float32)

        stats = {
            "total_sampled_frames": total_sampled,
            "sharp_frames_count": len(sharp_frames),
            "blur_thresh_used": blur_thresh_used,
            "selected_views_count": len(selected_crops),
            "duration_s": round(duration_s, 2),
        }

        return ProcessedVideoResult(
            crops=selected_crops,
            embeddings=final_cls,
            patches=final_patches,
            blur_scores=selected_blur,
            selected_indices=selected_indices,
            stats=stats,
        )

    def _crop_object(self, frame: np.ndarray, detector: Any) -> Optional[np.ndarray]:
        """Detect central personal item or extract salient central region."""
        h, w = frame.shape[:2]
        if detector is not None and hasattr(detector, "detect"):
            dets = detector.detect(frame)
            if dets:
                cx, cy = w / 2.0, h / 2.0
                best_det = None
                best_score = -1e9
                for d in dets:
                    dw = d.x2 - d.x1
                    dh = d.y2 - d.y1
                    area = dw * dh
                    area_fraction = area / float(w * h)
                    # Avoid learning the user/background as the object. The
                    # user is asked to hold the item centrally and fill much of
                    # the frame; large person boxes are not useful object crops.
                    if area_fraction < 0.01 or area_fraction > 0.55 or d.score < 0.20:
                        continue
                    if (d.class_name or "").casefold() == "person":
                        continue
                    dist = np.hypot((d.x1 + d.x2) / 2.0 - cx, (d.y1 + d.y2) / 2.0 - cy)
                    normalized_dist = dist / max(float(np.hypot(cx, cy)), 1.0)
                    score = 2.0 * area_fraction - normalized_dist
                    if score > best_score:
                        best_score = score
                        best_det = d
                if best_det is not None:
                    pad_x = int(0.05 * (best_det.x2 - best_det.x1))
                    pad_y = int(0.05 * (best_det.y2 - best_det.y1))
                    x1 = max(0, int(best_det.x1) - pad_x)
                    y1 = max(0, int(best_det.y1) - pad_y)
                    x2 = min(w, int(best_det.x2) + pad_x)
                    y2 = min(h, int(best_det.y2) + pad_y)
                    return frame[y1:y2, x1:x2]

        # Central salient crop if detector found no bounding box
        x1, y1 = int(w * 0.20), int(h * 0.20)
        x2, y2 = int(w * 0.80), int(h * 0.80)
        return frame[y1:y2, x1:x2]

    def register_from_video(
        self,
        store: GalleryStore,
        embedder: Any,
        name: str,
        video_source: Union[str, Path, bytes],
        detector: Any = None,
        confuser_of: Optional[str] = None,
        adapt_metric: bool = True,
        seed: int = 1234,
    ) -> dict:
        """Complete workflow: processes video, registers views + augmentations, and adapts metric space."""
        name = name.strip()
        if not name:
            raise ValueError("Object name must not be empty")

        processed = self.extract_sharp_diverse_crops(video_source, detector=detector, embedder=embedder)
        if len(processed.crops) == 0:
            raise ValueError("Could not extract any clear object crops from the provided video")

        # Build enrollment batch: each original crop + deterministic augmentations
        all_embeddings_to_save: List[Tuple[np.ndarray, Optional[np.ndarray]]] = []
        for i, crop in enumerate(processed.crops):
            cls_vec = processed.embeddings[i]
            patch_vec = processed.patches[i]
            all_embeddings_to_save.append((cls_vec, patch_vec))

            # Augmentations for robustness
            augmented_crops = augment(crop, self.augmentations_per_view, seed + i * 17)
            if augmented_crops:
                aug_cls, _ = embedder.embed(augmented_crops)
                for a_cls in aug_cls:
                    all_embeddings_to_save.append((a_cls, None))

        # Create the object only after every view and augmentation succeeded so
        # a failed video never leaves an empty personal object in the gallery.
        obj_id = store.get_id(name)
        if obj_id is None:
            obj_id = store.add_object(name)

        saved_count = store.add_embeddings_batch(obj_id, all_embeddings_to_save, source="enrol")

        if confuser_of:
            try:
                store.set_confusers(name, confuser_of)
            except Exception as e:
                log.warning("Could not set confuser %s: %s", confuser_of, e)

        # Trigger few-shot metric adaptation
        adaptation_info = None
        adaptation_error = None
        if adapt_metric:
            try:
                metrics = store.adapt_metric(epochs=50)
                if metrics is not None:
                    adaptation_info = {
                        "num_classes": metrics.num_classes,
                        "num_samples": metrics.num_samples,
                        "loss_improvement": round(metrics.initial_loss - metrics.final_loss, 4),
                        "margin_before": round(metrics.margin_before, 3),
                        "margin_after": round(metrics.margin_after, 3),
                        "margin_gain": round(metrics.margin_gain, 3),
                        "summary": metrics.summary(),
                    }
            except Exception as exc:  # Enrollment is already safely saved; report the fit problem separately.
                log.exception("Metric adaptation failed after saving %s", name)
                adaptation_error = str(exc)

        # Generate thumbnail previews for UI
        thumbnails = [frame_to_base64_jpeg(c, quality=75) for c in processed.crops[:4]]

        return {
            "success": True,
            "object_id": obj_id,
            "name": name,
            "views_extracted": len(processed.crops),
            "embeddings_saved": saved_count,
            "blur_scores": [round(b, 1) for b in processed.blur_scores],
            "stats": processed.stats,
            "adaptation": adaptation_info,
            "adaptation_error": adaptation_error,
            "thumbnails": thumbnails,
        }
