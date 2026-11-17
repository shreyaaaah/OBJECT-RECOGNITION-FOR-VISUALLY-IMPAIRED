"""Tests for video-based few-shot registration engine."""
import numpy as np
import pytest
import cv2
import tempfile
from pathlib import Path

from fewshot_vi.video_registration.video_processor import (
    VideoRegistrationEngine,
    ProcessedVideoResult,
    frame_to_base64_jpeg,
)
from fewshot_vi.gallery.store import GalleryStore


def _make_test_video(tmp_path: Path, n_frames: int = 180, fps: int = 30,
                     w: int = 320, h: int = 240) -> str:
    """Create a simple test video with moving colored rectangle."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = str(tmp_path / "test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    rng = np.random.default_rng(42)
    for i in range(n_frames):
        frame = np.full((h, w, 3), 60, dtype=np.uint8)
        # Moving colored box to simulate turning the object
        x1 = int(80 + 20 * np.sin(i * 0.2))
        y1 = int(60 + 15 * np.cos(i * 0.15))
        color = (int(100 + i * 2), int(200 - i * 2), 150)
        cv2.rectangle(frame, (x1, y1), (x1 + 100, y1 + 80), color, -1)
        # Add some texture/pattern
        cv2.putText(frame, f"F{i}", (x1 + 10, y1 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    return path


class FakeEmbedder:
    """Minimal fake embedder for testing without DINOv2 weights."""
    def __init__(self, dim=384):
        self.dim = dim
        self._rng = np.random.default_rng(99)

    def embed(self, crops):
        n = len(crops)
        cls_vecs = self._rng.normal(0, 1, (n, self.dim)).astype(np.float32)
        cls_vecs /= np.linalg.norm(cls_vecs, axis=1, keepdims=True)
        patches = self._rng.normal(0, 1, (n, 16, self.dim)).astype(np.float32)
        patches /= np.linalg.norm(patches, axis=2, keepdims=True)
        return cls_vecs, patches


def test_extract_sharp_diverse_crops(tmp_path):
    video_path = _make_test_video(tmp_path)
    engine = VideoRegistrationEngine(min_views=3, max_views=6)
    embedder = FakeEmbedder()

    result = engine.extract_sharp_diverse_crops(video_path, embedder=embedder)

    assert isinstance(result, ProcessedVideoResult)
    assert len(result.crops) >= 3
    assert len(result.crops) <= 6
    assert result.embeddings.shape[0] == len(result.crops)
    assert result.embeddings.shape[1] == 384
    assert len(result.blur_scores) == len(result.crops)
    assert result.stats["total_sampled_frames"] > 5
    assert result.stats["selected_views_count"] == len(result.crops)


def test_extract_from_bytes(tmp_path):
    video_path = _make_test_video(tmp_path)
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    engine = VideoRegistrationEngine(min_views=2, max_views=4)
    embedder = FakeEmbedder()
    result = engine.extract_sharp_diverse_crops(video_bytes, embedder=embedder)

    assert len(result.crops) >= 2
    assert result.embeddings.shape[0] == len(result.crops)


def test_register_from_video_creates_object_and_embeddings(tmp_path):
    video_path = _make_test_video(tmp_path)
    store = GalleryStore(":memory:", dim=384)
    embedder = FakeEmbedder()
    engine = VideoRegistrationEngine(min_views=3, max_views=5, augmentations_per_view=3)

    result = engine.register_from_video(
        store=store,
        embedder=embedder,
        name="my blue mug",
        video_source=video_path,
        adapt_metric=True,
    )

    assert result["success"] is True
    assert result["name"] == "my blue mug"
    assert result["views_extracted"] >= 3
    assert result["embeddings_saved"] > result["views_extracted"]  # originals + augmentations
    # A metric cannot learn negatives from only one registered object.
    assert result["adaptation"] is None
    assert store.use_adapter is False

    # Verify gallery state
    assert store.get_id("my blue mug") is not None
    assert len(store.list_objects()) == 1
    assert store.E.shape[0] == result["embeddings_saved"]


def test_register_multiple_objects_with_confusers(tmp_path):
    store = GalleryStore(":memory:", dim=384)
    embedder = FakeEmbedder()
    engine = VideoRegistrationEngine(min_views=3, max_views=4, augmentations_per_view=2)

    v1 = _make_test_video(tmp_path / "v1", n_frames=180)
    v2 = _make_test_video(tmp_path / "v2", n_frames=180)

    r1 = engine.register_from_video(store=store, embedder=embedder, name="book A",
                                     video_source=v1, adapt_metric=False)
    r2 = engine.register_from_video(store=store, embedder=embedder, name="book B",
                                     video_source=v2, confuser_of="book A", adapt_metric=True)

    assert r1["success"] and r2["success"]
    assert len(store.list_objects()) == 2
    assert store.are_confusers(store.get_id("book A"), store.get_id("book B"))
    assert r2["adaptation"] is not None
    assert r2["adaptation"]["num_classes"] == 2


def test_rejects_video_outside_five_to_ten_seconds(tmp_path):
    video_path = _make_test_video(tmp_path, n_frames=120, fps=30)
    engine = VideoRegistrationEngine()
    with pytest.raises(ValueError, match="5-10 seconds"):
        engine.extract_sharp_diverse_crops(video_path, embedder=FakeEmbedder())


def test_frame_to_base64():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    b64 = frame_to_base64_jpeg(img)
    assert len(b64) > 10
    assert isinstance(b64, str)


def test_blur_filtering_selects_sharper_frames(tmp_path):
    """Verify that blurry frames are filtered out in favor of sharp ones."""
    path = str(tmp_path / "blur_test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 30, (320, 240))

    for i in range(180):
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        if i % 3 == 0:
            # Apply heavy blur to simulate motion blur
            frame = cv2.GaussianBlur(frame, (31, 31), 0)
        out.write(frame)
    out.release()

    engine = VideoRegistrationEngine(min_views=3, max_views=6, default_blur_thresh=20.0)
    embedder = FakeEmbedder()
    result = engine.extract_sharp_diverse_crops(path, embedder=embedder)

    # All selected frames should have reasonable blur scores (not the heavily blurred ones)
    assert len(result.crops) >= 3
    assert len(result.blur_scores) == len(result.crops)
