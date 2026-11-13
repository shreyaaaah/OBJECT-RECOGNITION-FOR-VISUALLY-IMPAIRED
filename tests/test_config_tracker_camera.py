import threading
import time
from pathlib import Path

import numpy as np
import pytest

from fewshot_vi.capture.camera import Camera, CameraError
from fewshot_vi.config import ConfigError, config_from_dict, load_config
from fewshot_vi.tracker.tracker import IoUTracker, iou
from fewshot_vi.types import Detection

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ config
def test_shipped_config_loads_and_matches_spec_defaults():
    c = load_config(ROOT / "config.yaml")
    assert (c.matching.accept_sim, c.matching.ambiguous_sim, c.matching.min_margin, c.matching.knn_k) == (0.70, 0.55, 0.08, 3)
    assert c.temporal.window == 7 and c.temporal.min_votes == 4
    assert c.gallery.max_embeddings_per_object == 20 and c.gallery.update_min_sim == 0.85
    assert c.features.hand_guidance is False and c.features.depth_model is False
    assert c.detector.weights == "yolov8n.pt" and c.embedder.input_size == 224
    assert c.detector.imgsz == 640
    assert c.to_dict()["camera"]["width"] == 640


def test_defaults_without_file():
    assert load_config(None).tracker.reembed_iou == 0.6


def test_cpu_realtime_profile_uses_configurable_smaller_images():
    c = load_config(ROOT / "config.realtime-cpu.yaml")
    assert c.detector.imgsz == 320 and c.embedder.input_size == 140


@pytest.mark.parametrize("over", [
    {"nosuch": {"a": 1}}, {"matching": {"bogus": 1}}, {"matching": {"accept_sim": "high"}},
    {"matching": {"accept_sim": 1.5}}, {"features": {"memory": 1}}, {"temporal": {"window": 0}},
    {"matching": {"accept_sim": 0.5, "ambiguous_sim": 0.6}}, {"temporal": {"window": 3, "min_votes": 5}},
    {"gallery": {"update_min_sim": 0.5}}, {"enrol": {"min_views": 6, "max_views": 5}},
    {"detector": {"backend": "magic"}}, {"features": {"depth_model": True}},
    {"guidance_policy": {"near_zone_frac": 0.5, "slight_zone_frac": 0.2}}, {"matching": "x"},
])
def test_invalid_configs_rejected(over):
    with pytest.raises(ConfigError):
        config_from_dict(over)


def test_bad_files(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("matching: [1, 2")
    with pytest.raises(ConfigError):
        load_config(bad)
    lst = tmp_path / "list.yaml"
    lst.write_text("- 1\n- 2\n")
    with pytest.raises(ConfigError):
        load_config(lst)


# ----------------------------------------------------------------- tracker
def D(x1, y1, x2, y2):
    return Detection(x1, y1, x2, y2, 0.9)


def test_iou_values():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_stable_ids_new_ids_and_no_reuse():
    t = IoUTracker(max_missed=2)
    a = t.update([D(0, 0, 50, 50)])[0].track_id
    assert t.update([D(2, 2, 52, 52)])[0].track_id == a            # moved slightly -> same track
    ids = {x.track_id for x in t.update([D(2, 2, 52, 52), D(200, 200, 260, 260)])}
    assert a in ids and len(ids) == 2
    for _ in range(4):
        t.update([])
    assert t.tracks == {}
    assert t.update([D(0, 0, 50, 50)])[0].track_id not in ids       # ids never reused


def test_prediction_on_skipped_frames_moves_boxes():
    t = IoUTracker()
    t.update([D(0, 0, 50, 50)])
    t.update([D(10, 0, 60, 50)])
    t.update([D(20, 0, 70, 50)])
    before = t.active()[0].bbox
    t.predict()
    after = t.active()[0].bbox
    assert after[0] > before[0]


def test_needs_embed_rules():
    t = IoUTracker(reembed_iou=0.6, reembed_every=30)
    tr = t.update([D(0, 0, 100, 100)])[0]
    assert tr.needs_embed
    t.mark_embedded(tr, np.ones(4, np.float32))
    assert not tr.needs_embed
    t.update([D(2, 2, 102, 102)])
    assert not tr.needs_embed                                      # small move: reuse cached embedding
    t.update([D(40, 0, 140, 100)])                                  # still the same track (IoU~0.44), but < 0.6 vs embedded box
    assert tr.needs_embed
    t.mark_embedded(tr, np.ones(4, np.float32))
    for _ in range(31):
        t.update([D(40, 0, 140, 100)])
    assert tr.needs_embed                                          # 30-frame refresh


def test_tracker_thresholds_are_configurable():
    t = IoUTracker(reembed_every=3)
    tr = t.update([D(0, 0, 100, 100)])[0]
    t.mark_embedded(tr, np.ones(4, np.float32))
    for _ in range(3):
        t.update([D(0, 0, 100, 100)])
    assert tr.needs_embed


# ------------------------------------------------------------------ camera
class FakeCap:
    def __init__(self, frames, opened=True, fail_after=None):
        self.frames, self.i, self.opened, self.released = list(frames), 0, opened, False
        self.fail_after = fail_after

    def isOpened(self):
        return self.opened

    def set(self, *a):
        pass

    def read(self):
        if self.fail_after is not None and self.i >= self.fail_after:
            return False, None
        if self.i >= len(self.frames):
            time.sleep(0.005)
            return False, None
        f = self.frames[self.i]
        self.i += 1
        return True, f

    def release(self):
        self.released = True


def frames(n):
    return [np.full((4, 4, 3), i, np.uint8) for i in range(n)]


def test_camera_open_failure_has_helpful_message():
    with pytest.raises(CameraError, match="cannot open camera 7"):
        Camera(7, capture_factory=lambda i: FakeCap([], opened=False)).start()


def test_camera_drops_oldest_when_consumer_is_slow():
    cap = FakeCap(frames(10))
    cam = Camera(0, capture_factory=lambda i: cap).start()
    time.sleep(0.3)                                                 # nobody reads -> queue (max 2) overflows
    got = [cam.read(0.1)[0][0, 0, 0], cam.read(0.1)[0][0, 0, 0]]
    cam.stop()
    assert got == [8, 9] and cam.dropped == 8                       # only the NEWEST two frames survive
    assert cap.released


def test_camera_reports_error_when_device_stops_delivering():
    cam = Camera(0, capture_factory=lambda i: FakeCap(frames(2), fail_after=2)).start()
    assert cam.finished.wait(2) and cam.error and "stopped delivering" in cam.error
    cam.stop()


def test_camera_stop_is_idempotent_and_joins_thread():
    cam = Camera(0, capture_factory=lambda i: FakeCap(frames(3))).start()
    cam.stop()
    cam.stop()
    assert not cam._thread.is_alive()
