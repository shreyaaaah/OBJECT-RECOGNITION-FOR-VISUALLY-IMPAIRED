"""End-to-end behaviour of the real Pipeline using only fakes (no camera/GPU/mic/models/network)."""
import io
import time

import numpy as np
import pytest

from fewshot_vi.config import config_from_dict
from fewshot_vi.gallery.store import GalleryStore
from fewshot_vi.pipeline import Pipeline
from fewshot_vi.types import Decision, Mode
from fewshot_vi.voice import beeps, stt, tts

from conftest import DIM, basis, near, unit
from fakes import (FakeCamera, FakeEmbedder, RecordingBackend, ScriptedDetector, det, textured_frame)

BOX = (200, 180, 300, 280)    # 100 px wide, 15% left of centre; 0.08 m * 502.3 px / 100 px = 0.40 m -> "half a metre"
GREEN, RED = 1, 2


def make_cfg(**over):
    base = {
        "detector": {"detect_every_n_frames": 1},
        "temporal": {"window": 3, "min_votes": 2},
        "gallery": {"update_cooldown_s": 0},
        "memory": {"log_cooldown_s": 0, "min_conf_to_log": 0.8},
        "guidance": {"announce_cooldown_s": 1000, "hint_cooldown_s": 0},
        "rerank": {"deadline_s": 5.0},
    }
    for k, v in over.items():
        base.setdefault(k, {}).update(v)
    return config_from_dict(base)


class Rig:
    def __init__(self, cfg=None, frames=(), script=None, embedder=None, store=None, with_reranker=True):
        self.cfg = cfg or make_cfg()
        self.store = store or GalleryStore(":memory:", dim=DIM)
        self.backend = RecordingBackend()
        self.speaker = tts.Speaker(self.backend, stale_s=1000)
        self.camera = FakeCamera(frames)
        self.detector = ScriptedDetector(script or [[det(*BOX)]])
        self.embedder = embedder or FakeEmbedder({GREEN: basis(0), RED: basis(1)})
        self.beeps = []
        self.pipe = Pipeline(self.cfg, self.store, self.detector, self.embedder, self.camera, self.speaker,
                             beeper=beeps.Beeper(player=lambda d, sr: self.beeps.append(d)))

    def enrol(self, name, vec):
        oid = self.store.add_object(name)
        self.store.add_embedding(oid, vec, "enrol")
        return oid

    def run_frames(self, frame, n):
        out = []
        for _ in range(n):
            out += self.pipe.step(frame, time.perf_counter())
        return out

    def said(self):
        self.speaker.wait_idle(2)
        return list(self.backend.spoken)

    def close(self):
        self.pipe.close()


@pytest.fixture
def rig():
    r = Rig()
    yield r
    r.close()


def test_recognises_enrolled_object_and_announces_with_direction_distance(rig):
    rig.enrol("keys", basis(0))
    frame = textured_frame([(*BOX, GREEN)])
    events = rig.run_frames(frame, 6)
    announces = [e for e in events if e.kind == "announce"]
    assert len(announces) == 1                      # once, not every frame (cooldown)
    assert announces[0].text == "Keys, slightly left, about half a metre."
    assert rig.pipe.frame_errors == 0 and rig.pipe.frames_ok == 6
    assert "Keys, slightly left, about half a metre." in rig.said()


def test_no_speech_before_enough_stable_votes(rig):
    rig.enrol("keys", basis(0))
    frame = textured_frame([(*BOX, GREEN)])
    assert rig.run_frames(frame, 1) == []           # 1 vote < min_votes (2)
    assert [e.kind for e in rig.run_frames(frame, 1)] == ["announce"]


def test_unknown_object_is_never_named(rig):
    rig.enrol("keys", basis(0))
    frame = textured_frame([(*BOX, RED)])           # RED maps to basis(1): unrelated to the gallery
    events = rig.run_frames(frame, 8)
    assert all(e.kind != "announce" for e in events)
    assert not any("Keys" in s for s in rig.said())
    tr = rig.pipe.tracker.active()[0]
    assert tr.match.decision == Decision.UNKNOWN and tr.smoothed_name is None


def test_embedding_is_computed_once_per_track_not_per_frame(rig):
    rig.enrol("keys", basis(0))
    rig.run_frames(textured_frame([(*BOX, GREEN)]), 12)
    assert rig.embedder.calls == 1


def test_all_new_tracks_are_embedded_in_one_batch():
    r = Rig(script=[[det(100, 100, 200, 200), det(400, 100, 500, 200), det(250, 300, 350, 400)]])
    try:
        r.enrol("keys", basis(0))
        r.run_frames(textured_frame([(100, 100, 200, 200, GREEN), (400, 100, 500, 200, RED),
                                     (250, 300, 350, 400, GREEN)]), 1)
        assert r.embedder.batch_sizes == [3]
    finally:
        r.close()


def test_single_frame_error_does_not_stop_the_loop():
    r = Rig(script=[RuntimeError("detector exploded"), [det(*BOX)]])
    try:
        r.enrol("keys", basis(0))
        frame = textured_frame([(*BOX, GREEN)])
        assert r.pipe.step(frame, 0.0) == [] and r.pipe.frame_errors == 1
        events = r.run_frames(frame, 5)
        assert any(e.kind == "announce" for e in events) and r.pipe.frame_errors == 1
    finally:
        r.close()


def test_detect_only_every_n_frames_and_tracks_persist():
    r = Rig(cfg=make_cfg(detector={"detect_every_n_frames": 3}))
    try:
        r.enrol("keys", basis(0))
        r.run_frames(textured_frame([(*BOX, GREEN)]), 7)
        assert r.detector.calls == 3                 # frames 1, 4, 7
        assert len(r.pipe.tracker.active()) == 1     # one stable track across skipped frames
    finally:
        r.close()


def test_stage_timings_and_fps_are_recorded(rig):
    rig.enrol("keys", basis(0))
    rig.run_frames(textured_frame([(*BOX, GREEN)]), 5)
    for stage in ("detect", "embed", "match", "guidance", "total"):
        assert rig.pipe.stats.mean_ms(stage) is not None, stage
    assert rig.pipe.stats.count("recognition_to_speech_enqueue") >= 1
    assert "frames ok=5" in rig.pipe.timing_report()


def test_sightings_are_logged_asynchronously_with_scene_and_bbox():
    r = Rig(cfg=make_cfg(memory={"log_cooldown_s": 10}))
    try:
        r.pipe.scene.label = "desk"
        oid = r.enrol("keys", basis(0))
        r.run_frames(textured_frame([(*BOX, GREEN)]), 5)
        r.pipe.memory.flush()
        s = r.pipe.memory.last_seen("keys")
        assert s is not None and s.scene_label == "desk" and s.conf >= 0.8
        assert s.bbox == pytest.approx(BOX, abs=1.0)
        assert r.store.sighting_count(oid) == 1       # 10 s per-object cooldown debounces repeated frames
    finally:
        r.close()


def test_no_sighting_for_unknown_objects(rig):
    oid = rig.enrol("keys", basis(0))
    rig.run_frames(textured_frame([(*BOX, RED)]), 6)
    rig.pipe.memory.flush()
    assert rig.store.sighting_count() == 0


def test_self_improving_gallery_adds_auto_embedding_once_stable():
    emb = FakeEmbedder({GREEN: near(basis(0), 0.9, axis=5)})   # sim 0.9 >= update_min_sim 0.85
    r = Rig(embedder=emb)
    try:
        oid = r.enrol("keys", basis(0))
        r.run_frames(textured_frame([(*BOX, GREEN)]), 10)
        assert r.store.count(oid, "enrol") == 1 and r.store.count(oid, "auto") == 1
        assert r.store.patches_of(oid) == []          # enrolment stored no patches here, auto never does
    finally:
        r.close()


def test_self_improve_off_in_config_adds_nothing():
    emb = FakeEmbedder({GREEN: near(basis(0), 0.9, axis=5)})
    r = Rig(cfg=make_cfg(features={"self_improve": False}), embedder=emb)
    try:
        oid = r.enrol("keys", basis(0))
        r.run_frames(textured_frame([(*BOX, GREEN)]), 10)
        assert r.store.count(oid, "auto") == 0
    finally:
        r.close()


def test_low_quality_blocks_gallery_update():
    emb = FakeEmbedder({GREEN: near(basis(0), 0.9, axis=5)})
    r = Rig(embedder=emb, script=[[det(0, 180, 100, 280)]])   # box touches the left frame edge
    try:
        oid = r.enrol("keys", basis(0))
        r.run_frames(textured_frame([(0, 180, 100, 280, GREEN)]), 10)
        assert r.store.count(oid, "auto") == 0
    finally:
        r.close()


def test_active_hint_for_ambiguous_edge_cut_object():
    # a mid-similarity (ambiguous band) object cut by the left edge -> concise re-view hint
    emb = FakeEmbedder({GREEN: near(basis(0), 0.62, axis=5)})
    r = Rig(embedder=emb, script=[[det(0, 180, 100, 280)]])
    try:
        r.enrol("keys", basis(0))
        events = r.run_frames(textured_frame([(0, 180, 100, 280, GREEN)]), 4)
        hints = [e.text for e in events if e.kind == "hint"]
        assert hints == ["Move the camera left."]    # one hint; not repeated back-to-back
        assert all(e.kind != "announce" for e in events)
    finally:
        r.close()


def test_lookalike_rerank_runs_async_and_can_change_the_result():
    a, b = near(basis(0), 0.99, axis=7), basis(0)
    rng = np.random.default_rng(3)
    pa, pb = rng.normal(size=(8, DIM)).astype(np.float32), rng.normal(size=(8, DIM)).astype(np.float32)
    # query global embedding is slightly closer to mug_b, but its patches belong to mug_a
    q = unit(0.97 * basis(0) + 0.02 * a + 0.0 * basis(3))
    emb = FakeEmbedder({GREEN: near(b, 0.985, axis=11)}, {GREEN: pa + 0.02 * rng.normal(size=pa.shape)})
    r = Rig(embedder=emb)
    try:
        store = r.store
        ia, ib = store.add_object("mug_a"), store.add_object("mug_b")
        store.add_embedding(ia, a, "enrol", patches=pa)
        store.add_embedding(ib, b, "enrol", patches=pb)
        store.set_confusers("mug_a", "mug_b")
        frame = textured_frame([(*BOX, GREEN)])
        first = r.pipe.step(frame, 0.0)
        tr = r.pipe.tracker.active()[0]
        assert tr.match.decision == Decision.AMBIGUOUS and tr.match.name == "mug_b"   # provisional (global) result
        assert first == []                                                              # nothing spoken yet
        for _ in range(200):                           # async result is folded in on a later frame
            r.pipe.step(frame, 0.0)
            if tr.smoothed_name:
                break
            time.sleep(0.01)
        assert tr.match.name == "mug_a" and tr.smoothed_name == "mug_a"
        assert any(s.startswith("Mug_a") or s.startswith("Mug_A") or "mug_a" in s.lower() for s in r.said())
        assert r.pipe.stats.count("rerank") >= 1
    finally:
        r.close()


def test_confuser_without_rerank_agreement_blocks_auto_update():
    a, b = near(basis(0), 0.99, axis=7), basis(0)
    pa = np.random.default_rng(4).normal(size=(8, DIM)).astype(np.float32)
    emb = FakeEmbedder({GREEN: near(b, 0.985, axis=11)}, {GREEN: pa})
    r = Rig(embedder=emb)
    try:
        ia, ib = r.store.add_object("mug_a"), r.store.add_object("mug_b")
        r.store.add_embedding(ia, a, "enrol", patches=pa)
        r.store.add_embedding(ib, b, "enrol", patches=np.random.default_rng(5).normal(size=(8, DIM)).astype(np.float32))
        r.store.set_confusers("mug_a", "mug_b")
        frame = textured_frame([(*BOX, GREEN)])
        for _ in range(60):
            r.pipe.step(frame, 0.0)
            time.sleep(0.005)
        assert r.store.count(ia, "auto") == 0 and r.store.count(ib, "auto") == 0
    finally:
        r.close()


# --------------------------------------------------------------- voice commands
def test_find_mode_ignores_other_objects_and_beeps():
    r = Rig(script=[[det(100, 180, 200, 280), det(400, 180, 500, 280)]])
    try:
        r.enrol("keys", basis(0))
        r.enrol("wallet", basis(1))
        r.pipe.handle_text("find my wallet")
        assert r.pipe.state.mode == Mode.FIND and r.pipe.state.target == "wallet"
        frame = textured_frame([(100, 180, 200, 280, GREEN), (400, 180, 500, 280, RED)])
        events = r.run_frames(frame, 6)
        finds = [e for e in events if e.kind == "find"]
        assert finds and all(e.text.startswith("Wallet") for e in finds)
        assert not any(e.kind == "announce" for e in events)
        assert r.beeps and r.beeps[0].shape[1] == 2
    finally:
        r.close()


def test_memory_query_voice_command(rig):
    rig.enrol("keys", basis(0))
    rig.run_frames(textured_frame([(*BOX, GREEN)]), 5)
    rig.pipe.memory.flush()
    rig.pipe.handle_text("where did I leave my keys")
    assert any("keys" in s.lower() and "just saw" in s.lower() for s in rig.said())


def test_unknown_name_and_gibberish_get_spoken_errors(rig):
    rig.enrol("keys", basis(0))
    rig.pipe.handle_text("find my passport")
    rig.pipe.handle_text("banana smoothie recipe")
    said = " | ".join(rig.said())
    assert "don't have an object called passport" in said and "did not understand" in said
    assert rig.pipe.state.mode == Mode.IDENTIFY


def test_mute_and_resume(rig):
    rig.enrol("keys", basis(0))
    rig.pipe.handle_text("stop")
    assert rig.pipe.state.muted
    assert rig.run_frames(textured_frame([(*BOX, GREEN)]), 5) == []
    rig.pipe.handle_text("resume")
    assert not rig.pipe.state.muted
    assert any(e.kind == "announce" for e in rig.run_frames(textured_frame([(*BOX, GREEN)]), 5))


def test_list_objects(rig):
    rig.pipe.handle_text("list objects")
    rig.enrol("keys", basis(0))
    rig.enrol("wallet", basis(1))
    rig.pipe.handle_text("list objects")
    said = rig.said()
    assert "You have no objects yet." in said and "You have keys, wallet." in said


def test_delete_requires_confirmation(rig):
    rig.enrol("keys", basis(0))
    rig.pipe.handle_text("delete keys")
    rig.pipe.handle_text("no")
    rig.pipe._service_requests()
    assert rig.store.names() == ["keys"]
    rig.pipe.handle_text("delete keys")
    rig.pipe.handle_text("yes")
    rig.pipe._service_requests()
    assert rig.store.names() == [] and rig.store.E.shape[0] == 0
    said = rig.said()
    assert any("Say yes to confirm" in s for s in said) and "keys deleted." in said and "Okay, I will keep it." in said


def test_confuser_command_after_enrolment(rig):
    rig.enrol("blue mug", basis(0))
    rig.enrol("red mug", basis(1))
    rig.pipe.handle_text("this is similar to blue mug")      # nothing enrolled this session yet
    assert rig.store.confuser_group(rig.store.get_id("blue mug")) is None
    rig.pipe.last_enrolled = "red mug"
    rig.pipe.handle_text("this is similar to blue mug")
    assert rig.store.are_confusers(rig.store.get_id("red mug"), rig.store.get_id("blue mug"))


def test_voice_input_goes_through_listener_thread_not_vision_loop():
    r = Rig()
    try:
        r.enrol("keys", basis(0))
        r.pipe.stt = stt.KeyboardSTT(io.StringIO("find keys\n"), printer=lambda s: None)
        r.pipe.start_background()
        for _ in range(200):
            if r.pipe.state.mode == Mode.FIND:
                break
            time.sleep(0.01)
        assert r.pipe.state.mode == Mode.FIND and r.pipe.state.target == "keys"
    finally:
        r.close()


# ---------------------------------------------------------------- run / shutdown
def test_run_loop_processes_all_frames_then_shuts_down_cleanly():
    frame = textured_frame([(*BOX, GREEN)])
    r = Rig(frames=[frame] * 8)
    r.enrol("keys", basis(0))
    r.pipe.run()
    p = r.pipe
    assert p.frames_ok == 8 and p.frame_errors == 0
    assert r.camera.stopped and r.detector.closed and r.embedder.closed and p.state.shutting_down
    p.close()  # idempotent


def test_run_stops_on_camera_error_and_releases_resources():
    r = Rig(frames=[])
    r.camera.error = "camera stopped delivering frames"
    r.pipe.run()
    assert r.camera.stopped and r.pipe.frames_ok == 0


def test_run_honours_shutdown_request_and_max_frames():
    frame = textured_frame([(*BOX, GREEN)])
    r = Rig(frames=[frame] * 50)
    r.pipe.run(max_frames=5)
    assert r.pipe.frames_ok == 5 and r.camera.stopped


def test_keyboard_interrupt_still_closes_everything():
    class Boom(FakeCamera):
        def read(self, timeout=1.0):
            raise KeyboardInterrupt
    r = Rig()
    r.pipe.camera = Boom([])
    r.pipe.run()
    assert r.pipe.state.shutting_down and r.pipe.camera.stopped


def test_overlay_draws_without_error():
    from fewshot_vi.pipeline import draw_overlay
    r = Rig()
    try:
        r.enrol("keys", basis(0))
        frame = textured_frame([(*BOX, GREEN)])
        r.run_frames(frame, 4)
        out = draw_overlay(frame, r.pipe.tracker.active(), 12.3, r.pipe.stats)
        assert out.shape == frame.shape and (out != frame).any()
    finally:
        r.close()
