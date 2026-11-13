"""Guided enrolment workflow, dialogue plumbing, hand guidance in the pipeline, evaluation and the CLI."""
import threading
import time

import cv2
import numpy as np
import pytest

import run as run_cli
from fewshot_vi.config import config_from_dict
from fewshot_vi.embedder.color_hist import ColorHistEmbedder
from fewshot_vi.eval import benchmark as bm
from fewshot_vi.eval.episodes import DatasetError, sample_episodes, scan_dataset
from fewshot_vi.eval.metrics import FpsMeter, LatencyStats, OpenSetCounts, top1_accuracy
from fewshot_vi.eval.run_eval import run_ablation
from fewshot_vi.gallery import enrol as en
from fewshot_vi.gallery.store import GalleryStore
from fewshot_vi.types import Mode

from conftest import DIM, FakeClock, basis, near, unit
from fakes import FakeCamera, ScriptedDetector, det, textured_frame
from test_pipeline_smoke import Rig, make_cfg, BOX, GREEN


# ------------------------------------------------------------------ enrolment
class ViewEmbedder:
    """Returns a DIFFERENT view vector per call (like a rotated object); augmentations are close to it."""
    def __init__(self):
        self.calls = 0

    def embed(self, crops):
        j = self.calls
        self.calls += 1
        base = near(basis(0), 0.8, axis=1 + (j % 8))                  # successive views differ (cos ~0.64 apart)
        cls = np.stack([base] + [near(base, 0.97, axis=20 + i % 5) for i in range(len(crops) - 1)])
        pat = np.random.default_rng(j).normal(size=(len(crops), 100, DIM)).astype(np.float32)
        return cls.astype(np.float32), pat


def enrol_cfg(**over):
    base = {"enrol": {"stable_frames": 2, "timeout_s": 30}}
    for k, v in over.items():
        base.setdefault(k, {}).update(v)
    return config_from_dict(base)


class Env:
    def __init__(self, frames, answers, cfg=None, embedder=None, store=None):
        self.cfg = cfg or enrol_cfg()
        self.store = store or GalleryStore(":memory:", dim=DIM)
        self.clock = FakeClock(0.0)
        self.frames = list(frames)
        self.i = 0
        self.said, self.asked = [], []
        self.answers = list(answers)
        self.det = ScriptedDetector([[det(140, 100, 340, 300)]])
        self.emb = embedder or ViewEmbedder()

    def get_frame(self):
        self.clock.advance(0.1)
        f = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return f

    def ask(self, prompt):
        self.asked.append(prompt)
        return self.answers.pop(0) if self.answers else None

    def run(self):
        return en.run_enrolment(cfg=self.cfg, store=self.store, detector=self.det, embedder=self.emb,
                                get_frame=self.get_frame, speak=self.said.append, ask=self.ask, clock=self.clock)


SHARP = textured_frame([(140, 100, 340, 300, GREEN)])
BLURRY = cv2.GaussianBlur(SHARP, (41, 41), 0)


def test_enrolment_saves_views_with_augmentations_and_patches_only_for_enrol():
    e = Env([SHARP], ["blue mug", "yes"])
    r = e.run()
    assert r.success and r.name == "blue mug" and 3 <= r.views <= 5
    assert r.embeddings == r.views * (1 + e.cfg.enrol.augmentations_per_view)   # 9 embeddings per view
    oid = e.store.get_id("blue mug")
    assert e.store.count(oid, "enrol") == r.embeddings and e.store.count(oid, "auto") == 0
    assert len(e.store.patches_of(oid)) == r.embeddings and all(p.shape[0] <= 32 for p in e.store.patches_of(oid))
    assert any("Rotate it slowly" in s for s in e.said) and "Hold the object steadily" in e.said[0]


def test_blurred_views_rejected_with_spoken_hint_then_good_views_accepted():
    e = Env([BLURRY] * 12 + [SHARP], ["keys", "yes"])
    r = e.run()
    assert r.success and "Hold the object steady." in e.said
    assert e.det.calls > 12


def test_edge_cut_views_rejected():
    cut = textured_frame([(0, 100, 200, 300, GREEN)])
    e = Env([cut] * 30, [])
    e.det = ScriptedDetector([[det(0, 100, 200, 300)]])
    r = e.run()
    assert not r.success and r.views == 0 and "Keep the whole object inside the camera view." in e.said


def test_timeout_with_too_few_views_saves_nothing():
    e = Env([BLURRY], ["keys", "yes"], cfg=enrol_cfg(enrol={"timeout_s": 3}))
    r = e.run()
    assert not r.success and e.store.names() == [] and "Nothing was saved" in r.message


def test_no_detector_proposals_prompt_user_and_explain_timeout():
    e = Env([SHARP], [], cfg=enrol_cfg(enrol={"timeout_s": 3}))
    e.det = ScriptedDetector([[]])
    r = e.run()
    assert not r.success and r.views == 0 and e.store.names() == []
    assert any("can't see an object" in message.lower() for message in e.said)
    assert "did not propose any boxes" in r.message


def test_duplicate_views_ask_for_rotation():
    class SameEmb(ViewEmbedder):
        def embed(self, crops):
            cls, pat = super().embed(crops)
            return np.stack([basis(0)] * len(crops)).astype(np.float32), pat
    e = Env([SHARP], ["keys", "yes"], embedder=SameEmb(), cfg=enrol_cfg(enrol={"timeout_s": 5}))
    r = e.run()
    assert r.views == 1 and not r.success and "Rotate it a little more." in e.said


def test_name_must_be_confirmed_and_unique():
    e = Env([SHARP], ["keys", "no", "wallet", "no", "purse", "no"])
    r = e.run()
    assert not r.success and e.store.names() == [] and "No confirmed name" in r.message
    s = GalleryStore(":memory:", dim=DIM)
    s.add_object("keys")
    e2 = Env([SHARP], ["Keys", "wallet", "yes", "no"], store=s)
    r2 = e2.run()
    assert r2.success and r2.name == "wallet"
    assert any("already have an object called Keys" in x for x in e2.said)


def test_confuser_offer_links_objects():
    s = GalleryStore(":memory:", dim=DIM)
    other = s.add_object("blue mug")
    s.add_embedding(other, basis(5), "enrol")
    e = Env([SHARP], ["red mug", "yes", "yes", "blue mug"], store=s)
    r = e.run()
    assert r.success and r.confuser_of == "blue mug" and s.are_confusers(other, s.get_id("red mug"))
    s2 = GalleryStore(":memory:", dim=DIM)
    o2 = s2.add_object("blue mug")
    s2.add_embedding(o2, basis(5), "enrol")
    r2 = Env([SHARP], ["red mug", "yes", "yes", "passport"], store=s2).run()
    assert r2.success and r2.confuser_of is None and not s2.are_confusers(o2, s2.get_id("red mug"))


def test_augmentations_are_deterministic_and_varied():
    img = np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    a, b, c = en.augment(img, 8, 7), en.augment(img, 8, 7), en.augment(img, 8, 8)
    assert len(a) == 8 and all(x.shape == img.shape for x in a)
    assert all((x == y).all() for x, y in zip(a, b))
    assert any((x != y).any() for x, y in zip(a, c))
    assert not any((x == img).all() for x in a)


def test_pick_central_box_prefers_large_central_and_ignores_small():
    dets = [det(0, 0, 30, 30), det(240, 160, 400, 320), det(500, 50, 560, 120)]
    assert en.pick_central_box(dets, (480, 640, 3), 0.05) == dets[1]
    assert en.pick_central_box([det(0, 0, 10, 10)], (480, 640, 3), 0.05) is None
    assert en.is_yes("yes please") and en.is_yes("Yeah") and not en.is_yes("no") and not en.is_yes(None)


# -------------------------------------------- pipeline: dialogue + enrol request + hands
def test_ask_waits_for_reply_from_voice_thread_and_enrol_request_runs_in_vision_loop():
    from fewshot_vi.voice import stt
    r = Rig()
    try:
        r.pipe._listener = object()                                    # pretend a voice thread is running
        out = []
        t = threading.Thread(target=lambda: out.append(r.pipe.ask("What is your name?")))
        t.start()
        for _ in range(200):
            if r.pipe._dialog_active.is_set():
                break
            time.sleep(0.005)
        r.pipe.handle_text("blue mug")                                 # goes to the dialogue, NOT parsed as a command
        t.join(2)
        assert out == ["blue mug"] and r.pipe.state.mode == Mode.IDENTIFY
        r.pipe.handle_text("register object")
        assert r.pipe.state.drain() == [("enrol", None)]
    finally:
        r.pipe._listener = None
        r.close()


def test_hand_guidance_is_off_by_default_and_speaks_when_enabled():
    class FakeHands:
        def centroid(self, frame):
            return (50.0, 230.0)
        def close(self):
            pass
    frame = textured_frame([(*BOX, GREEN)])
    off = Rig()
    off.pipe.hands = None
    off.enrol("keys", basis(0))
    off.pipe.handle_text("find keys")
    assert not any(e.kind == "hand" for e in off.run_frames(frame, 6))
    off.close()
    on = Rig(cfg=make_cfg(features={"hand_guidance": True}))
    on.pipe.hands = FakeHands()
    on.enrol("keys", basis(0))
    on.pipe.handle_text("find keys")
    ev = [e for e in on.run_frames(frame, 6) if e.kind == "hand"]
    assert ev and ev[0].text == "Move your hand right."
    on.close()


# ----------------------------------------------------------------------- eval
def make_dataset(root, classes=8, per=14, seed=0):
    """SYNTHETIC colour-texture images: used only to test the evaluation PLUMBING, not to claim accuracy."""
    rng = np.random.default_rng(seed)
    hues = np.linspace(0, 170, classes).astype(int)
    for c in range(classes):
        d = root / f"class{c}"
        d.mkdir(parents=True)
        for i in range(per):
            hsv = np.zeros((64, 64, 3), np.uint8)
            hsv[..., 0] = np.clip(hues[c] + rng.integers(-3, 4, (64, 64)), 0, 179)
            hsv[..., 1] = rng.integers(150, 255, (64, 64))
            hsv[..., 2] = rng.integers(120, 255, (64, 64))
            cv2.imwrite(str(d / f"{i}.jpg"), cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))


def test_dataset_scan_and_helpful_errors(tmp_path):
    with pytest.raises(DatasetError, match="dataset folder not found"):
        scan_dataset(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(DatasetError, match="no class folders"):
        scan_dataset(tmp_path / "empty")
    make_dataset(tmp_path / "ds", classes=3, per=4)
    idx = scan_dataset(tmp_path / "ds")
    assert sorted(idx) == ["class0", "class1", "class2"]
    with pytest.raises(DatasetError, match="too small"):
        sample_episodes(idx, n_way=5, k_shot=3, n_query=5, n_unknown=3, n_episodes=2, seed=0)
    with pytest.raises(DatasetError, match="n_unknown must be >= 1"):
        sample_episodes(idx, n_way=2, k_shot=1, n_query=2, n_unknown=0, n_episodes=1, seed=0)


def test_episodes_are_reproducible_disjoint_and_have_unknown_classes(tmp_path):
    make_dataset(tmp_path / "ds", classes=9, per=12)
    idx = scan_dataset(tmp_path / "ds")
    a = sample_episodes(idx, 4, 3, 5, 2, 5, seed=11)
    b = sample_episodes(idx, 4, 3, 5, 2, 5, seed=11)
    c = sample_episodes(idx, 4, 3, 5, 2, 5, seed=12)
    assert a == b and a != c
    for ep in a:
        assert len(ep.known) == 4 and len(ep.unknown) == 2 and not set(ep.known) & set(ep.unknown)
        for k in ep.known:
            assert len(ep.support[k]) == 3 and len(ep.query_known[k]) == 5
            assert not set(ep.support[k]) & set(ep.query_known[k])
        assert all(len(v) == 5 for v in ep.query_unknown.values())


def test_metrics_helpers():
    c = OpenSetCounts()
    for t, p in [("a", "a"), ("a", None), ("a", "b"), (None, None), (None, "a"), (None, None)]:
        c.add(t, p)
    assert (c.top1_accuracy, c.frr, c.misid_rate) == (pytest.approx(1 / 3), pytest.approx(1 / 3), pytest.approx(1 / 3))
    assert c.far == pytest.approx(1 / 3)
    assert top1_accuracy([1, 2], [1, 3]) == 0.5
    s = LatencyStats()
    for v in (0.01, 0.02, 0.03):
        s.record("x", v)
    assert s.mean_ms("x") == pytest.approx(20) and s.p95_ms("x") == pytest.approx(29, abs=0.5) and s.mean_ms("y") is None
    f = FpsMeter()
    for i in range(11):
        f.tick(i * 0.1)
    assert f.fps == pytest.approx(10.0)


def test_full_ablation_runs_on_real_image_files_and_writes_csv(tmp_path):
    make_dataset(tmp_path / "ds", classes=8, per=14)
    cfg = config_from_dict({"temporal": {"window": 2, "min_votes": 2}})
    out = tmp_path / "res" / "abl.csv"
    rows, report = run_ablation(cfg, ColorHistEmbedder(), ColorHistEmbedder.name, tmp_path / "ds", ks=(1, 3),
                                n_way=4, n_query=4, n_unknown=2, episodes=3, seed=5, sessions=2, out_csv=out)
    abls = {r["ablation"] for r in rows}
    assert abls == {"1 baseline", "2 cascade", "3 look-alike subset", "4 degraded views (simulated re-view)",
                    "5 sequential self-update"}
    assert "color-hist (non-learned baseline, NOT DINOv2)" in report and out.exists()
    base = {r["metric"]: r["value"] for r in rows if r["ablation"] == "1 baseline" and r["k_shot"] == 3}
    assert 0.0 <= base["top1_acc"] <= 1.0 and base["known_n"] == 3 * 4 * 4 and base["unknown_n"] == 3 * 2 * 4
    assert "n/a" not in report.lower().replace("undef", "")
    ref = {r["config"]: r["value"] for r in rows if r["metric"] == "slow_path_frac" and r["k_shot"] == 3}
    assert ref["reference: always patch re-rank (no cascade)"] >= ref["+ cascade (patch re-rank only if ambiguous)"]
    assert ref["reference: always patch re-rank (no cascade)"] > 0.9
    again, _ = run_ablation(cfg, ColorHistEmbedder(), "x", tmp_path / "ds", ks=(3,), n_way=4, n_query=4,
                            n_unknown=2, episodes=3, seed=5, sessions=2)
    a1 = [r["value"] for r in rows if r["k_shot"] == 3 and r["metric"] == "top1_acc" and r["ablation"] == "1 baseline"]
    a2 = [r["value"] for r in again if r["metric"] == "top1_acc" and r["ablation"] == "1 baseline"]
    assert a1 == a2                                                    # reproducible with the same seed


def test_ablation_refuses_random_init_model_and_small_dataset(tmp_path):
    make_dataset(tmp_path / "ds", classes=3, per=6)
    cfg = config_from_dict()
    with pytest.raises(DatasetError, match="too small"):
        run_ablation(cfg, ColorHistEmbedder(), "x", tmp_path / "ds", n_way=5)
    make_dataset(tmp_path / "ds2", classes=9, per=12)
    with pytest.raises(DatasetError, match="random-initialised"):
        run_ablation(cfg, ColorHistEmbedder(), bm.RANDOM_INIT_LABEL, tmp_path / "ds2", ks=(1,), n_way=4)


# ------------------------------------------------------------------ benchmark
def test_benchmark_reports_missing_stages_honestly_and_measures_available_ones():
    cfg = config_from_dict()
    rows, report = bm.run_benchmark(cfg, None, None, iterations=3, detector_error="ModuleNotFoundError: no ultralytics",
                                    embedder_error="no weights")
    by = {r.stage: r for r in rows}
    assert by["detect (1 frame)"].status == "NOT MEASURED" and "ultralytics" in by["detect (1 frame)"].note
    assert by["embed (batch of 1 crops)"].status == "NOT MEASURED"
    assert by["end-to-end loop"].status == "NOT MEASURED"
    assert by["match (global embedding)"].status == "measured" and by["match (global embedding)"].mean_ms > 0
    assert by["rerank (patch, 2 candidates)"].p95_ms >= 0 and "NOT MEASURED" in report


def test_benchmark_synthetic_overhead_run_end_to_end():
    cfg = config_from_dict()
    rows, report = bm.run_benchmark(cfg, bm.OverheadDetector(), bm.OverheadEmbedder(), iterations=3, frames=10,
                                    detector_label="SYNTHETIC stub", embedder_label="SYNTHETIC stub")
    tot = [r for r in rows if "end-to-end frame" in r.stage]
    assert len(tot) == 2 and all(r.status == "measured" and r.n >= 8 for r in tot)
    assert any("stress" in r.stage for r in tot) and "synthetic overhead stubs" in report


# ------------------------------------------------------------------------ CLI
def test_cli_error_paths_and_exit_codes(tmp_path, capsys):
    assert run_cli.main(["--mode", "eval", "--ablation"]) == 2
    assert "--dataset" in capsys.readouterr().err
    assert run_cli.main(["--mode", "eval", "--ablation", "--dataset", str(tmp_path / "x"), "--embedder", "color-hist"]) == 2
    assert "dataset folder not found" in capsys.readouterr().err
    bad = tmp_path / "bad.yaml"
    bad.write_text("matching: {accept_sim: 3}\n")
    assert run_cli.main(["--mode", "benchmark", "--config", str(bad)]) == 2
    assert "invalid configuration" in capsys.readouterr().err
    assert run_cli.main(["--mode", "live", "--camera", "99", "--keyboard-only", "--db", str(tmp_path / "g.db")]) == 1
    assert "cannot open camera" in capsys.readouterr().err
    with pytest.raises(SystemExit) as ei:
        run_cli.main(["--help"])
    assert ei.value.code == 0


def test_cli_eval_ablation_end_to_end(tmp_path, capsys):
    make_dataset(tmp_path / "ds", classes=8, per=14)
    out = tmp_path / "o.csv"
    code = run_cli.main(["--mode", "eval", "--ablation", "--dataset", str(tmp_path / "ds"), "--embedder", "color-hist",
                         "--k-shot", "3", "--n-way", "4", "--n-query", "4", "--n-unknown", "2", "--episodes", "2",
                         "--out", str(out)])
    txt = capsys.readouterr().out
    assert code == 0 and out.exists() and "== ablation 5 sequential self-update ==" in txt and "NOT DINOv2" in txt


def test_cli_benchmark_synthetic(capsys):
    assert run_cli.main(["--mode", "benchmark", "--synthetic", "--iterations", "3", "--bench-frames", "8"]) == 0
    assert "SYNTHETIC stub" in capsys.readouterr().out
