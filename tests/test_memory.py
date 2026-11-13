import time

import pytest

from fewshot_vi.memory.object_memory import ObjectMemory, format_last_seen, humanise_delta
from fewshot_vi.memory.scene import ClipSceneClassifier, FixedSceneClassifier, NullSceneClassifier
from fewshot_vi.types import Sighting

import numpy as np

from conftest import FakeClock


@pytest.mark.parametrize("secs,text", [
    (0, "just now"), (59, "just now"), (60, "1 minute ago"), (12 * 60, "12 minutes ago"),
    (59 * 60, "59 minutes ago"), (3600, "1 hour ago"), (5 * 3600, "5 hours ago"),
    (23 * 3600, "23 hours ago"), (24 * 3600, "yesterday"), (47 * 3600, "yesterday"),
    (3 * 86400, "3 days ago"), (-5, "just now"),
])
def test_humanise_delta(secs, text):
    assert humanise_delta(secs) == text


def sight(ts, scene="desk", name="keys"):
    return Sighting(1, name, ts, scene, (10, 10, 50, 50), 0.9)


def test_format_never_seen():
    assert format_last_seen("keys", None, 1000.0) == "I haven't seen your keys yet."


def test_format_minutes_hours_yesterday_just_now():
    now = 100000.0
    assert format_last_seen("keys", sight(now - 12 * 60), now) == "Your keys were last seen 12 minutes ago, on the desk."
    assert "2 hours ago" in format_last_seen("wallet", sight(now - 7200, "kitchen", "wallet"), now)
    assert format_last_seen("wallet", sight(now - 7200, "kitchen", "wallet"), now).endswith("in the kitchen.")
    assert "yesterday" in format_last_seen("wallet", sight(now - 30 * 3600, None, "wallet"), now)
    assert format_last_seen("wallet", sight(now - 30 * 3600, None, "wallet"), now) == "Your wallet was last seen yesterday."
    assert format_last_seen("wallet", sight(now - 5, "desk", "wallet"), now) == "I just saw your wallet, on the desk."


def test_format_without_scene_has_no_location():
    assert format_last_seen("keys", sight(0, None), 120.0) == "Your keys were last seen 2 minutes ago."


@pytest.fixture
def mem(store):
    clk = FakeClock(0.0)
    wall = FakeClock(1_000_000.0)
    m = ObjectMemory(store, min_conf=0.80, cooldown_s=10.0, clock=clk, wall=wall)
    m._clk, m._wall = clk, wall
    yield m
    m.close()


def test_logs_only_confident_sightings(store, mem):
    oid = store.add_object("keys")
    assert not mem.offer(oid, "keys", 0.79, (1, 2, 3, 4), "desk")
    assert mem.offer(oid, "keys", 0.81, (1, 2, 3, 4), "desk")
    mem.flush()
    assert store.sighting_count(oid) == 1


def test_per_object_cooldown(store, mem):
    a, b = store.add_object("keys"), store.add_object("wallet")
    assert mem.offer(a, "keys", 0.9, (1, 2, 3, 4), "desk")
    mem._clk.advance(5)
    assert not mem.offer(a, "keys", 0.9, (1, 2, 3, 4), "desk")   # within cooldown
    assert mem.offer(b, "wallet", 0.9, (1, 2, 3, 4), "desk")     # different object is independent
    mem._clk.advance(6)
    assert mem.offer(a, "keys", 0.9, (1, 2, 3, 4), "desk")
    mem.flush()
    assert store.sighting_count(a) == 2 and store.sighting_count(b) == 1


def test_last_seen_returns_structured_data(store, mem):
    oid = store.add_object("keys")
    assert mem.last_seen("keys") is None and mem.last_seen("nothing") is None
    mem.offer(oid, "keys", 0.95, (10.0, 20.0, 60.0, 80.0), "kitchen")
    mem.flush()
    s = mem.last_seen("keys")
    assert isinstance(s, Sighting)
    assert s.name == "keys" and s.scene_label == "kitchen" and s.bbox == (10.0, 20.0, 60.0, 80.0)
    assert s.ts == 1_000_000.0 and s.conf == pytest.approx(0.95)


def test_latest_sighting_wins(store, mem):
    oid = store.add_object("keys")
    mem.offer(oid, "keys", 0.9, (0, 0, 1, 1), "desk")
    mem._clk.advance(20)
    mem._wall.advance(20)
    mem.offer(oid, "keys", 0.9, (5, 5, 9, 9), "sofa")
    mem.flush()
    assert mem.last_seen("keys").scene_label == "sofa"


def test_answer_texts(store, mem):
    oid = store.add_object("keys")
    assert mem.answer("keys") == "I haven't seen your keys yet."
    assert mem.answer("passport") == "I don't have an object called passport."
    mem.offer(oid, "keys", 0.9, (0, 0, 1, 1), "desk")
    mem.flush()
    assert mem.answer("keys", now=1_000_000.0 + 12 * 60) == "Your keys were last seen 12 minutes ago, on the desk."


def test_offer_never_blocks_on_slow_db(store):
    release = []
    orig = store.add_sighting

    def slow(s, blob=None):
        while not release:
            time.sleep(0.01)
        return orig(s, blob)
    store.add_sighting = slow
    m = ObjectMemory(store, 0.5, 0.0, max_pending=2)
    oid = store.add_object("keys")
    t0 = time.perf_counter()
    for _ in range(10):
        m.offer(oid, "keys", 0.9, (0, 0, 1, 1), "desk")
    assert time.perf_counter() - t0 < 0.5   # no stall even though the writer is stuck
    assert m.dropped > 0                     # overflow is dropped, not blocking
    release.append(1)
    m.close()


def test_scene_classifiers_never_fabricate_labels():
    frame = np.zeros((48, 64, 3), np.uint8)
    null = NullSceneClassifier()
    assert null.available is False and null.classify(frame) is None
    fixed = FixedSceneClassifier("desk")
    label, emb = fixed.classify(frame)
    assert label == "desk" and emb.shape == (4,)


def test_clip_classifier_disables_itself_when_unavailable(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake_import(name, *a, **k):
        if name in ("open_clip", "torch"):
            raise ImportError(f"No module named {name!r}")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    c = ClipSceneClassifier()
    assert c.classify(np.zeros((48, 64, 3), np.uint8)) is None
    assert c.available is False and "ImportError" in (c.reason or "")
