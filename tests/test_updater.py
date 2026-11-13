import logging

import numpy as np
import pytest

from fewshot_vi.gallery.updater import GalleryUpdateWorker, GalleryUpdater
from fewshot_vi.types import Decision, Match, Quality

from conftest import DIM, basis, near, unit


def good_q(**over):
    kw = dict(blur=100.0, blurry=False, edge_cut=frozenset(), too_small=False, low_margin=False, score=1.0)
    kw.update(over)
    return Quality(**kw)


def accept(oid, score=0.9, margin=0.3):
    return Match(oid, "keys", score, margin, Decision.ACCEPT, None, [(oid, score)])


def mk(store, clock, **kw):
    args = dict(window=3, max_per_object=4, min_sim=0.85, min_margin=0.15, cooldown_s=5.0,
                duplicate_sim=0.97, drift_min_sim=0.60, clock=clock)
    args.update(kw)
    return GalleryUpdater(store, **args)


@pytest.fixture
def obj(store):
    oid = store.add_object("keys")
    store.add_embedding(oid, basis(0), "enrol")
    return oid


def test_adds_auto_embedding_when_every_guard_passes(store, clock, obj):
    up = mk(store, clock)
    r = up.maybe_add(obj, near(basis(0), 0.9), accept(obj), good_q(), stable_accepts=3)
    assert r.added, r.reason
    assert store.count(obj, "auto") == 1 and store.count(obj, "enrol") == 1
    assert store.patches_of(obj) == []  # auto embeddings never carry patches


@pytest.mark.parametrize("label,kwargs,reason", [
    ("unstable", dict(stable_accepts=2), "stable"),
    ("low score", dict(match=accept(1, score=0.80)), "score"),
    ("low margin", dict(match=accept(1, margin=0.10)), "margin"),
    ("bad quality", dict(quality=good_q(blurry=True)), "quality"),
    ("missing quality", dict(quality=None), "quality"),
    ("not accepted", dict(match=Match(1, "keys", 0.9, 0.3, Decision.AMBIGUOUS)), "accepted"),
])
def test_rejections(store, clock, obj, label, kwargs, reason):
    up = mk(store, clock)
    args = dict(embedding=near(basis(0), 0.9), match=accept(obj), quality=good_q(), stable_accepts=3)
    args.update(kwargs)
    r = up.maybe_add(obj, **args)
    assert not r.added and reason in r.reason
    assert store.count(obj, "auto") == 0


def test_infinite_margin_single_object_passes(store, clock, obj):
    up = mk(store, clock)
    r = up.maybe_add(obj, near(basis(0), 0.9), accept(obj, margin=float("inf")), good_q(), 3)
    assert r.added


def test_cooldown(store, clock, obj):
    up = mk(store, clock)
    assert up.maybe_add(obj, near(basis(0), 0.90), accept(obj), good_q(), 3).added
    r = up.maybe_add(obj, near(basis(0), 0.88, axis=DIM - 2), accept(obj), good_q(), 3)
    assert not r.added and "cooldown" in r.reason
    clock.advance(5.1)
    assert up.maybe_add(obj, near(basis(0), 0.88, axis=DIM - 2), accept(obj), good_q(), 3).added


def test_near_duplicate_rejected(store, clock, obj):
    up = mk(store, clock)
    r = up.maybe_add(obj, basis(0), accept(obj), good_q(), 3)  # identical to the enrol vector
    assert not r.added and "duplicate" in r.reason


def test_confuser_must_agree(store, clock, obj):
    other = store.add_object("keys2")
    store.add_embedding(other, basis(1), "enrol")
    store.set_confusers("keys", "keys2")
    up = mk(store, clock)
    r = up.maybe_add(obj, near(basis(0), 0.9), accept(obj), good_q(), 3, confuser_agrees=False)
    assert not r.added and "confuser" in r.reason
    assert up.maybe_add(obj, near(basis(0), 0.9), accept(obj), good_q(), 3, confuser_agrees=True).added


def test_cap_replaces_most_redundant_auto_and_never_enrol(store, clock):
    oid = store.add_object("keys")
    e1 = store.add_embedding(oid, basis(0), "enrol")
    e2 = store.add_embedding(oid, near(basis(0), 0.5, axis=1), "enrol")
    # two autos: a_red is very close to the enrol cluster (redundant), a_div is far from everything
    a_red = store.add_embedding(oid, near(basis(0), 0.95, axis=2), "auto")
    a_div = store.add_embedding(oid, near(basis(0), 0.62, axis=5), "auto")
    assert store.count(oid) == 4
    up = mk(store, clock, max_per_object=4, drift_min_sim=0.0)
    newv = near(basis(0), 0.90, axis=9)
    r = up.maybe_add(oid, newv, accept(oid), good_q(), 3)
    assert r.added and r.replaced_id == a_red
    assert store.count(oid) == 4
    ids = {row[0] for row in store.embedding_rows(oid)}
    assert {e1, e2} <= ids and a_div in ids and a_red not in ids
    assert store.count(oid, "enrol") == 2


def test_cap_reached_with_only_enrol_embeddings_skips_update(store, clock):
    oid = store.add_object("keys")
    for i in range(3):
        store.add_embedding(oid, near(basis(0), 0.5 + 0.1 * i, axis=i + 1), "enrol")
    up = mk(store, clock, max_per_object=3)
    r = up.maybe_add(oid, near(basis(0), 0.9, axis=9), accept(oid), good_q(), 3)
    assert not r.added and "no auto" in r.reason
    assert store.count(oid, "enrol") == 3


def test_drift_guard_freezes_updates_and_warns(store, clock, caplog):
    oid = store.add_object("keys")
    store.add_embedding(oid, basis(0), "enrol")
    up = mk(store, clock, drift_min_sim=0.60)
    with caplog.at_level(logging.WARNING):
        r = up.maybe_add(oid, near(basis(0), 0.30), accept(oid), good_q(), 3)  # far from the enrolment prototype
    assert r.added and up.is_frozen(oid)
    assert any("drift" in rec.message for rec in caplog.records)
    clock.advance(60)
    r2 = up.maybe_add(oid, near(basis(0), 0.9), accept(oid), good_q(), 3)
    assert not r2.added and "frozen" in r2.reason


def test_check_drift_false_without_autos(store, clock, obj):
    assert mk(store, clock).check_drift(obj) is False


def test_gallery_worker_updates_off_the_calling_thread(store, clock, obj):
    import threading

    up = mk(store, clock)
    caller = threading.get_ident()
    worker_thread = []
    original = up.maybe_add

    def record_thread(*args, **kwargs):
        worker_thread.append(threading.get_ident())
        return original(*args, **kwargs)

    up.maybe_add = record_thread
    worker = GalleryUpdateWorker(up)
    try:
        assert worker.submit(obj, near(basis(0), 0.9), accept(obj), good_q(), 3)
        assert worker.wait_idle(2.0)
        assert worker_thread and worker_thread[0] != caller
        assert store.count(obj, "auto") == 1
    finally:
        worker.close()
