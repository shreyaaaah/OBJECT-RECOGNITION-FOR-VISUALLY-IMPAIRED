import numpy as np
import pytest

from fewshot_vi.gallery.matcher import Matcher
from fewshot_vi.gallery.store import GalleryError, GalleryStore
from fewshot_vi.types import Decision

from conftest import DIM, basis, near, unit


def add(store, name, vec, n=1, source="enrol"):
    oid = store.add_object(name)
    for _ in range(n):
        store.add_embedding(oid, vec, source)
    return oid


def make(store, **kw):
    return Matcher(store, accept_sim=0.70, ambiguous_sim=0.55, min_margin=0.08, knn_k=3, **kw)


def test_accept_when_high_score_and_margin(store):
    add(store, "keys", basis(0), 3)
    add(store, "wallet", basis(1), 3)
    m = make(store).match(near(basis(0), 0.9))
    assert m.decision == Decision.ACCEPT and m.name == "keys"
    assert m.margin >= 0.08


def test_unknown_when_below_ambiguous_band(store):
    add(store, "keys", basis(0), 3)
    add(store, "wallet", basis(1), 3)
    q = unit(basis(5))  # orthogonal to everything
    m = make(store).match(q)
    assert m.decision == Decision.UNKNOWN


def test_ambiguous_band(store):
    add(store, "keys", basis(0), 3)
    add(store, "wallet", basis(1), 3)
    m = make(store).match(near(basis(0), 0.62))
    assert m.decision == Decision.AMBIGUOUS


def test_small_margin_between_two_objects_is_not_accepted(store):
    add(store, "a", basis(0), 3)
    add(store, "b", basis(1), 3)
    q = unit(basis(0) + basis(1) + 0.05 * basis(2))  # ~equally similar to both (0.70 each)
    m = make(store).match(q)
    assert m.margin < 0.08
    assert m.decision != Decision.ACCEPT


def test_empty_gallery_is_unknown(store):
    m = make(store).match(basis(0))
    assert m.decision == Decision.UNKNOWN and m.name is None


def test_single_object_margin_rule_is_explicit(store):
    """With ONE object there is no runner-up: margin is +inf, so a high score is
    ACCEPT (never ambiguous) and a low score is still UNKNOWN (never accepted)."""
    add(store, "keys", basis(0), 3)
    hi = make(store).match(near(basis(0), 0.9))
    assert hi.decision == Decision.ACCEPT and np.isinf(hi.margin) and hi.runner_up_id is None
    mid = make(store).match(near(basis(0), 0.62))
    assert mid.decision == Decision.AMBIGUOUS
    lo = make(store).match(unit(basis(7)))
    assert lo.decision == Decision.UNKNOWN


def test_confusers_force_ambiguous_even_with_high_score(store):
    a = add(store, "mug_blue", basis(0), 3)
    b = add(store, "mug_blue2", near(basis(0), 0.99), 3)
    store.set_confusers("mug_blue", "mug_blue2")
    m = make(store).match(near(basis(0), 0.97))
    assert m.decision == Decision.AMBIGUOUS
    assert sorted(make(store).confuser_candidates(m)) == sorted([a, b])


def test_confuser_pair_not_flagged_when_runner_up_is_unrelated(store):
    add(store, "a", basis(0), 3)
    add(store, "b", basis(1), 3)
    add(store, "c", basis(2), 3)
    store.set_confusers("a", "c")  # b is NOT a confuser of a
    m = make(store).match(near(basis(0), 0.9))
    assert m.decision == Decision.ACCEPT


def test_score_is_better_of_prototype_and_topk(store):
    oid = store.add_object("x")
    store.add_embedding(oid, basis(0), "enrol")
    store.add_embedding(oid, basis(1), "enrol")
    store.add_embedding(oid, basis(2), "enrol")
    mt = make(store)
    # query equal to one stored vector: top-k mean = (1+0+0)/3, prototype sim = 1/sqrt(3)=0.577
    (oid_, s), = mt.scores(basis(0))
    assert s == pytest.approx(1 / np.sqrt(3), abs=1e-4)
    # with k=1 the top-k term wins
    mt1 = Matcher(store, knn_k=1)
    (_, s1), = mt1.scores(basis(0))
    assert s1 == pytest.approx(1.0, abs=1e-5)


def test_matrix_and_owner_stay_synchronised(store):
    a = add(store, "a", basis(0), 2)
    b = add(store, "b", basis(1), 3)
    assert store.E.shape == (5, DIM) and list(store.owner) == [a, a, b, b, b]
    assert store.delete_object("a")
    assert store.E.shape == (3, DIM) and set(store.owner) == {b}
    assert not store.delete_object("missing")


def test_prototype_is_normalised(store):
    oid = add(store, "a", basis(0), 1)
    store.add_embedding(oid, basis(1), "enrol")
    assert np.linalg.norm(store.prototype(oid)) == pytest.approx(1.0, abs=1e-5)
    ids, protos = store.prototypes()
    assert list(ids) == [oid] and protos.shape == (1, DIM)


def test_source_constraint_and_auto_has_no_patches(store):
    oid = store.add_object("a")
    with pytest.raises(GalleryError):
        store.add_embedding(oid, basis(0), "manual")
    with pytest.raises(GalleryError):
        store.add_embedding(oid, basis(0), "auto", patches=np.ones((4, DIM), np.float32))
    # the DB itself also enforces the constraint
    with pytest.raises(Exception):
        store._db.execute("INSERT INTO embeddings(object_id,cls_blob,source,created_at) VALUES(?,?,?,0)",
                          (oid, b"x", "bogus"))


def test_patches_stored_for_enrol_only_and_capped(store):
    oid = store.add_object("a")
    store.add_embedding(oid, basis(0), "enrol", patches=np.random.rand(100, DIM).astype(np.float32))
    store.add_embedding(oid, basis(1), "auto")
    views = store.patches_of(oid)
    assert len(views) == 1 and views[0].shape == (32, DIM)


def test_enrol_embeddings_are_protected_from_deletion(store):
    oid = store.add_object("a")
    eid = store.add_embedding(oid, basis(0), "enrol")
    with pytest.raises(GalleryError):
        store.delete_embedding(eid)
    assert store.count(oid) == 1


def test_persistence_roundtrip(tmp_path):
    p = tmp_path / "g.db"
    s = GalleryStore(p, dim=DIM)
    oid = s.add_object("keys")
    s.add_embedding(oid, basis(0), "enrol", patches=np.ones((3, DIM), np.float32))
    s.close()
    s2 = GalleryStore(p, dim=DIM)
    assert s2.names() == ["keys"] and s2.E.shape == (1, DIM) and len(s2.patches_of(oid)) == 1
    s2.close()


def test_duplicate_and_empty_names_rejected(store):
    store.add_object("keys")
    with pytest.raises(GalleryError):
        store.add_object("KEYS")
    with pytest.raises(GalleryError):
        store.add_object("  ")


def test_set_confusers_errors(store):
    store.add_object("a")
    with pytest.raises(GalleryError):
        store.set_confusers("a", "missing")
    with pytest.raises(GalleryError):
        store.set_confusers("a", "a")
