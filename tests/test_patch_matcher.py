import time

import numpy as np
import pytest

from fewshot_vi.gallery import patch_matcher as pm
from fewshot_vi.gallery.patch_matcher import (PatchReranker, RerankResult, apply_rerank, rerank,
                                              symmetric_patch_score)
from fewshot_vi.types import Decision, Match

D = 16


def tokens(seed, n=12):
    return np.random.default_rng(seed).normal(size=(n, D)).astype(np.float32)


def noisy(t, eps=0.05, seed=99):
    return t + eps * np.random.default_rng(seed).normal(size=t.shape).astype(np.float32)


def test_matching_views_score_above_unrelated():
    a = tokens(1)
    assert symmetric_patch_score(noisy(a), a) > symmetric_patch_score(a, tokens(2)) + 0.3


def test_score_is_symmetric_and_bounded():
    a, b = tokens(3), tokens(4, n=7)
    assert symmetric_patch_score(a, b) == pytest.approx(symmetric_patch_score(b, a), abs=1e-6)
    assert -1.0 <= symmetric_patch_score(a, b) <= 1.0
    assert symmetric_patch_score(a, a) == pytest.approx(1.0, abs=1e-5)


def test_asymmetric_token_counts_use_both_directions():
    a = tokens(5, n=4)
    b = np.vstack([a, tokens(6, n=20)])  # b contains a plus unrelated tokens
    # row-max of a->b is 1.0 but column-max of b->a is low for the extra tokens => avg < 1
    assert symmetric_patch_score(a, b) < 0.9


def test_empty_inputs():
    assert symmetric_patch_score(np.zeros((0, D)), tokens(1)) == -1.0
    assert pm.score_candidate(tokens(1), []) == -1.0


def test_candidate_score_is_best_stored_view():
    q = tokens(7)
    views = [tokens(8), noisy(q), tokens(9)]
    assert pm.score_candidate(q, views) == pytest.approx(symmetric_patch_score(q, views[1]), abs=1e-6)


def test_rerank_orders_candidates():
    q = tokens(10)
    ranking = rerank(q, {1: [tokens(11)], 2: [noisy(q)]})
    assert [oid for oid, _ in ranking] == [2, 1]


def _prov(top=1, runner=2, s1=0.90, s2=0.89):
    return Match(top, f"obj{top}", s1, s1 - s2, Decision.AMBIGUOUS, runner, [(top, s1), (runner, s2)])


def test_confuser_rerank_can_change_the_result():
    q = tokens(20)
    cands = {1: [tokens(21)], 2: [noisy(q)]}  # object 2 really matches the query's patches
    prov = _prov(top=1, runner=2)  # global embedding said object 1
    res = RerankResult(7, rerank(q, cands), 0.01, True)
    out = apply_rerank(prov, res, lambda i: f"obj{i}", accept_patch_sim=0.8, min_gain=0.05)
    assert prov.object_id == 1 and out.object_id == 2 and out.decision == Decision.ACCEPT
    assert out.name == "obj2"


def test_rerank_confirming_provisional_accepts_it():
    q = tokens(22)
    cands = {1: [noisy(q)], 2: [tokens(23)]}
    out = apply_rerank(_prov(1, 2), RerankResult(1, rerank(q, cands), 0.01, True),
                       lambda i: f"obj{i}", accept_patch_sim=0.8)
    assert out.object_id == 1 and out.decision == Decision.ACCEPT


def test_weak_patch_evidence_keeps_provisional_result():
    q = tokens(24)
    cands = {1: [tokens(25)], 2: [tokens(26)]}  # nothing really matches
    prov = _prov()
    out = apply_rerank(prov, RerankResult(1, rerank(q, cands), 0.01, True),
                       lambda i: f"obj{i}", accept_patch_sim=0.8)
    assert out is prov and out.decision == Decision.AMBIGUOUS


def test_missed_deadline_preserves_provisional():
    q = tokens(27)
    cands = {1: [tokens(28)], 2: [noisy(q)]}
    prov = _prov()
    out = apply_rerank(prov, RerankResult(1, rerank(q, cands), 0.5, False), lambda i: "x", 0.8)
    assert out is prov


def test_worker_is_asynchronous_and_flags_missed_deadline(monkeypatch):
    def slow(query, cands):
        time.sleep(0.15)
        return [(next(iter(cands)), 0.9)]
    monkeypatch.setattr(pm, "rerank", slow)
    rr = PatchReranker(deadline_s=0.03)
    try:
        t0 = time.perf_counter()
        assert rr.submit(5, tokens(1), {1: [tokens(2)]})
        assert time.perf_counter() - t0 < 0.05  # submit never blocks on the work
        assert rr.poll() == []  # nothing yet
        deadline = time.time() + 2
        got = []
        while not got and time.time() < deadline:
            got = rr.poll()
            time.sleep(0.01)
        assert got and got[0].track_id == 5 and got[0].met_deadline is False
    finally:
        rr.close()


def test_worker_fast_job_meets_deadline_and_survives_errors(monkeypatch):
    rr = PatchReranker(deadline_s=1.0)
    try:
        rr.submit(1, tokens(1), {1: [tokens(2)], 2: [tokens(3)]})
        rr.submit(2, "not-an-array", {1: [tokens(2)]})  # raises inside the worker
        rr.submit(3, tokens(1), {1: [tokens(2)]})
        deadline = time.time() + 2
        got = []
        while len(got) < 2 and time.time() < deadline:
            got += rr.poll()
            time.sleep(0.01)
        assert {r.track_id for r in got} == {1, 3}
        assert all(r.met_deadline for r in got)
    finally:
        rr.close()


def test_full_queue_drops_instead_of_blocking(monkeypatch):
    monkeypatch.setattr(pm, "rerank", lambda q, c: (time.sleep(0.3), [])[1])
    rr = PatchReranker(deadline_s=0.03, max_pending=1)
    try:
        results = [rr.submit(i, tokens(1), {1: [tokens(2)]}) for i in range(6)]
        assert not all(results)
    finally:
        rr.close()
