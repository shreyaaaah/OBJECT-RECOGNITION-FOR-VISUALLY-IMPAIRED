import cv2
import numpy as np
import pytest

from fewshot_vi.guidance.reviewhint import (HINT_BLUR, HINT_SMALL, HINT_UNSURE, ReviewHinter, choose_hint)
from fewshot_vi.quality import quality as q
from fewshot_vi.types import Decision, Quality

from conftest import FakeClock


def sharp(seed=0):
    return np.random.default_rng(seed).integers(0, 255, (120, 120, 3), dtype=np.uint8)


def blurred():
    return cv2.GaussianBlur(sharp(), (31, 31), 0)


def test_blur_metric_separates_sharp_from_blurred():
    assert q.laplacian_blur(sharp()) > 1000
    assert q.laplacian_blur(blurred()) < 60
    assert q.laplacian_blur(np.zeros((0, 0, 3), np.uint8)) == 0.0


def test_edge_cuts_each_side_and_none():
    assert q.edge_cuts((100, 100, 200, 200), 640, 480, 8) == frozenset()
    assert q.edge_cuts((0, 100, 200, 200), 640, 480, 8) == {"left"}
    assert q.edge_cuts((100, 100, 640, 200), 640, 480, 8) == {"right"}
    assert q.edge_cuts((100, 3, 200, 200), 640, 480, 8) == {"top"}
    assert q.edge_cuts((100, 100, 200, 478), 640, 480, 8) == {"bottom"}
    assert q.edge_cuts((0, 0, 640, 480), 640, 480, 8) == {"left", "right", "top", "bottom"}


def assess(crop, box, margin=None, frame=(480, 640, 3)):
    return q.assess(crop, box, frame, blur_var_min=60.0, edge_margin_px=8, min_box_area_frac=0.02,
                    margin=margin, min_margin=0.08)


def test_assess_good_view():
    r = assess(sharp(), (200, 150, 330, 280), margin=0.3)
    assert r.ok and not r.blurry and not r.edge_cut and not r.too_small and not r.low_margin
    assert r.score == pytest.approx(1.0)


def test_assess_flags_each_problem():
    assert assess(blurred(), (200, 150, 330, 280)).blurry
    assert assess(sharp(), (0, 150, 130, 280)).edge_cut == {"left"}
    small = assess(sharp(), (200, 150, 220, 170))  # 400 px of 307200 = 0.13%
    assert small.too_small and not small.ok
    low = assess(sharp(), (200, 150, 330, 280), margin=0.02)
    assert low.low_margin and not low.ok


def test_infinite_or_missing_margin_is_not_low_margin():
    assert not assess(sharp(), (200, 150, 330, 280), margin=float("inf")).low_margin
    assert not assess(sharp(), (200, 150, 330, 280), margin=None).low_margin


def mkq(**kw):
    base = dict(blur=100.0, blurry=False, edge_cut=frozenset(), too_small=False, low_margin=False, score=1.0)
    base.update(kw)
    return Quality(**base)


def test_choose_hint_priority_and_text():
    assert choose_hint(mkq(edge_cut=frozenset({"left"})), None) == "Move the camera left."
    assert choose_hint(mkq(edge_cut=frozenset({"right"})), None) == "Move the camera right."
    assert choose_hint(mkq(edge_cut=frozenset({"top"})), None) == "Tilt the camera up."
    assert choose_hint(mkq(edge_cut=frozenset({"bottom"})), None) == "Tilt the camera down."
    # edge cut outranks blur; blur outranks small; small outranks unsure
    assert choose_hint(mkq(edge_cut=frozenset({"left"}), blurry=True), None) == "Move the camera left."
    assert choose_hint(mkq(blurry=True, too_small=True), None) == HINT_BLUR
    assert choose_hint(mkq(too_small=True, low_margin=True), None) == HINT_SMALL
    assert choose_hint(mkq(low_margin=True), None) == HINT_UNSURE
    assert choose_hint(mkq(), Decision.AMBIGUOUS) == HINT_UNSURE
    assert choose_hint(mkq(), Decision.ACCEPT) is None and choose_hint(None, None) is None


def test_hints_return_at_most_one_text():
    assert isinstance(choose_hint(mkq(blurry=True, too_small=True, low_margin=True,
                                      edge_cut=frozenset({"left", "top"})), None), str)


def test_hinter_cooldown_and_no_consecutive_repeat():
    clk = FakeClock()
    h = ReviewHinter(cooldown_s=4.0, clock=clk)
    blur = mkq(blurry=True)
    assert h.next_hint(blur, None) == HINT_BLUR
    clk.advance(1)
    assert h.next_hint(mkq(too_small=True), None) is None       # inside cooldown
    clk.advance(4)
    assert h.next_hint(blur, None) is None                      # same hint twice in a row is blocked
    assert h.next_hint(mkq(too_small=True), None) == HINT_SMALL  # a different hint is allowed
    clk.advance(5)
    assert h.next_hint(mkq(too_small=True), None) is None
    h.note_success()
    assert h.next_hint(mkq(too_small=True), None) == HINT_SMALL  # a confirmed success resets the repeat guard
