import math

import pytest

from fewshot_vi.guidance import geometry as g


def test_focal_and_distance_known_numbers():
    # hfov 90deg -> tan(45deg)=1 -> focal = W/2 = 320 px
    assert g.focal_px(640, 90) == pytest.approx(320.0)
    # 8 cm object, 64 px wide: 0.08 * 320 / 64 = 0.4 m
    assert g.distance_m(64, 640, 90, 8) == pytest.approx(0.4)
    # twice as wide on screen -> half the distance
    assert g.distance_m(128, 640, 90, 8) == pytest.approx(0.2)
    # default camera: hfov 65deg -> focal = 320 / tan(32.5deg)
    assert g.focal_px(640, 65) == pytest.approx(320 / math.tan(math.radians(32.5)), rel=1e-9)
    assert g.distance_m(100, 640, 65, 8) == pytest.approx(0.08 * 502.3 / 100, rel=2e-3)


def test_zero_width_gives_infinite_distance():
    assert math.isinf(g.distance_m(0, 640, 65, 8))


@pytest.mark.parametrize("cx,expected_h", [
    (320, "centre"),
    (320 + 0.05 * 640, "centre"),            # 5% off -> centre
    (320 - 0.15 * 640, "slightly left"),     # 15% off
    (320 + 0.15 * 640, "slightly right"),
    (320 - 0.30 * 640, "left"),
    (320 + 0.30 * 640, "right"),
])
def test_horizontal_zones(cx, expected_h):
    box = (cx - 10, 230, cx + 10, 250)
    d = g.direction(box, 640, 480, 0.10, 0.20)
    assert d.horizontal == expected_h and d.vertical == "centre"


@pytest.mark.parametrize("cy,expected_v", [
    (240, "centre"), (240 - 0.15 * 480, "slightly up"), (240 + 0.15 * 480, "slightly down"),
    (240 - 0.35 * 480, "up"), (240 + 0.35 * 480, "down"),
])
def test_vertical_zones(cy, expected_v):
    d = g.direction((310, cy - 10, 330, cy + 10), 640, 480, 0.10, 0.20)
    assert d.vertical == expected_v


def test_pan_range_and_sign():
    left = g.direction((0, 200, 40, 240), 640, 480, 0.1, 0.2)
    right = g.direction((600, 200, 640, 240), 640, 480, 0.1, 0.2)
    mid = g.direction((300, 200, 340, 240), 640, 480, 0.1, 0.2)
    assert -1.0 <= left.pan < -0.8 and 0.8 < right.pan <= 1.0 and mid.pan == pytest.approx(0.0)


@pytest.mark.parametrize("d,phrase", [
    (0.1, "very close"), (0.5, "about half a metre"), (0.6, "about half a metre"),
    (1.0, "about one metre"), (1.4, "about one and a half metres"),
    (2.0, "about 2 metres"), (3.2, "about 3 metres"), (math.inf, "distance unknown"),
])
def test_distance_phrases(d, phrase):
    assert g.distance_phrase(d) == phrase
