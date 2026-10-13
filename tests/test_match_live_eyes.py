"""match_live_eyes: find each frozen eye anchor's eye detection in a frame.

QC only -- the anchors never move. Each anchor searches ``max_radius_px``,
clamped to 0.45x the distance to its nearest other anchor, and pairs are
assigned globally, nearest first, one-to-one.
"""

import numpy as np

from fbpipe.utils.multi_fly import match_live_eyes


def _boxes(*centers, half=5.0):
    return np.array(
        [[cx - half, cy - half, cx + half, cy + half] for cx, cy in centers],
        dtype=np.float32,
    ).reshape(-1, 4)


def test_picks_the_nearest_detection_within_the_radius():
    live = match_live_eyes([(100.0, 100.0)], _boxes((130.0, 100.0), (103.0, 104.0)), 50.0)
    assert live == [(103.0, 104.0)]
    assert all(type(v) is float for v in live[0])


def test_radius_is_inclusive_and_nothing_beyond_it_matches():
    assert match_live_eyes([(100.0, 100.0)], _boxes((100.0, 150.0)), 50.0) == [(100.0, 150.0)]
    assert match_live_eyes([(100.0, 100.0)], _boxes((100.0, 151.0)), 50.0) == [None]


def test_none_when_no_detection_is_in_range_of_any_anchor():
    anchors = [(500.0, 100.0), (500.0, 450.0)]
    assert match_live_eyes(anchors, _boxes((900.0, 900.0), (100.0, 300.0)), 50.0) == [None, None]


def test_each_anchor_gets_its_own_eye_whatever_the_detection_order():
    # Four flies stacked 350 px apart (Flybehavior2 framing), detections reversed
    # and each a little off its anchor.
    anchors = [(500.0, y) for y in (100.0, 450.0, 800.0, 1150.0)]
    dets = _boxes((502.0, 1149.0), (499.0, 803.0), (500.0, 452.0), (501.0, 100.0))
    assert match_live_eyes(anchors, dets, 50.0) == [
        (501.0, 100.0),
        (500.0, 452.0),
        (499.0, 803.0),
        (502.0, 1149.0),
    ]


def test_a_detection_is_given_to_one_anchor_only():
    # Degenerate but legal input: two anchors on the same spot clamp each other's
    # radius to 0, so a detection exactly there is in range of both.
    live = match_live_eyes([(10.0, 10.0), (10.0, 10.0)], _boxes((10.0, 10.0)), 50.0)
    assert live == [(10.0, 10.0), None]


def test_neighbours_eye_is_never_stolen_when_a_slots_own_eye_is_missing():
    # Fly B's eye is missing; fly A's eye is detected twice, once 40 px off
    # towards B. With a generous configured radius the stray box would land on B
    # (60 px away) and read as a 60 px drift; the spacing clamp (0.45 * 100 =
    # 45 px) keeps B unmatched instead.
    anchors = [(100.0, 100.0), (100.0, 200.0)]
    live = match_live_eyes(anchors, _boxes((100.0, 100.0), (100.0, 140.0)), 500.0)
    assert live == [(100.0, 100.0), None]


def test_radius_is_clamped_by_the_spacing_to_the_nearest_other_anchor():
    det = _boxes((100.0, 72.0))  # 28 px above A, 88 px from B
    # Alone, A searches the full 50 px and finds it ...
    assert match_live_eyes([(100.0, 100.0)], det, 50.0) == [(100.0, 72.0)]
    # ... but with B 60 px below, A's radius is 0.45 * 60 = 27 px.
    assert match_live_eyes([(100.0, 100.0), (100.0, 160.0)], det, 50.0) == [None, None]


def test_empty_inputs():
    assert match_live_eyes([], _boxes((10.0, 10.0)), 50.0) == []
    assert match_live_eyes([], np.zeros((0, 4), np.float32), 50.0) == []
    anchors = [(500.0, 100.0), (500.0, 450.0)]
    assert match_live_eyes(anchors, np.zeros((0, 4), np.float32), 50.0) == [None, None]
    assert match_live_eyes(anchors, [], 50.0) == [None, None]
