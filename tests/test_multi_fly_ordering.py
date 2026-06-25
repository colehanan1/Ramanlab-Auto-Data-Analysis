"""Eye-slot ordering for EyeAnchorManager.

Flies are stacked vertically in the rig, so eye slots must be numbered
top-to-bottom (slot 0 = topmost). Ordering is by y-center, with x-center as a
deterministic tie-breaker when two eyes share a row.
"""

import numpy as np

from fbpipe.utils.multi_fly import EyeAnchorManager


def test_eyes_ordered_top_to_bottom():
    # Three vertically stacked eyes at distinct y; x differs so a y-vs-x sort
    # would disagree. Confidence order is scrambled relative to position.
    eye_boxes = np.array(
        [
            [300.0, 700.0, 360.0, 760.0],  # bottom (y~730, x~330)
            [200.0, 100.0, 260.0, 160.0],  # top    (y~130, x~230)
            [340.0, 400.0, 400.0, 460.0],  # middle (y~430, x~370)
        ],
        dtype=np.float32,
    )
    eye_scores = np.array([0.99, 0.90, 0.80], dtype=np.float32)

    mgr = EyeAnchorManager(max_eyes=3)
    mgr.try_update_from_dets(eye_boxes, eye_scores)

    assert mgr.confirmed
    ys = [c[1] for c in mgr.get_centers()]
    assert ys == sorted(ys)  # slot order increases top-to-bottom
    assert ys[0] == min(ys)  # slot 0 is the topmost eye


def test_equal_y_breaks_ties_left_to_right():
    # Same row (equal y) -> deterministic left-to-right fallback.
    eye_boxes = np.array(
        [
            [500.0, 100.0, 560.0, 160.0],  # right
            [100.0, 100.0, 160.0, 160.0],  # left
        ],
        dtype=np.float32,
    )
    eye_scores = np.array([0.95, 0.90], dtype=np.float32)

    mgr = EyeAnchorManager(max_eyes=2)
    mgr.try_update_from_dets(eye_boxes, eye_scores)

    xs = [c[0] for c in mgr.get_centers()]
    assert xs == sorted(xs)  # left-to-right when y ties
