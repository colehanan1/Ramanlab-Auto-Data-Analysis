from __future__ import annotations

import math

import numpy as np
import pytest
from types import SimpleNamespace

pytest.importorskip("ultralytics")

from fbpipe.config import ProboscisFilterSettings, Settings
from fbpipe.steps import yolo_infer
from fbpipe.steps.yolo_infer import (
    _build_proboscis_tracker,
    _limit_proboscis_detections,
)
from fbpipe.utils.multi_fly import EyeAnchorManager, StablePairing


def test_limit_proboscis_detections_matches_detected_eye_slots() -> None:
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [10.0, 0.0, 20.0, 10.0],
            [20.0, 0.0, 30.0, 10.0],
            [30.0, 0.0, 40.0, 10.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.4, 0.95, 0.8, 0.7], dtype=np.float32)

    limited_boxes, limited_scores = _limit_proboscis_detections(boxes, scores, active_max_flies=2)

    assert limited_boxes.shape == (2, 4)
    np.testing.assert_allclose(limited_scores, np.array([0.95, 0.8], dtype=np.float32))
    np.testing.assert_allclose(limited_boxes, boxes[[1, 2]])


def test_build_proboscis_tracker_ignores_legacy_proboscis_cap() -> None:
    settings = Settings(
        model_path="model.pt",
        main_directories="/tmp/videos",
        max_flies=4,
        max_proboscis_tracks=3,
    )

    tracker = _build_proboscis_tracker(settings, active_max_flies=4)

    assert tracker.max_tracks == 4


class _DummyProboscisTracker:
    def __init__(self, tracks):
        self._tracks = tracks

    def step(self, boxes, scores):
        return self._tracks


def test_process_frame_invalidates_far_proboscis_match_for_three_fly_video(monkeypatch) -> None:
    eye_boxes = np.array(
        [
            [95.0, 95.0, 105.0, 105.0],
            [495.0, 95.0, 505.0, 105.0],
            [895.0, 95.0, 905.0, 105.0],
        ],
        dtype=np.float32,
    )
    eye_scores = np.array([0.95, 0.9, 0.85], dtype=np.float32)
    prob_box = np.array([[295.0, 95.0, 305.0, 105.0]], dtype=np.float32)
    prob_scores = np.array([0.98], dtype=np.float32)

    monkeypatch.setattr(
        yolo_infer,
        "collect_detections",
        lambda result, classes: {
            0: {"boxes": eye_boxes, "scores": eye_scores},
            1: {"boxes": prob_box, "scores": prob_scores},
        },
    )

    eye_mgr = EyeAnchorManager(max_eyes=3)
    eye_mgr.try_update_from_dets(eye_boxes, eye_scores)
    pairer = StablePairing(max_pairs=3, rebind_max_dist_px=300.0)
    cls8_tracker = _DummyProboscisTracker(
        [SimpleNamespace(id=17, box_xyxy=prob_box[0], time_since_update=0)]
    )
    settings = Settings(
        model_path="model.pt",
        main_directories="/tmp/videos",
        max_flies=3,
        class2_max=250.0,
        three_fly_max_eye_prob_distance_px=180.0,
        use_optical_flow=False,
    )

    _, row, _ = yolo_infer._process_frame(
        frame=np.zeros((160, 1000, 3), dtype=np.uint8),
        frame_number=0,
        current_timestamp=0.0,
        single_trackers={},
        prev_gray=None,
        anchor=(0.0, 0.0),
        settings=settings,
        result=object(),
        eye_mgr=eye_mgr,
        cls8_tracker=cls8_tracker,
        pairer=pairer,
        active_max_flies=3,
    )

    assert np.isnan(row["cls8_0_track_id"])
    assert np.isnan(row["cls8_0_x"])
    assert np.isnan(row["cls8_0_y"])
    assert np.isnan(row["dist_eye_0_cls8_0"])
    assert np.isnan(row["angle_eye_0_cls8_vs_anchor"])
    assert pairer.eye_to_cls8[eye_mgr.anchor_ids[0]] is None


def test_process_frame_keeps_far_proboscis_match_when_video_is_not_three_fly(monkeypatch) -> None:
    eye_boxes = np.array(
        [
            [95.0, 95.0, 105.0, 105.0],
            [495.0, 95.0, 505.0, 105.0],
        ],
        dtype=np.float32,
    )
    eye_scores = np.array([0.95, 0.9], dtype=np.float32)
    prob_box = np.array([[295.0, 95.0, 305.0, 105.0]], dtype=np.float32)
    prob_scores = np.array([0.98], dtype=np.float32)

    monkeypatch.setattr(
        yolo_infer,
        "collect_detections",
        lambda result, classes: {
            0: {"boxes": eye_boxes, "scores": eye_scores},
            1: {"boxes": prob_box, "scores": prob_scores},
        },
    )

    eye_mgr = EyeAnchorManager(max_eyes=2)
    eye_mgr.try_update_from_dets(eye_boxes, eye_scores)
    pairer = StablePairing(max_pairs=2, rebind_max_dist_px=300.0)
    cls8_tracker = _DummyProboscisTracker(
        [SimpleNamespace(id=17, box_xyxy=prob_box[0], time_since_update=0)]
    )
    settings = Settings(
        model_path="model.pt",
        main_directories="/tmp/videos",
        max_flies=2,
        class2_max=250.0,
        three_fly_max_eye_prob_distance_px=180.0,
        # Wide enough that the 200 px side reach passes the anisotropic gate:
        # this test is about the >=3-fly circular cap, not the geometry gate.
        proboscis_filter=ProboscisFilterSettings(max_eye_prob_distance_px=250.0),
        use_optical_flow=False,
    )

    _, row, _ = yolo_infer._process_frame(
        frame=np.zeros((160, 600, 3), dtype=np.uint8),
        frame_number=0,
        current_timestamp=0.0,
        single_trackers={},
        prev_gray=None,
        anchor=(0.0, 0.0),
        settings=settings,
        result=object(),
        eye_mgr=eye_mgr,
        cls8_tracker=cls8_tracker,
        pairer=pairer,
        active_max_flies=2,
    )

    assert row["cls8_0_track_id"] == 17
    assert row["dist_eye_0_cls8_0"] == pytest.approx(200.0)
    assert not np.isnan(row["angle_eye_0_cls8_vs_anchor"])
    assert pairer.eye_to_cls8[eye_mgr.anchor_ids[0]] == 17


def _stacked_four_fly_scene(monkeypatch, prob_center):
    """Four flies stacked 350 px apart (Flybehavior2 framing), one proboscis track."""
    eye_boxes = np.array(
        [[495.0, y - 5.0, 505.0, y + 5.0] for y in (100.0, 450.0, 800.0, 1150.0)],
        dtype=np.float32,
    )
    eye_scores = np.array([0.95, 0.9, 0.85, 0.8], dtype=np.float32)
    px, py = prob_center
    prob_box = np.array([[px - 5.0, py - 5.0, px + 5.0, py + 5.0]], dtype=np.float32)
    monkeypatch.setattr(
        yolo_infer,
        "collect_detections",
        lambda result, classes: {
            0: {"boxes": eye_boxes, "scores": eye_scores},
            1: {"boxes": prob_box, "scores": np.array([0.98], dtype=np.float32)},
        },
    )
    eye_mgr = EyeAnchorManager(max_eyes=4)
    eye_mgr.try_update_from_dets(eye_boxes, eye_scores)
    cls8_tracker = _DummyProboscisTracker(
        [SimpleNamespace(id=88, box_xyxy=prob_box[0], time_since_update=0)]
    )
    # The two gate keys deliberately differ: the circular >=3-fly cap (300) is
    # loose enough to pass the mispairing; only the anisotropic gate (280 px
    # left/right/down, 70 px up) can catch it.
    settings = Settings(
        model_path="model.pt",
        main_directories="/tmp/videos",
        max_flies=4,
        three_fly_max_eye_prob_distance_px=300.0,
        proboscis_filter=ProboscisFilterSettings(max_eye_prob_distance_px=280.0, up_divisor=4.0),
        use_optical_flow=False,
    )
    return eye_mgr, cls8_tracker, settings


def _run_frame(eye_mgr, cls8_tracker, settings, pairer):
    _, row, _ = yolo_infer._process_frame(
        frame=np.zeros((1300, 1000, 3), dtype=np.uint8),
        frame_number=0,
        current_timestamp=0.0,
        single_trackers={},
        prev_gray=None,
        anchor=(0.0, 600.0),
        settings=settings,
        result=object(),
        eye_mgr=eye_mgr,
        cls8_tracker=cls8_tracker,
        pairer=pairer,
        active_max_flies=4,
    )
    return row


def test_process_frame_hands_proboscis_from_eye_outside_gate_to_its_owner(monkeypatch) -> None:
    # october_07_batch_1_pretest_1 @ 0:45: eye3 holds track 88, which sits 78 px
    # BELOW eye2 (inside eye2's gate) and 272 px ABOVE eye3 (outside eye3's 70 px
    # upward gate, but inside the 300 px circular cap).
    eye_mgr, cls8_tracker, settings = _stacked_four_fly_scene(monkeypatch, (494.0, 878.0))
    eye_ids = eye_mgr.anchor_ids
    pairer = StablePairing(max_pairs=4)
    pairer.eye_to_cls8 = {eye_ids[0]: None, eye_ids[1]: None, eye_ids[2]: None, eye_ids[3]: 88}

    row = _run_frame(eye_mgr, cls8_tracker, settings, pairer)

    assert row["cls8_2_track_id"] == 88
    assert row["dist_eye_2_cls8_2"] == pytest.approx(np.hypot(6.0, 78.0))
    assert np.isnan(row["cls8_3_track_id"])
    assert np.isnan(row["cls8_3_x"])
    assert pairer.eye_to_cls8[eye_ids[3]] is None
    assert pairer.eye_to_cls8[eye_ids[2]] == 88


def test_process_frame_never_binds_proboscis_outside_every_gate(monkeypatch) -> None:
    # 290 px right of eye3 and eye2: past the 280 px side reach of every gate,
    # yet 294 px from eye3 -- inside the 300 px circular cap.
    eye_mgr, cls8_tracker, settings = _stacked_four_fly_scene(monkeypatch, (790.0, 1100.0))
    pairer = StablePairing(max_pairs=4, rebind_max_dist_px=400.0)  # only the gate may refuse it

    row = _run_frame(eye_mgr, cls8_tracker, settings, pairer)

    for idx in range(4):
        assert np.isnan(row[f"cls8_{idx}_track_id"])
    assert all(v is None for v in pairer.eye_to_cls8.values())


def test_process_frame_pairs_close_flies_by_gate_depth_not_visit_order(monkeypatch) -> None:
    # IAA-Sensitivity-24-1/october_07_batch_2_pretest_5 @ 0:35 (Flybehavior2): eyes 1
    # and 2 are 178 px apart. Eye 1 held track 18, which hangs 55 px under eye 2,
    # while eye 1's own proboscis (track 19, 73 px under it) went unpaired and
    # eye 2 got nothing -- in 2,395 of 3,598 frames.
    eyes = [(754.0, 116.0), (688.0, 423.0), (675.0, 601.0), (625.0, 875.0)]
    eye_boxes = np.array([[x - 5, y - 5, x + 5, y + 5] for x, y in eyes], dtype=np.float32)
    eye_scores = np.array([0.95, 0.9, 0.85, 0.8], dtype=np.float32)
    probs = {10: (753.0, 170.0), 18: (699.0, 656.0), 19: (649.0, 496.0), 8: (615.0, 953.0)}
    prob_boxes = {k: np.array([x - 5, y - 5, x + 5, y + 5], dtype=np.float32) for k, (x, y) in probs.items()}
    monkeypatch.setattr(
        yolo_infer,
        "collect_detections",
        lambda result, classes: {
            0: {"boxes": eye_boxes, "scores": eye_scores},
            1: {"boxes": np.stack(list(prob_boxes.values())), "scores": np.full(4, 0.9, np.float32)},
        },
    )
    eye_mgr = EyeAnchorManager(max_eyes=4)
    eye_mgr.try_update_from_dets(eye_boxes, eye_scores)
    ids = eye_mgr.anchor_ids
    pairer = StablePairing(max_pairs=4, rebind_max_dist_px=305.0)
    pairer.eye_to_cls8 = {ids[0]: 10, ids[1]: 18, ids[2]: None, ids[3]: 8}
    cls8_tracker = _DummyProboscisTracker(
        [SimpleNamespace(id=k, box_xyxy=b, time_since_update=0) for k, b in prob_boxes.items()]
    )
    settings = Settings(
        model_path="model.pt",
        main_directories="/tmp/videos",
        max_flies=4,
        three_fly_max_eye_prob_distance_px=300.0,
        proboscis_filter=ProboscisFilterSettings(max_eye_prob_distance_px=280.0, up_divisor=4.0),
        use_optical_flow=False,
    )

    _, row, _ = yolo_infer._process_frame(
        frame=np.zeros((1080, 1080, 3), dtype=np.uint8),
        frame_number=0,
        current_timestamp=0.0,
        single_trackers={},
        prev_gray=None,
        anchor=(0.0, 540.0),
        settings=settings,
        result=object(),
        eye_mgr=eye_mgr,
        cls8_tracker=cls8_tracker,
        pairer=pairer,
        active_max_flies=4,
    )

    assert [row[f"cls8_{i}_track_id"] for i in range(4)] == [10, 19, 18, 8]
    assert row["dist_eye_2_cls8_2"] == pytest.approx(np.hypot(24.0, 55.0))


# --- the pairing cost itself: depth in the anisotropic gate ------------------
# 280 px down and to the sides, 280 / up_divisor = 70 px up. The >=3-fly cap
# (300 px here) is a separate key and deliberately differs.


def _gate_settings(three_fly_cap: float = 300.0) -> Settings:
    return Settings(
        model_path="model.pt",
        main_directories="/tmp/videos",
        max_flies=4,
        three_fly_max_eye_prob_distance_px=three_fly_cap,
        proboscis_filter=ProboscisFilterSettings(max_eye_prob_distance_px=280.0, up_divisor=4.0),
        use_optical_flow=False,
    )


@pytest.mark.parametrize(
    "offset, depth",
    [
        ((0.0, 0.0), 0.0),
        ((0.0, 280.0), 1.0),     # the down edge
        ((280.0, 0.0), 1.0),     # the side edges
        ((-280.0, 0.0), 1.0),
        ((0.0, -70.0), 1.0),     # the up edge: max_px / up_divisor
        ((0.0, 140.0), 0.5),     # half-way down ...
        ((0.0, -35.0), 0.5),     # ... costs what 35 px up does
    ],
)
def test_gate_depth_is_0_at_the_eye_and_1_at_the_anisotropic_edge(offset, depth) -> None:
    eye = (500.0, 500.0)
    prob = (eye[0] + offset[0], eye[1] + offset[1])
    assert yolo_infer._eye_prob_gate_depth(eye, prob, _gate_settings(), 4) == pytest.approx(depth)


@pytest.mark.parametrize("offset", [(0.0, 281.0), (281.0, 0.0), (-281.0, 0.0), (0.0, -71.0)])
def test_gate_depth_is_infinite_just_outside_the_gate(offset) -> None:
    eye = (500.0, 500.0)
    prob = (eye[0] + offset[0], eye[1] + offset[1])
    assert math.isinf(yolo_infer._eye_prob_gate_depth(eye, prob, _gate_settings(), 4))


def test_gate_depth_is_infinite_beyond_a_tighter_three_fly_cap() -> None:
    eye, prob = (500.0, 500.0), (500.0, 700.0)  # 200 px down: inside the 280 px gate
    assert math.isinf(yolo_infer._eye_prob_gate_depth(eye, prob, _gate_settings(three_fly_cap=180.0), 4))
    assert yolo_infer._eye_prob_gate_depth(eye, prob, _gate_settings(three_fly_cap=180.0), 2) == pytest.approx(200 / 280)


def test_process_frame_keeps_a_mid_per_above_the_lower_eye_with_the_upper_fly(monkeypatch) -> None:
    # The pretest_5 close mount. Track 21 hangs 138 px under eye 1 but sits
    # 40 px ABOVE eye 2 (fly 1 mid-PER), and fly 2's own proboscis is undetected.
    # Raw distance would hand it to eye 2 (40 px vs 138 px); in gate depth the
    # tight upward reach makes 40 px up (0.57) deeper than 138 px down (0.49).
    eyes = [(754.0, 116.0), (688.0, 423.0), (675.0, 601.0), (625.0, 875.0)]
    eye_boxes = np.array([[x - 5, y - 5, x + 5, y + 5] for x, y in eyes], dtype=np.float32)
    eye_scores = np.array([0.95, 0.9, 0.85, 0.8], dtype=np.float32)
    probs = {10: (753.0, 170.0), 21: (680.0, 561.0), 8: (615.0, 953.0)}
    prob_boxes = {k: np.array([x - 5, y - 5, x + 5, y + 5], dtype=np.float32) for k, (x, y) in probs.items()}
    monkeypatch.setattr(
        yolo_infer,
        "collect_detections",
        lambda result, classes: {
            0: {"boxes": eye_boxes, "scores": eye_scores},
            1: {"boxes": np.stack(list(prob_boxes.values())), "scores": np.full(3, 0.9, np.float32)},
        },
    )
    eye_mgr = EyeAnchorManager(max_eyes=4)
    eye_mgr.try_update_from_dets(eye_boxes, eye_scores)
    pairer = StablePairing(max_pairs=4, rebind_max_dist_px=305.5)
    cls8_tracker = _DummyProboscisTracker(
        [SimpleNamespace(id=k, box_xyxy=b, time_since_update=0) for k, b in prob_boxes.items()]
    )

    _, row, _ = yolo_infer._process_frame(
        frame=np.zeros((1080, 1080, 3), dtype=np.uint8),
        frame_number=0,
        current_timestamp=0.0,
        single_trackers={},
        prev_gray=None,
        anchor=(0.0, 540.0),
        settings=_gate_settings(),
        result=object(),
        eye_mgr=eye_mgr,
        cls8_tracker=cls8_tracker,
        pairer=pairer,
        active_max_flies=4,
    )

    assert row["cls8_1_track_id"] == 21
    assert np.isnan(row["cls8_2_track_id"])
