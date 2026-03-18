from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

pytest.importorskip("ultralytics")

from fbpipe.config import Settings
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
        predict_fn=lambda image, conf_thres: [object()],
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
        predict_fn=lambda image, conf_thres: [object()],
        eye_mgr=eye_mgr,
        cls8_tracker=cls8_tracker,
        pairer=pairer,
        active_max_flies=2,
    )

    assert row["cls8_0_track_id"] == 17
    assert row["dist_eye_0_cls8_0"] == pytest.approx(200.0)
    assert not np.isnan(row["angle_eye_0_cls8_vs_anchor"])
    assert pairer.eye_to_cls8[eye_mgr.anchor_ids[0]] == 17
