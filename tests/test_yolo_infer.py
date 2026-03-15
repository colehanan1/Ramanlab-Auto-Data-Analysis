from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ultralytics")

from fbpipe.config import Settings
from fbpipe.steps.yolo_infer import (
    _build_proboscis_tracker,
    _limit_proboscis_detections,
)


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
