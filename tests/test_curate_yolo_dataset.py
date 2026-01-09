"""
Unit tests for YOLO dataset curation module.

Tests quality metrics computation, video flagging, frame extraction,
and augmentation functionality.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fbpipe.steps.curate_yolo_dataset import (
    compute_jitter,
    compute_quality_metrics,
    is_bad_tracking,
)
from src.fbpipe.utils.augmentation import (
    augment_horizontal_flip,
    augment_brightness_contrast,
    augment_rotation,
    parse_yolo_annotation,
    save_yolo_annotation,
)


class TestQualityMetrics:
    """Tests for quality metrics computation."""

    def test_compute_jitter_basic(self):
        """Test basic jitter computation."""
        df = pd.DataFrame({
            "proboscis_x": [100, 105, 110, 115, 120],
            "proboscis_y": [200, 200, 200, 200, 200],
        })

        jitter = compute_jitter(df, "proboscis_x", "proboscis_y")

        # Expected: 5px movement per frame (except first which is 0)
        assert len(jitter) == 5
        assert jitter[0] == 0.0  # First frame has no previous frame
        assert abs(jitter[1] - 5.0) < 0.01
        assert abs(jitter[2] - 5.0) < 0.01

    def test_compute_jitter_with_missing(self):
        """Test jitter computation with missing values."""
        df = pd.DataFrame({
            "proboscis_x": [100, np.nan, 110, 115, 120],
            "proboscis_y": [200, np.nan, 200, 200, 200],
        })

        jitter = compute_jitter(df, "proboscis_x", "proboscis_y")

        assert len(jitter) == 5
        assert not pd.isna(jitter[0])  # First is 0
        # Check that NaN propagates correctly

    def test_compute_quality_metrics_good_tracking(self):
        """Test metrics for video with good tracking."""
        df = pd.DataFrame({
            "proboscis_x": np.arange(100, 200, 1),
            "proboscis_y": np.ones(100) * 200,
        })

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            video_path = Path(tmp.name)
            metrics = compute_quality_metrics(
                df, video_path, {"max_jitter_px": 50.0, "max_missing_pct": 0.10}
            )

            assert metrics["total_frames"] == 100
            assert metrics["missing_frames"] == 0
            assert metrics["pct_missing"] == 0.0
            assert metrics["median_jitter_px"] < 2.0  # Small, consistent movement

    def test_compute_quality_metrics_bad_tracking(self):
        """Test metrics for video with poor tracking."""
        # Create data with 50% missing frames and high jitter
        x_vals = np.random.uniform(50, 150, 50)
        x_vals = np.concatenate([x_vals, [np.nan] * 50])
        y_vals = np.random.uniform(50, 150, 50)
        y_vals = np.concatenate([y_vals, [np.nan] * 50])

        df = pd.DataFrame({
            "proboscis_x": x_vals,
            "proboscis_y": y_vals,
        })

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            video_path = Path(tmp.name)
            metrics = compute_quality_metrics(
                df, video_path, {"max_jitter_px": 50.0, "max_missing_pct": 0.10}
            )

            assert metrics["total_frames"] == 100
            assert metrics["missing_frames"] == 50
            assert metrics["pct_missing"] == 0.5


class TestVideoFlagging:
    """Tests for video flagging logic."""

    def test_is_bad_tracking_high_jitter(self):
        """Test flagging based on high jitter."""
        metrics = {
            "pct_missing": 0.05,
            "median_jitter_px": 60.0,
        }
        thresholds = {"max_jitter_px": 50.0, "max_missing_pct": 0.10}

        assert is_bad_tracking(metrics, thresholds) is True

    def test_is_bad_tracking_high_missing(self):
        """Test flagging based on missing frames."""
        metrics = {
            "pct_missing": 0.15,
            "median_jitter_px": 30.0,
        }
        thresholds = {"max_jitter_px": 50.0, "max_missing_pct": 0.10}

        assert is_bad_tracking(metrics, thresholds) is True

    def test_is_bad_tracking_good_quality(self):
        """Test good quality video is not flagged."""
        metrics = {
            "pct_missing": 0.05,
            "median_jitter_px": 30.0,
        }
        thresholds = {"max_jitter_px": 50.0, "max_missing_pct": 0.10}

        assert is_bad_tracking(metrics, thresholds) is False


class TestAugmentation:
    """Tests for data augmentation functions."""

    def test_horizontal_flip(self):
        """Test horizontal flip augmentation."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        annotations = [
            (0, 320, 240, 50, 50, 0.0),  # Center of image
            (1, 100, 100, 30, 30, 0.0),  # Left side
        ]

        flipped_image, flipped_ann = augment_horizontal_flip(image, annotations)

        assert flipped_image.shape == image.shape
        assert len(flipped_ann) == len(annotations)

        # Check x coordinate is flipped
        assert abs(flipped_ann[0][1] - 320) < 1  # Center stays center
        assert flipped_ann[1][1] > 500  # Left side moves to right

    def test_brightness_contrast(self):
        """Test brightness/contrast augmentation."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        annotations = [(0, 320, 240, 50, 50, 0.0)]

        aug_image, aug_ann = augment_brightness_contrast(image, annotations, 10.0, 0.1)

        assert aug_image.shape == image.shape
        assert aug_ann == annotations  # Annotations unchanged

    def test_rotation(self):
        """Test rotation augmentation."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        annotations = [(0, 320, 240, 50, 50, 0.0)]

        rotated_image, rotated_ann = augment_rotation(image, annotations, 5.0)

        assert rotated_image.shape == image.shape
        assert len(rotated_ann) == len(annotations)

        # Center point should be close to center after small rotation
        center_x, center_y = rotated_ann[0][1], rotated_ann[0][2]
        assert abs(center_x - 320) < 20
        assert abs(center_y - 240) < 20

    def test_yolo_annotation_parsing(self):
        """Test YOLO annotation file parsing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = Path(tmpdir) / "test.txt"

            # Write sample YOLO annotation
            with open(txt_path, "w") as f:
                f.write("0 0.5 0.5 0.1 0.1\n")  # Standard bbox
                f.write("1 0.3 0.7 0.15 0.15 45.0\n")  # OBB with angle

            annotations = parse_yolo_annotation(txt_path, 640, 480)

            assert len(annotations) == 2
            assert annotations[0][0] == 0  # class_id
            assert abs(annotations[0][1] - 320) < 1  # x_center (denormalized)
            assert abs(annotations[0][2] - 240) < 1  # y_center
            assert annotations[0][5] == 0.0  # angle

            assert annotations[1][0] == 1
            assert annotations[1][5] == 45.0  # OBB angle

    def test_yolo_annotation_saving(self):
        """Test YOLO annotation file saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = Path(tmpdir) / "test.txt"

            annotations = [
                (0, 320, 240, 64, 48, 0.0),
                (1, 192, 336, 96, 72, 30.0),
            ]

            save_yolo_annotation(txt_path, annotations, 640, 480)

            # Read back and verify
            with open(txt_path, "r") as f:
                lines = f.readlines()

            assert len(lines) == 2

            # Check normalization
            parts = lines[0].strip().split()
            assert float(parts[1]) == pytest.approx(0.5, abs=0.01)  # x normalized
            assert float(parts[2]) == pytest.approx(0.5, abs=0.01)  # y normalized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
