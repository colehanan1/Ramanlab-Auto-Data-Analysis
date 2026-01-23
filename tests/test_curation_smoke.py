"""Simple smoke test for YOLO curation module."""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fbpipe.steps.curate_yolo_dataset import (
    compute_jitter,
    compute_quality_metrics,
    is_bad_tracking,
)
from fbpipe.utils.columns import PROBOSCIS_CLASS

def test_compute_jitter():
    """Test jitter computation."""
    print("Testing jitter computation...")
    prob_x = f"x_class{PROBOSCIS_CLASS}"
    prob_y = f"y_class{PROBOSCIS_CLASS}"
    df = pd.DataFrame({
        prob_x: [100, 105, 110, 115, 120],
        prob_y: [200, 200, 200, 200, 200],
    })

    jitter = compute_jitter(df, prob_x, prob_y)

    assert len(jitter) == 5
    assert jitter[0] == 0.0
    assert abs(jitter[1] - 5.0) < 0.01
    print("✓ Jitter computation works correctly")

def test_quality_metrics():
    """Test quality metrics computation."""
    print("Testing quality metrics...")

    # Create synthetic data with correct column names
    prob_x = f"x_class{PROBOSCIS_CLASS}"
    prob_y = f"y_class{PROBOSCIS_CLASS}"
    df = pd.DataFrame({
        prob_x: np.arange(100, 200, 1),
        prob_y: np.ones(100) * 200,
    })

    video_path = Path("/tmp/test.mp4")
    metrics = compute_quality_metrics(
        df, video_path, {"max_jitter_px": 50.0, "max_missing_pct": 0.10}
    )

    assert metrics["total_frames"] == 100
    assert metrics["missing_frames"] == 0
    assert metrics["pct_missing"] == 0.0
    print("✓ Quality metrics computation works correctly")

def test_flagging():
    """Test video flagging logic."""
    print("Testing video flagging...")

    # Good quality
    metrics_good = {
        "pct_missing": 0.05,
        "median_jitter_px": 30.0,
    }
    thresholds = {"max_jitter_px": 50.0, "max_missing_pct": 0.10}

    assert is_bad_tracking(metrics_good, thresholds) is False

    # Bad quality (high jitter)
    metrics_bad = {
        "pct_missing": 0.05,
        "median_jitter_px": 60.0,
    }

    assert is_bad_tracking(metrics_bad, thresholds) is True
    print("✓ Video flagging logic works correctly")

def test_augmentation_imports():
    """Test augmentation module imports."""
    print("Testing augmentation imports...")

    from fbpipe.utils.augmentation import (
        augment_horizontal_flip,
        augment_brightness_contrast,
        augment_rotation,
    )

    # Test horizontal flip
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    annotations = [(0, 320, 240, 50, 50, 0.0)]

    flipped_image, flipped_ann = augment_horizontal_flip(image, annotations)
    assert flipped_image.shape == image.shape
    assert len(flipped_ann) == len(annotations)
    print("✓ Augmentation functions work correctly")

if __name__ == "__main__":
    print("=" * 60)
    print("YOLO Dataset Curation - Smoke Tests")
    print("=" * 60)

    try:
        test_compute_jitter()
        test_quality_metrics()
        test_flagging()
        test_augmentation_imports()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
