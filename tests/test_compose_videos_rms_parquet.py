"""Focused tests for the Parquet I/O migration in compose_videos_rms._process_fly_angles.

Verifies:
- The updated glob finds .parquet files in addition to .csv files.
- Already-processed .parquet files (with angle_multiplier column) are skipped.
- New pipeline files get angle columns written as .parquet via write_table.
- Mixed .parquet/.csv directories deduplicate by stem (prefer .parquet).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Minimal synthetic DataFrame matching the per-fly distance file schema
# ---------------------------------------------------------------------------

_COLUMNS = [
    "frame",
    "timestamp",
    "track_id_class0",
    "x_class0",
    "y_class0",
    "corners_class0",
    "track_id_class1",
    "x_class1",
    "y_class1",
    "corners_class1",
    "x_anchor",
    "y_anchor",
    "distance_0_1",
    "distance_0_anchor",
    "angle_deg_c0_c1_vs_anchor",
    # columns expected by the angle computation helpers
    "x_class2",
    "y_class2",
    "distance_class1_class2_pct",
]

_N = 20  # row count — tiny enough for speed


def _make_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "frame": np.arange(_N),
            "timestamp": np.arange(_N) / 40.0,
            "track_id_class0": np.zeros(_N, dtype=int),
            "x_class0": rng.uniform(800, 1200, _N),
            "y_class0": rng.uniform(400, 700, _N),
            "corners_class0": ["(0,0,1,1)"] * _N,
            "track_id_class1": np.ones(_N, dtype=int),
            "x_class1": rng.uniform(900, 1100, _N),
            "y_class1": rng.uniform(450, 650, _N),
            "corners_class1": ["(0,0,1,1)"] * _N,
            "x_anchor": np.full(_N, 1080.0),
            "y_anchor": np.full(_N, 540.0),
            "distance_0_1": rng.uniform(50, 200, _N),
            "distance_0_anchor": rng.uniform(50, 200, _N),
            "angle_deg_c0_c1_vs_anchor": rng.uniform(-90, 90, _N),
            "x_class2": rng.uniform(1000, 1100, _N),
            "y_class2": rng.uniform(500, 600, _N),
            "distance_class1_class2_pct": rng.uniform(-50, 50, _N),
        }
    )


# ---------------------------------------------------------------------------
# Test: glob collects .parquet files (not just .csv)
# ---------------------------------------------------------------------------


def test_process_fly_angles_finds_parquet_files(tmp_path: Path) -> None:
    """After migration, .parquet files in RMS_calculations are discovered and processed."""
    import importlib
    import fbpipe.steps.compose_videos_rms as module
    importlib.reload(module)

    from fbpipe.utils.tables import write_table

    rms_dir = tmp_path / "RMS_calculations"
    rms_dir.mkdir()

    # Write a pipeline file as .parquet (as write_table would produce)
    df = _make_df()
    parquet_path = write_table(df, rms_dir / "training_1_distances.csv")
    assert parquet_path.suffix == ".parquet"
    assert parquet_path.exists()

    # Patch out the angle computation helpers so the test is pure I/O
    dummy_angles = pd.Series(np.zeros(_N))
    with patch.object(module, "compute_angle_deg_at_point2", return_value=dummy_angles), \
         patch.object(module, "compute_angle_multiplier_series", return_value=dummy_angles), \
         patch.object(module, "_ead_compute_trim_min_max", return_value=None), \
         patch.object(module, "find_fly_reference_angle", return_value=0.0):

        module._process_fly_angles(tmp_path)

    # The file should have been overwritten with angle columns
    result = pd.read_parquet(parquet_path)
    assert "angle_multiplier" in result.columns
    assert "angle_ARB_deg" in result.columns
    assert "angle_centered_deg" in result.columns


# ---------------------------------------------------------------------------
# Test: already-processed .parquet files (with angle_multiplier) are skipped
# ---------------------------------------------------------------------------


def test_process_fly_angles_skips_already_processed_parquet(tmp_path: Path) -> None:
    """Files with angle_multiplier column are skipped (idempotency)."""
    import importlib
    import fbpipe.steps.compose_videos_rms as module
    importlib.reload(module)

    from fbpipe.utils.tables import write_table

    rms_dir = tmp_path / "RMS_calculations"
    rms_dir.mkdir()

    # Write a file that already has angle_multiplier
    df = _make_df()
    df["angle_ARB_deg"] = 0.0
    df["angle_centered_deg"] = 0.0
    df["angle_multiplier"] = 1.0
    parquet_path = write_table(df, rms_dir / "training_1_distances.csv")

    call_count = {"compute": 0}

    def _counting_compute(df_inner):
        call_count["compute"] += 1
        return pd.Series(np.zeros(len(df_inner)))

    with patch.object(module, "compute_angle_deg_at_point2", side_effect=_counting_compute), \
         patch.object(module, "_ead_compute_trim_min_max", return_value=None), \
         patch.object(module, "find_fly_reference_angle", return_value=0.0):

        module._process_fly_angles(tmp_path)

    # compute_angle_deg_at_point2 must NOT have been called because the file was skipped
    assert call_count["compute"] == 0, (
        "Expected already-processed parquet file to be skipped, "
        f"but compute was called {call_count['compute']} time(s)"
    )


# ---------------------------------------------------------------------------
# Test: stem deduplication (prefer .parquet over .csv when both exist)
# ---------------------------------------------------------------------------


def test_process_fly_angles_deduplicates_by_stem(tmp_path: Path) -> None:
    """When both .parquet and .csv exist for same stem, only .parquet is processed."""
    import importlib
    import fbpipe.steps.compose_videos_rms as module
    importlib.reload(module)

    from fbpipe.utils.tables import write_table

    rms_dir = tmp_path / "RMS_calculations"
    rms_dir.mkdir()

    df = _make_df()
    # Write both .parquet and .csv for the same stem
    parquet_path = write_table(df, rms_dir / "training_1_distances.csv")
    csv_path = rms_dir / "training_1_distances.csv"
    df.to_csv(csv_path, index=False)

    assert parquet_path.exists()
    assert csv_path.exists()

    files_read: list[str] = []

    original_read_table = module.read_table

    def _tracking_read_table(path, **kwargs):
        files_read.append(str(path))
        return original_read_table(path, **kwargs)

    dummy_angles = pd.Series(np.zeros(_N))
    with patch.object(module, "read_table", side_effect=_tracking_read_table), \
         patch.object(module, "compute_angle_deg_at_point2", return_value=dummy_angles), \
         patch.object(module, "compute_angle_multiplier_series", return_value=dummy_angles), \
         patch.object(module, "_ead_compute_trim_min_max", return_value=None), \
         patch.object(module, "find_fly_reference_angle", return_value=0.0):

        module._process_fly_angles(tmp_path)

    # Only the .parquet file should have been read (csv stem is deduplicated)
    assert all(".parquet" in f for f in files_read), (
        f"Expected only .parquet reads, got: {files_read}"
    )
    assert not any(".csv" in f for f in files_read), (
        f"Expected .csv to be skipped by stem deduplication, got: {files_read}"
    )


# ---------------------------------------------------------------------------
# Test: write_table produces .parquet output for processed files
# ---------------------------------------------------------------------------


def test_process_fly_angles_writes_parquet_not_csv(tmp_path: Path) -> None:
    """Processed files are written as .parquet, not .csv."""
    import importlib
    import fbpipe.steps.compose_videos_rms as module
    importlib.reload(module)

    rms_dir = tmp_path / "RMS_calculations"
    rms_dir.mkdir()

    # Write a legacy CSV file (simulate input from before Parquet migration)
    df = _make_df()
    csv_path = rms_dir / "training_1_distances.csv"
    df.to_csv(csv_path, index=False)

    dummy_angles = pd.Series(np.zeros(_N))
    with patch.object(module, "compute_angle_deg_at_point2", return_value=dummy_angles), \
         patch.object(module, "compute_angle_multiplier_series", return_value=dummy_angles), \
         patch.object(module, "_ead_compute_trim_min_max", return_value=None), \
         patch.object(module, "find_fly_reference_angle", return_value=0.0):

        module._process_fly_angles(tmp_path)

    expected_parquet = rms_dir / "training_1_distances.parquet"
    assert expected_parquet.exists(), (
        f"Expected .parquet output at {expected_parquet}, but it does not exist"
    )
    result = pd.read_parquet(expected_parquet)
    assert "angle_multiplier" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
