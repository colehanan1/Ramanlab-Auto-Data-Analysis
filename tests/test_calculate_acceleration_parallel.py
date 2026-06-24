"""Determinism test for the parallel refactor of calculate_acceleration.

Verifies that running main() with parallel=enabled produces byte-identical
Parquet output to running it serially on an identical fixture.

Run with:
    pytest tests/test_calculate_acceleration_parallel.py -q
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fbpipe.config import ForceSettings, ParallelSettings, Settings
from fbpipe.steps.calculate_acceleration import main
from fbpipe.utils.tables import read_table, write_table


# ---------------------------------------------------------------------------
# Realistic fixture columns from the assignment spec
# ---------------------------------------------------------------------------

_FIXTURE_COLS = [
    "frame",
    "timestamp",
    "x_class0",
    "y_class0",
    "x_class1",
    "y_class1",
    "corners_class1",
    "distance_0_1",
    "angle_deg_c0_c1_vs_anchor",
    # columns required by calculate_acceleration
    "angle_multiplier",
    "distance_percentage_0_1",  # canonical PROBOSCIS_DISTANCE_PCT_COL = "distance_percentage_0_1"
]


def _make_fly_df(n: int = 50, seed: int = 0) -> pd.DataFrame:
    """Return a minimal DataFrame that mimics an RMS_calculations distance file.

    Uses the canonical column names from the assignment spec:
      ['frame','timestamp','x_class0','y_class0','x_class1','y_class1',
       'corners_class1','distance_0_1','angle_deg_c0_c1_vs_anchor']
    plus the two columns required by calculate_acceleration:
      'angle_multiplier'  and  'distance_percentage_0_1' (== PROBOSCIS_DISTANCE_PCT_COL).
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "frame": np.arange(n),
            "timestamp": np.linspace(0.0, n / 40.0, n),
            "x_class0": rng.uniform(0, 1280, n),
            "y_class0": rng.uniform(0, 720, n),
            "x_class1": rng.uniform(0, 1280, n),
            "y_class1": rng.uniform(0, 720, n),
            "corners_class1": [f"[[{i},{i+1}]]" for i in range(n)],
            "distance_0_1": rng.uniform(50, 250, n),
            "angle_deg_c0_c1_vs_anchor": rng.uniform(-180, 180, n),
            "angle_multiplier": rng.uniform(0.5, 1.5, n),
            # canonical PROBOSCIS_DISTANCE_PCT_COL value
            "distance_percentage_0_1": rng.uniform(0, 100, n),
        }
    )


def _build_fixture(root: Path, n_flies: int = 3) -> list[Path]:
    """Create n_flies fly directories under root, each with one RMS file."""
    fly_dirs = []
    for i in range(n_flies):
        fly_dir = root / f"fly{i+1}"
        rms_dir = fly_dir / "RMS_calculations"
        rms_dir.mkdir(parents=True)
        df = _make_fly_df(n=50, seed=i)
        write_table(df, rms_dir / "fly1_distances.parquet")
        fly_dirs.append(fly_dir)
    return fly_dirs


def _make_cfg(root: Path, *, parallel_enabled: bool, n_jobs: int = 2) -> Settings:
    return Settings(
        model_path="",
        main_directories=[str(root)],
        force=ForceSettings(pipeline=False),
        parallel=ParallelSettings(enabled=parallel_enabled, n_jobs=n_jobs),
        auto_discover_flagged=False,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_outputs(root: Path) -> dict[str, pd.DataFrame]:
    """Return {relative_path_str: DataFrame} for every parquet in the fixture."""
    result = {}
    for p in sorted(root.rglob("*.parquet")):
        key = str(p.relative_to(root))
        result[key] = read_table(p)
    return result


# ---------------------------------------------------------------------------
# The determinism test
# ---------------------------------------------------------------------------

def test_parallel_matches_serial(tmp_path: Path) -> None:
    """Serial and parallel runs produce identical Parquet outputs."""
    serial_root = tmp_path / "serial"
    parallel_root = tmp_path / "parallel"

    # Build two identical fixture trees
    _build_fixture(serial_root)
    _build_fixture(parallel_root)

    # Serial run
    cfg_serial = _make_cfg(serial_root, parallel_enabled=False)
    main(cfg_serial)

    # Parallel run (n_jobs=2 to exercise the multiprocess path even with 3 items)
    cfg_parallel = _make_cfg(parallel_root, parallel_enabled=True, n_jobs=2)
    main(cfg_parallel)

    # Collect outputs
    serial_out = _collect_outputs(serial_root)
    parallel_out = _collect_outputs(parallel_root)

    assert set(serial_out.keys()) == set(parallel_out.keys()), (
        f"Different files produced.\nSerial: {set(serial_out)}\nParallel: {set(parallel_out)}"
    )

    for rel_path in serial_out:
        pd.testing.assert_frame_equal(
            serial_out[rel_path].reset_index(drop=True),
            parallel_out[rel_path].reset_index(drop=True),
            check_like=False,
            obj=rel_path,
        )


def test_acceleration_columns_present(tmp_path: Path) -> None:
    """After main(), each RMS file must have the three new acceleration columns."""
    root = tmp_path / "root"
    _build_fixture(root, n_flies=2)
    cfg = _make_cfg(root, parallel_enabled=False)
    main(cfg)

    for p in root.rglob("*.parquet"):
        df = read_table(p)
        for col in ("combined_distance_x_angle", "acceleration_pct_per_frame", "acceleration_flag"):
            assert col in df.columns, f"Missing '{col}' in {p.relative_to(root)}"


def test_skip_already_processed(tmp_path: Path) -> None:
    """A file that already has acceleration_pct_per_frame is not reprocessed."""
    root = tmp_path / "root"
    fly_dir = root / "fly1"
    rms_dir = fly_dir / "RMS_calculations"
    rms_dir.mkdir(parents=True)

    df = _make_fly_df(n=10, seed=99)
    # Pre-add the acceleration column so the skip logic fires
    df["combined_distance_x_angle"] = df["distance_percentage_0_1"] * df["angle_multiplier"]
    accel = np.full(len(df), np.nan)
    accel[1:] = np.diff(df["combined_distance_x_angle"].to_numpy())
    df["acceleration_pct_per_frame"] = accel
    df["acceleration_flag"] = np.abs(accel) > 20.0

    out_path = write_table(df, rms_dir / "fly1_distances.parquet")
    mtime_before = out_path.stat().st_mtime

    cfg = _make_cfg(root, parallel_enabled=False)
    main(cfg)

    mtime_after = out_path.stat().st_mtime
    assert mtime_before == mtime_after, "File was unexpectedly rewritten despite already being processed"
