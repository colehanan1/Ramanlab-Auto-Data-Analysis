"""Focused tests for the Parquet I/O migration in calculate_acceleration.

Verifies:
  - read_schema_columns correctly detects a pre-existing acceleration column
    in both Parquet and CSV so the skip logic works as before.
  - write_table produces a .parquet file and read_table round-trips it.
  - calculate_acceleration_for_csv uses read_table / write_table end-to-end.

Run with:
    python -m pytest tests/test_calculate_acceleration_parquet.py -q
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fbpipe.utils.tables import read_schema_columns, read_table, write_table


# Representative per-fly columns from the assignment spec
_BASE_COLS = [
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
    "angle_multiplier",
    "proboscis_distance_pct",
]


def _make_base_df(n: int = 4) -> pd.DataFrame:
    """Return a minimal DataFrame that mimics RMS_calculations file schema."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "frame": np.arange(n),
            "timestamp": np.linspace(0.0, 1.0, n),
            "track_id_class0": rng.integers(0, 10, n),
            "x_class0": rng.uniform(0, 640, n),
            "y_class0": rng.uniform(0, 480, n),
            "corners_class0": [f"[[{i},{i+1}]]" for i in range(n)],
            "track_id_class1": rng.integers(0, 10, n),
            "x_class1": rng.uniform(0, 640, n),
            "y_class1": rng.uniform(0, 480, n),
            "corners_class1": [f"[[{i},{i+2}]]" for i in range(n)],
            "x_anchor": rng.uniform(0, 640, n),
            "y_anchor": rng.uniform(0, 480, n),
            "distance_0_1": rng.uniform(0, 200, n),
            "distance_0_anchor": rng.uniform(0, 200, n),
            "angle_deg_c0_c1_vs_anchor": rng.uniform(-180, 180, n),
            "angle_multiplier": rng.uniform(0.5, 1.5, n),
            # Use the canonical distance-pct column name that calculate_acceleration expects.
            "distance_percentage_0_1": rng.uniform(0, 100, n),
        }
    )
    return df


# ---------------------------------------------------------------------------
# read_schema_columns skip-logic: parquet without acceleration column
# ---------------------------------------------------------------------------

def test_skip_logic_no_accel_col_parquet(tmp_path: Path) -> None:
    """read_schema_columns on a parquet without the accel column returns False."""
    df = _make_base_df()
    out = write_table(df, tmp_path / "fly1_distances.csv")
    assert out.suffix == ".parquet"

    cols = read_schema_columns(out)
    assert "acceleration_pct_per_frame" not in cols


def test_skip_logic_has_accel_col_parquet(tmp_path: Path) -> None:
    """read_schema_columns on a parquet WITH the accel column returns True."""
    df = _make_base_df()
    df["acceleration_pct_per_frame"] = np.zeros(len(df))
    df["combined_distance_x_angle"] = df["distance_percentage_0_1"] * df["angle_multiplier"]
    df["acceleration_flag"] = False

    out = write_table(df, tmp_path / "fly1_distances.csv")
    cols = read_schema_columns(out)
    assert "acceleration_pct_per_frame" in cols


# ---------------------------------------------------------------------------
# read_schema_columns skip-logic: CSV (backward compat for legacy files)
# ---------------------------------------------------------------------------

def test_skip_logic_no_accel_col_csv(tmp_path: Path) -> None:
    """read_schema_columns on a CSV without accel column returns correct list."""
    df = _make_base_df()
    csv_path = tmp_path / "fly1_distances.csv"
    df.to_csv(csv_path, index=False)

    cols = read_schema_columns(csv_path)
    assert "acceleration_pct_per_frame" not in cols


def test_skip_logic_has_accel_col_csv(tmp_path: Path) -> None:
    """read_schema_columns on a CSV WITH accel column returns it in the list."""
    df = _make_base_df()
    df["acceleration_pct_per_frame"] = np.zeros(len(df))
    csv_path = tmp_path / "fly1_distances.csv"
    df.to_csv(csv_path, index=False)

    cols = read_schema_columns(csv_path)
    assert "acceleration_pct_per_frame" in cols


# ---------------------------------------------------------------------------
# write_table -> read_table round-trip preserves numeric values
# ---------------------------------------------------------------------------

def test_write_read_round_trip(tmp_path: Path) -> None:
    """write_table produces .parquet; read_table recovers numerics exactly."""
    df = _make_base_df()
    df["combined_distance_x_angle"] = (
        df["distance_percentage_0_1"] * df["angle_multiplier"]
    )
    accel = np.full(len(df), np.nan)
    accel[1:] = np.diff(df["combined_distance_x_angle"].to_numpy())
    df["acceleration_pct_per_frame"] = accel
    df["acceleration_flag"] = np.abs(accel) > 20.0

    out = write_table(df, tmp_path / "fly1_distances.csv")
    assert out.suffix == ".parquet"
    assert out.is_file()

    recovered = read_table(out)
    # All non-NaN floats must match exactly after parquet round-trip.
    for col in ["combined_distance_x_angle", "distance_percentage_0_1", "angle_multiplier"]:
        original = df[col].to_numpy()
        loaded = recovered[col].to_numpy()
        np.testing.assert_array_equal(original, loaded)

    # NaN in first acceleration row survives.
    assert math.isnan(recovered["acceleration_pct_per_frame"].iloc[0])
    # Remaining rows are identical.
    np.testing.assert_array_equal(
        df["acceleration_pct_per_frame"].iloc[1:].to_numpy(),
        recovered["acceleration_pct_per_frame"].iloc[1:].to_numpy(),
    )


# ---------------------------------------------------------------------------
# Integration: calculate_acceleration_for_csv uses read_table end-to-end
# ---------------------------------------------------------------------------

def test_calculate_acceleration_for_csv_produces_parquet(tmp_path: Path) -> None:
    """calculate_acceleration_for_csv reads via read_table; manual write via write_table."""
    from fbpipe.steps.calculate_acceleration import calculate_acceleration_for_csv

    df = _make_base_df()
    # Write as parquet (as the pipeline now produces).
    src = write_table(df, tmp_path / "fly1_distances.csv")

    result = calculate_acceleration_for_csv(src)
    assert result is not None, "calculate_acceleration_for_csv returned None unexpectedly"
    assert "acceleration_pct_per_frame" in result.columns
    assert "combined_distance_x_angle" in result.columns
    assert "acceleration_flag" in result.columns

    # First acceleration value is NaN (no previous frame).
    assert math.isnan(result["acceleration_pct_per_frame"].iloc[0])

    # Write the result and confirm it is a parquet file that is round-trippable.
    out = write_table(result, src)
    assert out.suffix == ".parquet"
    recovered = read_table(out)
    assert "acceleration_pct_per_frame" in recovered.columns
