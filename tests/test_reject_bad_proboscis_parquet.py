"""Focused test: reject_bad_proboscis I/O layer reads/writes Parquet correctly.

Exercises the read_table -> sanitize -> write_table path used by
fbpipe.steps.reject_bad_proboscis.main, without requiring a Settings object or
a real filesystem tree.  Uses the representative per-fly distance column schema.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fbpipe.utils.tables import read_table, write_table, table_path


# Representative columns per the assignment brief.
_COLS = [
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
]


def _make_df(n: int = 4) -> pd.DataFrame:
    """Return a tiny synthetic per-fly distance DataFrame."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "frame": list(range(n)),
            "timestamp": [float(i) / 30.0 for i in range(n)],
            "track_id_class0": [0] * n,
            "x_class0": rng.uniform(100, 200, n).tolist(),
            "y_class0": rng.uniform(100, 200, n).tolist(),
            "corners_class0": ["[[0,0],[1,1]]"] * n,  # object column
            "track_id_class1": [1] * n,
            "x_class1": rng.uniform(100, 200, n).tolist(),
            "y_class1": rng.uniform(100, 200, n).tolist(),
            "corners_class1": ["[[2,2],[3,3]]"] * n,
            "x_anchor": [150.0] * n,
            "y_anchor": [150.0] * n,
            "distance_0_1": rng.uniform(10, 80, n).tolist(),
            "distance_0_anchor": rng.uniform(10, 80, n).tolist(),
            "angle_deg_c0_c1_vs_anchor": rng.uniform(0, 360, n).tolist(),
        }
    )


def test_write_table_produces_parquet(tmp_path):
    """write_table always writes .parquet even when given a .csv path."""
    df = _make_df()
    csv_path = tmp_path / "fly1_distances.csv"
    out = write_table(df, csv_path)
    assert out.suffix == ".parquet"
    assert out.exists()
    assert not csv_path.exists()  # no .csv produced


def test_round_trip_preserves_values(tmp_path):
    """Values survive write_table -> read_table round-trip."""
    df = _make_df()
    csv_path = tmp_path / "fly1_distances.csv"
    out = write_table(df, csv_path)

    recovered = read_table(out)
    pd.testing.assert_frame_equal(df, recovered, check_like=False)


def test_read_table_resolves_parquet_from_csv_path(tmp_path):
    """read_table resolves a .csv path to the existing .parquet sibling."""
    df = _make_df()
    csv_path = tmp_path / "fly1_distances.csv"
    write_table(df, csv_path)

    # Pass the .csv path; the .parquet sibling should be auto-resolved.
    recovered = read_table(csv_path)
    pd.testing.assert_frame_equal(df, recovered, check_like=False)


def test_blanked_values_round_trip(tmp_path):
    """NaN-blanked rows survive write_table -> read_table without coercion."""
    df = _make_df(6)
    # Blank row 2 (simulating geometry gate rejection).
    blanked = df.copy()
    blanked.loc[2, ["x_class1", "y_class1", "distance_0_1"]] = np.nan

    csv_path = tmp_path / "fly1_distances.csv"
    out = write_table(blanked, csv_path)
    recovered = read_table(out)

    assert pd.isna(recovered.loc[2, "x_class1"])
    assert pd.isna(recovered.loc[2, "distance_0_1"])
    # Non-blanked rows unchanged.
    assert not pd.isna(recovered.loc[0, "x_class1"])


def test_in_place_overwrite_replaces_parquet(tmp_path):
    """Second write_table call (in-place overwrite) replaces the .parquet file."""
    df_orig = _make_df(4)
    csv_path = tmp_path / "fly1_distances.csv"
    out = write_table(df_orig, csv_path)
    mtime_1 = out.stat().st_mtime_ns

    # Simulate the blanking pass.
    df_mod = df_orig.copy()
    df_mod.loc[1, "distance_0_1"] = np.nan
    write_table(df_mod, csv_path)

    mtime_2 = out.stat().st_mtime_ns
    assert mtime_2 >= mtime_1  # file was touched

    recovered = read_table(out)
    assert pd.isna(recovered.loc[1, "distance_0_1"])
    assert not pd.isna(recovered.loc[0, "distance_0_1"])
