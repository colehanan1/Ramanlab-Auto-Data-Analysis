"""Focused test: distance_normalize I/O layer uses read_schema_columns / read_table / write_table.

Exercises:
  - Rule 3: nrows=0 header check replaced by read_schema_columns.
  - Rule 4: usecols+nrows=1 value snapshot replaced by read_table(columns=...).iloc[:1].
  - Rule 1: full read via read_table; write via write_table (produces .parquet).
  - _is_already_normalized returns True when parquet artifact matches current stats,
    allowing the skip path to fire correctly.

Does NOT require GPU, Settings object, or the /securedstorage mount.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fbpipe.utils.columns import (
    PROBOSCIS_DISTANCE_COL,
    PROBOSCIS_DISTANCE_PCT_COL,
    PROBOSCIS_MAX_DISTANCE_COL,
    PROBOSCIS_MIN_DISTANCE_COL,
    EYE_CLASS,
    PROBOSCIS_CLASS,
)
from fbpipe.utils.tables import read_schema_columns, read_table, write_table, table_path
from fbpipe.steps.distance_normalize import _is_already_normalized

# Representative per-fly distance columns (from assignment brief)
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
]

_GMIN = 5.0
_GMAX = 100.0
_EFFECTIVE_MAX = 100.0
_EFFECTIVE_COL = f"effective_max_distance_{EYE_CLASS}_{PROBOSCIS_CLASS}"


def _make_normalized_df(n: int = 4) -> pd.DataFrame:
    """Return a tiny normalized-distance DataFrame with all expected columns."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "frame": list(range(n)),
            "timestamp": [float(i) / 30.0 for i in range(n)],
            "track_id_class0": [0] * n,
            "x_class0": rng.uniform(100, 200, n),
            "y_class0": rng.uniform(100, 200, n),
            "corners_class0": ["[[0,0],[1,1]]"] * n,
            "track_id_class1": [1] * n,
            "x_class1": rng.uniform(100, 200, n),
            "y_class1": rng.uniform(100, 200, n),
            "corners_class1": ["[[2,2],[3,3]]"] * n,
            "x_anchor": [150.0] * n,
            "y_anchor": [150.0] * n,
            "distance_0_1": rng.uniform(_GMIN + 1, _GMAX - 1, n),
            "distance_0_anchor": rng.uniform(10, 80, n),
            "angle_deg_c0_c1_vs_anchor": rng.uniform(0, 360, n),
            # Normalized columns
            PROBOSCIS_DISTANCE_COL: rng.uniform(_GMIN, _GMAX, n),
            PROBOSCIS_DISTANCE_PCT_COL: rng.uniform(0, 100, n),
            PROBOSCIS_MIN_DISTANCE_COL: [_GMIN] * n,
            PROBOSCIS_MAX_DISTANCE_COL: [_GMAX] * n,
            _EFFECTIVE_COL: [_EFFECTIVE_MAX] * n,
        }
    )
    return df


# ---------------------------------------------------------------------------
# Test: read_schema_columns round-trips column names from .parquet
# ---------------------------------------------------------------------------


def test_read_schema_columns_parquet(tmp_path: "Path") -> None:
    """read_schema_columns returns correct column names for a written Parquet file."""
    df = _make_normalized_df()
    out = write_table(df, tmp_path / "fly1_distances.csv")
    assert out.suffix == ".parquet"

    cols = read_schema_columns(out)
    assert set(cols) == set(df.columns)


# ---------------------------------------------------------------------------
# Test: value snapshot via read_table(columns=...).iloc[:1]
# ---------------------------------------------------------------------------


def test_snapshot_read_parquet(tmp_path: "Path") -> None:
    """Selective column read returns correct first-row values (rule 4)."""
    df = _make_normalized_df()
    out = write_table(df, tmp_path / "fly1_distances.csv")

    check_cols = [PROBOSCIS_MIN_DISTANCE_COL, PROBOSCIS_MAX_DISTANCE_COL, _EFFECTIVE_COL]
    snapshot = read_table(out, columns=check_cols).iloc[:1]

    assert len(snapshot) == 1
    assert np.isclose(float(snapshot[PROBOSCIS_MIN_DISTANCE_COL].iloc[0]), _GMIN)
    assert np.isclose(float(snapshot[PROBOSCIS_MAX_DISTANCE_COL].iloc[0]), _GMAX)
    assert np.isclose(float(snapshot[_EFFECTIVE_COL].iloc[0]), _EFFECTIVE_MAX)


# ---------------------------------------------------------------------------
# Test: _is_already_normalized returns True for matching parquet artifact
# ---------------------------------------------------------------------------


def test_is_already_normalized_true(tmp_path: "Path") -> None:
    """Already-normalized artifact with matching stats causes skip (returns True)."""
    df = _make_normalized_df()
    out = write_table(df, tmp_path / "fly1_distances.csv")

    cols = read_schema_columns(out)
    check_cols = [
        c for c in (PROBOSCIS_MIN_DISTANCE_COL, PROBOSCIS_MAX_DISTANCE_COL, _EFFECTIVE_COL)
        if c in cols
    ]
    snapshot = read_table(out, columns=check_cols).iloc[:1]

    result = _is_already_normalized(
        cols,
        snapshot,
        gmin=_GMIN,
        gmax=_GMAX,
        effective_max=_EFFECTIVE_MAX,
    )
    assert result is True


def test_is_already_normalized_false_wrong_gmin(tmp_path: "Path") -> None:
    """Different gmin forces re-normalization (returns False)."""
    df = _make_normalized_df()
    out = write_table(df, tmp_path / "fly1_distances.csv")

    cols = read_schema_columns(out)
    check_cols = [
        c for c in (PROBOSCIS_MIN_DISTANCE_COL, PROBOSCIS_MAX_DISTANCE_COL, _EFFECTIVE_COL)
        if c in cols
    ]
    snapshot = read_table(out, columns=check_cols).iloc[:1]

    result = _is_already_normalized(
        cols,
        snapshot,
        gmin=_GMIN + 1.0,  # different
        gmax=_GMAX,
        effective_max=_EFFECTIVE_MAX,
    )
    assert result is False


def test_is_already_normalized_false_missing_pct_col() -> None:
    """Returns False when percentage column is absent (file not yet normalized)."""
    cols = _BASE_COLS  # no PROBOSCIS_DISTANCE_PCT_COL
    snapshot = pd.DataFrame()

    result = _is_already_normalized(
        cols,
        snapshot,
        gmin=_GMIN,
        gmax=_GMAX,
        effective_max=_EFFECTIVE_MAX,
    )
    assert result is False


# ---------------------------------------------------------------------------
# Test: write_table produces .parquet with correct data (rule 1)
# ---------------------------------------------------------------------------


def test_write_table_produces_parquet_and_roundtrip(tmp_path: "Path") -> None:
    """write_table writes Parquet; read_table reads it back with identical numeric values."""
    df = _make_normalized_df()
    out = write_table(df, tmp_path / "fly1_distances.csv")

    assert out.suffix == ".parquet"
    df2 = read_table(out)

    assert list(df2.columns) == list(df.columns)
    # Numeric columns round-trip exactly
    for col in (PROBOSCIS_MIN_DISTANCE_COL, PROBOSCIS_MAX_DISTANCE_COL, _EFFECTIVE_COL):
        np.testing.assert_array_equal(df2[col].to_numpy(), df[col].to_numpy())
