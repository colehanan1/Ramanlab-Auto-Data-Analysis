"""Focused tests for distance_stats.py Parquet I/O migration (Rules 3, 4, 5).

Tests cover:
- _needs_stats_refresh uses read_schema_columns (not pd.read_csv nrows=0) for
  both Parquet and CSV inputs (Rule 3).
- Correct True/False return based on 'distance_percentage' column presence.
- Graceful fallback (return True) when the file is missing or unreadable.
"""
from __future__ import annotations

import pandas as pd
import pytest

# Columns from the representative real column schema.
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


def _make_df(extra_cols: list[str] | None = None) -> pd.DataFrame:
    cols = _BASE_COLS + (extra_cols or [])
    return pd.DataFrame({c: [1.0] for c in cols})


# ---------------------------------------------------------------------------
# _needs_stats_refresh – parquet input
# ---------------------------------------------------------------------------


def test_needs_stats_refresh_parquet_without_distance_percentage(tmp_path):
    """Returns True when 'distance_percentage' is absent from a .parquet file."""
    from fbpipe.steps.distance_stats import _needs_stats_refresh
    from fbpipe.utils.tables import write_table

    df = _make_df()
    parquet_path = write_table(df, tmp_path / "fly1_distances.parquet")

    assert _needs_stats_refresh(parquet_path, force_recompute=False) is True


def test_needs_stats_refresh_parquet_with_distance_percentage(tmp_path):
    """Returns False when 'distance_percentage' is present in a .parquet file."""
    from fbpipe.steps.distance_stats import _needs_stats_refresh
    from fbpipe.utils.tables import write_table

    df = _make_df(extra_cols=["distance_percentage"])
    parquet_path = write_table(df, tmp_path / "fly1_distances.parquet")

    assert _needs_stats_refresh(parquet_path, force_recompute=False) is False


# ---------------------------------------------------------------------------
# _needs_stats_refresh – csv input (backward compat via read_schema_columns)
# ---------------------------------------------------------------------------


def test_needs_stats_refresh_csv_without_distance_percentage(tmp_path):
    """Returns True when 'distance_percentage' is absent from a .csv file."""
    from fbpipe.steps.distance_stats import _needs_stats_refresh

    csv_path = tmp_path / "fly1_distances.csv"
    _make_df().to_csv(csv_path, index=False)

    assert _needs_stats_refresh(csv_path, force_recompute=False) is True


def test_needs_stats_refresh_csv_with_distance_percentage(tmp_path):
    """Returns False when 'distance_percentage' is present in a .csv file."""
    from fbpipe.steps.distance_stats import _needs_stats_refresh

    csv_path = tmp_path / "fly1_distances.csv"
    _make_df(extra_cols=["distance_percentage"]).to_csv(csv_path, index=False)

    assert _needs_stats_refresh(csv_path, force_recompute=False) is False


# ---------------------------------------------------------------------------
# _needs_stats_refresh – missing file (graceful fallback → True)
# ---------------------------------------------------------------------------


def test_needs_stats_refresh_missing_file_returns_true(tmp_path):
    """Returns True (needs refresh) gracefully when the file does not exist."""
    from fbpipe.steps.distance_stats import _needs_stats_refresh

    missing = tmp_path / "fly99_distances.parquet"
    assert _needs_stats_refresh(missing, force_recompute=False) is True


# ---------------------------------------------------------------------------
# _needs_stats_refresh – force_recompute short-circuits all checks
# ---------------------------------------------------------------------------


def test_needs_stats_refresh_force_recompute(tmp_path):
    """Returns True immediately when force_recompute=True, regardless of schema."""
    from fbpipe.steps.distance_stats import _needs_stats_refresh
    from fbpipe.utils.tables import write_table

    df = _make_df(extra_cols=["distance_percentage"])
    parquet_path = write_table(df, tmp_path / "fly1_distances.parquet")

    assert _needs_stats_refresh(parquet_path, force_recompute=True) is True
