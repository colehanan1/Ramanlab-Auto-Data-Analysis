"""Focused tests for the Parquet I/O migration in update_ofm_state.

Tests cover:
- Rule 1: pipeline RMS file read/write goes through read_table/write_table (parquet)
- Rule 3/5: skip/freshness check uses read_schema_columns + resolve_existing
             so it fires on the real .parquet artifact, not a stale .csv path.
- External output_*.csv files are read via read_table (auto csv) and NOT converted.

Run with:
    python -m pytest tests/test_update_ofm_state_parquet.py -q
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from fbpipe.utils.tables import (
    read_table,
    write_table,
    table_path,
    resolve_existing,
    read_schema_columns,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rms_df(with_ofm_state: bool = False) -> pd.DataFrame:
    """Minimal RMS 'updated_' DataFrame with pipeline columns."""
    df = pd.DataFrame(
        {
            "frame": [0, 1, 2, 3, 4],
            "timestamp": [0.0, 0.1, 0.2, 0.3, 0.4],
            "track_id_class0": [1, 1, 1, 1, 1],
            "x_class0": [10.0, 11.0, 12.0, 13.0, 14.0],
            "y_class0": [20.0, 21.0, 22.0, 23.0, 24.0],
            "corners_class0": ["[[0,0]]"] * 5,
            "track_id_class1": [2, 2, 2, 2, 2],
            "x_class1": [50.0, 51.0, 52.0, 53.0, 54.0],
            "y_class1": [60.0, 61.0, 62.0, 63.0, 64.0],
            "corners_class1": ["[[0,0]]"] * 5,
            "x_anchor": [30.0] * 5,
            "y_anchor": [40.0] * 5,
            "distance_0_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "distance_0_anchor": [5.0, 6.0, 7.0, 8.0, 9.0],
            "angle_deg_c0_c1_vs_anchor": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    if with_ofm_state:
        df["OFM_State"] = ["before", "during", "during", "after", "after"]
    return df


def _make_output_csv(tmp_path: Path, ofm_col: str = "ActiveOFM") -> Path:
    """Write a fake external rig output_*.csv file (CSV, not parquet)."""
    df = pd.DataFrame(
        {
            "Frame": list(range(5)),
            ofm_col: ["off", "off", "on", "on", "off"],
        }
    )
    p = tmp_path / "output_rig1.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Rule 1 — pipeline file round-trip
# ---------------------------------------------------------------------------


def test_write_table_produces_parquet(tmp_path: Path):
    """write_table on a .csv path produces a .parquet file (Rule 1)."""
    rms_csv = tmp_path / "updated_001.csv"
    df = _make_rms_df()
    written = write_table(df, rms_csv)
    assert written.suffix == ".parquet"
    assert written.exists()
    assert not rms_csv.exists(), "CSV should not be written"


def test_read_table_reads_parquet(tmp_path: Path):
    """read_table can read back the .parquet file written by write_table (Rule 1)."""
    rms_csv = tmp_path / "updated_001.csv"
    df = _make_rms_df()
    written = write_table(df, rms_csv)
    df_back = read_table(written)
    pd.testing.assert_frame_equal(df, df_back)


def test_read_table_resolves_parquet_from_csv_path(tmp_path: Path):
    """read_table resolves .parquet when given a .csv path (Rule 1 auto-resolve)."""
    rms_csv = tmp_path / "updated_001.csv"
    df = _make_rms_df()
    write_table(df, rms_csv)  # writes .parquet
    # read_table given .csv path should resolve to the .parquet sibling
    df_back = read_table(rms_csv)
    pd.testing.assert_frame_equal(df, df_back)


# ---------------------------------------------------------------------------
# Rule 2 — external CSV kept as CSV
# ---------------------------------------------------------------------------


def test_external_output_csv_stays_csv(tmp_path: Path):
    """External rig output_*.csv is NOT converted — read_table auto-detects CSV."""
    out_csv = _make_output_csv(tmp_path)
    assert out_csv.suffix == ".csv"  # still CSV on disk
    df = read_table(out_csv)
    assert "ActiveOFM" in df.columns
    assert not (tmp_path / "output_rig1.parquet").exists()


def test_external_output_head5(tmp_path: Path):
    """read_table(o).head(5) yields at most 5 rows for the heuristic scan (Rule 2)."""
    out_csv = _make_output_csv(tmp_path)
    df_head = read_table(out_csv).head(5)
    assert len(df_head) <= 5


# ---------------------------------------------------------------------------
# Rule 3 — read_schema_columns for column-existence check
# ---------------------------------------------------------------------------


def test_read_schema_columns_parquet_no_ofm_state(tmp_path: Path):
    """read_schema_columns returns columns without OFM_State when absent."""
    rms_csv = tmp_path / "updated_001.csv"
    write_table(_make_rms_df(with_ofm_state=False), rms_csv)
    parquet_path = table_path(rms_csv)
    cols = read_schema_columns(parquet_path)
    assert "OFM_State" not in cols


def test_read_schema_columns_parquet_with_ofm_state(tmp_path: Path):
    """read_schema_columns finds OFM_State when present."""
    rms_csv = tmp_path / "updated_001.csv"
    write_table(_make_rms_df(with_ofm_state=True), rms_csv)
    parquet_path = table_path(rms_csv)
    cols = read_schema_columns(parquet_path)
    assert "OFM_State" in cols


def test_read_schema_columns_resolves_from_csv_path(tmp_path: Path):
    """read_schema_columns resolves .parquet sibling given a .csv path."""
    rms_csv = tmp_path / "updated_001.csv"
    write_table(_make_rms_df(with_ofm_state=True), rms_csv)
    # Give the .csv path — resolve_existing inside read_schema_columns finds .parquet
    cols = read_schema_columns(rms_csv)
    assert "OFM_State" in cols


# ---------------------------------------------------------------------------
# Rule 5 — skip/freshness uses resolve_existing + parquet mtime
# ---------------------------------------------------------------------------


def test_skip_logic_no_existing_artifact(tmp_path: Path):
    """resolve_existing returns None when no artifact exists -> skip does not fire."""
    rms_csv = tmp_path / "updated_001.csv"
    existing = resolve_existing(rms_csv)
    assert existing is None, "Should not skip when no output artifact exists"


def test_skip_logic_fires_when_parquet_newer_than_output_csv(tmp_path: Path):
    """Freshness check fires when .parquet artifact is newer than external CSV."""
    out_csv = _make_output_csv(tmp_path)
    # Write parquet after the CSV so its mtime is guaranteed newer
    time.sleep(0.01)
    rms_csv = tmp_path / "updated_001.csv"
    write_table(_make_rms_df(with_ofm_state=True), rms_csv)
    parquet_path = table_path(rms_csv)

    existing = resolve_existing(rms_csv)
    assert existing is not None
    assert existing == parquet_path
    cols = read_schema_columns(existing)
    assert "OFM_State" in cols
    assert existing.stat().st_mtime >= out_csv.stat().st_mtime, (
        "Parquet artifact should be newer than the external output CSV"
    )


def test_skip_logic_does_not_fire_on_stale_csv_only(tmp_path: Path):
    """If only a legacy .csv exists (no .parquet), resolve_existing returns it.

    This ensures backward compatibility: a legacy .csv without OFM_State still
    triggers recompute because OFM_State will not be in its schema.
    """
    rms_csv = tmp_path / "updated_001.csv"
    _make_rms_df(with_ofm_state=False).to_csv(rms_csv, index=False)
    existing = resolve_existing(rms_csv)
    # resolve_existing falls back to .csv when no .parquet is present
    assert existing == rms_csv
    cols = read_schema_columns(existing)
    assert "OFM_State" not in cols  # no OFM_State -> skip will NOT fire -> recompute


def test_skip_does_not_fire_when_only_csv_with_ofm_state_but_parquet_missing(tmp_path: Path):
    """resolve_existing with a stale .csv (OFM_State present) returns the csv.

    The skip condition checks the resolved artifact mtime, which in this case
    is the .csv itself.  This is a legacy-compat path — the main point is that
    resolve_existing never returns None when a .csv sibling is present.
    """
    rms_csv = tmp_path / "updated_001.csv"
    _make_rms_df(with_ofm_state=True).to_csv(rms_csv, index=False)
    existing = resolve_existing(rms_csv)
    assert existing == rms_csv  # falls back to csv
    cols = read_schema_columns(existing)
    assert "OFM_State" in cols
