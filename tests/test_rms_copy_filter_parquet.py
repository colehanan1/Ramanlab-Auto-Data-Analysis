"""Focused tests for rms_copy_filter.py parquet skip/freshness logic.

Verifies that:
1. Skip check uses the .parquet output path (not .csv), so a missing parquet
   is not mistakenly skipped.
2. write_table produces a .parquet file, not a .csv.
3. The freshness check correctly uses the .parquet mtime.

Run with:
    python -m pytest tests/test_rms_copy_filter_parquet.py -v
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from fbpipe.utils.tables import read_table, table_path, write_table


# Representative per-fly distance columns (subset used by rms_copy_filter)
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


def _make_distance_df(n: int = 4) -> pd.DataFrame:
    """Return a minimal per-fly distance DataFrame using the real column schema."""
    return pd.DataFrame(
        {
            "frame": list(range(n)),
            "timestamp": [i * 0.025 for i in range(n)],
            "track_id_class0": [0] * n,
            "x_class0": [100.0 + i for i in range(n)],
            "y_class0": [200.0 + i for i in range(n)],
            "corners_class0": ["[[1,2],[3,4]]"] * n,
            "track_id_class1": [1] * n,
            "x_class1": [300.0 + i for i in range(n)],
            "y_class1": [400.0 + i for i in range(n)],
            "corners_class1": ["[[5,6],[7,8]]"] * n,
            "x_anchor": [150.0] * n,
            "y_anchor": [250.0] * n,
            "distance_0_1": [50.0 + i * 0.5 for i in range(n)],
            "distance_0_anchor": [55.0 + i * 0.5 for i in range(n)],
            "angle_deg_c0_c1_vs_anchor": [45.0 + i for i in range(n)],
        }
    )


class TestTablePathHelper:
    """table_path() always returns .parquet regardless of input suffix."""

    def test_csv_suffix_becomes_parquet(self):
        p = table_path(Path("/some/dir/updated_fly1_distances.csv"))
        assert p.suffix == ".parquet"
        assert p.stem == "updated_fly1_distances"

    def test_parquet_suffix_unchanged(self):
        p = table_path(Path("/some/dir/updated_fly1_distances.parquet"))
        assert p.suffix == ".parquet"

    def test_no_suffix_gets_parquet(self):
        p = table_path(Path("/some/dir/updated_fly1_distances"))
        assert p.suffix == ".parquet"


class TestSkipLogicUsesParquetPath:
    """Freshness guard must reference .parquet output, not .csv."""

    def test_parquet_exists_is_treated_as_up_to_date(self, tmp_path: Path):
        """If the .parquet output already exists and is newer, skip should fire."""
        # Create an input CSV
        input_csv = tmp_path / "fly1_distances.csv"
        df = _make_distance_df()
        df.to_csv(input_csv, index=False)

        # Produce the output parquet (simulating a prior run)
        out_name = "updated_" + input_csv.name   # updated_fly1_distances.csv
        out_path = tmp_path / out_name
        out_parquet = table_path(out_path)       # updated_fly1_distances.parquet

        cols = ["frame", "timestamp", "x_class0", "y_class0"]
        written = write_table(df[cols], out_path)
        assert written == out_parquet
        assert out_parquet.exists()

        # Freshness check: parquet mtime >= input csv mtime  -> should skip
        assert out_parquet.stat().st_mtime >= input_csv.stat().st_mtime

    def test_no_parquet_output_means_not_up_to_date(self, tmp_path: Path):
        """If only a stale .csv output exists (legacy), parquet check should fail -> rerun."""
        input_csv = tmp_path / "fly1_distances.csv"
        df = _make_distance_df()
        df.to_csv(input_csv, index=False)

        out_name = "updated_" + input_csv.name
        out_path = tmp_path / out_name
        out_parquet = table_path(out_path)   # .parquet

        # Only write a legacy CSV output (NOT parquet) — simulates old run
        df[["frame", "timestamp"]].to_csv(out_path, index=False)
        assert out_path.exists()
        assert not out_parquet.exists()

        # The new skip logic checks out_parquet.exists() — must be False -> rerun
        should_skip = (
            out_parquet.exists()
            and out_parquet.stat().st_mtime >= input_csv.stat().st_mtime
        )
        assert not should_skip, (
            "Should NOT skip when only a legacy CSV output exists, not the parquet"
        )

    def test_stale_parquet_output_triggers_rerun(self, tmp_path: Path):
        """If parquet exists but is older than the input, should NOT skip."""
        # Write parquet first, then update the input CSV so input is newer
        input_csv = tmp_path / "fly1_distances.csv"
        df = _make_distance_df()
        df.to_csv(input_csv, index=False)

        out_name = "updated_" + input_csv.name
        out_path = tmp_path / out_name
        out_parquet = table_path(out_path)

        cols = ["frame", "timestamp", "x_class0", "y_class0"]
        write_table(df[cols], out_path)
        assert out_parquet.exists()

        # Simulate input being updated after the output was produced
        time.sleep(0.01)  # small delay so mtime differs
        input_csv.touch()  # bump mtime

        should_skip = (
            out_parquet.exists()
            and out_parquet.stat().st_mtime >= input_csv.stat().st_mtime
        )
        assert not should_skip, "Stale parquet should not be considered up to date"


class TestWriteTableProducesParquet:
    """write_table always produces a .parquet file."""

    def test_write_table_from_csv_named_output(self, tmp_path: Path):
        """Even when out_path has .csv in its name, write_table writes .parquet."""
        df = _make_distance_df()
        out_csv_name = tmp_path / "updated_fly1_distances.csv"
        written = write_table(df[["frame", "timestamp"]], out_csv_name)

        assert written.suffix == ".parquet"
        assert written.name == "updated_fly1_distances.parquet"
        assert written.exists()
        assert not out_csv_name.exists(), "Should NOT have written a .csv file"

    def test_round_trip_preserves_values(self, tmp_path: Path):
        """Values written by write_table can be read back identically by read_table."""
        df = _make_distance_df()
        cols = ["frame", "timestamp", "x_class0", "y_class0", "corners_class0"]
        out = tmp_path / "updated_fly1_distances.csv"
        write_table(df[cols], out)

        # read_table resolves to the .parquet sibling
        read_back = read_table(out)
        pd.testing.assert_frame_equal(df[cols].reset_index(drop=True), read_back)


class TestReadTableOnExternalInputs:
    """read_table auto-detects .csv for external inputs."""

    def test_read_csv_input(self, tmp_path: Path):
        """External .csv input files are read transparently by read_table."""
        df = _make_distance_df()
        csv_path = tmp_path / "fly1_distances.csv"
        df.to_csv(csv_path, index=False)

        loaded = read_table(csv_path)
        assert list(loaded.columns) == list(df.columns)
        assert len(loaded) == len(df)

    def test_read_parquet_input(self, tmp_path: Path):
        """Parquet inputs (from iter_fly_distance_csvs) are also read transparently."""
        df = _make_distance_df()
        parquet_path = tmp_path / "fly1_distances.parquet"
        df.to_parquet(parquet_path, index=False)

        loaded = read_table(parquet_path)
        pd.testing.assert_frame_equal(df.reset_index(drop=True), loaded)
