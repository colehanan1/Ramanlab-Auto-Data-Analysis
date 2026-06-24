"""Tests for fbpipe.utils.tables — written first (TDD).

Run with:
    python -m pytest tests/test_tables.py -q
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rich_df() -> pd.DataFrame:
    """Return a DataFrame that exercises float64, int, object, and NaN columns."""
    return pd.DataFrame(
        {
            "frame": [0, 1, 2, 3],                         # int column
            "distance": [1.0, float("nan"), 3.0, 4.5],     # float64 with NaN
            "all_nan": [float("nan")] * 4,                  # all-NaN float column
            "label": ["a", "bb", "ccc", "dddd"],            # object/string column
            "corners_class0": [                             # object column (stringified data)
                "[[1,2],[3,4]]",
                "[[5,6],[7,8]]",
                "[[9,10],[11,12]]",
                "[[13,14],[15,16]]",
            ],
        }
    )


# ---------------------------------------------------------------------------
# Import guard — tests fail if module doesn't exist yet (expected on first run)
# ---------------------------------------------------------------------------

from fbpipe.utils.tables import (  # noqa: E402
    read_table,
    read_schema_columns,
    resolve_existing,
    table_path,
    write_table,
)


# ---------------------------------------------------------------------------
# table_path
# ---------------------------------------------------------------------------

class TestTablePath:
    def test_swaps_csv_to_parquet(self, tmp_path):
        p = tmp_path / "data.csv"
        assert table_path(p) == tmp_path / "data.parquet"

    def test_keeps_parquet_unchanged(self, tmp_path):
        p = tmp_path / "data.parquet"
        assert table_path(p) == p

    def test_swaps_arbitrary_suffix(self, tmp_path):
        p = tmp_path / "data.txt"
        assert table_path(p) == tmp_path / "data.parquet"


# ---------------------------------------------------------------------------
# resolve_existing
# ---------------------------------------------------------------------------

class TestResolveExisting:
    def test_returns_none_when_nothing_exists(self, tmp_path):
        assert resolve_existing(tmp_path / "ghost.csv") is None

    def test_prefers_parquet_over_csv(self, tmp_path):
        csv = tmp_path / "data.csv"
        pq = tmp_path / "data.parquet"
        csv.write_text("a,b\n1,2\n")
        pq.write_bytes(b"")  # content doesn't matter for this test
        # actual parquet bytes needed; write a real one
        pq.unlink()
        df = pd.DataFrame({"a": [1], "b": [2]})
        df.to_parquet(pq, engine="pyarrow", index=False)
        assert resolve_existing(tmp_path / "data.csv") == pq
        assert resolve_existing(tmp_path / "data.parquet") == pq
        assert resolve_existing(tmp_path / "data.txt") == pq

    def test_falls_back_to_csv_when_no_parquet(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("a,b\n1,2\n")
        assert resolve_existing(tmp_path / "data.parquet") == csv

    def test_returns_parquet_when_only_parquet_exists(self, tmp_path):
        pq = tmp_path / "data.parquet"
        df = pd.DataFrame({"x": [9]})
        df.to_parquet(pq, engine="pyarrow", index=False)
        assert resolve_existing(tmp_path / "data.csv") == pq


# ---------------------------------------------------------------------------
# write_table
# ---------------------------------------------------------------------------

class TestWriteTable:
    def test_always_writes_parquet(self, tmp_path):
        df = _make_rich_df()
        out = write_table(df, tmp_path / "result.csv")  # .csv path given
        assert out.suffix == ".parquet"
        assert out.exists()

    def test_returns_path_written(self, tmp_path):
        df = pd.DataFrame({"x": [1, 2]})
        out = write_table(df, tmp_path / "data.parquet")
        assert isinstance(out, Path)
        assert out == tmp_path / "data.parquet"

    def test_no_index_column(self, tmp_path):
        df = _make_rich_df()
        out = write_table(df, tmp_path / "data.parquet")
        back = pd.read_parquet(out, engine="pyarrow")
        assert list(back.columns) == list(df.columns), (
            f"Extra/missing columns: got {list(back.columns)}, expected {list(df.columns)}"
        )
        assert "__index_level_0__" not in back.columns

    def test_creates_parent_dirs(self, tmp_path):
        df = pd.DataFrame({"v": [42]})
        nested = tmp_path / "a" / "b" / "c" / "data.parquet"
        out = write_table(df, nested)
        assert out.exists()

    def test_normalizes_csv_path_to_parquet(self, tmp_path):
        df = pd.DataFrame({"v": [1]})
        csv_path = tmp_path / "data.csv"
        out = write_table(df, csv_path)
        assert out == tmp_path / "data.parquet"
        assert not csv_path.exists()  # should NOT have written a CSV


# ---------------------------------------------------------------------------
# read_table
# ---------------------------------------------------------------------------

class TestReadTable:
    def test_reads_parquet(self, tmp_path):
        df = _make_rich_df()
        pq = tmp_path / "data.parquet"
        df.to_parquet(pq, engine="pyarrow", index=False)
        back = read_table(pq)
        pd.testing.assert_frame_equal(back, df)

    def test_reads_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)
        back = read_table(csv)
        pd.testing.assert_frame_equal(back, df)

    def test_resolves_to_parquet_when_csv_requested_but_parquet_exists(self, tmp_path):
        df = _make_rich_df()
        pq = tmp_path / "data.parquet"
        df.to_parquet(pq, engine="pyarrow", index=False)
        # Request .csv path — should auto-resolve to the parquet
        back = read_table(tmp_path / "data.csv")
        pd.testing.assert_frame_equal(back, df)

    def test_reads_csv_when_no_parquet(self, tmp_path):
        df = pd.DataFrame({"x": [10, 20]})
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)
        back = read_table(csv)
        pd.testing.assert_frame_equal(back, df)

    def test_raises_fnf_when_neither_exists(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_table(tmp_path / "ghost.parquet")

    def test_selective_columns_parquet(self, tmp_path):
        df = _make_rich_df()
        pq = tmp_path / "data.parquet"
        df.to_parquet(pq, engine="pyarrow", index=False)
        back = read_table(pq, columns=["frame", "distance"])
        assert list(back.columns) == ["frame", "distance"]

    def test_csv_kwargs_forwarded(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("a;b\n1;2\n3;4\n")
        back = read_table(csv, sep=";")
        assert list(back.columns) == ["a", "b"]


# ---------------------------------------------------------------------------
# Round-trip correctness
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_float64_exact_including_nan(self, tmp_path):
        df = pd.DataFrame({"v": [1.1, float("nan"), -9.9, 0.0]})
        out = write_table(df, tmp_path / "floats.parquet")
        back = read_table(out)
        v_orig = df["v"].to_numpy(dtype=float)
        v_back = back["v"].to_numpy(dtype=float)
        assert np.array_equal(v_orig, v_back, equal_nan=True), (
            f"Float64 values differ: orig={v_orig} back={v_back}"
        )

    def test_int_columns_stay_int(self, tmp_path):
        df = pd.DataFrame({"i": pd.array([10, 20, 30], dtype="int64")})
        out = write_table(df, tmp_path / "ints.parquet")
        back = read_table(out)
        assert back["i"].dtype == np.dtype("int64"), (
            f"Expected int64, got {back['i'].dtype}"
        )
        np.testing.assert_array_equal(back["i"].to_numpy(), [10, 20, 30])

    def test_object_string_columns_unchanged(self, tmp_path):
        df = pd.DataFrame({"s": ["hello", "world", "foo"]})
        out = write_table(df, tmp_path / "strings.parquet")
        back = read_table(out)
        assert list(back["s"]) == ["hello", "world", "foo"]

    def test_corners_class0_roundtrip(self, tmp_path):
        """Object column holding stringified coordinate data must round-trip exactly."""
        values = ["[[1,2],[3,4]]", "[[5,6],[7,8]]", None]
        df = pd.DataFrame({"corners_class0": values})
        out = write_table(df, tmp_path / "corners.parquet")
        back = read_table(out)
        for orig, got in zip(values, back["corners_class0"]):
            if orig is None:
                assert pd.isna(got)
            else:
                assert got == orig

    def test_empty_dataframe_roundtrip(self, tmp_path):
        df = pd.DataFrame({"a": pd.Series([], dtype="float64"), "b": pd.Series([], dtype="object")})
        out = write_table(df, tmp_path / "empty.parquet")
        back = read_table(out)
        assert list(back.columns) == ["a", "b"]
        assert len(back) == 0

    def test_all_nan_float_column(self, tmp_path):
        df = pd.DataFrame({"v": [float("nan"), float("nan"), float("nan")]})
        out = write_table(df, tmp_path / "allnan.parquet")
        back = read_table(out)
        assert all(math.isnan(x) for x in back["v"])

    def test_columns_exact_no_extra(self, tmp_path):
        """No __index_level_0__ or other phantom columns after write+read."""
        df = _make_rich_df()
        out = write_table(df, tmp_path / "rich.parquet")
        back = read_table(out)
        assert list(back.columns) == list(df.columns)


# ---------------------------------------------------------------------------
# read_schema_columns
# ---------------------------------------------------------------------------

class TestReadSchemaColumns:
    def test_parquet_names_without_loading_data(self, tmp_path):
        df = pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0], "z": ["a", "b"]})
        pq = tmp_path / "schema.parquet"
        df.to_parquet(pq, engine="pyarrow", index=False)
        cols = read_schema_columns(pq)
        assert cols == ["x", "y", "z"]

    def test_csv_names_with_nrows0(self, tmp_path):
        csv = tmp_path / "schema.csv"
        csv.write_text("a,b,c\n1,2,3\n4,5,6\n")
        cols = read_schema_columns(csv)
        assert cols == ["a", "b", "c"]

    def test_returns_list_of_str(self, tmp_path):
        df = pd.DataFrame({"col1": [1], "col2": [2]})
        pq = tmp_path / "cols.parquet"
        df.to_parquet(pq, engine="pyarrow", index=False)
        result = read_schema_columns(pq)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)


# ---------------------------------------------------------------------------
# iter_fly_distance_csvs — extended to Parquet
# ---------------------------------------------------------------------------

class TestIterFlyDistanceCsvs:
    """These tests import the extended iter_fly_distance_csvs from fly_files."""

    def _write_distances(self, path: Path) -> None:
        """Write a minimal valid parquet or csv distances file."""
        df = pd.DataFrame({"frame": [0, 1], "distance_0_1": [10.0, 20.0]})
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            df.to_parquet(path, engine="pyarrow", index=False)
        else:
            df.to_csv(path, index=False)

    def test_finds_parquet_distance_file(self, tmp_path):
        from fbpipe.utils.fly_files import iter_fly_distance_csvs

        pq = tmp_path / "trial_fly1_distances.parquet"
        self._write_distances(pq)

        results = list(iter_fly_distance_csvs(tmp_path, recursive=False))
        assert len(results) == 1
        path, token, slot_idx = results[0]
        assert path == pq
        assert token == "fly1_distances"
        assert slot_idx == 1

    def test_prefers_parquet_over_csv_same_stem(self, tmp_path):
        from fbpipe.utils.fly_files import iter_fly_distance_csvs

        csv_file = tmp_path / "trial_fly2_distances.csv"
        pq_file = tmp_path / "trial_fly2_distances.parquet"
        self._write_distances(csv_file)
        self._write_distances(pq_file)

        results = list(iter_fly_distance_csvs(tmp_path, recursive=False))
        assert len(results) == 1
        path, token, slot_idx = results[0]
        assert path == pq_file
        assert slot_idx == 2

    def test_yields_csv_when_no_parquet(self, tmp_path):
        from fbpipe.utils.fly_files import iter_fly_distance_csvs

        csv_file = tmp_path / "trial_fly3_distances.csv"
        self._write_distances(csv_file)

        results = list(iter_fly_distance_csvs(tmp_path, recursive=False))
        assert len(results) == 1
        path, token, slot_idx = results[0]
        assert path == csv_file
        assert slot_idx == 3

    def test_yields_both_different_fly_indices(self, tmp_path):
        from fbpipe.utils.fly_files import iter_fly_distance_csvs

        pq1 = tmp_path / "trial_fly1_distances.parquet"
        pq2 = tmp_path / "trial_fly2_distances.parquet"
        self._write_distances(pq1)
        self._write_distances(pq2)

        results = list(iter_fly_distance_csvs(tmp_path, recursive=False))
        assert len(results) == 2
        indices = {r[2] for r in results}
        assert indices == {1, 2}

    def test_recursive_finds_parquet_in_subdir(self, tmp_path):
        from fbpipe.utils.fly_files import iter_fly_distance_csvs

        sub = tmp_path / "RMS_calculations"
        pq = sub / "session_fly4_distances.parquet"
        self._write_distances(pq)

        results = list(iter_fly_distance_csvs(tmp_path, recursive=True))
        assert len(results) == 1
        assert results[0][0] == pq

    def test_existing_csv_detection_still_works(self, tmp_path):
        """Backward compat: existing CSV-only callers must still work."""
        from fbpipe.utils.fly_files import iter_fly_distance_csvs

        csv_file = tmp_path / "october_07_fly_1_testing_2_fly1_distances.csv"
        df = pd.DataFrame({"frame": [0, 1], "distance_0_1": [10.0, 20.0]})
        df.to_csv(csv_file, index=False)

        results = list(iter_fly_distance_csvs(tmp_path, recursive=False))
        assert len(results) == 1
        path, token, slot_idx = results[0]
        assert path == csv_file
        assert token == "fly1_distances"
        assert slot_idx == 1
