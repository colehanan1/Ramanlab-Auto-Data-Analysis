"""Tests for the shared analysis helpers (read_wide_table, baseline_correct)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fbpipe.analysis.traces import baseline_correct, read_wide_table
from fbpipe.utils.tables import write_table


def _wide(n=4):
    return pd.DataFrame(
        {"fly": ["a"] * n, "dataset": ["d"] * n,
         "dir_val_0": np.arange(n, dtype=float), "dir_val_1": np.arange(n, dtype=float) + 10}
    )


def test_read_wide_prefers_parquet(tmp_path):
    df = _wide()
    csv = tmp_path / "wide.csv"
    df.to_csv(csv, index=False)
    write_table(df, csv, replace_legacy_csv=False)  # also create wide.parquet
    back = read_wide_table(csv)  # given the .csv path, must resolve to parquet
    pd.testing.assert_frame_equal(back.reset_index(drop=True), df)


def test_read_wide_csv_fallback(tmp_path):
    df = _wide()
    csv = tmp_path / "wide.csv"
    df.to_csv(csv, index=False)
    back = read_wide_table(csv)
    pd.testing.assert_frame_equal(back.reset_index(drop=True), df)


def test_read_wide_column_projection(tmp_path):
    df = _wide()
    write_table(df, tmp_path / "wide.parquet")
    back = read_wide_table(tmp_path / "wide.parquet", columns=["dir_val_0", "dir_val_1"])
    assert list(back.columns) == ["dir_val_0", "dir_val_1"]


def test_read_wide_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_wide_table(tmp_path / "nope.csv")


def test_baseline_correct_subtracts_pre_odor_mean():
    trace = np.array([2.0, 4.0, 10.0, 20.0])
    out = baseline_correct(trace, 2)  # baseline mean = (2+4)/2 = 3
    assert np.allclose(out, trace - 3.0)


def test_baseline_correct_noops_on_nonpositive_or_none():
    trace = np.array([1.0, 2.0, 3.0])
    assert np.allclose(baseline_correct(trace, 0), trace)
    assert np.allclose(baseline_correct(trace, None), trace)


def test_baseline_correct_ignores_nan_in_window():
    trace = np.array([np.nan, 4.0, 8.0])
    out = baseline_correct(trace, 2)  # finite baseline mean = 4
    assert np.allclose(out, trace - 4.0, equal_nan=True)


def test_baseline_correct_all_nan_window_noop():
    trace = np.array([np.nan, np.nan, 5.0])
    out = baseline_correct(trace, 2)
    assert np.allclose(out, trace, equal_nan=True)
