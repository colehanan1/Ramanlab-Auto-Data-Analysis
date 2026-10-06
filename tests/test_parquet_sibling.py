"""Reading a wide CSV through its Parquet sibling, exactly as read_csv would.

build_wide_csv writes each wide table as CSV and then as a Parquet sibling
made from ``pd.read_csv`` of that CSV. Two nightly readers still parsed the
1.3 GB CSV (20-25 s) instead of the sibling (<1 s): the light-only traces and
predict_reactions. ``read_csv_via_parquet`` serves the sibling only when it is
at least as new as the CSV, and hands back the frame ``pd.read_csv`` would --
same dtypes, same values, and NaN (never None) for a missing value in an
object column, which is the one thing a Parquet round trip changes
(``astype(str)`` turns NaN into "nan" but None into "None").
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fbpipe.utils.parquet_sibling import read_csv_via_parquet
from fbpipe.utils.tables import write_table

CSV_TEXT = (
    "dataset,fly,fly_number,fly_type,trial_type,trial_label,frozen,"
    "tracking_flagged,maybe_flag,trace_len,trial_light_on_s,global_min,"
    "dir_val_0,dir_val_1,dir_val_2\n"
    "A,day_1,1,GR5a-Old,testing,testing_1_hexanol,False,False,True,3,,0.5,"
    "0.1,0.2,0.30000000000000004\n"
    "A,day_1,2,,testing,testing_2_citral,False,True,,3,30.0,,"
    "1e-17,,\n"
    "B,day_2,1,Unknown,pretest,pretest_1_hexanol,True,False,False,2,,"
    "-0.0,5.5,6.25,\n"
)


def _write(tmp_path: Path, *, sibling: bool = True) -> Path:
    csv = tmp_path / "wide.csv"
    csv.write_text(CSV_TEXT)
    if sibling:
        # Exactly how build_wide_csv writes it (envelope_combined.py).
        write_table(pd.read_csv(csv), csv, replace_legacy_csv=False)
        parquet = csv.with_suffix(".parquet")
        st = csv.stat()
        os.utime(parquet, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    return csv


def _cells(df: pd.DataFrame) -> list[list[object]]:
    """Every cell, with missing values spelled by their Python type."""
    out = []
    for name in df.columns:
        col = []
        for value in df[name].tolist():
            if value is None:
                col.append(("None",))
            elif isinstance(value, float) and math.isnan(value):
                col.append(("nan",))
            else:
                col.append((type(value).__name__, value))
        out.append(col)
    return out


def test_frame_matches_read_csv_exactly(tmp_path):
    csv = _write(tmp_path)
    expected = pd.read_csv(csv)
    got = read_csv_via_parquet(csv)
    pd.testing.assert_frame_equal(got, expected)
    assert list(got.dtypes.astype(str)) == list(expected.dtypes.astype(str))
    assert _cells(got) == _cells(expected)


def test_missing_object_values_are_nan_not_none(tmp_path):
    csv = _write(tmp_path)
    got = read_csv_via_parquet(csv)
    for name in got.columns[got.dtypes == object]:
        assert not any(v is None for v in got[name].tolist()), name
    # The cases that motivate it -- a missing string, a bool column with a gap
    # -- must stringify exactly as read_csv's frame does ("nan", not "None",
    # under pandas 2; NaN kept under pandas 3's str dtype).
    expected = pd.read_csv(csv)
    for name in ("fly_type", "maybe_flag"):
        assert _cells(got[[name]].astype(str)) == _cells(expected[[name]].astype(str))
    assert "None" not in got["maybe_flag"].astype(str).tolist()


def test_the_sibling_is_what_gets_read(tmp_path, monkeypatch):
    csv = _write(tmp_path)

    def no_csv(*_a, **_k):
        raise AssertionError("parsed the CSV although a fresh sibling exists")

    monkeypatch.setattr(pd, "read_csv", no_csv)
    assert len(read_csv_via_parquet(csv)) == 3


def test_no_sibling_reads_the_csv(tmp_path):
    csv = _write(tmp_path, sibling=False)
    pd.testing.assert_frame_equal(read_csv_via_parquet(csv), pd.read_csv(csv))


def test_a_sibling_older_than_the_csv_is_ignored(tmp_path, capsys):
    """A failed sidecar write leaves last night's Parquet beside tonight's
    CSV (build_wide_csv only warns); serving it would score stale rows."""
    csv = _write(tmp_path)
    parquet = csv.with_suffix(".parquet")
    # Tonight's CSV gains a row; the sibling stays as it was, and older.
    csv.write_text(CSV_TEXT + "C,day_3,1,GR5a-Old,testing,testing_1_hexanol,"
                   "False,False,True,1,,0.0,0.0,,\n")
    st = csv.stat()
    os.utime(parquet, ns=(st.st_atime_ns, st.st_mtime_ns - 1_000_000_000))
    got = read_csv_via_parquet(csv)
    assert len(got) == 4
    pd.testing.assert_frame_equal(got, pd.read_csv(csv))
    assert "older than" in capsys.readouterr().out


def test_an_unreadable_sibling_falls_back_to_the_csv(tmp_path, capsys):
    csv = _write(tmp_path)
    parquet = csv.with_suffix(".parquet")
    parquet.write_bytes(b"not a parquet file")
    st = csv.stat()
    os.utime(parquet, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    pd.testing.assert_frame_equal(read_csv_via_parquet(csv), pd.read_csv(csv))
    assert "unreadable" in capsys.readouterr().out


def test_a_missing_csv_still_raises(tmp_path):
    """Same failure as pd.read_csv: a sibling never stands in for an absent CSV."""
    csv = _write(tmp_path)
    csv.unlink()
    with pytest.raises(FileNotFoundError):
        read_csv_via_parquet(csv)


def test_a_parquet_path_is_read_as_a_table(tmp_path):
    csv = _write(tmp_path)
    got = read_csv_via_parquet(csv.with_suffix(".parquet"))
    assert list(got.columns) == list(pd.read_csv(csv).columns)


# --------------------------------------------------------------------------
# The two readers
# --------------------------------------------------------------------------


def test_predict_reads_the_same_frame_as_before(tmp_path):
    from fbpipe.steps.predict_reactions import _read_input_tables

    main = _write(tmp_path)
    pre_dir = tmp_path / "pre"
    pre_dir.mkdir()
    pretest = _write(pre_dir)
    expected = pd.concat(
        [pd.read_csv(main), pd.read_csv(pretest)], ignore_index=True, sort=False
    )
    got = _read_input_tables(main, [pretest])
    pd.testing.assert_frame_equal(got, expected)
    assert _cells(got) == _cells(expected)
    # The temp CSV the CLI re-reads is therefore byte-identical too.
    assert got.to_csv(index=False) == expected.to_csv(index=False)


def test_light_only_reads_the_same_frame_as_before(tmp_path):
    from fbpipe.utils.frozen_folders import drop_frozen
    from scripts.analysis.light_trial_traces import _read_wide_for_light

    csv = _write(tmp_path)
    expected = drop_frozen(pd.read_csv(csv))
    got = _read_wide_for_light(csv)
    pd.testing.assert_frame_equal(got, expected)
    assert len(got) == 2  # the frozen row stays out, as before
    assert _cells(got) == _cells(expected)
