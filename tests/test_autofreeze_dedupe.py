"""Lossless removal of byte-identical duplicate rows from a frozen slice.

15 of the frozen datasets' cached slices carried each trial twice (or, for
RandomPanel-24-0.1, 850 rows plus a copy of the last 649). Adoption removes an
exact duplicate -- a row equal to an EARLIER row in every column, NaN equal to
NaN -- and nothing else: a row sharing a trial key but differing anywhere stays,
and the first copy of every row stays, in order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fbpipe import freeze as _freeze


def _rows(n_unique=5, dup_tail=2, index=None):
    df = pd.DataFrame({
        "dataset": ["DS"] * n_unique,
        "fly": [f"b{i // 2}" for i in range(n_unique)],
        "fly_number": list(range(1, n_unique + 1)),
        "trial_type": ["testing"] * n_unique,
        "trial_label": ["testing_1_x"] * n_unique,
        "dir_val_0": [0.1 * i if i % 3 else np.nan for i in range(n_unique)],
    })
    out = pd.concat([df, df.tail(dup_tail)], ignore_index=True)  # RandomPanel shape
    if index is not None:
        out.index = index(len(out))
    return out


@pytest.mark.parametrize("index", [None, lambda n: pd.Index(range(100, 100 + n)), lambda n: pd.Index([7] * n)],
                         ids=["range", "offset", "repeated labels"])
def test_partial_duplicates_are_found_exactly(index):
    """Only the copied tail is a duplicate; the unique head is not touched --
    including when the index is not a plain RangeIndex."""
    rows = _rows(index=index)
    dup = _freeze._exact_duplicates(rows)
    assert dup.dtype == bool
    assert dup.tolist() == [False] * 5 + [True] * 2
    pd.testing.assert_frame_equal(rows.loc[~dup.to_numpy()].reset_index(drop=True),
                                  rows.drop_duplicates().reset_index(drop=True))


def test_same_key_different_values_is_not_a_duplicate():
    rows = _rows(dup_tail=0)
    clash = rows.iloc[[0]].assign(dir_val_0=9.9)
    both = pd.concat([rows, clash], ignore_index=True)
    assert not _freeze._exact_duplicates(both).any()


def test_nan_equals_nan():
    rows = _rows(n_unique=3, dup_tail=3)
    assert rows["dir_val_0"].isna().any()
    assert _freeze._exact_duplicates(rows).sum() == 3


def test_dedupe_slice_is_lossless_and_keeps_the_fingerprint(tmp_path):
    rows = _rows()
    fp = {"protocol": "v2"}
    _freeze.save_slice(tmp_path, "combined_base", "DS", rows, fp)
    assert _freeze.count_duplicate_rows(tmp_path, "combined_base", "DS") == (7, 5)
    assert _freeze.dedupe_slice(tmp_path, "combined_base", "DS") == (7, 5)
    kept = pd.read_parquet(_freeze.slice_dir(tmp_path, "combined_base", "DS") / "rows.parquet")
    pd.testing.assert_frame_equal(kept, rows.head(5))
    assert _freeze.slice_status(tmp_path, "combined_base", "DS", fp) == (True, "")
    assert _freeze.dedupe_slice(tmp_path, "combined_base", "DS") == (5, 5)  # idempotent
