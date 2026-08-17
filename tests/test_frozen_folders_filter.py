"""Frozen rows stay in the CSV and are dropped on the way into a figure.

The filter lives at the two shared readers rather than in each figure script:

* ``fbpipe.analysis.traces.read_wide_table`` — dataset_means, dataset_means_specific_flies,
  dataset_mean_traces_tvc, avg_training_traces_dataset, training_vs_learning;
* ``envelope_visuals._load_wide_table`` — the pipeline's Raw Testing / Raw
  Training PER trace figures (``wide_input``).

Filtering by DEFAULT is the whole point: a new figure script that forgets to
opt in still excludes the retired flies. ``include_frozen=True`` is the escape
hatch for anything that genuinely wants the complete record.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fbpipe.analysis.traces import read_wide_table
from fbpipe.utils.frozen_folders import drop_frozen


def _wide(tmp_path, name="wide.csv"):
    path = tmp_path / name
    pd.DataFrame(
        {
            "dataset": ["Hex-Training-24-0.1"] * 3,
            "fly": ["may_22_batch_1", "june_01_batch_1", "august_10_batch_1"],
            "fly_number": ["1", "1", "1"],
            "trial_type": ["testing"] * 3,
            "frozen": [True, True, False],
            "dir_val_0": [1.0, 2.0, 3.0],
            "dir_val_1": [4.0, 5.0, 6.0],
        }
    ).to_csv(path, index=False)
    return path


# ── drop_frozen ───────────────────────────────────────────────────────────


def test_drop_frozen_removes_frozen_rows():
    df = pd.DataFrame({"fly": ["a", "b"], "frozen": [True, False]})
    assert list(drop_frozen(df)["fly"]) == ["b"]


def test_drop_frozen_keeps_everything_when_asked():
    df = pd.DataFrame({"fly": ["a", "b"], "frozen": [True, False]})
    assert list(drop_frozen(df, include_frozen=True)["fly"]) == ["a", "b"]


def test_drop_frozen_passes_through_a_table_with_no_frozen_column():
    """Legacy-protocol tables and older CSVs have no such column; they must
    load unchanged rather than raise."""
    df = pd.DataFrame({"fly": ["a", "b"]})
    assert list(drop_frozen(df)["fly"]) == ["a", "b"]


def test_drop_frozen_treats_nan_as_live():
    """A cached slice predating the column reindexes to NaN. NaN means 'not
    marked', which must keep the row -- dropping it would silently delete a
    whole data-frozen dataset from the figures."""
    df = pd.DataFrame({"fly": ["a", "b"], "frozen": [np.nan, False]})
    assert list(drop_frozen(df)["fly"]) == ["a", "b"]


def test_drop_frozen_reads_string_booleans():
    """Round-tripping through CSV can yield the strings 'True'/'False'."""
    df = pd.DataFrame({"fly": ["a", "b"], "frozen": ["True", "False"]})
    assert list(drop_frozen(df)["fly"]) == ["b"]


def test_drop_frozen_does_not_mutate_the_caller_s_frame():
    df = pd.DataFrame({"fly": ["a", "b"], "frozen": [True, False]})
    drop_frozen(df)
    assert len(df) == 2


# ── read_wide_table ───────────────────────────────────────────────────────


def test_read_wide_table_drops_frozen_rows_by_default(tmp_path):
    df = read_wide_table(_wide(tmp_path))
    assert list(df["fly"]) == ["august_10_batch_1"]


def test_read_wide_table_can_include_frozen_rows(tmp_path):
    df = read_wide_table(_wide(tmp_path), include_frozen=True)
    assert len(df) == 3


def test_read_wide_table_filters_even_when_projecting_columns(tmp_path):
    """Column projection must not defeat the filter: a caller asking only for
    ``fly`` + ``dir_val_0`` would otherwise read no ``frozen`` column and get
    the retired flies straight back into its figure."""
    df = read_wide_table(_wide(tmp_path), columns=["fly", "dir_val_0"])
    assert list(df["fly"]) == ["august_10_batch_1"]


def test_projection_does_not_leak_the_frozen_column(tmp_path):
    """The filter reads `frozen` internally; the caller asked for two columns
    and must get exactly two, or downstream column-count logic shifts."""
    df = read_wide_table(_wide(tmp_path), columns=["fly", "dir_val_0"])
    assert list(df.columns) == ["fly", "dir_val_0"]


def test_read_wide_table_prefers_parquet_and_still_filters(tmp_path):
    csv = _wide(tmp_path)
    pd.read_csv(csv).to_parquet(csv.with_suffix(".parquet"), index=False)
    df = read_wide_table(csv)
    assert list(df["fly"]) == ["august_10_batch_1"]


# ── the figure-facing loader ──────────────────────────────────────────────


def test_envelope_visuals_wide_loader_drops_frozen_rows(tmp_path):
    import scripts.analysis.envelope_visuals as ev

    df, env_cols = ev._load_wide_table(_wide(tmp_path))
    assert list(df["fly"]) == ["august_10_batch_1"]
    assert env_cols == ["dir_val_0", "dir_val_1"]
