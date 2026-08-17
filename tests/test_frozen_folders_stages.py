"""The two remaining figure-facing entry points exclude frozen folders.

* ``light_trial_traces`` reads the wide CSV directly for the Light-Only PER
  figures;
* ``predict_reactions`` reads it to build ``model_predictions.csv``, which is
  the hub every score/matrix/publication figure downstream reads. Filtering
  there is what makes those figures clean without editing each one.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from fbpipe.steps.predict_reactions import _drop_frozen_rows


def _wide_frame():
    return pd.DataFrame(
        {
            "dataset": ["Hex-Training-24-0.1"] * 3,
            "fly": ["may_22_batch_1", "june_01_batch_1", "august_10_batch_1"],
            "fly_number": ["1", "1", "1"],
            "trial_type": ["testing"] * 3,
            "trial_label": ["testing_1"] * 3,
            "frozen": [True, True, False],
            "dir_val_0": [1.0, 2.0, 3.0],
        }
    )


# ── predict_reactions ─────────────────────────────────────────────────────


def test_predict_reactions_drops_frozen_rows():
    kept, n_dropped = _drop_frozen_rows(_wide_frame())
    assert list(kept["fly"]) == ["august_10_batch_1"]
    assert n_dropped == 2


def test_predict_reactions_keeps_everything_when_told_to():
    kept, n_dropped = _drop_frozen_rows(_wide_frame(), include_frozen=True)
    assert len(kept) == 3
    assert n_dropped == 0


def test_predict_reactions_handles_a_table_with_no_frozen_column():
    df = _wide_frame().drop(columns=["frozen"])
    kept, n_dropped = _drop_frozen_rows(df)
    assert len(kept) == 3
    assert n_dropped == 0


# ── the stale-CSV warning ─────────────────────────────────────────────────


class _Ov:
    freeze_data = False
    freeze_figures = False

    def __init__(self, folders=()):
        self.freeze_folders = tuple(folders)


class _Cfg:
    def __init__(self, *, before=None, overrides=None):
        self.freeze_folders_before = before
        self.dataset_overrides = dict(overrides or {})


def test_warns_when_rules_exist_but_the_table_predates_the_column(capsys):
    """``--figures-only`` reuses the wide CSV on disk. If that CSV was built
    before folder freeze existed, nothing is marked and every retired fly
    silently returns to the figures. Say so rather than quietly under-filter."""
    from fbpipe.utils.frozen_folders import warn_if_unmarked

    df = _wide_frame().drop(columns=["frozen"])
    warn_if_unmarked(df, _Cfg(before=date(2026, 6, 26)), source="wide.csv")
    out = capsys.readouterr().out
    assert "wide.csv" in out and "frozen" in out


def test_no_warning_when_the_column_is_present(capsys):
    from fbpipe.utils.frozen_folders import warn_if_unmarked

    warn_if_unmarked(_wide_frame(), _Cfg(before=date(2026, 6, 26)), source="wide.csv")
    assert capsys.readouterr().out == ""


def test_no_warning_when_no_folder_rules_are_configured(capsys):
    """Nothing is frozen, so an unmarked table is simply correct."""
    from fbpipe.utils.frozen_folders import warn_if_unmarked

    df = _wide_frame().drop(columns=["frozen"])
    warn_if_unmarked(df, _Cfg(), source="wide.csv")
    assert capsys.readouterr().out == ""


def test_warns_when_only_a_per_dataset_rule_is_configured(capsys):
    from fbpipe.utils.frozen_folders import FolderRule, warn_if_unmarked

    df = _wide_frame().drop(columns=["frozen"])
    cfg = _Cfg(overrides={"Hex-Training-24-0.1": _Ov([FolderRule("*_rig_3")])})
    warn_if_unmarked(df, cfg, source="wide.csv")
    assert "frozen" in capsys.readouterr().out


# ── light_trial_traces ────────────────────────────────────────────────────


def test_light_trial_traces_excludes_frozen_flies(tmp_path):
    """The Light-Only figures are per-fly, so a frozen folder must not get one."""
    import scripts.analysis.light_trial_traces as lt

    csv = tmp_path / "wide.csv"
    frame = _wide_frame()
    frame["trial_label"] = ["testing_light_1"] * 3
    frame.to_csv(csv, index=False)

    df = lt._read_wide_for_light(csv)
    assert list(df["fly"]) == ["august_10_batch_1"]
