"""Every pipeline figure stage must respect ``freeze.figures``.

A dataset frozen for figures keeps its rows in the wide CSV (that is
``freeze.data``'s job) but must not be RE-RENDERED: its published figures are
final, and a re-render silently restyles them with whatever the current palette,
threshold rule or label map happens to be.

The rule matches ``envelope_visuals.should_skip_frozen_figure``: a figure drawn
from several datasets is dropped only when EVERY contributor is frozen, so
adding flies to a live arm still redraws its comparison against a frozen one.
``--thaw``/``--thaw-all`` lift the freeze for one run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.pipeline.run_workflows as rw


class _Ov:
    def __init__(self, data=False, figures=False):
        self.freeze_data = data
        self.freeze_figures = figures


class _ReactionPrediction:
    python = ""
    output_csv = "/tmp/preds/model_predictions.csv"


class _Settings:
    """Minimal stand-in: only the attributes the builders/freeze_flags read."""

    def __init__(self, frozen=(), thawed=(), thaw_all=False, datasets=()):
        self.reaction_prediction = _ReactionPrediction()
        self.dataset_overrides = {n: _Ov(True, True) for n in frozen}
        self._thawed = thawed
        self._thaw_all = thaw_all
        self.datasets = list(datasets)
        self.flagged_flies_csv = ""
        self.protocol = "v2"


CONFIG = Path("/tmp/config_new.yaml")


def flag_values(cmd, flag):
    return [cmd[i + 1] for i, tok in enumerate(cmd) if tok == flag]


def all_flag_values(cmds, flag):
    return {v for cmd in cmds for v in flag_values(cmd, flag)}


# --- _figure_frozen_datasets ------------------------------------------


def test_figure_frozen_datasets_ignores_data_only_freeze():
    s = _Settings()
    s.dataset_overrides = {"A": _Ov(data=True, figures=False),
                           "B": _Ov(data=True, figures=True)}
    assert rw._figure_frozen_datasets(s) == ["B"]


def test_figure_frozen_datasets_honors_thaw():
    s = _Settings(frozen=("A", "B"), thawed=("A",))
    assert rw._figure_frozen_datasets(s) == ["B"]
    assert rw._figure_frozen_datasets(_Settings(frozen=("A",), thaw_all=True)) == []


def test_figure_frozen_datasets_no_settings():
    assert rw._figure_frozen_datasets(None) == []


# --- cohort_figures (binarized / graded rasters, training AUC) --------


def _cohort_cfg(**over):
    block = {
        "enabled": True,
        "datasets": ["EB-Control-24-1", "Hex-Control-24-0.01"],
        "trained_raster_datasets": ["EB-Training-24-1", "Hex-Training-24-0.01"],
        "training_wide_csv": "/tmp/csv/wide_training.parquet",
        "testing_wide_csv": "/tmp/csv/wide.parquet",
        "binarized_out_dir": "/tmp/fig/Binarized",
        "graded_out_dir": "/tmp/fig/Graded",
        "training_auc_out_dir": "/tmp/fig/AUC",
    }
    block.update(over)
    return {"cohort_figures": block}


def test_cohort_figures_drop_frozen_datasets():
    cmds = rw._cohort_figure_commands(
        _cohort_cfg(), _Settings(frozen=("Hex-Control-24-0.01", "Hex-Training-24-0.01")),
        python_exec="/usr/bin/python3", config_path=CONFIG,
    )
    names = all_flag_values(cmds, "--dataset")
    assert names == {"EB-Control-24-1", "EB-Training-24-1"}


def test_cohort_figures_emit_nothing_when_all_frozen():
    frozen = ("EB-Control-24-1", "Hex-Control-24-0.01",
              "EB-Training-24-1", "Hex-Training-24-0.01")
    assert rw._cohort_figure_commands(
        _cohort_cfg(), _Settings(frozen=frozen),
        python_exec="/usr/bin/python3", config_path=CONFIG,
    ) == []


def test_cohort_figures_thaw_restores_a_frozen_dataset():
    cmds = rw._cohort_figure_commands(
        _cohort_cfg(),
        _Settings(frozen=("Hex-Control-24-0.01",), thawed=("Hex-Control-24-0.01",)),
        python_exec="/usr/bin/python3", config_path=CONFIG,
    )
    assert "Hex-Control-24-0.01" in all_flag_values(cmds, "--dataset")


def test_cohort_figures_live_datasets_unaffected():
    live = rw._cohort_figure_commands(
        _cohort_cfg(), _Settings(), python_exec="/usr/bin/python3", config_path=CONFIG,
    )
    assert all_flag_values(live, "--dataset") == {
        "EB-Control-24-1", "Hex-Control-24-0.01",
        "EB-Training-24-1", "Hex-Training-24-0.01",
    }


# --- dataset_mean_traces (trained vs control mean traces) -------------


def _mt_cfg(**over):
    block = {
        "enabled": True,
        "wide_csv": "/tmp/csv/wide.csv",
        "out_root": "/tmp/fig/means",
        "cohorts": [
            {"train_dataset": "EB-Training-24-1", "control_dataset": "EB-Control-24-1"},
            {"train_dataset": "Hex-Training-24-0.01",
             "control_dataset": "Hex-Control-24-0.01"},
        ],
    }
    block.update(over)
    return {"dataset_mean_traces": block}


def test_mean_traces_drop_pair_frozen_on_both_arms():
    cmds = rw._dataset_mean_traces_commands(
        _mt_cfg(), _Settings(frozen=("Hex-Training-24-0.01", "Hex-Control-24-0.01")),
        python_exec="/usr/bin/python3", config_path=CONFIG,
    )
    assert all_flag_values(cmds, "--train-dataset") == {"EB-Training-24-1"}


def test_mean_traces_keep_pair_when_one_arm_is_live():
    """A frozen control against a live training arm must still redraw."""
    cmds = rw._dataset_mean_traces_commands(
        _mt_cfg(), _Settings(frozen=("Hex-Control-24-0.01",)),
        python_exec="/usr/bin/python3", config_path=CONFIG,
    )
    assert "Hex-Training-24-0.01" in all_flag_values(cmds, "--train-dataset")


def test_mean_traces_conc_series_dropped_when_every_member_frozen():
    cfg = _mt_cfg(cohorts=[], conc_series=[
        {"datasets": {"RandomPanel-24-1": 1.0, "RandomPanel-24-0.1": 0.1},
         "out_dir": "conc"},
    ])
    frozen = ("RandomPanel-24-1", "RandomPanel-24-0.1")
    assert rw._dataset_mean_traces_commands(
        cfg, _Settings(frozen=frozen),
        python_exec="/usr/bin/python3", config_path=CONFIG,
    ) == []
    # One live member keeps the series -- the figure is the comparison.
    kept = rw._dataset_mean_traces_commands(
        cfg, _Settings(frozen=("RandomPanel-24-1",)),
        python_exec="/usr/bin/python3", config_path=CONFIG,
    )
    assert len(kept) == 1


# --- publication figures (pubfig_score_train_vs_control) --------------


class _Cohort:
    def __init__(self, train_dataset, metrics=("mean-score",)):
        self.train_dataset = train_dataset
        self.metrics = list(metrics)
        self.out_stem = ""
        self.fly_months = ()
        self.genotype = ""
        self.odor_remap = ()
        self.label = ""


class _Pub:
    def __init__(self, cohorts):
        self.cohorts = cohorts
        self.figures_dir = "/tmp/fig/pub"


def _pub_settings(frozen=(), cohorts=("EB-Training-24-1", "Hex-Training-24-0.01")):
    s = _Settings(frozen=frozen)
    s.reaction_prediction = _ReactionPrediction()
    s.reaction_prediction.publication_figures = _Pub([_Cohort(c) for c in cohorts])
    return s


def test_pubfig_commands_skip_frozen_cohorts():
    cmds = rw._pubfig_commands(
        _pub_settings(frozen=("Hex-Training-24-0.01",)),
        python_exec="/usr/bin/python3",
        csv_path=Path("/tmp/preds/model_predictions.csv"),
        config_path=CONFIG,
    )
    assert all_flag_values(cmds, "--train-dataset") == {"EB-Training-24-1"}


def test_pubfig_commands_thaw_all_renders_everything():
    s = _pub_settings(frozen=("EB-Training-24-1", "Hex-Training-24-0.01"))
    s._thaw_all = True
    cmds = rw._pubfig_commands(
        s, python_exec="/usr/bin/python3",
        csv_path=Path("/tmp/preds/model_predictions.csv"), config_path=CONFIG,
    )
    assert len(all_flag_values(cmds, "--train-dataset")) == 2
