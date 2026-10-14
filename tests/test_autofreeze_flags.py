"""freeze_flags / figure_frozen_from_raw consult the active auto-freeze plan.

In-process (run_workflows sets it) and in figure subprocesses
(``FBPIPE_FREEZE_PLAN``), the plan answers for every dataset it covers, so no
figure script needs to know about auto-freeze. A dataset the plan does not
cover, and any run without a plan (hand-run scripts, ``--figures-only``,
auto-freeze disabled), keeps the explicit ``freeze:`` flags. ``--thaw`` still
wins over both.
"""

from __future__ import annotations

import json

import pytest

from fbpipe import autofreeze as af
from fbpipe.config import DatasetOverride
from fbpipe.freeze import figure_frozen_from_raw, freeze_flags


class _Cfg:
    def __init__(self, overrides):
        self.dataset_overrides = overrides


@pytest.fixture
def plan_file(tmp_path, monkeypatch):
    af.set_active_plan(None)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "schema": af.PLAN_SCHEMA, "run_id": "r", "config": "c",
        "datasets": {
            "3Oct-Training-24-0.1": {"data_frozen": True, "figures_frozen": True, "decision": "frozen"},
            "IAA-Sensitivity-24-1": {"data_frozen": True, "figures_frozen": True, "decision": "frozen"},
            "EB-Control-24-1": {"data_frozen": False, "figures_frozen": False, "decision": "thawed"},
        },
    }))
    monkeypatch.setenv(af.PLAN_ENV, str(path))
    yield path
    af.set_active_plan(None)


CFG = _Cfg({
    "EB-Control-24-1": DatasetOverride(freeze_data=True, freeze_figures=True),
    "Unplanned-24-1": DatasetOverride(freeze_data=True, freeze_figures=True),
})
RAW = {"dataset_overrides": {
    "EB-Control-24-1": {"freeze": {"data": True, "figures": True}},
    "Unplanned-24-1": {"freeze": {"data": True, "figures": True}},
}}


def test_the_plan_answers_for_the_datasets_it_covers(plan_file):
    assert freeze_flags(CFG, "EB-Control-24-1") == (False, False)  # thawed tonight
    assert freeze_flags(CFG, "IAA-Sensitivity-24-1") == (True, True)  # no block needed
    # Canonical spelling (3Oct -> 3OCT) and the -flagged split resolve too.
    assert freeze_flags(CFG, "3OCT-Training-24-0.1") == (True, True)
    assert freeze_flags(CFG, "3Oct-Training-24-0.1-flagged") == (True, True)


def test_a_dataset_the_plan_does_not_cover_keeps_its_flags(plan_file):
    assert freeze_flags(CFG, "Unplanned-24-1") == (True, True)
    got = figure_frozen_from_raw(RAW)
    assert {"3Oct-Training-24-0.1", "3OCT-Training-24-0.1", "IAA-Sensitivity-24-1", "Unplanned-24-1"} <= got
    assert "EB-Control-24-1" not in got


def test_thaw_wins_over_the_plan(plan_file):
    assert freeze_flags(CFG, "IAA-Sensitivity-24-1", thawed=["IAA-Sensitivity-24-1"]) == (False, False)
    assert "IAA-Sensitivity-24-1" not in figure_frozen_from_raw(RAW, thawed=["IAA-Sensitivity-24-1"])
    assert figure_frozen_from_raw(RAW, thaw_all=True) == set()


def test_without_a_plan_the_flags_answer(monkeypatch):
    af.set_active_plan(None)
    monkeypatch.delenv(af.PLAN_ENV, raising=False)
    assert freeze_flags(CFG, "EB-Control-24-1") == (True, True)
    assert freeze_flags(CFG, "IAA-Sensitivity-24-1") == (False, False)
    assert figure_frozen_from_raw(RAW) == {"EB-Control-24-1", "Unplanned-24-1"}


def test_an_unreadable_plan_file_falls_back_to_the_flags(tmp_path, monkeypatch):
    af.set_active_plan(None)
    bad = tmp_path / "plan.json"
    bad.write_text("{not json")
    monkeypatch.setenv(af.PLAN_ENV, str(bad))
    assert freeze_flags(CFG, "EB-Control-24-1") == (True, True)


# --- run_workflows figure helpers under a plan --------------------------------


def _pub_settings(tmp_path):
    import yaml

    from fbpipe.config import load_settings

    cfg = {
        "model_path": "/tmp/model.pt",
        "main_directories": "/tmp/data",
        "datasets": ["Hex-Training-24-0.1", "Hex-Control-24-0.1", "IAA-Sensitivity-24-1"],
        "reaction_prediction": {"publication_figures": {"cohorts": [
            {"train_dataset": "Hex-Training-24-0.1", "metrics": ["mean-score"]},
        ]}},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    settings = load_settings(path)
    settings._thawed, settings._thaw_all = (), False
    return settings


def _plan(tmp_path, monkeypatch, decisions):
    af.set_active_plan(None)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"schema": af.PLAN_SCHEMA, "run_id": "r", "config": "c", "datasets": {
        ds: {"data_frozen": f, "figures_frozen": f, "decision": "frozen" if f else "thawed"}
        for ds, f in decisions.items()
    }}))
    monkeypatch.setenv(af.PLAN_ENV, str(path))


@pytest.mark.parametrize(
    "train_frozen,control_frozen,drawn",
    [(True, True, False), (True, False, True), (False, True, True), (False, False, True)],
)
def test_a_pubfig_redraws_when_either_auto_frozen_arm_thaws(tmp_path, monkeypatch, train_frozen, control_frozen, drawn):
    from scripts.pipeline import run_workflows as rw

    settings = _pub_settings(tmp_path)
    _plan(tmp_path, monkeypatch, {"Hex-Training-24-0.1": train_frozen, "Hex-Control-24-0.1": control_frozen})
    cmds = rw._pubfig_commands(settings, python_exec="python3", csv_path=tmp_path / "p.csv", config_path=None)
    assert bool(cmds) is drawn
    af.set_active_plan(None)


def test_figure_frozen_datasets_covers_datasets_without_a_freeze_block(tmp_path, monkeypatch):
    from scripts.pipeline import run_workflows as rw

    settings = _pub_settings(tmp_path)
    _plan(tmp_path, monkeypatch, {"IAA-Sensitivity-24-1": True, "Hex-Training-24-0.1": False})
    assert "IAA-Sensitivity-24-1" in rw._figure_frozen_datasets(settings)
    assert "Hex-Training-24-0.1" not in rw._figure_frozen_datasets(settings)
    af.set_active_plan(None)
