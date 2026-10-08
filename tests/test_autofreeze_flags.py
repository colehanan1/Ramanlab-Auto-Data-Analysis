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
