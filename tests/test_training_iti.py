#!/usr/bin/env python3
"""``experiment.inter_trial_baseline``: the gap between training trials.

Training and testing carry separate inter-trial gaps. The training gap used to be
the ``_DEFAULTS`` value with no override path at all, so a protocol wanting
back-to-back conditioning trials (0 s ITI) could not ask for one without editing
the default and dragging every other protocol along with it.

These tests pin the two halves apart: the training gap is overridable per config,
the testing gap is not touched by that override.
"""

import sys
from pathlib import Path

import pytest
import yaml

PICODE = Path(__file__).resolve().parent.parent / "PiCode"
sys.path.insert(0, str(PICODE))

from expand_config import (  # noqa: E402
    _DEFAULTS,
    apply_sensitivity_overrides,
    expand_config,
)

TRAINING_CYCLES = (1, 6)          # standard protocol / pre-test protocol
TESTING_CYCLES = (2, 5, 7)


def _cfg(**experiment):
    exp = {"trained_odor": "OFM_H"}
    exp.update(experiment)
    return {"format": "v2", "experiment": exp}


def _cycles(cfg):
    return {c["cycle"]: c for c in expand_config(cfg)["cycles"]}


def _training_gap(cycle):
    """The ITI a training cycle spends between repeats.

    The training cycle is one trial with ``repeat`` applied, so the ITI is the
    trailing step -- identified by position, not by duration: the trial's own
    opening baseline is also a "Baseline Period".
    """
    last = cycle["steps"][-1]
    if last["name"] == "Stop Recording":
        return 0
    assert last["name"] == "Baseline Period", last
    return last["duration"]


def _testing_gaps(cycle):
    """Every inter-trial gap inside a testing cycle (gaps are inline there)."""
    gaps = []
    for prev, step in zip(cycle["steps"], cycle["steps"][1:]):
        if prev["name"] == "Stop Recording" and step["name"] == "Baseline Period":
            gaps.append(step["duration"])
    return gaps


# ── current behaviour, unchanged ──────────────────────────────────────

def test_training_gap_defaults_to_the_four_minute_baseline():
    assert _training_gap(_cycles(_cfg())[1]) == _DEFAULTS["inter_trial_baseline"]
    assert _DEFAULTS["inter_trial_baseline"] == 240


def test_testing_gap_defaults_to_the_four_minute_baseline():
    gaps = _testing_gaps(_cycles(_cfg())[2])
    assert gaps, "testing cycle has no inter-trial gaps"
    assert set(gaps) == {_DEFAULTS["testing_inter_trial_baseline"]}


# ── the override ──────────────────────────────────────────────────────

def test_zero_training_iti_leaves_no_gap_step_at_all():
    cycle = _cycles(_cfg(inter_trial_baseline=0))[1]
    assert _training_gap(cycle) == 0
    assert cycle["steps"][-1]["name"] == "Stop Recording"
    assert not any(
        s["name"] == "Baseline Period" and s["duration"] == 240
        for s in cycle["steps"]
    )


def test_zero_training_iti_keeps_the_trial_shape():
    """Only the ITI goes: 30 s baseline / 30 s odor / 30 s post are untouched."""
    cycle = _cycles(_cfg(inter_trial_baseline=0))[1]
    assert [s["duration"] for s in cycle["steps"]] == [0, 30, 30, 30, 0]
    assert cycle["repeat"] == _DEFAULTS["classical_trials"]


def test_zero_training_iti_does_not_touch_the_testing_gap():
    cycles = _cycles(_cfg(inter_trial_baseline=0))
    assert _training_gap(cycles[1]) == 0
    assert set(_testing_gaps(cycles[2])) == {240}


@pytest.mark.parametrize("gap", [0, 30, 120, 600])
def test_training_iti_is_honoured_verbatim(gap):
    """Not a 0-or-default branch -- any value passes through."""
    assert _training_gap(_cycles(_cfg(inter_trial_baseline=gap))[1]) == gap


def test_override_cannot_leak_into_the_testing_gap():
    """The two gaps are separate keys, and only the training one is reachable.

    ``testing_inter_trial_baseline`` is read straight from ``_DEFAULTS`` -- no
    config path -- so the testing spacing is 240 s whatever the config says. The
    two keys are given different values here so a single shared lookup would show
    up as a failure rather than passing by coincidence.
    """
    cycles = _cycles(_cfg(inter_trial_baseline=0, testing_inter_trial_baseline=99))
    assert _training_gap(cycles[1]) == 0
    assert set(_testing_gaps(cycles[2])) == {240}


# ── pre-test ("sensitivity") protocol ─────────────────────────────────

def test_pretest_protocol_training_gap_is_overridable():
    cycles = _cycles(_cfg(pretest=True, inter_trial_baseline=0))
    assert sorted(cycles) == [5, 6, 7, 8]
    assert _training_gap(cycles[6]) == 0


def test_pretest_protocol_keeps_both_testing_panels_spaced():
    cycles = _cycles(_cfg(pretest=True, inter_trial_baseline=0))
    assert set(_testing_gaps(cycles[5])) == {240}
    assert set(_testing_gaps(cycles[7])) == {240}


def test_pretest_rest_period_is_unaffected_by_the_training_iti():
    """The pre-test -> training rest is delay_after + the panel's trailing gap."""
    cycles = _cycles(_cfg(pretest=True, inter_trial_baseline=0))
    assert cycles[5]["delay_after"] == _DEFAULTS["pretest_wait"]
    assert cycles[5]["steps"][-1] == {"name": "Baseline Period", "duration": 240}


# ── --sensitivity flag ────────────────────────────────────────────────

def test_sensitivity_flag_zeroes_the_training_iti():
    exp = apply_sensitivity_overrides(_cfg())["experiment"]
    assert exp["inter_trial_baseline"] == 0


def test_sensitivity_flag_keeps_an_explicit_config_value():
    cfg = _cfg(inter_trial_baseline=240)
    assert apply_sensitivity_overrides(cfg)["experiment"]["inter_trial_baseline"] == 240


def test_sensitivity_flag_does_not_mutate_the_caller_config():
    cfg = _cfg()
    apply_sensitivity_overrides(cfg)
    assert "inter_trial_baseline" not in cfg["experiment"]


# ── shipped configs ───────────────────────────────────────────────────

@pytest.mark.parametrize("path", sorted(PICODE.glob("config_v2_*_pretest.yaml")))
def test_pretest_configs_train_back_to_back(path):
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cycles = _cycles(cfg)
    assert _training_gap(cycles[6]) == 0, f"{path.name} still spaces training trials"
    assert set(_testing_gaps(cycles[7])) == {240}, f"{path.name} lost its testing gap"


@pytest.mark.parametrize(
    "path",
    sorted(set(PICODE.glob("config_v2_?.yaml")) | set(PICODE.glob("config_v2_*_norest.yaml"))),
)
def test_non_pretest_configs_keep_the_spaced_training(path):
    """Only the pre-test (sensitivity) protocol goes back-to-back."""
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    assert _training_gap(_cycles(cfg)[1]) == 240, f"{path.name} lost its training gap"
