"""get_push_folder with experiment.cohort.

A pre-test cohort's no-light arm is named without a mode segment
(IAA-Sensitivity5s-1-1, Hex-HexTest-24-1): that protocol is its own condition.
A cohort with NO pre-test (config_v2_A_10s.yaml, cohort Odor10s) is a plain
trained/control pair, so its control arm must say Control -- the figure code
pairs X-Training-N with X-Control-N and treats any name without "control" as
light-paired.
"""
import sys
from pathlib import Path

import pytest

PICODE = Path(__file__).resolve().parent.parent / "PiCode"
sys.path.insert(0, str(PICODE))
from experiment_scheduler import get_push_folder  # noqa: E402


def _name(mode, **kw):
    return get_push_folder("OFM_A", mode, "/b", starvation_hours=1,
                           odor_vial_conc="1", **kw).rstrip("/").rsplit("/", 1)[-1]


def test_plain_cohort_control_says_control():
    assert _name("Control", cohort="Odor10s") == "IAA-Odor10s-Control-1-1"


def test_plain_cohort_training():
    assert _name("Training", cohort="Odor10s") == "IAA-Odor10s-Training-1-1"


@pytest.mark.parametrize("mode,expected", [
    ("Control", "IAA-Sensitivity5s-1-1"),
    ("Training", "IAA-Sensitivity5s-Training-1-1"),
])
def test_pretest_cohort_unchanged(mode, expected):
    assert _name(mode, cohort="Sensitivity5s", sensitivity=True) == expected


@pytest.mark.parametrize("mode,expected", [
    ("Control", "IAA-Sensitivity-1-1"),
    ("Training", "IAA-Sensitivity-Training-1-1"),
])
def test_sensitivity_without_cohort_unchanged(mode, expected):
    assert _name(mode, sensitivity=True) == expected


def test_no_cohort_unchanged():
    assert _name("Control") == "IAA-Control-1-1"
