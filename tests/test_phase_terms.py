"""Phase wording by cohort protocol.

The *-Sensitivity-* cohorts are pre-/post-exposure with a sensitized
odor; light-conditioned cohorts with a pre-test (Hex-HexTest-Training-*) keep
pre-test / post-training / CS+.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.analysis import phase_terms as pt


@pytest.mark.parametrize("name", [
    "IAA-Sensitivity-1-1", "IAA-Sensitivity5s-1-1", "3Oct-Sensitivity-24-0.1",
    "Hex-Sensitivity-24-0.1", "IAA-Sensitivity-1-1_Oct09b2-vs-Oct14b1", "IAA sensitivity",
])
def test_sensitivity_cohorts_are_recognised(name):
    assert pt.is_sensitivity(name)


@pytest.mark.parametrize("name", [
    "Hex-HexTest-Training-32-1", "EB-Control-24-1", "3Oct-Training-24-0.1",
    "RandomPanel-24-1", "", None,
])
def test_other_cohorts_are_not(name):
    assert not pt.is_sensitivity(name)


def test_sensitivity_wording():
    c = "IAA-Sensitivity-1-1"
    assert pt.pre_label(c, "Pre-test (naive)") == "Pre-exposure"
    assert pt.post_label(c, "Post-training") == "Post-exposure"
    assert pt.pre_label(c, "pre-test", case="lower") == "pre-exposure"
    assert pt.post_label(c, "Post-Training", case="title") == "Post-Exposure"
    assert pt.block_label(c) == "exposure"
    assert pt.odor_tag(c) == "sensitized odor"
    assert pt.odor_tag(c, capital=True) == "Sensitized odor"


def test_conditioning_cohorts_keep_their_own_wording():
    c = "Hex-HexTest-Training-32-1"
    assert pt.pre_label(c, "Pre-test (naive)") == "Pre-test (naive)"
    assert pt.post_label(c, "post-training", case="lower") == "post-training"
    assert pt.odor_tag(c) == "CS+"
    assert pt.odor_tag(c, "trained odor", capital=True) == "trained odor"
    assert pt.block_label(c) == "training"
    assert pt.block_label(c, "conditioning") == "conditioning"


def test_all_sensitivity_needs_every_cohort():
    assert pt.all_sensitivity(["IAA-Sensitivity-1-1", "IAA-Sensitivity-6-1"])
    assert not pt.all_sensitivity(["IAA-Sensitivity-1-1", "Hex-HexTest-Training-24-1"])
    assert not pt.all_sensitivity([])


def test_unknown_case_is_rejected():
    with pytest.raises(ValueError):
        pt.pre_label("IAA-Sensitivity-1-1", "Pre-test", case="upper")
