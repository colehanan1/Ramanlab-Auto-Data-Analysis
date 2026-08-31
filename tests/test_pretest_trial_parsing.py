"""The analysis layer must parse the ``pretest_N`` phase tag.

The pre-test protocol (config_v2_*_pretest.yaml) writes a naive odor panel as
``pretest_1..7``. Every trial-name regex in the pipeline previously matched only
``training|testing``, which would have silently dropped those recordings instead
of failing, so the parse is pinned here.
"""

import re

import pytest

from fbpipe.utils.trial_metadata import _CYCLE_RE
from fbpipe.steps.move_videos import TRIAL_DIR_RE


def parse_trial_dir_name(name):
    m = _CYCLE_RE.match(name)
    assert m is not None, f"{name!r} did not parse"
    return m.group("type").lower(), int(m.group("index")), m.group("odor")


def test_parse_trial_dir_name_handles_pretest():
    assert parse_trial_dir_name("pretest_3_Hexanol") == ("pretest", 3, "Hexanol")


def test_parse_trial_dir_name_still_handles_the_old_phases():
    assert parse_trial_dir_name("training_1_Hexanol") == ("training", 1, "Hexanol")
    assert parse_trial_dir_name("testing_7_ACV") == ("testing", 7, "ACV")


def test_pretest_is_its_own_phase_not_testing():
    """The whole point of the tag: a naive trial must never read as post-training."""
    phase, _, _ = parse_trial_dir_name("pretest_1_Citral")
    assert phase == "pretest"


def test_move_videos_matches_pretest_dirs():
    m = TRIAL_DIR_RE.match("october_01_fly1_pretest_2")
    assert m is not None
    assert m.group("phase").lower() == "pretest"
    assert m.group("idx") == "2"


@pytest.mark.parametrize("module,attr", [
    ("scripts.analysis.binarized_per_rasters", "_TRIAL_RE"),
    ("scripts.analysis.training_vs_learning", "_TRIAL_RE"),
    ("scripts.analysis.training_auc_vs_control_response", "_TRIAL_RE"),
])
def test_analysis_trial_regexes_accept_pretest(module, attr):
    import importlib
    mod = importlib.import_module(module)
    assert getattr(mod, attr).match("pretest_4_Linalool")
