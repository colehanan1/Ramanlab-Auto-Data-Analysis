"""The window-rescored figure sets run as pipeline steps.

``pretest_vs_test_plus_after``: rescore pre/post trials on odor + N s, then
render the standard pre-vs-post set from that CSV into its own folder.
``score_window_comparison``: odor vs after vs both, for the listed cohorts.
Model, predictions CSV and binary threshold come from reaction_prediction;
the AUC threshold rule from the config file itself.
"""

from pathlib import Path

import pytest
import yaml

from scripts.pipeline.run_workflows import (
    _pretest_vs_test_plus_after_commands,
    _score_window_comparison_command,
    _window_steps_expected,
)

CONFIG = Path(__file__).resolve().parent.parent / "config" / "config_new.yaml"
PRE = "/x/pretest.parquet"
POST = "/x/testing.parquet"
CFGP = Path("/x/config_new.yaml")


class _Reaction:
    output_csv = "/x/model_predictions.csv"
    model_path = "/x/model.json"
    binary_threshold = 2


class _Settings:
    reaction_prediction = _Reaction()
    flagged_flies_csv = "/x/flagged.csv"


def _plus(**over):
    block = {"out_dir": "/x/out", "rescored_csv": "/x/rescored.csv",
             "pretest_wide_csv": PRE, "testing_wide_csv": POST, "after_s": 5}
    block.update(over)
    return {"pretest_vs_test_plus_after": block}


def _cmp(**over):
    block = {"out_dir": "/x/cmp", "cohorts": ["IAA-Sensitivity5s-1-1"],
             "pretest_wide_csv": PRE, "testing_wide_csv": POST, "after_s": 5}
    block.update(over)
    return {"score_window_comparison": block}


def _arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


# ── +N s pre-vs-post set ──────────────────────────────────────────────────


def test_absent_or_disabled_block_builds_nothing():
    for cfg in ({}, _plus(enabled=False)):
        assert _pretest_vs_test_plus_after_commands(
            cfg, _Settings(), python_exec="py", config_path=CFGP) is None


@pytest.mark.parametrize("key", ["out_dir", "rescored_csv", "pretest_wide_csv", "testing_wide_csv"])
def test_a_missing_path_builds_nothing(key):
    assert _pretest_vs_test_plus_after_commands(
        _plus(**{key: ""}), _Settings(), python_exec="py", config_path=CFGP) is None


def test_rescore_then_render():
    rescore, render = _pretest_vs_test_plus_after_commands(
        _plus(), _Settings(), python_exec="py", config_path=CFGP)
    assert rescore[1].endswith("scripts/analysis/window_rescore.py")
    assert _arg(rescore, "--window") == "odor_plus_after"
    assert _arg(rescore, "--after-s") == "5.0"
    assert _arg(rescore, "--model-path") == "/x/model.json"
    assert _arg(rescore, "--config") == str(CFGP)
    assert _arg(rescore, "--output-csv") == "/x/rescored.csv"
    assert "--paired-only" in rescore
    assert rescore[rescore.index("--wide-csv") + 1:rescore.index("--wide-csv") + 3] == [PRE, POST]

    assert render[1].endswith("scripts/analysis/pretest_vs_test_comparison.py")
    assert _arg(render, "--predictions-csv") == "/x/rescored.csv"
    assert _arg(render, "--out-root") == "/x/out"
    assert _arg(render, "--window-note") == "scored on odor + 5 s after"
    assert _arg(render, "--flagged-flies-csv") == "/x/flagged.csv"


def test_no_config_path_builds_nothing():
    """The AUC rule lives in the config file; without it scores would drift."""
    assert _pretest_vs_test_plus_after_commands(
        _plus(), _Settings(), python_exec="py", config_path=None) is None


# ── window comparison ─────────────────────────────────────────────────────


def test_comparison_command():
    cmd = _score_window_comparison_command(_cmp(), _Settings(), python_exec="py",
                                           config_path=CFGP)
    assert cmd[1].endswith("scripts/analysis/score_window_comparison.py")
    assert cmd[cmd.index("--cohorts") + 1:] [:1] == ["IAA-Sensitivity5s-1-1"]
    assert _arg(cmd, "--predictions-csv") == "/x/model_predictions.csv"
    assert _arg(cmd, "--out-dir") == "/x/cmp"
    assert _arg(cmd, "--binary-threshold") == "2"


def test_comparison_needs_cohorts():
    assert _score_window_comparison_command(
        _cmp(cohorts=[]), _Settings(), python_exec="py", config_path=CFGP) is None


# ── cache key ─────────────────────────────────────────────────────────────


def test_cache_key_tracks_inputs_and_commands():
    a = _window_steps_expected(["a b"], _Settings(), [PRE, POST], CFGP)
    b = _window_steps_expected(["a c"], _Settings(), [PRE, POST], CFGP)
    assert a != b
    assert {"predictions_mtime", "model_mtime", "config_mtime", "wide_mtimes"} <= set(a)


# ── live config ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def analysis_cfg():
    if not CONFIG.exists():
        pytest.skip("config/ is gitignored")
    return yaml.safe_load(CONFIG.read_text())["analysis"]


@pytest.mark.parametrize("block", ["pretest_vs_test_plus_after", "score_window_comparison"])
def test_live_config_writes_under_new_opto(analysis_cfg, block):
    if block not in analysis_cfg:
        pytest.skip(f"{block} not in config_new.yaml yet")
    assert analysis_cfg[block]["out_dir"].startswith(
        "/home/ramanlab/Documents/cole/Results/New-Opto-Fly-Figures/")
