"""The pre-vs-post figures run as part of the pipeline, not by hand.

Hand-run scripts whose PNGs sit beside pipeline-owned ones drift silently — a
run rewrites model_predictions.csv underneath them and nobody re-renders. This
step is wired into run_workflows so the figures track the data.

Follows the ``naive_vs_trained`` shape exactly: a pure command builder that
returns None when unconfigured, plus an mtime-and-command cache key.
"""

from pathlib import Path

import pytest
import yaml

from scripts.pipeline.run_workflows import (
    _pretest_vs_test_command,
    _pretest_vs_test_expected,
)

CONFIG = Path(__file__).resolve().parent.parent / "config" / "config_new.yaml"
OUT_DIR = (
    "/home/ramanlab/Documents/cole/Results/New-Opto-Fly-Figures/Pre-vs-Post-Sensitivity"
)
PREDICTIONS = "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/model_predictions.csv"


class _Reaction:
    def __init__(self, output_csv=PREDICTIONS, python=""):
        self.output_csv = output_csv
        self.python = python


class _Settings:
    def __init__(self, **kw):
        self.reaction_prediction = _Reaction(**kw)


def _cfg(**over):
    base = {"enabled": True, "out_dir": OUT_DIR}
    base.update(over)
    return {"pretest_vs_test": base}


# ── the command builder ───────────────────────────────────────────────────


def test_no_command_when_the_block_is_absent():
    assert _pretest_vs_test_command(
        {}, _Settings(), python_exec="python3", config_path=None
    ) is None


def test_no_command_when_the_block_is_disabled():
    assert _pretest_vs_test_command(
        _cfg(enabled=False), _Settings(), python_exec="python3", config_path=None
    ) is None


def test_no_command_without_an_out_dir():
    assert _pretest_vs_test_command(
        _cfg(out_dir=""), _Settings(), python_exec="python3", config_path=None
    ) is None


def test_no_command_without_a_predictions_csv():
    assert _pretest_vs_test_command(
        _cfg(), _Settings(output_csv=""), python_exec="python3", config_path=None
    ) is None


def test_the_command_targets_the_comparison_script():
    cmd = _pretest_vs_test_command(
        _cfg(), _Settings(), python_exec="python3", config_path=None
    )
    assert cmd[0] == "python3"
    assert cmd[1].endswith("scripts/analysis/pretest_vs_test_comparison.py")


def test_the_command_passes_the_predictions_and_out_root():
    cmd = _pretest_vs_test_command(
        _cfg(), _Settings(), python_exec="python3", config_path=None
    )
    assert cmd[cmd.index("--predictions-csv") + 1] == PREDICTIONS
    assert cmd[cmd.index("--out-root") + 1] == OUT_DIR


def test_the_predictions_csv_can_be_overridden_in_the_block():
    cmd = _pretest_vs_test_command(
        _cfg(predictions_csv="/x/other.csv"), _Settings(),
        python_exec="python3", config_path=None,
    )
    assert cmd[cmd.index("--predictions-csv") + 1] == "/x/other.csv"


def test_cohorts_are_forwarded_when_listed():
    cmd = _pretest_vs_test_command(
        _cfg(cohorts=["Hex-Sensitivity-24-0.1", "IAA-Sensitivity-24-1"]),
        _Settings(), python_exec="python3", config_path=None,
    )
    i = cmd.index("--cohorts")
    assert cmd[i + 1:i + 3] == ["Hex-Sensitivity-24-0.1", "IAA-Sensitivity-24-1"]


def test_no_cohorts_flag_when_none_are_listed():
    """Omitted means every cohort with paired trials, which is the sane default."""
    cmd = _pretest_vs_test_command(
        _cfg(), _Settings(), python_exec="python3", config_path=None
    )
    assert "--cohorts" not in cmd


def test_the_binary_threshold_is_forwarded():
    cmd = _pretest_vs_test_command(
        _cfg(binary_threshold=3), _Settings(), python_exec="python3", config_path=None
    )
    assert cmd[cmd.index("--binary-threshold") + 1] == "3"


# ── the cache key ─────────────────────────────────────────────────────────


def test_the_cache_key_tracks_the_command():
    a = _pretest_vs_test_expected(_cfg(), _Settings(), config_path=None)
    b = _pretest_vs_test_expected(
        _cfg(cohorts=["Hex-Sensitivity-24-0.1"]), _Settings(), config_path=None
    )
    assert a["command"] != b["command"]


def test_the_cache_key_is_stable_for_an_unchanged_config():
    a = _pretest_vs_test_expected(_cfg(), _Settings(), config_path=None)
    b = _pretest_vs_test_expected(_cfg(), _Settings(), config_path=None)
    assert a == b


def test_the_cache_key_carries_the_predictions_mtime():
    key = _pretest_vs_test_expected(_cfg(), _Settings(), config_path=None)
    assert "predictions_mtime" in key


def test_an_unconfigured_block_still_yields_a_key():
    """_should_skip must be able to compare; it must not raise."""
    key = _pretest_vs_test_expected({}, _Settings(), config_path=None)
    assert key["command"] == ""


# ── config ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def analysis_cfg():
    if not CONFIG.exists():
        pytest.skip(f"{CONFIG} not present (config/ is gitignored)")
    with CONFIG.open() as fh:
        return yaml.safe_load(fh)["analysis"]


def test_config_declares_the_step(analysis_cfg):
    assert "pretest_vs_test" in analysis_cfg


def test_config_points_at_the_new_opto_figures_root(analysis_cfg):
    out_dir = analysis_cfg["pretest_vs_test"]["out_dir"]
    assert out_dir.startswith(
        "/home/ramanlab/Documents/cole/Results/New-Opto-Fly-Figures"
    )


def test_config_builds_a_real_command(analysis_cfg):
    cmd = _pretest_vs_test_command(
        analysis_cfg, _Settings(), python_exec="python3", config_path=None
    )
    assert cmd is not None
    assert "--out-root" in cmd


# ── flagged flies ─────────────────────────────────────────────────────────


class _SettingsFlagged(_Settings):
    def __init__(self, flagged="/x/flagged-flys-truth.csv", **kw):
        super().__init__(**kw)
        self.flagged_flies_csv = flagged


def test_the_flagged_flies_table_is_forwarded_from_settings():
    """Every other figure step passes this; the new one must too, or it keeps
    flies the rest of the pipeline drops."""
    cmd = _pretest_vs_test_command(
        _cfg(), _SettingsFlagged(), python_exec="python3", config_path=None
    )
    assert cmd[cmd.index("--flagged-flies-csv") + 1] == "/x/flagged-flys-truth.csv"


def test_the_block_can_override_the_flagged_table():
    cmd = _pretest_vs_test_command(
        _cfg(flagged_flies_csv="/x/other.csv"), _SettingsFlagged(),
        python_exec="python3", config_path=None,
    )
    assert cmd[cmd.index("--flagged-flies-csv") + 1] == "/x/other.csv"


def test_no_flag_when_settings_have_no_flagged_table():
    cmd = _pretest_vs_test_command(
        _cfg(), _SettingsFlagged(flagged=""), python_exec="python3", config_path=None
    )
    assert "--flagged-flies-csv" not in cmd


def test_the_cache_key_changes_when_the_flagged_table_changes():
    """Re-flagging a fly must invalidate the cache, or the figures go stale."""
    a = _pretest_vs_test_expected(_cfg(), _SettingsFlagged(), config_path=None)
    b = _pretest_vs_test_expected(
        _cfg(flagged_flies_csv="/x/other.csv"), _SettingsFlagged(), config_path=None
    )
    assert a["command"] != b["command"]
