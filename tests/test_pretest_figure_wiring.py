"""The two new sensitivity figure sets run inside the pipeline, not by hand.

CLAUDE.md records the failure mode these tests exist to prevent: hand-run
scripts whose PNGs sit beside pipeline-owned ones drift silently, because a
pipeline run rewrites ``model_predictions.csv`` underneath them and nobody
re-renders. ``naive_vs_trial_score_bars.py`` is already in that category.

So both new sets are wired in:
  * the pre-test vs post-training mean traces (``dataset_mean_traces`` gains a
    ``phase_cohorts`` list), and
  * the Pre-Test-vs-Trial-Score-Bars.
"""

from pathlib import Path

import pytest
import yaml

from scripts.pipeline.run_workflows import (
    _dataset_mean_traces_commands,
    _pretest_score_bars_command,
)

CONFIG = Path(__file__).resolve().parent.parent / "config" / "config_new.yaml"
SENSITIVITY = ["Hex-Sensitivity-24-0.1", "IAA-Sensitivity-24-1",
               "3Oct-Sensitivity-24-0.1"]


class _Settings:
    class _R:
        output_csv = "/x/model_predictions.csv"
        python = ""
    reaction_prediction = _R()
    flagged_flies_csv = "/x/flagged.csv"
    dataset_overrides = {}
    datasets = ()


def _mt_cfg(**over):
    base = {
        "wide_csv": "/x/testing.parquet",
        "out_root": "/x/dataset_mean_comparisons",
        "pretest_wide_csv": "/x/pretest.parquet",
        "phase_cohorts": list(SENSITIVITY),
    }
    base.update(over)
    return {"dataset_mean_traces": base}


def _cmds(cfg):
    return _dataset_mean_traces_commands(
        cfg, _Settings(), python_exec="python3", config_path=None
    )


# ── pre-vs-post mean traces ───────────────────────────────────────────────


def test_one_command_per_phase_cohort():
    cmds = [c for c in _cmds(_mt_cfg()) if "--pretest-wide-csv" in c]
    assert len(cmds) == len(SENSITIVITY)


def test_each_phase_command_names_its_cohort_as_the_train_dataset():
    cmds = [c for c in _cmds(_mt_cfg()) if "--pretest-wide-csv" in c]
    got = {c[c.index("--train-dataset") + 1] for c in cmds}
    assert got == set(SENSITIVITY)


def test_the_phase_command_carries_the_pretest_table():
    cmd = [c for c in _cmds(_mt_cfg()) if "--pretest-wide-csv" in c][0]
    assert cmd[cmd.index("--pretest-wide-csv") + 1] == "/x/pretest.parquet"


def test_the_phase_figures_go_to_their_own_subfolder():
    """They must not overwrite the trained-vs-control set for the same cohort."""
    cmd = [c for c in _cmds(_mt_cfg()) if "--pretest-wide-csv" in c][0]
    out = cmd[cmd.index("--out-dir") + 1]
    assert "Pre-Test_vs_Post-Training" in out


def test_no_phase_commands_without_the_pretest_table():
    cmds = _cmds(_mt_cfg(pretest_wide_csv=""))
    assert not [c for c in cmds if "--pretest-wide-csv" in c]


def test_no_phase_commands_when_none_are_listed():
    cmds = _cmds(_mt_cfg(phase_cohorts=[]))
    assert not [c for c in cmds if "--pretest-wide-csv" in c]


def test_the_flagged_table_reaches_the_phase_command():
    """Both arms are the same flies; a flagged fly must leave both."""
    cmd = [c for c in _cmds(_mt_cfg()) if "--pretest-wide-csv" in c][0]
    assert cmd[cmd.index("--flagged-flies-csv") + 1] == "/x/flagged.csv"


# ── pre-test score bars ───────────────────────────────────────────────────


def _sb_cfg(**over):
    base = {"enabled": True, "datasets": list(SENSITIVITY),
            "out_dir": "/x/Pre-Test-vs-Trial-Score-Bars"}
    base.update(over)
    return {"pretest_score_bars": base}


def test_the_score_bars_command_is_built():
    cmd = _pretest_score_bars_command(
        _sb_cfg(), _Settings(), python_exec="python3", config_path=None
    )
    assert cmd is not None
    assert cmd[1].endswith("naive_vs_trial_score_bars.py")


def test_the_score_bars_command_selects_the_pretest_baseline():
    """Without this flag it renders the naive-baselined cohorts instead."""
    cmd = _pretest_score_bars_command(
        _sb_cfg(), _Settings(), python_exec="python3", config_path=None
    )
    assert "--pretest-baseline" in cmd


def test_the_score_bars_command_lists_every_cohort():
    cmd = _pretest_score_bars_command(
        _sb_cfg(), _Settings(), python_exec="python3", config_path=None
    )
    got = [cmd[i + 1] for i, x in enumerate(cmd) if x == "--dataset"]
    assert sorted(got) == sorted(SENSITIVITY)


def test_no_score_bars_command_when_disabled():
    assert _pretest_score_bars_command(
        _sb_cfg(enabled=False), _Settings(), python_exec="python3", config_path=None
    ) is None


def test_no_score_bars_command_without_an_out_dir():
    assert _pretest_score_bars_command(
        _sb_cfg(out_dir=""), _Settings(), python_exec="python3", config_path=None
    ) is None


def test_no_score_bars_command_when_the_block_is_absent():
    assert _pretest_score_bars_command(
        {}, _Settings(), python_exec="python3", config_path=None
    ) is None


# ── config ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def analysis_cfg():
    if not CONFIG.exists():
        pytest.skip("config/ is gitignored")
    with CONFIG.open() as fh:
        return yaml.safe_load(fh)["analysis"]


def test_config_declares_the_phase_cohorts(analysis_cfg):
    listed = set(analysis_cfg["dataset_mean_traces"].get("phase_cohorts") or [])
    assert set(SENSITIVITY) <= listed


def test_config_gives_the_mean_traces_a_pretest_table(analysis_cfg):
    assert str(
        analysis_cfg["dataset_mean_traces"].get("pretest_wide_csv") or ""
    ).endswith("all_envelope_rows_wide_combined_base_pretest.parquet")


def test_config_declares_the_score_bars_step(analysis_cfg):
    assert "pretest_score_bars" in analysis_cfg


def test_the_score_bars_write_to_their_own_folder(analysis_cfg):
    out = analysis_cfg["pretest_score_bars"]["out_dir"]
    assert out.endswith("Pre-Test-vs-Trial-Score-Bars")
