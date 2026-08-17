"""The naive-vs-trained sweep must run on every pipeline run.

These figures read ``model_predictions.csv``, so they go stale the moment the
model re-scores -- exactly the failure the mean-trace step was wired in to fix.
``analysis.naive_vs_trained`` names the output tree and the step runs after
reactions, once the predictions CSV exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fbpipe.config import load_settings  # noqa: E402
from scripts.pipeline import run_workflows as rw  # noqa: E402

CONFIG_PATH = ROOT / "config" / "config_new.yaml"


class _Reaction:
    def __init__(self, output_csv="/preds.csv"):
        self.output_csv = output_csv
        self.python = ""


class _Settings:
    def __init__(self, **kw):
        self.reaction_prediction = _Reaction(kw.get("output_csv", "/preds.csv"))
        self.flagged_flies_csv = kw.get("flagged_flies_csv", "")


def _block(**kw):
    block = {"out_dir": "/figs/Naive"}
    block.update(kw)
    return {"naive_vs_trained": block}


def _cmd(analysis_cfg, settings=None):
    return rw._naive_vs_trained_command(
        analysis_cfg,
        settings or _Settings(),
        python_exec="/py",
        config_path=Path("/cfg.yaml"),
    )


def _value(cmd, flag):
    return cmd[cmd.index(flag) + 1]


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


def test_no_block_means_no_command():
    """An unconfigured pipeline must not start rendering figures."""
    assert _cmd({}) is None
    assert _cmd({"naive_vs_trained": {}}) is None


def test_enabled_false_turns_the_step_off():
    assert _cmd(_block(enabled=False)) is None


def test_the_command_sweeps_into_the_configured_tree():
    cmd = _cmd(_block())
    assert cmd[0] == "/py"
    assert cmd[1].endswith("pubfig_naive_vs_trained.py")
    assert "--sweep" in cmd
    assert _value(cmd, "--figures-dir") == "/figs/Naive"


def test_the_predictions_csv_comes_from_the_reaction_settings():
    """One source of truth: the CSV reactions just wrote is the one scored."""
    cmd = _cmd(_block(), _Settings(output_csv="/data/model_predictions.csv"))
    assert _value(cmd, "--predictions-csv") == "/data/model_predictions.csv"


def test_a_block_level_predictions_csv_wins():
    cmd = _cmd(_block(predictions_csv="/other.csv"))
    assert _value(cmd, "--predictions-csv") == "/other.csv"


def test_no_predictions_csv_anywhere_means_no_command():
    assert _cmd(_block(), _Settings(output_csv="")) is None


def test_the_config_is_forwarded_for_the_odor_remap():
    assert _value(_cmd(_block()), "--config") == "/cfg.yaml"


def test_genotype_and_correction_are_forwarded():
    cmd = _cmd(_block(genotype="GR5a-Old", correction="none"))
    assert _value(cmd, "--genotype") == "GR5a-Old"
    assert _value(cmd, "--correction") == "none"


def test_the_footnote_is_off_unless_asked_for():
    assert "--no-footnote" in _cmd(_block())
    assert "--footnote" in _cmd(_block(footnote=True))


def test_the_trend_p_label_is_on_unless_turned_off():
    assert "--trend-p" in _cmd(_block())
    assert "--no-trend-p" in _cmd(_block(trend_p=False))


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


def test_expected_state_tracks_the_predictions_csv(tmp_path):
    preds = tmp_path / "model_predictions.csv"
    preds.write_text("x")
    settings = _Settings(output_csv=str(preds))
    expected = rw._naive_vs_trained_expected(_block(), settings, config_path=None)
    assert expected["predictions_mtime"] is not None
    assert expected["command"]


def test_expected_state_changes_with_the_settings(tmp_path):
    preds = tmp_path / "model_predictions.csv"
    preds.write_text("x")
    settings = _Settings(output_csv=str(preds))
    a = rw._naive_vs_trained_expected(_block(), settings, config_path=None)
    b = rw._naive_vs_trained_expected(
        _block(correction="holm"), settings, config_path=None
    )
    assert a != b


# ---------------------------------------------------------------------------
# The shipped config must actually declare the step
# ---------------------------------------------------------------------------


def test_shipped_config_wires_the_sweep_into_the_new_opto_tree():
    import yaml

    raw = yaml.safe_load(CONFIG_PATH.read_text())
    block = (raw.get("analysis") or {}).get("naive_vs_trained")
    assert block, "naive_vs_trained is not wired into config_new.yaml"
    out_dir = str(block["out_dir"]).rstrip("/")
    assert out_dir.endswith("Comaprison-Train-Control-vs-Naive"), out_dir
    assert "New-Opto-Fly-Figures" in out_dir


def test_shipped_config_matches_the_conventions_the_figures_were_built_with():
    """GR5a-Old only, uncorrected p (matching score_summary), no footnote."""
    import yaml

    raw = yaml.safe_load(CONFIG_PATH.read_text())
    block = raw["analysis"]["naive_vs_trained"]
    assert block.get("genotype") == "GR5a-Old"
    assert block.get("correction") == "none"
    assert block.get("footnote", False) is False


def test_shipped_config_builds_a_runnable_command():
    import yaml

    raw = yaml.safe_load(CONFIG_PATH.read_text())
    settings = load_settings(CONFIG_PATH)
    cmd = rw._naive_vs_trained_command(
        raw["analysis"], settings, python_exec="/py", config_path=CONFIG_PATH
    )
    assert cmd is not None
    assert Path(cmd[1]).exists(), cmd[1]
    assert _value(cmd, "--predictions-csv").endswith("model_predictions.csv")


def test_the_force_flag_exists_so_the_step_can_be_cached_off():
    assert load_settings(CONFIG_PATH).force.naive_vs_trained in (True, False)
