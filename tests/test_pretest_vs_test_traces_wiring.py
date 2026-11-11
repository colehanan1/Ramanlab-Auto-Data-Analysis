"""The per-fly pre-vs-post trace overlays run as part of the pipeline.

Same shape as ``pretest_vs_test``: a pure command builder that returns None
when unconfigured, plus a cache key on both wide tables' mtimes and the command.
"""

from pathlib import Path

import pytest
import yaml

from scripts.pipeline.run_workflows import (
    _pretest_vs_test_traces_command,
    _pretest_vs_test_traces_expected,
)

CONFIG = Path(__file__).resolve().parent.parent / "config" / "config_new.yaml"
OUT_DIR = "/x/Results/New-Opto-Fly-Figures/Pre-vs-Post-PER-Traces"
PRE = "/x/all_envelope_rows_wide_combined_base_pretest.parquet"
POST = "/x/all_envelope_rows_wide_combined_base.parquet"


def _cfg(**over):
    base = {"out_dir": OUT_DIR, "pretest_wide_csv": PRE, "testing_wide_csv": POST}
    base.update(over)
    return {"pretest_vs_test_traces": base}


def _cmd(cfg):
    return _pretest_vs_test_traces_command(cfg, None, python_exec="python3")


def test_no_command_when_the_block_is_absent():
    assert _cmd({}) is None


def test_no_command_when_disabled():
    assert _cmd(_cfg(enabled=False)) is None


@pytest.mark.parametrize("key", ["out_dir", "pretest_wide_csv", "testing_wide_csv"])
def test_no_command_without_a_required_path(key):
    assert _cmd(_cfg(**{key: ""})) is None


def test_the_command_targets_the_trace_script_with_both_tables():
    cmd = _cmd(_cfg())
    assert cmd[0] == "python3"
    assert cmd[1].endswith("scripts/analysis/pretest_vs_test_traces.py")
    assert cmd[cmd.index("--pretest-wide-csv") + 1] == PRE
    assert cmd[cmd.index("--testing-wide-csv") + 1] == POST
    assert cmd[cmd.index("--out-dir") + 1] == OUT_DIR


def test_cohorts_are_forwarded_only_when_listed():
    assert "--cohorts" not in _cmd(_cfg())
    cmd = _cmd(_cfg(cohorts=["IAA-Sensitivity-24-1", "Hex-Sensitivity-24-0.1"]))
    i = cmd.index("--cohorts")
    assert cmd[i + 1:i + 3] == ["IAA-Sensitivity-24-1", "Hex-Sensitivity-24-0.1"]


def test_window_keys_are_forwarded():
    cmd = _cmd(_cfg(odor_on_s=30.0, odor_off_s=60.0, fps_default=40.0))
    assert cmd[cmd.index("--odor-off-s") + 1] == "60.0"
    assert cmd[cmd.index("--fps-default") + 1] == "40.0"


def test_no_fixed_length_window_is_forwarded():
    """The x-axis is the recording; a stale after_show_sec key must not leak."""
    cmd = _cmd(_cfg(after_show_sec=30.0, odor_latency_s=2.15))
    assert "--after-show-sec" not in cmd and "--odor-latency-s" not in cmd


def test_the_cache_key_tracks_the_command_and_both_tables():
    a = _pretest_vs_test_traces_expected(_cfg(), None)
    b = _pretest_vs_test_traces_expected(_cfg(cohorts=["A"]), None)
    assert a["command"] != b["command"]
    assert {"pretest_mtime", "testing_mtime"} <= set(a)
    assert a == _pretest_vs_test_traces_expected(_cfg(), None)


def test_an_unconfigured_block_still_yields_a_key():
    assert _pretest_vs_test_traces_expected({}, None)["command"] == ""


# ── the live config ───────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def analysis_cfg():
    if not CONFIG.exists():
        pytest.skip(f"{CONFIG} not present (config/ is gitignored)")
    with CONFIG.open() as fh:
        return yaml.safe_load(fh)["analysis"]


def test_config_writes_under_the_new_opto_figures_root(analysis_cfg):
    if "pretest_vs_test_traces" not in analysis_cfg:
        pytest.skip("pretest_vs_test_traces block not added to config_new.yaml yet")
    cmd = _pretest_vs_test_traces_command(analysis_cfg, None, python_exec="python3")
    assert cmd is not None
    assert cmd[cmd.index("--out-dir") + 1].startswith(
        "/home/ramanlab/Documents/cole/Results/New-Opto-Fly-Figures/"
    )


class _Reaction:
    output_csv = "/x/model_predictions.csv"


class _Settings:
    reaction_prediction = _Reaction()


def test_the_predictions_csv_is_forwarded_from_settings():
    cmd = _pretest_vs_test_traces_command(_cfg(), _Settings(), python_exec="python3")
    assert cmd[cmd.index("--predictions-csv") + 1] == "/x/model_predictions.csv"


def test_no_predictions_flag_without_one():
    assert "--predictions-csv" not in _cmd(_cfg())


def test_the_config_path_is_forwarded_for_odor_remaps():
    cmd = _pretest_vs_test_traces_command(
        _cfg(), None, python_exec="python3", config_path=Path("/x/config_new.yaml")
    )
    assert cmd[cmd.index("--config") + 1] == "/x/config_new.yaml"


def test_the_cache_key_carries_the_predictions_mtime():
    assert "predictions_mtime" in _pretest_vs_test_traces_expected(_cfg(), _Settings())


def test_the_cache_key_tracks_the_figure_code_and_the_config(tmp_path, monkeypatch):
    from fbpipe.utils import figure_cache

    config = tmp_path / "config_new.yaml"
    config.write_text("a: 1\n")
    key = _pretest_vs_test_traces_expected(_cfg(), None, config_path=config)
    assert key["config_digest"] == figure_cache.file_digest(config)
    assert key["code_digest"]

    config.write_text("a: 2\n")
    assert _pretest_vs_test_traces_expected(_cfg(), None, config_path=config) != key

    monkeypatch.setattr(figure_cache, "code_digest", lambda entries, *a, **k: "changed")
    assert _pretest_vs_test_traces_expected(_cfg(), None, config_path=config)["code_digest"] == "changed"


def test_the_code_digest_is_taken_of_the_traces_script(monkeypatch):
    from fbpipe.utils import figure_cache

    seen = []
    monkeypatch.setattr(figure_cache, "code_digest", lambda entries, *a, **k: seen.append(list(entries)) or "d")
    _pretest_vs_test_traces_expected(_cfg(), None)
    assert [p.name for p in seen[0]] == ["pretest_vs_test_traces.py"]


def test_without_a_config_path_the_config_digest_is_empty():
    assert _pretest_vs_test_traces_expected(_cfg(), None)["config_digest"] == ""


def test_a_failed_trace_run_is_not_recorded_as_done(monkeypatch):
    """A non-zero exit used to be cached as success, so later runs skipped the step."""
    import subprocess as _sp
    import scripts.pipeline.run_workflows as rw

    monkeypatch.setattr(rw, "_pretest_vs_test_traces_command", lambda *a, **k: ["py", "x.py"])
    for code, expected in ((1, False), (0, True)):
        monkeypatch.setattr(rw.subprocess, "run",
                            lambda *a, code=code, **k: _sp.CompletedProcess(a, code))
        assert rw._run_pretest_vs_test_traces({}, object(), config_path=None) is expected
    src = Path(rw.__file__).read_text()
    assert "if _run_pretest_vs_test_traces(analysis_cfg, settings, config_path=config_path):" in src
