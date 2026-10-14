"""run_workflows hands the abdomen/mounting overlay to its own stage and keeps PER on config_new.yaml.

``config_new_abdomen.yaml`` is ``extends: config_new.yaml`` plus an ``abdomen:`` block and/or a ``mounting:`` block.
``main()`` reads it through ``resolve_layers``: every PER reader and child gets the BASE file, only
``_run_abdomen_stages`` gets the overlay. The stage builds up to four commands (track; normalize + wide + figures;
the stats report; the mounting-posture analysis), runs them one after another against ONE ``abdomen.max_hours``
budget, each in its own process group, and never raises: a failed command is a warning, a held ``abdomen.lock`` or a
``--folder`` run skips the stage. The mounting command only reads what the others wrote, so it does not depend on
``abdomen.enabled``, survives ``--figures-only``, and may be the stage's only command; it is refused (a warning, no
command) when it would write outside ``Results/New-Opto-Fly-Figures`` or inside the hand-run ``Results/Figures``.
``--abdomen-only`` runs just that stage, before any PER work and without either lock. A full run takes the shared
PER lock inside main's whole-run lock (``run_lock``) for its PER work only, and the stage runs after both are let go:
it may take hours and the next PER run must not wait behind it. The driver imports the mounting analysis only when it
builds the mounting command, so a broken analysis module cannot stop a plain nightly. ``track`` is the long command: when
others follow it, it gets the budget less a tail of ``min(30 min, a tenth of the budget)``, so a timed-out track still
leaves time for normalize, the stats and the mounting analysis to read what it tracked. A killed driver must not orphan
the stage: SIGTERM / SIGHUP make the driver raise (so the group of the running command is killed and the lock freed) and
every command is spawned to die with its parent (``PR_SET_PDEATHSIG``), which also covers SIGKILL. The commands that read
``model_predictions.csv`` (the stats and the mounting analysis) are skipped, and the summary says so, whenever a PER run
holds the PER lock when they would start; the stats are handed the base config's predictions file, and no mounting
directory may lie inside production, the secured storage or the hand-run figures. Pinned here: the pure command builders,
``main()``'s source order and lock behaviour, the stage with a fake Popen and with real processes, the flags and the ntfy.
(The PER call trace itself is pinned in test_layered_parity.py.)
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import fcntl
import inspect
import logging
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import types
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import yaml

from fbpipe import geometry_pipeline
from fbpipe.config import ReactionPredictionSettings, Settings
from fbpipe.config_layers import ConfigLayers, LayeredConfigError
from fbpipe.utils.stage_locks import abdomen_lock, acquire_per_run_lock, per_run_active, per_run_lock, release_run_lock
from scripts.pipeline import run_workflows as rw
from test_eye_drift_qc_summary import _run_main_guard

LIVE = Path(__file__).resolve().parents[1] / "config" / "config_new.yaml"
OVERLAY = Path("/cfg/config_new_abdomen.yaml")
PY = "/venv/bin/python"
STATS = str(rw.REPO_ROOT / "scripts" / "analysis" / "abdomen_geometry_stats.py")
MOUNTING_SCRIPT = str(rw.REPO_ROOT / "scripts" / "analysis" / "mounting_posture.py")

GEOMETRY = [PY, "-m", "fbpipe.geometry_pipeline", "--config", str(OVERLAY)]
TRACK = GEOMETRY + ["--workers", "3", "track"]
NORMALIZE = GEOMETRY + ["normalize", "wide", "figures"]

# Strings only: the builder is pure and every Popen here is a fake, so none of these paths is ever touched.
NEW_OPTO = "/home/ramanlab/Documents/cole/Results/New-Opto-Fly-Figures"
HAND_RUN = "/home/ramanlab/Documents/cole/Results/Figures"
PREDICTIONS = "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/model_predictions.csv"
MOUNTING_TABLES = "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/Mounting-Posture"
STATS_CMD = [PY, STATS, "--config", str(OVERLAY), "--predictions", PREDICTIONS]     # the base config's predictions CSV
STATS_NO_PREDICTIONS = [PY, STATS, "--config", str(OVERLAY)]                        # a base config without one
MOUNTING = {
    "enabled": True,
    "out_dir": f"{NEW_OPTO}/Mounting-Posture",
    "abdomen_out_dir": f"{NEW_OPTO}/Abdomen-Movement",
    "csv_out_dir": MOUNTING_TABLES,
}
BOTH = {"abdomen": {"enabled": True}, "mounting": MOUNTING}
ABSENT = object()                                  # "the key is not in the block", as opposed to None / ""


def _mounting_cmd(python=PY, overlay=OVERLAY, predictions=PREDICTIONS) -> list[str]:
    return [str(python), MOUNTING_SCRIPT, "--config", str(overlay), "--predictions-csv", predictions,
            "--out-dir", f"{NEW_OPTO}/Mounting-Posture", "--abdomen-out-dir", f"{NEW_OPTO}/Abdomen-Movement",
            "--csv-out-dir", MOUNTING_TABLES]


MOUNTING_CMD = _mounting_cmd()


def _base_settings(output_csv: str = PREDICTIONS, **reaction) -> Settings:
    """What ``load_settings(config_new.yaml)`` returns, reduced to the one thing the mounting command reads."""
    return Settings(model_path="m.pt", main_directories=[],
                    reaction_prediction=ReactionPredictionSettings(output_csv=output_csv, **reaction))


def _cmds(overlay=None, settings=None, **over):
    kwargs = dict(python_exec=PY, overlay_path=OVERLAY, figures_only=False, annotated_video=False, n_workers=3)
    kwargs.update(over)
    overlay = {"abdomen": {"enabled": True}} if overlay is None else overlay
    return rw._abdomen_commands(overlay, _base_settings() if settings is None else settings, **kwargs)


# ---- the command builder ----------------------------------------------------------------------------------

@pytest.mark.parametrize("overlay", [{}, None, {"mounting": {"enabled": True}}, {"abdomen": None},
                                     {"abdomen": {"enabled": False}}])
def test_no_commands_without_an_enabled_abdomen_block_or_a_complete_mounting_block(overlay):
    assert rw._abdomen_commands(overlay, _base_settings(), python_exec=PY, overlay_path=OVERLAY, figures_only=False,
                                annotated_video=False, n_workers=3) == []


def test_an_enabled_block_gives_the_three_commands_in_order():
    assert _cmds() == [TRACK, NORMALIZE, STATS_CMD]


def test_a_block_without_an_enabled_key_counts_as_enabled():
    assert _cmds({"abdomen": {"model_path": "/m.pt"}}) == [TRACK, NORMALIZE, STATS_CMD]


def test_figures_only_drops_the_track_command():
    assert _cmds(figures_only=True) == [NORMALIZE, STATS_CMD]


def test_the_annotated_video_flag_reaches_only_the_track_command():
    cmds = _cmds(annotated_video=True)
    assert cmds[0] == GEOMETRY + ["--workers", "3", "--annotated-video", "track"]
    assert cmds[1:] == [NORMALIZE, STATS_CMD]


def test_the_worker_count_and_interpreter_are_the_callers():
    cmds = _cmds(python_exec="~/cam-env/bin/python", n_workers=5)
    assert {cmd[0] for cmd in cmds} == {str(Path("~/cam-env/bin/python").expanduser())}
    assert cmds[0][cmds[0].index("--workers") + 1] == "5"


def test_every_command_gets_the_overlay_and_never_the_base_config():
    for cmd in _cmds(BOTH):
        assert cmd[cmd.index("--config") + 1] == str(OVERLAY)
        assert "config_new.yaml" not in " ".join(cmd)


# ---- the mounting command ---------------------------------------------------------------------------------

def _mounting(block=None, settings=None, **over):
    """``rw._mounting_command`` for an overlay whose ``mounting:`` block is *block* (default: MOUNTING)."""
    kwargs = dict(python_exec=PY, overlay_path=OVERLAY)
    kwargs.update(over)
    return rw._mounting_command({"mounting": MOUNTING if block is None else block},
                                _base_settings() if settings is None else settings, **kwargs)


def _with(**changes):
    """MOUNTING with *changes* applied; ``ABSENT`` removes the key."""
    block = {**MOUNTING, **changes}
    return {key: value for key, value in block.items() if value is not ABSENT}


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]


def test_the_mounting_command_is_exactly_the_script_with_the_overlay_the_predictions_and_the_three_directories():
    assert Path(MOUNTING_SCRIPT).is_file()
    assert _mounting() == [
        PY, MOUNTING_SCRIPT,
        "--config", str(OVERLAY),
        "--predictions-csv", PREDICTIONS,
        "--out-dir", "/home/ramanlab/Documents/cole/Results/New-Opto-Fly-Figures/Mounting-Posture",
        "--abdomen-out-dir", "/home/ramanlab/Documents/cole/Results/New-Opto-Fly-Figures/Abdomen-Movement",
        "--csv-out-dir", "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/Mounting-Posture",
    ]


def test_the_pipelines_figure_root_is_the_new_opto_directory_with_a_trailing_separator():
    assert rw.MOUNTING_FIGURES_ROOT == "/home/ramanlab/Documents/cole/Results/New-Opto-Fly-Figures/"


def test_the_mounting_command_reads_the_output_csv_of_the_base_settings():
    settings = _base_settings(output_csv="/pred/model_predictions.csv", data_csv="/wide/all_rows.csv")
    cmd = _mounting(settings=settings)
    assert cmd[cmd.index("--predictions-csv") + 1] == "/pred/model_predictions.csv"
    assert "/wide/all_rows.csv" not in cmd


def test_the_mounting_command_runs_under_the_callers_interpreter_expanded_like_the_others():
    assert _mounting(python_exec="~/cam-env/bin/python")[0] == str(Path("~/cam-env/bin/python").expanduser())


def test_a_mounting_block_without_an_enabled_key_counts_as_enabled():
    assert _mounting(_with(enabled=ABSENT)) == MOUNTING_CMD


@pytest.mark.parametrize("overlay", [{}, None, {"abdomen": {"enabled": True}}, {"mounting": None},
                                     {"mounting": {**MOUNTING, "enabled": False}}],
                         ids=["empty", "none", "abdomen only", "empty block", "disabled"])
def test_no_mounting_command_without_an_enabled_mounting_block_and_nothing_to_warn_about(overlay, caplog):
    with caplog.at_level(logging.INFO):
        assert rw._mounting_command(overlay, _base_settings(), python_exec=PY, overlay_path=OVERLAY) is None
    assert _warnings(caplog) == [], "the user asked for nothing"


@pytest.mark.parametrize("settings", [_base_settings(""), _base_settings("   "), types.SimpleNamespace(), None],
                         ids=["empty", "blank", "no reaction_prediction", "no settings"])
def test_no_mounting_command_without_a_predictions_csv_and_the_warning_says_which(settings, caplog):
    with caplog.at_level(logging.INFO):
        assert rw._mounting_command({"mounting": MOUNTING}, settings, python_exec=PY, overlay_path=OVERLAY) is None
    assert len(_warnings(caplog)) == 1 and "reaction_prediction.output_csv" in _warnings(caplog)[0]


@pytest.mark.parametrize("key", ["out_dir", "abdomen_out_dir", "csv_out_dir"])
@pytest.mark.parametrize("value", [ABSENT, None, "", "  "], ids=["absent", "none", "empty", "blank"])
def test_no_mounting_command_when_a_directory_is_missing_and_the_warning_names_it(key, value, caplog):
    with caplog.at_level(logging.INFO):
        assert _mounting(_with(**{key: value})) is None
    assert len(_warnings(caplog)) == 1 and f"mounting.{key}" in _warnings(caplog)[0]


@pytest.mark.parametrize("change", [{"enabled": "yes"}, {"rest_quantile": 7}, {"typo_key": 1}, {"out_dir": 5}],
                         ids=["enabled is not a bool", "value out of range", "unknown key", "path is not a string"])
def test_a_mounting_block_that_does_not_parse_is_a_warning_and_no_command_never_an_error(change, caplog):
    with caplog.at_level(logging.INFO):
        assert _mounting(_with(**change)) is None
    assert len(_warnings(caplog)) == 1 and "mounting" in _warnings(caplog)[0]


def test_a_mounting_block_that_does_not_parse_is_logged_with_its_traceback(caplog):
    with caplog.at_level(logging.INFO):
        assert _mounting(_with(typo_key=1)) is None
    (record,) = [r for r in caplog.records if r.levelname == "WARNING"]
    assert record.exc_info and record.exc_info[0] is ValueError, "the log must say where it was raised"


@pytest.mark.parametrize("key", ["out_dir", "abdomen_out_dir", "csv_out_dir"])
def test_a_directory_whose_home_does_not_exist_is_a_warning_naming_the_error_never_an_exception(key, caplog):
    """``Path.expanduser`` raises RuntimeError -- not ValueError -- for ``~name`` when ``name`` is no user."""
    with caplog.at_level(logging.INFO):
        assert _mounting(_with(**{key: "~zzzznouser/x"})) is None
    assert len(_warnings(caplog)) == 1 and "RuntimeError" in _warnings(caplog)[0]


def test_a_number_beyond_the_float_range_is_a_warning_naming_the_error_never_an_exception(caplog):
    """``math.isfinite`` raises OverflowError -- not ValueError -- for an int too large for a float."""
    with caplog.at_level(logging.INFO):
        assert _mounting(_with(n_permutations=10**400)) is None
    assert len(_warnings(caplog)) == 1 and "OverflowError" in _warnings(caplog)[0]


@pytest.mark.parametrize("key, value", [
    ("out_dir", f"{HAND_RUN}/Mounting-Posture"),
    ("abdomen_out_dir", f"{HAND_RUN}/Abdomen-Movement"),
    ("out_dir", "/tmp/mounting"),
    ("abdomen_out_dir", f"{NEW_OPTO}-BACKUP-pre-born-cutoff-2026-08-26/Abdomen-Movement"),
    ("out_dir", NEW_OPTO),
    ("abdomen_out_dir", NEW_OPTO + "/"),
    ("out_dir", f"{NEW_OPTO}/../Figures/Mounting-Posture"),
    ("abdomen_out_dir", f"{NEW_OPTO}/Abdomen-Movement/../../../elsewhere"),
    ("out_dir", "Results/New-Opto-Fly-Figures/Mounting-Posture"),
    ("csv_out_dir", f"{HAND_RUN}/tables"),
    ("csv_out_dir", HAND_RUN),
    ("csv_out_dir", f"{NEW_OPTO}/../Figures/tables"),
    ("csv_out_dir", "Results/Figures/tables"),
], ids=["hand-run figures dir", "hand-run abdomen dir", "outside every root", "sibling sharing the prefix text",
        "the root itself", "the root with a slash", "walks into Results/Figures", "walks out of the root",
        "relative path", "tables in Results/Figures", "Results/Figures itself", "tables walk into Results/Figures",
        "relative tables in Results/Figures"])
def test_a_directory_the_pipeline_must_not_write_to_is_refused_with_one_warning_naming_it(key, value, caplog):
    with caplog.at_level(logging.INFO):
        assert _mounting(_with(**{key: value})) is None
    assert len(_warnings(caplog)) == 1 and f"mounting.{key}" in _warnings(caplog)[0]


FLYS_NEW = "/home/ramanlab/Documents/cole/Data/flys_New"


@pytest.mark.parametrize("value", [
    FLYS_NEW, FLYS_NEW + "/", f"{FLYS_NEW}/IAA-X", f"{FLYS_NEW}/IAA-X/october_07_batch_2/tables",
    f"/somewhere/else/flys_New/tables", "/securedstorage", "/securedstorage/DATAsec/cole/Data-secured-New/tables",
    f"/tmp/elsewhere/../../securedstorage/x", f"{FLYS_NEW}/../flys_New/IAA-X",
], ids=["flys_New", "flys_New with a slash", "a dataset folder", "a batch folder", "a flys_New elsewhere",
        "the secured mount", "inside the secured mount", "walks into the secured mount", "walks back into flys_New"])
def test_the_tables_may_not_go_inside_production_or_the_secured_storage_either(value, caplog):
    with caplog.at_level(logging.INFO):
        assert _mounting(_with(csv_out_dir=value)) is None
    (warning,) = _warnings(caplog)
    assert "mounting.csv_out_dir" in warning and ("flys_New" in warning or "/securedstorage" in warning)


@pytest.mark.parametrize("value", [
    f"{FLYS_NEW}2/tables", f"{FLYS_NEW}_copy/tables", "/securedstorage2/tables", MOUNTING_TABLES, "tables/mounting",
], ids=["flys_New2", "flys_New_copy", "securedstorage2", "the tables root", "relative"])
def test_a_directory_that_only_looks_like_production_is_not_refused(value):
    cmd = _mounting(_with(csv_out_dir=value))
    assert cmd is not None and cmd[cmd.index("--csv-out-dir") + 1] == value


def test_the_production_roots_of_the_base_settings_are_read_only_to_the_tables_too(caplog):
    settings = Settings(model_path="m.pt", main_directories=["/mnt/prod/Data/flys_Old/IAA-X", "/mnt/prod/Data/flys_Old/Hex-Y"],
                        reaction_prediction=ReactionPredictionSettings(output_csv=PREDICTIONS))
    with caplog.at_level(logging.INFO):
        assert _mounting(_with(csv_out_dir="/mnt/prod/Data/flys_Old/IAA-X/tables"), settings=settings) is None
    (warning,) = _warnings(caplog)
    assert "mounting.csv_out_dir" in warning and "/mnt/prod/Data/flys_Old/IAA-X" in warning
    assert _mounting(_with(csv_out_dir="/mnt/prod/Data/flys_Old/IAA-X2/tables"), settings=settings) is not None


def test_every_refused_directory_is_warned_about_not_just_the_first(caplog):
    with caplog.at_level(logging.INFO):
        assert _mounting(_with(out_dir="/tmp/a", abdomen_out_dir="/tmp/b", csv_out_dir=f"{HAND_RUN}/c")) is None
    warnings = _warnings(caplog)
    assert len(warnings) == 3
    for key in ("out_dir", "abdomen_out_dir", "csv_out_dir"):
        assert sum(f"mounting.{key}" in w for w in warnings) == 1


@pytest.mark.parametrize("key, value, passed", [
    ("out_dir", f"{NEW_OPTO}/Mounting-Posture/nested", f"{NEW_OPTO}/Mounting-Posture/nested"),
    ("abdomen_out_dir", f"{NEW_OPTO}/Abdomen-Movement/", f"{NEW_OPTO}/Abdomen-Movement"),
    ("out_dir", f"{NEW_OPTO}/a/../Mounting-Posture", f"{NEW_OPTO}/Mounting-Posture"),
    ("abdomen_out_dir", f"{NEW_OPTO}/a/../Abdomen-Movement", f"{NEW_OPTO}/Abdomen-Movement"),
    ("csv_out_dir", "/tmp/anywhere/tables", "/tmp/anywhere/tables"),
    ("csv_out_dir", "/tmp/anywhere/a/../tables", "/tmp/anywhere/tables"),
    ("csv_out_dir", "tables/mounting", "tables/mounting"),
], ids=["nested", "trailing slash", "figure dot-dot that stays inside", "abdomen dot-dot that stays inside",
        "tables anywhere else", "tables dot-dot", "relative tables"])
def test_a_directory_inside_the_allowed_places_is_passed_as_the_path_that_was_checked(key, value, passed):
    cmd = _mounting(_with(**{key: value}))
    assert cmd is not None
    assert cmd[cmd.index("--" + key.replace("_", "-")) + 1] == passed, "what is checked is what runs"


# ---- the command builder: the mounting command's place -----------------------------------------------------

def test_the_mounting_command_is_the_fourth_command_after_the_stats():
    assert _cmds(BOTH) == [TRACK, NORMALIZE, STATS_CMD, MOUNTING_CMD]


def test_figures_only_keeps_the_mounting_command_because_it_only_reads():
    assert _cmds(BOTH, figures_only=True) == [NORMALIZE, STATS_CMD, MOUNTING_CMD]


@pytest.mark.parametrize("abdomen", [ABSENT, None, {"enabled": False}], ids=["no block", "empty block", "disabled"])
def test_the_mounting_command_does_not_depend_on_the_abdomen_block(abdomen):
    overlay = {"mounting": MOUNTING} if abdomen is ABSENT else {"abdomen": abdomen, "mounting": MOUNTING}
    assert _cmds(overlay) == [MOUNTING_CMD]
    assert _cmds(overlay, figures_only=True) == [MOUNTING_CMD]


def test_the_track_flags_never_reach_the_mounting_command():
    cmds = _cmds(BOTH, annotated_video=True, n_workers=5)
    assert cmds[0] == GEOMETRY + ["--workers", "5", "--annotated-video", "track"]
    assert cmds[3] == MOUNTING_CMD


def test_the_mounting_command_gets_the_interpreter_the_abdomen_commands_get():
    cmds = _cmds(BOTH, python_exec="~/cam-env/bin/python")
    assert {cmd[0] for cmd in cmds} == {str(Path("~/cam-env/bin/python").expanduser())} and len(cmds) == 4


def test_a_refused_mounting_block_leaves_the_abdomen_commands_alone(caplog):
    overlay = {"abdomen": {"enabled": True}, "mounting": _with(out_dir=f"{HAND_RUN}/Mounting-Posture")}
    with caplog.at_level(logging.INFO):
        assert _cmds(overlay) == [TRACK, NORMALIZE, STATS_CMD]
    assert len(_warnings(caplog)) == 1


def test_an_empty_predictions_path_drops_only_the_mounting_command(caplog):
    with caplog.at_level(logging.INFO):
        assert _cmds(BOTH, settings=_base_settings("")) == [TRACK, NORMALIZE, STATS_NO_PREDICTIONS]
    assert len(_warnings(caplog)) == 1


def test_the_stats_command_reads_the_predictions_csv_of_the_base_settings_not_the_scripts_hard_coded_default():
    settings = _base_settings(output_csv="/pred/model_predictions.csv", data_csv="/wide/all_rows.csv")
    stats = _cmds(settings=settings)[2]
    assert stats == [PY, STATS, "--config", str(OVERLAY), "--predictions", "/pred/model_predictions.csv"]
    assert "/wide/all_rows.csv" not in stats


def test_the_stats_command_parses_with_the_scripts_own_parser(monkeypatch):
    from scripts.analysis import abdomen_geometry_stats

    class Parsed(Exception):
        pass

    real, seen = argparse.ArgumentParser.parse_args, []

    def parse_then_stop(self, args=None, namespace=None):
        seen.append(real(self, args, namespace))
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_then_stop)
    with pytest.raises(Parsed):
        abdomen_geometry_stats.main(_cmds(settings=_base_settings(output_csv="/pred/p.csv"))[2][2:])
    (parsed,) = seen
    assert (parsed.config, parsed.predictions) == (str(OVERLAY), "/pred/p.csv")


# ---- the stage: fakes --------------------------------------------------------------------------------------

class _Proc:
    """A subprocess.Popen stand-in. ``mode`` is an exit code, "timeout" (the first wait times out after the whole
    allowance has passed; the process is gone after SIGTERM), "ignores SIGTERM" (it also outlives the grace wait and
    dies on SIGKILL), a KeyboardInterrupt to raise from the first wait, ``("signal", n)`` (the driver is sent signal n
    during the first wait) or ``("signal twice", n)`` (and once more during the grace wait after SIGTERM)."""

    def __init__(self, procs, pid, mode, elapse):
        self.procs, self.pid, self.mode, self.elapse, self.returncode, self.waits = procs, pid, mode, elapse, None, []

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if isinstance(self.mode, tuple):
            kind, signum = self.mode
            if len(self.waits) == 1 or (kind == "signal twice" and len(self.waits) == 2):
                os.kill(os.getpid(), signum)                       # the handler under test raises right here
            self.returncode = -15
        elif isinstance(self.mode, KeyboardInterrupt):
            if len(self.waits) == 1:
                raise self.mode
            self.returncode = -15
        elif isinstance(self.mode, int):
            self.procs.clock.now += self.elapse                   # the command ran this long
            self.returncode = self.mode
        else:
            stubborn = self.mode == "ignores SIGTERM"
            if len(self.waits) == 1 or (stubborn and len(self.waits) == 2):
                self.procs.clock.now += timeout                    # it ran, or was given, this long before the timeout
                raise subprocess.TimeoutExpired("cmd", timeout)
            self.returncode = -9 if stubborn else -15
        return self.returncode

    def poll(self):
        return self.returncode


class _Clock:
    """``rw.time`` with a monotonic clock the test advances; everything else is the real time module."""

    def __init__(self, now=1000.0):
        self.now = now

    def monotonic(self):
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


class _Procs:
    """Fakes subprocess.Popen and os.killpg. ``script`` has one entry per Popen call: an exit code, a _Proc mode or an
    exception to raise instead of starting (default: exit 0); ``elapse`` the seconds each command takes."""

    def __init__(self, monkeypatch, script=(), elapse=()):
        self.script, self.elapse = list(script), list(elapse)
        self.calls, self.kwargs, self.procs, self.killed = [], [], [], []
        self.clock = _Clock()
        monkeypatch.setattr(rw, "time", self.clock)
        monkeypatch.setattr(subprocess, "Popen", self._popen)
        monkeypatch.setattr(os, "killpg", lambda pid, sig: self.killed.append((pid, sig)))

    def _popen(self, cmd, **kwargs):
        i = len(self.calls)
        self.calls.append(list(cmd))
        self.kwargs.append(kwargs)
        mode = self.script[i] if i < len(self.script) else 0
        if isinstance(mode, Exception):
            raise mode
        proc = _Proc(self, 4201 + i, mode, self.elapse[i] if i < len(self.elapse) else 0.0)
        self.procs.append(proc)
        return proc


@pytest.fixture
def procs(monkeypatch):
    return _Procs(monkeypatch)


@pytest.fixture(autouse=True)
def _isolated_from_the_developers_shell_and_the_previous_test(monkeypatch):
    monkeypatch.delenv("CACHE_DIR", raising=False)         # load_settings lets it override cache_dir, where the locks live
    monkeypatch.delenv("REACTION_OUTPUT_CSV", raising=False)    # ... and the predictions CSV the mounting command reads
    monkeypatch.setattr(rw, "LAST_ABDOMEN_SUMMARY", "")    # main() and the stage set these; restore them afterwards
    monkeypatch.setattr(rw, "LAST_RUN_ABDOMEN_ONLY", False)


def _settings(tmp_path, output_csv: str = PREDICTIONS) -> Settings:
    return dataclasses.replace(_base_settings(output_csv), cache_dir=str(tmp_path / "cache"))


def _layers(tmp_path, abdomen=None, **blocks) -> ConfigLayers:
    """An overlay with the given blocks plus an enabled ``abdomen:`` block (``abdomen=False``: no such block)."""
    overlay = dict(blocks)
    if abdomen is not False:
        overlay["abdomen"] = {"enabled": True} if abdomen is None else abdomen
    return ConfigLayers(tmp_path / "config_new_abdomen.yaml", tmp_path / "config_new.yaml", MappingProxyType(overlay))


def _run(tmp_path, layers=None, settings=None, **kw):
    kwargs = dict(figures_only=False, folder=None, annotated_video=False)
    kwargs.update(kw)
    return rw._run_abdomen_stages(layers or _layers(tmp_path), settings or _settings(tmp_path), **kwargs)


def _expected(tmp_path, **over):
    """The three abdomen commands the builder gives for the default stage (no mounting block)."""
    kwargs = dict(python_exec=sys.executable, overlay_path=tmp_path / "config_new_abdomen.yaml", figures_only=False,
                  annotated_video=False, n_workers=3)
    kwargs.update(over)
    return rw._abdomen_commands({"abdomen": {"enabled": True}}, _settings(tmp_path), **kwargs)


# ---- the stage: behaviour ----------------------------------------------------------------------------------

def test_the_stage_runs_the_three_commands_in_order_each_in_its_own_session(tmp_path, procs):
    result = _run(tmp_path)
    assert procs.calls == _expected(tmp_path)
    assert result["ran"] is True and result["skipped_reason"] == "" and result["error"] == ""
    assert [c["returncode"] for c in result["commands"]] == [0, 0, 0]
    assert [c["label"] for c in result["commands"]] == ["track", "normalize+wide+figures", "abdomen_geometry_stats"]
    for kwargs in procs.kwargs:
        assert kwargs["start_new_session"] is True, "killpg must reach the track workers too"


def test_the_commands_get_a_headless_matplotlib_and_the_repo_source_first_on_the_path(tmp_path, procs, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "qtagg")
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    _run(tmp_path)
    for kwargs in procs.kwargs:
        assert kwargs["env"]["MPLBACKEND"] == "Agg"
        assert kwargs["env"]["PYTHONPATH"].split(os.pathsep) == [str(rw.REPO_ROOT / "src"), "/somewhere/else"]


def test_the_stage_hands_every_command_the_overlay_not_the_base(tmp_path, procs):
    _run(tmp_path)
    for cmd in procs.calls:
        assert cmd[cmd.index("--config") + 1] == str(tmp_path / "config_new_abdomen.yaml")


def test_the_stage_reads_workers_python_and_budget_from_the_block(tmp_path, procs):
    block = {"enabled": True, "python": "/cam/bin/python", "yolo": {"num_workers": 2}, "max_hours": 2}
    _run(tmp_path, _layers(tmp_path, block))
    assert procs.calls == _expected(tmp_path, python_exec="/cam/bin/python", n_workers=2)
    assert procs.procs[0].waits == [6480.0], "track gets the 2 h budget less its tail (a tenth of it: 720 s)"


def test_figures_only_and_the_abdomen_video_flag_reach_the_stage_commands(tmp_path, procs):
    _run(tmp_path, figures_only=True)
    assert procs.calls == _expected(tmp_path, figures_only=True) and len(procs.calls) == 2
    procs.calls.clear()
    _run(tmp_path, annotated_video=True)
    assert procs.calls == _expected(tmp_path, annotated_video=True)
    assert "--annotated-video" in procs.calls[0] and all("--annotated-video" not in c for c in procs.calls[1:])


def test_a_held_abdomen_lock_skips_the_stage(tmp_path, procs, caplog):
    with abdomen_lock(tmp_path / "cache") as got, caplog.at_level(logging.INFO):
        assert got is True
        result = _run(tmp_path)
    assert result["skipped_reason"] == "locked" and result["ran"] is False
    assert procs.calls == [], "nothing is started while another abdomen run holds the lock"
    assert any("abdomen.lock" in r.getMessage() for r in caplog.records)
    assert rw.LAST_ABDOMEN_SUMMARY == result["summary"] and "skipped" in result["summary"]
    assert _run(tmp_path)["ran"] is True, "the lock is free again once the other run ends"


def test_the_stage_holds_the_exclusive_lock_while_it_runs_and_frees_it_after(tmp_path, monkeypatch):
    seen = []

    class Probe(_Procs):
        def _popen(self, cmd, **kwargs):
            with abdomen_lock(tmp_path / "cache") as got:
                seen.append(got)
            return super()._popen(cmd, **kwargs)

    Probe(monkeypatch)
    _run(tmp_path)
    assert seen == [False, False, False]
    with abdomen_lock(tmp_path / "cache") as got:
        assert got is True


def test_a_folder_run_skips_the_stage(tmp_path, procs):
    result = _run(tmp_path, folder="Hex-Training-24")
    assert result["skipped_reason"] == "folder" and result["ran"] is False and procs.calls == []


def test_a_disabled_abdomen_block_and_an_unrunnable_mounting_block_leave_nothing_to_run(tmp_path, procs):
    result = _run(tmp_path, _layers(tmp_path, {"enabled": False}, mounting={"enabled": True}))
    assert result["skipped_reason"] == "nothing to run" and procs.calls == []


def test_a_nonzero_exit_is_a_warning_and_the_next_commands_still_run(tmp_path, monkeypatch, caplog):
    procs = _Procs(monkeypatch, script=[1, 0, 3])
    with caplog.at_level(logging.INFO):
        result = _run(tmp_path)
    assert len(procs.calls) == 3, "a failed track must not stop normalize, wide, figures or the stats"
    assert [c["returncode"] for c in result["commands"]] == [1, 0, 3]
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert "[abdomen] track exited with code 1" in warnings
    assert "[abdomen] abdomen_geometry_stats exited with code 3" in warnings
    assert len(warnings) == 2
    assert "track rc=1" in result["summary"] and "abdomen_geometry_stats rc=3" in result["summary"]


def test_a_timeout_kills_the_process_group_of_track_which_leaves_its_tail_to_the_commands_after_it(
        tmp_path, monkeypatch, caplog):
    procs = _Procs(monkeypatch, script=["timeout"])
    with caplog.at_level(logging.INFO):
        result = _run(tmp_path)
    assert procs.killed == [(4201, signal.SIGTERM)], "the whole group of the first command, not just its leader"
    assert procs.procs[0].waits == [34200.0, 30.0], "10 h less the 30 min tail; then 30 s of grace after SIGTERM"
    assert result["timed_out"] is True and result["commands"][0]["timed_out"] is True
    assert len(procs.calls) == 3, "the tail is still there: normalize and the stats read what track did get done"
    assert [p.waits for p in procs.procs[1:]] == [[1800.0], [1800.0]]
    assert [c["ran"] for c in result["commands"]] == [True, True, True]
    assert any(r.levelname == "WARNING" and "track" in r.getMessage() for r in caplog.records)
    assert result["summary"] == ("ABDOMEN: track TIMED OUT (killed); normalize+wide+figures rc=0; "
                                 "abdomen_geometry_stats rc=0")


def test_a_process_that_ignores_sigterm_is_killed(tmp_path, monkeypatch):
    procs = _Procs(monkeypatch, script=["ignores SIGTERM"])
    result = _run(tmp_path)
    assert procs.killed == [(4201, signal.SIGTERM), (4201, signal.SIGKILL)]
    assert result["commands"][0]["returncode"] == -9 and result["timed_out"] is True


def test_the_commands_share_one_time_budget(tmp_path, monkeypatch):
    procs = _Procs(monkeypatch, elapse=[3000.0, 100.0, 0.0])            # one hour in all; the first uses 50 minutes
    _run(tmp_path, _layers(tmp_path, {"enabled": True, "max_hours": 1}))
    assert [p.waits for p in procs.procs] == [[3240.0], [600.0], [500.0]], "track: the hour less its 360 s tail"


@pytest.mark.parametrize("hours, tail", [(10, 1800.0), (5, 1800.0), (3, 1080.0), (1, 360.0), (0.01, 3.6)])
def test_the_tail_kept_from_track_is_a_tenth_of_the_budget_but_never_more_than_half_an_hour(
        tmp_path, procs, hours, tail):
    _run(tmp_path, _layers(tmp_path, {"enabled": True, "max_hours": hours}))
    total = hours * 3600.0
    assert procs.procs[0].waits == [pytest.approx(total - tail)]
    assert [p.waits for p in procs.procs[1:]] == [[pytest.approx(total)]] * 2, "nothing is held back from the others"


def test_the_commands_after_a_timed_out_track_share_what_is_left_and_the_budget_still_ends_the_stage(
        tmp_path, monkeypatch):
    procs = _Procs(monkeypatch, script=["timeout", "timeout", 0])        # normalize also runs into the end of the budget
    result = _run(tmp_path, _layers(tmp_path, {"enabled": True, "max_hours": 1}))
    assert [p.waits[0] for p in procs.procs] == [3240.0, 360.0], "track at the hour less its tail, normalize at the end"
    assert len(procs.calls) == 2 and [c["ran"] for c in result["commands"]] == [True, True, False]
    assert "track TIMED OUT (killed); normalize+wide+figures TIMED OUT (killed); abdomen_geometry_stats not run" \
        in result["summary"]


def test_track_as_the_last_command_gets_the_whole_remaining_budget(tmp_path, monkeypatch):
    track_last = [NORMALIZE, TRACK]
    monkeypatch.setattr(rw, "_abdomen_commands", lambda *args, **kwargs: track_last)
    procs = _Procs(monkeypatch, elapse=[600.0, 0.0])
    result = _run(tmp_path, _layers(tmp_path, {"enabled": True, "max_hours": 1}))
    assert [p.waits for p in procs.procs] == [[3600.0], [3000.0]], "nothing follows track: no tail is held back"
    assert [c["label"] for c in result["commands"]] == ["normalize+wide+figures", "track"]


def test_track_alone_gets_the_whole_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(rw, "_abdomen_commands", lambda *args, **kwargs: [TRACK])
    procs = _Procs(monkeypatch)
    _run(tmp_path, _layers(tmp_path, {"enabled": True, "max_hours": 1}))
    assert [p.waits for p in procs.procs] == [[3600.0]]


def test_a_budget_that_is_used_up_starts_nothing_more(tmp_path, monkeypatch):
    procs = _Procs(monkeypatch, elapse=[4000.0])                        # the first command ends 400 s past the hour
    result = _run(tmp_path, _layers(tmp_path, {"enabled": True, "max_hours": 1}))
    assert len(procs.calls) == 1 and result["timed_out"] is False
    assert [c["ran"] for c in result["commands"]] == [True, False, False]
    assert "not run" in result["summary"]


@pytest.mark.parametrize("block, why", [
    ({"enabled": True, "max_hours": "ten"}, "max_hours"),
    ({"enabled": True, "max_hours": 0}, "max_hours"),
    ({"enabled": True, "max_hours": float("nan")}, "max_hours"),
    ({"enabled": True, "yolo": {"num_workers": 0}}, "num_workers"),
    ({"enabled": True, "yolo": "three"}, "yolo"),
    ({"enabled": True, "python": 3}, "python"),
])
def test_a_bad_block_value_is_a_warning_and_nothing_runs(tmp_path, procs, caplog, block, why):
    with caplog.at_level(logging.INFO):
        result = _run(tmp_path, _layers(tmp_path, block))
    assert procs.calls == [] and result["ran"] is False and why in result["error"]
    assert any(r.levelname == "WARNING" and why in r.getMessage() for r in caplog.records)
    assert why in rw.LAST_ABDOMEN_SUMMARY


@pytest.mark.parametrize("exc", [OSError("disk gone"), LayeredConfigError("bad overlay"), yaml.YAMLError("bad yaml"),
                                 subprocess.SubprocessError("popen"), ValueError("bad value"),
                                 RuntimeError("Could not determine home directory."),
                                 OverflowError("int too large to convert to float"), KeyError("missing")])
def test_the_stage_never_raises(tmp_path, monkeypatch, caplog, exc):
    procs = _Procs(monkeypatch, script=[exc])
    with caplog.at_level(logging.INFO):
        result = _run(tmp_path)
    assert len(procs.calls) == 1 and result["ran"] is False and type(exc).__name__ in result["error"]
    assert any(r.levelname == "WARNING" and str(exc) in r.getMessage() for r in caplog.records)
    assert rw.LAST_ABDOMEN_SUMMARY.startswith("ABDOMEN:") and type(exc).__name__ in rw.LAST_ABDOMEN_SUMMARY


def test_a_stage_that_fails_is_logged_with_its_traceback(tmp_path, monkeypatch, caplog):
    _Procs(monkeypatch, script=[OSError("disk gone")])
    with caplog.at_level(logging.INFO):
        _run(tmp_path)
    (record,) = [r for r in caplog.records if r.levelname == "WARNING" and "stage failed" in r.getMessage()]
    assert record.exc_info and record.exc_info[0] is OSError and "disk gone" in str(record.exc_info[1])


def test_a_track_that_yielded_to_a_per_run_is_not_a_failure_and_the_commands_after_it_still_run(
        tmp_path, monkeypatch, caplog):
    """The geometry CLI exits 75 (EX_TEMPFAIL) when its track stopped for a PER run: what it did not get to resumes next
    run. That is neither success nor an error, so the stage logs it below warning level and words it on its own."""
    procs = _Procs(monkeypatch, script=[75, 0, 0])
    with caplog.at_level(logging.INFO):
        result = _run(tmp_path)
    assert len(procs.calls) == 3, "normalize and the stats still read what track got done"
    track = result["commands"][0]
    assert track["returncode"] == 75 and track["yielded"] is True and track["ran"] is True
    assert [c["yielded"] for c in result["commands"]] == [True, False, False]
    assert _warnings(caplog) == [], "a yield is not a warning"
    assert any("track yielded to a PER run" in r.getMessage() and r.levelname == "INFO" for r in caplog.records)
    assert result["summary"] == ("ABDOMEN: track yielded to a PER run (resumes next run); normalize+wide+figures rc=0; "
                                 "abdomen_geometry_stats rc=0")
    assert rw.LAST_ABDOMEN_SUMMARY == result["summary"] and "rc=75" not in result["summary"]
    assert result["error"] == "" and result["timed_out"] is False


def test_an_exit_75_of_any_other_command_is_an_ordinary_failure(tmp_path, monkeypatch, caplog):
    procs = _Procs(monkeypatch, script=[0, 75, 0])
    with caplog.at_level(logging.INFO):
        result = _run(tmp_path)
    assert [c["yielded"] for c in result["commands"]] == [False, False, False]
    assert _warnings(caplog) == ["[abdomen] normalize+wide+figures exited with code 75"]
    assert "normalize+wide+figures rc=75" in result["summary"]


def test_real_processes_that_all_exit_75_yield_only_for_track(tmp_path):
    if shutil.which("bash") is None:
        pytest.skip("needs bash")
    tempfail = tmp_path / "tempfail_python"
    tempfail.write_text("#!/bin/bash\nexit 75\n", encoding="utf-8")
    tempfail.chmod(tempfail.stat().st_mode | stat.S_IXUSR)
    result = _run(tmp_path, _layers(tmp_path, {"enabled": True, "python": str(tempfail)}))
    assert [c["returncode"] for c in result["commands"]] == [75, 75, 75]
    assert [c["yielded"] for c in result["commands"]] == [True, False, False]


def test_a_command_that_cannot_start_stops_the_stage_after_the_ones_that_ran(tmp_path, monkeypatch):
    procs = _Procs(monkeypatch, script=[0, FileNotFoundError("no such interpreter")])
    result = _run(tmp_path)
    assert len(procs.calls) == 2 and "FileNotFoundError" in result["error"]
    assert [c["ran"] for c in result["commands"]][:2] == [True, False]
    assert result["ran"] is True


def test_an_unusable_lock_directory_is_a_warning_not_an_unguarded_run(tmp_path, procs):
    (tmp_path / "cache").write_text("a file where the cache dir should be")
    result = _run(tmp_path)
    assert procs.calls == [] and result["ran"] is False and result["error"]


def test_ctrl_c_during_a_command_still_stops_its_process_group_and_frees_the_lock(tmp_path, monkeypatch):
    procs = _Procs(monkeypatch, script=[KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path)
    assert procs.killed == [(4201, signal.SIGTERM)], "a stage must not outlive the run that started it"
    with abdomen_lock(tmp_path / "cache") as got:
        assert got is True


# ---- the stage: a killed driver must not orphan its commands ------------------------------------------------

def test_every_stage_command_is_spawned_to_die_with_the_driver(tmp_path, procs, monkeypatch):
    sentinel = object()
    monkeypatch.setattr(rw, "die_with_parent_preexec", lambda: sentinel)
    _run(tmp_path, _layers(tmp_path, mounting=MOUNTING))
    assert len(procs.kwargs) == 4
    for kwargs in procs.kwargs:
        assert kwargs["preexec_fn"] is sentinel and kwargs["start_new_session"] is True


@pytest.fixture
def protected_signals():
    """A no-op Python handler for SIGTERM and SIGHUP, so that a test that sends one to itself cannot kill pytest when the
    code under test installs no handler of its own. ``hits`` lists what reached it; the old handlers come back after."""
    hits = []
    old = {sig: signal.signal(sig, lambda signum, frame: hits.append(signum)) for sig in (signal.SIGTERM, signal.SIGHUP)}
    protector = {sig: signal.getsignal(sig) for sig in old}
    yield SimpleNamespace(hits=hits, protector=protector)
    for sig, handler in old.items():
        signal.signal(sig, handler)


def test_sigterm_and_sighup_are_handled_only_while_the_stage_runs(tmp_path, monkeypatch, protected_signals):
    seen = []

    class Probe(_Procs):
        def _popen(self, cmd, **kwargs):
            seen.append({sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)})
            return super()._popen(cmd, **kwargs)

    Probe(monkeypatch)
    assert {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)} == protected_signals.protector
    _run(tmp_path)
    for during in seen:
        for sig, handler in during.items():
            assert callable(handler) and handler is not protected_signals.protector[sig], "the stage's own handler"
    assert {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)} == protected_signals.protector


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP], ids=["SIGTERM", "SIGHUP"])
def test_a_signal_to_the_driver_kills_the_group_of_the_running_command_and_frees_the_lock(
        tmp_path, monkeypatch, protected_signals, sig):
    procs = _Procs(monkeypatch, script=[("signal", sig)])
    with pytest.raises(SystemExit) as stop:
        _run(tmp_path)
    assert stop.value.code == 128 + sig, "the exit status a shell reports for a signal death"
    assert procs.killed == [(4201, signal.SIGTERM)], "the whole group, by the finally of the command's wait"
    assert len(procs.calls) == 1, "nothing else is started"
    assert protected_signals.hits == [], "the stage's handler took it, not whatever was there before"
    assert {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)} == protected_signals.protector
    with abdomen_lock(tmp_path / "cache") as got:
        assert got is True


def test_a_second_signal_while_the_group_is_being_killed_does_not_interrupt_the_kill(
        tmp_path, monkeypatch, protected_signals):
    procs = _Procs(monkeypatch, script=[("signal twice", signal.SIGTERM)])
    with pytest.raises(SystemExit) as stop:
        _run(tmp_path)
    assert stop.value.code == 128 + signal.SIGTERM
    assert procs.procs[0].waits == [34200.0, 30.0], "the grace wait ran to its end"
    assert procs.procs[0].returncode == -15 and procs.killed == [(4201, signal.SIGTERM)]


def test_the_handlers_come_back_when_the_stage_fails_on_its_own_too(tmp_path, monkeypatch, protected_signals):
    _Procs(monkeypatch, script=[OSError("disk gone")])
    assert _run(tmp_path)["error"].startswith("OSError")
    assert {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)} == protected_signals.protector


def test_a_stage_that_never_starts_leaves_the_signals_alone(tmp_path, procs, protected_signals):
    with abdomen_lock(tmp_path / "cache"):
        _run(tmp_path)                                        # skipped: the lock is held
    assert {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)} == protected_signals.protector


def _driver_script(tmp_path: Path, python: Path) -> str:
    """A driver-like process: it runs the real stage, one enabled abdomen block, with *python* as the interpreter."""
    return "\n".join([
        "import sys",
        "from pathlib import Path",
        "from types import MappingProxyType",
        f"sys.path[:0] = [{str(rw.REPO_ROOT / 'src')!r}, {str(rw.REPO_ROOT)!r}]",
        "from fbpipe.config import ReactionPredictionSettings, Settings",
        "from fbpipe.config_layers import ConfigLayers",
        "from scripts.pipeline import run_workflows as rw",
        f"tmp = Path({str(tmp_path)!r})",
        "settings = Settings(model_path='m.pt', main_directories=[], cache_dir=str(tmp / 'cache'),",
        "                    reaction_prediction=ReactionPredictionSettings(output_csv='/x/p.csv'))",
        f"block = {{'enabled': True, 'python': {str(python)!r}, 'max_hours': 1}}",
        "layers = ConfigLayers(tmp / 'config_new_abdomen.yaml', tmp / 'config_new.yaml',",
        "                      MappingProxyType({'abdomen': block}))",
        "print('driver ready', flush=True)",
        "rw._run_abdomen_stages(layers, settings)",
        "print('stage returned', flush=True)",
    ])


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP], ids=["SIGTERM", "SIGHUP"])
def test_a_driver_sent_sigterm_or_sighup_takes_its_stage_command_and_that_commands_children_down(
        tmp_path, fake_python, sig):
    """A real driver-like process runs the real stage with a fake interpreter whose track command starts a ``sleep`` of its
    own (a child in the command's process group). A signal to the driver must leave neither behind, and the lock free."""
    env = dict(os.environ, MPLBACKEND="Agg", FBPIPE_DISABLE_NTFY="1")
    driver = subprocess.Popen([sys.executable, "-c", _driver_script(tmp_path, fake_python)], env=env,
                              stdout=subprocess.PIPE, text=True)
    sleeper = command = None
    try:
        line = ""
        while line.strip() != "driver ready":                          # importing the driver prints its own lines first
            line = driver.stdout.readline()
            assert line, "the driver exited before it was ready"
        pidfile = tmp_path / "sleep.pid"
        deadline = time.monotonic() + 30
        while not (pidfile.exists() and pidfile.read_text().strip()) and time.monotonic() < deadline:
            time.sleep(0.05)
        sleeper = int(pidfile.read_text())
        command = int(Path(f"/proc/{sleeper}/stat").read_text().rsplit(")", 1)[1].split()[1])      # its parent: the command
        assert _alive(sleeper) and _alive(command)
        driver.send_signal(sig)
        assert driver.wait(timeout=30) == 128 + sig
        deadline = time.monotonic() + 10
        while (_alive(sleeper) or _alive(command)) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _alive(command), "the stage command outlived the driver"
        assert not _alive(sleeper), "the command's own child outlived the driver"
        with abdomen_lock(tmp_path / "cache") as got:
            assert got is True, "the abdomen lock is free again"
    finally:
        if driver.poll() is None:
            driver.kill()
        driver.wait()
        for pid in (sleeper, command):
            if pid and _alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


# ---- the stage: no reading of a predictions file that PER is rewriting ------------------------------------------

def test_a_running_per_run_holds_back_the_commands_that_read_the_predictions_and_only_those(tmp_path, procs, caplog):
    with per_run_lock(tmp_path / "cache"), caplog.at_level(logging.INFO):
        result = _run(tmp_path, _layers(tmp_path, mounting=MOUNTING))
    assert procs.calls == [_expected(tmp_path)[0], _expected(tmp_path)[1]], "track and normalize read no predictions"
    assert [c["ran"] for c in result["commands"]] == [True, True, False, False]
    assert [c["skipped"] for c in result["commands"]] == ["", "", "PER run active", "PER run active"]
    warnings = _warnings(caplog)
    assert "[abdomen] PER run active — skipping abdomen_geometry_stats: model_predictions.csv may be mid-write" in warnings
    assert "[abdomen] PER run active — skipping mounting_posture: model_predictions.csv may be mid-write" in warnings
    assert result["summary"] == ("ABDOMEN: track rc=0; normalize+wide+figures rc=0; "
                                 "abdomen_geometry_stats skipped (PER run active); mounting_posture skipped (PER run active)")
    assert rw.LAST_ABDOMEN_SUMMARY == result["summary"]


def test_without_a_per_run_the_commands_that_read_the_predictions_run(tmp_path, procs):
    result = _run(tmp_path, _layers(tmp_path, mounting=MOUNTING))
    assert len(procs.calls) == 4 and all(c["skipped"] == "" for c in result["commands"])


def test_the_per_run_is_asked_before_each_of_those_commands_not_once_at_the_start(tmp_path, monkeypatch):
    """A track of hours can outlast the start of a PER run: the lock is looked at when the stats are about to start."""
    held = []

    class Probe(_Procs):
        def _popen(self, cmd, **kwargs):
            if len(self.calls) == 1:                       # the second command (normalize) starts: PER begins now
                held.append(acquire_per_run_lock(tmp_path / "cache"))
            return super()._popen(cmd, **kwargs)

    procs = Probe(monkeypatch)
    try:
        result = _run(tmp_path, _layers(tmp_path, mounting=MOUNTING))
    finally:
        for handle in held:
            release_run_lock(handle)
    assert len(procs.calls) == 2 and [c["skipped"] for c in result["commands"]] == ["", "", "PER run active", "PER run active"]


def test_a_stage_with_only_the_mounting_command_is_held_back_too(tmp_path, procs):
    with per_run_lock(tmp_path / "cache"):
        result = _run(tmp_path, _layers(tmp_path, False, mounting=MOUNTING))
    assert procs.calls == [] and result["ran"] is False
    assert result["summary"] == "ABDOMEN: mounting_posture skipped (PER run active)"


def test_a_per_lock_that_is_gone_again_lets_the_next_run_read_the_predictions(tmp_path, procs):
    with per_run_lock(tmp_path / "cache"):
        assert _run(tmp_path, _layers(tmp_path, mounting=MOUNTING))["commands"][3]["skipped"] == "PER run active"
    procs.calls.clear()
    assert _run(tmp_path, _layers(tmp_path, mounting=MOUNTING))["commands"][3]["skipped"] == ""
    assert len(procs.calls) == 4


def test_the_summary_is_one_line(tmp_path, procs):
    result = _run(tmp_path)
    assert result["summary"] == "ABDOMEN: track rc=0; normalize+wide+figures rc=0; abdomen_geometry_stats rc=0"
    assert rw.LAST_ABDOMEN_SUMMARY == result["summary"]


# ---- the stage: the mounting command -----------------------------------------------------------------------

def _overlay_path(tmp_path) -> Path:
    return tmp_path / "config_new_abdomen.yaml"


def test_the_stage_runs_the_mounting_command_fourth_in_its_own_session_like_the_others(tmp_path, procs):
    result = _run(tmp_path, _layers(tmp_path, mounting=MOUNTING))
    assert procs.calls == [*_expected(tmp_path), _mounting_cmd(sys.executable, _overlay_path(tmp_path))]
    assert [c["label"] for c in result["commands"]] == [
        "track", "normalize+wide+figures", "abdomen_geometry_stats", "mounting_posture"]
    assert [c["returncode"] for c in result["commands"]] == [0, 0, 0, 0]
    for kwargs in procs.kwargs:
        assert kwargs["start_new_session"] is True and kwargs["env"]["MPLBACKEND"] == "Agg"
    assert result["summary"] == ("ABDOMEN: track rc=0; normalize+wide+figures rc=0; abdomen_geometry_stats rc=0; "
                                 "mounting_posture rc=0")
    assert rw.LAST_ABDOMEN_SUMMARY == result["summary"]


def test_a_mounting_block_alone_is_enough_for_the_stage_to_run(tmp_path, procs):
    result = _run(tmp_path, _layers(tmp_path, False, mounting=MOUNTING))
    assert procs.calls == [_mounting_cmd(sys.executable, _overlay_path(tmp_path))]
    assert result["skipped_reason"] == "" and result["error"] == "" and result["ran"] is True
    assert [c["label"] for c in result["commands"]] == ["mounting_posture"]
    assert procs.procs[0].waits == [36000.0], "no abdomen block: the default 10 h budget"
    assert procs.kwargs[0]["start_new_session"] is True
    assert result["summary"] == "ABDOMEN: mounting_posture rc=0"


def test_a_disabled_abdomen_block_still_supplies_the_interpreter_and_the_budget(tmp_path, procs):
    block = {"enabled": False, "python": "/cam/bin/python", "max_hours": 2}
    result = _run(tmp_path, _layers(tmp_path, block, mounting=MOUNTING))
    assert procs.calls == [_mounting_cmd("/cam/bin/python", _overlay_path(tmp_path))]
    assert procs.procs[0].waits == [7200.0] and result["ran"] is True


def test_the_mounting_command_still_runs_under_figures_only(tmp_path, procs):
    _run(tmp_path, _layers(tmp_path, False, mounting=MOUNTING), figures_only=True)
    assert procs.calls == [_mounting_cmd(sys.executable, _overlay_path(tmp_path))]
    procs.calls.clear()
    _run(tmp_path, _layers(tmp_path, mounting=MOUNTING), figures_only=True)
    mounting = _mounting_cmd(sys.executable, _overlay_path(tmp_path))
    assert procs.calls == [*_expected(tmp_path, figures_only=True), mounting]
    assert len(procs.calls) == 3, "track is dropped; the mounting command is not"


@pytest.mark.parametrize("abdomen", [False, {"enabled": True}], ids=["mounting alone", "with the abdomen commands"])
def test_a_folder_run_skips_the_mounting_command_with_the_rest(tmp_path, procs, abdomen):
    result = _run(tmp_path, _layers(tmp_path, abdomen, mounting=MOUNTING), folder="Hex-Training-24")
    assert result["skipped_reason"] == "folder" and procs.calls == []


def test_a_nonzero_mounting_exit_is_a_warning_like_the_others(tmp_path, monkeypatch, caplog):
    procs = _Procs(monkeypatch, script=[0, 0, 0, 2])
    with caplog.at_level(logging.INFO):
        result = _run(tmp_path, _layers(tmp_path, mounting=MOUNTING))
    assert len(procs.calls) == 4 and [c["returncode"] for c in result["commands"]] == [0, 0, 0, 2]
    assert _warnings(caplog) == ["[abdomen] mounting_posture exited with code 2"]
    assert result["summary"].endswith("mounting_posture rc=2") and result["error"] == ""


def test_a_failed_abdomen_command_does_not_stop_the_mounting_command_from_reading_what_exists(tmp_path, monkeypatch):
    procs = _Procs(monkeypatch, script=[1, 1, 1, 0])
    result = _run(tmp_path, _layers(tmp_path, mounting=MOUNTING))
    assert len(procs.calls) == 4 and procs.calls[3][1] == MOUNTING_SCRIPT
    assert [c["returncode"] for c in result["commands"]] == [1, 1, 1, 0]


def test_the_mounting_command_shares_the_one_time_budget(tmp_path, monkeypatch):
    procs = _Procs(monkeypatch, elapse=[3000.0, 100.0, 0.0, 0.0])
    _run(tmp_path, _layers(tmp_path, {"enabled": True, "max_hours": 1}, mounting=MOUNTING))
    assert [p.waits for p in procs.procs] == [[3240.0], [600.0], [500.0], [500.0]]


def test_a_budget_used_up_before_the_mounting_command_starts_nothing_more(tmp_path, monkeypatch):
    procs = _Procs(monkeypatch, elapse=[0.0, 0.0, 4000.0])              # the stats end 400 s past the hour
    result = _run(tmp_path, _layers(tmp_path, {"enabled": True, "max_hours": 1}, mounting=MOUNTING))
    assert len(procs.calls) == 3 and result["timed_out"] is False
    assert [c["ran"] for c in result["commands"]] == [True, True, True, False]
    assert "mounting_posture not run" in result["summary"]


def test_a_mounting_command_that_outlasts_the_budget_is_killed_with_its_group(tmp_path, monkeypatch):
    procs = _Procs(monkeypatch, script=[0, 0, 0, "timeout"])
    result = _run(tmp_path, _layers(tmp_path, mounting=MOUNTING))
    assert procs.killed == [(4204, signal.SIGTERM)] and result["timed_out"] is True
    assert result["commands"][3]["timed_out"] is True and "mounting_posture TIMED OUT" in result["summary"]


def test_the_mounting_command_runs_under_the_exclusive_abdomen_lock(tmp_path, monkeypatch):
    seen = []

    class Probe(_Procs):
        def _popen(self, cmd, **kwargs):
            with abdomen_lock(tmp_path / "cache") as got:
                seen.append(got)
            return super()._popen(cmd, **kwargs)

    Probe(monkeypatch)
    _run(tmp_path, _layers(tmp_path, False, mounting=MOUNTING))
    assert seen == [False], "another abdomen run cannot start while the mounting command reads the outputs"
    with abdomen_lock(tmp_path / "cache") as got:
        assert got is True


def test_a_held_abdomen_lock_holds_back_a_mounting_only_stage_too(tmp_path, procs):
    with abdomen_lock(tmp_path / "cache") as got:
        assert got is True
        result = _run(tmp_path, _layers(tmp_path, False, mounting=MOUNTING))
    assert result["skipped_reason"] == "locked" and procs.calls == []


def test_a_refused_mounting_block_alone_leaves_nothing_to_run_and_says_why(tmp_path, procs, caplog):
    block = _with(abdomen_out_dir=f"{HAND_RUN}/Abdomen-Movement")
    with caplog.at_level(logging.INFO):
        result = _run(tmp_path, _layers(tmp_path, False, mounting=block))
    assert result["skipped_reason"] == "nothing to run" and procs.calls == [] and result["ran"] is False
    assert any(w.startswith("[mounting]") and "mounting.abdomen_out_dir" in w for w in _warnings(caplog))


def test_a_refused_mounting_block_does_not_stop_the_abdomen_commands(tmp_path, procs, caplog):
    block = _with(out_dir="/tmp/not-the-pipelines-root")
    with caplog.at_level(logging.INFO):
        result = _run(tmp_path, _layers(tmp_path, mounting=block))
    assert procs.calls == _expected(tmp_path) and result["error"] == ""
    assert any("mounting.out_dir" in w for w in _warnings(caplog))


def test_an_empty_predictions_path_leaves_a_mounting_only_stage_with_nothing_to_run(tmp_path, procs, caplog):
    with caplog.at_level(logging.INFO):
        result = _run(tmp_path, _layers(tmp_path, False, mounting=MOUNTING), settings=_settings(tmp_path, ""))
    assert result["skipped_reason"] == "nothing to run" and procs.calls == []
    assert any("reaction_prediction.output_csv" in w for w in _warnings(caplog))


def test_a_mounting_block_that_does_not_parse_never_raises_out_of_the_stage(tmp_path, procs, caplog):
    with caplog.at_level(logging.INFO):
        result = _run(tmp_path, _layers(tmp_path, mounting=_with(typo_key=1)))
    assert procs.calls == _expected(tmp_path), "a typo in the mounting block must not cost the multi-hour tracking"
    assert result["error"] == "" and any("typo_key" in w for w in _warnings(caplog))


@pytest.mark.parametrize("change, error", [
    ({"out_dir": "~zzzznouser/x"}, "RuntimeError"),
    ({"csv_out_dir": "~zzzznouser/x"}, "RuntimeError"),
    ({"n_permutations": 10**400}, "OverflowError"),
], ids=["no such home in out_dir", "no such home in csv_out_dir", "int beyond the float range"])
def test_a_mounting_value_that_raises_no_value_error_costs_only_the_mounting_command(
        tmp_path, procs, caplog, change, error):
    with caplog.at_level(logging.INFO):
        result = _run(tmp_path, _layers(tmp_path, mounting=_with(**change)))
    assert procs.calls == _expected(tmp_path) and len(procs.calls) == 3, "the abdomen commands all still run"
    assert [c["label"] for c in result["commands"]] == ["track", "normalize+wide+figures", "abdomen_geometry_stats"]
    assert result["error"] == "" and any(w.startswith("[mounting]") and error in w for w in _warnings(caplog))


def test_a_mounting_only_stage_with_such_a_value_has_nothing_to_run_and_does_not_raise(tmp_path, procs, caplog):
    with caplog.at_level(logging.INFO):
        result = _run(tmp_path, _layers(tmp_path, False, mounting=_with(out_dir="~zzzznouser/x")))
    assert result["skipped_reason"] == "nothing to run" and procs.calls == [] and result["error"] == ""
    assert any("RuntimeError" in w for w in _warnings(caplog))


def test_an_abdomen_interpreter_whose_home_does_not_exist_is_a_warning_and_nothing_runs_not_an_exception(
        tmp_path, procs, caplog):
    with caplog.at_level(logging.INFO):
        result = _run(tmp_path, _layers(tmp_path, {"enabled": True, "python": "~zzzznouser/py"}, mounting=MOUNTING))
    assert procs.calls == [] and result["ran"] is False and result["error"].startswith("RuntimeError:")
    assert any(r.levelname == "WARNING" and "RuntimeError" in r.getMessage() for r in caplog.records)
    assert rw.LAST_ABDOMEN_SUMMARY.startswith("ABDOMEN: stopped: RuntimeError")


# ---- the commands against the real CLI and real processes --------------------------------------------------

def test_the_module_commands_parse_with_the_geometry_clis_own_parser():
    parser = geometry_pipeline.build_parser()
    track = parser.parse_args(_cmds(annotated_video=True, n_workers=2)[0][3:])
    assert (track.config, track.workers, track.annotated_video, track.steps) == (str(OVERLAY), 2, True, ["track"])
    later = parser.parse_args(_cmds()[1][3:])
    assert later.config == str(OVERLAY) and later.steps == ["normalize", "wide", "figures"]
    assert later.workers is None and later.annotated_video is False


def test_the_mounting_command_parses_with_the_scripts_own_parser(monkeypatch):
    """A flag the script does not know exits 2 -- a warning in the stage -- and the analysis would silently never
    run. The script builds its parser inside ``main``, so parsing is observed there and stopped right after."""
    from scripts.analysis import mounting_posture

    class Parsed(Exception):
        pass

    real, seen = argparse.ArgumentParser.parse_args, []

    def parse_then_stop(self, args=None, namespace=None):
        seen.append(real(self, args, namespace))
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_then_stop)
    with pytest.raises(Parsed):
        mounting_posture.main(_cmds(BOTH)[3][2:])
    (parsed,) = seen
    assert (parsed.config, parsed.predictions_csv, parsed.out_dir, parsed.abdomen_out_dir, parsed.csv_out_dir) == (
        str(OVERLAY), PREDICTIONS, MOUNTING["out_dir"], MOUNTING["abdomen_out_dir"], MOUNTING["csv_out_dir"])


FAKE_PYTHON = """#!/bin/bash
echo "$@" >> "$FAKE_LOG"
case "$*" in
  *" track") sleep 300 & echo $! > "$FAKE_PIDFILE"; wait ;;
  *) exit "${FAKE_EXIT:-0}" ;;
esac
"""


@pytest.fixture
def fake_python(tmp_path, monkeypatch):
    """An executable that stands in for the interpreter (the real tracker is never run): it logs its arguments and
    either exits ($FAKE_EXIT) or, for the track command, starts a background ``sleep`` and waits for it."""
    if shutil.which("bash") is None:
        pytest.skip("needs bash")
    script = tmp_path / "fake_python"
    script.write_text(FAKE_PYTHON, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "argv.log"))
    monkeypatch.setenv("FAKE_PIDFILE", str(tmp_path / "sleep.pid"))
    return script


def _alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def test_real_processes_get_exactly_the_planned_command_lines_and_a_failure_does_not_stop_the_rest(
        tmp_path, fake_python, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT", "3")
    result = _run(tmp_path, _layers(tmp_path, {"enabled": True, "python": str(fake_python)}), figures_only=True)
    overlay = tmp_path / "config_new_abdomen.yaml"
    assert (tmp_path / "argv.log").read_text().splitlines() == [
        f"-m fbpipe.geometry_pipeline --config {overlay} normalize wide figures",
        f"{rw.REPO_ROOT / 'scripts' / 'analysis' / 'abdomen_geometry_stats.py'} --config {overlay} "
        f"--predictions {PREDICTIONS}",
    ]
    assert [c["returncode"] for c in result["commands"]] == [3, 3]


def test_a_real_process_gets_exactly_the_mounting_command_line_last_and_its_failure_is_only_a_warning(
        tmp_path, fake_python, monkeypatch, caplog):
    monkeypatch.setenv("FAKE_EXIT", "2")
    layers = _layers(tmp_path, {"enabled": True, "python": str(fake_python)}, mounting=MOUNTING)
    with caplog.at_level(logging.INFO):
        result = _run(tmp_path, layers, figures_only=True)
    logged = (tmp_path / "argv.log").read_text().splitlines()
    assert len(logged) == 3 and logged[-1] == " ".join(_mounting_cmd(fake_python, _overlay_path(tmp_path))[1:])
    assert [c["returncode"] for c in result["commands"]] == [2, 2, 2] and result["error"] == ""
    assert "[abdomen] mounting_posture exited with code 2" in _warnings(caplog)


def test_a_real_timeout_kills_the_whole_process_group_including_what_the_command_started(
        tmp_path, fake_python, monkeypatch):
    monkeypatch.setattr(rw, "ABDOMEN_TAIL_FRACTION", 0.5)              # a roomy tail, so a loaded machine cannot eat it
    result = _run(tmp_path, _layers(tmp_path, {"enabled": True, "python": str(fake_python), "max_hours": 0.0006}))
    sleeper = int((tmp_path / "sleep.pid").read_text())
    deadline = time.monotonic() + 5
    while _alive(sleeper) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert result["timed_out"] is True and result["commands"][0]["returncode"] == -signal.SIGTERM
    assert not _alive(sleeper), "the track command's own child must not outlive the stage"
    assert len((tmp_path / "argv.log").read_text().splitlines()) == 3, "the tail is left to the commands after track"
    assert [c["returncode"] for c in result["commands"]] == [-signal.SIGTERM, 0, 0]
    with abdomen_lock(tmp_path / "cache") as got:
        assert got is True


def test_the_default_worker_count_is_the_geometry_loaders_own():
    from fbpipe.geometry.config import ABDOMEN_YOLO_DEFAULTS
    assert rw.ABDOMEN_NUM_WORKERS_DEFAULT == ABDOMEN_YOLO_DEFAULTS["num_workers"]


# ---- main(): flags and source order ------------------------------------------------------------------------

def test_the_abdomen_flags_are_off_unless_given():
    parser = rw._build_arg_parser()
    off = parser.parse_args([])
    assert off.abdomen_only is False and off.abdomen_video is False
    on = parser.parse_args(["--abdomen-only", "--abdomen-video"])
    assert on.abdomen_only is True and on.abdomen_video is True


def test_importing_the_driver_does_not_load_the_mounting_analysis():
    """A plain nightly never builds a mounting command, so an error in ``fbpipe.analysis.mounting`` (a syntax error, a
    scipy that will not import) must not be able to stop it before it starts: the driver imports the analysis only
    inside the builder. Checked in a fresh interpreter, where nothing else has imported it."""
    env = dict(os.environ, MPLBACKEND="Agg", FBPIPE_DISABLE_NTFY="1",
               PYTHONPATH=os.pathsep.join([str(rw.REPO_ROOT / "src"), str(rw.REPO_ROOT)]))
    done = subprocess.run(
        [sys.executable, "-c", "import sys, scripts.pipeline.run_workflows; "
                               "print('mounting loaded:', 'fbpipe.analysis.mounting' in sys.modules)"],
        cwd=rw.REPO_ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[-1] == "mounting loaded: False", done.stdout


def test_a_mounting_analysis_that_cannot_be_imported_costs_only_the_mounting_command(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "fbpipe.analysis.mounting", None)    # what importing a module that is broken halts on
    with caplog.at_level(logging.INFO):
        assert _mounting() is None
        assert _cmds(BOTH) == [TRACK, NORMALIZE, STATS_CMD], "the abdomen commands share the stage and must survive"
    warnings = _warnings(caplog)
    assert len(warnings) == 2 and all("[mounting] not run" in w and "fbpipe.analysis.mounting" in w for w in warnings)


def _main_source() -> str:
    return inspect.getsource(rw.main)


def test_the_layers_are_resolved_before_any_config_is_read_and_per_gets_the_base_path():
    src = _main_source()
    at = src.index("resolve_layers(")
    assert at < src.index("yaml.safe_load(") and at < src.index("load_settings(")
    assert src.index("config_path = layers.base_path") < src.index("yaml.safe_load(")


def _main_def() -> ast.FunctionDef:
    return ast.parse(inspect.getsource(rw.main)).body[0]


def _is_call(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return sorted((n for n in ast.walk(tree) if _is_call(n, name)), key=lambda n: (n.lineno, n.col_offset))


def _whole_run_with(main: ast.FunctionDef) -> ast.With:
    (block,) = [n for n in ast.walk(main)
                if isinstance(n, ast.With) and any(_is_call(item.context_expr, "run_lock") for item in n.items)]
    return block


def test_the_abdomen_only_early_return_precedes_both_locks_and_all_per_work():
    src = _main_source()
    early, whole_run = src.index("if args.abdomen_only:"), src.index("with run_lock(")
    assert src.index("_validate_thaw(") < early < whole_run < src.index("_main_locked(")
    assert early < src.index("acquire_per_run_lock("), "an --abdomen-only run is not a PER run: no PER lock"
    body = src[early:whole_run]
    assert "_run_abdomen_stages(" in body and re.search(r"^\s+return\s*$", body, re.MULTILINE), \
        "a bare return, so the legacy shim's SystemExit(main()) exits 0"
    assert "no abdomen/mounting block" in body


def test_the_per_lock_is_taken_inside_the_whole_run_lock_around_the_per_work_and_released_in_a_finally():
    main = _main_def()
    whole_run = _whole_run_with(main)
    (acquire,), (per_work,), (release,) = (_calls(main, name) for name in
                                           ("acquire_per_run_lock", "_main_locked", "release_run_lock"))
    for call in (acquire, per_work, release):
        assert whole_run.lineno < call.lineno <= whole_run.end_lineno
    assert acquire.lineno < per_work.lineno < release.lineno
    (guard,) = [n for n in ast.walk(whole_run)
                if isinstance(n, ast.Try) and any(release in ast.walk(stmt) for stmt in n.finalbody)]
    assert any(per_work in ast.walk(stmt) for stmt in guard.body), \
        "a crashing PER run must not leave the lock to the process' exit"


def test_the_nightly_body_knows_nothing_of_the_overlay_or_the_per_lock():
    """_main_locked is main's own code: everything about the overlay lives in main(), around it."""
    names = {n.id for n in ast.walk(ast.parse(inspect.getsource(rw._main_locked))) if isinstance(n, ast.Name)}
    assert not names & {"layers", "per_lock", "acquire_per_run_lock", "release_run_lock", "_run_abdomen_stages"}


def test_the_abdomen_stage_follows_the_per_run_outside_both_locks_and_only_runs_for_an_overlay():
    main = _main_def()
    whole_run = _whole_run_with(main)
    stage_calls = _calls(main, "_run_abdomen_stages")
    assert len(stage_calls) == 2, "the --abdomen-only run and the hook after a full run"
    hook = stage_calls[-1]
    assert _calls(main, "_main_locked")[0].lineno < whole_run.end_lineno < hook.lineno, \
        "an abdomen stage may run for hours: it must not hold the whole-run lock the next PER run waits on"
    (guard,) = [n for n in main.body if isinstance(n, ast.If) and ast.unparse(n.test) == "layers.overlay"]
    assert hook in list(ast.walk(guard)), "a plain config_new.yaml run must not even call the stage"


# ---- main(): --abdomen-only --------------------------------------------------------------------------------

def _configs(tmp_path, overlay=None, **base_extra):
    """A minimal base config_new.yaml (plus *base_extra* keys) and the overlay that extends it (default: an enabled
    abdomen block)."""
    base = {"protocol": "v2", "model_path": "m.pt", "datasets": [], "cache_dir": str(tmp_path / "cache")}
    base.update(base_extra)
    (tmp_path / "config_new.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    overlay = {"extends": "config_new.yaml", "abdomen": {"enabled": True}} if overlay is None else overlay
    (tmp_path / "config_new_abdomen.yaml").write_text(yaml.safe_dump(overlay), encoding="utf-8")
    return tmp_path / "config_new.yaml", tmp_path / "config_new_abdomen.yaml"


def _forbid_per_work(monkeypatch):
    def forbidden(name):
        def fail(*a, **k):
            raise AssertionError(f"{name} must not run under --abdomen-only")
        return fail

    for name in ("_run_pipeline", "_run_combined", "_run_reactions", "_run_light_only_traces", "_run_envelope_visuals",
                 "_run_training", "_run_dataset_means", "_run_secure_cleanup", "_batch_copy_to_smb",
                 "_run_data_secured_sync", "_eye_qc_summary", "acquire_per_run_lock"):
        monkeypatch.setattr(rw, name, forbidden(name))


def test_abdomen_only_calls_only_the_abdomen_stage_and_returns_none(tmp_path, monkeypatch):
    base, overlay = _configs(tmp_path)
    _forbid_per_work(monkeypatch)
    calls = []
    monkeypatch.setattr(rw, "_run_abdomen_stages", lambda layers, settings, **kw: calls.append((layers, settings, kw)) or {})
    assert rw.main(["--config", str(overlay), "--abdomen-only"]) is None
    ((layers, settings, kw),) = calls
    assert layers.path == overlay and layers.base_path == base.resolve()
    assert kw == {"figures_only": False, "folder": None, "annotated_video": False}
    assert settings.cache_dir == str(tmp_path / "cache")
    assert rw.LAST_RUN_ABDOMEN_ONLY is True
    assert not (tmp_path / "cache" / "locks" / "per.lock").exists(), "an --abdomen-only run is not a PER run"


def test_abdomen_only_forwards_the_video_figures_only_and_folder_flags(tmp_path, monkeypatch):
    _, overlay = _configs(tmp_path)
    kws = []
    monkeypatch.setattr(rw, "_run_abdomen_stages", lambda layers, settings, **kw: kws.append(kw) or {})
    rw.main(["--config", str(overlay), "--abdomen-only", "--abdomen-video", "--figures-only", "--folder", "X"])
    assert kws == [{"figures_only": True, "folder": "X", "annotated_video": True}]


def test_abdomen_only_runs_the_real_stage_on_the_overlay_with_a_fake_popen(tmp_path, procs):
    _, overlay = _configs(tmp_path, {"extends": "config_new.yaml",
                                     "abdomen": {"enabled": True, "yolo": {"num_workers": 2}}})
    assert rw.main(["--config", str(overlay), "--abdomen-only", "--abdomen-video"]) is None
    assert procs.calls == rw._abdomen_commands({"abdomen": {"enabled": True}}, _base_settings(""),
                                               python_exec=sys.executable, overlay_path=overlay, figures_only=False,
                                               annotated_video=True, n_workers=2)     # this base has no predictions CSV
    assert rw.LAST_ABDOMEN_SUMMARY.startswith("ABDOMEN: track rc=0") and rw.LAST_RUN_ABDOMEN_ONLY is True


@pytest.mark.parametrize("flags", [[], ["--figures-only"]], ids=["full", "figures-only"])
def test_abdomen_only_on_a_mounting_only_overlay_runs_the_mounting_command_on_the_base_predictions(
        tmp_path, procs, flags):
    _, overlay = _configs(tmp_path, {"extends": "config_new.yaml", "mounting": MOUNTING},
                          reaction_prediction={"output_csv": PREDICTIONS})
    assert rw.main(["--config", str(overlay), "--abdomen-only", *flags]) is None
    assert procs.calls == [_mounting_cmd(sys.executable, overlay)], "the overlay is --config; the CSV is the base's"
    assert rw.LAST_ABDOMEN_SUMMARY == "ABDOMEN: mounting_posture rc=0" and rw.LAST_RUN_ABDOMEN_ONLY is True


def test_abdomen_only_with_both_blocks_runs_the_mounting_command_after_the_abdomen_ones(tmp_path, procs):
    _, overlay = _configs(tmp_path, {"extends": "config_new.yaml", "abdomen": {"enabled": True}, "mounting": MOUNTING},
                          reaction_prediction={"output_csv": PREDICTIONS})
    rw.main(["--config", str(overlay), "--abdomen-only", "--figures-only"])
    assert [cmd[1] for cmd in procs.calls] == ["-m", STATS, MOUNTING_SCRIPT]
    assert procs.calls[-1] == _mounting_cmd(sys.executable, overlay)


def test_abdomen_only_without_a_predictions_csv_in_the_base_config_warns_and_runs_nothing(tmp_path, procs, caplog):
    _, overlay = _configs(tmp_path, {"extends": "config_new.yaml", "mounting": MOUNTING})
    with caplog.at_level(logging.INFO):
        assert rw.main(["--config", str(overlay), "--abdomen-only"]) is None
    assert procs.calls == [] and "reaction_prediction.output_csv" in " ".join(_warnings(caplog))
    assert rw.LAST_ABDOMEN_SUMMARY.startswith("ABDOMEN: skipped")


def test_abdomen_only_with_an_unreadable_mounting_value_returns_normally_after_the_abdomen_commands(
        tmp_path, procs, caplog):
    blocks = {"extends": "config_new.yaml", "abdomen": {"enabled": True}, "mounting": _with(out_dir="~zzzznouser/x")}
    _, overlay = _configs(tmp_path, blocks, reaction_prediction={"output_csv": PREDICTIONS})
    with caplog.at_level(logging.INFO):
        assert rw.main(["--config", str(overlay), "--abdomen-only", "--figures-only"]) is None
    assert [cmd[1] for cmd in procs.calls] == ["-m", STATS], "normalize and the stats ran; mounting was dropped"
    assert any("RuntimeError" in w for w in _warnings(caplog))


def test_abdomen_only_on_a_plain_config_exits_naming_the_missing_block(tmp_path, monkeypatch):
    base, _ = _configs(tmp_path)
    _forbid_per_work(monkeypatch)
    with pytest.raises(SystemExit, match="no abdomen/mounting block"):
        rw.main(["--config", str(base), "--abdomen-only"])


@pytest.mark.skipif(not LIVE.is_file(), reason="config/config_new.yaml is gitignored and absent")
def test_abdomen_only_on_the_live_config_new_exits_naming_the_missing_block(monkeypatch):
    _forbid_per_work(monkeypatch)
    with pytest.raises(SystemExit, match="no abdomen/mounting block"):
        rw.main(["--config", "config/config_new.yaml", "--abdomen-only"])


def test_abdomen_only_needs_an_enabled_block(tmp_path, monkeypatch):
    _, overlay = _configs(tmp_path, {"extends": "config_new.yaml", "abdomen": {"enabled": False}})
    _forbid_per_work(monkeypatch)
    with pytest.raises(SystemExit, match="no abdomen/mounting block"):
        rw.main(["--config", str(overlay), "--abdomen-only"])


def test_abdomen_only_accepts_an_enabled_mounting_block_alone(tmp_path, monkeypatch):
    _, overlay = _configs(tmp_path, {"extends": "config_new.yaml", "mounting": {"enabled": True}})
    calls = []
    monkeypatch.setattr(rw, "_run_abdomen_stages", lambda layers, settings, **kw: calls.append(kw) or {})
    assert rw.main(["--config", str(overlay), "--abdomen-only"]) is None
    assert len(calls) == 1


def test_an_invalid_overlay_aborts_at_startup_before_any_work(tmp_path, monkeypatch):
    _, overlay = _configs(tmp_path, {"extends": "config_new.yaml", "abdomen": {"enabled": True},
                                     "yolo": {"conf_thres": 0.5}})
    _forbid_per_work(monkeypatch)
    monkeypatch.setattr(rw, "load_settings", lambda path: pytest.fail("the config must be refused first"))
    for argv in (["--config", str(overlay)], ["--config", str(overlay), "--abdomen-only"]):
        with pytest.raises(LayeredConfigError, match="may only add"):
            rw.main(argv)


# ---- main(): the whole-run lock, the PER lock and the stage ----------------------------------------------

def _locks_held(cache: Path) -> tuple[bool, bool]:
    """(whole-run lock, PER lock): whether anybody holds each right now, probed through a new open file description."""
    whole = cache / "run_workflows.lock"
    held = False
    if whole.exists():
        fd = os.open(whole, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            held = True
        finally:
            os.close(fd)
    return held, per_run_active(cache)


def _stub_the_night_and_the_stage(monkeypatch, cache: Path, seen: dict, crash: Exception | None = None) -> None:
    """Replace the PER body and the abdomen stage with recorders of which locks are held when each is called."""

    def night(args, config_path, data, settings):
        seen["per"], seen["per_config"] = _locks_held(cache), config_path
        if crash is not None:
            raise crash

    def stage(layers, settings, **kwargs):
        seen["stage"], seen["stage_kwargs"] = _locks_held(cache), kwargs
        return {}

    monkeypatch.setattr(rw, "_main_locked", night)
    monkeypatch.setattr(rw, "_run_abdomen_stages", stage)


def test_an_overlay_run_holds_both_locks_for_the_per_work_and_neither_for_the_abdomen_stage(tmp_path, monkeypatch):
    base, overlay = _configs(tmp_path)
    seen: dict = {}
    _stub_the_night_and_the_stage(monkeypatch, tmp_path / "cache", seen)
    assert rw.main(["--config", str(overlay)]) is None
    assert seen["per"] == (True, True)
    assert seen["stage"] == (False, False), "a stage of hours must not keep the next PER run waiting on either lock"
    assert seen["per_config"] == base.resolve(), "PER gets the base file"
    assert _locks_held(tmp_path / "cache") == (False, False)


def test_a_plain_run_holds_both_locks_for_the_per_work_and_never_calls_the_stage(tmp_path, monkeypatch):
    base, _ = _configs(tmp_path)
    seen: dict = {}
    _stub_the_night_and_the_stage(monkeypatch, tmp_path / "cache", seen)
    assert rw.main(["--config", str(base)]) is None
    assert seen["per"] == (True, True) and "stage" not in seen


def test_the_abdomen_hook_forwards_the_flags_of_the_full_run(tmp_path, monkeypatch):
    _, overlay = _configs(tmp_path)
    seen: dict = {}
    _stub_the_night_and_the_stage(monkeypatch, tmp_path / "cache", seen)
    rw.main(["--config", str(overlay), "--figures-only", "--folder", "X", "--abdomen-video"])
    assert seen["stage_kwargs"] == {"figures_only": True, "folder": "X", "annotated_video": True}


def test_a_run_that_cannot_get_the_whole_run_lock_takes_no_per_lock_and_runs_no_stage(tmp_path, monkeypatch):
    _, overlay = _configs(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    holder = os.open(cache / "run_workflows.lock", os.O_RDWR | os.O_CREAT)
    fcntl.flock(holder, fcntl.LOCK_EX)                     # another PER run, in a different open file description
    monkeypatch.setenv("FBPIPE_RUN_LOCK_WAIT_MIN", "0")
    monkeypatch.setattr(rw, "_main_locked", lambda *a, **k: pytest.fail("PER must wait for the other run"))
    monkeypatch.setattr(rw, "_run_abdomen_stages", lambda *a, **k: pytest.fail("a skipped run runs no stage"))
    try:
        with pytest.raises(rw.RunLockTimeout):
            rw.main(["--config", str(overlay)])
    finally:
        os.close(holder)
    assert not (cache / "locks").exists(), "the PER lock marks PER work, not time spent queueing"
    assert not per_run_active(cache)


def test_a_crashing_per_run_lets_go_of_both_locks_and_runs_no_stage(tmp_path, monkeypatch):
    _, overlay = _configs(tmp_path)
    seen: dict = {}
    _stub_the_night_and_the_stage(monkeypatch, tmp_path / "cache", seen, crash=RuntimeError("yolo died"))
    with pytest.raises(RuntimeError, match="yolo died"):
        rw.main(["--config", str(overlay)])
    assert seen["per"] == (True, True) and "stage" not in seen
    assert _locks_held(tmp_path / "cache") == (False, False)


def test_freeze_status_reads_the_base_config_takes_no_lock_and_runs_no_stage(tmp_path, monkeypatch):
    base, overlay = _configs(tmp_path)
    seen: dict = {}
    monkeypatch.setattr(rw, "_print_freeze_status", lambda args, config_path, data, settings: seen.update(read=config_path))
    monkeypatch.setattr(rw, "_main_locked", lambda *a, **k: pytest.fail("a read-only report is no PER run"))
    monkeypatch.setattr(rw, "_run_abdomen_stages", lambda *a, **k: pytest.fail("a read-only report runs no stage"))
    assert rw.main(["--config", str(overlay), "--freeze-status"]) is None
    assert seen == {"read": base.resolve()}
    assert not (tmp_path / "cache").exists(), "it writes nothing and takes no lock"


# ---- the ntfy ----------------------------------------------------------------------------------------------

def test_the_pipeline_complete_ntfy_appends_the_abdomen_summary_after_the_eye_qc_line():
    sent = []

    def finishing(namespace):
        namespace["LAST_EYE_QC_SUMMARY"] = "EYE-QC: 1 of 4 flagged"
        namespace["LAST_ABDOMEN_SUMMARY"] = "ABDOMEN: track rc=0"

    _run_main_guard(finishing, sent)
    assert sent == [("Pipeline Complete",
                     "Pipeline finished successfully.\n\nEYE-QC: 1 of 4 flagged\n\nABDOMEN: track rc=0",
                     {"tags": "white_check_mark"})]


def test_the_pipeline_complete_ntfy_orders_the_eye_qc_freeze_and_abdomen_lines():
    sent = []

    def finishing(namespace):
        namespace.update(LAST_EYE_QC_SUMMARY="EYE-QC: 1 of 4 flagged", _RUN_SUMMARY="auto-freeze: 2 frozen",
                         LAST_ABDOMEN_SUMMARY="ABDOMEN: track rc=0")

    _run_main_guard(finishing, sent)
    assert sent == [("Pipeline Complete",
                     "Pipeline finished successfully.\n\nEYE-QC: 1 of 4 flagged\nauto-freeze: 2 frozen"
                     "\n\nABDOMEN: track rc=0", {"tags": "white_check_mark"})]


def test_the_abdomen_summary_alone_is_appended_when_there_is_no_eye_qc_line():
    sent = []
    _run_main_guard(lambda ns: ns.update(LAST_ABDOMEN_SUMMARY="ABDOMEN: skipped (locked)"), sent)
    assert sent == [("Pipeline Complete", "Pipeline finished successfully.\n\nABDOMEN: skipped (locked)",
                     {"tags": "white_check_mark"})]


def test_an_abdomen_only_run_sends_its_own_title_and_the_summary():
    sent = []
    _run_main_guard(lambda ns: ns.update(LAST_RUN_ABDOMEN_ONLY=True, LAST_ABDOMEN_SUMMARY="ABDOMEN: track rc=0"), sent)
    assert sent == [("Abdomen stage complete", "ABDOMEN: track rc=0", {"tags": "white_check_mark"})]


def test_an_abdomen_only_run_never_says_the_pipeline_completed_or_quotes_the_eye_qc():
    sent = []
    _run_main_guard(lambda ns: ns.update(LAST_RUN_ABDOMEN_ONLY=True, LAST_EYE_QC_SUMMARY="EYE-QC: stale"), sent)
    assert [title for title, _, _ in sent] == ["Abdomen stage complete"]
    assert "EYE-QC" not in sent[0][1] and sent[0][1]


def test_the_abdomen_globals_start_empty_so_a_plain_run_ntfy_is_untouched():
    tree = ast.parse(Path(rw.__file__).read_text(encoding="utf-8"))
    declared = {ast.unparse(n.target): ast.unparse(n) for n in tree.body if isinstance(n, ast.AnnAssign)}
    assert declared["LAST_ABDOMEN_SUMMARY"] == "LAST_ABDOMEN_SUMMARY: str = ''"
    assert declared["LAST_RUN_ABDOMEN_ONLY"] == "LAST_RUN_ABDOMEN_ONLY: bool = False"
