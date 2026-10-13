"""run_workflows hands the abdomen/mounting overlay to its own stage and keeps PER on config_new.yaml.

``config_new_abdomen.yaml`` is ``extends: config_new.yaml`` plus an ``abdomen:`` block. ``main()`` reads it through
``resolve_layers``: every PER reader and child gets the BASE file, only ``_run_abdomen_stages`` gets the overlay.
The stage builds three commands (track; normalize + wide + figures; the stats report), runs them one after another
against ONE ``abdomen.max_hours`` budget, each in its own process group, and never raises: a failed command is a
warning, a held ``abdomen.lock`` or a ``--folder`` run skips the stage. ``--abdomen-only`` runs just that stage,
before any PER work and without the PER lock; a full run holds the shared PER lock for its PER work and lets go of
it right before the stage. Pinned here: the pure command builder, ``main()``'s source order, the stage with a fake
Popen, the flags and the ntfy. (The PER call trace itself is pinned in test_layered_parity.py.)
"""
from __future__ import annotations

import ast
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
from pathlib import Path
from types import MappingProxyType

import pytest
import yaml

from fbpipe import geometry_pipeline
from fbpipe.config import Settings
from fbpipe.config_layers import ConfigLayers, LayeredConfigError
from fbpipe.utils.run_locks import abdomen_lock
from scripts.pipeline import run_workflows as rw
from test_eye_drift_qc_summary import _run_main_guard

LIVE = Path(__file__).resolve().parents[1] / "config" / "config_new.yaml"
OVERLAY = Path("/cfg/config_new_abdomen.yaml")
PY = "/venv/bin/python"
STATS = str(rw.REPO_ROOT / "scripts" / "analysis" / "abdomen_geometry_stats.py")

GEOMETRY = [PY, "-m", "fbpipe.geometry_pipeline", "--config", str(OVERLAY)]
TRACK = GEOMETRY + ["--workers", "3", "track"]
NORMALIZE = GEOMETRY + ["normalize", "wide", "figures"]
STATS_CMD = [PY, STATS, "--config", str(OVERLAY)]


def _cmds(overlay=None, **over):
    kwargs = dict(python_exec=PY, overlay_path=OVERLAY, figures_only=False, annotated_video=False, n_workers=3)
    kwargs.update(over)
    return rw._abdomen_commands({"abdomen": {"enabled": True}} if overlay is None else overlay, **kwargs)


# ---- the command builder ----------------------------------------------------------------------------------

@pytest.mark.parametrize("overlay", [{}, None, {"mounting": {"enabled": True}}, {"abdomen": None},
                                     {"abdomen": {"enabled": False}}])
def test_no_commands_without_an_enabled_abdomen_block(overlay):
    assert rw._abdomen_commands(overlay, python_exec=PY, overlay_path=OVERLAY, figures_only=False,
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
    for cmd in _cmds():
        assert cmd[cmd.index("--config") + 1] == str(OVERLAY)
        assert "config_new.yaml" not in " ".join(cmd)


# ---- the stage: fakes --------------------------------------------------------------------------------------

class _Proc:
    """A subprocess.Popen stand-in. ``mode`` is an exit code, "timeout" (the first wait times out; the process is
    gone after SIGTERM), "ignores SIGTERM" (it also outlives the grace wait and dies on SIGKILL) or a
    KeyboardInterrupt to raise from the first wait."""

    def __init__(self, procs, pid, mode, elapse):
        self.procs, self.pid, self.mode, self.elapse, self.returncode, self.waits = procs, pid, mode, elapse, None, []

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if isinstance(self.mode, KeyboardInterrupt):
            if len(self.waits) == 1:
                raise self.mode
            self.returncode = -15
        elif isinstance(self.mode, int):
            self.procs.clock.now += self.elapse                   # the command ran this long
            self.returncode = self.mode
        else:
            stubborn = self.mode == "ignores SIGTERM"
            if len(self.waits) == 1 or (stubborn and len(self.waits) == 2):
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
    monkeypatch.setattr(rw, "LAST_ABDOMEN_SUMMARY", "")    # main() and the stage set these; restore them afterwards
    monkeypatch.setattr(rw, "LAST_RUN_ABDOMEN_ONLY", False)


def _settings(tmp_path) -> Settings:
    return Settings(model_path="m.pt", main_directories=[], cache_dir=str(tmp_path / "cache"))


def _layers(tmp_path, abdomen=None, **blocks) -> ConfigLayers:
    overlay = dict(blocks)
    overlay["abdomen"] = {"enabled": True} if abdomen is None else abdomen
    return ConfigLayers(tmp_path / "config_new_abdomen.yaml", tmp_path / "config_new.yaml", MappingProxyType(overlay))


def _run(tmp_path, layers=None, **kw):
    kwargs = dict(figures_only=False, folder=None, annotated_video=False)
    kwargs.update(kw)
    return rw._run_abdomen_stages(layers or _layers(tmp_path), _settings(tmp_path), **kwargs)


def _expected(tmp_path, **over):
    kwargs = dict(python_exec=sys.executable, overlay_path=tmp_path / "config_new_abdomen.yaml", figures_only=False,
                  annotated_video=False, n_workers=3)
    kwargs.update(over)
    return rw._abdomen_commands({"abdomen": {"enabled": True}}, **kwargs)


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
    assert procs.procs[0].waits == [7200.0], "the first command may use the whole budget"


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


def test_a_disabled_block_leaves_nothing_to_run(tmp_path, procs):
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


def test_a_timeout_kills_the_process_group_and_ends_the_stage(tmp_path, monkeypatch, caplog):
    procs = _Procs(monkeypatch, script=["timeout"])
    with caplog.at_level(logging.INFO):
        result = _run(tmp_path)
    assert procs.killed == [(4201, signal.SIGTERM)], "the whole group of the first command, not just its leader"
    assert procs.procs[0].waits == [36000.0, 30.0], "the default budget is 10 h; then 30 s of grace after SIGTERM"
    assert result["timed_out"] is True and result["commands"][0]["timed_out"] is True
    assert len(procs.calls) == 1, "the budget is spent: nothing else is started"
    assert [c["ran"] for c in result["commands"]] == [True, False, False]
    assert any(r.levelname == "WARNING" and "track" in r.getMessage() for r in caplog.records)
    assert "track TIMED OUT" in result["summary"] and "normalize+wide+figures not run" in result["summary"]


def test_a_process_that_ignores_sigterm_is_killed(tmp_path, monkeypatch):
    procs = _Procs(monkeypatch, script=["ignores SIGTERM"])
    result = _run(tmp_path)
    assert procs.killed == [(4201, signal.SIGTERM), (4201, signal.SIGKILL)]
    assert result["commands"][0]["returncode"] == -9 and result["timed_out"] is True


def test_the_commands_share_one_time_budget(tmp_path, monkeypatch):
    procs = _Procs(monkeypatch, elapse=[3000.0, 100.0, 0.0])            # one hour in all; the first uses 50 minutes
    _run(tmp_path, _layers(tmp_path, {"enabled": True, "max_hours": 1}))
    assert [p.waits for p in procs.procs] == [[3600.0], [600.0], [500.0]]


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
                                 subprocess.SubprocessError("popen"), ValueError("bad value")])
def test_the_stage_never_raises(tmp_path, monkeypatch, caplog, exc):
    procs = _Procs(monkeypatch, script=[exc])
    with caplog.at_level(logging.INFO):
        result = _run(tmp_path)
    assert len(procs.calls) == 1 and result["ran"] is False and type(exc).__name__ in result["error"]
    assert any(r.levelname == "WARNING" and str(exc) in r.getMessage() for r in caplog.records)
    assert rw.LAST_ABDOMEN_SUMMARY.startswith("ABDOMEN:") and type(exc).__name__ in rw.LAST_ABDOMEN_SUMMARY


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


def test_the_summary_is_one_line(tmp_path, procs):
    result = _run(tmp_path)
    assert result["summary"] == "ABDOMEN: track rc=0; normalize+wide+figures rc=0; abdomen_geometry_stats rc=0"
    assert rw.LAST_ABDOMEN_SUMMARY == result["summary"]


# ---- the commands against the real CLI and real processes --------------------------------------------------

def test_the_module_commands_parse_with_the_geometry_clis_own_parser():
    parser = geometry_pipeline.build_parser()
    track = parser.parse_args(_cmds(annotated_video=True, n_workers=2)[0][3:])
    assert (track.config, track.workers, track.annotated_video, track.steps) == (str(OVERLAY), 2, True, ["track"])
    later = parser.parse_args(_cmds()[1][3:])
    assert later.config == str(OVERLAY) and later.steps == ["normalize", "wide", "figures"]
    assert later.workers is None and later.annotated_video is False


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
        f"{rw.REPO_ROOT / 'scripts' / 'analysis' / 'abdomen_geometry_stats.py'} --config {overlay}",
    ]
    assert [c["returncode"] for c in result["commands"]] == [3, 3]


def test_a_real_timeout_kills_the_whole_process_group_including_what_the_command_started(tmp_path, fake_python):
    result = _run(tmp_path, _layers(tmp_path, {"enabled": True, "python": str(fake_python), "max_hours": 0.0006}))
    sleeper = int((tmp_path / "sleep.pid").read_text())
    deadline = time.monotonic() + 5
    while _alive(sleeper) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert result["timed_out"] is True and result["commands"][0]["returncode"] == -signal.SIGTERM
    assert not _alive(sleeper), "the track command's own child must not outlive the stage"
    assert len((tmp_path / "argv.log").read_text().splitlines()) == 1, "the spent budget starts nothing else"
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


def _main_source() -> str:
    return inspect.getsource(rw.main)


def test_the_layers_are_resolved_before_any_config_is_read_and_per_gets_the_base_path():
    src = _main_source()
    at = src.index("resolve_layers(")
    assert at < src.index("yaml.safe_load(") and at < src.index("load_settings(")
    assert src.index("config_path = layers.base_path") < src.index("yaml.safe_load(")


def test_the_abdomen_only_early_return_precedes_all_per_work():
    src = _main_source()
    early = src.index("if args.abdomen_only:")
    assert src.index("_validate_thaw(") < early < src.index("_run_pipeline(")
    assert early < src.index("acquire_per_run_lock("), "an --abdomen-only run is not a PER run: no PER lock"
    body = src[early:src.index("acquire_per_run_lock(")]
    assert "_run_abdomen_stages(" in body and re.search(r"^\s+return\s*$", body, re.MULTILINE), \
        "a bare return, so the legacy shim's SystemExit(main()) exits 0"
    assert "no abdomen/mounting block" in body


def test_the_per_lock_spans_the_per_work_and_is_released_before_the_abdomen_stage():
    src = _main_source()
    acquire, sync = src.index("acquire_per_run_lock("), src.index("_run_data_secured_sync(data)")
    release, hook = src.index("release_run_lock("), src.rindex("_run_abdomen_stages(")
    assert acquire < src.index("_run_pipeline(") < sync < release < hook


def test_the_abdomen_stage_follows_the_data_secured_sync_and_only_runs_for_an_overlay():
    src = _main_source()
    assert src.rindex("_run_abdomen_stages(") > src.index("_run_data_secured_sync(data)")
    assert re.search(r"if layers\.overlay:\s*\n\s+_run_abdomen_stages\(", src), \
        "a plain config_new.yaml run must not even call the stage"


# ---- main(): --abdomen-only --------------------------------------------------------------------------------

def _configs(tmp_path, overlay=None):
    """A minimal base config_new.yaml and the overlay that extends it (default: an enabled abdomen block)."""
    base = {"protocol": "v2", "model_path": "m.pt", "datasets": [], "cache_dir": str(tmp_path / "cache")}
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
    assert procs.calls == rw._abdomen_commands({"abdomen": {"enabled": True}}, python_exec=sys.executable,
                                               overlay_path=overlay, figures_only=False, annotated_video=True,
                                               n_workers=2)
    assert rw.LAST_ABDOMEN_SUMMARY.startswith("ABDOMEN: track rc=0") and rw.LAST_RUN_ABDOMEN_ONLY is True


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
