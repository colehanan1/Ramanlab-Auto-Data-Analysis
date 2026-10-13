"""One pipeline run at a time: a whole-run lock under cache_dir.

The 00:00 nightly and a daytime ``run_workflows.py`` used to overlap (2026-10-05
/06), both rewriting the same per-trial tables, wide CSVs and freeze slices.
Every run_workflows mode that writes (full, ``--folder``, ``--figures-only``)
now takes an exclusive ``flock`` on ``<cache_dir>/run_workflows.lock`` first.

A second run WAITS for the first (logging who holds the lock) rather than
exiting: the nightly that finds a long daytime run still going should still do
its night's work once that run ends, and nothing is lost by waiting. After
``FBPIPE_RUN_LOCK_WAIT_MIN`` (default 360 min) it gives up with
``RunLockTimeout`` instead of piling up behind a hung run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from fbpipe.utils import run_lock as rl

SRC = Path(rl.__file__).resolve().parents[2]


def _holder(cache: Path, ready: Path, hold_s: float = 30.0) -> subprocess.Popen:
    code = textwrap.dedent(
        f"""
        import sys, time
        sys.path.insert(0, {str(SRC)!r})
        from fbpipe.utils.run_lock import run_lock
        with run_lock({str(cache)!r}, wait_seconds=0):
            open({str(ready)!r}, "w").write("held")
            time.sleep({hold_s})
        """
    )
    proc = subprocess.Popen([sys.executable, "-c", code])
    for _ in range(200):
        if ready.exists():
            return proc
        time.sleep(0.05)
    proc.kill()
    raise AssertionError("the holder never took the lock")


def test_acquire_writes_who_holds_it(tmp_path):
    with rl.run_lock(tmp_path, wait_seconds=0, label="unit test") as info:
        data = json.loads((tmp_path / "run_workflows.lock.info").read_text())
        assert data["pid"] == os.getpid()
        assert data["label"] == "unit test"
        assert info["pid"] == os.getpid()
    # Released: a fresh acquisition works at once.
    with rl.run_lock(tmp_path, wait_seconds=0):
        pass


def test_a_second_run_times_out_while_the_first_holds_it(tmp_path, capsys):
    holder = _holder(tmp_path, tmp_path / "ready")
    try:
        start = time.monotonic()
        with pytest.raises(rl.RunLockTimeout, match="another pipeline run"):
            with rl.run_lock(tmp_path, wait_seconds=0.6, poll_seconds=0.1):
                pytest.fail("must not get the lock")
        assert time.monotonic() - start >= 0.5
        out = capsys.readouterr().out
        assert f"pid {holder.pid}" in out and "waiting" in out
    finally:
        holder.kill()
        holder.wait()


def test_a_waiting_run_proceeds_once_the_first_ends(tmp_path):
    holder = _holder(tmp_path, tmp_path / "ready", hold_s=0.8)
    try:
        with rl.run_lock(tmp_path, wait_seconds=30, poll_seconds=0.05) as info:
            assert holder.poll() is not None  # the holder had to finish first
            assert info["waited_s"] > 0
    finally:
        holder.kill()
        holder.wait()


def test_a_dead_holder_never_blocks(tmp_path):
    """flock dies with its process: a killed run cannot leave a stale lock."""
    holder = _holder(tmp_path, tmp_path / "ready")
    holder.kill()
    holder.wait()
    with rl.run_lock(tmp_path, wait_seconds=0):
        pass


def test_the_lock_is_not_inherited_by_child_processes(tmp_path):
    """A child (YOLO worker, figure script) must not keep the lock alive after
    the driver exits."""
    with rl.run_lock(tmp_path, wait_seconds=0) as info:
        assert os.get_inheritable(info["fd"]) is False


def test_wait_default_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("FBPIPE_RUN_LOCK_WAIT_MIN", "2.5")
    assert rl.default_wait_seconds() == 150.0
    monkeypatch.delenv("FBPIPE_RUN_LOCK_WAIT_MIN")
    assert rl.default_wait_seconds() == 360 * 60.0
