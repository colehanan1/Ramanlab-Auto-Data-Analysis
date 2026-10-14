"""A child started to die with its parent is not orphaned when the parent is killed (``fbpipe.utils.process_guard``).

The abdomen stage and the geometry fan-out start their children in process groups of their own, so that a timeout can kill
a group as one. Nothing can catch a SIGKILL of the parent, so ``die_with_parent_preexec()`` is the ``preexec_fn`` that asks
the kernel (``PR_SET_PDEATHSIG``) to send the child SIGTERM when its parent dies. Pinned here: on Linux it is a function
that works together with ``start_new_session`` (a real child dies within seconds of a SIGKILLed parent, and one started
without it does not, so the test can tell); anywhere else, or where libc cannot be reached, it is None and Popen is
unchanged. The function that sets the signal is never called in the test process itself: that would tie pytest to its
own parent.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from fbpipe.utils import process_guard

SRC = Path(process_guard.__file__).resolve().parents[2]
LINUX = sys.platform.startswith("linux")


def _alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def _parent_with_a_child(guarded: bool) -> tuple[subprocess.Popen, int]:
    """A parent python process that starts ``sleep 300`` the way the driver starts a stage command (a session of its
    own), with the guard or without it, and prints the child's pid."""
    script = "\n".join([
        "import subprocess, sys, time",
        f"sys.path.insert(0, {str(SRC)!r})",
        "from fbpipe.utils.process_guard import die_with_parent_preexec",
        f"guard = die_with_parent_preexec() if {guarded!r} else None",
        "child = subprocess.Popen(['sleep', '300'], start_new_session=True, preexec_fn=guard)",
        "print(child.pid, flush=True)",
        "time.sleep(300)",
    ])
    parent = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    return parent, int(parent.stdout.readline())


def _wait_until_dead(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not _alive(pid)


def _cleanup(parent: subprocess.Popen, child: int) -> None:
    if parent.poll() is None:
        parent.kill()
    parent.wait()
    if _alive(child):
        os.kill(child, signal.SIGKILL)


@pytest.mark.skipif(not LINUX, reason="PR_SET_PDEATHSIG is Linux only")
def test_a_child_started_with_the_guard_dies_within_seconds_of_its_sigkilled_parent():
    parent, child = _parent_with_a_child(guarded=True)
    try:
        assert _alive(child), "precondition: the child runs while its parent does"
        parent.kill()                                       # SIGKILL: no handler, no finally, nothing runs in the parent
        parent.wait()
        assert _wait_until_dead(child, 10), "the child kept running after its parent was killed"
    finally:
        _cleanup(parent, child)


@pytest.mark.skipif(not LINUX, reason="PR_SET_PDEATHSIG is Linux only")
def test_without_the_guard_the_child_of_a_sigkilled_parent_runs_on():
    """The control: what the guard prevents really happens, so the test above can fail."""
    parent, child = _parent_with_a_child(guarded=False)
    try:
        parent.kill()
        parent.wait()
        assert not _wait_until_dead(child, 1.5), "an unguarded child outlives its parent"
    finally:
        _cleanup(parent, child)


@pytest.mark.skipif(not LINUX, reason="PR_SET_PDEATHSIG is Linux only")
def test_the_guard_is_a_function_on_linux():
    assert callable(process_guard.die_with_parent_preexec())


def test_the_guard_is_nothing_off_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert process_guard.die_with_parent_preexec() is None


def test_the_guard_is_nothing_where_libc_cannot_be_reached(monkeypatch):
    def no_libc(*args, **kwargs):
        raise OSError("no libc here")

    monkeypatch.setattr(process_guard.ctypes, "CDLL", no_libc)
    assert process_guard.die_with_parent_preexec() is None


def test_the_guard_is_nothing_where_libc_has_no_prctl(monkeypatch):
    monkeypatch.setattr(process_guard.ctypes, "CDLL", lambda *args, **kwargs: object())    # no .prctl attribute
    assert process_guard.die_with_parent_preexec() is None


def test_the_guard_module_is_not_per_trial_code_for_the_auto_freeze_report():
    """Auto-freeze reports -- never acts on -- frozen datasets built under other per-trial code, by hashing
    ``src/fbpipe/utils/*.py`` minus the files that cannot change a row. This module starts no PER work: left on the list
    it would put a "per-trial code changed" notice naming every frozen dataset in the first night after it lands."""
    from fbpipe import autofreeze

    hashed = {Path(rel).name for rel in autofreeze.code_provenance()}
    assert "process_guard.py" not in hashed
    assert "threshold.py" in hashed and "reject_bad_proboscis.py" in hashed, "the list is not empty"
