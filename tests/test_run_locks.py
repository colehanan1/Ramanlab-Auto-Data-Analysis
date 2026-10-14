"""Run locks keep an abdomen backfill and the nightly PER run apart without either waiting on the other.

Two flock files live under ``<cache_dir>/locks/``:

* ``per.lock`` -- every PER run holds it SHARED for its PER work, so two PER runs coexist. The abdomen
  tracker probes it (``per_run_active``) before each video and yields while any PER run holds it.
* ``abdomen.lock`` -- an abdomen run holds it EXCLUSIVE and non-blocking; a second abdomen run finds it
  taken and skips instead of queueing behind a multi-hour stage.

flock locks belong to the open file, not to a path or a pid, so a crashed process frees them and nothing is
ever stale. Probing must not create anything, an unwritable cache dir must never stop a PER run, and a child
process must not inherit the lock. Tests use tmp cache dirs only.
"""
import multiprocessing
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from fbpipe.utils import run_locks as RL
from fbpipe.utils.run_locks import (
    abdomen_lock,
    acquire_per_run_lock,
    per_run_active,
    per_run_lock,
    release_run_lock,
)

SRC = Path(RL.__file__).resolve().parents[2]


class LockCallBlocked(BaseException):
    """Not an OSError/Exception, so no ``except OSError`` in the code under test can swallow it."""


@pytest.fixture(autouse=True)
def _a_blocking_lock_fails_the_test_instead_of_freezing_the_suite():
    def _boom(signum, frame):
        raise LockCallBlocked("a lock call blocked")

    previous = signal.signal(signal.SIGALRM, _boom)
    signal.alarm(10)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _child(code: str, cache: Path) -> subprocess.Popen:
    """A python process that runs *code* (with ``CACHE`` and the run_locks names in scope) and prints ``held``."""
    script = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(SRC)!r})
        from fbpipe.utils.run_locks import abdomen_lock, acquire_per_run_lock
        CACHE = {str(cache)!r}
    """) + textwrap.dedent(code) + "\nprint('held', flush=True)\ntime.sleep(60)\n"
    return subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)


# ---- the shared PER lock ----------------------------------------------------------------------------------

def test_two_per_run_locks_coexist(tmp_path):
    with per_run_lock(tmp_path):
        with per_run_lock(tmp_path):
            assert per_run_active(tmp_path)


def test_per_run_active_is_true_while_a_holder_exists_and_false_after(tmp_path):
    assert not per_run_active(tmp_path)
    with per_run_lock(tmp_path):
        assert per_run_active(tmp_path)
    assert not per_run_active(tmp_path)


def test_per_run_stays_active_until_the_last_holder_leaves(tmp_path):
    first, second = acquire_per_run_lock(tmp_path), acquire_per_run_lock(tmp_path)
    release_run_lock(first)
    assert per_run_active(tmp_path), "one PER run is still going"
    release_run_lock(second)
    assert not per_run_active(tmp_path)


def test_probing_creates_no_file_and_no_directory(tmp_path):
    cache = tmp_path / "cache"
    assert per_run_active(cache) is False
    assert not cache.exists()


def test_the_lock_files_live_under_cache_dir_locks(tmp_path):
    with per_run_lock(tmp_path), abdomen_lock(tmp_path):
        assert sorted(p.name for p in (tmp_path / "locks").iterdir()) == ["abdomen.lock", "per.lock"]


def test_an_empty_cache_dir_falls_back_to_the_repo_cache_like_the_driver(tmp_path, monkeypatch):
    monkeypatch.setattr(RL, "REPO_ROOT", tmp_path)
    assert RL.lock_dir("") == tmp_path / "cache" / "locks"
    handle = acquire_per_run_lock("")
    assert (tmp_path / "cache" / "locks" / "per.lock").is_file() and per_run_active("")
    release_run_lock(handle)


def test_the_acquire_and_release_helpers_pair_up_and_release_is_idempotent(tmp_path):
    handle = acquire_per_run_lock(tmp_path)
    assert handle.held and per_run_active(tmp_path)
    release_run_lock(handle)
    assert not handle.held and not per_run_active(tmp_path)
    release_run_lock(handle)
    handle.release()
    release_run_lock(None)


def test_a_handle_is_a_context_manager_too(tmp_path):
    with acquire_per_run_lock(tmp_path) as handle:
        assert handle.held and per_run_active(tmp_path)
    assert not per_run_active(tmp_path)


def test_a_cache_dir_that_cannot_hold_locks_never_stops_the_run(tmp_path, caplog):
    blocker = tmp_path / "cache"
    blocker.write_text("a file where the cache directory should be")      # makedirs under it must fail
    with caplog.at_level("WARNING"):
        handle = acquire_per_run_lock(blocker)
    assert handle.held is False, "PER carries on without the advisory lock"
    assert any("per.lock" in r.getMessage() for r in caplog.records if r.levelname == "WARNING")
    release_run_lock(handle)
    assert per_run_active(blocker) is False


# ---- the exclusive abdomen lock ---------------------------------------------------------------------------

def test_a_second_abdomen_lock_yields_false(tmp_path):
    with abdomen_lock(tmp_path) as first:
        assert first is True
        with abdomen_lock(tmp_path) as second:
            assert second is False
        with abdomen_lock(tmp_path) as third:
            assert third is False, "a refused attempt must not have released the holder's lock"
    with abdomen_lock(tmp_path) as again:
        assert again is True, "released on exit"


def test_the_abdomen_lock_is_released_when_the_body_raises(tmp_path):
    with pytest.raises(RuntimeError):
        with abdomen_lock(tmp_path) as got:
            assert got
            raise RuntimeError("stage blew up")
    with abdomen_lock(tmp_path) as again:
        assert again is True


def test_the_two_locks_do_not_block_each_other(tmp_path):
    with per_run_lock(tmp_path):
        with abdomen_lock(tmp_path) as got:
            assert got is True, "a PER run does not stop an abdomen stage from starting"
    with abdomen_lock(tmp_path):
        assert per_run_active(tmp_path) is False, "an abdomen run is not a PER run"
        with per_run_lock(tmp_path):
            pass


# ---- other processes: a crash frees the lock, a child never inherits it -----------------------------------

def test_a_killed_per_run_frees_its_lock(tmp_path):
    child = _child("handle = acquire_per_run_lock(CACHE)", tmp_path)
    try:
        assert child.stdout.readline().strip() == "held"
        assert per_run_active(tmp_path), "another process holds the shared lock"
        child.kill()                                            # a crash: no chance to release anything
        child.wait()
        assert not per_run_active(tmp_path)
    finally:
        child.kill()
        child.wait()


def test_a_second_abdomen_run_in_another_process_is_refused_until_the_first_dies(tmp_path):
    child = _child("ctx = abdomen_lock(CACHE); assert ctx.__enter__() is True", tmp_path)
    try:
        assert child.stdout.readline().strip() == "held"
        with abdomen_lock(tmp_path) as got:
            assert got is False
        child.kill()
        child.wait()
        with abdomen_lock(tmp_path) as got:
            assert got is True
    finally:
        child.kill()
        child.wait()


def test_a_subprocess_never_inherits_the_lock_so_a_crash_frees_it_even_with_survivors(tmp_path):
    # close_fds=False hands the child every descriptor that is not close-on-exec, so this fails
    # if the lock descriptor is inheritable: the survivor would keep the dead holder's lock alive.
    child = _child("""
        import subprocess
        handle = acquire_per_run_lock(CACHE)
        survivor = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], close_fds=False)
        print(survivor.pid, flush=True)
    """, tmp_path)
    survivor_pid = int(child.stdout.readline())
    try:
        assert child.stdout.readline().strip() == "held" and per_run_active(tmp_path)
        child.kill()
        child.wait()
        assert not per_run_active(tmp_path), "the stage's own subprocesses must not extend the lock"
    finally:
        child.kill()
        child.wait()
        os.kill(survivor_pid, signal.SIGKILL)


def test_release_frees_the_lock_even_while_a_forked_worker_still_has_the_descriptor(tmp_path):
    ctx = multiprocessing.get_context("fork")
    handle = acquire_per_run_lock(tmp_path)
    worker = ctx.Process(target=time.sleep, args=(60,))          # a pool worker forked while the lock is held
    worker.start()
    try:
        release_run_lock(handle)
        assert not per_run_active(tmp_path), "closing alone would leave the worker's copy holding it"
    finally:
        worker.terminate()
        worker.join()
        assert os.path.isfile(tmp_path / "locks" / "per.lock")


def test_the_lock_modules_are_not_per_trial_code_for_the_auto_freeze_report():
    """Auto-freeze reports -- never acts on -- frozen datasets built under other per-trial code, by hashing
    ``src/fbpipe/utils/*.py`` minus the files that cannot change a row. This lock module is no per-trial code: left on
    the list it would put a "per-trial code changed" notice naming every frozen dataset in the first night after it
    lands."""
    from fbpipe import autofreeze

    hashed = {Path(rel).name for rel in autofreeze.code_provenance()}
    assert "run_locks.py" not in hashed
    assert "run_lock.py" not in hashed, "main's whole-run lock is skipped the same way"
    assert "threshold.py" in hashed and "reject_bad_proboscis.py" in hashed, "the list is not empty"
