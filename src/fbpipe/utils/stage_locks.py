"""Advisory run locks between the nightly PER run and the abdomen stage.

An on-demand abdomen backfill can overlap the nightly run. Two flock files under
``<cache_dir>/locks/`` keep them apart without either waiting on the other:

* ``per.lock`` -- every PER run holds it SHARED for its PER work (:func:`acquire_per_run_lock`), so two PER
  runs coexist. The abdomen tracker asks :func:`per_run_active` before each video and yields while any PER
  run holds it; whatever it has not tracked resumes on the next run.
* ``abdomen.lock`` -- an abdomen stage holds it EXCLUSIVE and non-blocking (:func:`abdomen_lock`), so a second
  abdomen run finds it taken and skips instead of queueing behind a multi-hour stage.

flock locks belong to the open file, not to a path or a pid: a crashed process frees its locks, so nothing is
ever stale, and the empty lock files are simply left behind. Descriptors are non-inheritable, so a stage's own
subprocesses never extend a lock, and :meth:`RunLock.release` unlocks before it closes so a worker forked while
the lock was held cannot keep it alive.

The lock is purely advisory and must never cost a PER run anything: acquiring it fails open, and probing it
creates nothing. (This module is ``stage_locks``, not ``run_locks``: that was one letter from main's whole-run lock,
``fbpipe.utils.run_lock``, which is a different lock with a different job.)
"""
from __future__ import annotations

import fcntl
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCKS_SUBDIR = "locks"
PER_LOCK = "per.lock"
ABDOMEN_LOCK = "abdomen.lock"


def lock_dir(cache_dir: str | Path | None) -> Path:
    """``<cache_dir>/locks``; an empty ``cache_dir`` means ``<repo>/cache`` like everywhere else in the driver."""
    base = Path(cache_dir).expanduser() if cache_dir else REPO_ROOT / "cache"
    return base / LOCKS_SUBDIR


class RunLock:
    """An advisory lock held through an open descriptor. ``release()`` is idempotent; a handle that never got
    its lock (:attr:`held` False) releases as a no-op."""

    def __init__(self, fd: Optional[int]) -> None:
        self._fd = fd

    @property
    def held(self) -> bool:
        return self._fd is not None

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)       # not just close(): a forked copy of the fd would keep the lock
        finally:
            os.close(fd)

    def __enter__(self) -> "RunLock":
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()


def _open_lock_file(cache_dir: str | Path | None, name: str) -> int:
    directory = lock_dir(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return os.open(directory / name, os.O_RDWR | os.O_CREAT, 0o644)


def acquire_per_run_lock(cache_dir: str | Path | None) -> RunLock:
    """Take the SHARED PER lock (blocking, but nothing ever holds it exclusively for longer than a probe).

    Fails open: if the lock file cannot be created or locked the run goes ahead without it -- a warning, never
    an exception, so the lock can add nothing to what can go wrong in a PER run."""
    fd: Optional[int] = None
    try:
        fd = _open_lock_file(cache_dir, PER_LOCK)
        fcntl.flock(fd, fcntl.LOCK_SH)
    except OSError as exc:
        LOGGER.warning("[locks] could not take %s under %s (%s: %s); this run is not visible to an abdomen "
                       "stage running at the same time", PER_LOCK, lock_dir(cache_dir), type(exc).__name__, exc)
        if fd is not None:
            os.close(fd)
        return RunLock(None)
    return RunLock(fd)


def release_run_lock(handle: Optional[RunLock]) -> None:
    """Release *handle* (idempotent; ``None`` and a lock that was never taken are no-ops)."""
    if handle is not None:
        handle.release()


@contextmanager
def per_run_lock(cache_dir: str | Path | None) -> Iterator[None]:
    """``with per_run_lock(cache_dir):`` -- the shared PER lock around a block."""
    handle = acquire_per_run_lock(cache_dir)
    try:
        yield
    finally:
        handle.release()


def per_run_active(cache_dir: str | Path | None) -> bool:
    """Whether some process holds the PER lock right now.

    A probe: it tries to take the lock exclusively without waiting and lets go at once, so it is True exactly
    when a shared holder exists. It never creates the lock file or its directory (no file, no PER run ever
    held it) and answers False when it cannot tell."""
    path = lock_dir(cache_dir) / PER_LOCK
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        except OSError:
            return False
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


@contextmanager
def abdomen_lock(cache_dir: str | Path | None) -> Iterator[bool]:
    """``with abdomen_lock(cache_dir) as got:`` -- True when this process now holds the exclusive abdomen lock,
    False when another abdomen run does (the caller skips; the lock is not waited for). OS errors creating the
    lock file propagate: an abdomen stage that cannot take its lock must not run unguarded."""
    fd = _open_lock_file(cache_dir, ABDOMEN_LOCK)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
