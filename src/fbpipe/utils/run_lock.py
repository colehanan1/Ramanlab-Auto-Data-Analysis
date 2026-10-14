"""One pipeline run at a time: an exclusive ``flock`` under ``cache_dir``.

The 00:00 nightly (cron -> ``make run``) and a daytime ``run_workflows.py``
overlapped on 2026-10-05/06, both rewriting the same per-trial tables, wide
CSVs and freeze slices -- and with auto-freeze, the same freeze records. Every
writing run_workflows mode takes this lock first and holds it to the end.

A run that finds the lock held WAITS, logging the holder: the nightly that
meets a long daytime run should still do the night's work once that run ends
(data recorded in the evening would otherwise wait another day), and waiting
loses nothing. After ``wait_seconds`` (``FBPIPE_RUN_LOCK_WAIT_MIN``, default
360 min: the longest observed run is ~2 h) it gives up with
:class:`RunLockTimeout` rather than queueing behind a hung run forever.

``flock`` belongs to the open file, so the lock dies with its process -- a
killed run never leaves a stale lock -- and the descriptor is opened
non-inheritable, so no YOLO worker or figure subprocess keeps it alive.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

LOCK_NAME = "run_workflows.lock"
DEFAULT_WAIT_MIN = 360.0


class RunLockTimeout(RuntimeError):
    """Another pipeline run held the lock for longer than this run would wait."""


def default_wait_seconds() -> float:
    raw = os.environ.get("FBPIPE_RUN_LOCK_WAIT_MIN")
    try:
        return float(raw) * 60.0 if raw not in (None, "") else DEFAULT_WAIT_MIN * 60.0
    except ValueError:
        return DEFAULT_WAIT_MIN * 60.0


def _holder_text(info_path: Path) -> str:
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- the holder may be writing it right now
        return "an unknown run"
    return (
        f"pid {info.get('pid')} ({info.get('label') or ' '.join(info.get('argv') or [])}, "
        f"started {info.get('started')} on {info.get('host')})"
    )


@contextlib.contextmanager
def run_lock(
    cache_dir: str | Path,
    *,
    wait_seconds: Optional[float] = None,
    poll_seconds: float = 30.0,
    label: str = "",
) -> Iterator[dict]:
    """Hold the whole-run lock for the ``with`` body; wait for it if needed."""
    directory = Path(cache_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / LOCK_NAME
    info_path = directory / f"{LOCK_NAME}.info"
    wait = default_wait_seconds() if wait_seconds is None else float(wait_seconds)

    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o664)  # non-inheritable (PEP 446)
    start = time.monotonic()
    announced = None
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                waited = time.monotonic() - start
                if waited >= wait:
                    raise RunLockTimeout(
                        f"another pipeline run holds {lock_path}: {_holder_text(info_path)}; "
                        f"gave up after {waited / 60:.1f} min"
                    ) from None
                if announced is None or time.monotonic() - announced >= 600:
                    print(
                        f"[LOCK] waiting for another pipeline run to finish: "
                        f"{_holder_text(info_path)} (holds {lock_path}; this run gives up "
                        f"after {wait / 60:.0f} min)",
                        flush=True,
                    )
                    announced = time.monotonic()
                time.sleep(min(poll_seconds, max(0.01, wait - waited)))
        waited_s = time.monotonic() - start
        info = {
            "pid": os.getpid(),
            "argv": list(sys.argv),
            "label": label,
            "host": socket.gethostname(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            tmp = info_path.with_name(f".{info_path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(info), encoding="utf-8")
            os.replace(tmp, info_path)
        except OSError:
            pass  # the info file is for humans; the flock is the lock
        if waited_s >= 1.0:
            print(f"[LOCK] acquired {lock_path} after waiting {waited_s / 60:.1f} min", flush=True)
        yield {**info, "fd": fd, "waited_s": waited_s, "path": str(lock_path)}
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)
