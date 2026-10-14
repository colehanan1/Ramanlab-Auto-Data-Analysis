"""Pull each finished trial from its Pi into staging and remux it there.

Additive rsync only — never a delete flag. Staging lives outside flys_New, so
nothing here can race the nightly pipeline or a legacy end-of-run remux.
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

log = logging.getLogger("rigmon.transfer")


def subprocess_runner(argv: list[str], timeout: float, env: dict | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout:.0f}s"
    except OSError as e:
        return 127, str(e)


def rsync_pull_argv(rig, ssh_key: Path, remote_path: str, dest_dir: Path) -> list[str]:
    argv = ["rsync", "-a", "--partial"]
    if rig.ip == "local":
        src = remote_path
    else:
        argv += ["-e", f"ssh -i {ssh_key} -o BatchMode=yes -o ConnectTimeout=10"]
        src = f"{rig.ssh_user}@{rig.ip}:{remote_path}"
    return argv + [src, f"{dest_dir}/"]


def remux_argv(cfg, h264: Path) -> list[str]:
    return ["nice", "-n", "10", cfg.remux_python, str(cfg.remux_script),
            "--input-dir", str(h264.parent), "--only", h264.name, *cfg.remux_args]


class TransferError(RuntimeError):
    pass


class TrialTransfer:
    def __init__(self, cfg, store, runner=subprocess_runner, sleep=time.sleep,
                 backoff=(5.0, 15.0, 45.0)):
        self.cfg, self.store, self.runner = cfg, store, runner
        self.sleep, self.backoff = sleep, tuple(backoff)

    def staging_dir(self, rig_name: str, batch_dir: str) -> Path:
        return self.cfg.staging_root / rig_name / batch_dir

    def pull(self, rig, output_dir: str, rel_path: str, staging: Path) -> None:
        dest_dir = (staging / rel_path).parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        remote = f"{str(output_dir).rstrip('/')}/{rel_path}"
        rc, out = self.runner(rsync_pull_argv(rig, self.cfg.ssh_key, remote, dest_dir), 900.0, None)
        if rc != 0:
            raise TransferError(f"rsync {rel_path} failed ({rc}): {out[-300:]}")

    def remux(self, h264: Path) -> Path:
        env = {**os.environ, "PATH": self.cfg.remux_path_env}
        rc, out = self.runner(remux_argv(self.cfg, h264), 1800.0, env)
        mp4 = h264.with_suffix(".mp4")
        if rc != 0 or not mp4.exists() or mp4.stat().st_size == 0:
            raise TransferError(f"remux {h264.name} failed ({rc}): {out[-300:]}")
        return mp4

    def _attempt(self, rig, batch_dir: str, output_dir: str, t: dict) -> str:
        staging = self.staging_dir(rig.name, batch_dir)
        idx = int(t["index"])
        video = staging / t["video"]
        mp4 = video.with_suffix(".mp4")
        if mp4.exists() and mp4.stat().st_size > 0 and not video.exists():
            return mp4.name  # finished before a hub restart
        self.store.set_trial_state(rig.name, batch_dir, idx, "pulling")
        for name, expected in ((t["frames_csv"], t.get("csv_bytes")),
                               (t["video"], t.get("video_bytes"))):
            self.pull(rig, output_dir, name, staging)
            got = (staging / name).stat().st_size
            if expected and got != int(expected):
                raise TransferError(f"{name}: {got} bytes here, Pi reported {expected}")
        self.store.set_trial_state(rig.name, batch_dir, idx, "remuxing")
        return self.remux(video).name

    def process(self, rig, batch_dir: str, output_dir: str, t: dict) -> str:
        self.store.upsert_trial(rig.name, batch_dir, t)
        idx = int(t["index"])
        if not t.get("closed", True):
            self.store.set_trial_state(
                rig.name, batch_dir, idx, "waiting",
                error="recorder still open on the Pi; finalize will collect it")
            return "waiting"
        last = ""
        for attempt in range(len(self.backoff) + 1):
            try:
                mp4 = self._attempt(rig, batch_dir, output_dir, t)
                self.store.set_trial_state(rig.name, batch_dir, idx, "done", mp4=mp4)
                return "done"
            except (TransferError, OSError) as e:
                last = str(e)
                log.warning("%s %s trial %s: %s", rig.name, batch_dir, idx, last)
                if attempt < len(self.backoff):
                    self.sleep(self.backoff[attempt])
        self.store.set_trial_state(rig.name, batch_dir, idx, "failed", error=last)
        return "failed"


class TransferWorkers:
    """One FIFO per rig: a rig's trials and its finalize run strictly in order;
    rigs run in parallel. A key already queued is not queued twice."""

    def __init__(self, threaded: bool = True):
        self._threaded = threaded
        self._queues: dict[str, queue.Queue] = {}
        self._pending: set = set()
        self._lock = threading.Lock()

    def submit(self, rig_name: str, key, fn: Callable[[], object]) -> bool:
        with self._lock:
            if key in self._pending:
                return False
            self._pending.add(key)
            q = self._queues.get(rig_name)
            if q is None:
                q = self._queues[rig_name] = queue.Queue()
                if self._threaded:
                    threading.Thread(target=self._loop, args=(q,), daemon=True,
                                     name=f"rigmon-{rig_name}").start()
        q.put((key, fn))
        return True

    def _run_one(self, key, fn) -> None:
        try:
            fn()
        except Exception:
            log.exception("job %s failed", key)
        finally:
            with self._lock:
                self._pending.discard(key)

    def _loop(self, q: queue.Queue) -> None:
        while True:
            self._run_one(*q.get())

    def drain(self) -> None:
        for q in list(self._queues.values()):
            while not q.empty():
                self._run_one(*q.get_nowait())
