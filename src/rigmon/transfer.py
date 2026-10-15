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


ABORTED_RC = -15


def subprocess_runner(argv: list[str], timeout: float, env: dict | None = None,
                      abort=None) -> tuple[int, str]:
    if abort is None:
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
            return p.returncode, (p.stdout or "") + (p.stderr or "")
        except subprocess.TimeoutExpired:
            return 124, f"timed out after {timeout:.0f}s"
        except OSError as e:
            return 127, str(e)
    try:
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    except OSError as e:
        return 127, str(e)
    deadline = time.monotonic() + timeout
    while True:
        try:
            out, _ = p.communicate(timeout=1.0)
            return p.returncode, out or ""
        except subprocess.TimeoutExpired:
            if abort():
                p.terminate()
                try:
                    p.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.communicate()
                return ABORTED_RC, "aborted: rig started recording"
            if time.monotonic() > deadline:
                p.kill()
                p.communicate()
                return 124, f"timed out after {timeout:.0f}s"


def rsync_pull_argv(rig, ssh_key: Path, remote_path: str, dest_dir: Path,
                    bwlimit_kbps: int | None = None) -> list[str]:
    argv = ["rsync", "-a", "--partial", "--no-links"]
    if bwlimit_kbps:
        argv.append(f"--bwlimit={int(bwlimit_kbps)}")
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


def _is_plain(name) -> bool:
    return (isinstance(name, str) and name not in ("", ".", "..")
            and "/" not in name and "\\" not in name and "\0" not in name)


class TrialTransfer:
    def __init__(self, cfg, store, runner=subprocess_runner, sleep=time.sleep,
                 backoff=(5.0, 15.0, 45.0), is_recording=lambda rig_name: False,
                 max_wait_s=1800.0, poll_s=2.0, clock=time.monotonic):
        self.cfg, self.store, self.runner = cfg, store, runner
        self.is_recording, self.max_wait_s, self.poll_s = is_recording, max_wait_s, poll_s
        self.clock = clock
        self.sleep, self.backoff = sleep, tuple(backoff)

    def staging_dir(self, rig_name: str, batch_dir: str) -> Path:
        if not _is_plain(rig_name) or not _is_plain(batch_dir):
            raise ValueError(f"not a plain name: {rig_name!r}/{batch_dir!r}")
        return self.cfg.staging_root / rig_name / batch_dir

    def pull(self, rig, output_dir: str, rel_path: str, staging: Path) -> None:
        dest_dir = (staging / rel_path).parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        remote = f"{str(output_dir).rstrip('/')}/{rel_path}"
        argv = rsync_pull_argv(rig, self.cfg.ssh_key, remote, dest_dir,
                               bwlimit_kbps=getattr(self.cfg, "transfer_bwlimit_kbps", None))
        start = self.clock()
        while True:
            while self.is_recording(rig.name):           # never read from a recording rig
                if self.clock() - start > self.max_wait_s:
                    raise TransferError(f"rig kept recording for {self.max_wait_s:.0f} s; {rel_path} not copied")
                self.sleep(self.poll_s)
            rc, out = self.runner(argv, 900.0, None, abort=lambda: self.is_recording(rig.name))
            if rc == ABORTED_RC:                          # recording started mid-copy: resume later
                continue
            if rc != 0:
                raise TransferError(f"rsync {rel_path} failed ({rc}): {out[-300:]}")
            return

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
            if not expected:
                raise TransferError(f"Pi reported no size for {name}")
            got = (staging / name).stat().st_size
            if got != int(expected):
                raise TransferError(f"{name}: {got} bytes here, Pi reported {expected}")
        self.store.set_trial_state(rig.name, batch_dir, idx, "remuxing")
        return self.remux(video).name

    def process(self, rig, batch_dir: str, output_dir: str, t: dict) -> str:
        idx = int(t["index"])
        try:
            self.staging_dir(rig.name, batch_dir)
            for key in ("video", "frames_csv"):
                if not _is_plain(t.get(key)):
                    raise TransferError(f"refusing {key} name {t.get(key)!r}")
        except (TransferError, ValueError) as e:
            if _is_plain(batch_dir):
                self.store.upsert_trial(rig.name, batch_dir, t)
                self.store.set_trial_state(rig.name, batch_dir, idx, "failed", error=str(e))
            log.warning("%s trial %s refused: %s", rig.name, idx, e)
            return "failed"
        self.store.upsert_trial(rig.name, batch_dir, t)
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
