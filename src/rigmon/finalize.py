"""Land a finished session: pull what is left, remux it, verify against the
Pi's manifest, then one rename from staging into flys_New. Never merges into
an existing folder; an abort from the Pi always wins."""

from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path, PurePosixPath

log = logging.getLogger("rigmon.finalize")


class FinalizeError(RuntimeError):
    pass


def dest_from_push_to(push_to: str, batch_dir: str, data_root) -> Path:
    path = push_to.split(":", 1)[1] if ":" in push_to else push_to
    if not path.strip():
        raise FinalizeError("the Pi sent no push_to destination")
    if not Path(path).is_absolute():
        raise FinalizeError(f"push_to path is not absolute: {path!r}")
    if batch_dir in ("", ".", "..") or "/" in batch_dir:
        raise FinalizeError(f"bad batch_dir {batch_dir!r}")
    dest = (Path(path) / batch_dir).resolve()
    root = Path(data_root).resolve()
    if root not in dest.parents:
        raise FinalizeError(f"{dest} is outside {root}")
    return dest


def _safe_rel(rel: str) -> str:
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise FinalizeError(f"refusing manifest path {rel!r}")
    return rel


def outstanding(manifest: list[dict], staging: Path) -> list[str]:
    """Manifest paths not in staging at the reported size. A .h264 counts as
    present once its .mp4 exists (the remux deletes the source)."""
    out = []
    for item in manifest:
        rel = item["path"]
        p = staging / rel
        if rel.endswith(".h264"):
            mp4 = p.with_suffix(".mp4")
            if mp4.exists() and mp4.stat().st_size > 0:
                continue
        if not p.exists() or p.stat().st_size != int(item["bytes"]):
            out.append(rel)
    return out


class Finalizer:
    def __init__(self, cfg, store, transfer,
                 disk_free=lambda p: shutil.disk_usage(p).free,
                 alert=lambda title, body: None):
        self.cfg, self.store, self.transfer = cfg, store, transfer
        self.disk_free, self.alert = disk_free, alert
        self._move_lock = threading.Lock()

    def _state(self, rig_name: str, batch_dir: str) -> str:
        s = self.store.get_session(rig_name, batch_dir)
        return s["state"] if s else ""

    def abort(self, rig_name: str, batch_dir: str) -> str:
        with self._move_lock:
            if self._state(rig_name, batch_dir) == "landed":
                return "landed"
            self.store.set_session(rig_name, batch_dir, "aborted")
            return "aborted"

    def run(self, rig, batch_dir, output_dir, push_to, manifest) -> str:
        try:
            return self._run(rig, batch_dir, output_dir, push_to, manifest)
        except Exception as e:
            log.warning("finalize %s/%s failed: %s", rig.name, batch_dir, e)
            self.store.set_session(rig.name, batch_dir, "failed", error=str(e))
            self.alert(f"{rig.label} finalize failed", f"{batch_dir}: {e}")
            return "failed"

    def _run(self, rig, batch_dir, output_dir, push_to, manifest) -> str:
        state = self._state(rig.name, batch_dir)
        if state in ("aborted", "landed"):
            return state
        dest = dest_from_push_to(push_to, batch_dir, self.cfg.data_root)
        stage_root = Path(self.cfg.staging_root).resolve()
        if dest == stage_root or stage_root in dest.parents:
            raise FinalizeError(f"destination {dest} is inside the staging area")
        for item in manifest:
            _safe_rel(item["path"])
        staging = self.transfer.staging_dir(rig.name, batch_dir)
        staging.mkdir(parents=True, exist_ok=True)

        self.store.set_session(rig.name, batch_dir, "pulling")
        for rel in outstanding(manifest, staging):
            self.transfer.pull(rig, output_dir, rel, staging)
        self.store.set_session(rig.name, batch_dir, "remuxing")
        for h264 in sorted(staging.rglob("*.h264")):
            self.transfer.remux(h264)

        problems = outstanding(manifest, staging)
        if problems:
            raise FinalizeError(f"{len(problems)} file(s) missing or wrong size: {problems[:5]}")
        if dest.exists():
            raise FinalizeError(f"destination already exists: {dest} (staging kept at {staging})")
        probe = dest.parent if dest.parent.exists() else self.cfg.data_root
        free_gb = self.disk_free(probe) / 1e9
        if free_gb < self.cfg.min_free_gb:
            raise FinalizeError(f"only {free_gb:.0f} GB free on {probe}")
        with self._move_lock:
            if self._state(rig.name, batch_dir) == "aborted":
                return "aborted"
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.rename(staging, dest)  # same filesystem → atomic; EXDEV → OSError → failed
            self.store.set_session(rig.name, batch_dir, "landed", dest=str(dest))
            return "landed"
