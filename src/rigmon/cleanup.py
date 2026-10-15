"""Daily cleanup: yesterday's (and older) trial traces and shadow staging copies.
Never mount-check files, never a landing session, never a rig's current batch,
never anything outside qc_root / staging_root, never through a symlink."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

log = logging.getLogger("rigmon.cleanup")
LANDING = ("pending", "pulling", "remuxing")


def _midnight(now: float) -> float:
    d = datetime.fromtimestamp(now)
    return d.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def _inside(path: Path, root: Path) -> bool:
    try:
        return root.resolve() in path.resolve().parents
    except OSError:
        return False


def _batches(root: Path | None):
    if root is None or root.is_symlink() or not root.is_dir():
        return
    for rig_dir in sorted(root.iterdir()):
        if rig_dir.is_symlink() or not rig_dir.is_dir():
            continue
        for batch_dir in sorted(rig_dir.iterdir()):
            if batch_dir.is_symlink() or not batch_dir.is_dir():
                continue
            yield rig_dir.name, batch_dir.name, batch_dir


def _old(store, rig, batch, path: Path, cutoff: float) -> bool:
    ended = store.batch_last_ended(rig, batch)
    if ended is None:
        ended = path.stat().st_mtime
    return ended < cutoff


def run_cleanup(cfg, store, current_batches: dict, now: float) -> dict:
    cutoff = _midnight(now)
    done = {"traces": [], "staging": []}
    for rig, batch, bdir in list(_batches(cfg.qc_root)):
        traces = bdir / "traces"
        try:
            if (current_batches.get(rig) == batch or traces.is_symlink() or not traces.is_dir()
                    or not _inside(traces, cfg.qc_root) or not _old(store, rig, batch, bdir, cutoff)):
                continue
            shutil.rmtree(traces)
            store.clear_traces(rig, batch)
            done["traces"].append(str(traces))
        except OSError:
            log.exception("cleanup of %s failed", traces)
    for rig, batch, bdir in list(_batches(cfg.staging_root)):
        try:
            session = store.get_session(rig, batch) or {}
            if (current_batches.get(rig) == batch or session.get("state") in LANDING
                    or not _inside(bdir, cfg.staging_root) or not _old(store, rig, batch, bdir, cutoff)):
                continue
            shutil.rmtree(bdir)
            done["staging"].append(str(bdir))
        except OSError:
            log.exception("cleanup of %s failed", bdir)
    log.info("cleanup removed %d trace dirs, %d staging dirs", len(done["traces"]), len(done["staging"]))
    return done
