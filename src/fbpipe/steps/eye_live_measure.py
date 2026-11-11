"""Per-trial step: measure from the per-frame eye, held while the light is on.

For each dataset listed in ``eye_measure.datasets`` with ``eye_measure.mode:
live_light_hold``, every per-fly trial table is re-measured from the eye YOLO
detected in that frame (``eye_live_x/y``; the frozen anchor where none was
matched), except from ``hold_before_s`` before light-on to ``hold_after_s``
after light-off, where the eye stays where it sat over the ``pre_hold_s``
before -- YOLO misplaces the eye under the stimulus light while the head stays
put. The frozen anchor is kept in ``x_class0_anchor``/``y_class0_anchor``.

Any other dataset whose tables carry that kept anchor is put back on it, so
taking a dataset off the list (or setting ``mode: frozen``) undoes the change
on the next run. Tables without per-frame eye columns are never touched.

Runs after YOLO and before ``reject_bad_proboscis``: the proboscis gate and
every downstream number then measure from the same eye. Re-running is a
no-op: the kept anchor is always the starting point.
"""
from __future__ import annotations

import re
from functools import partial
from pathlib import Path

from ..config import Settings, get_main_directories
from ..utils.frozen_folders import iter_live_batch_dirs
from ..utils.light_frames import trial_meta_for_table
from ..utils.live_eye import (
    BASE_X,
    apply_live_eye_held,
    has_live_eye,
    restore_fixed_eye,
    stimulus_hold,
)
from ..utils.parallel import parallel_map
from ..utils.tables import read_schema_columns, read_table, write_table

TRIAL_RE = re.compile(r"(pretest|training|testing)_\d+$", re.IGNORECASE)
TABLE_RE = re.compile(r"_fly\d+_distances\.(parquet|csv)$")


def _trial_tables(batch: Path):
    for trial in sorted(p for p in batch.iterdir() if p.is_dir() and TRIAL_RE.search(p.name)):
        seen = set()
        for path in sorted(trial.iterdir()):
            if not TABLE_RE.search(path.name) or path.name.startswith("updated_"):
                continue
            stem = path.name.rsplit(".", 1)[0]
            if stem in seen:  # a .csv twin of a .parquet table
                continue
            seen.add(stem)
            yield path


def _process_batch(batch: Path, cfg: Settings, live: bool) -> tuple[int, int]:
    """(tables re-measured, tables restored) for one batch folder."""
    em = cfg.eye_measure
    changed = restored = 0
    for path in _trial_tables(batch):
        try:
            cols = set(read_schema_columns(path))
        except Exception as exc:
            print(f"[EYE-LIVE] cannot read {path.name}: {exc}")
            continue
        if live:
            if not {"eye_live_x", "eye_live_y"} <= cols:
                continue
            df = read_table(path)
            if not has_live_eye(df) or "frame" not in df.columns:
                continue
            meta = trial_meta_for_table(path, batch)
            fps = float(meta.get("fps") or getattr(cfg, "fps_default", 40.0) or 40.0)
            hold = stimulus_hold(meta, odor=False, before_s=em.hold_before_s, after_s=em.hold_after_s)
            out = apply_live_eye_held(df, hold, int(round(em.pre_hold_s * fps)))
            write_table(out, path)
            changed += 1
        elif BASE_X in cols:
            write_table(restore_fixed_eye(read_table(path)), path)
            restored += 1
    return changed, restored


def main(cfg: Settings) -> None:
    em = getattr(cfg, "eye_measure", None)
    if em is None or (em.mode == "frozen" and not em.datasets):
        # the default: frozen eye everywhere. Nothing to do -- list datasets
        # under mode: frozen to put a previously switched dataset back.
        print("[EYE-LIVE] eye_measure is frozen for every dataset; nothing to do.")
        return
    listed = set(getattr(em, "datasets", ()) or ()) if em is not None and em.mode == "live_light_hold" else set()
    for root in get_main_directories(cfg):
        root = Path(root)
        if not root.is_dir():
            continue
        live = root.name in listed
        batches = iter_live_batch_dirs(cfg, root)
        if not batches:
            continue
        results = parallel_map(partial(_process_batch, cfg=cfg, live=live), batches,
                               enabled=cfg.parallel.enabled, n_jobs=cfg.parallel.n_jobs)
        changed = sum(r[0] for r in results)
        restored = sum(r[1] for r in results)
        if live:
            print(f"[EYE-LIVE] {root.name}: {changed} trial tables measured from the per-frame eye "
                  f"(held while the light is on).")
        elif restored:
            print(f"[EYE-LIVE] {root.name}: {restored} trial tables put back on the frozen eye anchor.")
