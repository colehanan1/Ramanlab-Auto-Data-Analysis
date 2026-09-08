"""Batch walking and the step entry points of the geometry pipeline.

Every step walks ``get_main_directories`` -> ``iter_live_batch_dirs`` like the
production steps, so freeze rules apply identically. Outputs live under
``<batch>/<geometry.out_subdir>/``.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from ..config import Settings, get_main_directories, infer_dataset_for_path
from ..utils.frozen_folders import is_frozen_folder, iter_live_batch_dirs
from . import columns as C
from .config import GeometrySettings
from .trial_meta import TrialInfo, protocol_sort_key, trial_info_for_video, trial_name_from_video

VIDEO_SUFFIXES = (".mp4", ".avi")


def dataset_for_root(settings: Settings, root: Path) -> str:
    return infer_dataset_for_path(settings, root) or Path(root).name


def iter_batches(settings: Settings, *, live_only: bool = True) -> Iterator[Tuple[Path, str, bool]]:
    """``(batch_dir, dataset, frozen)`` for every batch under every root.
    ``live_only`` (the heavy steps) skips frozen folders; the wide step passes
    False to keep frozen rows in the table, marked."""
    for root in sorted(get_main_directories(settings), key=str):
        if not root.is_dir():
            print(f"[GEOMETRY] main_directories entry does not exist: {root}")
            continue
        dataset = dataset_for_root(settings, root)
        if live_only:
            for batch in iter_live_batch_dirs(settings, root, dataset=dataset):
                yield batch, dataset, False
        else:
            for batch in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name):
                yield batch, dataset, is_frozen_folder(settings, dataset, batch)


def iter_videos(batch_dir: Path) -> List[Path]:
    return sorted((f for f in Path(batch_dir).iterdir() if f.is_file() and f.suffix.lower() in VIDEO_SUFFIXES),
                  key=lambda p: p.name)


def geometry_dir(batch_dir: Path, gcfg: GeometrySettings) -> Path:
    return Path(batch_dir) / gcfg.out_subdir


def trial_out_dir(batch_dir: Path, gcfg: GeometrySettings, video: Path) -> Path:
    return geometry_dir(batch_dir, gcfg) / trial_name_from_video(video)


def load_trial_entries(batch_dir: Path, gcfg: GeometrySettings) -> List[Tuple[Path, TrialInfo, Dict[int, Path]]]:
    """Every tracked trial of a batch: ``(trial_dir, info, {slot: table})`` in protocol order."""
    gdir = geometry_dir(batch_dir, gcfg)
    if not gdir.is_dir():
        return []
    out = []
    for trial_dir in sorted(p for p in gdir.iterdir() if p.is_dir()):
        meta_file = C.meta_path(trial_dir, trial_dir.name)
        if not meta_file.is_file():
            continue
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        info = TrialInfo.from_dict(meta["trial"])
        out.append((trial_dir, info, dict(C.iter_geometry_tables(trial_dir))))
    out.sort(key=lambda e: protocol_sort_key(e[1]))
    return out


def worker_shard() -> Tuple[int, int]:
    """``(NUM_WORKERS, WORKER_INDEX)`` from the environment, as ``yolo_infer`` reads them."""
    n = max(1, int(os.getenv("NUM_WORKERS", "1")))
    return n, int(os.getenv("WORKER_INDEX", "0")) % n


def run_track(settings: Settings, gcfg: GeometrySettings, *, force: bool = False,
              only_trial: Optional[str] = None, max_frames: Optional[int] = None) -> dict:
    """Detect + assemble + measure every live trial video; skips trials whose meta JSON exists."""
    import ultralytics
    from .detect import id_to_label_map, load_model, make_predict_fn, model_sha1
    from .track import track_video

    model = load_model(settings.model_path)
    id_to_label = id_to_label_map(model.names)
    predict_fn = make_predict_fn(model, imgsz=gcfg.imgsz, iou=gcfg.iou, allow_cpu=settings.allow_cpu)
    sha = model_sha1(Path(settings.model_path))
    class_map = {label: cid for cid, label in id_to_label.items()}
    print(f"[GEOMETRY] model {settings.model_path} (sha1 {sha[:10]}) classes {class_map} ultralytics {ultralytics.__version__}")
    n_workers, worker = worker_shard()
    job = -1
    done = skipped = 0
    for batch_dir, dataset, _ in iter_batches(settings):
        for video in iter_videos(batch_dir):
            job += 1
            if job % n_workers != worker:
                continue
            name = trial_name_from_video(video)
            if only_trial and only_trial not in name:
                continue
            out_dir = trial_out_dir(batch_dir, gcfg, video)
            if C.meta_path(out_dir, name).exists() and not force:
                print(f"[GEOMETRY] skip {video.name}: {out_dir.name} already tracked (use --force to redo)")
                skipped += 1
                continue
            info = trial_info_for_video(video, batch_dir, out_dir, settings)
            t0 = time.time()
            annotated = C.annotated_video_path(out_dir, name) if gcfg.annotated_video else None
            result = track_video(video, info, predict_fn=predict_fn, id_to_label=id_to_label, settings=settings,
                                 gcfg=gcfg, max_frames=max_frames, annotated_path=annotated)
            for slot, df in result.tables.items():
                C.write_table_fmt(df, C.geometry_table_path(out_dir, name, slot, gcfg.table_format), gcfg.table_format)
            meta = dict(result.meta, model_path=str(settings.model_path), model_sha1=sha,
                        ultralytics_version=ultralytics.__version__, class_map=class_map, dataset=dataset)
            C.meta_path(out_dir, name).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
            print(f"[GEOMETRY] {video.name}: {len(result.tables)} slot(s), {result.meta['n_frames_processed']} frames "
                  f"in {time.time() - t0:.1f}s -> {out_dir}")
            done += 1
    print(f"[GEOMETRY] track done: {done} trial(s) tracked, {skipped} skipped")
    return {"tracked": done, "skipped": skipped}


def run_normalize(settings: Settings, gcfg: GeometrySettings, *, force: bool = False) -> dict:
    """Per-fly stats + % columns for every live batch. Always recomputed (cheap, idempotent)."""
    from .normalize import normalize_batch
    summary = {}
    for batch_dir, dataset, _ in iter_batches(settings):
        result = normalize_batch(batch_dir, settings, gcfg)
        if result["n_trials"]:
            summary[str(batch_dir)] = result["slots"]
    print(f"[GEOMETRY] normalize done: {len(summary)} batch(es)")
    return summary


def run_wide(settings: Settings, gcfg: GeometrySettings) -> dict:
    """Wide tables over every batch (frozen ones included, marked) -> geometry.wide_out_dir."""
    from .wide import WIDE_MEASURES, batch_wide_rows, build_wide_tables, write_wide_tables
    merged = {m: [] for m in WIDE_MEASURES}
    for batch_dir, dataset, frozen in iter_batches(settings, live_only=False):
        for measure, rows in batch_wide_rows(batch_dir, dataset, frozen, gcfg).items():
            merged[measure].extend(rows)
    tables = build_wide_tables(merged)
    files = write_wide_tables(tables, Path(gcfg.wide_out_dir), gcfg.table_format)
    for measure, path in files.items():
        print(f"[GEOMETRY] wide {measure}: {len(tables[measure])} row(s) -> {path}")
    return {"files": files, "rows": {m: len(t) for m, t in tables.items()}}
