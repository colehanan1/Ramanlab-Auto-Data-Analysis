"""Batch walking and the step entry points of the geometry pipeline.

Every step walks ``get_main_directories`` -> ``iter_live_batch_dirs`` like the
production steps, so freeze rules apply identically, as :class:`BatchJob`s
(:func:`iter_batch_jobs`). Outputs live under ``<job.out_dir>/<geometry.out_subdir>/``:

* Legacy (``gcfg.out_base == ""``): ``out_dir`` is the production batch itself, next to its videos.
* Integrated (an ``abdomen:`` overlay sets ``out_base``): production holds sidecars and
  ``session_metadata.txt`` but no mp4s, so each trial's video is looked up per trial in the secured
  mirrors, then in the local batch (:func:`videos_for_job`), and every output goes to
  ``<out_base>/<dataset folder>/<batch>/``. Production and the mirrors are only ever read.

With ``abdomen.slot_seeds: production`` (:func:`uses_production_seeds`) a trial's slots are numbered by production's own
fly numbers, read from the trial's production tables (:func:`trial_seeds`). :func:`trial_status` is the one place that
decides what a run does with a trial -- both ``run_track`` and ``pending_trials`` ask it -- and :func:`_clear_trial_outputs`
removes what a re-track replaces.

Integrated tracking yields to PER: before each video it asks :func:`~fbpipe.utils.run_locks.per_run_active` and, while a
PER run holds the shared run lock, stops cleanly (the trials left have no meta, so the next run does them).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from ..config import Settings, get_main_directories, infer_dataset_for_path
from ..utils.frozen_folders import is_frozen_folder, iter_live_batch_dirs
from ..utils.rig_gates import read_batch_date
from ..utils.run_locks import per_run_active
from . import columns as C
from .config import GeometrySettings
from .trial_meta import TrialInfo, protocol_sort_key, trial_info_for_video, trial_name_from_video

VIDEO_SUFFIXES = (".mp4", ".avi")
_MONTHS = {name: i for i, name in enumerate(
    ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
     "december"), start=1)}
_BATCH_NAME_DATE_RE = re.compile(r"^(?P<month>[A-Za-z]+)_(?P<day>\d{1,2})_batch")


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


@dataclass(frozen=True)
class BatchJob:
    """One production batch a step works on.

    ``dataset`` is the label :func:`iter_batches` yields (canonicalised against ``settings.datasets``);
    ``root_name`` is the production root folder's own name, which the secured mirror and the out tree are
    laid out by -- every path is built from it, never from the label. ``prod_dir`` is where batch
    metadata (``session_metadata.txt``, freeze dates) is read from, ``out_dir`` the batch dir that holds
    ``<out_subdir>/`` (legacy: the same directory as ``prod_dir``)."""
    dataset: str
    root_name: str
    name: str
    prod_dir: Path
    out_dir: Path
    frozen: bool


def _batch_date(prod_dir: Path, year: int) -> Optional[date]:
    """Recording date of a production batch: ``read_batch_date`` (sidecar stamps, then metadata lines), else the
    ``<month>_<dd>_batch`` prefix of its folder name read in ``year`` (folder names carry none); None if neither."""
    found = read_batch_date(prod_dir)
    if found is not None:
        return found
    m = _BATCH_NAME_DATE_RE.match(prod_dir.name)
    if m is None or m.group("month").lower() not in _MONTHS:
        return None
    try:
        return date(year, _MONTHS[m.group("month").lower()], int(m.group("day")))
    except ValueError:                       # february_30_batch_1
        return None


def iter_batch_jobs(settings: Settings, gcfg: GeometrySettings, *, live_only: bool = True) -> Iterator[BatchJob]:
    """The batches the steps work on, in :func:`iter_batches` order (it is wrapped, so freeze rules are judged on
    the production batch dir and ``live_only`` means what it means there).

    Legacy (``gcfg.out_base == ""``): exactly what ``iter_batches`` yields, ``out_dir`` the batch itself.
    Integrated: additionally restricted to ``gcfg.datasets`` (when non-empty; matched against the root folder name
    or the dataset label) and to batches recorded on or after ``gcfg.batch_date_from`` (an undated batch is
    excluded), with ``out_dir = <out_base>/<root folder>/<batch>``."""
    for batch, dataset, frozen in iter_batches(settings, live_only=live_only):
        root_name = batch.parent.name
        if not gcfg.out_base:
            yield BatchJob(dataset, root_name, batch.name, batch, batch, frozen)
            continue
        if gcfg.datasets and root_name not in gcfg.datasets and dataset not in gcfg.datasets:
            continue
        cutoff = gcfg.batch_date_from
        if cutoff is not None:
            recorded = _batch_date(batch, cutoff.year)
            if recorded is None or recorded < cutoff:
                continue
        yield BatchJob(dataset, root_name, batch.name, batch, Path(gcfg.out_base) / root_name / batch.name, frozen)


def iter_tracked_jobs(settings: Settings, gcfg: GeometrySettings, *, live_only: bool = True) -> Iterator[BatchJob]:
    """The jobs whose out batch already has a geometry dir -- all that normalize, wide, figures and the stats read.
    Nothing but ``<out_dir>/<out_subdir>/`` is needed: the out tree holds no videos, sidecars or metadata."""
    for job in iter_batch_jobs(settings, gcfg, live_only=live_only):
        if geometry_dir(job.out_dir, gcfg).is_dir():
            yield job


def videos_for_job(job: BatchJob, gcfg: GeometrySettings) -> List[Path]:
    """The trial videos of a batch, one per trial, sorted by file name.

    Legacy: the batch dir's own. Integrated: production holds no mp4s, so each trial's video is the first one found
    in ``<video_base>/<root folder>/<batch>`` (``gcfg.video_bases`` order), then in the local production batch --
    decided per trial, so a batch whose videos are only partly on the mirror still yields every trial."""
    if not gcfg.out_base:
        return iter_videos(job.prod_dir)
    by_trial: Dict[str, Path] = {}
    for source in [Path(base) / job.root_name / job.name for base in gcfg.video_bases] + [job.prod_dir]:
        if source.is_dir():
            for video in iter_videos(source):
                by_trial.setdefault(trial_name_from_video(video), video)
    return sorted(by_trial.values(), key=lambda p: p.name)


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


def effective_batch_size(settings: Settings, is_engine: bool) -> int:
    """Production's rule: a TensorRT engine only takes batches when it was exported
    dynamic (``yolo.engine_supports_batch: true``); otherwise it runs frame by frame."""
    if is_engine and not settings.engine_supports_batch:
        return 1
    return max(1, int(settings.inference_batch_size))


def _cap_cpu_threads(n_workers: int) -> None:
    """N workers must not each grab every core (see scripts/pipeline/parallel_yolo.py)."""
    if n_workers <= 1:
        return
    threads = max(1, (os.cpu_count() or n_workers) // n_workers)
    try:
        import cv2
        cv2.setNumThreads(threads)
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(threads)
    except Exception:
        pass


def trial_has_meta(job: BatchJob, gcfg: GeometrySettings, video: Path) -> bool:
    """Whether the trial's meta JSON exists in the out tree -- the marker of "tracked at all". A trial that failed
    writes none, so the next run retries it. Whether a tracked trial is still *current* is :func:`trial_status`."""
    return C.meta_path(trial_out_dir(job.out_dir, gcfg, video), trial_name_from_video(video)).exists()


TRACKED, PENDING, AWAITING_SEEDS = "tracked", "pending", "awaiting_seeds"


@dataclass(frozen=True)
class TrialStatus:
    """What a run does with one trial video. ``state``: ``TRACKED`` (skip it without ``--force``), ``PENDING`` (track it)
    or ``AWAITING_SEEDS`` (production has not numbered this trial's flies yet: leave it, write no meta). ``seeds`` is
    what to number the slots by (None unless production seeds are in use); ``stale`` marks a pending trial whose meta
    exists but is out of date, so its outputs must be cleared before the re-track, and ``reason`` says why."""
    state: str
    seeds: Optional[Dict[int, Tuple[float, float]]] = None
    stale: bool = False
    reason: str = ""


def uses_production_seeds(gcfg: GeometrySettings) -> bool:
    """Integrated mode with ``abdomen.slot_seeds: production``. Warm-up numbering and legacy mode never look at seeds."""
    return bool(gcfg.out_base) and gcfg.slot_seeds == "production"


def trial_seeds(job: BatchJob, gcfg: GeometrySettings, trial_name: str) -> Dict[int, Tuple[float, float]]:
    """Production's eye anchors of one trial, from the first of these folders that has any: the production trial folder,
    then the same trial folder in each secured mirror (``video_bases`` order). ``{}`` = production has not numbered it."""
    from .track import production_eye_seeds
    folders = [job.prod_dir / trial_name, *(Path(base) / job.root_name / job.name / trial_name
                                            for base in gcfg.video_bases)]
    for folder in folders:
        seeds = production_eye_seeds(folder)
        if seeds:
            return seeds
    return {}


def trial_status(job: BatchJob, gcfg: GeometrySettings, video: Path) -> TrialStatus:
    """The single definition of what a trial needs, shared by ``run_track`` and ``pending_trials``.

    Warm-up numbering and legacy mode: tracked iff its meta JSON exists. Production seeds: a trial with no seeds is
    ``AWAITING_SEEDS`` (never pending, so the nightly does not start workers for trials PER has not processed); with
    seeds it is pending when it has no meta, or its meta was not tracked from production seeds, or recorded a different
    :func:`~fbpipe.geometry.track.seed_digest` -- an unreadable meta counts as out of date."""
    name = trial_name_from_video(video)
    meta_file = C.meta_path(trial_out_dir(job.out_dir, gcfg, video), name)
    has_meta = trial_has_meta(job, gcfg, video)
    if not uses_production_seeds(gcfg):
        return TrialStatus(TRACKED if has_meta else PENDING)
    seeds = trial_seeds(job, gcfg, name)
    if not seeds:
        return TrialStatus(AWAITING_SEEDS)
    if not has_meta:
        return TrialStatus(PENDING, seeds, reason="no meta")
    from .track import seed_digest
    try:
        recorded = json.loads(meta_file.read_text(encoding="utf-8"))
        source, digest = recorded.get("slot_source"), recorded.get("seed_digest")
    except (OSError, ValueError, AttributeError):
        return TrialStatus(PENDING, seeds, stale=True, reason="its meta cannot be read")
    if source != "production":
        return TrialStatus(PENDING, seeds, stale=True, reason="tracked with warm-up slot numbers, not production's")
    if digest != seed_digest(seeds):
        return TrialStatus(PENDING, seeds, stale=True, reason="production's fly numbers changed")
    return TrialStatus(TRACKED, seeds)


def pending_trials(settings: Settings, gcfg: GeometrySettings) -> int:
    """How many trial videos of the live batches ``run_track`` would track without ``--force`` (:func:`trial_status`).
    A trial awaiting production seeds is not counted: there is nothing a worker could do for it yet. The CLI starts no
    worker, and loads no model, when this is 0."""
    return sum(1 for job in iter_batch_jobs(settings, gcfg) for video in videos_for_job(job, gcfg)
               if trial_status(job, gcfg, video).state == PENDING)


def _clear_trial_outputs(trial_dir: Path, trial_name: str, *, out_base: str | Path) -> List[Path]:
    """Delete one trial's geometry outputs: its ``<trial>_fly{N}_geometry.{csv,parquet}`` tables, annotated mp4 and meta
    JSON. A re-track clears first, so a slot the new numbering lacks cannot keep a table that a join would pick up.
    REFUSES (``ValueError``) any ``trial_dir`` that is not a folder strictly inside the resolved ``out_base`` (symlinks
    and ``..`` resolved, compared by path component), and an empty ``out_base``; nothing else is ever deleted. Returns
    the removed paths."""
    if not str(out_base).strip():
        raise ValueError("refusing to clear a trial's outputs without an out_base to confine the deletion to")
    root = Path(out_base).expanduser().resolve()
    folder = Path(trial_dir).expanduser().resolve()
    if folder == root or not folder.is_relative_to(root):
        raise ValueError(f"refusing to clear {trial_dir}: it is not a folder inside the out tree {root}")
    if not folder.is_dir():
        return []
    victims = [p for p in sorted(folder.iterdir())
               if p.name.startswith(f"{trial_name}_fly") and C.fly_slot_from_geometry_name(p.name) is not None]
    victims += [C.annotated_video_path(folder, trial_name), C.meta_path(folder, trial_name)]
    removed = []
    for path in victims:
        if path.is_file() or path.is_symlink():
            path.unlink()
            removed.append(path)
    return removed


def write_source_json(job: BatchJob, gcfg: GeometrySettings, videos: List[Path]) -> Path:
    """Integrated mode: ``<out_dir>/<out_subdir>/source.json`` = ``{"prod_dir", "video_dir"}``, where this out
    batch's inputs came from (``video_dir``: the directory the batch's first video was read from). A record
    already on disk with the same content is left alone."""
    path = C.source_path(geometry_dir(job.out_dir, gcfg))
    record = {"prod_dir": str(job.prod_dir), "video_dir": str(videos[0].parent)}
    try:
        if json.loads(path.read_text(encoding="utf-8")) == record:
            return path
    except (OSError, ValueError):
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def run_track(settings: Settings, gcfg: GeometrySettings, *, force: bool = False,
              only_trial: Optional[str] = None, max_frames: Optional[int] = None) -> dict:
    """Detect + assemble + measure every live trial video; skips the trials that are already tracked (:func:`trial_status`).

    One bad video never stops the batch: any error while tracking it is logged (``[GEOMETRY] failed <video>:
    <error>``), counted in the result's ``failed`` and leaves no meta JSON behind, so the next run retries it.
    Integrated mode also records each batch's ``source.json`` before its first video is tracked.

    With production slot seeds a trial's slots take production's fly numbers. A trial whose seeds are not there yet is
    counted under ``awaiting_seeds`` and left exactly as it is -- nothing is written for it, even with ``force``, since
    it has no number to give its slots; one whose meta records other numbers has its old outputs cleared first, then is
    tracked again.

    Integrated mode yields to a PER run: before each video it would track, ``per_run_active(settings.cache_dir)`` is
    asked, and while it is True the loop stops (``[GEOMETRY] PER run active — yielding``), writing nothing for the
    video it declined or any after it; the result then carries ``"yielded": True`` and the trials left resume next run.
    Legacy mode never asks."""
    import ultralytics
    from .detect import id_to_label_map, is_engine_path, load_model, make_predict_fn, model_sha1, resolve_model_path
    from .track import track_video

    model_path = resolve_model_path(Path(settings.model_path), gcfg.prefer_engine)
    is_engine = is_engine_path(model_path)
    model = load_model(model_path)
    id_to_label = id_to_label_map(model.names)
    predict_fn = make_predict_fn(model, imgsz=gcfg.imgsz, iou=gcfg.iou, allow_cpu=settings.allow_cpu,
                                 is_engine=is_engine)
    batch_size = effective_batch_size(settings, is_engine)
    sha = model_sha1(model_path)
    class_map = {label: cid for cid, label in id_to_label.items()}
    n_workers, worker = worker_shard()
    _cap_cpu_threads(n_workers)
    print(f"[GEOMETRY] model {model_path} ({'TensorRT engine' if is_engine else 'PyTorch'}, sha1 {sha[:10]}) "
          f"classes {class_map} ultralytics {ultralytics.__version__} batch {batch_size} "
          f"worker {worker + 1}/{n_workers}")
    seq = -1
    done = skipped = failed = awaiting = 0
    yielded = False
    for job in iter_batch_jobs(settings, gcfg):
        videos = videos_for_job(job, gcfg)
        source_recorded = not gcfg.out_base                      # legacy batches record no source
        for video in videos:
            seq += 1
            if seq % n_workers != worker:
                continue
            name = trial_name_from_video(video)
            if only_trial and only_trial not in name:
                continue
            out_dir = trial_out_dir(job.out_dir, gcfg, video)
            status = trial_status(job, gcfg, video)
            if status.state == AWAITING_SEEDS:
                print(f"[GEOMETRY] skip {video.name}: awaiting production seeds ({name}_fly*_distances not found in "
                      f"{job.prod_dir / name} or the secured mirrors); retried next run")
                awaiting += 1
                continue
            if status.state == TRACKED and not force:
                print(f"[GEOMETRY] skip {video.name}: {out_dir.name} already tracked (use --force to redo)")
                skipped += 1
                continue
            if gcfg.out_base and per_run_active(settings.cache_dir):
                print("[GEOMETRY] PER run active — yielding; remaining trials resume next run")
                yielded = True
                break
            t0 = time.time()
            try:
                if status.stale:
                    removed = _clear_trial_outputs(out_dir, name, out_base=gcfg.out_base)
                    print(f"[GEOMETRY] {name}: {status.reason}; cleared {len(removed)} old output file(s), tracking again")
                if not source_recorded:
                    write_source_json(job, gcfg, videos)
                    source_recorded = True
                info = trial_info_for_video(video, job.prod_dir, out_dir, settings)
                annotated = C.annotated_video_path(out_dir, name) if gcfg.annotated_video else None
                result = track_video(video, info, predict_fn=predict_fn, id_to_label=id_to_label, settings=settings,
                                     gcfg=gcfg, max_frames=max_frames, annotated_path=annotated,
                                     batch_size=batch_size, seed_centers=status.seeds)
                for slot, df in result.tables.items():
                    C.write_table_fmt(df, C.geometry_table_path(out_dir, name, slot, gcfg.table_format),
                                      gcfg.table_format)
                meta = dict(result.meta, model_path=str(model_path), model_is_engine=is_engine, model_sha1=sha,
                            batch_size=batch_size, ultralytics_version=ultralytics.__version__, class_map=class_map,
                            dataset=job.dataset)
                C.meta_path(out_dir, name).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
            except Exception as exc:                             # one bad video must not stop the batch
                print(f"[GEOMETRY] failed {video}: {type(exc).__name__}: {exc}")
                failed += 1
                continue
            print(f"[GEOMETRY] {video.name}: {len(result.tables)} slot(s), {result.meta['n_frames_processed']} frames "
                  f"in {time.time() - t0:.1f}s -> {out_dir}")
            done += 1
        if yielded:
            break
    waiting = f", {awaiting} awaiting production seeds" if uses_production_seeds(gcfg) else ""
    stopped = ", stopped early for a PER run" if yielded else ""
    print(f"[GEOMETRY] track done: {done} trial(s) tracked, {skipped} skipped, {failed} failed{waiting}{stopped}")
    return {"tracked": done, "skipped": skipped, "failed": failed, "awaiting_seeds": awaiting, "yielded": yielded}


def run_normalize(settings: Settings, gcfg: GeometrySettings, *, force: bool = False) -> dict:
    """Per-fly stats + % columns for every live batch with a geometry dir. Always recomputed (cheap, idempotent)."""
    from .normalize import normalize_batch
    summary = {}
    for job in iter_tracked_jobs(settings, gcfg):
        result = normalize_batch(job.out_dir, settings, gcfg)
        if result["n_trials"]:
            summary[str(job.out_dir)] = result["slots"]
    print(f"[GEOMETRY] normalize done: {len(summary)} batch(es)")
    return summary


def run_wide(settings: Settings, gcfg: GeometrySettings) -> dict:
    """Wide tables over every batch with a geometry dir (frozen ones included, marked) -> geometry.wide_out_dir."""
    from .wide import WIDE_MEASURES, batch_wide_rows, build_wide_tables, write_wide_tables
    merged = {m: [] for m in WIDE_MEASURES}
    for job in iter_tracked_jobs(settings, gcfg, live_only=False):
        for measure, rows in batch_wide_rows(job.out_dir, job.dataset, job.frozen, gcfg).items():
            merged[measure].extend(rows)
    tables = build_wide_tables(merged)
    files = write_wide_tables(tables, Path(gcfg.wide_out_dir), gcfg.table_format)
    for measure, path in files.items():
        print(f"[GEOMETRY] wide {measure}: {len(tables[measure])} row(s) -> {path}")
    return {"files": files, "rows": {m: len(t) for m, t in tables.items()}}


def run_figures(settings: Settings, gcfg: GeometrySettings) -> dict:
    """Per-fly trace figures + QC + trial summary for every live batch with a geometry dir
    -> geometry.figures_out_dir."""
    from .figures import generate
    total = 0
    outputs = []
    for job in iter_tracked_jobs(settings, gcfg):
        result = generate(job.out_dir, job.dataset, settings, gcfg, Path(gcfg.figures_out_dir))
        if result["n_figures"]:
            total += result["n_figures"]
            outputs.append(result["out_dir"])
            print(f"[GEOMETRY] figures {job.name}: {result['n_figures']} figure(s) -> {result['out_dir']}")
    return {"n_figures": total, "out_dirs": outputs}
