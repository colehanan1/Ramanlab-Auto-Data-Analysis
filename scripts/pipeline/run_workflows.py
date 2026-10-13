#!/usr/bin/env python3
"""Run the full pipeline plus optional analysis workflows defined in config/config.yaml."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
LOGGER = logging.getLogger(__name__)

# Ensure local source tree takes precedence over any installed package.
def _find_repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return start.parents[2]


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
SRC_ROOT = REPO_ROOT / "src"
for path in (str(SRC_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

# v2: manifest tracks Parquet + CSV, records content hashes, and uses
# fly_dir/trial_dir/file depth (>=3 parts). Bumping invalidates v1 caches.
STATE_VERSION = 2

#: One line about this run's auto-freeze outcome, for the end-of-run ntfy.
_RUN_SUMMARY = ""

from fbpipe.analysis.threshold import ThresholdRule
from fbpipe.config import Settings, discover_flagged_directories, load_settings, resolve_config_path, _expand_datasets
from fbpipe.pipeline import ORDERED_STEPS
from fbpipe.steps import predict_reactions
from fbpipe.utils.data_secured_sync import sync_data_secured
from fbpipe.utils.run_lock import RunLockTimeout, run_lock
from fbpipe.utils.smb_copy import copy_to_smb

from scripts.analysis.envelope_visuals import (
    EnvelopePlotConfig,
    MatrixPlotConfig,
    NoTargetTrialsError,
    generate_envelope_plots,
    generate_reaction_matrices,
    load_light_check_fractions,
    set_dataset_light_windows,
    set_dataset_odor_remap,
    set_light_check_fractions,
    set_model_scores,
    set_protocol,
)
from scripts.analysis.envelope_training import latency_reports
from scripts.analysis.envelope_combined import (
    CombineConfig,
    LOW_MAX_FLAG_THRESHOLD_PX,
    build_wide_csv,
    combine_distance_angle,
    mirror_directory,
    overlay_sources,
    secure_copy_and_cleanup,
    wide_to_matrix,
)


def _resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _file_mtime_key(path: Path | str | None) -> float | None:
    """Return mtime of *path* as a cache fingerprint, or None if missing."""
    if path is None:
        return None
    p = Path(path)
    if p.exists():
        return p.stat().st_mtime
    return None


def _analysis_expected(cfg: Mapping[str, Any] | None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a lightweight expected-state dict for an analysis stage.

    Captures the mtimes of the input artifacts (matrix_npy, codes_json) so
    that re-runs with unchanged inputs can be skipped.
    """
    out: dict[str, Any] = {}
    if cfg:
        matrix_npy = _resolve_path(cfg.get("matrix_npy") or (cfg.get("matrices") or {}).get("matrix_npy"))
        codes_json = _resolve_path(cfg.get("codes_json") or (cfg.get("matrices") or {}).get("codes_json"))
        out["matrix_npy_mtime"] = _file_mtime_key(matrix_npy)
        out["codes_json_mtime"] = _file_mtime_key(codes_json)
        # Include config keys that would change the outputs
        out["overwrite"] = cfg.get("overwrite") or (cfg.get("matrices") or {}).get("overwrite")
    if extra:
        out.update(extra)
    return out


def _dataset_means_expected(analysis_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Build expected-state for dataset_means from the wide CSV mtime."""
    combined = analysis_cfg.get("combined") or {}
    # config_new keeps the wide table under combined.combined_base.wide; only
    # older configs have combined.wide. Reading just the latter made this key a
    # constant {None, None}, so the gate could never see that the data changed.
    wide = (
        (combined.get("combined_base") or {}).get("wide")
        or combined.get("wide")
        or {}
    )
    wide_csv = _resolve_path(wide.get("output_csv"))
    # Also check training export if present
    training_csv_mtime: float | None = None
    exports = wide.get("trial_type_exports") or []
    if isinstance(exports, Sequence) and not isinstance(exports, (str, bytes)):
        for entry in exports:
            if isinstance(entry, Mapping) and str(entry.get("trial_type", "")).lower() == "training":
                training_csv_mtime = _file_mtime_key(_resolve_path(entry.get("output_csv")))
    return {
        "wide_csv_mtime": _file_mtime_key(wide_csv),
        "training_csv_mtime": training_csv_mtime,
    }


def _ensure_path(value: str | Path, field: str) -> Path:
    path = _resolve_path(value)
    if path is None:
        raise ValueError(f"Missing required path for '{field}'.")
    return path


def _resolve_folder(folder_name: str, dataset_roots: Sequence[str]) -> str:
    """Find the dataset root whose directory name matches *folder_name* (case-insensitive).

    Raises ValueError with available folder names when no match is found.
    """
    target = folder_name.strip().lower()
    matches = [
        root for root in dataset_roots
        if Path(root).name.lower() == target
    ]
    if not matches:
        available = sorted(Path(root).name for root in dataset_roots)
        raise ValueError(
            f"Folder '{folder_name}' not found in main_directories. "
            f"Available: {available}"
        )
    if len(matches) > 1:
        LOGGER.warning(
            "Multiple roots match folder '%s': %s; using first.", folder_name, matches
        )
    return matches[0]


def _cache_root(settings: Settings) -> Path:
    path = Path(settings.cache_dir or (REPO_ROOT / "cache"))
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_key(value: str) -> str:
    text = str(Path(value).expanduser()) if value else ""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return cleaned.strip("_") or "default"


def _state_path(settings: Settings, category: str, key: str) -> Path:
    return _cache_root(settings) / category / _safe_key(key) / "state.json"


def _load_state(settings: Settings, category: str, key: str) -> dict[str, Any] | None:
    state_path = _state_path(settings, category, key)
    if not state_path.exists():
        return None
    try:
        with state_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _write_state(
    settings: Settings,
    category: str,
    key: str,
    payload: dict[str, Any],
    *,
    build_manifest: bool = True,
) -> None:
    """Write state with file manifest for relevant categories.

    ``build_manifest=False`` is for a FORCED run of a manifest category
    (pipeline / combined): its check returns before it ever loads the state,
    and hashing every tracked table for it cost ~45 s (pipeline) and ~50 s
    (combined) a night. Without a manifest the state could vouch for nothing,
    so none is kept -- an older one is removed -- and a later non-force run
    finds no state and recomputes. An empty manifest is no safe stand-in: a
    dataset that has since lost every tracked table would compare {} with {}
    and read as unchanged.
    """
    state_path = _state_path(settings, category, key)
    if not build_manifest and category in ("pipeline", "combined"):
        state_path.unlink(missing_ok=True)
        # INFO, not DEBUG: this timestamped line replaces "Built file manifest
        # for <dataset>" as the per-dataset marker in the nightly log.
        LOGGER.info(f"Forced {category} run for {Path(key).name}: no file manifest built, no state kept")
        return
    state_path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data.setdefault("version", STATE_VERSION)

    # Add file manifest for pipeline category
    if category == "pipeline":
        # Key is the dataset root path
        dataset_root = Path(key)
        if dataset_root.exists():
            LOGGER.debug(f"Building file manifest for state persistence: {dataset_root.name}")
            data["file_manifest"] = _build_file_manifest(dataset_root)

    # Add file manifest for combined category
    if category == "combined":
        dataset_roots = payload.get("dataset_roots", [])
        combined_manifest = {}
        for root_str in dataset_roots:
            root_path = Path(root_str)
            if root_path.exists():
                combined_manifest.update(_build_file_manifest(root_path))
        data["file_manifest"] = combined_manifest

    # Write to disk
    with state_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)

    LOGGER.debug(f"Wrote state: {state_path}")


# ============================================================================
# File Manifest Tracking Functions
# ============================================================================


def _should_track_file(file_path: Path, dataset_root: Path) -> bool:
    """
    Determine if a pipeline table file should be tracked in the manifest.

    Tracks per-trial pipeline tables (CSV legacy or Parquet) that live at
    ``fly_dir/trial_dir/file`` depth, excluding metadata/sensor/temp files.

    Args:
        file_path: Path to the file
        dataset_root: Root of the dataset

    Returns:
        True if file should be tracked
    """
    # Must be a pipeline table (CSV legacy or Parquet)
    if file_path.suffix.lower() not in ('.csv', '.parquet'):
        return False

    # Exclude metadata and temp files
    if file_path.name.startswith(('sensors_', '.', '~')):
        return False

    # Two things are tracked:
    #  1. Per-trial pipeline tables at fly_dir/trial_dir/file depth (>= 3 path
    #     parts relative to the dataset root). Shallower files (e.g. a stats
    #     sidecar directly in fly_dir) are not per-trial outputs.
    #  2. Raw recording inputs at batch depth (``<batch>/output_*.csv``, exactly
    #     2 parts). Without these, adding a freshly-recorded batch of raw videos
    #     does NOT change the manifest (raw outputs live shallower than the
    #     processed tables), so the dataset cache stays "valid" and the new batch
    #     is silently skipped. Raw output_* CSVs survive processing, so tracking
    #     them is stable and does not cause spurious post-processing invalidation.
    try:
        relative = file_path.relative_to(dataset_root)
    except ValueError:
        # File not in dataset_root
        return False

    depth = len(relative.parts)
    if depth >= 3:
        return True
    if depth == 2 and file_path.name.startswith("output_"):
        return True
    return False


def _file_content_hash(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Return a fast content hash (blake2b, 16-byte digest) of *path*.

    Read in chunks so large files do not spike memory. Used so a file that is
    merely *touched* (mtime bumped) but whose bytes are unchanged does NOT
    invalidate the cache — only a real content change does.
    """
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_file_manifest(dataset_root: Path) -> Dict[str, Dict[str, Any]]:
    """
    Build manifest of all trackable pipeline tables (CSV legacy + Parquet).

    Recursively scans the dataset for per-trial table files and captures size,
    mtime, and a content hash for change detection. The content hash makes the
    cache robust to mtime-only "touches" (re-sync, chmod) that leave bytes
    unchanged.

    Args:
        dataset_root: Root directory of dataset (e.g., /Data/flys/opto_EB/)

    Returns:
        Dict mapping absolute file path to {mtime: float, size: int, hash: str}
    """
    manifest = {}
    file_count = 0

    start_time = time.time()

    # Recursive walk through dataset (both legacy .csv and migrated .parquet)
    for table_file in (*dataset_root.rglob("*.csv"), *dataset_root.rglob("*.parquet")):
        if not _should_track_file(table_file, dataset_root):
            continue

        try:
            stat = table_file.stat()
            manifest[str(table_file.absolute())] = {
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "hash": _file_content_hash(table_file),
            }
            file_count += 1
        except (OSError, PermissionError) as e:
            LOGGER.warning(f"Failed to read {table_file}: {e}")
            continue

    elapsed = time.time() - start_time
    LOGGER.info(
        f"Built file manifest for {dataset_root.name}: "
        f"{file_count} files in {elapsed:.2f}s"
    )

    if file_count > 5000:
        LOGGER.warning(
            f"Large manifest: {file_count} files in {dataset_root.name}. "
            f"Consider dataset splitting for better performance."
        )

    return manifest


def _compare_manifests(
    current: Dict[str, Dict],
    cached: Dict[str, Dict]
) -> Tuple[bool, List[str]]:
    """
    Compare current manifest with cached manifest.

    Detects new files, modified files (mtime or size changed), and deleted files.

    Args:
        current: Current file manifest
        cached: Cached file manifest from state file

    Returns:
        (is_valid, change_descriptions)
        - is_valid: True if manifests match (cache valid)
        - change_descriptions: List of human-readable changes (empty if valid)
    """
    changes = []

    # Track counts for summary
    new_count = 0
    modified_count = 0
    deleted_count = 0

    # Check for new and modified files
    for file_path, current_info in current.items():
        if file_path not in cached:
            new_count += 1
            if LOGGER.level <= logging.DEBUG:
                changes.append(f"New file: {file_path}")
        else:
            cached_info = cached[file_path]
            # Content-aware change detection:
            #   * size differs                -> modified
            #   * both sides carry a hash     -> modified iff hashes differ
            #                                    (mtime-only "touches" are ignored)
            #   * no hash (legacy manifests)  -> fall back to mtime comparison
            size_changed = current_info["size"] != cached_info["size"]
            cur_hash = current_info.get("hash")
            cached_hash = cached_info.get("hash")
            if cur_hash is not None and cached_hash is not None:
                content_changed = cur_hash != cached_hash
            else:
                content_changed = current_info.get("mtime") != cached_info.get("mtime")
            if size_changed or content_changed:
                modified_count += 1
                if LOGGER.level <= logging.DEBUG:
                    changes.append(
                        f"Modified file: {file_path} "
                        f"(size: {cached_info['size']} → {current_info['size']}, "
                        f"hash: {str(cached_hash)[:8]} → {str(cur_hash)[:8]})"
                    )

    # Check for deleted files
    for file_path in cached:
        if file_path not in current:
            deleted_count += 1
            if LOGGER.level <= logging.DEBUG:
                changes.append(f"Deleted file: {file_path}")

    # Summary message (always at INFO level)
    if new_count + modified_count + deleted_count > 0:
        summary = (
            f"File changes detected: "
            f"{new_count} new, {modified_count} modified, {deleted_count} deleted"
        )
        changes.insert(0, summary)  # Prepend summary
        is_valid = False
    else:
        # Valid cache: report an explicit "unchanged" summary (callers and tests
        # rely on a non-empty changes list describing the no-change outcome).
        changes.append(f"unchanged: {len(current)} files")
        is_valid = True

    return is_valid, changes


def _combined_threshold_key(combined_cfg: Mapping[str, Any] | None) -> list:
    """Every distinct ThresholdRule the combined step will build a table under.

    One config can build several wide tables (``combined.wide``,
    ``combined.combined_base.wide``, each ``pair_groups`` entry) and they may
    carry different threshold blocks, so the cache key is the set of them, not
    one. Sorted so the key is stable across dict ordering.
    """
    cfg = combined_cfg or {}
    # Only blocks that actually exist: an absent block builds no table, and
    # folding in a default rule for it would make the key differ between two
    # configs that produce identical output.
    blocks = [cfg["wide"]] if isinstance(cfg.get("wide"), Mapping) else []
    for name in ("combined_base", "distance_base"):
        sub = cfg.get(name)
        if isinstance(sub, Mapping):
            blocks.append(sub.get("wide"))
    rules = {
        ThresholdRule.from_mapping(b if isinstance(b, Mapping) else {})
        for b in blocks
    }
    return sorted(repr(r) for r in rules)


def _freeze_fingerprint(
    settings,
    dataset: str,
    *,
    measure_cols,
    fps_fallback: float,
    distance_limits,
    non_reactive_threshold,
    low_max_threshold_px: float,
    use_per_trial_baseline: bool,
    trial_type_filter,
    threshold_rule,
) -> dict:
    """Fingerprint for one dataset under one set of build_wide_csv parameters.

    Shared by the load and save paths so they can never disagree -- if they did,
    every run would see spurious drift and re-derive.
    """
    from fbpipe import freeze as _freeze

    return _freeze.build_fingerprint(
        protocol=settings.protocol,
        measure_cols=measure_cols,
        fps_fallback=fps_fallback,
        distance_limits=distance_limits,
        non_reactive_threshold=non_reactive_threshold,
        low_max_threshold_px=low_max_threshold_px,
        use_per_trial_baseline=use_per_trial_baseline,
        # theta sets the AUC-* columns (envelope_combined._compute_trial_metrics).
        # Without it here, changing the threshold rule would leave frozen datasets
        # serving cached rows scored under the OLD rule while live folders got the
        # new one -- a wide table that is half one rule and half the other, with
        # nothing to show it. Required, not defaulted.
        threshold_rule=repr(threshold_rule) if threshold_rule is not None else None,
        # build_wide_csv uses trial_type_filter to gate which trials become
        # rows at all (envelope_combined.py:2661-2673, :2703). Required, not
        # defaulted, so a caller cannot silently omit a value-affecting input.
        trial_type_filter=trial_type_filter,
        override=(settings.dataset_overrides or {}).get(dataset),
        # build_wide_csv reads settings.tracking internally
        # (envelope_combined.py:2622) and derives the tracking_missing_frames /
        # tracking_pct_missing / tracking_flagged columns from it. Required, not
        # defaulted, so a caller cannot silently omit a value-affecting input.
        tracking=settings.tracking,
        # Folder freeze decides which rows are marked frozen, so a slice cached
        # under one policy must not be served under another.
        freeze_folders_before=getattr(settings, "freeze_folders_before", None),
        freeze_folders_born_on_or_after=getattr(
            settings, "freeze_folders_born_on_or_after", None
        ),
        keep_folders_born_on_or_after=getattr(
            settings, "keep_folders_born_on_or_after", None
        ),
    )


def _pipeline_expectation(settings, dataset_root: Path) -> dict[str, Any]:
    """Cache key for one dataset's pipeline (YOLO + per-trial) stage.

    The scalars gate row VALUES; the freeze entries gate which folders are
    walked at all. Both belong here: the manifest half of the cache only sees
    raw table bytes, and a freeze/thaw moves no bytes -- the ``output_*.csv``
    files of a frozen folder sit on disk and land in the manifest exactly as a
    live folder's do. Without this, un-retiring a folder read as "no file
    changes" and the dataset was skipped whole, so YOLO never saw it.

    Mirrors ``freeze.build_fingerprint``'s encoding of the same two inputs
    (which already guards the wide-CSV slice cache), including the sort: list
    ORDER never changes which folders are frozen, so it must not read as drift.
    Resolved through ``folder_freeze_rules`` so the per-run ``--thaw`` /
    ``--thaw-all`` escape hatches invalidate the cache too -- a cache hit would
    otherwise silently neuter them.
    """
    from fbpipe.utils.frozen_folders import (
        born_cutoff,
        born_keep_from,
        folder_freeze_rules,
    )

    cutoff, rules = folder_freeze_rules(
        settings,
        dataset_root.name,
        thawed=getattr(settings, "_thawed", ()),
        thaw_all=getattr(settings, "_thaw_all", False),
    )
    born_cut = born_cutoff(
        settings,
        dataset_root.name,
        thawed=getattr(settings, "_thawed", ()),
        thaw_all=getattr(settings, "_thaw_all", False),
    )
    keep_from = born_keep_from(
        settings,
        dataset_root.name,
        thawed=getattr(settings, "_thawed", ()),
        thaw_all=getattr(settings, "_thaw_all", False),
    )
    return {
        "non_reactive_span_px": settings.non_reactive_span_px,
        "class2_min": settings.class2_min,
        "class2_max": settings.class2_max,
        "freeze_folders": sorted(
            f"{rule.pattern}@{rule.before}" for rule in rules
        ),
        "freeze_folders_before": None if cutoff is None else str(cutoff),
        "freeze_folders_born_on_or_after": (
            None if born_cut is None else str(born_cut)
        ),
        "keep_folders_born_on_or_after": (
            None if keep_from is None else str(keep_from)
        ),
    }


def _resolve_frozen_folders(settings, roots, skip=()) -> dict[str, set[str]]:
    """``{dataset: {frozen experiment folder, ...}}`` across *roots*.

    Datasets with nothing frozen are OMITTED rather than mapped to an empty set,
    so a caller's ``if frozen_folders:`` guard stays honest and the log line
    cannot claim a folder freeze that did not happen. Datasets in *skip* (those
    spliced from the freeze cache, whose roots build_wide_csv never walks) are
    not resolved at all.
    """
    from fbpipe.utils.frozen_folders import frozen_folders_for_root

    thawed = getattr(settings, "_thawed", ())
    thaw_all = getattr(settings, "_thaw_all", False)
    skipped = {str(name) for name in (skip or ())}
    out: dict[str, set[str]] = {}
    for root in roots:
        root_path = Path(root)
        if root_path.name in skipped:
            continue
        names = frozen_folders_for_root(
            settings, root_path, thawed=thawed, thaw_all=thaw_all
        )
        if names:
            out[root_path.name] = names
    return out


def _resolve_frozen_slices(
    settings,
    roots,
    wide_block: str,
    *,
    measure_cols,
    fps_fallback: float,
    distance_limits,
    non_reactive_threshold,
    low_max_threshold_px: float,
    use_per_trial_baseline: bool,
    trial_type_filter,
    threshold_rule,
    thawed=(),
    thaw_all: bool = False,
) -> dict:
    """Return {dataset: FrozenSlice} for every root frozen for DATA with a
    trustworthy cache.

    A dataset is omitted when it is not frozen, is thawed, has no cache, or its
    fingerprint drifted. Omission IS the auto-rebuild: build_wide_csv then walks
    the root normally.
    """
    from fbpipe import freeze as _freeze

    out: dict = {}
    for root in roots:
        dataset = Path(str(root)).name
        if not dataset:
            continue
        freeze_data, _ = _freeze.freeze_flags(
            settings, dataset, thawed=thawed, thaw_all=thaw_all
        )
        if not freeze_data:
            continue
        fingerprint = _freeze_fingerprint(
            settings,
            dataset,
            measure_cols=measure_cols,
            fps_fallback=fps_fallback,
            distance_limits=distance_limits,
            non_reactive_threshold=non_reactive_threshold,
            low_max_threshold_px=low_max_threshold_px,
            use_per_trial_baseline=use_per_trial_baseline,
            trial_type_filter=trial_type_filter,
            threshold_rule=threshold_rule,
        )
        slice_ = _freeze.load_slice(settings.cache_dir, wide_block, dataset, fingerprint)
        if slice_ is None:
            print(
                f"[FREEZE] {dataset} is frozen but its {wide_block} cache is "
                f"missing or stale -- rebuilding it once, then re-caching."
            )
            continue
        out[dataset] = slice_
    return out


def _read_wide_for_cache(csv_path: Path, *, use_sibling: bool = True) -> pd.DataFrame:
    """``pd.read_csv(csv_path)``, served from its Parquet sibling when current.

    build_wide_csv writes ``<stem>.parquet`` as ``pd.read_csv(<stem>.csv)``
    right after the CSV, so the sibling holds the same frame (same dtypes) at a
    fraction of the cost of re-parsing ~2 GB of CSV. That write is best-effort
    (a warning on failure), so a missing, older or unreadable sibling falls
    back to the CSV. Not ``fbpipe.analysis.traces.read_wide_table``: it drops
    folder-frozen rows by default, and the cache must keep them.
    """
    try:
        # Where build_wide_csv writes it: next to the resolved output path.
        sibling = csv_path.expanduser().resolve().with_suffix(".parquet")
        if (
            use_sibling
            and sibling.is_file()
            and sibling.stat().st_mtime_ns >= csv_path.stat().st_mtime_ns
        ):
            return pd.read_parquet(sibling, engine="pyarrow")
    except Exception as exc:
        print(f"[FREEZE] Parquet sibling of {csv_path.name} unreadable ({exc}); reading the CSV.")
    return pd.read_csv(csv_path)


def _spliced_slice_unchanged(cache_dir, wide_block, dataset, group, fingerprint, cached) -> bool:
    """True when saving ``group`` with ``fingerprint`` would write ``cached`` back as is.

    A spliced dataset's rows came from its cache, but ``group`` is cut from the
    whole re-read output: its dtypes are settled over EVERY dataset's rows and
    its dir_val_* width is the run's global max. Both can move while the cached
    slice stands still -- a live NaN turns an int column float64 (the next
    splice then prints ``2.0``, not ``2``), a longer live trace widens every
    row -- and a config number re-spelled ``30`` -> ``30.0`` still matches the
    fingerprint yet rewrites meta.json. A re-save follows all of that, so only
    an identical slice is skipped: same columns, dtypes and values, and a
    meta.json that reads exactly as the re-save would write it -- so one an
    older save_slice wrote under the same fingerprint (another own_max_len,
    another layout) is rewritten, as it always was.
    """
    from fbpipe import freeze as _freeze

    try:
        old = cached[0]  # FrozenSlice.rows, or a (rows, own_max_len) tuple
        if (
            len(group) != len(old)
            or list(group.columns) != list(old.columns)
            or list(group.dtypes) != list(old.dtypes)
        ):
            return False
        # equals() also compares the row labels: the slice was saved after
        # reset_index(drop=True). Relabel a shallow copy -- frozen rows are
        # most of the table, and reset_index would copy them all.
        rows = group.copy(deep=False)
        rows.index = old.index
        if not rows.equals(old):
            return False
        # The meta.json fbpipe.freeze.save_slice writes for these rows: same
        # keys, same own_max_len, same json.dumps call. A save_slice that
        # writes more (a new key, another layout) no longer matches this, and
        # its slices are simply re-saved every run until this follows suit.
        meta = {
            "schema_version": _freeze.FREEZE_SCHEMA_VERSION,
            "dataset": str(dataset),
            "wide_block": str(wide_block),
            "own_max_len": _freeze.own_max_len(rows),
            "row_count": int(len(rows)),
            "columns": [str(c) for c in rows.columns],
            "fingerprint": dict(fingerprint),
        }
        meta_path = _freeze.slice_dir(cache_dir, wide_block, dataset) / "meta.json"
        return meta_path.read_text() == json.dumps(meta, indent=2, sort_keys=True)
    except Exception:  # anything unexpected: re-save, as before
        return False


def _write_freeze_cache(
    settings,
    wide_block: str,
    output_csv: str,
    extra_export_paths,
    *,
    fingerprint_for,
    spliced=None,
    walked=None,
) -> None:
    """Cache each dataset's slice of a freshly written wide CSV.

    Runs after EVERY live derivation, frozen or not, so that freezing a dataset
    later finds a cache already waiting rather than needing a priming run.

    Reads the main output plus every extra trial export, so a slice holds ALL
    trial types -- the splice routes them back the same way live rows are.

    ``spliced`` is the ``{dataset: slice}`` mapping build_wide_csv spliced from
    this block's cache (``_resolve_frozen_slices``: fingerprint already
    matched). Such a slice is not re-saved when the re-save would write it back
    unchanged (``_spliced_slice_unchanged``); otherwise it is, as before.

    Under auto-freeze (an active plan, fbpipe.autofreeze) a spliced slice is
    never re-saved: it IS the frozen dataset's record, and a re-save would
    carry the run's other datasets' dtypes and width into it (and any stale
    duplicate rows forward). Only the datasets walked this run -- *walked*, the
    block's roots minus the spliced ones -- are cached, minus any exact
    duplicate row, and a block that walked none is not even read back.

    "A cache write must never fail the run" covers the whole function, not just
    the per-dataset loop: building ``allrows`` (the concat + groupby below) can
    itself raise -- e.g. mismatched columns across the main output and an extra
    trial export -- so that step is guarded too.
    """
    from fbpipe import autofreeze as _autofreeze
    from fbpipe import freeze as _freeze

    paths = [Path(str(path)) for path in [output_csv, *(extra_export_paths or [])]]
    if _autofreeze.active_plan() is not None:
        names = None if walked is None else set(walked) - set(spliced or {})
        if names is None or names:
            _cache_walked_rows(settings, wide_block, paths, fingerprint_for, spliced or {}, names)
        return
    try:
        # Two outputs with one stem (wide.csv, wide.txt) share one sibling,
        # which holds only the rows of whichever was written last.
        resolved = {p.expanduser().resolve() for p in paths}
        use_siblings = len({r.with_suffix(".parquet") for r in resolved}) == len(resolved)
    except Exception:
        use_siblings = False
    frames = []
    for p in paths:
        if not p.is_file():
            continue
        try:
            frames.append(_read_wide_for_cache(p, use_sibling=use_siblings))
        except Exception as exc:
            print(f"[FREEZE] Skipping cache read of {p}: {exc}")
    if not frames:
        return
    try:
        allrows = pd.concat(frames, ignore_index=True)
        if "dataset" not in allrows.columns:
            return
        groups = list(allrows.groupby(allrows["dataset"].astype(str)))
    except Exception as exc:  # a cache write must never fail the run
        print(f"[FREEZE] Could not prepare rows to cache for {wide_block}: {exc}")
        return
    for dataset, group in groups:
        try:
            fingerprint = fingerprint_for(dataset)
            cached = (spliced or {}).get(dataset)
            if cached is not None and _spliced_slice_unchanged(
                settings.cache_dir, wide_block, dataset, group, fingerprint, cached
            ):
                continue
            _freeze.save_slice(
                settings.cache_dir,
                wide_block,
                dataset,
                group.reset_index(drop=True),
                fingerprint,
            )
        except Exception as exc:  # a cache write must never fail the run
            print(f"[FREEZE] Could not cache {dataset}/{wide_block}: {exc}")


def _csv_field(line: bytes, at: int) -> Optional[str]:
    try:
        return next(csv.reader([line.decode("utf-8")]))[at]
    except (IndexError, StopIteration, UnicodeDecodeError):
        return None


def _read_walked_rows(paths, names) -> list[pd.DataFrame]:
    """The rows of datasets *names* (None: every row) in each wide output.

    Lines are picked by their leading ``dataset`` field before parsing, so a
    night that rebuilt one dataset parses only that dataset's rows. Floats are
    parsed ``round_trip``: pandas' default parser can land one ULP off the
    written value, and a slice holding that value would splice back a
    different last digit than the walk wrote -- the frozen night's table would
    not be byte-identical to the night that built it.

    build_wide_csv writes ``dataset`` first. Were it another column, the
    leading-field pick would match nothing and every walked dataset would be
    cached empty, so each line's field is read with the csv module instead
    (slower, same rows); an output with no ``dataset`` column raises, and the
    caller caches nothing -- no dataset is recorded frozen from rows that
    could not be told apart.
    """
    import io

    keys = None
    if names is not None:
        keys = tuple(
            key for n in sorted(names) for key in (f"{n},".encode(), f'"{n}",'.encode())
        )
    frames = []
    for p in paths:
        if not p.is_file():
            continue
        with p.open("rb") as fh:
            header = fh.readline()
            if keys is None:
                body = fh.read()
            else:
                columns = next(csv.reader([header.decode("utf-8-sig")]), [])
                if "dataset" not in columns:
                    raise ValueError(f"{p.name} has no 'dataset' column")
                at = columns.index("dataset")
                if at == 0:
                    body = b"".join(line for line in fh if line.startswith(keys))
                else:
                    wanted = {str(n) for n in names}
                    body = b"".join(line for line in fh if _csv_field(line, at) in wanted)
        if body:
            frames.append(pd.read_csv(io.BytesIO(header + body), float_precision="round_trip"))
    return frames


def _cache_walked_rows(settings, wide_block, paths, fingerprint_for, spliced, names) -> None:
    """Auto-freeze's cache write: the walked datasets only, never a spliced one.

    See ``_write_freeze_cache``. Exact duplicate rows are not cached (a fresh
    build emits each trial once; a repeat could only come from overlapping
    outputs). Each slice carries this run's id -- the plan records a rebuilt
    dataset frozen only from a slice its own run wrote -- and a walked dataset
    that produced no rows gets an empty slice: no rows is then its truth. A
    cache write must never fail the run.
    """
    from fbpipe import autofreeze as _autofreeze
    from fbpipe import freeze as _freeze

    run_id = (_autofreeze.active_plan() or {}).get("run_id")
    try:
        frames = _read_walked_rows(paths, names)
        allrows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        groups = (
            list(allrows.groupby(allrows["dataset"].astype(str)))
            if "dataset" in allrows.columns else []
        )
    except Exception as exc:  # a cache write must never fail the run
        print(f"[FREEZE] Could not prepare rows to cache for {wide_block}: {exc}")
        return
    cached: set[str] = set()
    for dataset, group in groups:
        if dataset in spliced:
            continue
        try:
            rows = group.reset_index(drop=True)
            dup = _freeze._exact_duplicates(rows).to_numpy()
            if dup.any():
                print(f"[FREEZE] {dataset}/{wide_block}: {int(dup.sum())} exact duplicate row(s) not cached")
                rows = rows.iloc[(~dup).nonzero()[0]].reset_index(drop=True)
            _freeze.save_slice(settings.cache_dir, wide_block, dataset, rows, fingerprint_for(dataset),
                               built_by=run_id)
            cached.add(dataset)
        except Exception as exc:  # a cache write must never fail the run
            print(f"[FREEZE] Could not cache {dataset}/{wide_block}: {exc}")
    for dataset in sorted(set(names or ()) - cached - set(spliced) - {d for d, _ in groups}):
        try:
            _freeze.save_slice(settings.cache_dir, wide_block, dataset, pd.DataFrame(),
                               fingerprint_for(dataset), built_by=run_id)
        except Exception as exc:  # a cache write must never fail the run
            print(f"[FREEZE] Could not cache {dataset}/{wide_block}: {exc}")


def _plan_forces_combined(plan) -> bool:
    """Whether tonight's auto-freeze plan needs the combined step despite its cache.

    A rebuilt dataset's rows must be walked and cached before it can be
    recorded frozen again; an adoption that removed duplicate rows from a
    slice leaves them in the wide tables until those are rebuilt from it.
    """
    return plan is not None and any(d.commit or d.deduped for d in plan.decisions.values())


def _should_skip_with_manifest(
    settings: Settings,
    category: str,
    key: str,
    expected: Dict[str, Any],
    *,
    force_flag: bool,
    dataset_root: Optional[Path] = None
) -> bool:
    """
    Enhanced skip check with file manifest tracking.

    For 'pipeline' category: Checks dataset CSV file manifest
    For 'combined' category: Checks all dataset root manifests
    For other categories: Uses original expectation-only logic

    Args:
        settings: Pipeline settings
        category: State category (pipeline, combined, etc.)
        key: State key (dataset path, etc.)
        expected: Expected state values
        force_flag: If True, always bypass cache
        dataset_root: Dataset root path (for pipeline category)

    Returns:
        True if processing can be skipped (cache valid)
        False if processing required (cache invalid or miss)
    """
    # Force flag always bypasses cache
    if force_flag:
        LOGGER.debug(f"[{category}] force flag set, bypassing cache")
        return False

    # Load cached state
    state = _load_state(settings, category, key)
    if not state:
        LOGGER.debug(f"[{category}] No cached state found")
        return False

    # Check version compatibility
    if state.get("version") != STATE_VERSION:
        LOGGER.info(f"[{category}] State version mismatch, invalidating cache")
        return False

    # Check expectation fields (non_reactive_span_px, etc.)
    for field, value in expected.items():
        if field == "file_manifest":
            continue  # Handle separately below
        if state.get(field) != value:
            LOGGER.info(
                f"[{category}] Config changed: {field} "
                f"({state.get(field)} → {value})"
            )
            return False

    # NEW: File manifest checking for pipeline
    if category == "pipeline" and dataset_root:
        cached_manifest = state.get("file_manifest", {})

        # Build current manifest
        current_manifest = _build_file_manifest(dataset_root)

        # Compare manifests
        is_valid, changes = _compare_manifests(current_manifest, cached_manifest)

        if not is_valid:
            LOGGER.info(f"[CACHE MISS] {changes[0]}")  # Log summary
            for change in changes[1:]:  # Log details at DEBUG
                LOGGER.debug(f"[CACHE MISS] {change}")
            return False

        LOGGER.info(
            f"[CACHE HIT] No file changes in {dataset_root.name} "
            f"({len(current_manifest)} files checked)"
        )
        return True

    # NEW: File manifest checking for combined
    if category == "combined":
        dataset_roots = expected.get("dataset_roots", [])
        cached_manifest = state.get("file_manifest", {})

        # Build combined manifest from all datasets
        current_manifest = {}
        for root_str in dataset_roots:
            root_path = Path(root_str)
            if root_path.exists():
                current_manifest.update(_build_file_manifest(root_path))

        # Compare manifests
        is_valid, changes = _compare_manifests(current_manifest, cached_manifest)

        if not is_valid:
            LOGGER.info(f"[CACHE MISS] {changes[0]}")
            for change in changes[1:]:
                LOGGER.debug(f"[CACHE MISS] {change}")
            return False

        LOGGER.info(
            f"[CACHE HIT] No file changes across {len(dataset_roots)} datasets "
            f"({len(current_manifest)} files checked)"
        )
        return True

    # For other categories (reactions, etc.), use original logic
    LOGGER.info(f"[CACHE HIT] {category} state valid (no file tracking)")
    return True


def _should_skip(settings: Settings, category: str, key: str, expected: dict[str, Any], *, force_flag: bool) -> bool:
    if force_flag:
        return False
    state = _load_state(settings, category, key)
    if not state:
        return False
    if state.get("version") != STATE_VERSION:
        return False
    for field, value in expected.items():
        if state.get(field) != value:
            return False
    return True


def _apply_freeze_settings(config: Any, settings: Settings | None) -> None:
    """Populate a plot config's dataset_overrides/thawed/thaw_all from *settings*.

    Task 6 plumbing: ``should_skip_frozen_figure`` reads these three fields off
    the plot config, but they default to None/()/False (no dataset frozen) so
    that a caller which never wires *settings* through reproduces today's
    figure output byte-for-byte. Only touch the fields when *settings* is
    actually provided -- that is the default-off invariant.
    """
    if settings is None:
        return
    config.dataset_overrides = settings.dataset_overrides
    config.thawed = getattr(settings, "_thawed", ())
    config.thaw_all = getattr(settings, "_thaw_all", False)


def _matrix_plot_config(
    data: Mapping[str, Any], settings: Settings | None = None
) -> tuple[MatrixPlotConfig, str | None]:
    """Parse matrix plot config and return config plus SMB path."""
    opts = dict(data)
    for key in ("matrix_npy", "codes_json", "out_dir"):
        opts[key] = _ensure_path(opts.get(key), key)
    opts["overwrite"] = True
    if "trial_orders" in opts and opts["trial_orders"] is not None:
        opts["trial_orders"] = tuple(opts["trial_orders"])
    # Extract SMB path for later use
    smb_path = opts.pop("out_dir_smb", None)
    config = MatrixPlotConfig(**opts)  # type: ignore[arg-type]
    _apply_freeze_settings(config, settings)
    return config, smb_path


_PIPELINE_ENVELOPE_STYLE_DEFAULTS: dict[str, Any] = {
    "style_scale": 2.0,
    "trace_linewidth_scale": 2.0,
    "panel_title_scale": 0.9,
    "figure_title_scale": 0.72,
    "figure_subtitle_scale": 0.5,
    "legend_scale": 0.675,
    "plot_size_scale": 1.2,
    "single_ylabel_trial_num": 5,
    "fixed_y_max": 105.0,
    "y_label_override": "Max Distance x Angle %",
    "legend_anchor_y": 0.895,
    "figure_title_x": 0.5,
    "figure_title_y": 0.925,
    "figure_title_ha": "center",
    "figure_subtitle_x": 0.5,
    "figure_subtitle_y": 0.905,
    "figure_subtitle_ha": "center",
    "panel_title_x": 0.01,
    "panel_title_y": 105.0,
    "panel_title_va": "top",
    "panel_title_use_data_y": True,
    "tight_h_pad": 0.12,
}


def _envelope_plot_config(
    data: Mapping[str, Any], settings: Settings | None = None
) -> tuple[EnvelopePlotConfig, str | None]:
    """Parse envelope plot config and return config plus SMB path."""
    opts = dict(data)
    for key in ("matrix_npy", "codes_json", "out_dir"):
        opts[key] = _ensure_path(opts.get(key), key)
    # Optional: read the per-fly traces from a wide table (Parquet/CSV) instead
    # of the float16 .npy matrix. ``matrix_npy`` stays required as a label hint.
    if opts.get("wide_input"):
        opts["wide_input"] = _ensure_path(opts.get("wide_input"), "wide_input")
    trial_type = str(opts.get("trial_type", "testing")).strip().lower()
    # Ensure training envelope renders include light-line annotations by default.
    if trial_type == "training" and not str(opts.get("light_annotation_mode", "")).strip():
        opts["light_annotation_mode"] = "line"
    for key, value in _PIPELINE_ENVELOPE_STYLE_DEFAULTS.items():
        opts.setdefault(key, value)
    # Extract SMB path for later use
    smb_path = opts.pop("out_dir_smb", None)
    config = EnvelopePlotConfig(**opts)  # type: ignore[arg-type]
    _apply_freeze_settings(config, settings)
    return config, smb_path


def _copy_output_to_smb(local_path: Path | str, smb_path: str | None) -> None:
    """Copy output directory to SMB if configured."""
    if not smb_path:
        return

    try:
        local_path = Path(local_path)
        if not local_path.exists():
            LOGGER.warning(f"Output directory not found: {local_path}")
            return

        if copy_to_smb(local_path, smb_path):
            LOGGER.info(f"✓ Copied to SMB: {local_path.name} → {smb_path}")
        else:
            LOGGER.warning(f"⚠ Failed to copy to SMB: {smb_path}")
    except Exception as e:
        LOGGER.error(f"Error copying to SMB: {e}")


def _figure_frozen_datasets(settings: Any) -> list[str]:
    """Names of datasets frozen for FIGURES, honoring this run's thaw flags.

    ``freeze.data`` alone is not enough: a data-frozen dataset still redraws
    from its cached rows (that is the point of the two independent flags), so
    only ``freeze.figures`` suppresses a render. Under auto-freeze the plan
    holds a dataset's figures exactly while it is frozen -- a dataset need not
    have an override block for that, so every configured dataset is asked.
    """
    from fbpipe.freeze import freeze_flags

    from fbpipe.odor_constants import canon_dataset

    overrides = getattr(settings, "dataset_overrides", None) or {}
    thawed = getattr(settings, "_thawed", ())
    thaw_all = getattr(settings, "_thaw_all", False)
    candidates = list(overrides)
    candidates += [d for d in (getattr(settings, "datasets", ()) or ()) if d not in overrides]
    names: list[str] = []
    for name in candidates:
        if not freeze_flags(settings, name, thawed=thawed, thaw_all=thaw_all)[1]:
            continue
        names.append(str(name))
        # Also the canonical spelling: config says "3Oct-Training-24-0.1",
        # every figure path says "3OCT-Training-24-0.1".
        canon = canon_dataset(str(name))
        if canon and canon != str(name):
            names.append(canon)
    return names


def _run_light_only_traces(cfg: Mapping[str, Any] | None, settings: Any = None) -> None:
    """Render per-fly light-only PER trace figures, sorted by dataset.

    Driven by the ``analysis.light_only_traces`` config block:
      input_csv:   wide envelope CSV (``all_envelope_rows_wide_combined_base.csv``).
      out_dir:     output root; one subfolder per dataset is created beneath it.
      datasets:    optional allow-list; auto-detected from light trials if omitted.
      out_dir_smb: optional SMB mirror path.
      enabled:     set false to skip the stage.

    These are figures, so the stage always re-renders when configured (it reads
    the existing wide CSV and so runs under ``--figures-only`` too) -- EXCEPT
    for datasets frozen for figures, which are dropped before rendering. The
    drop is applied after the config ``datasets:`` allow-list, so listing a
    frozen dataset there cannot resurrect it; ``--thaw``/``--thaw-all`` still
    lift it for one run.
    """
    if not cfg or not cfg.get("enabled", True):
        return
    input_csv = cfg.get("input_csv")
    out_dir = cfg.get("out_dir")
    if not input_csv or not out_dir:
        LOGGER.warning(
            "[analysis] light_only_traces: input_csv and out_dir are required; skipping."
        )
        return
    csv_path = Path(input_csv)
    if not csv_path.exists():
        LOGGER.warning(
            "[analysis] light_only_traces: CSV not found (%s); skipping.", csv_path
        )
        return

    from scripts.analysis.light_trial_traces import generate as _generate_light_only

    datasets = cfg.get("datasets") or None
    skip = _figure_frozen_datasets(settings)
    print(f"[analysis] light_only_traces → {out_dir}")
    result = _generate_light_only(
        csv_path, Path(out_dir), datasets=datasets, skip_datasets=skip
    )
    if result.get("skipped_datasets"):
        LOGGER.info(
            "[analysis] light_only_traces: skipped %d dataset(s) frozen for "
            "figures: %s",
            len(result["skipped_datasets"]), ", ".join(result["skipped_datasets"]),
        )
    LOGGER.info(
        "[analysis] light_only_traces: %d figures across %d dataset(s) → %s",
        result["n_figures"], len(result["per_dataset"]), out_dir,
    )
    _copy_output_to_smb(out_dir, cfg.get("out_dir_smb"))


def _load_model_scores_for_envelopes(settings: Settings) -> None:
    """Populate envelope_visuals._MODEL_SCORES from reaction_prediction.output_csv.

    Called right before envelope plots are generated so each trial subplot can
    show the model's ordinal score (e.g. ``Score: 3``) in the upper-right
    corner. If the predictions CSV doesn't exist yet (e.g. fresh pipeline run
    before reactions have produced it), the registry stays empty and no score
    is drawn.
    """
    output_csv = getattr(settings.reaction_prediction, "output_csv", None)
    if not output_csv:
        set_model_scores({})
        return
    csv_path = Path(output_csv).expanduser()
    if not csv_path.exists():
        LOGGER.info(
            "[analysis] envelope_visuals: model predictions CSV not found (%s); "
            "score annotations will be skipped.",
            csv_path,
        )
        set_model_scores({})
        return
    try:
        df = pd.read_csv(
            csv_path,
            usecols=["dataset", "fly", "fly_number", "trial_label", "score"],
        )
    except Exception as exc:  # noqa: BLE001 — defensive load
        LOGGER.warning(
            "[analysis] envelope_visuals: failed to read %s for score annotations: %s",
            csv_path,
            exc,
        )
        set_model_scores({})
        return
    scores: dict[tuple[str, str, str, str], int] = {}
    for row in df.itertuples(index=False):
        try:
            scores[(str(row.dataset), str(row.fly), str(row.fly_number), str(row.trial_label))] = int(row.score)
        except (TypeError, ValueError):
            continue
    set_model_scores(scores)
    LOGGER.info(
        "[analysis] envelope_visuals: loaded %d per-trial model scores for annotation.",
        len(scores),
    )


def _load_light_checks_for_envelopes() -> None:
    """Populate envelope_visuals._LIGHT_CHECKS from the light-stimulus QC CSV.

    Called once at the start of the analysis phase — after the
    ``check_light_stimulus`` pipeline step has upserted this run's trials —
    so every trial subplot on the Raw-Testing/Raw-Training trace figures can
    confirm the LED physically fired for its commanded window (">= 95% ==
    full" per the display rule). The registry is module-global, so the
    deferred post-reactions re-render sees the same data. Missing CSV means
    the step has never run; annotations are simply skipped.
    """
    from fbpipe.steps.check_light_stimulus import DEFAULT_CSV_PATH

    csv_path = Path(os.getenv("LIGHT_CHECK_CSV", str(DEFAULT_CSV_PATH)))
    fractions = load_light_check_fractions(csv_path)
    set_light_check_fractions(fractions)
    if fractions:
        LOGGER.info(
            "[analysis] envelope_visuals: loaded %d per-trial light-check results "
            "for annotation (%s).",
            len(fractions),
            csv_path,
        )
    else:
        LOGGER.info(
            "[analysis] envelope_visuals: no light-check results found (%s); "
            "light-check annotations will be skipped.",
            csv_path,
        )


def _rerender_envelope_block_with_scores(
    envelopes_cfg: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    label: str,
    settings: Settings | None = None,
) -> None:
    """Re-render an envelope-config block with overwrite forced on.

    Accepts either a single envelope-config mapping (e.g.
    ``analysis.envelope_visuals.envelopes``) or a list of them (e.g.
    ``analysis.combined.combined_base.envelopes``, ``analysis.combined.envelopes``).
    Each entry is shallow-copied with ``overwrite=True`` injected so that
    ``generate_envelope_plots`` rewrites every PNG in its target directory —
    this is required because every envelopes block in ``config_new.yaml``
    sets ``overwrite: False`` and the first render of these PNGs happens
    before reactions writes ``model_predictions.csv``.

    *settings* is optional (defaults to None, preserving today's behaviour) so
    that ``dataset_overrides``/``thawed``/``thaw_all`` reach the re-rendered
    config only when a caller actually supplies it — see
    ``_apply_freeze_settings``.
    """
    if isinstance(envelopes_cfg, Sequence) and not isinstance(envelopes_cfg, (str, bytes)):
        entries: list[Mapping[str, Any]] = list(envelopes_cfg)
    else:
        entries = [envelopes_cfg]

    for entry in entries:
        if not isinstance(entry, Mapping):
            LOGGER.warning(
                "[analysis] %s: skipping non-mapping envelope entry: %r",
                label,
                entry,
            )
            continue
        forced = {**dict(entry), "overwrite": True}
        if settings is not None:
            config, _smb_path = _envelope_plot_config(forced, settings)
        else:
            config, _smb_path = _envelope_plot_config(forced)
        print(
            f"[analysis] envelope_visuals: re-rendering {label} → {config.out_dir} "
            "(scores annotated)."
        )
        try:
            generate_envelope_plots(config)
        except NoTargetTrialsError:
            # Testing-only dataset (e.g. RandomPanel): this block targets a
            # trial_type with no trials. Skip it rather than abort the run;
            # other entries (e.g. the testing blocks) still render.
            print(
                f"[analysis] envelope_visuals: re-rendering {label} → SKIPPED "
                f"(no '{config.trial_type}' trials in matrix; dataset is "
                f"{config.trial_type}-free)."
            )


def _run_envelope_visuals(
    cfg: Mapping[str, Any] | None,
    *,
    defer_envelopes: bool = False,
    settings: Settings | None = None,
) -> None:
    if not cfg:
        return

    matrices_cfg = cfg.get("matrices")
    if matrices_cfg:
        config, smb_path = _matrix_plot_config(matrices_cfg, settings)
        print(f"[analysis] envelope_visuals.matrices → {config.out_dir}")
        generate_reaction_matrices(config)

    envelopes_cfg = cfg.get("envelopes")
    if envelopes_cfg and defer_envelopes:
        print(
            "[analysis] envelope_visuals.envelopes → DEFERRED until after reactions "
            "(model scores will be annotated)."
        )
    elif envelopes_cfg:
        config, smb_path = _envelope_plot_config(envelopes_cfg, settings)
        print(f"[analysis] envelope_visuals.envelopes → {config.out_dir}")
        generate_envelope_plots(config)


def _csv_has_training_rows(csv_path: Path) -> bool:
    """Return True if ``csv_path`` contains at least one training trial.

    Used to skip training-only analyses (e.g. latency) for datasets that are
    testing-only by design — e.g. RandomPanel, whose trials carry a
    ``trial_type_override: testing`` so every row is typed ``testing`` even
    though the trial labels read ``training_N``. Returns True when the file has
    no ``trial_type`` column, since the analyses then treat every row as a
    candidate and we should not skip.
    """
    try:
        col = pd.read_csv(csv_path, usecols=["trial_type"])
    except ValueError:
        # No ``trial_type`` column — let the downstream analysis decide.
        return True
    if col.empty:
        return False
    return (
        col["trial_type"].astype(str).str.strip().str.lower().eq("training").any()
    )


def _run_training(cfg: Mapping[str, Any] | None, settings: Settings | None = None) -> None:
    if not cfg:
        return

    envelopes_cfg = cfg.get("envelopes")
    if envelopes_cfg:
        entries = (
            envelopes_cfg
            if isinstance(envelopes_cfg, Sequence)
            and not isinstance(envelopes_cfg, (str, bytes))
            else [envelopes_cfg]
        )
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("training.envelopes entries must be mappings.")
            opts = dict(entry)
            opts.setdefault("trial_type", "training")
            config, smb_path = _envelope_plot_config(opts, settings)
            print(f"[analysis] training.envelopes → {config.out_dir}")
            generate_envelope_plots(config)

    latency_cfg = cfg.get("latency")
    if latency_cfg:
        opts = dict(latency_cfg)
        matrix_raw = opts.pop("matrix_npy", None)
        codes_raw = opts.pop("codes_json", None)
        matrix = _resolve_path(matrix_raw)
        codes = _resolve_path(codes_raw)
        out_dir = _ensure_path(opts.pop("out_dir"), "out_dir")
        csv_path = _resolve_path(opts.pop("csv_path", None))
        fly_state_csv = _resolve_path(opts.pop("fly_state_csv", None))
        fly_state_column = str(opts.pop("fly_state_column", "FLY-State(1, 0, -1)"))
        trials = tuple(int(t) for t in opts.pop("trials", [4, 6, 8]))
        if csv_path is not None and not csv_path.exists():
            print(f"[WARN] training.latency csv_path missing: {csv_path}")
            csv_path = None
        if fly_state_csv is not None and not fly_state_csv.exists():
            print(f"[WARN] training.latency fly_state_csv missing: {fly_state_csv}")
            fly_state_csv = None

        if csv_path is not None and not _csv_has_training_rows(csv_path):
            print(
                f"[analysis] training.latency → SKIPPED: {csv_path.name} has no "
                "training trials (dataset is testing-only)."
            )
            return

        print(f"[analysis] training.latency → {out_dir}")
        latency_reports(
            matrix,
            codes,
            out_dir,
            csv_path=csv_path,
            fly_state_csv=fly_state_csv,
            fly_state_column=fly_state_column,
            trials_of_interest=trials,
            # Per-cohort latency figures are that cohort's figures: a dataset
            # frozen for figures must not redraw them.
            frozen_datasets=_figure_frozen_datasets(settings),
            **opts,
        )


def _auto_sync_wide_roots(
    combine_roots: Mapping[str, Path],
    wide_roots: Sequence[Path],
    handled_datasets: Sequence[str] | None = None,
) -> None:
    """Mirror freshly generated combine outputs into their wide-root peers."""

    if not combine_roots or not wide_roots:
        return

    handled = {entry.lower() for entry in (handled_datasets or ())}
    dest_map = {path.name.lower(): path for path in wide_roots}

    for dataset, source in combine_roots.items():
        key = dataset.lower()
        if key in handled:
            continue
        destination = dest_map.get(key)
        if destination is None:
            continue
        if destination == source:
            continue

        print(f"[analysis] combined.wide.auto_mirror[{key}] → {destination}")
        copied, bytes_copied = mirror_directory(str(source), str(destination))
        size_mb = bytes_copied / (1024 * 1024) if bytes_copied else 0.0
        print(
            "[analysis] combined.wide.auto_mirror[{}] copied {} file(s) ({:.1f} MiB).".format(
                key, copied, size_mb
            )
        )


def _render_pair_visuals(
    pair_name: str,
    pair_matrix_dir: Path,
    matrices_template: MatrixPlotConfig | None,
    envelope_templates: Sequence[EnvelopePlotConfig],
) -> None:
    """Generate reaction matrices + envelopes for a pair using template configs."""

    matrix_path = pair_matrix_dir / "envelope_matrix_float16.npy"
    codes_path = pair_matrix_dir / "code_maps.json"

    if matrices_template:
        if matrix_path.exists() and codes_path.exists():
            out_dir = pair_matrix_dir.parent / "results" / matrices_template.out_dir.name
            cfg = dc_replace(
                matrices_template,
                matrix_npy=matrix_path,
                codes_json=codes_path,
                out_dir=out_dir,
            )
            print(f"[analysis] combined.pair_groups[{pair_name}].matrices → {cfg.out_dir}")
            generate_reaction_matrices(cfg)
        else:
            print(
                f"[WARN] Pair '{pair_name}' missing matrix artifacts for reaction matrices: {pair_matrix_dir}"
            )

    for env_template in envelope_templates:
        trial_type = env_template.trial_type.strip().lower()
        target_dir = pair_matrix_dir if trial_type != "training" else pair_matrix_dir / "training"
        env_matrix = target_dir / "envelope_matrix_float16.npy"
        env_codes = target_dir / "code_maps.json"
        if not env_matrix.exists() or not env_codes.exists():
            print(
                f"[WARN] Pair '{pair_name}' missing {trial_type} matrix; skipping envelopes."
            )
            continue

        out_dir = pair_matrix_dir.parent / "results" / env_template.out_dir.name
        cfg = dc_replace(
            env_template,
            matrix_npy=env_matrix,
            codes_json=env_codes,
            out_dir=out_dir,
        )
        print(
            f"[analysis] combined.pair_groups[{pair_name}].envelopes[{trial_type}] → {cfg.out_dir}"
        )
        generate_envelope_plots(cfg)


def _run_combined(
    cfg: Mapping[str, Any] | None,
    settings: Settings | None,
    *,
    folder_filter: str | None = None,
    defer_envelopes: bool = False,
    defer_secure_cleanup: bool = False,
    config_path: str | Path | None = None,
) -> Callable[[], None] | None:
    if not cfg:
        return None

    combine_roots: dict[str, Path] = {}
    wide_root_paths: list[Path] = []
    wide_measure_cols: Sequence[str] = ["envelope_of_rms"]
    wide_fps_fallback = 40.0
    wide_exclude_cfg: list[str] = []
    # Default so the combined_base / distance_base helper below is safe even
    # when no top-level `wide` block is configured (the `wide` block overrides
    # this at load time when present).
    use_per_trial_baseline = False
    non_reactive_threshold = settings.non_reactive_span_px if settings is not None else None
    limits = (settings.class2_min, settings.class2_max) if settings is not None else None

    # Auto-discover flagged experiment directories.
    flagged_dirs: list[Path] = []
    if settings is not None and settings.flagged_root:
        flagged_dirs = discover_flagged_directories(settings.flagged_root)
        if flagged_dirs:
            LOGGER.info(
                "Auto-discovered %d flagged experiment(s) under %s: %s",
                len(flagged_dirs),
                settings.flagged_root,
                [p.name for p in flagged_dirs],
            )

    combine_cfg = cfg.get("combine")
    if combine_cfg:
        opts = dict(combine_cfg)
        roots_cfg = opts.pop("roots", None)
        root_value = opts.pop("root", None)
        if roots_cfg:
            if not isinstance(roots_cfg, Sequence) or isinstance(roots_cfg, (str, bytes, Path)):
                roots_iter = [roots_cfg]
            else:
                roots_iter = list(roots_cfg)
        elif root_value is not None:
            roots_iter = [root_value]
        else:
            raise ValueError("combined.combine requires 'root' or 'roots'.")

        # Append flagged directories so they are processed identically.
        roots_iter.extend(str(d) for d in flagged_dirs)

        for entry in roots_iter:
            entry_path = _ensure_path(entry, "combine.root")
            # In single-folder mode, skip combine for non-target folders.
            if folder_filter and entry_path.name.lower() != folder_filter.strip().lower():
                LOGGER.info(
                    "[analysis] combined.combine → SKIP %s (single-folder mode)",
                    entry_path.name,
                )
                continue
            run_opts = dict(opts)
            run_opts["root"] = entry_path
            if "odor_on" in run_opts and "odor_on_s" not in run_opts:
                run_opts["odor_on_s"] = float(run_opts.pop("odor_on"))
            if "odor_off" in run_opts and "odor_off_s" not in run_opts:
                run_opts["odor_off_s"] = float(run_opts.pop("odor_off"))
            if "odor_latency" in run_opts and "odor_latency_s" not in run_opts:
                run_opts["odor_latency_s"] = float(run_opts.pop("odor_latency"))
            config = CombineConfig(**run_opts)  # type: ignore[arg-type]
            dataset_name = config.root.name
            from fbpipe import freeze as _freeze

            freeze_data, _ = _freeze.freeze_flags(
                settings,
                dataset_name,
                thawed=getattr(settings, "_thawed", ()),
                thaw_all=getattr(settings, "_thaw_all", False),
            )
            if freeze_data:
                # freeze.data is a promise the raw data won't change, and a
                # frozen dataset's wide rows are spliced from the freeze cache
                # (build_wide_csv never reads angle_distance_rms_envelope/ for
                # it) -- so recomputing those per-trial files here is pure
                # waste. This mirrors build_wide_csv's own "never walk a
                # frozen root" skip (envelope_combined.py:2676).
                print(f"[FROZEN] combined.combine → skipping recompute for {config.root}")
            else:
                # Per-FOLDER freeze inside a still-live dataset: leave each
                # retired batch's derived CSVs as they are (the dataset-level
                # skip above is the all-or-nothing version of the same idea).
                from fbpipe.utils.frozen_folders import frozen_folders_for_root

                _skip = frozen_folders_for_root(
                    settings,
                    config.root,
                    thawed=getattr(settings, "_thawed", ()),
                    thaw_all=getattr(settings, "_thaw_all", False),
                )
                if _skip:
                    config = dc_replace(config, skip_folders=frozenset(_skip))
                print(f"[analysis] combined.combine → {config.root}")
                combine_distance_angle(config)
            combine_roots[config.root.name.lower()] = config.root

    wide_cfg = cfg.get("wide")
    #: theta for the AUC-* columns of every wide table built below. Read once so
    #: the build and the freeze fingerprint can never disagree about it.
    wide_threshold_rule = ThresholdRule.from_mapping(wide_cfg or {})
    if wide_cfg:
        roots_cfg = wide_cfg.get("roots", [])
        if not roots_cfg:
            raise ValueError("combined.wide.roots must list at least one directory.")
        roots_iter = (
            roots_cfg
            if isinstance(roots_cfg, Sequence) and not isinstance(roots_cfg, (str, bytes, Path))
            else [roots_cfg]
        )
        wide_root_paths = [_ensure_path(root, "roots") for root in roots_iter]

        # Mirror flagged directories into secured storage, then add the
        # secured paths to the wide build (matching the normal experiment flow).
        if flagged_dirs:
            flagged_secured = (
                Path(settings.flagged_secured_root).expanduser().resolve()
                if settings is not None and settings.flagged_secured_root
                else None
            )
            existing_names = {p.name.lower() for p in wide_root_paths}
            for fd in flagged_dirs:
                if fd.name.lower() in existing_names:
                    continue
                # In single-folder mode, skip flagged dirs that aren't the target.
                if folder_filter and fd.name.lower() != folder_filter.strip().lower():
                    # Still add to wide_root_paths so build_wide_csv reads existing data.
                    if flagged_secured is not None:
                        wide_root_paths.append(flagged_secured / fd.name)
                    else:
                        wide_root_paths.append(fd)
                    continue
                if flagged_secured is not None:
                    secured_dest = flagged_secured / fd.name
                    secured_dest.mkdir(parents=True, exist_ok=True)
                    print(f"[analysis] combined.mirror (flagged) → {secured_dest}")
                    copied, bytes_copied = mirror_directory(str(fd), str(secured_dest))
                    size_mb = bytes_copied / (1024 * 1024) if bytes_copied else 0.0
                    print(
                        f"[analysis] combined.mirror (flagged) copied {copied} file(s) "
                        f"({size_mb:.1f} MiB)."
                    )
                    wide_root_paths.append(secured_dest)
                    LOGGER.info("Added flagged secured root to wide build: %s", secured_dest)
                else:
                    # No secured storage configured; use source path directly.
                    wide_root_paths.append(fd)
                    LOGGER.info("Added flagged root to wide build: %s", fd)

        mirror_cfg = wide_cfg.get("mirror")
        handled_datasets: list[str] = []
        if mirror_cfg:
            entries = mirror_cfg if isinstance(mirror_cfg, Sequence) else [mirror_cfg]
            for entry in entries:
                src = _ensure_path(entry.get("source"), "mirror.source")
                # In single-folder mode, skip mirrors for non-target folders.
                if folder_filter and src.name.lower() != folder_filter.strip().lower():
                    LOGGER.info(
                        "[analysis] combined.mirror → SKIP %s (single-folder mode)", src.name
                    )
                    handled_datasets.append(src.name.lower())
                    continue
                dest = _ensure_path(entry.get("destination"), "mirror.destination")
                print(f"[analysis] combined.mirror → {dest}")
                copied, bytes_copied = mirror_directory(str(src), str(dest))
                size_mb = bytes_copied / (1024 * 1024) if bytes_copied else 0.0
                print(
                    f"[analysis] combined.mirror copied {copied} file(s) "
                    f"({size_mb:.1f} MiB)."
                )
                handled_datasets.append(src.name.lower())

        auto_sync = wide_cfg.get("auto_sync_roots", True)
        if auto_sync:
            _auto_sync_wide_roots(combine_roots, wide_root_paths, handled_datasets)

        roots = [str(path) for path in wide_root_paths]
        output_csv = _ensure_path(wide_cfg.get("output_csv"), "output_csv")
        wide_measure_cols = wide_cfg.get("measure_cols") or ["envelope_of_rms"]
        wide_fps_fallback = float(wide_cfg.get("fps_fallback", 40.0))
        wide_exclude_cfg = [
            str(_ensure_path(path, "exclude_roots"))
            for path in wide_cfg.get("exclude_roots", [])
        ]
        trial_type_filter = wide_cfg.get("trial_type_filter")
        use_per_trial_baseline = wide_cfg.get("use_per_trial_baseline", False)
        extra_exports_cfg = wide_cfg.get("trial_type_exports", [])
        extra_exports: dict[str, str] = {}
        extra_matrix_dirs: dict[str, str] = {}
        if extra_exports_cfg:
            entries = (
                extra_exports_cfg
                if isinstance(extra_exports_cfg, Sequence)
                and not isinstance(extra_exports_cfg, (str, bytes))
                else [extra_exports_cfg]
            )
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise ValueError("trial_type_exports entries must be mappings.")
                trial_type = entry.get("trial_type")
                export_csv = entry.get("output_csv")
                if not trial_type or not export_csv:
                    raise ValueError(
                        "trial_type_exports entries require 'trial_type' and 'output_csv'."
                    )
                export_path = _ensure_path(export_csv, "trial_type_exports.output_csv")
                trial_key = str(trial_type).strip().lower()
                extra_exports[trial_key] = str(export_path)
                matrix_dir = entry.get("matrix_out_dir")
                if matrix_dir:
                    matrix_path = _ensure_path(
                        matrix_dir, "trial_type_exports.matrix_out_dir"
                    )
                    extra_matrix_dirs[trial_key] = str(matrix_path)
        _block = "wide"
        _fp_kw = dict(
            measure_cols=wide_measure_cols,
            fps_fallback=wide_fps_fallback,
            distance_limits=limits,
            non_reactive_threshold=non_reactive_threshold,
            # build_wide_csv is not passed low_max_threshold_px at this call
            # site, so it uses the default. The fingerprint must record the
            # same value or every run reports drift.
            low_max_threshold_px=LOW_MAX_FLAG_THRESHOLD_PX,
            use_per_trial_baseline=use_per_trial_baseline,
            threshold_rule=wide_threshold_rule,
            # Same variable passed to the build_wide_csv(...) call below -- the
            # fingerprint must record exactly what the build used.
            trial_type_filter=trial_type_filter,
        )
        _frozen = _resolve_frozen_slices(
            settings,
            roots,
            _block,
            thawed=getattr(settings, "_thawed", ()),
            thaw_all=getattr(settings, "_thaw_all", False),
            **_fp_kw,
        )
        print(f"[analysis] combined.wide → {output_csv}")
        build_wide_csv(
            roots,
            str(output_csv),
            measure_cols=wide_measure_cols,
            fps_fallback=wide_fps_fallback,
            threshold_rule=wide_threshold_rule,
            exclude_roots=wide_exclude_cfg,
            distance_limits=limits,
            config_path=config_path,
            trial_type_filter=trial_type_filter,
            extra_trial_exports=extra_exports or None,
            non_reactive_threshold=non_reactive_threshold,
            use_per_trial_baseline=use_per_trial_baseline,
            frozen_slices=_frozen or None,
            frozen_folders=_resolve_frozen_folders(settings, roots, skip=_frozen) or None,
        )
        _write_freeze_cache(
            settings,
            _block,
            str(output_csv),
            list(extra_exports.values()),
            fingerprint_for=lambda ds: _freeze_fingerprint(settings, ds, **_fp_kw),
            spliced=_frozen,
            walked={Path(str(r)).name for r in roots},
        )

        for trial_key, matrix_dir in extra_matrix_dirs.items():
            export_csv = extra_exports.get(trial_key)
            if not export_csv:
                continue
            print(
                "[analysis] combined.wide.trial_type_matrix[{}] → {}".format(
                    trial_key, matrix_dir
                )
            )
            wide_to_matrix(export_csv, matrix_dir)

    def _process_base_block(combined_base_cfg, *, label, default_measure_cols):
        base_wide_cfg = combined_base_cfg.get("wide")
        if base_wide_cfg:
            roots_cfg = base_wide_cfg.get("roots")
            if roots_cfg:
                roots_iter = (
                    roots_cfg
                    if isinstance(roots_cfg, Sequence)
                    and not isinstance(roots_cfg, (str, bytes, Path))
                    else [roots_cfg]
                )
                base_root_paths = [
                    _ensure_path(root, "combined_base.wide.roots")
                    for root in roots_iter
                ]
            elif wide_root_paths:
                base_root_paths = list(wide_root_paths)
            else:
                raise ValueError(
                    "combined.combined_base.wide.roots must list at least one directory."
                )

            # Include flagged directories in combined_base wide build
            # (use secured storage paths when available, mirrored earlier).
            if flagged_dirs:
                flagged_secured = (
                    Path(settings.flagged_secured_root).expanduser().resolve()
                    if settings is not None and settings.flagged_secured_root
                    else None
                )
                existing_names = {p.name.lower() for p in base_root_paths}
                for fd in flagged_dirs:
                    if fd.name.lower() not in existing_names:
                        if flagged_secured is not None:
                            base_root_paths.append(flagged_secured / fd.name)
                        else:
                            base_root_paths.append(fd)

            roots = [str(path) for path in base_root_paths]
            output_csv = _ensure_path(
                base_wide_cfg.get("output_csv"),
                "combined_base.wide.output_csv",
            )
            base_measure_cols = base_wide_cfg.get("measure_cols") or list(default_measure_cols)
            base_fps_fallback = float(
                base_wide_cfg.get("fps_fallback", wide_fps_fallback)
            )
            # Read the base block's own baseline flag, falling back to the outer
            # ``combined.wide`` value. Without this, deleting ``combined.wide``
            # would silently flip combined_base to ``use_per_trial_baseline=False``
            # and change the numbers in all_envelope_rows_wide_combined_base.csv.
            base_use_per_trial_baseline = bool(
                base_wide_cfg.get("use_per_trial_baseline", use_per_trial_baseline)
            )
            # theta for the AUC-* columns. Read from this block, falling back to
            # the outer combined.wide block, so a cohort can carry its own floor
            # (the low-amplitude cohorts want a smaller one -- see
            # fbpipe.analysis.threshold).
            base_threshold_rule = ThresholdRule.from_mapping(
                {**(wide_cfg or {}), **base_wide_cfg}
            )
            if "exclude_roots" in base_wide_cfg:
                base_exclude_cfg = [
                    str(_ensure_path(path, "combined_base.wide.exclude_roots"))
                    for path in base_wide_cfg.get("exclude_roots", [])
                ]
            else:
                base_exclude_cfg = list(wide_exclude_cfg)

            trial_type_filter = base_wide_cfg.get("trial_type_filter")
            if trial_type_filter is None and wide_cfg:
                trial_type_filter = wide_cfg.get("trial_type_filter")

            extra_exports_cfg = base_wide_cfg.get("trial_type_exports", [])
            extra_exports: dict[str, str] = {}
            extra_matrix_dirs: dict[str, str] = {}
            if extra_exports_cfg:
                entries = (
                    extra_exports_cfg
                    if isinstance(extra_exports_cfg, Sequence)
                    and not isinstance(extra_exports_cfg, (str, bytes))
                    else [extra_exports_cfg]
                )
                for entry in entries:
                    if not isinstance(entry, Mapping):
                        raise ValueError("trial_type_exports entries must be mappings.")
                    trial_type = entry.get("trial_type")
                    export_csv = entry.get("output_csv")
                    if not trial_type or not export_csv:
                        raise ValueError(
                            "trial_type_exports entries require 'trial_type' and 'output_csv'."
                        )
                    export_path = _ensure_path(
                        export_csv, "trial_type_exports.output_csv"
                    )
                    trial_key = str(trial_type).strip().lower()
                    extra_exports[trial_key] = str(export_path)
                    matrix_dir = entry.get("matrix_out_dir")
                    if matrix_dir:
                        matrix_path = _ensure_path(
                            matrix_dir, "trial_type_exports.matrix_out_dir"
                        )
                        extra_matrix_dirs[trial_key] = str(matrix_path)

            _block = str(label)
            _fp_kw = dict(
                measure_cols=base_measure_cols,
                fps_fallback=base_fps_fallback,
                distance_limits=limits,
                non_reactive_threshold=non_reactive_threshold,
                # build_wide_csv is not passed low_max_threshold_px at this call
                # site, so it uses the default. The fingerprint must record the
                # same value or every run reports drift.
                low_max_threshold_px=LOW_MAX_FLAG_THRESHOLD_PX,
                use_per_trial_baseline=base_use_per_trial_baseline,
                threshold_rule=base_threshold_rule,
                # Same variable passed to the build_wide_csv(...) call below --
                # the fingerprint must record exactly what the build used.
                trial_type_filter=trial_type_filter,
            )
            _frozen = _resolve_frozen_slices(
                settings,
                roots,
                _block,
                thawed=getattr(settings, "_thawed", ()),
                thaw_all=getattr(settings, "_thaw_all", False),
                **_fp_kw,
            )
            print(f"[analysis] {label}.wide → {output_csv}")
            build_wide_csv(
                roots,
                str(output_csv),
                measure_cols=base_measure_cols,
                fps_fallback=base_fps_fallback,
                exclude_roots=base_exclude_cfg,
                distance_limits=limits,
                config_path=config_path,
                trial_type_filter=trial_type_filter,
                extra_trial_exports=extra_exports or None,
                non_reactive_threshold=non_reactive_threshold,
                use_per_trial_baseline=base_use_per_trial_baseline,
                threshold_rule=base_threshold_rule,
                frozen_slices=_frozen or None,
                frozen_folders=_resolve_frozen_folders(settings, roots, skip=_frozen) or None,
            )
            _write_freeze_cache(
                settings,
                _block,
                str(output_csv),
                list(extra_exports.values()),
                fingerprint_for=lambda ds: _freeze_fingerprint(settings, ds, **_fp_kw),
                spliced=_frozen,
                walked={Path(str(r)).name for r in roots},
            )

            for trial_key, matrix_dir in extra_matrix_dirs.items():
                export_csv = extra_exports.get(trial_key)
                if not export_csv:
                    continue
                print(
                    "[analysis] {}.wide.trial_type_matrix[{}] → {}".format(
                        label, trial_key, matrix_dir
                    )
                )
                wide_to_matrix(export_csv, matrix_dir)

        base_matrix_cfg = combined_base_cfg.get("matrix")
        if base_matrix_cfg:
            input_csv = _ensure_path(
                base_matrix_cfg.get("input_csv"),
                "combined_base.matrix.input_csv",
            )
            out_dir = _ensure_path(
                base_matrix_cfg.get("out_dir"),
                "combined_base.matrix.out_dir",
            )
            print(f"[analysis] {label}.matrix → {out_dir}")
            wide_to_matrix(str(input_csv), str(out_dir))

        base_envelopes_cfg = combined_base_cfg.get("envelopes")
        if base_envelopes_cfg and defer_envelopes:
            print(
                f"[analysis] {label}.envelopes → DEFERRED until after reactions "
                "(model scores will be annotated)."
            )
        elif base_envelopes_cfg:
            entries = (
                base_envelopes_cfg
                if isinstance(base_envelopes_cfg, Sequence)
                and not isinstance(base_envelopes_cfg, (str, bytes))
                else [base_envelopes_cfg]
            )
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise ValueError(f"{label}.envelopes entries must be mappings.")
                config, smb_path = _envelope_plot_config(entry, settings)
                print(f"[analysis] {label}.envelopes → {config.out_dir}")
                generate_envelope_plots(config)

    # ``combined_base`` plots the angle-weighted combined_pct signal; ``distance_base``
    # reuses the exact same wide -> matrix -> envelopes machinery to plot the raw
    # distance_percentage column (no angle multiplier). Both are optional.
    for _base_key, _default_cols in (
        ("combined_base", ["combined_pct", "combined_base"]),
        ("distance_base", ["distance_percentage"]),
    ):
        _base_cfg = cfg.get(_base_key)
        if _base_cfg:
            _process_base_block(
                _base_cfg, label=_base_key, default_measure_cols=_default_cols
            )

    pair_groups_cfg = cfg.get("pair_groups") or []
    if pair_groups_cfg:
        if not wide_root_paths:
            raise ValueError("combined.pair_groups requires combined.wide.roots to be configured.")

        dataset_lookup = {path.name: path for path in wide_root_paths}
        dataset_lookup.update({path.name: path for path in combine_roots.values()})

        matrices_cfg = cfg.get("matrices")
        matrices_template = None
        if matrices_cfg:
            matrices_template, _ = _matrix_plot_config(matrices_cfg, settings)

        envelopes_cfg = cfg.get("envelopes")
        envelope_templates: list[EnvelopePlotConfig] = []
        if envelopes_cfg:
            entries = (
                envelopes_cfg
                if isinstance(envelopes_cfg, Sequence)
                and not isinstance(envelopes_cfg, (str, bytes))
                else [envelopes_cfg]
            )
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise ValueError("combined.envelopes entries must be mappings.")
                config, _ = _envelope_plot_config(entry, settings)
                envelope_templates.append(config)

        entries = (
            pair_groups_cfg
            if isinstance(pair_groups_cfg, Sequence) and not isinstance(pair_groups_cfg, (str, bytes))
            else [pair_groups_cfg]
        )
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("combined.pair_groups entries must be mappings.")
            pair_name = str(entry.get("name") or entry.get("label") or "").strip()
            if not pair_name:
                raise ValueError("combined.pair_groups entries require 'name'.")

            datasets_cfg = entry.get("datasets")
            if not datasets_cfg:
                raise ValueError(f"combined.pair_groups[{pair_name}] requires 'datasets'.")
            dataset_iter = (
                datasets_cfg
                if isinstance(datasets_cfg, Sequence) and not isinstance(datasets_cfg, (str, bytes, Path))
                else [datasets_cfg]
            )
            resolved_roots: list[Path] = []
            missing: list[str] = []
            for ds in dataset_iter:
                key = Path(str(ds)).name
                path = dataset_lookup.get(key)
                if path is None:
                    missing.append(key)
                else:
                    resolved_roots.append(path)
            if missing:
                raise ValueError(
                    f"combined.pair_groups[{pair_name}] datasets not found among combined.wide.roots: {missing}"
                )

            out_dir = _ensure_path(entry.get("out_dir"), f"pair_groups[{pair_name}].out_dir")
            trial_type_filter = entry.get("trial_type_filter")

            pair_combined_dir = out_dir / "combined"
            pair_combined_dir.mkdir(parents=True, exist_ok=True)
            pair_matrix_dir = out_dir / "matrix"
            pair_wide_csv = pair_combined_dir / "all_envelope_rows_wide.csv"

            extra_exports = {
                "training": str(pair_combined_dir / "all_envelope_rows_wide_training.csv")
            }
            extra_matrix_dirs = {"training": str(pair_matrix_dir / "training")}

            _block = "pair_groups"
            _fp_kw = dict(
                measure_cols=wide_measure_cols,
                fps_fallback=wide_fps_fallback,
                distance_limits=limits,
                non_reactive_threshold=non_reactive_threshold,
                # build_wide_csv is not passed low_max_threshold_px at this call
                # site, so it uses the default. The fingerprint must record the
                # same value or every run reports drift.
                low_max_threshold_px=LOW_MAX_FLAG_THRESHOLD_PX,
                use_per_trial_baseline=use_per_trial_baseline,
                threshold_rule=wide_threshold_rule,
                # Same variable passed to the build_wide_csv(...) call below --
                # the fingerprint must record exactly what the build used.
                trial_type_filter=trial_type_filter,
            )
            _frozen = _resolve_frozen_slices(
                settings,
                [str(path) for path in resolved_roots],
                _block,
                thawed=getattr(settings, "_thawed", ()),
                thaw_all=getattr(settings, "_thaw_all", False),
                **_fp_kw,
            )
            print(f"[analysis] combined.pair_groups[{pair_name}].wide → {pair_wide_csv}")
            build_wide_csv(
                [str(path) for path in resolved_roots],
                str(pair_wide_csv),
                measure_cols=wide_measure_cols,
                fps_fallback=wide_fps_fallback,
                exclude_roots=wide_exclude_cfg,
                distance_limits=limits,
                config_path=config_path,
                trial_type_filter=trial_type_filter,
                extra_trial_exports=extra_exports,
                non_reactive_threshold=non_reactive_threshold,
                use_per_trial_baseline=use_per_trial_baseline,
                threshold_rule=wide_threshold_rule,
                frozen_slices=_frozen or None,
                # resolved_roots, matching the roots this call actually builds
                # from -- `roots` is a different list in this branch.
                frozen_folders=_resolve_frozen_folders(settings, resolved_roots, skip=_frozen) or None,
            )
            _write_freeze_cache(
                settings,
                _block,
                str(pair_wide_csv),
                list(extra_exports.values()),
                fingerprint_for=lambda ds: _freeze_fingerprint(settings, ds, **_fp_kw),
                spliced=_frozen,
                walked={Path(str(r)).name for r in resolved_roots},
            )

            print(f"[analysis] combined.pair_groups[{pair_name}].matrix → {pair_matrix_dir}")
            wide_to_matrix(str(pair_wide_csv), str(pair_matrix_dir))
            for trial_key, matrix_dir in extra_matrix_dirs.items():
                export_csv = extra_exports.get(trial_key)
                if not export_csv:
                    continue
                wide_to_matrix(export_csv, matrix_dir)

            _render_pair_visuals(pair_name, pair_matrix_dir, matrices_template, envelope_templates)

    matrix_cfg = cfg.get("matrix")
    if matrix_cfg:
        input_csv = _ensure_path(matrix_cfg.get("input_csv"), "input_csv")
        out_dir = _ensure_path(matrix_cfg.get("out_dir"), "out_dir")
        print(f"[analysis] combined.matrix → {out_dir}")
        wide_to_matrix(str(input_csv), str(out_dir))

    matrices_cfg = cfg.get("matrices")
    if matrices_cfg:
        config, smb_path = _matrix_plot_config(matrices_cfg, settings)
        print(f"[analysis] combined.matrices → {config.out_dir}")
        generate_reaction_matrices(config)

    envelopes_cfg = cfg.get("envelopes")
    if envelopes_cfg and defer_envelopes:
        print(
            "[analysis] combined.envelopes → DEFERRED until after reactions "
            "(model scores will be annotated)."
        )
    elif envelopes_cfg:
        entries = (
            envelopes_cfg
            if isinstance(envelopes_cfg, Sequence)
            and not isinstance(envelopes_cfg, (str, bytes))
            else [envelopes_cfg]
        )
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("combined.envelopes entries must be mappings.")
            config, smb_path = _envelope_plot_config(entry, settings)
            print(f"[analysis] combined.envelopes → {config.out_dir}")
            generate_envelope_plots(config)

    overlay_cfg = cfg.get("overlay")
    if overlay_cfg:
        combined_matrix = _ensure_path(overlay_cfg.get("combined_matrix"), "combined_matrix")
        combined_codes = _ensure_path(overlay_cfg.get("combined_codes"), "combined_codes")
        distance_matrix = _ensure_path(overlay_cfg.get("distance_matrix"), "distance_matrix")
        distance_codes = _ensure_path(overlay_cfg.get("distance_codes"), "distance_codes")
        out_dir = _ensure_path(overlay_cfg.get("out_dir"), "out_dir")
        latency_sec = float(overlay_cfg.get("latency_sec", 0.0))
        after_show = float(overlay_cfg.get("after_show_sec", 30.0))
        thresh = float(overlay_cfg.get("threshold_std_mult", 4.0))
        odor_on = float(overlay_cfg.get("odor_on_s", 30.0))
        odor_off = float(overlay_cfg.get("odor_off_s", 60.0))
        odor_latency = float(overlay_cfg.get("odor_latency_s", overlay_cfg.get("odor_latency", 0.0)))
        overwrite = bool(overlay_cfg.get("overwrite", False))

        missing_files = [
            p for p in (combined_matrix, combined_codes, distance_matrix, distance_codes)
            if not p.exists()
        ]
        if missing_files:
            print(
                "[analysis] combined.overlay → skipping; missing files: "
                + ", ".join(str(p) for p in missing_files)
            )
        else:
            print(f"[analysis] combined.overlay → {out_dir}")
            overlay_sources(
                {
                    "RMS × Angle": {
                        "MATRIX_NPY": str(combined_matrix),
                        "CODES_JSON": str(combined_codes),
                    },
                    "RMS": {
                        "MATRIX_NPY": str(distance_matrix),
                        "CODES_JSON": str(distance_codes),
                    },
                },
                latency_sec=latency_sec,
                after_show_sec=after_show,
                output_dir=str(out_dir),
                threshold_mult=thresh,
                odor_on_s=odor_on,
                odor_off_s=odor_off,
                odor_latency_s=odor_latency,
                overwrite=overwrite,
                non_reactive_threshold=settings.non_reactive_span_px if settings is not None else None,
            )

    # The local→secured copy (and optional local deletion) is deferred to the
    # very end of the run when requested, so it happens AFTER all figures have
    # been rendered from the local data — mirroring the file-server sync step.
    # When deferred, return a thunk the caller runs once all figures are done.
    if defer_secure_cleanup:
        return lambda: _run_secure_cleanup(
            cfg, settings, folder_filter=folder_filter, flagged_dirs=flagged_dirs
        )
    _run_secure_cleanup(
        cfg, settings, folder_filter=folder_filter, flagged_dirs=flagged_dirs
    )
    return None


def _run_secure_cleanup(
    cfg: Mapping[str, Any] | None,
    settings: Settings | None,
    *,
    folder_filter: str | None = None,
    flagged_dirs: Sequence[Path] | None = None,
) -> None:
    """Copy local experiment data to secured storage (and optionally delete the
    local source). Split out of ``_run_combined`` so it can run AFTER all figures
    are rendered from the local data, mirroring the file-server sync step.

    ``flagged_dirs`` is re-discovered from ``settings`` when not supplied (cheap),
    so this can be called standalone from ``main`` after the deferred figures.
    """
    if not cfg:
        return

    if flagged_dirs is None:
        flagged_dirs = []
        if settings is not None and settings.flagged_root:
            flagged_dirs = discover_flagged_directories(settings.flagged_root)

    secure_cfg = cfg.get("secure_cleanup")
    if secure_cfg:
        sources = secure_cfg.get("sources") or []
        if not sources:
            raise ValueError("combined.secure_cleanup.sources must list at least one directory.")
        dest = _ensure_path(secure_cfg.get("destination"), "secure_cleanup.destination")
        resolved_sources = [str(_ensure_path(src, "secure_cleanup.sources")) for src in sources]

        # In single-folder mode, only sync the target folder to secured storage.
        if folder_filter:
            target = folder_filter.strip().lower()
            resolved_sources = [
                src for src in resolved_sources
                if Path(src).name.lower() == target
            ]
            if not resolved_sources:
                LOGGER.info(
                    "[analysis] combined.secure_cleanup → SKIP (target folder '%s' not in sources)",
                    folder_filter,
                )

        perform_cleanup = bool(secure_cfg.get("perform_cleanup", True))
        if resolved_sources:
            print(
                f"[analysis] combined.secure_cleanup → dest={dest} sources={resolved_sources} perform_cleanup={perform_cleanup}"
            )
            secure_copy_and_cleanup(
                resolved_sources,
                str(dest),
                perform_cleanup=perform_cleanup,
            )

    # Secure-copy flagged experiments to their own secured storage folder.
    if flagged_dirs:
        # In single-folder mode, only sync the target flagged dir.
        sync_flagged = list(flagged_dirs)
        if folder_filter:
            target = folder_filter.strip().lower()
            sync_flagged = [fd for fd in flagged_dirs if fd.name.lower() == target]
        if settings is not None and settings.flagged_secured_root:
            flagged_dest: Path | None = Path(settings.flagged_secured_root).expanduser().resolve()
        else:
            flagged_dest = None
        if flagged_dest is not None and sync_flagged:
            flagged_dest.mkdir(parents=True, exist_ok=True)
            flagged_sources = [str(fd) for fd in sync_flagged]
            print(
                f"[analysis] combined.secure_cleanup (flagged) → dest={flagged_dest} "
                f"sources={[fd.name for fd in sync_flagged]}"
            )
            secure_copy_and_cleanup(
                flagged_sources,
                str(flagged_dest),
                perform_cleanup=False,  # Never auto-delete flagged experiment source data
            )


#: Filename prefix per metric. Two metrics sharing one stem would have the
#: second silently overwrite the first, so the prefix is part of the stem.
_PUBFIG_STEM_PREFIX = {
    "mean-score": "pubfig_mean_score_train_vs_ctrl_",
    "percent-responding": "pubfig_pct_responding_train_vs_ctrl_",
}


def _pubfig_commands(
    settings: Settings,
    *,
    python_exec: str,
    csv_path: Path,
    config_path: Path | None,
) -> list[list[str]]:
    """One ``pubfig_score_train_vs_control.py`` command per cohort x metric.

    Built as a pure function for the same reason as ``_thaw_cli_args``: the
    caller (``_run_reactions``) needs a model, a predictions CSV and an output
    tree before it will run, so the command shape is not otherwise testable.

    Returns ``[]`` when nothing is configured -- an unconfigured pipeline must
    not start rendering figures.
    """
    pub = getattr(settings.reaction_prediction, "publication_figures", None)
    if pub is None or not pub.cohorts:
        return []

    script = REPO_ROOT / "scripts" / "analysis" / "pubfig_score_train_vs_control.py"
    flagged_csv = str(getattr(settings, "flagged_flies_csv", "") or "")
    frozen_figures = set(_figure_frozen_datasets(settings))
    from fbpipe.autofreeze import active_plan

    auto = active_plan() is not None
    commands: list[list[str]] = []
    for cohort in pub.cohorts:
        arms = [cohort.train_dataset]
        control = cohort.train_dataset.replace("-Training-", "-Control-", 1)
        if auto and control != cohort.train_dataset:
            # Auto-frozen arms freeze independently: a control arm that got
            # new data must redraw the figure its trained arm keys.
            arms.append(control)
        if all(arm in frozen_figures for arm in arms):
            # The cohort is keyed by its trained arm; frozen there means the
            # published figure is final.
            continue
        for metric in cohort.metrics:
            stem = _PUBFIG_STEM_PREFIX[metric] + (
                cohort.out_stem or cohort.train_dataset
            )
            cmd = [
                str(Path(python_exec).expanduser()),
                str(script),
                "dataset",
                "--train-dataset", cohort.train_dataset,
                "--predictions-csv", str(csv_path),
                "--metric", metric,
                "--out-stem", stem,
            ]
            if pub.figures_dir:
                cmd.extend(["--figures-dir", pub.figures_dir])
            if cohort.fly_months:
                # Only when non-empty: the driver reads "" as "keep nothing".
                cmd.extend(["--fly-months", ",".join(cohort.fly_months)])
            if flagged_csv:
                cmd.extend(["--flagged-flies-csv", flagged_csv])
            if cohort.genotype:
                cmd.extend(["--genotype", cohort.genotype])
            for odor, label in getattr(cohort, "odor_remap", ()) or ():
                cmd.extend(["--odor-remap", f"{odor}={label}"])
            if cohort.label:
                cmd.extend(["--cohort-label", cohort.label])
            if config_path is not None:
                cmd.extend(["--config", str(config_path)])
            commands.append(cmd)
    return commands


def _thaw_cli_args(settings: Settings) -> list[str]:
    """CLI args forwarding this run's --thaw / --thaw-all selection.

    ``score_summary.py`` runs as a subprocess (see the ``score_cmd`` build
    below) and does not share the parent's ``settings`` object, so its own
    ``--thaw``/``--thaw-all`` flags (parsed in its ``_parse_args``) must be
    forwarded explicitly. ``settings._thawed``/``settings._thaw_all`` are the
    same run-only attributes ``main()`` stashes on ``settings`` from this
    process's own --thaw/--thaw-all (search for ``settings._thawed =`` in
    ``main()``) -- read defensively since they are not dataclass fields.
    """
    args: list[str] = []
    for name in getattr(settings, "_thawed", ()):
        args.extend(["--thaw", str(name)])
    if getattr(settings, "_thaw_all", False):
        args.append("--thaw-all")
    return args


def _run_reactions(settings: Settings, config_path: Path | None = None) -> None:
    reaction_cfg = settings.reaction_prediction

    required_pairs: list[tuple[str, str]] = []
    if reaction_cfg.run_prediction:
        required_pairs.extend(
            [
                ("reaction_prediction.data_csv", reaction_cfg.data_csv),
                ("reaction_prediction.model_path", reaction_cfg.model_path),
            ]
        )
    required_pairs.append(("reaction_prediction.output_csv", reaction_cfg.output_csv))

    missing = [name for name, value in required_pairs if not value]

    if missing:
        print(
            "[REACTION] Skipping reaction prediction; missing configuration values: "
            + ", ".join(missing)
        )
        return

    output_csv_path = Path(reaction_cfg.output_csv).expanduser()

    if reaction_cfg.run_prediction:
        prediction_expected = {
            "data_csv": str(Path(reaction_cfg.data_csv).expanduser()),
            "model_path": str(Path(reaction_cfg.model_path).expanduser()),
            "model_type": reaction_cfg.model_type,
        }
        data_path = Path(reaction_cfg.data_csv).expanduser()
        model_path = Path(reaction_cfg.model_path).expanduser()
        if data_path.exists():
            prediction_expected["data_mtime"] = data_path.stat().st_mtime
        if model_path.exists():
            prediction_expected["model_mtime"] = model_path.stat().st_mtime
        prediction_expected["output_exists"] = output_csv_path.exists()
        if reaction_cfg.threshold is not None:
            prediction_expected["threshold"] = float(reaction_cfg.threshold)
        if output_csv_path.exists():
            prediction_expected["output_mtime"] = output_csv_path.stat().st_mtime

        skip_prediction = output_csv_path.exists() and _should_skip(
            settings,
            category="reaction_prediction",
            key="predict",
            expected=prediction_expected,
            force_flag=settings.force.reaction_prediction,
        )

        if skip_prediction:
            print(
                "[analysis] reactions.predict cached → skipping. Set force.reaction_prediction=true to recompute."
            )
        else:
            print("[analysis] reactions.predict → flybehavior-response predict")
            predict_reactions.main(settings)
            prediction_expected["output_exists"] = output_csv_path.exists()
            if output_csv_path.exists():
                prediction_expected["output_mtime"] = output_csv_path.stat().st_mtime
            prediction_expected["version"] = STATE_VERSION
            _write_state(settings, "reaction_prediction", "predict", prediction_expected)
    else:
        print("[analysis] reactions.predict → skipped (run_prediction = False)")

    if not reaction_cfg.matrix.out_dir:
        print("[REACTION] Matrix generation skipped; reaction_prediction.matrix.out_dir is empty.")
        return

    python_exec = reaction_cfg.python or sys.executable
    script_path = REPO_ROOT / "scripts" / "analysis" / "reaction_matrix_from_spreadsheet.py"
    csv_path = output_csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Reaction prediction CSV not found: {csv_path}")

    matrix_cfg = reaction_cfg.matrix
    out_dir = Path(matrix_cfg.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Only emit the "_unordered" plot variant (produced by the ``trained-first``
    # order) and drop the redundant ``observed`` ordering: for v2 odor-column
    # matrices the two are identical, and only the "_unordered" files are the
    # ones symlinked into the Figures dir. Fall back to the configured orders if
    # ``trained-first`` isn't among them (e.g. an observed-only legacy config).
    plot_orders = [o for o in matrix_cfg.trial_orders if o == "trained-first"]
    if not plot_orders:
        plot_orders = list(matrix_cfg.trial_orders)
    elif list(plot_orders) != list(matrix_cfg.trial_orders):
        print(
            "[analysis] reactions: emitting only the '_unordered' (trained-first) "
            f"plot variant; skipping {sorted(set(matrix_cfg.trial_orders) - {'trained-first'})}."
        )

    matrix_expected = {
        "non_reactive_span_px": settings.non_reactive_span_px,
        "flagged_flies_csv": str(getattr(settings, "flagged_flies_csv", "") or ""),
        "csv_mtime": csv_path.stat().st_mtime,
        "latency_sec": matrix_cfg.latency_sec,
        "after_window_sec": matrix_cfg.after_window_sec,
        "row_gap": matrix_cfg.row_gap,
        "height_per_gap_in": matrix_cfg.height_per_gap_in,
        "bottom_shift_in": matrix_cfg.bottom_shift_in,
        "trial_orders": list(plot_orders),
        "include_hexanol": matrix_cfg.include_hexanol,
    }

    skip_matrix = _should_skip(
        settings,
        category="reaction_matrix",
        key="reaction_prediction",
        expected=matrix_expected,
        force_flag=settings.force.reaction_matrix,
    )

    flagged_csv = str(getattr(settings, "flagged_flies_csv", "") or "")
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    extra_path = str(REPO_ROOT / "src")
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [extra_path, pythonpath]))

    if skip_matrix:
        print(
            "[analysis] reactions.matrix cached → skipping. Set force.reaction_matrix=true to recompute."
        )
    else:
        cmd = [
            str(Path(python_exec).expanduser()),
            str(script_path),
            "--csv-path",
            str(csv_path.resolve()),
            "--out-dir",
            str(out_dir.resolve()),
            "--latency-sec",
            str(matrix_cfg.latency_sec),
            "--after-window-sec",
            str(matrix_cfg.after_window_sec),
            "--row-gap",
            str(matrix_cfg.row_gap),
            "--height-per-gap-in",
            str(matrix_cfg.height_per_gap_in),
            "--bottom-shift-in",
            str(matrix_cfg.bottom_shift_in),
            "--non-reactive-threshold",
            str(settings.non_reactive_span_px),
        ]

        if flagged_csv:
            cmd.extend(["--flagged-flies-csv", flagged_csv])

        for trial_order in plot_orders:
            cmd.extend(["--trial-order", trial_order])

        if not matrix_cfg.include_hexanol:
            cmd.append("--exclude-hexanol")
        if matrix_cfg.overwrite:
            cmd.append("--overwrite")
        if config_path is not None:
            cmd.extend(["--config", str(config_path)])
        # Run the reaction figures in the config's experiment protocol so legacy
        # datasets (labels without odor suffixes) don't hit the v2-only code path.
        cmd.extend(["--protocol", str(settings.protocol)])

        cmd.extend(_thaw_cli_args(settings))
        print("[analysis] reactions.matrix →", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)

        # --- Publication figures, one per configured cohort x metric ---
        for pub_cmd in _pubfig_commands(
            settings,
            python_exec=python_exec,
            csv_path=csv_path.resolve(),
            config_path=config_path,
        ):
            print("[analysis] publication_figures →", " ".join(pub_cmd))
            subprocess.run(pub_cmd, check=True, env=env)

        # --- Training vs Control comparison plots ---
        train_vs_ctrl_script = REPO_ROOT / "scripts" / "analysis" / "reaction_matrix_training_vs_control.py"
        if train_vs_ctrl_script.exists():
            tvc_cmd = [
                str(Path(python_exec).expanduser()),
                str(train_vs_ctrl_script),
                "--csv-path",
                str(csv_path.resolve()),
                "--out-dir",
                str(out_dir.resolve()),
                "--latency-sec",
                str(matrix_cfg.latency_sec),
                "--after-window-sec",
                str(matrix_cfg.after_window_sec),
                "--row-gap",
                str(matrix_cfg.row_gap),
                "--height-per-gap-in",
                str(matrix_cfg.height_per_gap_in),
                "--bottom-shift-in",
                str(matrix_cfg.bottom_shift_in),
            ]
            for trial_order in plot_orders:
                tvc_cmd.extend(["--trial-order", trial_order])
            if flagged_csv:
                tvc_cmd.extend(["--flagged-flies-csv", flagged_csv])
            if not matrix_cfg.include_hexanol:
                tvc_cmd.append("--exclude-hexanol")
            if matrix_cfg.overwrite:
                tvc_cmd.append("--overwrite")
            if config_path is not None:
                tvc_cmd.extend(["--config", str(config_path)])
            tvc_cmd.extend(["--protocol", str(settings.protocol)])
            tvc_cmd.extend(_thaw_cli_args(settings))
            print("[analysis] reactions.train_vs_ctrl →", " ".join(tvc_cmd))
            subprocess.run(tvc_cmd, check=True, env=env)
        else:
            print("[analysis] reactions.train_vs_ctrl script not found, skipping.")

        matrix_expected["version"] = STATE_VERSION
        _write_state(settings, "reaction_matrix", "reaction_prediction", matrix_expected)

    # --- Ordinal score summary plots (always runs when model_type=ordinal) ---
    if reaction_cfg.model_type == "ordinal" and csv_path.exists():
        score_script = REPO_ROOT / "scripts" / "analysis" / "score_summary.py"
        if score_script.exists():
            score_out_dir = out_dir / "score_summary"
            score_cmd = [
                str(Path(python_exec).expanduser()),
                str(score_script),
                "--csv-path",
                str(csv_path.resolve()),
                "--out-dir",
                str(score_out_dir.resolve()),
                "--overwrite",
            ]
            if flagged_csv:
                score_cmd.extend(["--flagged-flies-csv", flagged_csv])
            if config_path is not None:
                score_cmd.extend(["--config", str(config_path)])
            score_cmd.extend(["--protocol", str(settings.protocol)])
            score_cmd.extend(_thaw_cli_args(settings))
            print("[analysis] score_summary →", " ".join(score_cmd))
            subprocess.run(score_cmd, check=True, env=env)

            # --- RandomPanel per-odorant concentration comparison (KW/Holm) ---
            # Skips itself cleanly when the CSV isn't a RandomPanel run holding
            # all three concentrations (10 / 1 / 0.1).
            conc_script = REPO_ROOT / "scripts" / "analysis" / "randompanel_conc_comparison.py"
            if conc_script.exists():
                conc_cmd = [
                    str(Path(python_exec).expanduser()),
                    str(conc_script),
                    "--csv-path",
                    str(csv_path.resolve()),
                    "--out-dir",
                    str(score_out_dir.resolve()),
                    "--genotype-subdir",
                    "--overwrite",
                ]
                if config_path is not None:
                    conc_cmd.extend(["--config", str(config_path)])
                conc_cmd.extend(_thaw_cli_args(settings))
                print("[analysis] randompanel_conc_comparison →", " ".join(conc_cmd))
                subprocess.run(conc_cmd, check=True, env=env)
            else:
                print("[analysis] randompanel_conc_comparison script not found, skipping.")
        else:
            print("[analysis] score_summary script not found, skipping.")


def _run_pipeline(
    config_path: Path,
    *,
    main_directory: str | None = None,
    steps: Sequence[str] | None = None,
    num_workers: int = 1,
) -> None:
    cmd = [sys.executable, "-m", "fbpipe.pipeline", "--config", str(config_path)]
    if steps:
        cmd.extend(steps)
    else:
        cmd.append("all")
    suffix = f" (MAIN_DIRECTORY={main_directory})" if main_directory else ""
    env = os.environ.copy()
    if main_directory:
        env["MAIN_DIRECTORY"] = main_directory
    pythonpath = env.get("PYTHONPATH")
    extra_paths = [str(SRC_ROOT)]
    if pythonpath:
        extra_paths.append(pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(extra_paths)

    n = max(1, int(num_workers))
    if n <= 1:
        print(f"[analysis] pipeline{suffix} → {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Pipeline failed with exit code {e.returncode}")
            if e.stderr:
                print(e.stderr.decode() if hasattr(e.stderr, "decode") else e.stderr)
            raise
        return

    # Parallel YOLO: launch N worker processes, each handling a disjoint 1/N
    # slice of the deterministically-ordered video list (yolo_infer.main reads
    # NUM_WORKERS / WORKER_INDEX). Same mechanism as scripts/pipeline/
    # parallel_yolo.py, inlined so a single run_workflows run fans out the step.
    # Only valid for the YOLO step — other steps are not worker-sliced, so the
    # caller must pass num_workers>1 only for steps=("yolo",).
    print(f"[analysis] pipeline{suffix} → {n} parallel workers: {' '.join(cmd)}")
    procs = []
    for i in range(n):
        worker_env = dict(env)
        worker_env["NUM_WORKERS"] = str(n)
        worker_env["WORKER_INDEX"] = str(i)
        # One BLAS thread per worker, stated outright. Until now the cap held
        # only by accident: importing ultralytics here sets OMP_NUM_THREADS=1
        # when it is unset, and OpenBLAS falls back to that. Without it (a
        # preset OMP_NUM_THREADS, another ultralytics) numpy's 16-thread
        # OpenBLAS pool spins at 100% CPU between the tracker's tiny per-frame
        # np.linalg.inv calls. It has to be exactly 1: 2 still leaves a
        # spinner. Outputs are identical. OMP/torch threads are left alone.
        for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            worker_env[var] = "1"
        procs.append(subprocess.Popen(cmd, env=worker_env))
    return_codes = [p.wait() for p in procs]
    failed = [(i, rc) for i, rc in enumerate(return_codes) if rc != 0]
    if failed:
        idx, rc = failed[0]
        print(f"[ERROR] {len(failed)} of {n} YOLO worker(s) failed; first: worker {idx} exit {rc}")
        raise subprocess.CalledProcessError(rc, cmd)


#: What the pre-checks count as a video (matched case-insensitively): yolo_infer's
#: own .mp4/.avi, the rigs' raw .h264 the secure cleanup prunes with .mp4, and
#: every container the per-trial move_videos step relocates. Wider only ever
#: means more launches; tests/test_yolo_launch_precheck.py pins the coverage.
_PRECHECK_VIDEO_SUFFIXES = (".mp4", ".avi", ".h264", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v")


def _tree_has_video(root: str | Path) -> bool:
    """True if any entry anywhere under *root* has a video suffix.

    A deliberately wide SUPERSET of yolo_infer.main's discovery (a video
    directly inside a live batch dir): folder freeze, depth and entry type are
    ignored, symlinked dirs are followed (iter_live_batch_dirs keeps them too)
    and an unlistable dir reads as "may hold a video". A pending video this
    missed would be copied to secured storage and deleted by the secure
    cleanup without ever being inferred, so it may only err towards launching.
    A missing root has nothing to launch for -- yolo_infer skips it as well.
    """
    start = Path(root)
    try:
        if not start.is_dir():
            return False
    except OSError:  # e.g. EACCES: let the worker meet it, as today's launch would
        return True
    seen: set[tuple[int, int]] = set()
    stack = [start]
    while stack:
        current = stack.pop()
        try:
            st = current.stat()
            if (st.st_dev, st.st_ino) in seen:  # symlink loop
                continue
            seen.add((st.st_dev, st.st_ino))
            with os.scandir(current) as entries:
                for entry in entries:
                    if entry.name.lower().endswith(_PRECHECK_VIDEO_SUFFIXES):
                        return True
                    if entry.is_dir():
                        stack.append(Path(entry.path))
        except OSError:
            return True
    return False


def _pipeline_run_walks_beyond_root(settings, root: str | Path) -> bool:
    """Whether the ``fbpipe.pipeline`` run ``_run_pipeline`` starts for *root*
    walks anything but *root* itself, so that *root* cannot show it has nothing
    to do.

    The worker reads MAIN_DIRECTORIES ahead of the MAIN_DIRECTORY it is handed,
    splits and strips that value (config._parse_env_paths, so a root with a
    comma, an os.pathsep or edge whitespace arrives as other paths), and
    get_main_directories appends every auto-discovered flagged dir to each run.
    """
    from fbpipe.config import _parse_env_paths

    if os.environ.get("MAIN_DIRECTORIES") is not None:
        return True
    if _parse_env_paths(str(root)) != str(root):
        return True
    flagged_root = getattr(settings, "flagged_root", "")
    if flagged_root and getattr(settings, "auto_discover_flagged", True):
        return bool(discover_flagged_directories(flagged_root))
    return False


def _yolo_work_reason(settings, root: str | Path) -> str | None:
    """Why the YOLO launch for *root* must go ahead; None only when it would
    certainly run to completion without finding a video.

    Mirrors what yolo_infer.main does before it looks for one: without CUDA
    (allow_cpu off) it raises, and so does YOLO() when neither the .engine
    sibling nor the model file exists. Today's launch then fails the night
    loudly, even for an empty root. Skipped, the night would go on, and without
    CUDA the per-trial steps would run their CPU variants (fbpipe.pipeline
    picks them), whose outputs differ from the GPU path's. torch settles the
    CUDA answer once per process, and fbpipe.pipeline's import has already
    asked here: the check is free and reflects CUDA as the run started.
    """
    import torch

    if not getattr(settings, "allow_cpu", False) and not torch.cuda.is_available():
        return "CUDA is unavailable (allow_cpu off), so its YOLO launch must fail the run as today"
    model = Path(settings.model_path)
    if not model.with_suffix(".engine").exists() and not model.exists():
        return f"its YOLO model {model} is missing, so its YOLO launch must fail the run as today"
    if _pipeline_run_walks_beyond_root(settings, root):
        return "its pipeline run walks more than its root (MAIN_DIRECTORIES, flagged dirs or a split path)"
    if _tree_has_video(root):
        return "it holds video files"
    return None


def _yolo_may_have_work(settings, root: str | Path) -> bool:
    """False only when a YOLO launch for *root* is certainly a no-op."""
    return _yolo_work_reason(settings, root) is not None


def _freeze_cache_blocks(combined_cfg, settings) -> list[tuple[str, frozenset[str], dict]]:
    """``(wide_block, dataset names, fingerprint kwargs)`` for every
    ``_resolve_frozen_slices`` call ``_run_combined`` will make.

    Derived from the config the way ``_run_combined`` derives them, so the
    dataset loop can tell before any per-trial work whether build_wide_csv will
    splice a data-frozen dataset's cached rows or rebuild them from its
    per-trial files. Not for --folder runs, whose combined step differs.
    tests/test_frozen_dataset_skip.py pins this against the calls a real
    ``_run_combined`` makes -- change the two together.
    """
    cfg = combined_cfg if isinstance(combined_cfg, Mapping) else {}
    limits = (settings.class2_min, settings.class2_max)
    flagged_names = (
        [d.name for d in discover_flagged_directories(settings.flagged_root)]
        if settings.flagged_root
        else []
    )

    def _root_names(roots_cfg, field: str) -> list[str]:
        if isinstance(roots_cfg, Sequence) and not isinstance(roots_cfg, (str, bytes, Path)):
            roots_iter = list(roots_cfg)
        else:
            roots_iter = [roots_cfg]
        return [_ensure_path(root, field).name for root in roots_iter]

    def _with_flagged(names: list[str]) -> list[str]:
        present = {name.lower() for name in names}
        return names + [n for n in flagged_names if n.lower() not in present]

    def _kw(measure_cols, fps_fallback, use_per_trial_baseline, threshold_rule, trial_type_filter):
        return dict(
            measure_cols=measure_cols,
            fps_fallback=fps_fallback,
            distance_limits=limits,
            non_reactive_threshold=settings.non_reactive_span_px,
            low_max_threshold_px=LOW_MAX_FLAG_THRESHOLD_PX,
            use_per_trial_baseline=use_per_trial_baseline,
            threshold_rule=threshold_rule,
            trial_type_filter=trial_type_filter,
        )

    blocks: list[tuple[str, frozenset[str], dict]] = []
    wide_cfg = cfg.get("wide")
    wide_threshold_rule = ThresholdRule.from_mapping(wide_cfg or {})
    wide_names: list[str] = []
    wide_measure_cols: Sequence[str] = ["envelope_of_rms"]
    wide_fps_fallback = 40.0
    use_per_trial_baseline = False
    if wide_cfg:
        if not wide_cfg.get("roots", []):
            raise ValueError("combined.wide.roots must list at least one directory.")
        wide_names = _with_flagged(_root_names(wide_cfg.get("roots"), "roots"))
        wide_measure_cols = wide_cfg.get("measure_cols") or ["envelope_of_rms"]
        wide_fps_fallback = float(wide_cfg.get("fps_fallback", 40.0))
        use_per_trial_baseline = wide_cfg.get("use_per_trial_baseline", False)
        blocks.append((
            "wide",
            frozenset(wide_names),
            _kw(wide_measure_cols, wide_fps_fallback, use_per_trial_baseline,
                wide_threshold_rule, wide_cfg.get("trial_type_filter")),
        ))

    for base_key, default_cols in (
        ("combined_base", ["combined_pct", "combined_base"]),
        ("distance_base", ["distance_percentage"]),
    ):
        base_cfg = cfg.get(base_key)
        base_wide_cfg = base_cfg.get("wide") if base_cfg else None
        if not base_wide_cfg:
            continue
        if base_wide_cfg.get("roots"):
            names = _root_names(base_wide_cfg.get("roots"), f"{base_key}.wide.roots")
        elif wide_names:
            names = list(wide_names)
        else:
            raise ValueError(f"combined.{base_key}.wide.roots must list at least one directory.")
        trial_type_filter = base_wide_cfg.get("trial_type_filter")
        if trial_type_filter is None and wide_cfg:
            trial_type_filter = wide_cfg.get("trial_type_filter")
        blocks.append((
            base_key,
            frozenset(_with_flagged(names)),
            _kw(
                base_wide_cfg.get("measure_cols") or list(default_cols),
                float(base_wide_cfg.get("fps_fallback", wide_fps_fallback)),
                bool(base_wide_cfg.get("use_per_trial_baseline", use_per_trial_baseline)),
                ThresholdRule.from_mapping({**(wide_cfg or {}), **base_wide_cfg}),
                trial_type_filter,
            ),
        ))

    pair_groups_cfg = cfg.get("pair_groups") or []
    if pair_groups_cfg:
        if not wide_names:
            raise ValueError("combined.pair_groups requires combined.wide.roots to be configured.")
        # pair_groups resolve their datasets among the wide AND combine roots.
        known = set(wide_names)
        combine_cfg = cfg.get("combine")
        if combine_cfg:
            if combine_cfg.get("roots"):
                combine_names = _root_names(combine_cfg.get("roots"), "combine.root")
            elif combine_cfg.get("root") is not None:
                combine_names = _root_names([combine_cfg.get("root")], "combine.root")
            else:
                raise ValueError("combined.combine requires 'root' or 'roots'.")
            by_lower = {name.lower(): name for name in combine_names + flagged_names}
            known.update(by_lower.values())
        entries = (
            pair_groups_cfg
            if isinstance(pair_groups_cfg, Sequence) and not isinstance(pair_groups_cfg, (str, bytes))
            else [pair_groups_cfg]
        )
        for entry in entries:
            if not isinstance(entry, Mapping) or not entry.get("datasets"):
                raise ValueError("combined.pair_groups entries must be mappings with 'datasets'.")
            datasets_cfg = entry.get("datasets")
            if isinstance(datasets_cfg, Sequence) and not isinstance(datasets_cfg, (str, bytes, Path)):
                names = [Path(str(ds)).name for ds in datasets_cfg]
            else:
                names = [Path(str(datasets_cfg)).name]
            missing = [name for name in names if name not in known]
            if missing:
                raise ValueError(f"combined.pair_groups datasets not found among combined.wide.roots: {missing}")
            blocks.append((
                "pair_groups",
                frozenset(names),
                _kw(wide_measure_cols, wide_fps_fallback, use_per_trial_baseline,
                    wide_threshold_rule, entry.get("trial_type_filter")),
            ))
    return blocks


def _skip_frozen_dataset(settings, root: str, blocks, *, thawed, thaw_all) -> bool:
    """True when *root*'s YOLO launch and per-trial steps can sit out tonight.

    Only a data-frozen dataset (``--thaw`` / ``--thaw-all`` honoured) whose YOLO
    launch would certainly be a no-op (_yolo_work_reason: no video at all, a
    run that walks nothing beyond its own root, CUDA and the model file there)
    and for which every freeze-cache slice the combined step will consult loads
    under the current fingerprint: build_wide_csv then splices its rows and
    never walks its root, combine skips it, and no other nightly step reads its
    per-trial files, so re-deriving them changes no output. A missing or stale
    slice means build_wide_csv rebuilds its rows tonight; the dataset then runs
    exactly as today, so that rebuild and the slice it re-caches see exactly
    today's inputs. Says why for every data-frozen dataset.
    """
    from fbpipe import freeze as _freeze

    dataset = Path(root).name
    freeze_data, _ = _freeze.freeze_flags(settings, dataset, thawed=thawed, thaw_all=thaw_all)
    if not freeze_data:
        return False
    why = _yolo_work_reason(settings, root)
    if why is not None:
        print(f"[FROZEN] {dataset}: data-frozen, but {why} - running YOLO + per-trial steps as usual.")
        return False
    consulted = [(block, kw) for block, names, kw in blocks if dataset in names]
    if not consulted:
        print(
            f"[FROZEN] {dataset}: data-frozen but no wide table caches it - "
            f"running per-trial steps as usual."
        )
        return False
    for block, kw in consulted:
        try:
            valid = _freeze.load_slice(
                settings.cache_dir, block, dataset,
                _freeze_fingerprint(settings, dataset, **kw),
            ) is not None
        except Exception:  # never skip on doubt; the combined step reports it
            valid = False
        if not valid:
            print(
                f"[FROZEN] {dataset}: data-frozen but its {block} freeze cache is "
                f"missing or stale, so build_wide_csv rebuilds its rows tonight - "
                f"running per-trial steps as usual."
            )
            return False
    print(f"[FROZEN] {dataset}: data-frozen, no videos, cache valid - skipping YOLO + per-trial steps")
    return True


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path("config") / "config.yaml"),
        help="Path to pipeline configuration YAML.",
    )
    parser.add_argument(
        "--folder",
        default=None,
        help="Process only this dataset folder (e.g., 'Hex-Training-24'). "
             "Other datasets are left untouched; combined CSVs are rebuilt from all roots.",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        default=False,
        help="Skip YOLO and pipeline processing; regenerate all figures/plots "
             "from existing CSVs and data. Forces overwrite of all cached states.",
    )
    parser.add_argument(
        "--svg",
        action="store_true",
        default=False,
        help="Emit an editable .svg next to every .png figure (vector, fully "
             "editable in Illustrator/Inkscape). PNGs are still written.",
    )
    parser.add_argument(
        "--thaw",
        action="append",
        default=[],
        metavar="DATASET",
        help=(
            "Rebuild DATASET this run (repeatable): YOLO for unprocessed videos, "
            "the per-trial steps, combine and its wide rows, then it refreezes. "
            "Folder retirements and cutoffs stay live. Does not edit config."
        ),
    )
    parser.add_argument(
        "--thaw-all",
        action="store_true",
        help=(
            "Rebuild every dataset this run (folder retirements stay live; "
            "retired folders are still not scored -- see --include-frozen)."
        ),
    )
    parser.add_argument(
        "--freeze-status",
        action="store_true",
        help=(
            "Print what auto-freeze would decide for every dataset tonight "
            "(frozen / rebuilt and why, cascade notices) and exit. Writes "
            "nothing and takes no lock."
        ),
    )
    parser.add_argument(
        "--include-frozen",
        action="store_true",
        help=(
            "Score rows from frozen experiment folders too, so "
            "model_predictions.csv is rebuilt as a complete record. The wide "
            "CSV always holds every row; this only affects the derived "
            "predictions and the figures drawn from them."
        ),
    )
    return parser


def _validate_thaw(thaw, *, datasets) -> None:
    """Exit on an unknown --thaw name rather than silently ignoring it.

    A typo'd name would otherwise look exactly like a successful thaw: the run
    proceeds, the dataset stays frozen, and nothing says so.
    """
    known = set(datasets or ())
    unknown = [t for t in (thaw or []) if t not in known]
    if unknown:
        raise SystemExit(
            f"[FREEZE] Unknown --thaw dataset(s): {', '.join(sorted(unknown))}\n"
            f"         Known datasets: {', '.join(sorted(known)) or '(none)'}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # Install the SVG sidecar before any plotting happens (covers in-process
    # renders); propagate to subprocesses via env var below.
    if args.svg:
        from fbpipe.figure_export import install_svg_sidecar
        install_svg_sidecar()
        os.environ["FBPIPE_FIGURE_SVG"] = "1"
        LOGGER.info("[figures] SVG sidecar enabled — every PNG figure also written as .svg")

    config_path = resolve_config_path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        data = _expand_datasets(yaml.safe_load(fh) or {})

    settings = load_settings(config_path)
    set_protocol(settings.protocol)

    # Fail fast on a typo'd --thaw name before any YOLO/pipeline work runs.
    _validate_thaw(args.thaw, datasets=getattr(settings, "datasets", ()))

    if args.freeze_status:
        _print_freeze_status(args, config_path, data, settings)
        return

    # One pipeline run at a time (fbpipe.utils.run_lock): a daytime run and the
    # 00:00 nightly overlapped on 2026-10-05/06, both rewriting the same tables.
    # A second run waits for the first. The plan auto-freeze publishes for this
    # run's figure subprocesses never outlives it.
    from fbpipe import autofreeze as _autofreeze

    with run_lock(_cache_root(settings), label=" ".join(["run_workflows.py", *(argv or sys.argv[1:])])):
        try:
            _main_locked(args, config_path, data, settings)
        finally:
            _autofreeze.set_active_plan(None)


def _print_freeze_status(args: argparse.Namespace, config_path: Path, data: dict, settings: Settings) -> None:
    """``--freeze-status``: tonight's auto-freeze decisions, computed read-only."""
    from fbpipe import autofreeze as _autofreeze

    roots = data.get("main_directories") or []
    if isinstance(roots, str):
        roots = [roots]
    blocks = _freeze_cache_blocks((data.get("analysis") or {}).get("combined"), settings)
    plan = _autofreeze.plan_run(
        settings,
        config_path=config_path,
        roots=[str(Path(r).expanduser()) for r in roots],
        blocks=blocks,
        fingerprint=lambda ds, kw: _freeze_fingerprint(settings, ds, **kw),
        thawed=args.thaw or (),
        thaw_all=args.thaw_all,
        folder=args.folder,
        pairs=[(t, c) for t, c, _ in _mean_trace_cohorts(getattr(settings, "datasets", ()) or ())],
        read_only=True,
    )
    if not getattr(settings, "auto_freeze", True):
        print("[FREEZE] auto_freeze is disabled in this config: the explicit freeze: flags apply.")
    for line in plan.summary_lines():
        print(line)


def _main_locked(args: argparse.Namespace, config_path: Path, data: dict, settings: Settings) -> None:
    """The whole run, under the run lock (see ``main``)."""
    from fbpipe import autofreeze as _autofreeze

    global _RUN_SUMMARY
    _RUN_SUMMARY = ""

    # Register light-only datasets so envelope_visuals draws a light span
    # instead of the odor span for those plots.
    light_windows: dict[str, tuple[float, float]] = {}
    for ds_name, ov in settings.dataset_overrides.items():
        if not getattr(ov, "light_only", False):
            continue
        start = ov.light_start_s
        duration = ov.light_duration_s
        if start is None or duration is None:
            LOGGER.warning(
                "Dataset %s has light_only=true but light_start_s/light_duration_s are not both set; skipping.",
                ds_name,
            )
            continue
        light_windows[str(ds_name)] = (float(start), float(start) + float(duration))
    set_dataset_light_windows(light_windows)

    # Per-dataset odor display-label remap (e.g. Hex-Control-24-0.1 swaps
    # Citral -> "Sour Dough Yeast (25%)" because the rig delivered the
    # substitute liquid for that channel). Registered once at startup so
    # every figure-producing call site picks it up.
    odor_remap: dict[str, dict[str, str]] = {}
    for ds_name, ov in settings.dataset_overrides.items():
        mapping = getattr(ov, "odor_remap", None)
        if mapping:
            odor_remap[str(ds_name)] = dict(mapping)
    set_dataset_odor_remap(odor_remap)
    if odor_remap:
        for ds_name, mapping in odor_remap.items():
            LOGGER.info(
                "[analysis] odor remap for %s: %s",
                ds_name,
                ", ".join(f"{k!r} -> {v!r}" for k, v in mapping.items()),
            )

    # Propagate Settings to the analysis modules so per-trial metadata lookups
    # (TrialMetadata, dataset_overrides) work without threading cfg through
    # every helper signature.
    try:
        from scripts.analysis import envelope_combined as _ec_mod
        _ec_mod.set_runtime_settings(settings)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Could not set runtime settings on envelope_combined: %s", exc)
    try:
        from scripts.analysis import envelope_exports as _ee_mod
        _ee_mod.set_runtime_settings(settings)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Could not set runtime settings on envelope_exports: %s", exc)

    dataset_cfg = data.get("main_directories")
    dataset_roots: Sequence[str] | None = None
    if isinstance(dataset_cfg, str):
        dataset_roots = (str(Path(dataset_cfg).expanduser()),)
    elif isinstance(dataset_cfg, Sequence):
        dataset_roots = tuple(str(Path(root).expanduser()) for root in dataset_cfg)

    if not dataset_roots:
        combined_cfg = data.get("analysis", {}).get("combined")
        combine_cfg: Mapping[str, Any] | None = None
        if isinstance(combined_cfg, Mapping):
            combine_cfg = combined_cfg.get("combine") if isinstance(combined_cfg.get("combine"), Mapping) else None
        if combine_cfg and isinstance(combine_cfg.get("roots"), Sequence):
            roots_seq = combine_cfg.get("roots")
            if roots_seq:
                dataset_roots = tuple(str(Path(root).expanduser()) for root in roots_seq)

    dataset_roots = tuple(dataset_roots) if dataset_roots else None

    # --- Figures-only mode ---
    figures_only: bool = args.figures_only
    if figures_only:
        LOGGER.info("=" * 70)
        LOGGER.info(
            "FIGURES-ONLY MODE: skipping YOLO & pipeline; combined CSV / matrix "
            "rebuild stays cache-aware (runs only if new flies detected); all "
            "figure-output steps re-render."
        )
        LOGGER.info("=" * 70)
        # combined: cache-aware (force=False) so wide CSV / matrix are only
        # rebuilt when settings or dataset_roots changed (i.e. new flies).
        # Every downstream figure step is forced to re-render with overwrite.
        settings = dc_replace(
            settings,
            force=dc_replace(
                settings.force,
                combined=False,
                envelope_visuals=True,
                training=True,
                reaction_prediction=True,
                reaction_matrix=True,
                dataset_means=True,
                cohort_figures=True,
            ),
        )

    # This run's auto-freeze plan (None: --figures-only, auto_freeze disabled,
    # or planning failed -- the explicit freeze: flags then apply, as before).
    plan = None

    # --- Single-folder mode filtering ---
    folder_filter: str | None = args.folder
    if not figures_only:
        if folder_filter and dataset_roots:
            target_root = _resolve_folder(folder_filter, dataset_roots)
            LOGGER.info(
                "Single-folder mode: processing only '%s' (%s)", folder_filter, target_root
            )
            pipeline_dataset_roots: Sequence[str] = (target_root,)
            # Force reprocessing for the target folder
            settings = dc_replace(
                settings,
                force=dc_replace(settings.force, pipeline=True, combined=True),
            )
        elif folder_filter and not dataset_roots:
            raise ValueError(
                f"--folder '{folder_filter}' specified but no dataset roots found in config."
            )
        else:
            pipeline_dataset_roots = dataset_roots  # type: ignore[assignment]

        if dataset_roots:
            remaining_steps = [step.name for step in ORDERED_STEPS if step.name != "yolo"]

            yolo_targets: list[str] = []
            # Per-trial runs in root order; held_back marks a frozen dataset
            # skipped below, which runs only if a video turns up by its YOLO slot.
            planned_runs: list[tuple[str, dict[str, Any], bool]] = []

            # Every wide table the combined step builds, with the datasets it
            # reads -- which freeze-cache slices a frozen dataset needs.
            freeze_blocks = None
            try:
                freeze_blocks = _freeze_cache_blocks(
                    (data.get("analysis") or {}).get("combined"), settings
                )
            except Exception as exc:  # the combined step reports it properly
                print(f"[FROZEN] Cannot predict the freeze-cache splice ({exc}); no dataset skips its steps.")

            # Auto-freeze (fbpipe.autofreeze): each dataset is rebuilt only when
            # something new arrived or the user unfroze it; the rest sit out
            # YOLO, the per-trial steps and combine, and splice their cached
            # rows. Planned once, before any launch.
            if freeze_blocks is not None and getattr(settings, "auto_freeze", True):
                try:
                    plan = _autofreeze.plan_run(
                        settings,
                        config_path=config_path,
                        roots=dataset_roots,
                        blocks=freeze_blocks,
                        fingerprint=lambda ds, kw: _freeze_fingerprint(settings, ds, **kw),
                        thawed=args.thaw or (),
                        thaw_all=args.thaw_all,
                        folder=folder_filter,
                        pairs=[
                            (train, control)
                            for train, control, _ in _mean_trace_cohorts(getattr(settings, "datasets", ()) or ())
                        ],
                    )
                except Exception as exc:  # never lose a night to the planner itself
                    plan = None
                    print(
                        f"[FREEZE] ERROR: auto-freeze planning failed ({type(exc).__name__}: {exc}); "
                        f"this run uses the explicit freeze: flags."
                    )
                    _RUN_SUMMARY = f"auto-freeze planning FAILED ({exc}); explicit freeze flags used"
                if plan is not None:
                    _autofreeze.set_active_plan(plan)
                    for line in plan.summary_lines():
                        print(line)

            for root in pipeline_dataset_roots:
                resolved = str(Path(root).expanduser())
                # Per-root, not hoisted: freeze rules are scoped to one dataset,
                # so a shared expectation would carry the wrong dataset's rules.
                pipeline_expectation = _pipeline_expectation(
                    settings, Path(resolved)
                )
                decision = plan.decision_for_root(resolved) if plan is not None else None
                if decision is not None and decision.decision == "frozen":
                    if _pipeline_run_walks_beyond_root(settings, resolved):
                        print(
                            f"[FREEZE] {decision.dataset}: frozen, but its pipeline run walks more "
                            f"than its root - running YOLO + per-trial steps as usual."
                        )
                    else:
                        # Still a YOLO target: its slot re-checks for a video
                        # YOLO would process, just before that launch.
                        yolo_targets.append(resolved)
                        planned_runs.append((resolved, dict(pipeline_expectation), True))
                        continue
                if decision is not None and decision.decision == "thawed":
                    # Something new arrived (or an explicit thaw): YOLO and the
                    # per-trial steps run whatever the manifest says -- it
                    # tracks no videos, and a hit would leave them unprocessed.
                    yolo_targets.append(resolved)
                    planned_runs.append((resolved, dict(pipeline_expectation), False))
                    continue
                skip_frozen = False
                if plan is None and freeze_blocks is not None and not folder_filter:
                    try:
                        skip_frozen = _skip_frozen_dataset(
                            settings,
                            resolved,
                            freeze_blocks,
                            thawed=args.thaw or (),
                            thaw_all=args.thaw_all,
                        )
                    except Exception as exc:  # an optimisation must never cost a dataset
                        print(
                            f"[FROZEN] {Path(resolved).name}: could not decide on a skip "
                            f"({exc}) - running YOLO + per-trial steps as usual."
                        )
                if skip_frozen:
                    # Still a YOLO target: its slot re-checks for a video just
                    # before the launch would have happened, as the launch would.
                    yolo_targets.append(resolved)
                    planned_runs.append((resolved, dict(pipeline_expectation), True))
                    continue
                if plan is None and (args.thaw_all or Path(resolved).name in set(args.thaw or ())):
                    # --thaw without auto-freeze (its kill switch): rebuild whatever
                    # the cache says. Folder rules now stay live under --thaw, so the
                    # thaw no longer changes this dataset's pipeline key.
                    yolo_targets.append(resolved)
                    planned_runs.append((resolved, dict(pipeline_expectation), False))
                    continue
                skip_pipeline = _should_skip_with_manifest(
                    settings,
                    category="pipeline",
                    key=resolved,
                    expected=pipeline_expectation,
                    force_flag=settings.force.pipeline,
                    dataset_root=Path(resolved),
                )
                if skip_pipeline and not settings.force.yolo:
                    print(
                        f"[analysis] pipeline cached → skipping full run for {resolved}. Set force.pipeline=true to recompute."
                    )
                else:
                    yolo_targets.append(resolved)

                if not skip_pipeline:
                    planned_runs.append((resolved, dict(pipeline_expectation), False))

            held_back = {resolved for resolved, _, held in planned_runs if held}
            launched: set[str] = set()
            for resolved in yolo_targets:
                if plan is not None and resolved in held_back:
                    # A frozen dataset's slot: a video YOLO would process that
                    # arrived after the plan thaws it now, so it is processed,
                    # rebuilt and refrozen tonight rather than kept for tomorrow
                    # (a video YOLO already failed on twice does not).
                    try:
                        pending = plan.pending_now(resolved)
                        why = None
                    except Exception as exc:  # doubt runs the dataset, as today
                        pending, why = [], f"its video check failed: {exc}"
                    if not pending and why is None:
                        continue
                    plan.late_thaw(resolved, pending)
                    _autofreeze.refresh_active_plan(plan)
                    print(
                        f"[FREEZE] {Path(resolved).name}: thawed at its YOLO slot - "
                        f"{why or f'{len(pending)} video(s) YOLO has not processed arrived during the run'}; "
                        f"running YOLO + per-trial steps and rebuilding it tonight."
                    )
                    _run_pipeline(
                        config_path,
                        main_directory=resolved,
                        steps=("yolo",),
                        num_workers=settings.yolo_num_workers,
                    )
                    launched.add(resolved)
                    continue
                # Checked at the launch's own slot, not up front, so a video
                # landing mid-run is seen exactly when today's launch would.
                try:
                    why = _yolo_work_reason(settings, resolved)
                except Exception as exc:  # launch: the worker meets it as today
                    print(f"[analysis] yolo → video pre-check failed for {resolved} ({exc}); launching as usual.")
                    why = f"its video pre-check failed: {exc}"
                if why is None:
                    if resolved not in held_back:
                        print(f"[analysis] yolo → no video files under {resolved}; skipping the worker launch.")
                    continue
                if resolved in held_back:
                    print(
                        f"[FROZEN] {Path(resolved).name}: no longer skippable at its YOLO "
                        f"slot ({why}) - running YOLO + per-trial steps after all."
                    )
                _run_pipeline(
                    config_path,
                    main_directory=resolved,
                    steps=("yolo",),
                    num_workers=settings.yolo_num_workers,
                )
                launched.add(resolved)

            pipeline_targets = [
                (resolved, expected)
                for resolved, expected, held in planned_runs
                if not held or resolved in launched
            ]
            if pipeline_targets:
                for resolved, expected in pipeline_targets:
                    _run_pipeline(config_path, main_directory=resolved, steps=remaining_steps)
                    payload = dict(expected)
                    payload["version"] = STATE_VERSION
                    # A forced check never reads this state: skip the manifest.
                    _write_state(
                        settings, "pipeline", resolved, payload,
                        build_manifest=not settings.force.pipeline,
                    )
            elif not yolo_targets:
                print("[analysis] pipeline skipped for all datasets (cached).")
        else:
            _run_pipeline(config_path)
    else:
        LOGGER.info("[figures-only] Skipping YOLO inference and pipeline processing.")

    # Stash the thaw selection on the final `settings` object (after any
    # `dc_replace` above, which — being `dataclasses.replace` — would drop a
    # non-field attribute set any earlier) so `_run_combined`'s
    # `_resolve_frozen_slices` calls see it.
    settings._thawed = tuple(args.thaw or ())
    settings._thaw_all = bool(args.thaw_all)
    # Read by predict_reactions to decide whether frozen folders are scored.
    # Only --include-frozen asks for it: folder retirements stay live under
    # --thaw/--thaw-all (fbpipe.utils.frozen_folders), so a full rebuild keeps
    # the retired flies out of the predictions too.
    settings.include_frozen = bool(args.include_frozen)
    if settings.include_frozen:
        print("[FREEZE] Frozen experiment folders WILL be scored this run.")
    if settings._thaw_all:
        print("[FREEZE] --thaw-all: every dataset freeze ignored this run.")
    elif settings._thawed:
        print(f"[FREEZE] Thawed this run: {', '.join(settings._thawed)}")

    analysis_cfg = data.get("analysis") or {}

    # The check_light_stimulus pipeline step (above) has upserted this run's
    # trials into the QC CSV; load it now so every envelope-trace render —
    # first pass and deferred re-render alike — annotates each light trial
    # with its physical light-on confirmation.
    _load_light_checks_for_envelopes()

    # When reactions is configured to produce a predictions CSV, defer every
    # envelope-trace render until after reactions writes scores. Each render
    # site logs a "DEFERRED" line on the first pass; the canonical render
    # happens once, with model-score annotations, in the post-reactions block.
    rx_cfg = settings.reaction_prediction
    defer_envelopes = bool(
        getattr(rx_cfg, "data_csv", None)
        and getattr(rx_cfg, "model_path", None)
        and getattr(rx_cfg, "output_csv", None)
    )
    if defer_envelopes:
        print(
            "[analysis] envelope rendering deferred to after reactions "
            f"(predictions CSV: {rx_cfg.output_csv})."
        )

    # Deferred local→secured copy (runs after ALL figures; see _run_combined).
    deferred_secure_cleanup: Callable[[], None] | None = None
    combined_ran = False
    combined_cfg_input = analysis_cfg.get("combined")
    if combined_cfg_input:
        combined_expected = {
            "non_reactive_span_px": settings.non_reactive_span_px,
            "class2_min": settings.class2_min,
            "class2_max": settings.class2_max,
            "dataset_roots": sorted(dataset_roots) if dataset_roots else [],
            # Switching legacy<->v2 changes the wide CSV / matrix rules, so the
            # protocol must be part of the cache key; otherwise a legacy run
            # after a v2 run (or vice versa) would hash-skip and reuse the wrong
            # figures.
            "protocol": settings.protocol,
            # theta sets the AUC-* columns of every wide table. Changing the rule
            # in config must invalidate this cache, or the run hash-skips and the
            # CSVs keep the numbers from the old threshold with nothing to show
            # it. Same reasoning as the freeze fingerprint (_freeze_fingerprint).
            "threshold_rule": _combined_threshold_key(combined_cfg_input),
        }
        pair_cfg_input = combined_cfg_input.get("pair_groups")
        pair_expected: list[dict[str, Any]] = []
        if isinstance(pair_cfg_input, Sequence) and not isinstance(pair_cfg_input, (str, bytes)):
            for entry in pair_cfg_input:
                if not isinstance(entry, Mapping):
                    continue
                datasets_raw = entry.get("datasets")
                dataset_list: list[str] = []
                if isinstance(datasets_raw, Sequence) and not isinstance(datasets_raw, (str, bytes)):
                    dataset_list = [str(Path(ds).expanduser()) for ds in datasets_raw]
                elif datasets_raw:
                    dataset_list = [str(Path(str(datasets_raw)).expanduser())]
                pair_expected.append(
                    {
                        "name": str(entry.get("name") or entry.get("label") or ""),
                        "datasets": sorted(dataset_list),
                        "out_dir": str(entry.get("out_dir") or ""),
                        "trial_type_filter": entry.get("trial_type_filter"),
                    }
                )
        combined_expected["pair_groups"] = pair_expected
        if figures_only:
            # In --figures-only mode, trust the existing wide CSVs / matrix
            # npy unconditionally. The file-manifest cache check is too
            # sensitive (it re-runs combined on any source file mtime change,
            # even if no new flies were added) — defeats the purpose of the
            # mode. If the wide CSV is missing, the figure-output steps will
            # raise with a clear error.
            skip_combined = True
            print(
                "[analysis] combined → SKIPPED (--figures-only: using existing wide CSVs / matrix)."
            )
        else:
            skip_combined = _should_skip_with_manifest(
                settings,
                category="combined",
                key="analysis",
                expected=combined_expected,
                force_flag=settings.force.combined,
            )
            if skip_combined and _plan_forces_combined(plan):
                skip_combined = False
                print("[FREEZE] combined → running despite its cache: datasets are being rebuilt "
                      "or lost duplicate rows.")
            if skip_combined:
                print(
                    "[analysis] combined analysis cached → skipping. Set force.combined=true to recompute."
                )
        if not skip_combined:
            deferred_secure_cleanup = _run_combined(
                combined_cfg_input,
                settings,
                folder_filter=folder_filter,
                defer_envelopes=defer_envelopes,
                defer_secure_cleanup=True,
                config_path=config_path,
            )
            combined_ran = True
            # Only write combined cache when running ALL folders; a single-folder
            # run is partial so the next full run must recompute.
            if not folder_filter:
                payload = dict(combined_expected)
                payload["version"] = STATE_VERSION
                # A forced check never reads this state: skip the manifest.
                _write_state(
                    settings, "combined", "analysis", payload,
                    build_manifest=not settings.force.combined,
                )
            else:
                LOGGER.info(
                    "Single-folder mode: skipping combined cache write so next full run recomputes."
                )
    else:
        _run_combined(
            None,
            settings,
            folder_filter=folder_filter,
            defer_envelopes=defer_envelopes,
            config_path=config_path,
        )
    # Load per-trial model scores (if the reactions step has run on a prior
    # invocation) so envelope plots can annotate each subplot with the score
    # in the upper-right corner. Empty registry on a fresh pipeline run.
    _load_model_scores_for_envelopes(settings)

    # -- envelope_visuals with cache skip --
    env_vis_cfg = analysis_cfg.get("envelope_visuals")
    if env_vis_cfg:
        ev_expected = _analysis_expected(env_vis_cfg)
        skip_ev = _should_skip(
            settings, "envelope_visuals", "analysis", ev_expected,
            force_flag=settings.force.envelope_visuals,
        )
        if skip_ev:
            print(
                "[analysis] envelope_visuals cached → skipping. "
                "Set force.envelope_visuals=true to recompute."
            )
        else:
            _run_envelope_visuals(env_vis_cfg, defer_envelopes=defer_envelopes, settings=settings)
            payload = dict(ev_expected, version=STATE_VERSION)
            _write_state(settings, "envelope_visuals", "analysis", payload)
    else:
        _run_envelope_visuals(None, defer_envelopes=defer_envelopes, settings=settings)

    # -- training with cache skip --
    training_cfg = analysis_cfg.get("training")
    if training_cfg:
        tr_expected = _analysis_expected(training_cfg)
        skip_tr = _should_skip(
            settings, "training", "analysis", tr_expected,
            force_flag=settings.force.training,
        )
        if skip_tr:
            print(
                "[analysis] training cached → skipping. "
                "Set force.training=true to recompute."
            )
        else:
            _run_training(training_cfg, settings=settings)
            payload = dict(tr_expected, version=STATE_VERSION)
            _write_state(settings, "training", "analysis", payload)
    else:
        _run_training(None, settings=settings)

    # -- light-only PER traces (per-fly figures, sorted by dataset) --
    _run_light_only_traces(analysis_cfg.get("light_only_traces"), settings=settings)

    _run_reactions(settings, config_path=config_path)

    # Reactions just wrote model_predictions.csv. Re-render every envelope-
    # trace site that fired earlier in the pipeline (Raw + RMS testing/training
    # under analysis.combined.combined_base.envelopes and analysis.combined.envelopes,
    # plus analysis.envelope_visuals.envelopes) so each trial subplot picks up
    # its Score annotation in the upper-right corner. Skipped if reactions did
    # not produce a predictions CSV.
    scores_csv = Path(getattr(settings.reaction_prediction, "output_csv", "") or "")
    if scores_csv.exists():
        _load_model_scores_for_envelopes(settings)

        combined_cfg = analysis_cfg.get("combined") or {}
        raw_envelopes_cfg = (combined_cfg.get("combined_base") or {}).get("envelopes")
        distance_envelopes_cfg = (combined_cfg.get("distance_base") or {}).get("envelopes")
        rms_envelopes_cfg = combined_cfg.get("envelopes")

        if raw_envelopes_cfg:
            _rerender_envelope_block_with_scores(
                raw_envelopes_cfg, "Raw envelopes (combined_base)", settings=settings
            )
        if distance_envelopes_cfg:
            _rerender_envelope_block_with_scores(
                distance_envelopes_cfg, "Distance envelopes (distance_base)", settings=settings
            )
        if rms_envelopes_cfg:
            _rerender_envelope_block_with_scores(
                rms_envelopes_cfg, "RMS envelopes (combined)", settings=settings
            )
        if env_vis_cfg:
            vis_envelopes_cfg = env_vis_cfg.get("envelopes")
            if vis_envelopes_cfg:
                _rerender_envelope_block_with_scores(
                    vis_envelopes_cfg, "envelope_visuals.envelopes", settings=settings
                )
            payload = dict(ev_expected, version=STATE_VERSION)
            _write_state(settings, "envelope_visuals", "analysis", payload)

    # -- dataset_means with cache skip --
    dm_expected = _dataset_means_expected(analysis_cfg)
    skip_dm = _should_skip(
        settings, "dataset_means", "analysis", dm_expected,
        force_flag=settings.force.dataset_means,
    )
    if skip_dm:
        print(
            "[analysis] dataset_means cached → skipping. "
            "Set force.dataset_means=true to recompute."
        )
    else:
        # NOT no_overwrite=True: that froze every figure on first write, so a
        # PNG could never be redrawn by any run while its sidecar refreshed
        # every time. The step is already gated by _should_skip /
        # force.dataset_means, so redrawing here costs no extra work.
        _run_dataset_means(config_path, no_overwrite=False, settings=settings)
        _run_dataset_means_training(config_path, no_overwrite=False, settings=settings)
        payload = dict(dm_expected, version=STATE_VERSION)
        _write_state(settings, "dataset_means", "analysis", payload)

    # -- per-odor trained-vs-control mean traces, one set per dataset pair --
    mt_expected = _dataset_mean_traces_expected(
        analysis_cfg, settings, config_path=config_path
    )
    skip_mt = _should_skip(
        settings, "dataset_mean_traces", "analysis", mt_expected,
        force_flag=settings.force.dataset_mean_traces,
    )
    if skip_mt:
        print(
            "[analysis] dataset_mean_traces cached → skipping. "
            "Set force.dataset_mean_traces=true to recompute."
        )
    else:
        _run_dataset_mean_traces(analysis_cfg, settings, config_path=config_path)
        payload = dict(mt_expected, version=STATE_VERSION)
        _write_state(settings, "dataset_mean_traces", "analysis", payload)

    # -- naive vs trained vs control; needs the predictions reactions wrote --
    if scores_csv.exists():
        nvt_expected = _naive_vs_trained_expected(
            analysis_cfg, settings, config_path=config_path
        )
        skip_nvt = _should_skip(
            settings, "naive_vs_trained", "analysis", nvt_expected,
            force_flag=settings.force.naive_vs_trained,
        )
        if skip_nvt:
            print(
                "[analysis] naive_vs_trained cached → skipping. "
                "Set force.naive_vs_trained=true to recompute."
            )
        else:
            _run_naive_vs_trained(analysis_cfg, settings, config_path=config_path)
            payload = dict(nvt_expected, version=STATE_VERSION)
            _write_state(settings, "naive_vs_trained", "analysis", payload)
    else:
        LOGGER.info(
            "[analysis] naive_vs_trained skipped; no predictions CSV at %s",
            scores_csv,
        )

    # -- conditioning-trial scores; must precede every figure that reads them --
    _run_training_scores(analysis_cfg, settings, config_path=config_path)

    # -- pre-test-baselined trial score bars; same predictions CSV --
    if scores_csv.exists():
        _run_pretest_score_bars(analysis_cfg, settings, config_path=config_path)
    else:
        LOGGER.info(
            "[analysis] pretest_score_bars skipped; no predictions CSV at %s",
            scores_csv,
        )

    # -- pre-test vs post-test, paired within fly; same predictions CSV --
    if scores_csv.exists():
        pvt_expected = _pretest_vs_test_expected(
            analysis_cfg, settings, config_path=config_path
        )
        skip_pvt = _should_skip(
            settings, "pretest_vs_test", "analysis", pvt_expected,
            force_flag=getattr(settings.force, "pretest_vs_test", False),
        )
        if skip_pvt:
            print(
                "[analysis] pretest_vs_test cached → skipping. "
                "Set force.pretest_vs_test=true to recompute."
            )
        else:
            _run_pretest_vs_test(analysis_cfg, settings, config_path=config_path)
            payload = dict(pvt_expected, version=STATE_VERSION)
            _write_state(settings, "pretest_vs_test", "analysis", payload)
    else:
        LOGGER.info(
            "[analysis] pretest_vs_test skipped; no predictions CSV at %s",
            scores_csv,
        )

    # -- per-cohort PER rasters + training-AUC; needs the same predictions --
    if scores_csv.exists():
        cf_expected = _cohort_figures_expected(
            analysis_cfg, settings, config_path=config_path
        )
        skip_cf = _should_skip(
            settings, "cohort_figures", "analysis", cf_expected,
            force_flag=settings.force.cohort_figures,
        )
        if skip_cf:
            print(
                "[analysis] cohort_figures cached → skipping. "
                "Set force.cohort_figures=true to recompute."
            )
        else:
            _run_cohort_figures(analysis_cfg, settings, config_path=config_path)
            payload = dict(cf_expected, version=STATE_VERSION)
            _write_state(settings, "cohort_figures", "analysis", payload)
    else:
        LOGGER.info(
            "[analysis] cohort_figures skipped; no predictions CSV at %s",
            scores_csv,
        )

    # Data and figures are done: record every dataset rebuilt tonight as frozen
    # (with the inputs seen when the run started). A run that died before this
    # point left them ``thawing``, and the next run rebuilds them.
    if plan is not None:
        committed = plan.commit(combined_ran=combined_ran)
        for line in committed:
            print(line)
        status = plan.write_status(committed)
        _RUN_SUMMARY = plan.short_summary()
        print(f"[FREEZE] {_RUN_SUMMARY} (status: {status})")

    # All figures are now rendered (combined, envelope_visuals, deferred
    # model-annotated re-renders, dataset_means). NOW copy local experiment data
    # to secured storage (and optionally delete local) — deferred to here so the
    # figure pipeline always reads from the still-present local data first.
    if deferred_secure_cleanup is not None:
        deferred_secure_cleanup()

    # Copy all analysis outputs to SMB at the end
    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("Starting batch copy of all analysis outputs to SMB...")
    LOGGER.info("=" * 70)
    _batch_copy_to_smb(data)
    LOGGER.info("=" * 70)
    LOGGER.info("✓ Batch copy complete!")
    LOGGER.info("=" * 70 + "\n")
    _run_data_secured_sync(data)


# Output folders are named for the cohort itself ("EB-24-1"). The hand-run sets
# under Results/Figures carry a "_mean_traces_new" tail; set `dir_suffix` in the
# config to reproduce that naming.
_MEAN_TRACE_DIR_SUFFIX = ""


def _mean_trace_cohorts(datasets: Sequence[str]) -> list[tuple[str, str, str]]:
    """``(train_dataset, control_dataset, stem)`` for every two-armed dataset.

    A trained-vs-control mean trace needs both arms, and the config already
    names them by convention (``EB-Training-24-1`` / ``EB-Control-24-1``), so
    the pipeline can pair them itself instead of carrying a second hand-kept
    list. ``stem`` drops the arm token -- ``EB-24-1`` -- which is the name the
    published figure sets already use.

    A dataset whose partner is absent is skipped: half a pair cannot draw the
    figure. Config order is kept and duplicates collapse.
    """
    names = {str(d).strip() for d in datasets}
    pairs: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for raw in datasets:
        train = str(raw).strip()
        if "-Training-" not in train or train in seen:
            continue
        control = train.replace("-Training-", "-Control-", 1)
        if control not in names:
            continue
        seen.add(train)
        pairs.append((train, control, train.replace("-Training-", "-", 1)))
    return pairs


def _as_genotype_list(value: Any) -> list[str]:
    """Normalise a config `genotype:` entry (absent / one name / a list)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(v).strip() for v in value if str(v).strip()]


def _dataset_mean_traces_commands(
    analysis_cfg: Mapping[str, Any] | None,
    settings: Any,
    *,
    python_exec: str,
    config_path: Path | None,
) -> list[list[str]]:
    """One ``dataset_mean_traces_tvc.py`` command per trained/control pair.

    Built as a pure function for the same reason as ``_pubfig_commands``: the
    caller needs a wide envelope table and a whole figure tree before it will
    run, so the command shape is not otherwise testable.

    Returns ``[]`` when nothing is configured -- an unconfigured pipeline must
    not start rendering figures.
    """
    cfg = (analysis_cfg or {}).get("dataset_mean_traces") or {}
    if not cfg or not bool(cfg.get("enabled", True)):
        return []

    frozen_figures = set(_figure_frozen_datasets(settings))

    wide_csv = str(cfg.get("wide_csv", "") or "")
    out_root = str(cfg.get("out_root", "") or "")
    if not wide_csv or not out_root:
        print(
            "[analysis] dataset_mean_traces skipped; both wide_csv and out_root "
            "are required."
        )
        return []

    script = REPO_ROOT / "scripts" / "analysis" / "dataset_mean_traces_tvc.py"
    suffix = str(cfg.get("dir_suffix", _MEAN_TRACE_DIR_SUFFIX))
    flagged_csv = str(
        cfg.get("flagged_csv") or getattr(settings, "flagged_flies_csv", "") or ""
    )
    # One genotype per figure -- fbpipe.utils.fly_type exists so that different
    # genotypes are never averaged together, and the wide table pools them.
    block_genotypes = _as_genotype_list(cfg.get("genotype"))
    fps = float(cfg.get("fps", getattr(settings, "fps_default", 40.0)))
    odor_on_s = float(cfg.get("odor_on_s", getattr(settings, "odor_on_s", 30.0)))
    odor_off_s = float(cfg.get("odor_off_s", getattr(settings, "odor_off_s", 60.0)))

    # Explicit cohorts win: they are the escape hatch for pairs the naming
    # convention cannot express (a batch slice, a differently-named control).
    # Concentration series first: a dataset served by one has no control arm,
    # so it must not also be auto-paired below (the pairing would fail every
    # run against an empty control).
    conc_commands: list[list[str]] = []
    conc_datasets: set[str] = set()
    conc_script = REPO_ROOT / "scripts" / "analysis" / "randompanel_conc_traces.py"
    for raw in cfg.get("conc_series") or ():
        if not isinstance(raw, Mapping):
            continue
        members = raw.get("datasets") or {}
        if not isinstance(members, Mapping) or not members:
            print(
                "[analysis] dataset_mean_traces: conc_series needs a "
                f"`datasets: {{name: conc}}` mapping, got {raw!r}; skipping."
            )
            continue
        out_dir = str(raw.get("out_dir", "") or "conc_series")
        out_path = Path(out_dir)
        if not out_path.is_absolute():
            out_path = Path(out_root) / out_dir
        cmd = [
            str(Path(python_exec).expanduser()),
            str(conc_script),
            "--wide-csv", wide_csv,
            "--out-dir", str(out_path),
            "--fps", str(fps),
            "--odor-on-s", str(odor_on_s),
            "--odor-off-s", str(odor_off_s),
            "--protocol", str(getattr(settings, "protocol", "v2")),
        ]
        for name, conc in members.items():
            cmd.extend(["--dataset", f"{str(name).strip()}={float(conc):g}"])
            conc_datasets.add(str(name).strip())
        if members and all(str(n).strip() in frozen_figures for n in members):
            # Every member frozen -> the whole series is final. One live member
            # still redraws it: the figure IS the comparison between them.
            continue
        for genotype in (_as_genotype_list(raw.get("genotype")) or block_genotypes):
            cmd.extend(["--genotype", genotype])
        if flagged_csv:
            cmd.extend(["--flagged-flies-csv", flagged_csv])
        if config_path is not None:
            cmd.extend(["--config", str(config_path)])
        conc_commands.append(cmd)

    raw_cohorts = cfg.get("cohorts")
    if raw_cohorts:
        entries: list[tuple[str, str, str, int | None]] = []
        for raw in raw_cohorts:
            if not isinstance(raw, Mapping):
                continue
            train = str(raw.get("train_dataset", "") or "").strip()
            control = str(raw.get("control_dataset", "") or "").strip()
            if not train or not control:
                print(
                    "[analysis] dataset_mean_traces: cohort needs both "
                    f"train_dataset and control_dataset, got {raw!r}; skipping."
                )
                continue
            stem = train.replace("-Training-", "-", 1)
            out_dir = str(raw.get("out_dir", "") or f"{stem}{suffix}")
            batch = raw.get("batch")
            entries.append(
                (train, control, out_dir, None if batch is None else int(batch))
            )
    else:
        entries = [
            (train, control, f"{stem}{suffix}", None)
            for train, control, stem in _mean_trace_cohorts(
                getattr(settings, "datasets", ()) or ()
            )
            if train not in conc_datasets and control not in conc_datasets
        ]

    commands: list[list[str]] = []

    # Pre-test vs post-training: both arms are the SAME flies, so there is no
    # control cohort and no trained/control pairing. Emitted alongside the
    # trained-vs-control commands rather than as a separate step, so the two
    # sets always come from one wide table and one flagged-flies rule.
    pretest_wide = str(cfg.get("pretest_wide_csv", "") or "")
    phase_cohorts = [
        str(d).strip() for d in (cfg.get("phase_cohorts") or [])
        if str(d).strip() and str(d).strip() not in frozen_figures
    ]
    if pretest_wide and phase_cohorts:
        for dataset in phase_cohorts:
            phase_out = Path(out_root) / dataset / "Pre-Test_vs_Post-Training"
            phase_cmd = [
                str(Path(python_exec).expanduser()),
                str(script),
                "--wide-csv", wide_csv,
                "--pretest-wide-csv", pretest_wide,
                "--train-dataset", dataset,
                "--out-dir", str(phase_out),
                "--fps", str(fps),
                "--odor-on-s", str(odor_on_s),
            ]
            if flagged_csv:
                phase_cmd += ["--flagged-flies-csv", flagged_csv]
            if config_path is not None:
                phase_cmd += ["--config", str(config_path)]
            commands.append(phase_cmd)

    # --with-naive also draws a naive arm: the random panel run at each odor's
    # concentration (pubfig_naive_vs_trained.NAIVE_BY_CONC). It contributes to
    # the figure like the two arms do.
    with_naive = bool(cfg.get("with_naive", True))
    naive_panels: tuple[str, ...] = ()
    if with_naive:
        from scripts.analysis.pubfig_naive_vs_trained import NAIVE_BY_CONC

        naive_panels = tuple(NAIVE_BY_CONC.values())
    for train, control, out_dir, batch in entries:
        if all(arm in frozen_figures for arm in (train, control, *naive_panels)):
            # Every contributor frozen for figures -> the pair is final. A frozen
            # arm paired with a live one still redraws, or new flies in the live
            # arm would never appear in the comparison -- and so does a frozen
            # pair whose naive panel changed (auto-freeze dedupes those rows).
            continue
        out_path = Path(out_dir)
        if not out_path.is_absolute():
            out_path = Path(out_root) / out_dir
        cmd = [
            str(Path(python_exec).expanduser()),
            str(script),
            "--wide-csv", wide_csv,
            "--train-dataset", train,
            "--control-dataset", control,
            "--out-dir", str(out_path),
            "--fps", str(fps),
            "--odor-on-s", str(odor_on_s),
            "--odor-off-s", str(odor_off_s),
            "--protocol", str(getattr(settings, "protocol", "v2")),
        ]
        if flagged_csv:
            cmd.extend(["--flagged-flies-csv", flagged_csv])
        if batch is not None:
            cmd.extend(["--batch", str(batch)])
        for genotype in block_genotypes:
            cmd.extend(["--genotype", genotype])
        if bool(cfg.get("score_mean_trace", False)):
            cmd.append("--score-mean-trace")
        # The naive arm (black): the random panel run at each odor's own
        # concentration. Writes a second copy of every figure under
        # Trained_vs_Control_vs_Naive/, leaving the two-arm files untouched.
        if with_naive:
            cmd.append("--with-naive")
        if config_path is not None:
            # Carries dataset_overrides.odor_remap, so ACV reads
            # "Isoamyl Acetate (1%)" exactly as in the reaction matrices.
            # It is also what puts the concentration on the label, which is how
            # each odor finds its naive panel.
            cmd.extend(["--config", str(config_path)])
        commands.append(cmd)
    return commands + conc_commands


def _naive_vs_trained_command(
    analysis_cfg: Mapping[str, Any] | None,
    settings: Any,
    *,
    python_exec: str,
    config_path: Path | None,
) -> list[str] | None:
    """The ``pubfig_naive_vs_trained.py --sweep`` command, or None.

    Pure for the same reason as ``_pubfig_commands``: the caller needs a
    predictions CSV on disk before it will run. Returns None when the block is
    absent or there is no CSV to score -- an unconfigured pipeline must not
    start rendering figures.
    """
    cfg = (analysis_cfg or {}).get("naive_vs_trained") or {}
    if not cfg or not bool(cfg.get("enabled", True)):
        return None

    out_dir = str(cfg.get("out_dir", "") or "")
    if not out_dir:
        print("[analysis] naive_vs_trained skipped; out_dir is required.")
        return None

    predictions_csv = str(
        cfg.get("predictions_csv")
        or getattr(getattr(settings, "reaction_prediction", None), "output_csv", "")
        or ""
    )
    if not predictions_csv:
        print(
            "[analysis] naive_vs_trained skipped; no reaction_prediction.output_csv."
        )
        return None

    script = REPO_ROOT / "scripts" / "analysis" / "pubfig_naive_vs_trained.py"
    cmd = [
        str(Path(python_exec).expanduser()),
        str(script),
        "--sweep",
        "--figures-dir", out_dir,
        "--predictions-csv", predictions_csv,
    ]
    genotype = str(cfg.get("genotype", "") or "")
    if genotype:
        cmd.extend(["--genotype", genotype])
    correction = str(cfg.get("correction", "") or "")
    if correction:
        cmd.extend(["--correction", correction])
    cmd.append("--footnote" if bool(cfg.get("footnote", False)) else "--no-footnote")
    cmd.append("--trend-p" if bool(cfg.get("trend_p", True)) else "--no-trend-p")
    if config_path is not None:
        # Carries dataset_overrides.odor_remap: the concentration tags on the
        # labels are what pick each odor's naive panel. It also carries
        # freeze.figures -- the sweep skips cohorts whose three arms are all
        # frozen, so this run's --thaw goes with it.
        cmd.extend(["--config", str(config_path)])
        cmd.extend(_thaw_cli_args(settings))
    return cmd


#: The three per-cohort figure sets, as ``(script, mode, out_dir config key)``.
#: ``mode`` is None for the script that has no mode flag.
_COHORT_FIGURE_SETS = (
    ("binarized_per_rasters.py", "binary", "binarized_out_dir"),
    ("binarized_per_rasters.py", "graded", "graded_out_dir"),
    ("training_auc_vs_control_response.py", None, "training_auc_out_dir"),
)


def _cohort_figure_commands(
    analysis_cfg: Mapping[str, Any] | None,
    settings: Any,
    *,
    python_exec: str,
    config_path: Path | None,
) -> list[list[str]]:
    """The per-cohort raster / training-AUC commands, one per figure set.

    Pure, like ``_naive_vs_trained_command``: the caller checks for a predictions
    CSV on disk before running any of them. Returns ``[]`` when the block is
    absent, disabled, or names no cohorts -- an unconfigured pipeline must not
    start rendering figures.

    No ``--threshold-*`` flag is ever passed. The raster script resolves its
    :class:`ThresholdRule` from the ``--config`` file's own wide block, which is
    the point of the wiring: retuning the threshold in config moves the rasters,
    the AUC columns and the red line on the trace figures together. A flag here
    would silently pin the rasters to a stale value.
    """
    cfg = (analysis_cfg or {}).get("cohort_figures") or {}
    if not cfg or not bool(cfg.get("enabled", True)):
        return []

    frozen = set(_figure_frozen_datasets(settings))
    datasets = [
        str(d).strip() for d in (cfg.get("datasets") or [])
        if str(d).strip() and str(d).strip() not in frozen
    ]
    if not datasets:
        # Either nothing configured, or every configured cohort is frozen for
        # figures. Both mean "render nothing" -- a frozen cohort's rasters are
        # final and must not be restyled by a later threshold or palette.
        return []

    training_wide = str(cfg.get("training_wide_csv", "") or "")
    testing_wide = str(cfg.get("testing_wide_csv", "") or "")
    # Optional third phase for the *-Sensitivity-* cohorts. Raster-only: the
    # training-AUC script has no naive arm to plot.
    pretest_wide = str(cfg.get("pretest_wide_csv", "") or "")
    predictions_csv = str(
        cfg.get("predictions_csv")
        or getattr(getattr(settings, "reaction_prediction", None), "output_csv", "")
        or ""
    )
    if not training_wide or not testing_wide:
        print(
            "[analysis] cohort_figures skipped; training_wide_csv and "
            "testing_wide_csv are both required."
        )
        return []

    cmds: list[list[str]] = []
    for script_name, mode, out_key in _COHORT_FIGURE_SETS:
        out_dir = str(cfg.get(out_key, "") or "")
        if not out_dir:
            # One set with nowhere to write must not take the other two down.
            print(f"[analysis] cohort_figures: {out_key} unset; skipping that set.")
            continue
        script = REPO_ROOT / "scripts" / "analysis" / script_name
        cmd = [str(Path(python_exec).expanduser()), str(script)]

        if script_name == "training_auc_vs_control_response.py":
            cmd += ["--training-wide-csv", training_wide,
                    "--testing-wide-csv", testing_wide]
            if predictions_csv:
                cmd += ["--predictions-csv", predictions_csv]
        else:
            # The raster script spells the testing table --wide-csv.
            cmd += ["--training-wide-csv", training_wide,
                    "--wide-csv", testing_wide,
                    "--mode", str(mode),
                    # Fold-change of extension over each trial's own baseline,
                    # averaged across the fly's conditioning trials.
                    "--sort-by", str(cfg.get("sort_by") or "ratio")]
            if pretest_wide:
                cmd += ["--pretest-wide-csv", pretest_wide]
            if predictions_csv:
                cmd += ["--predictions-csv", predictions_csv]
            if config_path is not None:
                cmd += ["--config", str(config_path)]

        for dataset in datasets:
            cmd += ["--dataset", dataset]
        cmd += ["--out-dir", out_dir]
        cmds.append(cmd)

    # The trained arms of the same cohorts get the SAME three figure sets, on
    # the same row ordering, so a trained figure and its control partner can be
    # read side by side without mentally correcting for a different sort.
    trained = [
        str(d).strip()
        for d in (cfg.get("trained_raster_datasets") or [])
        if str(d).strip() and str(d).strip() not in frozen
    ]
    if trained:
        trained_sort = str(cfg.get("trained_sort_by") or cfg.get("sort_by") or "ratio")
        for script_name, mode, out_key in _COHORT_FIGURE_SETS:
            out_dir = str(cfg.get(out_key, "") or "")
            if not out_dir:
                continue
            script = REPO_ROOT / "scripts" / "analysis" / script_name
            cmd = [str(Path(python_exec).expanduser()), str(script)]
            if script_name == "training_auc_vs_control_response.py":
                # Valid for a trained arm too -- "does conditioning vigor
                # predict the test response" is well posed either way. Its
                # captions follow the arm (see that module's protocol_line), so
                # a trained figure does not claim the odor was unpaired.
                cmd += ["--training-wide-csv", training_wide,
                        "--testing-wide-csv", testing_wide]
                if predictions_csv:
                    cmd += ["--predictions-csv", predictions_csv]
            else:
                cmd += ["--training-wide-csv", training_wide,
                        "--wide-csv", testing_wide,
                        "--mode", str(mode),
                        "--sort-by", trained_sort]
                if pretest_wide:
                    cmd += ["--pretest-wide-csv", pretest_wide]
                if predictions_csv:
                    cmd += ["--predictions-csv", predictions_csv]
                if config_path is not None:
                    cmd += ["--config", str(config_path)]
            for dataset in trained:
                cmd += ["--dataset", dataset]
            cmd += ["--out-dir", out_dir]
            cmds.append(cmd)
    return cmds


def _cohort_figures_expected(
    analysis_cfg: Mapping[str, Any] | None,
    settings: Any,
    *,
    config_path: Path | None,
) -> dict[str, Any]:
    """Cache key: the wide tables' and predictions' mtimes plus every command.

    The commands carry the cohort list and the output directories, so editing
    either invalidates. The threshold rule is NOT in this key by name -- it
    arrives through the config file, and a config edit already rebuilds the wide
    tables, whose mtimes are here.
    """
    cmds = _cohort_figure_commands(
        analysis_cfg, settings, python_exec=sys.executable, config_path=config_path
    )
    cfg = (analysis_cfg or {}).get("cohort_figures") or {}
    watched = [
        cfg.get("training_wide_csv"),
        cfg.get("testing_wide_csv"),
        cfg.get("predictions_csv")
        or getattr(getattr(settings, "reaction_prediction", None), "output_csv", ""),
    ]
    return {
        "inputs": [
            _file_mtime_key(_resolve_path(str(w))) if w else None for w in watched
        ],
        "commands": sorted(" ".join(c) for c in cmds),
    }


#: The cross-cohort sidecar training_auc_vs_control_response.py rewrites from
#: the datasets of each run of it.
_TRAINING_AUC_SIDECAR = "training_auc_vs_control_response.json"


def _held_training_auc_entries(analysis_cfg, settings) -> tuple[Path | None, dict]:
    """``(sidecar, entries)``: what the training-AUC sidecar holds now for the
    cohorts held this run (configured, but frozen for figures)."""
    cfg = (analysis_cfg or {}).get("cohort_figures") or {}
    out_dir = str(cfg.get("training_auc_out_dir", "") or "")
    if not out_dir:
        return None, {}
    sidecar = Path(out_dir) / _TRAINING_AUC_SIDECAR
    configured = {
        str(d).strip()
        for d in [*(cfg.get("datasets") or []), *(cfg.get("trained_raster_datasets") or [])]
        if str(d).strip()
    }
    held = configured & set(_figure_frozen_datasets(settings))
    if not held or not sidecar.is_file():
        return sidecar, {}
    try:
        previous = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return sidecar, {}
    if not isinstance(previous, dict):
        return sidecar, {}
    return sidecar, {k: v for k, v in previous.items() if k in held}


def _keep_held_training_auc_entries(sidecar: Path | None, held: dict) -> None:
    """Put the held cohorts' entries back into the training-AUC sidecar.

    The script rewrites it from the datasets it drew; a held cohort is not
    drawn, and under auto-freeze most cohorts sit out most nights, so without
    this the sidecar would shrink to whatever was rebuilt.
    """
    if sidecar is None or not held or not sidecar.is_file():
        return
    try:
        current = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(current, dict):
        return
    missing = {k: v for k, v in held.items() if k not in current}
    if not missing:
        return
    sidecar.write_text(
        json.dumps({**current, **missing}, indent=2, default=float), encoding="utf-8"
    )
    LOGGER.info(
        "[analysis] cohort_figures: kept the training-AUC entries of %d held dataset(s)",
        len(missing),
    )


def _run_cohort_figures(
    analysis_cfg: Mapping[str, Any] | None,
    settings: Any,
    *,
    config_path: Path | None,
) -> None:
    """Binarised + graded PER rasters and training-AUC vs testing response."""
    python_exec = getattr(
        getattr(settings, "reaction_prediction", None), "python", ""
    ) or sys.executable
    cmds = _cohort_figure_commands(
        analysis_cfg, settings, python_exec=python_exec, config_path=config_path
    )
    if not cmds:
        LOGGER.info("[analysis] cohort_figures not configured; skipping.")
        return

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO_ROOT / "src"), env.get("PYTHONPATH")])
    )
    auc_sidecar, held_auc = _held_training_auc_entries(analysis_cfg, settings)
    for cmd in cmds:
        LOGGER.info("[analysis] cohort_figures → %s", " ".join(cmd))
        result = subprocess.run(cmd, env=env, capture_output=False)
        if result.returncode != 0:
            # One cohort set failing must not abort the remaining figure sets or
            # the SMB copy that follows -- same stance as naive_vs_trained.
            LOGGER.warning(
                "[analysis] cohort_figures command exited with code %d: %s",
                result.returncode, " ".join(cmd),
            )
    _keep_held_training_auc_entries(auc_sidecar, held_auc)
    LOGGER.info("[analysis] cohort_figures complete.")


def _training_scores_command(
    analysis_cfg: Mapping[str, Any] | None,
    settings: Any,
    *,
    python_exec: str,
    config_path: Path | None,
) -> list[str] | None:
    """Score the CONDITIONING trials into their own predictions sidecar.

    ``predict_reactions`` scores only testing and pretest trials, so nothing in
    the pipeline produced training-trial scores: the sidecar existed purely as
    a side-effect of hand-running a figure script, and was never refreshed as
    new cohorts landed. A cohort missing from it renders as a confident ``n=0``
    bar rather than an error, so it is made a first-class pipeline artefact.
    """
    cfg = (analysis_cfg or {}).get("training_scores") or {}
    if not cfg or not bool(cfg.get("enabled", True)):
        return None

    training_wide = str(cfg.get("training_wide_csv", "") or "")
    output_csv = str(cfg.get("output_csv", "") or "")
    if not training_wide or not output_csv:
        print(
            "[analysis] training_scores skipped; training_wide_csv and "
            "output_csv are both required."
        )
        return None

    script = REPO_ROOT / "scripts" / "analysis" / "naive_vs_trial_score_bars.py"
    cmd = [
        str(Path(python_exec).expanduser()),
        str(script),
        "--score-training-only",
        "--training-wide-csv", training_wide,
        "--training-predictions-csv", output_csv,
    ]
    model_path = str(cfg.get("model_path", "") or "")
    if model_path:
        cmd += ["--model-path", model_path]
    # Always rebuild: the freshness check inside the script compares mtimes, but
    # the pipeline has just rewritten the wide table, so this is the moment the
    # rebuild is meant to happen.
    cmd.append("--rescore")
    return cmd


def _run_training_scores(
    analysis_cfg: Mapping[str, Any] | None,
    settings: Any,
    *,
    config_path: Path | None,
) -> None:
    python_exec = getattr(
        getattr(settings, "reaction_prediction", None), "python", ""
    ) or sys.executable
    cmd = _training_scores_command(
        analysis_cfg, settings, python_exec=python_exec, config_path=config_path
    )
    if cmd is None:
        LOGGER.info("[analysis] training_scores not configured; skipping.")
        return
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO_ROOT / "src"), env.get("PYTHONPATH")])
    )
    LOGGER.info("[analysis] training_scores → %s", " ".join(cmd))
    result = subprocess.run(cmd, env=env, capture_output=False)
    if result.returncode != 0:
        LOGGER.warning(
            "[analysis] training_scores exited with code %d", result.returncode
        )


def _pretest_score_bars_command(
    analysis_cfg: Mapping[str, Any] | None,
    settings: Any,
    *,
    python_exec: str,
    config_path: Path | None,
) -> list[str] | None:
    """``naive_vs_trial_score_bars.py --pretest-baseline``, or None.

    The *-Sensitivity-* cohorts baseline each trial against the fly's OWN
    pre-test panel instead of a separate concentration-matched cohort. Wired
    into the pipeline because the hand-run original is exactly the kind of
    figure set that goes stale when a run rewrites model_predictions.csv.
    """
    cfg = (analysis_cfg or {}).get("pretest_score_bars") or {}
    if not cfg or not bool(cfg.get("enabled", True)):
        return None

    out_dir = str(cfg.get("out_dir", "") or "")
    if not out_dir:
        print("[analysis] pretest_score_bars skipped; out_dir is required.")
        return None

    datasets = [str(d).strip() for d in (cfg.get("datasets") or []) if str(d).strip()]
    if not datasets:
        return None

    predictions_csv = str(
        cfg.get("predictions_csv")
        or getattr(getattr(settings, "reaction_prediction", None), "output_csv", "")
        or ""
    )
    script = REPO_ROOT / "scripts" / "analysis" / "naive_vs_trial_score_bars.py"
    cmd = [
        str(Path(python_exec).expanduser()),
        str(script),
        "--pretest-baseline",
        "--out-dir", out_dir,
    ]
    if predictions_csv:
        cmd += ["--predictions-csv", predictions_csv]
    training_wide = str(cfg.get("training_wide_csv", "") or "")
    if training_wide:
        cmd += ["--training-wide-csv", training_wide]
    for dataset in datasets:
        cmd += ["--dataset", dataset]
    return cmd


def _run_pretest_score_bars(
    analysis_cfg: Mapping[str, Any] | None,
    settings: Any,
    *,
    config_path: Path | None,
) -> None:
    python_exec = getattr(
        getattr(settings, "reaction_prediction", None), "python", ""
    ) or sys.executable
    cmd = _pretest_score_bars_command(
        analysis_cfg, settings, python_exec=python_exec, config_path=config_path
    )
    if cmd is None:
        LOGGER.info("[analysis] pretest_score_bars not configured; skipping.")
        return
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO_ROOT / "src"), env.get("PYTHONPATH")])
    )
    LOGGER.info("[analysis] pretest_score_bars → %s", " ".join(cmd))
    result = subprocess.run(cmd, env=env, capture_output=False)
    if result.returncode != 0:
        LOGGER.warning(
            "[analysis] pretest_score_bars exited with code %d", result.returncode
        )


def _pretest_vs_test_command(
    analysis_cfg: Mapping[str, Any] | None,
    settings: Any,
    *,
    python_exec: str,
    config_path: Path | None,
) -> list[str] | None:
    """The ``pretest_vs_test_comparison.py`` command, or None when unconfigured.

    Pure, like ``_naive_vs_trained_command``: the caller checks for a
    predictions CSV on disk before running it.
    """
    cfg = (analysis_cfg or {}).get("pretest_vs_test") or {}
    if not cfg or not bool(cfg.get("enabled", True)):
        return None

    out_dir = str(cfg.get("out_dir", "") or "")
    if not out_dir:
        print("[analysis] pretest_vs_test skipped; out_dir is required.")
        return None

    predictions_csv = str(
        cfg.get("predictions_csv")
        or getattr(getattr(settings, "reaction_prediction", None), "output_csv", "")
        or ""
    )
    if not predictions_csv:
        print("[analysis] pretest_vs_test skipped; no reaction_prediction.output_csv.")
        return None

    script = REPO_ROOT / "scripts" / "analysis" / "pretest_vs_test_comparison.py"
    cmd = [
        str(Path(python_exec).expanduser()),
        str(script),
        "--predictions-csv", predictions_csv,
        "--out-root", out_dir,
    ]
    cohorts = [str(name) for name in (cfg.get("cohorts") or []) if str(name).strip()]
    if cohorts:
        cmd.append("--cohorts")
        cmd.extend(cohorts)
    if cfg.get("binary_threshold") is not None:
        cmd.extend(["--binary-threshold", str(int(cfg["binary_threshold"]))])
    # Falls back to the pipeline-wide table, like every other figure step: a
    # fly flagged there must not survive into these figures either.
    flagged = str(
        cfg.get("flagged_flies_csv")
        or getattr(settings, "flagged_flies_csv", "")
        or ""
    )
    if flagged:
        cmd.extend(["--flagged-flies-csv", flagged])
    return cmd


def _pretest_vs_test_expected(
    analysis_cfg: Mapping[str, Any] | None,
    settings: Any,
    *,
    config_path: Path | None,
) -> dict[str, Any]:
    """Cache key: the predictions CSV's mtime plus the exact command."""
    cmd = _pretest_vs_test_command(
        analysis_cfg, settings, python_exec=sys.executable, config_path=config_path
    )
    predictions = (
        _resolve_path(cmd[cmd.index("--predictions-csv") + 1]) if cmd else None
    )
    return {
        "predictions_mtime": _file_mtime_key(predictions),
        "command": " ".join(cmd) if cmd else "",
    }


def _run_pretest_vs_test(
    analysis_cfg: Mapping[str, Any] | None,
    settings: Any,
    *,
    config_path: Path | None,
) -> None:
    """Pre-test vs post-test PER, paired within fly, per sensitivity cohort."""
    python_exec = getattr(
        getattr(settings, "reaction_prediction", None), "python", ""
    ) or sys.executable
    cmd = _pretest_vs_test_command(
        analysis_cfg, settings, python_exec=python_exec, config_path=config_path
    )
    if cmd is None:
        LOGGER.info("[analysis] pretest_vs_test not configured; skipping.")
        return

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO_ROOT / "src"), env.get("PYTHONPATH")])
    )
    LOGGER.info("[analysis] pretest_vs_test → %s", " ".join(cmd))
    result = subprocess.run(cmd, env=env, capture_output=False)
    if result.returncode != 0:
        # A cohort with no pre-test panel is normal, not fatal: only the
        # *-Sensitivity-* datasets run one. Never abort the run over it.
        LOGGER.warning(
            "[analysis] pretest_vs_test exited with code %d", result.returncode
        )
    else:
        LOGGER.info("[analysis] pretest_vs_test complete.")


def _naive_vs_trained_expected(
    analysis_cfg: Mapping[str, Any] | None,
    settings: Any,
    *,
    config_path: Path | None,
) -> dict[str, Any]:
    """Cache key: the predictions CSV's mtime plus the exact command."""
    cmd = _naive_vs_trained_command(
        analysis_cfg, settings, python_exec=sys.executable, config_path=config_path
    )
    predictions = (
        _resolve_path(cmd[cmd.index("--predictions-csv") + 1]) if cmd else None
    )
    return {
        "predictions_mtime": _file_mtime_key(predictions),
        "command": " ".join(cmd) if cmd else "",
    }


def _run_naive_vs_trained(
    analysis_cfg: Mapping[str, Any] | None,
    settings: Any,
    *,
    config_path: Path | None,
) -> None:
    """Naive vs trained vs control, every cohort x odor x presentation."""
    python_exec = getattr(
        getattr(settings, "reaction_prediction", None), "python", ""
    ) or sys.executable
    cmd = _naive_vs_trained_command(
        analysis_cfg, settings, python_exec=python_exec, config_path=config_path
    )
    if cmd is None:
        LOGGER.info("[analysis] naive_vs_trained not configured; skipping.")
        return

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    extra_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [extra_path, env.get("PYTHONPATH")])
    )
    LOGGER.info("[analysis] naive_vs_trained → %s", " ".join(cmd))
    result = subprocess.run(cmd, env=env, capture_output=False)
    if result.returncode != 0:
        LOGGER.warning(
            "[analysis] naive_vs_trained exited with code %d", result.returncode
        )
    else:
        LOGGER.info("[analysis] naive_vs_trained complete.")


def _dataset_mean_traces_expected(
    analysis_cfg: Mapping[str, Any] | None,
    settings: Any,
    *,
    config_path: Path | None,
) -> dict[str, Any]:
    """Cache key: the wide table's mtime plus the exact commands to run."""
    cfg = (analysis_cfg or {}).get("dataset_mean_traces") or {}
    commands = _dataset_mean_traces_commands(
        analysis_cfg, settings, python_exec=sys.executable, config_path=config_path
    )
    return {
        "wide_csv_mtime": _file_mtime_key(_resolve_path(cfg.get("wide_csv"))),
        "commands": [" ".join(cmd) for cmd in commands],
    }


def _run_dataset_mean_traces(
    analysis_cfg: Mapping[str, Any] | None,
    settings: Any,
    *,
    config_path: Path | None,
) -> None:
    """Per-odor trained-vs-control mean traces, one figure set per dataset."""
    python_exec = getattr(
        getattr(settings, "reaction_prediction", None), "python", ""
    ) or sys.executable
    commands = _dataset_mean_traces_commands(
        analysis_cfg, settings, python_exec=python_exec, config_path=config_path
    )
    if not commands:
        LOGGER.info("[analysis] dataset_mean_traces not configured; skipping.")
        return

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    extra_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [extra_path, env.get("PYTHONPATH")])
    )

    for cmd in commands:
        LOGGER.info("[analysis] dataset_mean_traces → %s", " ".join(cmd))
        result = subprocess.run(cmd, env=env, capture_output=False)
        if result.returncode != 0:
            # One cohort with no usable traces (an arm that never ran testing,
            # a fully flagged batch) must not take the rest of the run down.
            LOGGER.warning(
                "[analysis] dataset_mean_traces for %s exited with code %d",
                cmd[cmd.index("--out-dir") + 1],
                result.returncode,
            )
    LOGGER.info("[analysis] dataset_mean_traces complete.")


def _run_dataset_means(
    config_path: Path, *, no_overwrite: bool = False, settings: Any = None
) -> None:
    """Generate dataset-level mean plots from the wide envelope CSV."""
    LOGGER.info("[analysis] Running dataset_means ...")
    repo_root = _find_repo_root(Path(__file__).resolve())
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "analysis" / "dataset_means.py"),
        "--config",
        str(config_path),
    ]
    if no_overwrite:
        cmd.append("--no-overwrite")
    # The script drops figure-frozen datasets itself (it reads the same config);
    # this only forwards THIS run's --thaw selection, which lives on settings.
    cmd.extend(_thaw_cli_args(settings))
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run(cmd, env=env, capture_output=False)
    if result.returncode != 0:
        LOGGER.warning("[analysis] dataset_means exited with code %d", result.returncode)
    else:
        LOGGER.info("[analysis] dataset_means complete.")


def _run_dataset_means_training(
    config_path: Path, *, no_overwrite: bool = False, settings: Any = None
) -> None:
    """Generate dataset-level mean plots for training odors."""
    LOGGER.info("[analysis] Running dataset_means_training ...")
    repo_root = _find_repo_root(Path(__file__).resolve())
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "analysis" / "dataset_means_training.py"),
        "--config",
        str(config_path),
    ]
    if no_overwrite:
        cmd.append("--no-overwrite")
    cmd.extend(_thaw_cli_args(settings))
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run(cmd, env=env, capture_output=False)
    if result.returncode != 0:
        LOGGER.warning("[analysis] dataset_means_training exited with code %d", result.returncode)
    else:
        LOGGER.info("[analysis] dataset_means_training complete.")


def _batch_copy_to_smb(raw_cfg: Dict[str, Any]) -> None:
    """Copy all analysis outputs to SMB at the end of the pipeline."""
    analysis_cfg = raw_cfg.get("analysis", {})

    # List of (local_path, smb_path) tuples to copy
    copies_to_make: List[Tuple[Path, str]] = []

    # Collect envelope_visuals outputs
    env_vis_cfg = analysis_cfg.get("envelope_visuals", {})
    if env_vis_cfg:
        matrices_cfg = env_vis_cfg.get("matrices")
        if matrices_cfg:
            out_dir = Path(matrices_cfg.get("out_dir", ""))
            smb_path = matrices_cfg.get("out_dir_smb")
            if out_dir.exists() and smb_path:
                copies_to_make.append((out_dir, smb_path))

        envelopes_cfg = env_vis_cfg.get("envelopes")
        if envelopes_cfg:
            entries = envelopes_cfg if isinstance(envelopes_cfg, list) else [envelopes_cfg]
            for env_cfg in entries:
                out_dir = Path(env_cfg.get("out_dir", ""))
                smb_path = env_cfg.get("out_dir_smb")
                if out_dir.exists() and smb_path:
                    copies_to_make.append((out_dir, smb_path))

    # Collect training outputs
    training_cfg = analysis_cfg.get("training", {})
    if training_cfg:
        envelopes_cfg = training_cfg.get("envelopes")
        if envelopes_cfg:
            entries = envelopes_cfg if isinstance(envelopes_cfg, list) else [envelopes_cfg]
            for env_cfg in entries:
                out_dir = Path(env_cfg.get("out_dir", ""))
                smb_path = env_cfg.get("out_dir_smb")
                if out_dir.exists() and smb_path:
                    copies_to_make.append((out_dir, smb_path))

    dataset_means_cfg = analysis_cfg.get("dataset_means", {})
    if dataset_means_cfg:
        out_dir = Path(dataset_means_cfg.get("out_dir", ""))
        smb_path = dataset_means_cfg.get("out_dir_smb")
        if out_dir.exists() and smb_path:
            copies_to_make.append((out_dir, smb_path))

    dataset_means_training_cfg = analysis_cfg.get("dataset_means_training", {})
    if dataset_means_training_cfg:
        out_dir = Path(dataset_means_training_cfg.get("out_dir", ""))
        smb_path = dataset_means_training_cfg.get("out_dir_smb")
        if out_dir.exists() and smb_path:
            copies_to_make.append((out_dir, smb_path))

    # Collect combined outputs
    combined_cfg = analysis_cfg.get("combined", {})
    if combined_cfg:
        matrices_cfg = combined_cfg.get("matrices")
        if matrices_cfg:
            out_dir = Path(matrices_cfg.get("out_dir", ""))
            smb_path = matrices_cfg.get("out_dir_smb")
            if out_dir.exists() and smb_path:
                copies_to_make.append((out_dir, smb_path))

        envelopes_list = combined_cfg.get("envelopes", [])
        if not isinstance(envelopes_list, list):
            envelopes_list = [envelopes_list]
        for env_cfg in envelopes_list:
            out_dir = Path(env_cfg.get("out_dir", ""))
            smb_path = env_cfg.get("out_dir_smb")
            if out_dir.exists() and smb_path:
                copies_to_make.append((out_dir, smb_path))

        overlay_cfg = combined_cfg.get("overlay")
        if overlay_cfg:
            out_dir = Path(overlay_cfg.get("out_dir", ""))
            smb_path = overlay_cfg.get("out_dir_smb")
            if out_dir.exists() and smb_path:
                copies_to_make.append((out_dir, smb_path))

        for base_key in ("combined_base", "distance_base"):
            base_cfg = combined_cfg.get(base_key, {})
            base_envelopes = base_cfg.get("envelopes", [])
            if not isinstance(base_envelopes, list):
                base_envelopes = [base_envelopes]
            for env_cfg in base_envelopes:
                out_dir = Path(env_cfg.get("out_dir", ""))
                smb_path = env_cfg.get("out_dir_smb")
                if out_dir.exists() and smb_path:
                    copies_to_make.append((out_dir, smb_path))

    # Collect reaction prediction outputs (CSV + matrix figures)
    reaction_cfg = raw_cfg.get("reaction_prediction", {})
    if reaction_cfg:
        output_csv = reaction_cfg.get("output_csv")
        output_csv_smb = reaction_cfg.get("output_csv_smb")
        if output_csv and output_csv_smb:
            csv_path = Path(output_csv)
            if csv_path.exists():
                copies_to_make.append((csv_path, output_csv_smb))

        matrix_cfg = reaction_cfg.get("matrix", {})
        if matrix_cfg:
            matrix_out_dir = Path(matrix_cfg.get("out_dir", ""))
            matrix_smb = matrix_cfg.get("out_dir_smb")
            if matrix_out_dir.exists() and matrix_smb:
                copies_to_make.append((matrix_out_dir, matrix_smb))

    # Execute all copies
    if copies_to_make:
        LOGGER.info(f"Found {len(copies_to_make)} items to copy to SMB")
        for local_path, smb_path in copies_to_make:
            try:
                item_name = local_path.name if local_path.is_file() else f"{local_path.name}/"
                LOGGER.info(f"Copying: {item_name} → {smb_path}")
                if copy_to_smb(local_path, smb_path):
                    LOGGER.info(f"✓ Success: {item_name}")
                else:
                    LOGGER.warning(f"⚠ Failed: {item_name}")
            except Exception as e:
                LOGGER.error(f"Error copying {local_path.name}: {e}")
    else:
        LOGGER.info("No items configured for SMB copy (out_dir_smb/output_csv_smb not set)")


def _run_data_secured_sync(raw_cfg: Dict[str, Any]) -> None:
    """Incrementally mirror Data-secured to the file server and Box."""
    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("Starting incremental Data-secured backup...")
    LOGGER.info("=" * 70)
    try:
        results = sync_data_secured(raw_cfg)
    except Exception as exc:
        LOGGER.error("Data-secured backup failed before transfer completed: %s", exc)
        LOGGER.info("=" * 70 + "\n")
        return

    if not results:
        LOGGER.info("Data-secured backup disabled or not configured.")
    else:
        failed = [name for name, ok in results.items() if not ok]
        if failed:
            LOGGER.warning("Data-secured backup finished with failures: %s", ", ".join(sorted(failed)))
        else:
            LOGGER.info("✓ Data-secured backup complete!")
    LOGGER.info("=" * 70 + "\n")


NTFY_TOPIC = "ramanlab-pipeline"

def ntfy_notify(title, message, priority="default", tags=""):
    try:
        data = message.encode("utf-8")
        req = urllib.request.Request(f"https://ntfy.sh/{NTFY_TOPIC}", data=data, method="POST")
        req.add_header("Title", title)
        req.add_header("Priority", priority)
        if tags:
            req.add_header("Tags", tags)
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass

if __name__ == "__main__" and "--freeze-status" in sys.argv[1:]:  # pragma: no cover
    main()  # a read-only report: no run happened, nothing to notify about
elif __name__ == "__main__":  # pragma: no cover
    try:
        main()
        _message = "Pipeline finished successfully."
        if _RUN_SUMMARY:
            _message += "\n" + _RUN_SUMMARY
        ntfy_notify("Pipeline Complete", _message, tags="white_check_mark")
    except RunLockTimeout as exc:
        ntfy_notify("Pipeline SKIPPED", f"Another pipeline run held the lock:\n{exc}", priority="high", tags="hourglass")
        raise SystemExit(75)
    except Exception as exc:
        ntfy_notify("Pipeline FAILED", f"Pipeline crashed with error:\n{exc}", priority="high", tags="x")
        raise
