"""Per-dataset freeze cache.

A frozen dataset keeps its rows in the wide CSV without being re-derived. This
module owns the cache only: fingerprinting, and saving/loading a dataset's row
slice. It deliberately knows nothing about ``build_wide_csv`` -- the splice
lives there, and the policy (which datasets are frozen) lives in the caller.

Layout, under the configured ``cache_dir``::

    <cache_dir>/frozen/<wide_block>/<dataset>/
        rows.parquet    # the dataset's slice of that block's wide CSV
        meta.json       # fingerprint + own_max_len

The cache is a DERIVED artifact. Deleting it is always safe: every frozen
dataset's raw data stays on disk, so a miss simply re-derives.

The ``wide_block`` component is a partitioning LABEL, not the correctness
mechanism. Correctness rests on the fingerprint (which includes
``measure_cols``). Keying by block is a safe over-partition: it may duplicate an
identical slice across two blocks that share parameters, but it can never serve
the wrong rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple, Optional

import pandas as pd

FREEZE_SCHEMA_VERSION = 1


class FrozenSlice(NamedTuple):
    """A dataset's cached wide rows plus its OWN maximum trace length.

    ``own_max_len`` is load-bearing. ``build_wide_csv`` pads *and truncates*
    every row to one global ``max_len``; if a frozen dataset's own length is not
    folded into that global max, its cached rows are silently chopped.
    """

    rows: pd.DataFrame
    own_max_len: int


def freeze_flags(
    cfg: Any,
    dataset: str,
    *,
    thawed: Iterable[str] = (),
    thaw_all: bool = False,
) -> tuple[bool, bool]:
    """Return ``(freeze_data, freeze_figures)`` for *dataset*, honoring thaw.

    ``thawed`` names datasets to treat as unfrozen for this run only;
    ``thaw_all`` unfreezes everything. Neither edits config.
    """
    if thaw_all or dataset in set(thawed):
        return (False, False)
    override = (getattr(cfg, "dataset_overrides", None) or {}).get(dataset)
    if override is None:
        return (False, False)
    return (bool(override.freeze_data), bool(override.freeze_figures))


def build_fingerprint(
    *,
    protocol: str,
    measure_cols: Iterable[str],
    fps_fallback: float,
    distance_limits: Optional[tuple[float, float]],
    non_reactive_threshold: Optional[float],
    low_max_threshold_px: float,
    use_per_trial_baseline: bool,
    trial_type_filter: Optional[str | bytes | Iterable[str]],
    threshold_rule: Any,
    override: Any,
    tracking: Any,
    freeze_folders_before: Any = None,
    freeze_folders_born_on_or_after: Any = None,
) -> dict:
    """Everything that determines a dataset's rows OTHER than its raw data.

    Raw data is deliberately absent: ``freeze.data`` never walks the root. Config
    IS checked, because it is already in memory and free -- without this, changing
    an analysis parameter while a dataset is frozen would leave one CSV silently
    mixing two parameterizations.

    ``figure_output_subdir`` is deliberately EXCLUDED: it routes figures and
    cannot change a row value, so it must not invalidate a data cache.
    """
    # build_wide_csv uses trial_type_filter to GATE which trials become rows at
    # all (envelope_combined.py:2661-2673 build the allow-set, :2703 skips any
    # trial whose type is not in it). Required, not defaulted -- same reasoning
    # as `tracking` below -- so a caller cannot silently omit a value-affecting
    # input. Without it: a dataset frozen while trial_type_filter=None (holds
    # every trial type) would still fingerprint-match after the config narrowed
    # the filter to e.g. "testing", so the stale cache would be trusted, spliced
    # in pre-filtered, and then re-cached under the narrower filter --
    # permanently evicting the dropped rows.
    #
    # Normalized identically to build_wide_csv's own ``trial_type_allow`` so the
    # recorded value reflects the actual filtering, and SORTED so multi-value
    # order (["testing","training"] vs ["training","testing"]) cannot cause
    # spurious drift -- order never changes which rows exist.
    if trial_type_filter is None:
        trial_type_norm: Optional[list[str]] = None
    else:
        if isinstance(trial_type_filter, (str, bytes)):
            _trial_type_allow = {str(trial_type_filter).strip().lower()}
        else:
            _trial_type_allow = {
                str(value).strip().lower()
                for value in trial_type_filter
                if str(value).strip()
            }
        trial_type_norm = sorted(_trial_type_allow) if _trial_type_allow else None

    return {
        "protocol": str(protocol),
        "measure_cols": [str(c) for c in measure_cols],
        "fps_fallback": float(fps_fallback),
        "distance_limits": (
            None if distance_limits is None else [float(x) for x in distance_limits]
        ),
        "non_reactive_threshold": (
            None if non_reactive_threshold is None else float(non_reactive_threshold)
        ),
        "low_max_threshold_px": float(low_max_threshold_px),
        "use_per_trial_baseline": bool(use_per_trial_baseline),
        # Gates which trials become rows at all -- see the comment above.
        "trial_type_filter": trial_type_norm,
        # theta determines the AUC-* columns, so a slice cached under one rule
        # must not be served under another -- that would put two thresholds in
        # one CSV with nothing to distinguish them. Stringified because the
        # fingerprint is JSON-serialised.
        "threshold_rule": None if threshold_rule is None else str(threshold_rule),
        "override": {
            "trial_type_override": getattr(override, "trial_type_override", None),
            "odor_on_s": getattr(override, "odor_on_s", None),
            "odor_off_s": getattr(override, "odor_off_s", None),
            "light_only": bool(getattr(override, "light_only", False)),
            "light_start_s": getattr(override, "light_start_s", None),
            "light_duration_s": getattr(override, "light_duration_s", None),
            "odor_remap": dict(getattr(override, "odor_remap", {}) or {}),
            # Folder freeze changes which rows carry frozen=True, so a slice
            # cached under one folder policy must not be served under another --
            # that would leave one CSV mixing two freeze policies, the same
            # failure mode trial_type_filter guards against above. Sorted so
            # list order (which never changes the outcome) cannot cause drift.
            "freeze_folders": sorted(
                f"{getattr(f, 'pattern', f)}@{getattr(f, 'before', None)}"
                for f in (getattr(override, "freeze_folders", ()) or ())
            ),
        },
        # The global date cutoff, for the same reason. Stringified so a
        # datetime.date survives the JSON round-trip load_slice compares over.
        "freeze_folders_before": (
            None if freeze_folders_before is None else str(freeze_folders_before)
        ),
        # The cohort cutoff, for the same reason: it decides which rows carry
        # frozen=True just as surely as the recording cutoff does.
        "freeze_folders_born_on_or_after": (
            None
            if freeze_folders_born_on_or_after is None
            else str(freeze_folders_born_on_or_after)
        ),
        # build_wide_csv reads settings.tracking internally (envelope_combined.py:2622)
        # and derives the tracking_missing_frames / tracking_pct_missing /
        # tracking_flagged columns from it (:3050, :3067-3068, :3111). Omitting it
        # would serve stale rows whose tracking flags used a different threshold.
        # Recorded unconditionally even when apply_missing_frame_check is False:
        # a needless rebuild is cheap and self-healing, a stale row is not.
        "tracking": {
            "apply_missing_frame_check": bool(
                getattr(tracking, "apply_missing_frame_check", True)
            ),
            "max_missing_frames_per_trial": getattr(
                tracking, "max_missing_frames_per_trial", None
            ),
            "max_missing_frames_pct_per_trial": getattr(
                tracking, "max_missing_frames_pct_per_trial", None
            ),
        },
    }


def _safe(name: str) -> str:
    """Filesystem-safe cache component. Dataset names carry dots and dashes
    (``Hex-Control-24-0.1``); only separators are unsafe."""
    return str(name).replace("/", "_").replace("\\", "_").strip() or "_"


def slice_dir(cache_dir: str | Path, wide_block: str, dataset: str) -> Path:
    return Path(cache_dir).expanduser() / "frozen" / _safe(wide_block) / _safe(dataset)


def own_max_len(rows: pd.DataFrame) -> int:
    """This dataset's own maximum trace length, in samples.

    ``trace_len`` is written as ``int(len(values))`` -- the UNPADDED length
    (``envelope_combined.py:3114``). But a row whose values exceeded the run's
    global max was truncated on write while ``trace_len`` kept the original
    figure, so clamp to the dir_val columns actually present. Over-reporting
    would pad the global max out to a width holding no data.
    """
    val_cols = [c for c in rows.columns if str(c).startswith("dir_val_")]
    present = len(val_cols)
    if "trace_len" not in rows.columns or rows.empty:
        return present
    tl = pd.to_numeric(rows["trace_len"], errors="coerce").max()
    if pd.isna(tl):
        return present
    return max(0, min(int(tl), present))


def save_slice(
    cache_dir: str | Path,
    wide_block: str,
    dataset: str,
    rows: pd.DataFrame,
    fingerprint: Mapping[str, Any],
) -> None:
    """Persist *rows* as *dataset*'s slice of *wide_block*.

    Stores the rows exactly as written, across ALL trial types -- the splice
    routes them to the main output or an extra trial export the same way live
    rows are routed.
    """
    target = slice_dir(cache_dir, wide_block, dataset)
    target.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(target / "rows.parquet", index=False)
    meta = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "dataset": str(dataset),
        "wide_block": str(wide_block),
        "own_max_len": own_max_len(rows),
        "row_count": int(len(rows)),
        "columns": [str(c) for c in rows.columns],
        "fingerprint": dict(fingerprint),
    }
    (target / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def load_slice(
    cache_dir: str | Path,
    wide_block: str,
    dataset: str,
    fingerprint: Mapping[str, Any],
) -> Optional[FrozenSlice]:
    """Return the cached slice, or ``None`` if it cannot be trusted.

    ``None`` means "re-derive this dataset": cache miss, fingerprint drift,
    schema-version bump, or an unreadable/corrupt cache. Callers implement
    auto-rebuild by simply omitting the dataset from ``frozen_slices``, which
    lets its root be walked normally.
    """
    target = slice_dir(cache_dir, wide_block, dataset)
    meta_path = target / "meta.json"
    rows_path = target / "rows.parquet"
    if not meta_path.is_file() or not rows_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return None
    if meta.get("schema_version") != FREEZE_SCHEMA_VERSION:
        return None
    if meta.get("fingerprint") != json.loads(json.dumps(dict(fingerprint))):
        return None
    try:
        rows = pd.read_parquet(rows_path)
    except Exception:
        return None
    return FrozenSlice(rows=rows, own_max_len=int(meta.get("own_max_len", 0)))


def drift_reason(
    cached: Mapping[str, Any], current: Mapping[str, Any]
) -> Optional[str]:
    """Name the first fingerprint field that differs, for a one-line log."""
    for key in sorted(set(cached) | set(current)):
        if cached.get(key) != current.get(key):
            return f"{key}: {cached.get(key)!r} -> {current.get(key)!r}"
    return None
