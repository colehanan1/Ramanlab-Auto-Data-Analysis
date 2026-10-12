"""Per-trial QC step: flag flies whose eye drifts off its frozen anchor.

YOLO freezes each fly's eye anchor at the start of a trial, and every distance,
angle and gate measures from it. The merged table also records where each
slot's eye is detected in every frame (``eye_{i}_live_x/_y``) and how far that
sits from the anchor (``eye_{i}_drift_px``). On a good trial that drift is
detection jitter (median 0.4-0.9 px, max 2.7 px); a fly that came loose or a
shifted mount moves the eye and keeps it moved. A fly is flagged when the
trailing median of its drift over ``eye_qc.sustain_s`` exceeds
``eye_qc.max_drift_px`` (a shift held for just over half that window is enough,
an unmatched frame counting as not over; a few-frame spike never is), or when
its eye goes unmatched in more than ``eye_qc.max_missing_frac`` of frames or for
longer than ``eye_qc.max_missing_run_s`` in a row. The run rule is for a fly
that comes loose late with a big jump: an eye beyond ``match_radius_px`` reads
as unmatched rather than as drift, and late in a trial that is too few frames
for the fraction to notice.

Report-only -- nothing is excluded. Every trial x fly is upserted into
``<report_dir>/eye_drift_qc.csv``; flagged flies are rewritten into
``eye_drift_qc_flagged.csv``, logged as ``[EYE-DRIFT][FLAG]`` lines, and sent as
one batched ntfy alert for flies *newly* flagged (re-running over unchanged
tables never re-notifies). A table written before per-frame eye tracking has no
``eye_*_live_x`` column; it gets one ``no_live_eye_columns`` row and is never
flagged.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..config import EyeQCSettings, Settings, get_main_directories, resolve_eye_qc_report_dir
from ..utils.frozen_folders import is_frozen_folder, iter_live_batch_dirs
from ..utils.notify import ntfy_notify
from ..utils.parallel import parallel_map
from ..utils.tables import read_schema_columns, read_table, resolve_existing
from ..utils.trial_metadata import load_trial_metadata

TRIAL_DIR_RE = re.compile(r"(pretest|training|testing)_(\d+)$", re.IGNORECASE)

MERGED_TABLE_SUFFIX = "_distances_merged"
REPORT_FILENAME = "eye_drift_qc.csv"
FLAGGED_FILENAME = "eye_drift_qc_flagged.csv"
QC_VERSION = 1
NTFY_MAX_LINES = 20
# Settings.fps_default, for when neither the timestamps nor the config give a sane rate.
DEFAULT_FPS = 40.0

_LIVE_X_RE = re.compile(r"^eye_\d+_live_x$")
_TRACK_ID_RE = re.compile(r"^eye_(\d+)_track_id$")
# The only merged-table columns the assessment reads.
_NEEDED_RE = re.compile(r"^(timestamp|eye_\d+_(track_id|x|y|drift_px))$")

#: One row per dataset x batch x trial x fly. ``fly`` is the batch folder, as
#: in the wide CSV; ``fly_number`` is slot + 1, as in the per-fly file names.
#: Trial-level statuses (no_live_eye_columns / no_eye_anchor / empty /
#: read_error) leave fly_number blank. No column is named like anything the
#: fly-exclusion loaders read (``flagged``, ``FLY-State``, ``tracking_flagged``).
REPORT_COLUMNS = (
    "qc_version",
    "dataset",
    "fly",
    "fly_number",
    "slot_idx",
    "eye_track_id",
    "phase",
    "trial_index",
    "trial_type",
    "odor",
    "trial_dir",
    "status",
    "n_frames",
    "eye_detected_frac",
    "eye_missing_frac",
    "longest_missing_run_s",
    "longest_missing_run_start_s",
    "drift_median_px",
    "drift_p95_px",
    "drift_max_px",
    "drift_frac_over_thresh",
    "drift_sustained_max_px",
    "longest_over_thresh_run_s",
    "first_over_thresh_s",
    "anchor_x",
    "anchor_y",
    "eye_qc_flag",
    "eye_qc_reason",
    "max_drift_px",
    "sustain_s",
    "max_missing_frac",
    "max_missing_run_s",
    "match_radius_px",
)
_INT_COLUMNS = ("qc_version", "fly_number", "slot_idx", "eye_track_id", "trial_index", "n_frames")


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


def _sanitized_eye_qc(eq: Optional[EyeQCSettings]) -> EyeQCSettings:
    """``eq`` with every unusable threshold replaced by its dataclass default.

    load_settings coerces types but never range-checks (a raise there would
    abort every pipeline subprocess), so odd values land here. Prints one
    warning per replaced value; never raises.
    """
    default = EyeQCSettings()
    if eq is None:
        eq = default
    checks = (
        ("max_drift_px", lambda v: math.isfinite(v) and v >= 0, "a non-negative number of px"),
        ("sustain_s", lambda v: math.isfinite(v) and v >= 0, "a non-negative number of seconds"),
        # Same rule yolo_infer applies to the radius it matches with.
        ("match_radius_px", lambda v: math.isfinite(v) and v > 0, "a positive number of px"),
        ("max_missing_frac", lambda v: 0.0 <= v <= 1.0, "a fraction between 0 and 1"),
        ("max_missing_run_s", lambda v: math.isfinite(v) and v >= 0, "a non-negative number of seconds"),
    )
    values = {}
    for name, usable, wanted in checks:
        raw = getattr(eq, name, getattr(default, name))
        try:
            value = math.nan if isinstance(raw, bool) else float(raw)
        except (TypeError, ValueError):
            value = math.nan
        if not usable(value):
            value = getattr(default, name)
            print(f"[EYE-DRIFT] WARNING: eye_qc.{name}={raw!r} is not {wanted}; using the default {value:g}.")
        values[name] = value
    if values["match_radius_px"] <= values["max_drift_px"]:
        print(
            f"[EYE-DRIFT] WARNING: eye_qc.match_radius_px ({values['match_radius_px']:g}) is not "
            f"above max_drift_px ({values['max_drift_px']:g}): an eye that drifts past the match "
            f"radius goes unmatched, so a loose fly reads as missing rather than as drift."
        )
    return EyeQCSettings(
        enabled=bool(getattr(eq, "enabled", default.enabled)),
        draw_live_eye=bool(getattr(eq, "draw_live_eye", default.draw_live_eye)),
        report_dir=getattr(eq, "report_dir", default.report_dir),
        **values,
    )


def _threshold_fields(eq: EyeQCSettings) -> dict:
    return {
        "max_drift_px": eq.max_drift_px,
        "sustain_s": eq.sustain_s,
        "max_missing_frac": eq.max_missing_frac,
        "max_missing_run_s": eq.max_missing_run_s,
        "match_radius_px": eq.match_radius_px,
    }


# ---------------------------------------------------------------------------
# Assessment (pure)
# ---------------------------------------------------------------------------


def _numeric(df: pd.DataFrame, column: str) -> np.ndarray:
    """``df[column]`` as floats; NaN where absent or not a number."""
    if column not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _rounded(value: float, ndigits: int) -> float:
    return round(float(value), ndigits) if math.isfinite(value) else math.nan


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    """(length, first index) of the longest run of consecutive True values:
    the earliest of equally long runs, and (0, -1) when there is none."""
    if not mask.any():
        return 0, -1
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    starts, lengths = edges[::2], edges[1::2] - edges[::2]
    k = int(np.argmax(lengths))
    return int(lengths[k]), int(starts[k])


def _seconds_in(ts: np.ndarray, k: int, fps: float) -> float:
    """Seconds from the first assessed frame to frame ``k``: by the timestamp
    column when both stamps are usable, else by counting frames at ``fps``."""
    if np.isfinite(ts[k]) and np.isfinite(ts[0]) and ts[k] >= ts[0]:
        return float(ts[k] - ts[0])
    return k / fps


def _trial_fps(ts: np.ndarray, fps_default: float) -> float:
    """1 / median positive timestamp step when that is a sane 1-240 fps, else
    ``fps_default`` (else DEFAULT_FPS)."""
    steps = np.diff(ts[np.isfinite(ts)])
    steps = steps[steps > 0]
    if steps.size:
        fps = 1.0 / float(np.median(steps))
        if math.isfinite(fps) and 1.0 <= fps <= 240.0:
            return fps
    try:
        fallback = float(fps_default)
    except (TypeError, ValueError):
        fallback = math.nan
    return fallback if math.isfinite(fallback) and fallback > 0 else DEFAULT_FPS


def _missing_too_often(row: dict) -> bool:
    return row.get("eye_missing_frac", math.nan) > row.get("max_missing_frac", math.inf)


def _missing_too_long(row: dict) -> bool:
    return row.get("longest_missing_run_s", math.nan) > row.get("max_missing_run_s", math.inf)


def _flag_kinds(row: dict) -> list[str]:
    """The flag rules a row's (rounded) metrics trip: "drift" and/or
    "eye_missing" (unmatched in too many frames, or for too long in a row)."""
    kinds = []
    if row.get("drift_sustained_max_px", math.nan) > row.get("max_drift_px", math.inf):
        kinds.append("drift")
    if _missing_too_often(row) or _missing_too_long(row):
        kinds.append("eye_missing")
    return kinds


def _missing_reason(row: dict, *, never_matched: bool) -> str:
    """Why a row is flagged eye_missing, in words; the run part names when the
    eye was lost, the likeliest cause being a jump past the match radius."""
    if never_matched:
        return "eye never matched (phantom slot?)"
    fraction = f"in {row['eye_missing_frac']:.0%} of frames"
    run = (
        f"{row['longest_missing_run_s']:.1f} s in a row from {row['longest_missing_run_start_s']:.1f} s "
        f"(moved > {row['match_radius_px']:g} px?)"
    )
    if not _missing_too_long(row):
        return f"eye unmatched {fraction}"
    if not _missing_too_often(row):
        return f"eye unmatched for {run}"
    return f"eye unmatched {fraction}, {run}"


def _assess_slot(
    idx: int,
    track_id: int,
    drift: np.ndarray,
    ts: np.ndarray,
    anchor_x: np.ndarray,
    anchor_y: np.ndarray,
    eq: EyeQCSettings,
    fps_default: float,
) -> dict:
    n = int(drift.size)
    fps = _trial_fps(ts, fps_default)
    threshold = eq.max_drift_px
    detected = np.isfinite(drift)
    seen = drift[detected]

    # Trailing rolling median over the whole window, an unmatched frame counting
    # as not over the threshold, and at least half the window must hold a
    # detection. A median only exceeds the threshold once about half the window
    # does, so a shift held for just over half of sustain_s flags and a
    # few-frame spike never does -- also right after a dropout: skipping the
    # unmatched frames instead would let 5 spike frames be half of a window
    # that kept only 10. Values before the window first spans sustain_s are
    # dropped: a 10-frame partial window at the trial start would let a
    # 5-frame spike read as a shift.
    window = max(1, int(round(eq.sustain_s * fps)))
    series = pd.Series(drift)
    enough = series.notna().rolling(window, min_periods=1).sum().to_numpy() >= max(1, math.ceil(window / 2))
    sustained = series.fillna(0.0).rolling(window, min_periods=1).median().to_numpy(dtype=float, copy=True)
    sustained[~enough] = np.nan
    sustained[: window - 1] = np.nan
    sustained = np.round(sustained, 3)  # flag from the values the report shows
    over = sustained > threshold
    first_over_s = _seconds_in(ts, int(np.argmax(over)), fps) if over.any() else math.nan
    missing_frames, missing_from = _longest_run(~detected)
    over_frames, _ = _longest_run(drift > threshold)

    anchors_x, anchors_y = anchor_x[np.isfinite(anchor_x)], anchor_y[np.isfinite(anchor_y)]
    row = {
        "fly_number": idx + 1,
        "slot_idx": idx,
        "eye_track_id": int(track_id),
        "status": "assessed",
        "n_frames": n,
        "eye_detected_frac": _rounded(detected.sum() / n, 4),
        "eye_missing_frac": _rounded((n - detected.sum()) / n, 4),
        "longest_missing_run_s": _rounded(missing_frames / fps, 3),
        # Timed like first_over_thresh_s; NaN when no frame went unmatched.
        "longest_missing_run_start_s": (
            _rounded(_seconds_in(ts, missing_from, fps), 3) if missing_frames else math.nan
        ),
        "drift_median_px": _rounded(np.median(seen), 3) if seen.size else math.nan,
        "drift_p95_px": _rounded(np.percentile(seen, 95), 3) if seen.size else math.nan,
        "drift_max_px": _rounded(seen.max(), 3) if seen.size else math.nan,
        "drift_frac_over_thresh": _rounded((seen > threshold).mean(), 4) if seen.size else math.nan,
        "drift_sustained_max_px": float(np.nanmax(sustained)) if np.isfinite(sustained).any() else math.nan,
        "longest_over_thresh_run_s": _rounded(over_frames / fps, 3),
        "first_over_thresh_s": _rounded(first_over_s, 3),
        "anchor_x": float(anchors_x[-1]) if anchors_x.size else math.nan,
        "anchor_y": float(anchors_y[-1]) if anchors_y.size else math.nan,
        **_threshold_fields(eq),
    }

    kinds = _flag_kinds(row)
    reasons = []
    if "drift" in kinds:
        # "median over", never "sustained": a shift held for just over half the
        # window already lifts the median past the threshold.
        reasons.append(
            f"drift > {threshold:.1f} px (median over {eq.sustain_s:g} s) from "
            f"{row['first_over_thresh_s']:.1f} s (max {row['drift_sustained_max_px']:.1f} px)"
        )
    if "eye_missing" in kinds:
        # The first frame does not count: yolo_infer matches the frame an anchor
        # was made from against that anchor, so even a phantom one (a false
        # positive the fly-count scan took for an eye) matches itself there once.
        reasons.append(_missing_reason(row, never_matched=not detected[1:].any()))
    row["eye_qc_flag"] = bool(kinds)
    row["eye_qc_reason"] = "; ".join(reasons)
    return row


def assess_trial(df: pd.DataFrame, eq: EyeQCSettings, fps_default: float) -> list[dict]:
    """QC metrics for one merged table: one dict per slot that has an eye anchor.

    Only frames whose ``eye_{i}_track_id`` equals the slot's FINAL track id
    count. Until the anchors confirm, a slot can hold another partial anchor
    (they are re-sorted top-to-bottom at confirmation), and frames before any
    anchor have none. ``eq`` is used as given; main() sanitizes it once.
    """
    timestamps = _numeric(df, "timestamp")
    slots = sorted(int(m.group(1)) for m in map(_TRACK_ID_RE.match, df.columns) if m)
    rows = []
    for idx in slots:
        track = _numeric(df, f"eye_{idx}_track_id")
        known = np.isfinite(track)
        if not known.any():
            continue  # no eye anchor in this slot
        track_id = track[known][-1]
        keep = track == track_id
        rows.append(
            _assess_slot(
                idx,
                int(track_id),
                _numeric(df, f"eye_{idx}_drift_px")[keep],
                timestamps[keep],
                _numeric(df, f"eye_{idx}_x")[keep],
                _numeric(df, f"eye_{idx}_y")[keep],
                eq,
                fps_default,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Trials on disk
# ---------------------------------------------------------------------------


def _assess_table(path: Path, eq: EyeQCSettings, fps_default: float) -> list[dict]:
    """Per-slot rows for one merged table, or a single trial-level status row."""
    columns = read_schema_columns(path)
    if not any(_LIVE_X_RE.match(c) for c in columns):
        # Written before per-frame eye tracking (or with eye_qc off): nothing to judge.
        return [{"status": "no_live_eye_columns"}]
    wanted = [c for c in columns if _NEEDED_RE.match(c)]
    if path.suffix.lower() == ".parquet":
        df = read_table(path, columns=wanted)
    else:
        df = read_table(path, usecols=wanted)
    if df.empty:
        return [{"status": "empty", "n_frames": 0}]
    return assess_trial(df, eq, fps_default) or [{"status": "no_eye_anchor", "n_frames": len(df)}]


def _trial_fields(trial_dir: Path, known_datasets: Iterable[str]) -> dict:
    """dataset / phase / trial_index / trial_type / odor, read the way
    check_light_stimulus reads them; from the folder names if that fails."""
    try:
        meta = load_trial_metadata(trial_dir, known_datasets=known_datasets)
    except Exception as exc:  # a bad sidecar must not cost the trial its QC row
        print(f"[EYE-DRIFT] {trial_dir.name}: no trial metadata ({type(exc).__name__}: {exc}); "
              f"using the folder names.")
        m = TRIAL_DIR_RE.search(trial_dir.name)
        phase = m.group(1).lower() if m else None
        return {
            "dataset": trial_dir.parent.parent.name,
            "phase": phase,
            "trial_index": int(m.group(2)) if m else None,
            "trial_type": phase,
            "odor": None,
        }
    return {
        "dataset": meta.dataset,
        "phase": meta.raw_trial_type or meta.trial_type,
        "trial_index": meta.trial_index,
        "trial_type": meta.trial_type,
        "odor": meta.odor,
    }


def _report_row(values: dict) -> dict:
    row = {col: values.get(col) for col in REPORT_COLUMNS}
    if row["eye_qc_flag"] is None:
        row["eye_qc_flag"] = False
    if row["eye_qc_reason"] is None:
        row["eye_qc_reason"] = ""
    return row


def _process_fly_dir(fly_dir: Path, cfg: Settings, eq: Optional[EyeQCSettings] = None) -> list[dict]:
    """QC rows for every trial folder in one batch directory. Picklable for joblib.

    Trial folders are the batch's direct children named ``..._<phase>_<N>``,
    the check_light_stimulus walk, so RMS_calculations copies are never read.
    A trial with no merged table (YOLO has not processed it) gets no row.
    """
    if eq is None:
        eq = _sanitized_eye_qc(getattr(cfg, "eye_qc", None))
    known_datasets = getattr(cfg, "datasets", ())
    fps_default = getattr(cfg, "fps_default", DEFAULT_FPS)
    try:
        trial_dirs = sorted(p for p in fly_dir.iterdir() if p.is_dir() and TRIAL_DIR_RE.search(p.name))
    except OSError as exc:
        print(f"[EYE-DRIFT] cannot list {fly_dir}: {exc}")
        return []

    rows: list[dict] = []
    for trial_dir in trial_dirs:
        table = None
        try:
            table = resolve_existing(trial_dir / f"{trial_dir.name}{MERGED_TABLE_SUFFIX}.parquet")
            if table is None:
                continue
            results = _assess_table(table, eq, fps_default)
        except Exception as exc:  # one bad table must not cost the rest of the run
            name = table.name if table is not None else trial_dir.name
            results = [{"status": "read_error",
                        "eye_qc_reason": f"could not read {name}: {type(exc).__name__}: {exc}"}]
        base = {
            "qc_version": QC_VERSION,
            **_trial_fields(trial_dir, known_datasets),
            "fly": fly_dir.name,
            "trial_dir": str(trial_dir),
            **_threshold_fields(eq),
        }
        rows.extend(_report_row({**base, **result}) for result in results)
    return rows


# ---------------------------------------------------------------------------
# Report files
# ---------------------------------------------------------------------------


def _truthy(values: pd.Series) -> pd.Series:
    """Bool per value; a CSV round-trip can hand back "True"/"False" strings."""
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})


def _normalised(df: pd.DataFrame) -> pd.DataFrame:
    """Report column order, whole numbers as nullable ints (a blank fly_number
    stays blank instead of turning every other one into 1.0) and a real bool flag."""
    df = df.copy()
    for col in REPORT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[list(REPORT_COLUMNS) + [c for c in df.columns if c not in REPORT_COLUMNS]]
    for col in _INT_COLUMNS:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        except (TypeError, ValueError):
            pass  # a hand-edited non-integer: leave the column as it is
    df["eye_qc_flag"] = _truthy(df["eye_qc_flag"])
    df["trial_dir"] = df["trial_dir"].astype(str)
    return df


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write through a temp file and a rename, so a crash never leaves half a report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _read_prior_report(path: Path) -> Optional[pd.DataFrame]:
    """The report already on disk, or None when there is none.

    An unreadable one is moved aside, never overwritten, and a fresh report
    starts; the other datasets' rows return as their pipelines re-run.
    """
    if not path.exists():
        return None
    try:
        old = pd.read_csv(path)
        missing = {"trial_dir", "fly_number", "eye_qc_flag"} - set(old.columns)
        if missing:
            raise ValueError(f"no {', '.join(sorted(missing))} column")
    except Exception as exc:
        aside = path.with_name(f"{path.name}.unreadable-{datetime.now():%Y%m%d-%H%M%S}")
        os.replace(path, aside)
        print(f"[EYE-DRIFT] WARNING: cannot read the existing report {path} "
              f"({type(exc).__name__}: {exc}); moved it to {aside.name} and started a new one.")
        return None
    return _normalised(old)


def _gone(path: Path) -> bool:
    """True only when ``path`` is certainly not on disk. Any other error (no
    permission, a flaky mount) reads as present: a row is never dropped on a guess."""
    try:
        path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return True
    except OSError:
        return False
    return False


def _stale_row_mask(
    old: pd.DataFrame, live_batches: Iterable[Path], scanned_roots: Iterable[Path]
) -> pd.Series:
    """True for a prior report row whose trial is gone from disk.

    Only rows under a root this call scanned are judged -- another dataset's
    rows are its own subprocess's business -- and only those whose batch
    folder was walked or is gone: a frozen batch is not walked and keeps its
    rows. Such a row is stale once its merged table is gone: the trial folder
    deleted (the orphan-trial cleanup), moved with its batch, or left without
    the table.
    """
    if old.empty:
        return pd.Series(False, index=old.index, dtype=bool)
    live, roots = set(live_batches), set(scanned_roots)
    verdicts: dict[str, bool] = {}

    def _stale(trial_dir) -> bool:
        if not isinstance(trial_dir, str) or not trial_dir:
            return False
        if trial_dir not in verdicts:
            trial = Path(trial_dir)
            batch = trial.parent
            verdicts[trial_dir] = (
                batch.parent in roots
                and (batch in live or _gone(batch))
                and all(
                    _gone(trial / f"{trial.name}{MERGED_TABLE_SUFFIX}{suffix}") for suffix in (".parquet", ".csv")
                )
            )
        return verdicts[trial_dir]

    return old["trial_dir"].map(_stale).astype(bool)


def _merge_report(
    old: Optional[pd.DataFrame],
    rows: list[dict],
    *,
    live_batches: Iterable[Path] = (),
    scanned_roots: Iterable[Path] = (),
) -> tuple[pd.DataFrame, list[dict]]:
    """``rows`` merged into the prior report ``old``: (full report, newly flagged rows).

    Each pipeline subprocess sees one dataset root, so the report is updated,
    never overwritten: every other trial's rows stay as they are. Rows are
    keyed by (trial_dir, fly_number), and a trial assessed here replaces ALL of
    its old rows -- a table re-made by re-inference can change status
    (``no_live_eye_columns`` -> one row per fly) or fly count, and the stale
    rows would otherwise linger. So do the rows of a trial gone from disk
    (_stale_row_mask): a deleted trial or a moved batch would otherwise be
    counted, and its flags listed, for good. "Newly flagged" is relative to the
    report's prior state, so unchanged tables never re-notify. Nothing is
    written here, so the new flags are known even when the report then cannot be.
    """
    new = _normalised(pd.DataFrame(rows, columns=list(REPORT_COLUMNS)))
    prior_flagged: set[tuple[str, int]] = set()
    if old is not None and not old.empty:
        was_flagged = old[_truthy(old["eye_qc_flag"]) & old["fly_number"].notna()]
        prior_flagged = {(str(t), int(n)) for t, n in zip(was_flagged["trial_dir"], was_flagged["fly_number"])}
        old = old[~old["trial_dir"].isin(set(new["trial_dir"]))]
        old = old[~_stale_row_mask(old, live_batches, scanned_roots)]
        report = pd.concat([old, new], ignore_index=True) if not old.empty else new
    else:
        report = new
    report = (
        report.drop_duplicates(subset=["trial_dir", "fly_number"], keep="last")
        .sort_values(["trial_dir", "fly_number"], kind="stable", na_position="first")
        .reset_index(drop=True)
    )
    newly_flagged = [
        r for r in rows
        if r["eye_qc_flag"] and (r["trial_dir"], r["fly_number"]) not in prior_flagged
    ]
    return report, newly_flagged


def frozen_row_mask(
    report: pd.DataFrame,
    cfg: Settings,
    *,
    thawed: Iterable[str] = (),
    thaw_all: bool = False,
) -> pd.Series:
    """True where a report row's batch folder is currently folder-frozen.

    Freeze rules are keyed by the dataset ROOT's folder name, exactly as
    iter_live_batch_dirs reads them, and that root is the trial's grandparent.
    """
    if report.empty or "trial_dir" not in report.columns:
        return pd.Series(False, index=report.index, dtype=bool)
    frozen_by_batch: dict[Path, bool] = {}

    def _frozen(trial_dir) -> bool:
        if not isinstance(trial_dir, str) or not trial_dir:
            return False
        batch = Path(trial_dir).parent
        if batch not in frozen_by_batch:
            frozen_by_batch[batch] = is_frozen_folder(
                cfg, batch.parent.name, batch, thawed=thawed, thaw_all=thaw_all
            )
        return frozen_by_batch[batch]

    return report["trial_dir"].map(_frozen).astype(bool)


def _write_flagged(report: pd.DataFrame, path: Path, cfg: Settings) -> None:
    """Rewrite the flagged-only CSV from the full report, minus folders frozen since."""
    flagged = report[_truthy(report["eye_qc_flag"])]
    if not flagged.empty:
        flagged = flagged[~frozen_row_mask(flagged, cfg)]
    _write_csv_atomic(flagged, path)


# ---------------------------------------------------------------------------
# Logging and alerts
# ---------------------------------------------------------------------------


def _plural(count: int, noun: str, plural: Optional[str] = None) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {plural or noun + 's'}"


def _label(row: dict) -> str:
    label = f"{row['dataset']}/{row['fly']}/{Path(row['trial_dir']).name}"
    return label if row["fly_number"] is None else f"{label} fly{row['fly_number']}"


def _summary(rows: list[dict], report_note: str) -> str:
    """The one line per call: what was assessed and flagged, then ``report_note``
    on what became of the report."""
    statuses = Counter(r["status"] for r in rows)
    assessed = [r for r in rows if r["status"] == "assessed"]
    flagged = [r for r in assessed if r["eye_qc_flag"]]
    kinds = Counter(kind for r in flagged for kind in _flag_kinds(r))
    parts = [
        f"assessed {_plural(len(assessed), 'trial-fly', 'trial-flies')} in "
        f"{_plural(len({r['trial_dir'] for r in assessed}), 'trial')}",
        f"{len(flagged)} flagged ({kinds['drift']} drift / {kinds['eye_missing']} eye_missing)",
        f"{_plural(statuses['no_live_eye_columns'], 'trial')} without per-frame eye columns",
    ]
    for status, what in (
        ("no_eye_anchor", "without an eye anchor"),
        ("empty", "with an empty table"),
        ("read_error", "unreadable"),
    ):
        if statuses[status]:
            parts.append(f"{_plural(statuses[status], 'trial')} {what}")
    return "[EYE-DRIFT] " + "; ".join(parts) + f". {report_note}"


def _has_batch_folders(root: Path) -> bool:
    """Whether ``root`` holds any folder at all, frozen or not. Only asked when
    no live batch folder came back, to tell an all-frozen root from an empty one."""
    try:
        return any(p.is_dir() for p in root.iterdir())
    except OSError:
        return True  # unlistable: no grounds to call it empty


def _notify(newly_flagged: list[dict], flagged_path: Path, report_problem: Optional[str] = None) -> None:
    """One alert for the newly flagged rows. ``report_problem`` is why the
    report could not be updated this run, if it could not."""
    lines = [f"- {_label(r)}: {r['eye_qc_reason']}" for r in newly_flagged[:NTFY_MAX_LINES]]
    extra = len(newly_flagged) - NTFY_MAX_LINES
    more = f"\n...and {extra} more" if extra > 0 else ""
    if report_problem is None:
        details = f"\n\nFull details: {flagged_path}"
    else:
        # Neither CSV was rewritten, and the report does not remember these
        # flags: say so, or the repeat alerts would read as new flies.
        details = (
            f"\n\n{REPORT_FILENAME} could NOT be written ({report_problem}), so these flags will be "
            "sent again every run until it can. Details: the [EYE-DRIFT] lines of the pipeline log."
        )
    ntfy_notify(
        f"Eye-drift QC: {len(newly_flagged)} fly-trial(s) flagged",
        "Eye moved off its frozen anchor (fly came loose? mount shifted?) or went unmatched:\n"
        + "\n".join(lines) + more + details,
        priority="high",
        tags="eyes,warning",
    )


# ---------------------------------------------------------------------------
# Step entry point
# ---------------------------------------------------------------------------


def _run(cfg: Settings) -> None:
    eq = _sanitized_eye_qc(getattr(cfg, "eye_qc", None))
    if not eq.enabled:
        print("[EYE-DRIFT] eye_qc.enabled is false; skipping the eye-drift QC.")
        return
    report_dir = resolve_eye_qc_report_dir(cfg)
    report_path = report_dir / REPORT_FILENAME
    flagged_path = report_dir / FLAGGED_FILENAME

    rows: list[dict] = []
    # Where this call looked: the report drops the rows of trials gone from
    # disk there, and only there.
    live_batches: list[Path] = []
    scanned_roots: list[Path] = []
    for root in get_main_directories(cfg):
        if not root.is_dir():
            print(f"[EYE-DRIFT] main_directories entry does not exist: {root}")
            continue
        print(f"[EYE-DRIFT] Scanning {root} for per-frame eye drift")
        # Frozen experiment folders are skipped, as in every per-trial step.
        fly_dirs = iter_live_batch_dirs(cfg, root)
        if not fly_dirs:
            # Every folder frozen is a retired dataset's normal state, and
            # iter_live_batch_dirs has already said so: no banner, no line.
            # A root without any batch folder may be an unmounted drive's
            # empty mount point, so its rows are never judged.
            if _has_batch_folders(root):
                scanned_roots.append(root)
            else:
                print(f"[EYE-DRIFT] No batch folders under {root}; nothing to check there.")
            continue
        scanned_roots.append(root)
        live_batches.extend(fly_dirs)
        root_rows = [
            row
            for result in parallel_map(
                partial(_process_fly_dir, cfg=cfg, eq=eq),
                fly_dirs,
                enabled=cfg.parallel.enabled,
                n_jobs=cfg.parallel.n_jobs,
            )
            for row in result
        ]
        if not root_rows:
            print("=" * 78)
            print(f"[EYE-DRIFT] WARNING: no *{MERGED_TABLE_SUFFIX} tables under {root} "
                  f"({_plural(len(fly_dirs), 'live batch folder')}): NOTHING was checked. Did YOLO run?")
            print("=" * 78)
        rows.extend(root_rows)

    if not rows:
        # Still one line, so a run that found nothing to assess says so.
        print(_summary(rows, f"Report unchanged: {report_path}"))
        return

    for row in rows:
        if row["eye_qc_flag"]:
            print(f"[EYE-DRIFT][FLAG] {_label(row)}: {row['eye_qc_reason']}")
        elif row["status"] == "read_error":
            print(f"[EYE-DRIFT] {_label(row)}: {row['eye_qc_reason']}")

    report = None
    report_problem = None
    # Every flag of this call is new until the prior report says otherwise --
    # also when that report can be neither read nor set aside, and is then left
    # as it is: writing a fresh one over it would lose every other dataset's rows.
    newly_flagged = [row for row in rows if row["eye_qc_flag"]]
    try:
        report, newly_flagged = _merge_report(
            _read_prior_report(report_path), rows, live_batches=live_batches, scanned_roots=scanned_roots
        )
        _write_csv_atomic(report, report_path)
    except Exception as exc:
        report = None
        report_problem = f"{type(exc).__name__}: {exc}"
        print(f"[EYE-DRIFT] ERROR: could not update {report_path} ({report_problem}).")
    if report is not None:
        try:
            _write_flagged(report, flagged_path, cfg)
        except Exception as exc:
            print(f"[EYE-DRIFT] ERROR: could not write {flagged_path} ({type(exc).__name__}: {exc}).")
    print(_summary(rows, f"Report: {report_path}" if report is not None else "Report NOT written"))

    # Alert even when the report could not be written: the new flags are known
    # from the prior report, and the log alone reaches no one.
    if newly_flagged:
        try:
            _notify(newly_flagged, flagged_path, report_problem)
        except Exception as exc:
            print(f"[EYE-DRIFT] ntfy alert failed ({type(exc).__name__}: {exc}).")


def main(cfg: Settings) -> None:
    """Assess every live trial under the configured roots. Never raises: an
    exception here would abort the whole nightly run, so a failure is printed
    and the pipeline carries on without the report."""
    try:
        _run(cfg)
    except Exception as exc:
        print(f"[EYE-DRIFT] ERROR: eye-drift QC failed ({type(exc).__name__}: {exc}); "
              f"the pipeline continues without it.")
