#!/usr/bin/env python3
"""Automate per-fly trial processing across multiple dataset roots.

This module walks one or more experiment roots, discovers every fly slot
(`flyN_distances.csv`) produced by the YOLO pipeline, and for each slot it:

* computes global distance statistics (min/max) across all training/testing
  trials,
* normalises every trial with those global bounds and derives a Hilbert
  envelope from a rolling RMS of the distance percentage,
* writes an augmented per-trial CSV containing the new features, and
* aggregates all trials for the fly into a single "wide" CSV containing
  metadata, summary metrics, and the envelope time-series.

The script finishes by concatenating every fly's wide table into a single
CSV, enabling downstream all-fly analyses without manual bookkeeping.

The implementation intentionally avoids hard-coded file names.  Instead it
leverages the existing :func:`fbpipe.utils.fly_files.iter_fly_distance_csvs`
helper to locate every ``flyN_distances`` CSV underneath a fly directory and
uses regular expressions to infer the trial number/type.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.signal import hilbert

from fbpipe.utils.columns import (
    PROBOSCIS_DISTANCE_COL,
    PROBOSCIS_DISTANCE_PCT_COL,
    PROBOSCIS_MAX_DISTANCE_COL,
    PROBOSCIS_MIN_DISTANCE_COL,
    find_proboscis_distance_column,
)
from fbpipe.utils.fly_files import iter_fly_distance_csvs


# ---------------------------------------------------------------------------
# Constants mirroring the existing envelope exporters
# ---------------------------------------------------------------------------


AUC_COLUMNS = (
    "AUC-Before",
    "AUC-During",
    "AUC-After",
    "AUC-During-Before-Ratio",
    "AUC-After-Before-Ratio",
    "TimeToPeak-During",
    "Peak-Value",
)

BEFORE_FRAMES = 1260
DURING_FRAMES = 1200
AFTER_FRAMES = 1200

TRIAL_REGEX = re.compile(r"(testing|training)_(\d+)", re.IGNORECASE)
TIME_COLS = ("time_s", "time_seconds", "t_s", "time")
TIMESTAMP_COLS = ("UTC_ISO", "Timestamp", "Number", "MonoNs")
FRAME_COLS = ("Frame", "FrameNumber", "Frame Number")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TrialEntry:
    """Metadata describing a single trial CSV."""

    path: Path
    slot_token: str
    slot_index: int
    trial_label: str
    trial_type: str


@dataclass(slots=True)
class TrialArtifacts:
    """Runtime artefacts derived from a trial CSV."""

    entry: TrialEntry
    dataframe: pd.DataFrame
    envelope: np.ndarray
    rms: np.ndarray
    fps_estimate: float
    fallback_fps: float
    default_fps: float
    fly_before_mean: float


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Not a directory: {resolved}")
    return resolved


def _pick_column(candidates: Sequence[str], df: pd.DataFrame) -> Optional[str]:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _seconds_from_timestamp(df: pd.DataFrame, column: str) -> pd.Series:
    series = df[column]
    if column in ("UTC_ISO", "Timestamp"):
        dt = pd.to_datetime(series, errors="coerce", utc=(column == "UTC_ISO"))
        secs = dt.astype("int64", copy=False) / 1e9
    elif column == "Number":
        secs = pd.to_numeric(series, errors="coerce").astype(float)
    elif column == "MonoNs":
        secs = pd.to_numeric(series, errors="coerce").astype(float) / 1e9
    else:
        raise ValueError(f"Unsupported timestamp column: {column}")
    values = secs.to_numpy(np.float64, copy=False)
    finite_mask = np.isfinite(values)
    origin = float(np.min(values[finite_mask])) if np.any(finite_mask) else 0.0
    return (secs - origin).astype(float)


def _time_axis(df: pd.DataFrame, fps_default: float) -> tuple[np.ndarray, float, float]:
    """Return a time axis (seconds) plus primary and fallback FPS estimates."""

    ts_col = _pick_column(TIMESTAMP_COLS, df)
    if ts_col:
        seconds = _seconds_from_timestamp(df, ts_col)
        fps_est = _estimate_fps(seconds)
        return seconds.to_numpy(np.float64, copy=False), float(fps_est or 0.0), fps_default

    time_col = _pick_column(TIME_COLS, df)
    if time_col:
        seconds = pd.to_numeric(df[time_col], errors="coerce").astype(float)
        fps_est = _estimate_fps(seconds)
        return seconds.to_numpy(np.float64, copy=False), float(fps_est or 0.0), fps_default

    frame_col = _pick_column(FRAME_COLS, df)
    if frame_col:
        frames = pd.to_numeric(df[frame_col], errors="coerce").astype(float)
        fps_est = _estimate_fps(frames)
        seconds = frames / max(fps_est or fps_default, 1e-9)
        return seconds.to_numpy(np.float64, copy=False), float(fps_est or 0.0), fps_default

    count = len(df)
    seconds = np.arange(count, dtype=float) / max(fps_default, 1e-9)
    return seconds, 0.0, fps_default


def _estimate_fps(series: pd.Series) -> Optional[float]:
    mask = series.notna()
    if mask.sum() < 2:
        return None
    duration = series[mask].iloc[-1] - series[mask].iloc[0]
    if duration <= 0:
        return None
    return float(mask.sum() / duration)


def _rolling_rms(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(np.nan_to_num(values, nan=0.0), copy=False)
    squared = series.pow(2.0)
    mean = squared.rolling(window=window, center=True, min_periods=1).mean()
    return mean.pow(0.5).to_numpy()


def _hilbert_envelope(values: np.ndarray, window: int) -> np.ndarray:
    env = np.abs(hilbert(np.nan_to_num(values, nan=0.0)))
    smoothed = (
        pd.Series(env)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    return smoothed


def _segment_bounds(total_len: int) -> tuple[int, int, int, int]:
    before_end = max(0, min(BEFORE_FRAMES, total_len))
    during_start = before_end
    during_end = max(during_start, min(during_start + DURING_FRAMES, total_len))
    after_end = max(during_end, min(during_end + AFTER_FRAMES, total_len))
    return before_end, during_start, during_end, after_end


def _segment_auc(values: np.ndarray, threshold: float, dt: float) -> float:
    if values.size == 0 or dt <= 0:
        return 0.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    diff = finite - threshold
    positive = diff[diff > 0]
    if positive.size == 0:
        return 0.0
    return float(np.sum(positive) * dt)


def _compute_trial_metrics(
    envelope: np.ndarray,
    fps: float,
    *,
    fallback_fps: float,
    default_fps: float,
    fly_before_mean: float,
) -> dict[str, float]:
    total_len = envelope.size
    before_end, during_start, during_end, after_end = _segment_bounds(total_len)

    before = envelope[:before_end]
    during = envelope[during_start:during_end]
    after = envelope[during_end:after_end]

    before_mean = float(np.nanmean(before)) if before.size else math.nan
    before_std = float(np.nanstd(before)) if before.size else 0.0

    baseline = fly_before_mean
    if not np.isfinite(baseline):
        baseline = before_mean
    if not np.isfinite(baseline):
        baseline = 0.0
    if not np.isfinite(before_std):
        before_std = 0.0

    threshold = baseline + 3.0 * before_std

    fps_eff = _effective_fps(fps, fallback=fallback_fps, default=default_fps)
    dt = 1.0 / fps_eff if fps_eff > 0 else 0.0

    auc_before = _segment_auc(before, threshold, dt)
    auc_during = _segment_auc(during, threshold, dt)
    auc_after = _segment_auc(after, threshold, dt)

    during_ratio = float(auc_during / auc_before) if auc_before > 0 else float("nan")
    after_ratio = float(auc_after / auc_before) if auc_before > 0 else float("nan")

    peak_value = float("nan")
    time_to_peak = float("nan")
    if during.size:
        finite_during = during[np.isfinite(during)]
        if finite_during.size:
            peak_value = float(np.max(finite_during))

    if total_len and fps_eff > 0:
        try:
            global_peak_idx = int(np.nanargmax(envelope))
        except ValueError:
            global_peak_idx = -1
        if during_start <= global_peak_idx < during_end:
            time_to_peak = (global_peak_idx - during_start) / fps_eff

    return {
        "AUC-Before": float(auc_before),
        "AUC-During": float(auc_during),
        "AUC-After": float(auc_after),
        "AUC-During-Before-Ratio": during_ratio,
        "AUC-After-Before-Ratio": after_ratio,
        "TimeToPeak-During": time_to_peak,
        "Peak-Value": peak_value,
    }


def _effective_fps(value: float, *, fallback: float, default: float) -> float:
    for candidate in (value, fallback, default):
        if np.isfinite(candidate) and candidate > 0:
            return float(candidate)
    return 0.0


def _trial_label_from_path(path: Path, default_category: str) -> str:
    match = TRIAL_REGEX.search(path.stem)
    if match:
        return f"{match.group(1).lower()}_{match.group(2)}"
    for parent in path.parents:
        match = TRIAL_REGEX.search(parent.name)
        if match:
            return f"{match.group(1).lower()}_{match.group(2)}"
    trailing = re.search(r"(\d+)$", path.stem)
    if trailing:
        return f"{default_category}_{trailing.group(1)}"
    return f"{default_category}_{path.stem.lower()}"


def _trial_type_from_path(path: Path) -> str:
    lowered = path.as_posix().lower()
    if "training" in lowered:
        return "training"
    if "testing" in lowered:
        return "testing"
    return "testing"


def _discover_trials(fly_dir: Path) -> Iterator[TrialEntry]:
    seen: set[Path] = set()
    for csv_path, token, slot_idx in iter_fly_distance_csvs(fly_dir, recursive=True):
        resolved = csv_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        trial_type = _trial_type_from_path(resolved)
        trial_label = _trial_label_from_path(resolved, trial_type)
        yield TrialEntry(resolved, token, slot_idx, trial_label, trial_type)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _valid_distance_values(
    series: pd.Series, *, lower: float, upper: float
) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
    mask = np.isfinite(values) & (values >= lower) & (values <= upper)
    return values[mask]


def _compute_global_distance_bounds(
    entries: Iterable[TrialEntry],
    *,
    distance_min: float,
    distance_max: float,
) -> tuple[float, float]:
    gmin = math.inf
    gmax = -math.inf
    total = 0
    for entry in entries:
        try:
            df = pd.read_csv(entry.path, usecols=None)
        except Exception:
            continue
        dist_col = find_proboscis_distance_column(df)
        if dist_col is None:
            continue
        vals = _valid_distance_values(df[dist_col], lower=distance_min, upper=distance_max)
        if vals.size == 0:
            continue
        gmin = min(gmin, float(np.min(vals)))
        gmax = max(gmax, float(np.max(vals)))
        total += 1
    if not total:
        return float("nan"), float("nan")
    return gmin, gmax


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def process_trial(
    entry: TrialEntry,
    *,
    dataset: str,
    fly_dir: Path,
    slot_label: str,
    global_min: float,
    global_max: float,
    fps_default: float,
    window_sec: float,
) -> TrialArtifacts | None:
    try:
        df = pd.read_csv(entry.path)
    except Exception as exc:
        print(f"[WARN] Failed to read {entry.path}: {exc}")
        return None

    dist_col = find_proboscis_distance_column(df)
    if dist_col is None:
        print(f"[WARN] {entry.path.name}: no proboscis distance column detected.")
        return None

    distances = pd.to_numeric(df[dist_col], errors="coerce").astype(float).to_numpy()
    perc = np.empty_like(distances, dtype=float)
    over = distances > global_max
    under = distances < global_min
    finite = ~(over | under)
    perc[over] = 101.0
    perc[under] = -1.0
    if np.isfinite(global_max) and np.isfinite(global_min) and global_max != global_min:
        perc[finite] = 100.0 * (distances[finite] - global_min) / (global_max - global_min)
    else:
        perc[finite] = 0.0

    df[PROBOSCIS_DISTANCE_COL] = distances
    df[PROBOSCIS_DISTANCE_PCT_COL] = perc
    df[PROBOSCIS_MIN_DISTANCE_COL] = global_min
    df[PROBOSCIS_MAX_DISTANCE_COL] = global_max
    df["dataset"] = dataset
    df["fly_directory"] = fly_dir.name
    df["fly_slot"] = slot_label
    df["trial_label"] = entry.trial_label
    df["trial_type"] = entry.trial_type

    seconds, fps_est, fallback = _time_axis(df, fps_default)
    df["time_seconds_multi_fly"] = seconds

    window_frames = max(int(round(window_sec * (fps_est if fps_est > 0 else fps_default))), 1)
    perc_clipped = np.clip(np.nan_to_num(perc, nan=0.0), 0.0, 100.0)
    rms = _rolling_rms(perc_clipped, window_frames)
    envelope = _hilbert_envelope(rms, window_frames)
    df["distance_rms_multi_fly"] = rms
    df["distance_envelope_multi_fly"] = envelope

    return TrialArtifacts(
        entry=entry,
        dataframe=df,
        envelope=envelope,
        rms=rms,
        fps_estimate=fps_est,
        fallback_fps=fallback,
        default_fps=fps_default,
        fly_before_mean=float("nan"),
    )


def process_slot(
    dataset: str,
    fly_dir: Path,
    slot_token: str,
    entries: list[TrialEntry],
    *,
    distance_min: float,
    distance_max: float,
    fps_default: float,
    window_sec: float,
) -> list[Mapping[str, float | str]]:
    if not entries:
        return []

    slot_label = slot_token.replace("_distances", "")
    print(f"[INFO] {fly_dir.name}/{slot_label}: processing {len(entries)} trials")

    bounds = _compute_global_distance_bounds(entries, distance_min=distance_min, distance_max=distance_max)
    global_min, global_max = bounds
    if not np.isfinite(global_min) or not np.isfinite(global_max):
        print(
            f"[WARN] {fly_dir.name}/{slot_label}: unable to derive global bounds; "
            "skipping fly."
        )
        return []

    trial_artifacts: list[TrialArtifacts] = []
    before_samples: list[np.ndarray] = []

    out_dir = fly_dir / "multi_fly_processed" / slot_label
    _ensure_dir(out_dir)

    for entry in sorted(entries, key=lambda e: (e.trial_type, e.trial_label)):
        art = process_trial(
            entry,
            dataset=dataset,
            fly_dir=fly_dir,
            slot_label=slot_label,
            global_min=global_min,
            global_max=global_max,
            fps_default=fps_default,
            window_sec=window_sec,
        )
        if art is None:
            continue
        trial_artifacts.append(art)
        before_end = min(BEFORE_FRAMES, art.envelope.size)
        if before_end > 0:
            before_samples.append(art.envelope[:before_end])

        out_csv = out_dir / f"{entry.trial_label}_{slot_label}_processed.csv"
        art.dataframe.to_csv(out_csv, index=False)
        print(f"[OK] Wrote per-trial CSV → {out_csv}")

    if not trial_artifacts:
        print(f"[WARN] {fly_dir.name}/{slot_label}: no trials produced artefacts.")
        return []

    baseline = float("nan")
    if before_samples:
        baseline = float(np.nanmean(np.concatenate(before_samples)))

    for art in trial_artifacts:
        art.fly_before_mean = baseline

    max_len = max(art.envelope.size for art in trial_artifacts)
    env_columns = [f"env_{idx}" for idx in range(max_len)]

    summary_rows: list[dict[str, float | str]] = []
    for art in trial_artifacts:
        metrics = _compute_trial_metrics(
            art.envelope,
            art.fps_estimate,
            fallback_fps=art.fallback_fps,
            default_fps=art.default_fps,
            fly_before_mean=art.fly_before_mean,
        )

        env_row = np.full(max_len, np.nan, dtype=float)
        env_row[: art.envelope.size] = art.envelope

        row: dict[str, float | str] = {
            "dataset": dataset,
            "fly_directory": fly_dir.name,
            "fly_slot": slot_label,
            "trial_label": art.entry.trial_label,
            "trial_type": art.entry.trial_type,
            "csv_path": str(art.entry.path),
            "global_min_distance": global_min,
            "global_max_distance": global_max,
            "baseline_mean": baseline,
            "fps_estimate": art.fps_estimate,
            "fallback_fps": art.fallback_fps,
        }
        row.update(metrics)
        row.update(zip(env_columns, env_row))
        summary_rows.append(row)

    wide_df = pd.DataFrame(summary_rows)
    combined_path = out_dir / f"{slot_label}_combined.csv"
    wide_df.to_csv(combined_path, index=False)
    print(f"[OK] Wrote combined fly summary → {combined_path}")

    return summary_rows


def process_root(
    root: Path,
    *,
    distance_min: float,
    distance_max: float,
    fps_default: float,
    window_sec: float,
) -> list[Mapping[str, float | str]]:
    dataset = root.name
    print(f"[ROOT] Scanning dataset '{dataset}' in {root}")
    aggregate: list[Mapping[str, float | str]] = []
    for fly_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        entries_by_slot: dict[str, list[TrialEntry]] = {}
        for entry in _discover_trials(fly_dir):
            entries_by_slot.setdefault(entry.slot_token, []).append(entry)
        if not entries_by_slot:
            print(f"[WARN] {fly_dir.name}: no flyN_distances trials discovered.")
            continue
        for slot_token, entries in sorted(entries_by_slot.items()):
            aggregate.extend(
                process_slot(
                    dataset,
                    fly_dir,
                    slot_token,
                    entries,
                    distance_min=distance_min,
                    distance_max=distance_max,
                    fps_default=fps_default,
                    window_sec=window_sec,
                )
            )
    return aggregate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="+",
        help="One or more dataset roots containing per-fly trial folders.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="multi_fly_outputs",
        help="Directory for the aggregated all-fly CSV (defaults to %(default)s).",
    )
    parser.add_argument(
        "--distance-min",
        type=float,
        default=70.0,
        help="Lower bound for in-range proboscis distances (defaults to %(default)s).",
    )
    parser.add_argument(
        "--distance-max",
        type=float,
        default=250.0,
        help="Upper bound for in-range proboscis distances (defaults to %(default)s).",
    )
    parser.add_argument(
        "--fps-default",
        type=float,
        default=40.0,
        help="Default frames-per-second when metadata is missing (defaults to %(default)s).",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=0.25,
        help="RMS/Hilbert rolling window size in seconds (defaults to %(default)s).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    roots = [_resolve_path(root) for root in args.roots]
    aggregate_rows: list[Mapping[str, float | str]] = []
    for root in roots:
        aggregate_rows.extend(
            process_root(
                root,
                distance_min=args.distance_min,
                distance_max=args.distance_max,
                fps_default=args.fps_default,
                window_sec=args.window_sec,
            )
        )

    if not aggregate_rows:
        print("[WARN] No trials processed across all roots; nothing to export.")
        return

    output_dir = Path(args.output).expanduser().resolve()
    _ensure_dir(output_dir)
    all_df = pd.DataFrame(aggregate_rows)
    out_csv = output_dir / "all_flies_combined.csv"
    all_df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote all-fly aggregation → {out_csv}")


if __name__ == "__main__":
    main()

