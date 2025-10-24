"""CLI tool to compute geometric features for fly trials.

This script recursively searches for trial CSV files containing eye and
proboscis coordinates, computes per-frame geometric features and per-fly
normalisation statistics, and emits enriched per-trial CSVs alongside a
consolidated summary CSV.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


TRIAL_TYPE_PATTERN = re.compile(r"(training|testing)", re.IGNORECASE)
TRIAL_LABEL_PATTERN = re.compile(r"(training|testing)_[^/\\]+", re.IGNORECASE)
YOLO_CSV_PATTERN = re.compile(r"_distances\.csv$", re.IGNORECASE)
FLY_SLOT_PATTERN = re.compile(r"(fly\d+)", re.IGNORECASE)
WIDE_COL_PATTERN = re.compile(r"(?P<prefix>eye_x|eye_y|prob_x|prob_y)_f(?P<frame>\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class TrialInfo:
    """Metadata about a discovered trial CSV."""

    dataset: str
    fly_directory: str
    fly_path: Path
    trial_type: str
    trial_label: str
    trial_dir: Path
    csv_path_in: Path
    fly_slot: str


@dataclass
class FlyStats:
    """Per-fly geometric statistics used for normalisation."""

    W_est_fly: float
    H_est_fly: float
    diag_est_fly: float
    r_min_fly: float
    r_max_fly: float
    r_p01_fly: float
    r_p99_fly: float
    r_mean_fly: float
    r_std_fly: float


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute geometric features for fly trials.")
    parser.add_argument("--roots", nargs="+", required=True, help="Root directories to scan for trial CSV files.")
    parser.add_argument("--outdir", required=True, help="Directory for consolidated output and optional per-trial outputs.")
    parser.add_argument("--dry-run", action="store_true", help="List processing actions without writing outputs.")
    parser.add_argument("--limit-flies", type=int, default=None, help="Process at most N flies per dataset root.")
    parser.add_argument("--limit-trials", type=int, default=None, help="Process at most N trials per fly.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default INFO).")
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")


def discover_trials(roots: Sequence[str]) -> List[TrialInfo]:
    trials: List[TrialInfo] = []
    for root_str in roots:
        root_path = Path(root_str).resolve()
        if not root_path.exists():
            LOGGER.warning("Root does not exist: %s", root_path)
            continue
        dataset = root_path.name
        LOGGER.info("Scanning dataset root: %s", root_path)
        for fly_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
            fly_directory = fly_dir.name
            for trial_dir in sorted(p for p in fly_dir.iterdir() if p.is_dir()):
                type_matches = TRIAL_TYPE_PATTERN.findall(trial_dir.name)
                if not type_matches:
                    continue

                trial_type = type_matches[-1].lower()
                trial_name_lower = trial_dir.name.lower()
                label_match = TRIAL_LABEL_PATTERN.search(trial_dir.name)
                if label_match:
                    # Normalise label casing for consistent downstream grouping
                    trial_label = label_match.group(0).lower()
                else:
                    trial_label = trial_dir.name
                csv_candidates = sorted(
                    p
                    for p in trial_dir.glob("*.csv")
                    if not p.name.endswith("_geom.csv")
                    and not p.name.endswith("_coords.csv")
                    and not p.name.endswith("_coords_long.csv")
                    and not p.name.endswith("_coords_wide.csv")
                )
                if not csv_candidates:
                    LOGGER.debug("No CSV files in trial directory: %s", trial_dir)
                    continue
                preferred = [p for p in csv_candidates if YOLO_CSV_PATTERN.search(p.name)]
                if preferred:
                    csv_path_in = preferred[0]
                else:
                    csv_path_in = csv_candidates[0]
                    LOGGER.debug(
                        "Defaulting to first CSV in %s (no *_distances.csv found): %s",
                        trial_dir,
                        csv_path_in,
                    )
                fly_slot_match = FLY_SLOT_PATTERN.search(csv_path_in.stem)
                fly_slot = fly_slot_match.group(1).lower() if fly_slot_match else "fly1"
                trials.append(
                    TrialInfo(
                        dataset=dataset,
                        fly_directory=fly_directory,
                        fly_path=fly_dir,
                        trial_type=trial_type,
                        trial_label=trial_label,
                        trial_dir=trial_dir,
                        csv_path_in=csv_path_in,
                        fly_slot=fly_slot,
                    )
                )
    LOGGER.info("Discovered %d trial CSV files.", len(trials))
    return trials


def apply_limits(trials: Sequence[TrialInfo], limit_flies: Optional[int], limit_trials: Optional[int]) -> List[TrialInfo]:
    if limit_flies is None and limit_trials is None:
        return list(trials)

    grouped: Dict[Tuple[str, str], List[TrialInfo]] = {}
    for trial in trials:
        grouped.setdefault((trial.dataset, trial.fly_directory), []).append(trial)

    limited_trials: List[TrialInfo] = []
    fly_count = 0
    for (dataset, fly_directory), fly_trials in grouped.items():
        if limit_flies is not None and fly_count >= limit_flies:
            break
        fly_count += 1
        fly_trials_sorted = sorted(fly_trials, key=lambda t: (t.trial_type, t.trial_label, t.csv_path_in.name))
        if limit_trials is not None:
            fly_trials_sorted = fly_trials_sorted[:limit_trials]
        limited_trials.extend(fly_trials_sorted)
    LOGGER.info("After applying limits: %d trials", len(limited_trials))
    return limited_trials


def load_coordinates(trial: TrialInfo) -> pd.DataFrame:
    df = pd.read_csv(trial.csv_path_in)
    coord_df = extract_coordinates(df)
    if coord_df is not None:
        return coord_df

    LOGGER.debug("Primary CSV lacks coordinates. Searching for auxiliary files for %s", trial.csv_path_in)
    aux_path = find_auxiliary_coordinate_csv(trial)
    if aux_path is None:
        raise FileNotFoundError(f"No coordinate columns found for trial {trial.csv_path_in}")
    aux_df = pd.read_csv(aux_path)
    coord_df = extract_coordinates(aux_df)
    if coord_df is None:
        raise ValueError(f"Auxiliary coordinate file lacks required schema: {aux_path}")
    return coord_df


def extract_coordinates(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    columns_lower = {c.lower(): c for c in df.columns}
    long_columns = ["frame", "eye_x", "eye_y", "prob_x", "prob_y"]
    if all(col in columns_lower for col in long_columns):
        selected = df[[columns_lower[col] for col in long_columns]].copy()
        selected.columns = long_columns
        return selected

    # Attempt to parse wide schema
    matches = [WIDE_COL_PATTERN.match(col) for col in df.columns]
    if any(matches):
        return convert_wide_to_long(df)

    return None


def convert_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    frames: Dict[int, Dict[str, float]] = {}
    for _, row in df.iterrows():
        for col, value in row.items():
            match = WIDE_COL_PATTERN.match(str(col))
            if not match:
                continue
            field = match.group("prefix").lower()
            frame = int(match.group("frame"))
            frames.setdefault(frame, {})[field] = value

    if not frames:
        raise ValueError("Wide-format coordinate columns not found.")

    records = []
    for frame_idx in sorted(frames):
        frame_data = frames[frame_idx]
        record = {
            "frame": frame_idx,
            "eye_x": frame_data.get("eye_x", math.nan),
            "eye_y": frame_data.get("eye_y", math.nan),
            "prob_x": frame_data.get("prob_x", math.nan),
            "prob_y": frame_data.get("prob_y", math.nan),
        }
        records.append(record)

    result = pd.DataFrame.from_records(records, columns=["frame", "eye_x", "eye_y", "prob_x", "prob_y"])
    return result


def find_auxiliary_coordinate_csv(trial: TrialInfo) -> Optional[Path]:
    candidates: List[Path] = []
    suffixes = ["_coords.csv", "_coords_long.csv", "_coords_wide.csv"]
    base = trial.csv_path_in.stem
    for suffix in suffixes:
        direct = trial.csv_path_in.with_name(base + suffix)
        if direct.exists():
            return direct

    # Same directory search
    for suffix in suffixes:
        candidates.extend(sorted(trial.trial_dir.glob(f"*{suffix}")))

    fly_dir = trial.fly_path
    if fly_dir.exists():
        for suffix in suffixes:
            candidates.extend(sorted(fly_dir.glob(f"*{suffix}")))

    candidates = [path for path in candidates if path != trial.csv_path_in]
    return candidates[0] if candidates else None


def compute_fly_stats(trial_data: Dict[TrialInfo, pd.DataFrame]) -> FlyStats:
    x_values: List[np.ndarray] = []
    y_values: List[np.ndarray] = []
    r_values: List[np.ndarray] = []

    for df in trial_data.values():
        eye_x = df["eye_x"].to_numpy(dtype=float)
        eye_y = df["eye_y"].to_numpy(dtype=float)
        prob_x = df["prob_x"].to_numpy(dtype=float)
        prob_y = df["prob_y"].to_numpy(dtype=float)
        dx = prob_x - eye_x
        dy = prob_y - eye_y
        r = np.sqrt(dx ** 2 + dy ** 2)

        if not np.all(np.isnan(eye_x)):
            x_values.append(eye_x[~np.isnan(eye_x)])
        if not np.all(np.isnan(prob_x)):
            x_values.append(prob_x[~np.isnan(prob_x)])
        if not np.all(np.isnan(eye_y)):
            y_values.append(eye_y[~np.isnan(eye_y)])
        if not np.all(np.isnan(prob_y)):
            y_values.append(prob_y[~np.isnan(prob_y)])
        if not np.all(np.isnan(r)):
            r_values.append(r[~np.isnan(r)])

    if not x_values or not y_values or not r_values:
        raise ValueError("Insufficient data to compute fly statistics.")

    x_concat = np.concatenate(x_values)
    y_concat = np.concatenate(y_values)
    r_concat = np.concatenate(r_values)

    W_est_fly = float(np.nanmax(x_concat) - np.nanmin(x_concat))
    H_est_fly = float(np.nanmax(y_concat) - np.nanmin(y_concat))
    diag_est_fly = float(math.hypot(W_est_fly, H_est_fly))

    r_min_fly = float(np.nanmin(r_concat))
    r_max_fly = float(np.nanmax(r_concat))
    r_p01_fly = float(np.nanpercentile(r_concat, 1))
    r_p99_fly = float(np.nanpercentile(r_concat, 99))
    r_mean_fly = float(np.nanmean(r_concat))
    r_std_fly = float(np.nanstd(r_concat))

    fly_stats = FlyStats(
        W_est_fly=W_est_fly,
        H_est_fly=H_est_fly,
        diag_est_fly=diag_est_fly,
        r_min_fly=r_min_fly,
        r_max_fly=r_max_fly,
        r_p01_fly=r_p01_fly,
        r_p99_fly=r_p99_fly,
        r_mean_fly=r_mean_fly,
        r_std_fly=r_std_fly,
    )
    LOGGER.info(
        "Fly stats for %s: W=%.3f H=%.3f diag=%.3f r[min=%.3f max=%.3f mean=%.3f std=%.3f]",
        next(iter(trial_data.keys())).fly_directory,
        W_est_fly,
        H_est_fly,
        diag_est_fly,
        r_min_fly,
        r_max_fly,
        r_mean_fly,
        r_std_fly,
    )
    return fly_stats


def enrich_trial(
    trial: TrialInfo,
    df: pd.DataFrame,
    stats: FlyStats,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = df.copy()
    df = df.astype({col: float for col in ["frame", "eye_x", "eye_y", "prob_x", "prob_y"]}, errors="ignore")

    dx = df["prob_x"] - df["eye_x"]
    dy = df["prob_y"] - df["eye_y"]
    r = np.sqrt(dx ** 2 + dy ** 2)
    r_safe = r + 1e-6

    cos_theta = dx / r_safe
    sin_theta = dy / r_safe

    dx_over_W = dx / (stats.W_est_fly + 1e-6)
    dy_over_H = dy / (stats.H_est_fly + 1e-6)
    r_over_diag = r / (stats.diag_est_fly + 1e-6)
    r_pct_minmax = 100.0 * (r - stats.r_min_fly) / ((stats.r_max_fly - stats.r_min_fly) + 1e-6)
    r_pct_robust = 100.0 * (r - stats.r_p01_fly) / ((stats.r_p99_fly - stats.r_p01_fly) + 1e-6)
    r_pct_robust = r_pct_robust.clip(lower=0.0, upper=100.0)
    r_z = (r - stats.r_mean_fly) / (stats.r_std_fly + 1e-6)

    summary = {
        "n_frames": int(len(df)),
        "r_mean_trial": float(np.nanmean(r)),
        "r_std_trial": float(np.nanstd(r)),
        "r_max_trial": float(np.nanmax(r)),
        "r95_trial": float(np.nanpercentile(r, 95)),
        "dx_mean_abs": float(np.nanmean(np.abs(dx))),
        "dy_mean_abs": float(np.nanmean(np.abs(dy))),
        "r_pct_robust_fly_max": float(np.nanmax(r_pct_robust)),
        "r_pct_robust_fly_mean": float(np.nanmean(r_pct_robust)),
    }

    nan_r = int(np.isnan(r).sum())
    nan_eye = int(df[["eye_x", "eye_y"]].isna().sum().sum())
    nan_prob = int(df[["prob_x", "prob_y"]].isna().sum().sum())
    if nan_r or nan_eye or nan_prob:
        LOGGER.debug(
            "Trial %s has NaNs -> r: %d, eye coords: %d, prob coords: %d",
            trial.csv_path_in,
            nan_r,
            nan_eye,
            nan_prob,
        )

    enriched = pd.DataFrame({
        "frame": df["frame"],
        "eye_x": df["eye_x"],
        "eye_y": df["eye_y"],
        "prob_x": df["prob_x"],
        "prob_y": df["prob_y"],
        "dx": dx,
        "dy": dy,
        "r": r,
        "cos_theta": cos_theta,
        "sin_theta": sin_theta,
        "dx_over_W_fly": dx_over_W,
        "dy_over_H_fly": dy_over_H,
        "r_over_diag_fly": r_over_diag,
        "r_pct_minmax_fly": r_pct_minmax,
        "r_pct_robust_fly": r_pct_robust,
        "r_z_fly": r_z,
    })

    for column in ["r_pct_robust_fly"]:
        series = enriched[column].dropna()
        if not series.empty and (series.lt(0).any() or series.gt(100).any()):
            raise AssertionError(f"{column} not clipped to [0, 100] for {trial.csv_path_in}")

    enriched["dataset"] = trial.dataset
    enriched["fly_directory"] = trial.fly_directory
    enriched["fly_slot"] = trial.fly_slot
    enriched["trial_type"] = trial.trial_type
    enriched["trial_label"] = trial.trial_label

    return enriched, summary


def determine_output_path(trial: TrialInfo, outdir: Path) -> Path:
    relative_parts = [trial.dataset, trial.fly_directory, trial.trial_label]
    filename = trial.csv_path_in.stem + "_geom.csv"
    return outdir.joinpath(*relative_parts, filename)


def ensure_parent(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def process_trials(
    trials: Sequence[TrialInfo],
    outdir: Path,
    dry_run: bool,
) -> List[Dict[str, object]]:
    consolidated_rows: List[Dict[str, object]] = []
    trials_by_fly: Dict[Tuple[str, str], List[TrialInfo]] = {}
    for trial in trials:
        trials_by_fly.setdefault((trial.dataset, trial.fly_directory), []).append(trial)

    for (dataset, fly_directory), fly_trials in trials_by_fly.items():
        fly_trials = sorted(fly_trials, key=lambda t: (t.trial_type, t.trial_label, t.csv_path_in.name))
        LOGGER.info("Processing fly %s/%s with %d trials", dataset, fly_directory, len(fly_trials))
        trial_data: Dict[TrialInfo, pd.DataFrame] = {}
        for trial in fly_trials:
            df = load_coordinates(trial)
            if df.empty:
                raise ValueError(f"Empty coordinate dataframe for trial {trial.csv_path_in}")
            trial_data[trial] = df

        stats = compute_fly_stats(trial_data)

        for trial in fly_trials:
            enriched_df, summary = enrich_trial(trial, trial_data[trial], stats)
            output_path = determine_output_path(trial, outdir)
            csv_path_out = output_path
            consolidated_row: Dict[str, object] = {
                "dataset": trial.dataset,
                "fly_directory": trial.fly_directory,
                "fly_slot": trial.fly_slot,
                "trial_type": trial.trial_type,
                "trial_label": trial.trial_label,
                "csv_path_in": str(trial.csv_path_in),
                "csv_path_out": str(csv_path_out),
                "W_est_fly": stats.W_est_fly,
                "H_est_fly": stats.H_est_fly,
                "diag_est_fly": stats.diag_est_fly,
                "r_min_fly": stats.r_min_fly,
                "r_max_fly": stats.r_max_fly,
                "r_p01_fly": stats.r_p01_fly,
                "r_p99_fly": stats.r_p99_fly,
                "r_mean_fly": stats.r_mean_fly,
                "r_std_fly": stats.r_std_fly,
            }
            consolidated_row.update(summary)
            consolidated_rows.append(consolidated_row)

            if dry_run:
                LOGGER.info("[DRY RUN] Would write per-trial CSV: %s", output_path)
                continue

            ensure_parent(output_path, dry_run=False)
            enriched_df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
            LOGGER.info("Wrote enriched trial CSV: %s", output_path)

    return consolidated_rows


def write_consolidated(consolidated_rows: Sequence[Dict[str, object]], outdir: Path, dry_run: bool) -> None:
    if dry_run:
        LOGGER.info("[DRY RUN] Would write consolidated CSV with %d rows", len(consolidated_rows))
        return

    outdir.mkdir(parents=True, exist_ok=True)
    consolidated_path = outdir / "geom_features_all_flies.csv"
    df = pd.DataFrame(consolidated_rows)
    if df.empty:
        raise ValueError("Consolidated dataframe is empty; no trials processed.")
    df.to_csv(consolidated_path, index=False, quoting=csv.QUOTE_MINIMAL)
    LOGGER.info("Wrote consolidated CSV: %s", consolidated_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)

    outdir = Path(args.outdir).resolve()
    LOGGER.info("Output directory: %s", outdir)

    trials = discover_trials(args.roots)
    trials = apply_limits(trials, args.limit_flies, args.limit_trials)

    if not trials:
        LOGGER.warning("No trials to process.")
        return

    consolidated_rows = process_trials(trials, outdir, args.dry_run)

    if consolidated_rows:
        write_consolidated(consolidated_rows, outdir, args.dry_run)
    else:
        LOGGER.warning("No consolidated rows generated.")


if __name__ == "__main__":
    main()

