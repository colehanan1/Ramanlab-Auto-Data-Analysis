"""CLI tool to compute geometric features for fly trials.

This script recursively searches for trial CSV files containing eye and
proboscis coordinates, computes per-frame geometric features and per-fly
normalisation statistics, and emits enriched per-trial CSVs alongside a
consolidated summary CSV. Testing trials are additionally streamed into a
single aggregated CSV for downstream batch analysis with a companion schema
manifest so large cohorts remain easy to load.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


TRIAL_TYPE_PATTERN = re.compile(r"(training|testing)", re.IGNORECASE)
TRIAL_LABEL_PATTERN = re.compile(r"(training|testing)_[^/\\]+", re.IGNORECASE)
YOLO_CSV_PATTERN = re.compile(r"_distances\.csv$", re.IGNORECASE)
FLY_SLOT_PATTERN = re.compile(r"(fly\d+)", re.IGNORECASE)
FLY_NUMBER_PATTERN = re.compile(r"fly[^0-9]*(\d+)", re.IGNORECASE)
WIDE_COL_PATTERN = re.compile(r"(?P<prefix>eye_x|eye_y|prob_x|prob_y)_f(?P<frame>\d+)", re.IGNORECASE)


BASELINE_END = 1260  # first ~30 seconds at 40 fps
STIM_END = 2460  # subsequent ~30 seconds during odor presentation
FPS = 40.0
EARLY_WIN_SEC = 1.0
EARLY_WIN = int(EARLY_WIN_SEC * FPS)
HIGH_EXT_THRESH = 75.0


METADATA_COLUMNS = [
    "dataset",
    "fly",
    "fly_number",
    "trial_type",
    "trial_label",
]

FLY_STATS_COLUMNS = [
    "W_est_fly",
    "H_est_fly",
    "diag_est_fly",
    "r_min_fly",
    "r_max_fly",
    "r_p01_fly",
    "r_p99_fly",
    "r_mean_fly",
    "r_std_fly",
]

SUMMARY_COLUMNS = [
    "n_frames",
    "r_mean_trial",
    "r_std_trial",
    "r_max_trial",
    "r95_trial",
    "dx_mean_abs",
    "dy_mean_abs",
    "r_pct_robust_fly_max",
    "r_pct_robust_fly_mean",
    "r_before_mean",
    "r_before_std",
    "r_during_mean",
    "r_during_std",
    "r_during_minus_before_mean",
    "cos_theta_during_mean",
    "sin_theta_during_mean",
    "direction_consistency",
    "frac_high_ext_during",
    "rise_speed",
]

PER_FRAME_COLUMNS = [
    "frame",
    "eye_x",
    "eye_y",
    "prob_x",
    "prob_y",
    "dx",
    "dy",
    "r",
    "cos_theta",
    "sin_theta",
    "dx_over_W_fly",
    "dy_over_H_fly",
    "r_over_diag_fly",
    "r_pct_minmax_fly",
    "r_pct_robust_fly",
    "r_z_fly",
    "is_before",
    "is_during",
    "is_after",
]


@dataclass(frozen=True)
class TrialInfo:
    """Metadata about a discovered trial CSV."""

    dataset: str
    fly_directory: str
    fly_path: Path
    fly_number: int
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
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--input",
        help="Process a single CSV to append behavioural window metrics and write *_with_behavior.csv.",
    )
    mode_group.add_argument(
        "--roots",
        nargs="+",
        help="Root directories to scan for trial CSV files.",
    )
    parser.add_argument(
        "--outdir",
        help="Directory for consolidated output and optional per-trial outputs (required with --roots).",
    )
    parser.add_argument("--dry-run", action="store_true", help="List processing actions without writing outputs.")
    parser.add_argument("--limit-flies", type=int, default=None, help="Process at most N flies per dataset root.")
    parser.add_argument("--limit-trials", type=int, default=None, help="Process at most N trials per fly.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default INFO).")
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")


def resolve_fly_number(fly_directory: str, fly_slot: str) -> int:
    """Infer the numeric fly identifier from directory or slot names."""

    for candidate in (fly_slot, fly_directory):
        if not candidate:
            continue
        match = FLY_NUMBER_PATTERN.search(candidate)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                LOGGER.debug("Failed to parse fly number from %s", candidate)
                continue
    LOGGER.debug("Defaulting fly number to 1 for directory=%s slot=%s", fly_directory, fly_slot)
    return 1


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
                fly_number = resolve_fly_number(fly_directory, fly_slot)
                trials.append(
                    TrialInfo(
                        dataset=dataset,
                        fly_directory=fly_directory,
                        fly_path=fly_dir,
                        fly_number=fly_number,
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

    # YOLO post-processing schema where eye is class2 and proboscis is class8
    yolo_eye_candidates = ["x_class2", "x_class_2", "class2_x"]
    yolo_eye_y_candidates = ["y_class2", "y_class_2", "class2_y"]
    yolo_prob_x_candidates = ["x_class8", "x_class_8", "class8_x"]
    yolo_prob_y_candidates = ["y_class8", "y_class_8", "class8_y"]

    def _resolve(candidates: Sequence[str]) -> Optional[str]:
        for candidate in candidates:
            if candidate in columns_lower:
                return columns_lower[candidate]
        return None

    frame_col = columns_lower.get("frame")
    eye_x_col = _resolve(yolo_eye_candidates)
    eye_y_col = _resolve(yolo_eye_y_candidates)
    prob_x_col = _resolve(yolo_prob_x_candidates)
    prob_y_col = _resolve(yolo_prob_y_candidates)

    if frame_col and eye_x_col and eye_y_col and prob_x_col and prob_y_col:
        selected = df[[frame_col, eye_x_col, eye_y_col, prob_x_col, prob_y_col]].copy()
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


def _safe_stat(series: pd.Series, mask: np.ndarray, func) -> float:
    """Apply ``func`` to ``series`` filtered by ``mask`` returning NaN when empty."""

    if series.empty:
        return float("nan")
    filtered = series[mask]
    if filtered.empty:
        return float("nan")
    value = func(filtered)
    if isinstance(value, np.ndarray):
        value = value.item()
    return float(value)


def _apply_behavioural_windows(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, int]]:
    """Append behavioural window masks and summary statistics to an enriched dataframe."""

    if "frame" not in df.columns:
        LOGGER.warning("Dataframe lacks frame column; behavioural windows will be empty.")
        frame_values = pd.Series(np.arange(len(df)), index=df.index, dtype=float)
    else:
        frame_values = pd.to_numeric(df["frame"], errors="coerce")

    if "r_pct_robust_fly" not in df.columns:
        raise KeyError("r_pct_robust_fly column required to compute behavioural metrics.")

    is_valid = frame_values.notna().to_numpy()
    frames = frame_values.to_numpy()

    before_mask = (frames < BASELINE_END) & is_valid
    during_mask = (frames >= BASELINE_END) & (frames < STIM_END) & is_valid
    after_mask = (frames >= STIM_END) & is_valid
    early_mask = (frames >= BASELINE_END) & (frames < BASELINE_END + EARLY_WIN) & is_valid

    df = df.copy()
    before_mask = before_mask.astype(bool)
    during_mask = during_mask.astype(bool)
    after_mask = after_mask.astype(bool)
    early_mask = early_mask.astype(bool)

    df["is_before"] = before_mask.astype(np.int8)
    df["is_during"] = during_mask.astype(np.int8)
    df["is_after"] = after_mask.astype(np.int8)

    r_pct = pd.to_numeric(df["r_pct_robust_fly"], errors="coerce") if "r_pct_robust_fly" in df else pd.Series(np.nan, index=df.index)
    cos_theta = pd.to_numeric(df.get("cos_theta", pd.Series(np.nan, index=df.index)), errors="coerce")
    sin_theta = pd.to_numeric(df.get("sin_theta", pd.Series(np.nan, index=df.index)), errors="coerce")

    r_before_mean = _safe_stat(r_pct, before_mask, np.nanmean)
    r_before_std = _safe_stat(r_pct, before_mask, np.nanstd)
    r_during_mean = _safe_stat(r_pct, during_mask, np.nanmean)
    r_during_std = _safe_stat(r_pct, during_mask, np.nanstd)
    r_during_minus_before_mean = (
        float(r_during_mean - r_before_mean)
        if not math.isnan(r_during_mean) and not math.isnan(r_before_mean)
        else float("nan")
    )

    cos_theta_during_mean = _safe_stat(cos_theta, during_mask, np.nanmean)
    sin_theta_during_mean = _safe_stat(sin_theta, during_mask, np.nanmean)
    if math.isnan(cos_theta_during_mean) or math.isnan(sin_theta_during_mean):
        direction_consistency = float("nan")
    else:
        direction_consistency = float(np.sqrt(cos_theta_during_mean ** 2 + sin_theta_during_mean ** 2))

    high_ext_mask = during_mask & (r_pct.to_numpy() >= HIGH_EXT_THRESH)
    during_frames = int(during_mask.sum())
    if during_frames == 0:
        frac_high_ext_during = float("nan")
    else:
        frac_high_ext_during = float(np.clip(high_ext_mask.sum() / (during_frames + 1e-6), 0.0, 1.0))

    r_early_mean = _safe_stat(r_pct, early_mask, np.nanmean)
    if math.isnan(r_early_mean) or math.isnan(r_before_mean):
        rise_speed = float("nan")
    else:
        rise_speed = float((r_early_mean - r_before_mean) / (EARLY_WIN_SEC + 1e-6))

    summary_updates: Dict[str, float] = {
        "r_before_mean": r_before_mean,
        "r_before_std": r_before_std,
        "r_during_mean": r_during_mean,
        "r_during_std": r_during_std,
        "r_during_minus_before_mean": r_during_minus_before_mean,
        "cos_theta_during_mean": cos_theta_during_mean,
        "sin_theta_during_mean": sin_theta_during_mean,
        "direction_consistency": direction_consistency,
        "frac_high_ext_during": frac_high_ext_during,
        "rise_speed": rise_speed,
    }

    for column, value in summary_updates.items():
        df[column] = value

    counts = {
        "frames_before": int(before_mask.sum()),
        "frames_during": int(during_mask.sum()),
        "frames_after": int(after_mask.sum()),
        "frames_early": int(early_mask.sum()),
    }

    LOGGER.debug(
        "Window diagnostics -> before: %d, during: %d, after: %d, early: %d",  # type: ignore[str-format]
        counts["frames_before"],
        counts["frames_during"],
        counts["frames_after"],
        counts["frames_early"],
    )

    return df, summary_updates, counts


def enrich_trial(
    trial: TrialInfo,
    df: pd.DataFrame,
    stats: FlyStats,
) -> Tuple[pd.DataFrame, Dict[str, float | int]]:
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

    summary: Dict[str, float | int] = {
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

    enriched, behaviour_summary, window_counts = _apply_behavioural_windows(enriched)

    summary.update(behaviour_summary)
    LOGGER.debug(
        "Trial %s window counts -> before=%d during=%d after=%d early=%d",
        trial.csv_path_in,
        window_counts["frames_before"],
        window_counts["frames_during"],
        window_counts["frames_after"],
        window_counts["frames_early"],
    )

    enriched["dataset"] = trial.dataset
    enriched["fly"] = trial.fly_directory
    enriched["fly_number"] = int(trial.fly_number)
    enriched["trial_type"] = trial.trial_type
    enriched["trial_label"] = trial.trial_label

    return enriched, summary


def determine_output_path(trial: TrialInfo, outdir: Path) -> Path:
    relative_parts = [trial.dataset, trial.fly_directory, trial.trial_label]
    filename = trial.csv_path_in.stem + "_geom.csv"
    return outdir.joinpath(*relative_parts, filename)


def ensure_directory(path: Path) -> None:
    """Ensure that ``path`` exists, raising with guidance on permission failures."""

    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:  # pragma: no cover - depends on environment permissions
        raise PermissionError(
            f"Cannot create directory {path}: {exc}. "
            "Ensure --outdir points to a writable location or pre-create the directory with correct permissions."
        ) from exc
    except OSError as exc:  # pragma: no cover - unexpected filesystem errors
        raise OSError(f"Failed to create directory {path}: {exc}") from exc


def ensure_parent(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    ensure_directory(path.parent)


class TestingAggregator:
    """Stream enriched testing-trial frames into a single CSV with schema metadata."""

    def __init__(self, outdir: Path, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.path = outdir / "geom_features_testing_all_frames.csv"
        self.schema_path = outdir / "geom_features_testing_all_frames.schema.json"
        self._header_written = False
        self._column_order: Optional[List[str]] = None
        self._column_groups: Optional[Dict[str, List[str]]] = None
        self.rows_written = 0

        if self.dry_run:
            LOGGER.info(
                "[DRY RUN] Would create aggregated testing file at %s",
                self.path,
            )
            return

        ensure_directory(self.path.parent)
        for existing in (self.path, self.schema_path):
            if existing.exists():
                existing.unlink()
                LOGGER.info("Overwriting existing testing artefact: %s", existing)

    def append(
        self,
        trial: TrialInfo,
        df: pd.DataFrame,
        summary: Dict[str, float | int],
        stats: FlyStats,
    ) -> None:
        if self.dry_run:
            LOGGER.info(
                "[DRY RUN] Would append %d rows from %s to %s",
                len(df),
                trial.csv_path_in,
                self.path,
            )
            return

        trimmed_df = self._trim_frames(df, trial)
        if trimmed_df.empty:
            LOGGER.warning(
                "Skipping testing aggregate append for %s because no frames remain within 0-3600",
                trial.csv_path_in,
            )
            return

        metadata = {
            "dataset": trial.dataset,
            "fly": trial.fly_directory,
            "fly_number": int(trial.fly_number),
            "trial_type": trial.trial_type,
            "trial_label": trial.trial_label,
        }
        fly_metrics = asdict(stats)

        augmented = trimmed_df.assign(**metadata, **fly_metrics, **summary)
        column_order = self._ensure_column_order(augmented)

        augmented = augmented.reindex(columns=column_order)
        mode = "w" if not self._header_written else "a"
        augmented.to_csv(
            self.path,
            index=False,
            mode=mode,
            header=not self._header_written,
            quoting=csv.QUOTE_MINIMAL,
        )
        self._header_written = True
        self.rows_written += len(augmented)
        LOGGER.info(
            "Appended %d testing frames to %s (total rows: %d)",
            len(augmented),
            self.path,
            self.rows_written,
        )

    def finalize(self) -> None:
        if self.dry_run:
            return
        if self.rows_written == 0:
            if self.path.exists():
                self.path.unlink()
            if self.schema_path.exists():
                self.schema_path.unlink()
            LOGGER.info("No testing trials found; skipping aggregated testing CSV.")
        else:
            LOGGER.info(
                "Finalised testing aggregate with %d rows and schema %s",
                self.rows_written,
                self.schema_path,
            )

    def _ensure_column_order(self, df: pd.DataFrame) -> List[str]:
        if self._column_order is not None:
            return self._column_order

        metadata_cols = METADATA_COLUMNS
        fly_cols = [col for col in FLY_STATS_COLUMNS if col in df.columns]
        summary_cols = [col for col in SUMMARY_COLUMNS if col in df.columns]

        base_per_frame = [col for col in PER_FRAME_COLUMNS if col in df.columns]
        excluded = set(metadata_cols + fly_cols + summary_cols + base_per_frame)
        additional_per_frame = [
            col for col in df.columns if col not in excluded
        ]
        per_frame_cols = base_per_frame + additional_per_frame

        self._column_order = metadata_cols + fly_cols + summary_cols + per_frame_cols
        self._column_groups = {
            "metadata": metadata_cols,
            "fly_metrics": fly_cols,
            "trial_summary": summary_cols,
            "per_frame": per_frame_cols,
        }
        self._write_schema(df)
        return self._column_order

    def _write_schema(self, df: pd.DataFrame) -> None:
        if self.dry_run or self._column_order is None or self._column_groups is None:
            return

        schema = {
            "column_order": self._column_order,
            "column_groups": self._column_groups,
            "dtypes": {col: str(df[col].dtype) for col in self._column_order},
        }
        with self.schema_path.open("w", encoding="utf-8") as handle:
            json.dump(schema, handle, indent=2, sort_keys=True)
        LOGGER.info("Wrote testing aggregate schema: %s", self.schema_path)

    def _trim_frames(self, df: pd.DataFrame, trial: TrialInfo) -> pd.DataFrame:
        """Keep only frames with indices between 0 and 3600 inclusive."""

        if "frame" not in df.columns:
            LOGGER.warning(
                "Trial %s lacks a frame column; writing only the first 3601 rows to testing aggregate",
                trial.csv_path_in,
            )
            return df.head(3601).copy()

        mask = df["frame"].between(0, 3600, inclusive="both")
        trimmed = df.loc[mask]
        if trimmed.empty:
            # Fallback to first 3601 rows to avoid dropping the trial entirely if
            # frame numbering is unexpected.
            LOGGER.warning(
                "Trial %s has no frames in [0, 3600]; defaulting to first 3601 rows",
                trial.csv_path_in,
            )
            return df.head(3601).copy()
        return trimmed.copy()


def process_single_trial(
    input_csv_path: str,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, int]]:
    """Load a CSV, append behavioural windows, and return the enriched dataframe."""

    path = Path(input_csv_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")

    df = pd.read_csv(path)
    enriched, summary_updates, counts = _apply_behavioural_windows(df)
    return enriched, summary_updates, counts


def process_trials(
    trials: Sequence[TrialInfo],
    outdir: Path,
    dry_run: bool,
    testing_aggregator: Optional[TestingAggregator],
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
                "fly": trial.fly_directory,
                "fly_number": int(trial.fly_number),
                "trial_type": trial.trial_type,
                "trial_label": trial.trial_label,
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

            behaviour_path = trial.csv_path_in.with_name(trial.csv_path_in.stem + "_with_behavior.csv")

            if dry_run:
                LOGGER.info("[DRY RUN] Would write per-trial CSV: %s", output_path)
                LOGGER.info("[DRY RUN] Would write behavioural CSV: %s", behaviour_path)
                continue

            ensure_parent(output_path, dry_run=False)
            enriched_df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
            LOGGER.info("Wrote enriched trial CSV: %s", output_path)

            enriched_df.to_csv(behaviour_path, index=False, quoting=csv.QUOTE_MINIMAL)
            LOGGER.info("Wrote behavioural trial CSV: %s", behaviour_path)

            if testing_aggregator is not None and trial.trial_type == "testing":
                testing_aggregator.append(trial, enriched_df, summary, stats)

    return consolidated_rows


def write_consolidated(consolidated_rows: Sequence[Dict[str, object]], outdir: Path, dry_run: bool) -> None:
    if dry_run:
        LOGGER.info("[DRY RUN] Would write consolidated CSV with %d rows", len(consolidated_rows))
        return

    ensure_directory(outdir)
    consolidated_path = outdir / "geom_features_all_flies.csv"
    df = pd.DataFrame(consolidated_rows)
    if df.empty:
        raise ValueError("Consolidated dataframe is empty; no trials processed.")
    df.to_csv(consolidated_path, index=False, quoting=csv.QUOTE_MINIMAL)
    LOGGER.info("Wrote consolidated CSV: %s", consolidated_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)

    if args.input:
        if args.limit_flies is not None or args.limit_trials is not None:
            LOGGER.warning("--limit-flies/--limit-trials are ignored when --input is provided.")
        if args.outdir:
            LOGGER.info("--outdir is ignored in --input mode; outputs are written beside the source CSV.")

        enriched_df, behaviour_summary, counts = process_single_trial(args.input)
        behaviour_path = Path(args.input).with_name(Path(args.input).stem + "_with_behavior.csv")

        if args.dry_run:
            LOGGER.info("[DRY RUN] Would write behavioural CSV: %s", behaviour_path)
        else:
            enriched_df.to_csv(behaviour_path, index=False, quoting=csv.QUOTE_MINIMAL)
            LOGGER.info("Wrote behavioural trial CSV: %s", behaviour_path)

        LOGGER.info(
            "Window counts -> before=%d during=%d after=%d early=%d",
            counts["frames_before"],
            counts["frames_during"],
            counts["frames_after"],
            counts["frames_early"],
        )
        LOGGER.info(
            "Behaviour metrics -> r_before_mean=%.3f r_during_mean=%.3f frac_high_ext_during=%.3f rise_speed=%.3f direction_consistency=%.3f",
            behaviour_summary["r_before_mean"],
            behaviour_summary["r_during_mean"],
            behaviour_summary["frac_high_ext_during"],
            behaviour_summary["rise_speed"],
            behaviour_summary["direction_consistency"],
        )
        return

    if not args.outdir:
        raise ValueError("--outdir is required when processing dataset roots.")

    outdir = Path(args.outdir).resolve()
    LOGGER.info("Output directory: %s", outdir)

    if not args.dry_run:
        ensure_directory(outdir)

    if args.roots is None:
        LOGGER.warning("No dataset roots provided.")
        return

    trials = discover_trials(args.roots)
    trials = apply_limits(trials, args.limit_flies, args.limit_trials)

    if not trials:
        LOGGER.warning("No trials to process.")
        return

    testing_aggregator = TestingAggregator(outdir, args.dry_run)

    consolidated_rows = process_trials(trials, outdir, args.dry_run, testing_aggregator)

    if consolidated_rows:
        write_consolidated(consolidated_rows, outdir, args.dry_run)
    else:
        LOGGER.warning("No consolidated rows generated.")

    testing_aggregator.finalize()


if __name__ == "__main__":
    main()

