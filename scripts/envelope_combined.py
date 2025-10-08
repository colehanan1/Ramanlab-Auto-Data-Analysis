#!/usr/bin/env python3
"""Combined distance/angle analytics derived from lab notebooks.

This CLI exposes the distance-percentage × angle workflow that previously
lived in ad-hoc notebooks.  It can

* merge raw testing trials into combined RMS + Hilbert envelope CSVs/PNGs,
* copy datasets into secured storage and prune the originals,
* aggregate the direction-value envelopes into a float16 matrix, and
* feed that matrix into the existing reaction-matrix / envelope exporters or
  an overlay plot comparing distance-only versus combined signals.

All commands emphasise single-pass scans and streaming writes so large folders
run efficiently.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.signal import hilbert

from scripts import envelope_visuals
from scripts.envelope_visuals import (
    EnvelopePlotConfig,
    MatrixPlotConfig,
    generate_envelope_plots,
    generate_reaction_matrices,
    resolve_dataset_output_dir,
    should_write,
)
from fbpipe.utils.csvs import extract_fly_slot, gather_distance_csvs


MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)

ANGLE_COLS = ["angle_centered_pct", "angle_centered_percentage", "angle_pct"]
DIST_COLS = [
    "distance_percentage_2_8",
    "distance_percentage",
    "distance_percent",
    "distance_pct",
    "distance_percentage_2_6",
    "measure",
    "value",
]
TIME_COLS = ["time_s", "time_seconds", "t_s", "time"]
TIMESTAMP_COLS = ["UTC_ISO", "Timestamp", "Number", "MonoNs"]
FRAME_COLS = ["Frame", "FrameNumber", "Frame Number"]
TRIAL_REGEX = re.compile(r"(testing|training)_(\d+)", re.IGNORECASE)
TESTING_REGEX = re.compile(r"testing_(\d+)", re.IGNORECASE)

ANCHOR_X = 1080.0
ANCHOR_Y = 540.0

MANDATORY_WIDE_EXCLUDES = {
    Path("/securedstorage/DATAsec/cole/Data-secured/opto_benz/").expanduser().resolve()
}


@dataclass(frozen=True)
class TrialEntry:
    """Descriptor for a per-trial CSV candidate."""

    label: str
    path: Path
    category: str
    fly_slot: Optional[int] = None
    distance_variant: Optional[str] = None

ODOR_CANON = {
    "acv": "ACV",
    "apple cider vinegar": "ACV",
    "apple-cider-vinegar": "ACV",
    "3-octonol": "3-octonol",
    "3 octonol": "3-octonol",
    "3-octanol": "3-octonol",
    "3 octanol": "3-octonol",
    "benz": "Benz",
    "benzaldehyde": "Benz",
    "benz-ald": "Benz",
    "benzadhyde": "Benz",
    "ethyl butyrate": "EB",
    "optogenetics benzaldehyde": "opto_benz",
    "optogenetics benzaldehyde 1": "opto_benz_1",
    "optogenetics ethyl butyrate": "opto_EB",
    "10s_odor_benz": "10s_Odor_Benz",
    "optogenetics hexanol": "opto_hex",
    "optogenetics hex": "opto_hex",
    "hexanol": "opto_hex",
    "opto_hex": "opto_hex",
}

DISPLAY_LABEL = {
    "ACV": "ACV",
    "3-octonol": "3-Octonol",
    "Benz": "Benzaldehyde",
    "10s_Odor_Benz": "Benzaldehyde",
    "EB": "Ethyl Butyrate",
    "opto_benz": "Benzaldehyde",
    "opto_benz_1": "Benzaldehyde",
    "opto_EB": "Ethyl Butyrate",
    "opto_hex": "Optogenetics Hexanol",
}

HEXANOL = "Optogenetics Hexanol"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _normalise_roots(roots: Sequence[str | os.PathLike[str]]) -> list[Path]:
    resolved: list[Path] = []
    for root in roots:
        path = Path(root).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Not a directory: {path}")
        resolved.append(path)
    return resolved


def _safe_dirname(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "export"


def _canon_dataset(value: str) -> str:
    if not isinstance(value, str):
        return "UNKNOWN"
    return ODOR_CANON.get(value.strip().lower(), value.strip())


def _trial_num(label: str) -> int:
    match = re.search(r"(\d+)", str(label))
    return int(match.group(1)) if match else -1


def _display_odor(dataset_canon: str, trial_label: str) -> str:
    number = _trial_num(trial_label)
    label_lower = str(trial_label).lower()
    if (
        dataset_canon == "opto_hex"
        and "testing" in label_lower
        and number in (1, 3)
    ):
        return "Apple Cider Vinegar"
    if number in (1, 3):
        return HEXANOL
    if number in (2, 4, 5):
        return DISPLAY_LABEL.get(dataset_canon, dataset_canon)

    mapping = {
        "ACV": {6: "3-Octonol", 7: "Benzaldehyde", 8: "Citral", 9: "Linalool"},
        "3-octonol": {6: "Benzaldehyde", 7: "Citral", 8: "Linalool"},
        "Benz": {6: "Citral", 7: "Linalool"},
        "EB": {6: "Apple Cider Vinegar", 7: "3-Octonol", 8: "Benzaldehyde", 9: "Citral", 10: "Linalool"},
        "10s_Odor_Benz": {6: "Benzaldehyde", 7: "Benzaldehyde"},
        "opto_EB": {6: "Apple Cider Vinegar", 7: "3-Octonol", 8: "Benzaldehyde", 9: "Citral", 10: "Linalool"},
        "opto_benz": {6: "3-Octonol", 7: "Benzaldehyde", 8: "Citral", 9: "Linalool"},
        "opto_benz_1": {6: "Apple Cider Vinegar", 7: "3-Octonol", 8: "Ethyl Butyrate", 9: "Citral", 10: "Linalool"},
        "opto_hex": {
            6: "Benzaldehyde",
            7: "3-Octonol",
            8: "Ethyl Butyrate",
            9: "Citral",
            10: "Linalool",
        },
    }
    return mapping.get(dataset_canon, {}).get(number, trial_label)


def _is_trained(dataset_canon: str, odor_name: str) -> bool:
    trained = DISPLAY_LABEL.get(dataset_canon, dataset_canon)
    return odor_name.strip().lower() == trained.strip().lower()


def _pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _time_axis(df: pd.DataFrame, fps_default: float) -> np.ndarray:
    col = _pick_column(df, TIME_COLS)
    if col:
        return pd.to_numeric(df[col], errors="coerce").to_numpy()
    if "frame" in df.columns:
        frames = pd.to_numeric(df["frame"], errors="coerce").to_numpy()
        return frames / max(fps_default, 1e-9)
    return np.arange(len(df), dtype=float) / max(fps_default, 1e-9)


def _rolling_rms(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(pd.to_numeric(values, errors="coerce"), copy=False).fillna(0.0)
    return (
        series.pow(2)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .pow(0.5)
        .to_numpy()
    )


def _hilbert_envelope(values: np.ndarray, window: int) -> np.ndarray:
    env = np.abs(hilbert(np.nan_to_num(values, nan=0.0)))
    return (
        pd.Series(env)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def _angle_multiplier(angle_pct: np.ndarray) -> np.ndarray:
    pct = np.asarray(angle_pct, dtype=float)
    pct = np.clip(pct, -100.0, 100.0)
    conditions = [
        pct < -40,
        (pct >= -40) & (pct < -25),
        (pct >= -25) & (pct < -10),
        (pct >= -10) & (pct <= 10),
        (pct > 10) & (pct <= 25),
        (pct > 25) & (pct <= 40),
        (pct > 40) & (pct <= 60),
        (pct > 60) & (pct <= 100),
    ]
    multipliers = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
    return np.select(conditions, multipliers, default=np.nan)


def _is_month_folder(path: Path) -> bool:
    return path.is_dir() and path.name.lower().startswith(MONTHS)


def _infer_category(path: Path) -> str:
    parts = [segment.lower() for segment in path.parts]
    if "testing" in parts or "testing" in path.name.lower():
        return "testing"
    if "training" in parts or "training" in path.name.lower():
        return "training"
    return "testing"


def _trial_label(path: Path) -> str:
    match = TRIAL_REGEX.search(path.stem)
    if not match:
        chain = f"{path.stem}/" + "/".join(parent.name for parent in path.parents)
        match = TRIAL_REGEX.search(chain)
    if match:
        return f"{match.group(1).lower()}_{match.group(2)}"
    trailing = re.search(r"(\d+)$", path.stem)
    if trailing:
        return f"{_infer_category(path)}_{trailing.group(1)}"
    return path.stem


def _trial_csv_candidates(
    fly_dir: Path,
    suffix_globs: Iterable[str] | str,
    *,
    prefer_distance: bool = False,
) -> list[Path]:
    patterns = [suffix_globs] if isinstance(suffix_globs, str) else list(suffix_globs)
    if not patterns:
        return []

    pattern_lower = [pat.lower() for pat in patterns]
    month_dirs = [p for p in fly_dir.rglob("*") if _is_month_folder(p)]
    search_roots = month_dirs or [fly_dir]

    def _matches(path: Path) -> bool:
        name = path.name.lower()
        return any(fnmatch.fnmatch(name, pat) for pat in pattern_lower)

    candidates: list[Path] = []
    seen: set[Path] = set()

    if prefer_distance:
        for root in search_roots:
            for path in gather_distance_csvs(root):
                if not path.is_file():
                    continue
                if not _matches(path):
                    continue
                name_lower = path.name.lower()
                if "training" not in name_lower and "testing" not in name_lower:
                    continue
                real = path.resolve()
                if real in seen:
                    continue
                seen.add(real)
                candidates.append(path)
    else:
        for root in search_roots:
            for pattern in patterns:
                for path in root.rglob(pattern):
                    if not path.is_file():
                        continue
                    name_lower = path.name.lower()
                    if "training" not in name_lower and "testing" not in name_lower:
                        continue
                    real = path.resolve()
                    if real in seen:
                        continue
                    seen.add(real)
                    candidates.append(path)

    return candidates if prefer_distance else sorted(candidates)


def _locate_trials(
    fly_dir: Path,
    suffix_globs: Iterable[str] | str,
    required_cols: Sequence[str],
    *,
    prefer_distance: bool = False,
) -> list[TrialEntry]:
    candidates = _trial_csv_candidates(
        fly_dir,
        suffix_globs,
        prefer_distance=prefer_distance,
    )

    results: list[TrialEntry] = []
    for csv_path in candidates:
        try:
            header = pd.read_csv(csv_path, nrows=5)
        except Exception:
            continue
        if not _pick_column(header, required_cols):
            continue

        slot = extract_fly_slot(csv_path) if prefer_distance else None
        variant = None
        if prefer_distance:
            variant = f"fly{slot}" if slot is not None else "merged"

        results.append(
            TrialEntry(
                label=csv_path.stem,
                path=csv_path,
                category=_infer_category(csv_path),
                fly_slot=slot,
                distance_variant=variant,
            )
        )
    return results


def _normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _resolve_column_alias(df: pd.DataFrame, *aliases: str) -> str | None:
    lookup = {_normalise_key(col): col for col in df.columns}
    for alias in aliases:
        column = lookup.get(_normalise_key(alias))
        if column:
            return column
    return None


def _compute_angle_deg(df: pd.DataFrame) -> pd.Series:
    x2_col = _resolve_column_alias(df, "x_class2", "x_class_2", "class2_x")
    y2_col = _resolve_column_alias(df, "y_class2", "y_class_2", "class2_y")
    x_prob_col = _resolve_column_alias(
        df,
        "x_class8",
        "x_class_8",
        "class8_x",
        "x_proboscis",
        "x_class6",
        "x_class_6",
        "class6_x",
    )
    y_prob_col = _resolve_column_alias(
        df,
        "y_class8",
        "y_class_8",
        "class8_y",
        "y_proboscis",
        "y_class6",
        "y_class_6",
        "class6_y",
    )
    if not all((x2_col, y2_col, x_prob_col, y_prob_col)):
        raise ValueError("Missing class2/proboscis coordinate columns for angle computation.")

    p2x = pd.to_numeric(df[x2_col], errors="coerce").astype(float)
    p2y = pd.to_numeric(df[y2_col], errors="coerce").astype(float)
    p3x = pd.to_numeric(df[x_prob_col], errors="coerce").astype(float)
    p3y = pd.to_numeric(df[y_prob_col], errors="coerce").astype(float)

    ux = ANCHOR_X - p2x
    uy = ANCHOR_Y - p2y
    vx = p3x - p2x
    vy = p3y - p2y

    dot = ux * vx + uy * vy
    cross = ux * vy - uy * vx
    n1 = np.hypot(ux, uy)
    n2 = np.hypot(vx, vy)
    valid = (n1 > 0) & (n2 > 0) & np.isfinite(dot) & np.isfinite(cross)

    angles = np.full(len(df), np.nan, dtype=float)
    if valid.any():
        with np.errstate(invalid="ignore"):
            ang = np.arctan2(np.abs(cross[valid]), dot[valid])
        angles[valid.to_numpy()] = np.degrees(ang)

    return pd.Series(angles, index=df.index, name="angle_ARB_deg")
def _find_reference_angle(csv_paths: Sequence[Path]) -> float:
    best: tuple[int, float, float] | None = None
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            angles = _compute_angle_deg(df).to_numpy(dtype=float)
        except Exception:
            continue

        dist_col = _pick_column(df, DIST_COLS)
        if not dist_col:
            continue
        dist = pd.to_numeric(df[dist_col], errors="coerce").to_numpy(dtype=float)
        if dist.size == 0:
            continue

        exact = np.flatnonzero(np.isfinite(dist) & (dist == 0))
        candidate: tuple[int, float, float] | None = None
        if exact.size > 0:
            idx = int(exact[0])
            angle_val = angles[idx] if np.isfinite(angles[idx]) else np.nan
            if np.isfinite(angle_val):
                candidate = (0, 0.0, float(angle_val))
        else:
            with np.errstate(invalid="ignore"):
                absdist = np.abs(dist)
            if not np.isfinite(absdist).any():
                continue
            idx = int(np.nanargmin(absdist))
            angle_val = angles[idx] if np.isfinite(angles[idx]) else np.nan
            if np.isfinite(angle_val):
                candidate = (1, float(absdist[idx]), float(angle_val))

        if candidate is None:
            continue
        if best is None or candidate < best:
            best = candidate

    return float("nan") if best is None else best[2]


def _fly_max_centered(csv_paths: Sequence[Path], reference_angle: float) -> float:
    if not np.isfinite(reference_angle):
        return float("nan")

    max_abs = 0.0
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            angles = _compute_angle_deg(df).to_numpy(dtype=float)
        except Exception:
            continue

        centered = angles - reference_angle
        with np.errstate(invalid="ignore"):
            local = np.nanmax(np.abs(centered))
        if np.isfinite(local):
            max_abs = max(max_abs, float(local))

    return float("nan") if max_abs <= 0 else max_abs


def _series_matches(existing: pd.Series | None, values: np.ndarray) -> bool:
    if existing is None:
        return False
    arr_existing = pd.to_numeric(existing, errors="coerce").to_numpy(dtype=float)
    arr_values = np.asarray(values, dtype=float)
    if arr_existing.shape != arr_values.shape:
        return False
    return np.allclose(arr_existing, arr_values, equal_nan=True)


def _ensure_angle_percentages(fly_dir: Path, suffix_globs: Iterable[str] | str) -> None:
    csv_paths = _trial_csv_candidates(fly_dir, suffix_globs)
    if not csv_paths:
        return

    reference = _find_reference_angle(csv_paths)
    fly_max = _fly_max_centered(csv_paths, reference)

    if not np.isfinite(reference):
        reference = 0.0
    valid_scale = np.isfinite(fly_max) and fly_max > 0
    if not valid_scale:
        print(
            f"[WARN] {fly_dir.name}: unable to derive centered angle scale; "
            "percentages will default to 0."
        )

    updates = 0
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            angles = _compute_angle_deg(df)
        except Exception:
            continue

        angle_vals = angles.to_numpy(dtype=float)
        centered_deg = angle_vals - reference
        existing_pct_series = df.get("angle_centered_pct")
        if not valid_scale:
            centered_pct = (
                pd.to_numeric(existing_pct_series, errors="coerce").to_numpy(dtype=float)
                if existing_pct_series is not None
                else np.zeros_like(centered_deg)
            )
            if not np.isfinite(centered_pct).any():
                centered_pct = np.zeros_like(centered_deg)
        else:
            with np.errstate(invalid="ignore"):
                centered_pct = (centered_deg / fly_max) * 100.0

        centered_deg = np.nan_to_num(centered_deg, nan=0.0)
        centered_pct = np.nan_to_num(centered_pct, nan=0.0)

        needs_write = False
        if not _series_matches(df.get("angle_ARB_deg"), angle_vals):
            df["angle_ARB_deg"] = angle_vals
            needs_write = True
        if not _series_matches(df.get("angle_centered_deg"), centered_deg):
            df["angle_centered_deg"] = centered_deg
            needs_write = True
        if not _series_matches(df.get("angle_centered_pct"), centered_pct):
            df["angle_centered_pct"] = centered_pct
            needs_write = True

        if needs_write:
            try:
                df.to_csv(path, index=False)
                updates += 1
                print(f"[angle] {fly_dir.name}: augmented {path.name} with centered angle columns.")
            except Exception as exc:
                print(f"[WARN] Failed to persist centered angle columns for {path}: {exc}")

    if updates:
        print(f"[{fly_dir.name}] centered angle normalisation applied to {updates} file(s).")


def _index_testing(
    entries: Sequence[TrialEntry],
) -> tuple[dict[str, list[TrialEntry]], dict[str, list[TrialEntry]]]:
    indexed: dict[str, list[TrialEntry]] = {}
    fallback: dict[str, list[TrialEntry]] = {}
    for entry in entries:
        if entry.category.lower() != "testing":
            continue
        match = TESTING_REGEX.search(entry.label)
        if match:
            key = match.group(0).lower()
            indexed.setdefault(key, []).append(entry)
        else:
            fallback.setdefault(entry.label.lower(), []).append(entry)
    return indexed, fallback


def _find_trial_csvs(fly_dir: Path) -> Iterator[Path]:
    base = fly_dir / "angle_distance_rms_envelope"
    if not base.is_dir():
        return

    seen: set[Path] = set()
    for pattern in ("**/*testing*.csv", "**/*training*.csv"):
        for csv in base.glob(pattern):
            if not csv.is_file():
                continue
            real = csv.resolve()
            if real in seen:
                continue
            seen.add(real)
            yield real


def _pick_timestamp(df: pd.DataFrame) -> str | None:
    return _pick_column(df, TIMESTAMP_COLS)


def _pick_frame(df: pd.DataFrame) -> str | None:
    return _pick_column(df, FRAME_COLS)


def _seconds_from_timestamp(df: pd.DataFrame, column: str) -> pd.Series:
    series = df[column]
    if column in ("UTC_ISO", "Timestamp"):
        dt = pd.to_datetime(series, errors="coerce", utc=(column == "UTC_ISO"))
        secs = dt.astype("int64") / 1e9
    elif column == "Number":
        secs = pd.to_numeric(series, errors="coerce").astype(float)
    elif column == "MonoNs":
        secs = pd.to_numeric(series, errors="coerce").astype(float) / 1e9
    else:
        raise ValueError(f"Unsupported timestamp column: {column}")
    origin = np.nanmin(secs.values)
    return (secs - origin).astype(float)


def _estimate_fps(seconds: pd.Series) -> float | None:
    mask = seconds.notna()
    if mask.sum() < 2:
        return None
    duration = seconds[mask].iloc[-1] - seconds[mask].iloc[0]
    if duration <= 0:
        return None
    return mask.sum() / duration


def _extract_env(row: pd.Series, env_cols: Sequence[str]) -> np.ndarray:
    env = row[list(env_cols)].to_numpy(dtype=float)
    env = env[np.isfinite(env) & (env > 0)]
    return env


# ---------------------------------------------------------------------------
# Combined distance × angle processing
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CombineConfig:
    root: Path
    fps_default: float = 40.0
    window_sec: float = 0.25
    odor_on_s: float = 30.0
    odor_off_s: float = 60.0
    odor_latency_s: float = 0.0
    angle_suffixes: tuple[str, ...] = ("*merged.csv", "*class_2_8.csv", "*class_2_6.csv")
    distance_suffixes: tuple[str, ...] = (
        "*fly*_distances.csv",
        "*merged.csv",
        "*class_2_8.csv",
        "*class_2_6.csv",
    )

    @property
    def window_frames(self) -> int:
        return max(int(round(self.window_sec * self.fps_default)), 1)


def combine_distance_angle(cfg: CombineConfig) -> None:
    odor_latency = max(cfg.odor_latency_s, 0.0)
    odor_on_cmd = cfg.odor_on_s
    odor_off_cmd = cfg.odor_off_s
    odor_on_effective = odor_on_cmd + odor_latency
    odor_off_effective = odor_off_cmd + odor_latency

    for fly_dir in sorted(p for p in cfg.root.iterdir() if p.is_dir()):
        fly_name = fly_dir.name
        _ensure_angle_percentages(fly_dir, cfg.angle_suffixes)
        angle_entries = _locate_trials(
            fly_dir,
            cfg.angle_suffixes,
            ANGLE_COLS,
        )
        distance_entries = _locate_trials(
            fly_dir,
            cfg.distance_suffixes,
            DIST_COLS,
            prefer_distance=True,
        )

        if not distance_entries:
            print(f"[{fly_name}] No testing distance trials found — skipping.")
            continue

        angle_idx, angle_fallback = _index_testing(angle_entries)
        dist_idx, dist_fallback = _index_testing(distance_entries)

        out_csv_dir = fly_dir / "angle_distance_rms_envelope"
        out_fig_dir = out_csv_dir / "plots"
        out_csv_dir.mkdir(parents=True, exist_ok=True)
        out_fig_dir.mkdir(parents=True, exist_ok=True)

        completed = skipped = 0
        cases: list[tuple[str, TrialEntry]] = []
        for key in sorted(dist_idx):
            for entry in dist_idx[key]:
                cases.append((key, entry))
        for key in sorted(dist_fallback):
            for entry in dist_fallback[key]:
                cases.append((key, entry))

        for test_id, dist_entry in cases:
            angle_candidates = angle_idx.get(test_id)
            angle_path: Optional[Path] = angle_candidates[0].path if angle_candidates else None
            if angle_path is None:
                dist_stem = dist_entry.path.stem.lower()
                fallback_candidates = angle_fallback.get(dist_stem)
                if fallback_candidates:
                    angle_path = fallback_candidates[0].path
            if angle_path is None:
                print(
                    f"[WARN] {fly_name} {test_id}: no matching angle file for {dist_entry.path.name} — skipped."
                )
                skipped += 1
                continue

            slot = dist_entry.fly_slot if dist_entry.fly_slot is not None else 0
            variant = dist_entry.distance_variant or (f"fly{slot}" if slot else "merged")
            variant_suffix = f"_{variant}" if variant else ""
            title_variant = f" ({variant})" if variant and variant != "merged" else ""

            try:
                dist_df = pd.read_csv(dist_entry.path)
                dist_col = _pick_column(dist_df, DIST_COLS)
                if not dist_col:
                    raise ValueError("missing distance column")
                dist_pct = (
                    pd.to_numeric(dist_df[dist_col], errors="coerce")
                    .fillna(0.0)
                    .clip(lower=0.0, upper=100.0)
                    .to_numpy()
                )
                time_dist = _time_axis(dist_df, cfg.fps_default)

                angle_df = pd.read_csv(angle_path)
                angle_col = _pick_column(angle_df, ANGLE_COLS)
                if not angle_col:
                    raise ValueError("missing angle column")
                time_ang = _time_axis(angle_df, cfg.fps_default)
                angle_vals = pd.to_numeric(angle_df[angle_col], errors="coerce").to_numpy()

                order = np.argsort(time_ang)
                time_ang = time_ang[order]
                angle_vals = angle_vals[order]
                mask = np.isfinite(time_ang) & np.isfinite(angle_vals)
                if not np.any(mask):
                    raise ValueError("angle series has no finite values")
                time_ang = time_ang[mask]
                angle_vals = angle_vals[mask]

                interp_angle = np.interp(
                    time_dist, time_ang, angle_vals, left=angle_vals[0], right=angle_vals[-1]
                )
                multiplier = _angle_multiplier(interp_angle)
                combined = dist_pct * multiplier
                combined_rms = _rolling_rms(combined, cfg.window_frames)
                envelope = _hilbert_envelope(combined_rms, cfg.window_frames)

                out_df = pd.DataFrame(
                    {
                        "time_s": time_dist,
                        "angle_centered_pct_interp": interp_angle,
                        "distance_percentage": dist_pct,
                        "multiplier": multiplier,
                        "combined_base": combined,
                        "rolling_rms": combined_rms,
                        "envelope_of_rms": envelope,
                    }
                )
                out_df["fly_slot"] = slot
                out_df["distance_variant"] = variant
                out_df["source_distance_csv"] = str(dist_entry.path)
                out_df["source_angle_csv"] = str(angle_path)

                out_csv = (
                    out_csv_dir
                    / f"{test_id}{variant_suffix}_angle_distance_rms_envelope.csv"
                )
                out_df.to_csv(out_csv, index=False)

                plt.figure(figsize=(12, 4))
                plt.plot(time_dist, envelope, linewidth=1.5)
                plt.axvline(odor_on_effective, color="red", linewidth=2)
                plt.axvline(odor_off_effective, color="red", linewidth=2)
                if odor_latency > 0:
                    plt.axvspan(
                        odor_on_cmd,
                        min(odor_on_effective, time_dist[-1]),
                        alpha=0.25,
                        color="red",
                    )
                    plt.axvspan(
                        odor_off_cmd,
                        min(odor_off_effective, time_dist[-1]),
                        alpha=0.25,
                        color="red",
                    )
                plt.title(
                    f"{fly_name} — {test_id}{title_variant}: Envelope(RMS(distance × angle-mult))"
                )
                plt.xlabel("Time (s)")
                plt.ylabel("Envelope of RMS (arb.)")
                plt.margins(x=0)
                plt.grid(True, alpha=0.3)
                out_png = (
                    out_fig_dir
                    / f"{fly_name}_{test_id}{variant_suffix}_env_rms_angle_distance.png"
                )
                plt.savefig(out_png, dpi=300, bbox_inches="tight")
                plt.close()

                print(
                    f"[OK] {fly_name} {test_id}{title_variant} → CSV: {out_csv.name} | FIG: {out_png.name}"
                )
                completed += 1
            except Exception as exc:
                print(f"[WARN] {fly_name} {test_id}{title_variant} → {exc}")
                skipped += 1

        print(f"[{fly_name}] completed: {completed}, skipped: {skipped}")


# ---------------------------------------------------------------------------
# Secure copy + cleanup
# ---------------------------------------------------------------------------


def secure_copy_and_cleanup(sources: Sequence[str], destination: str) -> None:
    dest_root = Path(destination).expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    for source in _normalise_roots(sources):
        dest_path = dest_root / source.name
        print(f"\nCopying from {source} → {dest_path}")
        dest_path.mkdir(parents=True, exist_ok=True)
        for item in source.rglob("*"):
            relative = item.relative_to(source)
            target = dest_path / relative
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                print(f"Skipping (already exists): {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            print(f"Copied: {target}")

    print("\nCopy phase completed successfully.")

    for source in _normalise_roots(sources):
        print(f"\nCleaning up {source}...")
        for fly_folder in source.iterdir():
            if not fly_folder.is_dir():
                continue
            lower = fly_folder.name.lower()
            if not any(lower.startswith(month) for month in MONTHS):
                shutil.rmtree(fly_folder)
                print(f"Deleted non-month folder: {fly_folder}")
                continue
            for item in list(fly_folder.iterdir()):
                if item.name == "RMS_calculations":
                    print(f"Preserving folder: {item}")
                    continue
                if item.is_file() and item.suffix.lower() == ".csv":
                    print(f"Preserving CSV file: {item}")
                    continue
                if item.is_file():
                    item.unlink()
                    print(f"Deleted file: {item}")
                elif item.is_dir():
                    shutil.rmtree(item)
                    print(f"Deleted folder: {item}")

    print("\nCleanup completed successfully.")


def mirror_directory(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> tuple[int, int]:
    """Mirror *source* into *destination*, copying new or changed files.

    Returns a tuple of (files_copied, bytes_copied).
    """

    src = Path(source).expanduser().resolve()
    dest = Path(destination).expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Mirror source is not a directory: {src}")

    files_copied = 0
    bytes_copied = 0
    for path in src.rglob("*"):
        relative = path.relative_to(src)
        target = dest / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            needs_copy = True
            if target.exists():
                src_stat = path.stat()
                dest_stat = target.stat()
                needs_copy = not (
                    dest_stat.st_size == src_stat.st_size
                    and int(dest_stat.st_mtime) >= int(src_stat.st_mtime)
                )
            if not needs_copy:
                continue
            shutil.copy2(path, target)
            files_copied += 1
            bytes_copied += path.stat().st_size
        except Exception as exc:  # pragma: no cover - best effort sync
            print(f"[WARN] Failed to mirror {path} → {target}: {exc}")

    return files_copied, bytes_copied


# ---------------------------------------------------------------------------
# Wide CSV + float16 matrix
# ---------------------------------------------------------------------------


def build_wide_csv(
    roots: Sequence[str],
    output_csv: str,
    *,
    measure_cols: Sequence[str],
    fps_fallback: float = 40.0,
    exclude_roots: Sequence[str] | None = None,
) -> None:
    out_path = Path(output_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, object]] = []
    max_len = 0
    exclude = {
        Path(root).expanduser().resolve() for root in (exclude_roots or ())
    }
    exclude |= MANDATORY_WIDE_EXCLUDES
    for root in _normalise_roots(roots):
        if root in exclude:
            print(f"[SKIP] Excluding dataset root: {root}")
            continue
        dataset = root.name
        for fly_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            fly = fly_dir.name
            for csv_path in _find_trial_csvs(fly_dir):
                try:
                    header = pd.read_csv(csv_path, nrows=0)
                except Exception as exc:
                    print(f"[WARN] Skip {csv_path.name}: header read error: {exc}")
                    continue
                measure = _pick_column(header, measure_cols)
                if measure is None:
                    print(f"[SKIP] {csv_path.name}: none of {measure_cols} present.")
                    continue
                try:
                    n_rows = pd.read_csv(csv_path, usecols=[measure]).shape[0]
                except Exception as exc:
                    print(f"[WARN] Skip {csv_path.name}: count error: {exc}")
                    continue
                items.append(
                    {
                        "dataset": dataset,
                        "fly": fly,
                        "csv_path": csv_path,
                        "measure_col": measure,
                    }
                )
                max_len = max(max_len, n_rows)

    if not items:
        raise RuntimeError("No eligible testing/training CSVs found in provided roots.")

    metadata = ["dataset", "fly", "trial_type", "trial_label", "fps"]
    value_cols = [f"dir_val_{idx}" for idx in range(max_len)]
    header_df = pd.DataFrame(columns=metadata + value_cols)
    header_df.to_csv(out_path, index=False)

    for item in items:
        csv_path = Path(item["csv_path"])
        try:
            header = pd.read_csv(csv_path, nrows=0)
        except Exception:
            header = pd.DataFrame()

        frame_col = _pick_frame(header) if not header.empty else None
        ts_col = _pick_timestamp(header) if not header.empty else None
        fps = math.nan
        if frame_col and ts_col:
            try:
                ts_df = pd.read_csv(csv_path, usecols=[frame_col, ts_col])
                seconds = _seconds_from_timestamp(ts_df, ts_col)
                fps_est = _estimate_fps(seconds)
                if fps_est and np.isfinite(fps_est) and fps_est > 0:
                    fps = float(fps_est)
            except Exception as exc:
                print(f"[WARN] FPS inference failed for {csv_path.name}: {exc}")
        if not np.isfinite(fps):
            fps = float(fps_fallback)

        try:
            measure = str(item["measure_col"])
            df = pd.read_csv(csv_path, usecols=[measure])
            values = pd.to_numeric(df[measure], errors="coerce").astype(float).to_numpy()
        except Exception as exc:
            print(f"[WARN] Read failed {csv_path}: {exc}")
            continue

        trial_type = _infer_category(csv_path)
        label = _trial_label(csv_path)

        row = [item["dataset"], item["fly"], trial_type, label, float(fps)] + list(values)
        if len(values) < max_len:
            row.extend([np.nan] * (max_len - len(values)))
        elif len(values) > max_len:
            row = row[: len(metadata) + max_len]

        pd.DataFrame([row], columns=metadata + value_cols).to_csv(
            out_path, index=False, mode="a", header=False
        )

    print(f"[OK] Wrote combined direction-value table: {out_path}")


def wide_to_matrix(input_csv: str, output_dir: str) -> None:
    csv_path = Path(input_csv).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    meta_cols = [
        column
        for column in ("dataset", "fly", "trial_type", "trial_label", "fps")
        if column in df.columns
    ]
    if not meta_cols:
        raise RuntimeError(
            "No metadata columns found. Expected at least dataset/fly/trial_type/trial_label."
        )

    env_cols = [column for column in df.columns if column not in meta_cols]
    if not env_cols:
        raise RuntimeError("No envelope columns found.")

    code_maps: dict[str, dict[str, int]] = {}
    df_num = df.copy()
    for column in meta_cols:
        uniques = pd.Series(df[column].astype(str).fillna("UNKNOWN")).unique().tolist()
        mapping = {"UNKNOWN": 0}
        code = 1
        for value in uniques:
            if value not in mapping:
                mapping[value] = code
                code += 1
        code_maps[column] = mapping
        df_num[column] = df_num[column].astype(str).map(mapping).fillna(0).astype(np.int32)

    df_num[env_cols] = df_num[env_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    ordered_cols = meta_cols + env_cols
    matrix = df_num[ordered_cols].to_numpy(dtype=np.float16)

    matrix_path = out_dir / "envelope_matrix_float16.npy"
    codes_path = out_dir / "code_maps.json"
    key_path = out_dir / "code_key.txt"

    np.save(matrix_path, matrix)
    with open(codes_path, "w", encoding="utf-8") as fh:
        json.dump({"column_order": ordered_cols, "code_maps": code_maps}, fh, indent=2)

    with open(key_path, "w", encoding="utf-8") as fh:
        fh.write("# Envelope matrix schema (float16), row-wise\n")
        fh.write("# Columns (in order):\n")
        for idx, column in enumerate(ordered_cols):
            fh.write(f"{idx:>5}: {column}\n")
        fh.write("\n# Metadata code maps (string → integer code)\n")
        for column in meta_cols:
            fh.write(f"\n[{column}]\n")
            inverse = sorted(
                ((code, value) for value, code in code_maps[column].items()),
                key=lambda pair: pair[0],
            )
            for code, value in inverse:
                fh.write(f"{code:>5} : {value}\n")
        fh.write("\nNotes:\n")
        fh.write("- Matrix dtype is float16 (16-bit). Metadata codes are stored as float16 numbers in the matrix.\n")
        fh.write("- Envelope NaNs (shorter videos) were replaced with 0.0.\n")
        fh.write("- Code '0' means UNKNOWN for the metadata fields.\n")

    print(f"[OK] Saved 16-bit matrix: {matrix_path} (shape={matrix.shape}, dtype={matrix.dtype})")
    print(f"[OK] Saved key: {key_path}")
    print(f"[OK] Saved JSON maps: {codes_path}")


# ---------------------------------------------------------------------------
# Overlay plots (combined vs distance-only)
# ---------------------------------------------------------------------------


def overlay_sources(
    sources: Mapping[str, Mapping[str, str]],
    *,
    latency_sec: float,
    after_show_sec: float,
    output_dir: str,
    threshold_mult: float = 4.0,
    odor_on_s: float = 30.0,
    odor_off_s: float = 60.0,
    odor_latency_s: float = 0.0,
    overwrite: bool = False,
) -> None:
    frames = []
    env_cols_by_source: dict[str, list[str]] = {}
    for tag, paths in sources.items():
        df, env_cols = envelope_visuals._load_matrix(
            Path(paths["MATRIX_NPY"]).expanduser().resolve(),
            Path(paths["CODES_JSON"]).expanduser().resolve(),
        )
        df["_source"] = tag
        df["dataset_canon"] = df["dataset"].apply(_canon_dataset)
        df = df[df["trial_type"].str.lower() == "testing"].copy()
        df["fps"] = df["fps"].replace([np.inf, -np.inf], np.nan).fillna(40.0)
        frames.append(df)
        env_cols_by_source[tag] = env_cols

    if not frames:
        raise RuntimeError("No sources available for overlay plotting.")

    combined = pd.concat(frames, ignore_index=True)
    out_dir = Path(output_dir).expanduser().resolve()

    odor_latency = max(odor_latency_s, 0.0)
    odor_on_cmd = odor_on_s
    odor_off_cmd = odor_off_s
    odor_on_effective = odor_on_cmd + odor_latency
    odor_off_effective = odor_off_cmd + odor_latency
    linger = max(latency_sec, 0.0)
    x_max = odor_off_effective + linger + after_show_sec

    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "font.size": 10,
        }
    )

    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    style_map = {
        tag: dict(color=palette[idx % len(palette)], label=tag)
        for idx, tag in enumerate(sources.keys())
    }

    for fly, df_fly in combined.groupby("fly"):
        mu_per_source: MutableMapping[str, list[float]] = {}
        for _, row in df_fly.iterrows():
            env = _extract_env(row, env_cols_by_source[row["_source"]])
            fps = float(row.get("fps", 40.0))
            baseline_end = min(int(round(odor_on_cmd * fps)), env.size)
            if baseline_end:
                baseline = env[:baseline_end]
                baseline = baseline[np.isfinite(baseline)]
                if baseline.size:
                    mu_per_source.setdefault(row["_source"], []).extend(baseline)

        mu_lookup = {
            source: float(np.mean(values)) if values else math.nan
            for source, values in mu_per_source.items()
        }

        trials: dict[str, list[tuple[str, np.ndarray, np.ndarray, float, str, bool]]] = {}
        y_max = 0.0
        for _, row in df_fly.iterrows():
            env = _extract_env(row, env_cols_by_source[row["_source"]])
            if env.size == 0:
                continue
            fps = float(row.get("fps", 40.0))
            t_full = np.arange(env.size, dtype=float) / max(fps, 1e-9)
            mask = t_full <= x_max + 1e-9
            t = t_full[mask]
            env_visible = env[mask]
            if t.size == 0:
                continue

            baseline_end = min(int(round(odor_on_cmd * fps)), env.size)
            baseline = env[:baseline_end]
            sigma = float(np.nanstd(baseline)) if baseline.size else math.nan
            mu = mu_lookup.get(row["_source"], math.nan)
            if not math.isfinite(mu) and baseline.size:
                mu = float(np.nanmean(baseline))
            theta = mu + threshold_mult * sigma if math.isfinite(mu) and math.isfinite(sigma) else math.nan

            dataset_canon = row["dataset_canon"]
            odor_name = _display_odor(dataset_canon, row["trial_label"])
            is_trained = _is_trained(dataset_canon, odor_name)
            trials.setdefault(row["trial_label"], []).append(
                (row["_source"], t, env_visible, theta, odor_name, is_trained)
            )

            local_max = np.nanmax(env_visible) if np.isfinite(env_visible).any() else 0.0
            if np.isfinite(theta):
                local_max = max(local_max, theta)
            y_max = max(y_max, float(local_max))

        if not trials:
            continue

        datasets_present = df_fly["dataset_canon"].dropna().unique().tolist()
        target_dir = resolve_dataset_output_dir(out_dir, datasets_present or ("UNKNOWN",))
        out_png = target_dir / f"{fly}_overlay_envelope_by_trial_{int(after_show_sec)}s_shifted.png"
        if out_png.exists() and not overwrite:
            continue

        trial_labels = sorted(trials.keys(), key=_trial_num)
        fig, axes = plt.subplots(
            len(trial_labels),
            1,
            figsize=(10, max(3.0, len(trial_labels) * 1.6 + 1.5)),
            sharex=True,
        )
        if len(trial_labels) == 1:
            axes = [axes]

        for ax, trial_label in zip(axes, trial_labels):
            entries = trials[trial_label]
            odor_name = entries[0][4]
            is_trained = entries[0][5]
            ax.axvline(odor_on_effective, linestyle="--", linewidth=1.0, color="black")
            ax.axvline(odor_off_effective, linestyle="--", linewidth=1.0, color="black")

            transit_on_end = min(odor_on_effective, x_max)
            transit_off_end = min(odor_off_effective, x_max)
            if odor_latency > 0:
                ax.axvspan(odor_on_cmd, transit_on_end, alpha=0.25, color="red")
                ax.axvspan(odor_off_cmd, transit_off_end, alpha=0.25, color="red")

            steady_off_end = min(odor_off_effective, x_max)
            linger_off_end = min(odor_off_effective + linger, x_max)
            if steady_off_end > transit_on_end:
                ax.axvspan(transit_on_end, steady_off_end, alpha=0.15, color="gray")
            if linger > 0 and linger_off_end > steady_off_end:
                ax.axvspan(steady_off_end, linger_off_end, alpha=0.1, color="gray")

            for source, t, env_visible, theta, *_ in entries:
                style = style_map[source]
                line = ax.plot(t, env_visible, linewidth=1.3, color=style["color"], label=style["label"])
                style["label"] = None  # only show once in legend
                if np.isfinite(theta):
                    ax.axhline(theta, linestyle=":", linewidth=1.0, color=line[0].get_color(), alpha=0.9)

            ax.set_ylim(0, y_max * 1.02 if y_max > 0 else 1.0)
            ax.set_xlim(0, x_max)
            ax.margins(x=0, y=0.02)
            ax.set_ylabel("DIST or DIST×ANGLE", fontsize=10)
            title = f"{odor_name} — {trial_label}"
            if is_trained:
                ax.set_title(title, loc="left", fontsize=11, weight="bold", pad=2, color="tab:blue")
            else:
                ax.set_title(title, loc="left", fontsize=11, weight="bold", pad=2)

        axes[-1].set_xlabel("Time (s)", fontsize=11)

        legend_handles = [
            plt.Line2D([0], [0], linestyle="--", linewidth=1.0, color="black", label="Odor at fly"),
            plt.Rectangle((0, 0), 1, 1, alpha=0.25, color="red", label=f"Valve→fly transit (~{odor_latency:.2f}s)"),
            plt.Rectangle((0, 0), 1, 1, alpha=0.15, color="gray", label="Odor present"),
            plt.Line2D([0], [0], linestyle=":", linewidth=1.0, color="black", label=r"$\theta=\mu_{global}+k\sigma_{trial}$"),
        ]
        for tag, style in style_map.items():
            legend_handles.insert(0, plt.Line2D([0], [0], linewidth=1.3, color=style["color"], label=tag))

        fig = plt.gcf()
        fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.97),
            frameon=True,
            fontsize=9,
            title=f"Threshold: k = {threshold_mult}",
            title_fontsize=9,
        )
        fig.suptitle(
            f"{fly} — Envelope overlay by testing trial (global μ per source, σ per trial)",
            y=0.995,
            fontsize=14,
            weight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if should_write(out_png, overwrite):
            fig.savefig(out_png, dpi=300, bbox_inches="tight")

        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    combine_parser = subparsers.add_parser("combine", help="Merge distance/angle testing trials.")
    combine_parser.add_argument("--root", type=Path, required=True, help="Fly dataset root directory.")
    combine_parser.add_argument("--fps-default", type=float, default=40.0)
    combine_parser.add_argument("--window-sec", type=float, default=0.25)
    combine_parser.add_argument("--odor-on", type=float, default=30.0)
    combine_parser.add_argument("--odor-off", type=float, default=60.0)
    combine_parser.add_argument(
        "--odor-latency",
        type=float,
        default=0.0,
        help="Transit delay between valve command and odor at the fly (seconds).",
    )

    copy_parser = subparsers.add_parser("secure-sync", help="Copy datasets then clean source directories.")
    copy_parser.add_argument("--source", action="append", required=True, help="Source directory (repeatable).")
    copy_parser.add_argument("--dest", required=True, help="Destination root directory.")

    wide_parser = subparsers.add_parser("wide", help="Build wide CSV of direction values.")
    wide_parser.add_argument("--root", action="append", required=True, help="Root directory (repeatable).")
    wide_parser.add_argument("--output-csv", required=True, help="Destination CSV path.")
    wide_parser.add_argument(
        "--measure-col",
        dest="measure_cols",
        action="append",
        default=None,
        help="Measurement column to extract (repeatable).",
    )
    wide_parser.add_argument(
        "--fps-fallback",
        type=float,
        default=40.0,
        help="Fallback FPS when timestamps/frames cannot infer it.",
    )
    wide_parser.add_argument(
        "--exclude-root",
        action="append",
        default=None,
        help="Root directory to exclude from aggregation (repeatable).",
    )

    matrix_parser = subparsers.add_parser("matrix", help="Convert wide CSV → float16 matrix + metadata.")
    matrix_parser.add_argument("--input-csv", required=True)
    matrix_parser.add_argument("--output-dir", required=True)

    matrices_parser = subparsers.add_parser("matrices", help="Generate reaction matrices for combined data.")
    envelope_visuals._parse_matrices_args(matrices_parser)  # reuse existing options

    envelopes_parser = subparsers.add_parser("envelopes", help="Generate per-fly envelopes for combined data.")
    envelope_visuals._parse_envelopes_args(envelopes_parser)

    overlay_parser = subparsers.add_parser("overlay", help="Overlay combined matrix vs distance-only matrix.")
    overlay_parser.add_argument(
        "--combined-matrix",
        required=True,
        help="Path to combined matrix float16 .npy file.",
    )
    overlay_parser.add_argument(
        "--combined-codes",
        required=True,
        help="Path to combined matrix code_maps.json.",
    )
    overlay_parser.add_argument(
        "--distance-matrix",
        required=True,
        help="Path to distance-only matrix float16 .npy file.",
    )
    overlay_parser.add_argument(
        "--distance-codes",
        required=True,
        help="Path to distance-only matrix code_maps.json.",
    )
    overlay_parser.add_argument("--out-dir", required=True, help="Output directory for overlay figures.")
    overlay_parser.add_argument("--latency-sec", type=float, default=0.0)
    overlay_parser.add_argument("--after-show-sec", type=float, default=30.0)
    overlay_parser.add_argument("--threshold-std-mult", type=float, default=4.0)
    overlay_parser.add_argument(
        "--odor-latency-s",
        type=float,
        default=0.0,
        help="Transit delay between valve command and odor at the fly (seconds).",
    )
    overlay_parser.add_argument("--overwrite", action="store_true", help="Rebuild plots even if the target files exist.")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "combine":
        cfg = CombineConfig(
            root=args.root.expanduser().resolve(),
            fps_default=args.fps_default,
            window_sec=args.window_sec,
            odor_on_s=args.odor_on,
            odor_off_s=args.odor_off,
            odor_latency_s=args.odor_latency,
        )
        combine_distance_angle(cfg)
        return

    if args.command == "secure-sync":
        secure_copy_and_cleanup(args.source, args.dest)
        return

    if args.command == "wide":
        measure_cols = args.measure_cols or ["envelope_of_rms"]
        build_wide_csv(
            args.root,
            args.output_csv,
            measure_cols=measure_cols,
            fps_fallback=args.fps_fallback,
            exclude_roots=args.exclude_root,
        )
        return

    if args.command == "matrix":
        wide_to_matrix(args.input_csv, args.output_dir)
        return

    if args.command == "matrices":
        trial_order = args.trial_order or ("observed", "trained-first")
        cfg = MatrixPlotConfig(
            matrix_npy=args.matrix_npy,
            codes_json=args.codes_json,
            out_dir=args.out_dir,
            latency_sec=args.latency_sec,
            fps_default=args.fps_default,
            before_sec=args.before_sec,
            during_sec=args.during_sec,
            after_window_sec=args.after_window_sec,
            threshold_std_mult=args.threshold_std_mult,
            min_samples_over=args.min_samples_over,
            row_gap=args.row_gap,
            height_per_gap_in=args.height_per_gap_in,
            bottom_shift_in=args.bottom_shift_in,
            trial_orders=trial_order,
            include_hexanol=not args.exclude_hexanol,
            overwrite=args.overwrite,
        )
        generate_reaction_matrices(cfg)
        return

    if args.command == "envelopes":
        cfg = EnvelopePlotConfig(
            matrix_npy=args.matrix_npy,
            codes_json=args.codes_json,
            out_dir=args.out_dir,
            latency_sec=args.latency_sec,
            fps_default=args.fps_default,
            odor_on_s=args.odor_on_s,
            odor_off_s=args.odor_off_s,
            odor_latency_s=args.odor_latency_s,
            after_show_sec=args.after_show_sec,
            threshold_std_mult=args.threshold_std_mult,
            overwrite=args.overwrite,
        )
        generate_envelope_plots(cfg)
        return

    if args.command == "overlay":
        overlay_sources(
            {
                "RMS × Angle": {
                    "MATRIX_NPY": args.combined_matrix,
                    "CODES_JSON": args.combined_codes,
                },
                "RMS": {
                    "MATRIX_NPY": args.distance_matrix,
                    "CODES_JSON": args.distance_codes,
                },
            },
            latency_sec=args.latency_sec,
            after_show_sec=args.after_show_sec,
            output_dir=args.out_dir,
            threshold_mult=args.threshold_std_mult,
            odor_latency_s=args.odor_latency_s,
            overwrite=args.overwrite,
        )
        return

    parser.error(f"Unknown command: {args.command!r}")


if __name__ == "__main__":  # pragma: no cover
    main()
