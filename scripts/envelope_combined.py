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
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.signal import hilbert

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fbpipe.config import load_settings


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

from scripts import envelope_visuals
from scripts.envelope_visuals import (
    EnvelopePlotConfig,
    MatrixPlotConfig,
    compute_non_reactive_flags,
    generate_envelope_plots,
    generate_reaction_matrices,
    is_non_reactive_span,
    resolve_dataset_output_dir,
    should_write,
)


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
FLY_SLOT_REGEX = re.compile(r"(fly\d+)_distances", re.IGNORECASE)
FLY_NUMBER_REGEX = re.compile(r"fly\s*[_-]?\s*(\d+)", re.IGNORECASE)

ANCHOR_X = 1080.0
ANCHOR_Y = 540.0

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_DISTANCE_LIMITS = (70.0, 250.0)


def _resolve_distance_limits(
    distance_limits: tuple[float, float] | None,
    config_path: str | Path | None,
) -> tuple[float, float]:
    if distance_limits is not None:
        lower, upper = distance_limits
        return float(lower), float(upper)

    cfg_path = Path(config_path).expanduser().resolve() if config_path else DEFAULT_CONFIG_PATH
    try:
        settings = load_settings(cfg_path)
        return float(settings.class2_min), float(settings.class2_max)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(
            "[WARN] build_wide_csv: failed to load distance limits from "
            f"{cfg_path} ({exc}); falling back to defaults"
        )
        return float(DEFAULT_DISTANCE_LIMITS[0]), float(DEFAULT_DISTANCE_LIMITS[1])

MANDATORY_WIDE_EXCLUDES = {
    Path("/securedstorage/DATAsec/cole/Data-secured/opto_benz/").expanduser().resolve()
}

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
    "eb_control": "EB_control",
    "eb control": "EB_control",
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
    "EB_control": "EB Control",
    "opto_benz": "Benzaldehyde",
    "opto_benz_1": "Benzaldehyde",
    "opto_EB": "Ethyl Butyrate",
    "opto_hex": "Hexanol",
}

HEXANOL = "Hexanol"


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
        "EB_control": {
            6: "Apple Cider Vinegar",
            7: "3-Octonol",
            8: "Benzaldehyde",
            9: "Citral",
            10: "Linalool",
        },
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
    # Angle bins that previously suppressed the distance percentage (multipliers < 1)
    # now default to a neutral weight of 1.0 so `dir_val` never penalises low angles.
    multipliers = [1.00, 1.00, 1.00, 1.00, 1.25, 1.50, 1.75, 2.00]
    return np.select(conditions, multipliers, default=np.nan)


def _extract_fly_number(*candidates: Optional[str]) -> Optional[int]:
    for candidate in candidates:
        if not candidate:
            continue
        match = FLY_NUMBER_REGEX.search(candidate)
        if not match:
            continue
        try:
            return int(match.group(1))
        except ValueError:
            print(f"[WARN] fly number token in '{candidate}' was not a valid integer.")
    return None


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


def _locate_trials(
    fly_dir: Path,
    suffix_globs: Iterable[str] | str,
    required_cols: Sequence[str],
) -> list[tuple[str, Path, str]]:
    patterns = [suffix_globs] if isinstance(suffix_globs, str) else list(suffix_globs)
    month_dirs = [p for p in fly_dir.rglob("*") if _is_month_folder(p)]
    found: set[Path] = set()
    print(
        f"[DEBUG] {fly_dir.name}: scanning {len(month_dirs)} month directories for patterns {patterns}"
    )
    for month_dir in month_dirs:
        for pattern in patterns:
            found.update(Path(p) for p in month_dir.rglob(pattern))

    results: list[tuple[str, Path, str]] = []
    for csv_path in sorted(found):
        print(f"[DEBUG] {fly_dir.name}: evaluating {csv_path}")
        try:
            header = pd.read_csv(csv_path, nrows=5)
        except Exception:
            print(f"[WARN] {fly_dir.name}: failed to read header from {csv_path}")
            continue
        if not _pick_column(header, required_cols):
            print(
                f"[SKIP] {csv_path.name}: required columns {required_cols} missing. Available={list(header.columns)}"
            )
            continue
        results.append((csv_path.stem, csv_path, _infer_category(csv_path)))
        print(
            f"[DEBUG] {fly_dir.name}: accepted {csv_path.name} as {_infer_category(csv_path)}"
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


def _trial_csv_candidates(fly_dir: Path, suffix_globs: Iterable[str] | str) -> list[Path]:
    patterns = [suffix_globs] if isinstance(suffix_globs, str) else list(suffix_globs)
    candidates: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in fly_dir.rglob(pattern):
            if not path.is_file():
                continue
            if "training" not in path.name.lower() and "testing" not in path.name.lower():
                continue
            real = path.resolve()
            if real in seen:
                continue
            seen.add(real)
            candidates.append(path)
    return sorted(candidates)


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


def _index_testing(entries: Sequence[tuple[str, Path, str]]) -> tuple[dict[str, Path], dict[str, Path]]:
    indexed: dict[str, Path] = {}
    fallback: dict[str, Path] = {}
    for label, path, category in entries:
        if category.lower() != "testing":
            continue
        match = TESTING_REGEX.search(label)
        slot_match = FLY_SLOT_REGEX.search(path.stem)
        slot_label = slot_match.group(1).lower() if slot_match else None
        if match:
            base_key = match.group(0).lower()
            if slot_label:
                indexed[f"{base_key}_{slot_label}"] = path
            indexed.setdefault(base_key, path)
        else:
            key = label.lower()
            if slot_label:
                indexed[f"{key}_{slot_label}"] = path
            fallback[key] = path
    return indexed, fallback


_STATS_SUFFIX = "_global_distance_stats_class_2.json"
_FLY_STATS_PATTERN = re.compile(r"fly[_-]?(\d+)_global_distance_stats_class_2\.json$", re.IGNORECASE)


def _parse_distance_stats(stats_path: Path) -> tuple[float, float] | None:
    try:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] build_wide_csv: failed to parse {stats_path}: {exc}")
        return None

    try:
        gmin = float(payload.get("global_min", float("nan")))
        gmax = float(payload.get("global_max", float("nan")))
    except (TypeError, ValueError):
        print(
            f"[WARN] build_wide_csv: non-numeric stats in {stats_path.name}: {payload}"
        )
        return None

    if not (math.isfinite(gmin) and math.isfinite(gmax)):
        print(
            f"[WARN] build_wide_csv: invalid stats in {stats_path.name}: {payload}"
        )
        return None

    return float(gmin), float(gmax)


def _load_global_distance_stats(
    fly_dir: Path,
) -> tuple[dict[str, tuple[float, float]], tuple[float, float] | None]:
    """Return indexed global stats for each fly plus any legacy aggregate."""

    per_fly: dict[str, tuple[float, float]] = {}
    legacy: tuple[float, float] | None = None
    seen: set[Path] = set()

    for stats_path in sorted(fly_dir.rglob(f"*{_STATS_SUFFIX}")):
        if stats_path in seen or not stats_path.is_file():
            continue
        seen.add(stats_path)

        stats = _parse_distance_stats(stats_path)
        if stats is None:
            continue

        name_lower = stats_path.name.lower()
        if name_lower == "global_distance_stats_class_2.json":
            legacy = stats
            continue

        match = _FLY_STATS_PATTERN.match(name_lower)
        if match:
            fly_key = str(int(match.group(1)))
            if fly_key in per_fly and per_fly[fly_key] != stats:
                print(
                    f"[WARN] build_wide_csv: duplicate stats for fly{fly_key} in {fly_dir}"
                )
            per_fly[fly_key] = stats
            continue

        # Unknown naming convention – retain for diagnostics but do not map.
        print(
            f"[WARN] build_wide_csv: unrecognised stats filename {stats_path.name} in {fly_dir}"
        )

    return per_fly, legacy


def _resolve_distance_stats(
    per_fly: Mapping[str, tuple[float, float]],
    legacy: tuple[float, float] | None,
    fly_dir: Path,
    fly_number: Optional[int],
    csv_path: Path,
) -> tuple[float, float]:
    context = f"{fly_dir.name}/{csv_path.name}"
    if fly_number is not None:
        key = str(fly_number)
        stats = per_fly.get(key)
        if stats is not None:
            return stats
        if legacy is not None:
            print(
                f"[WARN] build_wide_csv: {context} missing fly-specific stats; using legacy aggregate."
            )
            return legacy
        print(
            f"[WARN] build_wide_csv: {context} missing fly{key} stats; marking as NaN."
        )
        return (float("nan"), float("nan"))

    if legacy is not None:
        return legacy

    if len(per_fly) == 1:
        key, stats = next(iter(per_fly.items()))
        print(
            f"[WARN] build_wide_csv: {context} lacks fly number; using stats from fly{key}."
        )
        return stats

    if per_fly:
        print(
            f"[WARN] build_wide_csv: {context} lacks fly number and multiple stats are available; marking as NaN."
        )

    return (float("nan"), float("nan"))


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


def _effective_fps(value: float, *, fallback: float, default: float) -> float:
    for candidate in (value, fallback, default):
        if np.isfinite(candidate) and candidate > 0:
            return float(candidate)
    return 0.0


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
    values: np.ndarray,
    fps: float,
    *,
    fallback_fps: float,
    default_fps: float,
    fly_before_mean: float,
) -> dict[str, float]:
    env = np.asarray(values, dtype=float)
    total_len = env.size
    before_end, during_start, during_end, after_end = _segment_bounds(total_len)

    before = env[:before_end]
    during = env[during_start:during_end]
    after = env[during_end:after_end]

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
            global_peak_idx = int(np.nanargmax(env))
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
    angle_suffixes: tuple[str, ...] = (
        "*fly*_distances.csv",
        "*merged.csv",
        "*class_2_8.csv",
        "*class_2_6.csv",
    )
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
    print(
        "[DEBUG] combine_distance_angle → root=%s, fps_default=%.3f, window_frames=%d"
        % (cfg.root, cfg.fps_default, cfg.window_frames)
    )
    odor_latency = max(cfg.odor_latency_s, 0.0)
    odor_on_cmd = cfg.odor_on_s
    odor_off_cmd = cfg.odor_off_s
    odor_on_effective = odor_on_cmd + odor_latency
    odor_off_effective = odor_off_cmd + odor_latency

    for fly_dir in sorted(p for p in cfg.root.iterdir() if p.is_dir()):
        fly_name = fly_dir.name
        print(f"\n[DEBUG] Processing fly: {fly_name} ({fly_dir})")
        _ensure_angle_percentages(fly_dir, cfg.angle_suffixes)
        angle_entries = _locate_trials(fly_dir, cfg.angle_suffixes, ANGLE_COLS)
        distance_entries = _locate_trials(fly_dir, cfg.distance_suffixes, DIST_COLS)

        if not distance_entries:
            print(f"[{fly_name}] No testing distance trials found — skipping.")
            continue

        angle_idx, angle_fallback = _index_testing(angle_entries)

        dist_idx: dict[str, tuple[Path, str, Optional[str]]] = {}
        for label, path, category in distance_entries:
            if category.lower() != "testing":
                continue
            match = TESTING_REGEX.search(label)
            base_key = match.group(0).lower() if match else label.lower()
            slot_match = FLY_SLOT_REGEX.search(path.stem)
            slot_label = slot_match.group(1).lower() if slot_match else None
            key = f"{base_key}_{slot_label}" if slot_label else base_key
            prefer_new = path.name.lower().startswith("updated_")
            existing = dist_idx.get(key)
            if existing is None or (prefer_new and not existing[0].name.lower().startswith("updated_")):
                dist_idx[key] = (path, base_key, slot_label)

        out_csv_dir = fly_dir / "angle_distance_rms_envelope"
        out_fig_dir = out_csv_dir / "plots"
        out_csv_dir.mkdir(parents=True, exist_ok=True)
        out_fig_dir.mkdir(parents=True, exist_ok=True)

        completed = skipped = 0
        for key, (dist_path, base_key, slot_label) in sorted(dist_idx.items()):
            lookup_keys = [key, base_key]
            angle_path: Optional[Path] = None
            for candidate in lookup_keys:
                angle_path = angle_idx.get(candidate)
                if angle_path:
                    break
            if angle_path is None:
                angle_path = angle_fallback.get(dist_path.stem.lower())
            if angle_path is None:
                print(f"[WARN] {fly_name} {key}: no matching angle file — skipped.")
                skipped += 1
                continue

            fly_number = _extract_fly_number(slot_label, dist_path.stem, fly_dir.name)
            fly_number_label = str(fly_number) if fly_number is not None else "UNKNOWN"
            if fly_number is None:
                print(
                    f"[WARN] {fly_name} {dist_path.name}: unable to infer fly number token; defaulting to 'UNKNOWN'"
                )
            else:
                print(
                    f"[DEBUG] {fly_name}: parsed fly number {fly_number_label} from slot={slot_label} stem={dist_path.stem}"
                )

            print(
                f"[DEBUG] {fly_name}: pairing distance={dist_path.name} with angle={angle_path.name}; slot_label={slot_label}"
            )

            try:
                dist_df = pd.read_csv(dist_path)
                dist_col = _pick_column(dist_df, DIST_COLS)
                if not dist_col:
                    raise ValueError("missing distance column")
                print(
                    f"[DEBUG] {fly_name}: distance columns={list(dist_df.columns)} selected={dist_col}"
                )
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
                print(
                    f"[DEBUG] {fly_name}: angle columns={list(angle_df.columns)} selected={angle_col}"
                )
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

                slot_suffix = f"_{slot_label}" if slot_label else ""
                test_id = f"{base_key}{slot_suffix}".replace("__", "_")
                out_df = pd.DataFrame(
                    {
                        "time_s": time_dist,
                        "angle_centered_pct_interp": interp_angle,
                        "distance_percentage": dist_pct,
                        "multiplier": multiplier,
                        "combined_base": combined,
                        "rolling_rms": combined_rms,
                        "envelope_of_rms": envelope,
                        "fly_number": fly_number_label,
                    }
                )
                out_csv = out_csv_dir / f"{test_id}_angle_distance_rms_envelope.csv"
                out_df.to_csv(out_csv, index=False)
                print(
                    f"[DEBUG] {fly_name}: wrote {out_csv.name} rows={len(out_df)} fly_number={fly_number_label}"
                )

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
                    f"{fly_name} — {test_id}: Envelope(RMS(distance × angle-mult))"
                )
                plt.xlabel("Time (s)")
                plt.ylabel("Envelope of RMS (arb.)")
                plt.margins(x=0)
                plt.grid(True, alpha=0.3)
                out_png = out_fig_dir / f"{fly_name}_{test_id}_env_rms_angle_distance.png"
                plt.savefig(out_png, dpi=300, bbox_inches="tight")
                plt.close()

                print(
                    f"[OK] {fly_name} {test_id} → CSV: {out_csv.name} | FIG: {out_png.name}"
                )
                completed += 1
            except Exception as exc:
                print(f"[WARN] {fly_name} {test_id} → {exc}")
                skipped += 1

        print(f"[{fly_name}] completed: {completed}, skipped: {skipped}")


# ---------------------------------------------------------------------------
# Secure copy + cleanup
# ---------------------------------------------------------------------------


def secure_copy_and_cleanup(
    sources: Sequence[str], destination: str, perform_cleanup: bool = False
) -> None:
    source_list = list(sources)
    print(
        f"[DEBUG] secure_copy_and_cleanup → sources={source_list}, dest={destination}, perform_cleanup={perform_cleanup}"
    )
    dest_root = Path(destination).expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    for source in _normalise_roots(source_list):
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

    if not perform_cleanup:
        print("[INFO] Cleanup disabled; source directories left untouched.")
        return

    for source in _normalise_roots(source_list):
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
    distance_limits: tuple[float, float] | None = None,
    config_path: str | Path | None = None,
) -> None:
    print(
        f"[DEBUG] build_wide_csv → roots={list(roots)} output={output_csv} measure_cols={list(measure_cols)}"
    )
    out_path = Path(output_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    class2_min, class2_max = _resolve_distance_limits(distance_limits, config_path)
    print(
        f"[DEBUG] build_wide_csv: applying distance limits [{class2_min:.3f}, {class2_max:.3f}] for local extrema"
    )

    items: list[dict[str, object]] = []
    flagged_summary: dict[tuple[str, str, str], tuple[float, float]] = {}
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
            print(f"[DEBUG] build_wide_csv: scanning fly_dir={fly_dir}")
            per_fly_stats, legacy_stats = _load_global_distance_stats(fly_dir)
            for csv_path in _find_trial_csvs(fly_dir):
                try:
                    header = pd.read_csv(csv_path, nrows=0)
                except Exception as exc:
                    print(f"[WARN] Skip {csv_path.name}: header read error: {exc}")
                    continue
                print(
                    f"[DEBUG] build_wide_csv: candidate={csv_path.name} columns={list(header.columns)}"
                )
                measure = _pick_column(header, measure_cols)
                if measure is None:
                    print(f"[SKIP] {csv_path.name}: none of {measure_cols} present.")
                    continue
                slot_match = FLY_SLOT_REGEX.search(csv_path.stem)
                slot_label = slot_match.group(1) if slot_match else None
                fly_number = _extract_fly_number(slot_label, csv_path.stem, fly_dir.name)
                fly_number_label = str(fly_number) if fly_number is not None else "UNKNOWN"
                if fly_number is None:
                    print(
                        f"[WARN] build_wide_csv: {csv_path.name} lacks fly number token; using 'UNKNOWN'"
                    )
                else:
                    print(
                        f"[DEBUG] build_wide_csv: {csv_path.name} fly_number={fly_number_label}"
                    )
                gmin, gmax = _resolve_distance_stats(
                    per_fly_stats, legacy_stats, fly_dir, fly_number, csv_path
                )
                flagged = is_non_reactive_span(gmin, gmax)
                try:
                    n_rows = pd.read_csv(csv_path, usecols=[measure]).shape[0]
                except Exception as exc:
                    print(f"[WARN] Skip {csv_path.name}: count error: {exc}")
                    continue
                items.append(
                    {
                        "dataset": dataset,
                        "fly": fly,
                        "fly_key": f"{dataset}::{fly}",
                        "fly_number": fly_number_label,
                        "global_min": float(gmin),
                        "global_max": float(gmax),
                        "non_reactive_flag": 1.0 if flagged else 0.0,
                        "csv_path": csv_path,
                        "measure_col": measure,
                    }
                )
                if flagged:
                    flagged_summary[(dataset, fly, fly_number_label)] = (float(gmin), float(gmax))
                max_len = max(max_len, n_rows)

    if not items:
        raise RuntimeError("No eligible testing/training CSVs found in provided roots.")

    fly_before_totals: dict[str, tuple[float, int]] = {}
    for item in items:
        csv_path = Path(item["csv_path"])
        measure = str(item["measure_col"])
        if _infer_category(csv_path).lower() != "testing":
            continue
        try:
            df_before = pd.read_csv(csv_path, usecols=[measure], nrows=BEFORE_FRAMES)
        except Exception as exc:
            print(f"[WARN] build_wide_csv: failed to read before segment from {csv_path.name}: {exc}")
            continue
        series = pd.to_numeric(df_before[measure], errors="coerce").astype(float).to_numpy()
        if series.size == 0:
            continue
        finite_mask = np.isfinite(series)
        if not np.any(finite_mask):
            continue
        total, count = fly_before_totals.get(item["fly_key"], (0.0, 0))
        fly_before_totals[item["fly_key"]] = (
            total + float(np.nansum(series[finite_mask])),
            count + int(np.sum(finite_mask)),
        )

    fly_before_means = {
        key: (total / count if count > 0 else float("nan"))
        for key, (total, count) in fly_before_totals.items()
    }

    meta_prefix = ["dataset", "fly", "fly_number"]
    stat_columns = [
        "global_min",
        "global_max",
        "local_min",
        "local_max",
        "non_reactive_flag",
    ]
    meta_suffix = ["trial_type", "trial_label", "fps"]
    metadata = meta_prefix + stat_columns + meta_suffix
    value_cols = [f"dir_val_{idx}" for idx in range(max_len)]
    header_df = pd.DataFrame(columns=metadata + list(AUC_COLUMNS) + value_cols)
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
        fly_number_label = str(item.get("fly_number", "UNKNOWN"))
        print(
            f"[DEBUG] build_wide_csv: writing dataset={item['dataset']} fly={item['fly']} fly_number={fly_number_label} fps={fps:.3f} frames={len(values)}"
        )

        metrics = _compute_trial_metrics(
            values,
            fps,
            fallback_fps=fps_fallback,
            default_fps=fps_fallback,
            fly_before_mean=fly_before_means.get(item["fly_key"], float("nan")),
        )

        gmin = float(item.get("global_min", float("nan")))
        gmax = float(item.get("global_max", float("nan")))
        in_range_mask = np.isfinite(values) & (values >= class2_min) & (values <= class2_max)
        in_range_values = values[in_range_mask]
        if in_range_values.size:
            local_min = float(np.min(in_range_values))
            local_max = float(np.max(in_range_values))
        else:
            local_min = float("nan")
            local_max = float("nan")
            print(
                "[WARN] build_wide_csv: "
                f"no values within [{class2_min:.3f}, {class2_max:.3f}] for {csv_path.name};"
                " local extrema set to NaN."
            )
        non_reactive = float(item.get("non_reactive_flag", 0.0))
        row = [
            item["dataset"],
            item["fly"],
            fly_number_label,
            gmin,
            gmax,
            local_min,
            local_max,
            non_reactive,
            trial_type,
            label,
            float(fps),
            metrics["AUC-Before"],
            metrics["AUC-During"],
            metrics["AUC-After"],
            metrics["AUC-During-Before-Ratio"],
            metrics["AUC-After-Before-Ratio"],
            metrics["TimeToPeak-During"],
            metrics["Peak-Value"],
            *list(values),
        ]
        if len(values) < max_len:
            row.extend([np.nan] * (max_len - len(values)))
        elif len(values) > max_len:
            row = row[: len(metadata) + len(AUC_COLUMNS) + max_len]

        pd.DataFrame([row], columns=metadata + list(AUC_COLUMNS) + value_cols).to_csv(
            out_path, index=False, mode="a", header=False
        )

    flagged_path = out_path.with_name(out_path.stem + "_flagged_flies.txt")
    with flagged_path.open("w", encoding="utf-8") as fh:
        fh.write(
            f"# Flies flagged as non-reactive (global span ≤ {envelope_visuals.NON_REACTIVE_SPAN_PX:g} px)\n"
        )
        fh.write("# dataset,fly,fly_number,global_min,global_max\n")
        if flagged_summary:
            for key in sorted(flagged_summary):
                dataset, fly, fly_number = key
                gmin, gmax = flagged_summary[key]
                gmin_txt = f"{gmin:.3f}" if math.isfinite(gmin) else "nan"
                gmax_txt = f"{gmax:.3f}" if math.isfinite(gmax) else "nan"
                fh.write(f"{dataset},{fly},{fly_number},{gmin_txt},{gmax_txt}\n")
        else:
            fh.write("# None detected\n")
    print(f"[OK] Wrote flagged fly summary: {flagged_path}")
    print(f"[OK] Wrote combined direction-value table: {out_path}")


def wide_to_matrix(input_csv: str, output_dir: str) -> None:
    csv_path = Path(input_csv).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, dtype={"fly_number": str})
    meta_preference = [
        "dataset",
        "fly",
        "fly_number",
        "trial_type",
        "trial_label",
        "fps",
    ]
    meta_cols = [column for column in meta_preference if column in df.columns]
    if not meta_cols:
        raise RuntimeError(
            "No metadata columns found. Expected at least dataset/fly/trial_type/trial_label."
        )

    metric_cols = [col for col in AUC_COLUMNS if col in df.columns]
    extra_metrics = [
        col
        for col in (
            "global_min",
            "global_max",
            "local_min",
            "local_max",
            "non_reactive_flag",
        )
        if col in df.columns
    ]
    metric_cols.extend(extra_metrics)
    env_cols = [
        column
        for column in df.columns
        if column not in meta_cols and column not in metric_cols
    ]
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

    env_numeric = (
        df_num[metric_cols + env_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(np.float32, copy=False)
    )
    df_num[metric_cols + env_cols] = env_numeric

    ordered_cols = meta_cols + metric_cols + env_cols
    matrix_float32 = df_num[ordered_cols].to_numpy(dtype=np.float32, copy=False)
    finite_info = np.finfo(np.float16)
    np.clip(matrix_float32, finite_info.min, finite_info.max, out=matrix_float32)
    matrix = matrix_float32.astype(np.float16)

    matrix_path = out_dir / "envelope_matrix_float16.npy"
    codes_path = out_dir / "code_maps.json"
    key_path = out_dir / "code_key.txt"

    np.save(matrix_path, matrix)
    with open(codes_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "column_order": ordered_cols,
                "code_maps": code_maps,
                "metric_columns": metric_cols,
                "env_columns": env_cols,
            },
            fh,
            indent=2,
        )

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
    combined["_non_reactive"] = compute_non_reactive_flags(combined)
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
        fly_flagged = bool(df_fly.get("_non_reactive", pd.Series(False)).any())
        fig.suptitle(
            f"{fly} — Envelope overlay by testing trial (global μ per source, σ per trial)",
            y=0.995,
            fontsize=14,
            weight="bold",
            color="tab:red" if fly_flagged else "black",
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
    copy_parser.add_argument(
        "--perform-cleanup",
        action="store_true",
        help="After copying, delete non-essential files from the source directories.",
    )

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
    wide_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to pipeline configuration YAML supplying distance_limits. "
            "Defaults to config.yaml beside this script."
        ),
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
        secure_copy_and_cleanup(args.source, args.dest, perform_cleanup=args.perform_cleanup)
        return

    if args.command == "wide":
        measure_cols = args.measure_cols or ["envelope_of_rms"]
        build_wide_csv(
            args.root,
            args.output_csv,
            measure_cols=measure_cols,
            fps_fallback=args.fps_fallback,
            exclude_roots=args.exclude_root,
            config_path=args.config,
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
