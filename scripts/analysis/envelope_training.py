#!/usr/bin/env python3
"""Training-focused envelope utilities backed by the float16 matrix exports."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

# Ensure all plots use Arial to match lab styling.
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
    }
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.envelope_visuals import EnvelopePlotConfig, generate_envelope_plots

ODOR_CANON: Mapping[str, str] = {
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
    "hex_control": "hex_control",
    "hex control": "hex_control",
    "benz_control": "benz_control",
    "benz control": "benz_control",
    "optogenetics benzaldehyde": "opto_benz",
    "optogenetics benzaldehyde 1": "opto_benz_1",
    "optogenetics ethyl butyrate": "opto_EB",
    "10s_odor_benz": "10s_Odor_Benz",
    "optogenetics apple cider vinegar": "opto_ACV",
    "optogenetics acv": "opto_ACV",
    "optogenetics hexanol": "opto_hex",
    "optogenetics hex": "opto_hex",
    "hexanol": "opto_hex",
    "opto_hex": "opto_hex",
    "opto_acv": "opto_ACV",
    "optogenetics 3-octanol": "opto_3-oct",
    "opto_3-oct": "opto_3-oct",
}

DISPLAY_LABEL = {
    "ACV": "Apple Cider Vinegar",
    "3-octonol": "3-Octonol",
    "Benz": "Benzaldehyde",
    "10s_Odor_Benz": "Benzaldehyde",
    "EB": "Ethyl Butyrate",
    "EB_control": "Ethyl Butyrate",
    "hex_control": "Hexanol",
    "benz_control": "Benzaldehyde",
    "opto_benz": "Benzaldehyde",
    "opto_benz_1": "Benzaldehyde",
    "opto_EB": "Ethyl Butyrate",
    "opto_EB_6_training": "Ethyl Butyrate (6-Training)",
    "opto_ACV": "Apple Cider Vinegar",
    "opto_hex": "Hexanol",
    "opto_3-oct": "3-Octonol",
}

HEXANOL_LABEL = "Hexanol"

PRIMARY_ODOR_LABEL = {
    "EB_control": "Ethyl Butyrate",
    "hex_control": HEXANOL_LABEL,
    "benz_control": "Benzaldehyde",
}

TRAINING_ODOR_SCHEDULE_DEFAULT = {
    1: "Benzaldehyde",
    2: "Benzaldehyde",
    3: "Benzaldehyde",
    4: "Benzaldehyde",
    5: HEXANOL_LABEL,
    6: "Benzaldehyde",
    7: HEXANOL_LABEL,
    8: "Benzaldehyde",
}

TRAINING_ODOR_SCHEDULE_OVERRIDES = {
    "hex_control": {
        1: HEXANOL_LABEL,
        2: HEXANOL_LABEL,
        3: HEXANOL_LABEL,
        4: HEXANOL_LABEL,
        5: "Apple Cider Vinegar",
        6: HEXANOL_LABEL,
        7: "Apple Cider Vinegar",
        8: HEXANOL_LABEL,
    },
    "opto_hex": {
        1: HEXANOL_LABEL,
        2: HEXANOL_LABEL,
        3: HEXANOL_LABEL,
        4: HEXANOL_LABEL,
        5: "Apple Cider Vinegar",
        6: HEXANOL_LABEL,
        7: "Apple Cider Vinegar",
        8: HEXANOL_LABEL,
    },
    "EB_control": {
        1: "Ethyl Butyrate",
        2: "Ethyl Butyrate",
        3: "Ethyl Butyrate",
        4: "Ethyl Butyrate",
        5: HEXANOL_LABEL,
        6: "Ethyl Butyrate",
        7: HEXANOL_LABEL,
        8: "Ethyl Butyrate",
    },
    "opto_EB": {
        1: "Ethyl Butyrate",
        2: "Ethyl Butyrate",
        3: "Ethyl Butyrate",
        4: "Ethyl Butyrate",
        5: HEXANOL_LABEL,
        6: "Ethyl Butyrate",
        7: HEXANOL_LABEL,
        8: "Ethyl Butyrate",
    },
    "opto_EB_6_training": {
        1: "Ethyl Butyrate",
        2: "Ethyl Butyrate",
        3: "Ethyl Butyrate",
        4: "Ethyl Butyrate",
        5: "Ethyl Butyrate",
        6: "Ethyl Butyrate",
    },
    "opto_3-oct": {
        1: "3-Octonol",
        2: "3-Octonol",
        3: "3-Octonol",
        4: "3-Octonol",
        5: HEXANOL_LABEL,
        6: "3-Octonol",
        7: HEXANOL_LABEL,
        8: "3-Octonol",
    },
    "ACV": {
        1: "Apple Cider Vinegar",
        2: "Apple Cider Vinegar",
        3: "Apple Cider Vinegar",
        4: "Apple Cider Vinegar",
        5: HEXANOL_LABEL,
        6: "Apple Cider Vinegar",
        7: HEXANOL_LABEL,
        8: "Apple Cider Vinegar",
    },
    "opto_ACV": {
        1: "Apple Cider Vinegar",
        2: "Apple Cider Vinegar",
        3: "Apple Cider Vinegar",
        4: "Apple Cider Vinegar",
        5: HEXANOL_LABEL,
        6: "Apple Cider Vinegar",
        7: HEXANOL_LABEL,
        8: "Apple Cider Vinegar",
    },
}

TESTING_DATASET_ALIAS = {
    "opto_hex": "hex_control",
    "opto_EB": "EB_control",
    "opto_benz": "benz_control",
    "opto_benz_1": "benz_control",
    "opto_ACV": "ACV",
    "opto_3-oct": "opto_3-oct",
}


# ---------------------------------------------------------------------------
# Matrix loading helpers
# ---------------------------------------------------------------------------


def _canon_dataset(value: str) -> str:
    if not isinstance(value, str):
        return "UNKNOWN"
    return ODOR_CANON.get(value.strip().lower(), value.strip())


def _safe_dirname(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "export"


def _dataset_label(dataset: str) -> str:
    canon = _canon_dataset(dataset)
    return DISPLAY_LABEL.get(canon, canon)


def _target_dir(base: Path, datasets: Sequence[str] | str) -> Path:
    values: set[str] = set()
    if isinstance(datasets, str):
        if datasets:
            values.add(_canon_dataset(datasets))
    else:
        for val in datasets:
            if isinstance(val, str) and val:
                values.add(_canon_dataset(val))

    values = {val for val in values if val}
    if not values:
        label = "UNKNOWN"
    elif len(values) == 1:
        label = next(iter(values))
    else:
        label = "+".join(sorted(values))
    return base / _safe_dirname(label)


def _should_write(path: Path, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    return True


def _trial_num(label: str) -> int:
    match = re.search(r"(\d+)", str(label))
    return int(match.group(1)) if match else -1


def _trained_label(dataset_canon: str) -> str:
    return PRIMARY_ODOR_LABEL.get(
        dataset_canon, DISPLAY_LABEL.get(dataset_canon, dataset_canon)
    )


def _training_odor(dataset_canon: str, number: int) -> str | None:
    schedule = TRAINING_ODOR_SCHEDULE_OVERRIDES.get(dataset_canon)
    if schedule and number in schedule:
        return schedule[number]
    return TRAINING_ODOR_SCHEDULE_DEFAULT.get(number)


def _display_odor(dataset_canon: str, trial_label: str) -> str:
    number = _trial_num(trial_label)
    label_lower = str(trial_label).lower()

    if "training" in label_lower:
        odor_name = _training_odor(dataset_canon, number)
        if odor_name:
            return odor_name
        return DISPLAY_LABEL.get(dataset_canon, dataset_canon)

    dataset_for_testing = TESTING_DATASET_ALIAS.get(dataset_canon, dataset_canon)

    if number in (1, 3):
        return HEXANOL_LABEL
    if number in (2, 4, 5):
        return DISPLAY_LABEL.get(
            dataset_for_testing, DISPLAY_LABEL.get(dataset_canon, dataset_canon)
        )

    mapping = {
        "ACV": {
            6: "3-Octonol",
            7: "Ethyl Butyrate",
            8: "Benzaldehyde",
            9: "Citral",
            10: "Linalool",
        },
        "3-octonol": {6: "Benzaldehyde", 7: "Citral", 8: "Linalool"},
        "Benz": {6: "Citral", 7: "Linalool"},
        "benz_control": {
            6: "Apple Cider Vinegar",
            7: "3-Octonol",
            8: "Ethyl Butyrate",
            9: "Citral",
            10: "Linalool",
        },
        "EB": {
            6: "Apple Cider Vinegar",
            7: "3-Octonol",
            8: "Benzaldehyde",
            9: "Citral",
            10: "Linalool",
        },
        "EB_control": {
            6: "Apple Cider Vinegar",
            7: "3-Octonol",
            8: "Benzaldehyde",
            9: "Citral",
            10: "Linalool",
        },
        "hex_control": {
            6: "Benzaldehyde",
            7: "3-Octonol",
            8: "Ethyl Butyrate",
            9: "Citral",
            10: "Linalool",
        },
        "opto_ACV": {
            6: "3-Octonol",
            7: "Ethyl Butyrate",
            8: "Benzaldehyde",
            9: "Citral",
            10: "Linalool",
        },
        "10s_Odor_Benz": {6: "Benzaldehyde", 7: "Benzaldehyde"},
        "opto_3-oct": {
            6: "Apple Cider Vinegar",
            7: "Ethyl Butyrate",
            8: "Benzaldehyde",
            9: "Citral",
            10: "Linalool",
        },
    }

    if dataset_for_testing in mapping:
        return mapping[dataset_for_testing].get(number, trial_label)
    return mapping.get(dataset_canon, {}).get(number, trial_label)


def _load_envelope_matrix(matrix_path: Path, codes_json: Path) -> tuple[pd.DataFrame, list[str]]:
    matrix = np.load(matrix_path, allow_pickle=False)
    with codes_json.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)

    ordered_cols: list[str] = meta["column_order"]
    code_maps: Mapping[str, Mapping[str, int]] = meta["code_maps"]
    df = pd.DataFrame(matrix, columns=ordered_cols)

    decode_cols = [c for c in ("dataset", "fly", "fly_number", "trial_type", "trial_label", "fps") if c in ordered_cols]
    for col in decode_cols:
        if col == "fps":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            continue
        mapping = code_maps.get(col)
        if not mapping:
            continue
        inverse = {code: label for label, code in mapping.items()}
        df[col] = df[col].astype(int).map(inverse).fillna("UNKNOWN")

    if "fps" not in df.columns:
        df["fps"] = np.nan

    env_cols = [c for c in ordered_cols if c not in {"dataset", "fly", "fly_number", "trial_type", "trial_label", "fps"}]
    return df, env_cols


def _extract_env(row: pd.Series, env_cols: Sequence[str]) -> np.ndarray:
    env = row[env_cols].to_numpy(dtype=float)
    if env.ndim == 0:
        return np.empty(0, dtype=float)
    mask = np.isfinite(env) & (env > 0)
    return env[mask]


def _norm_key_text(value: object) -> str:
    return str(value).strip().lower()


def _norm_fly_number(value: object) -> str:
    if value is None:
        return ""
    try:
        num = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return "" if text.lower() in {"", "nan", "none"} else text

    if not np.isfinite(num):
        return ""
    rounded = round(num)
    if abs(num - rounded) < 1e-9:
        return str(int(rounded))
    text = f"{num:.6f}".rstrip("0").rstrip(".")
    return text


def _sorted_dir_cols(columns: Sequence[str]) -> list[str]:
    cols = [c for c in columns if str(c).startswith("dir_val_")]
    if not cols:
        return []

    def key(col: str) -> tuple[int, str]:
        m = re.search(r"(\d+)$", col)
        return (int(m.group(1)), col) if m else (10**9, col)

    return sorted(cols, key=key)


def _extract_env_from_dirvals(row: pd.Series, dir_cols: Sequence[str]) -> np.ndarray:
    if not dir_cols:
        return np.empty(0, dtype=float)
    env = row.loc[list(dir_cols)].to_numpy(dtype=float, copy=False)
    if env.ndim == 0:
        return np.empty(0, dtype=float)
    mask = np.isfinite(env) & (env > 0)
    return env[mask]


def _latency_to_cross(
    env: np.ndarray,
    fps: float,
    before_sec: float,
    during_sec: float,
    threshold_mult: float,
    odor_on_s: float = 30.0,
    odor_off_s: float = 60.0,
    odor_latency_s: float = 0.0,
) -> Optional[float]:
    """Compute latency to threshold crossing during the odor window.

    Args:
        env: Envelope signal (full trial duration)
        fps: Frames per second
        before_sec: Baseline window length (used for threshold calculation)
        during_sec: Response search window length (legacy parameter, may be ignored)
        threshold_mult: Multiplier for threshold = mean + k*std
        odor_on_s: Commanded odor ON time (seconds)
        odor_off_s: Commanded odor OFF time (seconds)
        odor_latency_s: Transit delay from valve command to fly (seconds)

    Returns:
        Latency in seconds from odor arrival to threshold crossing, or None
    """
    during_latency, _ = _latency_profile(
        env,
        fps,
        before_sec,
        threshold_mult,
        odor_on_s=odor_on_s,
        odor_off_s=odor_off_s,
        odor_latency_s=odor_latency_s,
    )
    return during_latency


def _latency_profile(
    env: np.ndarray,
    fps: float,
    before_sec: float,
    threshold_mult: float,
    *,
    odor_on_s: float = 30.0,
    odor_off_s: float = 60.0,
    odor_latency_s: float = 0.0,
) -> tuple[Optional[float], Optional[float]]:
    """Return latency tuple: (during_odor_latency, any_latency_after_odor_on)."""
    if env.size == 0 or not np.isfinite(fps) or fps <= 0:
        return None, None

    b_end = min(int(round(before_sec * fps)), env.size)
    before = env[:b_end]
    if before.size == 0:
        return None, None

    odor_on_effective = odor_on_s + odor_latency_s
    odor_off_effective = odor_off_s + odor_latency_s
    response_start = min(int(round(odor_on_effective * fps)), env.size)
    response_end = min(int(round(odor_off_effective * fps)), env.size)
    if response_start >= env.size:
        return None, None

    mu = float(np.nanmean(before))
    sd = float(np.nanstd(before))
    theta = mu + threshold_mult * sd

    during = env[response_start:response_end]
    idx_during = np.where(during > theta)[0]
    during_latency = float(idx_during[0]) / fps if idx_during.size else None

    after_on = env[response_start:]
    idx_any = np.where(after_on > theta)[0]
    any_latency = float(idx_any[0]) / fps if idx_any.size else None

    return during_latency, any_latency


def _latency_records_from_csv(
    csv_path: Path,
    *,
    before_sec: float,
    during_sec: float,
    threshold_mult: float,
    latency_ceiling: float,
    trials_of_interest: Sequence[int],
    fps_default: float,
    odor_on_s: float,
    odor_off_s: float,
    odor_latency_s: float,
    fly_state_csv: Path | None = None,
    fly_state_column: str = "FLY-State(1, 0, -1)",
) -> pd.DataFrame:
    df_all = pd.read_csv(csv_path)
    for col in ("dataset", "fly", "trial_label"):
        if col not in df_all.columns:
            raise RuntimeError(f"CSV missing required column '{col}': {csv_path}")

    if "fps" not in df_all.columns:
        df_all["fps"] = np.nan
    df_all["fps"] = pd.to_numeric(df_all["fps"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(fps_default)

    if "time_to_threshold" not in df_all.columns:
        df_all["time_to_threshold"] = np.nan
        need_save = True
    else:
        need_save = False

    if "trial_type" in df_all.columns:
        trial_type = df_all["trial_type"].astype(str).str.lower()
        df = df_all[trial_type == "training"].copy()
    else:
        df = df_all.copy()

    if fly_state_csv is not None and fly_state_csv.exists():
        truth = pd.read_csv(fly_state_csv)
        for col in ("dataset", "fly", fly_state_column):
            if col not in truth.columns:
                raise RuntimeError(
                    f"Fly-state CSV missing required column '{col}': {fly_state_csv}"
                )
        truth_state = pd.to_numeric(truth[fly_state_column], errors="coerce")
        truth = truth[truth_state.isin((0, -1))].copy()
        use_fly_number = "fly_number" in truth.columns and "fly_number" in df.columns

        if use_fly_number:
            blocked = {
                (
                    _norm_key_text(_canon_dataset(ds)),
                    _norm_key_text(fly),
                    _norm_fly_number(fly_number),
                )
                for ds, fly, fly_number in zip(
                    truth["dataset"], truth["fly"], truth["fly_number"]
                )
            }
            ds_norm = df["dataset"].map(_canon_dataset).map(_norm_key_text)
            fly_norm = df["fly"].map(_norm_key_text)
            fly_num_norm = df["fly_number"].map(_norm_fly_number)
            mask = [
                (ds, fly, fly_num) not in blocked
                for ds, fly, fly_num in zip(ds_norm, fly_norm, fly_num_norm)
            ]
        else:
            blocked = {
                (_norm_key_text(_canon_dataset(ds)), _norm_key_text(fly))
                for ds, fly in zip(truth["dataset"], truth["fly"])
            }
            ds_norm = df["dataset"].map(_canon_dataset).map(_norm_key_text)
            fly_norm = df["fly"].map(_norm_key_text)
            mask = [(ds, fly) not in blocked for ds, fly in zip(ds_norm, fly_norm)]
        df = df[mask].copy()

    df["trial_num"] = df["trial_label"].map(_trial_num)
    df = df[df["trial_num"].isin(set(int(t) for t in trials_of_interest))].copy()
    if df.empty:
        raise RuntimeError("No training rows in CSV matched the requested trial numbers.")

    dir_cols = _sorted_dir_cols(df.columns)
    if not dir_cols and df["time_to_threshold"].isna().any():
        raise RuntimeError(
            "CSV lacks both dir_val_* trace columns and complete time_to_threshold values."
        )

    ttt_existing = pd.to_numeric(df["time_to_threshold"], errors="coerce")
    during_cache: dict[int, Optional[float]] = {}
    any_cache: dict[int, Optional[float]] = {}
    for idx, row in df.iterrows():
        lat_during: Optional[float] = None
        lat_any: Optional[float] = None
        if dir_cols:
            env = _extract_env_from_dirvals(row, dir_cols)
            lat_during, lat_any = _latency_profile(
                env,
                float(row.get("fps", fps_default)),
                before_sec,
                threshold_mult,
                odor_on_s=odor_on_s,
                odor_off_s=odor_off_s,
                odor_latency_s=odor_latency_s,
            )

        existing = ttt_existing.loc[idx] if idx in ttt_existing.index else math.nan
        if lat_during is None and np.isfinite(existing):
            lat_during = float(existing)
            if lat_any is None:
                lat_any = lat_during
        elif not np.isfinite(existing) and lat_during is not None:
            df.at[idx, "time_to_threshold"] = float(lat_during)
            need_save = True

        during_cache[idx] = lat_during
        any_cache[idx] = lat_any

    if need_save:
        vals = pd.to_numeric(df["time_to_threshold"], errors="coerce")
        df_all.loc[vals.index, "time_to_threshold"] = vals.to_numpy(dtype=float)
        df_all.to_csv(csv_path, index=False)

    records = []
    for idx, row in df.iterrows():
        latency = during_cache.get(idx)
        latency_any = any_cache.get(idx)

        if latency_any is None:
            response_kind = "no_response_any"
            plot_latency = None
            lat_for_mean = float(latency_ceiling)
        elif latency_any > latency_ceiling:
            response_kind = "response_after_ceiling"
            plot_latency = float(latency_ceiling)
            lat_for_mean = float(latency_ceiling)
        else:
            response_kind = "response_within_ceiling"
            plot_latency = float(latency_any)
            lat_for_mean = float(latency_any)

        dataset_val = row.get("dataset", "UNKNOWN")
        dataset_text = dataset_val if isinstance(dataset_val, str) else str(dataset_val)
        fly_val = row.get("fly", "UNKNOWN")
        fly_text = fly_val if isinstance(fly_val, str) else str(fly_val)
        fly_num_text = _norm_fly_number(row.get("fly_number", ""))
        records.append(
            {
                "dataset": dataset_text,
                "dataset_canon": _canon_dataset(dataset_text),
                "fly": fly_text,
                "fly_number": fly_num_text,
                "trial_num": int(row["trial_num"]),
                "latency": latency,
                "latency_any": latency_any,
                "plot_latency": plot_latency,
                "response_kind": response_kind,
                "lat_for_mean": lat_for_mean,
            }
        )

    if not records:
        raise RuntimeError("No latency records could be built from CSV.")
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Latency visualisations
# ---------------------------------------------------------------------------


def _plot_latency_per_fly(
    lat_df: pd.DataFrame,
    out_dir: Path,
    *,
    latency_ceiling: float,
    trials_of_interest: Sequence[int],
    overwrite: bool,
) -> None:
    work = lat_df.copy()
    if "fly_number" not in work.columns:
        work["fly_number"] = ""
    work["dataset_canon"] = work["dataset_canon"].fillna("UNKNOWN")
    work["fly"] = work["fly"].fillna("UNKNOWN")
    work["fly_number_norm"] = work["fly_number"].map(_norm_fly_number)

    for (dataset_canon, fly, fly_number), subset in work.groupby(
        ["dataset_canon", "fly", "fly_number_norm"],
        dropna=False,
        sort=True,
    ):
        if subset.empty:
            continue

        target_dir = _target_dir(out_dir, [str(dataset_canon)])
        fly_slug = _safe_dirname(str(fly))
        fly_suffix = f"_fly{fly_number}" if fly_number else ""
        out_png = target_dir / f"{fly_slug}{fly_suffix}_training_{'_'.join(map(str, trials_of_interest))}_latency.png"
        if out_png.exists() and not overwrite:
            continue

        fly_label = str(fly)
        if fly_number:
            fly_label = f"{fly_label} (fly {fly_number})"

        labels = [f"Training {n}" for n in trials_of_interest]
        values: list[float | None] = []
        annotations: list[str | None] = []
        colors: list[str | None] = []
        no_per_flags: list[bool] = []

        for trial_num in trials_of_interest:
            trial_rows = subset[subset["trial_num"] == trial_num]
            if trial_rows.empty:
                values.append(None)
                annotations.append(None)
                colors.append(None)
                no_per_flags.append(False)
                continue

            kinds = trial_rows["response_kind"].astype(str).tolist() if "response_kind" in trial_rows.columns else []
            if any(k == "response_within_ceiling" for k in kinds):
                vals = pd.to_numeric(
                    trial_rows.loc[trial_rows["response_kind"] == "response_within_ceiling", "plot_latency"],
                    errors="coerce",
                ).to_numpy(dtype=float)
                finite = vals[np.isfinite(vals)]
                plot_val = float(finite.mean()) if finite.size else float(latency_ceiling)
                values.append(plot_val)
                annotations.append(f"{plot_val:.2f}s")
                colors.append("#1A1A1A")
                no_per_flags.append(False)
            elif any(k == "response_after_ceiling" for k in kinds):
                values.append(float(latency_ceiling))
                annotations.append("NR")
                colors.append("#BDBDBD")
                no_per_flags.append(False)
            else:
                # Explicitly no threshold crossing after odor-on in this trial.
                values.append(None)
                annotations.append(None)
                colors.append(None)
                no_per_flags.append(True)

        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        x = np.arange(len(labels))
        bars: list[plt.Rectangle | None] = []
        for idx, val in enumerate(values):
            if val is None:
                bars.append(None)
                continue
            bar = ax.bar(
                x[idx],
                val,
                width=0.6,
                color=colors[idx] or "#1A1A1A",
                edgecolor="black",
                linewidth=1.0,
            )[0]
            bars.append(bar)

        for idx, (bar, text) in enumerate(zip(bars, annotations)):
            if bar is None or text is None:
                continue
            ypos = max(bar.get_height() * 0.5, 0.35)
            txt_color = "white" if text != "NR" else "#444444"
            ax.text(bar.get_x() + bar.get_width() / 2, ypos, text, ha="center", va="center", fontsize=10, color=txt_color)

        for idx, no_per in enumerate(no_per_flags):
            if no_per:
                ax.text(x[idx], 0.14, "No PER", ha="center", va="bottom", fontsize=9, color="#666666")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Time After Odor Sent (s)")
        ax.set_ylim(0, latency_ceiling + 2.5)
        ax.axhline(latency_ceiling, linestyle="--", linewidth=1.1, color="#444444")
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(0.995, latency_ceiling + 0.12, f"NR if > {latency_ceiling:.1f} s", transform=trans, ha="right", va="bottom", fontsize=10, color="#444444", clip_on=False)
        ax.set_title(f"{fly_label} — Time to PER", pad=10, fontsize=14, weight="bold")

        fig.tight_layout()
        if _should_write(out_png, overwrite):
            fig.savefig(out_png, dpi=300)
        plt.close(fig)


def _plot_latency_by_odor(
    lat_df: pd.DataFrame,
    out_dir: Path,
    *,
    latency_ceiling: float,
    trials_of_interest: Sequence[int],
    overwrite: bool,
) -> None:
    for odor in sorted(lat_df["dataset_canon"].unique()):
        target_dir = _target_dir(out_dir, odor)
        filename = f"{odor}_training_{'_'.join(map(str, trials_of_interest))}_mean_latency.png"
        out_png = target_dir / filename
        if out_png.exists() and not overwrite:
            continue

        subset = lat_df[lat_df["dataset_canon"] == odor]
        labels = [f"Training {n}" for n in trials_of_interest]
        odor_label = _dataset_label(odor)

        if subset.empty:
            fig, ax = plt.subplots(figsize=(6.8, 3.2))
            ax.set_title(f"{odor_label} — Mean Time to PER", pad=10, fontsize=14, weight="bold")
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_ylim(0, latency_ceiling + 2.0)
            ax.text(0.5, 0.55, "NR", transform=ax.transAxes, ha="center", va="center", fontsize=18, color="#666666", weight="bold")
            ax.set_ylabel("Time After Odor Sent (s)")
            fig.tight_layout()
            if _should_write(out_png, overwrite):
                fig.savefig(out_png, dpi=300)
            plt.close(fig)
            continue

        means = []
        sems = []
        counts = []
        for trial_num in trials_of_interest:
            values = subset[subset["trial_num"] == trial_num]["lat_for_mean"].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            counts.append(finite.size)
            if finite.size == 0:
                means.append(math.nan)
                sems.append(math.nan)
                continue
            means.append(float(finite.mean()))
            if finite.size > 1:
                sems.append(float(finite.std(ddof=1)) / math.sqrt(finite.size))
            else:
                sems.append(0.0)

        y = np.nan_to_num(np.array(means, dtype=float), nan=0.0)
        yerr_up = np.nan_to_num(np.array(sems, dtype=float), nan=0.0)
        yerr = np.vstack([np.zeros_like(yerr_up), yerr_up])

        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        x = np.arange(len(labels))
        bars = ax.bar(x, y, width=0.6, color="#1A1A1A", edgecolor="black", linewidth=1.0)
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.2, capsize=4)

        for idx, bar in enumerate(bars):
            n_resp = counts[idx]
            sem_val = yerr_up[idx]
            if n_resp == 0:
                label_y = max(0.5, y[idx] + 0.08)
                ax.text(bar.get_x() + bar.get_width() / 2, label_y, "NR", ha="center", va="bottom", fontsize=9, color="#444444")
                continue
            inside = max(y[idx] * 0.5, min(y[idx] - 0.10, y[idx] * 0.90))
            ax.text(bar.get_x() + bar.get_width() / 2, inside, f"{y[idx]:.2f}s", ha="center", va="top", fontsize=10, color="white")
            top = y[idx] + (sem_val if np.isfinite(sem_val) else 0.0) + 0.06
            ax.text(bar.get_x() + bar.get_width() / 2, top, f"SEM={sem_val:.2f}s\nn={n_resp}", ha="center", va="bottom", fontsize=9, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Time After Odor Sent (s)")
        ymax = max(latency_ceiling + 2.0, float((y + yerr_up).max()) + 1.2)
        ax.set_ylim(0, ymax)
        ax.axhline(latency_ceiling, linestyle="--", linewidth=1.0, color="#6f6f6f")
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(0.995, latency_ceiling + 0.08, f"NR if > {latency_ceiling:.1f} s", transform=trans, ha="right", va="bottom", fontsize=9, color="#6f6f6f")
        ax.set_title(f"{odor_label} — Mean Time to PER", pad=10, fontsize=14, weight="bold")

        fig.tight_layout()
        if _should_write(out_png, overwrite):
            fig.savefig(out_png, dpi=300)
        plt.close(fig)


def _plot_latency_grand_means(
    lat_df: pd.DataFrame,
    out_dir: Path,
    latency_ceiling: float,
    overwrite: bool,
) -> None:
    odors = sorted(lat_df["dataset_canon"].dropna().unique())
    rows = []
    means = []
    sems = []
    counts = []

    for odor in odors:
        values = lat_df[lat_df["dataset_canon"] == odor]["lat_for_mean"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        n_resp = finite.size
        counts.append(n_resp)
        if n_resp == 0:
            means.append(math.nan)
            sems.append(math.nan)
        else:
            mu = float(finite.mean())
            if n_resp > 1:
                sd = float(finite.std(ddof=1))
                sem = sd / math.sqrt(n_resp)
            else:
                sem = 0.0
            means.append(mu)
            sems.append(sem)
        rows.append({"odor": odor, "n_resp": n_resp, "mean_s": means[-1], "sem_s": sems[-1]})

    csv_path = out_dir / "grand_mean_by_odor_latency.csv"
    if _should_write(csv_path, overwrite):
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    out_png = out_dir / "grand_mean_by_odor_latency.png"
    if out_png.exists() and not overwrite:
        return

    if sum(counts) == 0:
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        ax.set_title("Grand Mean Time to Reaction by Trained Odor", pad=10, fontsize=14, weight="bold")
        ax.set_xticks(np.arange(len(odors)))
        ax.set_xticklabels(odors)
        ax.set_ylim(0, latency_ceiling + 2.0)
        ax.text(0.5, 0.55, "NR", transform=ax.transAxes, ha="center", va="center", fontsize=18, color="#666666", weight="bold")
        ax.set_ylabel("Time After Odor Sent (s)")
        ax.axhline(latency_ceiling, linestyle="--", linewidth=1.0, color="#6f6f6f")
        fig.tight_layout()
        if _should_write(out_png, overwrite):
            fig.savefig(out_png, dpi=300)
        plt.close(fig)
        return

    y = np.nan_to_num(np.array(means, dtype=float), nan=0.0)
    yerr_up = np.nan_to_num(np.array(sems, dtype=float), nan=0.0)
    yerr = np.vstack([np.zeros_like(yerr_up), yerr_up])

    fig, ax = plt.subplots(figsize=(max(7.2, 1.8 * len(odors)), 3.8))
    x = np.arange(len(odors))
    bars = ax.bar(x, y, width=0.6, color="#1A1A1A", edgecolor="black", linewidth=1.0)
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.2, capsize=4)

    for idx, bar in enumerate(bars):
        n_resp = counts[idx]
        sem_val = yerr_up[idx]
        if n_resp == 0:
            label_y = max(0.5, y[idx] + 0.08)
            ax.text(bar.get_x() + bar.get_width() / 2, label_y, "NR", ha="center", va="bottom", fontsize=9, color="#444444")
            continue
        label_y = y[idx] + (sem_val if np.isfinite(sem_val) else 0.0) + 0.06
        ax.text(bar.get_x() + bar.get_width() / 2, label_y, f"SEM={sem_val:.2f} s\nn={n_resp}", ha="center", va="bottom", fontsize=9, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(odors)
    ax.set_ylabel("Time After Odor Sent (s)")
    ymax = max(latency_ceiling + 2.0, float((y + yerr_up).max()) + 1.2)
    ax.set_ylim(0, ymax)
    ax.axhline(latency_ceiling, linestyle="--", linewidth=1.0, color="#6f6f6f")
    ax.set_title("Grand Mean Time to Reaction by Trained Odor", pad=10, fontsize=14, weight="bold")

    fig.tight_layout()
    if _should_write(out_png, overwrite):
        fig.savefig(out_png, dpi=300)
    plt.close(fig)


def latency_reports(
    matrix_path: Path | None,
    codes_json: Path | None,
    out_dir: Path,
    *,
    csv_path: Path | None = None,
    fly_state_csv: Path | None = None,
    fly_state_column: str = "FLY-State(1, 0, -1)",
    before_sec: float,
    during_sec: float,
    threshold_mult: float | None = None,
    threshold_std_mult: float | None = None,
    latency_ceiling: float,
    trials_of_interest: Sequence[int],
    fps_default: float,
    overwrite: bool,
    odor_on_s: float = 30.0,
    odor_off_s: float = 60.0,
    odor_latency_s: float = 0.0,
) -> None:
    """Generate latency plots and summaries for the requested trials.

    Historically this function only accepted ``threshold_std_mult`` while recent
    workflow updates switched to ``threshold_mult``.  Accept both names (with
    the former taking precedence when both supplied) so that existing configs
    and cached CLI invocations continue to function without raising
    ``TypeError`` when mixing versions of the scripts.

    The odor timing parameters (odor_on_s, odor_off_s, odor_latency_s) account
    for the actual time the odor reaches the fly, not the commanded valve time.
    """
    if threshold_mult is None and threshold_std_mult is None:
        # Maintain the previous implicit default of 4.0 when neither keyword is
        # provided.  This mirrors the CLI default and the behaviour prior to the
        # signature change.
        threshold_mult = 4.0

    # Allow either keyword while favouring the legacy ``threshold_std_mult`` if
    # both are present.  This keeps behaviour consistent for older configs that
    # may still pass the legacy name indirectly (e.g. via cached YAML renders).
    if threshold_std_mult is not None:
        threshold_mult = threshold_std_mult

    if threshold_mult is None:
        raise ValueError("Latency threshold multiplier must be provided.")

    threshold_mult = float(threshold_mult)

    if csv_path is not None and csv_path.exists():
        lat_df = _latency_records_from_csv(
            csv_path,
            before_sec=before_sec,
            during_sec=during_sec,
            threshold_mult=threshold_mult,
            latency_ceiling=latency_ceiling,
            trials_of_interest=trials_of_interest,
            fps_default=fps_default,
            odor_on_s=odor_on_s,
            odor_off_s=odor_off_s,
            odor_latency_s=odor_latency_s,
            fly_state_csv=fly_state_csv,
            fly_state_column=fly_state_column,
        )
    else:
        if matrix_path is None or codes_json is None:
            raise ValueError("Provide csv_path, or provide both matrix_path and codes_json.")
        df, env_cols = _load_envelope_matrix(matrix_path, codes_json)
        df = df[df["trial_type"].str.lower() == "training"].copy()
        if df.empty:
            raise RuntimeError("No training trials present in matrix; cannot compute latency.")

        df["fps"] = df["fps"].replace([np.inf, -np.inf], np.nan).fillna(fps_default)
        df["dataset_canon"] = df["dataset"].map(_canon_dataset)

        records = []
        for _, row in df.iterrows():
            trial_num = _trial_num(row["trial_label"])
            if trial_num not in trials_of_interest:
                continue
            env = _extract_env(row, env_cols)
            fps = float(row.get("fps", fps_default))
            latency, latency_any = _latency_profile(
                env, fps, before_sec, threshold_mult,
                odor_on_s=odor_on_s,
                odor_off_s=odor_off_s,
                odor_latency_s=odor_latency_s,
            )
            if latency_any is None:
                response_kind = "no_response_any"
                plot_latency = None
                lat_for_mean = float(latency_ceiling)
            elif latency_any > latency_ceiling:
                response_kind = "response_after_ceiling"
                plot_latency = float(latency_ceiling)
                lat_for_mean = float(latency_ceiling)
            else:
                response_kind = "response_within_ceiling"
                plot_latency = float(latency_any)
                lat_for_mean = float(latency_any)
            records.append(
                {
                    "dataset": row["dataset"],
                    "dataset_canon": row["dataset_canon"],
                    "fly": row["fly"],
                    "fly_number": _norm_fly_number(row.get("fly_number", "")),
                    "trial_num": trial_num,
                    "latency": latency,
                    "latency_any": latency_any,
                    "plot_latency": plot_latency,
                    "response_kind": response_kind,
                    "lat_for_mean": lat_for_mean,
                }
            )

        if not records:
            raise RuntimeError("No training trials matched the requested trial numbers.")

        lat_df = pd.DataFrame(records)

    _plot_latency_per_fly(
        lat_df,
        out_dir,
        latency_ceiling=latency_ceiling,
        trials_of_interest=trials_of_interest,
        overwrite=overwrite,
    )
    _plot_latency_by_odor(
        lat_df,
        out_dir,
        latency_ceiling=latency_ceiling,
        trials_of_interest=trials_of_interest,
        overwrite=overwrite,
    )
    _plot_latency_grand_means(lat_df, out_dir, latency_ceiling, overwrite)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    env_parser = sub.add_parser("envelopes", help="Render per-fly training envelopes from the matrix outputs.")
    env_parser.add_argument("--matrix-npy", type=Path, required=True, help="Float16 matrix produced by the convert step.")
    env_parser.add_argument("--codes-json", type=Path, required=True, help="JSON metadata file from the convert step.")
    env_parser.add_argument("--out-dir", type=Path, required=True, help="Destination directory for envelope figures.")
    env_parser.add_argument("--latency-sec", type=float, default=0.0, help="Mean odor transit latency in seconds.")
    env_parser.add_argument("--fps-default", type=float, default=40.0, help="Fallback FPS when decoding rows.")
    env_parser.add_argument("--odor-on-s", type=float, default=30.0, help="Commanded odor ON timestamp (seconds).")
    env_parser.add_argument("--odor-off-s", type=float, default=60.0, help="Commanded odor OFF timestamp (seconds).")
    env_parser.add_argument(
        "--odor-latency-s",
        type=float,
        default=0.0,
        help="Transit delay between valve command and odor at the fly (seconds).",
    )
    env_parser.add_argument("--after-show-sec", type=float, default=30.0, help="Duration to display after odor OFF (seconds).")
    env_parser.add_argument("--threshold-std-mult", type=float, default=4.0, help="Threshold multiplier for baseline std dev.")
    env_parser.add_argument(
        "--light-annotation-mode",
        choices=("none", "line", "paired-span"),
        default="none",
        help="Training light annotation style for envelopes.",
    )
    env_parser.add_argument(
        "--max-flies",
        type=int,
        default=None,
        help="Optional cap for number of fly figures to render.",
    )
    env_parser.add_argument("--overwrite", action="store_true", help="Rebuild plots even if the target files exist.")

    lat_parser = sub.add_parser("latency", help="Compute latency-to-threshold metrics from the matrix outputs.")
    lat_parser.add_argument("--matrix-npy", type=Path, required=False, help="Float16 matrix produced by the convert step.")
    lat_parser.add_argument("--codes-json", type=Path, required=False, help="JSON metadata file from the convert step.")
    lat_parser.add_argument("--csv-path", type=Path, required=False, help="Wide training CSV used for CSV-first latency plots.")
    lat_parser.add_argument("--fly-state-csv", type=Path, required=False, help="CSV with fly-state labels; excludes only flies marked 0 or -1 when provided.")
    lat_parser.add_argument("--fly-state-column", type=str, default="FLY-State(1, 0, -1)", help="Column name in fly-state CSV indicating keep/drop state.")
    lat_parser.add_argument("--out-dir", type=Path, required=True, help="Directory for figures and CSV summaries.")
    lat_parser.add_argument("--before-sec", type=float, default=30.0, help="Baseline window length in seconds.")
    lat_parser.add_argument("--during-sec", type=float, default=35.0, help="During window length in seconds.")
    lat_parser.add_argument("--threshold-mult", type=float, default=4.0, help="Threshold multiplier (mu + k*std).")
    lat_parser.add_argument(
        "--threshold-std-mult",
        type=float,
        default=None,
        help="Deprecated alias for --threshold-mult maintained for backward compatibility.",
    )
    lat_parser.add_argument("--latency-ceiling", type=float, default=10.0, help="Cap for marking NR trials in seconds.")
    lat_parser.add_argument("--trials", nargs="+", type=int, default=[4, 6, 8], help="Training trial numbers to analyse.")
    lat_parser.add_argument("--fps-default", type=float, default=40.0, help="Fallback FPS when metadata missing.")
    lat_parser.add_argument("--overwrite", action="store_true", help="Rebuild plots even if the target files exist.")

    # Add a new subcommand for adding time_to_threshold column
    add_col_parser = sub.add_parser("add-threshold-col", help="Add time_to_threshold column to training CSV.")
    add_col_parser.add_argument("--csv", type=Path, required=True, help="Training CSV file to augment.")
    add_col_parser.add_argument("--matrix-npy", type=Path, required=True, help="Float16 matrix produced by the convert step.")
    add_col_parser.add_argument("--codes-json", type=Path, required=True, help="JSON metadata file from the convert step.")
    add_col_parser.add_argument("--before-sec", type=float, default=30.0, help="Baseline window length in seconds.")
    add_col_parser.add_argument("--during-sec", type=float, default=35.0, help="During window length in seconds.")
    add_col_parser.add_argument("--threshold-mult", type=float, default=2.0, help="Threshold multiplier (mu + k*std).")
    add_col_parser.add_argument("--fps-default", type=float, default=40.0, help="Fallback FPS when metadata missing.")
    add_col_parser.add_argument(
        "--odor-on-s",
        type=float,
        default=30.0,
        help="Commanded odor ON time (seconds).",
    )
    add_col_parser.add_argument(
        "--odor-off-s",
        type=float,
        default=60.0,
        help="Commanded odor OFF time (seconds).",
    )
    add_col_parser.add_argument(
        "--odor-latency-s",
        type=float,
        default=2.15,
        help="Transit delay between valve command and odor at the fly (seconds).",
    )

    return parser


def add_time_to_threshold_column(
    csv_path: Path,
    matrix_path: Path,
    codes_json: Path,
    *,
    before_sec: float,
    during_sec: float,
    threshold_mult: float,
    fps_default: float,
    odor_on_s: float = 30.0,
    odor_off_s: float = 60.0,
    odor_latency_s: float = 0.0,
) -> None:
    """Add 'time_to_threshold' column to training CSV.

    This column represents the time (in seconds) from actual odor arrival
    (odor_on_s + odor_latency_s) to when the signal crosses the threshold.
    """
    # Load the CSV file
    df_csv = pd.read_csv(csv_path)

    # Load the matrix
    df_matrix, env_cols = _load_envelope_matrix(matrix_path, codes_json)
    df_matrix = df_matrix[df_matrix["trial_type"].str.lower() == "training"].copy()

    if df_matrix.empty:
        raise RuntimeError("No training trials found in matrix.")

    df_matrix["fps"] = df_matrix["fps"].replace([np.inf, -np.inf], np.nan).fillna(fps_default)

    use_fly_number = "fly_number" in df_matrix.columns and "fly_number" in df_csv.columns

    # Create a mapping from fly key to latency
    latency_map = {}
    for _, row in df_matrix.iterrows():
        env = _extract_env(row, env_cols)
        fps = float(row.get("fps", fps_default))
        latency = _latency_to_cross(
            env, fps, before_sec, during_sec, threshold_mult,
            odor_on_s=odor_on_s,
            odor_off_s=odor_off_s,
            odor_latency_s=odor_latency_s,
        )
        if use_fly_number:
            key = (
                row["dataset"],
                row["fly"],
                _norm_fly_number(row.get("fly_number", "")),
                row["trial_label"],
            )
        else:
            key = (row["dataset"], row["fly"], row["trial_label"])
        latency_map[key] = latency

    # Add the column to the CSV
    def get_latency(row):
        if use_fly_number:
            key = (
                row["dataset"],
                row["fly"],
                _norm_fly_number(row.get("fly_number", "")),
                row["trial_label"],
            )
        else:
            key = (row["dataset"], row["fly"], row["trial_label"])
        return latency_map.get(key)

    df_csv["time_to_threshold"] = df_csv.apply(get_latency, axis=1)

    # Save the updated CSV
    df_csv.to_csv(csv_path, index=False)
    print(f"✓ Added 'time_to_threshold' column to {csv_path}")
    print(f"  - Rows with data: {df_csv['time_to_threshold'].notna().sum()}")
    print(f"  - Mean latency: {df_csv['time_to_threshold'].mean():.2f}s")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "envelopes":
        cfg = EnvelopePlotConfig(
            matrix_npy=args.matrix_npy.expanduser().resolve(),
            codes_json=args.codes_json.expanduser().resolve(),
            out_dir=args.out_dir.expanduser().resolve(),
            latency_sec=args.latency_sec,
            fps_default=args.fps_default,
            odor_on_s=args.odor_on_s,
            odor_off_s=args.odor_off_s,
            odor_latency_s=args.odor_latency_s,
            after_show_sec=args.after_show_sec,
            threshold_std_mult=args.threshold_std_mult,
            trial_type="training",
            light_annotation_mode=args.light_annotation_mode,
            max_flies=args.max_flies,
            overwrite=args.overwrite,
        )
        generate_envelope_plots(cfg)
        return

    if args.command == "latency":
        latency_reports(
            args.matrix_npy.expanduser().resolve() if args.matrix_npy else None,
            args.codes_json.expanduser().resolve() if args.codes_json else None,
            args.out_dir.expanduser().resolve(),
            csv_path=args.csv_path.expanduser().resolve() if args.csv_path else None,
            fly_state_csv=args.fly_state_csv.expanduser().resolve() if args.fly_state_csv else None,
            fly_state_column=args.fly_state_column,
            before_sec=args.before_sec,
            during_sec=args.during_sec,
            threshold_mult=args.threshold_mult,
            threshold_std_mult=args.threshold_std_mult,
            latency_ceiling=args.latency_ceiling,
            trials_of_interest=tuple(args.trials),
            fps_default=args.fps_default,
            overwrite=args.overwrite,
        )
        return

    if args.command == "add-threshold-col":
        add_time_to_threshold_column(
            args.csv.expanduser().resolve(),
            args.matrix_npy.expanduser().resolve(),
            args.codes_json.expanduser().resolve(),
            before_sec=args.before_sec,
            during_sec=args.during_sec,
            threshold_mult=args.threshold_mult,
            fps_default=args.fps_default,
            odor_on_s=args.odor_on_s,
            odor_off_s=args.odor_off_s,
            odor_latency_s=args.odor_latency_s,
        )
        return

    parser.error(f"Unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
