#!/usr/bin/env python3
"""Training-focused envelope analytics and visualisations.

This module ports the user's notebook snippets for training-trial analyses
into an efficient, scriptable CLI.  It can

* render per-fly Hilbert envelope subplots for every ``training_*`` CSV, marking
  valve timing and global peaks;
* gather those plots into a single folder; and
* derive latency-to-threshold summaries directly from the float16 matrix emitted
  by :mod:`scripts.envelope_exports`.

The commands default to reading paths from ``config.yaml`` so they integrate
with the existing pipeline, but every file path and timing constant can be
overridden from the CLI.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from scipy.signal import hilbert

try:  # lightweight helper so the CLI can reuse pipeline defaults when desired
    from src.fbpipe.config import load_settings
except ModuleNotFoundError:  # pragma: no cover - fallback for isolated use
    load_settings = None  # type: ignore


# ---------------------------------------------------------------------------
# Shared mappings (mirrors scripts.envelope_visuals)
# ---------------------------------------------------------------------------

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
    "optogenetics benzaldehyde": "opto_benz",
    "optogenetics benzaldehyde 1": "opto_benz_1",
    "optogenetics ethyl butyrate": "opto_EB",
}

DISPLAY_LABEL: Mapping[str, str] = {
    "ACV": "ACV",
    "3-octonol": "3-Octonol",
    "Benz": "Benzaldehyde",
    "10s_Odor_Benz": "Benzaldehyde",
    "EB": "Ethyl Butyrate",
    "opto_benz": "Benzaldehyde",
    "opto_EB": "Ethyl Butyrate",
    "opto_benz_1": "Benzaldehyde",
}

HEXANOL_LABEL = "Hexanol"


# ---------------------------------------------------------------------------
# Dataclasses + helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    root: Path
    fps_default: float = 40.0
    window_sec: float = 0.25
    odor_on_s: float = 30.0
    odor_off_s: float = 60.0
    measure_cols: Sequence[str] = ("distance_percentage_2_6", "distance_percentage")

    @property
    def window_frames(self) -> int:
        return max(int(round(self.window_sec * self.fps_default)), 1)


TRAINING_REGEX = re.compile(r"training_(\d+)", re.I)
PLOT_SUBDIR = Path("RMS_calculations/envelope_over_time_plots")


def _default_root() -> Path:
    if load_settings is None:
        return Path.cwd()
    cfg = load_settings(Path("config.yaml"))
    return Path(cfg.main_directory).expanduser()


def _resolve_measure_column(columns: Iterable[str], measure_cols: Sequence[str]) -> Optional[str]:
    for candidate in measure_cols:
        if candidate in columns:
            return candidate
    return None


def _trial_label(csv_path: Path) -> str:
    match = TRAINING_REGEX.search(csv_path.stem)
    if match:
        return f"training_{match.group(1)}"
    return csv_path.stem


def _trial_sort_key(label: str) -> tuple[int, str]:
    match = TRAINING_REGEX.search(label)
    number = int(match.group(1)) if match else math.inf
    return (number, label)


def _compute_envelope(series: pd.Series, window_frames: int) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)
    if values.empty:
        return np.empty(0, dtype=float)
    analytic = hilbert(values.to_numpy(dtype=float))
    envelope = np.abs(analytic)
    smoothed = (
        pd.Series(envelope, index=values.index)
        .rolling(window=window_frames, center=True, min_periods=1)
        .mean()
    )
    return smoothed.to_numpy(dtype=float)


def _load_training_trial(
    csv_path: Path,
    cfg: TrainingConfig,
) -> Optional[tuple[str, np.ndarray, np.ndarray]]:
    try:
        header = pd.read_csv(csv_path, nrows=0)
    except Exception as exc:  # pragma: no cover - defensive logging path
        print(f"[WARN] Skip {csv_path.name}: failed to read header ({exc})")
        return None

    measure_col = _resolve_measure_column(header.columns, cfg.measure_cols)
    if measure_col is None:
        print(f"[WARN] Skip {csv_path.name}: expected one of {cfg.measure_cols}")
        return None

    time_col = "time_seconds" if "time_seconds" in header.columns else None
    usecols = [measure_col]
    if time_col:
        usecols.append(time_col)

    try:
        df = pd.read_csv(csv_path, usecols=usecols)
    except ValueError:
        # Some CSVs may lack the optional column despite being in the header row
        df = pd.read_csv(csv_path, usecols=[measure_col])
        time_col = None

    series = pd.to_numeric(df[measure_col], errors="coerce").fillna(0.0)
    env = _compute_envelope(series, cfg.window_frames)
    if env.size == 0:
        return None

    if time_col:
        time_s = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    else:
        time_s = np.arange(env.size, dtype=float) / max(cfg.fps_default, 1e-6)

    label = _trial_label(csv_path)
    return label, time_s[: env.size], env


def _peak(value: np.ndarray) -> Optional[tuple[int, float]]:
    if not np.any(np.isfinite(value)):
        return None
    idx = int(np.nanargmax(value))
    return idx, float(value[idx])


def plot_training_envelopes(
    fly_name: str,
    trials: MutableMapping[str, tuple[np.ndarray, np.ndarray]],
    cfg: TrainingConfig,
    out_path: Path,
) -> None:
    if not trials:
        print(f"[WARN] {fly_name}: no training trials to plot.")
        return

    ordered = OrderedDict(sorted(trials.items(), key=lambda item: _trial_sort_key(item[0])))
    global_max = max((float(np.nanmax(env)) for _, (_, env) in ordered.items() if env.size), default=0.0)
    ymax = global_max * 1.02 if global_max > 0 else 1.0

    plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300})
    fig, axes = plt.subplots(len(ordered), 1, figsize=(10, 2.6 * len(ordered)), sharex=True)
    if len(ordered) == 1:
        axes = [axes]

    for ax, (label, (time_s, env)) in zip(axes, ordered.items()):
        ax.plot(time_s, env, linewidth=1.1, color="black", clip_on=False)

        peak = _peak(env)
        if peak is not None:
            idx, value = peak
            ax.plot(time_s[idx], value, marker="o", markersize=8, color="red", zorder=5)

        ax.axvline(cfg.odor_on_s, linestyle="--", linewidth=1.0, color="#444444")
        ax.axvline(cfg.odor_off_s, linestyle="--", linewidth=1.0, color="#444444")
        ax.axvspan(cfg.odor_on_s, cfg.odor_off_s, alpha=0.18, color="#aaaaaa")

        ax.set_ylim(0.0, ymax)
        ax.set_ylabel("Envelope")
        ax.set_title(label)
        ax.grid(True, linewidth=0.4, alpha=0.6)

        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="red", linestyle="None", markersize=6, label="Peak"),
            plt.Line2D([0], [0], linestyle="--", color="#444444", label="Valve on/off"),
            plt.Rectangle((0, 0), 1, 1, alpha=0.18, color="#aaaaaa", label="Odor on window"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=True)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"{fly_name}: Analytic Envelope — TRAINING (window={cfg.window_sec:.2f}s; odor {cfg.odor_on_s:.0f}–{cfg.odor_off_s:.0f}s)",
        y=0.98,
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


def run_training_envelopes(cfg: TrainingConfig) -> None:
    for fly_dir in sorted(p for p in cfg.root.iterdir() if p.is_dir()):
        rms_dir = fly_dir / "RMS_calculations"
        if not rms_dir.is_dir():
            continue

        trials: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for csv_path in sorted(rms_dir.glob("*training*.csv")):
            loaded = _load_training_trial(csv_path, cfg)
            if loaded is None:
                continue
            label, time_s, env = loaded
            trials[label] = (time_s, env)

        if not trials:
            continue

        out_path = fly_dir / PLOT_SUBDIR / f"{fly_dir.name}_TRAINING_envelope_over_time_subplots.png"
        plot_training_envelopes(fly_dir.name, trials, cfg, out_path)


def collect_training_plots(root: Path, dest_folder: str) -> None:
    root = root.expanduser().resolve()
    dest = root / dest_folder
    dest.mkdir(parents=True, exist_ok=True)

    count = 0
    for fly_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        plot_path = fly_dir / PLOT_SUBDIR / f"{fly_dir.name}_TRAINING_envelope_over_time_subplots.png"
        if plot_path.is_file():
            target = dest / f"{fly_dir.name}_envelope_over_time_subplots_training.png"
            shutil.copy2(plot_path, target)
            count += 1
    print(f"[OK] Collected {count} training plots → {dest}")


# ---------------------------------------------------------------------------
# Latency analytics using the float16 matrix
# ---------------------------------------------------------------------------


def _canon_dataset(value: str) -> str:
    if not isinstance(value, str):
        return "UNKNOWN"
    key = value.strip().lower()
    return ODOR_CANON.get(key, value.strip())


def _safe_dirname(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "export"


def _trial_num(label: str) -> int:
    match = re.search(r"(\d+)", str(label))
    return int(match.group(1)) if match else -1


def _display_odor(dataset_canon: str, trial_label: str) -> str:
    number = _trial_num(trial_label)
    if number in (1, 3):
        return HEXANOL_LABEL
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
    }
    return mapping.get(dataset_canon, {}).get(number, trial_label)


def _load_envelope_matrix(matrix_path: Path, codes_json: Path) -> tuple[pd.DataFrame, list[str]]:
    matrix = np.load(matrix_path, allow_pickle=False)
    with codes_json.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)

    ordered_cols: list[str] = meta["column_order"]
    code_maps: Mapping[str, Mapping[str, int]] = meta["code_maps"]
    df = pd.DataFrame(matrix, columns=ordered_cols)

    decode_cols = [c for c in ("dataset", "fly", "trial_type", "trial_label", "fps") if c in ordered_cols]
    for col in decode_cols:
        if col == "fps":
            # fps may have been stored numerically; ensure float64 for precision
            df[col] = pd.to_numeric(df[col], errors="coerce")
            continue
        mapping = code_maps.get(col)
        if not mapping:
            continue
        inverse = {code: label for label, code in mapping.items()}
        df[col] = df[col].astype(int).map(inverse).fillna("UNKNOWN")

    if "fps" not in df.columns:
        df["fps"] = np.nan

    env_cols = [c for c in ordered_cols if c not in {"dataset", "fly", "trial_type", "trial_label", "fps"}]
    return df, env_cols


def _extract_env(row: pd.Series, env_cols: Sequence[str]) -> np.ndarray:
    env = row[env_cols].to_numpy(dtype=float)
    if env.ndim == 0:
        return np.empty(0, dtype=float)
    mask = np.isfinite(env) & (env > 0)
    return env[mask]


def _latency_to_cross(
    env: np.ndarray,
    fps: float,
    before_sec: float,
    during_sec: float,
    thresh_mult: float,
) -> Optional[float]:
    if env.size == 0 or not np.isfinite(fps) or fps <= 0:
        return None

    b_end = min(int(round(before_sec * fps)), env.size)
    d_end = min(b_end + int(round(during_sec * fps)), env.size)
    before = env[:b_end]
    during = env[b_end:d_end]
    if before.size == 0 or during.size == 0:
        return None

    mu = float(np.nanmean(before))
    sd = float(np.nanstd(before))
    theta = mu + thresh_mult * sd
    idx = np.where(during > theta)[0]
    if idx.size == 0:
        return None
    return float(idx[0]) / fps


def latency_reports(
    matrix_path: Path,
    codes_json: Path,
    out_dir: Path,
    *,
    before_sec: float,
    during_sec: float,
    thresh_mult: float,
    latency_ceiling: float,
    trials_of_interest: Sequence[int],
    fps_default: float,
) -> None:
    df, env_cols = _load_envelope_matrix(matrix_path, codes_json)
    df = df[df["trial_type"].str.lower() == "training"].copy()
    if df.empty:
        print("[WARN] No training trials found in matrix; nothing to plot.")
        return

    df["fps"] = df["fps"].replace([np.inf, -np.inf], np.nan).fillna(fps_default)
    df["dataset_canon"] = df["dataset"].apply(_canon_dataset)

    records = []
    for _, row in df.iterrows():
        trial_number = _trial_num(row["trial_label"])
        if trial_number not in trials_of_interest:
            continue
        env = _extract_env(row, env_cols)
        fps = float(row["fps"]) if np.isfinite(row["fps"]) else fps_default
        latency = _latency_to_cross(env, fps, before_sec, during_sec, thresh_mult)
        latency_for_mean = latency if latency is not None and latency <= latency_ceiling else math.nan
        records.append(
            {
                "dataset": row["dataset"],
                "dataset_canon": row["dataset_canon"],
                "fly": row["fly"],
                "trial_num": trial_number,
                "latency": latency,
                "lat_for_mean": latency_for_mean,
            }
        )

    lat_df = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)

    if lat_df.empty:
        print("[WARN] No training trials matched the requested indices; nothing to plot.")
        return

    _plot_latency_per_fly(df, lat_df, out_dir, trials_of_interest, latency_ceiling)
    _plot_latency_per_odor(lat_df, out_dir, trials_of_interest, latency_ceiling)
    _plot_latency_grand_means(lat_df, out_dir, latency_ceiling)


def _plot_latency_per_fly(
    df: pd.DataFrame,
    lat_df: pd.DataFrame,
    out_dir: Path,
    trials_of_interest: Sequence[int],
    latency_ceiling: float,
) -> None:
    for fly in sorted(df["fly"].unique()):
        sub_rows = df[df["fly"] == fly]
        if sub_rows.empty:
            continue
        odor = _safe_dirname(_canon_dataset(sub_rows["dataset"].iloc[0]))
        odir = out_dir / odor
        odir.mkdir(parents=True, exist_ok=True)

        latencies = [
            lat_df[(lat_df["fly"] == fly) & (lat_df["trial_num"] == trial)].get("latency").to_numpy(dtype=float)
            for trial in trials_of_interest
        ]

        flat = [vals[0] if vals.size else math.nan for vals in latencies]
        labels = [f"Training {trial}" for trial in trials_of_interest]

        any_response = any(
            val is not None and np.isfinite(val) and val <= latency_ceiling for val in flat if not math.isnan(val)
        )

        if not any_response:
            fig, ax = plt.subplots(figsize=(6.5, 3.2))
            ax.set_title(f"{fly} — Time to PER", pad=10, fontsize=14, weight="bold")
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_ylim(0, latency_ceiling + 2.0)
            ax.text(0.5, 0.55, "NR", transform=ax.transAxes, ha="center", va="center", fontsize=18, color="#666666", weight="bold")
            ax.set_ylabel("Time After Odor Sent (s)")
            ax.axhline(latency_ceiling, linestyle="--", linewidth=1.1, color="#444444")
            trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(
                0.995,
                latency_ceiling + 0.12,
                f"NR if > {latency_ceiling:.1f} s",
                transform=trans,
                ha="right",
                va="bottom",
                fontsize=10,
                color="#444444",
                clip_on=False,
            )
            fig.tight_layout()
            out_png = odir / f"{fly}_training_{'_'.join(map(str, trials_of_interest))}_latency.png"
            fig.savefig(out_png, dpi=300)
            plt.close(fig)
            print(f"[OK] saved {out_png} (NR panel)")
            continue

        bars, annotations, colours = [], [], []
        for value in flat:
            if value is None or math.isnan(value) or value > latency_ceiling:
                bars.append(latency_ceiling)
                annotations.append("NR")
                colours.append("#BDBDBD")
            else:
                bars.append(value)
                annotations.append(f"{value:.2f}s")
                colours.append("#1A1A1A")

        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        x = np.arange(len(labels))
        rects = ax.bar(x, bars, width=0.6, color=colours, edgecolor="black", linewidth=1.0)

        for rect, label in zip(rects, annotations):
            y_text = max(rect.get_height() * 0.5, 0.35)
            colour = "white" if label != "NR" else "#444444"
            ax.text(rect.get_x() + rect.get_width() / 2, y_text, label, ha="center", va="center", fontsize=10, color=colour)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Time After Odor Sent (s)")
        ax.set_ylim(0, latency_ceiling + 2.5)
        ax.axhline(latency_ceiling, linestyle="--", linewidth=1.1, color="#444444")
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(
            0.995,
            latency_ceiling + 0.12,
            f"NR if > {latency_ceiling:.1f} s",
            transform=trans,
            ha="right",
            va="bottom",
            fontsize=10,
            color="#444444",
            clip_on=False,
        )
        ax.set_title(f"{fly} — Time to PER", pad=10, fontsize=14, weight="bold")

        fig.tight_layout()
        out_png = odir / f"{fly}_training_{'_'.join(map(str, trials_of_interest))}_latency.png"
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"[OK] saved {out_png}")


def _plot_latency_per_odor(
    lat_df: pd.DataFrame,
    out_dir: Path,
    trials_of_interest: Sequence[int],
    latency_ceiling: float,
) -> None:
    odors = sorted(lat_df["dataset_canon"].dropna().unique())
    labels = [f"Training {trial}" for trial in trials_of_interest]

    for odor in odors:
        sub = lat_df[lat_df["dataset_canon"] == odor]
        odir = out_dir / _safe_dirname(odor)
        odir.mkdir(parents=True, exist_ok=True)

        if sub.empty:
            fig, ax = plt.subplots(figsize=(6.8, 3.2))
            ax.set_title(f"{odor} — Mean Time to PER", pad=10, fontsize=14, weight="bold")
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_ylim(0, latency_ceiling + 2.0)
            ax.text(0.5, 0.55, "NR", transform=ax.transAxes, ha="center", va="center", fontsize=18, color="#666666", weight="bold")
            ax.set_ylabel("Time After Odor Sent (s)")
            out_png = odir / f"{odor}_training_{'_'.join(map(str, trials_of_interest))}_mean_latency.png"
            fig.tight_layout()
            fig.savefig(out_png, dpi=300)
            plt.close(fig)
            print(f"[OK] saved {out_png} (NR panel)")
            continue

        means, sems, ns = [], [], []
        for trial in trials_of_interest:
            values = sub[sub["trial_num"] == trial]["lat_for_mean"].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            n_resp = finite.size
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
            ns.append(n_resp)

        if sum(ns) == 0:
            fig, ax = plt.subplots(figsize=(6.8, 3.2))
            ax.set_title(f"{odor} — Mean Time to PER", pad=10, fontsize=14, weight="bold")
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_ylim(0, latency_ceiling + 2.0)
            ax.text(0.5, 0.55, "NR", transform=ax.transAxes, ha="center", va="center", fontsize=18, color="#666666", weight="bold")
            ax.set_ylabel("Time After Odor Sent (s)")
            out_png = odir / f"{odor}_training_{'_'.join(map(str, trials_of_interest))}_mean_latency.png"
            fig.tight_layout()
            fig.savefig(out_png, dpi=300)
            plt.close(fig)
            print(f"[OK] saved {out_png} (NR panel)")
            continue

        y = np.nan_to_num(np.array(means, dtype=float), nan=0.0)
        yerr_up = np.nan_to_num(np.array(sems, dtype=float), nan=0.0)
        yerr = np.vstack([np.zeros_like(yerr_up), yerr_up])

        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        x = np.arange(len(labels))
        bars = ax.bar(x, y, width=0.6, color="#1A1A1A", edgecolor="black", linewidth=1.0)
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.2, capsize=4)

        for idx, rect in enumerate(bars):
            mu_val = y[idx]
            sem_val = yerr_up[idx]
            n_resp = ns[idx]
            if n_resp == 0 or not np.isfinite(mu_val):
                y_text = max(0.5, mu_val + 0.08)
                ax.text(rect.get_x() + rect.get_width() / 2, y_text, "NR", ha="center", va="bottom", fontsize=9, color="#444444")
                continue
            inner_y = max(mu_val * 0.5, min(mu_val - 0.10, mu_val * 0.9))
            ax.text(rect.get_x() + rect.get_width() / 2, inner_y, f"{mu_val:.2f}s", ha="center", va="top", fontsize=10, color="white")
            top_y = mu_val + (sem_val if np.isfinite(sem_val) else 0.0) + 0.06
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                top_y,
                f"SEM={sem_val:.2f}s\nn={n_resp}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#333333",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Time After Odor Sent (s)")
        ymax = max(latency_ceiling + 2.0, float((y + yerr_up).max()) + 1.2)
        ax.set_ylim(0, ymax)
        ax.axhline(latency_ceiling, linestyle="--", linewidth=1.0, color="#6f6f6f")
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(
            0.995,
            latency_ceiling + 0.08,
            f"NR if > {latency_ceiling:.1f} s",
            transform=trans,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#6f6f6f",
        )
        ax.set_title(f"{odor} — Mean Time to PER", pad=10, fontsize=14, weight="bold")

        fig.tight_layout()
        out_png = odir / f"{odor}_training_{'_'.join(map(str, trials_of_interest))}_mean_latency.png"
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"[OK] saved {out_png}")


def _plot_latency_grand_means(lat_df: pd.DataFrame, out_dir: Path, latency_ceiling: float) -> None:
    summary_rows = []
    odors = sorted(lat_df["dataset_canon"].dropna().unique())

    grand_means, grand_sems, grand_ns = [], [], []
    for odor in odors:
        values = lat_df[lat_df["dataset_canon"] == odor]["lat_for_mean"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        n_resp = finite.size
        if n_resp == 0:
            mu = math.nan
            sem = math.nan
        else:
            mu = float(finite.mean())
            if n_resp > 1:
                sd = float(finite.std(ddof=1))
                sem = sd / math.sqrt(n_resp)
            else:
                sem = 0.0
        grand_means.append(mu)
        grand_sems.append(sem)
        grand_ns.append(n_resp)
        summary_rows.append({"odor": odor, "n_resp": n_resp, "mean_s": mu, "sem_s": sem})

    pd.DataFrame(summary_rows).to_csv(out_dir / "grand_mean_by_odor_latency.csv", index=False)

    if sum(grand_ns) == 0:
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        ax.set_title("Grand Mean Time to Reaction by Trained Odor", pad=10, fontsize=14, weight="bold")
        ax.set_xticks(np.arange(len(odors)))
        ax.set_xticklabels(odors)
        ax.set_ylim(0, latency_ceiling + 2.0)
        ax.text(0.5, 0.55, "NR", transform=ax.transAxes, ha="center", va="center", fontsize=18, color="#666666", weight="bold")
        ax.set_ylabel("Time After Odor Sent (s)")
        ax.axhline(latency_ceiling, linestyle="--", linewidth=1.0, color="#6f6f6f")
        out_png = out_dir / "grand_mean_by_odor_latency.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"[OK] saved {out_png} (NR panel)")
        return

    y = np.nan_to_num(np.array(grand_means, dtype=float), nan=0.0)
    yerr_up = np.nan_to_num(np.array(grand_sems, dtype=float), nan=0.0)
    yerr = np.vstack([np.zeros_like(yerr_up), yerr_up])

    fig, ax = plt.subplots(figsize=(max(7.2, 1.8 * len(odors)), 3.8))
    x = np.arange(len(odors))
    bars = ax.bar(x, y, width=0.6, color="#1A1A1A", edgecolor="black", linewidth=1.0)
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.2, capsize=4)

    for idx, rect in enumerate(bars):
        n_resp = grand_ns[idx]
        sem_val = yerr_up[idx]
        if n_resp == 0:
            label_y = max(0.5, y[idx] + 0.08)
            ax.text(rect.get_x() + rect.get_width() / 2, label_y, "NR", ha="center", va="bottom", fontsize=9, color="#444444")
            continue
        label_y = y[idx] + (sem_val if np.isfinite(sem_val) else 0.0) + 0.06
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            label_y,
            f"SEM={sem_val:.2f} s\nn={n_resp}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(odors)
    ax.set_ylabel("Time After Odor Sent (s)")
    ymax = max(latency_ceiling + 2.0, float((y + yerr_up).max()) + 1.2)
    ax.set_ylim(0, ymax)
    ax.axhline(latency_ceiling, linestyle="--", linewidth=1.0, color="#6f6f6f")
    ax.set_title("Grand Mean Time to Reaction by Trained Odor", pad=10, fontsize=14, weight="bold")

    out_png = out_dir / "grand_mean_by_odor_latency.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[OK] saved {out_png}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training envelope utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    plots = sub.add_parser("plots", help="Render training envelope subplots per fly")
    plots.add_argument("--root", type=Path, default=None, help="Root data directory (defaults to config main_directory)")
    plots.add_argument("--fps-default", type=float, default=40.0, help="Fallback FPS when missing in CSV")
    plots.add_argument("--window-sec", type=float, default=0.25, help="Rolling window for smoothing (seconds)")
    plots.add_argument("--odor-on", type=float, default=30.0, help="Valve ON timestamp (seconds)")
    plots.add_argument("--odor-off", type=float, default=60.0, help="Valve OFF timestamp (seconds)")
    plots.add_argument(
        "--measure-cols",
        nargs="+",
        default=["distance_percentage_2_6", "distance_percentage"],
        help="Ordered list of measurement columns to prefer",
    )

    collect = sub.add_parser("collect", help="Gather generated training plots into one folder")
    collect.add_argument("--root", type=Path, default=None, help="Root data directory (defaults to config main_directory)")
    collect.add_argument(
        "--dest-folder",
        default="all_training_envelope_plots",
        help="Destination folder name under the root",
    )

    latency = sub.add_parser("latency", help="Compute latency summaries from the float16 envelope matrix")
    latency.add_argument("--matrix-npy", type=Path, required=True, help="Path to envelope_matrix_float16.npy")
    latency.add_argument("--codes-json", type=Path, required=True, help="Path to code_maps.json")
    latency.add_argument("--out-dir", type=Path, required=True, help="Output directory for figures and CSVs")
    latency.add_argument("--before-sec", type=float, default=30.0, help="Baseline window length in seconds")
    latency.add_argument("--during-sec", type=float, default=35.0, help="During window length in seconds")
    latency.add_argument("--threshold-mult", type=float, default=4.0, help="Threshold multiplier (mu + k*std)")
    latency.add_argument("--latency-ceiling", type=float, default=9.5, help="Cap for marking NR trials")
    latency.add_argument(
        "--trials",
        nargs="+",
        type=int,
        default=[4, 5, 6],
        help="Training trial numbers to analyse (e.g. 4 5 6)",
    )
    latency.add_argument("--fps-default", type=float, default=40.0, help="Fallback FPS when metadata missing")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "plots":
        root = args.root if args.root is not None else _default_root()
        cfg = TrainingConfig(
            root=root.expanduser().resolve(),
            fps_default=args.fps_default,
            window_sec=args.window_sec,
            odor_on_s=args.odor_on,
            odor_off_s=args.odor_off,
            measure_cols=tuple(args.measure_cols),
        )
        run_training_envelopes(cfg)
        return

    if args.command == "collect":
        root = args.root if args.root is not None else _default_root()
        collect_training_plots(root, args.dest_folder)
        return

    if args.command == "latency":
        latency_reports(
            args.matrix_npy.expanduser().resolve(),
            args.codes_json.expanduser().resolve(),
            args.out_dir.expanduser().resolve(),
            before_sec=args.before_sec,
            during_sec=args.during_sec,
            thresh_mult=args.threshold_mult,
            latency_ceiling=args.latency_ceiling,
            trials_of_interest=tuple(args.trials),
            fps_default=args.fps_default,
        )
        return

    parser.error(f"Unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
