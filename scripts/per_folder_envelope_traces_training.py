"""Standalone script to generate per-folder envelope trace figures for TRAINING.

Same layout as per_folder_envelope_traces.py but for training trials, matching
the CombinedBase-Training-PER-Envelopes pipeline style.

Training-specific features:
  - Extended odor off (65s) for trials 4, 6, 8
  - Discriminate odor (darker shading) for trials 5, 7
  - Light pulsing start annotation (green dash-dot line)
  - Per-dataset odor schedules

Usage:
    python scripts/per_folder_envelope_traces_training.py
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
CSV_PATH = Path(
    "/home/ramanlab/Documents/cole/Data/Opto/Combined/"
    "all_envelope_rows_wide_combined_base_training.csv"
)
FLAGGED_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/Opto/Combined/"
    "flagged-flys-truth.csv"
)
OUT_DIR = Path(
    "/home/ramanlab/Documents/cole/Data/Opto/Combined/per_folder_traces_training"
)

# ── fly colours ────────────────────────────────────────────────────────────
FLY_COLORS = {1: "red", 2: "blue", 3: "green"}

# ── odor timing (seconds) — matches pipeline defaults ─────────────────────
ODOR_ON_S = 30.0
ODOR_OFF_S = 60.0
ODOR_LATENCY_S = 0.0
LATENCY_SEC = 0.0
AFTER_SHOW_SEC = 30.0
THRESHOLD_STD_MULT = 3.0

# ── training-specific timing ──────────────────────────────────────────────
TRAINING_EXTENDED_ODOR_TRIALS = frozenset({4, 6, 8})
TRAINING_EXTENDED_ODOR_OFF_S = 65.0
TRAINING_DISCRIMINATE_ODOR_TRIALS = frozenset({5, 7})
LIGHT_START_EARLY_TRIALS = frozenset({1, 2, 3})
LIGHT_START_LATE_TRIALS = frozenset({4, 6, 8})
LIGHT_START_EARLY_S = 35.0
LIGHT_START_LATE_S = 40.0

# ── shading constants (from envelope_visuals.py) ──────────────────────────
ODOR_PLUS_LIGHT_COLOR = "#9e9e9e"
ODOR_PLUS_LIGHT_ALPHA = 0.20
ODOR_PLUS_LIGHT_LINGER_ALPHA = 0.12
ODOR_PLUS_LIGHT_LABEL = "Odor + light"
DISCRIMINATE_ODOR_COLOR = "#4d4d4d"
DISCRIMINATE_ODOR_ALPHA = 0.28
DISCRIMINATE_ODOR_LINGER_ALPHA = 0.18
DISCRIMINATE_ODOR_LABEL = "Discriminate odor"
FIXED_Y_MAX = 100.0

HEXANOL_LABEL = "Hexanol"

# ── display labels ────────────────────────────────────────────────────────
DISPLAY_LABEL = {
    "ACV": "Apple Cider Vinegar",
    "3-octonol": "3-Octonol",
    "Benz": "Benzaldehyde",
    "EB": "Ethyl Butyrate",
    "EB_control": "Ethyl Butyrate",
    "hex_control": "Hexanol",
    "Benz_control": "Benzaldehyde",
    "opto_benz_1": "Benzaldehyde",
    "opto_EB": "Ethyl Butyrate",
    "opto_EB_6_training": "Ethyl Butyrate (6-Training)",
    "opto_ACV": "Apple Cider Vinegar",
    "opto_hex": "Hexanol",
    "opto_AIR": "AIR",
    "opto_3-oct": "3-Octonol",
}

# Map new dataset folder names to legacy canonical keys.
DATASET_ALIAS = {
    "benz-training": "opto_benz_1",
    "benz-training-24": "opto_benz_1",
    "benz-control": "Benz_control",
    "hex-control": "hex_control",
    "hex-training": "opto_hex",
    "hex-training-24": "opto_hex",
    "eb-control": "EB_control",
    "eb-training": "opto_EB",
    "eb-training(no-operant)": "opto_EB_6_training",
    "acv-training": "opto_ACV",
    "air-training": "opto_AIR",
    "3oct-training": "opto_3-oct",
}

# ── training odor schedules per dataset ───────────────────────────────────
TRAINING_ODOR_SCHEDULE = {
    1: "Benzaldehyde", 2: "Benzaldehyde", 3: "Benzaldehyde",
    4: "Benzaldehyde", 5: HEXANOL_LABEL, 6: "Benzaldehyde",
    7: HEXANOL_LABEL, 8: "Benzaldehyde",
}
TRAINING_ODOR_SCHEDULE_HEX = {
    1: HEXANOL_LABEL, 2: HEXANOL_LABEL, 3: HEXANOL_LABEL,
    4: HEXANOL_LABEL, 5: "Apple Cider Vinegar", 6: HEXANOL_LABEL,
    7: "Apple Cider Vinegar", 8: HEXANOL_LABEL,
}
TRAINING_ODOR_SCHEDULE_EB = {
    1: "Ethyl Butyrate", 2: "Ethyl Butyrate", 3: "Ethyl Butyrate",
    4: "Ethyl Butyrate", 5: HEXANOL_LABEL, 6: "Ethyl Butyrate",
    7: HEXANOL_LABEL, 8: "Ethyl Butyrate",
}
TRAINING_ODOR_SCHEDULE_EB_6TRAINING = {
    1: "Ethyl Butyrate", 2: "Ethyl Butyrate", 3: "Ethyl Butyrate",
    4: "Ethyl Butyrate", 5: "Ethyl Butyrate", 6: "Ethyl Butyrate",
}
TRAINING_ODOR_SCHEDULE_ACV = {
    1: "Apple Cider Vinegar", 2: "Apple Cider Vinegar",
    3: "Apple Cider Vinegar", 4: "Apple Cider Vinegar",
    5: HEXANOL_LABEL, 6: "Apple Cider Vinegar",
    7: HEXANOL_LABEL, 8: "Apple Cider Vinegar",
}
TRAINING_ODOR_SCHEDULE_AIR = {
    1: "AIR", 2: "AIR", 3: "AIR", 4: "AIR",
    5: HEXANOL_LABEL, 6: "AIR", 7: HEXANOL_LABEL, 8: "AIR",
}
TRAINING_ODOR_SCHEDULE_3OCT = {
    1: "3-Octonol", 2: "3-Octonol", 3: "3-Octonol", 4: "3-Octonol",
    5: HEXANOL_LABEL, 6: "3-Octonol", 7: HEXANOL_LABEL, 8: "3-Octonol",
}


# ---------------------------------------------------------------------------
# Odor name resolution for training trials
# ---------------------------------------------------------------------------

def _strip_flagged(dataset: str) -> str:
    ds = str(dataset).strip()
    lower = ds.lower()
    if lower.endswith("-flagged"):
        return ds[: -len("-flagged")]
    if lower.endswith("_flagged"):
        return ds[: -len("_flagged")]
    return ds


def _trial_num(trial_label: str) -> int:
    m = re.search(r"(\d+)", str(trial_label))
    return int(m.group(1)) if m else 0


def _resolve_training_odor(dataset: str, trial_label: str) -> str:
    base = _strip_flagged(dataset).strip()
    ds = DATASET_ALIAS.get(base.lower(), base)
    number = _trial_num(trial_label)

    if ds == "opto_AIR":
        return TRAINING_ODOR_SCHEDULE_AIR.get(number, ds)
    if ds == "opto_3-oct":
        return TRAINING_ODOR_SCHEDULE_3OCT.get(number, ds)
    if ds in ("opto_EB", "EB_control"):
        return TRAINING_ODOR_SCHEDULE_EB.get(number, ds)
    if ds in ("opto_EB(6-training)", "opto_EB_6_training"):
        return TRAINING_ODOR_SCHEDULE_EB_6TRAINING.get(number, ds)
    if ds in ("opto_hex", "hex_control"):
        return TRAINING_ODOR_SCHEDULE_HEX.get(number, ds)
    if ds in ("opto_ACV", "ACV"):
        return TRAINING_ODOR_SCHEDULE_ACV.get(number, ds)
    return TRAINING_ODOR_SCHEDULE.get(number,
           DISPLAY_LABEL.get(ds, ds))


def _is_trained_odor(dataset: str, odor_name: str) -> bool:
    base = _strip_flagged(dataset).strip()
    ds = DATASET_ALIAS.get(base.lower(), base)
    trained = DISPLAY_LABEL.get(ds, ds)
    return str(odor_name).strip().lower() == str(trained).strip().lower()


def _trial_odor_off(trial_label: str) -> float:
    number = _trial_num(trial_label)
    if number in TRAINING_EXTENDED_ODOR_TRIALS:
        return TRAINING_EXTENDED_ODOR_OFF_S
    return ODOR_OFF_S


def _is_discriminate(trial_label: str) -> bool:
    return _trial_num(trial_label) in TRAINING_DISCRIMINATE_ODOR_TRIALS


def _light_start(trial_label: str) -> float | None:
    number = _trial_num(trial_label)
    if number in LIGHT_START_EARLY_TRIALS:
        return LIGHT_START_EARLY_S
    if number in LIGHT_START_LATE_TRIALS:
        return LIGHT_START_LATE_S
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def natural_sort_key(s: str):
    return [int(p) if p.isdigit() else p.lower()
            for p in re.split(r"(\d+)", s)]


_MONTH_NUM = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _date_sort_key(folder: str) -> tuple[int, int, str]:
    m = re.match(r"([a-z]+)_(\d+)", folder.lower())
    if m:
        month = _MONTH_NUM.get(m.group(1), 99)
        day = int(m.group(2))
        return (month, day, folder.lower())
    return (99, 99, folder.lower())


def get_dir_val_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("dir_val_")]
    cols.sort(key=lambda c: int(c.split("_")[-1]))
    return cols


def _extract_env(row_vals: np.ndarray) -> np.ndarray:
    env = row_vals.astype(float, copy=False)
    mask = np.isfinite(env) & (env > 0)
    if not mask.any():
        return np.empty(0, dtype=float)
    return env[mask]


def _compute_theta(env: np.ndarray, fps: float) -> float:
    n_before = int(round(ODOR_ON_S * fps))
    before = env[:n_before]
    before = before[np.isfinite(before)]
    if before.size < 3:
        return float("nan")
    med = float(np.median(before))
    mad = float(np.median(np.abs(before - med)))
    sigma_est = mad * 1.4826
    return med + THRESHOLD_STD_MULT * sigma_est


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_folder(
    folder_df: pd.DataFrame,
    dataset: str,
    folder: str,
    dir_val_cols: list[str],
    out_dir: Path,
):
    trial_labels = sorted(
        folder_df["trial_label"].unique(), key=natural_sort_key
    )
    n_trials = len(trial_labels)
    if n_trials == 0:
        return

    fps = float(folder_df["fps"].iloc[0])
    if not math.isfinite(fps) or fps <= 0:
        fps = 40.0

    fly_numbers = sorted(folder_df["fly_number"].unique())

    # Use the longest possible x range (extended odor off + after)
    max_off = max(ODOR_OFF_S, TRAINING_EXTENDED_ODOR_OFF_S)
    odor_latency = max(ODOR_LATENCY_S, 0.0)
    linger = max(LATENCY_SEC, 0.0)
    x_max_limit = max_off + odor_latency + linger + AFTER_SHOW_SEC

    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out", "ytick.direction": "out",
        "font.family": "Arial", "font.sans-serif": ["Arial"],
        "font.size": 10,
    })

    fig_h = max(3.0, n_trials * 1.6 + 1.5)
    fig, axes = plt.subplots(n_trials, 1, figsize=(10, fig_h), sharex=True)
    if n_trials == 1:
        axes = [axes]

    fig.text(0.02, 0.5, "Proboscis Extension (combined base)",
             va="center", rotation="vertical", fontsize=10)

    for ax_idx, trial_label in enumerate(trial_labels):
        ax = axes[ax_idx]
        trial_data = folder_df[folder_df["trial_label"] == trial_label]
        odor_name = _resolve_training_odor(dataset, trial_label)
        is_trained = _is_trained_odor(dataset, odor_name)
        is_disc = _is_discriminate(trial_label)
        trial_off = _trial_odor_off(trial_label)
        light_s = _light_start(trial_label)

        odor_on_eff = ODOR_ON_S + odor_latency
        odor_off_eff = trial_off + odor_latency

        for fly_num in fly_numbers:
            fly_data = trial_data[trial_data["fly_number"] == fly_num]
            if fly_data.empty:
                continue

            raw = fly_data[dir_val_cols].values.flatten()
            env = _extract_env(raw)
            if env.size == 0:
                continue

            t = np.arange(env.size, dtype=float) / fps
            mask = t <= x_max_limit + 1e-9
            env = env[mask]
            t = t[mask]
            if env.size == 0:
                continue

            colour = FLY_COLORS.get(fly_num, "gray")
            ax.plot(t, env, linewidth=1.2, color=colour, alpha=0.8,
                    label=f"Fly {fly_num}")

        # Odor on/off dashed lines
        ax.axvline(odor_on_eff, linestyle="--", linewidth=1.0, color="black")
        ax.axvline(odor_off_eff, linestyle="--", linewidth=1.0, color="black")

        # Odor shading
        transit_on_end = min(odor_on_eff, x_max_limit)
        linger_off_end = min(odor_off_eff + linger, x_max_limit)
        odor_bar_color = DISCRIMINATE_ODOR_COLOR if is_disc else ODOR_PLUS_LIGHT_COLOR
        odor_bar_alpha = DISCRIMINATE_ODOR_ALPHA if is_disc else ODOR_PLUS_LIGHT_ALPHA
        if linger_off_end > transit_on_end:
            ax.axvspan(transit_on_end, linger_off_end,
                       alpha=odor_bar_alpha, color=odor_bar_color)

        # Light start annotation
        if light_s is not None and light_s <= x_max_limit:
            ax.axvline(light_s, linestyle="-.", linewidth=1.3,
                       color="tab:green")

        # Threshold line
        first_fly = trial_data[trial_data["fly_number"] == fly_numbers[0]]
        if not first_fly.empty:
            ref_env = _extract_env(first_fly[dir_val_cols].values.flatten())
            if ref_env.size > 0:
                theta = _compute_theta(ref_env, fps)
                if math.isfinite(theta):
                    ax.axhline(theta, linestyle="-", linewidth=1.0,
                               color="tab:red", alpha=0.9)

        ax.set_ylim(0, FIXED_Y_MAX)
        ax.set_xlim(0, x_max_limit)
        ax.margins(x=0, y=0.02)

        if is_trained:
            ax.set_title(odor_name.upper(), loc="left", fontsize=11,
                         weight="bold", pad=2, color="tab:blue")
        else:
            ax.set_title(odor_name, loc="left", fontsize=11,
                         weight="bold", pad=2, color="black")

    axes[-1].set_xlabel("Time (s)", fontsize=11)

    # ── legend ──
    fly_handles = [
        plt.Line2D([0], [0], linewidth=1.2, color=FLY_COLORS.get(fn, "gray"),
                    label=f"Fly {fn}")
        for fn in fly_numbers
    ]
    extra_handles = [
        plt.Line2D([0], [0], linestyle="--", linewidth=1.0, color="black",
                    label="Odor at fly"),
        plt.Rectangle((0, 0), 1, 1, alpha=ODOR_PLUS_LIGHT_ALPHA,
                       color=ODOR_PLUS_LIGHT_COLOR, label=ODOR_PLUS_LIGHT_LABEL),
        plt.Rectangle((0, 0), 1, 1, alpha=DISCRIMINATE_ODOR_ALPHA,
                       color=DISCRIMINATE_ODOR_COLOR, label=DISCRIMINATE_ODOR_LABEL),
        plt.Line2D([0], [0], linestyle="-.", linewidth=1.3, color="tab:green",
                    label="Light pulsing starts"),
        plt.Line2D([0], [0], linestyle="-", linewidth=1.0, color="tab:red",
                    label=r"$\theta = \mathrm{median}_{before} + k\cdot\mathrm{MAD}_{before}$"),
    ]
    fig.legend(
        handles=fly_handles + extra_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.97),
        frameon=True,
        fontsize=9,
        title=f"Threshold: k = {THRESHOLD_STD_MULT:g}",
        title_fontsize=9,
    )

    # ── titles ──
    title_y = 0.995
    subtitle_y = title_y - 0.035
    fig.suptitle(
        "Proboscis Distance Across Training Trials",
        y=title_y, fontsize=14, weight="bold", color="black",
    )
    fig.text(
        0.5, subtitle_y,
        f"{dataset} — {folder}",
        ha="center", va="center", fontsize=12, weight="bold", color="black",
    )

    fig.tight_layout(rect=[0.04, 0, 1, 0.93])

    safe_name = f"{folder}.png"
    fig.savefig(out_dir / safe_name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {safe_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_exclusion_set(flagged_path: Path) -> set[tuple[str, str, int]]:
    flagged = pd.read_csv(flagged_path)
    score_col = [c for c in flagged.columns if "State" in c or "state" in c][0]
    flagged[score_col] = pd.to_numeric(flagged[score_col], errors="coerce")
    bad = flagged[flagged[score_col] <= 0]
    exclude = set()
    for _, row in bad.iterrows():
        exclude.add((str(row["dataset"]).strip(),
                      str(row["fly"]).strip(),
                      int(row["fly_number"])))
    return exclude


def main():
    print(f"Reading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)

    # Keep only training rows
    df = df[df["trial_type"] == "training"].copy()

    # Load exclusion set from flagged-flys-truth.csv (score <= 0)
    exclude = _load_exclusion_set(FLAGGED_CSV)
    print(f"Excluding {len(exclude)} flagged flies (score <= 0)")

    before_len = len(df)
    df = df[
        ~df.apply(
            lambda r: (str(r["dataset"]).strip(),
                       str(r["fly"]).strip(),
                       int(r["fly_number"])) in exclude,
            axis=1,
        )
    ]
    print(f"  Removed {before_len - len(df)} rows "
          f"({before_len} → {len(df)})")

    dir_val_cols = get_dir_val_cols(df)
    print(f"Found {len(dir_val_cols)} dir_val columns")

    datasets = sorted(df["dataset"].unique(), key=natural_sort_key)
    print(f"Datasets: {datasets}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        ds_df = df[df["dataset"] == dataset]
        folders = sorted(ds_df["fly"].unique(), key=_date_sort_key)
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}  ({len(folders)} folders)")
        print(f"{'='*60}")

        for idx, folder in enumerate(folders, 1):
            folder_df = ds_df[ds_df["fly"] == folder]
            print(f"  [{idx:02d}] {folder}  "
                  f"({folder_df['fly_number'].nunique()} flies, "
                  f"{folder_df['trial_label'].nunique()} trials)")
            plot_folder(folder_df, dataset, folder, dir_val_cols, OUT_DIR)

    print(f"\nAll figures saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
