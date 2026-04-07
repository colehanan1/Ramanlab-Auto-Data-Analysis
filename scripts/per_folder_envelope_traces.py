"""Standalone script to generate per-folder envelope trace figures.

For each dataset, groups flies by folder (the `fly` column). Produces one
figure per folder where subplots are stacked vertically (one per testing
trial) — matching the CombinedBase-PER-Envelopes pipeline style — but with
one coloured trace per fly instead of a single black trace.

Fly colours: fly 1 = red, fly 2 = blue, fly 3 = green.

Flagged datasets use the same odor-naming convention as their non-flagged
counterparts:
  Hex-Control-flagged       → Hex-Control
  Hex-Training-flagged      → Hex-Training
  Benz-Training-flagged     → Benz-Training
  EB-Training-flagged       → EB-Training

Usage:
    python scripts/per_folder_envelope_traces.py
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[1]
for _p in (str(_REPO / "src"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fbpipe.odor_constants import (
    DISPLAY_LABEL,
    DATASET_ALIAS,
    TESTING_DATASET_ALIAS,
)
from fbpipe.plot_style import apply_lab_style

apply_lab_style()

# ── paths ──────────────────────────────────────────────────────────────────
CSV_PATH = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-ALL-Opto-Flys/"
    "all_envelope_rows_wide_combined_base.csv"
)
FLAGGED_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-ALL-Opto-Flys/"
    "flagged-flys-truth.csv"
)
OUT_DIR = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-ALL-Opto-Flys/per_folder_traces"
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

# ── shading constants (from envelope_visuals.py) ──────────────────────────
ODOR_PLUS_LIGHT_COLOR = "#9e9e9e"
ODOR_PLUS_LIGHT_ALPHA = 0.20
ODOR_PLUS_LIGHT_LINGER_ALPHA = 0.12
ODOR_PLUS_LIGHT_LABEL = "Odor + light"
FIXED_Y_MAX = 100.0

HEXANOL_LABEL = "Hexanol"

# ── display labels / dataset aliases ──────────────────────────────────────
# DISPLAY_LABEL, DATASET_ALIAS, and TESTING_DATASET_ALIAS are imported
# from fbpipe.odor_constants (see top-of-file imports).

PRIMARY_ODOR_LABEL = {
    "EB-Control": "Ethyl Butyrate",
    "Hex-Control": HEXANOL_LABEL,
    "Hex-Control-24-02": HEXANOL_LABEL,
    "Benz-Control": "Benzaldehyde",
    "Benz-Control-24-02": "Benzaldehyde",
    "Hex-Control-24-2": HEXANOL_LABEL,
    "Hex-Control-24-002": HEXANOL_LABEL,
    "Benz-Control-24-2": "Benzaldehyde",
    "3OCT-Control-24-2": "3-Octonol",
}


# ---------------------------------------------------------------------------
# Odor name resolution (mirrors envelope_visuals._display_odor for testing)
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


def _resolve_odor_name(dataset: str, trial_label: str) -> str:
    """Return human-readable odor name for a testing trial."""
    base = _strip_flagged(dataset).strip()
    ds = DATASET_ALIAS.get(base, DATASET_ALIAS.get(base.lower(), base))
    number = _trial_num(trial_label)

    # AIR-Training testing
    if ds == "AIR-Training":
        if number in (1, 3):
            return HEXANOL_LABEL
        if number in (2, 4, 5):
            return "AIR"
        m = {6: "Apple Cider Vinegar", 7: "Ethyl Butyrate",
             8: "Benzaldehyde", 9: "Citral", 10: "3-Octonol"}
        return m.get(number, trial_label)

    # 3OCT-Training testing
    if ds in ("3OCT-Training", "3OCT-Training-24-2", "3OCT-Control-24-2"):
        if number in (1, 3):
            return HEXANOL_LABEL
        if number in (2, 4, 5):
            return "3-Octonol"
        m = {6: "Apple Cider Vinegar", 7: "Ethyl Butyrate",
             8: "Benzaldehyde", 9: "Citral", 10: "Linalool"}
        return m.get(number, trial_label)

    ds_alias = TESTING_DATASET_ALIAS.get(ds, ds)

    if ds_alias == "Hex-Control":
        if number in (1, 3):
            return "Apple Cider Vinegar"
        if number in (2, 4, 5):
            return HEXANOL_LABEL
    else:
        if number in (1, 3):
            return HEXANOL_LABEL
        if number in (2, 4, 5):
            return DISPLAY_LABEL.get(ds_alias, DISPLAY_LABEL.get(ds, ds))

    # Novel odor mappings for trials 6-10+
    novel = {
        "Benz-Control": {6: "Apple Cider Vinegar", 7: "3-Octonol",
                         8: "Ethyl Butyrate", 9: "Citral", 10: "Linalool"},
        "Benz-Training": {6: "Apple Cider Vinegar", 7: "3-Octonol",
                          8: "Ethyl Butyrate", 9: "Citral", 10: "Linalool"},
        "EB-Control": {6: "Apple Cider Vinegar", 7: "3-Octonol",
                       8: "Benzaldehyde", 9: "Citral", 10: "Linalool"},
        "Hex-Control": {6: "Benzaldehyde", 7: "3-Octonol",
                        8: "Ethyl Butyrate", 9: "Citral", 10: "Linalool"},
        "ACV-Training": {6: "3-Octonol", 7: "Ethyl Butyrate",
                         8: "Benzaldehyde", 9: "Citral", 10: "Linalool"},
    }
    if ds_alias in novel:
        return novel[ds_alias].get(number, trial_label)
    if ds in novel:
        return novel[ds].get(number, trial_label)

    return trial_label


def _is_trained_odor(dataset: str, odor_name: str) -> bool:
    base = _strip_flagged(dataset).strip()
    ds = DATASET_ALIAS.get(base, DATASET_ALIAS.get(base.lower(), base))
    ds_alias = TESTING_DATASET_ALIAS.get(ds, ds)
    trained = PRIMARY_ODOR_LABEL.get(ds_alias,
              DISPLAY_LABEL.get(ds_alias, DISPLAY_LABEL.get(ds, ds)))
    return str(odor_name).strip().lower() == str(trained).strip().lower()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def natural_sort_key(s: str):
    return [int(p) if p.isdigit() else p.lower()
            for p in re.split(r"(\d+)", s)]


# Month name → number for chronological sorting
_MONTH_NUM = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _date_sort_key(folder: str) -> tuple[int, int, str]:
    """Parse folder name like 'march_02_batch_2_rig_2' → (3, 2, rest)."""
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


def _resolve_trace_len(trace_len: object, max_len: int) -> int | None:
    try:
        value = float(trace_len)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return max(0, min(int(round(value)), max_len))


def _extract_env(row_vals: np.ndarray, trace_len: object = None) -> np.ndarray:
    env = row_vals.astype(float, copy=False)
    resolved_len = _resolve_trace_len(trace_len, env.size)
    if resolved_len is not None:
        env = env[:resolved_len]
    elif np.isnan(env).any():
        finite_idx = np.flatnonzero(np.isfinite(env))
        if finite_idx.size == 0:
            return np.empty(0, dtype=float)
        env = env[: finite_idx[-1] + 1]
    if env.size == 0 or not np.isfinite(env).any():
        return np.empty(0, dtype=float)
    return env


def _compute_theta(env: np.ndarray, fps: float) -> float:
    """Threshold = median_before + k * MAD_before (scaled to sigma)."""
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
    """One figure per folder: vertical subplots like CombinedBase-PER-Envelopes."""
    trial_labels = sorted(
        folder_df["trial_label"].unique(), key=natural_sort_key
    )
    n_trials = len(trial_labels)
    if n_trials == 0:
        return

    fps = float(folder_df["fps"].iloc[0])
    if not math.isfinite(fps) or fps <= 0:
        fps = 40.0

    odor_on_eff = ODOR_ON_S + ODOR_LATENCY_S
    odor_off_eff = ODOR_OFF_S + ODOR_LATENCY_S
    linger = max(LATENCY_SEC, 0.0)
    x_max_limit = odor_off_eff + linger + AFTER_SHOW_SEC

    fly_numbers = sorted(folder_df["fly_number"].unique())

    # ── rcParams matching pipeline ──
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": 10,
    })

    fig_h = max(3.0, n_trials * 1.6 + 1.5)
    fig, axes = plt.subplots(n_trials, 1, figsize=(10, fig_h), sharex=True)
    if n_trials == 1:
        axes = [axes]

    # Shared y-label on the left margin
    fig.text(0.02, 0.5, "Proboscis Extension (combined base)",
             va="center", rotation="vertical", fontsize=10)

    for ax_idx, trial_label in enumerate(trial_labels):
        ax = axes[ax_idx]
        trial_data = folder_df[folder_df["trial_label"] == trial_label]
        odor_name = _resolve_odor_name(dataset, trial_label)
        is_trained = _is_trained_odor(dataset, odor_name)

        for fly_num in fly_numbers:
            fly_data = trial_data[trial_data["fly_number"] == fly_num]
            if fly_data.empty:
                continue

            raw = fly_data[dir_val_cols].values.flatten()
            trace_len = fly_data["trace_len"].iloc[0] if "trace_len" in fly_data.columns else None
            env = _extract_env(raw, trace_len=trace_len)
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
        if linger_off_end > transit_on_end:
            ax.axvspan(transit_on_end, linger_off_end,
                       alpha=ODOR_PLUS_LIGHT_ALPHA,
                       color=ODOR_PLUS_LIGHT_COLOR)

        # Threshold line (compute from first fly's before-period as reference)
        first_fly = trial_data[trial_data["fly_number"] == fly_numbers[0]]
        if not first_fly.empty:
            ref_trace_len = (
                first_fly["trace_len"].iloc[0]
                if "trace_len" in first_fly.columns
                else None
            )
            ref_env = _extract_env(
                first_fly[dir_val_cols].values.flatten(),
                trace_len=ref_trace_len,
            )
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
        "Proboscis Distance Across Testing Trials",
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
    """Load (dataset, fly, fly_number) tuples to exclude (score <= 0)."""
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

    # Keep only testing rows
    df = df[df["trial_type"] == "testing"].copy()

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
