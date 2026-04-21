"""
Average Training PER Traces from wide CSVs, split by rig.

Two groups:
  "New Flys"  — Hex-Training-24-0.005 + EB-Training-24-0.1
                from the New CSV, after April 11 only.
                Trials: 1,2,3,4,5,6
  "Old Flys"  — Hex-Training, Benz-Training, 3OCT-Training,
                ACV-Training, AIR-Training, EB-Training
                from the ALL CSV, before January only, flagged excluded.
                Trials: 1,2,3,4,6,8

Each group gets a rig_1 and rig_2 plot (if data exists).
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────
ALL_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-ALL-Opto-Flys/"
    "all_envelope_rows_wide_combined_base_training.csv"
)
NEW_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/"
    "all_envelope_rows_wide_combined_base_training.csv"
)
FLAGGED_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-ALL-Opto-Flys/flagged-flys-truth.csv"
)
OUT_DIR = Path(
    "/home/ramanlab/Documents/cole/Results/Opto-Fly-Figures/"
    "Raw-Training-PER-Traces/Combined-Averages"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Timing ────────────────────────────────────────────────────────────────
FPS = 40.0
ODOR_ON_S = 30.0
ODOR_OFF_S = 60.0
ODOR_LATENCY_S = 2.15
AFTER_SHOW_SEC = 30.0
TRAINING_EXTENDED_ODOR_TRIALS = frozenset({4, 6, 8})
TRAINING_EXTENDED_ODOR_OFF_S = 65.0
LIGHT_START_EARLY_TRIALS = frozenset({1, 2, 3})
LIGHT_START_LATE_TRIALS = frozenset({4, 6, 8})
LIGHT_START_EARLY_S = 35.0
LIGHT_START_LATE_S = 40.0

ODOR_PLUS_LIGHT_COLOR = "#9e9e9e"
ODOR_PLUS_LIGHT_ALPHA = 0.20

# ── Style ─────────────────────────────────────────────────────────────────
FIXED_Y_MAX = 100.0
TRACE_LW = 1.4
ODOR_MARKER_LW = 1.0
LIGHT_MARKER_LW = 1.3
SEM_ALPHA = 0.25
SEM_COLOR = "steelblue"
MEAN_COLOR = "black"

MONTH_NUM = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}
BEFORE_JAN_MONTHS = {"september", "october", "november", "december"}

# ── Dataset groups ────────────────────────────────────────────────────────
NEW_DATASETS = {"Hex-Training-24-0.005", "EB-Training-24-0.1"}
OLD_DATASETS = {
    "Hex-Training", "Benz-Training", "Benz-Training-flagged",
    "3OCT-Training", "ACV-Training", "AIR-Training", "EB-Training",
}
NEW_TRIALS = [1, 2, 3, 4, 5, 6]
OLD_TRIALS = [1, 2, 3, 4, 6, 8]


def _parse_month(fly: str) -> str | None:
    m = re.match(r"([a-z]+)_", str(fly))
    return m.group(1) if m else None


def _parse_month_day(fly: str) -> tuple[int, int] | None:
    m = re.match(r"([a-z]+)_(\d+)", str(fly))
    if not m:
        return None
    month = MONTH_NUM.get(m.group(1))
    if month is None:
        return None
    return (month, int(m.group(2)))


def _classify_rig(fly: str) -> str:
    return "rig_2" if "_rig_2" in str(fly) else "rig_1"


def _trial_num(label: str) -> int | None:
    """Extract trial number from labels like 'training_3' or 'training_3_hexanol'."""
    m = re.match(r"training_(\d+)", str(label))
    return int(m.group(1)) if m else None


def _trial_odor_off(trial_num: int) -> float:
    if trial_num in TRAINING_EXTENDED_ODOR_TRIALS:
        return TRAINING_EXTENDED_ODOR_OFF_S
    return ODOR_OFF_S


def _light_start(trial_num: int) -> float | None:
    if trial_num in LIGHT_START_EARLY_TRIALS:
        return LIGHT_START_EARLY_S
    if trial_num in LIGHT_START_LATE_TRIALS:
        return LIGHT_START_LATE_S
    return None


def _load_flagged() -> set[tuple[str, str, int]]:
    """Return {(dataset, fly, fly_number)} for state < 1."""
    df = pd.read_csv(FLAGGED_CSV)
    df = df[df["FLY-State(1, 0, -1)"] < 1]
    return {
        (row["dataset"], row["fly"], int(row["fly_number"]))
        for _, row in df.iterrows()
    }


def load_group(csv_path: Path, datasets: set[str],
               flagged: set[tuple[str, str, int]],
               date_filter: str | None = None,
               after_day: tuple[int, int] | None = None,
               trial_nums: list[int] | None = None,
               ) -> dict[str, dict[int, list[np.ndarray]]]:
    """Load traces grouped by rig and trial number.

    date_filter: "before_jan" to only keep months before January
    after_day: (month_num, day) exclusive lower bound
    """
    print(f"  Loading from {csv_path.name}...")
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["dataset"].isin(datasets)]
    print(f"  Rows matching datasets: {len(df)}")

    if df.empty:
        return {"rig_1": {}, "rig_2": {}}

    # Extract trial number
    df["_trial_num"] = df["trial_label"].apply(_trial_num)
    df = df.dropna(subset=["_trial_num"])
    df["_trial_num"] = df["_trial_num"].astype(int)

    # Filter trials
    if trial_nums:
        df = df[df["_trial_num"].isin(trial_nums)]

    # Date filter
    if date_filter == "before_jan":
        mask = df["fly"].apply(lambda f: _parse_month(f) in BEFORE_JAN_MONTHS)
        before = len(df)
        df = df[mask]
        print(f"  Date filter (before Jan): {before} -> {len(df)}")

    if after_day is not None:
        def _after(f):
            md = _parse_month_day(f)
            return md is not None and md > after_day
        mask = df["fly"].apply(_after)
        before = len(df)
        df = df[mask]
        print(f"  Date filter (after {after_day}): {before} -> {len(df)}")

    # Flagged filter
    if flagged:
        before = len(df)
        mask = df.apply(
            lambda r: (r["dataset"], r["fly"], int(r["fly_number"])) not in flagged,
            axis=1,
        )
        df = df[mask]
        print(f"  Flagged filter: {before} -> {len(df)}")

    # Classify rig
    df["_rig"] = df["fly"].apply(_classify_rig)

    # Extract dir_val columns
    dir_cols = sorted(
        [c for c in df.columns if c.startswith("dir_val_")],
        key=lambda c: int(c.split("_")[-1]),
    )

    trial_set = set(trial_nums) if trial_nums else set(range(1, 9))
    result: dict[str, dict[int, list[np.ndarray]]] = {
        "rig_1": {t: [] for t in trial_set},
        "rig_2": {t: [] for t in trial_set},
    }

    for _, row in df.iterrows():
        rig = row["_rig"]
        tnum = int(row["_trial_num"])
        if tnum not in trial_set:
            continue
        vals = row[dir_cols].to_numpy(dtype=np.float64)
        # Trim trailing NaN
        last_valid = np.where(np.isfinite(vals))[0]
        if last_valid.size > 0:
            vals = vals[: last_valid[-1] + 1]
        else:
            continue
        result[rig][tnum].append(vals)

    for rig in result:
        total = sum(len(result[rig][t]) for t in result[rig])
        if total > 0:
            for t in sorted(result[rig]):
                print(f"    {rig} trial {t}: {len(result[rig][t])} traces")

    return result


def _resample(traces: list[np.ndarray], fps: float,
              x_max: float) -> tuple[np.ndarray, np.ndarray]:
    n_pts = int(x_max * fps) + 1
    t_common = np.linspace(0, x_max, n_pts)
    resampled = []
    for env in traces:
        t_orig = np.arange(len(env)) / fps
        interp = np.interp(t_common, t_orig, env, left=np.nan, right=np.nan)
        resampled.append(interp)
    return t_common, np.array(resampled)


def plot_group(group_name: str, rig: str,
               trials: dict[int, list[np.ndarray]],
               trial_nums: list[int]) -> None:
    if all(len(trials.get(t, [])) == 0 for t in trial_nums):
        print(f"  [{group_name}] No data for {rig}, skipping.")
        return

    odor_on_eff = ODOR_ON_S
    x_max = ODOR_OFF_S + ODOR_LATENCY_S + AFTER_SHOW_SEC

    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out", "ytick.direction": "out",
        "font.family": "Arial", "font.sans-serif": ["Arial"],
        "font.size": 10,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    })

    n_rows = len(trial_nums)
    fig_h = max(3.0, n_rows * 1.6 + 1.5)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, fig_h), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for i, tnum in enumerate(trial_nums):
        ax = axes[i]
        trace_list = trials.get(tnum, [])
        trial_off_eff = _trial_odor_off(tnum)
        light_s = _light_start(tnum)

        if len(trace_list) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        t_common, matrix = _resample(trace_list, FPS, x_max)
        mean = np.nanmean(matrix, axis=0)
        sem = np.nanstd(matrix, axis=0) / np.sqrt(
            np.sum(~np.isnan(matrix), axis=0).clip(1))

        ax.axvspan(odor_on_eff, min(trial_off_eff, x_max),
                   alpha=ODOR_PLUS_LIGHT_ALPHA, color=ODOR_PLUS_LIGHT_COLOR)
        ax.axvline(odor_on_eff, ls="--", lw=ODOR_MARKER_LW, color="black")
        ax.axvline(trial_off_eff, ls="--", lw=ODOR_MARKER_LW, color="black")

        if light_s is not None and light_s <= x_max:
            ax.axvline(light_s, ls="-.", lw=LIGHT_MARKER_LW, color="tab:green")

        ax.plot(t_common, mean, lw=TRACE_LW, color=MEAN_COLOR)
        ax.fill_between(t_common, mean - sem, mean + sem,
                        alpha=SEM_ALPHA, color=SEM_COLOR)

        ax.set_ylim(0, FIXED_Y_MAX)
        ax.set_xlim(0, x_max)
        ax.margins(x=0, y=0.02)

        n_flies = matrix.shape[0]
        label = f"Training {tnum}  (n={n_flies})"
        ax.text(0.01, 0.92, label, transform=ax.transAxes, ha="left",
                va="top", fontsize=11, weight="bold", color="tab:blue")

    axes[-1].set_xlabel("Time (s)", fontsize=11)
    fig.text(0.02, 0.5, "Proboscis Distance (combined %)", va="center",
             rotation="vertical", fontsize=10)

    legend_handles = [
        plt.Line2D([0], [0], lw=TRACE_LW, color=MEAN_COLOR, label="Mean"),
        plt.Rectangle((0, 0), 1, 1, alpha=SEM_ALPHA, color=SEM_COLOR, label="SEM"),
        plt.Line2D([0], [0], ls="--", lw=ODOR_MARKER_LW, color="black",
                   label="Odor at fly"),
        plt.Rectangle((0, 0), 1, 1, alpha=ODOR_PLUS_LIGHT_ALPHA,
                       color=ODOR_PLUS_LIGHT_COLOR, label="Odor + light"),
        plt.Line2D([0], [0], ls="-.", lw=LIGHT_MARKER_LW, color="tab:green",
                   label="Light pulsing starts"),
    ]
    fig.legend(handles=legend_handles, loc="upper right",
               bbox_to_anchor=(0.98, 0.98), frameon=True, fontsize=9)

    rig_label = "Rig 1" if rig == "rig_1" else "Rig 2"
    fig.text(0.12, 0.97,
             f"Average Training PER Traces \u2014 {rig_label}",
             ha="left", va="center", fontsize=14, weight="bold")
    fig.text(0.12, 0.945, group_name,
             ha="left", va="center", fontsize=12, weight="bold", color="black")

    fig.tight_layout(rect=[0.04, 0, 1, 0.93])

    safe_name = group_name.replace(" ", "_").replace("/", "_")
    out_path = OUT_DIR / f"avg_training_{safe_name}_{rig}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main() -> None:
    flagged = _load_flagged()

    # ── New Flys ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Group: New Flys (Hex-Training-24-0.005 + EB-Training-24-0.1)")
    print("=" * 60)
    new_data = load_group(
        NEW_CSV,
        NEW_DATASETS,
        flagged=set(),           # no flagged filtering for new
        after_day=(4, 11),       # after April 11
        trial_nums=NEW_TRIALS,
    )
    for rig in ("rig_1", "rig_2"):
        plot_group("New Flys", rig, new_data[rig], NEW_TRIALS)

    # ── Old Flys ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Group: Old Flys (Hex/Benz/3OCT/ACV/AIR/EB-Training)")
    print("=" * 60)
    old_data = load_group(
        ALL_CSV,
        OLD_DATASETS,
        flagged=flagged,
        date_filter="before_jan",
        trial_nums=OLD_TRIALS,
    )
    for rig in ("rig_1", "rig_2"):
        plot_group("Old Flys", rig, old_data[rig], OLD_TRIALS)


if __name__ == "__main__":
    main()
