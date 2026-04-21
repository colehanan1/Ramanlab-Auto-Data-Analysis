"""
Average Training PER Traces by Rig.

Produces figures with subplots per training trial.
Each subplot shows the mean envelope trace across all flies in that rig,
with SEM shading. Matches the Raw-Training-PER-Traces visual style.

Supports multiple dataset configs with per-dataset exclusions, date filters,
flagged-fly filtering, and custom trial selections.
"""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Timing constants (match existing pipeline) ────────────────────────────
FPS_DEFAULT = 40.0
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

# ── Style constants ───────────────────────────────────────────────────────
FIXED_Y_MAX = 100.0
TRACE_LW = 1.4
ODOR_MARKER_LW = 1.0
LIGHT_MARKER_LW = 1.3
SEM_ALPHA = 0.25
SEM_COLOR = "steelblue"
MEAN_COLOR = "black"

RESULTS_ROOT = Path(
    "/home/ramanlab/Documents/cole/Results/Opto-Fly-Figures/Raw-Training-PER-Traces"
)

FLAGGED_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-ALL-Opto-Flys/flagged-flys-truth.csv"
)

# ── Month name → number for date filtering ────────────────────────────────
MONTH_NUM = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}


def _load_flagged(dataset_name: str) -> set[tuple[str, int]]:
    """Return set of (fly_folder, fly_number) with state < 1."""
    df = pd.read_csv(FLAGGED_CSV)
    df = df[df["dataset"] == dataset_name]
    df = df[df["FLY-State(1, 0, -1)"] < 1]
    return {(row["fly"], int(row["fly_number"])) for _, row in df.iterrows()}


@dataclass
class DatasetConfig:
    name: str
    data_root: Path
    folder_glob: str                         # e.g. "april_*" or "october_*"
    trials: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    exclude_batches: set[str] = field(default_factory=set)
    date_min: tuple[int, int] | None = None  # (month, day) inclusive
    date_max: tuple[int, int] | None = None  # (month, day) inclusive
    use_flagged_csv: bool = False            # filter by flagged-flys-truth.csv


DATASETS: list[DatasetConfig] = [
    DatasetConfig(
        name="Hex-Training-24-0.005",
        data_root=Path("/home/ramanlab/Documents/cole/Data/flys_New/Hex-Training-24-0.005"),
        folder_glob="april_*",
        trials=[1, 2, 3, 4, 5, 6],
        exclude_batches={"april_11_batch_1"},
    ),
    DatasetConfig(
        name="Hex-Training",
        data_root=Path("/home/ramanlab/Documents/cole/Data/flys/Hex-Training"),
        folder_glob="october_*",
        trials=[1, 2, 3, 4, 6, 8],
        date_min=(10, 2),
        date_max=(10, 14),
        use_flagged_csv=True,
    ),
]


def _parse_folder_date(folder_name: str) -> tuple[int, int] | None:
    m = re.match(r"([a-z]+)_(\d+)", folder_name)
    if not m:
        return None
    month_str, day_str = m.group(1), m.group(2)
    month = MONTH_NUM.get(month_str)
    if month is None:
        return None
    return (month, int(day_str))


def _in_date_range(folder: str, cfg: DatasetConfig) -> bool:
    if cfg.date_min is None and cfg.date_max is None:
        return True
    parsed = _parse_folder_date(folder)
    if parsed is None:
        return False
    if cfg.date_min is not None and parsed < cfg.date_min:
        return False
    if cfg.date_max is not None and parsed > cfg.date_max:
        return False
    return True


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


def _classify_rig(batch_name: str) -> str:
    return "rig_2" if "_rig_2" in batch_name else "rig_1"


def _extract_fly_number(filename: str) -> int | None:
    """Extract fly number from envelope filename like 'training_1_hexanol_fly2_...'."""
    m = re.search(r"_fly(\d+)_", filename)
    return int(m.group(1)) if m else None


def load_traces(cfg: DatasetConfig) -> dict[str, dict[int, list[np.ndarray]]]:
    """Return {rig: {trial_num: [envelope_arrays]}}."""
    trial_set = set(cfg.trials)
    result: dict[str, dict[int, list[np.ndarray]]] = {
        "rig_1": {t: [] for t in cfg.trials},
        "rig_2": {t: [] for t in cfg.trials},
    }

    flagged: set[tuple[str, int]] = set()
    if cfg.use_flagged_csv:
        flagged = _load_flagged(cfg.name)
        if flagged:
            print(f"[{cfg.name}] Excluding {len(flagged)} flagged flies:")
            for fly, fn in sorted(flagged):
                print(f"    {fly} fly {fn}")

    pattern = str(
        cfg.data_root / cfg.folder_glob / "angle_distance_rms_envelope"
        / "training_*_fly*_angle_distance_rms_envelope.csv"
    )
    files = sorted(glob.glob(pattern))
    print(f"[{cfg.name}] Found {len(files)} envelope CSV files")

    skipped_flagged = 0
    for fpath in files:
        parts = Path(fpath).parts
        batch_name = parts[-3]

        if batch_name in cfg.exclude_batches:
            continue
        if not _in_date_range(batch_name, cfg):
            continue

        fname = Path(fpath).name
        fly_num = _extract_fly_number(fname)

        # Check flagged list
        if flagged and fly_num is not None and (batch_name, fly_num) in flagged:
            skipped_flagged += 1
            continue

        rig = _classify_rig(batch_name)

        m = re.match(r"training_(\d+)_", fname)
        if not m:
            continue
        trial_num = int(m.group(1))
        if trial_num not in trial_set:
            continue

        try:
            df = pd.read_csv(fpath)
        except Exception as exc:
            print(f"  SKIP {fpath}: {exc}")
            continue

        env = df["envelope_of_rms"].to_numpy(dtype=np.float64)
        result[rig][trial_num].append(env)

    if skipped_flagged:
        print(f"  Skipped {skipped_flagged} files from flagged flies")

    for rig in result:
        total = sum(len(result[rig][t]) for t in cfg.trials)
        if total == 0:
            continue
        for t in cfg.trials:
            n = len(result[rig][t])
            print(f"  {rig}  trial {t}: {n} traces")
    return result


def _resample_to_common(traces: list[np.ndarray], fps: float,
                         x_max: float) -> tuple[np.ndarray, np.ndarray]:
    n_pts = int(x_max * fps) + 1
    t_common = np.linspace(0, x_max, n_pts)
    resampled = []
    for env in traces:
        t_orig = np.arange(len(env)) / fps
        interp = np.interp(t_common, t_orig, env, left=np.nan, right=np.nan)
        resampled.append(interp)
    return t_common, np.array(resampled)


def plot_rig(rig: str, trials: dict[int, list[np.ndarray]],
             trial_nums: list[int], dataset_name: str, out_dir: Path) -> None:
    if all(len(trials[t]) == 0 for t in trial_nums):
        print(f"  [{dataset_name}] No data for {rig}, skipping.")
        return

    odor_on_eff = ODOR_ON_S
    x_max = ODOR_OFF_S + ODOR_LATENCY_S + AFTER_SHOW_SEC

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
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
    })

    n_rows = len(trial_nums)
    fig_h = max(3.0, n_rows * 1.6 + 1.5)
    fig_w = 10
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_w, fig_h), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for i, trial_num in enumerate(trial_nums):
        ax = axes[i]
        trace_list = trials[trial_num]
        trial_off_eff = _trial_odor_off(trial_num)
        light_s = _light_start(trial_num)

        if len(trace_list) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_ylabel("")
            continue

        t_common, matrix = _resample_to_common(trace_list, FPS_DEFAULT, x_max)
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
        label = f"Training {trial_num}  (n={n_flies})"
        ax.text(0.01, 0.92, label, transform=ax.transAxes, ha="left",
                va="top", fontsize=11, weight="bold", color="tab:blue")

    axes[-1].set_xlabel("Time (s)", fontsize=11)

    fig.text(0.02, 0.5, "Proboscis Distance (envelope RMS)", va="center",
             rotation="vertical", fontsize=10)

    legend_handles = [
        plt.Line2D([0], [0], lw=TRACE_LW, color=MEAN_COLOR, label="Mean"),
        plt.Rectangle((0, 0), 1, 1, alpha=SEM_ALPHA, color=SEM_COLOR,
                       label="SEM"),
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
    fig.text(0.12, 0.945, dataset_name,
             ha="left", va="center", fontsize=12, weight="bold", color="black")

    fig.tight_layout(rect=[0.04, 0, 1, 0.93])

    out_path = out_dir / f"avg_training_traces_{rig}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    for ds in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds.name}")
        print(f"{'='*60}")
        out_dir = RESULTS_ROOT / ds.name
        out_dir.mkdir(parents=True, exist_ok=True)

        data = load_traces(ds)
        for rig in ("rig_1", "rig_2"):
            plot_rig(rig, data[rig], ds.trials, ds.name, out_dir)


if __name__ == "__main__":
    main()
