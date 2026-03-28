"""Analyze PID odor-check runs from per-trial CSV exports.

This script summarizes each trial and each odor across repeats.

For each trial it computes:
- A local baseline from the first few seconds after the odor-on command.
- An onset latency: the first time the PID signal stays above a baseline-based
  threshold for several consecutive samples.
- Peak PID values during the odor-on window and across the full trial.

It also writes summary figures to make the onset timing and peak sizes easy to
inspect across odors.

Example
-------
python scripts/analysis/analyze_pid_odor_check.py \
    --input-dir /path/to/pid_odor_check_20260326_154331 \
    --outdir data/pid_analysis/pid_odor_check_20260326_154331
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)

TRIAL_FILENAME_RE = re.compile(r"trial_(?P<trial_index>\d+)_(?P<odor>.+)_rep(?P<repeat>\d+)\.csv$")
REQUIRED_COLUMNS = {"Timestamp", "Voltage", "Phase", "Odor", "Repeat"}
TRACE_COLOURS = ("#1f77b4", "#ff7f0e")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing per-trial PID CSV exports.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Directory for summary CSVs and plots.",
    )
    parser.add_argument(
        "--baseline-window-sec",
        type=float,
        default=5.0,
        help="Seconds from trial start used to estimate the local baseline.",
    )
    parser.add_argument(
        "--threshold-sigma",
        type=float,
        default=3.0,
        help="Baseline standard-deviation multiplier used for onset detection.",
    )
    parser.add_argument(
        "--min-abs-delta",
        type=float,
        default=0.003,
        help="Minimum voltage rise above baseline required for onset detection.",
    )
    parser.add_argument(
        "--sustain-points",
        type=int,
        default=3,
        help="Number of consecutive samples above threshold required for onset.",
    )
    parser.add_argument(
        "--plot-window-sec",
        type=float,
        default=70.0,
        help="Seconds from trial start to display in the trace overview figure.",
    )
    return parser.parse_args(argv)


def _load_trial(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required columns: {missing_str}")

    df = df.copy()
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    df["Voltage"] = pd.to_numeric(df["Voltage"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "Voltage"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{path} does not contain usable PID samples.")

    df["Phase"] = df["Phase"].astype(str).str.strip()
    df["Odor"] = df["Odor"].astype(str).str.strip()
    df["Repeat"] = pd.to_numeric(df["Repeat"], errors="coerce")
    df["elapsed_sec"] = df["Timestamp"] - float(df["Timestamp"].iloc[0])
    return df


def _first_sustained_crossing(
    signal: np.ndarray,
    threshold: float,
    sustain_points: int,
) -> int | None:
    if sustain_points <= 0:
        raise ValueError("sustain_points must be positive.")
    if len(signal) < sustain_points:
        return None

    for start_idx in range(0, len(signal) - sustain_points + 1):
        if np.all(signal[start_idx : start_idx + sustain_points] > threshold):
            return start_idx
    return None


def _compute_trial_metrics(
    path: Path,
    *,
    baseline_window_sec: float,
    threshold_sigma: float,
    min_abs_delta: float,
    sustain_points: int,
) -> dict[str, object]:
    match = TRIAL_FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected trial filename format: {path.name}")

    trial_index = int(match.group("trial_index"))
    odor_code = match.group("odor")
    repeat = int(match.group("repeat"))

    df = _load_trial(path)
    odor_on = df.loc[df["Phase"].str.lower() == "odor_on"].copy().reset_index(drop=True)
    if odor_on.empty:
        raise ValueError(f"{path} does not contain any odor_on samples.")

    baseline_df = odor_on.loc[odor_on["elapsed_sec"] <= baseline_window_sec]
    if baseline_df.empty:
        baseline_df = odor_on.iloc[: min(25, len(odor_on))].copy()

    baseline_mean = float(baseline_df["Voltage"].mean())
    baseline_sd = float(baseline_df["Voltage"].std(ddof=1))
    if math.isnan(baseline_sd):
        baseline_sd = 0.0

    threshold_delta = max(threshold_sigma * baseline_sd, min_abs_delta)
    detection_threshold = baseline_mean + threshold_delta

    onset_idx = _first_sustained_crossing(
        odor_on["Voltage"].to_numpy(),
        detection_threshold,
        sustain_points,
    )
    onset_latency_sec = (
        float(odor_on.loc[onset_idx, "elapsed_sec"]) if onset_idx is not None else np.nan
    )
    onset_timestamp = (
        float(odor_on.loc[onset_idx, "Timestamp"]) if onset_idx is not None else np.nan
    )

    peak_on_idx = int(odor_on["Voltage"].idxmax())
    peak_on_row = odor_on.loc[peak_on_idx]

    peak_all_idx = int(df["Voltage"].idxmax())
    peak_all_row = df.loc[peak_all_idx]

    return {
        "trial_index": trial_index,
        "odor": odor_code,
        "repeat": repeat,
        "csv_file": path.name,
        "sample_count": int(len(df)),
        "odor_on_sample_count": int(len(odor_on)),
        "trial_duration_sec": float(df["elapsed_sec"].iloc[-1]),
        "odor_on_duration_sec": float(odor_on["elapsed_sec"].iloc[-1]),
        "baseline_window_sec": float(baseline_window_sec),
        "baseline_mean_voltage": baseline_mean,
        "baseline_sd_voltage": baseline_sd,
        "detection_threshold_voltage": detection_threshold,
        "detection_threshold_delta_voltage": threshold_delta,
        "detected": bool(onset_idx is not None),
        "onset_latency_sec": onset_latency_sec,
        "onset_timestamp_sec": onset_timestamp,
        "peak_odor_on_voltage": float(peak_on_row["Voltage"]),
        "peak_odor_on_delta_voltage": float(peak_on_row["Voltage"] - baseline_mean),
        "peak_odor_on_latency_sec": float(peak_on_row["elapsed_sec"]),
        "peak_overall_voltage": float(peak_all_row["Voltage"]),
        "peak_overall_delta_voltage": float(peak_all_row["Voltage"] - baseline_mean),
        "peak_overall_latency_sec": float(peak_all_row["elapsed_sec"]),
        "peak_overall_phase": str(peak_all_row["Phase"]),
        "peak_overall_occurs_after_odor_off": str(peak_all_row["Phase"]).lower() != "odor_on",
    }


def _build_trial_summary(args: argparse.Namespace) -> pd.DataFrame:
    trial_paths = sorted(args.input_dir.glob("trial_*.csv"))
    if not trial_paths:
        raise FileNotFoundError(f"No trial_*.csv files found in {args.input_dir}")

    rows = [
        _compute_trial_metrics(
            path,
            baseline_window_sec=args.baseline_window_sec,
            threshold_sigma=args.threshold_sigma,
            min_abs_delta=args.min_abs_delta,
            sustain_points=args.sustain_points,
        )
        for path in trial_paths
    ]

    trial_summary = pd.DataFrame(rows).sort_values(["trial_index", "repeat"]).reset_index(drop=True)
    return trial_summary


def _build_odor_summary(trial_summary: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        trial_summary.groupby("odor", as_index=False)
        .agg(
            repeats=("repeat", "count"),
            detected_repeats=("detected", "sum"),
            onset_mean_sec=("onset_latency_sec", "mean"),
            onset_sd_sec=("onset_latency_sec", "std"),
            peak_odor_on_mean_voltage=("peak_odor_on_voltage", "mean"),
            peak_odor_on_sd_voltage=("peak_odor_on_voltage", "std"),
            peak_odor_on_mean_delta_voltage=("peak_odor_on_delta_voltage", "mean"),
            peak_odor_on_sd_delta_voltage=("peak_odor_on_delta_voltage", "std"),
            peak_overall_mean_voltage=("peak_overall_voltage", "mean"),
            peak_overall_sd_voltage=("peak_overall_voltage", "std"),
            peak_overall_mean_delta_voltage=("peak_overall_delta_voltage", "mean"),
            peak_overall_sd_delta_voltage=("peak_overall_delta_voltage", "std"),
            peak_overall_mean_latency_sec=("peak_overall_latency_sec", "mean"),
            peak_overall_sd_latency_sec=("peak_overall_latency_sec", "std"),
        )
        .copy()
    )
    grouped["robust_detection"] = grouped["detected_repeats"] == grouped["repeats"]

    odor_order = (
        trial_summary.sort_values("trial_index")
        .drop_duplicates("odor")["odor"]
        .tolist()
    )
    grouped["odor"] = pd.Categorical(grouped["odor"], categories=odor_order, ordered=True)
    grouped = grouped.sort_values("odor").reset_index(drop=True)
    grouped["odor"] = grouped["odor"].astype(str)
    return grouped


def _plot_trace_overview(
    trial_summary: pd.DataFrame,
    source_dir: Path,
    out_path: Path,
    *,
    plot_window_sec: float,
) -> None:
    odors = trial_summary["odor"].drop_duplicates().tolist()
    n_panels = len(odors)
    n_cols = 2
    n_rows = math.ceil(n_panels / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.2 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, odor in zip(axes_flat, odors):
        odor_trials = trial_summary.loc[trial_summary["odor"] == odor].sort_values("repeat")
        for colour, (_, row) in zip(TRACE_COLOURS, odor_trials.iterrows()):
            df = _load_trial(source_dir / str(row["csv_file"]))
            df["delta_voltage"] = df["Voltage"] - float(row["baseline_mean_voltage"])
            window_df = df.loc[df["elapsed_sec"] <= plot_window_sec].copy()

            ax.plot(
                window_df["elapsed_sec"],
                window_df["delta_voltage"],
                color=colour,
                linewidth=1.5,
                alpha=0.95,
                label=f"rep{int(row['repeat'])}",
            )

            onset_latency = row["onset_latency_sec"]
            if pd.notna(onset_latency):
                onset_delta = row["detection_threshold_voltage"] - row["baseline_mean_voltage"]
                ax.scatter(
                    [onset_latency],
                    [onset_delta],
                    color=colour,
                    s=20,
                    zorder=5,
                )

            peak_latency = row["peak_overall_latency_sec"]
            peak_delta = row["peak_overall_delta_voltage"]
            if peak_latency <= plot_window_sec:
                ax.scatter(
                    [peak_latency],
                    [peak_delta],
                    color=colour,
                    marker="x",
                    s=32,
                    zorder=5,
                )

        ax.axvline(30.0, color="#555555", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.axhline(0.0, color="#999999", linestyle=":", linewidth=0.8)
        ax.set_title(odor, fontsize=11, weight="bold")
        ax.set_xlabel("Time Since Odor On (s)")
        ax.set_ylabel("Voltage Above Local Baseline (V)")
        ax.set_xlim(0, plot_window_sec)
        ax.legend(frameon=False, fontsize=8, loc="upper left")

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    fig.suptitle("PID Odor Check Traces", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_summary_bars(
    trial_summary: pd.DataFrame,
    odor_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    odors = odor_summary["odor"].tolist()
    x = np.arange(len(odors))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].bar(
        x,
        odor_summary["onset_mean_sec"].to_numpy(),
        color="#4c78a8",
        alpha=0.85,
    )
    for i, odor in enumerate(odors):
        reps = trial_summary.loc[trial_summary["odor"] == odor, "onset_latency_sec"].dropna()
        axes[0].scatter(np.full(len(reps), i), reps.to_numpy(), color="#1f1f1f", s=20, zorder=5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(odors)
    axes[0].set_ylabel("Onset Latency (s)")
    axes[0].set_title("Time To First Detectable PID Rise")

    axes[1].bar(
        x,
        odor_summary["peak_overall_mean_voltage"].to_numpy(),
        color="#f58518",
        alpha=0.85,
    )
    for i, odor in enumerate(odors):
        reps = trial_summary.loc[trial_summary["odor"] == odor, "peak_overall_voltage"]
        axes[1].scatter(np.full(len(reps), i), reps.to_numpy(), color="#1f1f1f", s=20, zorder=5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(odors)
    axes[1].set_ylabel("Peak PID Voltage (V)")
    axes[1].set_title("Maximum PID Reading Per Odor")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    args.input_dir = args.input_dir.expanduser().resolve()
    args.outdir = args.outdir.expanduser().resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)

    trial_summary = _build_trial_summary(args)
    odor_summary = _build_odor_summary(trial_summary)

    trial_summary_path = args.outdir / "pid_trial_summary.csv"
    odor_summary_path = args.outdir / "pid_odor_summary.csv"
    trial_summary.to_csv(trial_summary_path, index=False)
    odor_summary.to_csv(odor_summary_path, index=False)

    _plot_trace_overview(
        trial_summary,
        args.input_dir,
        args.outdir / "pid_trace_overview.png",
        plot_window_sec=args.plot_window_sec,
    )
    _plot_summary_bars(
        trial_summary,
        odor_summary,
        args.outdir / "pid_summary_bars.png",
    )

    print(f"Wrote {trial_summary_path}")
    print(f"Wrote {odor_summary_path}")
    print(f"Wrote {args.outdir / 'pid_trace_overview.png'}")
    print(f"Wrote {args.outdir / 'pid_summary_bars.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
