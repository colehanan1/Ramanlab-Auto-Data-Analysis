"""Cohort-average training PER traces — the per-fly training figure, for a whole dataset.

The published per-fly figure
(``<fly>_training_envelope_trials_by_odor_30_shifted.png``) stacks one panel per
training trial for a *single* fly: raw ``Max Distance x Angle %`` over 90 s, the
odor window shaded, the light-onset marker, one panel each for training 1–6.
Reading a cohort out of it means paging through 20–70 PNGs.

This driver draws the same six-panel layout once per dataset, with each panel
carrying the **cohort mean ± SEM** across every fly instead of one fly's trace.
The per-fly red threshold line has no cohort analogue (``θ`` is per-trial,
per-fly) and is dropped; the SEM band replaces it as the spread indicator.

Two representations are available:

``--baseline-subtract`` off (default)
    Raw ``Max Distance × Angle %``, y fixed to 0–100 — the same axis as the
    per-fly figure, so the two are directly comparable. Per-fly resting
    proboscis offsets survive into the mean, which is what makes the baseline
    sit near 25–30 %.

``--baseline-subtract`` on
    Each fly's trace is first shifted by its own pre-odor mean, matching the
    ``Training_vs_Control`` cohort-mean convention. Offsets cancel, so the
    panel shows the odor/light-evoked deflection alone.

Usage::

    python scripts/analysis/avg_training_traces_dataset.py \
        --wide-csv /home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/all_envelope_rows_wide_combined_base_training.parquet \
        --dataset 3Oct-Training-24-0.1 \
        --config config/config_new.yaml \
        --flagged-flies-csv /home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/flagged-flys-truth.csv \
        --out-dir /home/ramanlab/Documents/cole/Results/New-Opto-Fly-Figures/Raw-Training-PER-Traces/Cohort-Averages
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fbpipe.analysis.traces import baseline_correct, read_wide_table  # noqa: E402
from scripts.analysis.odor_bar_palette import odor_color  # noqa: E402
from scripts.analysis.rig_batch_breakdowns import batch_of  # noqa: E402
from scripts.analysis.envelope_visuals import (  # noqa: E402
    ODOR_PLUS_LIGHT_ALPHA,
    ODOR_PLUS_LIGHT_COLOR,
    _canon_dataset,
    _display_odor,
    _normalise_fly_columns,
    _trial_num,
    compute_non_reactive_flags,
    set_protocol,
)

LOGGER = logging.getLogger("avg_training_traces_dataset")

#: The v2 training schedule — six paired odor+light presentations.
TRAINING_TRIALS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

DPI = 300
MAX_TIME_S = 90.0
FIXED_Y_MAX = 100.0
MEAN_LW = 1.6
MEAN_COLOR = "black"
SEM_ALPHA = 0.30
SEM_FALLBACK_COLOR = "#7f7f7f"
LIGHT_COLOR = "tab:green"
LIGHT_LW = 1.3
TIME_TICK_STEP_S = 15.0


# ---------------------------------------------------------------------------
# selection


def select_dataset_rows(
    wide_df: pd.DataFrame, dataset: str, *, batch: int | None = None
) -> pd.DataFrame:
    """Training rows for one dataset, optionally narrowed to one starvation batch.

    The batch lives in the fly folder name as the ``batch_N`` token, so
    ``july_13_batch_1`` and ``july_13_batch_1_rig_2`` are both batch 1 — the
    same rule ``rig_batch_breakdowns`` uses, kept shared so a fly can never land
    in different batches in two figures.
    """
    frame = wide_df[wide_df["dataset"].astype(str).str.strip() == str(dataset)]
    if "trial_type" in frame.columns:
        frame = frame[frame["trial_type"].astype(str).str.strip() == "training"]
    if batch is not None and not frame.empty:
        frame = frame[frame["fly"].map(batch_of) == int(batch)]
    return frame.copy()


# ---------------------------------------------------------------------------
# collection


def collect_training_traces(
    ds_df: pd.DataFrame,
    *,
    dir_cols: Sequence[str],
    trials: Sequence[int] = TRAINING_TRIALS,
    baseline_frames: int | None = None,
    dataset: str = "",
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, list[float]], str]:
    """``(traces, light_on_s, odor_label)`` for one training cohort.

    *traces* is ``{trial number: {fly_id: trace}}`` with one trace per fly and
    trial — a duplicated wide row cannot inflate ``n``. *light_on_s* mirrors it
    with each fly's measured first-light-on, which is what the green marker is
    averaged from: the v2 protocol has no fixed light schedule to fall back on,
    so the plotted onset has to come from the data.
    """
    trial_set = {int(t) for t in trials}
    traces: dict[int, dict[str, np.ndarray]] = {t: {} for t in sorted(trial_set)}
    light: dict[int, list[float]] = {t: [] for t in sorted(trial_set)}
    odor_label = ""
    if ds_df.empty:
        return traces, light, odor_label

    dir_cols = list(dir_cols)
    has_light = "trial_light_on_s" in ds_df.columns
    dataset_canon = _canon_dataset(dataset) if dataset else ""

    for _, row in ds_df.iterrows():
        trial_label = str(row["trial_label"])
        trial = _trial_num(trial_label)
        if trial not in trial_set:
            continue
        fly_id = f"{row['fly']}_fly{row['fly_number']}"
        if fly_id in traces[trial]:
            continue

        trace = row[dir_cols].to_numpy(dtype=np.float64)
        finite = np.isfinite(trace)
        if not finite.any():
            continue
        trace = trace[: np.where(finite)[0][-1] + 1]
        if baseline_frames:
            trace = baseline_correct(trace, baseline_frames)
        traces[trial][fly_id] = trace

        if has_light:
            value = pd.to_numeric(row["trial_light_on_s"], errors="coerce")
            if np.isfinite(value) and value > 0:
                light[trial].append(float(value))

        if not odor_label and dataset_canon:
            odor_label = _display_odor(dataset_canon, trial_label)

    return traces, light, odor_label


# ---------------------------------------------------------------------------
# aggregation


def resample_traces(
    per_fly: dict[str, np.ndarray], *, fps: float, max_time_s: float
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Put every fly's trace on one time grid, NaN-padded past its own end.

    Recordings differ by a frame or two (3576–3608 samples for a nominal 90 s at
    40 fps), so a naive column stack would either truncate or misalign. The NaN
    padding is what lets ``mean_sem`` drop a fly only for the samples it is
    actually missing.
    """
    n_pts = int(round(max_time_s * fps)) + 1
    t_common = np.linspace(0.0, max_time_s, n_pts)
    fly_ids = sorted(per_fly)
    if not fly_ids:
        return t_common, np.empty((0, n_pts)), []
    rows = []
    for fly_id in fly_ids:
        trace = np.asarray(per_fly[fly_id], dtype=np.float64)
        t_orig = np.arange(len(trace)) / float(fps)
        rows.append(np.interp(t_common, t_orig, trace, left=np.nan, right=np.nan))
    return t_common, np.vstack(rows), fly_ids


def mean_sem(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(mean, sem, n)`` down the fly axis, ignoring NaN padding.

    ``n`` is the per-sample finite count, so a sample where a short recording
    has dropped out reports the smaller cohort rather than pretending the SEM
    was computed over everybody.
    """
    if matrix.size == 0:
        width = matrix.shape[1] if matrix.ndim == 2 else 0
        empty = np.full(width, np.nan)
        return empty, empty.copy(), np.zeros(width, dtype=int)
    counts = np.sum(np.isfinite(matrix), axis=0).astype(int)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(matrix, axis=0)
        sd = np.nanstd(matrix, axis=0)
    sem = np.divide(
        sd, np.sqrt(counts, dtype=float),
        out=np.full(counts.shape, np.nan), where=counts > 0,
    )
    return mean, sem, counts


def _mean_light_on(light: dict[int, list[float]]) -> float | None:
    values = [v for vals in light.values() for v in vals]
    return float(np.mean(values)) if values else None


# ---------------------------------------------------------------------------
# plotting


def _shared_ylim(
    curves: Sequence[tuple[np.ndarray, np.ndarray]], *, pad: float = 0.08
) -> tuple[float, float] | None:
    """One y-range covering every panel's mean ± SEM.

    Autoscaling each trial separately would make a response that shrinks from
    trial 1 to trial 6 look constant — the exact comparison the six-panel
    layout exists to support.
    """
    lows, highs = [], []
    for mean, sem in curves:
        band_lo = np.nanmin(mean - sem) if np.isfinite(mean - sem).any() else np.nan
        band_hi = np.nanmax(mean + sem) if np.isfinite(mean + sem).any() else np.nan
        if np.isfinite(band_lo):
            lows.append(float(band_lo))
        if np.isfinite(band_hi):
            highs.append(float(band_hi))
    if not lows or not highs:
        return None
    low, high = min(lows), max(highs)
    span = high - low
    if span <= 0:
        span = max(abs(high), 1.0)
    return low - pad * span, high + pad * span


def plot_training_means(
    traces: dict[int, dict[str, np.ndarray]],
    *,
    odor: str,
    title: str,
    subtitle: str,
    fps: float,
    odor_on_s: float,
    odor_off_s: float,
    max_time_s: float = MAX_TIME_S,
    light_on_s: float | None = None,
    baseline_subtracted: bool = False,
    ylim: tuple[float, float] | None = None,
    trials: Sequence[int] = TRAINING_TRIALS,
) -> tuple[plt.Figure, dict[int, dict[str, float]]]:
    """Six stacked panels of cohort mean ± SEM, one per training trial.

    Returns the figure and a per-trial summary (``n_flies``, ``peak_mean`` and
    when it occurs) so the caller can write a sidecar without re-deriving it.
    """
    trials = list(trials)
    band_color = odor_color(odor) or SEM_FALLBACK_COLOR

    plt.rcParams.update({
        "figure.dpi": DPI, "savefig.dpi": DPI,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "font.family": "Arial", "font.sans-serif": ["Arial"],
        "font.size": 10,
        "xtick.direction": "out", "ytick.direction": "out",
    })

    # Aggregate every panel first: the shared y-range needs all of them.
    aggregates: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}
    for trial in trials:
        per_fly = traces.get(trial, {})
        if not per_fly:
            continue
        t, matrix, fly_ids = resample_traces(per_fly, fps=fps, max_time_s=max_time_s)
        mean, sem, counts = mean_sem(matrix)
        aggregates[trial] = (t, mean, sem, counts, fly_ids)

    if ylim is None:
        ylim = (
            _shared_ylim([(m, s) for _, m, s, _, _ in aggregates.values()])
            if baseline_subtracted
            else (0.0, FIXED_Y_MAX)
        )

    fig, axes = plt.subplots(
        len(trials), 1, figsize=(10, 1.6 * len(trials) + 1.6), sharex=True
    )
    axes = np.atleast_1d(axes)
    summary: dict[int, dict[str, float]] = {}

    for ax, trial in zip(axes, trials):
        ax.axvspan(
            odor_on_s, min(odor_off_s, max_time_s),
            color=ODOR_PLUS_LIGHT_COLOR, alpha=ODOR_PLUS_LIGHT_ALPHA, linewidth=0,
        )
        if light_on_s is not None and 0 <= light_on_s <= max_time_s:
            ax.axvline(light_on_s, linestyle="-.", linewidth=LIGHT_LW, color=LIGHT_COLOR)

        if trial not in aggregates:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            summary[trial] = {"n_flies": 0}
        else:
            t, mean, sem, counts, fly_ids = aggregates[trial]
            ax.fill_between(
                t, mean - sem, mean + sem, color=band_color, alpha=SEM_ALPHA, linewidth=0
            )
            ax.plot(t, mean, color=MEAN_COLOR, linewidth=MEAN_LW)
            during = (t >= odor_on_s) & (t <= min(odor_off_s, max_time_s))
            peak_idx = int(np.nanargmax(np.where(during, mean, -np.inf)))
            summary[trial] = {
                "n_flies": len(fly_ids),
                "min_n_per_sample": int(counts.min()) if counts.size else 0,
                "peak_mean": float(mean[peak_idx]),
                "peak_time_s": float(t[peak_idx]),
                "peak_sem": float(sem[peak_idx]),
            }

        if baseline_subtracted:
            ax.axhline(0.0, color="0.35", linestyle=":", linewidth=0.8)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlim(0.0, max_time_s)
        ax.xaxis.set_major_locator(MultipleLocator(TIME_TICK_STEP_S))
        ax.margins(x=0)

        n_flies = summary[trial]["n_flies"]
        ax.text(
            0.012, 0.94, f"{odor}  —  Training {trial}  (n = {n_flies})",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=11, weight="bold", color="black",
        )

    axes[-1].set_xlabel("Time (s)", fontsize=12)
    y_label = (
        "Δ Max Distance x Angle %" if baseline_subtracted
        else "Max Distance x Angle %"
    )
    fig.text(0.015, 0.5, y_label, va="center", rotation="vertical", fontsize=12)

    legend_handles = [
        plt.Line2D([0], [0], color=MEAN_COLOR, lw=MEAN_LW, label="Cohort mean"),
        plt.Rectangle((0, 0), 1, 1, color=band_color, alpha=SEM_ALPHA, label="± SEM"),
        plt.Rectangle(
            (0, 0), 1, 1, color=ODOR_PLUS_LIGHT_COLOR, alpha=ODOR_PLUS_LIGHT_ALPHA,
            label="Odor",
        ),
    ]
    if light_on_s is not None:
        legend_handles.append(
            plt.Line2D([0], [0], linestyle="-.", lw=LIGHT_LW, color=LIGHT_COLOR,
                       label="Light pulsing starts")
        )
    fig.legend(
        handles=legend_handles, loc="upper right", bbox_to_anchor=(0.99, 0.99),
        frameon=True, fontsize=9,
    )
    fig.text(0.10, 0.985, title, ha="left", va="center", fontsize=15, weight="bold")
    fig.text(0.10, 0.958, subtitle, ha="left", va="center", fontsize=11, weight="bold")
    fig.tight_layout(rect=[0.045, 0, 1, 0.94])
    return fig, summary


# ---------------------------------------------------------------------------
# CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--wide-csv", type=Path, required=True,
                   help="all_envelope_rows_wide_combined_base_training CSV or Parquet.")
    p.add_argument("--dataset", required=True, help="e.g. 3Oct-Training-24-0.1")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--config", type=str, default="",
                   help="Pipeline config YAML; loads dataset_overrides.odor_remap.")
    p.add_argument("--flagged-flies-csv", type=str, default="",
                   help="flagged-flys-truth CSV (FLY-State != 1 excluded).")
    p.add_argument("--batch", type=int, default=None,
                   help="Restrict to one starvation batch (batch_N in the fly name).")
    p.add_argument("--trials", type=int, nargs="+", default=list(TRAINING_TRIALS))
    p.add_argument("--fps", type=float, default=40.0)
    p.add_argument("--odor-on-s", type=float, default=30.0)
    p.add_argument("--odor-off-s", type=float, default=60.0)
    p.add_argument("--max-time-s", type=float, default=MAX_TIME_S)
    p.add_argument("--baseline-subtract", action="store_true",
                   help="Shift each fly by its own pre-odor mean before averaging.")
    p.add_argument("--protocol", default="v2", choices=["v2", "legacy"])
    p.add_argument("--overwrite", action="store_true", default=True)
    p.add_argument("--verbose", action="store_true")
    return p


def _apply_config_remap(config_path: str) -> None:
    from fbpipe.config import load_settings

    from scripts.analysis.envelope_visuals import set_dataset_odor_remap

    settings = load_settings(config_path)
    remap = {
        str(ds): dict(ov.odor_remap)
        for ds, ov in settings.dataset_overrides.items()
        if getattr(ov, "odor_remap", None)
    }
    if remap:
        set_dataset_odor_remap(remap)
        LOGGER.info("Loaded odor_remap for %d datasets from %s", len(remap), config_path)


def _stem(dataset: str, *, batch: int | None, baseline: bool) -> str:
    stem = f"avg_training_traces_{dataset}"
    if batch is not None:
        stem += f"_batch{batch}"
    if baseline:
        stem += "_baseline"
    return stem


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    set_protocol(args.protocol)
    if args.config:
        _apply_config_remap(args.config)

    wide_df = _normalise_fly_columns(read_wide_table(args.wide_csv))
    LOGGER.info("Loaded %d rows from %s", len(wide_df), args.wide_csv)

    n_flagged = 0
    if args.flagged_flies_csv:
        flagged = compute_non_reactive_flags(
            wide_df, flagged_flies_csv=args.flagged_flies_csv
        )
        dropped = wide_df.loc[
            flagged & (wide_df["dataset"].astype(str).str.strip() == args.dataset),
            ["fly", "fly_number"],
        ].drop_duplicates()
        n_flagged = len(dropped)
        LOGGER.info("Excluding %d flagged flies from %s", n_flagged, args.dataset)
        wide_df = wide_df.loc[~flagged].copy()

    ds_df = select_dataset_rows(wide_df, args.dataset, batch=args.batch)
    if ds_df.empty:
        raise SystemExit(
            f"No training rows for {args.dataset}"
            + (f" batch {args.batch}" if args.batch is not None else "")
        )

    dir_cols = sorted(
        [c for c in ds_df.columns if c.startswith("dir_val_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    baseline_frames = (
        max(1, int(round(args.odor_on_s * args.fps))) if args.baseline_subtract else None
    )
    traces, light, odor = collect_training_traces(
        ds_df,
        dir_cols=dir_cols,
        trials=args.trials,
        baseline_frames=baseline_frames,
        dataset=args.dataset,
    )
    all_flies = sorted({fly for per_fly in traces.values() for fly in per_fly})
    if not all_flies:
        raise SystemExit(f"No usable traces for {args.dataset}")
    for trial in args.trials:
        LOGGER.info("  Training %d: n=%d flies", trial, len(traces.get(trial, {})))
    LOGGER.info("Cohort: %d flies, odor %s", len(all_flies), odor or "(unknown)")

    light_on_s = _mean_light_on(light)
    batch_note = f", batch {args.batch}" if args.batch is not None else ""
    fig, summary = plot_training_means(
        traces,
        odor=odor or args.dataset,
        title="Proboscis Distance Across Training Trials — Cohort Average",
        subtitle=f"{args.dataset}{batch_note}  —  n = {len(all_flies)} flies",
        fps=args.fps,
        odor_on_s=args.odor_on_s,
        odor_off_s=args.odor_off_s,
        max_time_s=args.max_time_s,
        light_on_s=light_on_s,
        baseline_subtracted=args.baseline_subtract,
        trials=args.trials,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = _stem(args.dataset, batch=args.batch, baseline=args.baseline_subtract)
    out_png = args.out_dir / f"{stem}.png"
    if args.overwrite or not out_png.exists():
        fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
        fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight")
        LOGGER.info("Saved %s", out_png)
    plt.close(fig)

    sidecar = {
        "dataset": args.dataset,
        "odor": odor,
        "batch": args.batch,
        "trials": list(args.trials),
        "n_flies": len(all_flies),
        "flies": all_flies,
        "flagged_flies_excluded": n_flagged,
        "flagged_flies_csv": args.flagged_flies_csv,
        "baseline_subtracted": bool(args.baseline_subtract),
        "fps": args.fps,
        "odor_on_s": args.odor_on_s,
        "odor_off_s": args.odor_off_s,
        "max_time_s": args.max_time_s,
        "light_on_s": light_on_s,
        "per_trial": {str(t): summary.get(t, {"n_flies": 0}) for t in args.trials},
    }
    sidecar_path = args.out_dir / f"{stem}.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    LOGGER.info("Saved %s", sidecar_path)


if __name__ == "__main__":
    main()
