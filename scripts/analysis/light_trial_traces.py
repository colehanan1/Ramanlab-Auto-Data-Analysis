#!/usr/bin/env python3
"""Per-fly proboscis-extension traces for the post-training LIGHT trials.

For the three RandomPanel datasets (24-0.1, 24-1, Training-24-10), plot each
fly's post-training light trials — the trials after the 14 odor-training trials,
in which the opto light turns on (``trial_light_on_s`` is populated).

One figure per fly folder (the ``fly`` column). Subplots are stacked vertically,
one per light trial, with one coloured trace per ``fly_number``
(1 = red, 2 = blue, 3 = green, 4 = purple) — matching the lab's
``per_folder_envelope_traces.py`` style. Each subplot marks the light-ON time
(vertical gold line) and the light-on window (shaded), using the per-trial
onset from the data and the config's ``light_only_duration`` (25 s).

Proboscis extension distance = ``dir_val_*`` (combined_pct envelope), scaled 0-100.

Usage:
    python scripts/analysis/light_trial_traces.py
    python scripts/analysis/light_trial_traces.py --csv /path/to.csv --outdir /path/out
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "src"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:
    from fbpipe.plot_style import apply_lab_style

    apply_lab_style()
except Exception:  # pragma: no cover - style is cosmetic only
    pass

# Shared per-trial light-stimulus check registry (populated by the pipeline
# driver, or by main() below for standalone runs). Keys join on
# (dataset, fly/batch folder, phase, trial number) — all available here.
from scripts.analysis.envelope_visuals import (  # noqa: E402
    _light_check_annotation,
    _lookup_light_check,
    load_light_check_fractions,
    set_light_check_fractions,
)

# ── defaults ────────────────────────────────────────────────────────────────
DEFAULT_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/"
    "all_envelope_rows_wide_combined_base.csv"
)
DEFAULT_OUTDIR = Path(
    "/home/ramanlab/Documents/cole/Results/New-Opto-Fly-Figures/"
    "Raw-Testing-PER-Traces/Light-Only"
)
# Datasets are auto-detected from the wide CSV (any dataset with light trials).
# This tuple is only a documented reference of the original three RandomPanel
# light datasets; pass --datasets to restrict, else all light datasets render.
TARGET_DATASETS = (
    "RandomPanel-24-0.1",
    "RandomPanel-24-1",
    "RandomPanel-Training-24-10",
)

# ── fly colours (matches per_folder_envelope_traces.py, + fly 4) ─────────────
FLY_COLORS = {1: "red", 2: "blue", 3: "green", 4: "#8e44ad"}

# ── light window (config_random_panel.yaml) ─────────────────────────────────
LIGHT_ONLY_DURATION_S = 25.0   # light on duration (light_only_duration)
X_MAX_S = 90.0                 # window shown per subplot (light response window)
FIXED_Y_MAX = 100.0
THRESHOLD_STD_MULT = 3.0

LIGHT_LINE_COLOR = "#e8a000"   # amber
LIGHT_SPAN_COLOR = "#ffd24d"   # light amber
LIGHT_SPAN_ALPHA = 0.30

# Light-trailer conditions in delivery order (config `light_trials`).
# trial 15 = first trailer, 16 = second, ...  Trials without a mapping
# (e.g. 20) are labelled by number only.
LIGHT_CONDITION = {
    15: "Solid 1 Hz (100% duty)",
    16: "Pulse 5 Hz (50% duty)",
    17: "Pulse 10 Hz (50% duty)",
    18: "Pulse 20 Hz (50% duty)",
    19: "Pulse 50 Hz (50% duty)",
}


# ── helpers (mirroring per_folder_envelope_traces.py) ────────────────────────

def natural_sort_key(s: str):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", str(s))]


_MONTH_NUM = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}


def _date_sort_key(folder: str) -> tuple[int, int, str]:
    m = re.match(r"([a-z]+)_(\d+)", str(folder).lower())
    if m:
        return (_MONTH_NUM.get(m.group(1), 99), int(m.group(2)), str(folder).lower())
    return (99, 99, str(folder).lower())


def _trial_num(trial_label: str) -> int:
    m = re.search(r"(\d+)", str(trial_label))
    return int(m.group(1)) if m else 0


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


def _compute_theta(env: np.ndarray, fps: float, before_s: float) -> float:
    """Threshold = median_before + k * MAD_before (scaled to sigma)."""
    n_before = int(round(max(before_s, 0.0) * fps))
    before = env[:n_before]
    before = before[np.isfinite(before)]
    if before.size < 3:
        return float("nan")
    med = float(np.median(before))
    mad = float(np.median(np.abs(before - med)))
    return med + THRESHOLD_STD_MULT * (mad * 1.4826)


def light_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of light trials — rows where the opto light turns on.

    Vectorised (avoids row-wise apply over the wide dir_val_* frame).
    """
    if "trial_light_on_s" not in df.columns:
        return pd.Series(False, index=df.index)
    vals = pd.to_numeric(df["trial_light_on_s"], errors="coerce")
    return np.isfinite(vals)


def detect_light_datasets(df: pd.DataFrame) -> list[str]:
    """Return, sorted, every dataset that has at least one light trial."""
    if "dataset" not in df.columns:
        return []
    light = df[light_mask(df)]
    return sorted(light["dataset"].astype(str).unique().tolist(), key=natural_sort_key)


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_folder(
    folder_df: pd.DataFrame,
    dataset: str,
    folder: str,
    dir_val_cols: list[str],
    out_dir: Path,
) -> Path | None:
    """One figure per folder: vertical subplots, one per light trial."""
    light_df = folder_df[light_mask(folder_df)]
    if light_df.empty:
        return None

    trial_labels = sorted(light_df["trial_label"].unique(), key=natural_sort_key)
    n_trials = len(trial_labels)

    fps = float(folder_df["fps"].iloc[0])
    if not math.isfinite(fps) or fps <= 0:
        fps = 40.0

    fly_numbers = sorted(int(f) for f in light_df["fly_number"].unique())

    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
        "font.family": "Arial", "font.sans-serif": ["Arial"], "font.size": 10,
    })

    header_in = 1.25  # reserved header height for the two-line title
    fig_h = max(3.4, n_trials * 1.7 + header_in + 0.35)
    fig, axes = plt.subplots(n_trials, 1, figsize=(10, fig_h), sharex=True)
    if n_trials == 1:
        axes = [axes]

    fig.text(0.02, 0.5, "Proboscis extension distance (combined base, 0-100)",
             va="center", rotation="vertical", fontsize=10)

    for ax_idx, trial_label in enumerate(trial_labels):
        ax = axes[ax_idx]
        trial_data = light_df[light_df["trial_label"] == trial_label]
        tnum = _trial_num(trial_label)

        # Light onset for this trial (a trial-level property; take the first).
        light_on_vals = pd.to_numeric(
            trial_data["trial_light_on_s"], errors="coerce"
        ).dropna()
        light_on = float(light_on_vals.iloc[0]) if not light_on_vals.empty else 30.0
        light_off = min(light_on + LIGHT_ONLY_DURATION_S, X_MAX_S)

        # ── shaded light-on window + onset/offset lines ──
        ax.axvspan(light_on, light_off, color=LIGHT_SPAN_COLOR,
                   alpha=LIGHT_SPAN_ALPHA, lw=0, zorder=0)
        ax.axvline(light_on, linestyle="-", linewidth=1.6, color=LIGHT_LINE_COLOR,
                   zorder=1)
        ax.axvline(light_off, linestyle="--", linewidth=1.0, color=LIGHT_LINE_COLOR,
                   alpha=0.8, zorder=1)

        # ── one trace per fly ──
        theta_ref = None
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
            mask = t <= X_MAX_S + 1e-9
            env, t = env[mask], t[mask]
            if env.size == 0:
                continue
            ax.plot(t, env, linewidth=1.3, color=FLY_COLORS.get(fly_num, "gray"),
                    alpha=0.85, label=f"Fly {fly_num}", zorder=3)
            if theta_ref is None:
                theta_ref = _compute_theta(env, fps, before_s=light_on)

        # ── response threshold (from first available fly's pre-light period) ──
        if theta_ref is not None and math.isfinite(theta_ref):
            ax.axhline(theta_ref, linestyle=":", linewidth=1.0, color="tab:red",
                       alpha=0.7, zorder=2)

        ax.set_ylim(0, FIXED_Y_MAX)
        ax.set_xlim(0, X_MAX_S)
        ax.margins(x=0, y=0.02)

        cond = LIGHT_CONDITION.get(tnum)
        title = f"Trial {tnum}" if tnum else str(trial_label)
        if cond:
            title += f"  —  {cond}"
        title += f"   (light on @ {light_on:.0f} s)"
        ax.set_title(title, loc="left", fontsize=11, weight="bold", pad=2)

        # Light-stimulus check (right of the title band): the video-verified
        # fraction of the commanded window the LED was actually on for this
        # batch's recording (>= 95% displays as "full"). Pulse-duty trailers
        # are excluded: the stride-sampled QC cannot measure a pulsing LED, so
        # its fraction there reflects duty cycle/aliasing, not a malfunction.
        light_fraction = None
        if "pulse" not in str(cond or "").lower():
            light_fraction = _lookup_light_check(str(dataset), str(folder), str(trial_label))
        if light_fraction is not None:
            check_text, check_color = _light_check_annotation(light_fraction)
            ax.text(
                1.0,
                1.0,
                check_text,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=11,
                weight="bold",
                color=check_color,
                clip_on=False,
            )

    axes[-1].set_xlabel("Time (s)", fontsize=11)

    # ── legend ──
    fly_handles = [
        plt.Line2D([0], [0], linewidth=1.3, color=FLY_COLORS.get(fn, "gray"),
                   label=f"Fly {fn}")
        for fn in fly_numbers
    ]
    extra_handles = [
        plt.Line2D([0], [0], linestyle="-", linewidth=1.6, color=LIGHT_LINE_COLOR,
                   label="Light ON"),
        plt.Rectangle((0, 0), 1, 1, alpha=LIGHT_SPAN_ALPHA, color=LIGHT_SPAN_COLOR,
                      label=f"Light window ({LIGHT_ONLY_DURATION_S:.0f} s)"),
        plt.Line2D([0], [0], linestyle=":", linewidth=1.0, color="tab:red",
                   label=r"Response threshold ($\mathrm{med}+3\,\mathrm{MAD}$)"),
    ]
    fig.legend(handles=fly_handles + extra_handles, loc="upper right",
               bbox_to_anchor=(0.985, 0.975), frameon=True, fontsize=9)

    # Inch-based offsets keep the two title lines from colliding on short
    # (few-subplot) figures, where fixed figure-fraction offsets compress.
    fig.tight_layout(rect=[0.04, 0, 1, 1.0 - header_in / fig_h])
    fig.suptitle("Light-Trial Proboscis Extension (post-training)",
                 y=1.0 - 0.30 / fig_h, va="top", fontsize=14, weight="bold")
    fig.text(0.5, 1.0 - 0.80 / fig_h, f"{dataset}  —  {folder}",
             ha="center", va="center", fontsize=12, weight="bold")

    ds_dir = out_dir / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    out_path = ds_dir / f"{folder}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── summary table ────────────────────────────────────────────────────────────

def build_summary(df: pd.DataFrame, dir_val_cols: list[str]) -> pd.DataFrame:
    """Per (fly, fly_number, light trial): onset + peak response in light window."""
    rows = []
    light = df[light_mask(df)]
    for _, r in light.iterrows():
        fps = float(r.get("fps", 40.0)) or 40.0
        if not math.isfinite(fps) or fps <= 0:
            fps = 40.0
        env = _extract_env(r[dir_val_cols].values.flatten(),
                           trace_len=r.get("trace_len"))
        light_on = float(pd.to_numeric(r.get("trial_light_on_s"), errors="coerce"))
        peak_light = peak_pre = np.nan
        if env.size:
            t = np.arange(env.size) / fps
            on_idx = t >= light_on
            off_idx = t < light_on
            if on_idx.any():
                peak_light = float(np.nanmax(env[on_idx]))
            if off_idx.any():
                peak_pre = float(np.nanmax(env[off_idx]))
        rows.append({
            "dataset": r["dataset"], "fly": r["fly"], "fly_number": r["fly_number"],
            "trial_label": r["trial_label"], "trial": _trial_num(r["trial_label"]),
            "light_condition": LIGHT_CONDITION.get(_trial_num(r["trial_label"]), ""),
            "light_on_s": round(light_on, 2),
            "peak_pre_light": round(peak_pre, 2) if math.isfinite(peak_pre) else np.nan,
            "peak_during_after_light": round(peak_light, 2) if math.isfinite(peak_light) else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["dataset", "fly", "fly_number", "trial"]).reset_index(drop=True)
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def _read_wide_for_light(csv_path) -> pd.DataFrame:
    """The wide table minus its frozen experiment folders.

    These figures are per-fly, so a retired folder would otherwise still get its
    own light-trace figure rendered into the results tree.
    """
    from fbpipe.utils.frozen_folders import drop_frozen

    df = pd.read_csv(csv_path)
    before = len(df)
    df = drop_frozen(df)
    if len(df) != before:
        print(f"[FROZEN] Excluding {before - len(df)} row(s) from frozen folders")
    return df


def generate(csv_path, out_dir, datasets=None, skip_datasets=None) -> dict:
    """Render per-fly light-only PER traces, sorted into per-dataset subfolders.

    Args:
        csv_path: wide envelope CSV (``all_envelope_rows_wide_combined_base.csv``).
        out_dir:  root output dir; one subfolder per dataset is created under it.
        datasets: explicit dataset allow-list. When ``None``, every dataset that
                  has light trials is auto-detected and rendered.
        skip_datasets: datasets to drop even if they are in ``datasets`` -- the
                  caller's figure-freeze set. Applied AFTER the allow-list, so
                  a config allow-list can never resurrect a frozen dataset.

    Returns a summary dict: ``n_figures``, ``per_dataset`` (dataset -> count),
    ``datasets`` (rendered, sorted), ``summary_csv``, ``out_dir``.
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    print(f"Reading {csv_path} ...")
    df = _read_wide_for_light(csv_path)

    if datasets:
        wanted = list(datasets)
    else:
        wanted = detect_light_datasets(df)
        print(f"Auto-detected {len(wanted)} light dataset(s): {wanted}")
    skipped = sorted({str(d) for d in (skip_datasets or [])} & set(wanted),
                     key=natural_sort_key)
    if skipped:
        wanted = [d for d in wanted if d not in set(skipped)]
        print(f"  frozen for figures, skipping: {skipped}")
    df = df[df["dataset"].isin(wanted)].copy()
    rendered = sorted(df["dataset"].unique().tolist(), key=natural_sort_key)
    print(f"  {len(df)} rows across datasets: {rendered}")

    dir_val_cols = get_dir_val_cols(df)
    print(f"Found {len(dir_val_cols)} dir_val columns")

    out_dir.mkdir(parents=True, exist_ok=True)

    per_dataset: dict[str, int] = {}
    for dataset in rendered:
        ds_df = df[df["dataset"] == dataset]
        folders = sorted(ds_df["fly"].unique(), key=_date_sort_key)
        print(f"\n=== {dataset}  ({len(folders)} folders) ===")
        for idx, folder in enumerate(folders, 1):
            folder_df = ds_df[ds_df["fly"] == folder]
            out_path = plot_folder(folder_df, dataset, folder, dir_val_cols, out_dir)
            if out_path is None:
                print(f"  [{idx:02d}] {folder}: no light trials — skipped")
            else:
                per_dataset[dataset] = per_dataset.get(dataset, 0) + 1
                n_lt = folder_df[light_mask(folder_df)]["trial_label"].nunique()
                n_fly = folder_df["fly_number"].nunique()
                print(f"  [{idx:02d}] {folder}: {n_lt} light trials, {n_fly} flies -> {out_path.name}")

    summary = build_summary(df, dir_val_cols)
    summary_path = out_dir / "light_trial_summary.csv"
    summary.to_csv(summary_path, index=False)
    made = sum(per_dataset.values())
    print(f"\nSaved {made} figures under {out_dir}")
    print(f"Saved summary ({len(summary)} light-trial rows) -> {summary_path}")
    return {
        "n_figures": made,
        "per_dataset": per_dataset,
        "datasets": [d for d in rendered if d in per_dataset] or rendered,
        "skipped_datasets": skipped,
        "summary_csv": summary_path,
        "out_dir": out_dir,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument(
        "--datasets", nargs="*", default=None,
        help="Datasets to render. Default: auto-detect all datasets with light trials.",
    )
    p.add_argument(
        "--light-check-csv",
        default=str(_REPO / "logs" / "light_stimulus_flags.csv"),
        help=(
            "Light-stimulus QC CSV from the check_light_stimulus step; each "
            "light trial gets a per-panel confirmation note (>= 95%% of the "
            "commanded window on == 'full'). Pass an empty string to disable."
        ),
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if str(args.light_check_csv).strip():
        set_light_check_fractions(load_light_check_fractions(args.light_check_csv))
    generate(args.csv, args.outdir, datasets=args.datasets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
