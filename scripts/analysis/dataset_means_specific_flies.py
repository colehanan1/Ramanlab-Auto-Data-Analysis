"""Dataset-mean plots restricted to the specific flies listed in a
``binary_reactions_<dataset>_unordered.csv`` file (one of the artifacts that
``reaction_matrix_from_spreadsheet.py`` emits next to the PER-reaction matrix
figure).

Generates two kinds of figures per dataset:

1. Overlay mean plot (all odors on one axes), matching the style of
   ``scripts/analysis/dataset_means.py`` but restricted to the binary-reactions
   fly list.

2. Per-odor "average + individuals" plots — one figure per odor showing every
   selected fly's first-presentation trace in a light tint plus the across-fly
   average in a dark tint of the same odor colour. Multi-presentation odors
   use a hard-coded first-presentation trial number (Hexanol -> trial 2,
   Apple Cider Vinegar -> trial 1); single-presentation odors use the only
   trial they appear in.

Defaults to the Hex-Control / Hex-Training reaction-matrix folders under
``Opto-Fly-Figures-OctNov``.
"""
from __future__ import annotations

import argparse
import colorsys
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

plt.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial"]})

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fbpipe.config import resolve_config_path  # noqa: E402
from fbpipe.analysis.traces import baseline_correct, read_wide_table  # noqa: E402
from fbpipe.utils.tables import resolve_existing  # noqa: E402
from fbpipe.utils.nanstats import (  # noqa: E402
    count_finite_contributors,
    nan_pad_stack,
    nanmean_sem,
)
from scripts.analysis.dataset_means import (  # noqa: E402
    ODOR_COLOURS,
    compute_dataset_means,
    plot_dataset_means,
    write_sidecar,
)
from scripts.analysis.envelope_combined import (  # noqa: E402
    _canon_dataset,
    _display_odor,
    _trial_num,
)

LOGGER = logging.getLogger("dataset_means_specific_flies")

DPI = 300
MAX_TIME_S = 90.0

# First-presentation trial number per odor. Multi-presentation odors are
# explicit per user request; single-presentation odors fall back to "the only
# trial number this odor was sent on" (computed per fly at runtime).
FIRST_PRESENTATION_TRIAL = {
    "Hexanol": 2,
    "Apple Cider Vinegar": 1,
}

# Odors to drop from "all odor means" style overlay plots (per-odor and
# Trained-vs-Control figures still get generated, they're just excluded from
# the multi-odor overlays).
EXCLUDE_FROM_OVERLAYS = {"Benzaldehyde"}

DEFAULT_BASE_DIR = Path(
    "/home/ramanlab/Documents/cole/Results/Opto-Fly-Figures-OctNov/Matrix-PER-Reactions-Model"
)
DEFAULT_DATASETS = ("Hex-Control", "Hex-Training")


def _configure_logging(verbose: bool) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
    LOGGER.propagate = False


def _display_name(name: str) -> str:
    """Replace 'Training' with 'Trained' for user-facing labels/titles only."""
    return name.replace("Training", "Trained")


def _adjust_lightness(color: str, factor: float) -> tuple[float, float, float]:
    """Return ``color`` brightened/darkened by ``factor`` (HLS lightness scale).

    factor < 1 -> darker, factor > 1 -> lighter.
    """
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l * factor))
    return colorsys.hls_to_rgb(h, l, s)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _select_wide_csv(cfg: dict) -> Path:
    candidate = (
        cfg.get("analysis", {}).get("dataset_means", {}).get("wide_csv")
        or cfg.get("analysis", {})
        .get("combined", {})
        .get("combined_base", {})
        .get("wide", {})
        .get("output_csv")
    )
    if not candidate:
        raise FileNotFoundError("Could not resolve wide CSV path from config")
    return Path(candidate)


def _load_specific_flies(binary_csv: Path) -> set[tuple[str, int]]:
    """Return ``{(fly, fly_number)}`` from a binary-reactions CSV."""
    df = pd.read_csv(binary_csv)
    needed = {"fly", "fly_number"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{binary_csv} is missing columns: {sorted(missing)}")
    return {
        (str(row["fly"]).strip(), int(row["fly_number"]))
        for _, row in df[["fly", "fly_number"]].drop_duplicates().iterrows()
    }


def _filter_to_flies(
    wide_df: pd.DataFrame, flies: set[tuple[str, int]]
) -> pd.DataFrame:
    if not flies:
        return wide_df.iloc[0:0]
    key = list(
        zip(
            wide_df["fly"].astype(str).str.strip(),
            wide_df["fly_number"].astype(int),
        )
    )
    mask = pd.Series([k in flies for k in key], index=wide_df.index)
    return wide_df[mask]


def _baseline_correct(trace: np.ndarray, baseline_frames: int) -> np.ndarray:
    # Single source of truth: fbpipe.analysis.traces.baseline_correct
    return baseline_correct(trace, baseline_frames)


def _odor_first_presentation_trial(
    odor: str,
    odor_trials_by_fly: dict[tuple[str, int], dict[str, list[int]]],
) -> int | None:
    """Return the canonical first-presentation trial number for an odor.

    Multi-presentation odors use ``FIRST_PRESENTATION_TRIAL``; otherwise the
    lowest trial_num across selected flies is used.
    """
    if odor in FIRST_PRESENTATION_TRIAL:
        return FIRST_PRESENTATION_TRIAL[odor]
    all_trials: set[int] = set()
    for fly_map in odor_trials_by_fly.values():
        for t in fly_map.get(odor, []):
            all_trials.add(t)
    if not all_trials:
        return None
    return min(all_trials)


def _collect_per_fly_first_presentation(
    ds_df: pd.DataFrame,
    dataset_canon: str,
    *,
    baseline_frames: int,
    dir_cols: list[str],
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[tuple[str, int], dict[str, list[int]]],
]:
    """Return per-fly first-presentation traces per odor.

    Output[0] is keyed by odor -> {fly_id: trace}; Output[1] is the
    odor-trials-by-fly map used to resolve fallback first-presentation trial
    numbers for single-presentation odors.
    """
    odor_trials_by_fly: dict[tuple[str, int], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    fly_traces_by_trial: dict[
        tuple[str, int], dict[str, dict[int, np.ndarray]]
    ] = defaultdict(lambda: defaultdict(dict))

    for _, row in ds_df.iterrows():
        trial_label = str(row["trial_label"])
        odor = _display_odor(dataset_canon, trial_label)
        if odor == trial_label:  # unmapped
            continue
        trial_n = _trial_num(trial_label)
        if trial_n < 0:
            continue
        fly_key = (str(row["fly"]).strip(), int(row["fly_number"]))
        trace = row[dir_cols].to_numpy(dtype=np.float64)
        finite_mask = np.isfinite(trace)
        if not finite_mask.any():
            continue
        trace = trace[: np.where(finite_mask)[0][-1] + 1]
        trace = _baseline_correct(trace, baseline_frames)
        odor_trials_by_fly[fly_key][odor].append(trial_n)
        # If a fly has the same odor on the same trial multiple times (it
        # shouldn't), keep the first.
        fly_traces_by_trial[fly_key][odor].setdefault(trial_n, trace)

    # Resolve which trial number to use per odor
    all_odors = set()
    for fly_map in odor_trials_by_fly.values():
        all_odors.update(fly_map.keys())

    per_odor_per_fly: dict[str, dict[str, np.ndarray]] = {}
    for odor in sorted(all_odors):
        target_trial = _odor_first_presentation_trial(odor, odor_trials_by_fly)
        if target_trial is None:
            continue
        per_fly: dict[str, np.ndarray] = {}
        for fly_key, odor_map in fly_traces_by_trial.items():
            trace_map = odor_map.get(odor, {})
            trace = trace_map.get(target_trial)
            if trace is None:
                # Fly didn't have this exact trial — skip
                continue
            fly_id = f"{fly_key[0]}_fly{fly_key[1]}"
            per_fly[fly_id] = trace
        if per_fly:
            per_odor_per_fly[odor] = per_fly
            LOGGER.info(
                "  %s -> first-presentation trial=%d, n_flies=%d",
                odor,
                target_trial,
                len(per_fly),
            )
    return per_odor_per_fly, odor_trials_by_fly


def _round_outward_to_10(value: float, *, upper: bool) -> float:
    """Round outward to the nearest multiple of 10.

    upper=True rounds up (ceil), upper=False rounds down (floor).
    """
    if not np.isfinite(value):
        return value
    if upper:
        return float(np.ceil(value / 10.0) * 10.0)
    return float(np.floor(value / 10.0) * 10.0)


def _shared_ylim_from_per_fly(
    per_fly_by_dataset_odor: dict[str, dict[str, dict[str, np.ndarray]]],
    *,
    fps: float,
) -> tuple[float, float]:
    """Compute a shared ylim across all per-fly traces, padded to multiples of 10."""
    max_frames = int(MAX_TIME_S * fps)
    lo = np.inf
    hi = -np.inf
    for ds_map in per_fly_by_dataset_odor.values():
        for per_fly in ds_map.values():
            for trace in per_fly.values():
                window = trace[:max_frames]
                finite = window[np.isfinite(window)]
                if finite.size == 0:
                    continue
                lo = min(lo, float(finite.min()))
                hi = max(hi, float(finite.max()))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (-10.0, 110.0)
    return _round_outward_to_10(lo, upper=False), _round_outward_to_10(hi, upper=True)


def _plot_odor_avg_with_individuals(
    *,
    odor: str,
    per_fly: dict[str, np.ndarray],
    fps: float,
    odor_on_s: float,
    odor_off_s: float,
    dataset_name: str,
    trial_num: int,
    ylim: tuple[float, float] | None = None,
) -> plt.Figure:
    base_color = ODOR_COLOURS.get(odor, "#444444")
    light = _adjust_lightness(base_color, 1.55)
    dark = _adjust_lightness(base_color, 0.55)

    fig, ax = plt.subplots(figsize=(8, 5))
    max_frames = int(MAX_TIME_S * fps)

    traces = list(per_fly.values())
    stacked = nan_pad_stack(traces)[:, :max_frames]
    n_frames = stacked.shape[1]
    time = np.arange(n_frames) / fps

    for fly_id, trace in per_fly.items():
        t = trace[:max_frames]
        ax.plot(
            np.arange(len(t)) / fps,
            t,
            color=light,
            linewidth=0.9,
            alpha=0.75,
        )

    with np.errstate(all="ignore"):
        mean = np.nanmean(stacked, axis=0)
    n_flies = count_finite_contributors(stacked)
    ax.plot(
        time,
        mean,
        color=dark,
        linewidth=2.2,
        label=f"Mean (n={n_flies})",
    )

    ax.axvline(odor_on_s, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(odor_off_s, color="black", linestyle="--", linewidth=0.8)
    ax.axvspan(odor_on_s, odor_off_s, alpha=0.10, color="grey")
    ax.axhline(0.0, color="0.35", linestyle=":", linewidth=0.8)

    ax.set_xlim(0, MAX_TIME_S)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Max Distance x Angle % (Baseline mean subtracted)")
    ax.set_title(f"{_display_name(dataset_name)} - {odor}", fontsize=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    return fig


def _mean_trace(per_fly: dict[str, np.ndarray], max_frames: int) -> np.ndarray:
    """Return the across-fly mean trace (NaN-padded) clipped to ``max_frames``."""
    stacked = nan_pad_stack(list(per_fly.values()))[:, :max_frames]
    with np.errstate(all="ignore"):
        return np.nanmean(stacked, axis=0)


def _mean_sem_trace(
    per_fly: dict[str, np.ndarray], max_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mean, sem)`` across flies, both clipped to ``max_frames``.

    ``sem`` is computed per time-point as ``nanstd(ddof=1) / sqrt(n_finite)``;
    where fewer than 2 flies contributed (or all NaN) the SEM is set to ``0``
    so callers can safely add/subtract from the mean to draw a shaded band.
    """
    if not per_fly:
        empty = np.full(max_frames, np.nan)
        return empty, np.zeros(max_frames)
    stacked = nan_pad_stack(list(per_fly.values()))[:, :max_frames]
    with np.errstate(all="ignore"):
        mean = np.nanmean(stacked, axis=0)
        n_per_t = np.sum(np.isfinite(stacked), axis=0)
        std = np.nanstd(stacked, axis=0, ddof=1)
    safe_n = np.where(n_per_t > 1, n_per_t, np.nan)
    sem = std / np.sqrt(safe_n)
    sem = np.where(np.isfinite(sem), sem, 0.0)
    return mean, sem


def _shared_ylim_from_means(
    mean_traces: list[np.ndarray], *, fps: float
) -> tuple[float, float]:
    max_frames = int(MAX_TIME_S * fps)
    lo = np.inf
    hi = -np.inf
    for trace in mean_traces:
        window = trace[:max_frames]
        finite = window[np.isfinite(window)]
        if finite.size == 0:
            continue
        lo = min(lo, float(finite.min()))
        hi = max(hi, float(finite.max()))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (-10.0, 30.0)
    return _round_outward_to_10(lo, upper=False), _round_outward_to_10(hi, upper=True)


def _plot_all_odors_means(
    *,
    per_odor_per_fly: dict[str, dict[str, np.ndarray]],
    fps: float,
    odor_on_s: float,
    odor_off_s: float,
    dataset_name: str,
    ylim: tuple[float, float] | None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    max_frames = int(MAX_TIME_S * fps)
    for odor in sorted(per_odor_per_fly.keys()):
        if odor in EXCLUDE_FROM_OVERLAYS:
            continue
        per_fly = per_odor_per_fly[odor]
        mean = _mean_trace(per_fly, max_frames)
        time = np.arange(len(mean)) / fps
        color = ODOR_COLOURS.get(odor, "#444444")
        ax.plot(time, mean, color=color, linewidth=2.0, label=f"{odor} (n={len(per_fly)})")
    ax.axvline(odor_on_s, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(odor_off_s, color="black", linestyle="--", linewidth=0.8)
    ax.axvspan(odor_on_s, odor_off_s, alpha=0.10, color="grey")
    ax.axhline(0.0, color="0.35", linestyle=":", linewidth=0.8)
    ax.set_xlim(0, MAX_TIME_S)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Max Distance x Angle % (Baseline mean subtracted)")
    ax.set_title(f"{_display_name(dataset_name)} - All odor means", fontsize=12)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    return fig


def _plot_training_vs_control_for_odor(
    *,
    odor: str,
    train_per_fly: dict[str, np.ndarray],
    ctrl_per_fly: dict[str, np.ndarray],
    fps: float,
    odor_on_s: float,
    odor_off_s: float,
    ylim: tuple[float, float] | None,
) -> plt.Figure:
    base_color = ODOR_COLOURS.get(odor, "#444444")
    dark = _adjust_lightness(base_color, 0.55)
    light = _adjust_lightness(base_color, 1.45)

    fig, ax = plt.subplots(figsize=(8, 5))
    max_frames = int(MAX_TIME_S * fps)

    train_mean, train_sem = _mean_sem_trace(train_per_fly, max_frames)
    ctrl_mean, ctrl_sem = _mean_sem_trace(ctrl_per_fly, max_frames)
    time_train = np.arange(len(train_mean)) / fps
    time_ctrl = np.arange(len(ctrl_mean)) / fps

    ax.fill_between(
        time_ctrl,
        ctrl_mean - ctrl_sem,
        ctrl_mean + ctrl_sem,
        color=light,
        alpha=0.22,
        linewidth=0,
    )
    ax.plot(
        time_ctrl,
        ctrl_mean,
        color=light,
        linewidth=2.2,
        label=f"Control (n={len(ctrl_per_fly)})",
    )
    ax.fill_between(
        time_train,
        train_mean - train_sem,
        train_mean + train_sem,
        color=dark,
        alpha=0.22,
        linewidth=0,
    )
    ax.plot(
        time_train,
        train_mean,
        color=dark,
        linewidth=2.2,
        label=f"Trained (n={len(train_per_fly)})",
    )

    ax.axvline(odor_on_s, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(odor_off_s, color="black", linestyle="--", linewidth=0.8)
    ax.axvspan(odor_on_s, odor_off_s, alpha=0.10, color="grey")
    ax.axhline(0.0, color="0.35", linestyle=":", linewidth=0.8)
    ax.set_xlim(0, MAX_TIME_S)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Max Distance x Angle % (Baseline mean subtracted)")
    ax.set_title(f"{odor} - Trained vs Control", fontsize=12)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    return fig


# Canonical odor-pair groupings requested for cross-odor trained-vs-control
# comparisons. Match keys are lower-cased, punctuation-stripped strings so the
# tolerant lookup below catches "3-Octanol" / "3-octonol" / "3 octanol" etc.
ODOR_PAIRS_TVC: tuple[tuple[str, str], ...] = (
    ("Hexanol", "Ethyl Butyrate"),
    ("Citral", "Apple Cider Vinegar"),
    ("Linalool", "3-Octonol"),
)

# Additional composite control-only overlays (N odors on one axes). Rendered
# from the control flies only, with mean + SEM band per odor.
ODOR_GROUPS_CONTROL: tuple[tuple[str, ...], ...] = (
    ("Hexanol", "Linalool", "3-Octonol"),
    ("Ethyl Butyrate",),
)


def _norm_odor_key(name: str) -> str:
    """Normalise an odor name for tolerant matching (case + punctuation)."""
    s = str(name).strip().lower()
    s = s.replace("octanol", "octonol")  # tolerate the alternate spelling
    return re.sub(r"[^a-z0-9]+", "", s)


def _resolve_odor_in_dict(
    requested: str,
    available: dict[str, dict[str, np.ndarray]],
) -> str | None:
    """Return the actual key in ``available`` matching ``requested`` loosely."""
    target = _norm_odor_key(requested)
    for key in available:
        if _norm_odor_key(key) == target:
            return key
    return None


def _plot_pair_single_role(
    *,
    pair: tuple[str, str],
    role: str,
    per_odor: dict[str, dict[str, np.ndarray]],
    fps: float,
    odor_on_s: float,
    odor_off_s: float,
    ylim: tuple[float, float] | None,
) -> plt.Figure | None:
    """Render a two-odor average-trace plot for a single condition (role).

    Used to produce per-pair Trained-only / Control-only figures so the two
    odors in the pair can be compared without trained-vs-control overlap.
    Each odor gets one mean line with a SEM band in its canonical colour.
    """
    max_frames = int(MAX_TIME_S * fps)

    resolved: list[tuple[str, dict[str, np.ndarray]]] = []
    for requested in pair:
        key = _resolve_odor_in_dict(requested, per_odor)
        if key is None:
            LOGGER.warning(
                "Odor-pair (%s) plot: missing %s data for '%s' — skipping this odor.",
                role, role, requested,
            )
            continue
        resolved.append((requested, per_odor[key]))
    if not resolved:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for requested, per_fly in resolved:
        base_color = ODOR_COLOURS.get(requested, "#444444")
        # Pull the line a touch darker so it stays readable on the SEM band.
        line_color = _adjust_lightness(base_color, 0.65 if role == "trained" else 0.95)
        mean, sem = _mean_sem_trace(per_fly, max_frames)
        time = np.arange(len(mean)) / fps
        ax.fill_between(
            time, mean - sem, mean + sem,
            color=line_color, alpha=0.22, linewidth=0,
        )
        ax.plot(
            time, mean,
            color=line_color, linewidth=2.2,
            label=f"{requested} (n={len(per_fly)})",
        )

    ax.axvline(odor_on_s, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(odor_off_s, color="black", linestyle="--", linewidth=0.8)
    ax.axvspan(odor_on_s, odor_off_s, alpha=0.10, color="grey")
    ax.axhline(0.0, color="0.35", linestyle=":", linewidth=0.8)
    ax.set_xlim(0, MAX_TIME_S)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Max Distance x Angle % (Baseline mean subtracted)")
    role_title = "Trained" if role == "trained" else "Control"
    ax.set_title(f"{pair[0]} vs {pair[1]} — {role_title} flies", fontsize=12)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.92)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    return fig


def _plot_odors_single_role(
    *,
    odors: Sequence[str],
    role: str,
    per_odor: dict[str, dict[str, np.ndarray]],
    fps: float,
    odor_on_s: float,
    odor_off_s: float,
    ylim: tuple[float, float] | None,
) -> plt.Figure | None:
    """Render an N-odor mean+SEM overlay for a single condition (role)."""
    max_frames = int(MAX_TIME_S * fps)

    resolved: list[tuple[str, dict[str, np.ndarray]]] = []
    for requested in odors:
        key = _resolve_odor_in_dict(requested, per_odor)
        if key is None:
            LOGGER.warning(
                "Odor-group (%s) plot: missing %s data for '%s' — skipping this odor.",
                role, role, requested,
            )
            continue
        resolved.append((requested, per_odor[key]))
    if not resolved:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for requested, per_fly in resolved:
        base_color = ODOR_COLOURS.get(requested, "#444444")
        line_color = _adjust_lightness(base_color, 0.65 if role == "trained" else 0.95)
        mean, sem = _mean_sem_trace(per_fly, max_frames)
        time = np.arange(len(mean)) / fps
        ax.fill_between(
            time, mean - sem, mean + sem,
            color=line_color, alpha=0.22, linewidth=0,
        )
        ax.plot(
            time, mean,
            color=line_color, linewidth=2.2,
            label=f"{requested} (n={len(per_fly)})",
        )

    ax.axvline(odor_on_s, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(odor_off_s, color="black", linestyle="--", linewidth=0.8)
    ax.axvspan(odor_on_s, odor_off_s, alpha=0.10, color="grey")
    ax.axhline(0.0, color="0.35", linestyle=":", linewidth=0.8)
    ax.set_xlim(0, MAX_TIME_S)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Max Distance x Angle % (Baseline mean subtracted)")
    role_title = "Trained" if role == "trained" else "Control"
    odors_title = ", ".join(o for o, _ in resolved)
    ax.set_title(f"{odors_title} — {role_title} flies", fontsize=12)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.92)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    return fig


def _safe_odor_filename(odor: str) -> str:
    return (
        odor.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "-")
    )


def prepare_dataset(
    *,
    dataset_name: str,
    binary_csv: Path,
    wide_df: pd.DataFrame,
    outdir: Path,
    fps: float,
    odor_on_s: float,
    odor_off_s: float,
    overwrite: bool,
) -> dict | None:
    """Collect per-odor per-fly traces + emit the overlay plot.

    Returns a dict with everything needed to render per-odor figures later,
    so the caller can compute a shared y-axis across datasets before drawing.
    """
    LOGGER.info("=== %s ===", dataset_name)
    LOGGER.info("Binary reactions CSV: %s", binary_csv)
    flies = _load_specific_flies(binary_csv)
    LOGGER.info("Specific flies: %d", len(flies))

    ds_df = wide_df[wide_df["dataset"] == dataset_name].copy()
    ds_df = ds_df[ds_df["trial_type"] == "testing"]
    ds_df = _filter_to_flies(ds_df, flies)
    LOGGER.info("Rows after dataset + fly filter: %d", len(ds_df))
    if ds_df.empty:
        LOGGER.warning("No rows left for %s — skipping.", dataset_name)
        return None

    outdir.mkdir(parents=True, exist_ok=True)

    # ---- (1) Overlay mean plot (unaffected by shared per-odor ylim) ----
    results = compute_dataset_means(
        ds_df,
        dataset_name,
        excluded_flies=set(),  # already pre-filtered to selected flies
        fps=fps,
        odor_on_s=odor_on_s,
        subtract_baseline=True,
    )

    overlay_base = outdir / f"{dataset_name}_testing_odors_mean_specific_flies"
    overlay_results = {
        odor: data for odor, data in results.items()
        if odor not in EXCLUDE_FROM_OVERLAYS
    }
    fig = plot_dataset_means(
        overlay_results,
        dataset_name=f"{_display_name(dataset_name)} (specific flies)",
        fps=fps,
        odor_on_s=odor_on_s,
        odor_off_s=odor_off_s,
        baseline_subtracted=True,
        trial_type="testing",
    )
    if fig is not None:
        png = overlay_base.with_suffix(".png")
        if overwrite or not png.exists():
            fig.savefig(png, dpi=DPI, bbox_inches="tight")
            LOGGER.info("Saved %s", png)
        plt.close(fig)
    write_sidecar(
        overlay_base.with_suffix(".json"),
        dataset_name=dataset_name,
        fps=fps,
        odor_on_s=odor_on_s,
        odor_off_s=odor_off_s,
        trial_type="testing",
        results=results,
        baseline_subtracted=True,
    )

    # ---- (2) Collect per-odor per-fly first-presentation traces ----
    dataset_canon = _canon_dataset(dataset_name)
    baseline_frames = max(1, int(round(float(odor_on_s) * float(fps))))
    dir_cols = sorted(
        [c for c in ds_df.columns if c.startswith("dir_val_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    per_odor_per_fly, odor_trials_by_fly = _collect_per_fly_first_presentation(
        ds_df,
        dataset_canon,
        baseline_frames=baseline_frames,
        dir_cols=dir_cols,
    )

    return {
        "dataset_name": dataset_name,
        "binary_csv": binary_csv,
        "outdir": outdir,
        "per_odor_per_fly": per_odor_per_fly,
        "odor_trials_by_fly": odor_trials_by_fly,
        "n_selected_flies": len(flies),
    }


def emit_per_odor_plots(
    *,
    prepared: dict,
    fps: float,
    odor_on_s: float,
    odor_off_s: float,
    overwrite: bool,
    ylim: tuple[float, float] | None,
) -> None:
    dataset_name = prepared["dataset_name"]
    outdir: Path = prepared["outdir"]
    per_odor_per_fly: dict[str, dict[str, np.ndarray]] = prepared["per_odor_per_fly"]
    odor_trials_by_fly = prepared["odor_trials_by_fly"]

    per_odor_meta: dict[str, dict] = {}
    odor_dir = outdir / "per_odor_first_presentation"
    odor_dir.mkdir(parents=True, exist_ok=True)
    for odor, per_fly in per_odor_per_fly.items():
        target_trial = _odor_first_presentation_trial(odor, odor_trials_by_fly)
        assert target_trial is not None
        out_base = odor_dir / f"{dataset_name}_{_safe_odor_filename(odor)}_trial{target_trial}_avg_individuals"
        png = out_base.with_suffix(".png")
        if not overwrite and png.exists():
            LOGGER.info("Skip existing %s", png)
        else:
            fig = _plot_odor_avg_with_individuals(
                odor=odor,
                per_fly=per_fly,
                fps=fps,
                odor_on_s=odor_on_s,
                odor_off_s=odor_off_s,
                dataset_name=dataset_name,
                trial_num=target_trial,
                ylim=ylim,
            )
            fig.savefig(png, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            LOGGER.info("Saved %s", png)
        per_odor_meta[odor] = {
            "trial_num": target_trial,
            "n_flies": len(per_fly),
            "fly_ids": sorted(per_fly.keys()),
        }

    sidecar_path = odor_dir / f"{dataset_name}_per_odor_first_presentation.json"
    with open(sidecar_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "dataset": dataset_name,
                "binary_reactions_csv": str(prepared["binary_csv"]),
                "fps": fps,
                "odor_on_s": odor_on_s,
                "odor_off_s": odor_off_s,
                "n_selected_flies": prepared["n_selected_flies"],
                "first_presentation_overrides": FIRST_PRESENTATION_TRIAL,
                "shared_ylim": list(ylim) if ylim is not None else None,
                "per_odor": per_odor_meta,
            },
            fh,
            indent=2,
        )
    LOGGER.info("Saved %s", sidecar_path)


def build_parser(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to config YAML",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Base directory containing <dataset>/binary_reactions_<dataset>_unordered.csv",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Datasets to process (folder names under base-dir)",
    )
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--odor-on-s", type=float, default=None)
    parser.add_argument("--odor-off-s", type=float, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing figures (default true)",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Skip figures that already exist",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _timing_from_cfg(args, cfg: dict) -> tuple[float, float, float]:
    combine = cfg.get("analysis", {}).get("combined", {}).get("combine", {})
    fps = args.fps if args.fps is not None else combine.get("fps_default", cfg.get("fps_default", 40.0))
    odor_on = (
        args.odor_on_s
        if args.odor_on_s is not None
        else combine.get("odor_on", cfg.get("odor_on_s", 30.0))
    )
    odor_off = (
        args.odor_off_s
        if args.odor_off_s is not None
        else combine.get("odor_off", cfg.get("odor_off_s", 60.0))
    )
    return float(fps), float(odor_on), float(odor_off)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser(argv)
    _configure_logging(args.verbose)
    cfg_path = resolve_config_path(args.config)
    cfg = _load_yaml(cfg_path)
    LOGGER.info("Config: %s", cfg_path)

    wide_csv = _select_wide_csv(cfg)
    if resolve_existing(wide_csv) is None:
        LOGGER.error("Wide CSV not found: %s", wide_csv)
        sys.exit(1)
    LOGGER.info("Wide CSV: %s", wide_csv)

    wide_df = read_wide_table(wide_csv)  # Parquet-preferred (3-4x faster), CSV fallback
    LOGGER.info("Loaded %d rows", len(wide_df))

    fps, odor_on_s, odor_off_s = _timing_from_cfg(args, cfg)
    LOGGER.info("fps=%.1f odor_on=%.1fs odor_off=%.1fs", fps, odor_on_s, odor_off_s)

    prepared_all: dict[str, dict] = {}
    for ds in args.datasets:
        ds_dir = args.base_dir / ds
        binary_csv = ds_dir / f"binary_reactions_{ds}_unordered.csv"
        if not binary_csv.exists():
            LOGGER.error("Missing %s — skipping.", binary_csv)
            continue
        prepared = prepare_dataset(
            dataset_name=ds,
            binary_csv=binary_csv,
            wide_df=wide_df,
            outdir=ds_dir,
            fps=fps,
            odor_on_s=odor_on_s,
            odor_off_s=odor_off_s,
            overwrite=args.overwrite,
        )
        if prepared is not None:
            prepared_all[ds] = prepared

    # Shared ylim across all per-odor plots in all datasets.
    per_fly_by_ds_odor = {
        ds: prep["per_odor_per_fly"] for ds, prep in prepared_all.items()
    }
    shared_ylim = _shared_ylim_from_per_fly(per_fly_by_ds_odor, fps=fps)
    LOGGER.info("Shared per-odor ylim: %s", shared_ylim)

    for ds, prepared in prepared_all.items():
        emit_per_odor_plots(
            prepared=prepared,
            fps=fps,
            odor_on_s=odor_on_s,
            odor_off_s=odor_off_s,
            overwrite=args.overwrite,
            ylim=shared_ylim,
        )

    # ---- Across-dataset comparison + per-dataset all-odors mean overlays ----
    max_frames = int(MAX_TIME_S * fps)
    all_mean_traces: list[np.ndarray] = []
    for prep in prepared_all.values():
        for per_fly in prep["per_odor_per_fly"].values():
            all_mean_traces.append(_mean_trace(per_fly, max_frames))
    mean_ylim = _shared_ylim_from_means(all_mean_traces, fps=fps)
    LOGGER.info("Shared mean-only ylim: %s", mean_ylim)

    # All-odors mean overlay (one figure per dataset).
    for ds, prep in prepared_all.items():
        outdir: Path = prep["outdir"]
        out_png = outdir / f"{ds}_all_odor_means_first_presentation.png"
        if not args.overwrite and out_png.exists():
            LOGGER.info("Skip existing %s", out_png)
            continue
        fig = _plot_all_odors_means(
            per_odor_per_fly=prep["per_odor_per_fly"],
            fps=fps,
            odor_on_s=odor_on_s,
            odor_off_s=odor_off_s,
            dataset_name=ds,
            ylim=mean_ylim,
        )
        fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("Saved %s", out_png)

    # Training vs Control per-odor comparisons. Identify training/control by name.
    def _role(name: str) -> str | None:
        low = name.lower()
        if "training" in low:
            return "training"
        if "control" in low:
            return "control"
        return None

    by_role: dict[str, dict] = {}
    for ds, prep in prepared_all.items():
        role = _role(ds)
        if role is None:
            LOGGER.warning("Cannot classify %s as training/control — skipping comparison.", ds)
            continue
        by_role.setdefault(role, {})[ds] = prep

    if "training" in by_role and "control" in by_role:
        # Pair the first training dataset with the first control dataset.
        train_ds = next(iter(by_role["training"]))
        ctrl_ds = next(iter(by_role["control"]))
        train_prep = by_role["training"][train_ds]
        ctrl_prep = by_role["control"][ctrl_ds]
        cmp_dir = args.base_dir / "Training_vs_Control"
        cmp_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            "Training vs Control: training=%s control=%s -> %s",
            train_ds,
            ctrl_ds,
            cmp_dir,
        )
        common_odors = sorted(
            set(train_prep["per_odor_per_fly"].keys())
            & set(ctrl_prep["per_odor_per_fly"].keys())
        )
        comparison_meta: dict[str, dict] = {}
        for odor in common_odors:
            out_png = cmp_dir / f"{_safe_odor_filename(odor)}_training_vs_control.png"
            if not args.overwrite and out_png.exists():
                LOGGER.info("Skip existing %s", out_png)
                continue
            fig = _plot_training_vs_control_for_odor(
                odor=odor,
                train_per_fly=train_prep["per_odor_per_fly"][odor],
                ctrl_per_fly=ctrl_prep["per_odor_per_fly"][odor],
                fps=fps,
                odor_on_s=odor_on_s,
                odor_off_s=odor_off_s,
                ylim=mean_ylim,
            )
            fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            LOGGER.info("Saved %s", out_png)
            comparison_meta[odor] = {
                "n_training_flies": len(train_prep["per_odor_per_fly"][odor]),
                "n_control_flies": len(ctrl_prep["per_odor_per_fly"][odor]),
            }
        with open(cmp_dir / "training_vs_control.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "training_dataset": train_ds,
                    "control_dataset": ctrl_ds,
                    "fps": fps,
                    "odor_on_s": odor_on_s,
                    "odor_off_s": odor_off_s,
                    "shared_mean_ylim": list(mean_ylim),
                    "per_odor": comparison_meta,
                },
                fh,
                indent=2,
            )

        # ---- Odor-pair plots, one per condition --------------------------------
        # Three odor pairs (Hexanol+Ethyl Butyrate, Citral+Apple Cider Vinegar,
        # Linalool+3-Octonol) each rendered as TWO figures: one for trained
        # flies (both odors overlaid with SEM bands) and one for control flies
        # (same layout). Trained / control are kept on separate axes so each
        # plot stays uncluttered.
        pair_dir = cmp_dir / "odor_pairs"
        pair_dir.mkdir(parents=True, exist_ok=True)
        pair_meta: list[dict] = []
        roles: tuple[tuple[str, dict], ...] = (
            ("trained", train_prep["per_odor_per_fly"]),
            ("control", ctrl_prep["per_odor_per_fly"]),
        )
        for pair in ODOR_PAIRS_TVC:
            pair_slug = f"{_safe_odor_filename(pair[0])}__vs__{_safe_odor_filename(pair[1])}"
            written: list[str] = []
            for role, per_odor in roles:
                out_png = pair_dir / f"{pair_slug}_{role}_only.png"
                if not args.overwrite and out_png.exists():
                    LOGGER.info("Skip existing %s", out_png)
                    written.append(out_png.name)
                    continue
                fig = _plot_pair_single_role(
                    pair=pair,
                    role=role,
                    per_odor=per_odor,
                    fps=fps,
                    odor_on_s=odor_on_s,
                    odor_off_s=odor_off_s,
                    ylim=mean_ylim,
                )
                if fig is None:
                    LOGGER.warning("Pair %s (%s) has no usable data; skipping plot.", pair, role)
                    continue
                fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
                plt.close(fig)
                LOGGER.info("Saved %s", out_png)
                written.append(out_png.name)
            if written:
                pair_meta.append({"pair": list(pair), "pngs": written})
        if pair_meta:
            with open(pair_dir / "odor_pairs.json", "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "training_dataset": train_ds,
                        "control_dataset": ctrl_ds,
                        "fps": fps,
                        "odor_on_s": odor_on_s,
                        "odor_off_s": odor_off_s,
                        "pairs": pair_meta,
                    },
                    fh,
                    indent=2,
                )

        # ---- Control-only composite overlays (N odors on one axes) -----------
        # E.g. Hexanol + Linalool + 3-Octonol, mean + SEM, control flies only.
        group_meta: list[dict] = []
        ctrl_per_odor = ctrl_prep["per_odor_per_fly"]
        for group in ODOR_GROUPS_CONTROL:
            group_slug = "__".join(_safe_odor_filename(o) for o in group)
            out_png = pair_dir / f"{group_slug}_control_only.png"
            if not args.overwrite and out_png.exists():
                LOGGER.info("Skip existing %s", out_png)
                group_meta.append({"odors": list(group), "png": out_png.name})
                continue
            fig = _plot_odors_single_role(
                odors=group,
                role="control",
                per_odor=ctrl_per_odor,
                fps=fps,
                odor_on_s=odor_on_s,
                odor_off_s=odor_off_s,
                ylim=mean_ylim,
            )
            if fig is None:
                LOGGER.warning("Control group %s has no usable data; skipping plot.", group)
                continue
            fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            LOGGER.info("Saved %s", out_png)
            group_meta.append({"odors": list(group), "png": out_png.name})
        if group_meta:
            with open(pair_dir / "odor_groups_control.json", "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "control_dataset": ctrl_ds,
                        "fps": fps,
                        "odor_on_s": odor_on_s,
                        "odor_off_s": odor_off_s,
                        "groups": group_meta,
                    },
                    fh,
                    indent=2,
                )
    else:
        LOGGER.info(
            "Skipping training-vs-control comparison: missing %s",
            sorted({"training", "control"} - set(by_role.keys())),
        )

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
