"""Plot dir_val_* heatmaps per fly and dataset.

This module loads a wide CSV containing direction values (``dir_val_*`` columns)
for odor trials. It produces per-fly odor-specific heatmaps, combined-condition
heatmaps, and aggregated time-series plots with mean ± SEM curves. Optional
normalisation and sorting controls are exposed via the CLI.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import numbers
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


LOGGER = logging.getLogger("plot_dirval_heatmaps")
DEFAULT_OUTDIR = Path("results") / "heatmaps"
DEFAULT_VPERCENTILES = (2.0, 98.0)
DEFAULT_DIRVAL_RANGE = (0.0, 200.0)
DEFAULT_LABELS_TRAIN = (2, 4, 5)
DEFAULT_LABELS_TEST = (1, 3)
DEFAULT_GRID_COLS = 2
DEFAULT_FPS_FALLBACK = 30.0
MAX_HEATMAP_FRAMES = 3600
TESTING_AGGREGATE_RANGE = range(6, 11)
TESTING_LABEL_PATTERN = re.compile(r"testing[\s_-]?(\d+)", re.IGNORECASE)
TRAILING_INT_PATTERN = re.compile(r"(\d+)(?!.*\d)")


def colourbar_params(normalise: str, vlimits: Tuple[float, float]) -> Tuple[str, Optional[List[float]]]:
    """Return colour-bar label and tick positions based on scaling."""

    if normalise == "zscore":
        return "z-score", None

    vmin, vmax = vlimits
    label = f"dir_val ({vmin:.0f}–{vmax:.0f})"

    if not (math.isfinite(vmin) and math.isfinite(vmax)) or vmax <= vmin:
        return label, None

    ticks = np.linspace(vmin, vmax, num=5)
    ticks = [float(round(tick, 2)) for tick in ticks]
    return label, ticks


@dataclass
class HeatmapData:
    """Container for heatmap-ready arrays."""

    matrix: np.ndarray
    time_axis: np.ndarray
    fps_values: List[float]
    trial_indices: List[int]
    truncation: Optional[int]
    mean_length: float


class MissingFpsTracker:
    """Track missing FPS warnings to avoid noisy logs."""

    def __init__(self) -> None:
        self.warned: bool = False

    def maybe_warn(self, logger: logging.Logger, fly: str, dataset: str) -> None:
        if not self.warned:
            logger.warning(
                "Missing/invalid fps detected for dataset=%s fly=%s; using default %.1f",
                dataset,
                fly,
                DEFAULT_FPS_FALLBACK,
            )
            self.warned = True


class ValidationError(RuntimeError):
    """Raised when required input schema is missing."""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Configure CLI parser and return parsed arguments."""

    parser = argparse.ArgumentParser(
        description="Generate dir_val heatmaps and averaged plots per fly and dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", required=True, type=Path, help="Input wide CSV path")
    parser.add_argument("--dataset", type=str, help="Dataset identifier to filter")
    parser.add_argument(
        "--fly",
        action="append",
        dest="flies",
        help="Restrict processing to specific fly identifiers (repeatable)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Output directory for generated figures",
    )
    parser.add_argument(
        "--labels-train",
        nargs="*",
        type=int,
        default=list(DEFAULT_LABELS_TRAIN),
        help="Training odor labels",
    )
    parser.add_argument(
        "--labels-test",
        nargs="*",
        type=int,
        default=list(DEFAULT_LABELS_TEST),
        help="Testing odor labels",
    )
    parser.add_argument(
        "--colprefix",
        type=str,
        default="dir_val_",
        help="Prefix used to identify time-series columns",
    )
    parser.add_argument(
        "--normalize",
        choices=("none", "zscore"),
        default="none",
        help="Per-trial normalisation strategy",
    )
    parser.add_argument(
        "--sort-by",
        choices=("none", "peak", "auc"),
        default="none",
        help="Trial ordering for heatmaps",
    )
    parser.add_argument(
        "--vclip",
        type=str,
        help="Comma-separated vmin,vmax override for heatmap colour scaling",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=DEFAULT_GRID_COLS,
        help="Number of columns in combined dataset figure",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output figure resolution")
    parser.add_argument(
        "--dry-run",
        type=int,
        default=0,
        help="Limit processing to first K flies and first K trials per heatmap",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging verbosity (DEBUG, INFO, WARNING, ...)",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional path to write overall summary JSON",
    )
    args = parser.parse_args(argv)

    if args.vclip:
        try:
            parts = [float(p.strip()) for p in args.vclip.split(",")]
        except ValueError as exc:  # pragma: no cover - guard
            raise argparse.ArgumentTypeError("--vclip must be 'low,high'") from exc
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("--vclip must contain exactly two numbers")
        args.vclip = tuple(parts)
    else:
        args.vclip = None

    return args


def setup_logging(level: str) -> None:
    """Initialise logging configuration."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
    )


def validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Raise ValidationError when required columns are absent."""

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {missing}")


def detect_dirval_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    """Return sorted dir_val columns by numeric suffix."""

    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    matched: List[Tuple[int, str]] = []
    for col in df.columns:
        match = pattern.match(col)
        if match:
            matched.append((int(match.group(1)), col))
    if not matched:
        raise ValidationError(f"No columns detected with prefix '{prefix}'")
    matched.sort(key=lambda item: item[0])
    return [name for _, name in matched]


def canonical_label(label: object) -> str:
    """Normalise label values to readable strings for comparisons."""

    if isinstance(label, numbers.Integral):
        return str(int(label))
    if isinstance(label, float) and float(label).is_integer():
        return str(int(label))

    text = str(label).strip()
    match = TRAILING_INT_PATTERN.search(text)
    if match:
        return match.group(1)
    return text


def filter_by_labels(df: pd.DataFrame, labels: Sequence[object]) -> pd.DataFrame:
    """Return subset of ``df`` where ``trial_label`` matches any of ``labels``."""

    if not labels:
        return df.iloc[0:0]
    canonical_labels = {canonical_label(label) for label in labels}
    mask = df["trial_label"].map(canonical_label).isin(canonical_labels)
    return df[mask]


def testing_label_index(label: object) -> Optional[int]:
    """Return testing label index when value matches ``testing_*`` pattern."""

    match = TESTING_LABEL_PATTERN.match(str(label))
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def extract_timeseries(row: Mapping[str, float], columns: Sequence[str]) -> np.ndarray:
    """Extract a numeric time-series for a single row."""

    data = [row.get(col, np.nan) for col in columns]
    return np.asarray(data, dtype=float)


def normalise_array(values: np.ndarray, mode: str) -> np.ndarray:
    """Normalise a 1D array according to the selected mode."""

    if mode == "none":
        return values
    if mode == "zscore":
        finite = np.isfinite(values)
        if not finite.any():
            return values
        mu = float(np.nanmean(values[finite]))
        sigma = float(np.nanstd(values[finite]))
        if sigma == 0 or math.isnan(sigma):
            return values - mu
        return (values - mu) / sigma
    raise ValueError(f"Unsupported normalisation mode: {mode}")


def sort_trials(matrix: np.ndarray, sort_by: str) -> np.ndarray:
    """Return indices that order the matrix rows as requested."""

    n_trials = matrix.shape[0]
    if sort_by == "none" or n_trials <= 1:
        return np.arange(n_trials)

    finite = np.where(np.isfinite(matrix), matrix, np.nan)
    if sort_by == "peak":
        peak_idx = np.nanargmax(finite, axis=1)
        peak_val = np.nanmax(finite, axis=1)
        order = np.lexsort((-peak_val, peak_idx))
    elif sort_by == "auc":
        auc = np.nansum(finite, axis=1)
        peak_idx = np.nanargmax(finite, axis=1)
        order = np.lexsort((peak_idx, -auc))
    else:
        raise ValueError(f"Unsupported sort option: {sort_by}")
    return order


def compute_vlimits(
    data: np.ndarray, vclip: Optional[Tuple[float, float]], normalise: str
) -> Tuple[float, float]:
    """Compute colour limits based on either fixed values or percentiles."""

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return DEFAULT_DIRVAL_RANGE
    if vclip is not None:
        return float(vclip[0]), float(vclip[1])
    if normalise == "zscore":
        vmin, vmax = np.percentile(finite, DEFAULT_VPERCENTILES)
        if vmin == vmax:
            vmax = vmin + 1e-6
        return float(vmin), float(vmax)
    return DEFAULT_DIRVAL_RANGE


def prepare_heatmap_matrix(
    subset: pd.DataFrame,
    dir_cols: Sequence[str],
    normalise: str,
    sort_by: str,
    fps_tracker: MissingFpsTracker,
    dataset: str,
    fly: str,
    dry_trials: int,
) -> Optional[HeatmapData]:
    """Generate heatmap matrix and associated metadata for a subset of trials."""

    if subset.empty:
        return None

    series_list: List[np.ndarray] = []
    fps_values: List[float] = []
    trial_indices: List[int] = []
    for idx, row in subset.iterrows():
        values = extract_timeseries(row, dir_cols)
        if not np.isfinite(values).any():
            LOGGER.debug(
                "Skipping trial index=%s for dataset=%s fly=%s due to all-NaN timeseries",
                idx,
                dataset,
                fly,
            )
            continue
        values = normalise_array(values, normalise)
        series_list.append(values)
        fps = row.get("fps", np.nan)
        if not isinstance(fps, (int, float)) or not math.isfinite(float(fps)) or float(fps) <= 0:
            fps_tracker.maybe_warn(LOGGER, fly, dataset)
            fps = DEFAULT_FPS_FALLBACK
        fps_values.append(float(fps))
        trial_indices.append(idx)

    if not series_list:
        LOGGER.warning(
            "No valid trials remained after filtering for dataset=%s fly=%s",
            dataset,
            fly,
        )
        return None

    lengths = [len(arr) for arr in series_list]
    min_length = min(lengths)
    if min_length <= 0:
        LOGGER.warning("Encountered zero-length trial for dataset=%s fly=%s", dataset, fly)
        return None
    truncations = [length - min_length for length in lengths]
    if any(t > 0 for t in truncations):
        LOGGER.info(
            "Truncating trials for dataset=%s fly=%s to min length %d",
            dataset,
            fly,
            min_length,
        )

    target_length = min(min_length, MAX_HEATMAP_FRAMES)
    if target_length < min_length:
        LOGGER.info(
            "Limiting trials for dataset=%s fly=%s to first %d frames (requested cap)",
            dataset,
            fly,
            target_length,
        )

    matrix = np.vstack([arr[:target_length] for arr in series_list])
    mean_length = float(np.mean(lengths))

    order_idx = sort_trials(matrix, sort_by)
    matrix = matrix[order_idx]
    fps_values = [fps_values[i] for i in order_idx]
    trial_indices = [trial_indices[i] for i in order_idx]

    if dry_trials > 0:
        matrix = matrix[:dry_trials]
        fps_values = fps_values[:dry_trials]
        trial_indices = trial_indices[:dry_trials]

    fps_avg = float(np.nanmedian(fps_values)) if fps_values else DEFAULT_FPS_FALLBACK
    if np.nanstd(fps_values) > 1e-3:
        LOGGER.debug(
            "Variable fps detected for dataset=%s fly=%s (median %.2f)",
            dataset,
            fly,
            fps_avg,
        )
    time_axis = np.arange(matrix.shape[1]) / fps_avg
    truncated = any(length > matrix.shape[1] for length in lengths)
    return HeatmapData(
        matrix=matrix,
        time_axis=time_axis,
        fps_values=fps_values,
        trial_indices=trial_indices,
        truncation=matrix.shape[1] if truncated else None,
        mean_length=mean_length,
    )


def ensure_directory(path: Path) -> None:
    """Create directory if it does not exist."""

    path.mkdir(parents=True, exist_ok=True)


def plot_heatmap(
    heatmap: HeatmapData,
    title: str,
    vlimits: Tuple[float, float],
    normalise: str,
    cmap: str = "viridis",
) -> plt.Figure:
    """Render a heatmap figure and return the figure object."""

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(
        heatmap.matrix,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vlimits[0],
        vmax=vlimits[1],
        extent=(
            heatmap.time_axis[0],
            heatmap.time_axis[-1] if heatmap.time_axis.size > 1 else heatmap.time_axis[0] + 1,
            0,
            heatmap.matrix.shape[0],
        ),
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trials")
    label, ticks = colourbar_params(normalise, vlimits)
    colourbar = fig.colorbar(im, ax=ax)
    colourbar.set_label(label)
    if ticks is not None:
        colourbar.set_ticks(ticks)
    ax.set_xlim(heatmap.time_axis[0], heatmap.time_axis[-1] if heatmap.time_axis.size > 1 else heatmap.time_axis[0] + 1)
    fig.tight_layout()
    return fig


def compute_mean_sem(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, SEM, and counts across trials for a heatmap matrix."""

    mean = np.nanmean(matrix, axis=0)
    counts = np.sum(np.isfinite(matrix), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sem = np.nanstd(matrix, axis=0)
        sem = np.divide(sem, np.sqrt(np.maximum(counts, 1)), where=counts > 0)
    sem = np.where(counts > 0, sem, np.nan)
    return mean, sem, counts


def plot_mean_sem(heatmap: HeatmapData, title: str) -> plt.Figure:
    """Render mean ± SEM plot for the provided heatmap data."""

    matrix = heatmap.matrix
    mean, sem, _ = compute_mean_sem(matrix)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(heatmap.time_axis, mean, label="Mean")
    if np.any(np.isfinite(sem)):
        lower = mean - sem
        upper = mean + sem
        ax.fill_between(heatmap.time_axis, lower, upper, alpha=0.3, label="±SEM")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, base_path: Path, dpi: int) -> None:
    """Save a figure as PNG and SVG with shared base path."""

    png_path = base_path.with_suffix(".png")
    svg_path = base_path.with_suffix(".svg")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_summary_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write JSON payload to disk."""

    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def build_summary_record(
    dataset: str,
    fly: str,
    label: str,
    n_trials: int,
    truncation: Optional[int],
    vlimits: Tuple[float, float],
    normalise: str,
    sort_by: str,
) -> Mapping[str, object]:
    """Create a summary record for reporting."""

    record: Dict[str, object] = {
        "dataset": dataset,
        "fly": fly,
        "label": label,
        "n_trials": int(n_trials),
        "vmin": float(vlimits[0]),
        "vmax": float(vlimits[1]),
        "normalize": normalise,
        "sort_by": sort_by,
    }
    if truncation is not None:
        record["truncated_length"] = int(truncation)
    return record


def summarise_counts(records: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Convert summary records into a DataFrame for logging."""

    if not records:
        return pd.DataFrame(columns=["dataset", "fly", "label", "n_trials"])
    df = pd.DataFrame(records)
    return df[[col for col in df.columns if col in ("dataset", "fly", "label", "n_trials")]]


def combined_label_groups(
    df: pd.DataFrame,
    labels_train: Sequence[int],
    labels_test: Sequence[int],
) -> Dict[str, Sequence[int]]:
    """Return mapping of combined label names to label sets."""

    groups: Dict[str, Sequence[int]] = {
        "TRAIN-COMBINED": tuple(labels_train),
        "TEST-COMBINED": tuple(labels_test),
    }
    return groups


def dataset_combined_figure(
    dataset: str,
    fly_heatmaps: Mapping[str, Mapping[str, HeatmapData]],
    dpi: int,
    outdir: Path,
    labels_order: Sequence[str],
    grid_cols: int,
    vclip: Optional[Tuple[float, float]],
    normalise: str,
) -> None:
    """Create a dataset-level figure showing combined heatmaps per fly."""

    if not fly_heatmaps:
        LOGGER.info("No heatmaps available for dataset-level figure for %s", dataset)
        return

    available_labels = [label for label in labels_order if any(label in hm for hm in fly_heatmaps.values())]
    if not available_labels:
        LOGGER.info("No combined labels available for dataset=%s", dataset)
        return

    n_flies = len(fly_heatmaps)
    cols = max(grid_cols, len(available_labels))
    rows = n_flies
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)

    for row_idx, (fly, heatmap_dict) in enumerate(sorted(fly_heatmaps.items())):
        for col_idx in range(cols):
            ax = axes[row_idx][col_idx]
            if col_idx >= len(available_labels):
                ax.axis("off")
                continue
            label = available_labels[col_idx]
            heatmap = heatmap_dict.get(label)
            ax.set_title(f"Fly {fly} | {label}")
            if heatmap is None:
                ax.axis("off")
                continue
            vlimits = compute_vlimits(heatmap.matrix, vclip, normalise)
            im = ax.imshow(
                heatmap.matrix,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=vlimits[0],
                vmax=vlimits[1],
                extent=(
                    heatmap.time_axis[0],
                    heatmap.time_axis[-1] if heatmap.time_axis.size > 1 else heatmap.time_axis[0] + 1,
                    0,
                    heatmap.matrix.shape[0],
                ),
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Trials")
            label_text, ticks = colourbar_params(normalise, vlimits)
            colourbar = fig.colorbar(im, ax=ax)
            colourbar.set_label(label_text)
            if ticks is not None:
                colourbar.set_ticks(ticks)

    for row_axes in axes:
        for ax in row_axes:
            if ax.has_data():
                continue
            ax.axis("off")

    fig.suptitle(f"Dataset {dataset} Combined Heatmaps")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    base_path = outdir / dataset / "combined" / "dataset_combined"
    ensure_directory(base_path.parent)
    save_figure(fig, base_path, dpi)


def dataset_allflies_combined(
    dataset: str,
    dataset_df: pd.DataFrame,
    dir_cols: Sequence[str],
    labels_train: Sequence[int],
    labels_test: Sequence[int],
    normalise: str,
    sort_by: str,
    fps_tracker: MissingFpsTracker,
    dpi: int,
    outdir: Path,
    vclip: Optional[Tuple[float, float]],
    dry_trials: int,
) -> Tuple[List[Mapping[str, object]], Dict[str, HeatmapData]]:
    """Build combined heatmaps that aggregate all flies for train/test labels."""

    if dataset_df.empty:
        return []

    combined_dir = outdir / dataset / "combined"
    ensure_directory(combined_dir)

    summary_records: List[Mapping[str, object]] = []
    aggregated_heatmaps: Dict[str, HeatmapData] = {}
    group_map = {
        "TRAIN-COMBINED": tuple(labels_train),
        "TEST-COMBINED": tuple(labels_test),
    }

    for name, labels in group_map.items():
        subset = filter_by_labels(dataset_df, labels)
        if subset.empty:
            continue
        heatmap = prepare_heatmap_matrix(
            subset,
            dir_cols,
            normalise,
            sort_by,
            fps_tracker,
            dataset,
            "ALL-FLIES",
            dry_trials,
        )
        if heatmap is None:
            continue
        aggregated_heatmaps[name] = heatmap
        vlimits = compute_vlimits(heatmap.matrix, vclip, normalise)
        title = f"{dataset} | All Flies | {name} | n={heatmap.matrix.shape[0]}"
        base = combined_dir / f"{name.lower().replace('-', '_')}_all_flies_heatmap"
        fig = plot_heatmap(heatmap, title, vlimits, normalise)
        save_figure(fig, base, dpi)
        avg_fig = plot_mean_sem(heatmap, title + " Mean ± SEM")
        save_figure(avg_fig, combined_dir / f"{name.lower().replace('-', '_')}_all_flies_heatmap_avg", dpi)
        summary_path = base.with_suffix(".json")
        write_summary_json(
            summary_path,
            {
                "dataset": dataset,
                "fly": "ALL-FLIES",
                "label": name,
                "labels": [canonical_label(l) for l in labels],
                "n_trials": int(heatmap.matrix.shape[0]),
                "vlimits": [float(v) for v in vlimits],
                "normalize": normalise,
                "sort_by": sort_by,
                "fps_median": float(np.nanmedian(heatmap.fps_values)),
                "mean_length": float(heatmap.mean_length),
                "truncated_length": int(heatmap.truncation) if heatmap.truncation is not None else None,
            },
        )
        summary_records.append(
            build_summary_record(
                dataset,
                "ALL-FLIES",
                name,
                heatmap.matrix.shape[0],
                heatmap.truncation,
                vlimits,
                normalise,
                sort_by,
            )
        )

    if not aggregated_heatmaps:
        return summary_records, aggregated_heatmaps

    labels = [name for name in ("TRAIN-COMBINED", "TEST-COMBINED") if name in aggregated_heatmaps]
    fig, axes = plt.subplots(1, len(labels), figsize=(len(labels) * 4, 3), squeeze=False)
    for idx, name in enumerate(labels):
        heatmap = aggregated_heatmaps[name]
        ax = axes[0][idx]
        vlimits = compute_vlimits(heatmap.matrix, vclip, normalise)
        im = ax.imshow(
            heatmap.matrix,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=vlimits[0],
            vmax=vlimits[1],
            extent=(
                heatmap.time_axis[0],
                heatmap.time_axis[-1] if heatmap.time_axis.size > 1 else heatmap.time_axis[0] + 1,
                0,
                heatmap.matrix.shape[0],
            ),
        )
        ax.set_title(f"{name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Trials")
        label_text, ticks = colourbar_params(normalise, vlimits)
        colourbar = fig.colorbar(im, ax=ax)
        colourbar.set_label(label_text)
        if ticks is not None:
            colourbar.set_ticks(ticks)

    fig.suptitle(f"Dataset {dataset} | All Flies Combined")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    overview_base = combined_dir / "dataset_all_flies_combined"
    save_figure(fig, overview_base, dpi)

    return summary_records, aggregated_heatmaps


def dataset_testing_label_figure(
    dataset: str,
    dataset_df: pd.DataFrame,
    dir_cols: Sequence[str],
    normalise: str,
    sort_by: str,
    fps_tracker: MissingFpsTracker,
    dpi: int,
    outdir: Path,
    vclip: Optional[Tuple[float, float]],
    dry_trials: int,
) -> Tuple[List[Mapping[str, object]], Dict[str, HeatmapData]]:
    """Generate dataset-level aggregated testing heatmaps and overview figure."""

    if dataset_df.empty:
        LOGGER.info("No rows supplied for dataset=%s aggregated testing figure", dataset)
        return [], {}

    working_df = dataset_df.copy()
    working_df["_testing_index"] = working_df["trial_label"].map(testing_label_index)
    mask = working_df["_testing_index"].isin(TESTING_AGGREGATE_RANGE)
    if not mask.any():
        LOGGER.info("Dataset=%s lacks testing_6-10 labels; skipping aggregated figure", dataset)
        return [], {}

    summary_records: List[Mapping[str, object]] = []
    heatmaps: Dict[str, HeatmapData] = {}
    combined_dir = outdir / dataset / "combined"
    ensure_directory(combined_dir)
    for test_idx in TESTING_AGGREGATE_RANGE:
        label_key = f"testing_{test_idx}"
        label_df = working_df[working_df["_testing_index"] == test_idx]
        if label_df.empty:
            continue
        subset = label_df.drop(columns=["_testing_index"])
        heatmap = prepare_heatmap_matrix(
            subset,
            dir_cols,
            normalise,
            sort_by,
            fps_tracker,
            dataset,
            label_key,
            dry_trials,
        )
        if heatmap is None:
            continue
        heatmaps[label_key] = heatmap
        n_trials = heatmap.matrix.shape[0]
        title = f"{dataset} | {label_key.replace('_', ' ').title()} | n={n_trials}"
        vlimits = compute_vlimits(heatmap.matrix, vclip, normalise)
        fig = plot_heatmap(heatmap, title, vlimits, normalise)
        base_path = combined_dir / f"{label_key}_across_flies_heatmap"
        save_figure(fig, base_path, dpi)
        avg_fig = plot_mean_sem(heatmap, title + " Mean ± SEM")
        save_figure(avg_fig, combined_dir / f"{label_key}_across_flies_heatmap_avg", dpi)
        summary_path = base_path.with_suffix(".json")
        write_summary_json(
            summary_path,
            {
                "dataset": dataset,
                "fly": "ALL-FLIES",
                "label": label_key,
                "n_trials": n_trials,
                "vlimits": [float(v) for v in vlimits],
                "normalize": normalise,
                "sort_by": sort_by,
                "fps_median": float(np.nanmedian(heatmap.fps_values)),
                "mean_length": float(heatmap.mean_length),
                "truncated_length": int(heatmap.truncation) if heatmap.truncation is not None else None,
            },
        )
        summary_records.append(
            build_summary_record(
                dataset,
                "ALL-FLIES",
                label_key,
                n_trials,
                heatmap.truncation,
                vlimits,
                normalise,
                sort_by,
            )
        )

    if not heatmaps:
        LOGGER.info("No aggregated testing heatmaps generated for dataset=%s", dataset)
        return summary_records, heatmaps

    labels = sorted(heatmaps.keys())
    cols = min(3, len(labels))
    rows = math.ceil(len(labels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)

    for idx, label_key in enumerate(labels):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col]
        heatmap = heatmaps[label_key]
        vlimits = compute_vlimits(heatmap.matrix, vclip, normalise)
        im = ax.imshow(
            heatmap.matrix,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=vlimits[0],
            vmax=vlimits[1],
            extent=(
                heatmap.time_axis[0],
                heatmap.time_axis[-1] if heatmap.time_axis.size > 1 else heatmap.time_axis[0] + 1,
                0,
                heatmap.matrix.shape[0],
            ),
        )
        ax.set_title(label_key.replace("_", " ").title())
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Trials")
        label_text, ticks = colourbar_params(normalise, vlimits)
        colourbar = fig.colorbar(im, ax=ax)
        colourbar.set_label(label_text)
        if ticks is not None:
            colourbar.set_ticks(ticks)

    for ax in axes.flatten()[len(labels):]:
        ax.axis("off")

    fig.suptitle(f"Dataset {dataset} Testing Aggregates")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    overview_base = combined_dir / "dataset_testing_overview"
    save_figure(fig, overview_base, dpi)

    return summary_records, heatmaps


def dataset_combined_average_heatmap(
    dataset: str,
    combined_heatmaps: Mapping[str, HeatmapData],
    testing_heatmaps: Mapping[str, HeatmapData],
    outdir: Path,
    dpi: int,
    normalise: str,
    sort_by: str,
    vclip: Optional[Tuple[float, float]],
) -> List[Mapping[str, object]]:
    """Create a combined average heatmap across train/test groupings."""

    sources: List[Tuple[str, HeatmapData]] = []
    for label in ("TRAIN-COMBINED", "TEST-COMBINED"):
        heatmap = combined_heatmaps.get(label)
        if heatmap is not None and heatmap.matrix.size:
            sources.append((label, heatmap))

    for idx in TESTING_AGGREGATE_RANGE:
        label = f"testing_{idx}"
        heatmap = testing_heatmaps.get(label)
        if heatmap is not None and heatmap.matrix.size:
            sources.append((label, heatmap))

    if not sources:
        LOGGER.info(
            "Dataset=%s lacks combined averages; skipping average heatmap", dataset
        )
        return []

    min_length = min(hm.matrix.shape[1] for _, hm in sources if hm.matrix.shape[1] > 0)
    if min_length <= 0:
        LOGGER.info(
            "Dataset=%s combined averages have zero-length matrices; skipping", dataset
        )
        return []

    mean_rows: List[np.ndarray] = []
    labels: List[str] = []
    group_metadata: List[Mapping[str, object]] = []

    for label, heatmap in sources:
        clipped_matrix = heatmap.matrix[:, :min_length]
        with np.errstate(all="ignore"):
            mean_series = np.nanmean(clipped_matrix, axis=0)
        if not np.isfinite(mean_series).any():
            LOGGER.warning(
                "All-NaN mean for dataset=%s label=%s when building combined average",
                dataset,
                label,
            )
            continue
        mean_rows.append(mean_series)
        labels.append(label)
        group_metadata.append(
            {
                "label": label,
                "n_trials": int(heatmap.matrix.shape[0]),
                "source_mean_length": float(heatmap.mean_length),
                "truncated_length": int(min_length),
            }
        )

    if not mean_rows:
        LOGGER.info("No valid averages computed for dataset=%s", dataset)
        return []

    average_matrix = np.vstack(mean_rows)
    reference_axis = sources[0][1].time_axis
    if reference_axis.size >= min_length:
        time_axis = reference_axis[:min_length]
    else:
        step = 1.0 if reference_axis.size == 0 else np.nanmean(np.diff(reference_axis))
        if not math.isfinite(step) or step <= 0:
            step = 1.0
        time_axis = np.linspace(0.0, step * (min_length - 1), num=min_length)

    vlimits = compute_vlimits(average_matrix, vclip, normalise)
    fig, ax = plt.subplots(figsize=(6, max(2, len(labels) * 0.6)))
    extent = (
        float(time_axis[0]),
        float(time_axis[-1]) if time_axis.size > 1 else float(time_axis[0] + 1.0),
        -0.5,
        len(labels) - 0.5,
    )
    im = ax.imshow(
        average_matrix,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=vlimits[0],
        vmax=vlimits[1],
        extent=extent,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Combined groups")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels([label.replace("_", " ").title() for label in labels])
    title = f"{dataset} | Combined Averages"
    ax.set_title(title)
    label_text, ticks = colourbar_params(normalise, vlimits)
    colourbar = fig.colorbar(im, ax=ax)
    colourbar.set_label(label_text)
    if ticks is not None:
        colourbar.set_ticks(ticks)
    fig.tight_layout()

    combined_dir = outdir / dataset / "combined"
    ensure_directory(combined_dir)
    base_path = combined_dir / "dataset_combined_average_heatmap"
    save_figure(fig, base_path, dpi)
    summary_path = base_path.with_suffix(".json")
    write_summary_json(
        summary_path,
        {
            "dataset": dataset,
            "mode": "combined-average",
            "normalize": normalise,
            "sort_by": sort_by,
            "vlimits": [float(v) for v in vlimits],
            "groups": group_metadata,
        },
    )

    summary_records: List[Mapping[str, object]] = []
    for meta in group_metadata:
        summary_records.append(
            build_summary_record(
                dataset,
                "ALL-FLIES",
                f"{meta['label']}-AVERAGE",
                meta["n_trials"],
                meta["truncated_length"],
                vlimits,
                normalise,
                sort_by,
            )
        )

    return summary_records


def dataset_all_flies_mean_overview(
    dataset: str,
    combined_heatmaps: Mapping[str, HeatmapData],
    testing_heatmaps: Mapping[str, HeatmapData],
    outdir: Path,
    dpi: int,
    normalise: str,
    sort_by: str,
    vclip: Optional[Tuple[float, float]],
) -> List[Mapping[str, object]]:
    """Create a multi-panel figure of mean ± SEM traces across all combined groups."""

    order: List[Tuple[str, str, Mapping[str, HeatmapData]]] = [
        ("TEST-COMBINED", "Test Combined", combined_heatmaps),
        ("TRAIN-COMBINED", "Train Combined", combined_heatmaps),
    ]
    for idx in TESTING_AGGREGATE_RANGE:
        key = f"testing_{idx}"
        display = f"Testing {idx}"
        order.append((key, display, testing_heatmaps))

    entries: List[Mapping[str, object]] = []
    for key, display, source in order:
        heatmap = source.get(key) if source else None
        if heatmap is None or heatmap.matrix.size == 0:
            LOGGER.debug(
                "Skipping combined mean overview entry for dataset=%s key=%s due to missing data",
                dataset,
                key,
            )
            continue
        mean, sem, counts = compute_mean_sem(heatmap.matrix)
        entries.append(
            {
                "key": key,
                "display": display,
                "heatmap": heatmap,
                "mean": mean,
                "sem": sem,
                "counts": counts,
                "n_trials": int(heatmap.matrix.shape[0]),
                "vlimits": compute_vlimits(heatmap.matrix, vclip, normalise),
            }
        )

    if not entries:
        LOGGER.info(
            "Dataset=%s lacks data for combined mean overview; skipping figure", dataset
        )
        return []

    fig, axes = plt.subplots(
        len(entries),
        1,
        figsize=(8, max(3, len(entries) * 2.2)),
        sharex=False,
        squeeze=False,
    )

    for ax, entry in zip(axes.flatten(), entries):
        heatmap = entry["heatmap"]
        mean = entry["mean"]
        sem = entry["sem"]
        time_axis = heatmap.time_axis
        ax.plot(time_axis, mean, color="C0", label="Mean")
        if np.any(np.isfinite(sem)):
            lower = mean - sem
            upper = mean + sem
            ax.fill_between(time_axis, lower, upper, alpha=0.3, color="C0", label="±SEM")
        ax.set_title(entry["display"])
        ax.set_ylabel("dir_val")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.2)
    axes.flatten()[-1].set_xlabel("Time (s)")

    fig.suptitle(f"{dataset} | All Flies Mean ± SEM Overview")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    combined_dir = outdir / dataset / "combined"
    ensure_directory(combined_dir)
    base_path = combined_dir / "dataset_all_flies_combined_mean_overview"
    save_figure(fig, base_path, dpi)

    summary_path = base_path.with_suffix(".json")
    write_summary_json(
        summary_path,
        {
            "dataset": dataset,
            "mode": "all-flies-mean-overview",
            "normalize": normalise,
            "sort_by": sort_by,
            "groups": [
                {
                    "label": entry["key"],
                    "display": entry["display"],
                    "n_trials": entry["n_trials"],
                    "value_range": [float(v) for v in entry["vlimits"]],
                }
                for entry in entries
            ],
        },
    )

    summary_records: List[Mapping[str, object]] = []
    for entry in entries:
        summary_records.append(
            build_summary_record(
                dataset,
                "ALL-FLIES",
                f"{entry['key']}-MEAN-OVERVIEW",
                entry["n_trials"],
                entry["heatmap"].truncation,
                entry["vlimits"],
                normalise,
                sort_by,
            )
        )

    return summary_records


def process(
    args: argparse.Namespace,
) -> Mapping[str, object]:
    """Execute main workflow."""

    setup_logging(args.log_level)
    LOGGER.info("Loading CSV from %s", args.csv)
    df = pd.read_csv(args.csv)

    validate_columns(df, ["dataset", "fly", "trial_label"])
    dir_cols = detect_dirval_columns(df, args.colprefix)

    if args.dataset:
        df = df[df["dataset"] == args.dataset]
        LOGGER.info("Filtered dataset to %s rows", len(df))
    if args.flies:
        df = df[df["fly"].isin(args.flies)]
        LOGGER.info("Filtered to specified flies (%d rows)", len(df))

    if df.empty:
        raise ValidationError("No rows remain after filtering; nothing to process")

    labels_train = tuple(args.labels_train)
    labels_test = tuple(args.labels_test)

    fps_tracker = MissingFpsTracker()
    summary_records: List[Mapping[str, object]] = []

    fly_limit = args.dry_run if args.dry_run else None
    processed_flies = 0

    dataset_groups: Dict[str, Dict[str, HeatmapData]] = {}
    processed_dataset_rows: Dict[str, List[pd.DataFrame]] = {}

    for (dataset, fly), fly_df in df.groupby(["dataset", "fly"]):
        if fly_limit is not None and processed_flies >= fly_limit:
            LOGGER.info("Dry-run limit reached (%d flies)", fly_limit)
            break
        processed_flies += 1
        LOGGER.info("Processing dataset=%s fly=%s (%d rows)", dataset, fly, len(fly_df))

        out_base = args.outdir / dataset / str(fly)
        ensure_directory(out_base)

        label_groups: Dict[str, Sequence[int]] = {}
        for label in labels_train:
            label_groups[f"Odor {label}"] = (label,)
        for label in labels_test:
            if f"Odor {label}" not in label_groups:
                label_groups[f"Odor {label}"] = (label,)
        combined_groups = combined_label_groups(fly_df, labels_train, labels_test)

        for group_name, label_set in label_groups.items():
            label = label_set[0]
            label_display = canonical_label(label)
            if label in labels_train or label in labels_test:
                folder = out_base
            else:
                continue
            subset = filter_by_labels(fly_df, label_set)
            heatmap = prepare_heatmap_matrix(
                subset,
                dir_cols,
                args.normalize,
                args.sort_by,
                fps_tracker,
                dataset,
                str(fly),
                args.dry_run if args.dry_run else 0,
            )
            if heatmap is None:
                continue
            n_trials = heatmap.matrix.shape[0]
            title = f"{dataset} | Fly {fly} | Odor {label_display} | n={n_trials}"
            vlimits = compute_vlimits(heatmap.matrix, args.vclip, args.normalize)
            fig = plot_heatmap(heatmap, title, vlimits, args.normalize)
            base_path = folder / f"odor_{label_display}_heatmap"
            save_figure(fig, base_path, args.dpi)
            avg_fig = plot_mean_sem(heatmap, title + " Mean ± SEM")
            save_figure(avg_fig, folder / f"odor_{label_display}_heatmap_avg", args.dpi)
            summary_path = base_path.with_suffix(".json")
            write_summary_json(
                summary_path,
                {
                    "dataset": dataset,
                    "fly": fly,
                    "label": label_display,
                    "n_trials": n_trials,
                    "vlimits": [float(v) for v in vlimits],
                    "normalize": args.normalize,
                    "sort_by": args.sort_by,
                    "fps_median": float(np.nanmedian(heatmap.fps_values)),
                    "mean_length": float(heatmap.mean_length),
                    "truncated_length": int(heatmap.truncation) if heatmap.truncation is not None else None,
                },
            )
            summary_records.append(
                build_summary_record(
                    dataset,
                    str(fly),
                    f"Odor {label_display}",
                    n_trials,
                    heatmap.truncation,
                    vlimits,
                    args.normalize,
                    args.sort_by,
                )
            )

        dataset_groups.setdefault(dataset, {})
        processed_dataset_rows.setdefault(dataset, []).append(fly_df)

        for combined_name, labels in combined_groups.items():
            subset = filter_by_labels(fly_df, labels)
            if subset.empty:
                continue
            heatmap = prepare_heatmap_matrix(
                subset,
                dir_cols,
                args.normalize,
                args.sort_by,
                fps_tracker,
                dataset,
                str(fly),
                args.dry_run if args.dry_run else 0,
            )
            if heatmap is None:
                continue
            dataset_groups[dataset].setdefault(str(fly), {})[combined_name] = heatmap
            n_trials = heatmap.matrix.shape[0]
            title = f"{dataset} | Fly {fly} | {combined_name} | n={n_trials}"
            vlimits = compute_vlimits(heatmap.matrix, args.vclip, args.normalize)
            fig = plot_heatmap(heatmap, title, vlimits, args.normalize)
            base_path = out_base / f"{combined_name.lower().replace('-', '_')}_heatmap"
            save_figure(fig, base_path, args.dpi)
            avg_fig = plot_mean_sem(heatmap, title + " Mean ± SEM")
            save_figure(avg_fig, out_base / f"{combined_name.lower().replace('-', '_')}_heatmap_avg", args.dpi)
            summary_path = base_path.with_suffix(".json")
            write_summary_json(
                summary_path,
                {
                    "dataset": dataset,
                    "fly": fly,
                    "label": combined_name,
                    "labels": [canonical_label(l) for l in labels],
                    "n_trials": n_trials,
                    "vlimits": [float(v) for v in vlimits],
                    "normalize": args.normalize,
                    "sort_by": args.sort_by,
                    "fps_median": float(np.nanmedian(heatmap.fps_values)),
                    "mean_length": float(heatmap.mean_length),
                    "truncated_length": int(heatmap.truncation) if heatmap.truncation is not None else None,
                },
            )
            summary_records.append(
                build_summary_record(dataset, str(fly), combined_name, n_trials, heatmap.truncation, vlimits, args.normalize, args.sort_by)
            )

    for dataset, fly_heatmaps in dataset_groups.items():
        dataset_combined_figure(
            dataset,
            fly_heatmaps,
            args.dpi,
            args.outdir,
            labels_order=("TRAIN-COMBINED", "TEST-COMBINED"),
            grid_cols=args.grid_cols,
            vclip=args.vclip,
            normalise=args.normalize,
        )

    for dataset, frames in processed_dataset_rows.items():
        dataset_df = pd.concat(frames, ignore_index=True)
        allflies_summary, combined_heatmaps = dataset_allflies_combined(
            dataset,
            dataset_df,
            dir_cols,
            labels_train,
            labels_test,
            args.normalize,
            args.sort_by,
            fps_tracker,
            args.dpi,
            args.outdir,
            args.vclip,
            args.dry_run if args.dry_run else 0,
        )
        summary_records.extend(allflies_summary)

        testing_summary, testing_heatmaps = dataset_testing_label_figure(
            dataset,
            dataset_df,
            dir_cols,
            args.normalize,
            args.sort_by,
            fps_tracker,
            args.dpi,
            args.outdir,
            args.vclip,
            args.dry_run if args.dry_run else 0,
        )
        summary_records.extend(testing_summary)

        summary_records.extend(
            dataset_combined_average_heatmap(
                dataset,
                combined_heatmaps,
                testing_heatmaps,
                args.outdir,
                args.dpi,
                args.normalize,
                args.sort_by,
                args.vclip,
            )
        )

        summary_records.extend(
            dataset_all_flies_mean_overview(
                dataset,
                combined_heatmaps,
                testing_heatmaps,
                args.outdir,
                args.dpi,
                args.normalize,
                args.sort_by,
                args.vclip,
            )
        )

    summary_df = summarise_counts(summary_records)
    if not summary_df.empty:
        LOGGER.info("Summary of generated heatmaps:\n%s", summary_df.to_string(index=False))
    else:
        LOGGER.info("No heatmaps were generated.")

    payload = {"records": summary_records}
    if args.summary_json:
        write_summary_json(args.summary_json, payload)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""

    try:
        args = parse_args(argv)
        process(args)
        return 0
    except ValidationError as exc:
        LOGGER.error("Validation error: %s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("Unhandled error: %s", exc)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
