"""Generate dataset-level mean plots of Distance % for testing odors.

Reads the pre-built wide-format CSV (all_envelope_rows_wide.csv) and
optionally filters out flagged flies using flagged-flys-truth.csv.

For each dataset, produces one figure with all odors overlaid and a JSON
sidecar with metadata.

Example
-------
    python scripts/analysis/dataset_means.py \
        --config config/config.yaml \
        --outdir artifacts/dataset_means
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Ensure all plots use Arial to match lab styling.
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
    }
)

# ---------------------------------------------------------------------------
# Repo root setup (so we can import fbpipe and sibling scripts)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
_root_str = str(ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)
_src_str = str(ROOT / "src")
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)

from fbpipe.config import load_settings, resolve_config_path  # noqa: E402
from fbpipe.utils.nanstats import (  # noqa: E402
    count_finite_contributors,
    nan_pad_stack,
    nanmean_sem,
)
from scripts.analysis.envelope_combined import (  # noqa: E402
    _canon_dataset,
    _display_odor,
)

LOGGER = logging.getLogger("dataset_means")

# ---------------------------------------------------------------------------
# Plotting style (consistent with existing repo conventions)
# ---------------------------------------------------------------------------
DPI = 300
FIGWIDTH = 8
MAX_TIME_S = 90.0  # Only plot t=0 to t=90s

# Default paths for the wide CSV and flagged-flies CSV
DEFAULT_WIDE_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/Opto/Combined/all_envelope_rows_wide.csv"
)
DEFAULT_FLAGGED_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/Opto/Combined/flagged-flys-truth.csv"
)


def _configure_logging(verbose: bool) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
    LOGGER.propagate = False


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_figure(fig: plt.Figure, base_path: Path) -> None:
    png = base_path.with_suffix(".png")
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved %s", png)


# ---------------------------------------------------------------------------
# Flagged-fly filtering
# ---------------------------------------------------------------------------

def load_excluded_flies(flagged_csv: Path) -> set[tuple[str, str, int]]:
    """Load flagged-flys-truth.csv and return set of (dataset, fly, fly_number)
    tuples that should be EXCLUDED (state 0 or -1).

    Flies not in the CSV are included. Flies with state=1 are included.
    """
    if not flagged_csv.exists():
        LOGGER.warning("Flagged-flies CSV not found: %s — no filtering applied", flagged_csv)
        return set()

    df = pd.read_csv(flagged_csv)
    # The state column has a verbose name; find it
    state_col = None
    for col in df.columns:
        if "FLY-State" in col or "fly-state" in col.lower():
            state_col = col
            break
    if state_col is None:
        LOGGER.warning("No FLY-State column found in %s — no filtering applied", flagged_csv)
        return set()

    # Exclude flies with state 0 or -1
    excluded = df[df[state_col].isin([0, -1])]
    result = set()
    for _, row in excluded.iterrows():
        result.add((str(row["dataset"]).strip(), str(row["fly"]).strip(), int(row["fly_number"])))

    LOGGER.info(
        "Flagged-fly filter: %d entries in CSV, %d excluded (state 0 or -1), %d kept (state 1)",
        len(df),
        len(result),
        len(df) - len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Core aggregation (from wide CSV)
# ---------------------------------------------------------------------------


def compute_dataset_means(
    wide_df: pd.DataFrame,
    dataset_name: str,
    *,
    excluded_flies: set[tuple[str, str, int]] | None = None,
    fps: float = 40.0,
) -> dict[str, dict]:
    """Aggregate per-odor mean traces for a single dataset from the wide CSV.

    Parameters
    ----------
    wide_df : pd.DataFrame
        The full wide-format DataFrame (all_envelope_rows_wide.csv) already
        filtered to `dataset == dataset_name`.
    dataset_name : str
        Name of the dataset being processed.
    excluded_flies : set, optional
        Set of (dataset, fly, fly_number) tuples to exclude.
    fps : float
        Frames per second (used only for logging).

    Returns a dict keyed by odor label with values::

        {
            "mean": np.ndarray,
            "sem": np.ndarray,
            "n_flies": int,
            "fly_names": list[str],
        }
    """
    if excluded_flies is None:
        excluded_flies = set()

    dataset_canon = _canon_dataset(dataset_name)
    LOGGER.info("Dataset: %s (canon=%s)", dataset_name, dataset_canon)

    # Identify dir_val columns (the trace)
    dir_cols = sorted(
        [c for c in wide_df.columns if c.startswith("dir_val_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    if not dir_cols:
        LOGGER.warning("No dir_val_* columns found")
        return {}

    # Filter out excluded flies
    pre_filter = len(wide_df)
    if excluded_flies:
        mask = wide_df.apply(
            lambda row: (
                str(row["dataset"]).strip(),
                str(row["fly"]).strip(),
                int(row["fly_number"]),
            ) not in excluded_flies,
            axis=1,
        )
        wide_df = wide_df[mask]
    post_filter = len(wide_df)
    if pre_filter != post_filter:
        LOGGER.info(
            "  Filtered: %d → %d rows (%d excluded)",
            pre_filter, post_filter, pre_filter - post_filter,
        )

    if wide_df.empty:
        LOGGER.warning("No data after filtering for %s", dataset_name)
        return {}

    # Group by (fly, fly_number) to identify unique flies,
    # then by trial_label to get odor
    # odor -> list of per-fly mean traces
    odor_fly_traces: dict[str, list[np.ndarray]] = defaultdict(list)
    odor_fly_names: dict[str, list[str]] = defaultdict(list)

    # Group rows by fly identity
    fly_groups = wide_df.groupby(["fly", "fly_number"])

    for (fly_name, fly_num), fly_rows in fly_groups:
        fly_id = f"{fly_name}_fly{fly_num}"
        LOGGER.debug("  Fly: %s", fly_id)

        # Group this fly's trials by odor
        odor_trials: dict[str, list[np.ndarray]] = defaultdict(list)
        for _, row in fly_rows.iterrows():
            trial_label = str(row["trial_label"])
            odor = _display_odor(dataset_canon, trial_label)
            trace = row[dir_cols].to_numpy(dtype=np.float64)
            # Trim trailing NaNs
            finite_mask = np.isfinite(trace)
            if not finite_mask.any():
                continue
            last_finite = np.where(finite_mask)[0][-1]
            trace = trace[: last_finite + 1]
            odor_trials[odor].append(trace)

        # Per-fly mean across trials for each odor
        for odor, traces in odor_trials.items():
            if not traces:
                continue
            stacked = nan_pad_stack(traces)
            with np.errstate(all="ignore"):
                fly_mean = np.nanmean(stacked, axis=0)
            if np.any(np.isfinite(fly_mean)):
                odor_fly_traces[odor].append(fly_mean)
                odor_fly_names[odor].append(fly_id)

    if not odor_fly_traces:
        LOGGER.warning("No usable data found for dataset %s", dataset_name)
        return {}

    results: dict[str, dict] = {}
    for odor in sorted(odor_fly_traces.keys()):
        fly_traces = odor_fly_traces[odor]
        stacked = nan_pad_stack(fly_traces)
        mean, sem = nanmean_sem(stacked)
        n_flies = count_finite_contributors(stacked)
        results[odor] = {
            "mean": mean,
            "sem": sem,
            "n_flies": n_flies,
            "fly_names": odor_fly_names[odor],
        }
        LOGGER.info("  %s: n=%d flies", odor, n_flies)
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_dataset_means(
    results: dict[str, dict],
    *,
    dataset_name: str,
    fps: float,
    odor_on_s: float,
    odor_off_s: float,
) -> Optional[plt.Figure]:
    """Create a single plot with all odors overlaid in different colours."""
    if not results:
        return None

    odors = list(results.keys())
    fig, ax = plt.subplots(figsize=(FIGWIDTH, 5))

    max_frames = int(MAX_TIME_S * fps)

    for idx, odor in enumerate(odors):
        data = results[odor]
        mean = data["mean"][:max_frames]
        n_flies = data["n_flies"]
        n_frames = len(mean)
        time = np.arange(n_frames) / fps
        colour = f"C{idx % 10}"

        ax.plot(time, mean, linewidth=1.3, color=colour, label=f"{odor} (n={n_flies})")

    # Odor window
    ax.axvline(odor_on_s, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(odor_off_s, color="black", linestyle="--", linewidth=0.8)
    ax.axvspan(odor_on_s, odor_off_s, alpha=0.12, color="grey")

    ax.set_xlim(0, MAX_TIME_S)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance %")
    ax.set_title(f"{dataset_name} \u2014 Testing Odors Mean", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# JSON sidecar
# ---------------------------------------------------------------------------


def write_sidecar(
    path: Path,
    *,
    dataset_name: str,
    fps: float,
    odor_on_s: float,
    odor_off_s: float,
    trial_type: str,
    results: dict[str, dict],
) -> None:
    info: dict = {
        "dataset": dataset_name,
        "fps": fps,
        "odor_on_s": odor_on_s,
        "odor_off_s": odor_off_s,
        "trial_type": trial_type,
        "n_definition": "count of flies with at least one finite value in the trace",
        "odors": {},
    }
    for odor, data in results.items():
        info["odors"][odor] = {
            "n_flies": data["n_flies"],
            "fly_names": data["fly_names"],
            "n_timepoints": len(data["mean"]),
        }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    LOGGER.info("Saved sidecar %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to config YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--wide-csv",
        type=Path,
        default=DEFAULT_WIDE_CSV,
        help="Path to all_envelope_rows_wide.csv",
    )
    parser.add_argument(
        "--flagged-csv",
        type=Path,
        default=DEFAULT_FLAGGED_CSV,
        help="Path to flagged-flys-truth.csv (flies with state 0/-1 are excluded)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/home/ramanlab/Documents/cole/Results/Opto/Dataset_means"),
        help="Output directory for plots and sidecars",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override frames per second")
    parser.add_argument("--odor-on-s", type=float, default=None, help="Odor onset in seconds")
    parser.add_argument("--odor-off-s", type=float, default=None, help="Odor offset in seconds")
    parser.add_argument(
        "--trial-type",
        default="testing",
        help="Trial type to aggregate (default: testing)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    return parser.parse_args(argv)


def _read_timing(args: argparse.Namespace) -> tuple[float, float, float]:
    """Return (fps, odor_on_s, odor_off_s) from CLI overrides or config."""
    import yaml

    fps = args.fps
    odor_on = args.odor_on_s
    odor_off = args.odor_off_s

    if fps is not None and odor_on is not None and odor_off is not None:
        return fps, odor_on, odor_off

    config_path = resolve_config_path(args.config)
    cfg_data: dict = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg_data = yaml.safe_load(fh) or {}

    combine = cfg_data.get("analysis", {}).get("combined", {}).get("combine", {})

    if fps is None:
        fps = combine.get("fps_default", cfg_data.get("fps_default", 40.0))
    if odor_on is None:
        odor_on = combine.get("odor_on", cfg_data.get("odor_on_s", 30.0))
    if odor_off is None:
        odor_off = combine.get("odor_off", cfg_data.get("odor_off_s", 60.0))

    return float(fps), float(odor_on), float(odor_off)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser(argv)
    _configure_logging(args.verbose)

    # Load the wide CSV
    wide_csv = args.wide_csv.expanduser().resolve()
    if not wide_csv.exists():
        LOGGER.error("Wide CSV not found: %s", wide_csv)
        sys.exit(1)

    LOGGER.info("Reading %s ...", wide_csv)
    wide_df = pd.read_csv(wide_csv)
    LOGGER.info("Loaded %d rows across %d datasets", len(wide_df), wide_df["dataset"].nunique())

    # Filter by trial type
    if args.trial_type:
        wide_df = wide_df[wide_df["trial_type"] == args.trial_type]
        LOGGER.info("After trial_type='%s' filter: %d rows", args.trial_type, len(wide_df))

    # Load excluded flies
    excluded = load_excluded_flies(args.flagged_csv.expanduser().resolve())

    fps, odor_on_s, odor_off_s = _read_timing(args)
    LOGGER.info("FPS=%.1f  odor_on=%.1fs  odor_off=%.1fs", fps, odor_on_s, odor_off_s)

    outdir = args.outdir.expanduser().resolve()
    _ensure_directory(outdir)

    processed = 0
    for dataset_name in sorted(wide_df["dataset"].unique()):
        LOGGER.info("=== Dataset: %s ===", dataset_name)

        ds_df = wide_df[wide_df["dataset"] == dataset_name].copy()
        results = compute_dataset_means(
            ds_df,
            dataset_name,
            excluded_flies=excluded,
            fps=fps,
        )

        if not results:
            LOGGER.warning("No data for %s — skipping.", dataset_name)
            continue

        base = outdir / f"{dataset_name}_{args.trial_type}_odors_mean"
        fig = plot_dataset_means(
            results,
            dataset_name=dataset_name,
            fps=fps,
            odor_on_s=odor_on_s,
            odor_off_s=odor_off_s,
        )
        if fig is not None:
            _save_figure(fig, base)

        write_sidecar(
            base.with_suffix(".json"),
            dataset_name=dataset_name,
            fps=fps,
            odor_on_s=odor_on_s,
            odor_off_s=odor_off_s,
            trial_type=args.trial_type,
            results=results,
        )
        processed += 1

    LOGGER.info("Done. Processed %d dataset(s).", processed)


if __name__ == "__main__":
    main()
