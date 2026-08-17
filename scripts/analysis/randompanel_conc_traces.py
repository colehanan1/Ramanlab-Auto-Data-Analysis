"""Concentration-series mean traces, one figure per odor.

For datasets with no control arm -- the RandomPanel panels, whose
``RandomPanel-Control-24-10`` carries zero rows -- the trained-vs-control
figure cannot be drawn. What *is* comparable is the same odor at the three
delivered concentrations, which live in three separate datasets
(``RandomPanel-Training-24-10`` / ``RandomPanel-24-1`` / ``RandomPanel-24-0.1``).

This driver overlays those means on one axes per odor. Concentration is encoded
twice, deliberately: shade of the odor's own palette colour (darkest = highest)
*and* dash pattern (solid -> dashed -> dotted, high to low). Either channel
alone would rank the lines, so the figure survives greyscale printing and
colour-vision deficiency without a legend lookup.

Both presentations of an odor are pooled per fly -- as
``randompanel_conc_comparison`` pools them for its bars -- so an odor shown
twice yields three lines, not six.

Usage::

    python scripts/analysis/randompanel_conc_traces.py \
        --wide-csv /home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/all_envelope_rows_wide_combined_base.parquet \
        --dataset RandomPanel-Training-24-10=10 \
        --dataset RandomPanel-24-1=1 \
        --dataset RandomPanel-24-0.1=0.1 \
        --config config/config_new.yaml \
        --out-dir /home/ramanlab/Documents/cole/Results/New-Opto-Fly-Figures/RandomPanel
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fbpipe.analysis.traces import read_wide_table  # noqa: E402
from fbpipe.utils.nanstats import nan_pad_stack  # noqa: E402
from scripts.analysis.dataset_means_specific_flies import (  # noqa: E402
    DPI,
    MAX_TIME_S,
    Y_LABEL,
    _adjust_lightness,
    _mean_sem_trace,
    _mean_trace,
    _safe_odor_filename,
    _shared_ylim_from_means,
    _trace_odor_color,
)
from scripts.analysis.dataset_mean_traces_tvc import (  # noqa: E402
    _PRESENTATION_SUFFIX,
    _apply_config_remap,
    _prepare,
    base_odor_key,
    filter_by_genotype,
    prune_stale_figures,
)
from scripts.analysis.envelope_visuals import (  # noqa: E402
    _normalise_fly_columns,
    compute_non_reactive_flags,
    set_protocol,
)

LOGGER = logging.getLogger("randompanel_conc_traces")

# High -> low. A fourth series would take "-." and then repeat; the shade ramp
# keeps ranking it either way.
LINESTYLES = ("-", "--", ":", "-.")

# HLS lightness factors spanning "clearly darker than the palette colour" to
# "clearly lighter", assigned high -> low concentration. The light end stops
# well short of white: the lowest dose still has to read as a coloured line
# against the page, and its SEM band sits paler still.
_DARKEST, _LIGHTEST = 0.62, 1.32


@dataclass(frozen=True)
class ConcStyle:
    color: tuple[float, float, float] | str
    linestyle: str


@dataclass(frozen=True)
class ConcSeries:
    conc: float
    per_fly: dict[str, np.ndarray]
    color: tuple[float, float, float] | str
    linestyle: str


def conc_series_styles(
    concentrations: Sequence[float], base_color: str
) -> dict[float, ConcStyle]:
    """``{conc: ConcStyle}`` ordered high -> low.

    Insertion order is the draw order, so the darkest, solid, highest-dose line
    is plotted last and sits on top of the paler ones.
    """
    ordered = sorted({float(c) for c in concentrations}, reverse=True)
    if not ordered:
        return {}
    if len(ordered) == 1:
        factors = [_DARKEST]
    else:
        factors = np.linspace(_DARKEST, _LIGHTEST, len(ordered))
    return {
        conc: ConcStyle(
            color=_adjust_lightness(base_color, float(factor)),
            linestyle=LINESTYLES[i % len(LINESTYLES)],
        )
        for i, (conc, factor) in enumerate(zip(ordered, factors))
    }


def pool_presentations(
    per_key: dict[str, dict[str, np.ndarray]]
) -> dict[str, dict[str, np.ndarray]]:
    """Collapse ``"Hexanol 1"``/``"Hexanol 2"`` into one trace per fly.

    Each fly's exposures are averaged element-wise (NaN-padded, so a short
    trial does not truncate the other exposure). Odors that were presented once
    keep their single trace unchanged.
    """
    grouped: dict[str, dict[str, list[np.ndarray]]] = {}
    for key, per_fly in per_key.items():
        odor = _PRESENTATION_SUFFIX.sub("", str(key).strip())
        bucket = grouped.setdefault(odor, {})
        for fly_id, trace in per_fly.items():
            bucket.setdefault(fly_id, []).append(np.asarray(trace, dtype=np.float64))

    pooled: dict[str, dict[str, np.ndarray]] = {}
    for odor, per_fly_traces in grouped.items():
        pooled[odor] = {}
        for fly_id, traces in per_fly_traces.items():
            if len(traces) == 1:
                pooled[odor][fly_id] = traces[0]
                continue
            with np.errstate(all="ignore"):
                pooled[odor][fly_id] = np.nanmean(nan_pad_stack(traces), axis=0)
    return pooled


def plot_conc_series_for_odor(
    *,
    odor: str,
    series: Sequence[ConcSeries],
    fps: float,
    odor_on_s: float,
    odor_off_s: float,
    ylim: tuple[float, float] | None,
    title: str | None = None,
) -> plt.Figure:
    """One odor, one axes, one mean+SEM line per concentration."""
    fig, ax = plt.subplots(figsize=(8, 5))
    max_frames = int(MAX_TIME_S * fps)

    for entry in series:
        mean, sem = _mean_sem_trace(entry.per_fly, max_frames)
        time = np.arange(len(mean)) / fps
        ax.fill_between(
            time, mean - sem, mean + sem, color=entry.color, alpha=0.16, linewidth=0
        )
        ax.plot(
            time,
            mean,
            color=entry.color,
            linestyle=entry.linestyle,
            linewidth=2.2,
            label=f"{entry.conc:g}% (n={len(entry.per_fly)})",
        )

    ax.axvline(odor_on_s, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(odor_off_s, color="black", linestyle="--", linewidth=0.8)
    ax.axvspan(odor_on_s, odor_off_s, alpha=0.10, color="grey")
    ax.axhline(0.0, color="0.35", linestyle=":", linewidth=0.8)
    ax.set_xlim(0, MAX_TIME_S)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(Y_LABEL)
    ax.set_title(title if title is not None else f"{odor} - Concentration series",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    return fig


def _parse_dataset_spec(spec: str) -> tuple[str, float]:
    name, _, conc = str(spec).partition("=")
    if not name.strip() or not conc.strip():
        raise SystemExit(f"--dataset expects NAME=CONC, got {spec!r}")
    try:
        return name.strip(), float(conc)
    except ValueError:
        raise SystemExit(f"--dataset concentration must be numeric, got {spec!r}") from None


def build_parser(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--wide-csv", type=Path, required=True,
                   help="all_envelope_rows_wide_combined_base CSV or Parquet.")
    p.add_argument("--dataset", action="append", default=[], required=True,
                   metavar="NAME=CONC",
                   help="Dataset and its delivered concentration (repeatable), "
                        "e.g. RandomPanel-24-1=1")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--config", type=str, default="",
                   help="Pipeline config YAML; loads dataset_overrides.odor_remap.")
    p.add_argument("--flagged-flies-csv", type=str, default="",
                   help="flagged-flys-truth CSV (FLY-State != 1 excluded).")
    p.add_argument("--fps", type=float, default=40.0)
    p.add_argument("--odor-on-s", type=float, default=30.0)
    p.add_argument("--odor-off-s", type=float, default=60.0)
    p.add_argument("--genotype", action="append", default=[], metavar="FLY_TYPE",
                   help="Keep only flies of this canonical fly_type "
                        "(repeatable), e.g. GR5a-Old. The 10%% panel pools "
                        "GR5a-GCaMP8 with GR5a-Old unless this is set.")
    p.add_argument("--protocol", default="v2", choices=["v2", "legacy"])
    p.add_argument("--overwrite", action="store_true", default=True)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    set_protocol(args.protocol)
    if args.config:
        _apply_config_remap(args.config)

    conc_by_dataset = dict(_parse_dataset_spec(spec) for spec in args.dataset)

    wide_df = read_wide_table(args.wide_csv)
    LOGGER.info("Loaded %d rows from %s", len(wide_df), args.wide_csv)
    wide_df = _normalise_fly_columns(wide_df)
    wide_df = filter_by_genotype(wide_df, args.genotype)

    if args.flagged_flies_csv:
        flagged = compute_non_reactive_flags(
            wide_df, flagged_flies_csv=args.flagged_flies_csv
        )
        if flagged.any():
            wide_df = wide_df.loc[~flagged].copy()

    # Pooled per-fly traces for every (dataset, odor).
    by_dataset: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for dataset in conc_by_dataset:
        LOGGER.info("=== %s (%g%%)", dataset, conc_by_dataset[dataset])
        per_key, _rows, _df = _prepare(
            wide_df, dataset, fps=args.fps, odor_on_s=args.odor_on_s
        )
        pooled = pool_presentations(per_key)
        if not pooled:
            # A dataset that contributes nothing would silently drop a whole
            # concentration from every figure -- that is a data problem, not a
            # figure to quietly render with one line fewer.
            raise SystemExit(f"No usable traces for {dataset}")
        for odor in sorted(pooled):
            LOGGER.info("  %-28s n=%d", odor, len(pooled[odor]))
        by_dataset[dataset] = pooled

    max_frames = int(MAX_TIME_S * args.fps)
    shared_ylim = _shared_ylim_from_means(
        [
            _mean_trace(per_fly, max_frames)
            for pooled in by_dataset.values()
            for per_fly in pooled.values()
        ],
        fps=args.fps,
    )
    LOGGER.info("Shared mean ylim: %s", shared_ylim)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_odors = sorted({odor for pooled in by_dataset.values() for odor in pooled})
    per_odor_meta: dict[str, dict[str, int]] = {}
    skipped: list[str] = []
    written: list[Path] = []

    for odor in all_odors:
        present = {
            conc_by_dataset[ds]: pooled[odor]
            for ds, pooled in by_dataset.items()
            if pooled.get(odor)
        }
        if len(present) < 2:
            # One line is not a concentration comparison, and drawn alone it
            # would read as one.
            LOGGER.warning(
                "Skipping %s — present in %d of %d concentrations",
                odor, len(present), len(by_dataset),
            )
            skipped.append(odor)
            continue

        styles = conc_series_styles(list(present), _trace_odor_color(base_odor_key(odor)))
        series = [
            ConcSeries(
                conc=conc,
                per_fly=present[conc],
                color=styles[conc].color,
                linestyle=styles[conc].linestyle,
            )
            for conc in styles
        ]
        fig = plot_conc_series_for_odor(
            odor=odor,
            series=series,
            fps=args.fps,
            odor_on_s=args.odor_on_s,
            odor_off_s=args.odor_off_s,
            ylim=shared_ylim,
        )
        out_png = args.out_dir / f"{_safe_odor_filename(odor)}_conc_series.png"
        written.append(out_png)
        if args.overwrite or not out_png.exists():
            fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
            LOGGER.info("Saved %s", out_png)
        plt.close(fig)
        per_odor_meta[odor] = {f"{s.conc:g}": len(s.per_fly) for s in series}

    prune_stale_figures(args.out_dir, written, pattern="*_conc_series.png")

    sidecar = {
        "fps": args.fps,
        "odor_on_s": args.odor_on_s,
        "odor_off_s": args.odor_off_s,
        "shared_mean_ylim": list(shared_ylim),
        "flagged_flies_csv": args.flagged_flies_csv,
        "genotypes": list(args.genotype),
        "datasets": conc_by_dataset,
        "per_odor": per_odor_meta,
        "skipped_odors": skipped,
    }
    sidecar_path = args.out_dir / "conc_series.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    LOGGER.info("Saved %s", sidecar_path)


if __name__ == "__main__":
    main()
