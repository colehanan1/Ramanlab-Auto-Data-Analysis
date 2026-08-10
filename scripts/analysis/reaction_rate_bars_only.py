"""Re-render the bar half of a reaction-matrix figure as a standalone figure.

``reaction_matrix_from_spreadsheet`` emits one stacked figure per dataset — the
per-fly binary matrix on top, the per-odor PER bar chart below. This script
rebuilds **only the bar chart**, from the ``binary_reactions_<dataset>_<order>.csv``
that was exported alongside it, so the bars can be used on their own.

The bars are computed with the same ``reaction_rate_stats_from_rows`` call the
stacked figure uses (v2 protocol, ``separate_presentations=True``), so every bar,
label, percentage and ``n`` matches the published figure exactly. Nothing is
overwritten: outputs use the ``reaction_rate_bars_`` prefix.

Run::

    /home/ramanlab/anaconda3/bin/python \
        scripts/analysis/reaction_rate_bars_only.py \
        ".../Matrix-PER-Reactions-Model/RandomPanel-24-0.1/binary_reactions_RandomPanel-24-0.1_unordered.csv"
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fbpipe.utils.tables import read_table  # noqa: E402
from scripts.analysis.envelope_visuals import (  # noqa: E402
    _canon_dataset,
    _matrix_title,
    plot_reaction_rate_bars,
    reaction_rate_stats_from_rows,
    set_protocol,
)

_RC_CONTEXT = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
}

_CSV_NAME_RE = re.compile(r"^binary_reactions_(?P<dataset>.+?)_(?P<order>[A-Za-z-]+)\.csv$")


def load_binary_reactions(csv_path: Path) -> pd.DataFrame:
    """Read an exported ``binary_reactions_*.csv`` and normalise its columns."""
    df = read_table(csv_path)
    required = {"fly", "fly_number", "trial_num", "odor_sent", "during_hit"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

    df = df.copy()
    df["fly"] = df["fly"].astype(str).str.strip()
    df["odor_sent"] = df["odor_sent"].astype(str).str.strip()
    df["trial_num"] = pd.to_numeric(df["trial_num"], errors="coerce")
    df["during_hit"] = pd.to_numeric(df["during_hit"], errors="coerce")
    return df.dropna(subset=["trial_num"])


def bar_stats(
    df: pd.DataFrame, dataset: str, *, include_hexanol: bool = True
) -> pd.DataFrame:
    """Per-presentation PER rates, identical to the stacked figure's bottom panel.

    ``odor_sent`` in the exported CSV is already the resolved display name, so the
    trial labels are re-synthesised as ``testing_<n>_<odor>`` — the form
    ``reaction_rate_stats_from_rows`` parses under the v2 protocol. Grouping is by
    (odor, per-fly occurrence), not by trial number: RandomPanel randomises trial
    order, so trial 1 holds a different odor for every fly.
    """
    set_protocol("v2")
    dataset_canon = _canon_dataset(dataset)

    working = df.copy()
    working["trial"] = [
        f"testing_{int(num)}_{odor}"
        for num, odor in zip(working["trial_num"], working["odor_sent"])
    ]
    return reaction_rate_stats_from_rows(
        working,
        dataset_canon,
        include_hexanol=include_hexanol,
        context=f"{dataset_canon} (bars-only)",
        trial_col="trial",
        reaction_col="during_hit",
        separate_presentations=True,
    )


def resolve_genotype(csv_path: Path, dataset: str) -> str | None:
    """Return the genotype folder name for genotype-split datasets, else ``None``.

    Split datasets land in ``<dataset>/<genotype>/binary_reactions_*.csv``; pooled
    ones sit directly in ``<dataset>/``. Any other layout (a scratch directory, an
    explicit ``--out-dir``) carries no genotype.
    """
    parent = csv_path.resolve().parent
    if parent.name and parent.parent.name == dataset:
        return parent.name
    return None


def render_bars_figure(
    stats: pd.DataFrame, *, dataset_canon: str, genotype: str | None, n_flies: int
) -> plt.Figure:
    n_bars = len(stats)
    fig_w = max(10.0, 0.70 * n_bars + 6.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.6))

    heading = _matrix_title(dataset_canon)
    if genotype:
        heading = f"{heading} — {genotype}"
    # One stacked title: the matrix figure's heading over its bar-panel title,
    # so the standalone bars still say which dataset and how many flies.
    plot_reaction_rate_bars(
        ax, stats, title=f"{heading} ({n_flies} Flies)\nReaction Rates by Odor"
    )
    fig.tight_layout()
    return fig


def make_bars(
    *,
    csv_path: Path,
    out_dir: Path | None = None,
    include_hexanol: bool = True,
    formats: Sequence[str] = ("png", "pdf"),
) -> list[Path]:
    """Write the standalone bar figure for one ``binary_reactions_*.csv``."""
    csv_path = Path(csv_path)
    out_dir = Path(out_dir) if out_dir is not None else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    name_match = _CSV_NAME_RE.match(csv_path.name)
    if name_match is None:
        raise ValueError(
            f"Cannot parse dataset/order from {csv_path.name!r}; "
            "expected binary_reactions_<dataset>_<order>.csv"
        )
    dataset = name_match.group("dataset")
    order = name_match.group("order")

    df = load_binary_reactions(csv_path)
    dataset_canon = _canon_dataset(dataset)
    genotype = resolve_genotype(csv_path, dataset)
    n_flies = len(df[["fly", "fly_number"]].drop_duplicates())

    stats = bar_stats(df, dataset, include_hexanol=include_hexanol)

    stem = f"reaction_rate_bars_{dataset}_{order}"
    if genotype:
        stem = f"{stem}_{genotype.replace(' ', '_')}"

    written: list[Path] = []
    with plt.rc_context(_RC_CONTEXT):
        fig = render_bars_figure(
            stats, dataset_canon=dataset_canon, genotype=genotype, n_flies=n_flies
        )
        for suffix in formats:
            path = out_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            written.append(path)
        plt.close(fig)

    csv_out = out_dir / f"{stem}.csv"
    stats.to_csv(csv_out, index=False, float_format="%.4f")
    written.append(csv_out)

    label = f"{dataset}{f' [{genotype}]' if genotype else ''}"
    print(f"[INFO] {label}: {len(stats)} bars, {n_flies} flies")
    for path in written:
        print(f"[INFO] wrote {path}")
    return written


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_paths",
        type=Path,
        nargs="+",
        help="Exported binary_reactions_<dataset>_<order>.csv files to re-render.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: alongside each CSV).",
    )
    parser.add_argument(
        "--exclude-hexanol",
        action="store_true",
        help="Drop Hexanol bars.",
    )
    parser.add_argument(
        "--format",
        action="append",
        dest="formats",
        default=None,
        help="Output format, repeatable. Default: png and pdf.",
    )
    args = parser.parse_args(argv)

    for csv_path in args.csv_paths:
        make_bars(
            csv_path=csv_path,
            out_dir=args.out_dir,
            include_hexanol=not args.exclude_hexanol,
            formats=tuple(args.formats) if args.formats else ("png", "pdf"),
        )


if __name__ == "__main__":
    main()
