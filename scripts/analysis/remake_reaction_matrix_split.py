"""Remake a reaction-matrix figure as two standalone plots, minus excluded odors.

The pipeline's ``reaction_matrix_from_spreadsheet`` emits one stacked figure —
the per-fly binary matrix on top, the per-odor PER bar chart below. This script
re-renders that same content from the already-exported
``binary_reactions_<dataset>_<order>.csv`` as **two separate figures**, with
Benzaldehyde (or any ``--exclude`` odor) dropped from both.

Nothing is overwritten: every output carries a ``_copy`` suffix.

Run::

    /home/ramanlab/anaconda3/bin/python \
        scripts/analysis/remake_reaction_matrix_split.py \
        --csv-path .../AIR-Training/binary_reactions_AIR-Training_unordered.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from fbpipe.utils.tables import read_table  # noqa: E402
from scripts.analysis.envelope_visuals import (  # noqa: E402
    _canon_dataset,
    _fly_row_label,
    _fly_sort_key,
    _matrix_title,
    _style_trained_xticks,
    _trained_label,
    plot_reaction_rate_bars,
)

DEFAULT_EXCLUDE: frozenset[str] = frozenset({"Benzaldehyde"})

_RC_CONTEXT = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
}

_CSV_NAME_RE = re.compile(r"^binary_reactions_(?P<dataset>.+?)_(?P<order>[A-Za-z-]+)\.csv$")


def _is_excluded(odor: object, exclude: Iterable[str]) -> bool:
    text = str(odor).strip().casefold()
    return any(text == str(item).strip().casefold() for item in exclude)


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
    return df


def _drop_excluded(df: pd.DataFrame, exclude: Iterable[str]) -> pd.DataFrame:
    mask = df["odor_sent"].apply(lambda odor: _is_excluded(odor, exclude))
    kept = df.loc[~mask].copy()
    if kept.empty:
        raise RuntimeError("Every row was excluded; nothing left to plot.")
    return kept


def _presentation_order(df: pd.DataFrame) -> list[tuple[float, str]]:
    """Distinct (trial_num, odor) presentations in trial order."""
    pairs = df[["trial_num", "odor_sent"]].drop_duplicates()
    pairs = pairs.sort_values(["trial_num", "odor_sent"], kind="mergesort")
    return [(row.trial_num, row.odor_sent) for row in pairs.itertuples(index=False)]


def build_matrix(
    df: pd.DataFrame, *, exclude: Iterable[str] = DEFAULT_EXCLUDE
) -> tuple[np.ndarray, list[str], list[tuple[str, object]]]:
    """Return the per-fly binary matrix, its column labels and its row keys."""
    kept = _drop_excluded(df, exclude)

    fly_pairs = [
        (row.fly, row.fly_number)
        for row in kept[["fly", "fly_number"]].drop_duplicates().itertuples(index=False)
    ]
    fly_pairs.sort(key=lambda pair: _fly_sort_key(*pair))

    presentations = _presentation_order(kept)
    labels = [odor for _, odor in presentations]

    row_map = {pair: idx for idx, pair in enumerate(fly_pairs)}
    col_map = {pres: idx for idx, pres in enumerate(presentations)}

    matrix = np.full((len(fly_pairs), len(presentations)), np.nan, dtype=float)
    for row in kept.itertuples(index=False):
        i = row_map.get((row.fly, row.fly_number))
        j = col_map.get((row.trial_num, row.odor_sent))
        if i is None or j is None or pd.isna(row.during_hit):
            continue
        matrix[i, j] = float(row.during_hit)

    return matrix, labels, fly_pairs


def rate_stats(
    df: pd.DataFrame,
    *,
    trained_label: str,
    exclude: Iterable[str] = DEFAULT_EXCLUDE,
) -> pd.DataFrame:
    """Per-presentation PER rates, one row per (trial_num, odor) in trial order."""
    kept = _drop_excluded(df, exclude)
    kept = kept.assign(reaction_flag=kept["during_hit"].fillna(0).astype(int))

    stats = (
        kept.groupby(["trial_num", "odor_sent"], dropna=False)["reaction_flag"]
        .agg(num_reactions="sum", num_trials="size")
        .reset_index()
        .rename(columns={"odor_sent": "odor"})
    )
    stats["rate"] = np.where(
        stats["num_trials"] > 0, stats["num_reactions"] / stats["num_trials"], 0.0
    )
    stats = stats.sort_values(["trial_num", "odor"], kind="mergesort").reset_index(drop=True)
    stats["is_trained"] = (
        stats["odor"].astype(str).str.casefold().str.startswith(str(trained_label).casefold())
    )
    return stats


def render_matrix_figure(
    matrix: np.ndarray,
    labels: Sequence[str],
    *,
    dataset_canon: str,
    trained_label: str,
    n_flies: int,
) -> plt.Figure:
    cmap = ListedColormap(["white", "black"])
    cmap.set_bad(color="0.7")
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    n_trials = len(labels)
    fig_w = max(10.0, 0.70 * n_trials + 6.0)
    fig_h = max(4.5, n_flies * 0.26 + 2.8)
    xtick_fs = 9 if n_trials <= 10 else (8 if n_trials <= 16 else 7)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_title(_matrix_title(dataset_canon), fontsize=14, weight="bold")
    _style_trained_xticks(ax, list(labels), trained_label, xtick_fs)
    ax.set_yticks([])
    ax.set_ylabel(f"{n_flies} Flies", fontsize=11)
    return fig


def render_bars_figure(stats: pd.DataFrame) -> plt.Figure:
    n_bars = len(stats)
    fig_w = max(10.0, 0.70 * n_bars + 6.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.4))
    plot_reaction_rate_bars(
        ax,
        stats,
        title="Reaction Rates by Odor",
        ylabel="Average PER Response %",
        xlabel=None,
    )
    return fig


def remake(
    *,
    csv_path: Path,
    out_dir: Path | None = None,
    latency_sec: float = 2.15,
    after_window_sec: float = 30.0,
    exclude: Iterable[str] = DEFAULT_EXCLUDE,
) -> list[Path]:
    """Write the split matrix / bar figures (plus a row key) as ``_copy`` files."""
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
    trained = _trained_label(dataset_canon)

    matrix, labels, fly_pairs = build_matrix(df, exclude=exclude)
    stats = rate_stats(df, trained_label=trained, exclude=exclude)

    stem = f"{dataset}_{int(after_window_sec)}_latency_{latency_sec:.3f}s_{order}_copy"
    matrix_png = out_dir / f"reaction_matrix_{stem}.png"
    bars_png = out_dir / f"reaction_rates_{stem}.png"
    row_key_txt = out_dir / f"row_key_{dataset}_{int(after_window_sec)}_{order}_copy.txt"

    with plt.rc_context(_RC_CONTEXT):
        fig = render_matrix_figure(
            matrix,
            labels,
            dataset_canon=dataset_canon,
            trained_label=trained,
            n_flies=len(fly_pairs),
        )
        fig.savefig(matrix_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig = render_bars_figure(stats)
        fig.savefig(bars_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

    with row_key_txt.open("w", encoding="utf-8") as fh:
        for idx, (fly, fly_number) in enumerate(fly_pairs):
            fh.write(f"Row {idx}: {_fly_row_label(fly, fly_number)}\n")

    dropped = sorted({str(o) for o in df["odor_sent"] if _is_excluded(o, exclude)})
    print(f"[INFO] {dataset}: excluded {dropped or ['<none>']}")
    print(f"[INFO] wrote {matrix_png}")
    print(f"[INFO] wrote {bars_png}")
    print(f"[INFO] wrote {row_key_txt}")
    return [matrix_png, bars_png, row_key_txt]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-path",
        type=Path,
        required=True,
        help="Exported binary_reactions_<dataset>_<order>.csv to re-render.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: alongside the CSV).",
    )
    parser.add_argument("--latency-sec", type=float, default=2.15)
    parser.add_argument("--after-window-sec", type=float, default=30.0)
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Odor to drop from both plots. Repeatable. Default: Benzaldehyde.",
    )
    args = parser.parse_args(argv)

    remake(
        csv_path=args.csv_path,
        out_dir=args.out_dir,
        latency_sec=args.latency_sec,
        after_window_sec=args.after_window_sec,
        exclude=args.exclude if args.exclude else DEFAULT_EXCLUDE,
    )


if __name__ == "__main__":
    main()
