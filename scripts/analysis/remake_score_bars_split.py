"""Mean-ordinal-score companion to ``remake_reaction_matrix_split``.

That script re-renders a dataset's *binary* PER bars — one bar per presentation,
height = fraction of flies that reacted. This one draws the same bars from the
ordinal PER model instead: height = mean predicted score (-1..5) across flies,
with SEM error bars.

The predictions CSV labels AIR-Training trials ``testing_<n>_<rig>_...`` with no
odor token, so ``score_summary`` drops the dataset entirely. The odor per trial
is therefore taken from the ``binary_reactions_<dataset>_<order>.csv`` exported
next to the reaction figure, joined on (fly, fly_number, trial_num). That also
guarantees this figure's bars line up one-for-one with that figure's bars.

``--first-presentation-only`` keeps just the earliest presentation of each odor,
collapsing the repeated Hexanol / AIR trials to a single bar apiece.

Nothing is overwritten: every output carries a ``_copy`` suffix.

Run::

    /home/ramanlab/anaconda3/bin/python \
        scripts/analysis/remake_score_bars_split.py \
        --csv-path .../AIR-Training/binary_reactions_AIR-Training_unordered.csv \
        --first-presentation-only
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
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fbpipe.utils.tables import read_table  # noqa: E402
from scripts.analysis import odor_bar_palette  # noqa: E402
from scripts.analysis.envelope_visuals import (  # noqa: E402
    _canon_dataset,
    _matrix_title,
    _normalise_fly_columns,
    _trained_label,
    _trial_num,
)
from scripts.analysis.remake_reaction_matrix_split import (  # noqa: E402
    DEFAULT_EXCLUDE,
    _is_excluded,
    load_binary_reactions,
)

DEFAULT_PREDICTIONS_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-ALL-Opto-Flys/model_predictions.csv"
)

SCORE_MIN, SCORE_MAX = -1.0, 5.0
SCORE_Y_LABEL = "Mean PER Score"

# AIR is not an odorant, so it has no entry in the shared palette. It keeps the
# blue the reaction-rate figure gives the trained bar; every named odorant takes
# its palette colour, so the same odor reads the same across every bar figure.
AIR_COLOR = "tab:blue"
UNKNOWN_ODOR_COLOR = "0.6"

_RC_CONTEXT = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
}

_CSV_NAME_RE = re.compile(r"^binary_reactions_(?P<dataset>.+?)_(?P<order>[A-Za-z-]+)\.csv$")

__all__ = [
    "AIR_COLOR",
    "DEFAULT_EXCLUDE",
    "DEFAULT_PREDICTIONS_CSV",
    "attach_odors",
    "bar_color",
    "load_scores",
    "remake_score_bars",
    "render_score_bars_figure",
    "score_stats",
]


def load_scores(source: Path | pd.DataFrame, *, dataset: str) -> pd.DataFrame:
    """Testing-trial ordinal scores for one dataset, keyed by fly and trial number."""
    df = source.copy() if isinstance(source, pd.DataFrame) else read_table(Path(source))

    required = {"dataset", "fly", "fly_number", "trial_label", "score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Predictions CSV is missing required columns: {', '.join(sorted(missing))}"
        )

    df = df.loc[df["dataset"].astype(str).str.strip() == str(dataset).strip()].copy()
    if df.empty:
        raise RuntimeError(f"No rows for dataset {dataset!r} in the predictions CSV.")

    if "trial_type" in df.columns:
        is_testing = df["trial_type"].astype(str).str.strip().str.lower() == "testing"
        df = df.loc[is_testing].copy()
    if df.empty:
        raise RuntimeError(f"No testing trials for dataset {dataset!r}.")

    df = _normalise_fly_columns(df)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["trial_num"] = df["trial_label"].apply(_trial_num)
    df = df.dropna(subset=["trial_num", "score"])
    df["trial_num"] = df["trial_num"].astype(int)

    # One score per (fly, trial); the predictions CSV can repeat a row when a
    # fly was re-scored.
    df = df.drop_duplicates(subset=["fly", "fly_number", "trial_num"], keep="first")
    return df[["fly", "fly_number", "trial_num", "score"]].reset_index(drop=True)


def attach_odors(scores: pd.DataFrame, binary: Path | pd.DataFrame) -> pd.DataFrame:
    """Label each score with the odor that trial presented, per the binary export."""
    if isinstance(binary, pd.DataFrame):
        binary_df = _normalise_fly_columns(binary.copy())
        binary_df["odor_sent"] = binary_df["odor_sent"].astype(str).str.strip()
        binary_df["trial_num"] = pd.to_numeric(binary_df["trial_num"], errors="coerce")
    else:
        binary_df = _normalise_fly_columns(load_binary_reactions(Path(binary)))

    key = ["fly", "fly_number", "trial_num"]
    lookup = binary_df.dropna(subset=["trial_num"]).copy()
    lookup["trial_num"] = lookup["trial_num"].astype(int)
    lookup = lookup[key + ["odor_sent"]].drop_duplicates(subset=key, keep="first")

    merged = scores.merge(lookup, on=key, how="left").rename(columns={"odor_sent": "odor"})

    unmatched = merged.loc[merged["odor"].isna(), key].drop_duplicates()
    if not unmatched.empty:
        sample = ", ".join(
            f"{fly}/{fly_number} trial {trial}"
            for fly, fly_number, trial in unmatched.head(5).itertuples(index=False)
        )
        raise RuntimeError(
            f"{len(unmatched)} scored trial(s) have no odor in the binary export: {sample}"
        )
    return merged


def score_stats(
    df: pd.DataFrame,
    *,
    trained_label: str,
    exclude: Iterable[str] = DEFAULT_EXCLUDE,
    first_presentation_only: bool = False,
) -> pd.DataFrame:
    """Mean score / SEM / fly count per presentation, in trial order."""
    kept = df.loc[~df["odor"].apply(lambda odor: _is_excluded(odor, exclude))].copy()
    if kept.empty:
        raise RuntimeError("Every row was excluded; nothing left to plot.")

    if first_presentation_only:
        first_trial = kept.groupby("odor")["trial_num"].min()
        kept = kept.loc[
            kept["trial_num"] == kept["odor"].map(first_trial)
        ].copy()

    stats = (
        kept.groupby(["trial_num", "odor"], dropna=False)["score"]
        .agg(mean_score="mean", sem_score="sem", n_flies="size")
        .reset_index()
    )
    stats["sem_score"] = stats["sem_score"].fillna(0.0)
    stats = stats.sort_values(["trial_num", "odor"], kind="mergesort").reset_index(drop=True)
    stats["is_trained"] = (
        stats["odor"].astype(str).str.casefold().str.startswith(str(trained_label).casefold())
    )
    return stats


def bar_color(odor: str, is_trained: bool) -> str:
    """Palette colour for an odorant; blue for the trained non-odorant (AIR)."""
    palette = odor_bar_palette.odor_color(odor)
    if palette is not None:
        return palette
    return AIR_COLOR if bool(is_trained) else UNKNOWN_ODOR_COLOR


def tick_color(odor: str, is_trained: bool) -> str:
    """Only a bar whose colour is dark enough to read as text gets a coloured tick."""
    if not bool(is_trained) or odor_bar_palette.odor_color(odor) is not None:
        return "black"
    return AIR_COLOR


def render_score_bars_figure(
    stats: pd.DataFrame, *, title: str = "Mean Ordinal Score by Odor"
) -> plt.Figure:
    n_bars = len(stats)
    fig_w = max(10.0, 0.70 * n_bars + 6.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.4))

    x = range(n_bars)
    colors = [bar_color(odor, is_trained) for odor, is_trained in
              zip(stats["odor"], stats["is_trained"])]
    means = stats["mean_score"].to_numpy(float)
    sems = stats["sem_score"].to_numpy(float)

    ax.bar(
        x,
        means,
        yerr=sems,
        color=colors,
        edgecolor="black",
        linewidth=0.75,
        error_kw={"ecolor": "black", "elinewidth": 1.0, "capsize": 4},
    )
    # Scores run -1..5, so the bars need a visible baseline to read against.
    ax.axhline(0.0, color="black", linewidth=0.8)

    ax.set_xticks(list(x))
    labels = [
        str(odor).upper() if bool(is_trained) else str(odor)
        for odor, is_trained in zip(stats["odor"], stats["is_trained"])
    ]
    ax.set_xticklabels(labels, rotation=35, ha="right")
    for tick, odor, is_trained in zip(
        ax.get_xticklabels(), stats["odor"], stats["is_trained"]
    ):
        tick.set_color(tick_color(odor, is_trained))
        if bool(is_trained):
            tick.set_weight("bold")

    ax.set_ylim(SCORE_MIN, SCORE_MAX)
    ax.set_ylabel(SCORE_Y_LABEL)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.margins(x=0.02)

    # The fly count is per cohort, not per bar: state it once in the legend and
    # keep it on a bar only where that bar's count differs from the rest.
    counts = {int(n) for n in stats["n_flies"]}
    shared_n = counts.pop() if len(counts) == 1 else None
    odor_bar_palette.add_training_legend(
        ax,
        colors,
        train_label="Training" + (f" (n={shared_n})" if shared_n else ""),
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )

    for xi, mean, sem, n in zip(x, means, sems, stats["n_flies"]):
        annotation = f"{mean:.2f}" if shared_n else f"{mean:.2f}\n(n={int(n)})"
        if mean >= 0:
            ax.text(
                xi,
                min(mean + sem + 0.12, SCORE_MAX - 0.55),
                annotation,
                ha="center",
                va="bottom",
                fontsize=9,
            )
        else:
            # Below-zero bars would collide with their own bar; label underneath.
            ax.text(
                xi,
                max(mean - sem - 0.12, SCORE_MIN + 0.55),
                annotation,
                ha="center",
                va="top",
                fontsize=9,
            )

    fig.tight_layout()
    return fig


def remake_score_bars(
    *,
    csv_path: Path,
    predictions_csv: Path = DEFAULT_PREDICTIONS_CSV,
    out_dir: Path | None = None,
    exclude: Iterable[str] = DEFAULT_EXCLUDE,
    first_presentation_only: bool = False,
    formats: Sequence[str] = ("png", "pdf"),
) -> list[Path]:
    """Write the standalone score-bar figure for one ``binary_reactions_*.csv``."""
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

    dataset_canon = _canon_dataset(dataset)
    trained = _trained_label(dataset_canon)

    scores = attach_odors(
        load_scores(Path(predictions_csv), dataset=dataset), csv_path
    )
    stats = score_stats(
        scores,
        trained_label=trained,
        exclude=exclude,
        first_presentation_only=first_presentation_only,
    )
    n_flies = len(scores[["fly", "fly_number"]].drop_duplicates())

    stem = f"score_bars_{dataset}_{order}"
    if first_presentation_only:
        stem = f"{stem}_first-presentation"
    stem = f"{stem}_copy"

    heading = f"{_matrix_title(dataset_canon)}\nMean PER Score by Odor"

    written: list[Path] = []
    with plt.rc_context(_RC_CONTEXT):
        fig = render_score_bars_figure(stats, title=heading)
        for suffix in formats:
            path = out_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            written.append(path)
        plt.close(fig)

    csv_out = out_dir / f"{stem}.csv"
    stats.to_csv(csv_out, index=False, float_format="%.4f")
    written.append(csv_out)

    dropped = sorted({str(o) for o in scores["odor"] if _is_excluded(o, exclude)})
    print(f"[INFO] {dataset}: {len(stats)} bars, {n_flies} flies")
    print(f"[INFO] {dataset}: excluded {dropped or ['<none>']}")
    for path in written:
        print(f"[INFO] wrote {path}")
    return written


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-path",
        type=Path,
        required=True,
        help="Exported binary_reactions_<dataset>_<order>.csv, used for the odor map.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=DEFAULT_PREDICTIONS_CSV,
        help="Ordinal-score predictions CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: alongside the CSV).",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Odor to drop. Repeatable. Default: Benzaldehyde.",
    )
    parser.add_argument(
        "--first-presentation-only",
        action="store_true",
        help="Keep only the earliest presentation of each odor.",
    )
    parser.add_argument(
        "--format",
        action="append",
        dest="formats",
        default=None,
        help="Output format, repeatable. Default: png and pdf.",
    )
    args = parser.parse_args(argv)

    remake_score_bars(
        csv_path=args.csv_path,
        predictions_csv=args.predictions_csv,
        out_dir=args.out_dir,
        exclude=args.exclude if args.exclude else DEFAULT_EXCLUDE,
        first_presentation_only=args.first_presentation_only,
        formats=tuple(args.formats) if args.formats else ("png", "pdf"),
    )


if __name__ == "__main__":
    main()
