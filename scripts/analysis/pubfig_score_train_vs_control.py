#!/usr/bin/env python3
"""Publication-grade training-vs-control mean-score bars.

``score_summary.py`` and ``rig_batch_breakdowns.py`` both emit a working
version of this figure (``mean_score_train_vs_ctrl_<dataset>.png``,
``mean_score_tvc_batch_<n>.png``). Those stay as they are — this script
re-renders the same numbers in the house publication style used by the other
figures in ``Results/Figures``:

* training bars coloured by odor (``odor_bar_palette``), control bars gray;
* y axis "Mean PER Score", topping out at the score maximum;
* the cohort n stated once in the legend instead of on every tick label;
* significance as stars only — no p-values, and no bracket where the
  comparison is not significant.

Sources, both recomputed from the predictions CSV so nothing is transcribed:

    # per-dataset (score_summary)
    pubfig_score_train_vs_control.py dataset --train-dataset 3OCT-Training-24-0.1

    # per-batch (rig_batch_breakdowns)
    pubfig_score_train_vs_control.py batch --train-dataset 3OCT-Training-24-0.1 \
        --control-dataset 3OCT-Control-24-0.1 --tag tvc_batch_1
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.analysis import odor_bar_palette as pal  # noqa: E402

FIGURES_DIR = Path("/home/ramanlab/Documents/cole/Results/Figures")
PREDICTIONS_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/model_predictions.csv"
)
CONFIG = REPO_ROOT / "config" / "config_new.yaml"
GENOTYPE = "GR5a-Old"

SCORE_MIN, SCORE_MAX = -1.0, 5.0
Y_LABEL = "Mean PER Score"
ALPHA = 0.05

# The ordinal PER model emits only {-1, 0, 2, 3, 4, 5} -- it never outputs 1 --
# so ">= 1" and ">= 2" select exactly the same trials on real data. 2 is the
# lowest score it can actually produce that means "the proboscis extended".
# -1 marks a non-reactive trial, which is a non-response, not missing data.
RESPONSE_THRESHOLD = 2.0


def responded(score: float) -> bool:
    """Did this fly extend its proboscis for this odor?"""
    return bool(float(score) >= RESPONSE_THRESHOLD)


@dataclass(frozen=True)
class Metric:
    """Everything about a panel that depends on *what* is being plotted.

    Both metrics share the row schema (``ROW_COLUMNS``) and the whole filter /
    pairing / save path; only the axis, the value formatting and the vertical
    padding of annotations differ. Keeping the pads proportional to the span
    means the percentage panel's labels and brackets sit exactly where the
    score panel's do rather than being retuned by eye.
    """

    key: str
    y_label: str
    y_min: float
    y_max: float
    value_fmt: str
    zero_line: bool

    @property
    def span(self) -> float:
        return self.y_max - self.y_min

    @property
    def label_pad(self) -> float:
        return 0.0154 * self.span

    @property
    def bracket_pad(self) -> float:
        return 0.0846 * self.span

    @property
    def bracket_tick(self) -> float:
        return 0.0185 * self.span


SCORE_METRIC = Metric(
    key="mean-score",
    y_label=Y_LABEL,
    y_min=SCORE_MIN - 0.5,
    y_max=SCORE_MAX,
    value_fmt="{:.2f}",
    zero_line=True,
)
PERCENT_METRIC = Metric(
    key="percent-responding",
    y_label="% of Flies Responding",
    y_min=0.0,
    y_max=100.0,
    value_fmt="{:.0f}%",
    # A percentage axis starts at its own floor, so a rule at 0 would just
    # overdraw the spine.
    zero_line=False,
)
METRICS = {m.key: m for m in (SCORE_METRIC, PERCENT_METRIC)}

# A fly folder is named ``<month>_<day>_batch_<n>[_rig_<n>]``, so the collection
# month is the leading token. That is the only date the predictions CSV carries,
# and it is what ``--fly-months`` cuts a cohort on: the Hex-*-24-0.01 datasets
# were collected in April/May and then restarted in July/August, and the two
# blocks are worth looking at apart.
MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)

ROW_COLUMNS = [
    "odor", "is_trained",
    "mean_train", "sem_train", "n_train",
    "mean_ctrl", "sem_ctrl", "n_ctrl", "p_value",
]


def _stars(p: float) -> str:
    """Stars for a p-value, or "" when it is not significant."""
    if pd.isna(p) or p >= ALPHA:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    return "*"


def fly_month(fly: str) -> str | None:
    """Collection month of a fly folder, or ``None`` when it carries no date.

    ``"july_31_batch_1_rig_2"`` -> ``"july"``. The undated ``flagged`` folder,
    and anything else that does not lead with a whole month token, comes back
    ``None`` so it can never be filed under the wrong era.
    """
    token = str(fly).strip().split("_", 1)[0].casefold()
    return token if token in MONTHS else None


def filter_by_fly_months(df: pd.DataFrame, months) -> pd.DataFrame:
    """Keep only rows whose fly folder was collected in one of ``months``.

    ``months=None`` is a no-op. Undated flies are always dropped — there is no
    era to put them in.
    """
    if not months:
        return df
    wanted = {str(m).strip().casefold() for m in months}
    unknown = sorted(wanted.difference(MONTHS))
    if unknown:
        raise SystemExit(f"Not a month name: {', '.join(unknown)}")
    kept = df.loc[df["fly"].map(fly_month).isin(wanted)].copy()
    if kept.empty:
        raise SystemExit(f"no flies collected in {', '.join(sorted(wanted))}")
    return kept


def _fly_keys(df: pd.DataFrame):
    """``(dataset, fly, fly_number)`` per row, keyed the way the truth CSV is."""
    from fbpipe.config import canon_fly_number

    return list(
        zip(
            df["dataset"].astype(str).str.strip(),
            df["fly"].astype(str).str.strip(),
            df["fly_number"].map(canon_fly_number),
        )
    )


def apply_flagged_exclusions(
    df: pd.DataFrame, flagged_flies_csv: str, *, known_keys=None, datasets=None
) -> pd.DataFrame:
    """Drop flies the truth CSV marks ``FLY-State != 1``.

    Matching is exact on ``(dataset, fly, fly_number)``. A row whose fly folder
    is spelled differently in the truth CSV than in the predictions CSV matches
    nothing and would otherwise excludenothing at all, silently — the figure
    still renders, just with a fly the truth table calls dead. Those rows are
    printed as ``[UNMATCHED]`` so a typo is loud instead of invisible.

    ``known_keys`` is the fly set to judge "matched nothing" against, and should
    come from the frame *before* any genotype or month filtering, so a fly that
    a later filter legitimately removed is not mistaken for a misspelling.
    ``datasets`` scopes the report to the cohorts this figure draws.
    """
    from fbpipe.config import load_flagged_fly_exclusions

    # A missing file is an error — the loader only warns and returns an empty
    # set, which would look exactly like "nothing to exclude" and quietly leave
    # every dead fly in the figure. An empty result from a file that *does*
    # exist is legitimate: it means every fly listed is alive.
    if not Path(flagged_flies_csv).expanduser().exists():
        raise SystemExit(f"Flagged-flies CSV not found: {flagged_flies_csv}")
    exclusions = load_flagged_fly_exclusions(flagged_flies_csv)

    keys = _fly_keys(df)
    mask = pd.Series([k in exclusions for k in keys], index=df.index)

    scope = set(datasets) if datasets else None
    known = set(known_keys) if known_keys is not None else set(keys)
    unmatched = sorted(
        e for e in exclusions
        if (scope is None or e[0] in scope) and e not in known
    )
    kept = df.loc[~mask].copy()
    dropped = sorted({(k[0], k[1], k[2]) for k, m in zip(keys, mask) if m})
    print(f"[FLAGGED] {len(dropped)} flies excluded via {flagged_flies_csv}")
    for ds, fly, num in dropped:
        print(f"[FLAGGED]   {ds}  {fly}  fly {num}")
    for ds, fly, num in unmatched:
        print(
            f"[UNMATCHED] {ds}  {fly}  fly {num} — no such fly in the predictions "
            f"CSV, so this exclusion did nothing"
        )
    return kept


def _cohort_label(name: str, counts) -> str:
    """``"Training (n=23)"`` — the count belongs in the legend when it is shared."""
    uniq = {int(n) for n in pd.Series(counts).dropna()}
    return f"{name} (n={uniq.pop()})" if len(uniq) == 1 else name


def plot_train_vs_control(
    ax, rows: pd.DataFrame, *, title: str, metric: Metric = SCORE_METRIC
) -> None:
    """Draw one publication-grade training-vs-control panel."""
    rows = rows.reset_index(drop=True)
    x = np.arange(len(rows))
    bar_w = 0.35

    train_colors = pal.training_bar_colors(rows["odor"], rows["is_trained"])
    bars_train = ax.bar(
        x - bar_w / 2, rows["mean_train"].to_numpy(float), width=bar_w,
        yerr=rows["sem_train"].to_numpy(float), capsize=3,
        color=train_colors, edgecolor="black", linewidth=0.75,
        error_kw={"linewidth": 0.9},
    )
    bars_ctrl = ax.bar(
        x + bar_w / 2, rows["mean_ctrl"].to_numpy(float), width=bar_w,
        yerr=rows["sem_ctrl"].to_numpy(float), capsize=3,
        color=pal.CTRL_COLOR, edgecolor="black", linewidth=0.75,
        error_kw={"linewidth": 0.9},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(list(rows["odor"]), rotation=35, ha="right")
    for tick, odor, trained in zip(ax.get_xticklabels(), rows["odor"], rows["is_trained"]):
        if bool(trained):
            tick.set_color(pal.trained_tick_color(odor))
            tick.set_weight("bold")

    ax.set_ylim(metric.y_min, metric.y_max)
    if metric.zero_line:
        ax.axhline(0, color="0.4", linewidth=0.7)
    ax.set_ylabel(metric.y_label, fontsize=12)
    ax.set_xlabel("")
    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.margins(x=0.04)

    # Mean above each bar, clear of its whisker. The n is in the legend, so it
    # is not repeated here.
    for bars, mean_col, sem_col in (
        (bars_train, "mean_train", "sem_train"),
        (bars_ctrl, "mean_ctrl", "sem_ctrl"),
    ):
        for bar, mean_v, sem_v in zip(bars, rows[mean_col], rows[sem_col]):
            top = max(float(mean_v) + float(sem_v), 0.0)
            ax.text(
                bar.get_x() + bar.get_width() / 2, top + metric.label_pad,
                metric.value_fmt.format(float(mean_v)),
                ha="center", va="bottom", fontsize=8,
            )

    _draw_significant_brackets(ax, x, bar_w, rows, metric=metric)

    pal.add_training_legend(
        ax, train_colors, ctrl_color=pal.CTRL_COLOR,
        train_label=_cohort_label("Training", rows["n_train"]),
        ctrl_label=_cohort_label("Control", rows["n_ctrl"]),
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
    )


def _draw_significant_brackets(
    ax, x, bar_w: float, rows: pd.DataFrame, *, metric: Metric = SCORE_METRIC
) -> None:
    """Bracket + stars over the significant pairs only."""
    for i, row in enumerate(rows.itertuples(index=False)):
        stars = _stars(row.p_value)
        if not stars:
            continue
        top = max(
            float(row.mean_train) + float(row.sem_train),
            float(row.mean_ctrl) + float(row.sem_ctrl),
            0.0,
        )
        bracket_y = top + metric.bracket_pad
        tick = metric.bracket_tick
        ax.plot(
            [x[i] - bar_w / 2, x[i] - bar_w / 2, x[i] + bar_w / 2, x[i] + bar_w / 2],
            [bracket_y - tick, bracket_y, bracket_y, bracket_y - tick],
            color="black", linewidth=0.9, clip_on=False,
        )
        ax.text(
            x[i], bracket_y + 0.0062 * metric.span, stars,
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def rows_from_score_summary(
    predictions_csv: Path, train_dataset: str, *, config: Path | None = None,
    genotype: str = GENOTYPE, fly_months=None, flagged_flies_csv: str = "",
    odor_remap=None,
) -> pd.DataFrame:
    """Per-dataset rows, recomputed with ``score_summary``'s own pipeline.

    ``fly_months`` restricts both cohorts to flies collected in those months,
    which is how a dataset that was collected in two separated blocks gets one
    block's figure.
    """
    from scripts.analysis import score_summary as ss

    df = _prepare_scores(
        predictions_csv, train_dataset, config=config, genotype=genotype,
        fly_months=fly_months, flagged_flies_csv=flagged_flies_csv,
        odor_remap=odor_remap,
    )
    summary = ss._compute_training_vs_control_summary(df)
    sub = summary[summary["training_dataset"] == train_dataset].copy()
    if sub.empty:
        raise SystemExit(f"No rows for {train_dataset} in {predictions_csv}")
    if "trial_num" in sub.columns:
        sub = sub.sort_values(["trial_num", "odor"])
    return pd.DataFrame(
        {
            "odor": sub["odor"].to_numpy(),
            "is_trained": sub["is_trained"].astype(bool).to_numpy(),
            "mean_train": sub["mean_score_train"].astype(float).to_numpy(),
            "sem_train": sub["sem_score_train"].fillna(0.0).astype(float).to_numpy(),
            "n_train": sub["n_flies_train"].astype(int).to_numpy(),
            "mean_ctrl": sub["mean_score_ctrl"].astype(float).to_numpy(),
            "sem_ctrl": sub["sem_score_ctrl"].fillna(0.0).astype(float).to_numpy(),
            "n_ctrl": sub["n_flies_ctrl"].astype(int).to_numpy(),
            "p_value": sub["score_p_value"].astype(float).to_numpy(),
        },
        columns=ROW_COLUMNS,
    )


def parse_odor_remap(pairs) -> dict[str, str]:
    """``["Apple Cider Vinegar=Isoamyl Acetate (1%)"]`` -> a mapping."""
    out: dict[str, str] = {}
    for raw in pairs or ():
        text = str(raw)
        if "=" not in text:
            raise SystemExit(f"--odor-remap wants KEY=VALUE, got {text!r}")
        key, value = text.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key:
            raise SystemExit(f"--odor-remap has an empty odor name: {text!r}")
        out[key] = value
    return out


def _install_cohort_odor_remap(train_dataset: str, overrides: dict[str, str]) -> None:
    """Layer ``overrides`` onto this cohort's two datasets, and only those.

    A dataset can outlive its own rig plumbing: ``Hex-*-24-0.1`` delivered
    sourdough yeast on the Citral channel in May/June and the newer panel in
    August, so no single per-dataset ``odor_remap`` describes both. Overriding
    here keeps the config's dataset-level mapping intact for every other figure
    over the same dataset.

    Both cohorts are patched because figures pair training and control by
    display label -- relabelling one side alone would split an odor into two
    unpaired columns and still render.
    """
    if not overrides:
        return
    from scripts.analysis.envelope_visuals import (
        _DATASET_ODOR_REMAP,
        _canon_dataset,
        set_dataset_odor_remap,
    )

    names = {train_dataset, train_dataset.replace("Training", "Control")}
    keys = {n for name in names for n in (name, _canon_dataset(name))}
    merged = {ds: dict(m) for ds, m in _DATASET_ODOR_REMAP.items()}
    for key in keys:
        merged.setdefault(key, {}).update(overrides)
    set_dataset_odor_remap(merged)


def _prepare_scores(
    predictions_csv: Path, train_dataset: str, *, config: Path | None = None,
    genotype: str = GENOTYPE, fly_months=None, flagged_flies_csv: str = "",
    odor_remap=None,
) -> pd.DataFrame:
    """Load the predictions CSV and apply every cohort filter, in order.

    Shared by both metrics so a figure pair can never disagree about which
    flies it is describing.
    """
    from scripts.analysis import score_summary as ss

    _apply_config(config)
    # After the config, so a cohort override wins over the dataset-level map,
    # and before _load_scores, which is what resolves trial labels to odors.
    _install_cohort_odor_remap(train_dataset, dict(odor_remap or {}))
    df = ss._load_scores(predictions_csv, threshold=None, flagged_flies_csv="")
    # Judge "this exclusion matched nothing" against every fly the CSV holds,
    # before the genotype and month filters legitimately remove some.
    all_keys = _fly_keys(df)
    if "fly_type" in df.columns:
        df = df[df["fly_type"].astype(str).str.strip() == genotype].copy()
    if flagged_flies_csv:
        from scripts.analysis.envelope_visuals import _canon_dataset as _cd

        train_canon = _cd(train_dataset)
        pair = {train_canon, ss._auto_pairs(sorted(set(df["dataset_canon"]))).get(train_canon)}
        df = apply_flagged_exclusions(
            df, flagged_flies_csv, known_keys=all_keys, datasets={d for d in pair if d},
        )
    if fly_months:
        df = filter_by_fly_months(df, fly_months)
        # Report only the pair this figure draws — the frame still holds every
        # other dataset in the CSV, and listing those folders would be noise.
        from scripts.analysis.envelope_visuals import _canon_dataset

        train_canon = _canon_dataset(train_dataset)
        pairs = ss._auto_pairs(sorted(set(df["dataset_canon"])))
        pair = {train_canon, pairs.get(train_canon)}
        for ds, group in df[df["dataset_canon"].isin(pair)].groupby("dataset_canon"):
            kept = sorted(group["fly"].unique())
            print(f"[MONTHS] {ds}: {len(kept)} fly folders in {', '.join(fly_months)}"
                  f" -> {', '.join(kept)}")
    return df


def rows_from_percent_responding(
    predictions_csv: Path, train_dataset: str, *, config: Path | None = None,
    genotype: str = GENOTYPE, fly_months=None, flagged_flies_csv: str = "",
    odor_remap=None,
) -> pd.DataFrame:
    """Per-odor share of flies that extended, training vs control.

    The fly is the unit, not the trial: a fly's scores for one odor are averaged
    first and that single value is thresholded, so a fly that saw an odor twice
    cannot vote twice. Significance is Fisher's exact on the 2x2 of
    responders/non-responders by cohort -- the counts are what is being
    compared here, so a rank test on the underlying scores would be answering
    the other metric's question.
    """
    from scipy.stats import fisher_exact

    from scripts.analysis import score_summary as ss
    from scripts.analysis.envelope_visuals import _canon_dataset, _trained_label

    df = _prepare_scores(
        predictions_csv, train_dataset, config=config, genotype=genotype,
        fly_months=fly_months, flagged_flies_csv=flagged_flies_csv,
        odor_remap=odor_remap,
    )
    train_canon = _canon_dataset(train_dataset)
    ctrl_canon = ss._auto_pairs(sorted(set(df["dataset_canon"]))).get(train_canon)
    if ctrl_canon is None:
        raise SystemExit(f"No control cohort pairs with {train_dataset}")

    fly_level = (
        df.groupby(["dataset_canon", "odor_col", "fly", "fly_number"])["score"]
        .mean()
        .reset_index()
    )
    fly_level["responded"] = fly_level["score"].map(responded)

    def counts(dataset_canon: str) -> dict[str, tuple[int, int]]:
        sub = fly_level[fly_level["dataset_canon"] == dataset_canon]
        return {
            str(odor): (int(g["responded"].sum()), int(len(g)))
            for odor, g in sub.groupby("odor_col")
        }

    train_counts, ctrl_counts = counts(train_canon), counts(ctrl_canon)
    columns = sorted(set(train_counts).union(ctrl_counts), key=str.casefold)
    if not columns:
        raise SystemExit(f"No odors for {train_dataset} in {predictions_csv}")
    trained = _trained_label(train_canon)

    def pct_and_se(k: int, n: int) -> tuple[float, float]:
        if not n:
            return float("nan"), 0.0
        p = k / n
        return 100.0 * p, 100.0 * float(np.sqrt(p * (1.0 - p) / n))

    records = []
    for odor in columns:
        k_t, n_t = train_counts.get(odor, (0, 0))
        k_c, n_c = ctrl_counts.get(odor, (0, 0))
        pct_t, se_t = pct_and_se(k_t, n_t)
        pct_c, se_c = pct_and_se(k_c, n_c)
        if n_t and n_c:
            p_value = float(
                fisher_exact([[k_t, n_t - k_t], [k_c, n_c - k_c]])[1]
            )
        else:
            p_value = float("nan")
        records.append(
            {
                "odor": odor,
                "is_trained": str(odor).casefold().startswith(trained.casefold()),
                "mean_train": pct_t, "sem_train": se_t, "n_train": n_t,
                "mean_ctrl": pct_c, "sem_ctrl": se_c, "n_ctrl": n_c,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(records, columns=ROW_COLUMNS)


def rows_from_batch(
    predictions_csv: Path, train_dataset: str, control_dataset: str, tag: str,
    *, config: Path | None = None,
) -> tuple[pd.DataFrame, str]:
    """Per-batch rows, recomputed with ``rig_batch_breakdowns``' own pipeline."""
    from scripts.analysis import rig_batch_breakdowns as rbb
    from scripts.analysis.envelope_visuals import _canon_dataset, _trained_label

    _apply_config(config)
    train_canon = _canon_dataset(train_dataset)
    ctrl_canon = _canon_dataset(control_dataset)
    reactions = rbb.load_reaction_frame(predictions_csv, flagged_flies_csv="")
    scores = rbb.load_score_frame(predictions_csv, flagged_flies_csv="")

    comps = {c.tag: c for c in rbb.build_comparisons(reactions, train_canon, ctrl_canon)}
    if tag not in comps:
        raise SystemExit(f"Unknown tag {tag!r}; have {sorted(comps)}")
    comp = comps[tag]

    df_a = rbb.select_group(scores, comp.a)
    df_b = rbb.select_group(scores, comp.b)
    columns = sorted(set(df_a["odor_col"]).union(df_b["odor_col"]), key=str.casefold)
    stats_a = rbb.group_score_stats(df_a, columns)
    stats_b = rbb.group_score_stats(df_b, columns)
    p_values = rbb.mannwhitney_per_column(
        rbb.group_score_samples(df_a, columns),
        rbb.group_score_samples(df_b, columns),
        columns,
    )
    trained = _trained_label(train_canon)
    rows = pd.DataFrame(
        {
            "odor": columns,
            "is_trained": [
                str(c).casefold().startswith(trained.casefold()) for c in columns
            ],
            "mean_train": stats_a["mean_score"].fillna(0.0).astype(float).to_numpy(),
            "sem_train": stats_a["sem_score"].fillna(0.0).astype(float).to_numpy(),
            "n_train": stats_a["n_flies"].fillna(0).astype(int).to_numpy(),
            "mean_ctrl": stats_b["mean_score"].fillna(0.0).astype(float).to_numpy(),
            "sem_ctrl": stats_b["sem_score"].fillna(0.0).astype(float).to_numpy(),
            "n_ctrl": stats_b["n_flies"].fillna(0).astype(int).to_numpy(),
            "p_value": [float(p_values.get(c, float("nan"))) for c in columns],
        },
        columns=ROW_COLUMNS,
    )
    return rows, comp.title_suffix


def _apply_config(config: Path | None, *, protocol: str = "v2") -> None:
    """Match the pipeline's odor labelling: protocol first, then odor_remap.

    Both matter — the protocol decides how a trial label resolves to an odor,
    and without it the bars come out named ``testing_6_citral``.
    """
    from scripts.analysis.envelope_visuals import set_protocol

    set_protocol(protocol)
    if not config:
        return
    try:
        from fbpipe.config import load_settings
        from scripts.analysis.envelope_visuals import set_dataset_odor_remap

        settings = load_settings(str(config))
        remap = {
            str(ds): dict(ov.odor_remap)
            for ds, ov in settings.dataset_overrides.items()
            if getattr(ov, "odor_remap", None)
        }
        if remap:
            set_dataset_odor_remap(remap)
    except Exception as exc:  # noqa: BLE001 — labels just fall back to defaults
        print(f"[WARN] could not load odor_remap from {config}: {exc}")


def save(
    rows: pd.DataFrame, *, title: str, out_stem: str, figures_dir: Path,
    metric: Metric = SCORE_METRIC,
) -> None:
    rc = {
        "figure.dpi": 300, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "Arial", "font.sans-serif": ["Arial"],
        "svg.fonttype": "none",
    }
    figures_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(max(7.5, 1.05 * len(rows) + 3.0), 5.5))
        plot_train_vs_control(ax, rows, title=title, metric=metric)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            out = figures_dir / f"{out_stem}.{suffix}"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"[SAVED] {out}")
        plt.close(fig)
    csv_out = figures_dir / f"{out_stem}.csv"
    rows.to_csv(csv_out, index=False, float_format="%.4f")
    print(f"[SAVED] {csv_out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="source", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--predictions-csv", type=Path, default=PREDICTIONS_CSV)
    common.add_argument("--config", type=Path, default=CONFIG)
    common.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    common.add_argument("--train-dataset", required=True)
    common.add_argument("--title", default=None)
    common.add_argument("--out-stem", default=None)
    common.add_argument(
        "--odor-remap", action="append", default=[], metavar="ODOR=LABEL",
        help="relabel one odor for THIS figure only, e.g. "
             "'Apple Cider Vinegar=Isoamyl Acetate (1%%)'. Repeatable. Wins "
             "over the config's per-dataset odor_remap and applies to both "
             "cohorts. Use when one dataset spans two rig configurations.",
    )
    common.add_argument(
        "--cohort-label", default="",
        help="what the default title is about, in place of the dataset name "
             "(e.g. 'Hex-24-0.01, August'). Each metric keeps its own wording, "
             "so one label titles both. Ignored when --title is given.",
    )

    common.add_argument(
        "--metric", choices=sorted(METRICS), default=SCORE_METRIC.key,
        help="mean-score: average PER score per odor. percent-responding: "
             "share of flies scoring >= 2 for that odor.",
    )

    p_ds = sub.add_parser("dataset", parents=[common])
    p_ds.add_argument("--genotype", default=GENOTYPE)
    p_ds.add_argument(
        "--flagged-flies-csv", default="",
        help="flagged-flys-truth.csv; flies with FLY-State != 1 are excluded",
    )
    p_ds.add_argument(
        "--fly-months", default="",
        help="comma-separated collection months to keep, e.g. 'july,august'; "
             "flies whose folder carries no date are dropped",
    )

    p_batch = sub.add_parser("batch", parents=[common])
    p_batch.add_argument("--control-dataset", required=True)
    p_batch.add_argument("--tag", required=True)

    args = parser.parse_args(argv)

    metric = METRICS[args.metric]
    percent = metric is PERCENT_METRIC

    if args.source == "dataset":
        months = tuple(m.strip() for m in args.fly_months.split(",") if m.strip())
        source = rows_from_percent_responding if percent else rows_from_score_summary
        rows = source(
            args.predictions_csv, args.train_dataset,
            config=args.config, genotype=args.genotype, fly_months=months or None,
            flagged_flies_csv=args.flagged_flies_csv,
            odor_remap=parse_odor_remap(args.odor_remap),
        )
        default_stem = (
            f"pubfig_pct_responding_train_vs_ctrl_{args.train_dataset}" if percent
            else f"pubfig_mean_score_train_vs_ctrl_{args.train_dataset}"
        )
        stem = args.out_stem or default_stem
        subject = args.cohort_label or args.train_dataset
        title = args.title or f"{metric.y_label} – {subject} (Training vs Control)"
    else:
        if percent:
            raise SystemExit("--metric percent-responding is only available for 'dataset'")
        rows, suffix = rows_from_batch(
            args.predictions_csv, args.train_dataset, args.control_dataset,
            args.tag, config=args.config,
        )
        stem = args.out_stem or f"pubfig_mean_score_{args.tag}_{args.train_dataset}"
        title = args.title or f"Mean PER Score – {args.train_dataset} ({suffix})"

    save(rows, title=title, out_stem=stem, figures_dir=args.figures_dir, metric=metric)
    for row in rows.itertuples(index=False):
        if percent:
            summary = (
                f"train {row.mean_train:5.1f}%±{row.sem_train:.1f} (n={row.n_train})"
                f"  ctrl {row.mean_ctrl:5.1f}%±{row.sem_ctrl:.1f} (n={row.n_ctrl})"
            )
        else:
            summary = (
                f"train {row.mean_train:+.2f}±{row.sem_train:.2f} (n={row.n_train})"
                f"  ctrl {row.mean_ctrl:+.2f}±{row.sem_ctrl:.2f} (n={row.n_ctrl})"
            )
        print(
            f"  {row.odor:24s} {summary}  p={row.p_value:.4f} "
            f"{_stars(row.p_value) or 'ns'}"
        )


if __name__ == "__main__":
    main()
