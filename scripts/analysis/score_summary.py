"""Generate average ordinal-score plots from model predictions.

Reads a predictions CSV that contains a ``score`` column (ordinal -1..5) and
produces:

1. Per-odor bar chart of mean score by testing number (with SEM error bars).
2. Heatmap of mean score (datasets x testing numbers).
3. Summary CSV: ``score_summary_by_odor_testing.csv``.
4. Training-vs-control grouped bar charts of mean score by presented odor.
5. Training-vs-control summary CSV with score-based p-values.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from scipy.stats import mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.envelope_visuals import (
    DISPLAY_LABEL,
    ODOR_ORDER,
    _canon_dataset,
    _display_label_ci,
    _display_odor,
    _extract_odor_from_label,
    _is_light_only_label,
    _is_testing_11_label,
    _normalise_fly_columns,
    _safe_dirname,
    _trained_label,
    _trial_num,
    apply_dataset_odor_remap,
    compute_non_reactive_flags,
    get_protocol,
    set_protocol,
    should_skip_frozen_figure,
    should_write,
)
from scripts.analysis import odor_bar_palette  # noqa: E402
from scripts.analysis import significance_brackets  # noqa: E402
from scripts.analysis.per_axis_labels import SCORE_Y_LABEL  # noqa: E402

_RC_CONTEXT = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
}

_STATIC_PAIRS = {
    "EB-Training": "EB-Control",
    "Hex-Training": "Hex-Control",
    "Hex-Training-24": "Hex-Control-24",
    "Hex-Training-24-2": "Hex-Control-24-2",
    "Hex-Training-24-02": "Hex-Control-24-02",
    "Hex-Training-36": "Hex-Control-36",
    "Hex-Training-24-002": "Hex-Control-24-002",
    "Hex-Training-24-0002": "Hex-Control-24-0002",
    "Hex-Training-24-0.005": "Hex-Control-24-0.005",
    "Hex-Training-24-0.01": "Hex-Control-24-0.01",
    "Benz-Training": "Benz-Control",
    "Benz-Training-24": "Benz-Control",
    "Benz-Training-24-2": "Benz-Control-24-2",
    "Benz-Training-24-02": "Benz-Control-24-02",
    "ACV-Training": "ACV-Control",
    "3OCT-Training": "3OCT-Control",
    "3OCT-Training-24-2": "3OCT-Control-24-2",
    "Cit-Training": "Cit-Control",
    "Lin-Training": "Lin-Control",
}


# --- Per-fly score matrix palette -------------------------------------------
# Diverging at the reaction boundary (score >= 2 is a reaction, see
# fbpipe.config.binary_threshold), NOT at zero. Purple = no reaction,
# green = reaction.
#
# Red->green was rejected: it collapses to worst-pair CVD dE 4.1 (protanopia),
# i.e. a protanope cannot tell -1 from 5. This ramp measures dE 19.5.
# The no-reaction arm is deliberately pale: score 0 is ~71% of cells, so an
# even-stepped arm would make the modal "nothing happened" a wall of colour
# that the signal has to fight.
SCORE_MIN = -1
SCORE_MAX = 5
SCORES = list(range(SCORE_MIN, SCORE_MAX + 1))
SCORE_COLORS = [
    "#762a83",  # -1  strong purple  (rare, notable)
    "#e2d4e8",  #  0  pale lavender  (modal, recedes)
    "#f2ebf5",  #  1  palest lavender
    # ---- reaction boundary (score >= 2) ----
    "#a6dba0",  #  2  light green
    "#5aae61",  #  3
    "#1b7837",  #  4
    "#00441b",  #  5  dark green
]
MISSING_COLOR = "0.70"        # matches reaction_matrix's cmap.set_bad(color="0.7")
REACTION_BOUNDARY_Y = 1.5


def _score_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    """Fixed score->colour map. Never rank-based: a 3 is the same green in
    every dataset, whether or not that dataset happens to contain a 1."""
    cmap = ListedColormap(SCORE_COLORS)
    cmap.set_bad(color=MISSING_COLOR)
    bounds = [s - 0.5 for s in SCORES] + [SCORE_MAX + 0.5]
    return cmap, BoundaryNorm(bounds, cmap.N)


def _auto_pairs(datasets: list[str]) -> dict[str, str]:
    pairs = dict(_STATIC_PAIRS)
    pat = re.compile(r"^(.+)-Training(.*)$", re.IGNORECASE)
    ds_set = set(datasets)
    for ds in datasets:
        if ds in pairs:
            continue
        m = pat.match(ds)
        if m:
            ctrl = f"{m.group(1)}-Control{m.group(2)}"
            if ctrl in ds_set:
                pairs[ds] = ctrl
    return pairs

TRAINING_CONTROL_PAIRS = dict(_STATIC_PAIRS)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _normalise_trial_label(label: str) -> str:
    """Keep odor suffix for v2: testing_3_benzaldehyde → testing_3_benzaldehyde."""
    m = re.match(
        r"((?:pretest|testing|training)_\d+(?:_(?!fly\d|distances)[A-Za-z0-9._-]+?)?)"
        r"(?:_fly\d|_distances|$)",
        str(label), re.IGNORECASE,
    )
    return m.group(1) if m else str(label)


def _load_scores(
    csv_path: Path,
    *,
    threshold: float | None = None,
    flagged_flies_csv: str = "",
    trial_types: Sequence[str] = ("testing",),
) -> pd.DataFrame:
    """Load scored trials for one phase.

    ``trial_types`` defaults to the post-training panel, which is what every
    existing figure wants. Pass ``("pretest",)`` for the naive panel. Load ONE
    phase per call: the odor-occurrence numbering below groups by trial index
    within a fly, and ``pretest_1`` and ``testing_1`` share an index, so mixing
    the phases in one frame would merge them.
    """

    df = pd.read_csv(csv_path)
    required = {"dataset", "fly", "fly_number", "trial_label", "score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

    df = df.copy()

    # Filter to the requested phase
    wanted = {str(value).strip().lower() for value in trial_types}
    if "trial_type" in df.columns:
        mask = df["trial_type"].astype(str).str.strip().str.lower().isin(wanted)
        df = df.loc[mask].copy()

    if df.empty:
        raise RuntimeError(
            f"No {'/'.join(sorted(wanted))} trials found in predictions CSV."
        )

    df["dataset"] = df["dataset"].astype(str).str.strip()
    df["fly"] = df["fly"].astype(str).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = _normalise_fly_columns(df)

    # Drop non-reactive flies
    if threshold is not None:
        flags = compute_non_reactive_flags(
            df, threshold=threshold, flagged_flies_csv=flagged_flies_csv
        )
        if flags.any():
            df = df.loc[~flags.astype(bool)].copy()

    df["dataset_canon"] = df["dataset"].map(_canon_dataset)
    df["trial"] = df["trial_label"].astype(str).apply(_normalise_trial_label)
    df["trial_num"] = df["trial"].apply(_trial_num)

    # Drop light-only trials (testing_11 legacy, testing_9 v2)
    light_mask = df["trial"].apply(_is_light_only_label)
    df = df.loc[~light_mask].copy()

    # Also drop trials where the rig recording filename had a non-parseable
    # odor token (e.g. RandomPanel's training_15 / training_16 light-only
    # trials carry no odor suffix because the rig name has underscores). For
    # these the extracted "odor" is the trial label itself.
    def _has_no_odor_suffix(trial: str) -> bool:
        return _extract_odor_from_label(trial) == str(trial)

    no_odor_mask = df["trial"].apply(_has_no_odor_suffix)
    df = df.loc[~no_odor_mask].copy()

    # Add odor display name from trial label, then apply any per-dataset
    # remap (e.g. Hex-Control-24-0.1 swaps Citral -> "Sour Dough Yeast (25%)").
    df["odor_display"] = df.apply(
        lambda r: apply_dataset_odor_remap(
            r["dataset_canon"],
            _display_label_ci(_extract_odor_from_label(r["trial"])),
        ),
        axis=1,
    )

    # For v2: build odor_col with "Name 1"/"Name 2" suffixes.
    #
    # Default behaviour numbers ONLY the trained odor (it's presented twice
    # per fly in classic Hex/EB/etc. testing). For panels where every odor
    # is presented twice (e.g. RandomPanel), we instead number ALL repeated
    # odors so the bar plot shows exposure 1 vs exposure 2 side-by-side.
    if get_protocol() == "v2":
        _odor_col_rows = []
        for (ds, fly, fn), grp in df.groupby(["dataset_canon", "fly", "fly_number"]):
            seen: dict[str, int] = {}
            for idx, row in grp.sort_values("trial_num").iterrows():
                od = row["odor_display"]
                seen[od] = seen.get(od, 0) + 1
                _odor_col_rows.append((idx, seen[od]))
        occ_series = pd.Series(dict(_odor_col_rows), name="occurrence")
        df = df.join(occ_series)

        # Per dataset: which odors are duplicated within any fly?
        max_occ = df.groupby(["dataset_canon", "odor_display"])["occurrence"].max()
        dup_per_ds: dict[str, set[str]] = {}
        for (ds_canon, odor), val in max_occ.items():
            if val > 1:
                dup_per_ds.setdefault(ds_canon, set()).add(odor)

        # "Panel" datasets are ones where more than one odor is duplicated.
        # For these we number all duplicates. For single-trained-odor
        # datasets, we only number the trained odor (legacy behaviour).
        panel_datasets = {ds for ds, odors in dup_per_ds.items() if len(odors) > 1}

        def _should_number(row):
            ds_canon = row["dataset_canon"]
            od = row["odor_display"]
            if od not in dup_per_ds.get(ds_canon, set()):
                return False
            if ds_canon in panel_datasets:
                return True
            trained = _trained_label(ds_canon)
            # startswith, not ==: a per-dataset odor_remap may append text to the
            # trained odor's display name (e.g. "Ethyl Butyrate (1%)"). Exact
            # equality would fail there, the odor would stop being numbered, and
            # its two presentations would silently merge into one column with
            # n = 2 x flies. Matches the `is_trained` convention used elsewhere
            # in this module.
            return od.casefold().startswith(trained.casefold())

        df["odor_col"] = df.apply(
            lambda r: f"{r['odor_display']} {int(r['occurrence'])}" if _should_number(r) else r["odor_display"],
            axis=1,
        )
    else:
        df["odor_col"] = df["odor_display"]

    # De-duplicate (same logic as reaction matrix)
    df = df.drop_duplicates(
        subset=["dataset", "fly", "fly_number", "trial"], keep="first"
    )

    return df


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (dataset_canon, odor_col) and compute mean/sem/n."""
    group_key = "odor_col" if "odor_col" in df.columns and get_protocol() == "v2" else "trial_num"
    grouped = (
        df.groupby(["dataset_canon", group_key])["score"]
        .agg(["mean", "sem", "count"])
        .rename(columns={"mean": "mean_score", "sem": "sem_score", "count": "n_flies"})
        .reset_index()
    )
    grouped["sem_score"] = grouped["sem_score"].fillna(0.0)
    if group_key == "odor_col":
        grouped = grouped.sort_values(["dataset_canon", "odor_col"], key=lambda s: s.str.casefold() if s.dtype == object else s)
    return grouped


def _per_fly_score_matrix(
    df: pd.DataFrame, dataset: str, columns: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    """Build a (flies x columns) score matrix for one dataset.

    ``columns`` must be the same ordered odor_col list the bar chart uses, so
    the matrix and the bars align by construction. Absent (fly, odor) pairs are
    NaN. Rows are keyed on (fly, fly_number) -- ``fly`` alone is not unique.
    """
    sub = df[df["dataset_canon"] == dataset]
    if sub.empty or not len(columns):
        return np.full((0, len(columns)), np.nan), []

    fly_keys = sub["fly"].astype(str) + "|" + sub["fly_number"].astype(str)
    sub = sub.assign(_fly_key=fly_keys)
    flies = sorted(sub["_fly_key"].unique())

    row_of = {f: i for i, f in enumerate(flies)}
    col_of = {c: j for j, c in enumerate(columns)}
    matrix = np.full((len(flies), len(columns)), np.nan, dtype=float)
    for fly_key, odor, score in zip(
        sub["_fly_key"], sub["odor_col"], sub["score"]
    ):
        i = row_of.get(fly_key)
        j = col_of.get(odor)
        if i is not None and j is not None:
            matrix[i, j] = score
    return matrix, flies


def _sig_stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _empty_train_control_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "training_dataset",
            "control_dataset",
            "trial_num",
            "odor",
            "is_trained",
            "mean_score_train",
            "sem_score_train",
            "n_flies_train",
            "mean_score_ctrl",
            "sem_score_ctrl",
            "n_flies_ctrl",
            "score_p_value",
            "significance",
        ]
    )


def _compute_training_vs_control_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_train_control_summary()

    compare = df.copy()
    if get_protocol() == "v2" and "odor_col" in compare.columns:
        compare["odor"] = compare["odor_col"]
    else:
        compare["odor"] = [
            _display_odor(dataset, trial)
            for dataset, trial in zip(compare["dataset_canon"], compare["trial"])
        ]

    present = set(compare["dataset_canon"])
    active_pairs = _auto_pairs(list(present))
    summaries: list[pd.DataFrame] = []

    for train_ds, ctrl_ds in active_pairs.items():
        if train_ds not in present or ctrl_ds not in present:
            continue

        pair_df = compare[compare["dataset_canon"].isin((train_ds, ctrl_ds))].copy()
        if pair_df.empty:
            continue

        group_key = "odor" if get_protocol() == "v2" else "trial_num"
        group_cols = ["dataset_canon", group_key, "odor", "fly", "fly_number"] if group_key == "trial_num" else ["dataset_canon", "odor", "fly", "fly_number"]
        fly_level = (
            pair_df.groupby(group_cols)["score"]
            .mean()
            .rename("fly_mean_score")
            .reset_index()
        )
        sample_group_cols = ["dataset_canon", "odor"] if get_protocol() == "v2" else ["dataset_canon", "trial_num"]
        fly_samples = {
            tuple(key): group["fly_mean_score"].to_numpy(float)
            for key, group in fly_level.groupby(sample_group_cols)
        }
        stats_group = ["dataset_canon", "odor"] if get_protocol() == "v2" else ["dataset_canon", "trial_num", "odor"]
        score_stats = (
            fly_level.groupby(stats_group)["fly_mean_score"]
            .agg(["mean", "sem", "count"])
            .rename(
                columns={
                    "mean": "mean_score",
                    "sem": "sem_score",
                    "count": "n_flies",
                }
            )
            .reset_index()
        )
        score_stats["sem_score"] = score_stats["sem_score"].fillna(0.0)

        train_scores = (
            score_stats[score_stats["dataset_canon"] == train_ds]
            .drop(columns="dataset_canon")
            .rename(
                columns={
                    "mean_score": "mean_score_train",
                    "sem_score": "sem_score_train",
                    "n_flies": "n_flies_train",
                }
            )
        )
        ctrl_scores = (
            score_stats[score_stats["dataset_canon"] == ctrl_ds]
            .drop(columns="dataset_canon")
            .rename(
                columns={
                    "mean_score": "mean_score_ctrl",
                    "sem_score": "sem_score_ctrl",
                    "n_flies": "n_flies_ctrl",
                }
            )
        )

        merge_on = ["odor"] if get_protocol() == "v2" else ["trial_num", "odor"]
        merged = pd.merge(train_scores, ctrl_scores, on=merge_on, how="outer")

        if merged.empty:
            continue

        merged["training_dataset"] = train_ds
        merged["control_dataset"] = ctrl_ds
        merged["is_trained"] = (
            merged["odor"].astype(str).str.casefold()
            .str.startswith(_trained_label(train_ds).casefold())
        )

        for col in (
            "sem_score_train",
            "sem_score_ctrl",
            "n_flies_train",
            "n_flies_ctrl",
        ):
            merged[col] = merged[col].fillna(0)

        for col in ("n_flies_train", "n_flies_ctrl"):
            merged[col] = merged[col].astype(int)

        p_values: list[float] = []
        significance: list[str] = []
        for row in merged.itertuples(index=False):
            if get_protocol() == "v2":
                train_vals = fly_samples.get((train_ds, row.odor), np.array([], dtype=float))
                ctrl_vals = fly_samples.get((ctrl_ds, row.odor), np.array([], dtype=float))
            else:
                train_vals = fly_samples.get((train_ds, row.trial_num), np.array([], dtype=float))
                ctrl_vals = fly_samples.get((ctrl_ds, row.trial_num), np.array([], dtype=float))
            if len(train_vals) == 0 or len(ctrl_vals) == 0:
                p = np.nan
            else:
                _, p = mannwhitneyu(
                    train_vals,
                    ctrl_vals,
                    alternative="two-sided",
                    method="auto",
                )
            p_values.append(p)
            significance.append(_sig_stars(p))

        merged["score_p_value"] = p_values
        merged["significance"] = significance
        if get_protocol() == "v2":
            merged = merged.sort_values("odor", key=lambda s: s.str.casefold()).reset_index(drop=True)
        else:
            merged = merged.sort_values(["trial_num", "odor"]).reset_index(drop=True)

        out_cols = ["training_dataset", "control_dataset"]
        if "trial_num" in merged.columns:
            out_cols.append("trial_num")
        out_cols += [
            "odor", "is_trained",
            "mean_score_train", "sem_score_train", "n_flies_train",
            "mean_score_ctrl", "sem_score_ctrl", "n_flies_ctrl",
            "score_p_value", "significance",
        ]
        summaries.append(merged[[c for c in out_cols if c in merged.columns]])

    if not summaries:
        return _empty_train_control_summary()

    return pd.concat(summaries, ignore_index=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _draw_score_matrix(
    fig, ax_m, cax, matrix: np.ndarray, n_col: int, x: np.ndarray,
    columns: Sequence[str], is_trained: pd.Series,
) -> None:
    """Draw the per-fly score band + its discrete key."""
    cmap, norm = _score_cmap()
    n_fly = matrix.shape[0]
    ax_m.imshow(
        np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, aspect="auto",
        interpolation="nearest",
        extent=(-0.5, n_col - 0.5, n_fly - 0.5, -0.5),
    )
    # 2px surface gap between fills.
    for j in range(n_col + 1):
        ax_m.axvline(j - 0.5, color="white", lw=1.6)
    for i in range(n_fly + 1):
        ax_m.axhline(i - 0.5, color="white", lw=1.6)

    # Rows are anonymous, as in reaction_matrix: the row count is the message.
    ax_m.set_yticks([])
    ax_m.set_ylabel(f"{n_fly} Flies", fontsize=10)
    ax_m.tick_params(axis="x", labelbottom=False, length=0)
    for sp in ax_m.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.6)
        sp.set_color("0.6")

    # Odor labels BELOW the matrix on a secondary axis: sharex shares the tick
    # formatter, so labelling ax_m directly would also relabel the bars.
    ax_lab = ax_m.secondary_xaxis("bottom")
    ax_lab.set_xticks(x)
    # Bold, not upper-cased: these ticks head the bar panel directly below, and
    # the two halves of one figure must say "trained" the same way.
    ax_lab.set_xticklabels(
        [str(o) for o in columns], rotation=35, ha="right", fontsize=8,
    )
    for tick, odor, t in zip(ax_lab.get_xticklabels(), columns, is_trained):
        if t:
            tick.set_color(odor_bar_palette.trained_tick_color(odor))
            tick.set_weight("bold")
    ax_lab.tick_params(axis="x", length=0, pad=2)
    for sp in ax_lab.spines.values():
        sp.set_visible(False)

    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=SCORES)
    cb.set_label("Odor Response Score", fontsize=9, labelpad=4)
    cb.ax.tick_params(labelsize=8, pad=2)
    cb.ax.axhline(REACTION_BOUNDARY_Y, color="black", lw=2.2)
    cb.ax.annotate(
        "Reaction\nBoundary", xy=(0, REACTION_BOUNDARY_Y),
        xycoords=cb.ax.get_yaxis_transform(),
        xytext=(-8, 0), textcoords="offset points",
        fontsize=9.5, fontweight="bold", va="center", ha="right",
        linespacing=1.0,
    )


def _plot_bar_charts(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    overwrite: bool,
    cfg: Any = None,
    thawed: Sequence[str] = (),
    thaw_all: bool = False,
) -> None:
    """One figure per dataset: mean score per odor, with a per-fly score
    matrix band above it under the v2 protocol."""
    present = df["dataset_canon"].unique().tolist()
    ordered = [o for o in ODOR_ORDER if o in present]
    extras = sorted(o for o in present if o not in ODOR_ORDER)

    for odor in ordered + extras:
        is_v2 = get_protocol() == "v2" and "odor_col" in summary.columns
        sort_col = "odor_col" if is_v2 else "trial_num"
        sub = summary[summary["dataset_canon"] == odor].sort_values(
            sort_col, key=lambda s: s.str.casefold() if s.dtype == object else s
        )
        if sub.empty:
            continue

        label = DISPLAY_LABEL.get(odor, odor)
        png_path = out_dir / f"mean_score_{odor.replace(' ', '_')}.png"
        if not should_write(png_path, overwrite):
            continue
        if should_skip_frozen_figure(cfg, [odor], thawed=thawed, thaw_all=thaw_all):
            print(f"[FROZEN] Skipping figure (all contributors frozen): {png_path}")
            continue

        x = np.arange(len(sub))
        trained = _trained_label(odor)
        if is_v2:
            is_trained = sub["odor_col"].str.casefold().str.startswith(
                trained.casefold())
        else:
            is_trained = pd.Series([False] * len(sub), index=sub.index)
        # Legacy panels are keyed on trial number, not odor, so there is no
        # odor to look a colour up by: they keep the flat gray they always had.
        bar_colors = (
            single_cohort_bar_colors(sub["odor_col"], is_trained)
            if is_v2 and "odor_col" in sub.columns
            else ["#b0b0b0"] * len(sub)
        )

        matrix, fly_keys = (
            _per_fly_score_matrix(df, odor, sub["odor_col"].tolist())
            if is_v2 else (np.zeros((0, 0)), [])
        )
        show_matrix = is_v2 and matrix.size > 0

        with plt.rc_context(_RC_CONTEXT):
            if show_matrix:
                n_fly = len(fly_keys)
                fig = plt.figure(
                    figsize=(max(8, len(sub) * 1.15 + 2.5), n_fly * 0.16 + 5.0))
                # Three columns: plots | gutter | colorbar. ax_m and ax_b BOTH
                # live in column 0 so the matrix cells line up with the bars by
                # construction. Anchoring the colorbar to ax_m alone
                # (fig.colorbar(ax=ax_m)) shrinks only the matrix and silently
                # breaks that alignment.
                gs = gridspec.GridSpec(
                    2, 3,
                    width_ratios=[1.0, 0.125, 0.030],
                    # Floor keeps a 1-fly band readable; cap stops a large
                    # cohort dwarfing the bars.
                    height_ratios=[min(1.4, max(0.35, n_fly * 0.17)), 1.0],
                    hspace=0.50, wspace=0.0,
                )
                ax_m = fig.add_subplot(gs[0, 0])
                ax = fig.add_subplot(gs[1, 0], sharex=ax_m)
                cax = fig.add_subplot(gs[0, 2])
                _draw_score_matrix(
                    fig, ax_m, cax, matrix, len(sub), x,
                    sub["odor_col"].tolist(), is_trained,
                )
                ax_m.set_title(f"Mean Ordinal Score - {label}", fontsize=13,
                               weight="bold", pad=10)
            else:
                fig, ax = plt.subplots(figsize=(max(6, len(sub) * 0.7 + 2), 5))
                ax.set_title(f"Mean Ordinal Score - {label}", fontsize=13,
                             weight="bold")

            ax.bar(
                x, sub["mean_score"].values, yerr=sub["sem_score"].values,
                capsize=4, color=bar_colors,
                edgecolor="black" if is_v2 else "white",
                linewidth=0.75 if is_v2 else 0.5,
            )
            # Print the mean value above each bar (clear of its SEM whisker).
            for xi, mean_v, sem_v in zip(
                x, sub["mean_score"].values, sub["sem_score"].values
            ):
                ax.text(
                    xi,
                    mean_v + (sem_v if np.isfinite(sem_v) else 0.0) + 0.12,
                    f"{mean_v:.2f}",
                    ha="center", va="bottom", fontsize=8,
                )
            ax.set_xticks(x)
            if is_v2:
                ax.set_xticklabels(
                    [f"{o}\n(n={int(n)})" for o, n in zip(sub["odor_col"],
                                                          sub["n_flies"])],
                    fontsize=8, rotation=35, ha="right",
                )
            else:
                ax.set_xticklabels(
                    [f"T{int(t)}\n(n={int(n)})" for t, n in zip(sub["trial_num"],
                                                                sub["n_flies"])],
                    fontsize=9,
                )
            if show_matrix:
                # Same scale the matrix is coloured on.
                ax.set_ylim(SCORE_MIN, SCORE_MAX)
                ax.set_yticks(SCORES)
            else:
                ax.set_ylim(-1.5, 5.5)
            ax.set_ylabel(SCORE_Y_LABEL)
            ax.set_xlabel("Presented Odor" if is_v2 else "Testing Trial")
            ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
            if not show_matrix:
                plt.tight_layout()
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.close(fig)


def _draw_score_significance_brackets(
    ax: plt.Axes,
    x_positions: np.ndarray,
    bar_w: float,
    sub: pd.DataFrame,
) -> None:
    """Stars over the significant pairs only, clear of the value labels."""
    significance_brackets.draw(
        ax, x_positions, bar_w,
        [row.score_p_value for row in sub.itertuples(index=False)],
        fontsize=8,
    )

def single_cohort_bar_colors(odors, is_trained) -> list[str]:
    """Bar colours for a one-cohort score panel, in plotted order.

    These bars used to be dark blue for the trained odor and a flat gray for
    every other one, so a panel of eight odors carried two colours. They now
    take the same per-odor palette the paired panels use; the gray is reserved
    for the control series, where it actually means something.
    """
    return odor_bar_palette.training_bar_colors(odors, is_trained)


def _cohort_n_label(name: str, counts) -> tuple[str, bool]:
    """``("Training (n=12)", True)`` when one count covers the whole cohort.

    The second value says whether the legend now carries the n. When it does
    not — an odor dropped a fly, say — the caller keeps the per-bar n rather
    than letting a number that varies vanish from the figure.
    """
    uniq = {int(v) for v in pd.Series(counts).dropna()}
    if len(uniq) == 1:
        return f"{name} (n={uniq.pop()})", True
    return name, False


def plot_score_train_vs_control(
    ax: plt.Axes,
    rows: pd.DataFrame,
    *,
    title: str,
    label: str = "",
) -> tuple[np.ndarray, float]:
    """One training-vs-control score panel, styled like the pubfig.

    Training bars take each odor's palette colour, control bars are the single
    shared gray, and the trained odor is marked by a bold tick rather than by
    upper-casing its name in dark blue. Returns ``(x, bar_w)`` so the caller can
    place significance brackets over the same geometry.
    """
    x = np.arange(len(rows))
    bar_w = 0.35

    train_vals = rows["mean_score_train"].fillna(0.0).to_numpy(float)
    ctrl_vals = rows["mean_score_ctrl"].fillna(0.0).to_numpy(float)
    train_err = rows["sem_score_train"].fillna(0.0).to_numpy(float)
    ctrl_err = rows["sem_score_ctrl"].fillna(0.0).to_numpy(float)

    y_top = 0.0
    if len(rows):
        y_top = float(
            np.max(
                np.concatenate([
                    np.maximum(train_vals + train_err, 0.0),
                    np.maximum(ctrl_vals + ctrl_err, 0.0),
                ])
            )
        )

    train_colors = odor_bar_palette.training_bar_colors(
        rows["odor"], rows["is_trained"]
    )
    ax.bar(
        x - bar_w / 2, train_vals, width=bar_w, yerr=train_err, capsize=4,
        color=train_colors, edgecolor="black", linewidth=0.75,
    )
    ax.bar(
        x + bar_w / 2, ctrl_vals, width=bar_w, yerr=ctrl_err, capsize=4,
        color=odor_bar_palette.CTRL_COLOR, edgecolor="black", linewidth=0.75,
    )

    # Print the mean value above each bar (clear of its SEM whisker).
    for xi, mean_v, sem_v in zip(x - bar_w / 2, train_vals, train_err):
        ax.text(xi, mean_v + sem_v + 0.12, f"{mean_v:.2f}",
                ha="center", va="bottom", fontsize=7, rotation=90)
    for xi, mean_v, sem_v in zip(x + bar_w / 2, ctrl_vals, ctrl_err):
        ax.text(xi, mean_v + sem_v + 0.12, f"{mean_v:.2f}",
                ha="center", va="bottom", fontsize=7, rotation=90)

    n_train = rows["n_flies_train"].fillna(0).astype(int)
    n_ctrl = rows["n_flies_ctrl"].fillna(0).astype(int)
    train_key, train_has_n = _cohort_n_label("Training", n_train)
    ctrl_key, ctrl_has_n = _cohort_n_label("Control", n_ctrl)

    # The n belongs in the legend. It only stays on the ticks when it varies
    # by odor, where one legend number would be wrong.
    ax.set_xticks(x)
    if train_has_n and ctrl_has_n:
        tick_labels = [str(odor) for odor in rows["odor"]]
    else:
        tick_labels = [f"{odor}\n(n={nt}/{nc})"
                       for odor, nt, nc in zip(rows["odor"], n_train, n_ctrl)]
    ax.set_xticklabels(tick_labels, rotation=35, ha="right")
    for tick, odor, is_trained in zip(
        ax.get_xticklabels(), rows["odor"], rows["is_trained"]
    ):
        if bool(is_trained):
            tick.set_color(odor_bar_palette.trained_tick_color(odor))
            tick.set_weight("bold")

    ax.set_ylabel(SCORE_Y_LABEL)
    ax.set_xlabel("Testing Trial / Presented Odor")
    ax.set_title(title, fontsize=13, weight="bold")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylim(-1.5, max(6.1, y_top + 1.0))
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    odor_bar_palette.add_training_legend(
        ax, train_colors, ctrl_color=odor_bar_palette.CTRL_COLOR,
        train_label=train_key, ctrl_label=ctrl_key, loc="upper right",
    )
    return x, bar_w


def _plot_training_vs_control_bars(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    overwrite: bool,
    cfg: Any = None,
    thawed: Sequence[str] = (),
    thaw_all: bool = False,
) -> None:
    if summary.empty:
        return

    all_train_ds = set(summary["training_dataset"].unique()) if not summary.empty else set()
    for train_ds in all_train_ds:
        sub = summary[summary["training_dataset"] == train_ds].copy()
        if sub.empty:
            continue

        label = DISPLAY_LABEL.get(train_ds, train_ds)
        png_path = out_dir / f"mean_score_train_vs_ctrl_{train_ds.replace(' ', '_')}.png"
        if not should_write(png_path, overwrite):
            continue
        # This figure draws Training beside its paired Control -- the exact
        # "frozen Control beside live Training" case: both datasets must
        # count, or adding flies to a live Training silently fails to appear.
        contributing = {train_ds, *sub["control_dataset"].dropna().unique().tolist()}
        if should_skip_frozen_figure(cfg, contributing, thawed=thawed, thaw_all=thaw_all):
            print(f"[FROZEN] Skipping figure (all contributors frozen): {png_path}")
            continue

        if "trial_num" in sub.columns:
            sub = sub.sort_values(["trial_num", "odor"]).reset_index(drop=True)
        else:
            sub = sub.sort_values("odor", key=lambda s: s.str.casefold()).reset_index(drop=True)
        with plt.rc_context(_RC_CONTEXT):
            fig, ax = plt.subplots(figsize=(max(7, len(sub) * 1.0 + 2), 5.5))
            x, bar_w = plot_score_train_vs_control(
                ax,
                sub,
                title=f"Mean Model Score - {label} (Training vs Control, Trials Separate)",
                label=label,
            )

            _draw_score_significance_brackets(ax, x, bar_w, sub)

            plt.tight_layout()
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.close(fig)


def _plot_heatmap(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    overwrite: bool,
    cfg: Any = None,
    thawed: Sequence[str] = (),
    thaw_all: bool = False,
) -> None:
    """Heatmap: datasets (rows) x testing numbers (columns), colored by mean score."""
    png_path = out_dir / "mean_score_heatmap.png"
    if not should_write(png_path, overwrite):
        return
    # One figure pooling every dataset in `summary`: skip only when every one
    # of them is frozen for figures, or a live dataset's row would silently
    # never appear.
    contributing = set(summary["dataset_canon"].unique()) if "dataset_canon" in summary.columns else set()
    if should_skip_frozen_figure(cfg, contributing, thawed=thawed, thaw_all=thaw_all):
        print(f"[FROZEN] Skipping figure (all contributors frozen): {png_path}")
        return

    col_key = "odor_col" if "odor_col" in summary.columns and get_protocol() == "v2" else "trial_num"
    pivot = summary.pivot_table(
        index="dataset_canon",
        columns=col_key,
        values="mean_score",
        aggfunc="first",
    )

    # Order rows by ODOR_ORDER
    ordered = [o for o in ODOR_ORDER if o in pivot.index]
    extras = sorted(o for o in pivot.index if o not in ODOR_ORDER)
    pivot = pivot.reindex(ordered + extras)

    # Sort columns
    if col_key == "odor_col":
        pivot = pivot.reindex(sorted(pivot.columns, key=str.casefold), axis=1)
    else:
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    display_idx = [DISPLAY_LABEL.get(o, o) for o in pivot.index]

    with plt.rc_context(_RC_CONTEXT):
        fig, ax = plt.subplots(
            figsize=(max(8, len(pivot.columns) * 0.8 + 3), max(4, len(pivot) * 0.5 + 2))
        )
        norm = TwoSlopeNorm(vmin=-1, vcenter=1.5, vmax=5)
        im = ax.imshow(
            pivot.values,
            cmap="RdYlGn",
            norm=norm,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_xticks(range(len(pivot.columns)))
        if get_protocol() == "v2":
            ax.set_xticklabels([str(c) for c in pivot.columns], fontsize=8, rotation=35, ha="right")
        else:
            ax.set_xticklabels([f"T{int(c)}" for c in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(display_idx)))
        ax.set_yticklabels(display_idx, fontsize=9)
        ax.set_xlabel("Presented Odor" if get_protocol() == "v2" else "Testing Trial")
        ax.set_title("Mean Ordinal Score by Dataset and Trial", fontsize=13, weight="bold")

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Mean Score (-1 to 5)")

        # Annotate cells
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(
                        j, i, f"{val:.1f}",
                        ha="center", va="center", fontsize=8,
                        color="black" if 0.5 < val < 4 else "white",
                    )

        plt.tight_layout()
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_score_pair(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    overwrite: bool,
    cfg: Any = None,
    thawed: Sequence[str] = (),
    thaw_all: bool = False,
) -> None:
    """One figure per training dataset: control score matrix LEFT, training
    RIGHT, sharing columns, cell height, and a single key."""
    if get_protocol() != "v2" or "odor_col" not in df.columns:
        return
    present = sorted(set(df["dataset_canon"]))
    for train_ds, ctrl_ds in _auto_pairs(present).items():
        if train_ds not in present or ctrl_ds not in present:
            continue
        png_path = out_dir / f"mean_score_pair_{train_ds.replace(' ', '_')}.png"
        if not should_write(png_path, overwrite):
            continue
        # Same pairing subtlety as _plot_training_vs_control_bars: a frozen
        # Control beside a live Training must still redraw.
        if should_skip_frozen_figure(cfg, [train_ds, ctrl_ds], thawed=thawed, thaw_all=thaw_all):
            print(f"[FROZEN] Skipping figure (all contributors frozen): {png_path}")
            continue
        pair_df = df[df["dataset_canon"].isin((train_ds, ctrl_ds))]
        columns = sorted(pair_df["odor_col"].unique(), key=str.casefold)
        train_m, train_flies = _per_fly_score_matrix(df, train_ds, columns)
        ctrl_m, ctrl_flies = _per_fly_score_matrix(df, ctrl_ds, columns)
        if not len(train_flies) or not len(ctrl_flies):
            continue
        n_max = max(len(train_flies), len(ctrl_flies))
        cmap, norm = _score_cmap()
        trained = _trained_label(train_ds)
        is_trained = [str(c).casefold().startswith(trained.casefold()) for c in columns]

        with plt.rc_context(_RC_CONTEXT):
            fig = plt.figure(figsize=(max(9, len(columns) * 1.5 + 3),
                                      n_max * 0.26 + 4.0))
            # Colorbar owns its own column: fig.colorbar(ax=...) would shrink one
            # panel and break the shared-cell-height guarantee.
            gs = gridspec.GridSpec(1, 4, width_ratios=[1.0, 1.0, 0.10, 0.035],
                                   wspace=0.10)
            ax_c = fig.add_subplot(gs[0, 0])
            ax_t = fig.add_subplot(gs[0, 1])
            cax = fig.add_subplot(gs[0, 3])
            for ax, mat, flies, title in (
                (ax_c, ctrl_m, ctrl_flies, f"Control ({len(ctrl_flies)} Flies)"),
                (ax_t, train_m, train_flies, f"Training ({len(train_flies)} Flies)"),
            ):
                # CELL HEIGHT PARITY -- get this exactly right.
                # extent's y-span must match THIS matrix's own row count
                # (mat.shape[0]), NOT n_max: imshow STRETCHES the image to fill
                # the extent, so giving a 2-row matrix a 3-row extent renders its
                # cells 1.5x too tall. Equal cell height comes from set_ylim
                # sharing n_max across both panels while each image keeps its own
                # true extent -- the shorter panel then genuinely ends early.
                # (This exact mistake shipped in Task 2 and was caught in review:
                # control cells measured 1.54in vs training 1.03in.)
                ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, norm=norm,
                          aspect="auto", interpolation="nearest",
                          extent=(-0.5, len(columns) - 0.5,
                                  mat.shape[0] - 0.5, -0.5))
                ax.set_ylim(n_max - 0.5, -0.5)
                for j in range(len(columns) + 1):
                    ax.axvline(j - 0.5, color="white", lw=1.6)
                for i in range(n_max + 1):
                    ax.axhline(i - 0.5, color="white", lw=1.6)
                ax.set_yticks([])
                ax.set_xticks(np.arange(len(columns)))
                ax.set_xticklabels(
                    [str(c).upper() if t else str(c)
                     for c, t in zip(columns, is_trained)],
                    rotation=35, ha="right", fontsize=8)
                for tick, t in zip(ax.get_xticklabels(), is_trained):
                    if t:
                        tick.set_color("#1a3a6b")
                        tick.set_weight("bold")
                ax.set_title(title, fontsize=12, weight="bold")
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                              cax=cax, ticks=SCORES)
            cb.set_label("Odor Response Score", fontsize=9, labelpad=4)
            cb.ax.tick_params(labelsize=8, pad=2)
            cb.ax.axhline(REACTION_BOUNDARY_Y, color="black", lw=2.2)
            fig.suptitle(
                f"Per-Fly Odor Response - {DISPLAY_LABEL.get(train_ds, train_ds)}",
                fontsize=14, weight="bold")
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _genotype_score_groups(
    df: pd.DataFrame, out_dir: Path
) -> list[tuple[pd.DataFrame, Path]]:
    """Split scores into per-genotype (df, out_dir) groups.

    When the data spans more than one genotype (``fly_type``), each genotype's
    flies get their own ``<out_dir>/<genotype>/`` subfolder so genotypes are
    never pooled in one summary. Single-genotype data — and legacy predictions
    with no ``fly_type`` column — return one group at ``out_dir`` (unchanged).
    """
    if "fly_type" not in df.columns:
        return [(df, out_dir)]
    genos = sorted({str(g).strip() for g in df["fly_type"] if str(g).strip()})
    if len(genos) <= 1:
        return [(df, out_dir)]
    return [
        (df[df["fly_type"].astype(str).str.strip() == g].copy(), out_dir / _safe_dirname(g))
        for g in genos
    ]


def _summarise_and_plot(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    overwrite: bool,
    cfg: Any = None,
    thawed: Sequence[str] = (),
    thaw_all: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _compute_summary(df)

    # Export CSV
    csv_out = out_dir / "score_summary_by_odor_testing.csv"
    if should_write(csv_out, overwrite):
        summary.to_csv(csv_out, index=False, float_format="%.4f")
        print(f"[score_summary] Wrote {csv_out}")

    train_ctrl_summary = _compute_training_vs_control_summary(df)
    if not train_ctrl_summary.empty:
        train_ctrl_csv = out_dir / "score_summary_train_vs_control.csv"
        if should_write(train_ctrl_csv, overwrite):
            train_ctrl_summary.to_csv(train_ctrl_csv, index=False, float_format="%.4f")
            print(f"[score_summary] Wrote {train_ctrl_csv}")

    _plot_bar_charts(df, summary, out_dir, overwrite=overwrite, cfg=cfg, thawed=thawed, thaw_all=thaw_all)
    _plot_heatmap(summary, out_dir, overwrite=overwrite, cfg=cfg, thawed=thawed, thaw_all=thaw_all)
    _plot_training_vs_control_bars(
        train_ctrl_summary, out_dir, overwrite=overwrite, cfg=cfg, thawed=thawed, thaw_all=thaw_all
    )
    _plot_score_pair(df, out_dir, overwrite=overwrite, cfg=cfg, thawed=thawed, thaw_all=thaw_all)
    print(f"[score_summary] Plots saved to {out_dir}")


def generate_score_summary(
    csv_path: Path,
    out_dir: Path,
    *,
    overwrite: bool = True,
    non_reactive_threshold: float | None = None,
    flagged_flies_csv: str = "",
    cfg: Any = None,
    thawed: Sequence[str] = (),
    thaw_all: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_scores(
        csv_path,
        threshold=non_reactive_threshold,
        flagged_flies_csv=flagged_flies_csv,
    )
    if df.empty:
        print("[score_summary] No data after filtering; nothing to plot.")
        return

    # Sort flies by genotype: one full score-summary set per genotype when the
    # data holds more than one (else a single pooled set, as before).
    for sub_df, sub_out in _genotype_score_groups(df, out_dir):
        if sub_df.empty:
            continue
        _summarise_and_plot(
            sub_df, sub_out, overwrite=overwrite, cfg=cfg, thawed=thawed, thaw_all=thaw_all
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-path", type=Path, required=True,
        help="Path to predictions CSV containing a 'score' column.",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="Directory for exported plots and summary CSV.",
    )
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument(
        "--non-reactive-threshold", type=float, default=None,
        help="Span threshold for non-reactive fly exclusion.",
    )
    parser.add_argument(
        "--flagged-flies-csv", type=str, default="",
        help="Path to flagged-flies CSV for exclusion.",
    )
    parser.add_argument(
        "--protocol", type=str, default="v2", choices=["v2", "legacy"],
        help="Protocol version (default: v2).",
    )
    parser.add_argument(
        "--config", type=str, default="",
        help="Pipeline config YAML; used to load dataset_overrides.odor_remap "
             "and dataset_overrides.freeze.figures.",
    )
    parser.add_argument(
        "--thaw", action="append", default=[], metavar="DATASET",
        help=(
            "Ignore freeze.figures for DATASET this run (repeatable). Does "
            "not edit config."
        ),
    )
    parser.add_argument(
        "--thaw-all", action="store_true",
        help="Ignore every dataset's freeze.figures this run.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    set_protocol(args.protocol)
    try:
        from fbpipe.figure_export import maybe_install_from_env
        maybe_install_from_env()
    except Exception:  # noqa: BLE001 — sidecar is optional
        pass
    # This process runs as a subprocess (run_workflows.py invokes it with
    # --config), so it does not share the parent's `settings` object -- it
    # must load config itself. `settings` doubles as the `cfg` argument to
    # should_skip_frozen_figure (duck-typed via getattr on
    # `dataset_overrides`, see fbpipe.freeze.freeze_flags), so a load failure
    # must leave it None -- that reads as "no freeze info", never as "skip".
    settings = None
    if args.config:
        try:
            from fbpipe.config import load_settings
            from scripts.analysis.envelope_visuals import set_dataset_odor_remap
            settings = load_settings(args.config)
            remap = {
                str(ds): dict(ov.odor_remap)
                for ds, ov in settings.dataset_overrides.items()
                if getattr(ov, "odor_remap", None)
            }
            if remap:
                set_dataset_odor_remap(remap)
        except Exception as exc:  # noqa: BLE001 — defensive
            print(f"[WARN] Failed to load odor_remap from {args.config}: {exc}")
            settings = None
    generate_score_summary(
        csv_path=args.csv_path,
        out_dir=args.out_dir,
        overwrite=args.overwrite,
        non_reactive_threshold=args.non_reactive_threshold,
        flagged_flies_csv=args.flagged_flies_csv,
        cfg=settings,
        thawed=tuple(args.thaw or ()),
        thaw_all=bool(args.thaw_all),
    )


if __name__ == "__main__":
    main()
