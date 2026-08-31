"""Pre-test vs post-test PER for the ``*-Sensitivity-*`` cohorts.

The sensitivity protocol runs a naive 7-odor panel, then training, then the same
7 odors again on the same fly:

    pretest_1..7  ->  30 min rest  ->  training_1..6  ->  testing_1..7

so the pre-test IS the control, measured inside each fly. That is a stronger
design than a separate control cohort, and it demands paired statistics:

  * response rate (responded / did not) -> **McNemar's exact test**;
  * ordinal PER score                   -> **Wilcoxon signed-rank**.

Both are paired per (fly, odor), so **n counts flies** — the fly is the unit
that is paired. This is not the "never average an average" trial-pooling case
used for unpaired cohort means: each fly contributes one pre value and one post
value per odor, and every fly is weighted once. Axis labels say ``n=N flies`` so
the difference is visible on the figure.

Odor order is randomised between the two panels, so pairing is on the ODOR, not
the trial index — ``pretest_1`` is very often a different odor than
``testing_1``.

p-values are uncorrected, matching the convention in ``score_summary.py``.

Usage
-----
    python3 scripts/analysis/pretest_vs_test_comparison.py \
        --predictions-csv .../model_predictions.csv \
        --out-root .../Results/New-Opto-Fly-Figures/Pre-vs-Post-Sensitivity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis import odor_bar_palette, significance_brackets  # noqa: E402
from scripts.analysis.envelope_visuals import (  # noqa: E402
    NON_REACTIVE_SPAN_PX,
    _safe_dirname,
    _trained_label,
)
from scripts.analysis.per_axis_labels import PERCENT_Y_LABEL, SCORE_Y_LABEL  # noqa: E402
from scripts.analysis.score_summary import _load_scores, _score_cmap  # noqa: E402

#: Ordinal score at or above which a trial counts as a response. Matches
#: ``reaction_prediction.binary_threshold`` in config_new.yaml.
BINARY_THRESHOLD = 2

#: Wilcoxon cannot reach p < 0.05 below this many non-zero pairs, so reporting a
#: number there would only mislead.
MIN_WILCOXON_PAIRS = 6

# ── Palette ────────────────────────────────────────────────────────────────
# Validated with the dataviz skill's scripts/validate_palette.js (light mode):
# ALL CHECKS PASS — lightness band, chroma floor, CVD separation (ΔE 24.7
# protan / 32.7 tritan), normal-vision floor (33.6) and 3:1 contrast.
# The obvious alternative, gray #b0b0b0 + navy #1a3a6b (the train-vs-control
# convention), FAILS three of those: navy sits outside the lightness band, gray
# has zero chroma so it reads as "no series", and gray is only 2.11:1 on white.
PRE_COLOR = "#2a78d6"   # blue  — the naive panel, before training
POST_COLOR = "#eb6834"  # orange — after training; the warmer hue carries the result
PRE_LABEL = "Pre-test (naive)"
POST_LABEL = "Post-training"

# Ink, never a series colour: values and labels wear text tokens so identity is
# carried by the mark beside them, not by coloured type.
INK = "#1a1a19"
INK_MUTED = "#6b6b66"
GRID = "#e4e4e0"

# ── Fixed axes ─────────────────────────────────────────────────────────────
# Every panel, every cohort, always. Auto-scaled axes make two cohorts printed
# side by side look like different effect sizes when they are not, and a bar's
# height stops being comparable between panels. The score range is the model's
# full ordinal scale (-1 is a real output, seen in 3Oct-Sensitivity).
SCORE_YLIM = (-1, 5)
SCORE_YTICKS = (-1, 0, 1, 2, 3, 4, 5)
RATE_YLIM = (0, 100)
RATE_YTICKS = (0, 20, 40, 60, 80, 100)

_RC_CONTEXT = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    # Journals reject Type-3 fonts; 42 keeps text as editable TrueType.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": INK_MUTED,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
}


def _apply_score_axis(ax) -> None:
    """Pin the PER-score axis to the model's full ordinal range."""
    ax.set_ylim(*SCORE_YLIM)
    ax.set_yticks(list(SCORE_YTICKS))
    ax.set_ylabel(SCORE_Y_LABEL)
    _recessive_grid(ax)
    # The axis floor is -1 but bars are anchored at 0, so without an explicit
    # rule at zero the bars read as floating above the bottom of the plot.
    # (The rate axis needs none — its floor IS zero, where the spine already is.)
    ax.axhline(0.0, color=INK_MUTED, linewidth=0.8, zorder=0.5)


def _apply_rate_axis(ax) -> None:
    """Pin the response-rate axis to the full percentage range."""
    ax.set_ylim(*RATE_YLIM)
    ax.set_yticks(list(RATE_YTICKS))
    ax.set_ylabel(PERCENT_Y_LABEL)
    _recessive_grid(ax)


def _zero_label_offset(kind: str) -> float:
    """Where a "0" label sits: just clear of the baseline, in axis units."""
    lo, hi = SCORE_YLIM if kind == "score" else RATE_YLIM
    return 0.012 * (hi - lo)


def _recessive_grid(ax) -> None:
    """Horizontal rules only, behind the data, barely there."""
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.xaxis.grid(False)


# ---------------------------------------------------------------------------
# Loading and pairing
# ---------------------------------------------------------------------------


def load_paired_scores(
    predictions_csv: Path | str,
    *,
    threshold: float | None = None,
    flagged_flies_csv: str = "",
) -> pd.DataFrame:
    """One row per (fly, odor) carrying both the naive and the post score.

    The two phases are loaded separately — see ``_load_scores`` — then joined on
    the odor. A (fly, odor) with only one of the two halves has no within-subject
    comparison and is dropped; the count lands in ``df.attrs["n_unpaired"]`` so
    the caller can report it rather than absorb it silently.
    """

    # _load_scores gates the flagged-flies table behind `threshold is not
    # None`, so passing the CSV alone silently drops nobody. When a truth table
    # is actually present, open that gate: compute_non_reactive_flags then
    # excludes on FLY-State != 1 and ignores the threshold entirely.
    if (
        threshold is None
        and flagged_flies_csv
        and Path(flagged_flies_csv).exists()
    ):
        threshold = NON_REACTIVE_SPAN_PX

    kwargs = dict(threshold=threshold, flagged_flies_csv=flagged_flies_csv)
    key = ["dataset_canon", "fly", "fly_number", "odor_display"]

    def _phase(name: str) -> pd.DataFrame:
        # Absent phase is normal, not an error: only the *-Sensitivity-*
        # cohorts run a naive panel, and model_predictions.csv carries no
        # pretest rows at all until a full run produces them.
        try:
            return _load_scores(Path(predictions_csv), trial_types=(name,), **kwargs)
        except RuntimeError as exc:
            print(f"[pre-vs-post] {exc}")
            return pd.DataFrame(columns=key + ["score"])

    pre = _phase("pretest")
    post = _phase("testing")

    pre_slim = pre[key + ["score"]].rename(columns={"score": "score_pre"})
    post_slim = post[key + ["score"]].rename(columns={"score": "score_post"})

    # One presentation per odor per phase in this protocol; guard anyway so a
    # protocol change shows up as dropped duplicates rather than a row explosion.
    pre_slim = pre_slim.drop_duplicates(subset=key, keep="first")
    post_slim = post_slim.drop_duplicates(subset=key, keep="first")

    merged = pre_slim.merge(post_slim, on=key, how="inner")
    n_unpaired = (len(pre_slim) + len(post_slim)) - 2 * len(merged)

    merged = merged.rename(columns={"odor_display": "odor"})
    merged = merged.sort_values(["dataset_canon", "fly", "fly_number", "odor"])
    merged = merged.reset_index(drop=True)
    merged.attrs["n_unpaired"] = int(n_unpaired)
    return merged


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------


def mcnemar_p(pre: Sequence[bool], post: Sequence[bool]) -> float | None:
    """Two-sided exact McNemar p for paired binary outcomes.

    Only discordant pairs carry information: ``b`` flies that gained a response
    and ``c`` that lost one. With no discordant pairs the test is undefined —
    None, not 1.0, so the figure draws no bracket rather than a misleading one.
    """

    pre_arr = np.asarray(pre, dtype=bool)
    post_arr = np.asarray(post, dtype=bool)
    if pre_arr.size == 0 or pre_arr.size != post_arr.size:
        return None

    b = int(np.sum(~pre_arr & post_arr))
    c = int(np.sum(pre_arr & ~post_arr))
    if b + c == 0:
        return None
    return float(binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)


def wilcoxon_p(pre: Sequence[float], post: Sequence[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank p, or None when the test cannot run."""

    pre_arr = np.asarray(pre, dtype=float)
    post_arr = np.asarray(post, dtype=float)
    if pre_arr.size == 0 or pre_arr.size != post_arr.size:
        return None

    diff = post_arr - pre_arr
    diff = diff[np.isfinite(diff)]
    non_zero = diff[diff != 0]
    if non_zero.size < MIN_WILCOXON_PAIRS:
        return None
    try:
        return float(wilcoxon(non_zero, alternative="two-sided").pvalue)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def per_odor_summary(
    paired: pd.DataFrame, *, binary_threshold: int = BINARY_THRESHOLD
) -> pd.DataFrame:
    """Per-odor pre/post means, response percentages and paired p-values."""

    if paired.empty:
        return pd.DataFrame(columns=[
            "odor", "n_flies", "mean_pre", "sem_pre", "mean_post", "sem_post",
            "pct_pre", "pct_post", "p_score", "p_rate",
        ])

    rows = []
    for odor, grp in paired.groupby("odor", sort=True):
        pre = grp["score_pre"].to_numpy(dtype=float)
        post = grp["score_post"].to_numpy(dtype=float)
        resp_pre = pre >= binary_threshold
        resp_post = post >= binary_threshold
        rows.append({
            "odor": odor,
            "n_flies": len(grp),
            "mean_pre": float(np.mean(pre)),
            "sem_pre": float(_sem(pre)),
            "mean_post": float(np.mean(post)),
            "sem_post": float(_sem(post)),
            "pct_pre": 100.0 * float(np.mean(resp_pre)),
            "pct_post": 100.0 * float(np.mean(resp_post)),
            "p_score": wilcoxon_p(pre, post),
            "p_rate": mcnemar_p(resp_pre, resp_post),
        })
    return pd.DataFrame(rows)


def _sem(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _n_label(summary: pd.DataFrame) -> str:
    counts = sorted(set(int(v) for v in summary["n_flies"]))
    if len(counts) == 1:
        return f"n={counts[0]} flies"
    return f"n={min(counts)}–{max(counts)} flies"


def build_grouped_bars(summary, *, cohort, kind, p_key):
    """Pre vs post bars, one group per odor. Returns ``(fig, ax)`` unsaved.

    ``kind`` is "score" or "rate"; it selects the value columns AND the fixed
    axis, so the two can never disagree.
    """

    is_score = kind == "score"
    pre_key, post_key = ("mean_pre", "mean_post") if is_score else ("pct_pre", "pct_post")
    err_keys = ("sem_pre", "sem_post") if is_score else None

    x = np.arange(len(summary))
    bar_w = 0.38
    # 2 px of surface between the paired bars, per the mark spec: the gap does
    # the separating, so the fills need no heavy stroke.
    gap = 0.012

    fig, ax = plt.subplots(figsize=(1.35 * max(len(summary), 1) + 2.4, 4.2))
    for key, err_key, offset, color, label in (
        (pre_key, err_keys[0] if err_keys else None, -(bar_w / 2 + gap), PRE_COLOR, PRE_LABEL),
        (post_key, err_keys[1] if err_keys else None, +(bar_w / 2 + gap), POST_COLOR, POST_LABEL),
    ):
        values = summary[key].to_numpy(dtype=float)
        ax.bar(
            x + offset, values, bar_w,
            yerr=(summary[err_key] if err_key else None),
            color=color, linewidth=0, label=label,
            error_kw=dict(ecolor=INK_MUTED, elinewidth=0.9, capsize=2.5, capthick=0.9),
        )
        # Selective direct labels: ONLY the zero bars. A zero-height bar is
        # otherwise indistinguishable from "no data for this odor", which is a
        # different claim. Every other bar is read off the axis.
        for xi, value in zip(x + offset, values):
            if value == 0:
                ax.text(
                    xi, _zero_label_offset(kind), "0", ha="center", va="bottom",
                    fontsize=7.5, color=INK_MUTED,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(summary["odor"], rotation=28, ha="right")
    ax.set_xlim(-0.6, len(summary) - 0.4)
    ax.set_xlabel("")
    _color_trained_tick(ax, cohort, summary["odor"])
    (_apply_score_axis if is_score else _apply_rate_axis)(ax)

    metric = "Mean PER score" if is_score else "PER response rate"
    ax.set_title(
        f"{cohort} — {metric}, pre-test vs post-training\n"
        f"{_n_label(summary)}, paired within fly",
        loc="left", color=INK, pad=8,
    )
    ax.legend(loc="upper right", ncol=2, handlelength=1.1, handleheight=0.9,
              borderpad=0.2, columnspacing=1.0)

    # expand=False: the fixed axis is the point — a significant result must not
    # silently rescale the panel and break comparability with its neighbours.
    significance_brackets.draw(
        ax, x, 2 * bar_w, list(summary[p_key]), expand=False
    )
    return fig, ax


def _grouped_bars(summary, *, cohort, kind, p_key, path):
    with plt.rc_context(_RC_CONTEXT):
        fig, _ax = build_grouped_bars(summary, cohort=cohort, kind=kind, p_key=p_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
    return path


def _color_trained_tick(ax, cohort: str, odors: Sequence[str]) -> None:
    """Tint the CS+ odor's tick so the odor that should have changed stands out."""
    try:
        trained = _trained_label(cohort)
    except Exception:
        return
    if not trained:
        return
    for label, odor in zip(ax.get_xticklabels(), odors):
        if str(odor).casefold().startswith(str(trained).casefold()):
            label.set_color(odor_bar_palette.trained_tick_color(trained))
            label.set_fontweight("bold")


def _trained_odor_row(summary: pd.DataFrame, cohort: str) -> pd.Series | None:
    try:
        trained = _trained_label(cohort)
    except Exception:
        return None
    if not trained:
        return None
    for _, row in summary.iterrows():
        if str(row["odor"]).casefold().startswith(str(trained).casefold()):
            return row
    return None


def _spotlight(summary, *, cohort, path):
    """The CS+ odor alone: score and response rate, side by side."""
    row = _trained_odor_row(summary, cohort)
    if row is None:
        return None

    with plt.rc_context(_RC_CONTEXT):
        fig, axes = plt.subplots(1, 2, figsize=(5.6, 4.0))
        panels = (
            (axes[0], ("mean_pre", "mean_post"), ("sem_pre", "sem_post"),
             row["p_score"], _apply_score_axis),
            (axes[1], ("pct_pre", "pct_post"), None, row["p_rate"], _apply_rate_axis),
        )
        for ax, (pre_k, post_k), err_k, p, apply_axis in panels:
            errs = [row[err_k[0]], row[err_k[1]]] if err_k else None
            ax.bar(
                [0, 1], [row[pre_k], row[post_k]], 0.55,
                yerr=errs, color=[PRE_COLOR, POST_COLOR], linewidth=0,
                error_kw=dict(ecolor=INK_MUTED, elinewidth=0.9, capsize=2.5,
                              capthick=0.9),
            )
            ax.set_xticks([0, 1])
            ax.set_xticklabels([PRE_LABEL.replace(" (naive)", ""), "Post-training"])
            ax.set_xlim(-0.6, 1.6)
            apply_axis(ax)
            significance_brackets.draw(ax, [0.5], 1.0, [p], expand=False)

        fig.suptitle(
            f"{cohort} — {row['odor']} (CS+)\n"
            f"n={int(row['n_flies'])} flies, paired within fly",
            x=0.01, ha="left", color=INK,
        )
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
    return path


#: Half-width of the slope-plot jitter, in x units (the categorical axis).
#: Jitter goes on X, never on the score axis: nudging a score of 0 down to -0.16
#: would draw it as negative, and -1 is a REAL value on this scale.
SLOPE_JITTER = 0.055


def slope_jitter(n: int) -> np.ndarray:
    """Per-fly x offsets that separate flies sharing an identical trajectory.

    PER scores are integers, so several flies routinely land on exactly the same
    (pre, post) pair and collapse into one visible line — contradicting the
    figure's "one line per fly" caption. Offsets are evenly spaced and assigned
    by position, so they are deterministic: re-running the pipeline reproduces
    the same figure. Every fly's plotted score stays exactly its measured score.
    """

    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.zeros(1, dtype=float)
    return np.linspace(-SLOPE_JITTER, SLOPE_JITTER, n)


def _slopes(paired, summary, *, cohort, path):
    """One line per fly, pre -> post on the CS+ odor.

    A mean hides which flies actually learned; at these n that is the whole
    story. Direction is named in the legend, so it is never colour-alone.
    """
    row = _trained_odor_row(summary, cohort)
    if row is None:
        return None
    sub = paired[paired["odor"] == row["odor"]]
    if sub.empty:
        return None

    with plt.rc_context(_RC_CONTEXT):
        fig, ax = plt.subplots(figsize=(3.4, 4.2))
        raw_pre = sub["score_pre"].to_numpy(dtype=float)
        raw_post = sub["score_post"].to_numpy(dtype=float)
        dx = slope_jitter(len(sub))

        seen_up = seen_flat = False
        for offset, y0, y1, r0, r1 in zip(dx, raw_pre, raw_post, raw_pre, raw_post):
            rose = r1 > r0
            label = None
            if rose and not seen_up:
                label, seen_up = "increased", True
            elif not rose and not seen_flat:
                label, seen_flat = "unchanged / decreased", True
            ax.plot(
                [0 + offset, 1 + offset], [y0, y1],
                marker="o", markersize=4.5, linewidth=1.2,
                color=POST_COLOR if rose else INK_MUTED, alpha=0.9, zorder=2,
                markeredgecolor="white", markeredgewidth=0.6, label=label,
                # A score of 5 or -1 sits exactly on the axis limit; without
                # this its marker is sliced in half. The limit stays hard —
                # only the mark overhangs.
                clip_on=False,
            )
        ax.plot([0, 1], [raw_pre.mean(), raw_post.mean()], marker="s",
                markersize=7, linewidth=2.4, color=INK, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8, label="mean",
                clip_on=False)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pre-test", "Post-training"])
        ax.set_xlim(-0.3, 1.3)
        _apply_score_axis(ax)
        ax.set_title(
            f"{cohort} — {row['odor']} (CS+)\nn={len(sub)} flies, one line per fly",
            loc="left", color=INK, pad=8,
        )
        ax.legend(loc="upper left", handlelength=1.4, borderpad=0.2)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
    return path


def _with_fly_id(paired: pd.DataFrame) -> pd.DataFrame:
    out = paired.copy()
    out["fly_id"] = out["fly"].astype(str) + " #" + out["fly_number"].astype(str)
    return out


def heatmap_fly_order(paired: pd.DataFrame) -> list[str]:
    """Fly rows ordered by fly NUMBER, not by the string form of it.

    Plain lexical sorting puts ``#10`` between ``#1`` and ``#2``, which on a
    per-fly matrix reads as a data error. ``fly_number`` is UNKNOWN for
    sidecar-less batches, so non-numeric values sort last rather than raising.
    """

    def sort_key(fly_id: str) -> tuple[int, float, str]:
        tail = str(fly_id).rsplit("#", 1)[-1].strip()
        try:
            return (0, float(tail), str(fly_id))
        except ValueError:
            return (1, 0.0, str(fly_id))

    return sorted(_with_fly_id(paired)["fly_id"].unique(), key=sort_key)


def _heatmap(paired, *, cohort, path):
    """Fly x odor scores, pre and post, on the pinned CVD-validated PRGn ramp.

    The ramp is repo-fixed (score_summary.SCORE_COLORS) and diverges at 1.5,
    the response boundary — so purple reads "no response" and green "responded"
    without the viewer consulting the colourbar.
    """
    cmap, norm = _score_cmap()
    order = heatmap_fly_order(paired)
    paired = _with_fly_id(paired)
    pre = paired.pivot_table(index="fly_id", columns="odor", values="score_pre")
    pre = pre.reindex(index=order)
    post = paired.pivot_table(index="fly_id", columns="odor", values="score_post")
    post = post.reindex(index=pre.index, columns=pre.columns)

    n_rows, n_cols = len(pre.index), len(pre.columns)
    with plt.rc_context(_RC_CONTEXT):
        fig = plt.figure(figsize=(1.9 + 0.62 * n_cols * 2, 1.9 + 0.34 * n_rows))
        # A dedicated colourbar column, NOT fig.colorbar(ax=...): passing ax=
        # steals width from that one axes and breaks the two panels' alignment.
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.055], wspace=0.12)
        ax_pre, ax_post, cax = (fig.add_subplot(gs[0, i]) for i in range(3))

        for ax, frame, title in ((ax_pre, pre, PRE_LABEL), (ax_post, post, POST_LABEL)):
            mesh = ax.imshow(frame.to_numpy(dtype=float), aspect="auto",
                             cmap=cmap, norm=norm, interpolation="nearest")
            ax.set_xticks(np.arange(n_cols))
            ax.set_xticklabels(frame.columns, rotation=28, ha="right")
            ax.set_title(title, loc="left", color=INK, pad=6)
            # 2 px surface gap between cells, per the mark spec.
            ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.4)
            ax.tick_params(which="minor", length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
        ax_pre.set_yticks(np.arange(n_rows))
        ax_pre.set_yticklabels(pre.index, fontsize=7.5)
        ax_post.set_yticks([])

        cb = fig.colorbar(mesh, cax=cax, label="PER score")
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0)
        fig.suptitle(f"{cohort} — per-fly PER score, {n_rows} flies",
                     x=0.01, ha="left", color=INK)
        fig.subplots_adjust(left=0.24, right=0.93, top=0.84, bottom=0.26)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
    return path


def render_cohort(
    paired: pd.DataFrame,
    *,
    cohort: str,
    out_dir: Path,
    binary_threshold: int = BINARY_THRESHOLD,
) -> list[Path]:
    """Write the whole pre-vs-post figure set for one cohort."""

    if paired.empty:
        print(f"[pre-vs-post] {cohort}: no paired trials — nothing to draw.")
        return []

    summary = per_odor_summary(paired, binary_threshold=binary_threshold)
    out_dir = Path(out_dir)
    stem = _safe_dirname(cohort)

    written: list[Path] = []
    written.append(_grouped_bars(
        summary, cohort=cohort, kind="score", p_key="p_score",
        path=out_dir / f"{stem}_score_bars.png",
    ))
    written.append(_grouped_bars(
        summary, cohort=cohort, kind="rate", p_key="p_rate",
        path=out_dir / f"{stem}_rate_bars.png",
    ))
    for maybe in (
        _spotlight(summary, cohort=cohort, path=out_dir / f"{stem}_trained_odor.png"),
        _slopes(paired, summary, cohort=cohort, path=out_dir / f"{stem}_slopes.png"),
        _heatmap(paired, cohort=cohort, path=out_dir / f"{stem}_heatmap.png"),
    ):
        if maybe is not None:
            written.append(maybe)

    summary.to_csv(out_dir / f"{stem}_summary.csv", index=False)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument(
        "--cohorts", nargs="*", default=None,
        help="Cohorts to render. Default: every cohort with paired pre/post trials.",
    )
    parser.add_argument("--binary-threshold", type=int, default=BINARY_THRESHOLD)
    parser.add_argument("--flagged-flies-csv", default="")
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    paired = load_paired_scores(
        args.predictions_csv,
        threshold=args.threshold,
        flagged_flies_csv=args.flagged_flies_csv,
    )
    unpaired = paired.attrs.get("n_unpaired", 0)
    if unpaired:
        print(f"[pre-vs-post] {unpaired} trial(s) had no counterpart and were dropped.")

    if paired.empty:
        print(
            "[pre-vs-post] no paired pre/post trials in the predictions CSV — "
            "nothing to draw. (Pre-test rows only appear after a full pipeline "
            "run; --figures-only cannot produce them.)"
        )
        return

    cohorts = args.cohorts or sorted(paired["dataset_canon"].unique())
    total = 0
    for cohort in cohorts:
        sub = paired[paired["dataset_canon"] == cohort]
        if sub.empty:
            print(f"[pre-vs-post] {cohort}: no paired trials — skipped.")
            continue
        written = render_cohort(
            sub, cohort=cohort,
            out_dir=args.out_root / _safe_dirname(cohort),
            binary_threshold=args.binary_threshold,
        )
        total += len(written)
        print(f"[pre-vs-post] {cohort}: wrote {len(written)} figure(s).")
    print(f"[pre-vs-post] {total} figure(s) written under {args.out_root}")


if __name__ == "__main__":
    main()
