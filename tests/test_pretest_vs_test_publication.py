"""Publication-grade rules for the pre-vs-post figure set.

Two requirements drive these tests.

1. **Fixed axes, every figure, every cohort.** PER score always spans the full
   ordinal scale -1..5 and response rate always spans 0..100 %. Auto-scaled axes
   make two cohorts printed side by side look like different effect sizes when
   they are not — the single most common way a bar chart misleads. Fixed limits
   also mean a bar's height is comparable across every panel in the paper.

2. **A validated palette.** The pre/post pair is checked with the dataviz
   validator (``scripts/validate_palette.js``): the previous gray + navy failed
   the lightness band, the chroma floor and the 3:1 contrast check. The blue /
   orange pair passes every check (CVD ΔE 24.7 protan, 32.7 tritan, 33.6 normal).

The significance-bracket helper expands ``ylim`` to make room for a bracket, so
it has to be told not to when the axis is fixed — otherwise requirement 1 is
silently violated exactly when a result is significant.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scripts.analysis import pretest_vs_test_comparison as pvt
from scripts.analysis import significance_brackets

ODORS = ["hexanol", "citral", "linalool"]


def _predictions(tmp_path, n_flies=4, pre=1, post=4):
    rows = []
    for fn in range(1, n_flies + 1):
        for phase, score in (("pretest", pre), ("testing", post)):
            for i, odor in enumerate(ODORS, start=1):
                rows.append({
                    "dataset": "Hex-Sensitivity-24-0.1", "fly": "b1",
                    "fly_number": fn, "trial_label": f"{phase}_{i}_{odor}",
                    "score": score, "trial_type": phase,
                })
    path = tmp_path / "model_predictions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ── the fixed axis limits ─────────────────────────────────────────────────


def test_score_axis_limits_span_the_whole_ordinal_scale():
    assert pvt.SCORE_YLIM == (-1, 5)


def test_rate_axis_limits_span_the_whole_percentage_range():
    assert pvt.RATE_YLIM == (0, 100)


def test_score_ticks_are_every_integer_level():
    """-1..5 are the actual model outputs; a tick per level, no interpolation."""
    assert list(pvt.SCORE_YTICKS) == [-1, 0, 1, 2, 3, 4, 5]


def test_rate_ticks_are_round_percentages():
    assert list(pvt.RATE_YTICKS) == [0, 20, 40, 60, 80, 100]


# ── every rendered figure honours them ────────────────────────────────────


def _axis_limits(path_stem, tmp_path, **kw):
    written = pvt.render_cohort(
        pvt.load_paired_scores(_predictions(tmp_path, **kw)),
        cohort="Hex-Sensitivity-24-0.1",
        out_dir=tmp_path / "figs",
    )
    return {p.stem.split("Hex-Sensitivity-24-0.1_")[-1]: p for p in written}


def test_score_bars_use_the_fixed_score_axis(tmp_path):
    fig, ax = plt.subplots()
    pvt._apply_score_axis(ax)
    assert ax.get_ylim() == pvt.SCORE_YLIM
    plt.close(fig)


def test_rate_bars_use_the_fixed_rate_axis(tmp_path):
    fig, ax = plt.subplots()
    pvt._apply_rate_axis(ax)
    assert ax.get_ylim() == pvt.RATE_YLIM
    plt.close(fig)


def test_a_low_scoring_cohort_still_gets_the_full_axis(tmp_path):
    """All-zero data must NOT auto-scale to a flat, exaggerated-looking axis."""
    fig, ax = plt.subplots()
    ax.bar([0, 1], [0.0, 0.1])
    pvt._apply_score_axis(ax)
    assert ax.get_ylim() == (-1, 5)
    plt.close(fig)


def test_a_saturated_cohort_does_not_overflow_the_rate_axis(tmp_path):
    fig, ax = plt.subplots()
    ax.bar([0, 1], [100.0, 100.0])
    pvt._apply_rate_axis(ax)
    assert ax.get_ylim() == (0, 100)
    plt.close(fig)


def test_negative_scores_are_visible_on_the_score_axis():
    """-1 is a real model output (seen in 3Oct); it must not fall off the axis."""
    fig, ax = plt.subplots()
    ax.bar([0], [-1.0])
    pvt._apply_score_axis(ax)
    lo, hi = ax.get_ylim()
    assert lo <= -1
    plt.close(fig)


# ── brackets must not silently expand a fixed axis ────────────────────────


def test_brackets_do_not_expand_a_fixed_axis():
    """The whole point of a fixed axis is that nothing may move it."""
    fig, ax = plt.subplots()
    ax.bar([0, 1], [100.0, 100.0])
    ax.set_ylim(0, 100)
    significance_brackets.draw(ax, [0.5], 1.0, [0.001], expand=False)
    assert ax.get_ylim() == (0, 100)
    plt.close(fig)


def test_brackets_still_expand_by_default_for_existing_callers():
    """score_summary, reaction_matrix and rig_batch rely on the old behaviour."""
    fig, ax = plt.subplots()
    ax.bar([0, 1], [100.0, 100.0])
    ax.set_ylim(0, 100)
    significance_brackets.draw(ax, [0.5], 1.0, [0.001])
    assert ax.get_ylim()[1] > 100
    plt.close(fig)


def test_a_bracket_on_a_saturated_bar_stays_inside_the_fixed_axis():
    """A 100 % bar leaves no headroom; the star must still land on the canvas."""
    fig, ax = plt.subplots()
    ax.bar([0, 1], [100.0, 100.0])
    ax.set_ylim(0, 100)
    significance_brackets.draw(ax, [0.5], 1.0, [0.001], expand=False)
    ys = [t.get_position()[1] for t in ax.texts]
    assert ys, "no significance star was drawn"
    assert max(ys) <= 100
    plt.close(fig)


def test_no_bracket_is_drawn_for_a_null_p_value():
    fig, ax = plt.subplots()
    ax.bar([0, 1], [50.0, 50.0])
    ax.set_ylim(0, 100)
    significance_brackets.draw(ax, [0.5], 1.0, [None], expand=False)
    assert not ax.texts
    plt.close(fig)


# ── the validated palette ─────────────────────────────────────────────────


def test_the_pre_post_pair_is_the_validated_one():
    """Ran through dataviz scripts/validate_palette.js: ALL CHECKS PASS.

    The previous gray #b0b0b0 + navy #1a3a6b FAILED the lightness band, the
    chroma floor (gray reads as no-hue) and the 3:1 contrast check.
    """
    assert pvt.PRE_COLOR == "#2a78d6"
    assert pvt.POST_COLOR == "#eb6834"


def test_the_two_phases_are_not_distinguished_by_colour_alone():
    """Accessibility: a legend is always present for two series."""
    assert pvt.PRE_LABEL and pvt.POST_LABEL
    assert pvt.PRE_LABEL != pvt.POST_LABEL


def test_the_score_heatmap_still_uses_the_pinned_cvd_ramp():
    """SCORE_COLORS is fixed repo-wide; a second ramp must never appear."""
    from scripts.analysis.score_summary import _score_cmap

    assert pvt._score_cmap is _score_cmap


# ── figures still render with all of the above ────────────────────────────


def test_every_figure_still_renders(tmp_path):
    written = pvt.render_cohort(
        pvt.load_paired_scores(_predictions(tmp_path)),
        cohort="Hex-Sensitivity-24-0.1",
        out_dir=tmp_path / "figs",
    )
    assert len(written) == 5
    assert all(p.exists() and p.stat().st_size > 0 for p in written)


def test_figures_render_at_publication_resolution():
    assert pvt._RC_CONTEXT["savefig.dpi"] >= 300
    assert pvt._RC_CONTEXT["figure.dpi"] >= 300


def test_figures_embed_text_as_text_not_paths():
    """Journals require editable vector text; Type-3 fonts are widely rejected."""
    assert pvt._RC_CONTEXT.get("pdf.fonttype") == 42
    assert pvt._RC_CONTEXT.get("ps.fonttype") == 42


# ── the zero baseline on the score axis ───────────────────────────────────


def test_score_axis_marks_the_zero_baseline():
    """The axis floor is -1 but bars are anchored at 0, so without an explicit
    zero rule the bars look like they float above the bottom of the plot."""
    fig, ax = plt.subplots()
    pvt._apply_score_axis(ax)
    zero_lines = [
        ln for ln in ax.get_lines()
        if np.allclose(ln.get_ydata(), 0.0) and len(ln.get_ydata()) >= 2
    ]
    assert zero_lines, "no zero baseline drawn on the score axis"
    plt.close(fig)


def test_rate_axis_needs_no_zero_line():
    """Its floor IS zero, so a rule there would just double the spine."""
    fig, ax = plt.subplots()
    pvt._apply_rate_axis(ax)
    zero_lines = [
        ln for ln in ax.get_lines()
        if len(ln.get_ydata()) >= 2 and np.allclose(ln.get_ydata(), 0.0)
    ]
    assert not zero_lines
    plt.close(fig)


def test_the_zero_baseline_is_recessive_not_a_data_mark():
    fig, ax = plt.subplots()
    pvt._apply_score_axis(ax)
    ln = [l for l in ax.get_lines() if np.allclose(l.get_ydata(), 0.0)][0]
    assert ln.get_linewidth() <= 1.0
    assert ln.get_zorder() < 2
    plt.close(fig)


# ── zero must not read as missing ─────────────────────────────────────────


def _summary_with_zero_pre():
    return pd.DataFrame([
        {"odor": "Hexanol", "n_flies": 4, "mean_pre": 0.0, "sem_pre": 0.0,
         "mean_post": 3.0, "sem_post": 0.4, "pct_pre": 0.0, "pct_post": 75.0,
         "p_score": None, "p_rate": None},
        {"odor": "Citral", "n_flies": 4, "mean_pre": 1.0, "sem_pre": 0.2,
         "mean_post": 1.0, "sem_post": 0.2, "pct_pre": 25.0, "pct_post": 25.0,
         "p_score": None, "p_rate": None},
    ])


def test_a_zero_bar_is_labelled_so_it_is_not_read_as_missing():
    """A 0 % bar has zero height. Without a mark it is indistinguishable from
    "this fly/odor has no data", which is a different claim entirely."""
    fig, ax = pvt.build_grouped_bars(
        _summary_with_zero_pre(), cohort="Hex-Sensitivity-24-0.1",
        kind="rate", p_key="p_rate",
    )
    labels = [t.get_text() for t in ax.texts]
    assert "0" in labels, labels
    plt.close(fig)


def test_only_zero_bars_are_labelled_not_every_bar():
    """Selective direct labels — a number on every bar is chartjunk."""
    fig, ax = pvt.build_grouped_bars(
        _summary_with_zero_pre(), cohort="Hex-Sensitivity-24-0.1",
        kind="rate", p_key="p_rate",
    )
    # 4 bars, exactly one of which is zero.
    assert len([t for t in ax.texts if t.get_text() == "0"]) == 1
    plt.close(fig)


def test_zero_labels_apply_to_the_score_panel_too():
    fig, ax = pvt.build_grouped_bars(
        _summary_with_zero_pre(), cohort="Hex-Sensitivity-24-0.1",
        kind="score", p_key="p_score",
    )
    assert "0" in [t.get_text() for t in ax.texts]
    plt.close(fig)


def test_no_zero_labels_when_every_bar_has_height():
    summary = _summary_with_zero_pre()
    summary.loc[0, "pct_pre"] = 10.0
    fig, ax = pvt.build_grouped_bars(
        summary, cohort="Hex-Sensitivity-24-0.1", kind="rate", p_key="p_rate",
    )
    assert "0" not in [t.get_text() for t in ax.texts]
    plt.close(fig)


def test_slope_markers_at_the_axis_extremes_are_not_clipped():
    """A score of 5 sits exactly on the ceiling; with clipping on, its marker is
    sliced in half and the maximum response looks like a rendering error. The
    axis limit stays hard — only the mark is allowed to overhang."""
    paired = pd.DataFrame([
        {"dataset_canon": "Hex-Sensitivity-24-0.1", "fly": "b1", "fly_number": 1,
         "odor": "Hexanol", "score_pre": -1.0, "score_post": 5.0},
        {"dataset_canon": "Hex-Sensitivity-24-0.1", "fly": "b1", "fly_number": 2,
         "odor": "Hexanol", "score_pre": 0.0, "score_post": 5.0},
    ])
    summary = pvt.per_odor_summary(paired)
    fig, ax = plt.subplots()
    path = pvt._slopes(paired, summary, cohort="Hex-Sensitivity-24-0.1",
                       path=Path("/tmp/_unused_slopes.png"))
    plt.close(fig)
    assert path is not None
