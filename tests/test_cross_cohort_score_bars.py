"""Tests for the cross-cohort score-bar figures.

Two figures live in ``cross_cohort_score_bars``:

* the CS+ comparison — hexanol in hexanol-conditioned flies against air in
  air-conditioned flies;
* ethyl butyrate across four conditioning protocols, where EB was never the
  CS+, so the bars are expected to be indistinguishable.

The numbers pinned here are the ones the already-published figures show
(``score_bars_Hex-Training_oct_nov_all_odors``, ``score_bars_AIR-Training_...``,
``score_bars_ACV-Training_hex_acv_all``). If a future predictions rebuild moves
a cohort, these fail rather than the figure silently re-rendering at new values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.analysis import cross_cohort_score_bars as ccsb  # noqa: E402
from scripts.analysis import odor_bar_palette  # noqa: E402


# --------------------------------------------------------------------------
# Fixtures: load the real predictions once, they are the point of these tests
# --------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not ccsb.PREDICTIONS_CSV.exists()
    or not ccsb.FROZEN_PREDICTIONS_CSV.exists()
    or not ccsb.AIR_BINARY_CSV.exists(),
    reason="predictions / binary-reaction exports are not on this machine",
)


@pytest.fixture(scope="module")
def preds() -> dict[str, pd.DataFrame]:
    return ccsb.load_sources()


@pytest.fixture(scope="module")
def cs_plus(preds: pd.DataFrame) -> pd.DataFrame:
    return ccsb.cs_plus_table(preds)


@pytest.fixture(scope="module")
def eb(preds: pd.DataFrame) -> pd.DataFrame:
    return ccsb.eb_table(preds)


# --------------------------------------------------------------------------
# Cohort extraction reproduces the published bars
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cohort_key, odor, mean, n",
    [
        # score_bars_Hex-Training_oct_nov_all_odors: Hexanol 3.23, EB 1.92, n=13
        ("Hex-Training", "Hexanol", 3.230769, 13),
        ("Hex-Training", "Ethyl Butyrate", 1.923077, 13),
        # score_bars_AIR-Training_unordered_first-presentation_copy.csv
        ("AIR-Training", "AIR", -0.083333, 12),
        ("AIR-Training", "Hexanol", 0.750000, 12),
        ("AIR-Training", "Ethyl Butyrate", 2.666667, 12),
        # score_bars_ACV-Training_hex_acv_all: EB 2.05 at n=19 (one fly lacks EB)
        ("ACV-Training", "Ethyl Butyrate", 2.052632, 19),
        # score_bars_Hex-Training_oct_nov_all_odors control bars: EB 2.00,
        # Hexanol 0.07, n=15. Only the frozen snapshot still holds all 15.
        ("Hex-Control", "Ethyl Butyrate", 2.000000, 15),
        ("Hex-Control", "Hexanol", 0.066667, 15),
    ],
)
def test_cohort_reproduces_published_bar(preds, cohort_key, odor, mean, n):
    scores = ccsb.first_presentation_scores(preds, ccsb.COHORTS[cohort_key], odor)
    assert len(scores) == n
    assert scores.mean() == pytest.approx(mean, abs=5e-6)


def test_each_cohort_reads_its_published_source(preds):
    """The per-cohort source mapping is the whole reason both files are loaded."""
    assert ccsb.COHORTS["Hex-Training"].source == "frozen"
    assert ccsb.COHORTS["Hex-Control"].source == "frozen"
    assert ccsb.COHORTS["AIR-Training"].source == "live"
    assert ccsb.COHORTS["ACV-Training"].source == "live"
    assert set(preds) == {"live", "frozen"}


def test_hex_training_is_identical_in_both_sources(preds):
    """Hex-Training is the one cohort where the source choice is free."""
    frozen = ccsb.first_presentation_scores(
        preds["frozen"], ccsb.COHORTS["Hex-Training"], "Ethyl Butyrate"
    )
    live = ccsb.first_presentation_scores(
        preds["live"], ccsb.COHORTS["Hex-Training"], "Ethyl Butyrate"
    )
    assert sorted(frozen) == sorted(live)


def test_live_hex_control_is_the_lossy_path(preds):
    """Reading Hex-Control live loses flies however you slice it.

    With the flagged dataset it recovers 13 of 15 and shifts EB from 2.00 to
    2.46; without it the cohort silently collapses to four flies. Either way
    the published control bar cannot be rebuilt from the live file, which is
    what pins ``Hex-Control`` to the frozen snapshot.
    """
    lossy = ccsb.first_presentation_scores(
        preds, ccsb.LIVE_HEX_CONTROL, "Ethyl Butyrate"
    )
    assert len(lossy) == 13
    assert lossy.mean() == pytest.approx(2.461538, abs=5e-6)

    unflagged = ccsb.Cohort(
        key="Hex-Control",
        label="x",
        datasets=("Hex-Control",),
        date_prefixes=ccsb.OCTNOV_PREFIXES,
        color="k",
        odor_source="trial_map",
        source="live",
    )
    assert len(ccsb.first_presentation_scores(preds, unflagged, "Ethyl Butyrate")) == 4


def test_unknown_source_is_an_error_not_a_silent_fallback(preds):
    bogus = ccsb.Cohort(
        key="Hex-Control",
        label="x",
        datasets=("Hex-Control",),
        date_prefixes=ccsb.OCTNOV_PREFIXES,
        color="k",
        odor_source="trial_map",
        source="may-2026",
    )
    with pytest.raises(RuntimeError, match="may-2026"):
        ccsb.cohort_scores(preds, bogus)


def test_octnov_filter_actually_restricts_hex_training(preds):
    """Without the date filter the Hex cohort pulls in non-Oct/Nov flies."""
    undated = ccsb.Cohort(
        key="Hex-Training",
        label="x",
        datasets=("Hex-Training",),
        date_prefixes=None,
        color="k",
        odor_source="trial_map",
        source="live",
    )
    assert len(ccsb.first_presentation_scores(preds, undated, "Hexanol")) > 13


def test_air_odor_comes_from_the_binary_export(preds):
    """AIR trial labels carry no odor token, so the join is load-bearing."""
    air = ccsb.cohort_scores(preds, ccsb.COHORTS["AIR-Training"])
    assert set(air["odor"]) >= {"AIR", "Hexanol", "Ethyl Butyrate"}
    assert air["odor"].notna().all()


def test_first_presentation_picks_the_lowest_trial(preds):
    """Hexanol is presented repeatedly; the bar must be the first one only."""
    air = ccsb.cohort_scores(preds, ccsb.COHORTS["AIR-Training"])
    hexanol = air[air["odor"] == "Hexanol"]
    assert hexanol["trial_num"].nunique() > 1, "no repeat presentations to collapse"
    picked = ccsb.first_presentation_scores(
        preds, ccsb.COHORTS["AIR-Training"], "Hexanol"
    )
    assert len(picked) == hexanol["trial_num"].value_counts().loc[
        hexanol["trial_num"].min()
    ]


# --------------------------------------------------------------------------
# Figure 1: the CS+ comparison
# --------------------------------------------------------------------------
def test_cs_plus_table_is_three_protocols_in_order(cs_plus):
    assert list(cs_plus["cohort"]) == ["Hex-Training", "AIR-Training", "ACV-Training"]
    assert list(cs_plus["odor"]) == ["Hexanol", "AIR", "Apple Cider Vinegar"]
    assert list(cs_plus["n_flies"]) == [13, 12, 20]


def test_every_cs_plus_bar_shows_that_cohorts_own_conditioned_stimulus(cs_plus):
    """The figure is only meaningful if each bar is that cohort's own CS+."""
    for row in cs_plus.itertuples(index=False):
        assert row.odor == ccsb.COHORTS[row.cohort].cs_plus


def test_hex_control_has_no_cs_plus_and_is_absent_from_the_figure(cs_plus):
    """An unpaired control has no conditioned stimulus, so it cannot appear."""
    assert ccsb.COHORTS["Hex-Control"].cs_plus is None
    assert "Hex-Control" not in set(cs_plus["cohort"])
    assert "Hex-Control" not in ccsb.CS_PLUS_COHORT_ORDER


def test_cs_plus_means_and_sems(cs_plus):
    assert cs_plus.loc[0, "mean_score"] == pytest.approx(3.230769, abs=5e-6)
    assert cs_plus.loc[1, "mean_score"] == pytest.approx(-0.083333, abs=5e-6)
    # score_bars_ACV-Training_hex_acv_all shows the ACV bar at -0.20, n=20
    assert cs_plus.loc[2, "mean_score"] == pytest.approx(-0.200000, abs=5e-6)
    assert cs_plus.loc[0, "sem_score"] == pytest.approx(0.631988, abs=5e-6)
    assert cs_plus.loc[1, "sem_score"] == pytest.approx(0.228908, abs=5e-6)
    assert cs_plus.loc[2, "sem_score"] == pytest.approx(0.091766, abs=5e-6)


def test_cs_plus_omnibus_separates_the_protocols(preds):
    omnibus = ccsb.cs_plus_omnibus(preds)
    assert omnibus["k"] == 3
    assert omnibus["p"] < 0.001
    assert omnibus["p"] == pytest.approx(3.962e-05, rel=1e-3)


def test_only_hexanol_conditioning_moves_its_own_cs_plus(preds):
    """Both neutral protocols sit at zero and do not differ from each other."""
    pairs = ccsb.cs_plus_pairwise(preds).set_index(["left", "right"])
    assert pairs.loc[("Hex-Training", "AIR-Training"), "stars"] == "**"
    assert pairs.loc[("Hex-Training", "ACV-Training"), "stars"] == "***"
    assert pairs.loc[("AIR-Training", "ACV-Training"), "stars"] == ""


def test_holm_demotes_the_hexanol_air_pair(preds):
    """Correcting for three pairs moves hexanol-vs-air from *** to **.

    The uncorrected p is still 0.00082; reporting it as *** on a figure that
    draws three comparisons would be the multiple-comparison error this guards.
    """
    raw = ccsb.cs_plus_pvalue(preds)
    assert ccsb.stars(raw) == "***"
    pairs = ccsb.cs_plus_pairwise(preds).set_index(["left", "right"])
    row = pairs.loc[("Hex-Training", "AIR-Training")]
    assert row["p"] == pytest.approx(raw, abs=5e-9)
    assert row["p_holm"] == pytest.approx(2 * raw, abs=5e-9)
    assert row["stars"] == "**"


def test_cs_plus_bar_colors_follow_the_shared_palette(cs_plus):
    colors = list(cs_plus["color"])
    assert colors[0] == odor_bar_palette.HEX_COLOR
    assert colors[2] == odor_bar_palette.ACV_COLOR
    # AIR is not an odorant and has no palette entry; it keeps the blue every
    # other AIR bar uses.
    assert odor_bar_palette.odor_color("AIR") is None
    assert colors[1] == ccsb.AIR_COLOR


# --------------------------------------------------------------------------
# Figure 2: ethyl butyrate across cohorts
# --------------------------------------------------------------------------
def test_eb_table_is_four_cohorts_in_order(eb):
    assert list(eb["cohort"]) == [
        "Hex-Training",
        "Hex-Control",
        "AIR-Training",
        "ACV-Training",
    ]
    assert list(eb["n_flies"]) == [13, 15, 12, 19]


def test_eb_means(eb):
    assert list(np.round(eb["mean_score"], 6)) == [
        pytest.approx(1.923077, abs=5e-6),
        pytest.approx(2.000000, abs=5e-6),
        pytest.approx(2.666667, abs=5e-6),
        pytest.approx(2.052632, abs=5e-6),
    ]


def test_eb_control_bar_matches_the_published_hex_figure(eb):
    """The published Hex figure shows the control EB bar at 2.00, not 2.46."""
    control = eb.loc[eb["cohort"] == "Hex-Control"].iloc[0]
    assert control["mean_score"] == pytest.approx(2.0, abs=5e-6)
    assert control["n_flies"] == 15


def test_every_eb_bar_is_the_ethyl_butyrate_colour(eb):
    """Bar colour tracks the presented odor, not the cohort.

    Every bar is ethyl butyrate, so every bar is the palette's EB pink; the
    cohort is carried by the tick label. Colouring these by cohort would make a
    green bar mean "hexanol" in one figure and "EB in hexanol-trained flies" in
    this one.
    """
    assert set(eb["color"]) == {odor_bar_palette.PINK}


def test_eb_has_no_significant_pair(preds):
    """The point of the figure: EB was never the CS+, so nothing separates."""
    omnibus = ccsb.eb_omnibus(preds)
    assert omnibus["p"] > 0.05
    assert omnibus["p"] == pytest.approx(0.8728, abs=5e-4)

    pairs = ccsb.eb_pairwise(preds)
    assert len(pairs) == 6
    assert (pairs["p_holm"] > 0.05).all()
    assert (pairs["stars"] == "").all()


def test_no_brackets_are_drawn_when_nothing_is_significant(preds, eb):
    """A bracket in this figure would be a bug, so assert on the artists."""
    fig = ccsb.render_bars(eb, title="t", brackets=ccsb.significant_brackets(preds, which="eb"))
    ax = fig.axes[0]
    # Bars and the zero baseline are the only line/patch artists expected.
    assert len([ln for ln in ax.lines if ln.get_linewidth() == 0.9]) == 0
    assert not [t for t in ax.texts if set(t.get_text()) <= {"*"} and t.get_text()]
    plt.close(fig)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def test_stars_thresholds():
    assert ccsb.stars(0.0009) == "***"
    assert ccsb.stars(0.005) == "**"
    assert ccsb.stars(0.04) == "*"
    assert ccsb.stars(0.051) == ""
    assert ccsb.stars(float("nan")) == ""


def test_render_bars_uses_the_score_scale_and_label(cs_plus):
    fig = ccsb.render_bars(cs_plus, title="t")
    ax = fig.axes[0]
    assert ax.get_ylim() == (ccsb.SCORE_MIN, ccsb.SCORE_MAX)
    assert ax.get_ylabel() == ccsb.SCORE_Y_LABEL
    assert len([p for p in ax.patches]) == len(cs_plus)
    plt.close(fig)


def test_render_bars_draws_the_brackets_it_is_given(preds, cs_plus):
    brackets = ccsb.significant_brackets(preds, which="cs_plus")
    assert [b["stars"] for b in brackets] == ["**", "***"]
    fig = ccsb.render_bars(cs_plus, title="t", brackets=brackets)
    ax = fig.axes[0]
    drawn = {t.get_text() for t in ax.texts}
    assert {"**", "***"} <= drawn
    plt.close(fig)


def test_brackets_are_ordered_narrow_span_first(preds):
    """A wide bracket stacked under a narrow one would cross it."""
    brackets = ccsb.significant_brackets(preds, which="cs_plus")
    spans = [abs(b["j"] - b["i"]) for b in brackets]
    assert spans == sorted(spans)


def test_stacked_brackets_do_not_overlap_each_other(preds, cs_plus):
    """Two brackets on a 3.2-high bar have to stack, not collide."""
    brackets = ccsb.significant_brackets(preds, which="cs_plus")
    fig = ccsb.render_bars(cs_plus, title="t", brackets=brackets)
    ax = fig.axes[0]
    rails = sorted(
        max(ln.get_ydata()) for ln in ax.lines if ln.get_linewidth() == 0.9
    )
    assert len(rails) == len(brackets)
    for lower, upper in zip(rails, rails[1:]):
        assert upper - lower >= 0.3, f"brackets at {lower} and {upper} collide"
    assert max(rails) <= ccsb.SCORE_MAX


def test_tall_bar_label_yields_to_a_bracket(preds, cs_plus):
    """The hexanol label moves inside its bar rather than blocking the stack."""
    brackets = ccsb.significant_brackets(preds, which="cs_plus")
    fig = ccsb.render_bars(cs_plus, title="t", brackets=brackets)
    ax = fig.axes[0]
    label = next(t for t in ax.texts if "n=13" in t.get_text())
    _, y = label.get_position()
    rails = [max(ln.get_ydata()) for ln in ax.lines if ln.get_linewidth() == 0.9]
    assert y < min(rails), "the 3.23 label still sits in the bracket stack"
    plt.close(fig)


def test_label_moved_inside_a_bar_still_clears_its_whisker(preds, cs_plus):
    """Moving the label inside must not drop it back onto the error bar.

    The first three-bar render put "3.23 (n=13)" just under the bar top, which
    is squarely inside the 2.60..3.86 whisker.
    """
    brackets = ccsb.significant_brackets(preds, which="cs_plus")
    fig = ccsb.render_bars(cs_plus, title="t", brackets=brackets)
    ax = fig.axes[0]
    row = cs_plus.iloc[0]
    lo = row["mean_score"] - row["sem_score"]
    hi = row["mean_score"] + row["sem_score"]
    label = next(t for t in ax.texts if "n=13" in t.get_text())
    _, y = label.get_position()
    assert not (lo <= y <= hi), f"label at y={y} sits inside the whisker {lo}..{hi}"
    # ...and it stays within its own bar rather than hanging below the baseline.
    assert y - ccsb._ANNOTATION_HEIGHT >= 0.0
    plt.close(fig)


def test_unknown_comparison_is_rejected(preds):
    with pytest.raises(ValueError, match="cs_plus"):
        ccsb.significant_brackets(preds, which="nonsense")


def test_bar_annotations_report_mean_and_n(cs_plus):
    fig = ccsb.render_bars(cs_plus, title="t")
    ax = fig.axes[0]
    joined = " ".join(t.get_text() for t in ax.texts)
    assert "3.23" in joined and "-0.08" in joined
    assert "n=13" in joined and "n=12" in joined
    plt.close(fig)


def test_render_bars_keeps_labels_inside_the_axes(cs_plus):
    """A negative bar labels below itself and must not fall off the scale."""
    fig = ccsb.render_bars(cs_plus, title="t")
    ax = fig.axes[0]
    for text in ax.texts:
        _, y = text.get_position()
        assert ccsb.SCORE_MIN <= y <= ccsb.SCORE_MAX
    plt.close(fig)


def test_negative_bar_label_clears_its_error_bar(cs_plus):
    """The AIR bar is -0.08 with a 0.23 SEM, so there is no room to label below.

    Clamping the label to the bottom of the scale instead of flipping it above
    the bar drops the text straight onto the error-bar whisker, which is what
    the first render of this figure did.
    """
    fig = ccsb.render_bars(cs_plus, title="t")
    ax = fig.axes[0]
    row = cs_plus.iloc[1]
    lo = row["mean_score"] - row["sem_score"]
    hi = row["mean_score"] + row["sem_score"]
    labels = [t for t in ax.texts if "n=12" in t.get_text()]
    assert labels, "the AIR bar lost its annotation"
    for text in labels:
        _, y = text.get_position()
        assert not (lo <= y <= hi), f"label at y={y} sits inside the whisker {lo}..{hi}"
    plt.close(fig)


@pytest.mark.parametrize(
    "mean, sem",
    [
        (-0.083333, 0.228908),  # the real AIR bar: no room below
        (-0.9, 0.05),           # pinned to the floor of the scale
        (4.9, 0.4),             # pinned to the ceiling of the scale
        (0.0, 0.0),             # degenerate: no bar, no whisker
    ],
)
def test_labels_never_overlap_whiskers_at_the_scale_edges(mean, sem):
    table = pd.DataFrame(
        [
            {
                "cohort": "x",
                "label": "x",
                "odor": "Hexanol",
                "mean_score": mean,
                "sem_score": sem,
                "n_flies": 9,
                "color": "k",
            }
        ]
    )
    fig = ccsb.render_bars(table, title="t")
    ax = fig.axes[0]
    for text in ax.texts:
        _, y = text.get_position()
        assert ccsb.SCORE_MIN <= y <= ccsb.SCORE_MAX
        if sem > 0:
            assert not (mean - sem < y < mean + sem)
    plt.close(fig)
