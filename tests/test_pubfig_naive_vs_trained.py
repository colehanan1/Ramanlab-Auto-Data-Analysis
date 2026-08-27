"""Naive vs trained vs control, one odor at one concentration.

The published figures compare *trained vs control* within a cohort, and
*concentration vs concentration* within the naive RandomPanel flies. Neither
answers "does conditioning move the response away from what an untrained fly
already does to this odorant", because the naive bar never appears next to the
trained one.

This figure puts the three cohorts on one axes for a single odor/concentration
— mean PER score and response rate — with the first presentation only, so the
trained bar is the immediate post-training test and the naive bar is that
odorant's first appearance in the random panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scripts.analysis import odor_bar_palette as pal
from scripts.analysis.pubfig_naive_vs_trained import (
    COHORTS,
    RATE_Y_LABEL,
    REACTION_BOUNDARY,
    SCORE_Y_LABEL,
    Comparison,
    base_odor,
    cohort_scores,
    rate_stats,
    render_comparison,
    score_stats,
    stats_rows,
)

GENO = "GR5a-Old"


# --------------------------------------------------------------------------
# Label handling
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("3-Octanol (0.1%)", "3-Octanol"),
        ("3-Octanol (0.1%) 2", "3-Octanol"),
        ("Ethyl Butyrate (1%) 1", "Ethyl Butyrate"),
        ("Ethyl Butyrate", "Ethyl Butyrate"),
        ("  Hexanol  ", "Hexanol"),
    ],
)
def test_base_odor_strips_concentration_and_presentation(label, expected):
    assert base_odor(label) == expected


def test_base_odor_keeps_case_for_display():
    """The bar figures print this label, so it must not come back casefolded."""
    assert base_odor("Isoamyl Acetate (1%)") == "Isoamyl Acetate"


# --------------------------------------------------------------------------
# Cohort extraction
# --------------------------------------------------------------------------


def _scores_frame() -> pd.DataFrame:
    """Two flies per dataset, each with two presentations of two odors."""
    rows = []
    for dataset, flies, geno in (
        ("RandomPanel-24-0.1", ["may_01_batch_1", "may_02_batch_2"], GENO),
        ("3OCT-Training-24-0.1", ["july_01_batch_1", "july_02_batch_2"], GENO),
        ("3OCT-Control-24-0.1", ["july_03_batch_1"], GENO),
        ("RandomPanel-24-0.1", ["may_09_batch_1"], "GR5a-New"),
    ):
        for fly in flies:
            for odor, label in (("3-Octanol (0.1%)", "3-Octanol"), ("Hexanol (0.1%)", "Hexanol")):
                for occurrence in (1, 2):
                    rows.append(
                        {
                            "dataset_canon": dataset,
                            "fly": fly,
                            "fly_number": 1,
                            "fly_type": geno,
                            "odor_display": odor if dataset != "RandomPanel-24-0.1" else label,
                            "occurrence": occurrence,
                            # first presentation 3 / second 0, so a test that
                            # silently averages the two lands on 1.5
                            "score": (3.0 if occurrence == 1 else 0.0)
                            if label == "3-Octanol"
                            else 1.0,
                        }
                    )
    return pd.DataFrame(rows)


def test_cohort_scores_keeps_one_row_per_fly():
    got = cohort_scores(
        _scores_frame(), "3OCT-Training-24-0.1", "3-Octanol", genotype=GENO
    )
    assert len(got) == 2
    assert set(got.columns) >= {"fly", "fly_number", "score"}


def test_cohort_scores_uses_the_first_presentation_only():
    got = cohort_scores(
        _scores_frame(), "3OCT-Training-24-0.1", "3-Octanol", genotype=GENO
    )
    assert sorted(got["score"]) == [3.0, 3.0]  # not 1.5 (the two-presentation mean)


def test_cohort_scores_matches_a_concentration_tagged_label():
    """The trained cohort labels the odor "3-Octanol (0.1%)", the panel doesn't."""
    frame = _scores_frame()
    trained = cohort_scores(frame, "3OCT-Training-24-0.1", "3-Octanol", genotype=GENO)
    naive = cohort_scores(frame, "RandomPanel-24-0.1", "3-Octanol", genotype=GENO)
    assert len(trained) == 2 and len(naive) == 2


def test_cohort_scores_does_not_leak_another_odor():
    got = cohort_scores(
        _scores_frame(), "3OCT-Training-24-0.1", "Hexanol", genotype=GENO
    )
    assert sorted(got["score"]) == [1.0, 1.0]


def test_cohort_scores_filters_genotype():
    frame = _scores_frame()
    assert len(cohort_scores(frame, "RandomPanel-24-0.1", "3-Octanol", genotype=GENO)) == 2
    assert (
        len(cohort_scores(frame, "RandomPanel-24-0.1", "3-Octanol", genotype="GR5a-New"))
        == 1
    )


def test_cohort_scores_filters_batch():
    got = cohort_scores(
        _scores_frame(), "3OCT-Training-24-0.1", "3-Octanol", genotype=GENO, batch=1
    )
    assert list(got["fly"]) == ["july_01_batch_1"]


def test_cohort_scores_batch_none_keeps_every_batch():
    got = cohort_scores(
        _scores_frame(), "3OCT-Training-24-0.1", "3-Octanol", genotype=GENO, batch=None
    )
    assert len(got) == 2


def test_cohort_scores_unknown_dataset_is_empty_not_an_error():
    got = cohort_scores(_scores_frame(), "Nope-24-1", "3-Octanol", genotype=GENO)
    assert got.empty


# --------------------------------------------------------------------------
# Score statistics
# --------------------------------------------------------------------------


def _groups(naive, trained, control) -> dict[str, np.ndarray]:
    return {
        "Naive": np.asarray(naive, dtype=float),
        "Trained": np.asarray(trained, dtype=float),
        "Control": np.asarray(control, dtype=float),
    }


def test_score_stats_reports_mean_sem_and_n():
    got = score_stats(_groups([0, 2, 4], [5, 5, 5], [0, 0, 0]))
    assert got["means"]["Naive"] == pytest.approx(2.0)
    assert got["sems"]["Naive"] == pytest.approx(np.std([0, 2, 4], ddof=1) / np.sqrt(3))
    assert got["n"]["Trained"] == 3


def test_score_stats_separated_groups_are_significant():
    got = score_stats(_groups([0] * 12, [5] * 12, [0] * 12))
    assert got["omnibus_p"] < 0.05
    assert got["pairwise"][("Naive", "Trained")]["p_adj"] < 0.05


def test_score_stats_identical_groups_are_not_significant():
    got = score_stats(_groups([1, 2, 3] * 4, [1, 2, 3] * 4, [1, 2, 3] * 4))
    assert got["omnibus_p"] > 0.05
    for pair in got["pairwise"].values():
        assert pair["p_adj"] > 0.05


def test_score_stats_holm_correction_is_applied():
    """Three pairwise tests: the adjusted p must exceed the raw one."""
    got = score_stats(_groups([0] * 10, [3] * 10, [5] * 10))
    for pair in got["pairwise"].values():
        assert pair["p_adj"] >= pair["p_raw"]
    assert any(p["p_adj"] > p["p_raw"] for p in got["pairwise"].values())


def test_score_stats_covers_all_three_pairs():
    got = score_stats(_groups([1, 2], [2, 3], [3, 4]))
    assert set(got["pairwise"]) == {
        ("Naive", "Trained"),
        ("Naive", "Control"),
        ("Trained", "Control"),
    }


def test_score_stats_tolerates_an_empty_cohort():
    got = score_stats(_groups([], [1, 2, 3], [1, 2, 3]))
    assert np.isnan(got["means"]["Naive"])
    assert np.isnan(got["pairwise"][("Naive", "Trained")]["p_raw"])


# --------------------------------------------------------------------------
# Response-rate statistics
# --------------------------------------------------------------------------


def test_reaction_boundary_matches_score_summary():
    assert REACTION_BOUNDARY == 2


def test_rate_stats_counts_scores_at_or_above_the_boundary():
    got = rate_stats(_groups([0, 1, 2, 5], [0, 0, 0, 0], [2, 2, 2, 2]))
    assert got["k"]["Naive"] == 2  # the 2 and the 5
    assert got["rate"]["Naive"] == pytest.approx(0.5)
    assert got["rate"]["Trained"] == pytest.approx(0.0)
    assert got["rate"]["Control"] == pytest.approx(1.0)


def test_rate_stats_wilson_interval_brackets_the_estimate():
    got = rate_stats(_groups([5, 5, 0, 0], [0] * 4, [5] * 4))
    lo, hi = got["ci"]["Naive"]
    assert lo < got["rate"]["Naive"] < hi
    assert 0.0 <= lo and hi <= 1.0


def test_rate_stats_wilson_interval_stays_inside_the_unit_range_at_the_extremes():
    got = rate_stats(_groups([5] * 6, [0] * 6, [0] * 6))
    lo, hi = got["ci"]["Naive"]
    assert hi == pytest.approx(1.0)
    assert lo > 0.5


def test_rate_stats_separated_groups_are_significant():
    got = rate_stats(_groups([5] * 15, [0] * 15, [0] * 15))
    assert got["omnibus_p"] < 0.05
    assert got["pairwise"][("Naive", "Trained")]["p_adj"] < 0.05


def test_rate_stats_identical_groups_are_not_significant():
    got = rate_stats(_groups([5, 0] * 6, [5, 0] * 6, [5, 0] * 6))
    assert got["omnibus_p"] > 0.05


def test_rate_stats_tolerates_an_empty_cohort():
    got = rate_stats(_groups([], [5] * 5, [0] * 5))
    assert got["n"]["Naive"] == 0
    assert np.isnan(got["rate"]["Naive"])


# --------------------------------------------------------------------------
# The figure
# --------------------------------------------------------------------------


CMP = Comparison(
    odor="3-Octanol",
    concentration="0.1%",
    naive_dataset="RandomPanel-24-0.1",
    train_dataset="3OCT-Training-24-0.1",
    control_dataset="3OCT-Control-24-0.1",
)


@pytest.fixture
def figure():
    groups = _groups([0, 1, 2, 3, 0, 5] * 3, [5, 4, 3, 5, 2, 5] * 3, [0, 0, 1, 0, 2, 0] * 3)
    fig = render_comparison(CMP, groups)
    yield fig
    plt.close(fig)


def test_figure_has_a_score_panel_and_a_rate_panel(figure):
    axes = figure.axes
    assert len(axes) == 2
    assert axes[0].get_ylabel() == SCORE_Y_LABEL
    assert axes[1].get_ylabel() == RATE_Y_LABEL


def test_score_axis_spans_the_model_score_range(figure):
    assert figure.axes[0].get_ylim()[0] == pytest.approx(-1.0)
    assert figure.axes[0].get_ylim()[1] >= 5.0


def test_rate_axis_is_a_percentage(figure):
    lo, hi = figure.axes[1].get_ylim()
    assert lo == pytest.approx(0.0)
    assert hi >= 100.0


def test_each_panel_draws_one_bar_per_cohort(figure):
    for ax in figure.axes:
        bars = [p for p in ax.patches if p.get_width() > 0]
        assert len(bars) == len(COHORTS) == 3


def test_bar_colours_follow_the_house_scheme(figure):
    """Trained = odor colour, control = grey, naive = open bar in the odor colour."""
    for ax in figure.axes:
        naive, trained, control = [p for p in ax.patches if p.get_width() > 0]
        assert matplotlib.colors.to_hex(trained.get_facecolor()) == pal.OCTANOL_BLUE
        assert matplotlib.colors.to_hex(control.get_facecolor()) == pal.CTRL_COLOR
        assert matplotlib.colors.to_hex(naive.get_facecolor()) == "#ffffff"
        assert matplotlib.colors.to_hex(naive.get_edgecolor()) == pal.OCTANOL_BLUE
        assert naive.get_hatch()


def test_tick_labels_name_the_cohorts_with_their_n(figure):
    labels = [t.get_text() for t in figure.axes[0].get_xticklabels()]
    assert [l.split("\n")[0] for l in labels] == list(COHORTS)
    assert "n=18" in labels[0]


def test_title_names_the_odor_and_concentration(figure):
    text = " ".join(t.get_text() for t in figure.texts) + figure.axes[0].get_title()
    assert "3-Octanol" in text and "0.1%" in text


def test_significant_pairs_get_a_star_bracket(figure):
    stars = [
        t.get_text()
        for ax in figure.axes
        for t in ax.texts
        if set(t.get_text()) <= {"*"} and t.get_text()
    ]
    assert stars, "separated cohorts must be bracketed"
    assert all(s in {"*", "**", "***"} for s in stars)


def test_no_bracket_is_drawn_when_nothing_is_significant():
    flat = _groups([1] * 8, [1] * 8, [1] * 8)
    fig = render_comparison(CMP, flat)
    try:
        stars = [
            t.get_text() for ax in fig.axes for t in ax.texts
            if t.get_text() and set(t.get_text()) <= {"*"}
        ]
        assert stars == []
    finally:
        plt.close(fig)


def test_p_values_are_not_printed_on_the_figure(figure):
    """House style: stars on the figure, p-values in the CSV."""
    printed = " ".join(t.get_text() for ax in figure.axes for t in ax.texts)
    assert "p=" not in printed


def test_unpalettised_odor_falls_back_without_raising():
    # Benzaldehyde used to be the odor with no entry; it has brown now, so this
    # needs an odor that is genuinely absent from the palette.
    cmp_benz = Comparison(
        odor="Nonanal",
        concentration="0.1%",
        naive_dataset="RandomPanel-24-0.1",
        train_dataset="3OCT-Training-24-0.1",
        control_dataset="3OCT-Control-24-0.1",
    )
    fig = render_comparison(cmp_benz, _groups([1, 2], [3, 4], [0, 1]))
    try:
        trained = [p for p in fig.axes[0].patches if p.get_width() > 0][1]
        assert matplotlib.colors.to_hex(trained.get_facecolor()) == pal.TRAIN_COLOR
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------
# Stats sidecar
# --------------------------------------------------------------------------


def test_stats_rows_cover_both_metrics_and_every_pair():
    groups = _groups([0, 1, 2] * 4, [5, 4, 3] * 4, [0, 0, 1] * 4)
    rows = stats_rows(CMP, groups)
    assert set(rows["metric"]) == {"mean_score", "response_rate"}
    assert len(rows) == 2 * (3 + 1)  # three pairs + the omnibus, per metric


def test_stats_rows_carry_the_numbers_the_figure_shows():
    groups = _groups([0, 1, 2] * 4, [5, 4, 3] * 4, [0, 0, 1] * 4)
    rows = stats_rows(CMP, groups)
    score_pair = rows[
        (rows["metric"] == "mean_score") & (rows["comparison"] == "Naive vs Trained")
    ].iloc[0]
    assert score_pair["n_a"] == 12 and score_pair["n_b"] == 12
    assert 0.0 <= score_pair["p_adj"] <= 1.0
    assert score_pair["odor"] == "3-Octanol"
    assert score_pair["concentration"] == "0.1%"


def test_stats_rows_record_the_cohort_datasets():
    rows = stats_rows(CMP, _groups([1, 2], [3, 4], [0, 1]))
    assert "RandomPanel-24-0.1" in set(rows["naive_dataset"])
    assert "3OCT-Training-24-0.1" in set(rows["train_dataset"])
    assert "3OCT-Control-24-0.1" in set(rows["control_dataset"])
