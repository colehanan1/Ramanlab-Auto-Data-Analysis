"""Tests for :mod:`scripts.analysis.training_auc_vs_control_response`.

The question the module answers -- "do control flies that extend more during
odor-only conditioning respond more at test?" -- is one join and three
statistics. Each is checked against a hand-computable case or an independent
implementation (statsmodels, scipy), never against another function here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from scripts.analysis import training_auc_vs_control_response as mod  # noqa: E402

statsmodels = pytest.importorskip("statsmodels")


# ---------------------------------------------------------------------------
# Fixtures: a miniature cohort with a known answer
# ---------------------------------------------------------------------------

DATASET = "Hex-Control-24-0.1"

#: fly -> (AUC per training trial, score on presentation 1, score on 8)
_FLIES = {
    ("day_1", 1): ([0.0, 0.0, 0.0, 0.0], 0, 0),
    ("day_1", 2): ([10.0, 20.0, 30.0, 40.0], 0, 1),
    ("day_2", 1): ([100.0, 100.0, 100.0, 100.0], 4, 0),
    ("day_2", 2): ([200.0, 180.0, 160.0, 140.0], 5, 4),
}


def _training_frame() -> pd.DataFrame:
    rows = []
    for (fly, num), (aucs, _s1, _s8) in _FLIES.items():
        for trial, auc in enumerate(aucs, start=1):
            rows.append({
                "dataset": DATASET,
                "fly": fly,
                "fly_number": num,
                "trial_type": "training",
                "trial_label": f"training_{trial}_hexanol",
                "AUC-During": auc,
            })
    # A second dataset that must never leak into the Hex-Control table.
    rows.append({
        "dataset": "Hex-Training-24-0.1", "fly": "day_9", "fly_number": 1,
        "trial_type": "training", "trial_label": "training_1_hexanol",
        "AUC-During": 999.0,
    })
    return pd.DataFrame(rows)


def _testing_frame() -> pd.DataFrame:
    rows = []
    for (fly, num), (_aucs, s1, s8) in _FLIES.items():
        for trial, score in ((1, s1), (8, s8)):
            rows.append({
                "dataset": DATASET, "fly": fly, "fly_number": num,
                "trial_type": "testing",
                "trial_label": f"testing_{trial}_hexanol",
                "score": score, "prediction": int(score >= 2),
            })
        # A distractor odor: must not be mistaken for a hexanol presentation.
        rows.append({
            "dataset": DATASET, "fly": fly, "fly_number": num,
            "trial_type": "testing", "trial_label": "testing_4_citral",
            "score": 5, "prediction": 1,
        })
    return pd.DataFrame(rows)


def _testing_auc_frame() -> pd.DataFrame:
    """Testing-side envelope rows: the AUC the fly actually produced at test."""
    rows = []
    for (fly, num), (aucs, s1, s8) in _FLIES.items():
        for trial, score in ((1, s1), (8, s8)):
            rows.append({
                "dataset": DATASET, "fly": fly, "fly_number": num,
                "trial_type": "testing",
                "trial_label": f"testing_{trial}_hexanol",
                "AUC-During": float(score) * 40.0 + aucs[0],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trial-label parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,expected",
    [
        ("training_3_hexanol", (3, "hexanol")),
        ("testing_10_3-octonol", (10, "3-octonol")),
        ("testing_9_lightonly", (9, "lightonly")),
        ("TESTING_2_Hexanol", (2, "hexanol")),
        ("  training_1_hexanol  ", (1, "hexanol")),
        ("nonsense", None),
        ("training_x_hexanol", None),
    ],
)
def test_parse_trial(label, expected):
    assert mod.parse_trial(label) == expected


def test_training_odor_token_is_the_modal_token():
    """One stray mislabelled trial must not rename the conditioning odor."""
    df = _training_frame()
    df.loc[len(df)] = {
        "dataset": DATASET, "fly": "day_1", "fly_number": 1,
        "trial_type": "training", "trial_label": "training_5_citral",
        "AUC-During": 1.0,
    }
    assert mod.training_odor_token(df[df["dataset"] == DATASET]) == "hexanol"


def test_presentation_trials_finds_only_the_named_odor():
    test = _testing_frame()
    assert mod.presentation_trials(test, "hexanol") == [1, 8]
    assert mod.presentation_trials(test, "citral") == [4]
    assert mod.presentation_trials(test, "linalool") == []


# ---------------------------------------------------------------------------
# The join
# ---------------------------------------------------------------------------


def test_build_fly_table_shape_and_values():
    table = mod.build_fly_table(
        _training_frame(), _testing_frame(), dataset=DATASET
    )
    assert list(table.attrs["presentations"]) == [1, 8]
    assert table.attrs["odor_token"] == "hexanol"
    assert len(table) == 4

    row = table.set_index(["fly", "fly_number"]).loc[("day_2", 2)]
    assert row["mean_auc"] == pytest.approx(170.0)          # (200+180+160+140)/4
    assert row["auc_slope"] == pytest.approx(-20.0)          # -20 per trial
    assert row["score_1"] == 5
    assert row["score_8"] == 4
    assert row["per_1"] == 1
    assert row["mean_score"] == pytest.approx(4.5)

    zero = table.set_index(["fly", "fly_number"]).loc[("day_1", 1)]
    assert zero["mean_auc"] == pytest.approx(0.0)
    assert zero["auc_slope"] == pytest.approx(0.0)
    assert zero["per_any"] == 0


def test_build_fly_table_per_any_is_a_logical_or_over_presentations():
    """A fly that reacts on either hexanol presentation counts as a responder."""
    test = _testing_frame()
    # day_1/2 scored 0 then 1 -> no PER on either presentation.
    test.loc[
        (test["fly"] == "day_1") & (test["fly_number"] == 2)
        & (test["trial_label"] == "testing_8_hexanol"),
        ["score", "prediction"],
    ] = [3, 1]
    table = mod.build_fly_table(_training_frame(), test, dataset=DATASET)
    row = table.set_index(["fly", "fly_number"]).loc[("day_1", 2)]
    assert row["per_1"] == 0
    assert row["per_8"] == 1
    assert row["per_any"] == 1


def test_build_fly_table_drops_flies_missing_either_side():
    """A fly with conditioning but no test trials is dropped, not carried NaN."""
    train = _training_frame()
    train.loc[len(train)] = {
        "dataset": DATASET, "fly": "day_3", "fly_number": 1,
        "trial_type": "training", "trial_label": "training_1_hexanol",
        "AUC-During": 50.0,
    }
    table = mod.build_fly_table(train, _testing_frame(), dataset=DATASET)
    assert ("day_3", 1) not in set(zip(table["fly"], table["fly_number"]))
    assert len(table) == 4


def test_build_fly_table_ignores_other_datasets():
    table = mod.build_fly_table(
        _training_frame(), _testing_frame(), dataset=DATASET
    )
    assert "day_9" not in set(table["fly"])


def test_build_fly_table_honours_an_explicit_odor_token():
    """`--odor-token citral` scores the novel odor, not the conditioning one."""
    table = mod.build_fly_table(
        _training_frame(), _testing_frame(), dataset=DATASET, odor_token="citral"
    )
    assert table.attrs["odor_token"] == "citral"
    assert list(table.attrs["presentations"]) == [4]
    assert (table["score_4"] == 5).all()


def test_build_fly_table_empty_for_an_unknown_dataset():
    table = mod.build_fly_table(
        _training_frame(), _testing_frame(), dataset="Nope-24-1"
    )
    assert table.empty


# ---------------------------------------------------------------------------
# Slope
# ---------------------------------------------------------------------------


def test_auc_slope_matches_numpy_polyfit():
    series = pd.Series([5.0, 9.0, 11.0, 20.0], index=[1, 2, 3, 4])
    expected = np.polyfit(series.index.to_numpy(float), series.to_numpy(), 1)[0]
    assert mod.auc_slope(series) == pytest.approx(expected)


def test_auc_slope_needs_two_points():
    assert np.isnan(mod.auc_slope(pd.Series([3.0], index=[1])))
    assert np.isnan(mod.auc_slope(pd.Series([], dtype=float)))


# ---------------------------------------------------------------------------
# Logistic fit -- checked against statsmodels directly
# ---------------------------------------------------------------------------


def test_fit_logistic_matches_statsmodels():
    import statsmodels.api as sm

    rng = np.random.default_rng(7)
    x = rng.uniform(0, 300, 60)
    y = (rng.uniform(size=60) < 1 / (1 + np.exp(-(x - 150) / 60))).astype(int)

    fit = mod.fit_logistic(x, y)
    ref = sm.Logit(y, sm.add_constant(x)).fit(disp=0)

    assert fit.converged
    assert fit.intercept == pytest.approx(ref.params[0], rel=1e-6)
    assert fit.slope == pytest.approx(ref.params[1], rel=1e-6)
    assert fit.p_lr == pytest.approx(ref.llr_pvalue, rel=1e-6)
    assert fit.odds_ratio_100 == pytest.approx(np.exp(ref.params[1] * 100), rel=1e-6)


def test_fit_logistic_curve_is_monotone_and_bounded():
    x = np.array([0.0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    fit = mod.fit_logistic(x, y)
    assert fit.p_grid.min() >= 0.0 and fit.p_grid.max() <= 1.0
    assert np.all(np.diff(fit.p_grid) >= -1e-12)      # rising with AUC
    assert fit.slope > 0


def test_fit_logistic_degenerate_outcomes_do_not_raise():
    x = np.arange(10, dtype=float)
    for y in (np.zeros(10, dtype=int), np.ones(10, dtype=int)):
        fit = mod.fit_logistic(x, y)
        assert not fit.converged
        assert np.isnan(fit.p_lr)
        assert fit.p_grid.size == 0


def test_fit_logistic_ignores_nan_rows():
    x = np.array([0.0, 10, np.nan, 30, 40, 50, 60, 70, 80, 90])
    y = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1, 1])
    fit = mod.fit_logistic(x, y)
    assert fit.n == 9


# ---------------------------------------------------------------------------
# Equal-count bins and binomial rates
# ---------------------------------------------------------------------------


def test_equal_count_bins_splits_evenly_and_orders_by_value():
    values = np.array([50.0, 10, 30, 60, 20, 40])
    bins = mod.equal_count_bins(values, 3)
    assert sorted(np.bincount(bins).tolist()) == [2, 2, 2]
    # The two smallest values share bin 0, the two largest share bin 2.
    assert bins[np.argsort(values)[0]] == bins[np.argsort(values)[1]] == 0
    assert bins[np.argsort(values)[-1]] == bins[np.argsort(values)[-2]] == 2


def test_equal_count_bins_marks_nan_as_minus_one():
    bins = mod.equal_count_bins(np.array([1.0, np.nan, 3.0, 2.0]), 3)
    assert bins[1] == -1
    assert set(bins[[0, 2, 3]]) == {0, 1, 2}


def test_rate_by_bin_matches_hand_counts_and_wilson():
    from scripts.analysis.corrected_stats import wilson_ci

    values = np.arange(1.0, 7.0)                       # 1..6 -> bins 0,0,1,1,2,2
    outcome = np.array([0, 0, 0, 1, 1, 1])
    rates = mod.rate_by_bin(values, outcome, 3)
    assert [(r.k, r.n) for r in rates] == [(0, 2), (1, 2), (2, 2)]
    assert [r.rate for r in rates] == [0.0, 0.5, 1.0]
    assert rates[1].lo == pytest.approx(wilson_ci(1, 2)[0])
    assert rates[1].hi == pytest.approx(wilson_ci(1, 2)[1])
    # x is the bin's median AUC, so the marker sits over its own data.
    assert [r.x_center for r in rates] == [1.5, 3.5, 5.5]


def test_rate_by_bin_drops_empty_bins_rather_than_plotting_nan():
    rates = mod.rate_by_bin(np.array([1.0, 2.0]), np.array([0, 1]), 3)
    assert all(r.n > 0 for r in rates)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


@pytest.fixture()
def table():
    return mod.build_fly_table(_training_frame(), _testing_frame(), dataset=DATASET)


def test_per_figure_has_a_panel_per_presentation_plus_pooled(table):
    fig, stats = mod.figure_per(table, dataset=DATASET, odor_label="Hexanol")
    try:
        # 2 rows (group comparison, dose-response) x 3 columns (Hex 1, Hex 2, either)
        assert len(fig.axes) == 6
        assert [s["column"] for s in stats] == ["Hex 1", "Hex 2", "Hex 1 or 2"]
        assert all("mannwhitney_p" in s and "n_per" in s for s in stats)
    finally:
        matplotlib.pyplot.close(fig)


def test_score_figure_has_a_panel_per_presentation_plus_mean(table):
    fig, stats = mod.figure_score(table, dataset=DATASET, odor_label="Hexanol")
    try:
        assert len(fig.axes) == 6
        assert [s["column"] for s in stats] == ["Hex 1", "Hex 2", "Hex 1 & 2 mean"]
        assert all("spearman_rho" in s and "spearman_p" in s for s in stats)
    finally:
        matplotlib.pyplot.close(fig)


def test_figure_stats_match_scipy_on_the_fixture(table):
    from scipy.stats import mannwhitneyu, spearmanr

    _fig, per_stats = mod.figure_per(table, dataset=DATASET, odor_label="Hexanol")
    matplotlib.pyplot.close("all")
    auc = table["mean_auc"].to_numpy(float)
    per1 = table["per_1"].to_numpy(float)
    expected = mannwhitneyu(auc[per1 < 0.5], auc[per1 >= 0.5],
                            alternative="two-sided").pvalue
    assert per_stats[0]["mannwhitney_p"] == pytest.approx(expected)

    _fig, score_stats = mod.figure_score(table, dataset=DATASET, odor_label="Hexanol")
    matplotlib.pyplot.close("all")
    rho, p = spearmanr(auc, table["score_1"].to_numpy(float))
    assert score_stats[0]["spearman_rho"] == pytest.approx(rho)
    assert score_stats[0]["spearman_p"] == pytest.approx(p)


def test_palette_is_the_project_score_palette():
    """Score points reuse the pinned CVD-validated PRGn ramp, not new hues."""
    from scripts.analysis.score_scale_figure import SCORE_COLORS

    assert mod.SCORE_COLORS == SCORE_COLORS
    # The PER split reuses the same purple/green poles (validated dE 16.7).
    assert mod.NO_PER_COLOR == SCORE_COLORS[-1]
    assert mod.PER_COLOR == SCORE_COLORS[4]


def test_holm_adjustment_is_applied_across_the_three_columns(table):
    from scripts.analysis.corrected_stats import holm_adjust

    _fig, stats = mod.figure_per(table, dataset=DATASET, odor_label="Hexanol")
    matplotlib.pyplot.close("all")
    raw = [s["mannwhitney_p"] for s in stats]
    expected = holm_adjust(raw)
    got = [s["mannwhitney_p_holm"] for s in stats]
    np.testing.assert_allclose(got, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Per-training-trial analysis: does trial 1 alone predict the test response?
# ---------------------------------------------------------------------------


@pytest.fixture()
def full_table():
    return mod.build_fly_table(
        _training_frame(), _testing_frame(), dataset=DATASET,
        testing_auc_df=_testing_auc_frame(),
    )


def test_build_fly_table_keeps_one_auc_column_per_training_trial(full_table):
    assert list(full_table.attrs["training_trials"]) == [1, 2, 3, 4]
    row = full_table.set_index(["fly", "fly_number"]).loc[("day_2", 2)]
    assert [row[f"auc_t{t}"] for t in (1, 2, 3, 4)] == [200.0, 180.0, 160.0, 140.0]
    # The pooled column stays the mean of the per-trial columns.
    assert row["mean_auc"] == pytest.approx(np.mean([200.0, 180.0, 160.0, 140.0]))


def test_build_fly_table_carries_testing_auc_when_supplied(full_table):
    row = full_table.set_index(["fly", "fly_number"]).loc[("day_2", 2)]
    assert row["test_auc_1"] == pytest.approx(5 * 40.0 + 200.0)
    assert row["test_auc_8"] == pytest.approx(4 * 40.0 + 200.0)


def test_testing_auc_columns_absent_when_not_supplied(table):
    assert not [c for c in table.columns if c.startswith("test_auc_")]


def test_trial_correlation_matrix_shape_and_labels(full_table):
    mat = mod.trial_correlation_matrix(full_table)
    # One row per training trial plus the pooled mean.
    assert list(mat.rho.index) == ["Train 1", "Train 2", "Train 3", "Train 4", "Mean"]
    # Two presentations x three outcome kinds.
    assert list(mat.rho.columns) == [
        ("Hex 1", "AUC"), ("Hex 1", "Score"), ("Hex 1", "PER"),
        ("Hex 2", "AUC"), ("Hex 2", "Score"), ("Hex 2", "PER"),
    ]
    assert mat.rho.shape == mat.p_raw.shape == mat.p_bh.shape


def test_trial_correlation_matrix_rho_matches_scipy(full_table):
    from scipy.stats import spearmanr

    mat = mod.trial_correlation_matrix(full_table)
    for trial, label in ((1, "Train 1"), (3, "Train 3")):
        rho, p = spearmanr(full_table[f"auc_t{trial}"], full_table["score_1"])
        assert mat.rho.loc[label, ("Hex 1", "Score")] == pytest.approx(rho)
        assert mat.p_raw.loc[label, ("Hex 1", "Score")] == pytest.approx(p)
    rho, _p = spearmanr(full_table["mean_auc"], full_table["per_8"])
    assert mat.rho.loc["Mean", ("Hex 2", "PER")] == pytest.approx(rho)


def test_trial_correlation_matrix_bh_matches_statsmodels(full_table):
    from scripts.analysis.corrected_stats import bh_adjust

    mat = mod.trial_correlation_matrix(full_table)
    flat = mat.p_raw.to_numpy(float).ravel()
    expected = bh_adjust(flat)
    np.testing.assert_allclose(
        mat.p_bh.to_numpy(float).ravel(), expected, rtol=1e-12, equal_nan=True
    )


def test_trial_correlation_matrix_omits_auc_columns_without_testing_auc(table):
    mat = mod.trial_correlation_matrix(table)
    assert list(mat.rho.columns) == [
        ("Hex 1", "Score"), ("Hex 1", "PER"),
        ("Hex 2", "Score"), ("Hex 2", "PER"),
    ]


def test_trial_matrix_figure_renders(full_table):
    fig, payload = mod.figure_trial_matrix(
        full_table, dataset=DATASET, odor_label="Hexanol"
    )
    try:
        assert payload["n_flies"] == 4
        assert payload["n_tests"] == 5 * 6
        assert len(payload["cells"]) == 5 * 6
    finally:
        matplotlib.pyplot.close(fig)


def test_trial_profile_figure_has_one_panel_per_outcome_kind(full_table):
    fig, payload = mod.figure_trial_profile(
        full_table, dataset=DATASET, odor_label="Hexanol"
    )
    try:
        assert [s["outcome"] for s in payload] == ["AUC", "Score", "PER"]
        # Each panel carries a rho-per-training-trial series per presentation.
        for panel in payload:
            assert list(panel["series"]) == ["Hex 1", "Hex 2"]
            assert len(panel["series"]["Hex 1"]) == 4
    finally:
        matplotlib.pyplot.close(fig)


def test_trial_profile_falls_back_to_two_panels_without_testing_auc(table):
    fig, payload = mod.figure_trial_profile(
        table, dataset=DATASET, odor_label="Hexanol"
    )
    try:
        assert [s["outcome"] for s in payload] == ["Score", "PER"]
    finally:
        matplotlib.pyplot.close(fig)


# ---------------------------------------------------------------------------
# Plain-reading figures: bars + explicit significance marks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p,expected",
    [
        (0.0004, "***"),
        (0.001, "**"),
        (0.004, "**"),
        (0.01, "*"),
        (0.049, "*"),
        (0.05, "n.s."),
        (0.4, "n.s."),
        (float("nan"), "n.s."),
    ],
)
def test_stars(p, expected):
    assert mod.stars(p) == expected


def test_stars_boundaries_are_strict():
    """0.05 is not significant; 0.0499 is. No rounding a result into a star."""
    assert mod.stars(0.0499) == "*"
    assert mod.stars(0.05) == "n.s."
    assert mod.stars(0.0099) == "**"


def test_overview_figure_is_two_by_two(full_table):
    fig, stats = mod.figure_simple_overview(
        full_table, dataset=DATASET, odor_label="Hexanol", n_bins=2
    )
    try:
        assert len(fig.axes) == 4                      # PER / score x Hex 1 / Hex 2
        assert [s["presentation"] for s in stats] == ["Hex 1", "Hex 2"]
        for s in stats:
            assert {"per_bins", "score_bins", "per_low_vs_high", "score_low_vs_high",
                    "spearman_rho", "spearman_p"} <= set(s)
    finally:
        matplotlib.pyplot.close(fig)


def test_overview_low_vs_high_matches_scipy(full_table):
    from scipy.stats import fisher_exact, mannwhitneyu

    fig, stats = mod.figure_simple_overview(
        full_table, dataset=DATASET, odor_label="Hexanol", n_bins=2
    )
    matplotlib.pyplot.close(fig)

    auc = full_table["mean_auc"].to_numpy(float)
    bins = mod.equal_count_bins(auc, 2)
    low, high = bins == 0, bins == 1

    per = full_table["per_1"].to_numpy(float) >= 0.5
    k_hi, n_hi = int(per[high].sum()), int(high.sum())
    k_lo, n_lo = int(per[low].sum()), int(low.sum())
    expected = fisher_exact([[k_hi, n_hi - k_hi], [k_lo, n_lo - k_lo]])[1]
    assert stats[0]["per_low_vs_high"]["p_fisher"] == pytest.approx(expected)

    score = full_table["score_1"].to_numpy(float)
    expected = mannwhitneyu(score[high], score[low], alternative="two-sided").pvalue
    assert stats[0]["score_low_vs_high"]["p"] == pytest.approx(expected)


def test_overview_bins_are_ordered_low_to_high(full_table):
    _fig, stats = mod.figure_simple_overview(
        full_table, dataset=DATASET, odor_label="Hexanol", n_bins=2
    )
    matplotlib.pyplot.close("all")
    aucs = [b["auc_median"] for b in stats[0]["per_bins"]]
    assert aucs == sorted(aucs)


def test_trial_predictor_figure_one_panel_per_outcome(full_table):
    fig, payload = mod.figure_trial_predictors(
        full_table, presentation_index=0, dataset=DATASET, odor_label="Hexanol"
    )
    try:
        assert payload["presentation"] == "Hex 1"
        assert [p["outcome"] for p in payload["panels"]] == ["PER", "Score", "AUC"]
        for panel in payload["panels"]:
            # Six conditioning trials in the fixture is four, plus the mean bar.
            assert [b["label"] for b in panel["bars"]] == [
                "1", "2", "3", "4", "Mean"
            ]
            assert all("stars" in b for b in panel["bars"])
    finally:
        matplotlib.pyplot.close(fig)


def test_trial_predictor_figure_second_presentation(full_table):
    fig, payload = mod.figure_trial_predictors(
        full_table, presentation_index=1, dataset=DATASET, odor_label="Hexanol"
    )
    try:
        assert payload["presentation"] == "Hex 2"
        assert payload["trial"] == 8
    finally:
        matplotlib.pyplot.close(fig)


def test_trial_predictor_bars_match_the_matrix(full_table):
    mat = mod.trial_correlation_matrix(full_table)
    _fig, payload = mod.figure_trial_predictors(
        full_table, presentation_index=0, dataset=DATASET, odor_label="Hexanol"
    )
    matplotlib.pyplot.close("all")
    score_panel = next(p for p in payload["panels"] if p["outcome"] == "Score")
    expected = [
        float(mat.rho.loc[r, ("Hex 1", "Score")])
        for r in ["Train 1", "Train 2", "Train 3", "Train 4", "Mean"]
    ]
    got = [b["rho"] for b in score_panel["bars"]]
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_trial_predictor_drops_auc_panel_without_testing_auc(table):
    fig, payload = mod.figure_trial_predictors(
        table, presentation_index=0, dataset=DATASET, odor_label="Hexanol"
    )
    try:
        assert [p["outcome"] for p in payload["panels"]] == ["PER", "Score"]
    finally:
        matplotlib.pyplot.close(fig)


@pytest.mark.parametrize("k,n", [(0, 7), (0, 18), (10, 10), (13, 13)])
def test_asym_err_never_returns_a_negative_offset(k, n):
    """Wilson bounds miss an exact 0 or 1 by an ulp; raw yerr goes negative.

    matplotlib rejects the whole errorbar on a single negative offset, so an
    all-or-nothing bin took the figure down rather than drawing flat.
    """
    from scripts.analysis.corrected_stats import wilson_ci

    lo, hi = wilson_ci(k, n)
    rate = k / n
    assert min(rate - lo, hi - rate) < 0              # the trap this guards
    err = mod.asym_err([rate], [lo], [hi])
    assert err.shape == (2, 1)
    assert (err >= 0).all()


def test_asym_err_survives_every_small_wilson_interval():
    from scripts.analysis.corrected_stats import wilson_ci

    for n in range(1, 30):
        for k in range(n + 1):
            lo, hi = wilson_ci(k, n)
            assert (mod.asym_err([k / n], [lo], [hi]) >= 0).all()


def test_asym_err_scales_and_keeps_real_widths():
    err = mod.asym_err([0.5], [0.2], [0.9], scale=100.0)
    np.testing.assert_allclose(err[:, 0], [30.0, 40.0])


def test_overview_renders_with_an_all_or_nothing_bin(full_table):
    """A bin where every fly reacted must not blow up the error bars."""
    table = full_table.copy()
    table.attrs = dict(full_table.attrs)
    table["per_1"] = [0, 0, 1, 1]
    fig, _stats = mod.figure_simple_overview(
        table, dataset=DATASET, odor_label="Hexanol", n_bins=2
    )
    matplotlib.pyplot.close(fig)


def test_resolve_dataset_is_case_insensitive():
    train = _training_frame()
    assert mod.resolve_dataset(train, "hex-CONTROL-24-0.1") == DATASET
    assert mod.resolve_dataset(train, "  Hex-Control-24-0.1 ") == DATASET
    # Unknown names come back untouched, so the caller can log a clean warning.
    assert mod.resolve_dataset(train, "Nope-24-1") == "Nope-24-1"


def test_build_fly_table_matches_a_differently_cased_dataset():
    table = mod.build_fly_table(
        _training_frame(), _testing_frame(), dataset="HEX-control-24-0.1"
    )
    assert len(table) == 4


def test_bar_label_sits_outside_the_bar_on_both_sides():
    """A negative bar's label must hang below it, not sit inside it."""
    y, va = mod.bar_label_position(0.55)
    assert y > 0.55 and va == "bottom"
    y, va = mod.bar_label_position(-0.06)
    assert y < -0.06 and va == "top"
    # Exactly zero reads as a positive bar; the label goes above the baseline.
    y, va = mod.bar_label_position(0.0)
    assert y > 0 and va == "bottom"


def test_trial_predictor_axis_clears_the_lowest_label(full_table):
    """The y floor has to leave room for a label hanging under a negative bar."""
    table = full_table.copy()
    table.attrs = dict(full_table.attrs)
    # Make trial 2 anti-correlate with the Hex 1 score.
    table["auc_t2"] = -table["score_1"].to_numpy(float) * 10.0
    fig, payload = mod.figure_trial_predictors(
        table, presentation_index=0, dataset=DATASET, odor_label="Hexanol"
    )
    try:
        panel = next(p for p in payload["panels"] if p["outcome"] == "Score")
        lowest = min(b["rho"] for b in panel["bars"] if np.isfinite(b["rho"]))
        assert lowest < 0                                   # the case being tested
        label_y = mod.bar_label_position(lowest)[0]
        for ax in fig.axes:
            assert ax.get_ylim()[0] <= label_y
    finally:
        matplotlib.pyplot.close(fig)


def test_bin_ramp_is_the_validated_ordinal_scale():
    """low/mid/high is an ORDINAL bin, so it takes one hue, light->dark."""
    assert mod.BIN_RAMP == ("#86b6ef", "#3987e5", "#1c5cab")
    assert mod.bin_colors(2) == ["#86b6ef", "#1c5cab"]
    assert mod.bin_colors(3) == list(mod.BIN_RAMP)
