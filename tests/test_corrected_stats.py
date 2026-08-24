"""Known-answer tests for :mod:`scripts.analysis.corrected_stats`.

Every claim the corrected figure set makes rests on these functions, so each is
checked against an *independent* implementation (statsmodels, scipy) or against
a hand-computable case -- never against another function in the same module.
"""
from __future__ import annotations

import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.analysis import corrected_stats as cs  # noqa: E402

statsmodels = pytest.importorskip("statsmodels")
from statsmodels.stats.multitest import multipletests  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402


# ---------------------------------------------------------------------------
# Multiplicity -- checked against statsmodels.multipletests
# ---------------------------------------------------------------------------

P_FAMILIES = [
    [0.001, 0.01, 0.02, 0.04, 0.2, 0.9],
    [0.04, 0.04, 0.04],
    [0.5],
    [0.0001, 0.6, 0.03, 0.049, 0.051, 0.3, 0.7, 0.9],
]


@pytest.mark.parametrize("pvals", P_FAMILIES)
def test_holm_matches_statsmodels(pvals):
    expected = multipletests(pvals, method="holm")[1]
    np.testing.assert_allclose(cs.holm_adjust(pvals), expected, rtol=1e-12)


@pytest.mark.parametrize("pvals", P_FAMILIES)
def test_bh_matches_statsmodels(pvals):
    expected = multipletests(pvals, method="fdr_bh")[1]
    np.testing.assert_allclose(cs.bh_adjust(pvals), expected, rtol=1e-12)


def test_adjusters_pass_nan_through_and_shrink_the_family():
    """A NaN must not silently count as a test in the family size."""
    raw = [0.01, float("nan"), 0.02]
    holm = cs.holm_adjust(raw)
    assert math.isnan(holm[1])
    # Two live tests, not three: the smallest is multiplied by 2, not 3.
    assert holm[0] == pytest.approx(0.02)


def test_holm_is_never_weaker_than_bh():
    for pvals in P_FAMILIES:
        holm, bh = cs.holm_adjust(pvals), cs.bh_adjust(pvals)
        assert np.all(holm >= bh - 1e-12)


# ---------------------------------------------------------------------------
# Wilson CI -- checked against statsmodels.proportion_confint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k,n", [(0, 11), (1, 11), (8, 11), (11, 11), (9, 20), (20, 20)])
def test_wilson_matches_statsmodels(k, n):
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    got = cs.wilson_ci(k, n)
    assert got == pytest.approx((lo, hi), abs=1e-12)


def test_wilson_stays_inside_the_unit_interval_at_the_boundary():
    assert cs.wilson_ci(0, 11)[0] == 0.0
    assert cs.wilson_ci(11, 11)[1] == 1.0


def test_wilson_of_empty_group_is_nan():
    assert all(math.isnan(v) for v in cs.wilson_ci(0, 0))


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def test_cliffs_delta_hand_computable_cases():
    assert cs.cliffs_delta([1, 2, 3], [4, 5, 6]) == pytest.approx(-1.0)
    assert cs.cliffs_delta([4, 5, 6], [1, 2, 3]) == pytest.approx(1.0)
    assert cs.cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "a,b",
    [
        ([0, 1, 2, 3, 4, 5], [0, 0, 1, 1, 2, 2]),
        ([-1, 0, 0, 2, 5], [1, 1, 3, 3]),
        ([2, 2, 2], [2, 2, 2, 2]),
    ],
)
def test_cliffs_delta_equals_rank_biserial_from_mannwhitney_u(a, b):
    """delta = 2U/(n1 n2) - 1 is the identity that makes it reportable as r."""
    from scipy.stats import mannwhitneyu

    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    expected = 2.0 * u / (len(a) * len(b)) - 1.0
    assert cs.cliffs_delta(a, b) == pytest.approx(expected)


def test_hodges_lehmann_hand_computable():
    # Pairwise differences are {1, 2, 3}; the median is 2.
    assert cs.hodges_lehmann([1, 2, 3], [0, 0, 0]) == pytest.approx(2.0)


def test_hodges_lehmann_is_shift_equivariant():
    a, b = [0, 1, 2, 5], [1, 1, 3]
    base = cs.hodges_lehmann(a, b)
    shifted = cs.hodges_lehmann([v + 4 for v in a], b)
    assert shifted == pytest.approx(base + 4)


def test_bootstrap_cis_bracket_the_point_estimate():
    rng = np.random.default_rng(7)
    a = rng.integers(-1, 6, 25).astype(float)
    b = rng.integers(-1, 4, 30).astype(float)
    lo, hi = cs.cliffs_delta_ci(a, b, n_boot=2000, seed=1)
    assert lo <= cs.cliffs_delta(a, b) <= hi
    lo, hi = cs.hodges_lehmann_ci(a, b, n_boot=2000, seed=1)
    assert lo <= cs.hodges_lehmann(a, b) <= hi


def test_bootstrap_ci_is_deterministic_for_a_fixed_seed():
    a, b = [0, 1, 2, 3, 4], [0, 0, 1, 1, 2]
    first = cs.cliffs_delta_ci(a, b, n_boot=500, seed=3)
    second = cs.cliffs_delta_ci(a, b, n_boot=500, seed=3)
    assert first == second


# ---------------------------------------------------------------------------
# The binary battery
# ---------------------------------------------------------------------------


def test_battery_reproduces_scipy_on_a_textbook_table():
    from scipy.stats import chi2_contingency, fisher_exact

    k1, n1, k2, n2 = 8, 11, 4, 20
    res = cs.binary_test_battery(k1, n1, k2, n2)
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    assert res.p_fisher == pytest.approx(fisher_exact(table)[1])
    assert res.p_chi2 == pytest.approx(chi2_contingency(table, correction=False).pvalue)
    assert res.p_chi2_yates == pytest.approx(
        chi2_contingency(table, correction=True).pvalue
    )
    assert res.table == table


def test_chi2_admissibility_follows_cochrans_rule():
    """The whole reason to report min_expected next to the chi-square p."""
    thin = cs.binary_test_battery(1, 11, 0, 20)   # expected cells well under 5
    assert not thin.chi2_admissible
    assert thin.min_expected < 5.0

    fat = cs.binary_test_battery(30, 60, 15, 60)
    assert fat.chi2_admissible
    assert fat.min_expected >= 5.0


def test_yates_is_more_conservative_than_pearson():
    res = cs.binary_test_battery(8, 11, 4, 20)
    assert res.p_chi2_yates > res.p_chi2


def test_boschloo_is_at_least_as_powerful_as_fisher():
    """Boschloo dominates Fisher by construction; a violation means a bug."""
    for k1, n1, k2, n2 in [(8, 11, 4, 20), (9, 11, 9, 20), (2, 11, 9, 20)]:
        res = cs.binary_test_battery(k1, n1, k2, n2)
        assert res.p_boschloo <= res.p_fisher + 1e-9


def test_risk_difference_and_wilson_arms():
    res = cs.binary_test_battery(8, 11, 4, 20)
    assert res.risk_difference == pytest.approx(8 / 11 - 4 / 20)
    assert res.p1_ci == pytest.approx(cs.wilson_ci(8, 11))
    assert res.p2_ci == pytest.approx(cs.wilson_ci(4, 20))
    lo, hi = res.risk_difference_ci
    assert lo <= res.risk_difference <= hi


def test_degenerate_tables_return_nan_not_an_exception():
    empty = cs.binary_test_battery(0, 0, 4, 20)
    assert math.isnan(empty.p_fisher)
    # A zero column margin: Fisher is defined (p = 1), chi-square is not.
    zero_col = cs.binary_test_battery(0, 11, 0, 20)
    assert zero_col.p_fisher == pytest.approx(1.0)
    assert math.isnan(zero_col.p_chi2)
    assert not zero_col.chi2_admissible


# ---------------------------------------------------------------------------
# score_test
# ---------------------------------------------------------------------------


def test_score_test_reports_ties_as_inexact():
    """An integer -1..5 PER score is all ties, so the exact null is unusable."""
    tied = cs.score_test([1, 2, 2, 3], [0, 1, 1, 2])
    assert not tied.exact
    untied = cs.score_test([1.1, 2.2, 3.3], [0.4, 0.5, 0.6])
    assert untied.exact


def test_score_test_medians_and_iqr():
    res = cs.score_test([0, 1, 2, 3, 4], [0, 0, 0, 1, 1])
    assert res.median1 == pytest.approx(2.0)
    assert res.median2 == pytest.approx(0.0)
    assert res.iqr1 == pytest.approx((1.0, 3.0))


def test_score_test_p_matches_scipy():
    from scipy.stats import mannwhitneyu

    a, b = [0, 1, 2, 3, 4, 5], [0, 0, 1, 1, 2, 2]
    res = cs.score_test(a, b)
    assert res.p_value == pytest.approx(
        mannwhitneyu(a, b, alternative="two-sided").pvalue
    )


def test_score_test_on_empty_group_is_all_nan():
    res = cs.score_test([], [1, 2, 3])
    assert math.isnan(res.p_value)
    assert res.n1 == 0


# ---------------------------------------------------------------------------
# Cluster permutation
# ---------------------------------------------------------------------------


def test_cluster_permutation_enumerates_exhaustively_when_cheap():
    values = np.array([1.0, 1.0, 2.0, 5.0, 5.0, 6.0])
    clusters = ["a", "a", "b", "c", "c", "d"]
    arm = [True, True, True, False, False, False]
    res = cs.cluster_permutation_test(
        values, clusters, arm, cs.mean_difference
    )
    assert res.exhaustive
    assert res.n_permutations == math.comb(4, 2) == 6
    assert res.p_floor == pytest.approx(1 / 6)


def test_cluster_permutation_p_floor_is_the_real_resolution():
    """5 vs 8 batches admits 1287 assignments -- nothing below ~1/1288 exists."""
    values = np.arange(13, dtype=float)
    clusters = [f"c{i}" for i in range(13)]
    arm = [i < 5 for i in range(13)]
    res = cs.cluster_permutation_test(values, clusters, arm, cs.mean_difference)
    assert res.n_permutations == math.comb(13, 5) == 1287
    assert res.p_value >= res.p_floor


def test_cluster_permutation_null_effect_is_not_significant():
    rng = np.random.default_rng(11)
    values = rng.normal(size=12)
    clusters = [f"c{i // 2}" for i in range(12)]
    arm = [i < 6 for i in range(12)]
    res = cs.cluster_permutation_test(values, clusters, arm, cs.mean_difference)
    assert res.p_value > 0.05


def test_cluster_permutation_finds_a_clean_separation():
    """A perfectly separated 3-vs-3 design bottoms out at 2/(N+1), not 1/(N+1).

    When the two arms hold the same number of clusters the enumeration contains
    the exact mirror of the observed assignment, whose statistic is -observed.
    A two-sided test counts it, so the attainable floor is doubled. The 5-vs-8
    split this cohort actually has is unbalanced and has no mirror, which is why
    ``p_floor`` documents the one-sided bound.
    """
    values = np.array([10.0] * 6 + [0.0] * 6)
    clusters = [f"c{i // 2}" for i in range(12)]
    arm = [i < 6 for i in range(12)]
    res = cs.cluster_permutation_test(values, clusters, arm, cs.mean_difference)
    assert res.n_permutations == math.comb(6, 3) == 20
    assert res.p_value == pytest.approx(2 / 20)
    assert res.p_value == pytest.approx(2 * res.p_floor)


def test_cluster_permutation_is_weaker_than_ignoring_clusters():
    """The point of the test: batch-level exchangeability costs resolution.

    Six flies in three batches per arm cannot yield a p below 1/21, while the
    same data treated as 6 independent flies per arm would go far lower.
    """
    values = np.array([5.0] * 6 + [0.0] * 6)
    clusters = [f"c{i // 2}" for i in range(12)]
    arm = [i < 6 for i in range(12)]
    res = cs.cluster_permutation_test(values, clusters, arm, cs.mean_difference)
    fly_level = 1.0 / math.comb(12, 6)
    assert res.p_value > fly_level


def test_cluster_spanning_both_arms_is_rejected():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    clusters = ["a", "a", "b", "b"]
    arm = [True, False, True, False]  # every cluster straddles both arms
    with pytest.raises(ValueError, match="nested"):
        cs.cluster_permutation_test(values, clusters, arm, cs.mean_difference)


def test_cluster_permutation_observed_matches_the_direct_statistic():
    values = np.array([3.0, 4.0, 1.0, 0.0])
    clusters = ["a", "a", "b", "b"]
    arm = [True, True, False, False]
    res = cs.cluster_permutation_test(values, clusters, arm, cs.mean_difference)
    assert res.observed == pytest.approx(3.5 - 0.5)


def test_exhaustive_null_contains_the_observed_assignment():
    values = np.array([3.0, 4.0, 1.0, 0.0, 2.0, 2.0])
    clusters = ["a", "a", "b", "b", "c", "c"]
    arm = [True, True, False, False, False, False]
    res = cs.cluster_permutation_test(values, clusters, arm, cs.mean_difference)
    assert np.any(np.isclose(res.null, res.observed))
    assert res.n_permutations == len(list(combinations(range(3), 1)))
