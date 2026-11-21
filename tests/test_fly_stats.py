"""fbpipe.analysis.fly_stats: exact tie-aware Wilcoxon, fly-cluster CIs, controlled tests.

Reference values come from brute force (every sign flip / every batch split), never
from the implementation under test.
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest, rankdata, wilcoxon

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src")]

from fbpipe.analysis import fly_stats as fs  # noqa: E402


# ── brute-force references ───────────────────────────────────────────────────
def brute_signed_rank_p(d) -> float:
    """Two-sided p over all 2^n sign flips of the midranked |d| (zeros dropped)."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    r = rankdata(np.abs(d))
    mu = r.sum() / 2
    obs = r[d > 0].sum()
    hits = total = 0
    for signs in itertools.product([False, True], repeat=d.size):
        t = r[np.array(signs, dtype=bool)].sum()
        hits += abs(t - mu) >= abs(obs - mu) - 1e-9
        total += 1
    return hits / total


def wilson(k: float, n: float, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


# ── Wilcoxon signed-rank ─────────────────────────────────────────────────────
def test_pinned_tie_case_is_no_longer_called_significant():
    # The audit's case: scipy 1.12's exact null ignores the ties and says 0.047.
    d = [-2, -2, -2, -2, -1, -1, 1]
    assert fs.wilcoxon_signed_rank_p(d) == pytest.approx(0.0625, abs=1e-12)
    assert fs.wilcoxon_signed_rank_p(d) == pytest.approx(brute_signed_rank_p(d), abs=1e-12)


@pytest.mark.parametrize("seed", range(40))
def test_matches_brute_force_with_ties_and_zeros(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 13))
    d = rng.choice([-3, -2, -1, 0, 0, 1, 1, 2, 3, 4], size=n).astype(float)
    got = fs.wilcoxon_signed_rank_p(d)
    if not np.any(d != 0):
        assert got is None
    else:
        assert got == pytest.approx(brute_signed_rank_p(d), abs=1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_matches_scipy_exact_when_there_are_no_ties(seed):
    rng = np.random.default_rng(100 + seed)
    d = rng.normal(size=int(rng.integers(5, 25)))
    assert fs.wilcoxon_signed_rank_p(d) == pytest.approx(
        wilcoxon(d, method="exact").pvalue, abs=1e-10)


def test_zeros_and_nans_are_dropped():
    d = [1.0, 2.0, -0.5, 3.0, 4.0, 5.0]
    assert fs.wilcoxon_signed_rank_p(d + [0.0, 0.0, np.nan]) == pytest.approx(
        fs.wilcoxon_signed_rank_p(d), abs=1e-15)


def test_symmetric_in_sign():
    d = np.array([3, 1, 1, -2, 4, 4, 4, 2, -1], dtype=float)
    assert fs.wilcoxon_signed_rank_p(d) == pytest.approx(fs.wilcoxon_signed_rank_p(-d))


def test_all_positive_is_two_over_two_to_the_n():
    assert fs.wilcoxon_signed_rank_p([1, 2, 3, 4, 5]) == pytest.approx(2 / 32)
    assert fs.wilcoxon_signed_rank_p([1, 1, 1, 1, 1, 1]) == pytest.approx(2 / 64)


def test_undefined_cases_return_none():
    assert fs.wilcoxon_signed_rank_p([]) is None
    assert fs.wilcoxon_signed_rank_p([0, 0, 0]) is None
    assert fs.wilcoxon_signed_rank_p([np.nan]) is None
    assert fs.wilcoxon_signed_rank_p([1, 2, 3], min_pairs=6) is None
    assert fs.wilcoxon_signed_rank_p([1, 2, 3, 4, 5, 6], min_pairs=6) is not None


def test_p_never_exceeds_one():
    assert fs.wilcoxon_signed_rank_p([1, -1]) == pytest.approx(1.0)
    assert fs.wilcoxon_signed_rank_p([1, -1, 2, -2, 3, -3]) <= 1.0


def test_large_n_normal_approximation_tracks_the_exact_null():
    rng = np.random.default_rng(7)
    d = rng.choice([-2, -1, 1, 2, 3], size=150).astype(float)
    exact = fs.wilcoxon_signed_rank_p(d)
    approx = fs.wilcoxon_signed_rank_p(d, exact_max_n=10)
    assert approx == pytest.approx(exact, abs=0.01)


# ── McNemar ──────────────────────────────────────────────────────────────────
def test_mcnemar_is_the_binomial_test_on_discordant_pairs():
    pre = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 1], bool)
    post = np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 1], bool)
    # gained 5, lost 1
    assert fs.mcnemar_exact_p(pre, post) == pytest.approx(binomtest(5, 6, 0.5).pvalue)


def test_mcnemar_without_discordant_pairs_is_undefined():
    assert fs.mcnemar_exact_p([True, False], [True, False]) is None
    assert fs.mcnemar_exact_p([], []) is None


# ── fly-cluster confidence intervals ─────────────────────────────────────────
def test_rate_ci_with_one_trial_per_fly_is_plain_wilson():
    resp = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    ids = list("abcdefghij")
    pct, lo, hi = fs.fly_cluster_rate_ci(resp, ids)
    w_lo, w_hi = wilson(3, 10)
    assert pct == pytest.approx(30.0)
    assert (lo, hi) == pytest.approx((100 * w_lo, 100 * w_hi))


def test_rate_ci_counts_flies_not_trials_when_flies_are_all_or_none():
    # 10 flies x 8 odors; 3 flies respond to everything, 7 to nothing. 80 trials,
    # but the evidence is 10 flies: the interval must be about as wide as 3 of 10.
    ids = np.repeat(list("abcdefghij"), 8)
    resp = np.repeat([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 8)
    pct, lo, hi = fs.fly_cluster_rate_ci(resp, ids)
    t_lo, t_hi = wilson(24, 80)                 # what counting trials gives
    f_lo, f_hi = wilson(3, 10)                  # one value per fly
    assert pct == pytest.approx(30.0)
    assert (hi - lo) > 1.5 * 100 * (t_hi - t_lo)
    assert (hi - lo) == pytest.approx(100 * (f_hi - f_lo), rel=0.15)


def test_rate_ci_with_independent_trials_stays_near_the_trial_count():
    rng = np.random.default_rng(11)
    ids = np.repeat(np.arange(60), 8)
    resp = rng.random(ids.size) < 0.3
    pct, lo, hi = fs.fly_cluster_rate_ci(resp, ids)
    t_lo, t_hi = wilson(int(resp.sum()), resp.size)
    assert (hi - lo) == pytest.approx(100 * (t_hi - t_lo), rel=0.3)


def test_rate_ci_at_zero_uses_the_fly_count():
    ids = np.repeat(list("abcde"), 4)
    pct, lo, hi = fs.fly_cluster_rate_ci(np.zeros(20, bool), ids)
    assert pct == 0.0 and lo == 0.0
    assert hi == pytest.approx(100 * wilson(0, 5)[1])


def test_rate_ci_point_estimate_pools_trials():
    # fly a: 1 of 1, fly b: 0 of 3 -> pooled 1/4 = 25 %, not the per-fly mean 50 %.
    pct, _, _ = fs.fly_cluster_rate_ci([1, 0, 0, 0], ["a", "b", "b", "b"])
    assert pct == pytest.approx(25.0)


def test_mean_ci_with_one_trial_per_fly_is_the_t_interval():
    from scipy.stats import t as student_t
    v = np.array([0.0, 1, 2, 2, 3, 5, -1, 1])
    m, lo, hi = fs.fly_cluster_mean_ci(v, np.arange(v.size))
    half = student_t.ppf(0.975, v.size - 1) * v.std(ddof=1) / math.sqrt(v.size)
    assert m == pytest.approx(v.mean())
    assert (lo, hi) == pytest.approx((v.mean() - half, v.mean() + half))


def test_mean_ci_pools_trials_and_widens_for_clustered_flies():
    ids = np.repeat(np.arange(6), 5)
    v = np.repeat([0.0, 0, 0, 4, 4, 4], 5)
    m, lo, hi = fs.fly_cluster_mean_ci(v, ids)
    assert m == pytest.approx(2.0)
    per_fly = np.array([0.0, 0, 0, 4, 4, 4])
    m2, lo2, hi2 = fs.fly_cluster_mean_ci(per_fly, np.arange(6))
    assert (hi - lo) == pytest.approx(hi2 - lo2, rel=1e-9)


def test_mean_ci_needs_two_flies():
    m, lo, hi = fs.fly_cluster_mean_ci([1.0, 2.0, 3.0], ["a", "a", "a"])
    assert m == pytest.approx(2.0) and np.isnan(lo) and np.isnan(hi)


# ── controlled (difference-in-differences) test ──────────────────────────────
def _groups():
    exposed = pd.DataFrame({
        "change": [1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
        "batch": ["e1"] * 4 + ["e2"] * 4 + ["e3"] * 4,
    })
    control = pd.DataFrame({
        "change": [0, 0, 1, 0, 0, -1, 0, 0, 1, 0, 0, 0],
        "batch": ["c1"] * 3 + ["c2"] * 3 + ["c3"] * 3 + ["c4"] * 3,
    })
    return exposed, control


def brute_batch_p(exposed, control):
    eb = exposed.groupby("batch")["change"].agg(["sum", "count"])
    cb = control.groupby("batch")["change"].agg(["sum", "count"])
    allb = pd.concat([eb, cb])
    s, c = allb["sum"].to_numpy(float), allb["count"].to_numpy(float)
    k = len(eb)
    obs = s[:k].sum() / c[:k].sum() - s[k:].sum() / c[k:].sum()
    hits = total = 0
    for pick in itertools.combinations(range(len(allb)), k):
        mask = np.zeros(len(allb), bool)
        mask[list(pick)] = True
        stat = s[mask].sum() / c[mask].sum() - s[~mask].sum() / c[~mask].sum()
        hits += abs(stat) >= abs(obs) - 1e-12
        total += 1
    return obs, hits / total


def test_controlled_test_estimate_and_exact_batch_permutation():
    exposed, control = _groups()
    r = fs.controlled_change_test(exposed, control, value="change", batch="batch")
    obs, p = brute_batch_p(exposed, control)
    assert r["estimate"] == pytest.approx(obs)
    assert r["estimate"] == pytest.approx(7 / 12 - 1 / 12)
    assert r["exposed_change"] == pytest.approx(7 / 12)
    assert r["control_change"] == pytest.approx(1 / 12)
    assert r["p_batch"] == pytest.approx(p)
    assert r["perm_exact"] is True
    assert (r["n_flies_exposed"], r["n_flies_control"]) == (12, 12)
    assert (r["n_batches_exposed"], r["n_batches_control"]) == (3, 4)
    assert r["ci_lo"] <= r["estimate"] <= r["ci_hi"]
    assert 0 < r["p_fly"] <= 1


def test_controlled_test_is_reproducible_and_monte_carlo_when_large():
    rng = np.random.default_rng(5)
    exposed = pd.DataFrame({"change": rng.integers(-1, 2, 120),
                            "batch": np.repeat([f"e{i}" for i in range(15)], 8)})
    control = pd.DataFrame({"change": rng.integers(-1, 2, 120),
                            "batch": np.repeat([f"c{i}" for i in range(15)], 8)})
    a = fs.controlled_change_test(exposed, control, value="change", batch="batch",
                                  n_perm=3000, n_boot=2000, seed=1)
    b = fs.controlled_change_test(exposed, control, value="change", batch="batch",
                                  n_perm=3000, n_boot=2000, seed=1)
    assert a == b
    assert a["perm_exact"] is False                 # C(30, 15) is far too many
    assert 1 / 3001 <= a["p_batch"] <= 1


def test_controlled_test_ignores_missing_values():
    exposed, control = _groups()
    exposed.loc[0, "change"] = np.nan
    r = fs.controlled_change_test(exposed, control, value="change", batch="batch")
    assert r["n_flies_exposed"] == 11
    assert r["exposed_change"] == pytest.approx(6 / 11)


# ── multiplicity ─────────────────────────────────────────────────────────────
def test_holm_and_bh_match_the_textbook_and_pass_nans():
    p = [0.01, 0.04, np.nan, 0.03, 0.2]
    holm = fs.holm_adjust(p)
    bh = fs.bh_adjust(p)
    assert np.isnan(holm[2]) and np.isnan(bh[2])
    assert holm[[0, 3, 1, 4]] == pytest.approx([0.04, 0.09, 0.09, 0.2])
    assert bh[[0, 3, 1, 4]] == pytest.approx([0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.2])


def test_fly_permutation_p_is_two_sided_and_reproducible():
    a = np.array([1.0, 1, 1, 1, 0, 1])
    b = np.array([0.0, 0, 0, 1, 0, 0])
    p1 = fs.fly_permutation_p(a, b, n_perm=5000, seed=3)
    p2 = fs.fly_permutation_p(b, a, n_perm=5000, seed=3)
    assert p1 == fs.fly_permutation_p(a, b, n_perm=5000, seed=3)
    assert p1 == pytest.approx(p2, abs=0.02)
    assert 0 < p1 < 0.1


# ── permutation-inversion CI (consistent with the batch permutation p) ───────
def brute_inversion_ci(exposed, control, grid):
    """Every shift d on a grid: subtract d from each exposed fly, exact batch-permutation
    p; the CI is the range of d with p >= 0.05."""
    keep = []
    for d in grid:
        e = exposed.assign(change=exposed["change"] - d)
        _, p = brute_batch_p(e, control)
        if p >= 0.05:
            keep.append(d)
    return min(keep), max(keep)


def test_permutation_ci_matches_brute_force_inversion():
    rng = np.random.default_rng(21)
    exposed = pd.DataFrame({"change": rng.normal(1.0, 1.0, 18),
                            "batch": np.repeat([f"e{i}" for i in range(6)], 3)})
    control = pd.DataFrame({"change": rng.normal(0.0, 1.0, 15),
                            "batch": np.repeat([f"c{i}" for i in range(5)], 3)})
    r = fs.controlled_change_test(exposed, control, value="change", batch="batch",
                                  ci="permutation")
    assert r["perm_exact"] is True and r["ci_method"] == "permutation"
    grid = np.linspace(r["estimate"] - 4, r["estimate"] + 4, 1601)
    lo, hi = brute_inversion_ci(exposed, control, grid)
    step = grid[1] - grid[0]
    assert r["ci_lo"] == pytest.approx(lo, abs=1.5 * step)
    assert r["ci_hi"] == pytest.approx(hi, abs=1.5 * step)


@pytest.mark.parametrize("seed", range(12))
def test_permutation_ci_excludes_zero_exactly_when_p_is_below_005(seed):
    rng = np.random.default_rng(300 + seed)
    shift = rng.choice([0.0, 0.4, 0.8, 1.2])
    exposed = pd.DataFrame({"change": rng.normal(shift, 1.0, 45),
                            "batch": np.repeat([f"e{i}" for i in range(15)], 3)})
    control = pd.DataFrame({"change": rng.normal(0.0, 1.0, 42),
                            "batch": np.repeat([f"c{i}" for i in range(14)], 3)})
    r = fs.controlled_change_test(exposed, control, value="change", batch="batch",
                                  n_perm=4000, seed=seed, ci="permutation")
    assert r["perm_exact"] is False
    assert r["ci_lo"] <= r["estimate"] <= r["ci_hi"]
    excludes_zero = r["ci_lo"] > 0 or r["ci_hi"] < 0
    assert excludes_zero == (r["p_batch"] < 0.05)


def test_default_ci_is_still_the_bootstrap():
    exposed, control = _groups()
    a = fs.controlled_change_test(exposed, control, value="change", batch="batch")
    assert a["ci_method"] == "bootstrap"
