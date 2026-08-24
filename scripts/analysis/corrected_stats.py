#!/usr/bin/env python3
"""Statistics core for the corrected trained-vs-control PER figures.

Everything here is deliberately test-driven against an independent
implementation (``statsmodels``/``scipy``) rather than against itself, because
the whole point of the corrected figure set is that the numbers survive an
outside check.

What this module adds over what ``score_summary`` / ``pubfig_score_train_vs_control``
already do, and *why* each addition is not cosmetic:

1. **Multiplicity.** The existing trained-vs-control panels run one test per
   odor presentation (8 here) and star the raw p. :func:`holm_adjust` and
   :func:`bh_adjust` give the family-wise and the false-discovery reading of the
   same family.
2. **Effect sizes.** A p-value without an effect size is not reportable in the
   current PER literature. :func:`cliffs_delta` (identical to the rank-biserial
   correlation for Mann-Whitney) and :func:`hodges_lehmann` cover the ordinal
   scores; the odds ratio and risk difference cover the binary rate.
3. **Test choice for the binary rate.** :func:`binary_test_battery` runs
   Fisher, Pearson chi-square, Yates-corrected chi-square, Barnard and Boschloo
   on the same 2x2 and reports the minimum expected cell count, so "is the
   chi-square even admissible here" is answered in the output instead of
   assumed.
4. **Non-independence.** Flies are not independent: they arrive in batches
   (one fly folder = one day/rig/odor-bottle), and in this cohort every batch is
   *entirely* trained or *entirely* control, so batch cannot be conditioned on
   -- it is collinear with the effect of interest. The honest response is to
   make the batch the unit of exchangeability:
   :func:`cluster_permutation_test` permutes whole batches between arms,
   enumerating all C(n, k) assignments exactly when that is cheap. A p-value
   that survives batch-level permutation is not a batch artefact.

Nothing here re-implements anything scipy does correctly; ``fisher_exact``,
``mannwhitneyu``, ``barnard_exact`` and ``boschloo_exact`` are called directly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.stats import (
    barnard_exact,
    boschloo_exact,
    chi2_contingency,
    fisher_exact,
    mannwhitneyu,
)
from scipy.stats.contingency import odds_ratio as _sp_odds_ratio

Z95 = 1.959963984540054

__all__ = [
    "holm_adjust",
    "bh_adjust",
    "wilson_ci",
    "cliffs_delta",
    "cliffs_delta_ci",
    "hodges_lehmann",
    "hodges_lehmann_ci",
    "BinaryTests",
    "binary_test_battery",
    "ScoreTest",
    "score_test",
    "cluster_permutation_test",
    "PermutationResult",
]


# ---------------------------------------------------------------------------
# Multiplicity
# ---------------------------------------------------------------------------


def holm_adjust(pvals: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (NaNs pass through).

    Same contract as ``randompanel_conc_comparison.holm_adjust``; duplicated
    here only so this module has no import cycle back into a figure script.
    """
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    m = finite.size
    running = 0.0
    for rank, idx in enumerate(finite[np.argsort(p[finite])]):
        running = max(running, (m - rank) * p[idx])
        out[idx] = min(1.0, running)
    return out


def bh_adjust(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg step-up adjusted p-values (NaNs pass through).

    The FDR reading of the same family Holm controls family-wise. Reported
    beside Holm because a 7-odor generalisation panel is exactly the setting
    where FWER is the wrong loss function -- a single false "this odor also
    moved" does not overturn the CS+ claim.
    """
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    m = finite.size
    if m == 0:
        return out
    order = finite[np.argsort(p[finite])]
    running = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        running = min(running, m / (rank + 1) * p[idx])
        out[idx] = min(1.0, running)
    return out


# ---------------------------------------------------------------------------
# Proportions
# ---------------------------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k/n``.

    Wilson rather than Wald: at n=11 with a proportion near 0 or 1 the Wald
    interval runs off the end of the axis and has real coverage far below 95%.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def _newcombe_rd_ci(
    k1: int, n1: int, k2: int, n2: int, z: float = Z95
) -> tuple[float, float]:
    """Newcombe's method 10 CI for a difference of two proportions.

    Built from the two Wilson intervals, so it inherits their behaviour at the
    boundary instead of producing a difference interval that includes values
    outside [-1, 1].
    """
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    l1, u1 = wilson_ci(k1, n1, z)
    l2, u2 = wilson_ci(k2, n2, z)
    d = k1 / n1 - k2 / n2
    lower = d - math.sqrt((k1 / n1 - l1) ** 2 + (u2 - k2 / n2) ** 2)
    upper = d + math.sqrt((u1 - k1 / n1) ** 2 + (k2 / n2 - l2) ** 2)
    return (max(-1.0, lower), min(1.0, upper))


@dataclass(frozen=True)
class BinaryTests:
    """Every defensible p-value for one 2x2, plus what makes them differ."""

    k1: int
    n1: int
    k2: int
    n2: int
    p_fisher: float
    p_chi2: float          # Pearson, no continuity correction
    p_chi2_yates: float    # Yates continuity-corrected
    p_barnard: float
    p_boschloo: float
    min_expected: float    # smallest expected cell count
    chi2_admissible: bool  # Cochran's rule: all expected >= 5
    odds_ratio: float
    odds_ratio_ci: tuple[float, float]
    risk_difference: float
    risk_difference_ci: tuple[float, float]
    p1_ci: tuple[float, float]
    p2_ci: tuple[float, float]

    @property
    def p1(self) -> float:
        return self.k1 / self.n1 if self.n1 else float("nan")

    @property
    def p2(self) -> float:
        return self.k2 / self.n2 if self.n2 else float("nan")

    @property
    def table(self) -> list[list[int]]:
        return [[self.k1, self.n1 - self.k1], [self.k2, self.n2 - self.k2]]


def binary_test_battery(k1: int, n1: int, k2: int, n2: int) -> BinaryTests:
    """Fisher / chi-square / Barnard / Boschloo on one 2x2, with effect sizes.

    The four p-values disagree by construction, and the disagreement is the
    point of reporting them together:

    * **Fisher** conditions on both margins. Conservative -- its actual type-I
      rate sits below nominal alpha because the conditional reference set is
      coarse at small n. This is what the existing figures use.
    * **Pearson chi-square** is an asymptotic approximation and is *anti*-
      conservative when expected counts are small; ``min_expected`` and
      ``chi2_admissible`` say whether it is usable at all here (Cochran: every
      expected cell >= 5).
    * **Yates-corrected chi-square** over-corrects toward Fisher and is the
      least recommended of the four in the modern literature; included because
      it is still what many PER papers report as "chi-square".
    * **Barnard / Boschloo** are unconditional exact tests: they keep exact
      type-I control while recovering the power Fisher gives away. Boschloo is
      uniformly at least as powerful as Fisher.

    A degenerate table (an empty arm) yields NaNs rather than an exception, so
    a sweep over odors does not die on one missing cell.
    """
    if n1 <= 0 or n2 <= 0:
        nan = float("nan")
        return BinaryTests(
            k1, n1, k2, n2, nan, nan, nan, nan, nan, nan, False, nan,
            (nan, nan), nan, (nan, nan), (nan, nan), (nan, nan),
        )

    table = np.array([[k1, n1 - k1], [k2, n2 - k2]], dtype=int)
    p_fisher = float(fisher_exact(table, alternative="two-sided")[1])

    # A zero margin makes chi-square undefined; Fisher still returns 1.0.
    if table.sum(axis=0).min() == 0 or table.sum(axis=1).min() == 0:
        p_chi2 = p_chi2_yates = float("nan")
        min_expected = float("nan")
    else:
        chi_plain = chi2_contingency(table, correction=False)
        chi_yates = chi2_contingency(table, correction=True)
        p_chi2 = float(chi_plain.pvalue)
        p_chi2_yates = float(chi_yates.pvalue)
        min_expected = float(np.min(chi_plain.expected_freq))

    # Unconditional exact tests need both columns non-degenerate.
    try:
        p_barnard = float(barnard_exact(table, alternative="two-sided").pvalue)
    except ValueError:
        p_barnard = float("nan")
    try:
        p_boschloo = float(boschloo_exact(table, alternative="two-sided").pvalue)
    except ValueError:
        p_boschloo = float("nan")

    res = _sp_odds_ratio(table, kind="conditional")
    or_point = float(res.statistic)
    try:
        lo, hi = res.confidence_interval(confidence_level=0.95)
        or_ci = (float(lo), float(hi))
    except Exception:  # noqa: BLE001 -- boundary tables have no finite CI
        or_ci = (float("nan"), float("nan"))

    rd = k1 / n1 - k2 / n2
    return BinaryTests(
        k1=k1, n1=n1, k2=k2, n2=n2,
        p_fisher=p_fisher,
        p_chi2=p_chi2,
        p_chi2_yates=p_chi2_yates,
        p_barnard=p_barnard,
        p_boschloo=p_boschloo,
        min_expected=min_expected,
        chi2_admissible=bool(np.isfinite(min_expected) and min_expected >= 5.0),
        odds_ratio=or_point,
        odds_ratio_ci=or_ci,
        risk_difference=float(rd),
        risk_difference_ci=_newcombe_rd_ci(k1, n1, k2, n2),
        p1_ci=wilson_ci(k1, n1),
        p2_ci=wilson_ci(k2, n2),
    )


# ---------------------------------------------------------------------------
# Ordinal scores
# ---------------------------------------------------------------------------


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta: P(a > b) - P(a < b), in [-1, 1].

    The rank-biserial correlation for a Mann-Whitney U is the same number
    (``delta = 2U/(n1*n2) - 1``), which is what the tests assert. Reported
    because "trained scored higher" needs a magnitude, and a mean difference on
    an ordinal -1..5 score is not one.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    diff = np.sign(x[:, None] - y[None, :])
    return float(diff.sum() / (x.size * y.size))


def cliffs_delta_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for Cliff's delta (resampling flies)."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size < 2 or y.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draws[i] = cliffs_delta(
            x[rng.integers(0, x.size, x.size)], y[rng.integers(0, y.size, y.size)]
        )
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def hodges_lehmann(a: Sequence[float], b: Sequence[float]) -> float:
    """Hodges-Lehmann shift: the median of all pairwise differences a - b.

    The location estimate Mann-Whitney is actually testing, and the honest
    companion to a rank test -- unlike a difference of means it does not
    pretend the -1..5 score is an interval scale.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    return float(np.median((x[:, None] - y[None, :]).ravel()))


def hodges_lehmann_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the Hodges-Lehmann shift."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size < 2 or y.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draws[i] = hodges_lehmann(
            x[rng.integers(0, x.size, x.size)], y[rng.integers(0, y.size, y.size)]
        )
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


@dataclass(frozen=True)
class ScoreTest:
    """Mann-Whitney plus everything needed to report it properly."""

    n1: int
    n2: int
    u_statistic: float
    p_value: float
    exact: bool
    delta: float                     # Cliff's delta == rank-biserial r
    delta_ci: tuple[float, float]
    shift: float                     # Hodges-Lehmann
    shift_ci: tuple[float, float]
    median1: float
    median2: float
    iqr1: tuple[float, float]
    iqr2: tuple[float, float]
    mean1: float
    mean2: float
    sem1: float
    sem2: float


def _iqr(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(values, 25)), float(np.percentile(values, 75)))


def score_test(
    a: Sequence[float], b: Sequence[float], *, seed: int = 0, n_boot: int = 10_000
) -> ScoreTest:
    """Two-sided Mann-Whitney with effect size, shift, and median/IQR.

    ``exact`` records whether scipy could use the exact null: with ties -- and
    an integer -1..5 PER score is nothing but ties -- it cannot, and falls back
    to the tie-corrected normal approximation. Saying so is part of reporting
    the test honestly.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        nan = float("nan")
        return ScoreTest(
            x.size, y.size, nan, nan, False, nan, (nan, nan), nan, (nan, nan),
            nan, nan, (nan, nan), (nan, nan), nan, nan, 0.0, 0.0,
        )
    ties = np.unique(np.concatenate([x, y])).size < (x.size + y.size)
    res = mannwhitneyu(x, y, alternative="two-sided", method="auto")
    return ScoreTest(
        n1=int(x.size),
        n2=int(y.size),
        u_statistic=float(res.statistic),
        p_value=float(res.pvalue),
        exact=not ties,
        delta=cliffs_delta(x, y),
        delta_ci=cliffs_delta_ci(x, y, n_boot=n_boot, seed=seed),
        shift=hodges_lehmann(x, y),
        shift_ci=hodges_lehmann_ci(x, y, n_boot=n_boot, seed=seed),
        median1=float(np.median(x)),
        median2=float(np.median(y)),
        iqr1=_iqr(x),
        iqr2=_iqr(y),
        mean1=float(x.mean()),
        mean2=float(y.mean()),
        sem1=float(x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else 0.0,
        sem2=float(y.std(ddof=1) / math.sqrt(y.size)) if y.size > 1 else 0.0,
    )


# ---------------------------------------------------------------------------
# Cluster (batch-level) permutation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PermutationResult:
    observed: float
    p_value: float
    n_permutations: int
    exhaustive: bool
    null: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    @property
    def p_floor(self) -> float:
        """Smallest two-sided p this permutation set could have produced.

        Under exhaustive enumeration the observed assignment is one of the N
        members, so nothing below ``1/N`` exists. When the two arms hold the
        *same* number of clusters the set also contains the exact mirror of
        every assignment (statistic ``-observed``), which a two-sided test also
        counts, doubling the floor to ``2/N``; the 5-vs-8 batch split in this
        cohort is unbalanced and has no mirror.
        """
        if self.n_permutations == 0:
            return float("nan")
        return (
            1.0 / self.n_permutations
            if self.exhaustive
            else 1.0 / (self.n_permutations + 1)
        )


def cluster_permutation_test(
    values: np.ndarray,
    clusters: Sequence,
    in_group_a: Sequence[bool],
    statistic: Callable[[np.ndarray, np.ndarray], float],
    *,
    max_exhaustive: int = 200_000,
    n_random: int = 20_000,
    seed: int = 0,
) -> PermutationResult:
    """Permute whole clusters between arms and re-evaluate ``statistic``.

    ``values`` is per-observation (one fly, or one fly's CS+ minus novel
    contrast). ``clusters`` labels the exchangeable unit -- here the fly folder,
    i.e. one day / rig / odor bottle. ``in_group_a`` says which arm each
    observation belongs to; it must be constant within a cluster, which is
    asserted, because a cluster split across arms would mean the design *can*
    condition on batch and this test is the wrong tool.

    When the number of cluster assignments C(n, k) is at most
    ``max_exhaustive`` every assignment is enumerated, so the p-value is exact
    rather than Monte-Carlo. ``p_floor`` reports the resolution: with 5 trained
    and 8 control batches there are only 1287 assignments, so no batch-level
    p-value below ~0.0016 exists no matter how large the effect.

    The p-value is the two-sided proportion of assignments whose statistic is at
    least as extreme in absolute value, with the usual +1 in numerator and
    denominator (Phipson & Smyth) so it can never be exactly zero.
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    in_group_a = np.asarray(in_group_a, dtype=bool)
    if values.shape != clusters.shape or values.shape != in_group_a.shape:
        raise ValueError("values, clusters and in_group_a must be the same length")

    unique = list(dict.fromkeys(clusters.tolist()))
    cluster_arm: dict = {}
    for c in unique:
        arms = set(in_group_a[clusters == c].tolist())
        if len(arms) != 1:
            raise ValueError(
                f"cluster {c!r} spans both arms; batch is not nested in group, "
                "so stratify (Cochran-Mantel-Haenszel) instead of permuting"
            )
        cluster_arm[c] = arms.pop()

    a_clusters = [c for c in unique if cluster_arm[c]]
    n_total, n_a = len(unique), len(a_clusters)
    observed = float(statistic(values[in_group_a], values[~in_group_a]))

    index_of = {c: i for i, c in enumerate(unique)}
    cluster_index = np.array([index_of[c] for c in clusters.tolist()])

    def stat_for(mask_clusters: np.ndarray) -> float:
        mask = mask_clusters[cluster_index]
        return float(statistic(values[mask], values[~mask]))

    n_choose = math.comb(n_total, n_a) if 0 < n_a < n_total else 1
    exhaustive = n_choose <= max_exhaustive
    if exhaustive:
        null = np.empty(n_choose, dtype=float)
        for i, combo in enumerate(combinations(range(n_total), n_a)):
            mask_clusters = np.zeros(n_total, dtype=bool)
            mask_clusters[list(combo)] = True
            null[i] = stat_for(mask_clusters)
        n_perm = n_choose
    else:
        rng = np.random.default_rng(seed)
        null = np.empty(n_random, dtype=float)
        for i in range(n_random):
            mask_clusters = np.zeros(n_total, dtype=bool)
            mask_clusters[rng.choice(n_total, n_a, replace=False)] = True
            null[i] = stat_for(mask_clusters)
        n_perm = n_random

    finite = null[np.isfinite(null)]
    extreme = int(np.sum(np.abs(finite) >= abs(observed) - 1e-12))
    if exhaustive:
        # The observed assignment is itself a member of the enumerated set, so
        # p = extreme/N is already exact. Adding the Phipson-Smyth +1 here
        # would count it twice.
        p = extreme / finite.size if finite.size else float("nan")
    else:
        # Monte-Carlo: the observed assignment is not guaranteed to be in the
        # sample, so add it to both numerator and denominator (Phipson & Smyth
        # 2010) -- this is what keeps a sampled p from ever being exactly zero.
        p = (extreme + 1) / (finite.size + 1)
    return PermutationResult(
        observed=observed,
        p_value=float(min(1.0, p)),
        n_permutations=int(finite.size),
        exhaustive=bool(exhaustive),
        null=finite,
    )


def mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled mean of ``a`` minus pooled mean of ``b`` (never a mean of means)."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(a.mean() - b.mean())
