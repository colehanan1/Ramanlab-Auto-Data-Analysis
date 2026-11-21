"""Shared statistics with the fly (or the batch) as the unit.

One home for the tests every figure script needs, so they cannot drift apart:

* :func:`wilcoxon_signed_rank_p` -- exact two-sided Wilcoxon signed-rank p that is
  correct with ties. scipy 1.12's ``method="exact"`` (what the pipeline's envs
  run) uses the no-ties null even when the midranks tie, which on 0..5 PER
  scores is the usual case: ``[-2, -2, -2, -2, -1, -1, 1]`` reads 0.047 there and
  is really 0.0625.
* :func:`mcnemar_exact_p` -- the paired response-rate test (pre vs post, same flies).
* :func:`fly_cluster_rate_ci` / :func:`fly_cluster_mean_ci` -- 95 % intervals for a
  value pooled over trials when a fly contributes several trials (all odors
  pooled): the point estimate pools trials (never a mean of per-fly means), the
  width counts flies.
* :func:`controlled_change_test` -- did the exposed flies change MORE than the
  control flies? (difference in differences). p from permuting whole batches,
  because batches are nested in cohorts and a day effect would otherwise pass
  for an exposure effect.
* :func:`holm_adjust` / :func:`bh_adjust` -- multiplicity, NaNs passed through.
"""

from __future__ import annotations

import itertools
import math
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm, rankdata
from scipy.stats import t as student_t

Z95 = 1.959963984540054

#: Above this many non-zero differences the signed-rank null switches from the
#: exact distribution to its normal approximation (tie-corrected variance). No
#: pre/post comparison in this repo comes near it; the cap only bounds the cost.
WILCOXON_EXACT_MAX_N = 400

#: Exact batch permutation (every split enumerated) up to this many splits;
#: Monte Carlo above it.
MAX_EXACT_SPLITS = 200_000


# ── paired tests ─────────────────────────────────────────────────────────────
def wilcoxon_signed_rank_p(diffs: Sequence[float], *, min_pairs: int = 1,
                           exact_max_n: int = WILCOXON_EXACT_MAX_N) -> float | None:
    """Two-sided Wilcoxon signed-rank p of paired differences, exact with ties.

    Zeros and non-finite values are dropped (Wilcoxon's own convention, as the
    old helpers did). The |d| get midranks, and the null distribution of T+ (the
    rank sum of the positive differences) is enumerated over all 2^n sign flips
    *of those midranks* by dynamic programming over doubled ranks, so ties are
    exact rather than assumed away. p = P(|T+ - E| >= |t - E|).

    Returns None when fewer than ``min_pairs`` non-zero differences remain (the
    test cannot say anything), never a made-up 1.0.
    """
    d = np.asarray(diffs, dtype=float).ravel()
    d = d[np.isfinite(d) & (d != 0)]
    n = int(d.size)
    if n == 0 or n < int(min_pairs):
        return None
    ranks = rankdata(np.abs(d))                  # midranks: multiples of 0.5
    t_obs = float(ranks[d > 0].sum())
    mu = float(ranks.sum()) / 2.0
    if n > exact_max_n:
        sd = math.sqrt(float(np.sum(ranks ** 2)) / 4.0)
        if sd == 0:
            return 1.0
        z = abs(t_obs - mu) / sd
        return float(min(1.0, 2.0 * norm.sf(z)))
    w = np.rint(2.0 * ranks).astype(np.int64)    # doubled ranks are integers
    total = int(w.sum())
    prob = np.zeros(total + 1)
    prob[0] = 1.0
    for wi in w:
        shifted = np.zeros_like(prob)
        shifted[wi:] = prob[: total + 1 - wi]
        prob = 0.5 * (prob + shifted)
    s = np.arange(total + 1) / 2.0               # T+ on the original rank scale
    extreme = np.abs(s - mu) >= abs(t_obs - mu) - 1e-9
    return float(min(1.0, prob[extreme].sum()))


def mcnemar_exact_p(pre: Sequence[bool], post: Sequence[bool]) -> float | None:
    """Two-sided exact McNemar p for paired yes/no outcomes (pre vs post, same flies).

    Only discordant flies carry information: the binomial test of "gained" vs
    "lost" against 1:1. None with no discordant pair (undefined, not 1.0)."""
    a = np.asarray(pre, dtype=bool).ravel()
    b = np.asarray(post, dtype=bool).ravel()
    if a.size == 0 or a.size != b.size:
        return None
    gained = int(np.sum(~a & b))
    lost = int(np.sum(a & ~b))
    if gained + lost == 0:
        return None
    return float(binomtest(gained, gained + lost, 0.5, alternative="two-sided").pvalue)


def fly_permutation_p(a: Sequence[float], b: Sequence[float], *, n_perm: int = 20000,
                      seed: int = 0) -> float:
    """Two-sided permutation p for a difference in means between two sets of flies
    (one value per fly), (hits + 1) / (n_perm + 1)."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    obs = abs(x.mean() - y.mean())
    pool = np.concatenate([x, y])
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(int(n_perm)):
        rng.shuffle(pool)
        hits += abs(pool[: x.size].mean() - pool[x.size:].mean()) >= obs - 1e-12
    return (hits + 1) / (int(n_perm) + 1)


# ── fly-cluster intervals ────────────────────────────────────────────────────
def _per_fly(values: Sequence[float], fly_ids: Sequence) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=float).ravel()
    ids = np.asarray(fly_ids, dtype=object).ravel()
    if v.size != ids.size:
        raise ValueError(f"{v.size} values but {ids.size} fly ids")
    keep = np.isfinite(v)
    g = (pd.DataFrame({"v": v[keep], "id": ids[keep]})
         .groupby("id", sort=False)["v"].agg(["sum", "count"]))
    return g["sum"].to_numpy(dtype=float), g["count"].to_numpy(dtype=float)


def _wilson(p: float, n: float, z: float) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    lo = 0.0 if p <= 0.0 else max(0.0, centre - half)
    hi = 1.0 if p >= 1.0 else min(1.0, centre + half)
    return lo, hi


def fly_cluster_rate_ci(responded: Sequence[bool], fly_ids: Sequence, *,
                        z: float = Z95) -> tuple[float, float, float]:
    """(% responding, 95 % low, 95 % high), in percent, with flies as the unit.

    The point estimate pools trials (responses / trials). When every fly gave
    exactly one trial this is the plain Wilson interval on that count. When flies
    gave several (all odors pooled), the Wilson interval is taken at the
    effective sample size p(1-p)/V, with V the cluster-robust (ratio-estimator)
    variance over flies -- so seven trials from one fly that answers all or
    nothing count about as one fly, and genuinely independent trials still count
    as trials. With no responses (or all), where V is 0, the fly count is used.
    """
    k_i, n_i = _per_fly(np.asarray(responded, dtype=float), fly_ids)
    if n_i.size == 0:
        return float("nan"), float("nan"), float("nan")
    k, n, g = float(k_i.sum()), float(n_i.sum()), int(n_i.size)
    p = k / n
    if np.all(n_i == 1):
        n_eff = n
    elif 0.0 < p < 1.0 and g > 1:
        v = g / (g - 1) * float(np.sum((k_i - p * n_i) ** 2)) / (n * n)
        n_eff = n if v <= 0 else float(np.clip(p * (1.0 - p) / v, 1.0, n))
    else:
        n_eff = float(g)
    lo, hi = _wilson(p, n_eff, z)
    return 100.0 * p, 100.0 * lo, 100.0 * hi


def fly_cluster_mean_ci(values: Sequence[float], fly_ids: Sequence, *,
                        level: float = 0.95) -> tuple[float, float, float]:
    """(mean, low, high) of a value pooled over trials, with flies as the unit.

    Mean = sum of trial values / trials (never a mean of per-fly means). Interval
    = mean +/- t(flies - 1) x the cluster-robust standard error over flies; with
    one trial per fly that is exactly the ordinary t interval. Needs two flies."""
    s_i, n_i = _per_fly(values, fly_ids)
    if n_i.size == 0:
        return float("nan"), float("nan"), float("nan")
    n, g = float(n_i.sum()), int(n_i.size)
    m = float(s_i.sum()) / n
    if g < 2:
        return m, float("nan"), float("nan")
    v = g / (g - 1) * float(np.sum((s_i - m * n_i) ** 2)) / (n * n)
    half = float(student_t.ppf(0.5 + level / 2.0, g - 1)) * math.sqrt(v)
    return m, m - half, m + half


# ── controlled test: exposed change vs control change ────────────────────────
def _split_masks(k: int, kx: int, n_perm: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """Batch-to-group assignments: every split (exact) or ``n_perm`` random ones.
    Row = one split, True = that batch goes to the exposed side."""
    n_splits = math.comb(k, kx)
    if n_splits <= MAX_EXACT_SPLITS:
        masks = np.zeros((n_splits, k), dtype=bool)
        for i, pick in enumerate(itertools.combinations(range(k), kx)):
            masks[i, list(pick)] = True
        return masks, True
    idx = np.argsort(rng.random((int(n_perm), k)), axis=1)[:, :kx]
    masks = np.zeros((int(n_perm), k), dtype=bool)
    np.put_along_axis(masks, idx, True, axis=1)
    return masks, False


def _shifted_batch_p(s: np.ndarray, c: np.ndarray, kx: int, masks: np.ndarray, exact: bool):
    """p(d): the batch-permutation p after subtracting d from every exposed fly's value
    (H0: exposed = control + d). p(0) is the ordinary test."""
    m = masks.astype(float)
    den_in = m @ c
    den_out = c.sum() - den_in
    isx = np.zeros(c.size)
    isx[:kx] = 1.0

    def p_at(d: float) -> float:
        s_adj = s - d * c * isx
        num_in = m @ s_adj
        t = num_in / den_in - (s_adj.sum() - num_in) / den_out
        obs = abs(s_adj[:kx].sum() / c[:kx].sum() - s_adj[kx:].sum() / c[kx:].sum())
        hits = int(np.sum(np.abs(t) >= obs * (1 - 1e-12) - 1e-12))
        return hits / m.shape[0] if exact else (hits + 1) / (m.shape[0] + 1)

    return p_at


def _invert_ci(p_at, est: float, scale: float, alpha: float) -> tuple[float, float]:
    """The shifts d with p(d) >= alpha. Moving d away from the estimate, each split can
    only stop exceeding the observed statistic (its slope in d is at most 1 in size,
    the observed one's is 1), so p falls monotonically on each side and bisection finds
    each bound. NaN when even a far shift is not rejected (too few batches)."""
    bounds = []
    for direction in (-1.0, 1.0):
        step = max(scale, 1e-9)
        far = est + direction * step
        for _ in range(60):
            if p_at(far) < alpha:
                break
            step *= 2.0
            far = est + direction * step
        else:
            bounds.append(float("nan"))
            continue
        inside, outside = est, far
        for _ in range(100):
            mid = 0.5 * (inside + outside)
            if p_at(mid) >= alpha:
                inside = mid
            else:
                outside = mid
            if abs(outside - inside) <= 1e-10 * max(1.0, abs(scale)):
                break
        bounds.append(float(inside))
    return bounds[0], bounds[1]


def controlled_change_test(exposed: pd.DataFrame, control: pd.DataFrame, *,
                           value: str = "change", batch: str = "batch_folder",
                           n_perm: int = 20000, n_boot: int = 10000,
                           seed: int = 0, ci: str = "bootstrap",
                           alpha: float = 0.05) -> dict:
    """Difference in differences: (exposed flies' mean change) - (control flies').

    One row per fly, ``value`` = that fly's post - pre change on the endpoint
    (e.g. responded post - responded pre, in {-1, 0, 1}). Each group's change is
    pooled over its flies (sum / flies).

    * ``p_batch`` (the headline): whole batches swapped between the two groups,
      the same pooled statistic recomputed. Exact over every split when there
      are at most :data:`MAX_EXACT_SPLITS`, Monte Carlo ((hits+1)/(n+1)) above.
    * ``p_fly``: flies swapped one by one (ignores that flies share a batch).
    * ``ci_lo``/``ci_hi`` (``1 - alpha``):
      - ``ci="bootstrap"``: percentile bootstrap resampling batches within each
        group. With ~15 batches a side it runs narrow, so it can exclude 0 while
        p_batch >= 0.05.
      - ``ci="permutation"``: the shifts d not rejected when d is subtracted from
        every exposed fly and the same batch permutation is run (test inversion).
        Uses the same splits as p_batch, so the interval excludes 0 exactly when
        p_batch < alpha.
    """
    if ci not in ("bootstrap", "permutation"):
        raise ValueError(f"ci must be 'bootstrap' or 'permutation': {ci!r}")
    def by_batch(df: pd.DataFrame) -> pd.DataFrame:
        d = df[[value, batch]].copy()
        d[value] = pd.to_numeric(d[value], errors="coerce")
        d = d[np.isfinite(d[value].to_numpy(dtype=float))]
        return d.groupby(batch, sort=True)[value].agg(["sum", "count"])

    xb, cb = by_batch(exposed), by_batch(control)
    out = {
        "n_flies_exposed": int(xb["count"].sum()), "n_flies_control": int(cb["count"].sum()),
        "n_batches_exposed": int(len(xb)), "n_batches_control": int(len(cb)),
    }
    if xb.empty or cb.empty:
        return {**out, "exposed_change": float("nan"), "control_change": float("nan"),
                "estimate": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p_batch": float("nan"), "p_fly": float("nan"), "perm_exact": False,
                "ci_method": ci}
    s = np.concatenate([xb["sum"].to_numpy(float), cb["sum"].to_numpy(float)])
    c = np.concatenate([xb["count"].to_numpy(float), cb["count"].to_numpy(float)])
    kx, k = len(xb), len(s)
    ex_change = float(s[:kx].sum() / c[:kx].sum())
    ct_change = float(s[kx:].sum() / c[kx:].sum())
    obs = ex_change - ct_change

    rng = np.random.default_rng(seed)
    if ci == "permutation":
        masks, exact = _split_masks(k, kx, n_perm, rng)
        p_at = _shifted_batch_p(s, c, kx, masks, exact)
        vx = pd.to_numeric(exposed[value], errors="coerce").to_numpy(dtype=float)
        vc = pd.to_numeric(control[value], errors="coerce").to_numpy(dtype=float)
        both = np.concatenate([vx[np.isfinite(vx)], vc[np.isfinite(vc)]])
        scale = float(np.std(both)) if both.size > 1 else 1.0
        ci_lo, ci_hi = _invert_ci(p_at, obs, scale if scale > 0 else 1.0, alpha)
        return {**out, "exposed_change": ex_change, "control_change": ct_change,
                "estimate": float(obs), "ci_lo": ci_lo, "ci_hi": ci_hi,
                "p_batch": float(p_at(0.0)),
                "p_fly": float(fly_permutation_p(vx, vc, n_perm=n_perm, seed=seed)),
                "perm_exact": bool(exact), "ci_method": "permutation"}
    n_splits = math.comb(k, kx)
    exact = n_splits <= MAX_EXACT_SPLITS
    if exact:
        hits = 0
        for pick in itertools.combinations(range(k), kx):
            mask = np.zeros(k, dtype=bool)
            mask[list(pick)] = True
            stat = s[mask].sum() / c[mask].sum() - s[~mask].sum() / c[~mask].sum()
            hits += abs(stat) >= abs(obs) - 1e-12
        p_batch = hits / n_splits
    else:
        hits = 0
        idx = np.arange(k)
        for _ in range(int(n_perm)):
            rng.shuffle(idx)
            x, y = idx[:kx], idx[kx:]
            stat = s[x].sum() / c[x].sum() - s[y].sum() / c[y].sum()
            hits += abs(stat) >= abs(obs) - 1e-12
        p_batch = (hits + 1) / (int(n_perm) + 1)

    vx = pd.to_numeric(exposed[value], errors="coerce").to_numpy(dtype=float)
    vc = pd.to_numeric(control[value], errors="coerce").to_numpy(dtype=float)
    p_fly = fly_permutation_p(vx, vc, n_perm=n_perm, seed=seed)

    bx = rng.integers(0, kx, size=(int(n_boot), kx))
    bc = rng.integers(0, k - kx, size=(int(n_boot), k - kx)) + kx
    boot = (s[bx].sum(1) / c[bx].sum(1)) - (s[bc].sum(1) / c[bc].sum(1))
    ci_lo, ci_hi = (float(q) for q in np.percentile(boot, [2.5, 97.5]))
    return {**out, "exposed_change": ex_change, "control_change": ct_change,
            "estimate": float(obs), "ci_lo": ci_lo, "ci_hi": ci_hi,
            "p_batch": float(p_batch), "p_fly": float(p_fly), "perm_exact": bool(exact),
            "ci_method": "bootstrap"}


# ── multiplicity ─────────────────────────────────────────────────────────────
def holm_adjust(pvals: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values; NaN/None stay NaN and are not counted."""
    p = np.asarray([np.nan if v is None else v for v in pvals], dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    m = finite.size
    running = 0.0
    for rank, i in enumerate(finite[np.argsort(p[finite], kind="stable")]):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def bh_adjust(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg step-up adjusted p-values (q); NaN/None stay NaN."""
    p = np.asarray([np.nan if v is None else v for v in pvals], dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    m = finite.size
    if m == 0:
        return out
    order = finite[np.argsort(p[finite], kind="stable")]
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, m / (rank + 1) * p[i])
        out[i] = min(1.0, running)
    return out


def stars(p: float | None) -> str:
    """'***' < 0.001, '**' < 0.01, '*' < 0.05, else 'n.s.' ('' for no test)."""
    if p is None or not np.isfinite(float(p)):
        return ""
    p = float(p)
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


__all__ = (
    "WILCOXON_EXACT_MAX_N", "Z95", "bh_adjust", "controlled_change_test",
    "fly_cluster_mean_ci", "fly_cluster_rate_ci", "fly_permutation_p", "holm_adjust",
    "mcnemar_exact_p", "stars", "wilcoxon_signed_rank_p",
)
