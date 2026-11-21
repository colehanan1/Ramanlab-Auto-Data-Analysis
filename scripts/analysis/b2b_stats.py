"""Batch-aware tests for the b2b IAA odor-pair figures (user 2026-11-18, stats audit).

A batch folder (``november_13_batch_1_rig_3``) is one rig's run of one session; the
three rigs of a session (same folder name without ``_rig_N``) start together. Start
time, rig, starvation and the dataset a folder ran are all batch properties, and flies
of one batch share a day, a rig and a vial, so a test that treats flies as independent
overstates the evidence (audit 2026-11-17: Hexanol by rig p 6e-5 at fly level, 0.018 at
batch level). Everything here takes one value per fly and keeps a batch's flies together.

* :func:`sign_flip_p` -- exact paired sign-flip across flies (one value per fly).
* :func:`batch_sign_flip` -- the same with every fly of a batch flipped together; its
  95 % CI inverts that test (the shifts it does not reject), so the CI clears 0 exactly
  when p < 0.05.
* :func:`batch_label_perm` -- do groups of batches differ (group, rig, start-time bin)?
  Batch labels permuted, within strata when given: PairFirst vs PairSecond keeps each
  session's number of PairSecond folders; rig labels move only within a session.
* :func:`batch_trend` -- Spearman rho of batch value vs batch start, batches permuted.

Exact over every pattern up to :data:`MAX_EXACT`, else Monte Carlo with a fixed seed
((hits + 1) / (n + 1)), so a rebuild of the same flies gives the same p.
"""
from __future__ import annotations

import itertools
import math
import re
from collections import Counter
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

MAX_EXACT = 1 << 18
N_MC = 100_000
SEED = 0
_RIG = re.compile(r"_rig_\d+$")


def session_of(folder: str) -> str:
    """The session a batch folder belongs to: its name without ``_rig_N``."""
    return _RIG.sub("", str(folder))


# ── one value per fly ────────────────────────────────────────────────────────
def sign_flip_p(d) -> float:
    """Exact two-sided paired sign-flip p for mean(d) = 0, flies as the unit: the share
    of the 2^n sign patterns whose |sum| is at least the observed one. A zero carries no
    sign. Values on a half-step grid (scores, 0/1, means of 2) are counted exactly;
    others by a seeded Monte Carlo."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    h = np.abs(d) * 2
    a = np.rint(h).astype(int)
    if not np.allclose(a, h):
        signs = np.random.default_rng(SEED).choice([-1.0, 1.0], size=(200_000, d.size))
        null = np.abs(signs @ np.abs(d))
        return float((np.sum(null >= abs(d.sum()) - 1e-12) + 1) / (len(null) + 1))
    a = a[a > 0]
    if not a.size:
        return 1.0
    total = int(a.sum())
    dist = np.zeros(2 * total + 1)
    dist[total] = 1.0
    for ai in a:
        new = np.zeros_like(dist)
        new[ai:] += dist[:-ai] / 2
        new[:-ai] += dist[ai:] / 2
        dist = new
    obs = abs(int(np.rint(np.sum(np.sign(d) * h))))
    return float(min(1.0, dist[np.abs(np.arange(-total, total + 1)) >= obs].sum()))


# ── whole batches flipped together ───────────────────────────────────────────
def _by_batch(values, batches) -> tuple[np.ndarray, np.ndarray, list]:
    v = np.asarray(values, dtype=float).ravel()
    b = np.asarray(batches, dtype=object).ravel()
    if v.size != b.size:
        raise ValueError(f"{v.size} values but {b.size} batch ids")
    keep = np.isfinite(v)
    g = (pd.DataFrame({"v": v[keep], "b": b[keep].astype(str)})
         .groupby("b", sort=True)["v"].agg(["sum", "count"]))
    return g["sum"].to_numpy(float), g["count"].to_numpy(float), list(g.index)


def _signs(k: int) -> tuple[np.ndarray, bool]:
    if 2 ** k <= MAX_EXACT:
        return (((np.arange(2 ** k)[:, None] >> np.arange(k)) & 1) * 2 - 1).astype(float), True
    return np.random.default_rng(SEED).choice([-1.0, 1.0], size=(N_MC, k)), False


def batch_sign_flip(values, batches, *, ci: bool = False, alpha: float = 0.05) -> dict:
    """Paired sign-flip with whole batches as the unit.

    ``values``: one per fly (e.g. a fly's change minus its click's change); ``batches``:
    each fly's batch folder. Statistic: the sum over flies; every fly of a batch takes
    the same sign. ``estimate`` = the mean over flies. ``ci``: the 1 - alpha interval
    of shifts d for which subtracting d from every fly is not rejected; NaN ends when
    even the most extreme pattern cannot reach alpha (too few batches)."""
    s, n, _ = _by_batch(values, batches)
    k = len(s)
    out = {"n_flies": int(n.sum()), "n_batches": k, "estimate": float("nan"),
           "p": float("nan"), "exact": False, "ci_lo": float("nan"), "ci_hi": float("nan")}
    if k == 0:
        return out
    signs, exact = _signs(k)
    a, bn = signs @ s, signs @ n
    tot_s, tot_n = float(s.sum()), float(n.sum())

    def p_at(d: float) -> float:
        obs = abs(tot_s - d * tot_n)
        hits = int(np.sum(np.abs(a - d * bn) >= obs - 1e-9))
        return hits / len(a) if exact else (hits + 1) / (len(a) + 1)

    est = tot_s / tot_n
    out.update(estimate=est, p=float(min(1.0, p_at(0.0))), exact=exact)
    if ci:
        lo_v, hi_v = float(np.min(s / n)), float(np.max(s / n))
        span = max(hi_v - lo_v, abs(est), 1e-6)

        def bound(direction: int) -> float:
            near, far = est, est + direction * span
            for _ in range(60):                       # walk out until rejected
                if p_at(far) < alpha:
                    break
                near, far = far, est + 2 * (far - est)
            else:
                return float("nan")
            for _ in range(60):                       # then bisect the crossing
                mid = (near + far) / 2
                near, far = (mid, far) if p_at(mid) >= alpha else (near, mid)
            return near
        out.update(ci_lo=bound(-1), ci_hi=bound(+1))
    return out


# ── groups of batches ────────────────────────────────────────────────────────
def _between_ss(sums: np.ndarray, counts: np.ndarray, lab: np.ndarray, levels) -> np.ndarray:
    """Between-group sum of squares of fly-pooled means, one per label row."""
    m = sums.sum() / counts.sum()
    out = np.zeros(lab.shape[0])
    for level in levels:
        mask = (lab == level).astype(float)
        nr, sr = mask @ counts, mask @ sums
        with np.errstate(invalid="ignore", divide="ignore"):
            out += np.where(nr > 0, nr * (sr / np.where(nr > 0, nr, 1) - m) ** 2, 0.0)
    return out


def _multinomial(codes) -> int:
    """How many distinct orderings a multiset of labels has."""
    out = math.factorial(len(codes))
    for c in Counter(codes.tolist() if hasattr(codes, "tolist") else codes).values():
        out //= math.factorial(c)
    return out


def distinct_arrangements(codes) -> np.ndarray:
    """Every distinct ordering of a multiset of label codes, one row each (multinomial
    many), built level by level from the positions each level can take -- never the n!
    orderings."""
    counts = sorted(Counter(list(codes)).items())
    rows: list[np.ndarray] = []
    row = np.empty(len(codes), dtype=int)

    def place(free: tuple, k: int) -> None:
        if k == len(counts):
            rows.append(row.copy())
            return
        level, c = counts[k]
        for pick in itertools.combinations(free, c):
            row[list(pick)] = level
            taken = set(pick)
            place(tuple(i for i in free if i not in taken), k + 1)

    place(tuple(range(len(codes))), 0)
    return np.array(rows, dtype=int).reshape(len(rows), len(codes))


def batch_label_perm(values, batches, labels: Mapping, strata: Mapping | None = None) -> dict:
    """Do the batches' groups differ? ``labels``: batch -> group (dataset, rig, start
    bin); ``strata``: batch -> stratum (session) or None for one stratum. Statistic: the
    between-group sum of squares of the fly-pooled group means (sum over flies / flies,
    so two groups give a two-sided test of their difference). Labels are permuted among
    the batches of each stratum, keeping its label counts. ``means``: each group's
    fly-pooled mean; ``n_perm``: arrangements counted."""
    s, n, ids = _by_batch(values, batches)
    lab = np.array([labels[b] for b in ids], dtype=object)
    levels = sorted(set(lab), key=str)
    means = {lv: float(s[lab == lv].sum() / n[lab == lv].sum()) for lv in levels}
    out = {"n_flies": int(n.sum()), "n_batches": len(ids), "means": means,
           "p": float("nan"), "exact": False, "n_perm": 0}
    if len(levels) < 2:
        return out
    strat = np.array([(strata or {}).get(b, "_all") if strata else "_all" for b in ids],
                     dtype=object)
    groups = [np.flatnonzero(strat == st) for st in sorted(set(strat), key=str)]
    code = {lv: i for i, lv in enumerate(levels)}
    base = np.array([code[x] for x in lab], dtype=int)
    total = math.prod(_multinomial(base[idx]) for idx in groups)
    if total <= MAX_EXACT:                 # every distinct arrangement, never the n! orders
        per = [(idx, distinct_arrangements(base[idx])) for idx in groups]
        grid = np.indices([len(arr) for _, arr in per]).reshape(len(per), -1).T
        rows = np.empty((len(grid), len(ids)), dtype=int)
        for j, (idx, arr) in enumerate(per):
            rows[:, idx] = arr[grid[:, j]]
        exact = True
    else:                                  # seeded draws, labels shuffled within each stratum
        rng = np.random.default_rng(SEED)
        rows = np.tile(base, (N_MC, 1))
        for idx in groups:
            rows[:, idx] = rng.permuted(rows[:, idx], axis=1)
        exact = False
    obs = _between_ss(s, n, np.array([[code[x] for x in lab]]), range(len(levels)))[0]
    null = _between_ss(s, n, rows, range(len(levels)))
    hits = int(np.sum(null >= obs - 1e-9))
    p = hits / len(null) if exact else (hits + 1) / (len(null) + 1)
    out.update(p=float(min(1.0, p)), exact=exact, n_perm=int(len(null)))
    return out


def batch_trend(values, batches, start_of: Mapping) -> dict:
    """Spearman rho between each batch's start (``start_of``: batch -> clock hours) and
    its fly-pooled value, p from permuting the values across batches (two-sided, exact
    up to 8 batches)."""
    s, n, ids = _by_batch(values, batches)
    x = np.array([start_of.get(b, np.nan) for b in ids], dtype=float)
    keep = np.isfinite(x)
    y, x = (s / n)[keep], x[keep]
    k = len(y)
    out = {"n_batches": int(k), "n_flies": int(n[keep].sum()), "rho": float("nan"),
           "p": float("nan"), "exact": False}
    if k < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return out

    def ranks(v):
        return pd.Series(v).rank().to_numpy(float)

    rx, ry = ranks(x), ranks(y)
    rx_c = rx - rx.mean()

    def rho_of(r):
        r = r - r.mean(axis=-1, keepdims=True)
        return (r @ rx_c) / np.sqrt((r ** 2).sum(axis=-1) * (rx_c ** 2).sum())

    obs = float(rho_of(ry[None, :])[0])
    if math.factorial(k) <= 40320:
        perms = np.array(list(itertools.permutations(ry)))
        null = rho_of(perms)
        p = float(np.mean(np.abs(null) >= abs(obs) - 1e-12))
        exact = True
    else:
        rng = np.random.default_rng(SEED)
        perms = np.array([rng.permutation(ry) for _ in range(N_MC // 5)])
        null = rho_of(perms)
        p = float((np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (len(null) + 1))
        exact = False
    out.update(rho=obs, p=min(1.0, p), exact=exact)
    return out
