"""Statistics for the PER analysis package, with the FLY as the unit.

Every function here takes one value per fly (or a tidy trial table it reduces to one
value per fly first). Trials of one fly are never treated as independent: a paired test
averages a fly's repeated trials of a condition first, intervals and tests count flies,
and the bar heights (:func:`pooled_rate_ci`, :func:`pooled_mean_sem`) pool trials
(sum / trials), never a mean of per-fly means.

Building blocks are reused, not copied:

* ``fbpipe.analysis.fly_stats`` -- Holm/BH, fly-cluster Wilson and mean intervals,
  the fly permutation p, and the whole-batch difference-in-differences test.
* ``scripts.analysis.b2b_stats`` -- the exact half-step-grid sign-flip DP and the
  whole-batch sign-flip with its inverted CI.

Public API
----------
Paired tests      :func:`sign_flip_test`, :func:`batch_sign_flip_test`,
                  :func:`paired_effect`, :func:`paired_comparisons`
Bootstrap         :func:`bootstrap_ci` (BCa, jackknife acceleration)
Per-fly / pooled  :func:`per_fly`, :func:`pooled_rate_ci`, :func:`pooled_mean_sem`,
                  :func:`group_summary`
Multiplicity      :func:`holm`, :func:`bh`, :func:`stars`, :func:`add_corrections`
Two groups        :func:`did_bootstrap`
Models            :func:`fit_gee_logistic`, :func:`fit_mixed_logistic`,
                  :func:`fit_mixedlm`, :func:`fit_ordinal`, :func:`save_model`
Power             :func:`power_curve`, :func:`min_n_for_power`
Shared baseline   :func:`direction_count_test`

Batch-level robustness. Flies of one batch folder share a day, a rig and a vial, and the
2026-11-17 stats audit found fly-level p overstating the evidence (Hexanol by rig: p 6e-5
at fly level, 0.018 at batch level). Every comparison table therefore carries ``p_batch``
next to ``p``; stars come from the fly-level ``p_holm`` only, and a result whose
``p_batch`` does not agree should be reported as fragile.

Defaults (``SEED``, ``N_BOOT``, ``N_PERM``) are read lazily from
``per_analysis.config`` when it imports, else the fallbacks below; an explicit argument
always wins.
"""
from __future__ import annotations

import contextlib
import logging
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from scipy.stats import t as student_t

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fbpipe.analysis import fly_stats as _fs  # noqa: E402
from scripts.analysis import b2b_stats as _b2b  # noqa: E402

log = logging.getLogger(__name__)

#: Used when ``per_analysis.config`` is absent or lacks the name.
_FALLBACK: dict[str, Any] = {"SEED": 0, "N_BOOT": 10_000, "N_PERM": 100_000,
                             "RESPONSE_THRESHOLD": 2}

#: Enumerate every sign pattern exactly up to this many non-zero, off-grid flies
#: (2**20 = 1,048,576 patterns, an 8 MB array); seeded Monte Carlo above it.
EXACT_ENUM_MAX = 20

#: Rational-grid exactness: denominators tried (60 = lcm of 1..6 trials per fly) and
#: the cap on the integer DP size (sum of |d| * q).
GRID_MAX_DENOM = 60
GRID_MAX_TOTAL = 1_000_000

#: The direction-count null is enumerated over every joint fly flip up to this many flies.
DIRECTION_EXACT_MAX = 16

_MC_CHUNK = 10_000


def config_value(name: str, default: Any = None) -> Any:
    """``per_analysis.config.<name>``, imported lazily so this module (and its tests)
    work before or without config.py. Falls back to :data:`_FALLBACK`, then ``default``."""
    fallback = _FALLBACK.get(name, default)
    try:
        from per_analysis import config as _cfg  # local: config may not exist yet
    except Exception as exc:  # noqa: BLE001 -- missing file or a config that cannot load
        log.debug("per_analysis.config unavailable (%s); using fallback for %s", exc, name)
        return fallback
    return getattr(_cfg, name, fallback)


def _resolve(value: Any, name: str) -> Any:
    return config_value(name) if value is None else value


def _finite(values: Any) -> np.ndarray:
    v = np.asarray(values, dtype=float).ravel()
    return v[np.isfinite(v)]


# ═════════════════════════════════════════════════════════════════════════════
# 1. paired sign-flip
# ═════════════════════════════════════════════════════════════════════════════
def _grid_denominator(d: np.ndarray) -> int | None:
    """Smallest q <= :data:`GRID_MAX_DENOM` with every |d| * q an integer, or None.

    Per-fly means of k trials of integer scores (or 0/1) are multiples of 1/k, and a
    difference of two such means is a multiple of 1/lcm(k1, k2); q = 60 covers any mix of
    1-6 trials. None also when the integer DP would exceed :data:`GRID_MAX_TOTAL`."""
    ad = np.abs(d)
    for q in range(1, GRID_MAX_DENOM + 1):
        h = ad * q
        if np.allclose(np.rint(h), h, rtol=0.0, atol=1e-7):
            return q if float(np.rint(h).sum()) <= GRID_MAX_TOTAL else None
    return None


def _enumerated_p(a: np.ndarray, obs: float) -> float:
    """Exact sign-flip p by listing all 2^n signed sums of the magnitudes ``a``."""
    sums = np.zeros(1)
    for ai in a:
        sums = np.concatenate([sums + ai, sums - ai])
    tol = 1e-9 * max(1.0, float(a.sum()))
    return float(np.mean(np.abs(sums) >= obs - tol))


def _mc_p(a: np.ndarray, obs: float, n_perm: int, seed: int) -> float:
    """Monte Carlo sign-flip p = (hits + 1) / (n_perm + 1), seeded, in chunks."""
    rng = np.random.default_rng(seed)
    tol = 1e-9 * max(1.0, float(a.sum()))
    hits, done = 0, 0
    while done < n_perm:
        m = min(_MC_CHUNK, n_perm - done)
        signs = rng.choice((-1.0, 1.0), size=(m, a.size))
        hits += int(np.sum(np.abs(signs @ a) >= obs - tol))
        done += m
    return (hits + 1) / (n_perm + 1)


def sign_flip_test(diffs: Sequence[float], *, n_perm: int | None = None,
                   seed: int | None = None) -> dict:
    """Two-sided paired sign-flip (randomization) test of mean(diffs) = 0, flies as unit.

    ``diffs``: one within-fly difference per fly (average a fly's repeated trials of each
    condition first). Statistic |sum d|; under H0 each fly's difference is symmetric about
    0, so every sign pattern is equally likely. Zeros carry no sign (they add 0 to every
    pattern); non-finite values are dropped.

    Exactness, in order:

    1. on a rational grid -- every |d| a multiple of 1/q, q <= :data:`GRID_MAX_DENOM`
       (scores, 0/1, and differences of per-fly means of 1-6 trials): exact DP via
       ``b2b_stats.sign_flip_p``. The p is scale-invariant, so d is rescaled onto that
       function's half-step grid (d * q / 2, snapped); feasible for any n;
    2. off the grid with at most :data:`EXACT_ENUM_MAX` non-zero flies (2^n <= ~1e6):
       exact enumeration of every pattern;
    3. otherwise ``n_perm`` seeded Monte Carlo patterns, p = (hits + 1) / (n_perm + 1).

    Returns ``{"p", "exact", "n", "n_nonzero", "mean", "method"}``; ``n`` counts finite
    flies (zeros included), ``p`` is NaN for no flies and 1.0 when every fly is 0.
    """
    n_perm = int(_resolve(n_perm, "N_PERM"))
    seed = int(_resolve(seed, "SEED"))
    d = _finite(diffs)
    out = {"p": float("nan"), "exact": False, "n": int(d.size),
           "n_nonzero": int(np.sum(d != 0)), "mean": float(d.mean()) if d.size else float("nan"),
           "method": "none"}
    if d.size == 0:
        return out
    a = np.abs(d[d != 0])
    if a.size == 0:
        out.update(p=1.0, exact=True, method="exact-grid")
        return out
    q = _grid_denominator(d)
    if q is not None:
        on_grid = np.sign(d) * np.rint(np.abs(d) * q) / 2.0      # half-step grid, exact
        out.update(p=float(_b2b.sign_flip_p(on_grid)), exact=True, method="exact-grid")
    elif a.size <= EXACT_ENUM_MAX:
        out.update(p=_enumerated_p(a, abs(float(d.sum()))), exact=True,
                   method="exact-enumeration")
    else:
        out.update(p=float(_mc_p(a, abs(float(d.sum())), n_perm, seed)), exact=False,
                   method="monte-carlo")
    out["p"] = float(min(1.0, out["p"]))
    return out


def batch_sign_flip_test(diffs: Sequence[float], batches: Sequence, *, ci: bool = True,
                         alpha: float = 0.05) -> dict:
    """Paired sign-flip with whole batches as the unit (``b2b_stats.batch_sign_flip``).

    Every fly of a batch takes the same sign, so a batch whose flies all moved together
    counts once. ``estimate`` is the mean over flies; with ``ci`` the 1 - alpha interval
    inverts this same test (it clears 0 exactly when p < alpha; NaN ends when there are
    too few batches to reject anything). Exact up to 18 batches, else 100k seeded draws.
    Returns ``{"p", "exact", "n_flies", "n_batches", "estimate", "ci_lo", "ci_hi"}``.
    """
    v = np.asarray(diffs, dtype=float).ravel()
    b = np.asarray(batches, dtype=object).ravel()
    if v.size != b.size:
        raise ValueError(f"{v.size} differences but {b.size} batch labels")
    keep = np.isfinite(v) & pd.notna(b)
    return dict(_b2b.batch_sign_flip(v[keep], b[keep], ci=ci, alpha=alpha))


# ═════════════════════════════════════════════════════════════════════════════
# 2. bootstrap
# ═════════════════════════════════════════════════════════════════════════════
def _apply_stat(stat: Callable, mat: np.ndarray) -> np.ndarray:
    """``stat`` of every row; uses ``axis=1`` when the callable takes it."""
    try:
        out = np.asarray(stat(mat, axis=1), dtype=float)
        if out.shape == (mat.shape[0],):
            return out
    except TypeError:
        pass
    return np.array([float(stat(row)) for row in mat])


def _bca_bounds(theta: float, boot: np.ndarray, accel: float, level: float) -> tuple[float, float]:
    """BCa percentiles: bias correction z0 from the bootstrap share below theta (ties
    count half), acceleration ``accel`` from the jackknife / influence values."""
    b = boot[np.isfinite(boot)]
    nb = b.size
    prop = (np.sum(b < theta) + 0.5 * np.sum(b == theta)) / nb
    prop = float(np.clip(prop, 1.0 / (nb + 1), nb / (nb + 1)))
    z0 = norm.ppf(prop)
    zq = norm.ppf([(1 - level) / 2, 1 - (1 - level) / 2])
    denom = 1 - accel * (z0 + zq)
    if np.any(denom <= 0) or not np.isfinite(accel):
        q = np.array([(1 - level) / 2, 1 - (1 - level) / 2])
    else:
        q = norm.cdf(z0 + (z0 + zq) / denom)
    lo, hi = np.quantile(b, q)
    return float(lo), float(hi)


def _jackknife_accel(jack: np.ndarray) -> float:
    jm = jack.mean()
    num = float(np.sum((jm - jack) ** 3))
    den = 6.0 * float(np.sum((jm - jack) ** 2)) ** 1.5
    return num / den if den > 0 else 0.0


def bootstrap_ci(values: Sequence[float], *, stat: Callable = np.mean,
                 n_boot: int | None = None, seed: int | None = None, method: str = "BCa",
                 level: float = 0.95) -> tuple[float, float]:
    """``level`` bootstrap interval of ``stat`` over flies (one value per fly).

    ``method="BCa"`` (default): bias-corrected and accelerated, acceleration from the
    leave-one-fly-out jackknife; ``"percentile"``: plain percentiles. A constant sample
    returns ``(c, c)``; fewer than two finite flies return ``(nan, nan)`` -- one fly has
    no sampling distribution to speak of.
    """
    if method not in ("BCa", "percentile"):
        raise ValueError(f"method must be 'BCa' or 'percentile': {method!r}")
    n_boot = int(_resolve(n_boot, "N_BOOT"))
    seed = int(_resolve(seed, "SEED"))
    x = _finite(values)
    n = x.size
    if n < 2:
        return float("nan"), float("nan")
    theta = float(stat(x))
    if np.ptp(x) == 0:
        return theta, theta
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for start in range(0, n_boot, _MC_CHUNK):
        m = min(_MC_CHUNK, n_boot - start)
        boot[start:start + m] = _apply_stat(stat, x[rng.integers(0, n, size=(m, n))])
    if np.ptp(boot[np.isfinite(boot)]) == 0:
        return theta, theta
    if method == "percentile":
        lo, hi = np.quantile(boot[np.isfinite(boot)], [(1 - level) / 2, 1 - (1 - level) / 2])
        return float(lo), float(hi)
    loo = np.broadcast_to(x, (n, n))[~np.eye(n, dtype=bool)].reshape(n, n - 1)
    return _bca_bounds(theta, boot, _jackknife_accel(_apply_stat(stat, loo)), level)


def _as_fly_series(s: Any, name: str) -> pd.Series:
    s = pd.Series(s) if not isinstance(s, pd.Series) else s
    if s.index.has_duplicates:
        raise ValueError(f"{name}: one value per fly expected, duplicate fly ids found")
    return pd.to_numeric(s, errors="coerce").astype(float)


def _lookup_batches(flies: Sequence, batch_of: Mapping | pd.Series | None) -> np.ndarray:
    if batch_of is None:
        return np.full(len(flies), None, dtype=object)
    get = batch_of.get if hasattr(batch_of, "get") else dict(batch_of).get
    return np.array([get(f) for f in flies], dtype=object)


def paired_effect(per_fly_a: pd.Series, per_fly_b: pd.Series, *,
                  batch_of: Mapping | pd.Series | None = None, n_boot: int | None = None,
                  n_perm: int | None = None, seed: int | None = None,
                  level: float = 0.95) -> dict:
    """Within-fly change b - a: fly-level estimate, BCa CI, sign-flip p and batch p.

    ``per_fly_a`` / ``per_fly_b``: Series indexed by fly_id, one within-fly mean each
    (from :func:`per_fly`); flies present in both are kept (inner join). ``batch_of``:
    fly_id -> batch folder, for ``p_batch`` (:func:`batch_sign_flip_test`); NaN without it.

    ``mean_a`` / ``mean_b`` / ``mean_diff`` are means over flies -- the test's estimand.
    They are NOT the bar heights: bars pool trials (:func:`group_summary`).

    Returns ``{n_flies, n_batches, mean_a, mean_b, mean_diff, ci_lo, ci_hi, p, p_exact,
    p_method, p_batch, p_batch_exact}``.
    """
    a = _as_fly_series(per_fly_a, "per_fly_a")
    b = _as_fly_series(per_fly_b, "per_fly_b")
    j = pd.concat({"a": a, "b": b}, axis=1, join="inner").dropna()
    d = (j["b"] - j["a"]).to_numpy(float)
    sf = sign_flip_test(d, n_perm=n_perm, seed=seed)
    lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed, level=level)
    out = {"n_flies": int(len(j)), "n_batches": 0,
           "mean_a": float(j["a"].mean()) if len(j) else float("nan"),
           "mean_b": float(j["b"].mean()) if len(j) else float("nan"),
           "mean_diff": float(d.mean()) if d.size else float("nan"),
           "ci_lo": lo, "ci_hi": hi, "p": sf["p"], "p_exact": sf["exact"],
           "p_method": sf["method"], "p_batch": float("nan"), "p_batch_exact": False}
    batches = _lookup_batches(list(j.index), batch_of)
    has = pd.notna(batches)
    if has.any():
        bt = batch_sign_flip_test(d[has], batches[has], ci=False)
        out.update(n_batches=int(bt["n_batches"]), p_batch=float(bt["p"]),
                   p_batch_exact=bool(bt["exact"]))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 3. per-fly aggregation and pooled bar values
# ═════════════════════════════════════════════════════════════════════════════
def _numeric(col: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(col):
        return col.astype("float64")
    return pd.to_numeric(col, errors="coerce").astype("float64")


def per_fly(trials: pd.DataFrame, value: str, by: Sequence[str] | None = None, *,
            fly_col: str = "fly_id", carry: Sequence[str] = ("batch",),
            out_col: str | None = None) -> pd.DataFrame:
    """Within-fly mean of ``value`` (score, or responded as 0/1) per ``by`` group.

    One row per (``by``..., fly): the mean over that fly's trials (NaN trials skipped),
    ``n_trials``, and the ``carry`` columns (fly-level constants such as batch; first
    value) that exist in ``trials``. NaN group keys are kept, not dropped.
    """
    by = list(by or [])
    keys = by + [fly_col]
    out_col = out_col or value
    carry = [c for c in carry if c in trials.columns and c not in keys]
    d = trials[keys + carry].copy()
    d["_v"] = _numeric(trials[value])
    g = d.groupby(keys, sort=True, dropna=False, observed=True)
    out = g["_v"].agg(["mean", "count"]).rename(columns={"mean": out_col, "count": "n_trials"})
    if carry:
        out = out.join(g[carry].first())
    return out.reset_index()


def pooled_rate_ci(trials: pd.DataFrame, *, value: str = "responded",
                   fly_col: str = "fly_id") -> dict:
    """Response rate pooled over trials (responses / trials, in %), 95 % Wilson CI with
    flies as the unit (``fly_stats.fly_cluster_rate_ci``: Wilson at the cluster-robust
    effective n). Returns ``{rate, ci_lo, ci_hi, n_flies, n_trials}``."""
    v = _numeric(trials[value])
    keep = v.notna().to_numpy()
    ids = trials[fly_col].to_numpy(dtype=object)[keep]
    rate, lo, hi = _fs.fly_cluster_rate_ci(v.to_numpy()[keep], ids)
    return {"rate": float(rate), "ci_lo": float(lo), "ci_hi": float(hi),
            "n_flies": int(pd.unique(ids).size), "n_trials": int(keep.sum())}


def pooled_mean_sem(trials: pd.DataFrame, *, value: str = "score",
                    fly_col: str = "fly_id") -> dict:
    """Mean pooled over trials (sum / trials) with its SEM over flies: the cluster-robust
    (ratio-estimator) standard error of ``fly_stats.fly_cluster_mean_ci``; with one trial
    per fly it is the ordinary SD / sqrt(n). Returns ``{mean, sem, ci_lo, ci_hi, n_flies,
    n_trials}`` (``ci`` = the 95 % t interval on flies - 1 df)."""
    v = _numeric(trials[value])
    keep = v.notna().to_numpy()
    ids = trials[fly_col].to_numpy(dtype=object)[keep]
    g = int(pd.unique(ids).size)
    m, lo, hi = _fs.fly_cluster_mean_ci(v.to_numpy()[keep], ids, level=0.95)
    sem = (hi - m) / float(student_t.ppf(0.975, g - 1)) if g >= 2 else float("nan")
    return {"mean": float(m), "sem": float(sem), "ci_lo": float(lo), "ci_hi": float(hi),
            "n_flies": g, "n_trials": int(keep.sum())}


def group_summary(trials: pd.DataFrame, by: Sequence[str], metric: str, *,
                  value: str | None = None, fly_col: str = "fly_id") -> pd.DataFrame:
    """The bar table: one row per ``by`` group with ``value``, ``err_lo``, ``err_hi``.

    ``metric="rate"``: ``value`` = % of trials responding (pooled), error = 95 % Wilson
    CI with flies as the unit. ``metric="score"``: ``value`` = pooled mean score, error
    = +/- SEM over flies. This is the contract ``plots.paired_bars`` consumes.
    """
    if metric not in ("rate", "score"):
        raise ValueError(f"metric must be 'rate' or 'score': {metric!r}")
    col = value or ("responded" if metric == "rate" else "score")
    rows = []
    for key, sub in trials.groupby(list(by), sort=True, dropna=False, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        if metric == "rate":
            r = pooled_rate_ci(sub, value=col, fly_col=fly_col)
            vals = (r["rate"], r["ci_lo"], r["ci_hi"], "95% Wilson CI (flies as unit)")
        else:
            r = pooled_mean_sem(sub, value=col, fly_col=fly_col)
            vals = (r["mean"], r["mean"] - r["sem"], r["mean"] + r["sem"], "+/-SEM over flies")
        rows.append({**dict(zip(by, key)), "metric": metric, "value": vals[0],
                     "err_lo": vals[1], "err_hi": vals[2], "interval": vals[3],
                     "n_flies": r["n_flies"], "n_trials": r["n_trials"]})
    return pd.DataFrame(rows)


def paired_comparisons(trials: pd.DataFrame, value: str, *, condition_col: str,
                       a: Any, b: Any, by: Sequence[str] = (), fly_col: str = "fly_id",
                       batch_col: str = "batch", family_col: str | None = None,
                       n_boot: int | None = None, n_perm: int | None = None,
                       seed: int | None = None) -> pd.DataFrame:
    """:func:`paired_effect` (b - a) for every ``by`` group, corrected.

    Per-fly means of ``value`` per condition (:func:`per_fly`), inner-joined on fly, so
    each row is a within-fly comparison; ``p_batch`` uses ``batch_col`` when present.
    ``p_holm`` / ``q_bh`` are added within ``family_col`` (default: the whole table is
    one family). One row per group with the :func:`paired_effect` fields.
    """
    by = list(by)
    pf = per_fly(trials, value, by=by + [condition_col], fly_col=fly_col,
                 carry=(batch_col,) if batch_col in trials.columns else ())
    batch_of = (pf.drop_duplicates(fly_col).set_index(fly_col)[batch_col]
                if batch_col in pf.columns else None)
    rows = []
    groups = pf.groupby(by, sort=True, dropna=False, observed=True) if by else [((), pf)]
    for key, sub in groups:
        key = key if isinstance(key, tuple) else (key,)
        sa = sub[sub[condition_col] == a].set_index(fly_col)[value]
        sb = sub[sub[condition_col] == b].set_index(fly_col)[value]
        eff = paired_effect(sa, sb, batch_of=batch_of, n_boot=n_boot, n_perm=n_perm,
                            seed=seed)
        rows.append({**dict(zip(by, key)), "value": value, "a": a, "b": b, **eff})
    out = pd.DataFrame(rows)
    return add_corrections(out, p_col="p", family_col=family_col) if len(out) else out


# ═════════════════════════════════════════════════════════════════════════════
# 4. multiplicity
# ═════════════════════════════════════════════════════════════════════════════
def holm(ps: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p (``fly_stats.holm_adjust``); NaN passes through."""
    return _fs.holm_adjust(list(ps))


def bh(ps: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg q (``fly_stats.bh_adjust``); NaN passes through."""
    return _fs.bh_adjust(list(ps))


def stars(p_holm: float | None) -> str:
    """'***' p < 0.001, '**' < 0.01, '*' < 0.05, else '' (also for NaN / None).

    Pass a HOLM-adjusted p: figures take stars from ``p_holm`` only. Unlike
    ``fly_stats.stars`` this never writes 'n.s.'.
    """
    if p_holm is None:
        return ""
    try:
        p = float(p_holm)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def add_corrections(table: pd.DataFrame, p_col: str = "p",
                    family_col: str | Sequence[str] | None = None) -> pd.DataFrame:
    """Copy of ``table`` with ``p_holm`` and ``q_bh`` from ``p_col``, both computed
    within each ``family_col`` group (None: the whole table is one family). When a
    ``p_batch`` column exists, ``p_batch_holm`` is added the same way."""
    out = table.copy()
    targets = [(p_col, "p_holm", "q_bh")]
    if "p_batch" in out.columns and p_col != "p_batch":
        targets.append(("p_batch", "p_batch_holm", None))
    for src, holm_col, bh_col in targets:
        out[holm_col] = np.nan
        if bh_col:
            out[bh_col] = np.nan
        if family_col is None:
            families = [out.index]
        else:
            families = [idx for _, idx in
                        out.groupby(family_col, sort=False, dropna=False).groups.items()]
        for idx in families:
            ps = pd.to_numeric(out.loc[idx, src], errors="coerce").to_numpy(float)
            out.loc[idx, holm_col] = holm(ps)
            if bh_col:
                out.loc[idx, bh_col] = bh(ps)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 5. difference in differences
# ═════════════════════════════════════════════════════════════════════════════
def did_bootstrap(exposed_changes: Sequence[float] | pd.Series,
                  control_changes: Sequence[float] | pd.Series, *,
                  batch_of: Mapping | pd.Series | None = None, n_boot: int | None = None,
                  n_perm: int | None = None, seed: int | None = None,
                  level: float = 0.95, method: str = "BCa") -> dict:
    """Did exposed flies change more than control flies? (difference in differences)

    Inputs: one within-fly change per fly (post - pre) for two independent groups;
    Series indexed by fly_id when ``batch_of`` is used. ``estimate`` = mean exposed
    change - mean control change (means over flies).

    * CI: bootstrap resampling flies within each group (stratified). ``"BCa"``: bias
      correction from the bootstrap, acceleration from the two-sample empirical influence
      values (Davison & Hinkley 1997, sec. 3.2 / 5.3); ``"percentile"`` also available.
    * ``p``: two-sided fly-level label permutation of the difference in means
      (``fly_stats.fly_permutation_p``, (hits+1)/(n_perm+1)).
    * ``p_batch`` (with ``batch_of``): whole batches swapped between groups,
      ``fly_stats.controlled_change_test(ci="permutation")``; exact over every split up to
      200k splits. ``ci_batch_lo/hi`` invert that test.
    """
    if method not in ("BCa", "percentile"):
        raise ValueError(f"method must be 'BCa' or 'percentile': {method!r}")
    n_boot = int(_resolve(n_boot, "N_BOOT"))
    n_perm = int(_resolve(n_perm, "N_PERM"))
    seed = int(_resolve(seed, "SEED"))
    xs = pd.Series(exposed_changes) if not isinstance(exposed_changes, pd.Series) else exposed_changes
    cs = pd.Series(control_changes) if not isinstance(control_changes, pd.Series) else control_changes
    xs = pd.to_numeric(xs, errors="coerce").astype(float).dropna()
    cs = pd.to_numeric(cs, errors="coerce").astype(float).dropna()
    x, c = xs.to_numpy(), cs.to_numpy()
    nan = float("nan")
    out = {"n_exposed": int(x.size), "n_control": int(c.size),
           "mean_exposed": float(x.mean()) if x.size else nan,
           "mean_control": float(c.mean()) if c.size else nan,
           "estimate": nan, "ci_lo": nan, "ci_hi": nan, "p": nan,
           "p_batch": nan, "ci_batch_lo": nan, "ci_batch_hi": nan,
           "perm_exact_batch": False, "n_batches_exposed": 0, "n_batches_control": 0}
    if x.size == 0 or c.size == 0:
        return out
    est = float(x.mean() - c.mean())
    out["estimate"] = est

    if x.size >= 2 and c.size >= 2:
        rng = np.random.default_rng(seed)
        boot = np.empty(n_boot)
        for start in range(0, n_boot, _MC_CHUNK):
            m = min(_MC_CHUNK, n_boot - start)
            boot[start:start + m] = (x[rng.integers(0, x.size, (m, x.size))].mean(1)
                                     - c[rng.integers(0, c.size, (m, c.size))].mean(1))
        if np.ptp(boot) == 0:
            lo = hi = est
        elif method == "percentile":
            lo, hi = (float(q) for q in np.quantile(boot, [(1 - level) / 2,
                                                           1 - (1 - level) / 2]))
        else:
            lx, lc = x - x.mean(), -(c - c.mean())       # empirical influence values
            num = np.sum(lx ** 3) / x.size ** 3 + np.sum(lc ** 3) / c.size ** 3
            den = np.sum(lx ** 2) / x.size ** 2 + np.sum(lc ** 2) / c.size ** 2
            accel = float(num / (6.0 * den ** 1.5)) if den > 0 else 0.0
            lo, hi = _bca_bounds(est, boot, accel, level)
        out.update(ci_lo=float(lo), ci_hi=float(hi))

    bx = _lookup_batches(list(xs.index), batch_of)
    bc = _lookup_batches(list(cs.index), batch_of)
    if batch_of is not None and pd.notna(bx).any() and pd.notna(bc).any():
        ex = pd.DataFrame({"change": x, "batch": bx}).dropna()
        ct = pd.DataFrame({"change": c, "batch": bc}).dropna()
        res = _fs.controlled_change_test(ex, ct, value="change", batch="batch",
                                         n_perm=n_perm, seed=seed, ci="permutation",
                                         alpha=1 - level)
        out.update(p_batch=float(res["p_batch"]), ci_batch_lo=float(res["ci_lo"]),
                   ci_batch_hi=float(res["ci_hi"]), perm_exact_batch=bool(res["perm_exact"]),
                   n_batches_exposed=int(res["n_batches_exposed"]),
                   n_batches_control=int(res["n_batches_control"]))
        if len(ex) == x.size and len(ct) == c.size:
            out["p"] = float(res["p_fly"])        # same call, same seed: no second run
    if not np.isfinite(out["p"]):
        out["p"] = float(_fs.fly_permutation_p(x, c, n_perm=n_perm, seed=seed))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 6. models
# ═════════════════════════════════════════════════════════════════════════════
COEF_COLUMNS = ["term", "estimate", "se", "ci_lo", "ci_hi", "p", "model", "n_obs",
                "n_groups", "status", "note"]

ModelResult = tuple[Any, pd.DataFrame, str]


def _formula_columns(df: pd.DataFrame, *texts: str) -> list[str]:
    """Columns of ``df`` named as whole identifiers in any of ``texts``."""
    joined = " ".join(texts)
    return [c for c in df.columns
            if re.search(r"(?<![\w.])" + re.escape(str(c)) + r"(?![\w])", joined)]


def _model_frame(df: pd.DataFrame, formula: str, extra: Sequence[str] = (),
                 vc_texts: Sequence[str] = ()) -> pd.DataFrame:
    """The rows and columns a fit needs: NaN rows dropped, booleans cast to 0/1.

    The cast matters: patsy treats a bool response as categorical and models
    P(False), which silently flips every coefficient's sign."""
    cols = list(dict.fromkeys(_formula_columns(df, formula, *vc_texts) + list(extra)))
    missing = [c for c in extra if c not in df.columns]
    if missing:
        raise KeyError(f"columns not in the frame: {missing}")
    d = df[cols].dropna().copy()
    for c in d.columns:
        if pd.api.types.is_bool_dtype(d[c]):
            d[c] = d[c].astype(int)
    return d.reset_index(drop=True)


def _failed(model: str, exc: BaseException | str, n_obs: int = 0,
            n_groups: int = 0) -> ModelResult:
    msg = f"{type(exc).__name__}: {exc}" if isinstance(exc, BaseException) else str(exc)
    row = {"term": "(fit failed)", "estimate": np.nan, "se": np.nan, "ci_lo": np.nan,
           "ci_hi": np.nan, "p": np.nan, "model": model, "n_obs": n_obs,
           "n_groups": n_groups, "status": "failed", "note": msg}
    return None, pd.DataFrame([row], columns=COEF_COLUMNS), f"{model}\nFIT FAILED: {msg}\n"


def _convergence_notes(caught: list) -> list[str]:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    notes = []
    for w in caught:
        text = str(w.message)
        if (issubclass(w.category, ConvergenceWarning) or "converg" in text.lower()
                or "separation" in text.lower() or "singular" in text.lower()):
            notes.append(text.replace("\n", " "))
    return list(dict.fromkeys(notes))


def _coef_frame(terms: Sequence[str], est, se, lo, hi, p, *, model: str, n_obs: int,
                n_groups: int, status: str, note: str) -> pd.DataFrame:
    df = pd.DataFrame({"term": list(terms), "estimate": np.asarray(est, float),
                       "se": np.asarray(se, float), "ci_lo": np.asarray(lo, float),
                       "ci_hi": np.asarray(hi, float), "p": np.asarray(p, float)})
    df["model"], df["n_obs"], df["n_groups"] = model, int(n_obs), int(n_groups)
    df["status"], df["note"] = status, note
    return df[COEF_COLUMNS]


def _single_blas_thread() -> contextlib.AbstractContextManager:
    """Limit BLAS to one thread when threadpoolctl is installed, else do nothing."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return contextlib.nullcontext()
    return threadpool_limits(limits=1)


def _fit_guarded(model: str, fit: Callable[[], Any]) -> tuple[Any, list[str], BaseException | None]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            res = fit()
        except Exception as exc:  # noqa: BLE001 -- any fitting error becomes a failed row
            return None, [], exc
    return res, _convergence_notes(caught), None


def _status(converged: bool | None, notes: list[str]) -> tuple[str, str]:
    if converged is False:
        return "not_converged", "; ".join(notes) or "optimizer reported no convergence"
    if any("converg" in n.lower() for n in notes) and converged is None:
        return "not_converged", "; ".join(notes)
    return "ok", "; ".join(notes)


def fit_gee_logistic(df: pd.DataFrame, formula: str, group: str = "fly_id") -> ModelResult:
    """Population-averaged logistic GEE: Binomial family, exchangeable working
    correlation within ``group`` (the fly), robust (sandwich) SEs.

    Returns ``(result, coefs, summary_text)``; estimates are log-odds. A fit that raises
    returns ``(None, <one 'failed' row>, text)``; one that warns of non-convergence keeps
    its estimates with ``status == 'not_converged'`` -- filter on ``status`` before use.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    label = f"GEE logistic (Binomial, exchangeable within {group}, robust SE)"
    try:
        d = _model_frame(df, formula, extra=[group])
    except Exception as exc:  # noqa: BLE001
        return _failed(label, exc)
    n_obs, n_groups = len(d), int(d[group].nunique())
    res, notes, err = _fit_guarded(label, lambda: smf.gee(
        formula, groups=group, data=d, family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable()).fit())
    if err is not None:
        return _failed(label, err, n_obs, n_groups)
    status, note = _status(getattr(res, "converged", None), notes)
    ci = np.asarray(res.conf_int())
    coefs = _coef_frame(res.params.index, res.params, res.bse, ci[:, 0], ci[:, 1],
                        res.pvalues, model=label, n_obs=n_obs, n_groups=n_groups,
                        status=status, note=note)
    text = f"{label}\nformula: {formula}\nstatus: {status} {note}\n\n{res.summary()}\n"
    return res, coefs, text


def _vc_group_columns(df: pd.DataFrame, vc: Mapping[str, str]) -> list[str]:
    return _formula_columns(df, *vc.values())


def fit_mixed_logistic(df: pd.DataFrame, formula: str,
                       vc: Mapping[str, str] | None = None) -> ModelResult:
    """Logistic GLMM with random intercepts, fitted as statsmodels'
    ``BinomialBayesMixedGLM.fit_vb`` -- a BAYESIAN VARIATIONAL APPROXIMATION, not ML.

    ``vc`` defaults to ``{"fly": "0 + C(fly_id)"}`` (a random intercept per fly).
    ``estimate`` / ``se`` are the posterior mean / SD of each fixed effect; ``ci`` =
    mean +/- 1.96 SD and ``p`` = 2 Phi(-|mean / SD|) are normal approximations to the
    variational posterior, not frequentist tests. Variance components are reported as
    ``VC <name> (log SD)`` rows with p NaN.
    """
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    vc = dict(vc or {"fly": "0 + C(fly_id)"})
    label = "Bayesian mixed logistic (BinomialBayesMixedGLM, variational approximation fit_vb)"
    try:
        d = _model_frame(df, formula, vc_texts=list(vc.values()))
        gcols = _vc_group_columns(d, vc)
        if not gcols:
            raise KeyError(f"no grouping column of {list(vc.values())} in the frame")
    except Exception as exc:  # noqa: BLE001
        return _failed(label, exc)
    n_obs, n_groups = len(d), int(d[gcols[0]].nunique())
    res, notes, err = _fit_guarded(label, lambda: BinomialBayesMixedGLM.from_formula(
        formula, vc, d).fit_vb())
    if err is not None:
        return _failed(label, err, n_obs, n_groups)
    converged = None
    opt = getattr(res, "optim_retvals", None)
    if isinstance(opt, Mapping) and "success" in opt:
        converged = bool(opt["success"])
    elif opt is not None and hasattr(opt, "success"):
        converged = bool(opt.success)
    status, note = _status(converged, notes)
    fe_m, fe_s = np.asarray(res.fe_mean, float), np.asarray(res.fe_sd, float)
    z = np.divide(fe_m, fe_s, out=np.full_like(fe_m, np.nan), where=fe_s > 0)
    fe = _coef_frame(res.model.exog_names, fe_m, fe_s, fe_m - 1.959964 * fe_s,
                     fe_m + 1.959964 * fe_s, 2 * norm.sf(np.abs(z)), model=label,
                     n_obs=n_obs, n_groups=n_groups, status=status, note=note)
    vm, vs = np.asarray(res.vcp_mean, float), np.asarray(res.vcp_sd, float)
    vc_rows = _coef_frame([f"VC {n} (log SD)" for n in res.model.vcp_names], vm, vs,
                          vm - 1.959964 * vs, vm + 1.959964 * vs, np.full(vm.size, np.nan),
                          model=label, n_obs=n_obs, n_groups=n_groups, status=status,
                          note=note)
    coefs = pd.concat([fe, vc_rows], ignore_index=True)
    text = (f"{label}\nNOTE: Bayesian variational approximation (posterior mean / SD); "
            f"p = normal approximation to the posterior.\nformula: {formula}\nvc: {vc}\n"
            f"status: {status} {note}\n\n{res.summary()}\n")
    return res, coefs, text


def fit_mixedlm(df: pd.DataFrame, formula: str, group: str = "fly_id", *,
                re_formula: str | None = None, reml: bool = True) -> ModelResult:
    """Linear mixed model (statsmodels MixedLM), random intercept per ``group`` (plus
    ``re_formula`` slopes). Wald SEs / p from the fit; the variance rows ('Group Var')
    are kept in the table."""
    import statsmodels.formula.api as smf
    label = f"Linear mixed model (MixedLM, random intercept by {group}, {'REML' if reml else 'ML'})"
    try:
        d = _model_frame(df, formula, extra=[group], vc_texts=[re_formula or ""])
    except Exception as exc:  # noqa: BLE001
        return _failed(label, exc)
    n_obs, n_groups = len(d), int(d[group].nunique())
    res, notes, err = _fit_guarded(label, lambda: smf.mixedlm(
        formula, d, groups=d[group], re_formula=re_formula).fit(reml=reml))
    if err is not None:
        return _failed(label, err, n_obs, n_groups)
    status, note = _status(getattr(res, "converged", None), notes)
    ci = res.conf_int()
    coefs = _coef_frame(res.params.index, res.params, res.bse.reindex(res.params.index),
                        ci.iloc[:, 0].reindex(res.params.index),
                        ci.iloc[:, 1].reindex(res.params.index),
                        res.pvalues.reindex(res.params.index), model=label, n_obs=n_obs,
                        n_groups=n_groups, status=status, note=note)
    text = f"{label}\nformula: {formula}\nstatus: {status} {note}\n\n{res.summary()}\n"
    return res, coefs, text


def fit_ordinal(df: pd.DataFrame, formula: str, cluster: str = "fly_id", *,
                n_boot: int = 200, seed: int | None = None,
                distr: str = "logit") -> ModelResult:
    """Proportional-odds (cumulative logit) model of the ordinal score, statsmodels
    ``OrderedModel``, with SEs from a FLY-CLUSTER BOOTSTRAP.

    OrderedModel has no cluster-robust covariance, and its model-based SEs treat a fly's
    trials as independent. Here flies are resampled with replacement (all their trials
    together) ``n_boot`` times and the model refitted from the full-data estimates; ``se``
    = SD of the replicates, ``ci`` = estimate +/- 1.96 se, ``p`` = Wald z on that se.
    ``n_boot`` defaults to 200 to bound runtime: ~0.09 s per refit on 2,000 rows / 100
    flies (one BLAS thread), so ~20 s per model; raise it for a final figure. Slopes come
    from every replicate; a replicate that lacks a score level drops a threshold, so
    thresholds use only full-level replicates (the note says how many).

    Do not write an intercept-free formula (``0 + ...``): OrderedModel drops the
    intercept itself and keeps the reduced-rank coding.
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    seed = int(_resolve(seed, "SEED"))
    label = f"Ordinal logit (OrderedModel {distr}, fly-cluster bootstrap SE, n_boot={n_boot})"
    try:
        d = _model_frame(df, formula, extra=[cluster])
        model = OrderedModel.from_formula(formula, d, distr=distr)
    except Exception as exc:  # noqa: BLE001
        return _failed(label, exc)
    n_obs, n_groups = len(d), int(d[cluster].nunique())
    res, notes, err = _fit_guarded(label, lambda: model.fit(method="bfgs", disp=False,
                                                            maxiter=2000))
    if err is not None:
        return _failed(label, err, n_obs, n_groups)
    names = list(res.params.index)
    k = int(model.exog.shape[1])
    est = res.params.to_numpy(float)
    y = np.asarray(model.labels)[np.asarray(model.endog, dtype=int)]
    X = np.asarray(model.exog, float)
    levels = np.unique(y)
    rows_of = [np.asarray(ix) for ix in d.groupby(cluster, sort=True).indices.values()]
    rng = np.random.default_rng(seed)
    slopes, full, n_fail = [], [], 0
    # One BLAS thread: on these small matrices OpenBLAS's 16 threads only thrash
    # (0.36 s vs 0.09 s per refit, 2,000 rows, measured 2026-11-18).
    with _single_blas_thread():
        for _ in range(int(n_boot)):
            pick = rng.integers(0, len(rows_of), len(rows_of))
            rows = np.concatenate([rows_of[i] for i in pick])
            yb, xb = y[rows], X[rows]
            same = np.array_equal(np.unique(yb), levels)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    rb = OrderedModel(yb, xb, distr=distr).fit(   # no Hessian: unused here
                        method="bfgs", disp=False, maxiter=2000,
                        start_params=est if same else None, skip_hessian=True)
                except Exception:  # noqa: BLE001 -- a failed replicate is counted, not fatal
                    n_fail += 1
                    continue
            pb = np.asarray(rb.params, float)
            if not np.all(np.isfinite(pb[:k])):
                n_fail += 1
                continue
            slopes.append(pb[:k])
            if same:
                full.append(pb)
    se = np.full(est.size, np.nan)
    if len(slopes) >= 2:
        se[:k] = np.std(np.vstack(slopes), axis=0, ddof=1)
    if len(full) >= 2:
        se[k:] = np.std(np.vstack(full)[:, k:], axis=0, ddof=1)
    z = np.divide(est, se, out=np.full_like(est, np.nan), where=se > 0)
    status, note = _status(res.mle_retvals.get("converged"), notes)
    boot_note = (f"cluster bootstrap: {len(slopes)}/{n_boot} replicates for slopes, "
                 f"{len(full)} with every score level for thresholds, {n_fail} failed")
    note = "; ".join(x for x in (note, boot_note) if x)
    coefs = _coef_frame(names, est, se, est - 1.959964 * se, est + 1.959964 * se,
                        2 * norm.sf(np.abs(z)), model=label, n_obs=n_obs, n_groups=n_groups,
                        status=status, note=note)
    text = (f"{label}\nformula: {formula}\nstatus: {status}\n{boot_note}\n"
            "NOTE: the summary below shows OrderedModel's model-based SEs (trials treated "
            "as independent). The coefficient CSV uses the fly-cluster bootstrap SEs:\n"
            f"{coefs[['term', 'estimate', 'se', 'p']].to_string(index=False)}\n\n"
            f"{res.summary()}\n")
    return res, coefs, text


def save_model(result_tuple: ModelResult, path_stem: str | Path) -> tuple[Path, Path]:
    """Write ``<stem>.txt`` (the summary text) and ``<stem>_coefs.csv`` (term, estimate,
    se, ci_lo, ci_hi, p, model, n_obs, n_groups, status, note). Creates parent dirs.
    Returns the two paths."""
    _, coefs, text = result_tuple
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    txt = stem.with_name(stem.name + ".txt")
    csv = stem.with_name(stem.name + "_coefs.csv")
    txt.write_text(text if text.endswith("\n") else text + "\n")
    cols = [c for c in COEF_COLUMNS if c in coefs.columns] + \
        [c for c in coefs.columns if c not in COEF_COLUMNS]
    coefs[cols].to_csv(csv, index=False)
    return txt, csv


# ═════════════════════════════════════════════════════════════════════════════
# 7. power
# ═════════════════════════════════════════════════════════════════════════════
def _all_signs(n: int) -> np.ndarray:
    return (((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1).astype(float)


def power_curve(data: Sequence[float] | tuple[Sequence[float], Sequence[float]], *,
                deltas: Sequence[float], ns: Sequence[int], test: str = "paired",
                alpha: float = 0.05, n_sim: int = 1000, seed: int | None = None,
                center: bool = True, n_perm_inner: int = 2000) -> pd.DataFrame:
    """Simulated power, resampling the observed flies.

    ``test="paired"``: ``data`` = per-fly within-fly differences. Each simulation draws
    ``n`` flies with replacement, shifts them by ``delta`` and runs the paired sign-flip
    test (all 2^n patterns when that is <= ``n_perm_inner``, else ``n_perm_inner`` seeded
    patterns shared by every simulation -- Monte Carlo inside the simulation).

    ``test="two_group"``: ``data`` = (group_a, group_b) per-fly values; ``n`` flies are
    drawn from each, ``delta`` is added to group b, and a label-permutation test of the
    difference in means (``n_perm_inner`` seeded permutations) is run.

    ``center=True`` (default) first removes each sample's observed mean, so ``delta`` is
    the assumed true effect and ``delta = 0`` gives the test's size; pass
    ``center=False`` to shift the observed effect instead. Rejection: p <= alpha.
    Returns a long table: design, delta, n, power, n_sim.
    """
    seed = int(_resolve(seed, "SEED"))
    rng = np.random.default_rng(seed)
    rows = []
    if test == "paired":
        d = _finite(data)
        if d.size < 2:
            raise ValueError("power_curve needs at least two per-fly differences")
        if center:
            d = d - d.mean()
        for n in ns:
            n = int(n)
            exact = 2 ** n <= n_perm_inner
            signs = _all_signs(n) if exact else rng.choice((-1.0, 1.0), (n_perm_inner, n))
            for delta in deltas:
                D = d[rng.integers(0, d.size, (n_sim, n))] + float(delta)
                obs = np.abs(D.sum(1))
                null = np.abs(np.abs(D) @ signs.T)
                tol = 1e-9 * np.maximum(1.0, np.abs(D).sum(1))
                hits = np.sum(null >= (obs - tol)[:, None], axis=1)
                p = hits / signs.shape[0] if exact else (hits + 1) / (signs.shape[0] + 1)
                rows.append({"design": "paired", "delta": float(delta), "n": n,
                             "power": float(np.mean(p <= alpha)), "n_sim": int(n_sim)})
    elif test == "two_group":
        a, b = (_finite(v) for v in data)
        if a.size < 2 or b.size < 2:
            raise ValueError("power_curve needs at least two flies per group")
        if center:
            a, b = a - a.mean(), b - b.mean()
        for n in ns:
            n = int(n)
            lab = np.argsort(rng.random((n_perm_inner, 2 * n)), axis=1) < n
            W = np.where(lab, 1.0 / n, -1.0 / n)                # +: "b" side, -: "a" side
            w_obs = np.r_[np.full(n, -1.0 / n), np.full(n, 1.0 / n)]
            for delta in deltas:
                A = a[rng.integers(0, a.size, (n_sim, n))]
                B = b[rng.integers(0, b.size, (n_sim, n))] + float(delta)
                Z = np.hstack([A, B])
                obs = np.abs(Z @ w_obs)
                null = np.abs(Z @ W.T)
                hits = np.sum(null >= (obs - 1e-12)[:, None], axis=1)
                p = (hits + 1) / (n_perm_inner + 1)
                rows.append({"design": "two_group", "delta": float(delta), "n": n,
                             "power": float(np.mean(p <= alpha)), "n_sim": int(n_sim)})
    else:
        raise ValueError(f"test must be 'paired' or 'two_group': {test!r}")
    return pd.DataFrame(rows)


def min_n_for_power(table: pd.DataFrame, target: float = 0.8) -> pd.DataFrame:
    """Smallest simulated ``n`` reaching ``power >= target``, per delta (and design when
    present); NaN when no n in the grid reaches it. Grid values only, no interpolation."""
    keys = [k for k in ("design", "delta") if k in table.columns]
    rows = []
    for key, sub in table.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        ok = sub.loc[sub["power"] >= target, "n"]
        rows.append({**dict(zip(keys, key)), "target": target,
                     "min_n": float(ok.min()) if len(ok) else float("nan")})
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# 8. shared-baseline diagnostic
# ═════════════════════════════════════════════════════════════════════════════
DIRECTION_NULL = (
    "H0: within each fly the alone baseline and its pair measurements are exchangeable "
    "as a block, so the fly's whole row of changes (pair_odor - alone, every odor) is "
    "symmetric under joint negation. Each null draw multiplies every fly's entire row by "
    "one random sign (all odors flipped jointly), which keeps the cross-odor correlation "
    "the shared alone baseline induces. Statistic: |#odors whose mean change over flies "
    "> 0 - #odors < 0|; two-sided p = P(null >= observed).")


def direction_count_test(per_fly_changes_by_odor: pd.DataFrame, *,
                         n_perm: int | None = None, seed: int | None = None) -> dict:
    """Do more odors go up (or down) than a shared baseline alone would produce?

    ``per_fly_changes_by_odor``: index = fly_id, columns = odor, values = that fly's
    change (pair - alone) for the odor; NaN where the fly lacks the odor. Every odor's
    change uses the same alone baseline, so a fly with a low baseline pushes every odor
    up together, and the naive binomial count (odors as independent coin flips,
    ``p_naive_binomial``) overstates the evidence. The null is :data:`DIRECTION_NULL`:
    exact over all 2^flies joint flips up to :data:`DIRECTION_EXACT_MAX` flies, else
    ``n_perm`` seeded draws ((hits + 1) / (n + 1)).

    Returns ``{n_odors, n_flies, n_up, n_down, n_zero, statistic, p, exact,
    p_naive_binomial, mean_change, mean_cross_odor_corr, null_n_up, null}``.
    """
    n_perm = int(_resolve(n_perm, "N_PERM"))
    seed = int(_resolve(seed, "SEED"))
    X = per_fly_changes_by_odor.apply(pd.to_numeric, errors="coerce").astype(float)
    X = X.loc[X.notna().any(axis=1), X.notna().any(axis=0)]
    vals = X.to_numpy(float)
    fin = np.isfinite(vals)
    X0 = np.where(fin, vals, 0.0)
    n_fly, n_odor = X0.shape
    sums = X0.sum(0)
    tol = 1e-9 * max(1.0, float(np.abs(X0).sum()))
    up, down = int(np.sum(sums > tol)), int(np.sum(sums < -tol))
    obs = abs(up - down)
    corr = X.corr().to_numpy(float) if n_odor > 1 else np.full((1, 1), np.nan)
    off = corr[~np.eye(n_odor, dtype=bool)] if n_odor > 1 else np.array([np.nan])
    out = {"n_odors": n_odor, "n_flies": n_fly, "n_up": up, "n_down": down,
           "n_zero": n_odor - up - down, "statistic": obs, "p": float("nan"), "exact": False,
           "p_naive_binomial": (float(binomtest(up, up + down, 0.5).pvalue)
                                if up + down else float("nan")),
           "mean_change": dict(zip(X.columns, (sums / np.maximum(fin.sum(0), 1)).tolist())),
           "mean_cross_odor_corr": float(np.nanmean(off)) if np.isfinite(off).any() else float("nan"),
           "null_n_up": {}, "null": DIRECTION_NULL}
    if n_fly == 0:
        return out
    up_counts = np.zeros(n_odor + 1)
    hits = 0

    def tally(signs: np.ndarray) -> int:
        s = signs @ X0
        nu, nd = np.sum(s > tol, axis=1), np.sum(s < -tol, axis=1)
        np.add.at(up_counts, nu, 1)
        return int(np.sum(np.abs(nu - nd) >= obs))

    if n_fly <= DIRECTION_EXACT_MAX:
        total = 2 ** n_fly
        for start in range(0, total, 1 << 14):
            idx = np.arange(start, min(total, start + (1 << 14)))
            hits += tally((((idx[:, None] >> np.arange(n_fly)) & 1) * 2 - 1).astype(float))
        p, exact = hits / total, True
    else:
        rng = np.random.default_rng(seed)
        done = 0
        while done < n_perm:
            m = min(_MC_CHUNK, n_perm - done)
            hits += tally(rng.choice((-1.0, 1.0), size=(m, n_fly)))
            done += m
        p, exact = (hits + 1) / (n_perm + 1), False
    out.update(p=float(min(1.0, p)), exact=exact,
               null_n_up={int(k): float(v / up_counts.sum())
                          for k, v in enumerate(up_counts) if v > 0})
    return out


__all__ = (
    "COEF_COLUMNS", "DIRECTION_NULL", "EXACT_ENUM_MAX", "add_corrections",
    "batch_sign_flip_test", "bh", "bootstrap_ci", "config_value", "did_bootstrap",
    "direction_count_test", "fit_gee_logistic", "fit_mixed_logistic", "fit_mixedlm",
    "fit_ordinal", "group_summary", "holm", "min_n_for_power", "paired_comparisons",
    "paired_effect", "per_fly", "pooled_mean_sem", "pooled_rate_ci", "power_curve",
    "save_model", "sign_flip_test", "stars",
)
