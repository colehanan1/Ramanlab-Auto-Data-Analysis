"""Power curves -> ``<out>/06_power/`` and ``tables/power_min_n.csv``.

Simulated power, resampling the OBSERVED per-fly values (``stats.power_curve``): each
simulation draws n flies with replacement from the centred per-fly values, adds the
assumed true effect Δ and runs the design's test at α = 0.05. n = 20..150 flies in steps
of 10; Δ score 0.25 and 0.5; Δ rate 10, 15 and 20 percentage points (per-fly 0/1 means,
so Δ 0.15 = 15 pp).

Designs
-------
* Exp 1 (two groups, ``test="two_group"``): per-fly post - pre change on the exposed odor,
  exposed cohort vs the Paraffin-Sensitivity cohort (same odor), one design per exposed
  cohort (IAA, Hexanol, 3-Octanol). n = flies PER GROUP.
* Exp 2 (within fly, ``test="paired"``): per fly (X pair - X alone) - (click pair - click
  alone). "one partner": the 7 partners' per-fly values, each centred on its own partner
  mean, pooled -- the test of a single partner X; "mean of 7 partners": the per-fly mean
  over the 7 partners.
* Exp 3 (within fly): per fly IAA after X - IAA after click; same two designs.

Effect models
-------------
* ``shift`` (as specified; ``stats.power_curve``): Δ is added to every resampled fly. For
  rates this keeps the observed per-fly variance, which for zero-heavy contrasts (most
  flies never respond, e.g. Exp 3) is tiny, so a 15 pp shift moves EVERY fly a little and
  is detected with very few flies -- optimistic.
* ``binomial`` (rates only, sensitivity): each resampled fly gains Binomial(k, Δ) / k,
  k = the trials on the shifted side of its contrast (Exp 1: 1 post trial; one partner: 2
  pair presentations; mean of 7 partners: 14). Same mean shift Δ, plus the variance a real
  15 pp rise in responding brings (some flies flip 0 -> 1). Ceiling effects are ignored.

Monte Carlo sizes are module constants (:data:`N_SIM`, :data:`N_PERM_INNER`), modest so
the run takes under a minute; they are written into every table row.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from per_analysis import config as C  # noqa: E402
from per_analysis import cross as X  # noqa: E402
from per_analysis import plots as pl  # noqa: E402
from per_analysis import stats as st  # noqa: E402

log = logging.getLogger(__name__)

SECTION = "power"
SUBDIR = "06_power"
N_SIM = 2000                 # simulated experiments per (design, Δ, n)
N_PERM_INNER = 2000          # permutations / sign patterns inside each simulated test
NS = tuple(range(20, 151, 10))
ALPHA = 0.05
TARGET = 0.8
DELTAS = {"rate": (0.10, 0.15, 0.20), "score": (0.25, 0.5)}
HIGHLIGHT = ("rate", 0.15)
EXP_TITLES = {"exp1": "Exp 1: exposed vs paraffin cohort, post − pre on the exposed odor",
              "exp2": "Exp 2: (X after IAA − X alone) − (click after IAA − click alone)",
              "exp3": "Exp 3: IAA after X − IAA after click"}


# ── per-fly design values ────────────────────────────────────────────────────
def _changes(odor: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Exp 1 per fly x odor: post - pre (fly_id, cohort, odor, value)."""
    d = odor[(odor["experiment"] == "exp1") & odor["block"].isin(["pre", "post"])]
    pf = X.fly_values(d, metric, by=["cohort", "block", "odor"])
    w = pf.pivot_table(index=["cohort", "odor", "fly_id"], columns="block", values="value",
                       aggfunc="first")
    if not {"pre", "post"} <= set(w.columns):
        return pd.DataFrame(columns=["cohort", "odor", "fly_id", "value"])
    return (w["post"] - w["pre"]).dropna().rename("value").reset_index()


def exp1_values(odor: pd.DataFrame, metric: str) -> dict:
    """exposure odor -> {"exposed": fly->change, "control": fly->change, "cohort": name}:
    each exposed cohort's per-fly post - pre on its exposure odor, and the control
    (paraffin) cohort's on the same odor."""
    ch = _changes(odor, metric)
    e1 = odor[odor["experiment"] == "exp1"]
    out = {}
    for cohort in [c for c in C.COHORTS if C.COHORTS[c] == "exp1" and c != C.CONTROL_COHORT]:
        sub = e1[e1["cohort"] == cohort]
        if sub.empty:
            continue
        e_odor = str(sub["exposure_odor"].dropna().iloc[0])
        exposed = ch[(ch["cohort"] == cohort) & (ch["odor"] == e_odor)].set_index("fly_id")["value"]
        control = ch[(ch["cohort"] == C.CONTROL_COHORT) & (ch["odor"] == e_odor)] \
            .set_index("fly_id")["value"]
        if len(exposed) >= 2 and len(control) >= 2:
            out[e_odor] = {"exposed": exposed, "control": control, "cohort": cohort}
    return out


def _contrast_values(contrasts: Mapping) -> pd.DataFrame:
    rows = []
    for x, (a, b) in contrasts.items():
        d = (b - a).dropna()
        rows.append(pd.DataFrame({"fly_id": d.index, "partner": x, "value": d.to_numpy(float)}))
    return (pd.concat(rows, ignore_index=True) if rows
            else pd.DataFrame(columns=["fly_id", "partner", "value"]))


def exp2_values(odor: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Exp 2 per fly x partner: (X pair - X alone) - (click pair - click alone)."""
    return _contrast_values(X.exp2_contrasts(odor, metric))


def exp3_values(odor: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Exp 3 per fly x partner: IAA after X - IAA after click."""
    return _contrast_values(X.exp3_contrasts(odor, metric))


# ── simulation ───────────────────────────────────────────────────────────────
def simulate(data, *, kind: str, deltas: Sequence[float], ns: Sequence[int]) -> pd.DataFrame:
    """``stats.power_curve`` with this module's Monte Carlo sizes."""
    return st.power_curve(data, deltas=list(deltas), ns=list(ns), test=kind, alpha=ALPHA,
                          n_sim=N_SIM, seed=C.SEED, n_perm_inner=N_PERM_INNER)


def simulate_binomial(data, *, kind: str, k: int, deltas: Sequence[float],
                      ns: Sequence[int]) -> pd.DataFrame:
    """Like :func:`simulate` (same tests, same rejection rule), but each resampled fly of
    the shifted group gains Binomial(k, Δ) / k instead of exactly Δ."""
    rng = np.random.default_rng(C.SEED)
    rows = []
    if kind == "paired":
        d = np.asarray(data, float)
        d = d[np.isfinite(d)]
        d = d - d.mean()
        for n in ns:
            n = int(n)
            exact = 2 ** n <= N_PERM_INNER
            signs = ((((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1).astype(float)
                     if exact else rng.choice((-1.0, 1.0), (N_PERM_INNER, n)))
            for delta in deltas:
                D = (d[rng.integers(0, d.size, (N_SIM, n))]
                     + rng.binomial(k, float(delta), (N_SIM, n)) / k)
                obs = np.abs(D.sum(1))
                null = np.abs(np.abs(D) @ signs.T)
                tol = 1e-9 * np.maximum(1.0, np.abs(D).sum(1))
                hits = np.sum(null >= (obs - tol)[:, None], axis=1)
                pv = hits / len(signs) if exact else (hits + 1) / (len(signs) + 1)
                rows.append({"design": "paired", "delta": float(delta), "n": n,
                             "power": float(np.mean(pv <= ALPHA)), "n_sim": N_SIM})
    elif kind == "two_group":
        a, b = (np.asarray(v, float) for v in data)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        a, b = a - a.mean(), b - b.mean()
        for n in ns:
            n = int(n)
            lab = np.argsort(rng.random((N_PERM_INNER, 2 * n)), axis=1) < n
            W = np.where(lab, 1.0 / n, -1.0 / n)
            w_obs = np.r_[np.full(n, -1.0 / n), np.full(n, 1.0 / n)]
            for delta in deltas:
                A = a[rng.integers(0, a.size, (N_SIM, n))]
                B = (b[rng.integers(0, b.size, (N_SIM, n))]
                     + rng.binomial(k, float(delta), (N_SIM, n)) / k)
                Z = np.hstack([A, B])
                obs = np.abs(Z @ w_obs)
                hits = np.sum(np.abs(Z @ W.T) >= (obs - 1e-12)[:, None], axis=1)
                pv = (hits + 1) / (N_PERM_INNER + 1)
                rows.append({"design": "two_group", "delta": float(delta), "n": n,
                             "power": float(np.mean(pv <= ALPHA)), "n_sim": N_SIM})
    else:
        raise ValueError(f"kind must be 'paired' or 'two_group': {kind!r}")
    return pd.DataFrame(rows)


def min_n_table(curves: pd.DataFrame, target: float = TARGET) -> pd.DataFrame:
    """Smallest simulated n with power >= target per (experiment, design, metric, Δ);
    ``highlight_15pp`` marks the Δ rate 15 pp rows."""
    if "effect_model" not in curves.columns:
        curves = curves.assign(effect_model="shift")
    keys = ["experiment", "design", "metric", "effect_model"]
    rows = []
    for key, sub in curves.groupby(keys, sort=False):
        m = st.min_n_for_power(sub[["delta", "n", "power"]], target=target)
        for r in m.itertuples():
            rows.append(dict(zip(keys, key), delta=r.delta, target=target, min_n=r.min_n))
    out = pd.DataFrame(rows)
    out["highlight_15pp"] = (out["metric"] == HIGHLIGHT[0]) & np.isclose(out["delta"],
                                                                         HIGHLIGHT[1])
    return out


def _one_sample(d: pd.Series, batch_of: Mapping, scale: float) -> dict:
    """Observed mean of per-fly values d against 0 (sign-flip, BCa CI, batch sign-flip)."""
    d = d.dropna().astype(float)
    sf = st.sign_flip_test(d.to_numpy())
    lo, hi = st.bootstrap_ci(d.to_numpy(), n_boot=X.N_BOOT, seed=C.SEED)
    b = np.array([batch_of.get(f) for f in d.index], dtype=object)
    bt = st.batch_sign_flip_test(d.to_numpy(), b, ci=False)
    return {"n_flies": int(d.size), "n_batches": int(bt["n_batches"]),
            "effect": float(d.mean()) * scale, "ci_lo": lo * scale, "ci_hi": hi * scale,
            "p": sf["p"], "p_batch": float(bt["p"])}


def designs(odor: pd.DataFrame) -> list[dict]:
    """Every (experiment, design, metric) with its data for ``power_curve`` and the
    observed effect of that design (NaN for the pooled one-partner design)."""
    batch_of = X.batch_of_flies(odor)
    sess = X.session_of_batches(odor)
    out = []
    for metric in X.METRICS:
        sc = X.SCALE[metric]
        for e_odor, v in exp1_values(odor, metric).items():
            obs = X.two_group(v["control"], v["exposed"], batch_of, sess, scale=sc)
            out.append(dict(experiment="exp1", metric=metric, kind="two_group", k=1,
                            design=f"{e_odor}: {v['cohort']} vs {C.CONTROL_COHORT}",
                            data=(v["control"].to_numpy(float), v["exposed"].to_numpy(float)),
                            sd=float(np.std(np.r_[v["control"] - v["control"].mean(),
                                                  v["exposed"] - v["exposed"].mean()],
                                            ddof=1)) * sc,
                            observed=obs, test="two-group label permutation (difference "
                                               "of mean post - pre change, flies per group)"))
        for exp, values in (("exp2", exp2_values(odor, metric)),
                            ("exp3", exp3_values(odor, metric))):
            if values.empty:
                continue
            centred = values["value"] - values.groupby("partner")["value"].transform("mean")
            out.append(dict(experiment=exp, metric=metric, kind="paired", k=2,
                            design="one partner (7 partners pooled, each centred)",
                            data=centred.to_numpy(float),
                            sd=float(np.std(centred, ddof=1)) * sc,
                            observed={"n_flies": int(values["fly_id"].nunique()),
                                      "effect": float(values["value"].mean()) * sc},
                            test="paired sign-flip of the per-fly contrast"))
            mean7 = values.groupby("fly_id")["value"].mean()
            out.append(dict(experiment=exp, metric=metric, kind="paired", k=14,
                            design="mean of 7 partners", data=mean7.to_numpy(float),
                            sd=float(np.std(mean7, ddof=1)) * sc,
                            observed=_one_sample(mean7, batch_of, sc),
                            test="paired sign-flip of the per-fly mean contrast"))
    return out


def run_designs(ds: Sequence[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(long curves table, min-n table with the observed effects)."""
    curves = []
    for d in ds:
        tag = dict(experiment=d["experiment"], design=d["design"], metric=d["metric"])
        c = simulate(d["data"], kind=d["kind"], deltas=DELTAS[d["metric"]], ns=NS)
        curves.append(c.assign(**tag, effect_model="shift"))
        if d["metric"] == "rate":
            c = simulate_binomial(d["data"], kind=d["kind"], k=d["k"],
                                  deltas=DELTAS[d["metric"]], ns=NS)
            curves.append(c.assign(**tag, effect_model="binomial"))
        log.debug("power %s %s %s done", d["experiment"], d["design"], d["metric"])
    curves = pd.concat(curves, ignore_index=True)
    m = min_n_table(curves)
    info = []
    for d in ds:
        o = d["observed"]
        info.append({"experiment": d["experiment"], "design": d["design"], "metric": d["metric"],
                     "kind": d["kind"], "k_trials_shifted": d["k"], "sd_per_fly": d["sd"], "n_flies": o.get("n_flies"),
                     "n_flies_per_group_observed": (f"{o.get('n_flies_a')} control / "
                                                    f"{o.get('n_flies_b')} exposed"
                                                    if d["kind"] == "two_group" else ""),
                     "effect": o.get("effect", np.nan), "ci_lo": o.get("ci_lo", np.nan),
                     "ci_hi": o.get("ci_hi", np.nan), "p": o.get("p", np.nan),
                     "p_batch": o.get("p_batch", np.nan), "test": d["test"]})
    m = m.merge(pd.DataFrame(info), on=["experiment", "design", "metric"], how="left")
    m["delta_display"] = np.where(m["metric"] == "rate", m["delta"] * 100, m["delta"])
    m["delta_unit"] = np.where(m["metric"] == "rate", "pp", "score points")
    m["alpha"], m["n_sim"], m["n_perm_inner"] = ALPHA, N_SIM, N_PERM_INNER
    m["p_holm"], m["q_bh"] = m["p"], m["p"]
    m["correction"] = "none (one observed test per design; min n is a simulation result)"
    return curves, m


# ── figures ──────────────────────────────────────────────────────────────────
PANELS = (("rate", "shift"), ("rate", "binomial"), ("score", "shift"))
PANEL_TITLES = {("rate", "shift"): "rate, shift model", ("rate", "binomial"):
                "rate, binomial model (sensitivity)", ("score", "shift"): "score, shift model"}


def n_text(v: float) -> str:
    """Min n for display: '> 150' off the grid, '≤ 20' at its floor."""
    if not np.isfinite(v):
        return f"> {NS[-1]}"
    return f"≤ {NS[0]}" if int(v) <= NS[0] else f"{int(v)}"


def plot_experiment(exp: str, curves: pd.DataFrame, mins: pd.DataFrame, ds: Sequence[dict],
                    odor: pd.DataFrame, folder: Path) -> Path:
    c = curves[curves["experiment"] == exp]
    dnames = list(dict.fromkeys(c["design"]))
    fig, axes = pl.new_figure(len(dnames), len(PANELS),
                              figsize=(14.5, 3.0 * len(dnames) + 2.0), squeeze=False)
    plotted = []
    for i, dn in enumerate(dnames):
        for j, (metric, model) in enumerate(PANELS):
            ax = axes[i, j]
            t = c[(c["design"] == dn) & (c["metric"] == metric)
                  & (c["effect_model"] == model)].copy()
            if t.empty:
                ax.set_visible(False)
                continue
            if metric == "rate":
                t["delta"] = (t["delta"] * 100).round(6)
            p = pl.power_plot(ax, t, target=TARGET,
                              delta_label="Δ rate (pp)" if metric == "rate" else "Δ score",
                              n_label="Flies per group" if exp == "exp1" else "Flies")
            mm = mins[(mins["experiment"] == exp) & (mins["design"] == dn)
                      & (mins["metric"] == metric) & (mins["effect_model"] == model)]
            note = ", ".join(f"Δ {r.delta_display:g}: n {n_text(r.min_n)}"
                             for r in mm.itertuples())
            ax.set_title(f"{dn}\n{PANEL_TITLES[(metric, model)]}\n80 % power at {note}",
                         fontsize=7.5, loc="left", color=pl.INK_SECONDARY)
            for txt in ax.texts:            # the target label: left end, where curves are low
                if txt.get_text().endswith("power"):
                    txt.set_x(0.01)
                    txt.set_horizontalalignment("left")
            if j:
                ax.set_ylabel("")
            plotted.append(p.assign(metric=metric, design=dn, effect_model=model))
    sub = [d for d in ds if d["experiment"] == exp]
    n_obs = max(int(d["observed"].get("n_flies", 0) or 0) for d in sub)
    e_trials = odor[odor["experiment"] == exp]
    kind = "two-group label permutation" if exp == "exp1" else "paired sign-flip"
    pl.title_block(fig, f"Power vs flies, {EXP_TITLES[exp]}",
                   subtitle=f"bootstrap from n = {n_obs} observed flies, {len(e_trials)} trials · "
                            f"test: {kind}, α = {ALPHA} · correction: none · n_sim = {N_SIM}, "
                            f"{N_PERM_INNER} permutations per simulated test")
    pl.caption(fig, "Each simulation resamples the observed per-fly values (centred), adds the "
                    "effect and runs the test; power = share of simulations with p ≤ 0.05. Shift "
                    "model: every fly + Δ (stats.power_curve). Binomial model: each fly + "
                    "Binomial(k, Δ)/k, k = trials on the shifted side (1 / 2 / 14), i.e. some "
                    "flies flip to responding; more realistic for rates when most flies never "
                    "respond. Rate Δ in percentage points. No stars: no hypothesis is tested.")
    base = f"power_{exp}"
    pl.save_figure(fig, pd.concat(plotted, ignore_index=True), folder / base)
    return folder / base


def _finding(exp: str, mins: pd.DataFrame) -> dict:
    m = mins[mins["experiment"] == exp]
    hl = m[m["highlight_15pp"]]
    per = "; ".join(
        f"{dn}: shift model n {n_text(g.loc[g.effect_model == 'shift', 'min_n'].iloc[0])}, "
        f"binomial model n {n_text(g.loc[g.effect_model == 'binomial', 'min_n'].iloc[0])}"
        for dn, g in hl.groupby("design", sort=False))
    rest = []
    for (dn, model), g in m[~m["highlight_15pp"]].groupby(["design", "effect_model"],
                                                          sort=False):
        rest.append(f"{dn} [{model}]: " + ", ".join(
            f"Δ {r.delta_display:g} {'pp' if r.metric == 'rate' else 'score'} → "
            f"{n_text(r.min_n)}" for r in g.itertuples()))
    obs = m.drop_duplicates(["design", "metric"])
    obs_txt = "; ".join(f"{r.design} ({r.metric}): observed {r.effect:+.2f} "
                        f"{'pp' if r.metric == 'rate' else 'score'}"
                        + (f" (95 % CI {r.ci_lo:+.2f} to {r.ci_hi:+.2f}, p = {r.p:.3g}, "
                           f"p_batch = {r.p_batch:.3g})" if np.isfinite(r.p) else "")
                        for r in obs.itertuples() if np.isfinite(r.effect))
    unit = "flies per group" if exp == "exp1" else "flies"
    text = (f"Power, {EXP_TITLES[exp]}. Min n ({unit}) for 80 % power at Δ rate 15 pp "
            f"(α = 0.05): {per}. The shift model (every fly + Δ) is optimistic when most "
            "flies never respond; the binomial model (some flies flip to responding) is the "
            f"more realistic planning number. Other Δ: {'; '.join(rest)}. Observed: {obs_txt}. "
            f"Bootstrap from the observed per-fly values, n_sim = {N_SIM}, {N_PERM_INNER} "
            f"permutations inside each simulated test, n = {NS[0]}..{NS[-1]} step 10.")
    nums = {"min_n_15pp": {f"{r.design} [{r.effect_model}]":
                           (float(r.min_n) if np.isfinite(r.min_n) else None)
                           for r in hl.itertuples()},
            "min_n": m[["design", "metric", "effect_model", "delta_display", "min_n"]]
            .to_dict("records"), "n_sim": N_SIM, "n_perm_inner": N_PERM_INNER}
    return X.finding(f"power_{exp}", text, nums, section=SECTION)


README = """# 06_power
Simulated power vs number of flies (`per_analysis.power.run()`), bootstrapping the observed per-fly values: Exp 1 exposed vs paraffin cohort (two groups, post − pre on the exposed odor), Exp 2 (X change − click change) and Exp 3 (IAA after X − after click) within fly, for one partner and for the mean of 7 partners.
Δ score 0.25 / 0.5, Δ rate 10 / 15 / 20 pp, α 0.05, n 20..150; n_sim and permutation counts are in every table row.
Two effect models for rates: `shift` (every fly + Δ, `stats.power_curve`, as specified; optimistic when most flies never respond) and `binomial` (each fly + Binomial(k, Δ)/k, a sensitivity check). `tables/power_min_n.csv`: smallest n with 80 % power per design, model and Δ (`highlight_15pp` marks Δ rate 15 pp), with each design's observed effect; `tables/power_curves.csv`: every curve point."""


def run(trials: pd.DataFrame, out: Path, *, tables: Path) -> list[dict]:
    out, tables = Path(out), Path(tables)
    folder = out / SUBDIR
    X.readme(folder, README)
    odor = X.odor_trials(trials)
    ds = designs(odor)
    if not ds:
        log.warning("power: no design has data")
        return []
    log.info("power: %d designs, n_sim %d, %d inner permutations", len(ds), N_SIM, N_PERM_INNER)
    curves, mins = run_designs(ds)
    X.save_table(mins, tables, "power_min_n")
    curves.to_csv(Path(tables) / "power_curves.csv", index=False)
    findings = []
    for exp in ("exp1", "exp2", "exp3"):
        if (curves["experiment"] == exp).any():
            plot_experiment(exp, curves, mins, ds, odor, folder)
            findings.append(_finding(exp, mins))
    return findings


__all__ = ["run", "designs", "exp1_values", "exp2_values", "exp3_values", "simulate",
           "min_n_table", "N_SIM", "N_PERM_INNER", "NS", "DELTAS"]
