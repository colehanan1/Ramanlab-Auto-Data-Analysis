"""Day vs evening sessions -> ``<out>/08_day_vs_evening/`` (the user's main question:
"is there a difference between the flies we do in the time window vs the evening, mostly
comparing the evening b2b to the normal b2b").

``window`` is the clock window of the batch folder's first pre-test stamp (config):
"day" 09:40-16:20, "evening" 16:40-21:20; folders starting in neither ("other") are left
out, logged. Every cohort with evening flies is compared, evening - day, on

1. naive responses (exp1 pre / exp2 alone) per odor and all odors pooled (paraffin /
   click left out of the pool), rate and score;
2. the change: Exp 2 per fly the mean over the 7 partners of (X pair - X alone) -
   (click pair - click alone), for round 1, round 2 and both rounds pooled; Exp 1 per fly
   post - pre per odor (the exposed odor marked) and pooled;
3. the pair-block (Exp 2) / post (Exp 1) responses themselves;
4. Exp 1 only: the light-probe positive-control rate. The Exp 2 probes are not compared
   (``probe_note`` says why: before the 2026-11-18 fix their table window was 0.025 s).

Tests (one row per comparison, evening vs day flies are independent groups):

* ``p_batch`` -- PRIMARY: whole-batch label permutation (``b2b_stats.batch_label_perm``):
  window labels permuted among batch folders; stars come from its Holm adjustment
  (``p_batch_holm``);
* ``p_session`` -- the same with whole sessions (folder without ``_rig_N``) as units: with
  one evening session it can never fall below 1 / (number of sessions);
* ``p_same_date`` -- batch permutation within calendar dates that hold both windows (the
  evening sessions all have a day session on the same date);
* ``p`` -- secondary: fly-level label permutation (``fly_stats``), ``p_holm`` from it.

With a single evening session (Exp 2: 2026-11-17; 3Oct) a window difference is confounded
with that session; every caption and finding says so. Models: ``responded ~ C(window) *
C(block) + C(odor) + start_time_c + position_in_block`` (+ cohort for Exp 1), GEE
clustered by fly and a Bayesian mixed logistic; start_time_c is nearly collinear with the
window (the window is a start-time bin), so each model is also fitted without it.
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
from scripts.analysis import b2b_stats as b2b  # noqa: E402

log = logging.getLogger(__name__)

SECTION = "daynight"
SUBDIR = "08_day_vs_evening"
WINDOWS = ("day", "evening")
ALL_ODORS = "All odors"
ROUNDS = ("round 1", "round 2", "pooled")
NAIVE = {"exp1": "pre", "exp2": "alone", "exp3": "alone"}
LATER = {"exp1": "post", "exp2": "pair", "exp3": "pair"}
COHORT_TAGS = {"IAA-PairFirst-Control-1-1": "exp2_pairfirst",
               "IAA-PairSecond-Control-1-1": "exp3_pairsecond",
               "IAA-Sensitivity-1-1": "exp1_iaa_sensitivity",
               "Hex-Sensitivity-1-0.1": "exp1_hex_sensitivity",
               "3Oct-Sensitivity-1-0.1": "exp1_3oct_sensitivity",
               "Paraffin-Sensitivity-1-100": "exp1_paraffin_sensitivity"}
WINDOW_LABELS = {"day": "day (start 9:40–16:20)", "evening": "evening (start 16:40–21:20)"}
TEST = ("evening - day, difference in fly means; p_batch (primary): whole-batch label "
        "permutation; p_session: whole-session permutation; p_same_date: batch permutation "
        "within dates holding both windows; p: fly-level label permutation (secondary); CI: "
        "BCa bootstrap of flies")
CORRECTION = "Holm over the groups (odors / rounds) within cohort x measure x metric"
STAR_CAPTION = ("stars: Holm-adjusted whole-batch permutation p < 0.05 (*), < 0.01 (**), "
                "< 0.001 (***); no star = not significant after correction")


# ── windows ──────────────────────────────────────────────────────────────────
def in_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of the day and evening windows; 'other' flies left out, logged."""
    keep = df["window"].isin(WINDOWS)
    if (~keep).any():
        other = df.loc[~keep].drop_duplicates("fly_id")
        log.info("day vs evening: %d fly(ies) outside both windows left out: %s", len(other),
                 other.groupby(["cohort", "batch"]).size().to_dict())
    return df[keep]


def window_counts(odor: pd.DataFrame) -> pd.DataFrame:
    """Per cohort x window: flies, batch folders, sessions, their names and dates."""
    d = in_windows(odor)
    if "batch_key" not in d.columns:
        d = d.assign(batch_key=X.batch_keys(d))
    rows = []
    for (cohort, window), g in d.groupby(["cohort", "window"], sort=False):
        f = g.drop_duplicates("fly_id")
        sess = sorted(set(f["session"].astype(str)))
        rows.append({"cohort": cohort, "experiment": f["experiment"].iloc[0], "window": window,
                     "n_flies": int(len(f)), "n_batches": int(f["batch_key"].nunique()),
                     "n_sessions": len(sess), "sessions": "; ".join(sess),
                     "dates": "; ".join(sorted({str(pd.Timestamp(x).date())
                                                for x in f["date"]}))})
    order = {c: i for i, c in enumerate(C.COHORTS)}
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["cohort", "window"], key=lambda s: s.map(order) if s.name == "cohort"
                           else s).reset_index(drop=True)


def confound_note(counts: pd.DataFrame, cohort: str) -> str:
    ev = counts[(counts["cohort"] == cohort) & (counts["window"] == "evening")]
    if ev.empty:
        return ""
    r = ev.iloc[0]
    if int(r["n_sessions"]) == 1:
        return (f"Only one evening session ({r['sessions']}, {r['dates']}): an evening - day "
                "difference is confounded with that single session (its date, rigs and fly "
                "vials) and cannot be attributed to the clock window.")
    return (f"Evening flies come from {int(r['n_sessions'])} evening sessions "
            f"({r['dates']}); a window difference is still partly confounded with those dates.")


def counts_text(counts: pd.DataFrame, cohort: str) -> str:
    c = counts[counts["cohort"] == cohort].set_index("window")
    return "; ".join(f"{w}: {_n(c.loc[w, 'n_flies'], 'fly', 'flies')} / "
                     f"{_n(c.loc[w, 'n_batches'], 'batch', 'batches')} / "
                     f"{_n(c.loc[w, 'n_sessions'], 'session', 'sessions')}"
                     for w in WINDOWS if w in c.index)


def _n(k, one: str, many: str) -> str:
    k = int(k)
    return f"{k} {one if k == 1 else many}"


# ── the comparison ───────────────────────────────────────────────────────────
def compare_windows(pf: pd.DataFrame, *, scale: float = 1.0) -> dict:
    """Evening vs day on one value per fly (``pf``: fly_id, value, window, batch_key,
    session, date). Empty dict when a window has no fly."""
    pf = pf.dropna(subset=["value"])
    day = pf[pf["window"] == "day"].set_index("fly_id")["value"].astype(float)
    eve = pf[pf["window"] == "evening"].set_index("fly_id")["value"].astype(float)
    if day.empty or eve.empty:
        return {}
    batch_of = dict(zip(pf["fly_id"], pf["batch_key"]))
    r = X.two_group(day, eve, batch_of, None, scale=scale)
    vals = pf["value"].to_numpy(float)
    skey = (pf["session"].astype(str) + "|" + pf["window"].astype(str)).to_numpy(object)
    ps = b2b.batch_label_perm(vals, skey, dict(zip(skey, pf["window"])))
    labels = dict(zip(pf["batch_key"], pf["window"]))
    date_of = dict(zip(pf["batch_key"], pf["date"].astype(str)))
    both = {d for d in set(date_of.values())
            if len({labels[b] for b, dd in date_of.items() if dd == d}) > 1}
    p_date, n_dates = np.nan, len(both)
    if both:
        p_date = b2b.batch_label_perm(vals, pf["batch_key"].to_numpy(object), labels,
                                      date_of)["p"]
    sess = pf.drop_duplicates("fly_id").assign(sk=skey[~pf.duplicated("fly_id").to_numpy()])
    return {"n_flies": r["n_flies"], "n_flies_day": r["n_flies_a"],
            "n_flies_evening": r["n_flies_b"], "n_batches": r["n_batches"],
            "n_batches_day": r["n_batches_a"], "n_batches_evening": r["n_batches_b"],
            "n_sessions_day": int(sess.loc[sess["window"] == "day", "sk"].nunique()),
            "n_sessions_evening": int(sess.loc[sess["window"] == "evening", "sk"].nunique()),
            "mean_day": r["mean_a"], "mean_evening": r["mean_b"], "effect": r["effect"],
            "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p": r["p"], "p_batch": r["p_batch"],
            "p_session": float(ps["p"]), "p_session_exact": bool(ps["exact"]),
            "n_session_arrangements": int(ps["n_perm"]), "p_same_date": float(p_date),
            "n_dates_both_windows": n_dates}


# ── per-fly values per measure ───────────────────────────────────────────────
def _info(sub: pd.DataFrame) -> pd.DataFrame:
    cols = ["window", "batch_key", "batch", "session", "date", "cohort", "experiment"]
    return sub.drop_duplicates("fly_id").set_index("fly_id")[cols]


def _by_odor(trials: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per fly x odor (``group`` = odor) plus the all-odors pool (no paraffin / click)."""
    pf = X.fly_values(trials, metric, by=["odor"]).assign(group=lambda d: d["odor"])
    pool = trials[trials["odor"] != C.CLICK_ODOR]
    pp = X.fly_values(pool, metric).assign(odor=ALL_ODORS, group=ALL_ODORS)
    return pd.concat([pf, pp], ignore_index=True)


def exp2_controlled_change(sub: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per fly and round: mean over the partners of (X pair - X alone) - (click pair -
    click alone). Round 1 / 2 use that presentation; 'pooled' both."""
    d = sub[sub["experiment"] == "exp2"]
    alone = X.fly_values(d[d["block"] == "alone"], metric, by=["odor"]).pivot_table(
        index="fly_id", columns="odor", values="value", aggfunc="first")
    pair = d[d["block"] == "pair"]
    rows = []
    for rnd in ROUNDS:
        p = pair if rnd == "pooled" else pair[pair["presentation"] == float(rnd[-1])]
        w = X.fly_values(p, metric, by=["odor"]).pivot_table(index="fly_id", columns="odor",
                                                             values="value", aggfunc="first")
        ch = w.sub(alone.reindex(index=w.index, columns=w.columns))
        if C.CLICK_ODOR not in ch.columns:
            continue
        did = ch[[x for x in X.PARTNERS if x in ch.columns]].sub(ch[C.CLICK_ODOR], axis=0)
        v = did.mean(axis=1, skipna=True)
        rows.append(pd.DataFrame({"fly_id": v.index, "round": rnd, "value": v.to_numpy(float),
                                  "n_partners": did.notna().sum(axis=1).to_numpy()}))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["fly_id", "round", "value", "n_partners"])
    out = out.dropna(subset=["value"]).join(_info(d), on="fly_id")
    return out.assign(odor="mean of 7 partners", group=out["round"])


def exp1_change(sub: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per fly x odor: post - pre; ``All odors``: the per-fly mean change over the
    non-paraffin odors."""
    d = sub[(sub["experiment"] == "exp1") & sub["block"].isin(["pre", "post"])]
    pf = X.fly_values(d, metric, by=["block", "odor"])
    w = pf.pivot_table(index=["fly_id", "odor"], columns="block", values="value", aggfunc="first")
    ch = (w["post"] - w["pre"]).dropna().rename("value").reset_index()
    pool = ch[ch["odor"] != C.CLICK_ODOR].groupby("fly_id")["value"].mean().reset_index() \
        .assign(odor=ALL_ODORS)
    out = pd.concat([ch, pool], ignore_index=True).join(_info(d), on="fly_id")
    return out.assign(group=out["odor"])


def measures(sub: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame, pd.DataFrame | None]]:
    """(measure, metric, per-fly values with ``group``, trial rows with ``group`` for the
    pooled bars or None for a per-fly contrast) for one cohort's odor rows."""
    exp = sub["experiment"].iloc[0]
    out = []
    for measure, block in (("naive", NAIVE[exp]), (LATER[exp], LATER[exp])):
        tr = sub[sub["block"] == block]
        bars = pd.concat([tr.assign(group=tr["odor"]),
                          tr[tr["odor"] != C.CLICK_ODOR].assign(group=ALL_ODORS)],
                         ignore_index=True)
        for metric in X.METRICS:
            out.append((measure, metric, _by_odor(tr, metric), bars))
    for metric in X.METRICS:
        ch = exp2_controlled_change(sub, metric) if exp == "exp2" else exp1_change(sub, metric)
        out.append(("change", metric, ch, None))
    return out


def window_table(odor: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Every evening-vs-day comparison of one cohort (odor rows; light probes apart)."""
    sub = in_windows(odor[odor["cohort"] == cohort])
    if "batch_key" not in sub.columns:
        sub = sub.assign(batch_key=X.batch_keys(sub))
    exposed = str(sub["exposure_odor"].dropna().iloc[0]) if sub["exposure_odor"].notna().any() \
        else ""
    rows = []
    for measure, metric, pf, _ in measures(sub):
        for group, g in pf.groupby("group", sort=False):
            r = compare_windows(g, scale=X.SCALE[metric])
            if not r:
                continue
            rows.append({"cohort": cohort, "experiment": sub["experiment"].iloc[0],
                         "measure": measure, "metric": metric, "group": group,
                         "odor": g["odor"].iloc[0],
                         "exposed": bool(exposed) and g["odor"].iloc[0] == exposed,
                         "unit": X.UNIT[metric], **r, "test": TEST})
    t = pd.DataFrame(rows)
    return X.finish(t, ["measure", "metric"], correction=CORRECTION) if len(t) else t


def probe_note(trials: pd.DataFrame, cohort: str) -> str:
    """Why a pair cohort's light probes are not in its evening-vs-day comparison, from the table."""
    p = trials[(trials["cohort"] == cohort) & trials["is_light_probe"].astype(bool)]
    if p.empty:
        return "This cohort has no light probe in the trial table."
    w = pd.to_numeric(p["odor_window_s"], errors="coerce")
    short = int((w < 1.0).sum())
    if short:
        return (f"Its light probes are not used: {short} of {len(p)} have a window under 1 s in "
                "the trial table (fixed in code 2026-11-18: the window is the light, from the "
                "sensors log; it reaches the table on the next full pipeline run).")
    n = p[p["window"].isin(WINDOWS)].groupby("window")["fly_id"].size()
    by = ", ".join(f"{k} {int(n.get(k, 0))}" for k in WINDOWS)
    return f"Its light probes (pair sessions since 2026-11-17; {by}) are not compared between windows."


def light_table(trials: pd.DataFrame, cohort: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exp 1 light-probe rate, evening vs day (valid probes: window >= 1 s)."""
    lt = trials[(trials["cohort"] == cohort) & trials["is_light_probe"].astype(bool)
                & (pd.to_numeric(trials["odor_window_s"], errors="coerce") >= 1.0)]
    lt = in_windows(lt).assign(batch_key=lambda d: X.batch_keys(d), group="light probe",
                               odor="light probe")
    if lt.empty:
        return pd.DataFrame(), lt
    pf = X.fly_values(lt, "rate")
    r = compare_windows(pf.join(_info(lt)[["date"]], on="fly_id")
                        if "date" not in pf.columns else pf, scale=100.0)
    t = pd.DataFrame([{"cohort": cohort, "experiment": "exp1", "measure": "light probe",
                       "metric": "rate", "group": "light probe", "odor": "light probe",
                       "exposed": False, "unit": "pp", **r, "test": TEST}]) if r else pd.DataFrame()
    return (X.finish(t, None, correction="none (one comparison)") if len(t) else t), lt


# ── figures ──────────────────────────────────────────────────────────────────
def _group_order(groups: Sequence[str]) -> list[str]:
    g = list(dict.fromkeys(groups))
    if any(x in ROUNDS for x in g):
        return [r for r in ROUNDS if r in g]
    return pl.odor_order([x for x in g if x not in (ALL_ODORS, "light probe")]) + \
        [x for x in (ALL_ODORS, "light probe") if x in g]


def _group_labels(groups: Sequence[str], exposed: str, exp: str) -> dict:
    lab = {}
    for g in groups:
        s = g.replace("Ethyl Butyrate", "Ethyl\nButyrate").replace("Benzaldehyde", "Benz-\naldehyde")
        if g == C.CLICK_ODOR:
            s = "Paraffin\n(click)" if exp != "exp1" else "Paraffin\nOil"
        if g == ALL_ODORS:
            s = "All odors\n(no paraffin)"
        if g == exposed:
            s += "\n(exposed)"
        lab[g] = s
    return lab


def _change_bars(pf: pd.DataFrame, scale: float) -> pd.DataFrame:
    """Per-fly contrast -> bar table: mean over flies, +/- SEM over flies."""
    rows = []
    for (g, w), d in pf.groupby(["group", "window"], sort=False):
        v = d["value"].to_numpy(float) * scale
        m = float(v.mean())
        sem = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size >= 2 else np.nan
        rows.append({"group": g, "window": w, "metric": "change", "value": m,
                     "err_lo": m - sem, "err_hi": m + sem, "interval": "+/-SEM over flies",
                     "n_flies": int(v.size), "n_trials": np.nan})
    return pd.DataFrame(rows)


def _bar_table(measure: str, metric: str, pf: pd.DataFrame, bars: pd.DataFrame | None) -> pd.DataFrame:
    if bars is None:
        return _change_bars(pf, X.SCALE[metric])
    return st.group_summary(bars, ["group", "window"], metric)


def _ykey(exp: str) -> tuple:
    return (SECTION, exp)


def register_limits(tables_by_exp: Mapping[str, Sequence[tuple[str, str, pd.DataFrame]]]) -> None:
    """One y-scale per metric within an experiment (score, change_rate, change_score)."""
    for exp, items in tables_by_exp.items():
        groups: dict[str, list] = {}
        for measure, metric, bt in items:
            key = f"change_{metric}" if measure == "change" else metric
            groups.setdefault(key, []).extend(
                bt[["value", "err_lo", "err_hi"]].to_numpy(float).ravel().tolist())
        for key, vals in groups.items():
            if key == "rate":
                continue
            pl.register_ylim(key, _ykey(exp), data=vals, include_zero=True, overwrite=True)


MEASURE_TITLES = {"naive": "naive responses ({block} block)", "pair": "pair-block responses",
                  "post": "post-exposure responses", "change": "{change}",
                  "light probe": "light-probe positive control"}
CHANGE_DEFS = {"exp2": "Change: per fly, the mean over the 7 partners of (X pair − X alone) − "
                       "(click pair − click alone); round 1 / 2 use that presentation, pooled "
                       "both.",
               "exp1": "Change: per fly post − pre on each odor; All odors = the fly's mean "
                       "change over the non-paraffin odors."}


def _measure_title(measure: str, exp: str) -> str:
    return MEASURE_TITLES[measure].format(
        block=NAIVE[exp], change=("controlled change vs the click" if exp == "exp2"
                                  else "change post − pre"))


def _short(cohort: str, exp: str) -> str:
    return X.COHORT_LABELS.get(cohort, cohort).replace("\n", " ")


def plot_measure(cohort: str, exp: str, measure: str, items: Sequence[tuple[str, pd.DataFrame]],
                 table: pd.DataFrame, trials_n: tuple[int, int], counts: pd.DataFrame,
                 exposed: str, folder: Path, base: str) -> Path:
    """Day vs evening bars per group, one row per metric."""
    k = len(items)
    n_groups = max(bt["group"].nunique() for _, bt in items)
    width = 7.0 if n_groups == 1 else max(8.5, 0.95 * n_groups + 2.4)
    fig, axes = pl.new_figure(k, 1, figsize=(width, 3.3 * k + 2.3), squeeze=False)
    plotted = []
    for i, (metric, bt) in enumerate(items):
        ax = axes[i, 0]
        groups = _group_order(bt["group"])
        if measure == "change":
            m, ylabel = f"change_{metric}", ("Δ PER (pp, mean ± SEM over flies)" if metric == "rate"
                                             else "Δ PER score (mean ± SEM over flies)")
        else:
            m, ylabel = metric, None
        p = pl.paired_bars(ax, bt, conditions=list(WINDOWS), metric=m, groups=groups,
                           colors=C.WINDOW_COLORS, group_col="group", condition_col="window",
                           condition_labels=WINDOW_LABELS, hatches=pl.WINDOW_HATCHES,
                           group_labels=_group_labels(groups, exposed, exp),
                           experiment=_ykey(exp) if m != "rate" else None, ylabel=ylabel,
                           legend=i == 0)
        t = table[(table["measure"] == measure) & (table["metric"] == metric)]
        comps = t.assign(cond_a="day", cond_b="evening", p_holm=t["p_batch_holm"])[
            ["group", "cond_a", "cond_b", "p_holm"]]
        comps = comps[comps["group"].isin(set(p.loc[p["layer"] == "bar", "group"]))]
        pl.add_holm_stars(ax, comps, plotted=p, group_col="group", condition_col="window")
        plotted.append(p.assign(metric=metric))
    note = confound_note(counts, cohort)
    pl.title_block(fig, f"{_short(cohort, exp)}: evening vs day sessions, "
                        f"{_measure_title(measure, exp)}",
                   subtitle=f"n = {trials_n[0]} flies, {trials_n[1]} trials ({counts_text(counts, cohort)})"
                            f" · test: whole-batch label permutation (session-level and "
                            f"fly-level in the table) · correction: Holm over groups")
    extra = (CHANGE_DEFS[exp] + " " if measure == "change" else
             "The light probe is opto light without odor (Gr5a>CsChrimson). "
             if measure == "light probe" else "")
    shown = ("Bars: per-fly contrast, mean ± SEM over flies." if measure == "change" else
             "Bars: % of trials pooled over flies, Wilson 95 % CI with flies as unit."
             if measure == "light probe" else
             "Rates: % of trials pooled over flies, Wilson 95 % CI with flies as unit; scores: "
             "pooled mean ± SEM over flies.")
    pl.caption(fig, f"{STAR_CAPTION}. {note} {cohort}. {extra}{shown} Evening bars hatched.")
    pl.save_figure(fig, pd.concat(plotted, ignore_index=True), folder / base)
    return folder / base


# ── models ───────────────────────────────────────────────────────────────────
def fit_models(odor: pd.DataFrame, cohorts: Sequence[str], exp: str, folder: Path) -> dict:
    blocks = [NAIVE[exp], LATER[exp]]
    d = in_windows(odor[odor["cohort"].isin(list(cohorts)) & odor["block"].isin(blocks)])
    d = d.assign(start_time_c=X.centred(d["start_time_h"]))
    rhs = (f"C(window, Treatment('day')) * C(block, Treatment('{blocks[0]}')) + C(odor) "
           + ("+ C(cohort) " if d["cohort"].nunique() > 1 else "") + "+ position_in_block")
    out = {}
    note = ("start_time_c is nearly collinear with window (the window is a start-time bin), so "
            "in the '_with_start_time' fits the window term is a jump at the boundary net of a "
            "linear clock trend; the '_no_start_time' fits give the plain window contrast.")
    for tag, formula in (("with_start_time", f"responded ~ {rhs} + start_time_c"),
                         ("no_start_time", f"responded ~ {rhs}")):
        for kind, fit in (("gee", lambda f=formula: st.fit_gee_logistic(d, f)),
                          ("mixed_logistic", lambda f=formula: st.fit_mixed_logistic(d, f))):
            res, coefs, text = fit()
            full = note + (" " + X.MIXED_NOTE if kind == "mixed_logistic" else "")
            coefs = coefs.assign(note=coefs["note"].astype(str) + " | " + full)
            st.save_model((res, coefs, full + "\n\n" + text + "\n"),
                          folder / f"daynight_{exp}_{kind}_window_x_block_{tag}")
            out[(kind, tag)] = coefs
    return out


def _window_terms(coefs: pd.DataFrame) -> dict:
    t = coefs[coefs["term"].astype(str).str.contains("window", regex=False)]
    return {r.term: {"estimate": float(r.estimate), "ci_lo": float(r.ci_lo),
                     "ci_hi": float(r.ci_hi), "p": float(r.p), "status": str(r.status)}
            for r in t.itertuples()}


def _short_term(name: str) -> str:
    return (name.replace("C(window, Treatment('day'))[T.evening]", "evening")
            .replace("C(block, Treatment('alone'))[T.pair]", "pair block")
            .replace("C(block, Treatment('pre'))[T.post]", "post block"))


def _term_text(terms: Mapping) -> str:
    parts = []
    for name, v in terms.items():
        short = _short_term(name)
        parts.append(f"{short} {v['estimate']:+.2f} (95 % CI {v['ci_lo']:+.2f} to "
                     f"{v['ci_hi']:+.2f}, {X.fmt_p(v['p'])}, {v['status']})")
    return "; ".join(parts)


# ── findings ─────────────────────────────────────────────────────────────────
def _row(t: pd.DataFrame, measure: str, metric: str, group: str) -> pd.Series | None:
    r = t[(t["measure"] == measure) & (t["metric"] == metric) & (t["group"] == group)]
    return r.iloc[0] if len(r) else None


def _cmp_text(r: pd.Series | None, unit: str, level: str | None = None) -> str:
    if r is None:
        return "n/a"
    lv = level or unit
    return (f"day {r['mean_day']:.1f} {lv} vs evening {r['mean_evening']:.1f} {lv}, evening − day "
            f"{X.fmt_effect(r, unit)}, whole-batch {X.fmt_p(r['p_batch'])} (Holm "
            f"{X.fmt_p(r['p_batch_holm'])}), session-level {X.fmt_p(r['p_session'])}, same-date "
            f"{X.fmt_p(r['p_same_date'])}, fly-level {X.fmt_p(r['p'])}")


def _sig_groups(t: pd.DataFrame, measure: str, metric: str) -> str:
    s = t[(t["measure"] == measure) & (t["metric"] == metric) & (t["p_batch_holm"] < 0.05)]
    return ", ".join(f"{g} ({e:+.1f})" for g, e in zip(s["group"], s["effect"])) or "none"


def cohort_finding(cohort: str, t: pd.DataFrame, light: pd.DataFrame, counts: pd.DataFrame,
                   models: Mapping, *, primary: bool, probe_text: str = "") -> dict:
    exp = t["experiment"].iloc[0]
    later = LATER[exp]
    chg_group = "pooled" if exp == "exp2" else ALL_ODORS
    note = confound_note(counts, cohort)
    parts = [f"{X.EXP_LABELS[exp]} {cohort}, evening vs day ({counts_text(counts, cohort)})."]
    lab = ("controlled change (mean over 7 partners of (X pair − X alone) − (click pair − click "
           "alone)), rounds pooled" if exp == "exp2" else "change post − pre, all odors pooled")
    parts.append(f"{lab[0].upper() + lab[1:]}: rate {_cmp_text(_row(t, 'change', 'rate', chg_group), 'pp')}.")
    if exp == "exp2":
        for rnd in ("round 1", "round 2"):
            r = _row(t, "change", "rate", rnd)
            if r is not None:
                parts.append(f"{rnd}: {X.fmt_effect(r, 'pp')}, whole-batch "
                             f"{X.fmt_p(r['p_batch'])}.")
    else:
        ex = t[(t["measure"] == "change") & (t["metric"] == "rate") & t["exposed"]]
        if len(ex):
            parts.append(f"Exposed odor ({ex.iloc[0]['group']}) change: "
                         f"{_cmp_text(ex.iloc[0], 'pp')}.")
    parts.append(f"Naive responses, all odors pooled: rate "
                 f"{_cmp_text(_row(t, 'naive', 'rate', ALL_ODORS), 'pp', '%')}; score "
                 f"{X.fmt_effect(_row(t, 'naive', 'score', ALL_ODORS), 'points', digits=2)}.")
    parts.append(f"{'Pair-block' if exp == 'exp2' else 'Post'} responses, all odors pooled: rate "
                 f"{X.fmt_effect(_row(t, later, 'rate', ALL_ODORS), 'pp')}, whole-batch "
                 f"{X.fmt_p(_row(t, later, 'rate', ALL_ODORS)['p_batch'])}.")
    parts.append("Groups with a Holm-significant whole-batch window difference (rate): naive "
                 f"{_sig_groups(t, 'naive', 'rate')}; {later} {_sig_groups(t, later, 'rate')}; "
                 f"change {_sig_groups(t, 'change', 'rate')}.")
    if exp == "exp1":
        if len(light):
            r = light.iloc[0]
            parts.append(f"Light-probe positive control: {_cmp_text(r, 'pp', '%')}.")
    else:
        parts.append(probe_text or "The Exp 2 light probes are not used.")
    gee = models.get(("gee", "no_start_time"))
    if gee is not None:
        pooled = (" + cohort, fitted on all Exp 1 cohorts with evening flies together"
                  if exp == "exp1" else "")
        parts.append(f"GEE (responded ~ window x block + odor + position{pooled}, fly clusters; "
                     f"no start time): {_term_text(_window_terms(gee))}.")
    gee_t = models.get(("gee", "with_start_time"))
    if gee_t is not None:
        parts.append(f"With centred start time: {_term_text(_window_terms(gee_t))}.")
    mix = models.get(("mixed_logistic", "no_start_time"))
    if gee is not None and mix is not None:
        gw, mw = _window_terms(gee), _window_terms(mix)
        checks = [f"{_short_term(t)} {X.sign_check(gw[t], mw[t])}" for t in gw if t in mw]
        parts.append("Bayesian mixed logistic (sensitivity only; its SDs are too narrow to test "
                     "with) vs GEE, sign of the window terms: " + "; ".join(checks) + ".")
    parts.append(note)
    nums = {"counts": counts[counts["cohort"] == cohort][["window", "n_flies", "n_batches",
                                                          "n_sessions", "sessions", "dates"]]
            .to_dict("records")}
    for key, (m, met, g) in {"change": ("change", "rate", chg_group),
                             "naive_all": ("naive", "rate", ALL_ODORS),
                             "later_all": (later, "rate", ALL_ODORS)}.items():
        r = _row(t, m, met, g)
        if r is not None:
            nums[key] = {k: float(r[k]) for k in ("mean_day", "mean_evening", "effect", "ci_lo",
                                                  "ci_hi", "p", "p_batch", "p_batch_holm",
                                                  "p_session", "p_same_date")}
    return X.finding(f"daynight_{cohort}", " ".join(parts), nums, primary=primary,
                     section=SECTION)


# ── run ──────────────────────────────────────────────────────────────────────
README = """# 08_day_vs_evening
Evening vs day sessions (window of the batch folder's first pre-test stamp: day 9:40–16:20, evening 16:40–21:20; "other" left out), per cohort with evening flies, from `per_analysis.daynight.run()`: naive responses, the change (Exp 2: (X pair − X alone) − (click pair − click alone) over 7 partners; Exp 1: post − pre), the pair/post responses and (Exp 1) the light probe.
Primary test: whole-batch label permutation (`p_batch`, stars from its Holm adjustment); also session-level, same-date and fly-level p in `tables/daynight_*.csv`. With one evening session (Exp 2, 3Oct) a window difference is confounded with that session.
Day gold, evening plum + hatch (validated palette, see `config.WINDOW_COLORS`). Models in `models/`."""

COHORT_README = """# {tag}
{cohort} ({exp}), evening vs day: {counts}.
Figures `daynight_{tag}_<measure>` (PNG/SVG/CSV of the plotted numbers) and tables `tables/daynight_{tag}_<measure>.csv` (evening − day, CI, fly/batch/session/same-date p, Holm).
{note}"""


def run(trials: pd.DataFrame, out: Path, *, tables: Path) -> list[dict]:
    out, tables = Path(out), Path(tables)
    root = out / SUBDIR
    X.readme(root, README)
    X.readme(root / "models", "# models\nDay-vs-evening models via `stats.save_model`: GEE "
             "(fly clusters) and Bayesian mixed logistic (fit_vb) of responded ~ window x block "
             "+ odor + position (+ cohort for Exp 1), each with and without centred start time "
             "(nearly collinear with the window). Check `status`.")
    odor = X.odor_trials(trials)
    counts = window_counts(odor)
    X.save_table(counts, tables, "daynight_window_counts")
    eve = counts[counts["window"] == "evening"]
    with_eve = [c for c in C.COHORTS if c in set(eve["cohort"])]
    findings: list[dict] = []
    for c in [c for c in C.COHORTS if c in set(odor["cohort"]) and c not in with_eve]:
        findings.append(X.finding(f"daynight_{c}",
                                  f"{X.EXP_LABELS[C.COHORTS[c]]} {c}: no evening flies, so no "
                                  "day vs evening comparison.", {"n_evening_flies": 0},
                                  section=SECTION))
    # stats + bar tables first, so each experiment's y-scales are fixed before drawing
    plan, by_exp = [], {}
    for cohort in with_eve:
        exp = C.COHORTS[cohort]
        sub = in_windows(odor[odor["cohort"] == cohort])
        t = window_table(odor, cohort)
        lt_tab, lt_rows = light_table(trials, cohort) if exp == "exp1" else (pd.DataFrame(), None)
        items = [(m, met, _bar_table(m, met, pf, bars)) for m, met, pf, bars in measures(sub)]
        if lt_rows is not None and len(lt_rows):
            items.append(("light probe", "rate",
                          st.group_summary(lt_rows, ["group", "window"], "rate")))
        by_exp.setdefault(exp, []).extend(items)
        plan.append((cohort, exp, sub, t, lt_tab, lt_rows, items))
    register_limits(by_exp)
    models_by_exp = {exp: fit_models(odor, [p[0] for p in plan if p[1] == exp], exp,
                                     root / "models") for exp in by_exp}
    for cohort, exp, sub, t, lt_tab, lt_rows, items in plan:
        tag = COHORT_TAGS.get(cohort, cohort.lower())
        folder = root / tag
        X.readme(folder, COHORT_README.format(tag=tag, cohort=cohort, exp=X.EXP_LABELS[exp],
                                              counts=counts_text(counts, cohort),
                                              note=confound_note(counts, cohort)))
        exposed = str(sub["exposure_odor"].dropna().iloc[0]) if sub["exposure_odor"].notna().any() \
            else ""
        table_all = pd.concat([t, lt_tab], ignore_index=True) if len(lt_tab) else t
        for measure in dict.fromkeys(m for m, _, _ in items):
            its = [(met, bt) for m, met, bt in items if m == measure]
            rows = (lt_rows if measure == "light probe" else
                    sub[sub["block"] == (NAIVE[exp] if measure == "naive" else LATER[exp])]
                    if measure != "change" else
                    sub[sub["block"].isin([NAIVE[exp], LATER[exp]])])
            base = f"daynight_{tag}_{measure.replace(' ', '_')}"
            tm = table_all[table_all["measure"] == measure]
            plot_measure(cohort, exp, measure, its, table_all, (int(rows["fly_id"].nunique()),
                                                                int(len(rows))),
                         counts, exposed, folder, base)
            X.save_table(tm, tables, base)
        findings.append(cohort_finding(
            cohort, t, lt_tab, counts, models_by_exp[exp], primary=exp == "exp2",
            probe_text="" if exp == "exp1" else probe_note(trials, cohort)))
    if "exp3" not in {C.COHORTS[c] for c in with_eve} and any(
            C.COHORTS.get(c) == "exp3" for c in set(odor["cohort"])):
        log.info("day vs evening: Exp 3 has no evening flies")
    return findings


__all__ = ["run", "window_counts", "window_table", "compare_windows", "confound_note",
           "exp2_controlled_change", "exp1_change", "ALL_ODORS"]
