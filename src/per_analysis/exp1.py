"""Exp 1: the "Sensitivity" exposure cohorts.

Protocol: a pre-test (8 test odors, 30 s each), an exposure block (6 x 30 s pulses of the
cohort's odor), then a post-test with the same odors. Cohorts (``config.COHORTS``):
IAA-Sensitivity-1-1, Hex-Sensitivity-1-0.1, 3Oct-Sensitivity-1-0.1 and the control,
Paraffin-Sensitivity-1-100. The control's "exposure" delivered no odor: its pulses shut
the paraffin carrier off for 30 s (``paraffin_off_protocol``) or, after the 2026-11-10
rig fix, were a 50 ms master-valve click (``is_click``).

Entry point: :func:`run` writes everything under ``<out>/02_exp1_exposure_cohorts/``:

a. ``a_pre_vs_post``: pre vs post per test odor x cohort, paired sign-flip per odor
   (``stats.paired_effect``, BCa CI), Holm across odors within each cohort panel.
b. ``b_did_vs_paraffin`` (PRIMARY): per test odor, DiD = (post - pre, exposed cohort) -
   (post - pre, Paraffin control), ``stats.did_bootstrap`` (fly-bootstrap CI, fly-label
   and whole-batch permutation p). The three pre-specified primary contrasts are each
   exposed cohort on its own exposure odor, Holm across those 3; every other row is
   exploratory, Holm within its cohort's panel. Models in ``models/``.
c. ``c_global_drift``: each fly's mean post - pre over the odors every fly ran; tested
   against 0 per cohort and between cohorts (fly and whole-batch permutation), and
   against the fly's start time (``b2b_stats.batch_trend``).
d. ``d_time_since_exposure``: the exposure odor's post response vs
   ``minutes_since_exposure_end``, against the Paraffin control on the same odor.
e. ``e_odor_specific_suppression``: per exposure odor, the self-exposed cohort's change
   vs the pooled change of every other cohort, Holm across odors.

Data rules: ``experiment == "exp1"``, light probes out, odor windows < 1 s out (logged);
the fly is the unit (within-fly means before any paired test); bars pool trials
(``stats.group_summary``); response = ``responded``. Every stats table lands in
``tables/<figure base name>.csv`` with n_flies, effect, ci_lo, ci_hi, p, p_holm, q_bh,
p_batch, test, correction (+ ``batch_disagrees``: fly- and batch-level p on opposite
sides of 0.05). Rate effects are in percentage points, score effects in score points.

Design facts this module relies on (2026-11-18 tidy table):

* every exp1 fly ran each test odor once per block, so a fly's change is post - pre of
  one trial each;
* the IAA cohort never ran Paraffin Oil; Hex lacks it for 22 of 50 flies, 3Oct for 35 of
  62. Paraffin rows use the flies that ran it, and the IAA cohort has none;
* Paraffin Oil is always the FIRST pre-test trial and the LAST post-test trial;
* the post-test odor order is shuffled per batch folder but shared by its flies, and
  ``minutes_since_exposure_end`` is the post-block position on a fixed clock (r =
  0.9999998 with ``position_in_block``), so time since exposure is a batch property;
* each cohort has one exposure concentration, so ``exposure_conc`` is aliased with the
  cohort term and cannot enter beside it (:func:`add_covariates` says so in each model).
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from per_analysis import config as cfg
from per_analysis import plots as pl
from per_analysis import stats as st
from scripts.analysis import b2b_stats as _b2b  # importable once stats put the repo on sys.path

log = logging.getLogger("per_analysis.exp1")

SECTION = "exp1"
ROOT_DIR = "02_exp1_exposure_cohorts"
SUBDIRS = {"a": "a_pre_vs_post", "b": "b_did_vs_paraffin", "c": "c_global_drift",
           "d": "d_time_since_exposure", "e": "e_odor_specific_suppression"}
METRICS = ("rate", "score")
METRIC_VALUE = {"rate": "responded", "score": "score"}
SCALE = {"rate": 100.0, "score": 1.0}
UNIT = {"rate": "percentage points", "score": "score points"}
UNIT_SHORT = {"rate": "pp", "score": "score pts"}
METRIC_WORD = {"rate": "response rate", "score": "PER score"}
#: Every stats table carries these columns.
TABLE_COLUMNS = ["n_flies", "effect", "ci_lo", "ci_hi", "p", "p_holm", "q_bh", "p_batch",
                 "test", "correction"]
CONTROL = cfg.CONTROL_COHORT
PARAFFIN = "Paraffin Oil"
#: The four exp2/3 click trials have a 0.025 s table window; any window this short is broken.
SHORT_WINDOW_S = 1.0
#: Offered to every model, in this order; :func:`add_covariates` keeps the identifiable ones.
COVARIATES = ("start_time_c", "position_in_block", "exposure_conc", "minutes_since_exposure_end")
#: A covariate whose R^2 on the terms already in the model exceeds this is not entered.
ALIAS_R2 = 0.999
#: Bins of minutes since the last exposure pulse (left-closed) for the binned plot.
TIME_BINS = (0.0, 10.0, 20.0, 30.0, 45.0)
MARKERS = ("o", "s", "^", "D", "v", "P")
ALPHA = 0.05
TREND_PERM_CAP = 20_000

PARAFFIN_CAVEAT = ("Paraffin Oil test trials deliver no odor (carrier shut off 30 s, or a 50 ms "
                   "master-valve click from 2026-11-10)")


# ═════════════════════════════════════════════════════════════════════════════
# cohorts and data
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Cohort:
    name: str            # dataset name
    code: str            # formula level: IAA / Hex / 3Oct / Paraffin
    odor: str            # exposure odor (canonical)
    conc: float          # exposure concentration
    label: str           # display label
    color: str
    marker: str
    is_control: bool


def cohort_info(d: pd.DataFrame) -> dict[str, Cohort]:
    """The exp1 cohorts present in ``d``, in ``config.COHORTS`` order (control last as
    configured), with their exposure odor, concentration, label, colour and marker."""
    present = set(d["cohort"].dropna())
    names = [c for c, e in cfg.COHORTS.items() if e == SECTION and c in present]
    names += sorted(present - set(names))
    colors = pl.odor_colors()
    out = {}
    for i, name in enumerate(names):
        sub = d[d["cohort"] == name]
        code = str(name).split("-")[0]
        odor = (sub["exposure_odor"].dropna().iloc[0]
                if "exposure_odor" in sub and sub["exposure_odor"].notna().any()
                else cfg.NAME_ODORS.get(code, code))
        conc = (float(sub["exposure_conc"].dropna().iloc[0])
                if "exposure_conc" in sub and sub["exposure_conc"].notna().any() else float("nan"))
        ctrl = name == CONTROL
        label = f"{code} {'control' if ctrl else 'exposed'} ({conc:g})"
        out[name] = Cohort(name, code, str(odor), conc, label,
                           colors.get(odor, pl.CATEGORICAL[i % len(pl.CATEGORICAL)]),
                           MARKERS[i % len(MARKERS)], ctrl)
    return out


def _exposed(info: Mapping[str, Cohort]) -> list[Cohort]:
    return [c for c in info.values() if not c.is_control]


def prepare(trials: pd.DataFrame) -> pd.DataFrame:
    """Exp 1 odor trials: ``experiment == "exp1"``, light probes out, odor windows
    < :data:`SHORT_WINDOW_S` out (logged). Adds ``start_time_c`` (start hour centred on
    the mean over flies), ``post`` (0/1) and ``grp`` (the cohort's formula code)."""
    d = trials[trials["experiment"] == SECTION]
    n_exp = len(d)
    light = d["is_light_probe"].astype(bool)
    d = d[~light]
    short = pd.to_numeric(d["odor_window_s"], errors="coerce") < SHORT_WINDOW_S
    log.info("exp1: leaving out %d trial(s) with odor_window_s < %g s (the known-bad 0.025 s "
             "click trials)%s", int(short.sum()), SHORT_WINDOW_S,
             "" if not short.any() else ": " + "; ".join(
                 f"{r.fly_id} {r.get('trial_label', r.odor)}" for _, r in d[short].iterrows()))
    d = d[~short].copy()
    log.debug("exp1 rows: %d of %d trials; %d light probes and %d short windows dropped; "
              "%d flies", len(d), len(trials), int(light.sum()), int(short.sum()),
              d["fly_id"].nunique())
    starts = d.drop_duplicates("fly_id")["start_time_h"].astype(float)
    d["start_time_c"] = d["start_time_h"].astype(float) - float(starts.mean())
    d["post"] = (d["block"] == "post").astype(int)
    d["responded"] = d["responded"].astype(bool)
    info = cohort_info(d)
    d["grp"] = d["cohort"].map({n: c.code for n, c in info.items()})
    if n_exp and not len(d):
        log.warning("exp1: no odor trials left after the data rules")
    return d.reset_index(drop=True)


def _test_rows(d: pd.DataFrame) -> pd.DataFrame:
    return d[d["block"].isin(["pre", "post"]) & d["odor"].notna()]


def common_odors(d: pd.DataFrame) -> list[str]:
    """Test odors every cohort ran in both blocks (Paraffin Oil drops out: the IAA cohort
    never ran it), in config order."""
    pp = _test_rows(d)
    sets = [set(sub["odor"]) for _, sub in pp.groupby("cohort")]
    shared = set.intersection(*sets) if sets else set()
    return pl.odor_order(sorted(shared))


def fly_changes(d: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per (cohort, odor, fly) with both blocks: the within-fly means ``pre`` and
    ``post`` of the metric (0/1 for rate), ``change`` = post - pre, ``n_trials``, and the
    fly's ``batch`` and ``start_time_h``. Flies lacking either block for an odor (e.g.
    no Paraffin Oil) drop out of that odor only."""
    value = METRIC_VALUE[metric]
    pp = _test_rows(d)
    pf = st.per_fly(pp, value, by=["cohort", "odor", "block"],
                    carry=("batch", "start_time_h"))
    keys = ["cohort", "odor", "fly_id"]
    pre = pf[pf["block"] == "pre"].drop(columns="block").rename(
        columns={value: "pre", "n_trials": "n_pre"})
    post = pf[pf["block"] == "post"][keys + [value, "n_trials"]].rename(
        columns={value: "post", "n_trials": "n_post"})
    ch = pre.merge(post, on=keys, how="inner")
    ch["change"] = ch["post"] - ch["pre"]
    ch["n_trials"] = ch["n_pre"] + ch["n_post"]
    unpaired = len(pre) + len(post) - 2 * len(ch)
    log.debug("fly_changes(%s): %d fly x odor pairs, %d unpaired fly x odor x block rows "
              "dropped", metric, len(ch), unpaired)
    return ch.reset_index(drop=True)


def _batch_of(d: pd.DataFrame) -> pd.Series:
    return d.drop_duplicates("fly_id").set_index("fly_id")["batch"]


def _paraffin_note(cohort: Cohort, n_tested: int, n_total: int) -> str:
    if n_tested == 0:
        return f"not tested: the {cohort.code} cohort never ran {PARAFFIN}"
    part = (f"{PARAFFIN}: {n_tested} of {n_total} {cohort.code} flies ran it (the rest never did)"
            if n_tested < n_total else f"{PARAFFIN}: all {n_total} {cohort.code} flies")
    return f"{part}; no odor delivered (carrier shut-off or 50 ms valve click)"


def _date_range(d: pd.DataFrame, cohort: str) -> str:
    if "date" not in d.columns:
        return ""
    dates = pd.to_datetime(d.loc[d["cohort"] == cohort, "date"], errors="coerce").dropna()
    if dates.empty:
        return ""
    a, b = dates.min(), dates.max()
    return f"{a:%b} {a.day}-{b:%b} {b.day}"


# ═════════════════════════════════════════════════════════════════════════════
# small helpers: covariates, corrections, tables, text
# ═════════════════════════════════════════════════════════════════════════════
def add_covariates(df: pd.DataFrame, core: str,
                   covariates: Sequence[str] = COVARIATES) -> tuple[str, list[str]]:
    """``core`` plus each covariate that is defined on every row of ``df``, varies, and
    is not (near-)aliased with the terms already in (R^2 > :data:`ALIAS_R2`). Returns
    the formula and one note per covariate left out, saying why."""
    import patsy
    rhs = core.split("~", 1)[1]
    X = np.asarray(patsy.dmatrix(rhs, df, NA_action="raise", return_type="matrix"), float)
    terms, notes = [], []
    for cov in covariates:
        if cov not in df.columns:
            notes.append(f"{cov} not entered: no such column")
            continue
        v = pd.to_numeric(df[cov], errors="coerce").to_numpy(float)
        bad = int(np.sum(~np.isfinite(v)))
        if bad:
            notes.append(f"{cov} not entered: undefined on {bad} of {v.size} rows "
                         "(defined on post-exposure rows only)" if cov ==
                         "minutes_since_exposure_end" else
                         f"{cov} not entered: undefined on {bad} of {v.size} rows")
            continue
        if np.ptp(v) == 0:
            notes.append(f"{cov} not entered: constant ({v[0]:g}) in this model's rows")
            continue
        beta, *_ = np.linalg.lstsq(X, v, rcond=None)
        resid = v - X @ beta
        r2 = 1.0 - float(resid.var()) / float(v.var())
        if r2 > ALIAS_R2:
            why = (" (one exposure concentration per cohort)" if cov == "exposure_conc" else
                   " (post-block trials run on a fixed clock: position = time since exposure)"
                   if cov == "position_in_block" else "")
            notes.append(f"{cov} not entered: aliased with the model terms, R^2 = {r2:.7f}{why}")
            continue
        terms.append(cov)
        X = np.column_stack([X, v])
    formula = core + "".join(f" + {t}" for t in terms)
    return formula, notes


def _corrected(table: pd.DataFrame, family_col: str | None = None) -> pd.DataFrame:
    """``stats.add_corrections`` over the rows with a finite p (rows without a test,
    e.g. 'not tested', stay out of every family)."""
    t = table.copy()
    for c in ("p_holm", "q_bh", "p_batch_holm"):
        t[c] = np.nan
    if t.empty:
        return t
    ok = pd.to_numeric(t["p"], errors="coerce").notna()
    if ok.any():
        done = st.add_corrections(t.loc[ok].drop(columns=["p_holm", "q_bh", "p_batch_holm"]),
                                  p_col="p", family_col=family_col)
        for c in ("p_holm", "q_bh", "p_batch_holm"):
            if c in done.columns:
                t.loc[done.index, c] = done[c]
    return t


def _disagrees(p: Any, p_batch: Any) -> Any:
    p, pb = float(p), float(p_batch)
    if not (np.isfinite(p) and np.isfinite(pb)):
        return np.nan
    return bool((p < ALPHA) != (pb < ALPHA))


def _write_table(table: pd.DataFrame, tables: Path, stem: str,
                 lead: Sequence[str] = ()) -> Path:
    t = table.copy()
    for c in TABLE_COLUMNS:
        if c not in t.columns:
            t[c] = np.nan
    if "batch_disagrees" not in t.columns:
        t["batch_disagrees"] = [_disagrees(a, b) for a, b in zip(t["p"], t["p_batch"])]
    first = [c for c in lead if c in t.columns]
    cols = first + [c for c in TABLE_COLUMNS if c not in first]
    cols += [c for c in t.columns if c not in cols]
    tables.mkdir(parents=True, exist_ok=True)
    path = tables / f"{stem}.csv"
    t[cols].to_csv(path, index=False)
    log.debug("table %s: %d rows", path, len(t))
    return path


def _p(p: Any) -> str:
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(p):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _p_eq(p: Any) -> str:
    s = _p(p)
    return s if s.startswith("<") or s == "n/a" else f"= {s}"


def _f(x: Any, digits: int = 1, sign: bool = True) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(x):
        return "n/a"
    return f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"


def _num(x: Any) -> Any:
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    return x


def _digits(metric: str) -> int:
    return 1 if metric == "rate" else 2


def _fragile(row: Mapping) -> str:
    flag = _disagrees(row.get("p", np.nan), row.get("p_batch", np.nan))
    return (" FRAGILE: fly-level p and batch-level p_batch fall on opposite sides of 0.05."
            if flag is True else "")


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", str(s)).strip("-")


def _readme(folder: Path, lines: Iterable[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "README.md").write_text("\n".join(lines).rstrip() + "\n")


def _save_model(result: st.ModelResult, stem: Path, notes: Sequence[str]) -> st.ModelResult:
    res, coefs, text = result
    if notes:
        text = "NOTES:\n" + "\n".join(f"- {n}" for n in notes) + "\n\n" + text
        coefs = coefs.assign(covariate_notes="; ".join(notes))
    status = set(coefs["status"]) if "status" in coefs else set()
    if status - {"ok"}:
        log.warning("model %s: status %s", stem.name, sorted(status))
    st.save_model((res, coefs, text), stem)
    log.debug("model %s: %d coefficient rows, status %s", stem.name, len(coefs), sorted(status))
    return res, coefs, text


# ═════════════════════════════════════════════════════════════════════════════
# a. pre vs post
# ═════════════════════════════════════════════════════════════════════════════
def pre_vs_post_table(d: pd.DataFrame, metric: str, *, n_boot: int | None = None,
                      n_perm: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Per cohort x test odor: within-fly post - pre (``stats.paired_comparisons``),
    BCa CI, exact sign-flip p and whole-batch sign-flip p_batch; Holm and BH across the
    odors of each cohort panel. Odors a cohort never ran get a 'not tested' row."""
    info = cohort_info(d)
    value, k = METRIC_VALUE[metric], SCALE[metric]
    pp = _test_rows(d)
    odors = pl.odor_order(pp["odor"].unique())
    ch = fly_changes(d, metric)
    trials_n = ch.groupby(["cohort", "odor"])["n_trials"].sum()
    parts = []
    for name, co in info.items():
        sub = pp[pp["cohort"] == name]
        n_total = sub["fly_id"].nunique()
        tab = st.paired_comparisons(sub, value, condition_col="block", a="pre", b="post",
                                    by=["odor"], batch_col="batch", n_boot=n_boot,
                                    n_perm=n_perm, seed=seed)
        rows = []
        for odor in odors:
            r = tab[tab["odor"] == odor]
            base = {"cohort": name, "cohort_label": co.label, "odor": odor,
                    "exposed_odor": odor == co.odor, "metric": metric, "unit": UNIT[metric],
                    "family": co.label, "note": ""}
            if odor == PARAFFIN:
                n_t = int(r["n_flies"].iloc[0]) if len(r) else 0
                base["note"] = _paraffin_note(co, n_t, n_total)
            if r.empty:
                rows.append({**base, "n_flies": 0, "n_batches": 0, "n_trials": 0})
                continue
            r = r.iloc[0]
            rows.append({**base, "n_flies": int(r["n_flies"]), "n_batches": int(r["n_batches"]),
                         "n_trials": int(trials_n.get((name, odor), 0)),
                         "pre_mean": r["mean_a"] * k, "post_mean": r["mean_b"] * k,
                         "effect": r["mean_diff"] * k, "ci_lo": r["ci_lo"] * k,
                         "ci_hi": r["ci_hi"] * k, "p": r["p"], "p_method": r["p_method"],
                         "p_batch": r["p_batch"], "p_batch_exact": r["p_batch_exact"]})
        parts.append(_corrected(pd.DataFrame(rows), family_col="family"))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out["test"] = ("paired sign-flip on within-fly post - pre (exact), BCa 95% CI over flies; "
                   "p_batch: whole-batch sign-flip")
    out["correction"] = "Holm (q_bh: BH) across odors within each cohort panel"
    out["batch_disagrees"] = [_disagrees(a, b) for a, b in zip(out["p"], out["p_batch"])]
    log.debug("pre_vs_post_table(%s): %d rows, %d tested", metric, len(out),
              int(out["p"].notna().sum()))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# b. difference in differences vs the Paraffin control
# ═════════════════════════════════════════════════════════════════════════════
def _did_row(x: pd.Series, c: pd.Series, batch_of: pd.Series, k: float, *,
             n_boot, n_perm, seed) -> dict:
    r = st.did_bootstrap(x, c, batch_of=batch_of, n_boot=n_boot, n_perm=n_perm, seed=seed)
    return {"n_flies_exposed": r["n_exposed"], "n_flies_control": r["n_control"],
            "n_flies": r["n_exposed"] + r["n_control"],
            "n_batches_exposed": r["n_batches_exposed"],
            "n_batches_control": r["n_batches_control"],
            "change_exposed": r["mean_exposed"] * k, "change_control": r["mean_control"] * k,
            "effect": r["estimate"] * k, "ci_lo": r["ci_lo"] * k, "ci_hi": r["ci_hi"] * k,
            "p": r["p"], "p_batch": r["p_batch"], "ci_batch_lo": r["ci_batch_lo"] * k,
            "ci_batch_hi": r["ci_batch_hi"] * k, "p_batch_exact": r["perm_exact_batch"]}


def did_vs_control_table(d: pd.DataFrame, metric: str, *, n_boot: int | None = None,
                         n_perm: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Per exposed cohort x test odor: DiD of the within-fly change against the Paraffin
    control (``stats.did_bootstrap``). ``primary`` marks each cohort's own exposure odor
    (family 'primary', Holm across those 3); the rest are 'exploratory: <cohort>', Holm
    within the cohort's panel. Paraffin Oil uses only flies that ran it; a cohort that
    never ran an odor gets a 'not tested' row outside every family."""
    info = cohort_info(d)
    if CONTROL not in info:
        raise ValueError(f"no {CONTROL} rows: the DiD needs the control cohort")
    k = SCALE[metric]
    ch = fly_changes(d, metric)
    batch_of = _batch_of(d)
    odors = pl.odor_order(ch["odor"].unique())
    ctrl = ch[ch["cohort"] == CONTROL]
    totals = _test_rows(d).groupby("cohort")["fly_id"].nunique()
    trials_n = ch.groupby(["cohort", "odor"])["n_trials"].sum()
    rows = []
    for co in _exposed(info):
        for odor in odors:
            x = ch[(ch["cohort"] == co.name) & (ch["odor"] == odor)].set_index("fly_id")["change"]
            c = ctrl[ctrl["odor"] == odor].set_index("fly_id")["change"]
            primary = odor == co.odor
            note = ""
            if odor == PARAFFIN:
                note = (_paraffin_note(co, len(x), int(totals.get(co.name, 0)))
                        + f"; control {len(c)} of {int(totals.get(CONTROL, 0))} flies")
            base = {"cohort": co.name, "cohort_label": co.label, "odor": odor,
                    "primary": primary,
                    "family": "primary" if primary else f"exploratory: {co.label}",
                    "metric": metric, "unit": UNIT[metric], "note": note,
                    "n_trials": int(trials_n.get((co.name, odor), 0)
                                    + trials_n.get((CONTROL, odor), 0)) if len(x) else 0}
            if x.empty or c.empty:
                rows.append({**base, "n_flies_exposed": len(x), "n_flies_control": len(c),
                             "n_flies": len(x) + len(c)})
                continue
            rows.append({**base, **_did_row(x, c, batch_of, k, n_boot=n_boot, n_perm=n_perm,
                                            seed=seed)})
    out = _corrected(pd.DataFrame(rows), family_col="family")
    out["test"] = ("DiD of within-fly post - pre vs Paraffin control: stratified fly-bootstrap "
                   "BCa 95% CI, fly-label permutation p; p_batch: whole-batch permutation")
    out["correction"] = np.where(out["primary"], "Holm across the 3 pre-specified primary "
                                 "contrasts", "Holm within the cohort's panel (exploratory)")
    out["batch_disagrees"] = [_disagrees(a, b) for a, b in zip(out["p"], out["p_batch"])]
    log.debug("did_vs_control_table(%s): %d rows", metric, len(out))
    return out


def _contrast(coefs: pd.DataFrame, term: str) -> dict:
    row = coefs[coefs["term"] == term]
    if row.empty:
        status = coefs["status"].iloc[0] if len(coefs) else "failed"
        return {"term": term, "effect": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan,
                "status": status, "model_note": f"term not in the fit ({status})"}
    r = row.iloc[0]
    return {"term": term, "effect": r["estimate"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
            "p": r["p"], "status": r["status"], "model_note": r.get("note", ""),
            "n_obs": r["n_obs"], "n_groups": r["n_groups"]}


def did_models(d: pd.DataFrame, models_dir: Path, *,
               did_tables: Mapping[str, pd.DataFrame] | None = None,
               ordinal: bool = True, ordinal_boot: int = 200,
               seed: int | None = None) -> pd.DataFrame:
    """The DiD as models, every fit saved through ``stats.save_model``:

    * full ``responded ~ post * cohort * odor`` + covariates on the odors every cohort ran
      (Paraffin Oil cannot enter: the IAA cohort never ran it), as GEE logistic (fly
      clusters) and Bayesian mixed logistic (fly random intercept). It is refitted with
      each primary odor as the odor reference, so ``post:cohort[X]`` IS that primary
      contrast (log odds ratio);
    * Paraffin Oil on its own, ``post * cohort`` (exploratory), both kinds;
    * per primary odor, an ordinal (proportional-odds) model of the score with a
      fly-cluster bootstrap (``ordinal_boot`` replicates), when ``ordinal``.

    Returns one row per contrast and model, Holm across the 3 primaries per model kind;
    ``p_batch`` is the whole-batch permutation p of the same DiD from ``did_tables``."""
    info = cohort_info(d)
    ref = info[CONTROL].code
    pp = _test_rows(d)
    shared = common_odors(d)
    full = pp[pp["odor"].isin(shared)]
    primaries = [co for co in _exposed(info) if co.odor in shared]
    did_tables = did_tables or {}
    labels = {"gee": "GEE logistic (exchangeable within fly, robust SE)",
              "glmm": "Bayesian mixed logistic (fly random intercept; VB posterior, "
                      "normal-approximation p)",
              "gee_odor": "GEE logistic on this odor only, post x cohort (exchangeable within "
                          "fly, robust SE)",
              "glmm_odor": "Bayesian mixed logistic on this odor only, post x cohort (fly "
                           "random intercept; VB posterior)",
              "ordinal": f"ordinal logit of score (fly-cluster bootstrap SE, "
                         f"n_boot={ordinal_boot})"}
    fits = {"gee": st.fit_gee_logistic, "glmm": st.fit_mixed_logistic}
    rows = []

    def p_batch_of(cohort: str, odor: str, metric: str) -> float:
        t = did_tables.get(metric)
        if t is None:
            return float("nan")
        r = t[(t["cohort"] == cohort) & (t["odor"] == odor)]
        return float(r["p_batch"].iloc[0]) if len(r) else float("nan")

    full_note = (f"full model on the {len(shared)} odors every cohort ran ({PARAFFIN} left "
                 "out: the IAA cohort never ran it)")
    for co in primaries:
        core = (f"responded ~ post * C(grp, Treatment('{ref}')) * "
                f"C(odor, Treatment('{co.odor}'))")
        formula, notes = add_covariates(full, core)
        notes = [full_note, f"odor reference = {co.odor}, so post:cohort[{co.code}] is the "
                 f"{co.code} x {co.odor} DiD (log odds ratio)"] + notes
        term = f"post:C(grp, Treatment('{ref}'))[T.{co.code}]"
        for kind in ("gee", "glmm"):
            log.info("exp1 b: fitting full %s, odor reference %s", kind, co.odor)
            res = _save_model(fits[kind](full, formula),
                              models_dir / f"exp1_b_full_{kind}_ref-{_slug(co.odor)}", notes)
            rows.append({"cohort": co.name, "cohort_label": co.label, "odor": co.odor,
                         "primary": True, "model": kind, "model_label": labels[kind],
                         "formula": formula, "effect_scale": "log odds ratio",
                         "n_flies": int(full[full["grp"].isin([co.code, ref])]["fly_id"].nunique()),
                         "p_batch": p_batch_of(co.name, co.odor, "rate"),
                         **_contrast(res[1], term), "covariate_notes": "; ".join(notes)})

    # robustness: the same contrast from a model of that odor alone (8 fixed effects
    # instead of 56; the VB mixed fit of the full model proved unstable on this table)
    for co in primaries:
        sub = pp[pp["odor"] == co.odor]
        core = f"responded ~ post * C(grp, Treatment('{ref}'))"
        formula, notes = add_covariates(sub, core)
        notes = [f"{co.odor} trials of every cohort only (robustness check of the full "
                 "model's primary contrast)"] + notes
        term = f"post:C(grp, Treatment('{ref}'))[T.{co.code}]"
        for kind in ("gee", "glmm"):
            log.info("exp1 b: fitting per-odor %s for %s", kind, co.odor)
            res = _save_model(fits[kind](sub, formula),
                              models_dir / f"exp1_b_odor_{kind}_{_slug(co.odor)}", notes)
            rows.append({"cohort": co.name, "cohort_label": co.label, "odor": co.odor,
                         "primary": True, "model": f"{kind}_odor",
                         "model_label": labels[f"{kind}_odor"], "formula": formula,
                         "effect_scale": "log odds ratio",
                         "n_flies": int(sub[sub["grp"].isin([co.code, ref])]["fly_id"].nunique()),
                         "p_batch": p_batch_of(co.name, co.odor, "rate"),
                         **_contrast(res[1], term), "covariate_notes": "; ".join(notes)})

    par = pp[pp["odor"] == PARAFFIN]
    par_cohorts = [co for co in _exposed(info) if (par["cohort"] == co.name).any()]
    if (par["cohort"] == CONTROL).any() and par_cohorts:
        core = f"responded ~ post * C(grp, Treatment('{ref}'))"
        formula, notes = add_covariates(par, core)
        notes = [f"{PARAFFIN} only (exploratory): flies that ran it; no IAA-cohort rows"] + notes
        for kind in ("gee", "glmm"):
            log.info("exp1 b: fitting %s %s model", PARAFFIN, kind)
            res = _save_model(fits[kind](par, formula),
                              models_dir / f"exp1_b_paraffin_{kind}", notes)
            for co in par_cohorts:
                rows.append({"cohort": co.name, "cohort_label": co.label, "odor": PARAFFIN,
                             "primary": False, "model": kind, "model_label": labels[kind],
                             "formula": formula, "effect_scale": "log odds ratio",
                             "n_flies": int(par[par["grp"].isin([co.code, ref])]["fly_id"].nunique()),
                             "p_batch": p_batch_of(co.name, PARAFFIN, "rate"),
                             **_contrast(res[1], f"post:C(grp, Treatment('{ref}'))[T.{co.code}]"),
                             "covariate_notes": "; ".join(notes)})

    if ordinal:
        for co in primaries:
            sub = pp[pp["odor"] == co.odor]
            core = f"score ~ post * C(grp, Treatment('{ref}'))"
            formula, notes = add_covariates(sub, core)
            notes = [f"{co.odor} trials of every cohort; post:cohort[{co.code}] is the DiD "
                     "(log cumulative odds ratio)"] + notes
            log.info("exp1 b: fitting ordinal score model for %s (%d bootstrap replicates)",
                     co.odor, ordinal_boot)
            res = _save_model(st.fit_ordinal(sub, formula, n_boot=ordinal_boot, seed=seed),
                              models_dir / f"exp1_b_ordinal_{_slug(co.odor)}", notes)
            rows.append({"cohort": co.name, "cohort_label": co.label, "odor": co.odor,
                         "primary": True, "model": "ordinal", "model_label": labels["ordinal"],
                         "formula": formula, "effect_scale": "log cumulative odds ratio",
                         "n_flies": int(sub[sub["grp"].isin([co.code, ref])]["fly_id"].nunique()),
                         "p_batch": p_batch_of(co.name, co.odor, "score"),
                         **_contrast(res[1], f"post:C(grp, Treatment('{ref}'))[T.{co.code}]"),
                         "covariate_notes": "; ".join(notes)})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["family"] = np.where(out["primary"], "primary: " + out["model"],
                             "exploratory: " + out["model"] + " " + out["odor"])
    out = _corrected(out, family_col="family")
    out["odds_ratio"] = np.exp(out["effect"].astype(float))
    out["test"] = out["model_label"] + "; Wald p on the post:cohort term"
    out["correction"] = np.where(out["primary"], "Holm across the 3 primary contrasts "
                                 "within each model kind", "none (single exploratory contrast "
                                 "per model)")
    out["p_batch_source"] = ("whole-batch permutation p of the same contrast's bootstrap DiD "
                             "(rate for logistic, score for ordinal)")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# c. global drift
# ═════════════════════════════════════════════════════════════════════════════
def fly_drift(d: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per fly: the metric pooled over its trials of the odors every cohort ran, pre and
    post (``pre``, ``post``, 0/1 for rate), ``drift`` = post - pre (with one trial per
    odor per block, the mean of its per-odor changes), ``n_odors``, ``n_trials``."""
    value = METRIC_VALUE[metric]
    shared = common_odors(d)
    pp = _test_rows(d)
    pp = pp[pp["odor"].isin(shared)]
    pf = st.per_fly(pp, value, by=["cohort", "block"], carry=("batch", "start_time_h"))
    w = pf.pivot_table(index=["cohort", "fly_id"], columns="block", values=value,
                       aggfunc="first", observed=True)
    meta = pf.groupby(["cohort", "fly_id"]).agg(batch=("batch", "first"),
                                                start_time_h=("start_time_h", "first"),
                                                n_trials=("n_trials", "sum"))
    n_odors = pp.groupby(["cohort", "fly_id"])["odor"].nunique().rename("n_odors")
    out = meta.join(w[["pre", "post"]]).join(n_odors).dropna(subset=["pre", "post"])
    out["drift"] = out["post"] - out["pre"]
    log.debug("fly_drift(%s): %d flies over %d shared odors (%s)", metric, len(out),
              len(shared), ", ".join(shared))
    return out.reset_index()


def _omnibus_perm(values: np.ndarray, labels: np.ndarray, *, n_perm: int,
                  seed: int) -> float:
    """Fly-level permutation of the between-group sum of squares (labels shuffled across
    flies), (hits + 1) / (n_perm + 1); chunked to bound memory."""
    v = np.asarray(values, float)
    codes = pd.factorize(pd.Series(labels))[0]
    levels = np.unique(codes)
    m = v.mean()

    def ss(lab: np.ndarray) -> np.ndarray:
        out = np.zeros(lab.shape[0])
        for lv in levels:
            mask = (lab == lv)
            n = mask.sum(1)
            s = (mask * v).sum(1)
            out += np.where(n > 0, n * (s / np.maximum(n, 1) - m) ** 2, 0.0)
        return out

    obs = ss(codes[None, :])[0]
    rng = np.random.default_rng(seed)
    hits, done = 0, 0
    while done < n_perm:
        k = min(2_000, n_perm - done)
        lab = rng.permuted(np.tile(codes, (k, 1)), axis=1)
        hits += int(np.sum(ss(lab) >= obs - 1e-12))
        done += k
    return (hits + 1) / (n_perm + 1)


def _slope_ci(x: np.ndarray, y: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float, float]:
    """OLS slope of y on x and its 95 % percentile bootstrap CI, resampling flies."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3 or np.ptp(x) == 0:
        return float("nan"), float("nan"), float("nan")
    slope = float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, (int(n_boot), x.size))
    xb, yb = x[idx], y[idx]
    xc = xb - xb.mean(1, keepdims=True)
    den = (xc ** 2).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sb = (xc * (yb - yb.mean(1, keepdims=True))).sum(1) / den
    sb = sb[np.isfinite(sb)]
    lo, hi = (np.quantile(sb, [0.025, 0.975]) if sb.size else (np.nan, np.nan))
    return slope, float(lo), float(hi)


def drift_tables(d: pd.DataFrame, metric: str, *, n_boot: int | None = None,
                 n_perm: int | None = None, seed: int | None = None
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(tests, trend, per-fly drift). Tests: drift != 0 per cohort (sign-flip + batch
    sign-flip; Holm across cohorts), each exposed cohort vs the control
    (``did_bootstrap``; Holm across the exposed cohorts), and a 4-cohort omnibus (fly-level
    and whole-batch label permutation of the between-cohort sum of squares). Trend: drift
    vs fly start time per cohort and pooled (OLS slope per hour with a fly-bootstrap CI;
    Spearman permutation p over flies, and over batches via ``b2b_stats.batch_trend``)."""
    info = cohort_info(d)
    k = SCALE[metric]
    n_boot = int(st.config_value("N_BOOT") if n_boot is None else n_boot)
    n_perm = int(st.config_value("N_PERM") if n_perm is None else n_perm)
    seed = int(st.config_value("SEED") if seed is None else seed)
    dr = fly_drift(d, metric)
    batch_of = dr.set_index("fly_id")["batch"]
    rows = []
    for co in info.values():
        sub = dr[dr["cohort"] == co.name].set_index("fly_id")
        r = st.paired_effect(sub["pre"], sub["post"], batch_of=batch_of, n_boot=n_boot,
                             n_perm=n_perm, seed=seed)
        rows.append({"comparison": f"{co.label}: drift vs 0", "cohort": co.name,
                     "cohort_label": co.label, "family": "drift vs 0 within cohort",
                     "n_flies": r["n_flies"], "n_batches": r["n_batches"],
                     "n_trials": int(sub["n_trials"].sum()), "pre_mean": r["mean_a"] * k,
                     "post_mean": r["mean_b"] * k, "effect": r["mean_diff"] * k,
                     "ci_lo": r["ci_lo"] * k, "ci_hi": r["ci_hi"] * k, "p": r["p"],
                     "p_batch": r["p_batch"],
                     "test": "paired sign-flip on per-fly drift (exact), BCa 95% CI; "
                             "p_batch: whole-batch sign-flip"})
    if CONTROL in info:
        c = dr[dr["cohort"] == CONTROL].set_index("fly_id")["drift"]
        for co in _exposed(info):
            x = dr[dr["cohort"] == co.name].set_index("fly_id")["drift"]
            r = _did_row(x, c, batch_of, k, n_boot=n_boot, n_perm=n_perm, seed=seed)
            rows.append({"comparison": f"{co.label} vs {info[CONTROL].label}",
                         "cohort": co.name, "cohort_label": co.label,
                         "family": "exposed cohort vs Paraffin control",
                         "n_trials": int(dr[dr["cohort"].isin([co.name, CONTROL])]["n_trials"].sum()),
                         **r, "test": "difference in mean drift: fly-bootstrap BCa 95% CI, "
                                      "fly-label permutation p; p_batch: whole-batch "
                                      "permutation"})
    if dr["cohort"].nunique() >= 2:
        p_fly = _omnibus_perm(dr["drift"].to_numpy(float), dr["cohort"].to_numpy(object),
                              n_perm=min(n_perm, 100_000), seed=seed)
        lab = dr.drop_duplicates("batch").set_index("batch")["cohort"].to_dict()
        bt = _b2b.batch_label_perm(dr["drift"].to_numpy(float), dr["batch"].to_numpy(object),
                                   lab)
        means = dr.groupby("cohort")["drift"].mean() * k
        rows.append({"comparison": "omnibus: any cohort differs", "cohort": "all",
                     "cohort_label": "all cohorts", "family": "omnibus",
                     "n_flies": len(dr), "n_batches": dr["batch"].nunique(),
                     "n_trials": int(dr["n_trials"].sum()), "p": p_fly, "p_batch": bt["p"],
                     "p_batch_exact": bt["exact"],
                     "note": "cohort mean drift: " + "; ".join(
                         f"{info[n].label} {_f(v, _digits(metric))}" for n, v in means.items()),
                     "test": "between-cohort sum of squares: fly-label permutation p; "
                             "p_batch: whole-batch label permutation (b2b_stats)"})
    tests = _corrected(pd.DataFrame(rows), family_col="family")
    tests["metric"], tests["unit"] = metric, UNIT[metric]
    tests["correction"] = tests["family"].map({
        "drift vs 0 within cohort": "Holm across cohorts",
        "exposed cohort vs Paraffin control": "Holm across the exposed cohorts",
        "omnibus": "none (single test)"})
    tests["batch_disagrees"] = [_disagrees(a, b) for a, b in zip(tests["p"], tests["p_batch"])]

    trows = []
    groups = [(co.name, co.label, dr[dr["cohort"] == co.name]) for co in info.values()]
    groups.append(("all", "all cohorts pooled", dr))
    for name, label, sub in groups:
        y = sub["drift"].to_numpy(float) * k
        x = sub["start_time_h"].to_numpy(float)
        slope, lo, hi = _slope_ci(x, y, n_boot=n_boot, seed=seed)
        fly = _b2b.batch_trend(y, sub["fly_id"].to_numpy(object),
                               dict(zip(sub["fly_id"], sub["start_time_h"])))
        bat = _b2b.batch_trend(y, sub["batch"].to_numpy(object),
                               sub.groupby("batch")["start_time_h"].first().to_dict())
        trows.append({"cohort": name, "cohort_label": label,
                      "family": "trend within cohort" if name != "all" else "trend pooled",
                      "n_flies": len(sub), "n_batches": sub["batch"].nunique(),
                      "n_trials": int(sub["n_trials"].sum()), "effect": slope,
                      "effect_unit": f"{UNIT_SHORT[metric]} per hour of start time",
                      "ci_lo": lo, "ci_hi": hi, "rho_fly": fly["rho"], "p": fly["p"],
                      "rho_batch": bat["rho"], "p_batch": bat["p"],
                      "x_min": float(np.min(x)) if x.size else np.nan,
                      "x_max": float(np.max(x)) if x.size else np.nan,
                      "intercept": (float(np.polyfit(x, y, 1)[1])
                                    if x.size >= 3 and np.ptp(x) > 0 else np.nan)})
    trend = _corrected(pd.DataFrame(trows), family_col="family")
    trend["metric"], trend["unit"] = metric, UNIT[metric]
    trend["test"] = ("OLS slope of per-fly drift on start time (fly-bootstrap 95% CI); p: "
                     "Spearman, flies permuted; p_batch: Spearman of batch means, batches "
                     "permuted (b2b_stats.batch_trend)")
    trend["correction"] = np.where(trend["cohort"] == "all", "none (pooled, single test)",
                                   "Holm across cohorts")
    trend["batch_disagrees"] = [_disagrees(a, b) for a, b in zip(trend["p"], trend["p_batch"])]
    return tests, trend, dr


# ═════════════════════════════════════════════════════════════════════════════
# d. time since exposure
# ═════════════════════════════════════════════════════════════════════════════
def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx @ ry) / np.sqrt((rx @ rx) * (ry @ ry)))


def _batch_trend_difference(ex: pd.DataFrame, ct: pd.DataFrame, value: str, *,
                            n_perm: int, seed: int) -> dict:
    """Does the batch-level trend (Spearman of batch mean vs batch minutes since
    exposure) differ between the exposed and the control batches? Exposed/control labels
    are permuted across whole batches; two-sided (hits + 1) / (n + 1)."""
    def by_batch(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby("batch").agg(y=(value, lambda s: float(np.mean(s.astype(float)))),
                                       x=("minutes_since_exposure_end", "first"))
    bx, bc = by_batch(ex), by_batch(ct)
    xs = np.r_[bx["x"].to_numpy(float), bc["x"].to_numpy(float)]
    ys = np.r_[bx["y"].to_numpy(float), bc["y"].to_numpy(float)]
    kx = len(bx)
    obs = _spearman(xs[:kx], ys[:kx]) - _spearman(xs[kx:], ys[kx:])
    out = {"rho_diff": obs, "p": float("nan"), "n_batches": len(xs)}
    if not np.isfinite(obs):
        return out
    rng = np.random.default_rng(seed)
    hits, valid = 0, 0
    for _ in range(int(n_perm)):
        pick = rng.permutation(len(xs))
        a, b = pick[:kx], pick[kx:]
        s = _spearman(xs[a], ys[a]) - _spearman(xs[b], ys[b])
        if np.isfinite(s):
            valid += 1
            hits += abs(s) >= abs(obs) - 1e-12
    out["p"] = (hits + 1) / (valid + 1) if valid else float("nan")
    return out


def time_since_exposure_tables(d: pd.DataFrame, models_dir: Path, *,
                               n_perm: int | None = None, seed: int | None = None
                               ) -> dict[str, pd.DataFrame]:
    """Per exposed cohort, its exposure odor's post-exposure trials vs minutes since the
    last exposure pulse (per 10 min):

    * rate: GEE logistic ``responded ~ min10`` (exposed cohort alone) and ``~ min10 *
      exposed`` (with the Paraffin control's trials of the same odor), clustered by batch;
    * score: MixedLM ``score ~ min10`` / ``~ min10 * exposed``, random intercept by batch.

    Clustered by BATCH: each fly has one post trial of the odor, and time since exposure
    is a batch property (one shuffled order per folder). ``p_batch``: Spearman of batch
    means vs batch minutes (``b2b_stats.batch_trend``), or for the interaction the
    exposed/control labels permuted across batches. Holm across cohorts per test.
    Returns {"rate": table, "score": table}."""
    info = cohort_info(d)
    n_perm = int(st.config_value("N_PERM") if n_perm is None else n_perm)
    seed = int(st.config_value("SEED") if seed is None else seed)
    post = d[(d["block"] == "post") & d["odor"].notna()].copy()
    post["min10"] = post["minutes_since_exposure_end"].astype(float) / 10.0
    covs = ("start_time_c", "position_in_block", "exposure_conc")
    out: dict[str, list[dict]] = {"rate": [], "score": []}
    for co in _exposed(info):
        ex = post[(post["cohort"] == co.name) & (post["odor"] == co.odor)].copy()
        ct = post[(post["cohort"] == CONTROL) & (post["odor"] == co.odor)].copy()
        both = pd.concat([ex, ct], ignore_index=True)
        both["exposed"] = (both["cohort"] == co.name).astype(int)
        for metric, value, fit, kind in (("rate", "responded", st.fit_gee_logistic, "gee"),
                                         ("score", "score", st.fit_mixedlm, "mixedlm")):
            vals = ex[value].astype(float).to_numpy()
            bt = _b2b.batch_trend(vals, ex["batch"].to_numpy(object),
                                  ex.groupby("batch")["minutes_since_exposure_end"].first().to_dict())
            diff = (_batch_trend_difference(ex, ct, value, n_perm=min(n_perm, TREND_PERM_CAP),
                                            seed=seed) if len(ct) else {"p": np.nan})
            for test, frame, core, term, p_batch, extra in (
                    ("slope in the exposed cohort", ex, f"{value} ~ min10", "min10", bt["p"],
                     {"rho_batch": bt["rho"]}),
                    ("slope difference vs Paraffin control", both,
                     f"{value} ~ min10 * exposed", "min10:exposed", diff["p"],
                     {"rho_batch_diff": diff.get("rho_diff", np.nan)})):
                if frame["batch"].nunique() < 2 or frame["min10"].nunique() < 2:
                    continue
                formula, notes = add_covariates(frame, core, covs)
                stem = models_dir / f"exp1_d_{kind}_{_slug(co.code)}_{'slope' if term == 'min10' else 'vs-control'}"
                log.info("exp1 d: %s %s for %s", kind, test, co.label)
                res = _save_model(fit(frame, formula, group="batch"), stem, notes)
                cr = _contrast(res[1], term)
                out[metric].append({
                    "cohort": co.name, "cohort_label": co.label, "odor": co.odor,
                    "test_name": test, "family": test, "metric": metric,
                    "n_flies": int(frame["fly_id"].nunique()),
                    "n_batches": int(frame["batch"].nunique()), "n_trials": len(frame),
                    "effect": cr["effect"], "ci_lo": cr["ci_lo"], "ci_hi": cr["ci_hi"],
                    "p": cr["p"], "p_batch": p_batch, **extra,
                    "effect_unit": ("log odds ratio per 10 min" if metric == "rate"
                                    else "score points per 10 min"),
                    "status": cr["status"], "formula": formula,
                    "covariate_notes": "; ".join(notes),
                    "test": (f"{'GEE logistic' if metric == 'rate' else 'MixedLM'} "
                             f"clustered by batch, Wald p on {term}; p_batch: "
                             + ("Spearman of batch means vs minutes, batches permuted"
                                if term == "min10" else
                                "difference in batch-level Spearman, exposed/control labels "
                                "permuted across batches"))})
    tables = {}
    for metric, rows in out.items():
        t = _corrected(pd.DataFrame(rows), family_col="family") if rows else pd.DataFrame(rows)
        if len(t):
            t["correction"] = "Holm across cohorts within each test"
            t["batch_disagrees"] = [_disagrees(a, b) for a, b in zip(t["p"], t["p_batch"])]
        tables[metric] = t
    return tables


# ═════════════════════════════════════════════════════════════════════════════
# e. odor-specific suppression
# ═════════════════════════════════════════════════════════════════════════════
def suppression_table(d: pd.DataFrame, metric: str, *, n_boot: int | None = None,
                      n_perm: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Per exposure odor (the exposed cohorts', in cohort order): the self-exposed
    cohort's within-fly change vs the pooled change of every other exp1 cohort (control
    included) on that odor, ``stats.did_bootstrap``; Holm across the odors."""
    info = cohort_info(d)
    k = SCALE[metric]
    ch = fly_changes(d, metric)
    batch_of = _batch_of(d)
    rows = []
    for co in _exposed(info):
        on = ch[ch["odor"] == co.odor]
        x = on[on["cohort"] == co.name]
        c = on[on["cohort"] != co.name]
        others = c.groupby("cohort")["fly_id"].nunique()
        base = {"odor": co.odor, "self_cohort": co.name, "cohort_label": co.label,
                "others": "; ".join(f"{info[n].label} n={v}" for n, v in others.items()),
                "metric": metric, "unit": UNIT[metric],
                "n_trials": int(x["n_trials"].sum() + c["n_trials"].sum())}
        if x.empty or c.empty:
            rows.append({**base, "n_flies_exposed": len(x), "n_flies_control": len(c),
                         "n_flies": len(x) + len(c)})
            continue
        r = _did_row(x.set_index("fly_id")["change"], c.set_index("fly_id")["change"],
                     batch_of, k, n_boot=n_boot, n_perm=n_perm, seed=seed)
        r["change_others"] = r.pop("change_control")
        r["change_self"] = r.pop("change_exposed")
        rows.append({**base, **r})
    out = _corrected(pd.DataFrame(rows))
    out["test"] = ("DiD of within-fly post - pre, self-exposed cohort vs all other cohorts "
                   "pooled: fly-bootstrap BCa 95% CI, fly-label permutation p; p_batch: "
                   "whole-batch permutation")
    out["correction"] = "Holm across the exposure odors"
    out["batch_disagrees"] = [_disagrees(a, b) for a, b in zip(out["p"], out["p_batch"])]
    return out


# ═════════════════════════════════════════════════════════════════════════════
# figures
# ═════════════════════════════════════════════════════════════════════════════
def _paraffin_caption(d: pd.DataFrame, info: Mapping[str, Cohort]) -> str:
    pp = _test_rows(d)
    parts = []
    for co in info.values():
        if co.is_control:
            continue
        n_all = pp.loc[pp["cohort"] == co.name, "fly_id"].nunique()
        n_par = pp.loc[(pp["cohort"] == co.name) & (pp["odor"] == PARAFFIN), "fly_id"].nunique()
        parts.append(f"{co.code} {n_par}/{n_all}")
    return (f"{PARAFFIN} rows use only flies that ran it ({', '.join(parts)} flies); "
            f"{PARAFFIN_CAVEAT}.")


def _concurrency_caption(d: pd.DataFrame, info: Mapping[str, Cohort]) -> str:
    spans = [f"{co.code} {_date_range(d, co.name)}" for co in info.values()
             if _date_range(d, co.name)]
    if not spans:
        return ""
    return ("Cohorts were not all run concurrently (" + ", ".join(spans) +
            "): the Paraffin control does not overlap the IAA cohort.")


def _ylabel(metric: str, kind: str) -> str:
    if kind == "change":
        return ("Δ PER, post − pre (pp)" if metric == "rate" else "Δ PER score, post − pre")
    if kind == "did":
        return ("DiD (percentage points)" if metric == "rate" else "DiD (score points)")
    return {"rate": "PER (% of trials)", "score": "PER score (mean ± SEM)"}[metric]


def plot_pre_vs_post(d: pd.DataFrame, table: pd.DataFrame, metric: str, stem: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    info = cohort_info(d)
    value = METRIC_VALUE[metric]
    pp = _test_rows(d)
    odors = pl.odor_order(pp["odor"].unique())
    n = len(info)
    fig, axes = pl.new_figure(n, 1, figsize=(8.6, 2.55 * n + 2.4), squeeze=False)
    plotted = []
    for i, co in enumerate(info.values()):
        ax = axes[i, 0]
        sub = pp[pp["cohort"] == co.name]
        gs = st.group_summary(sub, by=["odor", "block"], metric=metric)
        pf = st.per_fly(sub, value, by=["odor", "block"])
        drawn = pl.paired_bars(ax, gs, conditions=["pre", "post"], metric=metric, groups=odors,
                               per_fly=pf, condition_col="block", experiment=SECTION,
                               condition_labels={"pre": "pre-exposure",
                                                 "post": "post-exposure"},
                               group_labels={o: o.replace(" ", "\n") for o in odors},
                               legend=(i == 0))
        tab = table[(table["cohort"] == co.name) & table["p_holm"].notna()]
        comps = pd.DataFrame({"group": tab["odor"], "cond_a": "pre", "cond_b": "post",
                              "p_holm": tab["p_holm"]})
        stars = pl.add_holm_stars(ax, comps, plotted=drawn, group_col="odor",
                                  condition_col="block")
        ylo, yhi = ax.get_ylim()
        top = pl.shared_ylim(metric, SECTION)[1] + 0.04 * (yhi - ylo)
        if co.odor in odors:
            gi = odors.index(co.odor)
            ax.add_patch(Rectangle((gi - 0.47, ylo), 0.94, top - ylo, fill=False,
                                   linestyle=(0, (4, 2)), linewidth=0.9,
                                   edgecolor=pl.INK_SECONDARY, zorder=0.6))
        for gi, o in enumerate(odors):
            if not ((gs["odor"] == o).any()):
                ax.text(gi, ylo + 0.05 * (yhi - ylo), "not\ntested", ha="center", va="bottom",
                        fontsize=7, color=pl.INK_MUTED)
        ax.set_title(f"{co.label} · {sub['fly_id'].nunique()} flies"
                     + (" · dashed box = mock exposure" if co.is_control else
                        f" · dashed box = exposure odor ({co.odor})"),
                     loc="right", fontsize=8.5, color=pl.INK_PRIMARY, pad=3)
        plotted.append(drawn.assign(cohort=co.name))
        plotted.append(stars.assign(cohort=co.name, layer="stars"))
    pl.title_block(fig, f"Pre- vs post-exposure {METRIC_WORD[metric]} per test odor, "
                        "by exposure cohort",
                   pl.stats_subtitle(n_flies=pp["fly_id"].nunique(), n_trials=len(pp),
                                     test="paired sign-flip per odor on within-fly change "
                                          "(exact), BCa 95% CI",
                                     correction="Holm across odors within each cohort panel"))
    bars = ("bars pool trials: % of trials responding, 95% Wilson CI (flies as unit)"
            if metric == "rate" else "bars: pooled mean score ± SEM over flies")
    pl.caption(fig, f"{pl.HOLM_CAPTION}. {bars}; grey lines = individual flies. "
                    f"{_paraffin_caption(d, info)}")
    paths = pl.save_figure(fig, pd.concat(plotted, ignore_index=True), stem)
    plt.close("all")
    return paths["png"]


def plot_did(d: pd.DataFrame, table: pd.DataFrame, metric: str, stem: Path) -> Path:
    from matplotlib.transforms import blended_transform_factory
    info = cohort_info(d)
    exposed = _exposed(info)
    odors = pl.odor_order(table["odor"].unique())
    vals = pd.concat([table["ci_lo"], table["ci_hi"], table["effect"]]).astype(float)
    lims = pl.register_ylim(f"did_{metric}", SECTION, data=vals[np.isfinite(vals)].tolist() or [0])
    span = lims[1] - lims[0]
    ypos = {o: len(odors) - 1 - i for i, o in enumerate(odors)}
    fig, axes = pl.new_figure(1, len(exposed), figsize=(3.2 * len(exposed) + 1.4,
                                                        0.42 * len(odors) + 3.0),
                              sharey=True, squeeze=False)
    fig.get_layout_engine().set(wspace=0.08)       # keep panel tick labels apart
    plotted = []
    ctrl_n = d.loc[d["cohort"] == CONTROL, "fly_id"].nunique()
    for j, co in enumerate(exposed):
        ax = axes[0, j]
        pl.style_axes(ax, grid="x")
        ax.axvline(0.0, color=pl.BASELINE, linewidth=0.8, zorder=1)
        sub = table[table["cohort"] == co.name]
        tr = blended_transform_factory(ax.transAxes, ax.transData)
        for _, r in sub.iterrows():
            y = ypos[r["odor"]]
            if r["primary"]:
                ax.axhspan(y - 0.42, y + 0.42, color=pl.GRID, alpha=0.55, linewidth=0, zorder=0)
            if not np.isfinite(float(r["effect"])):
                ax.text(lims[0] + 0.03 * span, y, "not tested", va="center", ha="left",
                        fontsize=7.5, color=pl.INK_MUTED)
                plotted.append({"cohort": co.name, "odor": r["odor"], "y": y,
                                "primary": bool(r["primary"]), "effect": np.nan,
                                "ci_lo": np.nan, "ci_hi": np.nan, "p_holm": np.nan,
                                "stars": "", "note": "not tested"})
                continue
            color = pl.INK_PRIMARY if r["primary"] else co.color
            ax.hlines(y, r["ci_lo"], r["ci_hi"], color=color,
                      linewidth=1.8 if r["primary"] else 1.3, zorder=2)
            ax.scatter([r["effect"]], [y], marker="D" if r["primary"] else "o",
                       s=48 if r["primary"] else 30, color=color, edgecolors=pl.SURFACE,
                       linewidths=1.2, zorder=3)
            s = st.stars(r["p_holm"])
            if s:
                ax.text(1.01, y, s, transform=tr, ha="left", va="center", fontsize=9,
                        color=pl.INK_PRIMARY, clip_on=False)
            plotted.append({"cohort": co.name, "odor": r["odor"], "y": y,
                            "primary": bool(r["primary"]), "effect": r["effect"],
                            "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p_holm": r["p_holm"],
                            "stars": s, "n_flies_exposed": r["n_flies_exposed"],
                            "n_flies_control": r["n_flies_control"]})
        ax.set_xlim(lims)
        ax.set_xlabel(_ylabel(metric, "did"))
        n_x = d.loc[d["cohort"] == co.name, "fly_id"].nunique()
        ax.set_title(f"{co.label}\n{n_x} exposed vs {ctrl_n} control flies", loc="left",
                     fontsize=8.5, color=pl.INK_PRIMARY, pad=4)
    ax0 = axes[0, 0]
    ax0.set_yticks([ypos[o] for o in odors])
    ax0.set_yticklabels(odors)
    ax0.set_ylim(-0.6, len(odors) - 0.4)
    n_flies = int(_test_rows(d)["fly_id"].nunique())
    n_trials = int(_test_rows(d).shape[0])
    pl.title_block(fig, f"Exposure effect beyond the Paraffin control: difference in "
                        f"differences, {METRIC_WORD[metric]}",
                   pl.stats_subtitle(n_flies=n_flies, n_trials=n_trials,
                                     test="DiD of within-fly post − pre, fly-bootstrap BCa "
                                          "95% CI, fly-label permutation p",
                                     correction="Holm across the 3 primaries; exploratory "
                                                "rows Holm within each panel"))
    pl.caption(fig, f"{pl.HOLM_CAPTION}. DiD = (post − pre in the exposed cohort) − "
                    "(post − pre in the Paraffin control), means over flies. Black diamond on a "
                    "grey band = pre-specified primary contrast (the cohort's own exposure "
                    "odor); coloured dots = exploratory. Whole-batch permutation p (p_batch) is in the table. "
                    f"{_concurrency_caption(d, info)} {_paraffin_caption(d, info)}")
    paths = pl.save_figure(fig, pd.DataFrame(plotted), stem)
    return paths["png"]


def plot_drift(dr: pd.DataFrame, tests: pd.DataFrame, metric: str, stem: Path,
               info: Mapping[str, Cohort]) -> Path:
    k = SCALE[metric]
    cohorts = [co for co in info.values() if (dr["cohort"] == co.name).any()]
    within = tests[tests["family"] == "drift vs 0 within cohort"].set_index("cohort")
    between = tests[tests["family"] == "exposed cohort vs Paraffin control"]
    lims = pl.shared_ylim(f"drift_{metric}", SECTION)
    span = lims[1] - lims[0]
    fig, ax = pl.new_figure(figsize=(7.4, 5.6))
    pl.style_axes(ax, grid="y")
    rng = np.random.default_rng(0)
    plotted = []
    xs = {co.name: i for i, co in enumerate(cohorts)}
    for co in cohorts:
        x0 = xs[co.name]
        y = dr.loc[dr["cohort"] == co.name, "drift"].to_numpy(float) * k
        jit = x0 + rng.uniform(-0.28, 0.28, y.size)
        ax.scatter(jit, y, s=9, color=pl.INK_MUTED, alpha=0.35, linewidths=0, zorder=1.5)
        r = within.loc[co.name]
        ax.bar([x0], [r["effect"]], width=0.62, color=co.color, alpha=0.85, zorder=2,
               linewidth=0)
        ax.errorbar([x0], [r["effect"]], yerr=[[r["effect"] - r["ci_lo"]],
                                               [r["ci_hi"] - r["effect"]]],
                    fmt="none", ecolor=pl.INK_PRIMARY, elinewidth=1.0, capsize=3, zorder=4)
        s = st.stars(r["p_holm"])
        if s:
            ax.text(x0, lims[1] + 0.01 * span, s, ha="center", va="bottom", fontsize=9,
                    color=pl.INK_PRIMARY)
        plotted += [{"layer": "fly", "cohort": co.name, "x": xx, "value": yy}
                    for xx, yy in zip(jit, y)]
        plotted.append({"layer": "bar", "cohort": co.name, "x": x0, "value": r["effect"],
                        "err_lo": r["ci_lo"], "err_hi": r["ci_hi"], "p_holm": r["p_holm"],
                        "stars": s, "n_flies": r["n_flies"]})
    ax.axhline(0.0, color=pl.BASELINE, linewidth=0.8, zorder=1.6)
    ax.set_ylim(lims[0], lims[1] + pl.STAR_HEADROOM * span)
    ax.spines["left"].set_bounds(*lims)
    comps = []
    if CONTROL in xs:
        for _, r in between.iterrows():
            comps.append({"x1": xs[r["cohort"]], "x2": xs[CONTROL],
                          "y": lims[1] + 0.07 * span, "p_holm": r["p_holm"]})
    if comps:
        br = pl.add_holm_stars(ax, comps)
        top = np.nanmax(br["y"]) if br["y"].notna().any() else np.nan
        if np.isfinite(top) and top + 0.08 * span > ax.get_ylim()[1]:
            ax.set_ylim(lims[0], top + 0.1 * span)
        plotted += [{"layer": "bracket", **row} for row in br.to_dict("records")]
    ax.set_xticks(range(len(cohorts)))
    ax.set_xticklabels([co.label.replace(" (", "\n(") for co in cohorts])
    ax.set_xlim(-0.6, len(cohorts) - 0.4)
    ax.set_ylabel(_ylabel(metric, "change") + ", mean over odors")
    n_tr = int(dr["n_trials"].sum())
    pl.title_block(fig, f"Global drift by cohort: post − pre {METRIC_WORD[metric]}, "
                        "mean over odors",
                   pl.stats_subtitle(n_flies=len(dr), n_trials=n_tr,
                                     test="sign-flip vs 0 per cohort; exposed vs control "
                                          "fly-label permutation",
                                     correction="Holm (4 cohorts; 3 vs-control)"))
    shared = int(dr["n_odors"].max()) if len(dr) else 0
    pl.caption(fig, f"{pl.HOLM_CAPTION}. Bar = mean drift over flies (each fly: post − pre over "
                    f"the {shared} odors every cohort ran; {PARAFFIN} left out), whisker = BCa "
                    "95% CI over flies, dots = flies (x jittered). Star above a bar: drift ≠ 0; "
                    "bracket: exposed cohort vs Paraffin control. Batch-level p in the table.")
    paths = pl.save_figure(fig, pd.DataFrame(plotted), stem)
    return paths["png"]


def plot_drift_vs_start(dr: pd.DataFrame, trend: pd.DataFrame, metric: str, stem: Path,
                        info: Mapping[str, Cohort]) -> Path:
    from matplotlib.lines import Line2D
    k = SCALE[metric]
    lims = pl.shared_ylim(f"drift_{metric}", SECTION)
    fig, ax = pl.new_figure(figsize=(7.6, 5.4))
    pl.style_axes(ax, grid="both")
    rng = np.random.default_rng(1)
    plotted, handles = [], []
    tr = trend.set_index("cohort")
    for co in info.values():
        sub = dr[dr["cohort"] == co.name]
        if sub.empty:
            continue
        x = sub["start_time_h"].to_numpy(float)
        y = sub["drift"].to_numpy(float) * k
        xj = x + rng.uniform(-0.04, 0.04, x.size)
        ax.scatter(xj, y, s=16, marker=co.marker, color=co.color, alpha=0.55,
                   edgecolors=pl.SURFACE, linewidths=0.4, zorder=2)
        r = tr.loc[co.name]
        if np.isfinite(r["effect"]):
            xx = np.array([r["x_min"], r["x_max"]])
            ax.plot(xx, r["intercept"] + r["effect"] * xx, color=co.color, linewidth=2,
                    solid_capstyle="round", zorder=3)
        s = st.stars(r["p_holm"])
        handles.append(Line2D([], [], color=co.color, marker=co.marker, linewidth=2,
                              markersize=5, label=f"{co.label}: batch-level ρ "
                                                  f"{_f(r['rho_batch'], 2)}{' ' + s if s else ''}"))
        plotted += [{"layer": "fly", "cohort": co.name, "x": a, "x_true": b, "value": c}
                    for a, b, c in zip(xj, x, y)]
        plotted.append({"layer": "trend", "cohort": co.name, "x": r["x_min"],
                        "x2": r["x_max"], "slope": r["effect"], "intercept": r["intercept"],
                        "rho_batch": r["rho_batch"], "p_batch": r["p_batch"],
                        "p_holm": r["p_holm"], "stars": s})
    ax.axhline(0.0, color=pl.BASELINE, linewidth=0.8, zorder=1)
    ax.set_ylim(*lims)
    ax.set_xlabel("Fly start time (clock hours, Pi clock)")
    ax.set_ylabel(_ylabel(metric, "change") + ", mean over odors")
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.13), ncols=2,
              frameon=False, fontsize=7.5, labelcolor=pl.INK_SECONDARY)
    allr = tr.loc["all"] if "all" in tr.index else None
    sub_extra = (f"all cohorts pooled: batch-level ρ {_f(allr['rho_batch'], 2)}"
                 if allr is not None else None)
    pl.title_block(fig, f"Global drift vs fly start time, {METRIC_WORD[metric]}",
                   pl.stats_subtitle(n_flies=len(dr), n_trials=int(dr["n_trials"].sum()),
                                     test="Spearman trend, batches permuted (b2b_stats)",
                                     correction="Holm across cohorts (fly-level p)",
                                     extra=sub_extra))
    pl.caption(fig, f"{pl.HOLM_CAPTION} (fly-level Spearman p). Lines = OLS fit per cohort over "
                    "its start-time range; legend: Spearman ρ of batch-mean drift vs batch start "
                    "time. Fly- and batch-level permutation p are in the table. Start time is a "
                    "batch property and "
                    "cohort start times differ, so the pooled trend is confounded with cohort. "
                    "x jittered ±0.04 h.")
    paths = pl.save_figure(fig, pd.DataFrame(plotted), stem)
    return paths["png"]


def plot_time_since_exposure(d: pd.DataFrame, table: pd.DataFrame, metric: str,
                             stem: Path) -> Path:
    info = cohort_info(d)
    exposed = _exposed(info)
    value = METRIC_VALUE[metric]
    post = d[(d["block"] == "post") & d["odor"].notna()].copy()
    edges = list(TIME_BINS)
    post["bin"] = pd.cut(post["minutes_since_exposure_end"].astype(float), edges, right=False)
    lims = pl.shared_ylim(metric, SECTION)
    span = lims[1] - lims[0]
    fig, axes = pl.new_figure(1, len(exposed), figsize=(3.4 * len(exposed) + 0.8, 5.0),
                              sharey=True, squeeze=False)
    plotted = []
    ctrl = info.get(CONTROL)
    for j, co in enumerate(exposed):
        ax = axes[0, j]
        pl.style_axes(ax, grid="y")
        pre = d[(d["block"] == "pre") & (d["odor"] == co.odor)]
        for cohort, style in ((co, {"ls": (0, (4, 2))}), (ctrl, {"ls": (0, (1, 2))})):
            if cohort is None:
                continue
            rows = post[(post["cohort"] == cohort.name) & (post["odor"] == co.odor)]
            if rows.empty:
                continue
            g = st.group_summary(rows, by=["bin"], metric=metric)
            xm = rows.groupby("bin", observed=True)["minutes_since_exposure_end"].mean()
            g["x"] = g["bin"].map(xm).astype(float)
            g = g.dropna(subset=["x"]).sort_values("x")
            ax.plot(g["x"], g["value"], color=cohort.color, marker=cohort.marker,
                    markersize=5, linewidth=1.8, markeredgecolor=pl.SURFACE, zorder=3,
                    label=f"{cohort.label}, post")
            yerr = np.vstack([np.clip(g["value"] - g["err_lo"], 0, None),
                              np.clip(g["err_hi"] - g["value"], 0, None)])
            ok = np.isfinite(yerr).all(axis=0)
            ax.errorbar(g["x"].to_numpy()[ok], g["value"].to_numpy()[ok], yerr=yerr[:, ok],
                        fmt="none", ecolor=cohort.color, elinewidth=1.0, capsize=2.5, zorder=2)
            pre_c = pre[pre["cohort"] == cohort.name]
            if len(pre_c):
                ref = st.group_summary(pre_c.assign(_all=1), by=["_all"], metric=metric)
                ax.axhline(float(ref["value"].iloc[0]), color=cohort.color, linewidth=1.0,
                           linestyle=style["ls"], zorder=1.5,
                           label=f"{cohort.label}, pre (all flies)")
                plotted.append({"panel": co.name, "series": cohort.name, "layer": "pre",
                                "value": float(ref["value"].iloc[0]),
                                "n_flies": int(ref["n_flies"].iloc[0])})
            for _, r in g.iterrows():
                plotted.append({"panel": co.name, "series": cohort.name, "layer": "post_bin",
                                "bin": str(r["bin"]), "x": r["x"], "value": r["value"],
                                "err_lo": r["err_lo"], "err_hi": r["err_hi"],
                                "n_flies": r["n_flies"], "n_trials": r["n_trials"]})
        t = table[(table["cohort"] == co.name)] if len(table) else table
        lines = []
        for _, r in t.iterrows():
            s = st.stars(r["p_holm"])
            word = "exposed slope" if r["test_name"].startswith("slope in") else "vs control"
            lines.append(f"{word}: {_f(r['effect'], 2)}/10 min{(' ' + s) if s else ''}")
        if lines:
            ax.text(0.98, 0.98, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.5, color=pl.INK_PRIMARY)
        ax.set_ylim(lims[0], lims[1] + 0.05 * span)
        ax.set_xlim(0, edges[-1])
        ax.set_xlabel("Minutes since the last exposure pulse")
        ax.set_title(f"{co.odor}: {co.label} vs control", loc="left", fontsize=8.5,
                     color=pl.INK_PRIMARY, pad=4)
        if j == 0:
            ax.set_ylabel(_ylabel(metric, "level"))
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), frameon=False, fontsize=7,
                  labelcolor=pl.INK_SECONDARY)
    n_flies = int(post[post["cohort"].isin([c.name for c in exposed] + [CONTROL])]["fly_id"].nunique())
    used = sum(len(post[(post["cohort"].isin([co.name, CONTROL])) & (post["odor"] == co.odor)])
               for co in exposed)
    pl.title_block(fig, f"Post-exposure {METRIC_WORD[metric]} to the exposure odor vs time "
                        "since exposure",
                   pl.stats_subtitle(n_flies=n_flies, n_trials=used,
                                     test=("GEE logistic" if metric == "rate" else "MixedLM")
                                     + " on minutes, clustered by batch",
                                     correction="Holm across cohorts"))
    unit = "log-OR" if metric == "rate" else "score points"
    pl.caption(fig, f"{pl.HOLM_CAPTION}. Points = post-exposure trials binned by minutes since "
                    "the last exposure pulse (x = bin mean); "
                    + ("% responding, 95% Wilson CI" if metric == "rate" else
                       "mean score ± SEM over flies")
                    + f". Panel text: model slope ({unit} per 10 min). Odor order is shuffled "
                    "per batch, so time since exposure is a batch property (models clustered "
                    "by batch; batch-level p in the table).")
    paths = pl.save_figure(fig, pd.DataFrame(plotted), stem)
    return paths["png"]


def plot_suppression(d: pd.DataFrame, table: pd.DataFrame, metric: str, stem: Path) -> Path:
    info = cohort_info(d)
    ch = fly_changes(d, metric)
    k = SCALE[metric]
    rows = []
    for _, r in table.iterrows():
        on = ch[ch["odor"] == r["odor"]]
        for cond, sub in (("self", on[on["cohort"] == r["self_cohort"]]),
                          ("others", on[on["cohort"] != r["self_cohort"]])):
            v = sub["change"].to_numpy(float) * k
            lo, hi = st.bootstrap_ci(v) if v.size >= 2 else (np.nan, np.nan)
            rows.append({"odor": r["odor"], "condition": cond, "value": float(np.mean(v))
                         if v.size else np.nan, "err_lo": lo, "err_hi": hi,
                         "interval": "BCa 95% CI over flies", "n_flies": int(v.size),
                         "n_trials": int(sub["n_trials"].sum())})
    bars = pd.DataFrame(rows)
    vals = pd.concat([bars["err_lo"], bars["err_hi"], bars["value"]]).astype(float)
    pl.register_ylim(f"change_{metric}", SECTION, data=vals[np.isfinite(vals)].tolist() or [0])
    fig, ax = pl.new_figure(figsize=(7.6, 5.2))
    labels = {r["odor"]: f"{r['odor']}\n(self: {info[r['self_cohort']].code})"
              for _, r in table.iterrows()}
    drawn = pl.paired_bars(ax, bars, conditions=["self", "others"], metric=f"change_{metric}",
                           groups=list(table["odor"]), experiment=SECTION,
                           colors={"self": cfg.CONDITION_COLORS["post"],
                                   "others": cfg.CONDITION_COLORS["pre"]},
                           condition_labels={"self": "self-exposed cohort",
                                             "others": "all other cohorts pooled"},
                           group_labels=labels, ylabel=_ylabel(metric, "change"))
    comps = pd.DataFrame({"group": table["odor"], "cond_a": "self", "cond_b": "others",
                          "p_holm": table["p_holm"]})
    stars = pl.add_holm_stars(ax, comps, plotted=drawn, group_col="odor",
                              condition_col="condition")
    n_flies = int(_test_rows(d)["fly_id"].nunique())
    pl.title_block(fig, f"Self-exposed cohort vs all others pooled: {METRIC_WORD[metric]} "
                        "change",
                   pl.stats_subtitle(n_flies=n_flies, n_trials=int(bars["n_trials"].sum()),
                                     test="DiD of within-fly change, fly-label permutation",
                                     correction="Holm across the 3 exposure odors"))
    pl.caption(fig, f"{pl.HOLM_CAPTION}. Bars = mean within-fly post − pre change over flies, "
                    "whisker = BCa 95% CI over flies. Others pooled = every other exp1 cohort, "
                    "Paraffin control included. Whole-batch permutation p in the table.")
    plotted = pd.concat([drawn, stars.assign(layer="stars")], ignore_index=True)
    paths = pl.save_figure(fig, plotted, stem)
    return paths["png"]


# ═════════════════════════════════════════════════════════════════════════════
# findings
# ═════════════════════════════════════════════════════════════════════════════
def _finding(analysis: str, text: str, numbers: Mapping, primary: bool = False) -> dict:
    return {"section": SECTION, "analysis": analysis, "primary": bool(primary),
            "text": " ".join(text.split()), "numbers": {k: _num(v) for k, v in numbers.items()}}


def _did_text(r: Mapping, metric: str) -> str:
    dg = _digits(metric)
    return (f"{METRIC_WORD[metric]} DiD {_f(r['effect'], dg)} {UNIT_SHORT[metric]} "
            f"(95% CI {_f(r['ci_lo'], dg)} to {_f(r['ci_hi'], dg)}; "
            f"n = {int(r['n_flies_exposed'])} exposed / {int(r['n_flies_control'])} control flies; "
            f"Holm p {_p_eq(r['p_holm'])}; p_batch {_p_eq(r['p_batch'])})"
            + (" [fly-level and batch-level p disagree at 0.05]"
               if _disagrees(r["p"], r["p_batch"]) is True else ""))


def _findings_b(d: pd.DataFrame, did: Mapping[str, pd.DataFrame], models: pd.DataFrame,
                info: Mapping[str, Cohort]) -> list[dict]:
    out = []
    conc = _concurrency_caption(d, info)
    for co in _exposed(info):
        rr = {m: did[m][(did[m]["cohort"] == co.name) & (did[m]["odor"] == co.odor)]
              for m in METRICS}
        if any(v.empty for v in rr.values()):
            continue
        rr = {m: v.iloc[0] for m, v in rr.items()}
        nums = {}
        for m in METRICS:
            for key in ("effect", "ci_lo", "ci_hi", "p", "p_holm", "p_batch",
                        "n_flies_exposed", "n_flies_control", "change_exposed",
                        "change_control"):
                nums[f"{m}_{key}"] = rr[m][key]
            nums[f"{m}_batch_disagrees"] = _disagrees(rr[m]["p"], rr[m]["p_batch"])
        model_txt = ""
        if len(models):
            for kind in ("gee", "glmm", "gee_odor", "glmm_odor", "ordinal"):
                mr = models[(models["cohort"] == co.name) & (models["odor"] == co.odor)
                            & (models["model"] == kind)]
                if mr.empty:
                    continue
                mr = mr.iloc[0]
                for key in ("effect", "ci_lo", "ci_hi", "p", "p_holm", "status"):
                    nums[f"model_{kind}_{key}"] = mr[key]
            g = models[(models["cohort"] == co.name) & (models["odor"] == co.odor)
                       & (models["model"] == "gee")]
            if len(g):
                g = g.iloc[0]
                model_txt = (f" GEE logistic: log-OR {_f(g['effect'], 2)} (95% CI "
                             f"{_f(g['ci_lo'], 2)} to {_f(g['ci_hi'], 2)}, Holm p "
                             f"{_p_eq(g['p_holm'])}).")
        text = (f"Primary contrast {co.code} cohort x {co.odor}: change in the exposed cohort "
                f"minus change in the Paraffin control. {_did_text(rr['rate'], 'rate')}; "
                f"{_did_text(rr['score'], 'score')}.{model_txt} {conc}")
        out.append(_finding("b_did_vs_paraffin", text, nums, primary=True))
    if len(models):
        lines = []
        names = {"gee": "GEE full", "glmm": "mixed logistic (VB) full",
                 "gee_odor": "GEE per odor", "glmm_odor": "mixed logistic (VB) per odor",
                 "ordinal": "ordinal score per odor"}
        for kind in ("gee", "glmm", "gee_odor", "glmm_odor", "ordinal"):
            sub = models[(models["model"] == kind) & models["primary"]]
            if sub.empty:
                continue
            lines.append(f"{names[kind]}: " + ", ".join(
                f"{info[r['cohort']].code}x{r['odor']} {_f(r['effect'], 2)} (Holm p "
                f"{_p_eq(r['p_holm'])}{'' if r['status'] == 'ok' else ', ' + str(r['status'])})"
                for _, r in sub.iterrows()))
        flips = []
        for co in _exposed(info):
            e = {k: models[(models["cohort"] == co.name) & (models["odor"] == co.odor)
                           & (models["model"] == k)]["effect"] for k in ("gee", "glmm")}
            if all(len(v) for v in e.values()) and \
                    np.sign(float(e["gee"].iloc[0])) != np.sign(float(e["glmm"].iloc[0])):
                flips.append(f"{co.code}x{co.odor}")
        flip_txt = (f" The full-model mixed logistic disagrees in SIGN with the GEE for "
                    f"{', '.join(flips)}." if flips else "")
        out.append(_finding("b_did_models",
                            "Model-based primary DiD contrasts (log odds ratios; ordinal = "
                            "score): " + "; ".join(lines) + "." + flip_txt + " The Bayesian "
                            "mixed logistic is a variational approximation whose posterior SDs "
                            "run far narrower than the GEE robust SEs; read its p with "
                            "caution and prefer the GEE.",
                            {f"{r['model']}_{info[r['cohort']].code}_{_slug(r['odor'])}_{k}": r[k]
                             for _, r in models.iterrows() for k in ("effect", "p", "p_holm")}))
        expl = {m: did[m][~did[m]["primary"] & (did[m]["p_holm"] < ALPHA)] for m in METRICS}
        txt = "; ".join(f"{m}: " + (", ".join(f"{info[r['cohort']].code}x{r['odor']} "
                                               f"{_f(r['effect'], _digits(m))} (Holm p "
                                               f"{_p_eq(r['p_holm'])})"
                                               for _, r in t.iterrows()) or "none")
                        for m, t in expl.items())
        out.append(_finding("b_did_exploratory",
                            f"Exploratory DiDs significant after per-panel Holm: {txt}.",
                            {f"n_significant_{m}": len(t) for m, t in expl.items()}))
    return out


def _findings_a(tables: Mapping[str, pd.DataFrame], info: Mapping[str, Cohort]) -> list[dict]:
    out = []
    for m, t in tables.items():
        ex = t[t["exposed_odor"] & t["p"].notna()]
        parts = [f"{info[r['cohort']].code} on {r['odor']} {_f(r['pre_mean'], _digits(m), False)}"
                 f" -> {_f(r['post_mean'], _digits(m), False)} ({_f(r['effect'], _digits(m))} "
                 f"{UNIT_SHORT[m]}, 95% CI {_f(r['ci_lo'], _digits(m))} to "
                 f"{_f(r['ci_hi'], _digits(m))}, n = {int(r['n_flies'])}, Holm p "
                 f"{_p_eq(r['p_holm'])}, p_batch {_p_eq(r['p_batch'])})" + _fragile(r)
                 for _, r in ex.iterrows()]
        sig = t[t["p_holm"] < ALPHA]
        sig_txt = ", ".join(f"{info[r['cohort']].code} {r['odor']} {_f(r['effect'], _digits(m))}"
                            for _, r in sig.iterrows()) or "none"
        out.append(_finding("a_pre_vs_post",
                            f"Pre vs post ({METRIC_WORD[m]}), exposure odor in its own cohort: "
                            + "; ".join(parts) + f". Significant after Holm within cohort: "
                            f"{len(sig)} of {int(t['p'].notna().sum())} odor x cohort tests "
                            f"({sig_txt}).",
                            {f"{info[r['cohort']].code}_{_slug(r['odor'])}_{k}": r[k]
                             for _, r in ex.iterrows()
                             for k in ("effect", "ci_lo", "ci_hi", "p_holm", "p_batch")}
                            | {"n_significant": len(sig)}))
    return out


def _findings_c(drift: Mapping[str, tuple], info: Mapping[str, Cohort]) -> list[dict]:
    out = []
    for m, (tests, trend, _) in drift.items():
        dg = _digits(m)
        w = tests[tests["family"] == "drift vs 0 within cohort"]
        b = tests[tests["family"] == "exposed cohort vs Paraffin control"]
        o = tests[tests["family"] == "omnibus"]
        wt = "; ".join(f"{info[r['cohort']].code} {_f(r['effect'], dg)} {UNIT_SHORT[m]} (95% CI "
                       f"{_f(r['ci_lo'], dg)} to {_f(r['ci_hi'], dg)}, n = {int(r['n_flies'])}, "
                       f"Holm p {_p_eq(r['p_holm'])}, p_batch {_p_eq(r['p_batch'])})"
                       + (" [p/p_batch disagree]" if r["batch_disagrees"] is True else "")
                       for _, r in w.iterrows())
        bt = "; ".join(f"{info[r['cohort']].code} - control {_f(r['effect'], dg)} (95% CI "
                       f"{_f(r['ci_lo'], dg)} to {_f(r['ci_hi'], dg)}, Holm p "
                       f"{_p_eq(r['p_holm'])}, p_batch {_p_eq(r['p_batch'])})"
                       + (" [p/p_batch disagree]" if r["batch_disagrees"] is True else "")
                       for _, r in b.iterrows())
        ot = (f" Omnibus across cohorts: p {_p_eq(o['p'].iloc[0])}, p_batch "
              f"{_p_eq(o['p_batch'].iloc[0])}." if len(o) else "")
        tt = "; ".join(f"{('pooled' if r['cohort'] == 'all' else info[r['cohort']].code)} "
                       f"{_f(r['effect'], dg)} {UNIT_SHORT[m]}/h (batch rho "
                       f"{_f(r['rho_batch'], 2)}, p_batch {_p_eq(r['p_batch'])}; fly-level "
                       f"Holm p {_p_eq(r['p_holm'])})" for _, r in trend.iterrows())
        nums = {f"{info[r['cohort']].code}_{k}": r[k] for _, r in w.iterrows()
                for k in ("effect", "ci_lo", "ci_hi", "p", "p_holm", "p_batch")}
        nums |= {f"{info[r['cohort']].code}_vs_control_{k}": r[k] for _, r in b.iterrows()
                 for k in ("effect", "ci_lo", "ci_hi", "p_holm", "p_batch")}
        nums |= {f"trend_{r['cohort'] if r['cohort'] == 'all' else info[r['cohort']].code}_{k}": r[k]
                 for _, r in trend.iterrows() for k in ("effect", "rho_batch", "p_batch", "p_holm")}
        if len(o):
            nums |= {"omnibus_p": o["p"].iloc[0], "omnibus_p_batch": o["p_batch"].iloc[0]}
        out.append(_finding("c_global_drift",
                            f"Global drift ({METRIC_WORD[m]}, per-fly mean post - pre over the "
                            f"shared odors): {wt}. Vs control: {bt}.{ot} Drift vs start time: "
                            f"{tt}.", nums))
    return out


def _findings_d(tables: Mapping[str, pd.DataFrame], info: Mapping[str, Cohort]) -> list[dict]:
    out = []
    for m, t in tables.items():
        if t.empty:
            continue
        unit = "log-OR" if m == "rate" else "score points"
        txt = "; ".join(f"{info[r['cohort']].code} {r['test_name']}: {_f(r['effect'], 3)} {unit}"
                        f"/10 min (95% CI {_f(r['ci_lo'], 3)} to {_f(r['ci_hi'], 3)}, n = "
                        f"{int(r['n_flies'])} flies / {int(r['n_batches'])} batches, Holm p "
                        f"{_p_eq(r['p_holm'])}, p_batch {_p_eq(r['p_batch'])})"
                        + (" [p/p_batch disagree]" if r["batch_disagrees"] is True else "")
                        for _, r in t.iterrows())
        nums = {f"{info[r['cohort']].code}_{'slope' if r['test_name'].startswith('slope in') else 'vs_control'}_{k}": r[k]
                for _, r in t.iterrows() for k in ("effect", "ci_lo", "ci_hi", "p_holm", "p_batch")}
        out.append(_finding("d_time_since_exposure",
                            f"Time since exposure ({METRIC_WORD[m]} to the exposure odor, "
                            f"{'GEE logistic' if m == 'rate' else 'MixedLM'} clustered by "
                            f"batch): {txt}.", nums))
    return out


def _findings_e(tables: Mapping[str, pd.DataFrame], info: Mapping[str, Cohort]) -> list[dict]:
    out = []
    for m, t in tables.items():
        dg = _digits(m)
        txt = "; ".join(f"{r['odor']}: self {_f(r['change_self'], dg)} vs others "
                        f"{_f(r['change_others'], dg)} {UNIT_SHORT[m]}, DiD {_f(r['effect'], dg)}"
                        f" (95% CI {_f(r['ci_lo'], dg)} to {_f(r['ci_hi'], dg)}, n = "
                        f"{int(r['n_flies_exposed'])}/{int(r['n_flies_control'])}, Holm p "
                        f"{_p_eq(r['p_holm'])}, p_batch {_p_eq(r['p_batch'])})"
                        + (" [p/p_batch disagree]" if r["batch_disagrees"] is True else "")
                        for _, r in t.iterrows() if np.isfinite(float(r.get("effect", np.nan))))
        nums = {f"{_slug(r['odor'])}_{k}": r[k] for _, r in t.iterrows()
                for k in ("effect", "ci_lo", "ci_hi", "p_holm", "p_batch")}
        out.append(_finding("e_odor_specific_suppression",
                            f"Odor-specific change ({METRIC_WORD[m]}), self-exposed cohort vs "
                            f"all other cohorts pooled: {txt}.", nums))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# run
# ═════════════════════════════════════════════════════════════════════════════
def _write_readmes(root: Path, d: pd.DataFrame, info: Mapping[str, Cohort]) -> None:
    conc = _concurrency_caption(d, info)
    _readme(root, [
        "# 02 Exp 1: exposure cohorts (pre-test -> 6 x 30 s exposure -> post-test)",
        "Cohorts IAA (1), Hex (0.1), 3Oct (0.1) and the Paraffin (100) control, whose "
        "exposure delivered no odor. Unit = fly; stars from Holm-adjusted p only; every "
        "stats table (with p_batch, batches = folders) is in `tables/<figure name>.csv`.",
        f"b_did_vs_paraffin holds the PRIMARY result. {conc} The IAA cohort never ran "
        "Paraffin Oil; Hex and 3Oct ran it for some flies only."])
    _readme(root / SUBDIRS["a"], [
        "# a. Pre vs post per test odor x cohort",
        "Bars pool trials (rate: % with 95% Wilson CI; score: mean ± SEM), grey lines = flies, "
        "dashed box = the cohort's exposure odor. Paired sign-flip per odor (BCa CI), Holm "
        "within each cohort panel; whole-batch sign-flip p_batch in the table."])
    _readme(root / SUBDIRS["b"], [
        "# b. Difference in differences vs the Paraffin control (PRIMARY)",
        "DiD = (post − pre, exposed cohort) − (post − pre, Paraffin control) per test odor, "
        "fly-bootstrap BCa CI, fly-label and whole-batch permutation p. Primary = each cohort "
        "on its own exposure odor (Holm across 3); the rest are exploratory (Holm per panel).",
        "Model versions (GEE, Bayesian mixed logistic, ordinal score) in "
        "`tables/exp1_did_vs_paraffin_models.csv` and `../models/`. The control was not run "
        "concurrently with the IAA cohort."])
    _readme(root / SUBDIRS["c"], [
        "# c. Global drift",
        "Per fly: mean post − pre over the odors every cohort ran (Paraffin Oil left out). "
        "Tested vs 0 per cohort (sign-flip + batch sign-flip) and between cohorts (fly and "
        "whole-batch permutation), and against fly start time (`b2b_stats.batch_trend`)."])
    _readme(root / SUBDIRS["d"], [
        "# d. Time since exposure",
        "Post-exposure response to each cohort's exposure odor vs minutes since the last "
        "exposure pulse, against the Paraffin control on the same odor. Models (GEE logistic, "
        "MixedLM score) are clustered by batch: odor order, and so time since exposure, is "
        "set per batch folder. Binned points are for display only."])
    _readme(root / SUBDIRS["e"], [
        "# e. Odor-specific suppression",
        "Per exposure odor: the self-exposed cohort's within-fly change vs every other "
        "cohort's (pooled, Paraffin control included), as a DiD with fly-bootstrap CI and "
        "fly / whole-batch permutation p; Holm across the 3 odors."])
    _readme(root / "models", [
        "# Exp 1 models",
        "Every fit via `stats.save_model`: `<name>.txt` (summary, with NOTES on covariates "
        "left out and why) and `<name>_coefs.csv`. Check the `status` column before use.",
        "exposure_conc is aliased with the cohort term (one concentration per cohort) and "
        "position_in_block with minutes since exposure, so neither enters beside them."])


def run(trials: pd.DataFrame, out: Path, *, tables: Path, n_boot: int | None = None,
        n_perm: int | None = None, seed: int | None = None, ordinal: bool = True,
        ordinal_boot: int = 200) -> list[dict]:
    """Run analyses a-e on ``trials`` (the tidy table), writing figures under
    ``<out>/02_exp1_exposure_cohorts/``, tables under ``tables`` and models under
    ``.../models/``. ``n_boot`` / ``n_perm`` / ``seed`` default to config; ``ordinal`` /
    ``ordinal_boot`` control the ordinal score model. Returns the findings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out, tables = Path(out), Path(tables)
    root = out / ROOT_DIR
    models_dir = root / "models"
    opts = {"n_boot": n_boot, "n_perm": n_perm, "seed": seed}
    pl.reset_ylims(SECTION)
    d = prepare(trials)
    if d.empty:
        log.warning("exp1: no data; nothing written")
        return []
    info = cohort_info(d)
    log.info("exp1: %d trials, %d flies, cohorts %s", len(d), d["fly_id"].nunique(),
             ", ".join(f"{c.label} n={d.loc[d['cohort'] == c.name, 'fly_id'].nunique()}"
                       for c in info.values()))
    _write_readmes(root, d, info)
    if not (tables / "README.md").exists():
        _readme(tables, ["# Stats tables",
                         "One CSV per figure (same base name), plus model contrasts. Columns "
                         "include n_flies, effect, ci_lo, ci_hi, p, p_holm, q_bh, p_batch, "
                         "test, correction; p_batch permutes whole batch folders."])
    findings: list[dict] = []

    log.info("exp1 a: pre vs post")
    a_tabs = {}
    for m in METRICS:
        a_tabs[m] = pre_vs_post_table(d, m, **opts)
        stem = f"exp1_pre_vs_post_{m}"
        _write_table(a_tabs[m], tables, stem, lead=("cohort", "cohort_label", "odor",
                                                    "exposed_odor", "metric", "unit"))
        plot_pre_vs_post(d, a_tabs[m], m, root / SUBDIRS["a"] / stem)
    findings += _findings_a(a_tabs, info)

    log.info("exp1 b: DiD vs the Paraffin control (primary)")
    did = {}
    for m in METRICS:
        did[m] = did_vs_control_table(d, m, **opts)
        stem = f"exp1_did_vs_paraffin_{m}"
        _write_table(did[m], tables, stem, lead=("cohort", "cohort_label", "odor", "primary",
                                                 "family", "metric", "unit"))
        plot_did(d, did[m], m, root / SUBDIRS["b"] / stem)
    models = did_models(d, models_dir, did_tables=did, ordinal=ordinal,
                        ordinal_boot=ordinal_boot, seed=seed)
    _write_table(models, tables, "exp1_did_vs_paraffin_models",
                 lead=("cohort", "cohort_label", "odor", "primary", "model", "family"))
    findings += _findings_b(d, did, models, info)

    log.info("exp1 c: global drift")
    drift = {m: drift_tables(d, m, **opts) for m in METRICS}
    for m, (tests, trend, dr) in drift.items():
        data = np.r_[dr["drift"].to_numpy(float) * SCALE[m], tests["ci_lo"].astype(float),
                     tests["ci_hi"].astype(float)]
        pl.register_ylim(f"drift_{m}", SECTION, data=data[np.isfinite(data)].tolist() or [0])
        _write_table(tests, tables, f"exp1_global_drift_{m}",
                     lead=("comparison", "cohort", "cohort_label", "family", "metric", "unit"))
        _write_table(trend, tables, f"exp1_global_drift_vs_start_{m}",
                     lead=("cohort", "cohort_label", "family", "metric", "unit"))
        plot_drift(dr, tests, m, root / SUBDIRS["c"] / f"exp1_global_drift_{m}", info)
        plot_drift_vs_start(dr, trend, m, root / SUBDIRS["c"] / f"exp1_global_drift_vs_start_{m}",
                            info)
    findings += _findings_c(drift, info)

    log.info("exp1 d: time since exposure")
    d_tabs = time_since_exposure_tables(d, models_dir, n_perm=n_perm, seed=seed)
    for m in METRICS:
        stem = f"exp1_time_since_exposure_{m}"
        _write_table(d_tabs[m], tables, stem, lead=("cohort", "cohort_label", "odor",
                                                    "test_name", "metric"))
        plot_time_since_exposure(d, d_tabs[m], m, root / SUBDIRS["d"] / stem)
    findings += _findings_d(d_tabs, info)

    log.info("exp1 e: odor-specific suppression")
    e_tabs = {}
    for m in METRICS:
        e_tabs[m] = suppression_table(d, m, **opts)
        stem = f"exp1_odor_specific_suppression_{m}"
        _write_table(e_tabs[m], tables, stem, lead=("odor", "self_cohort", "cohort_label",
                                                    "metric", "unit"))
        plot_suppression(d, e_tabs[m], m, root / SUBDIRS["e"] / stem)
    findings += _findings_e(e_tabs, info)

    plt.close("all")
    log.info("exp1: done, %d findings", len(findings))
    return findings


__all__ = ("COVARIATES", "Cohort", "TABLE_COLUMNS", "add_covariates", "cohort_info",
           "common_odors", "did_models", "did_vs_control_table", "drift_tables", "fly_changes",
           "fly_drift", "pre_vs_post_table", "prepare", "run", "suppression_table",
           "time_since_exposure_tables")
