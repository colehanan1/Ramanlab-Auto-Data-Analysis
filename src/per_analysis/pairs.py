"""Shared helpers of the IAA odor-pair analyses: exp2 (IAA first) and exp3 (IAA second).

Design of both cohorts
----------------------
An alone block (~5-35 min into the session: each of the 8 odors once, the click first)
and a pair block (~65-140 min: odor 1 for 10 s, a 1 s gap, then odor 2, scored against the
baseline before odor 1; each partner twice, ``presentation`` 1/2). "Paraffin Oil" in these
sessions is a 50 ms master-valve CLICK with no odor (``is_click``): the within-block,
time-matched reference. "Alone vs pair" therefore also holds time in the rig and
re-testing; the click reference controls for both.

* exp2: odor 1 = IAA, odor 2 = X (scored). ``group`` = X (= ``partner``).
* exp3: odor 1 = X, odor 2 = IAA (scored). ``group`` = X (= ``prime``). IAA's ONE alone
  trial is the baseline of every X, so :func:`scored_long` copies it once per X. In exp3
  the controlled change (X pair - IAA alone) - (click pair - IAA alone) is therefore
  identical to the direct form (IAA after X - IAA after the click).

Data rules
----------
:func:`prepare` leaves out the light probes (``is_light_probe``) and the alone trials with
an odor window under 1 s (the four 0.025 s clicks of november_13_batch_1_rig_3), logged.
The fly is the unit: paired tests take per-fly means (``stats.paired_comparisons``:
two-sided exact sign-flip, BCa CI, whole-batch ``p_batch``); bars pool trials
(``stats.group_summary``). Every analysis runs per clock window (:func:`split_windows`);
file names are ``<exp>_<analysis>_<metric>_<window>`` (:func:`stem`).

Table columns (every comparison table; ``tables/<stem>.csv``)
-------------------------------------------------------------
``effect`` = mean over flies of the within-fly difference b - a (the sign-flip test's
estimand, with its 95 % BCa CI ``ci_lo``/``ci_hi``); ``effect_pooled`` = the difference of
the trial-pooled bar heights (``pooled_b - pooled_a``; equal to ``effect`` when every fly
has the same number of trials per condition). Rate: percentage points; score: score
units. ``p_holm`` / ``q_bh`` within the figure's family; ``p_batch``: whole batch folders
flipped together; ``p_batch_disagrees``: p and p_batch fall on opposite sides of 0.05.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from . import config as cfg  # noqa: E402  (not `C`: patsy's C(...) must stay free)
from . import plots as pl  # noqa: E402
from . import stats as st  # noqa: E402

LOG = logging.getLogger(__name__)

REFERENCE = cfg.CLICK_ODOR                     # "Paraffin Oil": the master-valve click
IAA = "IAA"
ALONE, PAIR = "alone", "pair"
PRES_OF = {1: "1st", 2: "2nd"}
METRICS = ("rate", "score")
SHORT_WINDOW_S = 1.0
P_ALPHA = 0.05

REQUIRED = ("experiment", "fly_id", "batch", "window", "block", "odor", "prime", "partner",
            "presentation", "is_light_probe", "is_click", "odor_window_s", "score",
            "responded", "odor1_score", "start_time_h", "position_in_block",
            "minutes_since_block_start", "scored_label")

TABLE_COLUMNS = ["experiment", "window", "analysis", "metric", "unit", "group", "a", "b",
                 "n_flies", "n_batches", "n_trials_a", "n_trials_b", "pooled_a", "pooled_b",
                 "effect", "ci_lo", "ci_hi", "effect_pooled", "p", "p_method", "p_holm",
                 "q_bh", "p_batch", "p_batch_holm", "p_batch_disagrees", "test",
                 "correction"]

TEST_PAIRED = "two-sided exact sign-flip on per-fly means"
TEST_DIRECTION = ("joint sign-flip of each fly's whole row of changes "
                  "(stats.direction_count_test)")
BARS_NOTE = ("Bars pool trials: rate = % of trials responding (95 % Wilson CI, flies as "
             "unit); score = pooled mean ± SEM over flies. Thin lines: one fly's mean per "
             "condition. Tests: two-sided exact sign-flip on per-fly means; 95 % BCa CIs "
             "and the whole-batch p are in the table.")

#: plots.HOLM_CAPTION, capitalised to follow a sentence.
HOLM_SENTENCE = pl.HOLM_CAPTION[0].upper() + pl.HOLM_CAPTION[1:] + "."

GROUP_LABELS = {REFERENCE: "Click\n(Paraffin Oil)", "Ethyl Butyrate": "Ethyl\nButyrate",
                "Benzaldehyde": "Benz-\naldehyde"}
EFFECT_COLORS = (pl.CATEGORICAL[0], pl.CATEGORICAL[1])
_UNIT = {"rate": "percentage points", "score": "score"}
_SHORT_UNIT = {"rate": "pp", "score": ""}


# ── small utilities ──────────────────────────────────────────────────────────
def value_col(metric: str) -> str:
    if metric not in METRICS:
        raise ValueError(f"metric must be 'rate' or 'score': {metric!r}")
    return "responded" if metric == "rate" else "score"


def scale_of(metric: str) -> float:
    return 100.0 if metric == "rate" else 1.0


def stem(experiment: str, analysis: str, metric: str, window: str) -> str:
    """``<exp>_<analysis>_<metric>_<window>``, e.g. exp3_vs_paraffin_reference_rate_day."""
    return f"{experiment}_{analysis}_{metric}_{window}"


def window_text(name: str, windows: Sequence[cfg.Window] | None = None) -> str:
    for w in windows or cfg.WINDOWS:
        if w.name == name:
            return f"{name} window (folder starts {w.start:%H:%M}-{w.end:%H:%M})"
    return f"{name} window"


def group_order(values) -> list:
    return pl.odor_order(list(values))


def design_groups(long: pd.DataFrame) -> list:
    """The design's partners (config ODOR_ORDER: the 7 odors + the click) plus any other
    group present, in odor order: a partner whose trials were all left out keeps its
    table row (n_flies 0)."""
    return group_order(list(cfg.ODOR_ORDER) + list(long["group"].dropna().unique()))


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("float64") if pd.api.types.is_bool_dtype(s) else s,
                         errors="coerce").astype(float)


def jsonable(x: Any) -> Any:
    """numpy / NaN -> plain JSON values (NaN -> None), recursively."""
    if isinstance(x, Mapping):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else float(x)
    return x


def write_readme(folder: Path, lines: Sequence[str]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "README.md"
    path.write_text("\n".join(lines).rstrip() + "\n")
    return path


# ── data preparation ─────────────────────────────────────────────────────────
def prepare(trials: pd.DataFrame, experiment: str, *, log: logging.Logger | None = None,
            threshold: float | None = None) -> pd.DataFrame:
    """The experiment's rows, light probes and short alone clicks left out (logged), with
    ``condition`` (alone/pair), ``pres`` (alone/1st/2nd), ``second_presentation`` (0/1),
    ``odor1_responded`` (pair: odor-1 score >= threshold, NaN when unscored; alone: 0, no
    odor 1 came before) and ``_uid`` (one id per scored trial)."""
    log = log or LOG
    thr = cfg.RESPONSE_THRESHOLD if threshold is None else threshold
    missing = [c for c in REQUIRED if c not in trials.columns]
    if missing:
        raise KeyError(f"trial table lacks {missing}")
    d = trials[trials["experiment"] == experiment].copy()
    light = d["is_light_probe"].astype(bool).to_numpy()
    win = pd.to_numeric(d["odor_window_s"], errors="coerce").to_numpy(float)
    short = (d["block"].to_numpy() == ALONE) & (win < SHORT_WINDOW_S) & ~light
    log.info("%s: leaving out %d light-probe rows (%d flies; no odor) and %d alone rows "
             "with an odor window < %g s (%s)", experiment, int(light.sum()),
             d.loc[light, "fly_id"].nunique(), int(short.sum()), SHORT_WINDOW_S,
             ", ".join(sorted(set(d.loc[short, "batch"].astype(str) + ":"
                                  + d.loc[short, "trial_label"].astype(str)))) or "none")
    d = d[~light & ~short].copy()
    bad = ~d["block"].isin([ALONE, PAIR])
    if bad.any():
        log.warning("%s: %d rows in blocks other than alone/pair left out: %s", experiment,
                    int(bad.sum()), sorted(set(d.loc[bad, "block"])))
        d = d[~bad].copy()
    d["condition"] = d["block"].astype(str)
    pres = pd.to_numeric(d["presentation"], errors="coerce")
    d["pres"] = np.where(d["condition"] == ALONE, ALONE,
                         pres.map(lambda v: PRES_OF.get(int(v), str(v)) if np.isfinite(v)
                                  else "?").to_numpy(object))
    d["second_presentation"] = (pres == 2).astype(int)
    o1 = pd.to_numeric(d["odor1_score"], errors="coerce")
    d["odor1_responded"] = np.where(d["condition"] == ALONE, 0.0,
                                    np.where(o1.notna(), (o1 >= thr).astype(float), np.nan))
    d["_uid"] = d["fly_id"].astype(str) + "|" + d["scored_label"].astype(str)
    log.debug("%s prepared: %s, %d flies", experiment, d.shape, d["fly_id"].nunique())
    return d


def split_windows(df: pd.DataFrame, *, log: logging.Logger | None = None,
                  windows: Sequence[cfg.Window] | None = None) -> list[tuple[str, pd.DataFrame]]:
    """[(window, rows)] in config order; a window with no flies is skipped (logged), and
    rows outside every window ('other') are left out (logged)."""
    log = log or LOG
    names = [w.name for w in (windows or cfg.WINDOWS)]
    other = df[~df["window"].isin(names)]
    if len(other):
        log.info("leaving out %d rows (%d flies) outside the %s windows (window=%s)",
                 len(other), other["fly_id"].nunique(), "/".join(names),
                 sorted(set(other["window"].astype(str))))
    out = []
    for name in names:
        sub = df[df["window"] == name]
        if sub["fly_id"].nunique() == 0:
            log.info("%s window: no flies, skipped", name)
            continue
        log.debug("%s window: %d rows, %d flies", name, len(sub), sub["fly_id"].nunique())
        out.append((name, sub))
    return out


def scored_long(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """One row per scored trial with ``group`` = the varying odor.

    exp2: group = partner (the scored odor; its alone row is that odor alone).
    exp3: the scored odor is IAA and group = prime; IAA's alone trial is copied once per
    prime present in the pair block (each prime's baseline is that same trial; ``_uid``
    keeps the copies countable as one trial). Other alone odors are not used."""
    if experiment == "exp2":
        out = df.assign(group=df["partner"])
    elif experiment == "exp3":
        pair = df[(df["condition"] == PAIR) & (df["odor"] == IAA)]
        n_other = int(((df["condition"] == PAIR) & (df["odor"] != IAA)).sum())
        if n_other:
            LOG.warning("exp3: %d pair rows not scored on IAA left out", n_other)
        pair = pair.assign(group=pair["prime"])
        alone = df[(df["condition"] == ALONE) & (df["odor"] == IAA)]
        primes = group_order(pair["group"].dropna().unique())
        out = pd.concat([alone.assign(group=p) for p in primes] + [pair], ignore_index=True)
    else:
        raise ValueError(f"not a pair experiment: {experiment!r}")
    return out[out["group"].notna()].reset_index(drop=True)


def drop_carryover(df: pd.DataFrame, *, log: logging.Logger | None = None
                   ) -> tuple[pd.DataFrame, int]:
    """Leave out pair trials whose odor 1 responded (``odor1_responded == 1``), the
    carry-over sensitivity analysis. Pair trials with an unscored odor 1 are kept (logged)."""
    log = log or LOG
    pair = df["condition"] == PAIR
    drop = pair & (df["odor1_responded"] == 1)
    unknown = int((pair & df["odor1_responded"].isna()).sum())
    if unknown:
        log.info("carry-over: %d pair trials without an odor-1 score kept", unknown)
    log.info("carry-over: leaving out %d of %d pair trials whose odor 1 responded",
             int(drop.sum()), int(pair.sum()))
    return df[~drop].copy(), int(drop.sum())


# ── analyses ─────────────────────────────────────────────────────────────────
@dataclass
class Result:
    """One figure's worth of numbers."""
    analysis: str
    metric: str                  # rate | score
    plot_metric: str             # rate | score | rate_change | score_change
    groups: list
    conditions: list
    table: pd.DataFrame          # TABLE_COLUMNS (minus experiment/window, added on write)
    bars: pd.DataFrame           # group, condition, value, err_lo, err_hi, n_flies, n_trials
    per_fly: pd.DataFrame        # group, condition, fly_id, value (plotted units)
    comps: pd.DataFrame          # group, cond_a, cond_b, p_holm
    n_flies: int
    n_trials: int
    test: str
    correction: str
    extra: dict = field(default_factory=dict)


def _complete_cells(d: pd.DataFrame, conditions: Sequence, value: str) -> pd.DataFrame:
    """Rows of the (group, fly) cells that have a scored trial in every condition."""
    d = d[d["condition"].isin(list(conditions)) & _num(d[value]).notna()]
    if d.empty:
        return d
    n = d.groupby(["group", "fly_id"])["condition"].transform("nunique")
    return d[n == len(conditions)]


def _count_family(table: pd.DataFrame) -> int:
    return int(np.isfinite(pd.to_numeric(table["p"], errors="coerce")).sum())


def _disagree(p, pb) -> Any:
    if not (np.isfinite(p) and np.isfinite(pb)):
        return np.nan
    return bool((p < P_ALPHA) != (pb < P_ALPHA))


def contract_table(comp: pd.DataFrame, bars: pd.DataFrame, *, groups: Sequence,
                   metric: str, analysis: str, a: str, b: str, test: str,
                   family: str) -> pd.DataFrame:
    """``stats.paired_comparisons`` output + the bar table -> one row per group in
    ``groups`` (a group without flies keeps a row with n_flies 0 and NaN tests)."""
    sc = scale_of(metric)
    ci = comp.set_index("group") if len(comp) else pd.DataFrame()
    bi = bars.set_index(["group", "condition"]) if len(bars) else pd.DataFrame()
    rows = []
    for g in groups:
        r = {"analysis": analysis, "metric": metric, "unit": _UNIT[metric], "group": g,
             "a": a, "b": b, "n_flies": 0, "n_batches": 0, "effect": np.nan,
             "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan, "p_method": "",
             "p_holm": np.nan, "q_bh": np.nan, "p_batch": np.nan, "p_batch_holm": np.nan}
        if g in ci.index and int(ci.loc[g, "n_flies"]) > 0:
            c = ci.loc[g]
            r.update(n_flies=int(c["n_flies"]), n_batches=int(c["n_batches"]),
                     effect=sc * float(c["mean_diff"]), ci_lo=sc * float(c["ci_lo"]),
                     ci_hi=sc * float(c["ci_hi"]), p=float(c["p"]),
                     p_method=str(c["p_method"]), p_holm=float(c["p_holm"]),
                     q_bh=float(c["q_bh"]), p_batch=float(c["p_batch"]),
                     p_batch_holm=float(c["p_batch_holm"]))
        for side, cond in (("a", a), ("b", b)):
            if len(bi) and (g, cond) in bi.index:
                br = bi.loc[(g, cond)]
                r[f"pooled_{side}"] = float(br["value"])
                r[f"n_trials_{side}"] = int(br["n_trials"])
            else:
                r[f"pooled_{side}"], r[f"n_trials_{side}"] = np.nan, 0
        r["effect_pooled"] = r["pooled_b"] - r["pooled_a"]
        r["p_batch_disagrees"] = _disagree(r["p"], r["p_batch"])
        rows.append(r)
    t = pd.DataFrame(rows)
    k = _count_family(t)
    t["test"] = test
    t["correction"] = f"Holm across {k} {family}"
    return t[[c for c in TABLE_COLUMNS if c in t.columns]]


def paired_analysis(d: pd.DataFrame, metric: str, *, conditions: Sequence, a: str, b: str,
                    groups: Sequence, analysis: str, family: str, test: str = TEST_PAIRED,
                    n_boot: int | None = None, n_perm: int | None = None) -> Result:
    """Bars of every condition per group, per-fly lines, and the paired test b vs a per
    group (Holm across the groups). ``d`` has group, condition, fly_id, batch, _uid."""
    value = value_col(metric)
    sub = _complete_cells(d, conditions, value)
    if sub.empty:
        bars = pd.DataFrame(columns=["group", "condition", "value", "err_lo", "err_hi",
                                     "n_flies", "n_trials"])
        pf = pd.DataFrame(columns=["group", "condition", "fly_id", "value"])
        comp = pd.DataFrame()
    else:
        bars = st.group_summary(sub, by=["group", "condition"], metric=metric, value=value)
        pf = st.per_fly(sub, value, by=["group", "condition"]).rename(columns={value: "value"})
        comp = st.paired_comparisons(sub, value, condition_col="condition", a=a, b=b,
                                     by=["group"], n_boot=n_boot, n_perm=n_perm)
    table = contract_table(comp, bars, groups=groups, metric=metric, analysis=analysis,
                           a=a, b=b, test=test, family=family)
    comps = table.loc[np.isfinite(table["p_holm"].astype(float)), ["group", "p_holm"]] \
        .assign(cond_a=a, cond_b=b)
    return Result(analysis=analysis, metric=metric, plot_metric=metric, groups=list(groups),
                  conditions=list(conditions), table=table, bars=bars, per_fly=pf,
                  comps=comps, n_flies=int(sub["fly_id"].nunique()) if len(sub) else 0,
                  n_trials=int(sub["_uid"].nunique()) if len(sub) else 0, test=test,
                  correction=table["correction"].iloc[0] if len(table) else "Holm")


def alone_vs_pair(long: pd.DataFrame, metric: str, *, family: str = "partners",
                  **kw) -> Result:
    """Each group: alone vs in pair (both presentations pooled within fly)."""
    groups = design_groups(long)
    return paired_analysis(long, metric, conditions=[ALONE, PAIR], a=ALONE, b=PAIR,
                           groups=groups, analysis="alone_vs_pair_pooled", family=family,
                           **kw)


def presentation_1_vs_2(long: pd.DataFrame, metric: str, *, family: str = "partners",
                        **kw) -> Result:
    """Bars alone / 1st / 2nd per group; the paired test is 1st vs 2nd."""
    d = long.assign(condition=long["pres"])
    groups = design_groups(long)
    return paired_analysis(d, metric, conditions=[ALONE, "1st", "2nd"], a="1st", b="2nd",
                           groups=groups, analysis="presentation_1_vs_2", family=family,
                           **kw)


def reference_contrast(long: pd.DataFrame, metric: str, *,
                       analysis: str = "vs_paraffin_reference", reference: str = REFERENCE,
                       family: str = "odors", **kw) -> Result:
    """Within fly, pair rows only: each X's trials vs the reference's (the click) trials.
    exp2: X right after IAA vs the click right after IAA; exp3: IAA right after X vs IAA
    right after the click. Effect = per fly mean(X) - mean(click); Holm across the odors."""
    pair = long[long["condition"] == PAIR]
    ref = pair[pair["group"] == reference]
    groups = [g for g in design_groups(long) if g != reference]
    frames = []
    for g in groups:
        frames.append(pair[pair["group"] == g].assign(condition="odor"))
        frames.append(ref.assign(group=g, condition="click"))
    frame = pd.concat(frames, ignore_index=True) if frames else pair.iloc[0:0]
    return paired_analysis(frame, metric, conditions=["click", "odor"], a="click", b="odor",
                           groups=groups, analysis=analysis, family=family, **kw)


def _changes(long: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(complete alone+pair rows, per (group, fly) change = mean pair - mean alone, raw
    units, with batch)."""
    value = value_col(metric)
    sub = _complete_cells(long, [ALONE, PAIR], value)
    if sub.empty:
        return sub, pd.DataFrame(columns=["group", "fly_id", "change", "batch"])
    pf = st.per_fly(sub, value, by=["group", "condition"])
    w = pf.pivot_table(index=["group", "fly_id"], columns="condition", values=value,
                       aggfunc="first")
    ch = (w[PAIR] - w[ALONE]).rename("change").reset_index()
    batch_of = sub.drop_duplicates("fly_id").set_index("fly_id")["batch"]
    ch["batch"] = ch["fly_id"].map(batch_of)
    return sub, ch


def _pooled_change(sub: pd.DataFrame, group: str, flies, value: str) -> tuple[float, int]:
    """Pooled pair mean - pooled alone mean (sum / trials each) over ``flies``."""
    g = sub[(sub["group"] == group) & sub["fly_id"].isin(flies)]
    v = _num(g[value])
    pair, alone = v[g["condition"] == PAIR], v[g["condition"] == ALONE]
    if pair.empty or alone.empty:
        return np.nan, 0
    return float(pair.mean() - alone.mean()), int(g["_uid"].nunique())


def _sem(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else np.nan


def controlled_change(long: pd.DataFrame, metric: str, *, reference: str = REFERENCE,
                      analysis: str = "vs_paraffin_reference_controlled",
                      family: str = "odors", n_boot: int | None = None,
                      n_perm: int | None = None) -> Result:
    """Per fly: (X pair - X alone) - (click pair - click alone), the odor's change minus the
    click's change in the same fly (removes odor identity and time in the rig). Bars: the
    pooled change of X and of the click over the same flies, ± SEM over flies of the
    per-fly change; lines: per-fly changes (plotted units)."""
    value = value_col(metric)
    sc = scale_of(metric)
    sub, ch = _changes(long, metric)
    groups = [g for g in design_groups(long) if g != reference]
    ref = ch[ch["group"] == reference].set_index("fly_id")
    frames, bars, flies_all, uids = [], [], set(), set()
    for g in groups:
        own = ch[ch["group"] == g].set_index("fly_id")
        flies = own.index.intersection(ref.index)
        if not len(flies):
            continue
        flies_all |= set(flies)
        frames.append(own.loc[flies].reset_index().assign(condition="odor"))
        frames.append(ref.loc[flies].reset_index().assign(group=g, condition="click"))
        for cond, src, per in (("click", reference, ref), ("odor", g, own)):
            v, _ = _pooled_change(sub, src, flies, value)
            sem = _sem(per.loc[flies, "change"].to_numpy(float))
            bars.append({"group": g, "condition": cond, "value": sc * v,
                         "err_lo": sc * (v - sem), "err_hi": sc * (v + sem),
                         "interval": "±SEM over flies of the per-fly change",
                         "n_flies": int(len(flies)),
                         "n_trials": int(sub.loc[(sub["group"] == src)
                                                 & sub["fly_id"].isin(flies), "_uid"]
                                         .nunique())})
        uids |= set(sub.loc[sub["group"].isin([g, reference]) & sub["fly_id"].isin(flies),
                            "_uid"])
    frame = pd.concat(frames, ignore_index=True) if frames else \
        pd.DataFrame(columns=["group", "fly_id", "change", "batch", "condition"])
    comp = (st.paired_comparisons(frame, "change", condition_col="condition", a="click",
                                  b="odor", by=["group"], n_boot=n_boot, n_perm=n_perm)
            if len(frame) else pd.DataFrame())
    bars = pd.DataFrame(bars, columns=["group", "condition", "value", "err_lo", "err_hi",
                                       "interval", "n_flies", "n_trials"])
    # pooled_a / pooled_b = the pooled click / odor change bars (plotted units)
    table = contract_table(comp, bars.replace({"condition": {"click": "click change",
                                                             "odor": "odor change"}}),
                           groups=groups, metric=metric, analysis=analysis,
                           a="click change", b="odor change",
                           test=TEST_PAIRED + " (per-fly changes)", family=family)
    pf = frame.assign(value=sc * frame["change"].astype(float))[
        ["group", "condition", "fly_id", "value"]] if len(frame) else \
        pd.DataFrame(columns=["group", "condition", "fly_id", "value"])
    comps = table.loc[np.isfinite(table["p_holm"].astype(float)), ["group", "p_holm"]] \
        .assign(cond_a="click", cond_b="odor")
    return Result(analysis=analysis, metric=metric, plot_metric=f"{metric}_change",
                  groups=groups, conditions=["click", "odor"], table=table, bars=bars,
                  per_fly=pf, comps=comps, n_flies=len(flies_all), n_trials=len(uids),
                  test=TEST_PAIRED + " (per-fly changes)",
                  correction=table["correction"].iloc[0] if len(table) else "Holm")


def _direction_batch_p(wide: pd.DataFrame, batch_of: pd.Series, n_perm=None) -> dict:
    """The direction-count test with whole batch folders as the unit: every fly of a
    batch takes the same joint sign. The statistic only needs each odor's SUM over flies,
    and with one sign per batch that sum is sum_b s_b * (sum of the batch's rows), so the
    test on per-batch row sums (NaN -> 0, as the fly test does) is exactly that null."""
    sums = wide.fillna(0.0).groupby(wide.index.map(batch_of)).sum()
    return st.direction_count_test(sums, n_perm=n_perm)


def shared_baseline(long: pd.DataFrame, metric: str, *, reference: str = REFERENCE,
                    analysis: str = "shared_baseline_diagnostic", family: str = "odors",
                    n_boot: int | None = None, n_perm: int | None = None) -> Result:
    """Every odor's change (pair - alone) per fly; how many odors go up vs down, against
    the joint-flip null of ``stats.direction_count_test`` (fly level; ``p_batch``: whole
    batches). The click is drawn as the reference and left out of the count."""
    value = value_col(metric)
    sc = scale_of(metric)
    sub, ch = _changes(long, metric)
    all_groups = design_groups(long)
    odors = [g for g in all_groups if g != reference]
    wide = ch[ch["group"].isin(odors)].pivot_table(index="fly_id", columns="group",
                                                   values="change", aggfunc="first")
    wide = wide.reindex(columns=[g for g in odors if g in wide.columns])
    batch_of = sub.drop_duplicates("fly_id").set_index("fly_id")["batch"] if len(sub) \
        else pd.Series(dtype=object)
    if wide.size:
        dirn = st.direction_count_test(wide, n_perm=n_perm)
        dirb = _direction_batch_p(wide, batch_of, n_perm=n_perm)
    else:
        dirn = {"n_up": 0, "n_down": 0, "n_zero": 0, "n_flies": 0, "n_odors": 0,
                "p": np.nan, "p_naive_binomial": np.nan, "exact": False,
                "mean_cross_odor_corr": np.nan}
        dirb = {"p": np.nan, "n_flies": 0}
    per_odor = paired_analysis(sub[sub["group"].isin(odors)], metric,
                               conditions=[ALONE, PAIR], a=ALONE, b=PAIR, groups=odors,
                               analysis=analysis, family=family, n_boot=n_boot,
                               n_perm=n_perm)
    summary = {c: np.nan for c in per_odor.table.columns}
    summary.update(analysis=analysis, metric=metric, unit="odors (n up - n down)",
                   group="direction count", a="", b="", n_flies=int(dirn["n_flies"]),
                   n_batches=int(dirb.get("n_flies", 0)),
                   effect=float(dirn["n_up"] - dirn["n_down"]), p=float(dirn["p"]),
                   p_method="exact" if dirn.get("exact") else "monte-carlo",
                   p_holm=float(dirn["p"]), q_bh=float(dirn["p"]),
                   p_batch=float(dirb["p"]), p_batch_holm=float(dirb["p"]),
                   p_batch_disagrees=_disagree(float(dirn["p"]), float(dirb["p"])),
                   test=TEST_DIRECTION, correction="none (one test)",
                   n_trials_a=0, n_trials_b=0)
    table = pd.concat([per_odor.table, pd.DataFrame([summary])], ignore_index=True)
    bars, pf = [], []
    chi = ch.set_index(["group", "fly_id"])
    for g in all_groups:
        flies = ch.loc[ch["group"] == g, "fly_id"]
        if flies.empty:
            continue
        v, n_trials = _pooled_change(sub, g, flies, value)
        per = chi.loc[g, "change"].to_numpy(float)
        sem = _sem(per)
        bars.append({"group": g, "condition": "change", "value": sc * v,
                     "err_lo": sc * (v - sem), "err_hi": sc * (v + sem),
                     "interval": "±SEM over flies of the per-fly change",
                     "n_flies": int(len(flies)), "n_trials": n_trials})
        pf += [{"group": g, "condition": "change", "fly_id": f, "value": sc * float(c)}
               for f, c in zip(flies, per)]
    bars = pd.DataFrame(bars)
    extra = {"direction": {k: v for k, v in dirn.items() if k not in ("null", "null_n_up")},
             "direction_batch_p": float(dirb["p"]), "odors": list(wide.columns)}
    return Result(analysis=analysis, metric=metric, plot_metric=f"{metric}_change",
                  groups=all_groups, conditions=["change"], table=table, bars=bars,
                  per_fly=pd.DataFrame(pf), comps=per_odor.comps,
                  n_flies=int(sub["fly_id"].nunique()) if len(sub) else 0,
                  n_trials=int(sub["_uid"].nunique()) if len(sub) else 0,
                  test=TEST_DIRECTION, correction="none (one test)", extra=extra)


# ── figures ──────────────────────────────────────────────────────────────────
@dataclass
class Job:
    """One figure + its table: drawn after every window's numbers exist, so the shared
    y-limits of each metric are registered from all of them first."""
    folder: Path
    stem: str
    kind: str                     # "bars" | "diagnostic"
    result: Result
    title: str
    window: str
    xlabel: str
    caption: str
    cond_labels: Mapping
    effect_series: list = field(default_factory=list)   # [(label, table[, colour])]


def condition_colors(conditions: Sequence) -> dict:
    cc = st.config_value("CONDITION_COLORS", {}) or {}
    base = {ALONE: cc.get(ALONE, pl.INK_SECONDARY), PAIR: cc.get(PAIR, pl.CATEGORICAL[0]),
            "1st": cc.get(1, pl.CATEGORICAL[2]), "2nd": cc.get(2, pl.CATEGORICAL[1]),
            "click": cc.get(ALONE, pl.INK_SECONDARY), "odor": cc.get(PAIR, pl.CATEGORICAL[0])}
    return {c: base.get(c, pl.CATEGORICAL[i % 8]) for i, c in enumerate(conditions)}


def ylim_values(jobs: Sequence[Job]) -> dict[str, list[float]]:
    """Every plotted value per non-default metric (``<m>_change`` bars + fly points,
    ``<m>_effect`` estimates + CIs), for :func:`register_limits`."""
    vals: dict[str, list[float]] = {}
    for j in jobs:
        r = j.result
        if r.plot_metric.endswith("_change"):
            v = vals.setdefault(r.plot_metric, [])
            for c in ("value", "err_lo", "err_hi"):
                if c in r.bars.columns:
                    v += list(pd.to_numeric(r.bars[c], errors="coerce"))
            if len(r.per_fly):
                v += list(pd.to_numeric(r.per_fly["value"], errors="coerce"))
        if j.effect_series:
            v = vals.setdefault(f"{r.metric}_effect", [])
            for series in j.effect_series:
                t = series[1]
                for c in ("effect", "ci_lo", "ci_hi"):
                    v += list(pd.to_numeric(t[c], errors="coerce"))
    return vals


def register_limits(experiment: str, jobs: Sequence[Job]) -> dict:
    """Forget the experiment's limits, then register one per metric from every job."""
    pl.reset_ylims(experiment)
    out = {}
    for metric, v in ylim_values(jobs).items():
        v = [x for x in v if np.isfinite(x)]
        if v:
            out[metric] = pl.register_ylim(metric, experiment, data=v, overwrite=True)
    return out


def _set_ylim(ax, lims: tuple[float, float]) -> None:
    lo, hi = lims
    ax.set_ylim(lo, hi + pl.STAR_HEADROOM * (hi - lo))
    ticks = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    ax.set_yticks([t for t in ticks if lo - 1e-9 <= t <= hi + 1e-9])
    ax.spines["left"].set_bounds(lo, hi)


def _effect_ylabel(metric: str) -> str:
    return ("Within-fly difference\n(% points, 95 % BCa CI)" if metric == "rate"
            else "Within-fly difference\n(score, 95 % BCa CI)")


def effect_panel(ax, series: Sequence[tuple], groups: Sequence, *,
                 metric: str, experiment: str) -> pd.DataFrame:
    """Per group: the effect (mean within-fly difference) with its 95 % BCa CI, one
    series per (label, table[, colour]), offset side by side; Holm stars at each CI's
    top."""
    pl.style_axes(ax, grid="y")
    k = len(series)
    rows = []
    lims = pl.shared_ylim(f"{metric}_effect", experiment)
    span = lims[1] - lims[0]
    for i, item in enumerate(series):
        label, t = item[0], item[1]
        off = (i - (k - 1) / 2) * 0.28
        color = item[2] if len(item) > 2 else EFFECT_COLORS[i % len(EFFECT_COLORS)]
        ti = t.set_index("group")
        xs, ys = [], []
        for gi, g in enumerate(groups):
            if g not in ti.index or not np.isfinite(ti.loc[g, "effect"]):
                continue
            r = ti.loc[g]
            x = gi + off
            if np.isfinite(r["ci_lo"]) and np.isfinite(r["ci_hi"]):
                ax.vlines(x, r["ci_lo"], r["ci_hi"], color=color, linewidth=1.5, zorder=2)
            xs.append(x)
            ys.append(float(r["effect"]))
            s = st.stars(r["p_holm"])
            if s:                     # beyond the CI end away from 0, clear of the 0 line
                down = float(r["effect"]) < 0
                end = float(r["ci_lo"] if down else r["ci_hi"])
                end = end if np.isfinite(end) else float(r["effect"])
                ax.text(x, end + (-0.02 if down else 0.02) * span, s, ha="center",
                        va="top" if down else "bottom", fontsize=9, color=pl.INK_PRIMARY,
                        zorder=5)
            rows.append({"panel": "effect", "layer": "effect", "series": label, "group": g,
                         "x": x, "value": float(r["effect"]), "err_lo": float(r["ci_lo"]),
                         "err_hi": float(r["ci_hi"]), "p_holm": float(r["p_holm"]),
                         "stars": s, "n_flies": int(r["n_flies"]), "metric": metric})
        ax.scatter(xs, ys, s=30, color=color, edgecolors=pl.SURFACE, linewidths=1.2,
                   zorder=3, label=label)
    ax.axhline(0.0, color=pl.BASELINE, linewidth=0.8, zorder=1.5)
    _set_ylim(ax, lims)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([GROUP_LABELS.get(g, g) for g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel(_effect_ylabel(metric))
    if k >= 2:
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=k, frameon=False,
                  fontsize=8, handletextpad=0.3, borderaxespad=0.2,
                  labelcolor=pl.INK_SECONDARY)
    return pd.DataFrame(rows)


def _ylabel(plot_metric: str) -> str | None:
    return {"rate_change": "Change in PER\n(% points, ± SEM)",
            "score_change": "Change in PER score\n(± SEM)"}.get(plot_metric)


def bars_figure(job: Job, experiment: str):
    """Paired bars (per-fly lines under them, Holm stars) and, with ``effect_series``,
    the within-fly effects with BCa CIs below. Returns (figure, plotted numbers)."""
    r = job.result
    has_eff = bool(job.effect_series)
    n_cond = len(r.conditions)
    width = max(7.5, 1.05 * len(r.groups) * (0.55 + 0.25 * n_cond) + 1.6)
    if has_eff:
        fig, (ax, ax2) = pl.new_figure(2, 1, figsize=(width, 8.4),
                                       gridspec_kw={"height_ratios": [3, 2]})
    else:
        fig, ax = pl.new_figure(figsize=(width, 5.6))
    plotted = pl.paired_bars(ax, r.bars, conditions=r.conditions, metric=r.plot_metric,
                             groups=r.groups, per_fly=r.per_fly, group_col="group",
                             condition_col="condition", per_fly_col="value",
                             colors=condition_colors(r.conditions),
                             condition_labels=job.cond_labels, group_labels=GROUP_LABELS,
                             experiment=experiment, ylabel=_ylabel(r.plot_metric))
    comps = r.comps[r.comps["group"].isin(set(plotted.loc[plotted["layer"] == "bar",
                                                          "group"]))]
    starred = pl.add_holm_stars(ax, comps, plotted=plotted, group_col="group",
                                condition_col="condition") if len(comps) else pd.DataFrame()
    parts = [plotted.assign(panel="bars")]
    if len(starred):
        parts.append(starred.assign(panel="bars", layer="stars"))
    if has_eff:
        ax.tick_params(axis="x", labelbottom=False)
        eff = effect_panel(ax2, job.effect_series, r.groups, metric=r.metric,
                           experiment=experiment)
        ax2.set_xlabel(job.xlabel)
        parts.append(eff)
    else:
        ax.set_xlabel(job.xlabel)
    pl.title_block(fig, job.title, window_text(job.window), n_flies=r.n_flies,
                   n_trials=r.n_trials, test=r.test, correction=r.correction)
    pl.caption(fig, job.caption)
    return fig, pd.concat(parts, ignore_index=True)


def diagnostic_figure(job: Job, experiment: str):
    """Per-odor change (pair - alone) bars in odor colours, per-fly dots, the click as a
    hatched reference, and the direction count with its joint-flip p."""
    r = job.result
    groups = [g for g in r.groups if g in set(r.bars["group"])] if len(r.bars) else []
    fig, ax = pl.new_figure(figsize=(max(7.5, 1.0 * len(groups) + 2.0), 5.8))
    pl.style_axes(ax, grid="y")
    colors = pl.odor_colors()
    bi = r.bars.set_index("group") if len(r.bars) else pd.DataFrame()
    rng = np.random.default_rng(0)
    rows = []
    for i, g in enumerate(groups):
        b = bi.loc[g]
        ref = g == REFERENCE
        c = colors.get(g, pl.INK_MUTED)
        ax.bar(i, b["value"], width=0.62, color=pl.SURFACE if ref else c,
               edgecolor=c, hatch="////" if ref else None, linewidth=0.9 if ref else 0,
               zorder=2)
        lo, hi = float(b["err_lo"]), float(b["err_hi"])
        if np.isfinite(lo) and np.isfinite(hi):
            ax.errorbar([i], [b["value"]], yerr=[[b["value"] - lo], [hi - b["value"]]],
                        fmt="none", ecolor=pl.INK_PRIMARY, elinewidth=1.0, capsize=2.5,
                        zorder=4)
        rows.append({"panel": "bars", "layer": "bar", "group": g, "x": float(i),
                     "value": float(b["value"]), "err_lo": lo, "err_hi": hi,
                     "n_flies": int(b["n_flies"]), "n_trials": int(b["n_trials"]),
                     "reference": ref, "metric": r.plot_metric})
        pts = r.per_fly[r.per_fly["group"] == g]
        jit = rng.uniform(-0.17, 0.17, len(pts))
        ax.scatter(i + jit, pts["value"], s=7, color=pl.INK_MUTED, alpha=0.45,
                   linewidths=0, zorder=3)
        rows += [{"panel": "bars", "layer": "fly", "group": g, "x": float(i + j),
                  "value": float(v), "fly_id": f, "metric": r.plot_metric}
                 for j, v, f in zip(jit, pts["value"], pts["fly_id"])]
    ax.axhline(0.0, color=pl.BASELINE, linewidth=0.8, zorder=1.6)
    _set_ylim(ax, pl.shared_ylim(r.plot_metric, experiment))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([GROUP_LABELS.get(g, g) for g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel(_ylabel(r.plot_metric))
    ax.set_xlabel(job.xlabel)
    d = r.extra.get("direction", {})
    n_odors = len(r.extra.get("odors", []))
    note = (f"{d.get('n_up', 0)} of {n_odors} odors up, {d.get('n_down', 0)} down, "
            f"{d.get('n_zero', 0)} unchanged · joint fly-flip p = {fmt_p(d.get('p'))} · "
            f"whole-batch p = {fmt_p(r.extra.get('direction_batch_p'))} · naive "
            f"binomial p = {fmt_p(d.get('p_naive_binomial'))}")
    ax.text(0.0, 1.02, note, transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
            color=pl.INK_PRIMARY)
    ax.legend(handles=[Patch(facecolor=pl.SURFACE, edgecolor=pl.INK_SECONDARY,
                             hatch="////", label="click: reference, not counted")],
              loc="upper right", frameon=False, fontsize=8, labelcolor=pl.INK_SECONDARY)
    rows.append({"panel": "bars", "layer": "direction", "group": "direction count",
                 "value": float(d.get("n_up", 0) - d.get("n_down", 0)),
                 "p": d.get("p"), "p_batch": r.extra.get("direction_batch_p"),
                 "p_naive_binomial": d.get("p_naive_binomial")})
    pl.title_block(fig, job.title, window_text(job.window), n_flies=r.n_flies,
                   n_trials=r.n_trials, test=r.test, correction=r.correction)
    pl.caption(fig, job.caption)
    return fig, pd.DataFrame(rows)


def write_table(table: pd.DataFrame, path: Path, *, experiment: str, window: str) -> Path:
    t = table.assign(experiment=experiment, window=window)
    cols = [c for c in TABLE_COLUMNS if c in t.columns] + \
        [c for c in t.columns if c not in TABLE_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    t[cols].to_csv(path, index=False)
    return path


def render(jobs: Sequence[Job], experiment: str, tables: Path, *,
           log: logging.Logger | None = None) -> list[Path]:
    """Register the shared limits from every job, then draw and save each figure
    (PNG 300 dpi + SVG + CSV) and write its table to ``tables/<stem>.csv``."""
    log = log or LOG
    lims = register_limits(experiment, jobs)
    log.debug("%s shared y-limits: %s", experiment, lims)
    written = []
    for j in jobs:
        draw = diagnostic_figure if j.kind == "diagnostic" else bars_figure
        fig, plotted = draw(j, experiment)
        paths = pl.save_figure(fig, plotted, j.folder / j.stem)
        written += list(paths.values())
        written.append(write_table(j.result.table, Path(tables) / f"{j.stem}.csv",
                                   experiment=experiment, window=j.window))
        log.info("wrote %s (+ table)", paths["png"].relative_to(j.folder.parent.parent)
                 if j.folder.parent.parent in paths["png"].parents else paths["png"])
    plt.close("all")
    return written


# ── findings ─────────────────────────────────────────────────────────────────
def fmt_p(p) -> str:
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(p):
        return "NA"
    return "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def fmt_effect(v: float, metric: str) -> str:
    if not np.isfinite(v):
        return "NA"
    return f"{v:+.1f} pp" if metric == "rate" else f"{v:+.2f}"


def describe_rows(table: pd.DataFrame, metric: str) -> str:
    """'Hexanol +4.5 pp (95 % CI -2.3 to 11.4; 22 flies; Holm p = 1.00; batch p = 0.75)'
    for every group, ';'-joined; fragile rows are marked."""
    parts = []
    for r in table.itertuples():
        if not r.n_flies:
            parts.append(f"{r.group}: no flies")
            continue
        lo, hi = (f"{x:.1f}" if metric == "rate" else f"{x:.2f}" for x in (r.ci_lo, r.ci_hi))
        tag = ""
        if r.p_batch_disagrees is True:
            tag = (" — fly-level only, fragile at batch level" if r.p < P_ALPHA
                   else " — batch level only")
        parts.append(f"{r.group} {fmt_effect(r.effect, metric)} (95 % CI {lo} to {hi}; "
                     f"{r.n_flies} flies; Holm p = {fmt_p(r.p_holm)}; batch p = "
                     f"{fmt_p(r.p_batch)}{tag})")
    return "; ".join(parts)


def holm_verdict(table: pd.DataFrame) -> str:
    sig = table.loc[pd.to_numeric(table["p_holm"], errors="coerce") < P_ALPHA, "group"]
    if sig.empty:
        return "None survives Holm."
    return f"Holm p < 0.05: {', '.join(map(str, sig))}."


def table_numbers(table: pd.DataFrame, *, window: str, metric: str) -> dict:
    """Findings ``numbers``: window, metric, then per-group effect / CI / Holm p / batch p."""
    t = table.set_index("group")
    out = {"window": window, "metric": metric, "unit": _UNIT.get(metric, metric),
           "n_flies": int(t["n_flies"].max()) if len(t) else 0}
    for col in ("effect", "ci_lo", "ci_hi", "p_holm", "p_batch", "n_flies"):
        out[col] = {str(g): t.loc[g, col] for g in t.index}
    return jsonable(out)


def finding(section: str, analysis: str, *, primary: bool, window: str, metric: str,
            text: str, numbers: Mapping) -> dict:
    return {"section": section, "analysis": analysis, "primary": bool(primary),
            "text": text, "numbers": jsonable(dict(numbers)), "window": window,
            "metric": metric}


# ── models ───────────────────────────────────────────────────────────────────
FITS = ("gee", "glmm", "mixedlm", "ordinal")
#: GEE is the model of record. The Bayesian mixed logistic (variational) gave the opposite
#: sign to GEE on the Exp 1 full model with posterior SDs ~5x narrower than GEE's robust
#: SEs, so it and the score models are sensitivity checks only: never starred, and a sign
#: that disagrees with GEE is flagged.
FIT_ROLE = {"gee": "model of record", "glmm": "sensitivity check",
            "mixedlm": "sensitivity check", "ordinal": "sensitivity check"}


@dataclass(frozen=True)
class Term:
    text: str                     # formula text
    columns: tuple[str, ...]      # columns it needs
    factor: bool = False          # categorical: needs >= 2 levels


def prune_terms(frame: pd.DataFrame, terms: Sequence[Term]) -> tuple[list[str], list[str]]:
    """(kept formula texts, dropped notes): a term is dropped when one of its columns is
    constant in the frame (one level, or zero variance), which would make the design
    singular (e.g. start time in a one-session window)."""
    kept, dropped = [], []
    for t in terms:
        bad = []
        for c in t.columns:
            v = frame[c].dropna()
            if v.nunique() <= 1:
                bad.append(c)
        if bad:
            dropped.append(f"{t.text} (constant: {', '.join(bad)})")
        else:
            kept.append(t.text)
    return kept, dropped


def model_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Covariates: start_time_c = start_time_h centred on the frame's mean."""
    d = rows.copy()
    d["start_time_c"] = d["start_time_h"] - d["start_time_h"].mean()
    d["responded"] = d["responded"].astype(int)
    d["score"] = pd.to_numeric(d["score"], errors="coerce")
    return d


@dataclass(frozen=True)
class Separation:
    """How to find quasi-complete separation for the GEE: a cell (``cells`` columns)
    whose responses are all 0 or all 1 makes the coefficient of its ``level`` infinite,
    and statsmodels' GEE then returns NaN for EVERY estimate (Exp 2 day: IAA alone 0 of
    36). If ``reference`` itself is separated, the model is not estimable."""
    cells: tuple[str, ...]
    level: str
    reference: str


def separated_levels(frame: pd.DataFrame, sep: Separation,
                     outcome: str = "responded") -> list:
    """Levels of ``sep.level`` with a ``sep.cells`` cell of all-0 or all-1 ``outcome``,
    in odor order."""
    if frame.empty:
        return []
    m = frame.groupby(list(sep.cells), observed=True)[outcome].mean()
    bad = m[(m == 0) | (m == 1)].reset_index()[sep.level].unique()
    return group_order(bad)


def _failed_coefs(label: str, frame: pd.DataFrame, msg: str) -> tuple:
    return None, pd.DataFrame([{
        "term": "(fit failed)", "estimate": np.nan, "se": np.nan, "ci_lo": np.nan,
        "ci_hi": np.nan, "p": np.nan, "model": label, "n_obs": len(frame),
        "n_groups": frame["fly_id"].nunique(), "status": "failed", "note": msg}]), \
        f"FIT FAILED: {msg}\n"


def fit_model_set(frame: pd.DataFrame, terms: Sequence[Term], *, name: str,
                  experiment: str, window: str, folder: Path, ordinal_boot: int = 200,
                  log: logging.Logger | None = None, note: str = "",
                  separation: Separation | None = None) -> list[dict]:
    """GEE logistic (fly clusters; the model of record), Bayesian mixed logistic (fly
    random intercept, VB), MixedLM on score, and the ordinal model with fly-cluster
    bootstrap SEs, each saved via ``stats.save_model`` as
    ``<exp>_model_<name>_<fit>_<window>``. With ``separation``, the GEE leaves out the
    separated levels (recorded in ``gee_left_out`` and the note) and fails with the
    reason when the reference is one of them; the other fits keep every row. Failures
    are recorded, never raised. Returns one status row per fit."""
    log = log or LOG
    kept, dropped = prune_terms(frame, terms)
    for dnote in dropped:
        log.info("%s %s model %s: dropped %s", experiment, window, name, dnote)
    rhs = " + ".join(kept) if kept else "1"
    formulas = {"gee": f"responded ~ {rhs}", "glmm": f"responded ~ {rhs}",
                "mixedlm": f"score ~ {rhs}", "ordinal": f"score ~ {rhs}"}
    sep_bad = separated_levels(frame, separation) if separation else []
    gee_frame, gee_note = frame, ""
    if sep_bad:
        what = (f"{separation.level} level(s) {', '.join(map(str, sep_bad))} have a "
                f"{' x '.join(separation.cells)} cell with 0 % or 100 % responses "
                "(quasi-complete separation: the coefficient is infinite)")
        if separation.reference in sep_bad:
            gee_note = f"{what}; the reference {separation.reference!r} is one of them, so " \
                       "the GEE is not estimable"
        else:
            gee_frame = frame[~frame[separation.level].isin(sep_bad)]
            gee_note = f"GEE fitted without {', '.join(map(str, sep_bad))}: {what}"
        log.info("%s %s model %s: %s", experiment, window, name, gee_note)
    gee_label = "GEE logistic (Binomial, exchangeable within fly_id, robust SE)"
    fits: dict[str, Callable] = {
        "gee": (lambda: _failed_coefs(gee_label, frame, gee_note))
        if separation and separation.reference in sep_bad
        else (lambda: st.fit_gee_logistic(gee_frame, formulas["gee"])),
        "glmm": lambda: st.fit_mixed_logistic(frame, formulas["glmm"]),
        "mixedlm": lambda: st.fit_mixedlm(frame, formulas["mixedlm"]),
        "ordinal": lambda: st.fit_ordinal(frame, formulas["ordinal"],
                                          n_boot=int(ordinal_boot)),
    }
    header = (f"{experiment} · {window} window · model '{name}'\n{note}\n"
              f"dropped terms: {'; '.join(dropped) or 'none'}\n"
              f"rows: {len(frame)}, flies: {frame['fly_id'].nunique()}\n\n")
    rows = []
    for fit, fn in fits.items():
        t0 = time.perf_counter()
        try:
            res, coefs, text = fn()
        except Exception as exc:  # noqa: BLE001 -- a fit must never stop the run
            res, coefs, text = None, pd.DataFrame([{
                "term": "(fit failed)", "estimate": np.nan, "se": np.nan, "ci_lo": np.nan,
                "ci_hi": np.nan, "p": np.nan, "model": fit, "n_obs": len(frame),
                "n_groups": frame["fly_id"].nunique(), "status": "failed",
                "note": f"{type(exc).__name__}: {exc}"}]), f"FIT FAILED: {exc}\n"
        secs = time.perf_counter() - t0
        coefs, text = _guard_nonfinite(coefs, text)
        if fit == "gee" and gee_note:
            coefs = coefs.assign(note=[gee_note if not (isinstance(n, str) and n)
                                       else n if gee_note in n else f"{gee_note}; {n}"
                                       for n in coefs["note"]])
            text = f"SEPARATION: {gee_note}\n{text}"
        st.save_model((res, coefs, header + text),
                      folder / f"{experiment}_model_{name}_{fit}_{window}")
        status = str(coefs["status"].iloc[0]) if len(coefs) else "failed"
        log.info("%s %s model %s/%s: %s (%.1f s)", experiment, window, name, fit, status, secs)
        rows.append({"experiment": experiment, "window": window, "model": name, "fit": fit,
                     "role": FIT_ROLE[fit], "formula": formulas[fit], "status": status,
                     "n_obs": int(coefs["n_obs"].iloc[0]) if len(coefs) else 0,
                     "n_groups": int(coefs["n_groups"].iloc[0]) if len(coefs) else 0,
                     "dropped_terms": "; ".join(dropped),
                     "gee_left_out": (", ".join(map(str, sep_bad))
                                      if fit == "gee" and sep_bad
                                      and separation.reference not in sep_bad else ""),
                     "seconds": round(secs, 2),
                     "note": str(coefs["note"].iloc[0]) if len(coefs) else "",
                     "_coefs": coefs})
    return rows


def _guard_nonfinite(coefs: pd.DataFrame, text: str) -> tuple[pd.DataFrame, str]:
    """A fit whose fixed-effect estimates are not all finite is 'failed', whatever the
    fitter said: statsmodels' GEE can diverge to all-NaN estimates (separation in a small
    window) without raising, and ``stats`` then reports 'ok'."""
    if coefs.empty or (coefs["status"] == "failed").all():
        return coefs, text
    fe = coefs[~coefs["term"].astype(str).str.startswith("VC ")]
    est = pd.to_numeric(fe["estimate"], errors="coerce")
    bad = int((~np.isfinite(est)).sum())
    if bad == 0:
        return coefs, text
    note = (f"non-finite estimates in {bad} of {len(fe)} fixed-effect terms (fitter "
            f"status was '{coefs['status'].iloc[0]}'): treated as failed")
    out = coefs.assign(status="failed",
                       note=[f"{n}; {note}" if isinstance(n, str) and n else note
                             for n in coefs["note"]])
    return out, f"STATUS OVERRIDDEN: {note}\n{text}"


def write_model_status(rows: Sequence[dict], folder: Path, experiment: str,
                       window: str) -> Path:
    path = folder / f"{experiment}_model_status_{window}.csv"
    pd.DataFrame([{k: v for k, v in r.items() if k != "_coefs"} for r in rows]) \
        .to_csv(path, index=False)
    return path


def key_terms(rows: Sequence[dict], *, name: str, fit: str, pattern: str) -> pd.DataFrame:
    """The coefficient rows of one fit whose term contains ``pattern``."""
    for r in rows:
        if r["model"] == name and r["fit"] == fit:
            c = r["_coefs"]
            return c[c["term"].astype(str).str.contains(pattern, regex=False)]
    return pd.DataFrame()


def term_level(term: str) -> str:
    """'C(a)[T.pair]:C(partner, ...)[T.Hexanol]' -> 'Hexanol' (the last level named);
    a plain covariate keeps its name."""
    levels = re.findall(r"\[T\.([^\]]+)\]", str(term))
    return levels[-1] if levels else str(term)


def sign_disagreements(rows: Sequence[dict], *, name: str, pattern: str,
                       reference: str = "gee", check: str = "glmm") -> list[dict]:
    """Terms containing ``pattern`` whose ``check`` fit (default the mixed logistic) has
    the opposite sign to the model of record (GEE); both estimates must be finite and
    non-zero. A failed fit contributes nothing."""
    est = {}
    for fit in (reference, check):
        c = key_terms(rows, name=name, fit=fit, pattern=pattern)
        if len(c) and not (c["status"] == "failed").all():
            est[fit] = dict(zip(c["term"], pd.to_numeric(c["estimate"], errors="coerce")))
    if len(est) < 2:
        return []
    out = []
    for term, g in est[reference].items():
        m = est[check].get(term, np.nan)
        if np.isfinite(g) and np.isfinite(m) and g * m < 0:
            out.append({"term": term, reference: float(g), check: float(m)})
    return out


def gee_left_out(rows: Sequence[dict], name: str) -> str:
    """'' or ' The GEE left out X (...separation...).' for model ``name``."""
    for r in rows:
        if r["model"] == name and r["fit"] == "gee" and r.get("gee_left_out"):
            return (f" The GEE left out {r['gee_left_out']}: a cell with 0 % or 100 % "
                    "responses (quasi-complete separation) makes that coefficient infinite "
                    "and, left in, drags every estimate to NaN.")
    return ""


def describe_sign_flips(flips: Sequence[dict]) -> str:
    """'' when none; else the sentence the findings carry."""
    if not flips:
        return ""
    terms = ", ".join(f"{term_level(f['term'])} (GEE {f['gee']:+.2f}, mixed logistic "
                      f"{f['glmm']:+.2f})" for f in flips)
    return (f" SIGN DISAGREEMENT: the Bayesian mixed logistic (sensitivity check only) has "
            f"the opposite sign to the GEE for {terms}; trust the GEE.")


def describe_odds(coefs: pd.DataFrame) -> str:
    """'Hexanol OR 1.30 (0.60-2.80, p 0.48)' per term (log-odds -> odds ratio)."""
    parts = []
    for r in coefs.itertuples():
        name = term_level(r.term)
        if not np.isfinite(r.estimate):
            continue
        parts.append(f"{name} OR {np.exp(r.estimate):.2f} ({np.exp(r.ci_lo):.2f}-"
                     f"{np.exp(r.ci_hi):.2f}, p {fmt_p(r.p)})")
    return "; ".join(parts)


def dump_findings(findings: Sequence[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(list(findings)), indent=1))
    return path


__all__ = (
    "ALONE", "FITS", "IAA", "Job", "PAIR", "REFERENCE", "Result", "TABLE_COLUMNS", "Term",
    "FIT_ROLE", "Separation", "alone_vs_pair", "contract_table", "controlled_change",
    "describe_rows", "describe_sign_flips", "sign_disagreements",
    "drop_carryover", "finding", "fit_model_set", "fmt_p", "holm_verdict", "jsonable",
    "model_frame", "paired_analysis", "prepare", "presentation_1_vs_2",
    "reference_contrast", "register_limits", "render", "scored_long", "shared_baseline",
    "separated_levels", "split_windows", "stem", "table_numbers", "write_model_status",
    "write_readme",
    "write_table",
)
