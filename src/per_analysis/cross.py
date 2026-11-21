"""Cross-experiment analyses -> ``<out>/05_cross_experiment/``.

a. ``a_iaa_baseline_by_cohort``: the naive IAA response (exp1 pre / exp2-3 alone block),
   one trial per fly, across the six cohorts. Kruskal-Wallis over per-fly values, the 15
   cohort pairs by fly-level label permutation (Holm), and whole-batch label permutation
   with sessions as strata where two cohorts share a session (``p_batch``).
b. ``b_iaa_iaa_across_experiments``: the IAA->IAA pair response of Exp 2 and Exp 3 beside
   each cohort's alone IAA, and click->IAA (Exp 3 only: Exp 2 has no click-primed IAA).
   Within-fly comparisons, between-experiment comparisons (session strata), a GEE, a
   Bayesian mixed logistic and a MixedLM of condition x experiment, and an ANCOVA-style
   model of the per-fly pair response on the per-fly naive IAA.
c. ``c_eb_iaa_bidirectional``: every prime->target pair of Exp 2 and Exp 3 against its
   click reference -- Exp 2 per fly (X pair - X alone) - (click pair - click alone),
   Exp 3 per fly IAA after X - IAA after click -- in one forest sorted by effect, Holm
   over the forest; bars of IAA->EB and EB->IAA.
d. ``d_paraffin_noise_floor``: per cohort, each odor's naive response minus the same
   fly's naive paraffin/click response; odors not distinguishable from it are flagged
   (Holm within cohort x metric). The IAA cohort was never tested on paraffin: skipped.

Data rules: light probes and the known-bad 0.025 s click trials (non-light rows with
``odor_window_s < 1``) are left out of every odor analysis, logged (:func:`odor_trials`).
The fly is the unit: per-fly means first, bars pool trials (``stats.group_summary``).
Rates are tested on per-fly 0/1 means and reported in percentage points (pp).
Every table has ``p`` (fly level), ``p_batch`` (whole batches) and ``batch_disagrees``
(the two fall on different sides of 0.05). Stars come from ``p_holm`` only.

The helpers here (:func:`odor_trials`, :func:`fly_values`, :func:`two_group`,
:func:`paired`, :func:`save_table`, ...) are reused by ``power`` and ``daynight``.
"""
from __future__ import annotations

import itertools
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

from per_analysis import config as C  # noqa: E402
from per_analysis import plots as pl  # noqa: E402
from per_analysis import stats as st  # noqa: E402 -- also puts the repo on sys.path
from scripts.analysis import b2b_stats as b2b  # noqa: E402

log = logging.getLogger(__name__)

# ── settings ──────────────────────────────────────────────────────────────────
N_BOOT = C.N_BOOT            # bootstrap replicates for CIs (tests lower it)
N_PERM_FLY = 20_000          # fly-level two-group permutations (fly_stats loops in Python)
SEED = C.SEED
SECTION = "cross"
SUBDIR = "05_cross_experiment"

PARTNERS = [o for o in C.ODOR_ORDER if o != C.CLICK_ODOR]
NAIVE_BLOCKS = ("pre", "alone")
METRICS = {"rate": "responded", "score": "score"}
SCALE = {"rate": 100.0, "score": 1.0}
UNIT = {"rate": "pp", "score": "score points"}
CONTRACT = ["n_flies", "effect", "ci_lo", "ci_hi", "p", "p_holm", "q_bh", "p_batch", "test",
            "correction"]
FLY_CARRY = ("batch_key", "batch", "session", "cohort", "experiment", "window", "date",
             "start_time_h")

COHORT_LABELS = {"IAA-Sensitivity-1-1": "IAA-Sens\n(Exp 1)",
                 "Hex-Sensitivity-1-0.1": "Hex-Sens\n(Exp 1)",
                 "3Oct-Sensitivity-1-0.1": "3Oct-Sens\n(Exp 1)",
                 "Paraffin-Sensitivity-1-100": "Paraffin-Sens\n(Exp 1)",
                 "IAA-PairFirst-Control-1-1": "PairFirst\n(Exp 2)",
                 "IAA-PairSecond-Control-1-1": "PairSecond\n(Exp 3)"}
EXP_LABELS = {"exp1": "Exp 1", "exp2": "Exp 2", "exp3": "Exp 3"}
CLICK_LABEL = "click"
GRAY = C.CONDITION_COLORS["alone"]
PAIR_BLUE = C.CONDITION_COLORS["pair"]
CLICK_RED = C.ODOR_COLORS[C.CLICK_ODOR]

TEST_TWO_GROUP = ("difference in fly means; p: fly-level label permutation; CI: BCa "
                  "bootstrap of flies; p_batch: whole-batch label permutation")
TEST_PAIRED = ("within-fly difference; p: exact paired sign-flip over flies; CI: BCa "
               "bootstrap of flies; p_batch: whole-batch sign-flip")


# ═════════════════════════════════════════════════════════════════════════════
# shared helpers
# ═════════════════════════════════════════════════════════════════════════════
def batch_keys(df: pd.DataFrame) -> pd.Series:
    """``cohort|batch``: a batch folder within its dataset."""
    return df["cohort"].astype(str) + "|" + df["batch"].astype(str)


def odor_trials(trials: pd.DataFrame) -> pd.DataFrame:
    """Odor rows only: light probes and the known-bad short clicks (a non-light row with
    ``odor_window_s < 1``) left out, logged. Adds ``batch_key``."""
    light = trials["is_light_probe"].astype(bool).to_numpy()
    win = pd.to_numeric(trials["odor_window_s"], errors="coerce").to_numpy(float)
    short = ~light & (win < 1.0)
    if short.any():
        bad = trials.loc[short, ["fly_id", "trial_label", "odor", "odor_window_s"]]
        log.info("left out %d odor trial(s) with an odor window < 1 s (known-bad 0.025 s "
                 "clicks): %s", int(short.sum()), bad.to_dict("records"))
    log.debug("left out %d light-probe rows from the odor analyses", int(light.sum()))
    out = trials[~light & ~short].copy()
    out["batch_key"] = batch_keys(out)
    return out


def fly_values(df: pd.DataFrame, metric: str, by: Sequence[str] = ()) -> pd.DataFrame:
    """One row per (``by``..., fly): the within-fly mean of ``responded`` (0/1, metric
    "rate") or ``score``, in column ``value``, plus the fly's batch/session/window."""
    if "batch_key" not in df.columns:
        df = df.assign(batch_key=batch_keys(df))
    carry = tuple(c for c in FLY_CARRY if c in df.columns)
    return st.per_fly(df, METRICS[metric], by=list(by), carry=carry, out_col="value")


def series(pf: pd.DataFrame, **match) -> pd.Series:
    """fly_id -> value of the rows of ``pf`` matching ``match``."""
    m = np.ones(len(pf), dtype=bool)
    for k, v in match.items():
        m &= (pf[k] == v).to_numpy()
    s = pf.loc[m].set_index("fly_id")["value"].astype(float)
    if s.index.has_duplicates:
        raise ValueError(f"several values per fly for {match}")
    return s


def batch_of_flies(df: pd.DataFrame) -> dict:
    d = df if "batch_key" in df.columns else df.assign(batch_key=batch_keys(df))
    return d.drop_duplicates("fly_id").set_index("fly_id")["batch_key"].to_dict()


def session_of_batches(df: pd.DataFrame) -> dict:
    """batch_key -> session (``b2b_stats.session_of``: the folder without ``_rig_N``)."""
    d = df if "batch_key" in df.columns else df.assign(batch_key=batch_keys(df))
    u = d.drop_duplicates("batch_key")
    return {k: b2b.session_of(b) for k, b in zip(u["batch_key"], u["batch"])}


def hybrid_strata(labels: Mapping, session_of_batch: Mapping | None) -> dict | None:
    """batch -> stratum for ``b2b_stats.batch_label_perm``: a session that holds batches
    of two or more groups is its own stratum (labels move only inside it); every other
    batch sits in one shared stratum. None when no session is shared."""
    if not session_of_batch:
        return None
    labs: dict = defaultdict(set)
    for b, lab in labels.items():
        labs[session_of_batch.get(b, b)].add(lab)
    shared = {s for s, ls in labs.items() if len(ls) > 1}
    if not shared:
        return None
    return {b: (session_of_batch.get(b, b) if session_of_batch.get(b, b) in shared
                else "_unshared") for b in labels}


def batch_perm(groups: Mapping[Any, pd.Series], batch_of: Mapping,
               session_of_batch: Mapping | None = None) -> dict:
    """Whole-batch label permutation over ``groups`` (label -> fly_id -> value), with
    :func:`hybrid_strata`. Returns ``b2b_stats.batch_label_perm`` + ``strata``."""
    vals, bats, labels = [], [], {}
    for lab, s in groups.items():
        s = s.dropna()
        for fly, v in s.items():
            b = batch_of[fly]
            if labels.setdefault(b, lab) != lab:
                raise ValueError(f"batch {b} holds flies of two groups")
            vals.append(float(v))
            bats.append(b)
    strata = hybrid_strata(labels, session_of_batch)
    res = b2b.batch_label_perm(np.asarray(vals), np.asarray(bats, dtype=object), labels,
                               strata)
    res["strata"] = "session" if strata else "none"
    res["n_batches_by_group"] = {lab: len({batch_of[f] for f in s.dropna().index})
                                 for lab, s in groups.items()}
    return res


def two_group(a: pd.Series, b: pd.Series, batch_of: Mapping,
              session_of_batch: Mapping | None = None, *, scale: float = 1.0) -> dict:
    """Group b vs group a (independent flies): ``effect`` = mean b - mean a over flies
    (x ``scale``), BCa CI (``stats.did_bootstrap``), ``p`` fly-level label permutation,
    ``p_batch`` whole-batch label permutation (:func:`batch_perm`)."""
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    r = st.did_bootstrap(b, a, n_boot=N_BOOT, n_perm=N_PERM_FLY, seed=SEED)
    bp = batch_perm({"a": a, "b": b}, batch_of, session_of_batch)
    nb = bp["n_batches_by_group"]
    return {"n_flies": int(len(a) + len(b)), "n_flies_a": int(len(a)), "n_flies_b": int(len(b)),
            "n_batches": int(bp["n_batches"]), "n_batches_a": int(nb["a"]),
            "n_batches_b": int(nb["b"]), "mean_a": r["mean_control"] * scale,
            "mean_b": r["mean_exposed"] * scale, "effect": r["estimate"] * scale,
            "ci_lo": r["ci_lo"] * scale, "ci_hi": r["ci_hi"] * scale, "p": r["p"],
            "p_batch": bp["p"], "p_batch_exact": bp["exact"], "strata": bp["strata"]}


def paired(a: pd.Series, b: pd.Series, batch_of: Mapping, *, scale: float = 1.0) -> dict:
    """Within-fly b - a (``stats.paired_effect``), scaled."""
    r = st.paired_effect(a, b, batch_of=batch_of, n_boot=N_BOOT, seed=SEED)
    return {"n_flies": r["n_flies"], "n_batches": r["n_batches"], "mean_a": r["mean_a"] * scale,
            "mean_b": r["mean_b"] * scale, "effect": r["mean_diff"] * scale,
            "ci_lo": r["ci_lo"] * scale, "ci_hi": r["ci_hi"] * scale, "p": r["p"],
            "p_batch": r["p_batch"]}


def finish(table: pd.DataFrame, family: str | Sequence[str] | None, *,
           correction: str) -> pd.DataFrame:
    """Holm / BH within ``family`` (also p_batch_holm), the correction text and the
    p vs p_batch disagreement flag."""
    if table.empty:
        return table
    t = st.add_corrections(table, p_col="p", family_col=family)
    t["correction"] = correction
    p, pb = (pd.to_numeric(t[c], errors="coerce") for c in ("p", "p_batch"))
    both = p.notna() & pb.notna()
    t["batch_disagrees"] = both & ((p < 0.05) != (pb < 0.05))
    return t


def save_table(table: pd.DataFrame, tables: Path, base: str) -> Path:
    """``tables/<base>.csv`` with the contract columns (NaN where not applicable)."""
    tables = Path(tables)
    if not tables.exists():
        tables.mkdir(parents=True, exist_ok=True)
    readme = tables / "README.md"
    if not readme.exists():
        readme.write_text("# tables\nOne CSV per figure (same base name) with n_flies, effect, "
                          "95 % CI, p (fly level), p_holm, q_bh, p_batch (whole batches), the "
                          "test and the correction.\n")
    t = table.copy()
    for c in CONTRACT:
        if c not in t.columns:
            t[c] = np.nan
    t = t[[c for c in t.columns if c not in CONTRACT] + CONTRACT]
    path = tables / f"{base}.csv"
    t.to_csv(path, index=False)
    n_dis = int(t.get("batch_disagrees", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    if n_dis:
        log.warning("%s: %d row(s) where p and p_batch disagree at 0.05", base, n_dis)
    return path


def readme(folder: Path, text: str) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "README.md").write_text(text.strip() + "\n")


def fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "p n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def fmt_effect(r: Mapping, unit: str, *, digits: int = 1) -> str:
    e, lo, hi = (float(r.get(k, np.nan)) for k in ("effect", "ci_lo", "ci_hi"))
    f = f"{{:+.{digits}f}}"
    ci = (f"95 % CI {f.format(lo)} to {f.format(hi)}" if np.isfinite(lo) and np.isfinite(hi)
          else "95 % CI n/a")
    return f"{f.format(e)} {unit} ({ci})"


def fmt_stats(r: Mapping) -> str:
    """'Holm p = .., p_batch = ..' (+ a fragile note when they disagree)."""
    s = f"Holm {fmt_p(r.get('p_holm'))}, fly-level {fmt_p(r.get('p'))}, " \
        f"batch-level {fmt_p(r.get('p_batch'))}"
    if bool(r.get("batch_disagrees", False)):
        s += " [fly and batch levels disagree at 0.05: fragile]"
    return s


def centred(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    return x - x.mean()


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def model_numbers(coefs: pd.DataFrame, term: str) -> dict:
    row = coefs[coefs["term"] == term]
    if row.empty:
        return {"term": term, "estimate": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "p": np.nan, "status": "missing"}
    r = row.iloc[0]
    return {"term": term, "estimate": _num(r["estimate"]), "ci_lo": _num(r["ci_lo"]),
            "ci_hi": _num(r["ci_hi"]), "p": _num(r["p"]), "status": str(r["status"])}


MIXED_NOTE = ("SENSITIVITY ONLY: BinomialBayesMixedGLM.fit_vb is a variational approximation; its "
              "posterior SDs run about 2-5x narrower than the GEE robust SEs on these data, so its "
              "p values are not tests. GEE is the model of record; never star this fit.")


def sign_check(gee: Mapping, mixed: Mapping) -> str:
    """'agrees' / 'DISAGREES' on the sign of a term between GEE and the mixed logistic."""
    g, m = gee.get("estimate", np.nan), mixed.get("estimate", np.nan)
    if not (np.isfinite(g) and np.isfinite(m)):
        return "n/a"
    return "agrees" if np.sign(g) == np.sign(m) else "DISAGREES"


def n_trials(df: pd.DataFrame) -> int:
    return int(len(df))


def finding(analysis: str, text: str, numbers: Mapping, *, primary: bool = False,
            section: str = SECTION) -> dict:
    return {"section": section, "analysis": analysis, "primary": bool(primary),
            "text": " ".join(text.split()), "numbers": dict(numbers)}


# ═════════════════════════════════════════════════════════════════════════════
# a. naive IAA by cohort
# ═════════════════════════════════════════════════════════════════════════════
def iaa_naive_trials(odor: pd.DataFrame) -> pd.DataFrame:
    return odor[odor["block"].isin(NAIVE_BLOCKS) & (odor["odor"] == "IAA")]


def iaa_baseline_table(odor: pd.DataFrame) -> pd.DataFrame:
    d = iaa_naive_trials(odor)
    cohorts = [c for c in C.COHORTS if c in set(d["cohort"])]
    batch_of, sess = batch_of_flies(d), session_of_batches(d)
    rows = []
    for metric in METRICS:
        pf = fly_values(d, metric, by=["cohort"])
        groups = {c: series(pf, cohort=c) for c in cohorts}
        vals = [g.to_numpy(float) for g in groups.values() if len(g)]
        try:
            h, p_kw = sps.kruskal(*vals)
        except ValueError:                       # all values identical
            h, p_kw = np.nan, 1.0
        bp = batch_perm(groups, batch_of, sess)
        rows.append({"analysis": "omnibus", "metric": metric, "comparison": "all cohorts",
                     "cohort_a": "", "cohort_b": "", "n_flies": int(sum(len(g) for g in
                                                                       groups.values())),
                     "n_batches": int(bp["n_batches"]), "statistic": float(h),
                     "effect": np.nan, "p": float(p_kw), "p_batch": bp["p"],
                     "strata": bp["strata"],
                     "test": "Kruskal-Wallis on per-fly values; p_batch: whole-batch label "
                             "permutation (between-cohort SS of fly-pooled means)",
                     "unit": UNIT[metric]})
        for ca, cb in itertools.combinations(cohorts, 2):
            r = two_group(groups[ca], groups[cb], batch_of, sess, scale=SCALE[metric])
            rows.append({"analysis": "pairwise", "metric": metric,
                         "comparison": f"{cb} - {ca}", "cohort_a": ca, "cohort_b": cb, **r,
                         "test": TEST_TWO_GROUP, "unit": UNIT[metric]})
    return finish(pd.DataFrame(rows), ["analysis", "metric"],
                  correction="Holm over the 15 cohort pairs within metric (omnibus: none)")


def _strip_box(ax, pf: pd.DataFrame, groups: Sequence, summary: pd.DataFrame,
               group_col: str, labels: Mapping, color: str) -> list[dict]:
    """Per-fly score points (jittered) over a box, with the pooled mean +/- SEM."""
    pl.style_axes(ax, grid="y")
    rng = np.random.default_rng(SEED)
    rows = []
    data = [pf.loc[pf[group_col] == g, "value"].to_numpy(float) for g in groups]
    ax.boxplot(data, positions=range(len(groups)), widths=0.5, showfliers=False,
               patch_artist=True, medianprops=dict(color=pl.INK_SECONDARY, linewidth=1),
               boxprops=dict(facecolor="none", edgecolor=pl.BASELINE, linewidth=0.8),
               whiskerprops=dict(color=pl.BASELINE, linewidth=0.8),
               capprops=dict(color=pl.BASELINE, linewidth=0.8), zorder=2)
    for i, (g, v) in enumerate(zip(groups, data)):
        x = i + rng.uniform(-0.17, 0.17, v.size)
        y = v + rng.uniform(-0.12, 0.12, v.size)          # integer scores: vertical jitter
        ax.scatter(x, y, s=9, color=color, alpha=0.45, linewidths=0, zorder=3)
        s = summary[summary[group_col] == g]
        if len(s):
            m, lo, hi = (float(s.iloc[0][k]) for k in ("value", "err_lo", "err_hi"))
            ax.errorbar([i + 0.3], [m], yerr=[[m - lo], [hi - m]], fmt="D", ms=4.5,
                        color=pl.INK_PRIMARY, elinewidth=1, capsize=2.5, zorder=4)
            rows.append({"layer": "mean", group_col: g, "x": i + 0.3, "value": m,
                         "err_lo": lo, "err_hi": hi, "err_hi_plot": hi})
        rows += [{"layer": "fly", group_col: g, "x": float(xx), "value": float(vv)}
                 for xx, vv in zip(x, v)]
    lo, hi = pl.shared_ylim("score")
    ax.set_ylim(lo, hi + pl.STAR_HEADROOM * (hi - lo))
    ax.spines["left"].set_bounds(lo, hi)
    ax.set_yticks(np.arange(np.ceil(lo), np.floor(hi) + 1))   # no ticks in the bracket room
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([labels.get(g, g) for g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel("PER score (dots: flies; diamond: mean ± SEM)")
    return rows


def _room_for_brackets(ax, n_brackets: int, data_top: float) -> None:
    """Raise the axes top so ``n_brackets`` stacked Holm brackets (add_holm_stars: pad
    0.025, step 0.07 of the span) fit above ``data_top`` inside the axes."""
    if n_brackets <= 0:
        return
    lo, top = ax.get_ylim()
    need = (data_top - lo) / max(0.2, 1 - 0.025 - 0.07 * n_brackets - 0.08)
    if lo + need > top:
        ax.set_ylim(lo, lo + need)


def _pair_brackets(ax, table: pd.DataFrame, x_of: Mapping, plotted: pd.DataFrame) -> pd.DataFrame:
    comps = pd.DataFrame({"x1": [x_of[a] for a in table["cohort_a"]],
                          "x2": [x_of[b] for b in table["cohort_b"]],
                          "p_holm": table["p_holm"].to_numpy(float)})
    pts = plotted.copy()
    if "err_hi" not in pts.columns:
        pts["err_hi"] = pts["value"]
    return pl.add_holm_stars(ax, comps, plotted=pts[["x", "value", "err_hi"]])


def plot_iaa_baseline(odor: pd.DataFrame, table: pd.DataFrame, folder: Path) -> Path:
    d = iaa_naive_trials(odor)
    cohorts = [c for c in C.COHORTS if c in set(d["cohort"])]
    color = C.ODOR_COLORS["IAA"]
    fig, axes = pl.new_figure(1, 2, figsize=(12.5, 5.6))
    plotted = []
    rate = st.group_summary(d.assign(condition="naive"), ["cohort", "condition"], "rate")
    pr = pl.paired_bars(axes[0], rate, conditions=["naive"], metric="rate", groups=cohorts,
                        colors={"naive": color}, group_col="cohort", group_labels=COHORT_LABELS,
                        legend=False)
    bars = pr[pr["layer"] == "bar"]
    x_of = dict(zip(bars["cohort"], bars["x"]))
    pw = table[(table["analysis"] == "pairwise")]
    _pair_brackets(axes[0], pw[pw["metric"] == "rate"], x_of, bars)
    axes[0].set_title("Naive IAA response rate (% of flies, Wilson 95 % CI)", fontsize=9,
                      loc="left", color=pl.INK_SECONDARY)
    plotted.append(pr.assign(panel="rate"))
    pf = fly_values(d, "score", by=["cohort"])
    summ = st.group_summary(d, ["cohort"], "score")
    rows = _strip_box(axes[1], pf, cohorts, summ, "cohort", COHORT_LABELS, color)
    ps = pd.DataFrame(rows)
    pws = pw[pw["metric"] == "score"]
    _room_for_brackets(axes[1], int((pws["p_holm"] < 0.05).sum()),
                       float(ps["value"].max()) if len(ps) else 5.0)
    _pair_brackets(axes[1], pws, {c: i for i, c in enumerate(cohorts)}, ps)
    axes[1].set_title("Naive IAA score per fly", fontsize=9, loc="left",
                      color=pl.INK_SECONDARY)
    plotted.append(ps.assign(panel="score"))
    om = table[table["analysis"] == "omnibus"].set_index("metric")
    pl.title_block(fig, "Naive IAA response across the six cohorts (pre / alone block)",
                   n_flies=int(d["fly_id"].nunique()), n_trials=n_trials(d),
                   test="Kruskal-Wallis + pairwise fly-level permutation",
                   correction="Holm over 15 cohort pairs")
    pl.caption(fig, f"{pl.HOLM_CAPTION}. Kruskal-Wallis: rate {fmt_p(om.loc['rate', 'p'])}, "
                    f"score {fmt_p(om.loc['score', 'p'])}; whole-batch label permutation "
                    f"(session strata where cohorts share a session): rate "
                    f"{fmt_p(om.loc['rate', 'p_batch'])}, score {fmt_p(om.loc['score', 'p_batch'])}. "
                    "One naive IAA trial per fly; the table has every pair with its p_batch.")
    base = "cross_a_iaa_baseline_by_cohort"
    pl.save_figure(fig, pd.concat(plotted, ignore_index=True), folder / base)
    return folder / base


def _finding_a(table: pd.DataFrame, odor: pd.DataFrame) -> dict:
    d = iaa_naive_trials(odor)
    rate = st.group_summary(d, ["cohort"], "rate").set_index("cohort")
    om = table[table["analysis"] == "omnibus"].set_index("metric")
    pw = table[(table["analysis"] == "pairwise") & (table["metric"] == "rate")]
    sig = pw[pw["p_holm"] < 0.05]
    parts = [f"{c.split('-')[0] if 'Pair' not in c else c.split('-')[1]} "
             f"{rate.loc[c, 'value']:.0f} % (n = {int(rate.loc[c, 'n_flies'])})"
             for c in C.COHORTS if c in rate.index]
    sig_txt = ("; ".join(f"{r.cohort_b} - {r.cohort_a} {fmt_effect(r._asdict(), 'pp')}, "
                         f"{fmt_stats(r._asdict())}" for r in sig.itertuples())
               or "no cohort pair differs after Holm")
    text = (f"Naive IAA response rate by cohort (one trial per fly): {', '.join(parts)}. "
            f"Kruskal-Wallis on per-fly responses {fmt_p(om.loc['rate', 'p'])} "
            f"(score {fmt_p(om.loc['score', 'p'])}); whole-batch label permutation with "
            f"session strata {fmt_p(om.loc['rate', 'p_batch'])} (score "
            f"{fmt_p(om.loc['score', 'p_batch'])}). Pairwise (Holm over 15): {sig_txt}.")
    return finding("a_iaa_baseline_by_cohort", text,
                   {"rate_by_cohort": rate["value"].round(2).to_dict(),
                    "n_flies_by_cohort": rate["n_flies"].astype(int).to_dict(),
                    "kruskal_p": {m: float(om.loc[m, "p"]) for m in om.index},
                    "p_batch": {m: float(om.loc[m, "p_batch"]) for m in om.index}})


# ═════════════════════════════════════════════════════════════════════════════
# b. IAA -> IAA across experiments
# ═════════════════════════════════════════════════════════════════════════════
CONDITIONS = ["alone", "IAA-primed", "click-primed"]
COND_COLORS = {"alone": GRAY, "IAA-primed": PAIR_BLUE, "click-primed": CLICK_RED}
COND_LABELS = {"alone": "alone (naive)", "IAA-primed": "IAA → IAA",
               "click-primed": "click → IAA"}


def iaa_condition_trials(odor: pd.DataFrame) -> pd.DataFrame:
    """Scored-IAA rows of Exp 2 and Exp 3: ``alone`` (alone block), ``IAA-primed`` (the
    IAA->IAA pair), ``click-primed`` (click->IAA, Exp 3 only)."""
    d = odor[odor["experiment"].isin(["exp2", "exp3"]) & (odor["odor"] == "IAA")]
    alone = (d["block"] == "alone").to_numpy()
    pair = (d["block"] == "pair").to_numpy()
    iaa = (d["prime"] == "IAA").fillna(False).to_numpy(bool)
    click = ((d["prime"] == C.CLICK_ODOR).fillna(False).to_numpy(bool)
             & (d["experiment"] == "exp3").to_numpy())
    cond = np.select([alone, pair & iaa, pair & click], CONDITIONS, default="")
    out = d.assign(condition=cond, exp_label=d["experiment"].map(EXP_LABELS))
    return out[out["condition"] != ""].copy()


def iaa_iaa_table(cond: pd.DataFrame) -> pd.DataFrame:
    batch_of, sess = batch_of_flies(cond), session_of_batches(cond)
    rows = []
    for metric in METRICS:
        pf = fly_values(cond, metric, by=["experiment", "condition"])
        sc = SCALE[metric]
        get = {(e, c): series(pf, experiment=e, condition=c)
               for e in ("exp2", "exp3") for c in CONDITIONS}
        within = [("exp2", "alone", "IAA-primed"), ("exp3", "alone", "IAA-primed"),
                  ("exp3", "alone", "click-primed"), ("exp3", "click-primed", "IAA-primed")]
        for e, ca, cb in within:
            if get[(e, ca)].empty or get[(e, cb)].empty:
                continue
            r = paired(get[(e, ca)], get[(e, cb)], batch_of, scale=sc)
            rows.append({"family": "within experiment", "metric": metric, "experiment": e,
                         "group": EXP_LABELS[e], "cond_a": ca, "cond_b": cb,
                         "comparison": f"{EXP_LABELS[e]}: {COND_LABELS[cb]} - {COND_LABELS[ca]}",
                         **r, "test": TEST_PAIRED, "unit": UNIT[metric]})
        # between experiments (independent flies; the pair cohorts share sessions)
        chg = {e: (get[(e, "IAA-primed")] - get[(e, "alone")]).dropna() for e in ("exp2", "exp3")}
        between = [("alone", get[("exp2", "alone")], get[("exp3", "alone")]),
                   ("IAA-primed", get[("exp2", "IAA-primed")], get[("exp3", "IAA-primed")]),
                   ("IAA-primed - alone", chg["exp2"], chg["exp3"])]
        for what, a, b in between:
            r = two_group(a, b, batch_of, sess, scale=sc)
            rows.append({"family": "between experiments", "metric": metric,
                         "experiment": "exp3 - exp2", "group": "", "cond_a": what,
                         "cond_b": what, "comparison": f"{what}: Exp 3 - Exp 2", **r,
                         "test": TEST_TWO_GROUP + " (session strata: the pair cohorts share "
                                 "sessions)", "unit": UNIT[metric]})
    return finish(pd.DataFrame(rows), ["family", "metric"],
                  correction="Holm within family (within experiment: 4; between: 3) x metric")


def iaa_iaa_models(cond: pd.DataFrame, models: Path) -> dict:
    """GEE + Bayesian mixed logistic of responded, MixedLM of score, and the per-fly
    ANCOVA. Exp 2 has no click-primed IAA, so condition x experiment is coded as
    exp3 + iaa_primed + iaa_primed:exp3 + click_primed (click is an Exp 3-only term)."""
    md = cond.assign(exp3=(cond["experiment"] == "exp3").astype(int),
                     iaa_primed=(cond["condition"] == "IAA-primed").astype(int),
                     click_primed=(cond["condition"] == "click-primed").astype(int),
                     start_time_c=centred(cond["start_time_h"]))
    rhs = "exp3 + iaa_primed + iaa_primed:exp3 + click_primed + start_time_c"
    out = {}
    note = ("Exp 2 has no click-primed IAA: the click_primed term is Exp 3's click->IAA vs its "
            "alone IAA, and there is no click x experiment term. Reference: Exp 2 alone IAA. "
            "position_in_block is left out: alone positions 1-8 and pair positions 1-17 make it "
            "a proxy of condition.")
    for name, fit in (("gee_responded", lambda: st.fit_gee_logistic(md, f"responded ~ {rhs}")),
                      ("mixed_logistic_responded",
                       lambda: st.fit_mixed_logistic(md, f"responded ~ {rhs}")),
                      ("mixedlm_score", lambda: st.fit_mixedlm(md, f"score ~ {rhs}"))):
        res, coefs, text = fit()
        full = note + (" " + MIXED_NOTE if name.startswith("mixed_logistic") else "")
        coefs = coefs.assign(note=coefs["note"].astype(str).str.cat([full] * len(coefs),
                                                                   sep=" | "))
        st.save_model((res, coefs, full + "\n\n" + text + "\n"),
                      models / f"cross_b_{name}_condition_x_experiment")
        out[name] = coefs
    # ANCOVA-style: per-fly IAA->IAA response on the per-fly naive IAA
    for metric in METRICS:
        pf = fly_values(cond, metric, by=["condition"])
        wide = pf.pivot_table(index="fly_id", columns="condition", values="value",
                              aggfunc="first")
        info = pf.drop_duplicates("fly_id").set_index("fly_id")
        ad = pd.DataFrame({"pair_value": wide.get("IAA-primed"), "naive_value": wide.get("alone")})
        ad = ad.join(info[["experiment", "batch_key", "start_time_h"]]).dropna(
            subset=["pair_value", "naive_value"]).reset_index()
        ad["exp3"] = (ad["experiment"] == "exp3").astype(int)
        ad["start_time_c"] = centred(ad["start_time_h"])
        res = st.fit_mixedlm(ad, "pair_value ~ naive_value + exp3 + start_time_c",
                             group="batch_key")
        anc_note = (f"ANCOVA-style: one row per fly, pair_value = mean of the fly's two IAA->IAA "
                    f"presentations ({'responded 0/1' if metric == 'rate' else 'score'}), "
                    "naive_value = its alone IAA trial; random intercept per batch folder.")
        st.save_model((res[0], res[1].assign(note=res[1]["note"].astype(str) + " | " + anc_note),
                       res[2] + "\n" + anc_note + "\n"), models / f"cross_b_ancova_{metric}")
        out[f"ancova_{metric}"] = res[1]
    return out


def plot_iaa_iaa(cond: pd.DataFrame, table: pd.DataFrame, folder: Path) -> Path:
    fig, axes = pl.new_figure(1, 2, figsize=(11.5, 5.8))
    plotted = []
    for ax, metric in zip(axes, METRICS):
        summ = st.group_summary(cond, ["exp_label", "condition"], metric)
        pf = fly_values(cond, metric, by=["exp_label", "condition"])
        p = pl.paired_bars(ax, summ, conditions=CONDITIONS, metric=metric,
                           groups=["Exp 2", "Exp 3"], per_fly=pf, colors=COND_COLORS,
                           group_col="exp_label", condition_col="condition",
                           per_fly_col="value", condition_labels=COND_LABELS,
                           legend=metric == "rate")
        w = table[(table["family"] == "within experiment") & (table["metric"] == metric)]
        pl.add_holm_stars(ax, w.rename(columns={"group": "exp_label"}), plotted=p,
                          group_col="exp_label", condition_col="condition")
        plotted.append(p.assign(panel=metric))
    b = table[(table["family"] == "between experiments")].set_index(["metric", "cond_a"])
    pl.title_block(fig, "IAA after IAA: Exp 2 (IAA first) vs Exp 3 (IAA second), each "
                        "beside its naive IAA",
                   n_flies=int(cond["fly_id"].nunique()), n_trials=n_trials(cond),
                   test="paired sign-flip within fly; between experiments fly permutation",
                   correction="Holm within family x metric")
    chg = b.loc[("rate", "IAA-primed - alone")]
    pl.caption(fig, f"{pl.HOLM_CAPTION} (within-fly comparisons). Exp 2 has no click-primed "
                    "IAA (its click comes after IAA). Between experiments (Exp 3 - Exp 2): "
                    f"naive {fmt_effect(b.loc[('rate', 'alone')], 'pp')}; IAA→IAA "
                    f"{fmt_effect(b.loc[('rate', 'IAA-primed')], 'pp')}; IAA→IAA − naive "
                    f"{fmt_effect(chg, 'pp')}, Holm {fmt_p(chg['p_holm'])}, batch "
                    f"{fmt_p(chg['p_batch'])}. Lines: one fly. Exp 3's IAA→IAA scores the 2nd "
                    "IAA, so the 1st IAA's late response can carry over.")
    base = "cross_b_iaa_iaa_across_experiments"
    pl.save_figure(fig, pd.concat(plotted, ignore_index=True), folder / base)
    return folder / base


def _finding_b(cond: pd.DataFrame, table: pd.DataFrame, models: Mapping) -> dict:
    rate = st.group_summary(cond, ["experiment", "condition"], "rate").set_index(
        ["experiment", "condition"])
    w = table[table["family"] == "within experiment"].set_index(["metric", "experiment",
                                                                  "cond_a", "cond_b"])
    b = table[table["family"] == "between experiments"].set_index(["metric", "cond_a"])
    e2 = w.loc[("rate", "exp2", "alone", "IAA-primed")]
    e3 = w.loc[("rate", "exp3", "alone", "IAA-primed")]
    chg, lvl, base = (b.loc[("rate", k)] for k in ("IAA-primed - alone", "IAA-primed", "alone"))
    gee = model_numbers(models["gee_responded"], "iaa_primed:exp3")
    gee_e2 = model_numbers(models["gee_responded"], "iaa_primed")
    mix = model_numbers(models["mixed_logistic_responded"], "iaa_primed:exp3")
    mix_e2 = model_numbers(models["mixed_logistic_responded"], "iaa_primed")
    lmm = model_numbers(models["mixedlm_score"], "iaa_primed:exp3")
    lmm_e2 = model_numbers(models["mixedlm_score"], "iaa_primed")
    anc = model_numbers(models["ancova_rate"], "exp3")
    anc_n = model_numbers(models["ancova_rate"], "naive_value")

    def r(e, c):
        return (f"{rate.loc[(e, c), 'value']:.0f} %" if (e, c) in rate.index else "n/a")
    same_level = not (lvl["p_holm"] < 0.05)
    low_base = base["effect"] > 0
    if same_level and low_base:
        verdict = (f"the IAA→IAA level is the same in both experiments (Exp 3 − Exp 2 "
                   f"{lvl['effect']:+.1f} pp) while Exp 2's naive IAA is lower "
                   f"({base['effect']:+.1f} pp in Exp 3's favour), so Exp 2's apparent "
                   "sensitization is what a low naive baseline rising to the common IAA→IAA "
                   "level looks like, not a larger primed response; neither the baseline "
                   f"difference (Holm {fmt_p(base['p_holm'])}) nor the difference in change "
                   f"(Holm {fmt_p(chg['p_holm'])}) is significant, so this is a reading of "
                   "the estimates, not a test result")
    else:
        verdict = "the data do not separate a priming effect from the baseline difference"
    text = (f"IAA→IAA vs naive IAA. Exp 2 (n = {int(e2['n_flies'])} flies): naive "
            f"{r('exp2', 'alone')} → IAA→IAA {r('exp2', 'IAA-primed')}, within-fly "
            f"{fmt_effect(e2, 'pp')}, {fmt_stats(e2)}. Exp 3 (n = {int(e3['n_flies'])}): naive "
            f"{r('exp3', 'alone')} → IAA→IAA {r('exp3', 'IAA-primed')} (click→IAA "
            f"{r('exp3', 'click-primed')}), {fmt_effect(e3, 'pp')}, {fmt_stats(e3)}. "
            f"Exp 3 − Exp 2: naive {fmt_effect(base, 'pp')}, {fmt_stats(base)}; IAA→IAA "
            f"{fmt_effect(lvl, 'pp')}, {fmt_stats(lvl)}; change (IAA→IAA − naive) "
            f"{fmt_effect(chg, 'pp')}, {fmt_stats(chg)}. GEE (responded ~ condition x experiment "
            f"+ start time, fly clusters): Exp 2 priming log-odds {gee_e2['estimate']:+.2f} "
            f"(95 % CI {gee_e2['ci_lo']:+.2f} to {gee_e2['ci_hi']:+.2f}, {fmt_p(gee_e2['p'])}), "
            f"priming x Exp 3 {gee['estimate']:+.2f} ({gee['ci_lo']:+.2f} to {gee['ci_hi']:+.2f}, "
            f"{fmt_p(gee['p'])}, status {gee['status']}). Bayesian mixed logistic (sensitivity "
            f"only, SDs too narrow to test with): priming {mix_e2['estimate']:+.2f}, priming x "
            f"Exp 3 {mix['estimate']:+.2f}; signs {sign_check(gee_e2, mix_e2)} / "
            f"{sign_check(gee, mix)} with GEE. MixedLM on score (fly intercepts): Exp 2 priming "
            f"{lmm_e2['estimate']:+.2f} points ({lmm_e2['ci_lo']:+.2f} to {lmm_e2['ci_hi']:+.2f}, "
            f"{fmt_p(lmm_e2['p'])}), priming x Exp 3 {lmm['estimate']:+.2f} ({lmm['ci_lo']:+.2f} "
            f"to {lmm['ci_hi']:+.2f}, {fmt_p(lmm['p'])}, status {lmm['status']}). ANCOVA of the "
            f"per-fly IAA→IAA rate on the per-fly naive IAA (batch intercepts): Exp 3 − Exp 2 "
            f"{100 * anc['estimate']:+.1f} pp ({100 * anc['ci_lo']:+.1f} to "
            f"{100 * anc['ci_hi']:+.1f}, {fmt_p(anc['p'])}, status {anc['status']}), naive slope "
            f"{anc_n['estimate']:+.2f} ({fmt_p(anc_n['p'])}). Reading: {verdict}. "
            "Exp 2 has no click-primed IAA.")
    return finding("b_iaa_iaa_across_experiments", text,
                   {"exp2_change": e2[["effect", "ci_lo", "ci_hi", "p", "p_holm", "p_batch",
                                       "n_flies"]].to_dict(),
                    "exp3_change": e3[["effect", "ci_lo", "ci_hi", "p", "p_holm", "p_batch",
                                       "n_flies"]].to_dict(),
                    "change_exp3_minus_exp2": chg[["effect", "ci_lo", "ci_hi", "p", "p_holm",
                                                   "p_batch"]].to_dict(),
                    "gee_priming_x_exp3": gee, "gee_exp2_priming": gee_e2,
                    "mixed_logistic_priming_x_exp3_sensitivity": mix,
                    "mixedlm_score_priming_x_exp3": lmm, "ancova_exp3": anc},
                   primary=True)


# ═════════════════════════════════════════════════════════════════════════════
# c. every prime -> target pair against its click reference
# ═════════════════════════════════════════════════════════════════════════════
def _exp2_change(odor: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Exp 2 per fly x partner: pair (mean of presentations) - alone, as 'change'."""
    d = odor[(odor["experiment"] == "exp2") & odor["block"].isin(["alone", "pair"])]
    pf = fly_values(d, metric, by=["block", "odor"])
    w = pf.pivot_table(index=["fly_id", "odor"], columns="block", values="value",
                       aggfunc="first")
    return (w["pair"] - w["alone"]).rename("value").reset_index()


def exp2_contrasts(odor: pd.DataFrame, metric: str) -> dict:
    """partner -> (a = click change, b = partner change), fly_id-indexed Series."""
    ch = _exp2_change(odor, metric)
    click = ch[ch["odor"] == C.CLICK_ODOR].set_index("fly_id")["value"]
    return {x: (click, ch[ch["odor"] == x].set_index("fly_id")["value"])
            for x in PARTNERS if (ch["odor"] == x).any()}


def exp3_contrasts(odor: pd.DataFrame, metric: str) -> dict:
    """prime -> (a = IAA after click, b = IAA after prime), fly_id-indexed Series."""
    d = odor[(odor["experiment"] == "exp3") & (odor["block"] == "pair") & (odor["odor"] == "IAA")]
    pf = fly_values(d, metric, by=["prime"])
    click = series(pf, prime=C.CLICK_ODOR)
    return {x: (click, series(pf, prime=x)) for x in PARTNERS if (pf["prime"] == x).any()}


def forest_table(odor: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Every prime->target pair of Exp 2 and Exp 3 vs its click reference, sorted by
    effect (largest first), Holm over all rows."""
    batch_of = batch_of_flies(odor)
    thr = C.RESPONSE_THRESHOLD
    rows = []
    for x, (a, b) in exp2_contrasts(odor, metric).items():
        r = paired(a, b, batch_of, scale=SCALE[metric])
        rows.append({"experiment": "exp2", "prime": "IAA", "target": x,
                     "label": f"Exp 2: IAA → {x}", "reference": "IAA → click",
                     "contrast": "(X pair - X alone) - (click pair - click alone)", **r})
    d3 = odor[(odor["experiment"] == "exp3") & (odor["block"] == "pair")]
    for x, (a, b) in exp3_contrasts(odor, metric).items():
        r = paired(a, b, batch_of, scale=SCALE[metric])
        o1 = pd.to_numeric(d3.loc[d3["prime"] == x, "odor1_score"], errors="coerce")
        rows.append({"experiment": "exp3", "prime": x, "target": "IAA",
                     "label": f"Exp 3: {x} → IAA", "reference": "click → IAA",
                     "contrast": "IAA after X - IAA after click",
                     "prime_response_rate": float(100 * (o1 >= thr).mean()) if len(o1) else np.nan,
                     **r})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t = t.assign(metric=metric, unit=UNIT[metric], test=TEST_PAIRED)
    t = finish(t, None, correction=f"Holm over the whole forest ({len(t)} pairs) within metric")
    return t.sort_values("effect", ascending=False, kind="stable").reset_index(drop=True)


EB = "Ethyl Butyrate"
EB_CONDS = {"exp2": ["alone", "after IAA"], "exp3": ["alone", "after EB", "after click"]}
EB_COLORS = {"alone": GRAY, "after IAA": PAIR_BLUE, "after EB": PAIR_BLUE,
             "after click": CLICK_RED}


def eb_iaa_trials(odor: pd.DataFrame) -> pd.DataFrame:
    """Exp 2: EB and click, alone and after IAA; Exp 3: IAA alone, after EB, after click."""
    e2 = odor[(odor["experiment"] == "exp2") & odor["odor"].isin([EB, C.CLICK_ODOR])]
    e2 = e2.assign(group=np.where(e2["odor"] == EB, EB, CLICK_LABEL),
                   condition=np.where(e2["block"] == "alone", "alone", "after IAA"))
    e3 = odor[(odor["experiment"] == "exp3") & (odor["odor"] == "IAA")]
    cond = np.select([(e3["block"] == "alone").to_numpy(),
                      (e3["prime"] == EB).fillna(False).to_numpy(bool),
                      (e3["prime"] == C.CLICK_ODOR).fillna(False).to_numpy(bool)],
                     EB_CONDS["exp3"], default="")
    e3 = e3.assign(group="IAA", condition=cond)
    e3 = e3[e3["condition"] != ""]
    return pd.concat([e2, e3], ignore_index=True)


def eb_iaa_table(d: pd.DataFrame) -> pd.DataFrame:
    batch_of = batch_of_flies(d)
    comps = [("exp2", EB, "alone", "after IAA"), ("exp2", CLICK_LABEL, "alone", "after IAA"),
             ("exp3", "IAA", "alone", "after EB"), ("exp3", "IAA", "alone", "after click"),
             ("exp3", "IAA", "after click", "after EB")]
    rows = []
    for metric in METRICS:
        pf = fly_values(d, metric, by=["experiment", "group", "condition"])
        for e, g, ca, cb in comps:
            a = series(pf, experiment=e, group=g, condition=ca)
            b = series(pf, experiment=e, group=g, condition=cb)
            if a.empty or b.empty:
                continue
            rows.append({"metric": metric, "experiment": e, "group": g, "cond_a": ca,
                         "cond_b": cb, "comparison": f"{EXP_LABELS[e]} {g}: {cb} - {ca}",
                         **paired(a, b, batch_of, scale=SCALE[metric]), "test": TEST_PAIRED,
                         "unit": UNIT[metric]})
    return finish(pd.DataFrame(rows), "metric", correction="Holm over the 5 comparisons "
                                                             "within metric")


def plot_eb_iaa(d: pd.DataFrame, table: pd.DataFrame, folder: Path) -> Path:
    fig, axes = pl.new_figure(2, 2, figsize=(10.5, 8.6), width_ratios=[2, 1.25])
    plotted = []
    for i, metric in enumerate(METRICS):
        for j, (e, groups) in enumerate((("exp2", [EB, CLICK_LABEL]), ("exp3", ["IAA"]))):
            ax = axes[i, j]
            sub = d[d["experiment"] == e]
            summ = st.group_summary(sub, ["group", "condition"], metric)
            pf = fly_values(sub, metric, by=["group", "condition"])
            p = pl.paired_bars(ax, summ, conditions=EB_CONDS[e], metric=metric, groups=groups,
                               per_fly=pf, colors=EB_COLORS, group_col="group",
                               condition_col="condition", per_fly_col="value",
                               legend=i == 0)
            t = table[(table["metric"] == metric) & (table["experiment"] == e)]
            pl.add_holm_stars(ax, t, plotted=p, group_col="group", condition_col="condition")
            if i == 0:
                ax.set_title(f"{EXP_LABELS[e]}: " + ("IAA → EB and IAA → click"
                                                     if e == "exp2" else "EB → IAA and click → IAA"),
                             fontsize=9, loc="left", color=pl.INK_SECONDARY, pad=24)
            if j == 1:
                ax.set_ylabel("")
            plotted.append(p.assign(panel=f"{metric} {e}"))
    pl.title_block(fig, "Ethyl butyrate and IAA in both orders, each beside its click reference",
                   n_flies=int(d["fly_id"].nunique()), n_trials=n_trials(d),
                   test="paired sign-flip within fly", correction="Holm over 5 comparisons x metric")
    pl.caption(fig, f"{pl.HOLM_CAPTION}. Exp 2 scores the 2nd odor (EB or the click) after IAA; "
                    "Exp 3 scores IAA after EB or after the click, so EB's own late response can "
                    "carry into the IAA score. Lines: one fly. The forest gives every pair's "
                    "click-referenced effect.")
    base = "cross_c_eb_iaa_bars"
    pl.save_figure(fig, pd.concat(plotted, ignore_index=True), folder / base)
    return folder / base


def plot_forest(forest: pd.DataFrame, odor: pd.DataFrame, folder: Path) -> Path:
    fig, axes = pl.new_figure(1, 2, figsize=(12.5, 6.4))
    plotted = []
    for ax, metric in zip(axes, METRICS):
        t = forest[forest["metric"] == metric]
        p = pl.forest_plot(ax, t, "effect", "ci_lo", "ci_hi", "label", p_holm="p_holm",
                           xlabel=("Δ PER vs click reference (pp)" if metric == "rate"
                                   else "Δ PER score vs click reference"),
                           color=PAIR_BLUE)
        ax.set_title("Response rate" if metric == "rate" else "Score", fontsize=9,
                     loc="left", color=pl.INK_SECONDARY)
        plotted.append(p.assign(metric=metric))
    d = odor[odor["experiment"].isin(["exp2", "exp3"])]
    pl.title_block(fig, "Every prime → target pair vs its click reference (Exp 2 and Exp 3)",
                   n_flies=int(d["fly_id"].nunique()), n_trials=n_trials(d),
                   test="paired sign-flip within fly", correction="Holm over all 14 pairs x metric")
    pl.caption(fig, f"{pl.HOLM_CAPTION}. Exp 2: per fly (X after IAA − X alone) − (click after "
                    "IAA − click alone). Exp 3: per fly IAA after X − IAA after click (X's own "
                    "late response can carry into the IAA score). Points: mean over flies; "
                    "whiskers: BCa 95 % CI. Sorted by effect.")
    base = "cross_c_forest"
    pl.save_figure(fig, pd.concat(plotted, ignore_index=True), folder / base)
    return folder / base


def _finding_c(forest: pd.DataFrame) -> dict:
    def row(e, metric):
        t = forest[(forest["metric"] == metric) & (forest["experiment"] == e)
                   & ((forest["target"] == EB) | (forest["prime"] == EB))]
        return t.iloc[0] if len(t) else None
    parts, nums = [], {}
    for e, lab in (("exp2", "IAA → EB (Exp 2)"), ("exp3", "EB → IAA (Exp 3)")):
        for metric in METRICS:
            r = row(e, metric)
            if r is None:
                continue
            nums[f"{e}_{metric}"] = {k: float(r[k]) for k in ("effect", "ci_lo", "ci_hi", "p",
                                                              "p_holm", "p_batch", "n_flies")}
            if metric == "rate":
                parts.append(f"{lab}: {fmt_effect(r, 'pp')}, n = {int(r['n_flies'])} flies, "
                             f"{fmt_stats(r)}")
            else:
                parts.append(f"score {fmt_effect(r, 'points', digits=2)}, Holm "
                             f"{fmt_p(r['p_holm'])}")
    fr = forest[forest["metric"] == "rate"]
    sig = fr[fr["p_holm"] < 0.05]
    top = fr.iloc[0]
    text = (f"Click-referenced effects ({len(fr)} prime→target pairs, Holm over the forest). "
            + "; ".join(parts) + f". Largest rate effect: {top['label']} "
            f"{fmt_effect(top, 'pp')}. Pairs significant after Holm (rate): "
            + (", ".join(sig["label"]) if len(sig) else "none") + ".")
    return finding("c_eb_iaa_bidirectional", text, nums)


# ═════════════════════════════════════════════════════════════════════════════
# d. paraffin / click noise floor
# ═════════════════════════════════════════════════════════════════════════════
FLAG_NONE = "not distinguishable from paraffin"


def noise_floor_table(odor: pd.DataFrame) -> pd.DataFrame:
    """Per cohort x odor x metric: naive odor - the same fly's naive paraffin/click."""
    naive = odor[odor["block"].isin(NAIVE_BLOCKS)]
    batch_of = batch_of_flies(naive)
    rows = []
    for cohort in [c for c in C.COHORTS if c in set(naive["cohort"])]:
        sub = naive[naive["cohort"] == cohort]
        exp = sub["experiment"].iloc[0]
        has_p = set(sub.loc[sub["odor"] == C.CLICK_ODOR, "fly_id"])
        if not has_p:
            rows.append({"cohort": cohort, "experiment": exp, "odor": "", "metric": "",
                         "flag": "skipped", "n_flies": 0,
                         "note": "never tested on paraffin (no naive paraffin/click trial): "
                                 "no within-fly noise floor"})
            log.info("noise floor: %s skipped (no naive paraffin)", cohort)
            continue
        n_all = sub["fly_id"].nunique()
        ref_kind = "click" if exp != "exp1" else "paraffin (click after 2026-11-10)"
        note = (f"{len(has_p)} of {n_all} flies have a naive {ref_kind} trial; only they enter"
                if len(has_p) < n_all else f"all {n_all} flies have a naive {ref_kind} trial")
        for metric in METRICS:
            pf = fly_values(sub, metric, by=["odor"])
            ref = series(pf, odor=C.CLICK_ODOR)
            for x in [o for o in PARTNERS if (pf["odor"] == o).any()]:
                r = paired(ref, series(pf, odor=x), batch_of, scale=SCALE[metric])
                rows.append({"cohort": cohort, "experiment": exp, "odor": x, "metric": metric,
                             **r, "test": TEST_PAIRED, "unit": UNIT[metric], "note": note})
    t = pd.DataFrame(rows)
    tested = t[t["flag"] != "skipped"].copy() if "flag" in t else t.copy()
    skipped = t[t["flag"] == "skipped"] if "flag" in t else t.iloc[0:0]
    if tested.empty:
        return skipped.reset_index(drop=True)
    tested = finish(tested, ["cohort", "metric"],
                    correction="Holm over the 7 odors within cohort x metric")
    sig = tested["p_holm"] < 0.05
    tested["flag"] = np.where(~sig, FLAG_NONE, np.where(tested["effect"] > 0, "above paraffin",
                                                        "below paraffin"))
    return pd.concat([tested, skipped], ignore_index=True)


def plot_noise_floor(table: pd.DataFrame, odor: pd.DataFrame, folder: Path) -> Path:
    tested = table[table["flag"] != "skipped"]
    cohorts = [c for c in C.COHORTS if c in set(tested["cohort"])]
    skipped = [c for c in table.loc[table["flag"] == "skipped", "cohort"]]
    odors = [o for o in PARTNERS if o in set(tested["odor"])]
    k = max(len(cohorts), 1)
    fig, axes = pl.new_figure(2, k, figsize=(2.15 * k + 1.6, 7.6), sharey=True, squeeze=False)
    plotted = []
    y_of = {o: len(odors) - 1 - i for i, o in enumerate(odors)}
    for i, metric in enumerate(METRICS):
        tm = tested[tested["metric"] == metric]
        span = np.nanmax(np.abs(tm[["ci_lo", "ci_hi"]].to_numpy(float))) if len(tm) else 1.0
        lim = float(np.ceil(span / (10 if metric == "rate" else 0.5)) *
                    (10 if metric == "rate" else 0.5)) or 1.0
        for j, cohort in enumerate(cohorts):
            ax = axes[i, j]
            pl.style_axes(ax, grid="x")
            ax.axvline(0, color=pl.BASELINE, linewidth=0.8, zorder=1)
            t = tm[tm["cohort"] == cohort]
            for r in t.itertuples():
                y = y_of[r.odor]
                c = C.ODOR_COLORS.get(r.odor, pl.CATEGORICAL[0])
                filled = r.flag != FLAG_NONE
                ax.hlines(y, r.ci_lo, r.ci_hi, color=c, linewidth=1.5, zorder=2)
                ax.scatter([r.effect], [y], s=34, zorder=3, linewidths=1.4,
                           facecolors=c if filled else pl.SURFACE, edgecolors=c)
                s = st.stars(r.p_holm)
                if s:
                    ax.text(lim * 1.02, y, s, ha="left", va="center", fontsize=8,
                            color=pl.INK_PRIMARY, clip_on=False)
                plotted.append({"cohort": cohort, "metric": metric, "odor": r.odor, "y": y,
                                "effect": r.effect, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi,
                                "p_holm": r.p_holm, "flag": r.flag, "n_flies": r.n_flies})
            ax.set_xlim(-lim, lim)
            ax.set_yticks(list(y_of.values()))
            ax.set_yticklabels(list(y_of.keys()))
            ax.set_ylim(-0.6, len(odors) - 0.4)
            if i == 0:
                n = int(t["n_flies"].max()) if len(t) else 0
                ax.set_title(f"{COHORT_LABELS.get(cohort, cohort).replace(chr(10), ' ')}\n"
                             f"n = {n} flies", fontsize=8.5, color=pl.INK_SECONDARY)
            if j == 0:
                ax.set_ylabel("Response rate (pp)" if metric == "rate" else "Score",
                              color=pl.INK_SECONDARY)
            ax.set_xlabel("odor − paraffin (pp)" if metric == "rate" else "odor − paraffin",
                          fontsize=8)
    naive =odor[odor["block"].isin(NAIVE_BLOCKS) & odor["cohort"].isin(cohorts)]
    pl.title_block(fig, "Naive response to each odor minus the same fly's paraffin / click",
                   n_flies=int(tested.drop_duplicates("cohort")["n_flies"].sum()
                               if len(tested) else 0),
                   n_trials=n_trials(naive), test="paired sign-flip within fly",
                   correction="Holm over 7 odors within cohort x metric")
    pl.caption(fig, f"{pl.HOLM_CAPTION}. Filled circle: differs from paraffin (Holm); open "
                    f"circle: {FLAG_NONE}. Skipped: {', '.join(skipped) or 'none'} (never tested on "
                    "paraffin). Hex/3Oct: only flies tested on paraffin. Exp 2/3 reference is "
                    "the master-valve click; the 4 known-bad 0.025 s clicks are left out. "
                    "Whiskers: BCa 95 % CI over flies.")
    base = "cross_d_paraffin_noise_floor"
    pl.save_figure(fig, pd.DataFrame(plotted), folder / base)
    return folder / base


def _finding_d(table: pd.DataFrame) -> dict:
    t = table[(table["flag"] != "skipped") & (table["metric"] == "rate")]
    parts, nums = [], {}
    for cohort, g in t.groupby("cohort", sort=False):
        none = g[g["flag"] == FLAG_NONE]["odor"].tolist()
        above = g[g["flag"] == "above paraffin"]["odor"].tolist()
        below = g[g["flag"] == "below paraffin"]["odor"].tolist()
        nums[cohort] = {"n_flies": int(g["n_flies"].max()), "not_distinguishable": none,
                        "above": above, "below": below}
        parts.append(f"{cohort} (n = {int(g['n_flies'].max())}): above paraffin "
                     f"{', '.join(above) or 'none'}; not distinguishable "
                     f"{', '.join(none) or 'none'}" + (f"; below {', '.join(below)}" if below
                                                       else ""))
    skipped = table.loc[table["flag"] == "skipped", "cohort"].tolist()
    raw = table[(table["flag"] != "skipped") & (table["p"] < 0.05)]
    near = "; ".join(f"{r.cohort} {r.odor} {r.metric} {r.effect:+.2f} {r.unit} (p = {r.p:.3f}, "
                     f"Holm {r.p_holm:.3f})" for r in raw.itertuples())
    text = ("Noise floor (naive odor − same fly's naive paraffin/click, response rate, Holm over "
            "7 odors per cohort): " + "; ".join(parts) + ". Skipped: "
            + (", ".join(skipped) or "none") + " (never tested on paraffin). Uncorrected p < 0.05 "
            "(rate or score) that do not survive Holm: " + (near or "none") + ".")
    nums["uncorrected_p_below_0.05"] = raw[["cohort", "odor", "metric", "effect", "p",
                                            "p_holm"]].to_dict("records")
    return finding("d_paraffin_noise_floor", text, nums)


# ═════════════════════════════════════════════════════════════════════════════
# run
# ═════════════════════════════════════════════════════════════════════════════
READMES = {
    "": """# 05_cross_experiment
Cross-experiment analyses from `per_analysis.cross.run()`; stats tables are in `tables/cross_*.csv`.
a: naive IAA by cohort; b: IAA→IAA in Exp 2 vs Exp 3 beside the naive IAA; c: every prime→target pair vs its click reference (EB↔IAA highlighted); d: each odor vs the same fly's paraffin/click.
Light probes and the 4 known-bad 0.025 s clicks are left out; the fly is the unit; stars = Holm-adjusted p.""",
    "a_iaa_baseline_by_cohort": """# a_iaa_baseline_by_cohort
Naive IAA (exp1 pre / exp2-3 alone; one trial per fly) across the six cohorts: rate bars (Wilson CI) and per-fly score strip + box.
Kruskal-Wallis, 15 pairwise fly-level permutations (Holm) and whole-batch label permutation with session strata where cohorts share a session (`p_batch`).
Table: `tables/cross_a_iaa_baseline_by_cohort.csv`.""",
    "b_iaa_iaa_across_experiments": """# b_iaa_iaa_across_experiments
IAA→IAA (and Exp 3's click→IAA) beside each cohort's naive IAA, Exp 2 vs Exp 3: does Exp 2's "sensitization" reflect its low baseline?
Within-fly sign-flip tests, between-experiment permutations (session strata), and models in `../models/cross_b_*` (GEE, Bayesian mixed logistic, MixedLM, per-fly ANCOVA).
Table: `tables/cross_b_iaa_iaa_across_experiments.csv`. Exp 2 has no click-primed IAA.""",
    "c_eb_iaa_bidirectional": """# c_eb_iaa_bidirectional
`cross_c_forest`: every prime→target pair of Exp 2 and Exp 3 vs its click reference (Exp 2: (X pair − X alone) − (click pair − click alone); Exp 3: IAA after X − IAA after click), sorted by effect, Holm over the forest.
`cross_c_eb_iaa_bars`: IAA→EB (Exp 2) and EB→IAA (Exp 3) beside their click references, per-fly lines.
Tables: `tables/cross_c_forest.csv`, `tables/cross_c_eb_iaa_bars.csv`.""",
    "d_paraffin_noise_floor": """# d_paraffin_noise_floor
Per cohort, each odor's naive response minus the same fly's naive paraffin/click response (within fly), BCa 95 % CI, Holm over 7 odors per cohort x metric; `flag` marks odors not distinguishable from paraffin.
IAA-Sensitivity was never tested on paraffin (skipped); Hex/3Oct use only flies tested on paraffin; the 4 known-bad 0.025 s clicks are left out.
Table: `tables/cross_d_paraffin_noise_floor.csv`.""",
    "models": """# models
Every cross-experiment model, via `stats.save_model`: `<name>.txt` (summary) and `<name>_coefs.csv` (term, estimate, CI, p, status).
`cross_b_*`: responded / score ~ condition x experiment + centred start time (GEE with fly clusters; Bayesian mixed logistic, fit_vb; MixedLM) and the per-fly ANCOVA of IAA→IAA on naive IAA. Check `status` before use.""",
}


def run(trials: pd.DataFrame, out: Path, *, tables: Path) -> list[dict]:
    """All four cross-experiment analyses; figures in ``<out>/05_cross_experiment``,
    tables in ``tables``. Returns the findings."""
    out, tables = Path(out), Path(tables)
    root = out / SUBDIR
    for sub, text in READMES.items():
        readme(root / sub if sub else root, text)
    odor = odor_trials(trials)
    findings = []

    log.info("cross a: naive IAA by cohort")
    ta = iaa_baseline_table(odor)
    save_table(ta, tables, "cross_a_iaa_baseline_by_cohort")
    plot_iaa_baseline(odor, ta, root / "a_iaa_baseline_by_cohort")
    findings.append(_finding_a(ta, odor))

    log.info("cross b: IAA->IAA across experiments")
    cond = iaa_condition_trials(odor)
    if cond["experiment"].nunique() == 2:
        tb = iaa_iaa_table(cond)
        save_table(tb, tables, "cross_b_iaa_iaa_across_experiments")
        models = iaa_iaa_models(cond, root / "models")
        plot_iaa_iaa(cond, tb, root / "b_iaa_iaa_across_experiments")
        findings.append(_finding_b(cond, tb, models))
    else:
        log.warning("cross b skipped: need both pair experiments")

    log.info("cross c: prime->target forest")
    forest = pd.concat([forest_table(odor, m) for m in METRICS], ignore_index=True)
    if len(forest):
        save_table(forest, tables, "cross_c_forest")
        plot_forest(forest, odor, root / "c_eb_iaa_bidirectional")
        ebd = eb_iaa_trials(odor)
        tc = eb_iaa_table(ebd)
        save_table(tc, tables, "cross_c_eb_iaa_bars")
        plot_eb_iaa(ebd, tc, root / "c_eb_iaa_bidirectional")
        findings.append(_finding_c(forest))

    log.info("cross d: paraffin noise floor")
    td = noise_floor_table(odor)
    save_table(td, tables, "cross_d_paraffin_noise_floor")
    plot_noise_floor(td, odor, root / "d_paraffin_noise_floor")
    findings.append(_finding_d(td))
    return findings


__all__ = ["run", "odor_trials", "fly_values", "two_group", "paired", "batch_perm",
           "hybrid_strata", "iaa_baseline_table", "iaa_condition_trials", "iaa_iaa_table",
           "forest_table", "noise_floor_table", "save_table", "finish", "finding", "PARTNERS"]
