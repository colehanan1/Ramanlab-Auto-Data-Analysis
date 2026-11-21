"""Exp 2: IAA first (IAA-PairFirst-Control-1-1).

Alone block: all 8 odors once (~5-35 min into the session). Pair block (~65-140 min):
IAA for 10 s, a 1 s gap, then partner X, scored (``odor`` = X, ``prime`` = IAA); each X
twice (``presentation`` 1/2). X = the 7 odors and the click (Paraffin Oil, master valve
only, no odor), the within-block, time-matched reference.

``c_vs_paraffin_reference`` reports both forms, Holm across the 7 odors:
(i) direct: per fly, X in pair - click in pair (both right after IAA);
(ii) controlled: (X pair - X alone) - (click pair - click alone), which also removes
odor identity (each X's own alone response).

Every analysis runs per clock window (day / evening); a window without flies is skipped.
The evening window is one session (Nov 17), so its whole-batch p cannot reach 0.05.
Entry point: :func:`run`.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import pairs  # noqa: E402
from . import plots as pl  # noqa: E402

LOG = logging.getLogger(__name__)

EXPERIMENT = "exp2"
SECTION = "exp2"
FOLDER = "03_exp2_iaa_first"
SUB = {"a": "a_alone_vs_pair_pooled", "b": "b_presentation_1_vs_2",
       "c": "c_vs_paraffin_reference", "d": "d_shared_baseline_diagnostic",
       "models": "models"}
LABEL = "Exp 2 (IAA first)"
XLABEL = "Scored odor (alone, or 1 s after IAA's 10 s)"
XLABEL_PAIR = "Scored odor, 1 s after IAA's 10 s"
METRIC_WORDS = {"rate": "PER rate (% of trials; effects in percentage points)",
                "score": "PER score (-1..5)"}
METRIC_TITLE = {"rate": "PER rate", "score": "PER score"}
MAIN_COLOR = pl.CATEGORICAL[0]

COND_LABELS = {
    "a": {"alone": "Alone (pre-block)", "pair": "Right after IAA (1st + 2nd pooled)"},
    "b": {"alone": "Alone (pre-block)", "1st": "Right after IAA, 1st time",
          "2nd": "Right after IAA, 2nd time"},
    "c": {"click": "Click right after IAA", "odor": "Odor right after IAA"},
    "cc": {"click": "Click: change alone → after IAA",
           "odor": "Odor: change alone → after IAA"},
}
TITLES = {
    "a": "Each odor alone vs right after IAA (presentations pooled)",
    "b": "Each odor alone, then right after IAA: 1st vs 2nd time",
    "c": "Each odor right after IAA vs the click right after IAA (within fly)",
    "cc": "Each odor's change (alone → after IAA) minus the click's change",
    "d": "Direction of each odor's change, alone → right after IAA",
}
CAPTIONS = {
    "a": pairs.BARS_NOTE + " Alone (~5-35 min) vs pair (~65-140 min) also holds time in "
         "the rig and re-testing; see c_vs_paraffin_reference for the click-controlled "
         "forms. " + pairs.HOLM_SENTENCE,
    "b": pairs.BARS_NOTE + " Brackets: 1st vs 2nd time after IAA. "
         + pairs.HOLM_SENTENCE,
    "c": pairs.BARS_NOTE + " Lower panel: per fly, the odor right after IAA minus the "
         "click right after IAA (same block, same slot), mean over flies with its 95 % "
         "BCa CI. " + pairs.HOLM_SENTENCE,
    "cc": "Bars: pooled change alone → right after IAA (trials pooled on each side) ± SEM "
          "over flies of the per-fly change; lines: one fly's change. Lower panel: per "
          "fly, (odor pair - odor alone) - (click pair - click alone), mean over flies "
          "with its 95 % BCa CI; two-sided exact sign-flip. " + pairs.HOLM_SENTENCE,
    "d": "Bars: pooled change pair - alone per odor (trials pooled on each side) ± SEM "
         "over flies; dots: flies. Each odor has its own alone trial, but all sit in one "
         "alone block, so a fly-level shift moves every bar together; the count is "
         "tested by flipping each fly's whole row of changes jointly "
         "(stats.direction_count_test), and whole batches for the batch p. Click "
         "(hatched) = reference, not counted.",
}
READMES = {
    "": ["# Exp 2 · IAA first (IAA-PairFirst-Control-1-1)",
         "Alone block (8 odors, ~5-35 min), then pairs: IAA 10 s, 1 s gap, odor X scored "
         "(~65-140 min), each X twice; the click (Paraffin Oil, master valve, no odor) is "
         "the time-matched reference.",
         "Every analysis runs per clock window (`_day` / `_evening` in each file name; "
         "evening = one session, Nov 17); light probes are left out. Tables: "
         "`../tables/<figure base name>.csv`."],
    "a": ["# a · Each odor alone vs right after IAA (presentations pooled)",
          "Paired sign-flip on per-fly means, 95 % BCa CI, Holm across the 8 partners. "
          "Not controlled for time in the rig or re-testing (see c).",
          "Files: `exp2_alone_vs_pair_pooled_<rate|score>_<window>`."],
    "b": ["# b · Each odor alone, then 1st and 2nd time right after IAA",
          "Bars for alone / 1st / 2nd; paired sign-flip of 1st vs 2nd, Holm across the 8 "
          "partners per panel.",
          "Files: `exp2_presentation_1_vs_2_<rate|score>_<window>`."],
    "c": ["# c · Each odor vs the click, both right after IAA",
          "`*_direct_*`: per fly, X in pair - click in pair. `*_controlled_*`: per fly, "
          "(X pair - X alone) - (click pair - click alone), which removes odor identity.",
          "Sign-flip on per-fly values, BCa CI, Holm across the 7 odors; p_batch flips "
          "whole batch folders."],
    "d": ["# d · Shared-baseline diagnostic",
          "All alone trials sit in one block, so a fly-level shift moves every odor's "
          "change together. How many odors go up vs down, against a null that flips each "
          "fly's whole row of changes jointly (`stats.direction_count_test`).",
          "Files: `exp2_shared_baseline_diagnostic_<rate|score>_<window>`."],
    "models": ["# models · Exp 2",
               "`alone_vs_pair`: responded ~ C(condition, ref alone) * C(partner, ref "
               "click) + 2nd presentation + start_time_c + position_in_block; the "
               "condition x partner terms are the controlled change on the log-odds scale.",
               "GEE logistic (fly clusters, robust SE) is the model of record; Bayesian "
               "mixed logistic (VB), MixedLM and ordinal (fly-cluster bootstrap) on score "
               "are sensitivity checks. Status and role per fit in "
               "`exp2_model_status_<window>.csv` (check it before reading a coefficient)."],
}


def _texts(window: str, metric: str, what: str, res: pairs.Result) -> str:
    return (f"{LABEL}, {window} window, {METRIC_WORDS[metric]}: {what} "
            f"({res.n_flies} flies, {res.n_trials} trials; {res.test}; {res.correction}). ")


def _models(wdf: pd.DataFrame, window: str, folder: Path, *, ordinal_boot: int) -> list[dict]:
    T = pairs.Term
    frame = pairs.model_frame(wdf)
    r = np.corrcoef(frame["position_in_block"], frame["minutes_since_block_start"])[0, 1] \
        if len(frame) > 2 else np.nan
    note = (f"minutes_since_block_start left out: r = {r:.5f} with position_in_block (one "
            "trial every 5 min); second_presentation codes C(presentation) (0 for alone "
            "and 1st, so it is not collinear with condition).")
    terms = [T(f"C(condition, Treatment('alone')) * "
               f"C(partner, Treatment('{pairs.REFERENCE}'))", ("condition", "partner"), True),
             T("second_presentation", ("second_presentation",)),
             T("start_time_c", ("start_time_c",)),
             T("position_in_block", ("position_in_block",))]
    rows = pairs.fit_model_set(frame, terms, name="alone_vs_pair", experiment=EXPERIMENT,
                               window=window, folder=folder, ordinal_boot=ordinal_boot,
                               log=LOG, note=note,
                               separation=pairs.Separation(("condition", "partner"), "partner",
                                                           pairs.REFERENCE))
    pairs.write_model_status(rows, folder, EXPERIMENT, window)
    return rows


def _model_finding(rows: list[dict], window: str) -> dict:
    status = {f"{r['model']}.{r['fit']}": r["status"] for r in rows}
    inter = pairs.key_terms(rows, name="alone_vs_pair", fit="gee", pattern="]:C(partner")
    gee_ok = next((r["status"] for r in rows if r["fit"] == "gee"), "failed")
    text = (f"{LABEL}, {window} window, models (status per fit: "
            + ", ".join(f"{k} {v}" for k, v in status.items()) + "). ")
    flips = pairs.sign_disagreements(rows, name="alone_vs_pair", pattern="]:C(partner")
    if gee_ok != "failed" and len(inter):
        text += ("Model of record, GEE logistic: condition x partner terms = the "
                 "controlled change "
                 "(odor's change alone → after IAA vs the click's), as odds-ratio ratios "
                 "with Wald 95 % CI and uncorrected p: " + pairs.describe_odds(inter) + ".")
    else:
        reason = next((r["note"] for r in rows if r["fit"] == "gee"), "")
        text += ("The GEE (model of record) did not fit"
                 + (f" ({reason})" if reason else "") + "; the sensitivity-check fits are "
                 "not read in its place.")
    text += pairs.gee_left_out(rows, "alone_vs_pair")
    text += pairs.describe_sign_flips(flips) or (
        " The mixed logistic (sensitivity check) agrees in sign with the GEE on these terms."
        if gee_ok != "failed" else "")
    nums = {"window": window, "status": status,
            "glmm_sign_disagreements": [pairs.term_level(f["term"]) for f in flips]}
    if len(inter):
        nums["gee_interaction_log_odds"] = {pairs.term_level(t): e for t, e in
                                            zip(inter["term"], inter["estimate"])}
        nums["gee_interaction_p"] = {pairs.term_level(t): p for t, p in
                                     zip(inter["term"], inter["p"])}
    return pairs.finding(SECTION, "models", primary=False, window=window, metric="rate",
                         text=text, numbers=nums)


def _direction_finding(dd: pairs.Result, window: str, metric: str) -> dict:
    d = dd.extra["direction"]
    n = len(dd.extra["odors"])
    row = dd.table.set_index("group").loc["direction count"]
    text = (f"{LABEL}, {window} window, {METRIC_WORDS[metric]}: shared-baseline "
            f"diagnostic ({dd.n_flies} flies). Each odor's change alone → right after IAA: "
            f"{d['n_up']} of {n} odors up, {d['n_down']} down, {d['n_zero']} unchanged; "
            f"joint fly-flip p = {pairs.fmt_p(row['p'])} (whole-batch p = "
            f"{pairs.fmt_p(row['p_batch'])}; the naive binomial count would give p = "
            f"{pairs.fmt_p(d['p_naive_binomial'])}); mean cross-odor correlation of per-fly "
            f"changes {d['mean_cross_odor_corr']:.2f}.")
    return pairs.finding(SECTION, "shared_baseline_diagnostic", primary=False, window=window,
                         metric=metric, text=text,
                         numbers={"window": window, "metric": metric,
                                  "n_flies": dd.n_flies, "n_up": d["n_up"],
                                  "n_down": d["n_down"], "n_zero": d["n_zero"],
                                  "p": row["p"], "p_batch": row["p_batch"],
                                  "p_naive_binomial": d["p_naive_binomial"],
                                  "mean_cross_odor_corr": d["mean_cross_odor_corr"]})


def run(trials: pd.DataFrame, out, *, tables, fit_models: bool = True,
        ordinal_boot: int = 200, n_boot: int | None = None,
        n_perm: int | None = None) -> list[dict]:
    """Exp 2 analyses a-d and the models, per clock window, into ``out/03_exp2_iaa_first``
    (tables into ``tables``). Returns the findings (dicts: section, analysis, primary,
    text, numbers, window, metric)."""
    t0 = time.perf_counter()
    out, tables = Path(out), Path(tables)
    root = out / FOLDER
    kw = {"n_boot": n_boot, "n_perm": n_perm}
    d = pairs.prepare(trials, EXPERIMENT, log=LOG)
    LOG.info("exp2: %d rows, %d flies", len(d), d["fly_id"].nunique())
    for key, lines in READMES.items():
        pairs.write_readme(root / SUB[key] if key else root, lines)
    windows = pairs.split_windows(d, log=LOG)
    if not windows:
        LOG.info("exp2: no flies in any window, nothing to analyse")
        return [pairs.finding(SECTION, "skipped", primary=False, window="", metric="",
                              text="Exp 2: no flies in any clock window.", numbers={})]
    jobs: list[pairs.Job] = []
    findings: list[dict] = []
    for window, wdf in windows:
        n_batches = wdf["batch"].nunique()
        LOG.info("exp2 %s window: %d flies, %d batches, %d sessions", window,
                 wdf["fly_id"].nunique(), n_batches,
                 wdf["session"].nunique() if "session" in wdf.columns else -1)
        long = pairs.scored_long(wdf, EXPERIMENT)
        LOG.debug("exp2 %s: long %s", window, long.shape)
        for metric in pairs.METRICS:
            a = pairs.alone_vs_pair(long, metric, family="partners", **kw)
            b = pairs.presentation_1_vs_2(long, metric, family="partners", **kw)
            c = pairs.reference_contrast(long, metric, family="odors",
                                         analysis="vs_paraffin_reference_direct", **kw)
            cc = pairs.controlled_change(long, metric, family="odors",
                                         analysis="vs_paraffin_reference_controlled", **kw)
            dd = pairs.shared_baseline(long, metric, family="odors", **kw)

            def job(key, res, sub, analysis, xlabel=XLABEL, **extra):
                return pairs.Job(folder=root / SUB[sub],
                                 stem=pairs.stem(EXPERIMENT, analysis, metric, window),
                                 kind="diagnostic" if key == "d" else "bars", result=res,
                                 title=f"{LABEL} · {TITLES[key]} · {METRIC_TITLE[metric]}",
                                 window=window, xlabel=xlabel, caption=CAPTIONS[key],
                                 cond_labels=COND_LABELS.get(key, COND_LABELS["c"]),
                                 **extra)
            jobs += [
                job("a", a, "a", "alone_vs_pair_pooled"),
                job("b", b, "b", "presentation_1_vs_2"),
                job("c", c, "c", "vs_paraffin_reference_direct", xlabel=XLABEL_PAIR,
                    effect_series=[("odor - click, both right after IAA", c.table,
                                    MAIN_COLOR)]),
                job("cc", cc, "c", "vs_paraffin_reference_controlled",
                    effect_series=[("(odor pair - alone) - (click pair - alone)", cc.table,
                                    MAIN_COLOR)]),
                job("d", dd, "d", "shared_baseline_diagnostic"),
            ]
            one_session = (" Evening = one session: whole-batch p cannot reach 0.05."
                           if window == "evening" else "")
            findings += [
                pairs.finding(SECTION, "alone_vs_pair_pooled", primary=False, window=window,
                              metric=metric,
                              text=_texts(window, metric, "each odor right after IAA "
                                          "minus the same odor alone, presentations pooled",
                                          a)
                              + pairs.describe_rows(a.table, metric) + ". "
                              + pairs.holm_verdict(a.table)
                              + " Not controlled for time in the rig (see the click "
                                "reference)." + one_session,
                              numbers=pairs.table_numbers(a.table, window=window,
                                                          metric=metric)),
                pairs.finding(SECTION, "presentation_1_vs_2", primary=False, window=window,
                              metric=metric,
                              text=_texts(window, metric, "each odor right after IAA, 2nd "
                                          "minus 1st time", b)
                              + pairs.describe_rows(b.table, metric) + ". "
                              + pairs.holm_verdict(b.table) + one_session,
                              numbers=pairs.table_numbers(b.table, window=window,
                                                          metric=metric)),
                pairs.finding(SECTION, "vs_paraffin_reference_direct", primary=False,
                              window=window, metric=metric,
                              text=_texts(window, metric, "(i) direct form: each odor right "
                                          "after IAA minus the click right after IAA, "
                                          "within fly, rounds pooled", c)
                              + pairs.describe_rows(c.table, metric) + ". "
                              + pairs.holm_verdict(c.table) + one_session,
                              numbers=pairs.table_numbers(c.table, window=window,
                                                          metric=metric)),
                pairs.finding(SECTION, "vs_paraffin_reference_controlled", primary=False,
                              window=window, metric=metric,
                              text=_texts(window, metric, "(ii) controlled change: (odor "
                                          "pair - odor alone) - (click pair - click alone), "
                                          "within fly", cc)
                              + pairs.describe_rows(cc.table, metric) + ". "
                              + pairs.holm_verdict(cc.table) + one_session,
                              numbers=pairs.table_numbers(cc.table, window=window,
                                                          metric=metric)),
                _direction_finding(dd, window, metric),
            ]
        if fit_models:
            rows = _models(wdf, window, root / SUB["models"], ordinal_boot=ordinal_boot)
            findings.append(_model_finding(rows, window))
    pairs.render(jobs, EXPERIMENT, tables, log=LOG)
    LOG.info("exp2 done: %d figures, %d findings, %.1f s", len(jobs), len(findings),
             time.perf_counter() - t0)
    return findings


__all__ = ("run",)
