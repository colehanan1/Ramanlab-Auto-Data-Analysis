"""Exp 3: IAA second (IAA-PairSecond-Control-1-1).

Alone block: all 8 odors once (~5-35 min into the session). Pair block (~65-140 min):
odor X for 10 s, a 1 s gap, then IAA, scored (``odor`` = IAA, ``prime`` = ``partner`` =
X); each X twice. X = the 7 odors and the click (Paraffin Oil, master valve only, no odor),
the within-block, time-matched reference.

PRIMARY (``c_vs_paraffin_reference``): per fly, IAA after X minus IAA after the click,
Holm across the 7 odor primes. Every prime shares IAA's one alone trial, so the
controlled change equals this direct form. Odor 2 starts 11 s after odor 1's onset but is
scored against the baseline before odor 1, so odor 1's late response can leak into IAA's
score: a sensitivity version leaves out pair trials whose odor 1 responded, and odor-1
responded is a covariate in every model.

Every analysis runs per clock window (day / evening); a window without flies is skipped.
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

EXPERIMENT = "exp3"
SECTION = "exp3"
FOLDER = "04_exp3_iaa_second"
SUB = {"a": "a_alone_vs_pair_pooled", "b": "b_presentation_1_vs_2",
       "c": "c_vs_paraffin_reference", "d": "d_shared_baseline_diagnostic",
       "models": "models"}
LABEL = "Exp 3 (IAA second)"
XLABEL = "Prime: the odor that ran 1 s before IAA (IAA scored)"
METRIC_WORDS = {"rate": "PER rate (% of trials; effects in percentage points)",
                "score": "PER score (-1..5)"}
METRIC_TITLE = {"rate": "PER rate", "score": "PER score"}
CARRY_COLOR = pl.CATEGORICAL[1]
MAIN_COLOR = pl.CATEGORICAL[0]

COND_LABELS = {
    "a": {"alone": "IAA alone (pre-block)", "pair": "IAA right after the odor (1st + 2nd)"},
    "b": {"alone": "IAA alone (pre-block)", "1st": "IAA after the odor, 1st time",
          "2nd": "IAA after the odor, 2nd time"},
    "c": {"click": "IAA right after the click", "odor": "IAA right after the odor"},
}
TITLES = {
    "a": "IAA alone vs IAA right after each odor (presentations pooled)",
    "b": "IAA alone, then right after each odor: 1st vs 2nd time",
    "c": "IAA right after each odor vs IAA right after the click (within fly)",
    "cs": "IAA after each odor vs after the click, odor-1 responders left out",
    "d": "Direction of IAA's change after each odor (one shared IAA-alone baseline)",
}
CAPTIONS = {
    "a": pairs.BARS_NOTE + " Each prime's alone bar is the same IAA-alone trial. "
         "Alone (~5-35 min) vs pair (~65-140 min) also holds time in the rig; see "
         "c_vs_paraffin_reference for the click-controlled contrast. " + pairs.HOLM_SENTENCE,
    "b": pairs.BARS_NOTE + " Brackets: 1st vs 2nd time after the same odor. "
         + pairs.HOLM_SENTENCE,
    "c": pairs.BARS_NOTE + " Lower panel: per fly, IAA after the odor minus IAA after the "
         "click (both in the pair block), mean over flies with its 95 % BCa CI; orange = "
         "pair trials whose odor 1 responded left out (carry-over check). "
         + pairs.HOLM_SENTENCE,
    "cs": pairs.BARS_NOTE + " Pair trials whose odor 1 (the prime, or the click) scored "
          ">= threshold are left out: odor 1's late response could leak into IAA's score, "
          "which is measured against the baseline before odor 1. " + pairs.HOLM_SENTENCE,
    "d": "Bars: pooled change of IAA (after the odor - alone, trials pooled on each side) "
         "± SEM over flies; dots: flies. Every odor shares IAA's one alone trial, so a "
         "fly with a low baseline moves every bar up together; the count is tested by "
         "flipping each fly's whole row of changes jointly (stats.direction_count_test), "
         "and whole batches for the batch p. Click (hatched) = reference, not counted.",
}
READMES = {
    "": ["# Exp 3 · IAA second (IAA-PairSecond-Control-1-1)",
         "Alone block (8 odors, ~5-35 min), then pairs: odor X 10 s, 1 s gap, IAA scored "
         "(~65-140 min), each X twice; the click (Paraffin Oil, master valve, no odor) is "
         "the time-matched reference. PRIMARY: c_vs_paraffin_reference.",
         "Every analysis runs per clock window (`_day` / `_evening` in each file name); "
         "light probes and the four 0.025 s alone clicks are left out. Tables: "
         "`../tables/<figure base name>.csv`."],
    "a": ["# a · IAA alone vs IAA right after each odor (presentations pooled)",
          "Paired sign-flip on per-fly means, 95 % BCa CI, Holm across the 8 primes. "
          "Not controlled for time in the rig or re-testing (see c).",
          "Files: `exp3_alone_vs_pair_pooled_<rate|score>_<window>`."],
    "b": ["# b · IAA alone, then 1st and 2nd time after each odor",
          "Bars for alone / 1st / 2nd; paired sign-flip of 1st vs 2nd, Holm across the 8 "
          "primes per panel.",
          "Files: `exp3_presentation_1_vs_2_<rate|score>_<window>`."],
    "c": ["# c · PRIMARY: IAA after each odor vs IAA after the click, within fly",
          "Per fly: mean IAA after X - mean IAA after the click (pair block, rounds pooled); "
          "sign-flip, BCa CI, Holm across the 7 odor primes; p_batch flips whole batches. "
          "Equals the controlled change (one shared IAA-alone baseline).",
          "`*_carryover_excluded_*`: pair trials whose odor 1 responded left out "
          "(odor 1's late response can leak into IAA's score)."],
    "d": ["# d · Shared-baseline diagnostic",
          "Every prime's change uses the same IAA-alone trial. How many odors go up vs "
          "down, against a null that flips each fly's whole row of changes jointly "
          "(`stats.direction_count_test`; whole batches for p_batch).",
          "Files: `exp3_shared_baseline_diagnostic_<rate|score>_<window>`."],
    "models": ["# models · Exp 3",
               "`alone_vs_pair`: IAA rows, responded ~ C(level: alone or prime) + 2nd "
               "presentation + odor1_responded + start_time_c + position_in_block. "
               "`vs_paraffin_reference`: pair rows, responded ~ C(prime, ref = click) + "
               "odor1_responded + 2nd presentation + start_time_c + position_in_block.",
               "GEE logistic (fly clusters, robust SE) is the model of record; Bayesian "
               "mixed logistic (VB), MixedLM and ordinal (fly-cluster bootstrap) on score "
               "are sensitivity checks. Status and role per fit in "
               "`exp3_model_status_<window>.csv` (check it before reading a coefficient)."],
}


def _texts(window: str, metric: str, what: str, res: pairs.Result) -> str:
    return (f"{LABEL}, {window} window, {METRIC_WORDS[metric]}: {what} "
            f"({res.n_flies} flies, {res.n_trials} trials; {res.test}; {res.correction}). ")


def _models(wdf: pd.DataFrame, window: str, folder: Path, *, ordinal_boot: int) -> list[dict]:
    T = pairs.Term
    iaa = wdf[wdf["odor"] == pairs.IAA]
    iaa = iaa.assign(level=np.where(iaa["condition"] == pairs.ALONE, "alone", iaa["prime"]))
    pair = wdf[wdf["condition"] == pairs.PAIR]
    rows = []
    for name, rows_df, terms, sep in (
        ("alone_vs_pair", iaa,
         [T("C(level, Treatment('alone'))", ("level",), True),
          T("second_presentation", ("second_presentation",)),
          T("odor1_responded", ("odor1_responded",)),
          T("start_time_c", ("start_time_c",)),
          T("position_in_block", ("position_in_block",))],
         pairs.Separation(("level",), "level", "alone")),
        ("vs_paraffin_reference", pair,
         [T(f"C(prime, Treatment('{pairs.REFERENCE}'))", ("prime",), True),
          T("odor1_responded", ("odor1_responded",)),
          T("second_presentation", ("second_presentation",)),
          T("start_time_c", ("start_time_c",)),
          T("position_in_block", ("position_in_block",))],
         pairs.Separation(("prime",), "prime", pairs.REFERENCE)),
    ):
        frame = pairs.model_frame(rows_df)
        r = np.corrcoef(frame["position_in_block"], frame["minutes_since_block_start"])[0, 1] \
            if len(frame) > 2 else np.nan
        note = (f"minutes_since_block_start left out: r = {r:.5f} with position_in_block "
                "(one trial every 5 min); second_presentation codes C(presentation).")
        rows += pairs.fit_model_set(frame, terms, name=name, experiment=EXPERIMENT,
                                    window=window, folder=folder, ordinal_boot=ordinal_boot,
                                    log=LOG, note=note, separation=sep)
    pairs.write_model_status(rows, folder, EXPERIMENT, window)
    return rows


def _model_finding(rows: list[dict], window: str) -> dict:
    status = {f"{r['model']}.{r['fit']}": r["status"] for r in rows}
    ref = f"C(prime, Treatment('{pairs.REFERENCE}'))"
    gee_prime = pairs.key_terms(rows, name="vs_paraffin_reference", fit="gee", pattern=ref)
    gee_o1 = pairs.key_terms(rows, name="vs_paraffin_reference", fit="gee",
                             pattern="odor1_responded")
    gee_ok = next((r["status"] for r in rows if r["model"] == "vs_paraffin_reference"
                   and r["fit"] == "gee"), "failed")
    text = (f"{LABEL}, {window} window, models (status per fit: "
            + ", ".join(f"{k} {v}" for k, v in status.items()) + "). ")
    flips = pairs.sign_disagreements(rows, name="vs_paraffin_reference", pattern=ref) + \
        pairs.sign_disagreements(rows, name="vs_paraffin_reference",
                                 pattern="odor1_responded")
    if gee_ok != "failed" and len(gee_prime):
        text += ("Model of record: the primary contrast as a GEE logistic model on pair "
                 "rows (reference = IAA "
                 "after the click; covariates odor-1 responded, 2nd presentation, start "
                 "time, position; odds ratios, Wald 95 % CI, uncorrected p): "
                 + pairs.describe_odds(gee_prime) + ". ")
        if len(gee_o1):
            text += ("Odor 1 responded (carry-over covariate): "
                     + pairs.describe_odds(gee_o1).replace("odor1_responded ", "") + ".")
    else:
        reason = next((r["note"] for r in rows if r["model"] == "vs_paraffin_reference"
                       and r["fit"] == "gee"), "")
        text += ("The primary-contrast GEE (model of record) did not fit"
                 + (f" ({reason})" if reason else "") + "; the sensitivity-check fits are "
                 "not read in its place.")
    text += pairs.gee_left_out(rows, "vs_paraffin_reference")
    text += pairs.describe_sign_flips(flips) or (
        " The mixed logistic (sensitivity check) agrees in sign with the GEE on these terms.")
    nums = {"window": window, "status": status,
            "glmm_sign_disagreements": [pairs.term_level(f["term"]) for f in flips]}
    if len(gee_prime):
        nums["gee_log_odds_vs_click"] = {pairs.term_level(t): e for t, e in
                                         zip(gee_prime["term"], gee_prime["estimate"])}
        nums["gee_p_vs_click"] = {pairs.term_level(t): p for t, p in
                                  zip(gee_prime["term"], gee_prime["p"])}
    if len(gee_o1):
        nums["gee_odor1_responded_log_odds"] = float(gee_o1["estimate"].iloc[0])
        nums["gee_odor1_responded_p"] = float(gee_o1["p"].iloc[0])
    return pairs.finding(SECTION, "models", primary=False, window=window, metric="rate",
                         text=text, numbers=nums)


def run(trials: pd.DataFrame, out, *, tables, fit_models: bool = True,
        ordinal_boot: int = 200, n_boot: int | None = None,
        n_perm: int | None = None) -> list[dict]:
    """Exp 3 analyses a-d and the models, per clock window, into ``out/04_exp3_iaa_second``
    (tables into ``tables``). Returns the findings (dicts: section, analysis, primary,
    text, numbers, window, metric)."""
    t0 = time.perf_counter()
    out, tables = Path(out), Path(tables)
    root = out / FOLDER
    kw = {"n_boot": n_boot, "n_perm": n_perm}
    d = pairs.prepare(trials, EXPERIMENT, log=LOG)
    LOG.info("exp3: %d rows, %d flies", len(d), d["fly_id"].nunique())
    for key, lines in READMES.items():
        pairs.write_readme(root / SUB[key] if key else root, lines)
    windows = pairs.split_windows(d, log=LOG)
    if not windows:
        LOG.info("exp3: no flies in any window, nothing to analyse")
        return [pairs.finding(SECTION, "skipped", primary=False, window="", metric="",
                              text="Exp 3: no flies in any clock window.", numbers={})]
    jobs: list[pairs.Job] = []
    findings: list[dict] = []
    for window, wdf in windows:
        LOG.info("exp3 %s window: %d flies, %d batches", window, wdf["fly_id"].nunique(),
                 wdf["batch"].nunique())
        long = pairs.scored_long(wdf, EXPERIMENT)
        kept, n_drop = pairs.drop_carryover(wdf, log=LOG)
        long_kept = pairs.scored_long(kept, EXPERIMENT)
        LOG.debug("exp3 %s: long %s, carry-over kept %s", window, long.shape, long_kept.shape)
        for metric in pairs.METRICS:
            a = pairs.alone_vs_pair(long, metric, family="primes", **kw)
            b = pairs.presentation_1_vs_2(long, metric, family="primes", **kw)
            c = pairs.reference_contrast(long, metric, family="odor primes", **kw)
            cs = pairs.reference_contrast(long_kept, metric, family="odor primes",
                                          analysis="vs_paraffin_reference_carryover_excluded",
                                          **kw)
            dd = pairs.shared_baseline(long, metric, family="odor primes", **kw)

            def job(key, res, sub, analysis, **extra):
                return pairs.Job(folder=root / SUB[sub],
                                 stem=pairs.stem(EXPERIMENT, analysis, metric, window),
                                 kind="diagnostic" if key == "d" else "bars", result=res,
                                 title=f"{LABEL} · {TITLES[key]} · {METRIC_TITLE[metric]}",
                                 window=window, xlabel=XLABEL, caption=CAPTIONS[key],
                                 cond_labels=COND_LABELS.get(key, COND_LABELS["c"]),
                                 **extra)
            jobs += [
                job("a", a, "a", "alone_vs_pair_pooled"),
                job("b", b, "b", "presentation_1_vs_2"),
                job("c", c, "c", "vs_paraffin_reference",
                    effect_series=[("all pair trials", c.table, MAIN_COLOR),
                                   ("odor-1 responders left out", cs.table, CARRY_COLOR)]),
                job("cs", cs, "c", "vs_paraffin_reference_carryover_excluded",
                    effect_series=[("odor-1 responders left out", cs.table, CARRY_COLOR)]),
                job("d", dd, "d", "shared_baseline_diagnostic"),
            ]
            findings += [
                pairs.finding(SECTION, "alone_vs_pair_pooled", primary=False, window=window,
                              metric=metric,
                              text=_texts(window, metric, "IAA right after each prime minus "
                                          "IAA alone, presentations pooled", a)
                              + pairs.describe_rows(a.table, metric) + ". "
                              + pairs.holm_verdict(a.table)
                              + " Not controlled for time in the rig (see the click "
                                "reference).",
                              numbers=pairs.table_numbers(a.table, window=window,
                                                          metric=metric)),
                pairs.finding(SECTION, "presentation_1_vs_2", primary=False, window=window,
                              metric=metric,
                              text=_texts(window, metric, "IAA after each prime, 2nd minus "
                                          "1st time", b)
                              + pairs.describe_rows(b.table, metric) + ". "
                              + pairs.holm_verdict(b.table),
                              numbers=pairs.table_numbers(b.table, window=window,
                                                          metric=metric)),
                pairs.finding(SECTION, "vs_paraffin_reference", primary=True, window=window,
                              metric=metric,
                              text="PRIMARY. " + _texts(window, metric, "IAA right after "
                                                        "each odor minus IAA right after "
                                                        "the click, within fly, rounds "
                                                        "pooled", c)
                              + pairs.describe_rows(c.table, metric) + ". "
                              + pairs.holm_verdict(c.table)
                              + f" Carry-over check ({n_drop} pair trials whose odor 1 "
                                "responded left out): " + pairs.holm_verdict(cs.table),
                              numbers=pairs.table_numbers(c.table, window=window,
                                                          metric=metric)),
                pairs.finding(SECTION, "vs_paraffin_reference_carryover_excluded",
                              primary=False, window=window, metric=metric,
                              text=_texts(window, metric, f"carry-over sensitivity of the "
                                          f"primary contrast, {n_drop} pair trials whose "
                                          "odor 1 responded left out", cs)
                              + pairs.describe_rows(cs.table, metric) + ". "
                              + pairs.holm_verdict(cs.table),
                              numbers={**pairs.table_numbers(cs.table, window=window,
                                                             metric=metric),
                                       "n_trials_left_out": n_drop}),
                _direction_finding(dd, window, metric),
            ]
        if fit_models:
            rows = _models(wdf, window, root / SUB["models"], ordinal_boot=ordinal_boot)
            findings.append(_model_finding(rows, window))
    pairs.render(jobs, EXPERIMENT, tables, log=LOG)
    LOG.info("exp3 done: %d figures, %d findings, %.1f s", len(jobs), len(findings),
             time.perf_counter() - t0)
    return findings


def _direction_finding(dd: pairs.Result, window: str, metric: str) -> dict:
    d = dd.extra["direction"]
    n = len(dd.extra["odors"])
    row = dd.table.set_index("group").loc["direction count"]
    text = (f"{LABEL}, {window} window, {METRIC_WORDS[metric]}: shared-baseline "
            f"diagnostic ({dd.n_flies} flies). IAA's change after each odor uses one shared "
            f"IAA-alone trial: {d['n_up']} of {n} odors up, {d['n_down']} down, "
            f"{d['n_zero']} unchanged; joint fly-flip p = {pairs.fmt_p(row['p'])} "
            f"(whole-batch p = {pairs.fmt_p(row['p_batch'])}; the naive binomial count "
            f"would give p = {pairs.fmt_p(d['p_naive_binomial'])}); mean cross-odor "
            f"correlation of per-fly changes {d['mean_cross_odor_corr']:.2f}.")
    return pairs.finding(SECTION, "shared_baseline_diagnostic", primary=False, window=window,
                         metric=metric, text=text,
                         numbers={"window": window, "metric": metric,
                                  "n_flies": dd.n_flies, "n_up": d["n_up"],
                                  "n_down": d["n_down"], "n_zero": d["n_zero"],
                                  "p": row["p"], "p_batch": row["p_batch"],
                                  "p_naive_binomial": d["p_naive_binomial"],
                                  "mean_cross_odor_corr": d["mean_cross_odor_corr"]})


__all__ = ("run",)
