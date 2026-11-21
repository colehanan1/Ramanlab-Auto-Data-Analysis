"""QC of the tidy trial table: one row of flags per fly plus a markdown report.

Rules (per fly):

- missing trials: trials its batch folder recorded (raw stamped files; without them,
  the union of its folder-mates' rows) that have no scored row for this fly;
- protocol gaps: slots of the cohort protocol (block x odor / partner / light probe,
  with counts) its folder never ran. Known gaps (paraffin not yet in an exp1 panel; no
  light probe in pair sessions before 2026-11-17) are listed but are not "incomplete";
- duplicate trials, out-of-range scores (outside -1..5);
- never-responder: no presentation >= threshold, odors only (light probe and Paraffin
  Oil/click excluded; a pair trial's odor-1 score counts with its odor). Whether the fly
  answered the light probe is reported beside it;
- always-responder: every presentation >= threshold, paraffin/click included (light probe
  excluded; a pair trial's odor-1 score counts);
- short odor windows: an odor trial whose window is < half its experiment x block median.

``apply_exclusions`` drops flagged flies only for the rules switched on.
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from . import config as C
from . import io

log = logging.getLogger(__name__)

SHORT_WINDOW_FRACTION = 0.5
LIGHT_SLOT = "light probe"
#: apply_exclusions key -> per-fly flag column (and the config switch that defaults it).
EXCLUSION_RULES = {"never_responders": ("never_responder", "EXCLUDE_NEVER_RESPONDERS"),
                   "always_responders": ("always_responder", "EXCLUDE_ALWAYS_RESPONDERS"),
                   "incomplete": ("incomplete", "EXCLUDE_INCOMPLETE")}
_PHASE_ORDER = {"pretest": 0, "training": 1, "testing": 2}


# ── helpers ───────────────────────────────────────────────────────────────────
def _label_order(label: str):
    p = io.parse_trial_label(label)
    return (_PHASE_ORDER.get(p.phase, 9), p.trial_index, label) if p else (9, 0, label)


def _slot(experiment: str, label: str):
    """(block, odor | partner | 'light probe') of a trial label, or None."""
    p = io.parse_trial_label(label)
    if p is None:
        return None
    block = io.BLOCKS.get(experiment, {}).get(p.phase)
    if block is None:
        return None
    if p.is_light_probe:
        return block, LIGHT_SLOT
    if len(p.odors) == 2:
        return block, (p.odors[1] if experiment == "exp2" else p.odors[0])
    return block, p.odors[0]


def _known_gap(experiment: str, slot: tuple) -> bool:
    block, key = slot
    if experiment == "exp1":
        return block in ("pre", "post") and key == C.CLICK_ODOR    # paraffin joined later
    return key == LIGHT_SLOT                                        # pair probe since 11-17


def _events(trials: pd.DataFrame) -> pd.DataFrame:
    """Every scored presentation but the light probe: (fly_id, score, odor); a pair row
    adds its odor-1 score under its prime."""
    odor = trials[~trials["is_light_probe"].astype(bool)]
    a = odor[["fly_id", "score", "odor"]]
    pr = odor[odor["odor1_score"].notna()] if "odor1_score" in odor else odor.iloc[:0]
    b = pr[["fly_id", "odor1_score", "prime"]].rename(columns={"odor1_score": "score",
                                                               "prime": "odor"})
    return pd.concat([a, b], ignore_index=True)


def _cell(v) -> str:
    if isinstance(v, str):
        return v.replace("|", "\\|")
    if v is None or (not isinstance(v, (list, tuple)) and pd.isna(v)):
        return ""
    if isinstance(v, (float, np.floating)):
        return str(int(v)) if float(v).is_integer() else f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)


def _md_table(df: pd.DataFrame, index: bool = True) -> str:
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in d.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    return "\n".join(lines)


def flies_per_cohort_window(trials: pd.DataFrame) -> pd.DataFrame:
    """Unique flies per cohort x window (every configured window plus 'other') + total."""
    flies = trials.drop_duplicates("fly_id")
    tab = pd.crosstab(flies["cohort"], flies["window"])
    cols = [w.name for w in C.WINDOWS] + [C.OTHER_WINDOW]
    cols += [c for c in tab.columns if c not in cols]
    tab = tab.reindex(columns=cols, fill_value=0)
    order = [c for c in C.COHORTS if c in tab.index] + sorted(set(tab.index) - set(C.COHORTS))
    tab = tab.reindex(order)
    tab["total"] = tab.sum(axis=1)
    tab.columns.name = None
    return tab


# ── per-fly flags ─────────────────────────────────────────────────────────────
def _completeness(trials: pd.DataFrame, raw_trials: pd.DataFrame | None) -> pd.DataFrame:
    flies = trials.drop_duplicates("fly_id")[["fly_id", "cohort", "experiment", "batch"]]
    ds_col = "dataset" if "dataset" in trials else "cohort"
    if raw_trials is not None and len(raw_trials):
        r = raw_trials.assign(trial_label=raw_trials["trial_label"].str.lower())
        folder_labels = r.groupby(["dataset", "folder"])["trial_label"].agg(set).to_dict()
    else:
        t = trials.assign(_l=trials["trial_label"].str.lower())
        folder_labels = t.groupby([ds_col, "batch"])["_l"].agg(set).to_dict()
    fly_labels = trials.assign(_l=trials["trial_label"].str.lower()).groupby("fly_id")["_l"] \
        .agg(set).to_dict()

    # each folder's slot counts, the cohort protocol = the most any folder ran
    folder_slots, protocol = {}, {}
    for f in flies.drop_duplicates(["cohort", "batch"]).itertuples():
        labels = folder_labels.get((f.cohort, f.batch), set())
        cnt = Counter(s for s in (_slot(f.experiment, lab) for lab in labels) if s)
        folder_slots[(f.cohort, f.batch)] = cnt
        proto = protocol.setdefault(f.cohort, Counter())
        for s, n in cnt.items():
            proto[s] = max(proto[s], n)

    rows = []
    for f in flies.itertuples():
        expected = folder_labels.get((f.cohort, f.batch), set())
        missing = sorted(expected - fly_labels.get(f.fly_id, set()), key=_label_order)
        have = folder_slots.get((f.cohort, f.batch), Counter())
        known, unexpected = [], []
        for s, n in sorted(protocol.get(f.cohort, {}).items()):
            k = have.get(s, 0)
            if k < n:
                text = f"{s[0]}:{s[1]}" + ("" if k == 0 else f" ({k} of {n})")
                (known if _known_gap(f.experiment, s) else unexpected).append(text)
        rows.append(dict(fly_id=f.fly_id, n_missing_trials=len(missing),
                         missing_trials="; ".join(missing),
                         protocol_gaps_known="; ".join(sorted(known)),
                         protocol_gaps_unexpected="; ".join(sorted(unexpected))))
    return pd.DataFrame(rows)


def _short_windows(trials: pd.DataFrame) -> pd.DataFrame:
    odor = trials[~trials["is_light_probe"].astype(bool)]
    med = odor.groupby(["experiment", "block"])["odor_window_s"].transform("median")
    short = odor[odor["odor_window_s"] < SHORT_WINDOW_FRACTION * med]
    return short.assign(median_window_s=med[short.index])


def run_qc(trials: pd.DataFrame, raw_trials: pd.DataFrame | None = None, *,
           threshold: int | None = None, facts: pd.DataFrame | None = None,
           b2b: pd.DataFrame | None = None) -> tuple[pd.DataFrame, str]:
    """-> (one row of QC flags per fly, markdown report)."""
    thr = C.RESPONSE_THRESHOLD if threshold is None else threshold
    ident = [c for c in ["fly_id", "experiment", "cohort", "window", "batch", "session", "rig",
                         "host", "start_time"] if c in trials]
    per_fly = trials.drop_duplicates("fly_id")[ident].set_index("fly_id")
    light = trials["is_light_probe"].astype(bool)

    g = trials.groupby("fly_id")
    per_fly["n_trials"] = g.size()
    per_fly["n_odor_trials"] = trials[~light].groupby("fly_id").size().reindex(
        per_fly.index, fill_value=0)

    ev = _events(trials)
    ev = ev.assign(hit=ev["score"] >= thr, is_odor=ev["odor"].fillna("") != C.CLICK_ODOR)
    eg = ev.groupby("fly_id")
    per_fly["n_presentations"] = eg.size().reindex(per_fly.index, fill_value=0)
    per_fly["n_odor_responses"] = ev[ev.hit & ev.is_odor].groupby("fly_id").size().reindex(
        per_fly.index, fill_value=0)
    per_fly["n_paraffin_responses"] = ev[ev.hit & ~ev.is_odor].groupby("fly_id").size() \
        .reindex(per_fly.index, fill_value=0)
    per_fly["never_responder"] = per_fly["n_odor_responses"] == 0
    all_hit = eg["hit"].all().reindex(per_fly.index, fill_value=False)
    per_fly["always_responder"] = all_hit & (per_fly["n_presentations"] > 0)

    lp = trials[light].groupby("fly_id")
    per_fly["light_probe_score"] = lp["score"].max().reindex(per_fly.index).astype(float)
    per_fly["light_probe_responded"] = pd.array(
        [pd.NA if np.isnan(s) else bool(s >= thr) for s in per_fly["light_probe_score"]],
        dtype="boolean")
    if "odor_window_s" in trials:
        per_fly["light_probe_window_s"] = lp["odor_window_s"].min().reindex(per_fly.index)

    comp = _completeness(trials, raw_trials).set_index("fly_id")
    per_fly = per_fly.join(comp)
    per_fly["incomplete"] = (per_fly["n_missing_trials"] > 0) | \
        (per_fly["protocol_gaps_unexpected"] != "")

    key = ["fly_id", "trial_label"] + (["is_odor2"] if "is_odor2" in trials else [])
    dup = trials.duplicated(key, keep="first")
    per_fly["n_duplicate_trials"] = trials[dup].groupby("fly_id").size().reindex(
        per_fly.index, fill_value=0)
    sc = pd.to_numeric(trials["score"], errors="coerce")
    oor = sc.isna() | (sc != sc.round()) | ~sc.between(C.SCORE_MIN, C.SCORE_MAX)
    per_fly["n_out_of_range_scores"] = trials[oor].groupby("fly_id").size().reindex(
        per_fly.index, fill_value=0)

    short = _short_windows(trials.drop_duplicates(key))
    per_fly["n_short_odor_windows"] = short.groupby("fly_id").size().reindex(
        per_fly.index, fill_value=0)
    per_fly["short_window_trials"] = short.groupby("fly_id")["trial_label"].agg(
        lambda s: "; ".join(sorted(s, key=_label_order))).reindex(per_fly.index).fillna("")

    per_fly = per_fly.reset_index()
    md = _report(trials, per_fly, short, thr, raw_trials is not None, facts, b2b)
    for rule in ("never_responder", "always_responder", "incomplete"):
        log.info("QC %s: %d of %d flies", rule, int(per_fly[rule].sum()), len(per_fly))
    return per_fly, md


# ── report ────────────────────────────────────────────────────────────────────
def _by_cohort(per_fly: pd.DataFrame, flag: str) -> pd.DataFrame:
    t = per_fly.groupby("cohort").agg(flies=("fly_id", "size"),
                                      flagged=(flag, lambda s: int(s.astype(bool).sum())))
    return t.reindex([c for c in C.COHORTS if c in t.index]
                     + sorted(set(t.index) - set(C.COHORTS)))


def _report(trials, per_fly, short, thr, have_raw, facts, b2b) -> str:
    L = ["# Data QC", "",
         f"{len(trials)} scored trials, {trials['fly_id'].nunique()} flies, "
         f"{trials['cohort'].nunique()} cohorts. Response = score >= {thr}. "
         "Nothing is excluded here; `per_analysis.qc.apply_exclusions` drops flagged flies "
         "only for the switches on in `per_analysis.config` "
         f"(never {C.EXCLUDE_NEVER_RESPONDERS}, always {C.EXCLUDE_ALWAYS_RESPONDERS}, "
         f"incomplete {C.EXCLUDE_INCOMPLETE}).", ""]

    L += ["## Flies per cohort x window", "",
          "Window of the batch folder's first pre-test stamp (Pi clock, end exclusive).", "",
          _md_table(flies_per_cohort_window(trials)), ""]
    shape = io.shape_table(trials) if {"experiment", "block", "window"} <= set(trials) else None
    if shape is not None:
        L += ["## Rows per experiment x block x window", "", _md_table(shape), ""]

    # completeness
    src = "its folder's raw stamped files" if have_raw else "its folder-mates' rows"
    miss = per_fly[per_fly["n_missing_trials"] > 0]
    L += ["## Missing trials", "",
          f"Trials {src} recorded with no scored row for this fly: {len(miss)} of "
          f"{len(per_fly)} flies.", ""]
    if len(miss):
        L += [_md_table(miss[["fly_id", "n_missing_trials", "missing_trials"]].head(40),
                        index=False), ""]
        if len(miss) > 40:
            L += [f"... {len(miss) - 40} more in qc_report.csv", ""]
    L += ["## Protocol gaps vs the cohort protocol", "",
          "Slots (block:odor/partner/light probe) the fly's folder never ran, against the most "
          "any folder of the cohort ran. Known gaps: paraffin not yet in an exp1 panel, no "
          "pair light probe before 2026-11-17.", ""]
    rows = []
    for (cohort, kind), g in per_fly.melt(
            id_vars=["cohort", "fly_id"],
            value_vars=["protocol_gaps_known", "protocol_gaps_unexpected"],
            var_name="kind", value_name="gaps").groupby(["cohort", "kind"]):
        for gaps, n in g[g["gaps"] != ""]["gaps"].value_counts().items():
            rows.append(dict(cohort=cohort, kind=kind.replace("protocol_gaps_", ""), gaps=gaps,
                             flies=int(n)))
    L += [_md_table(pd.DataFrame(rows), index=False) if rows else "No gaps.", ""]
    inc = per_fly[per_fly["incomplete"]]
    L += [f"**Incomplete** (missing trials or an unexpected gap): {len(inc)} flies.", ""]

    # integrity
    nd, no = int(per_fly["n_duplicate_trials"].sum()), int(per_fly["n_out_of_range_scores"].sum())
    L += ["## Duplicate trials", "", f"{nd} duplicate rows (fly x trial label x odor-2 flag).", "",
          "## Out-of-range scores", "",
          f"{no} scores outside {C.SCORE_MIN}..{C.SCORE_MAX} (-1 = retraction is valid).", ""]

    # responders
    nev = _by_cohort(per_fly, "never_responder")
    nv = per_fly[per_fly["never_responder"]]
    lpr = nv["light_probe_responded"].astype("boolean")
    tally = pd.DataFrame({"cohort": nv["cohort"],
                          "of_which_light_probe_yes": lpr.fillna(False).astype(int),
                          "of_which_light_probe_no": (~lpr).fillna(False).astype(int),
                          "of_which_no_probe": lpr.isna().astype(int)})
    nev = nev.join(tally.groupby("cohort").sum()).fillna(0).astype(int)
    L += ["## Never-responders", "",
          f"No presentation >= {thr}, odors only (light probe and Paraffin Oil/click left out; "
          "a pair trial's odor-1 score counts). Light-probe response reported beside it.", "",
          _md_table(nev), ""]
    alw = _by_cohort(per_fly, "always_responder")
    L += ["## Always-responders", "",
          f"Every presentation >= {thr}, paraffin/click included (light probe left out).", "",
          _md_table(alw), ""]

    # light probe
    lt = trials[trials["is_light_probe"].astype(bool)]
    if len(lt):
        lpt = lt.groupby("cohort").agg(probes=("score", "size"),
                                       responded=("responded", "sum"),
                                       median_window_s=("odor_window_s", "median"))
        lpt["pct_responded"] = (100 * lpt["responded"] / lpt["probes"]).round(1)
        L += ["## Light probe (opto positive control)", "",
              "Pooled over flies (one probe per fly). A probe whose table window is < 1 s was "
              "scored on that window, so its score is not a light response.", "",
              _md_table(lpt), ""]

    # windows
    L += ["## Short odor windows", "",
          f"Odor trials with a window < {SHORT_WINDOW_FRACTION:g} x their experiment x block "
          f"median: {len(short)}.", ""]
    if len(short):
        L += [_md_table(short[["fly_id", "trial_label", "odor_window_s",
                               "median_window_s"]].head(40), index=False), ""]
    if len(lt):
        bad_lp = lt[lt["odor_window_s"] < 1.0]
        if len(bad_lp):
            L += [f"Light probes with a table window < 1 s: {len(bad_lp)} "
                  f"({', '.join(sorted(set(bad_lp['cohort'])))}).", ""]

    if facts is not None and len(facts):
        L += ["## Protocol facts", "", _md_table(facts, index=False), ""]
    if b2b is not None and len(b2b):
        L += ["## Counts check against the b2b step", "",
              "Alone ('pre') and pair odor-2 ('post') trials per pair dataset and window vs "
              "`Results/b2b*/b2b_presentations.csv`.", "", _md_table(b2b, index=False), ""]
    return "\n".join(L)


# ── exclusions and outputs ────────────────────────────────────────────────────
def apply_exclusions(trials: pd.DataFrame, per_fly: pd.DataFrame,
                     flags: Mapping[str, bool] | None = None) -> pd.DataFrame:
    """Drop the flies flagged by each rule switched on. ``flags`` keys:
    never_responders / always_responders / incomplete; None = the config switches."""
    if flags is None:
        flags = {k: bool(getattr(C, sw)) for k, (_col, sw) in EXCLUSION_RULES.items()}
    unknown = set(flags) - set(EXCLUSION_RULES)
    if unknown:
        raise KeyError(f"unknown exclusion rules {sorted(unknown)}; "
                       f"known: {sorted(EXCLUSION_RULES)}")
    drop: set = set()
    for k, on in flags.items():
        if on:
            col = EXCLUSION_RULES[k][0]
            ids = set(per_fly.loc[per_fly[col].astype(bool), "fly_id"])
            log.info("exclusion %s: %d flies", k, len(ids))
            drop |= ids
    kept = trials[~trials["fly_id"].isin(drop)]
    log.debug("apply_exclusions: %s -> %s rows (%d flies dropped)", trials.shape, kept.shape,
              len(drop))
    return kept


README = """# 01_data_qc
`tidy_trials.parquet`: one row per scored trial (columns in `column_mapping.yaml`), from `per_analysis.io.load_trials()`; rebuilt when its inputs change. `raw_trial_files.parquet`: the raw-file stamps behind it.
`qc_report.md` / `qc_report.csv`: per-fly QC flags (missing trials, protocol gaps, duplicates, scores, never/always responders, short odor windows) from `per_analysis.qc.run_qc()`; nothing is excluded unless a `config.EXCLUDE_*` switch is on.
`protocol_facts.csv` / `b2b_counts_check.csv`: known protocol facts observed vs expected, and the pair-trial counts vs the b2b step's `b2b_presentations.csv`.
"""


def write_qc_outputs(per_fly: pd.DataFrame, md: str, out_dir: Path | None = None) -> list[Path]:
    out = Path(out_dir) if out_dir else C.QC_DIR
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "qc_report.csv", out / "qc_report.md", out / "README.md"]
    per_fly.to_csv(paths[0], index=False)
    paths[1].write_text(md)
    paths[2].write_text(README)
    log.info("wrote %s", ", ".join(str(p) for p in paths))
    return paths


def main(argv=None) -> int:
    """Build/refresh the tidy table and write every 01_data_qc output. Exit 1 when a
    protocol fact changed or the b2b counts disagree."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = C.DEFAULT_CONFIG
    io.write_qc_inputs(config=cfg)
    trials = io.load_trials(cfg)
    raw = io.load_raw_trials(cfg)
    facts = io.load_protocol_facts(cfg)
    b2b, ok = io.b2b_counts_check(trials)
    b2b.to_csv(cfg.qc_dir / "b2b_counts_check.csv", index=False)
    per_fly, md = run_qc(trials, raw, threshold=cfg.threshold, facts=facts, b2b=b2b)
    write_qc_outputs(per_fly, md, cfg.qc_dir)
    facts_ok = bool(facts["ok"].all())
    if not ok:
        log.error("b2b counts check FAILED (see b2b_counts_check.csv)")
    if not facts_ok:
        log.error("a protocol fact changed (see protocol_facts.csv)")
    return 0 if ok and facts_ok else 1


if __name__ == "__main__":
    sys.exit(main())
