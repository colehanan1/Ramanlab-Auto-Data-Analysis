"""Reporting layer of the PER analysis: the results tree and its READMEs, SUMMARY.md and
the figure index.

- :func:`create_tree` / :func:`write_readmes`: every folder of the spec tree exists and
  carries a 2-4 line ``README.md``. A README a module wrote is never overwritten; the two
  PRIMARY folders get one "PRIMARY" line appended when their README lacks the word.
- :func:`write_summary`: ``SUMMARY.md`` -- data and assumptions, PRIMARY results, one
  paragraph per analysis by section, "Data vs described design", "Not analysed".
- :func:`write_index`: ``index.html`` -- a self-contained thumbnail index of every PNG,
  grouped by folder, plus links to the tables, SUMMARY.md and the QC report.

Findings are the analysis modules' contract: dicts with ``section``, ``analysis``,
``primary`` (bool), ``text`` and ``numbers``. ``run_all`` adds ``module`` and ``run``
("main" or "exclusions"). ``analysis == "design_contradiction"`` / ``"skipped"`` /
``"failed"`` are routed to their own SUMMARY sections.
"""

from __future__ import annotations

import datetime as dt
import html
import logging
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

import yaml

from . import config as C

log = logging.getLogger(__name__)

# ── the results tree ──────────────────────────────────────────────────────────
QC_DIR = "01_data_qc"
TABLES_DIR = "tables"
SENSITIVITY_DIR = "07_sensitivity_with_exclusions"
MODULE_DIRS = {"exp1": "02_exp1_exposure_cohorts", "exp2": "03_exp2_iaa_first",
               "exp3": "04_exp3_iaa_second", "cross": "05_cross_experiment",
               "power": "06_power", "daynight": "08_day_vs_evening"}
#: modules re-run with every QC exclusion on, into 07 (mirroring their folders)
SENSITIVITY_MODULES = ("exp1", "exp2", "exp3", "cross")
PRIMARY_FOLDERS = ("02_exp1_exposure_cohorts/b_did_vs_paraffin",
                   "04_exp3_iaa_second/c_vs_paraffin_reference")
PRIMARY_LINE = "PRIMARY analysis (marked by run_all)."

_EXP1_SUBS = {
    "a_pre_vs_post": "Exp 1 (a): post-test vs pre-test response to each odor within each "
                     "exposure cohort, paired by fly.",
    "b_did_vs_paraffin": "PRIMARY. Exp 1 (b): difference-in-differences: each exposed "
                         "cohort's post-minus-pre change against the Paraffin-Sensitivity "
                         "control's.",
    "c_global_drift": "Exp 1 (c): global drift: the post-minus-pre change pooled over every "
                      "odor, exposed or not.",
    "d_time_since_exposure": "Exp 1 (d): post-test response against minutes since the last "
                             "exposure pulse.",
    "e_odor_specific_suppression": "Exp 1 (e): odor-specific suppression: the exposed odor's "
                                   "change against the other odors' change, per fly.",
    "models": "Exp 1 model fits (statsmodels GEE / mixed / ordinal): coefficient tables "
              "and fit summaries.",
}


def _pair_subs(exp: str, role: str) -> dict:
    primary = exp == "Exp 3"
    return {
        "a_alone_vs_pair_pooled": f"{exp} (a): the scored odor alone (alone block) vs in a "
                                  f"pair ({role}), pooled over partners.",
        "b_presentation_1_vs_2": f"{exp} (b): first vs second presentation of each pair.",
        "c_vs_paraffin_reference": ("PRIMARY. " if primary else "")
        + f"{exp} (c): each partner's pair against the paraffin reference pair (a "
          "master-valve click, no odor).",
        "d_shared_baseline_diagnostic": f"{exp} (d): shared-baseline diagnostic: odor 2 is "
                                        "scored against the baseline before odor 1, so "
                                        "does its score track odor 1's (carry-over)?",
        "models": f"{exp} model fits (statsmodels GEE / mixed / ordinal): coefficient tables "
                  "and fit summaries.",
    }


_CROSS_SUBS = {
    "a_iaa_baseline_by_cohort": "Cross (a): the IAA baseline response (pre-test / alone "
                                "block) by cohort.",
    "b_iaa_iaa_across_experiments": "Cross (b): IAA-then-IAA pairs across experiments.",
    "c_eb_iaa_bidirectional": "Cross (c): Ethyl Butyrate and IAA in both orders (Exp 2 vs "
                              "Exp 3).",
    "d_paraffin_noise_floor": "Cross (d): the paraffin / click response rate as each "
                              "cohort's noise floor.",
    "models": "Cross-experiment model fits (statsmodels): coefficient tables and fit "
              "summaries.",
}

_STATS_LINE = ("Rates pool trials (sum / trials, Wilson CI over flies); paired tests average "
               "a fly's trials first; stars are Holm-adjusted fly-level p, with p_batch beside.")


def _window_text() -> str:
    return ", ".join(f"{w.name} {w.start:%H:%M}-{w.end:%H:%M}" for w in C.WINDOWS)


def _top_readmes() -> dict:
    return {
        QC_DIR: ("Data QC: the tidy trial table (`tidy_trials.parquet`) and per-fly QC flags "
                 "(`qc_report.md` / `qc_report.csv`).\n`column_mapping.yaml` maps every "
                 "column to its source. Nothing is excluded here."),
        MODULE_DIRS["exp1"]: ("Exp 1: pre-test -> odor exposure -> post-test cohorts (IAA, "
                              "Hexanol, 3-Octanol; Paraffin-Sensitivity is the control).\n"
                              "One subfolder per analysis; written by `per_analysis.exp1`."),
        MODULE_DIRS["exp2"]: ("Exp 2: IAA first in each pair (IAA -> partner); the scored "
                              "odor is the partner, odor 2.\nOne subfolder per analysis; "
                              "written by `per_analysis.exp2`."),
        MODULE_DIRS["exp3"]: ("Exp 3: IAA second in each pair (partner -> IAA); the scored "
                              "odor is IAA, odor 2.\nOne subfolder per analysis; written by "
                              "`per_analysis.exp3`."),
        MODULE_DIRS["cross"]: ("Cross-experiment comparisons of IAA and paraffin across the "
                               "exp1/exp2/exp3 cohorts.\nWritten by `per_analysis.cross`."),
        MODULE_DIRS["power"]: ("Power: simulated power against the number of flies, from the "
                               "observed per-fly effects, and the n for 80 % power.\n"
                               "Written by `per_analysis.power`."),
        SENSITIVITY_DIR: ("Sensitivity run: exp1, exp2, exp3 and cross re-run with every QC "
                          "exclusion on (never-responders, always-responders, incomplete "
                          "flies).\nSame folders as the main run, its own `tables/`; "
                          "`excluded_flies.csv` lists each dropped fly and its rules."),
        MODULE_DIRS["daynight"]: (f"Day vs evening sessions ({_window_text()}, by the batch "
                                  "folder's first pre-test stamp, Pi clock).\nEvening data "
                                  "is one Exp 2 session and no Exp 3; written by "
                                  "`per_analysis.daynight`."),
        TABLES_DIR: ("Tables (CSV) written by the analysis modules, one file per table.\n"
                     "A figure's own numbers sit beside it as `<figure>.csv` instead."),
    }


def default_readmes() -> "OrderedDict[str, str]":
    """Relative folder -> default README text, in spec order (07's mirror excluded)."""
    top = _top_readmes()
    out: OrderedDict[str, str] = OrderedDict()
    out[QC_DIR] = top[QC_DIR]
    for key, subs in (("exp1", _EXP1_SUBS),
                      ("exp2", _pair_subs("Exp 2", "IAA -> partner")),
                      ("exp3", _pair_subs("Exp 3", "partner -> IAA")),
                      ("cross", _CROSS_SUBS)):
        out[MODULE_DIRS[key]] = top[MODULE_DIRS[key]]
        for sub, text in subs.items():
            out[f"{MODULE_DIRS[key]}/{sub}"] = f"{text}\n{_STATS_LINE}" \
                if sub != "models" else text + "\nOne CSV of coefficients and one text " \
                                                   "summary per model."
    out[MODULE_DIRS["power"]] = top[MODULE_DIRS["power"]]
    out[SENSITIVITY_DIR] = top[SENSITIVITY_DIR]
    out[MODULE_DIRS["daynight"]] = top[MODULE_DIRS["daynight"]]
    out[TABLES_DIR] = top[TABLES_DIR]
    return out


def mirror_readmes(modules: Sequence[str] = SENSITIVITY_MODULES) -> "OrderedDict[str, str]":
    """07's mirror of ``modules``' folders (+ its tables/), README texts prefixed."""
    base = default_readmes()
    out: OrderedDict[str, str] = OrderedDict()
    for m in modules:
        top = MODULE_DIRS[m]
        for rel, text in base.items():
            if rel == top or rel.startswith(top + "/"):
                first, _, rest = text.partition("\n")
                first = first.replace("PRIMARY. ", "PRIMARY (re-run). ")
                out[f"{SENSITIVITY_DIR}/{rel}"] = (
                    f"With QC exclusions (all rules on): {first}\n"
                    + (rest or "Same analysis as the main run, on the flies that pass QC."))
    out[f"{SENSITIVITY_DIR}/{TABLES_DIR}"] = ("Tables of the sensitivity run (QC exclusions "
                                              "on), named as in the main `tables/`.\n"
                                              "Compare each with its main-run namesake.")
    return out


def spec_folders(sensitivity_modules: Sequence[str] = SENSITIVITY_MODULES) -> list[str]:
    return list(default_readmes()) + list(mirror_readmes(sensitivity_modules))


def create_tree(results_dir: Path | str,
                sensitivity_modules: Sequence[str] = SENSITIVITY_MODULES) -> list[Path]:
    """mkdir every spec folder (and 07's mirror). Returns the folders."""
    res = Path(results_dir)
    dirs = [res / rel for rel in spec_folders(sensitivity_modules)]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def write_readmes(results_dir: Path | str,
                  sensitivity_modules: Sequence[str] = SENSITIVITY_MODULES) -> list[Path]:
    """Give every spec folder a README.md. A README already there (a module's) is kept
    as is, except that a PRIMARY folder's README lacking "PRIMARY" gets one line appended.
    Returns the READMEs written or amended."""
    res = Path(results_dir)
    texts = OrderedDict(default_readmes())
    texts.update(mirror_readmes(sensitivity_modules))
    primary = set(PRIMARY_FOLDERS) | {f"{SENSITIVITY_DIR}/{p}" for p in PRIMARY_FOLDERS}
    written = []
    for rel, text in texts.items():
        d = res / rel
        d.mkdir(parents=True, exist_ok=True)
        readme = d / "README.md"
        if not readme.exists():
            readme.write_text(text.rstrip("\n") + "\n")
            written.append(readme)
        elif rel in primary:
            current = readme.read_text()
            if "PRIMARY" not in current:
                readme.write_text(current.rstrip("\n") + "\n" + PRIMARY_LINE + "\n")
                log.info("marked %s PRIMARY", readme)
                written.append(readme)
    log.info("READMEs: %d written or amended, %d folders checked", len(written), len(texts))
    return written


def readme_caption(folder: Path | str) -> str:
    """First non-empty line of ``folder/README.md`` without markdown heading marks; a
    heading that only names the folder is skipped. '' when there is no README."""
    folder = Path(folder)
    try:
        lines = (folder / "README.md").read_text().splitlines()
    except OSError:
        return ""
    names = {folder.name.lower()}
    for line in lines:
        text = line.strip().lstrip("#").strip()
        if not text:
            continue
        if line.lstrip().startswith("#") and text.lower().strip("/ ") in names:
            continue
        return text
    return ""


# ── findings ──────────────────────────────────────────────────────────────────
FINDING_KEYS = ("section", "analysis", "primary", "text", "numbers")
SPECIAL = ("design_contradiction", "skipped", "failed")


def normalize_findings(findings: Iterable[Any] | None, *, module: str | None = None,
                       run: str | None = None) -> list[dict]:
    """Copy each finding into a dict carrying every contract key (+ module, run)."""
    out = []
    for f in findings or ():
        d = dict(f) if isinstance(f, Mapping) else {"text": str(f)}
        missing = [k for k in FINDING_KEYS if k not in d]
        if missing:
            log.warning("finding from %s lacks %s: %r", module, missing, d)
        d["section"] = str(d.get("section") or MODULE_DIRS.get(module or "", module or "")
                           or "unsectioned")
        d["analysis"] = str(d.get("analysis") or "unnamed")
        d["primary"] = bool(d.get("primary", False))
        d["text"] = "" if d.get("text") is None else str(d["text"])
        nums = d.get("numbers")
        d["numbers"] = {} if nums is None else (dict(nums) if isinstance(nums, Mapping)
                                                else {"value": nums})
        if module is not None:
            d.setdefault("module", module)
        if run is not None:
            d.setdefault("run", run)
        out.append(d)
    return out


def _fmt_scalar(v: Any) -> str:
    if v is None:
        return "NA"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return str(v)
        if v == 0:
            return "0"
        return f"{v:.2e}" if abs(v) < 1e-3 or abs(v) >= 1e6 else f"{v:.3g}"
    try:                                    # numpy scalars
        import numpy as np
        if isinstance(v, np.integer):
            return str(int(v))
        if isinstance(v, np.floating):
            return _fmt_scalar(float(v))
        if isinstance(v, np.bool_):
            return str(bool(v))
    except ImportError:                     # pragma: no cover
        pass
    text = " ".join(str(v).split())
    return text if len(text) <= 80 else text[:77] + "..."


def _flatten(numbers: Mapping, prefix: str = "") -> list[tuple[str, Any]]:
    out = []
    for k, v in numbers.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out += _flatten(v, key + ".")
        else:
            out.append((key, v))
    return out


def format_numbers(numbers: Mapping | None, limit: int = 10) -> str:
    """``{"p": 0.0123, "ci": [0.1, 0.4]}`` -> ``"p = 0.0123; ci = [0.1, 0.4]"``."""
    items = _flatten(numbers or {})
    parts = []
    for k, v in items[:limit]:
        if isinstance(v, (list, tuple)):
            shown = ", ".join(_fmt_scalar(x) for x in list(v)[:6])
            text = f"[{shown}{', ...' if len(v) > 6 else ''}]"
        else:
            text = _fmt_scalar(v)
        parts.append(f"{k} = {text}")
    if len(items) > limit:
        parts.append(f"... {len(items) - limit} more in findings.json")
    return "; ".join(parts)


# ── SUMMARY.md ────────────────────────────────────────────────────────────────
def _window_caveats(results_dir: Path | None) -> list[str]:
    """Light-probe and short-odor-window bullets, counted from the cached trial table."""
    probe = ("**Light-only probe**: post-test sessions end with a light-only opto probe, used "
             "here as a positive control and kept out of every odor analysis.")
    t = None
    if results_dir is not None:
        try:
            import pandas as pd
            from . import io
            path = Path(results_dir) / QC_DIR / io.CACHE_NAME
            if path.exists():
                t = pd.read_parquet(path, columns=["is_light_probe", "odor_window_s"])
        except Exception:                   # pragma: no cover - a caveat must never fail
            t = None
    if t is None:
        return [f"{probe} Since 2026-11-18 a probe's window is its light, from the sensors log "
                "(the 2026-11-17 pair probes had a 0.025 s window before); not checked here: "
                "no trial table."]
    import pandas as pd
    w = pd.to_numeric(t["odor_window_s"], errors="coerce")
    lp = t["is_light_probe"].astype(bool)
    bad_p, bad_o = lp & (w < 1.0), ~lp & (w < 1.0)
    out = []
    if bad_p.any():
        out.append(f"{probe} {int(bad_p.sum())} of {int(lp.sum())} light probes have a window "
                   f"under 1 s ({w[bad_p].min():.3f} s) in the trial table and are left out: with "
                   "no odor in the frame log the window fell back to the clamped 30-60 s default "
                   "(the last frame). Fixed in code on 2026-11-18 (the window is the light, from "
                   "the sensors log); it reaches the tables on the next full pipeline run.")
    else:
        out.append(f"{probe} All {int(lp.sum())} light probes have a window of at least 1 s (a "
                   "probe's window is its light, from the sensors log, since 2026-11-18).")
    if bad_o.any():
        out.append(f"**Short odor windows**: {int(bad_o.sum())} odor trials have a window under "
                   f"1 s ({w[bad_o].min():.3f} s) and are left out: the paraffin click fell "
                   "between frame reads. Fixed in code on 2026-11-18 (window from the sensors "
                   "log); it reaches the tables on the next full pipeline run.")
    return out


def _known_contradictions(results_dir: Path | None = None) -> list[str]:
    try:
        from . import io
        nop = io.NO_PARAFFIN_TEST
        partly = f" ({nop.get('Hex-Sensitivity-1-0.1')} Hex and " \
                 f"{nop.get('3Oct-Sensitivity-1-0.1')} 3Oct trial-table flies have no " \
                 "paraffin test; `01_data_qc/protocol_facts.csv`)"
    except Exception:                       # pragma: no cover - io is foundation
        partly = ""
    return [
        "**Exp 3 alone block**: it presents all 8 odors, not just IAA.",
        "**Paraffin partner in the pair experiments**: since 2026-11-10 it is a 50 ms "
        "master-valve click with no odor, not a paraffin-oil puff (`is_click`).",
        "**Exp 1 paraffin test**: the IAA-Sensitivity cohort was never tested on paraffin, "
        f"and Hex-/3Oct-Sensitivity only partly: paraffin joined the panel partway{partly}.",
        "**Paraffin control cohort**: Paraffin-Sensitivity was not run concurrently with "
        "IAA-Sensitivity (Nov 5-12 vs Oct 9-Nov 1), and some of its flies ran the "
        "paraffin-off protocol (paraffin line shut off 30 s; "
        "`config/paraffin_off_protocol_trials.csv`).",
        *_window_caveats(results_dir),
        "**Odor 2 baseline**: odor 2 of a pair is scored against the baseline before odor 1, "
        "so its score can carry odor 1's response over (carry-over); see each experiment's "
        "`d_shared_baseline_diagnostic`.",
        "**Evening data**: a single session for Exp 2 and none for Exp 3, so day-vs-evening "
        "contrasts rest on one Exp 2 session.",
        # Exp 1 (found by the exp1 analysis, 2026-11-18)
        "**Exp 1 paraffin position**: Paraffin Oil is always the first pre-test and the last "
        "post-test trial, so its pre-to-post change is confounded with trial position.",
        "**Exp 1 position vs time**: `position_in_block` and `minutes_since_exposure_end` "
        "are the same clock (r = 0.9999998) because the odor order is fixed per batch; "
        "their effects cannot be separated.",
        "**Exp 1 concentration**: `exposure_conc` is collinear with cohort, so concentration "
        "and exposure odor cannot be separated.",
        "**Exp 1 start times**: unbalanced. The IAA cohort includes 26 evening flies, while "
        "Hex and Paraffin are day-only.",
    ]


def _methods_caveats() -> list[str]:
    return [
        "**GEE is the model of record** (logistic, fly-clustered, robust SE). The results "
        "rest on GEE plus the exact sign-flip and whole-batch permutation tests.",
        "**Mixed-logistic check unreliable**: lme4/pymer4 are not installed, so the "
        "mixed-logistic sensitivity check (`stats.fit_mixed_logistic`) is a variational "
        "Bayesian approximation (statsmodels `BinomialBayesMixedGLM.fit_vb`). On the Exp 1 "
        "full model it disagreed in sign with GEE (+0.36, SD 0.10 vs -0.41, SE 0.55) with "
        "SDs about 5x too narrow, so it is not used for inference here.",
    ]


def _assumptions(threshold: Any) -> list[str]:
    return [
        "Genotype: female Gr5a-GAL4 > UAS-CsChrimson in every cohort.",
        "The fly is the unit. A fly's repeated trials of a condition are averaged before a "
        "paired test; intervals and permutation tests count flies. Rates and mean scores pool "
        "trials (sum / trials), never a mean of per-fly means.",
        f"Response = score >= {threshold} (`reaction_prediction.binary_threshold`, "
        "config/config_new.yaml). Scores are ordinal -1..5; -1 (retraction) is a valid level.",
        "A pair trial is one row: the scored odor is odor 2; odor 1's score rides along as "
        "`odor1_score`. In the pair experiments \"Paraffin Oil\" is the master-valve click.",
        "The light-only probe is a positive control only, never an odor trial.",
        f"Clock windows of the batch folder's first pre-test stamp (Pi clock, end exclusive): "
        f"{_window_text()}; anything else is \"{C.OTHER_WINDOW}\".",
        "Stars come from Holm-adjusted fly-level p only. Every comparison also carries "
        "p_batch (whole-batch sign-flip); a result whose p_batch disagrees is fragile.",
        f"Fixed seeds: SEED = {C.SEED} (reset before every module), N_BOOT = {C.N_BOOT}, "
        f"N_PERM = {C.N_PERM}.",
        "Models are statsmodels (no pymer4/rpy2/mord). GEE (fly-clustered) is the model of "
        "record; the variational mixed logistic is a check only (see section 4).",
    ]


def _qc_report_parts(qc_md: str) -> tuple[str, str]:
    """(intro paragraph, 'Flies per cohort x window' section body) of qc_report.md."""
    lines = qc_md.splitlines()
    intro, i = [], 0
    while i < len(lines) and not lines[i].strip().startswith("# "):
        i += 1
    i += 1
    while i < len(lines) and not lines[i].startswith("#"):
        intro.append(lines[i])
        i += 1
    flies, on = [], False
    for line in lines:
        if line.startswith("## "):
            if on:
                break
            on = "flies per cohort" in line.lower()
            continue
        if on:
            flies.append(line)
    return "\n".join(intro).strip(), "\n".join(flies).strip()


def _column_mapping_md(path: Path) -> list[str]:
    try:
        mapping = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        return [f"Column mapping unavailable ({path.name}: {exc})."]
    contract = {k: v for k, v in mapping.items() if (v or {}).get("contract", True)}
    extra = [k for k in mapping if k not in contract]
    out = ["| column | source | meaning |", "|---|---|---|"]
    for col, v in contract.items():
        v = v or {}
        out.append(f"| `{col}` | {_md_cell(v.get('source', ''))} | "
                   f"{_md_cell(v.get('description', ''))} |")
    if extra:
        out += ["", "Extra columns: " + ", ".join(f"`{c}`" for c in extra)
                + f" (see `{QC_DIR}/{path.name}`)."]
    return out


def _md_cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _finding_block(f: Mapping, *, numbers_limit: int = 6) -> list[str]:
    text = f.get("text", "").strip() or "(no text)"
    nums = format_numbers(f.get("numbers"), numbers_limit)
    out = [text]
    if nums:
        out.append(f"*Numbers:* {nums}")
    return out


def _by_section(findings: Sequence[Mapping]) -> "OrderedDict[str, list]":
    groups: OrderedDict[str, list] = OrderedDict()
    for f in findings:
        groups.setdefault(f["section"], []).append(f)
    return groups


def _run_line(run_info: Mapping | None) -> list[str]:
    if not run_info:
        return []
    mods = run_info.get("modules") or {}
    parts = []
    for name, st in mods.items():
        cell = f"{name} {st.get('main', 'not run')}"
        if st.get("exclusions"):
            cell += f" / 07 {st['exclusions']}"
        if st.get("seconds"):
            cell += f" ({float(st['seconds']):.0f} s)"
        parts.append(cell)
    out = []
    if parts:
        out.append("Run status (main / 07): " + "; ".join(parts) + ".")
    if run_info.get("seconds") is not None:
        out.append(f"Total runtime {float(run_info['seconds']):.0f} s.")
    return out


def write_summary(results_dir: Path | str, findings: Sequence[Mapping],
                  qc_summary: Mapping | str | None = None, *,
                  run_info: Mapping | None = None) -> Path:
    """Write ``results_dir/SUMMARY.md``. ``qc_summary``: run_all's QC dict (n_trials,
    n_flies, n_cohorts, threshold, flags, excluded, main_exclusions, checks) or a
    markdown string; None reads only ``01_data_qc/qc_report.md``."""
    res = Path(results_dir)
    res.mkdir(parents=True, exist_ok=True)
    fs = normalize_findings(findings)
    for f in fs:
        f.setdefault("run", "main")
    main = [f for f in fs if f["run"] != "exclusions"]
    excl = [f for f in fs if f["run"] == "exclusions"]
    qs = qc_summary if isinstance(qc_summary, Mapping) else {}
    thr = qs.get("threshold", C.RESPONSE_THRESHOLD)

    L = ["# PER analysis: summary", "",
         f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `run_all.py`. Figures: "
         "[index.html](index.html). QC: "
         f"[{QC_DIR}/qc_report.md]({QC_DIR}/qc_report.md)."]
    L += _run_line(run_info)
    L.append("")

    # 1. data and assumptions
    L += ["## 1. Data and assumptions", "", "### Data and QC", ""]
    if qs.get("n_trials") is not None:
        L.append(f"- {qs['n_trials']} scored trials from {qs.get('n_flies', '?')} flies in "
                 f"{qs.get('n_cohorts', '?')} cohorts, one row per scored trial "
                 "(`per_analysis.io.load_trials()`).")
    flags = qs.get("flags") or {}
    if flags:
        L.append("- QC flags (per fly): " + ", ".join(f"{k.replace('_', ' ')} {v}"
                                                      for k, v in flags.items())
                 + f" of {qs.get('n_flies', '?')} flies.")
    L.append(f"- Main run exclusions: {qs.get('main_exclusions', 'none')}.")
    exc = qs.get("excluded") or {}
    if exc:
        rules = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in exc.items()
                          if k not in ("union", "kept"))
        L.append(f"- Sensitivity run (`{SENSITIVITY_DIR}/`, with exclusions): every QC rule "
                 f"on drops {exc.get('union', '?')} flies ({rules}); "
                 f"{exc.get('kept', '?')} kept.")
    for c in qs.get("checks") or ():
        L.append(f"- {c}")
    if qs.get("status") and qs["status"] != "ok":
        L.append(f"- QC status: {qs['status']}")
    qc_md_path = res / QC_DIR / "qc_report.md"
    qc_md = qc_summary if isinstance(qc_summary, str) else None
    if qc_md is None and qc_md_path.exists():
        qc_md = qc_md_path.read_text()
    if qc_md:
        intro, flies = _qc_report_parts(qc_md)
        if intro:
            L += ["", f"From `{QC_DIR}/qc_report.md`: {intro}"]
        if flies:
            L += ["", "Flies per cohort x window:", "", flies]
    else:
        L += ["", "QC report not found."]
    L += ["", "### Assumptions", ""] + [f"- {a}" for a in _assumptions(thr)]
    L += ["", "### Column mapping", "",
          f"Contract columns of the tidy table (`{QC_DIR}/column_mapping.yaml`):", ""]
    L += _column_mapping_md(res / QC_DIR / "column_mapping.yaml")
    L.append("")

    # 2. primary results
    L += ["## 2. PRIMARY results", ""]
    prim = [f for f in main if f["primary"] and f["analysis"] not in SPECIAL]
    prim_excl = [f for f in excl if f["primary"] and f["analysis"] not in SPECIAL]
    if not prim and not prim_excl:
        L += ["No primary finding was produced (see Not analysed).", ""]
    # the k-th main finding of a (section, analysis) pairs with the k-th exclusion-run one
    twins: dict = {}
    for g in prim_excl:
        twins.setdefault((g["section"], g["analysis"]), []).append(g)
    used: dict = {}
    for f in prim:
        key = (f["section"], f["analysis"])
        L += [f"### {f['section']}: {f['analysis']}", ""] + _finding_block(f, numbers_limit=10)
        k = used.get(key, 0)
        used[key] = k + 1
        if k < len(twins.get(key, ())):
            L += ["", f"**With exclusions ({SENSITIVITY_DIR}):** "
                      f"{twins[key][k]['text'].strip()}"]
        L.append("")
    for key, gs in twins.items():
        for g in gs[used.get(key, 0):]:
            L += [f"### {g['section']}: {g['analysis']} (with exclusions only)", ""] \
                + _finding_block(g, numbers_limit=10) + [""]

    # 3. by analysis
    L += ["## 3. Results by analysis", ""]
    regular = [f for f in main if f["analysis"] not in SPECIAL]
    if not regular:
        L += ["No analysis findings.", ""]
    for section, group in _by_section(regular).items():
        L += [f"### {section}", ""]
        for f in group:
            tag = " (PRIMARY, above)" if f["primary"] else ""
            block = _finding_block(f)
            L += [f"**{f['analysis']}**{tag}: {block[0]}"] + block[1:] + [""]
    regular_x = [f for f in excl if f["analysis"] not in SPECIAL]
    if regular_x:
        L += [f"### Sensitivity run with exclusions ({SENSITIVITY_DIR})", ""]
        for section, group in _by_section(regular_x).items():
            L += [f"#### {section}", ""]
            for f in group:
                block = _finding_block(f)
                L += [f"**{f['analysis']}**: {block[0]}"] + block[1:] + [""]

    # 4. data vs described design
    L += ["## 4. Data vs described design / methods caveats", "",
          "Where the data differ from the described design:", ""]
    L += [f"- {c}" for c in _known_contradictions(res)]
    L += ["", "Methods caveats:", ""] + [f"- {c}" for c in _methods_caveats()]
    seen_txt = set()
    reported = []
    for f in fs:
        if f["analysis"] == "design_contradiction" and f["text"].strip() not in seen_txt:
            seen_txt.add(f["text"].strip())
            reported.append(f)
    if reported:
        L += ["", "Reported by the analysis modules:", ""]
        for f in reported:
            nums = format_numbers(f.get("numbers"))
            L.append(f"- ({f['section']}) {f['text'].strip()}"
                     + (f" [{nums}]" if nums else ""))
    L.append("")

    # 5. not analysed
    L += ["## 5. Not analysed", ""]
    failed = [f for f in fs if f["analysis"] == "failed"]
    skipped = [f for f in fs if f["analysis"] == "skipped"]
    if not failed and not skipped:
        L += ["Every requested module ran and nothing was skipped.", ""]
    for f in failed:
        where = "07 with exclusions" if f.get("run") == "exclusions" else "main run"
        L.append(f"- **FAILED** `{f.get('module', f['section'])}` ({where}): "
                 f"{f['text'].strip()}")
    for f in skipped:
        where = " (07 with exclusions)" if f.get("run") == "exclusions" else ""
        L.append(f"- Skipped, {f['section']}{where}: {f['text'].strip()}")
    L.append("")

    path = res / "SUMMARY.md"
    path.write_text("\n".join(L))
    log.info("wrote %s (%d findings)", path, len(fs))
    return path


# ── index.html ────────────────────────────────────────────────────────────────
_CSS = """
:root{color-scheme:light dark;--bg:#f7f7f5;--surface:#ffffff;--ink:#1d1d1b;--muted:#5f5e5a;
--line:#e2e1dc;--accent:#1f5fb4;--chip:#eceae5}
@media (prefers-color-scheme: dark){:root{--bg:#141413;--surface:#1e1e1c;--ink:#ecebe6;
--muted:#a5a49e;--line:#33322e;--accent:#8ab6f2;--chip:#2a2a27}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1240px;margin:0 auto;padding:16px}
h1{font-size:1.5rem;margin:.2rem 0 .3rem}
h2{font-size:1.05rem;margin:0;word-break:break-word}
a{color:var(--accent)}
.meta{color:var(--muted);margin:0 0 .8rem}
.links,.toc{display:flex;flex-wrap:wrap;gap:6px;margin:.4rem 0;padding:0;list-style:none}
.links a,.toc a{display:inline-block;padding:3px 9px;border-radius:999px;background:var(--chip);
text-decoration:none;font-size:.85rem;color:var(--ink)}
.toc a span{color:var(--muted)}
section{background:var(--surface);border:1px solid var(--line);border-radius:10px;
padding:14px;margin:14px 0}
.caption{color:var(--muted);margin:.25rem 0 .8rem;font-size:.9rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px}
figure{margin:0;border:1px solid var(--line);border-radius:8px;overflow:hidden;
background:var(--surface)}
figure a{display:block;background:#ffffff}
figure img{display:block;width:100%;height:170px;object-fit:contain}
figcaption{padding:6px 8px;font-size:.78rem;color:var(--muted);word-break:break-all}
ul.files{margin:.3rem 0 .6rem;padding-left:1.1rem;columns:2 320px;font-size:.88rem}
ul.files li{word-break:break-all}
@media (max-width:520px){.grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
figure img{height:120px}.wrap{padding:12px 16px}ul.files{columns:1}}
"""


def _href(rel: Path | str) -> str:
    return quote(Path(rel).as_posix())


def _anchor(rel: str) -> str:
    return "f-" + "".join(ch if ch.isalnum() else "-" for ch in rel).strip("-")


def _visible(rel: Path) -> bool:
    return not any(part.startswith(".") for part in rel.parts)


def write_index(results_dir: Path | str) -> Path:
    """Write ``results_dir/index.html``: every PNG as a linked thumbnail, grouped by
    folder in folder order and captioned by the folder README's first line; links to
    every file in ``tables/`` (and 07's), SUMMARY.md and the QC report. Relative links,
    no external resources."""
    res = Path(results_dir)
    res.mkdir(parents=True, exist_ok=True)
    pngs = [p.relative_to(res) for p in res.rglob("*.png")]
    pngs = sorted((p for p in pngs if _visible(p)), key=lambda p: (p.parent.parts, p.name))
    groups: OrderedDict[Path, list[Path]] = OrderedDict()
    for p in pngs:
        groups.setdefault(p.parent, []).append(p)

    table_dirs = [Path(TABLES_DIR), Path(SENSITIVITY_DIR) / TABLES_DIR]
    tables: OrderedDict[Path, list[Path]] = OrderedDict()
    for td in table_dirs:
        if (res / td).is_dir():
            files = sorted(p.relative_to(res) for p in (res / td).rglob("*")
                           if p.is_file() and p.name != "README.md"
                           and _visible(p.relative_to(res)))
            tables[td] = files
    n_tables = sum(len(v) for v in tables.values())

    reports = [("SUMMARY.md", "SUMMARY.md"),
               (f"{QC_DIR}/qc_report.md", "QC report"),
               (f"{QC_DIR}/qc_report.csv", "QC flags per fly (CSV)"),
               (f"{QC_DIR}/column_mapping.yaml", "Column mapping"),
               (f"{QC_DIR}/protocol_facts.csv", "Protocol facts"),
               (f"{SENSITIVITY_DIR}/excluded_flies.csv", "Flies excluded in 07"),
               ("findings.json", "findings.json"),
               ("run_all.log", "run_all.log")]
    esc = html.escape
    out = ["<!doctype html>", '<html lang="en">', "<head>", '<meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width, initial-scale=1">',
           '<meta name="color-scheme" content="light dark">',
           "<title>PER analysis figures</title>", f"<style>{_CSS}</style>", "</head>",
           "<body>", '<div class="wrap">', "<header>", "<h1>PER analysis results</h1>",
           f'<p class="meta">{len(pngs)} figures in {len(groups)} folders, {n_tables} '
           f'tables. Generated {dt.datetime.now():%Y-%m-%d %H:%M}.</p>',
           '<ul class="links">']
    for rel, label in reports:
        if (res / rel).exists():
            out.append(f'<li><a href="{_href(rel)}">{esc(label)}</a></li>')
    if tables:
        out.append('<li><a href="#tables">Tables</a></li>')
    out += ["</ul>", '<nav><ul class="toc">']
    for folder, files in groups.items():
        rel = folder.as_posix() if folder.parts else "."
        out.append(f'<li><a href="#{_anchor(rel)}">{esc(rel)} <span>{len(files)}</span></a>'
                   "</li>")
    out += ["</ul></nav>", "</header>", "<main>"]
    if not groups:
        out.append('<section><p class="caption">No figures yet.</p></section>')
    for folder, files in groups.items():
        rel = folder.as_posix() if folder.parts else "."
        cap = readme_caption(res / folder)
        out += [f'<section id="{_anchor(rel)}">',
                f'<h2><a href="{_href(folder) if folder.parts else "."}/">{esc(rel)}</a></h2>']
        if cap:
            out.append(f'<p class="caption">{esc(cap)}</p>')
        out.append('<div class="grid">')
        for p in files:
            href = _href(p)
            out.append(f'<figure><a href="{href}"><img src="{href}" loading="lazy" '
                       f'alt="{esc(p.name)}"></a><figcaption>{esc(p.name)}</figcaption>'
                       "</figure>")
        out += ["</div>", "</section>"]
    if tables:
        out += ['<section id="tables">', "<h2>Tables</h2>"]
        for td, files in tables.items():
            cap = readme_caption(res / td)
            out.append(f'<p class="caption"><a href="{_href(td)}/">{esc(td.as_posix())}/</a>'
                       f"{': ' + esc(cap) if cap else ''}</p>")
            if files:
                out.append('<ul class="files">')
                out += [f'<li><a href="{_href(f)}">{esc(f.relative_to(td).as_posix())}</a>'
                        "</li>" for f in files]
                out.append("</ul>")
            else:
                out.append('<p class="caption">No tables.</p>')
        out.append("</section>")
    out += ["</main>", "</div>", "</body>", "</html>", ""]
    path = res / "index.html"
    path.write_text("\n".join(out))
    log.info("wrote %s (%d figures, %d tables)", path, len(pngs), n_tables)
    return path
