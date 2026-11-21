#!/usr/bin/env python3
"""Back-to-back (B2B) IAA odor-pair figures: naive (pre-block) vs paired response.

Output: /home/ramanlab/Documents/cole/Results/b2b. Run by the pipeline after every run
(``analysis.b2b_pair_figures`` in config_new.yaml, user 2026-11-16) and redrawn only when
what it is drawn from changed: the two datasets' rows of the wide tables (frozen
folders left out), their model scores, the reaction threshold and this code with what
it imports. Flies the flagged-flies CSV marks dead or bad are left out, as the scorer
leaves them out. A figure missing since the last build also redraws; ``--force`` always
does. Delete ``_build_cache.json`` in the folder to force a redraw by hand.

``--start HH:MM --end HH:MM`` keeps only batch folders whose first pre-test trial
started in that clock window (Pi clock, read from the raw trial file name; end
exclusive). The pipeline builds one folder per window (``windows:`` in the config,
user 2026-11-17: Results/b2b 9:40am-4:20pm, Results/b2b-Evening 4:40pm-9:20pm). Each
folder's README.md lists the folders it holds and the ones it left out. A window no
batch started in yet keeps only its README.

Protocol (PiCode config_v2_pair*_A.yaml): 8 single-odor pre-block trials, 30 min,
16 pair trials ``testing_<N>_<odor1>_then_<odor2>`` (10 s odor, 1 s gap, 10 s odor);
each partner appears twice (round 1, round 2). Each odor of a pair is scored on its
own against the trial's pre-odor block (fbpipe.utils.odor_pairs; odor 2's score is
``<label>__odor2`` in model_predictions.csv).

* IAA-PairFirst (IAA -> partner): per odor X, X alone (pre-block) vs X right after
  IAA (odor 2 of the pair, both rounds).
* IAA-PairSecond (partner -> IAA): per preceding odor X, IAA alone (pre-block) vs
  IAA right after X (odor 2 of the pair, both rounds).
* Both: per partner X, the second odor's naive response vs its response in the pair,
  pooled over both datasets (PairFirst: X; PairSecond: IAA).
* <group>_scores(_pooled)_odor_plus_5s: <group>_scores(_pooled) with every score taken
  on the odor window extended 5 s past odor-off (valve on -> off + 5 s, shifted by the
  model's 1 s odor-arrival delay as every score is), green / magenta bars: the naive
  pre-test trial and odor 2 of each pair (production's odor-2 row: the pre-odor block,
  then everything from odor 2's onset). Rescored with reaction_prediction.model_path
  (scripts/analysis/window_rescore.py, which reproduces production on 100 % of these
  trials on the plain odor window). Odor 1 is never extended: odor 2 reaches the fly
  2 s after odor 1's valve closes. Scores in <group>_odor_plus_5s_scores.csv.
* <group>_change_minus_paraffin(_pooled)(_odor_plus_5s): Paraffin Oil as the control
  (user 2026-11-18). Per odor, its change from alone to in-pair minus Paraffin Oil's
  change in the same flies (IAA-first: X after IAA; IAA-second: IAA after X vs IAA
  after Paraffin Oil), solid; Paraffin Oil's own change hatched on the right. Paraffin
  Oil follows IAA as every odor does, so what it gains (time in the rig, starvation,
  IAA just before) is subtracted, leaving what the odor itself adds. Bars pool trials;
  the test is an exact sign-flip of one value per fly against 0, Holm across the panel.
* Pre-Exposure-Baseline/ (this folder's flies) and Time-of-Day_2h-bins_whole-dataset/
  (every fly, all clock windows): scripts/analysis/b2b_baseline_time_of_day.py.
* <group>_odor1_vs_naive(_pooled): odor 1 of each pair (nothing precedes it within its
  trial) vs the same odor naive in the pre-test. IAA-first: IAA, per pair; IAA-second:
  each partner odor. Shows whether responding changed from the pre-test to the test
  block; rows in b2b_odor1_presentations.csv.
* pretest/: the naive pre-test, every odor alone, IAA-first vs IAA-second flies (a
  two-sample permutation test per odor: do the two groups start out alike?).
* odor1_vs_odor2/: within a pair trial, does a fly that reacts to odor 1 also react
  to odor 2? "Reacts" = model score >= reaction_prediction.binary_threshold
  (config_new.yaml). Counts are pair trials; trials of one fly are not independent,
  so no p-values are drawn.

Traces are aligned to the scored odor's valve onset (0 s, shaded 10 s). Means pool
trials (never a mean of per-fly means); n counts trials.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from fbpipe.config import canon_fly_number, load_flagged_fly_exclusions  # noqa: E402
from fbpipe.utils import figure_cache as fc  # noqa: E402
from fbpipe.utils import odor_pairs as op  # noqa: E402
from scripts.analysis import b2b_baseline_time_of_day as bt  # noqa: E402
from scripts.analysis import b2b_stats as bs  # noqa: E402
from fbpipe.analysis import fly_stats as fs  # noqa: E402

try:   # the production scorer; without it only the odor + 5 s figure is skipped
    from scripts.analysis import window_rescore as wr  # noqa: E402
except ImportError as exc:  # pragma: no cover - the pipeline env has it
    wr = None
    _WR_ERROR = str(exc)

CSV_DIR = Path("/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys")
DATA_ROOT = Path("/home/ramanlab/Documents/cole/Data/flys_New")
#: Where a batch's rig frame logs still are once local data is cleaned up.
SECURED_ROOT = Path("/securedstorage/DATAsec/cole/Data-secured-New")
OUT = Path("/home/ramanlab/Documents/cole/Results/b2b")
CACHE_FILE = "_build_cache.json"
LOCK_TIMEOUT_S = 30 * 60
WIDE_TABLES = ("all_envelope_rows_wide_combined_base", "all_envelope_rows_wide_combined_base_pretest")
REC_COLS = ["dataset", "fly", "fly_number", "role", "round", "odor", "partner", "label",
            "score", "onset_s", "fps", "trace"]
FIRST, SECOND = "IAA-PairFirst-Control-1-1", "IAA-PairSecond-Control-1-1"
DATASETS = (FIRST, SECOND)
WINDOW_S = (-10.0, 30.0)
ODOR_S = 10.0

NAMES = {"isoamylacetate": "IAA", "hexanol": "Hexanol", "linalool": "Linalool",
         "benzaldehyde": "Benzaldehyde", "ethylbutyrate": "Ethyl Butyrate", "citral": "Citral",
         "3-octonol": "3-Octanol", "paraffinoil": "Paraffin Oil"}
ORDER = ["IAA", "Hexanol", "Linalool", "Benzaldehyde", "Ethyl Butyrate", "Citral",
         "3-Octanol", "Paraffin Oil"]
COLORS = {"pre": "#222222", 1: "#1b9e77", 2: "#d95f02"}
_TRIAL = re.compile(r"^(pretest|testing)_(\d+)_(.+)$", re.IGNORECASE)


def odor_name(token: str) -> str:
    """'isoamylacetate-1' -> 'IAA' (concentration suffix dropped)."""
    base = re.sub(r"-[0-9.]+$", "", str(token).strip().lower())
    return NAMES.get(base, base)


def parse_label(label: str):
    """-> (phase, number, [odor names]) or None."""
    m = _TRIAL.match(str(label))
    if not m:
        return None
    odors = [odor_name(t) for t in re.split(r"_then_", m.group(3), flags=re.IGNORECASE)]
    return m.group(1).lower(), int(m.group(2)), odors


def _dir_cols(df):
    return sorted((c for c in df.columns if c.startswith("dir_val_")), key=lambda c: int(c[8:]))


def _pretest_by(pre_df: pd.DataFrame) -> dict:
    """(dataset, fly, fly_number, odor) -> that odor's single-odor pre-block row."""
    by = {}
    for _, r in pre_df.iterrows():
        p = parse_label(r["trial_label"])
        if p and p[0] == "pretest" and len(p[2]) == 1:
            by[(r["dataset"], r["fly"], r["fly_number"], p[2][0])] = r
    return by


def _pair_rows(pair_df: pd.DataFrame) -> list:
    """[(trial number, (odor 1, odor 2), row)] of the pair trials, in trial order."""
    pairs = []
    for _, r in pair_df.iterrows():
        p = parse_label(r["trial_label"])
        if p and p[0] == "testing" and len(p[2]) == 2:
            pairs.append((p[1], p[2], r))
    return sorted(pairs, key=lambda t: t[0])


def odor1_records(pre_df: pd.DataFrame, pair_df: pd.DataFrame, scores: dict) -> pd.DataFrame:
    """One record per scored presentation of odor 1, the odor that opens a pair trial,
    plus that same odor's naive pre-block trial (role 'pre', once per partner).

    IAA-first: odor 1 is IAA (partner = the odor after it). IAA-second: odor 1 is the
    partner itself. Odor 1 follows nothing within its trial, so it shows how responding
    to an odor changed from the naive pre-test to the test block. Columns as
    ``build_records``; no traces."""
    recs = []
    pre_by = _pretest_by(pre_df)
    seen_round = {}
    for _num, (o1, o2), r in _pair_rows(pair_df):
        ds, fly, fn = r["dataset"], r["fly"], r["fly_number"]
        partner = o2 if ds == FIRST else o1
        k = (ds, fly, fn, partner)
        seen_round[k] = seen_round.get(k, 0) + 1
        recs.append(dict(dataset=ds, fly=fly, fly_number=fn, role="post", round=seen_round[k],
                         odor=o1, partner=partner, label=r["trial_label"],
                         score=scores.get((ds, fly, fn, r["trial_label"])),
                         onset_s=float(r.get("trial_odor_on_s", np.nan)), fps=float(r["fps"]),
                         trace=None))
        pre = pre_by.get((ds, fly, fn, o1))
        if pre is not None and seen_round[k] == 1:
            recs.append(dict(dataset=ds, fly=fly, fly_number=fn, role="pre", round=0, odor=o1,
                             partner=partner, label=pre["trial_label"],
                             score=scores.get((ds, fly, fn, pre["trial_label"])),
                             onset_s=float(pre["trial_odor_on_s"]), fps=float(pre["fps"]),
                             trace=None))
    return pd.DataFrame(recs, columns=REC_COLS)


PRETEST_COLS = ["dataset", "fly", "fly_number", "odor", "label", "score"]


def pretest_records(pre_df: pd.DataFrame, scores: dict) -> pd.DataFrame:
    """One row per naive pre-block trial (each odor alone, once per fly), with its score."""
    rows = [dict(dataset=ds, fly=fly, fly_number=fn, odor=odor, label=r["trial_label"],
                 score=scores.get((ds, fly, fn, r["trial_label"])))
            for (ds, fly, fn, odor), r in _pretest_by(pre_df).items()]
    return pd.DataFrame(rows, columns=PRETEST_COLS)


def build_records(pre_df: pd.DataFrame, pair_df: pd.DataFrame, scores: dict,
                  odor2_onset) -> pd.DataFrame:
    """One record per scored presentation of the comparison odor.

    ``scores[(dataset, fly, fly_number, label)]`` -> model score.
    ``odor2_onset(row)`` -> odor 2 valve onset (s) of a pair row, or None.
    Columns: dataset, fly, fly_number, role ('pre'|'post'), round (0 pre, 1|2),
    odor (the odor whose response is shown), partner, label, score, onset_s, fps, trace.
    """
    recs = []
    pre_cols, pair_cols = _dir_cols(pre_df), _dir_cols(pair_df)
    pre_by = _pretest_by(pre_df)
    seen_round = {}
    for num, (o1, o2), r in _pair_rows(pair_df):
        ds, fly, fn = r["dataset"], r["fly"], r["fly_number"]
        partner = o2 if ds == FIRST else o1
        k = (ds, fly, fn, partner)
        seen_round[k] = seen_round.get(k, 0) + 1
        on2 = odor2_onset(r)
        if on2 is None:
            continue
        recs.append(dict(dataset=ds, fly=fly, fly_number=fn, role="post", round=seen_round[k],
                         odor=o2, partner=partner, label=r["trial_label"],
                         score=scores.get((ds, fly, fn, r["trial_label"] + op.ODOR2_SUFFIX)),
                         onset_s=float(on2), fps=float(r["fps"]),
                         trace=r[pair_cols].to_numpy(dtype=float)))
        pre = pre_by.get((ds, fly, fn, o2))
        if pre is not None and seen_round[k] == 1:
            recs.append(dict(dataset=ds, fly=fly, fly_number=fn, role="pre", round=0, odor=o2,
                             partner=partner, label=pre["trial_label"],
                             score=scores.get((ds, fly, fn, pre["trial_label"])),
                             onset_s=float(pre["trial_odor_on_s"]), fps=float(pre["fps"]),
                             trace=pre[pre_cols].to_numpy(dtype=float)))
    return pd.DataFrame(recs, columns=REC_COLS)


def aligned(rec) -> tuple[np.ndarray, np.ndarray]:
    """(t, y) of a record's trace over WINDOW_S around its onset (NaN outside the trace)."""
    fps = rec["fps"]
    i0 = int(round(rec["onset_s"] * fps))
    lo, hi = int(round(WINDOW_S[0] * fps)), int(round(WINDOW_S[1] * fps))
    idx = np.arange(i0 + lo, i0 + hi)
    tr = rec["trace"]
    y = np.full(idx.size, np.nan)
    ok = (idx >= 0) & (idx < tr.size)
    y[ok] = tr[idx[ok]]
    return (idx - i0) / fps, y


def _fmt(s):
    return "–" if s is None or pd.isna(s) else f"{int(s)}"


def _panel_setup(ax, title):
    ax.axvspan(0, ODOR_S, color="0.85", zorder=0)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlim(*WINDOW_S)
    ax.axvline(0, color="0.5", lw=0.6)


def _grid(n):
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), sharey=True,
                             squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


def _panel_title(ds, partner):
    if ds == FIRST:
        return f"{partner}: alone vs after IAA"
    if ds == SECOND:
        return f"IAA: alone vs after {partner}"
    return f"partner {partner}"


def per_fly(recs: pd.DataFrame, ds: str, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for (fly, fn), g in recs[recs.dataset == ds].groupby(["fly", "fly_number"]):
        partners = [p for p in ORDER if p in set(g.partner)]
        fig, axes = _grid(len(partners))
        for ax, partner in zip(axes, partners):
            _panel_setup(ax, _panel_title(ds, partner))
            for _, r in g[g.partner == partner].sort_values("round").iterrows():
                t, y = aligned(r)
                if r["role"] == "pre":
                    lab = f"{r['odor']} alone (pre-block)  score {_fmt(r['score'])}"
                    ax.plot(t, y, color=COLORS["pre"], lw=1.6, label=lab)
                else:
                    num = parse_label(r["label"])[1]
                    lab = f"in pair #{int(r['round'])} (testing_{num})  score {_fmt(r['score'])}"
                    ax.plot(t, y, color=COLORS[int(r["round"])], lw=1.2, label=lab)
            ax.legend(fontsize=7.5, loc="upper right", frameon=False)
        for ax in list(axes)[: len(partners)]:
            ax.set_xlabel("time from scored odor onset (s)")
        head = ("IAA first: each odor alone vs right after IAA" if ds == FIRST
                else "IAA second: IAA alone vs right after each odor")
        fig.suptitle(f"{head}\n{fly} fly {fn}   (shaded = scored odor, 10 s)", fontsize=13)
        fig.supylabel("proboscis extension (trace units)")
        fig.tight_layout()
        p = out_dir / f"{fly}_fly{fn}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths


def _mean_sem(rows):
    ys = np.array([aligned(r)[1] for _, r in rows.iterrows()])
    t = aligned(rows.iloc[0])[0]
    n = np.sum(np.isfinite(ys), axis=0)
    mean = np.nanmean(ys, axis=0)
    sem = np.nanstd(ys, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1)) if len(ys) > 1 else np.zeros_like(mean)
    return t, mean, sem


def _mean_score(rows):
    s = pd.to_numeric(rows["score"], errors="coerce").dropna()
    return (float(s.sum() / len(s)) if len(s) else float("nan")), len(s)


def dataset_mean(recs: pd.DataFrame, title: str, path: Path) -> Path:
    partners = [p for p in ORDER if p in set(recs.partner)]
    fig, axes = _grid(len(partners))
    for ax, partner in zip(axes, partners):
        g = recs[recs.partner == partner]
        dsets = sorted(set(g.dataset))
        _panel_setup(ax, _panel_title(dsets[0] if len(dsets) == 1 else "both", partner))
        for key, rows, color, name in (
                ("pre", g[g.role == "pre"], COLORS["pre"], "alone (pre-block)"),
                (1, g[(g.role == "post") & (g["round"] == 1)], COLORS[1], "in pair #1"),
                (2, g[(g.role == "post") & (g["round"] == 2)], COLORS[2], "in pair #2")):
            if rows.empty:
                continue
            t, m, se = _mean_sem(rows)
            ms, n = _mean_score(rows)
            ax.plot(t, m, color=color, lw=1.6 if key == "pre" else 1.2,
                    label=f"{name}  mean score {ms:.2f} (n={n} trials)")
            ax.fill_between(t, m - se, m + se, color=color, alpha=0.18, lw=0)
        ax.legend(fontsize=7.5, loc="upper right", frameon=False)
        ax.set_xlabel("time from scored odor onset (s)")
    fig.suptitle(title + "\nmean ± SEM over trials (pooled across flies); shaded = scored odor",
                 fontsize=13)
    fig.supylabel("proboscis extension (trace units)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ── score bars: mean score and % responding, ± SEM (user 2026-11-16, /dataviz) ──
#: Bar colors, checked with the dataviz palette validator (all-pairs, light surface):
#: CVD ΔE >= 9.2, normal-vision ΔE >= 27.2. "Alone" is a neutral reference on purpose
#: (it fails only the chroma floor); the two pairings keep the green/orange identity
#: of the trace figures. Aqua sits at 2.7:1 on white, so the legend and the CSV twin
#: carry identity and values too.
BAR_XLABEL = {FIRST: "Scored odor (alone, or right after IAA)",
              SECOND: "Odor presented right before IAA (scored odor: IAA)",
              "both": "Partner odor (IAA-first: the partner is scored; IAA-second: IAA is)"}
#: "pair" (both presentations pooled) is blue: vs the gray, CVD ΔE 21.3, normal 21.8, >= 3:1.
BAR_COLORS = {"pre": "#52514e", 1: "#1baf7a", 2: "#eb6834", "pair": "#2a78d6"}
BAR_LABELS = {"pre": "Alone (pre-block)", 1: "In pair, 1st presentation",
              2: "In pair, 2nd presentation",
              "pair": "In pair, 1st + 2nd presentation pooled"}
_N_WORDS = {"pre": "alone", 1: "1st in pair", 2: "2nd in pair", "pair": "in pair (2 per fly)"}
#: (value column, error column or None, y label). The % panel carries no error bars
#: (user 2026-11-16): an SEM of yes/no trials adds little to a share of 10-22 trials.
BAR_PANELS = (("mean", "sem", "Mean PER score"),
              ("pct", None, "Responding trials (%)\nscore \u2265 {threshold:g}"))
INK, INK2, RULE, GRID = "#0b0b0b", "#52514e", "#c3c2b7", "#e1e0d9"
_PUB_RC = {"font.family": "sans-serif",
           "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
           "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
           "axes.linewidth": 0.6, "font.size": 8}


def _bar_rows(g: pd.DataFrame, key) -> pd.DataFrame:
    if key == "pre":
        return g[g.role == "pre"]
    if key == "pair":                     # both presentations' trials, pooled
        return g[g.role == "post"]
    return g[(g.role == "post") & (g["round"] == key)]


def bar_stats(rows: pd.DataFrame, threshold: float) -> dict:
    """Mean score and % responding (score >= threshold) of pooled trials, each ± SEM
    over trials (sample SD / sqrt n). Each fly gives one trial per bar, so n trials is
    also n flies."""
    sc = pd.to_numeric(rows["score"], errors="coerce")
    s = sc.dropna().to_numpy(dtype=float)
    n = len(s)
    if not n:
        return {"n": 0, "mean": np.nan, "sem": np.nan, "pct": np.nan, "pct_sem": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan}
    r = (s >= threshold).astype(float)
    sem = (lambda v: float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else 0.0)
    # The drawn interval: 95 % CI with flies as the unit (fly_stats; audit 2026-11-17).
    # A pooled bar holds 2 trials per fly; one trial per fly gives the plain t interval.
    ci_lo = ci_hi = np.nan
    if n > 1 and {"fly", "fly_number"} <= set(rows.columns):
        ok = sc.notna().to_numpy()
        ds = rows["dataset"].astype(str) if "dataset" in rows.columns else ""
        ids = (ds + "|" + rows["fly"].astype(str) + "|" + rows["fly_number"].astype(str))
        _, ci_lo, ci_hi = fs.fly_cluster_mean_ci(s, ids.to_numpy()[ok])
    return {"n": n, "mean": float(s.sum() / n), "sem": sem(s),
            "pct": 100.0 * float(r.sum() / n), "pct_sem": 100.0 * sem(r),
            "ci_lo": ci_lo, "ci_hi": ci_hi}


def n_note(ns) -> str | None:
    """The one n every bar shares, for the corner; None when the bars differ."""
    ns = {int(n) for n in ns}
    return f"n = {ns.pop()} trials per bar" if len(ns) == 1 else None


def corner_n(t: pd.DataFrame, words: dict | None = None) -> str | None:
    """The corner's n: one for every bar, else one per condition when each condition's
    bars share it ("n = 12 trials alone, 24 in pair (2 per fly)"); None otherwise."""
    t = t[t["n"] > 0]
    one = n_note(t["n"])
    if one or t.empty:
        return one
    per = {}
    for key in pd.unique(t["key"]):
        ns = set(t.loc[t["key"] == key, "n"].astype(int))
        if len(ns) != 1:
            return None
        per[key] = ns.pop()
    names = {**_N_WORDS, **(words or {})}
    words = [f"{n} {names.get(k, k)}" for k, n in per.items()]
    return "n = " + words[0].replace(" ", " trials ", 1) + "".join(f", {w}" for w in words[1:])


# ── significance: exact paired sign-flip test across flies (user 2026-11-16) ──
#: Bars compared per odor: the 3-bar figure every pair, the pooled figure alone vs pair.
COMPARISONS = {False: (("pre", 1), ("pre", 2), (1, 2)), True: (("pre", "pair"),)}
STATS_NOTE = ("Secondary: alone vs in the test block is NOT controlled for time in the rig or "
              "re-testing (controlled: 00_Primary-Endpoint, *_change_minus_paraffin). "
              "Exploratory: * p < 0.05, ** p < 0.01, *** p < 0.001, exact paired sign-flip on "
              "per-fly differences, Holm-adjusted over the {n} tests of each panel; unmarked "
              "pairs are not significant. A whiskers: 95 % CI with flies as the unit.")


#: Moved verbatim to b2b_stats (shared with the batch-level tests).
sign_flip_p = bs.sign_flip_p


def holm(ps) -> list[float]:
    """Holm step-down adjusted p values (NaN stays NaN and is not counted):
    ``fly_stats.holm_adjust``."""
    return fs.holm_adjust(list(ps)).tolist()


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _fly_values(g: pd.DataFrame, key, threshold: float, panel: str) -> pd.Series:
    """One value per fly for a bar: its score (or responded 0/1); for the pooled pair
    bar, the mean over the fly's 2 trials, so each fly counts once in the test."""
    r = _bar_rows(g, key)
    s = pd.to_numeric(r["score"], errors="coerce")
    r, s = r[s.notna()], s[s.notna()]
    v = s if panel == "score" else (s >= threshold).astype(float)
    return v.groupby([r["dataset"], r["fly"], r["fly_number"]]).mean()


def paired_stats(recs: pd.DataFrame, partners, threshold: float,
                 pooled: bool, labels: dict | None = None) -> pd.DataFrame:
    """Every compared pair of bars, both panels: flies with both bars, mean per-fly
    difference (b - a; percentage points on the % panel), exact p, Holm-adjusted p
    within the panel, stars."""
    lab = {**BAR_LABELS, **(labels or {})}
    rows = []
    for panel in ("score", "pct"):
        for partner in partners:
            g = recs[recs.partner == partner]
            for ka, kb in COMPARISONS[pooled]:
                va, vb = (_fly_values(g, k, threshold, panel) for k in (ka, kb))
                both = va.index.intersection(vb.index)
                d = (vb[both] - va[both]).to_numpy(dtype=float)
                scale = 100.0 if panel == "pct" else 1.0
                rows.append({"panel": panel, "partner": partner, "a": lab[ka],
                             "b": lab[kb], "key_a": ka, "key_b": kb,
                             "n_flies": len(d),
                             "mean_diff": scale * float(d.mean()) if len(d) else np.nan,
                             "p": sign_flip_p(d) if len(d) else np.nan})
    st = pd.DataFrame(rows)
    st["p_holm"] = np.nan
    for panel in ("score", "pct"):
        m = st["panel"] == panel
        st.loc[m, "p_holm"] = holm(st.loc[m, "p"])
    st["stars"] = st["p_holm"].map(stars)
    return st


def _brackets(ax, st: pd.DataFrame, xpos: dict, tops: dict, unit: float) -> float:
    """Significance brackets over each odor's bars, stacked; the highest y drawn."""
    ymax = -np.inf
    for i, g in st[st["stars"] != ""].groupby("partner_i"):
        g = g.assign(span=[abs(xpos[(i, b)] - xpos[(i, a)]) for a, b in zip(g.key_a, g.key_b)])
        for level, (_, r) in enumerate(g.sort_values("span", kind="stable").iterrows()):
            y = tops[i] + unit * (0.7 + 1.5 * level)
            xa, xb = xpos[(i, r.key_a)], xpos[(i, r.key_b)]
            ax.plot([xa, xa, xb, xb], [y - 0.35 * unit, y, y, y - 0.35 * unit], color=INK,
                    lw=0.6, solid_capstyle="butt", zorder=4)
            ax.text((xa + xb) / 2, y - 0.15 * unit, r.stars, ha="center", va="bottom",
                    fontsize=9, color=INK, zorder=4)
            ymax = max(ymax, y + 1.1 * unit)
    return ymax


def _style_axis(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=RULE, labelcolor=INK2, length=2.5, width=0.6, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def score_bars(recs: pd.DataFrame, title: str, path: Path, *, threshold: float,
               xlabel: str = "", pool_presentations: bool = False,
               labels: dict | None = None, n_words: dict | None = None,
               started: str = "", colors: dict | None = None) -> list[Path]:
    """Two stacked panels over the same odors: (A) mean model score ± SEM over trials,
    (B) % of trials with a response (score >= threshold), no error bars. One n in the
    top-right corner when every bar shares it, else a small n over each bar. Writes the
    PNG (300 dpi), a vector PDF and the plotted values as CSV beside it.

    ``pool_presentations``: one "in pair" bar per odor holding the trials of both
    presentations (sum / all trials, never the mean of the two means), beside "alone".
    ``labels`` / ``n_words``: legend names and n-note words replacing ``BAR_LABELS`` /
    ``_N_WORDS`` for this figure. ``started`` ("flies started 9:40am-4:20pm") ends the
    subtitle. ``colors``: bar colors replacing ``BAR_COLORS``."""
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, MaxNLocator

    partners = [p for p in ORDER if p in set(recs.partner)]
    keys = ("pre", "pair") if pool_presentations else ("pre", 1, 2)
    lab = {**BAR_LABELS, **(labels or {})}
    col_of = {**BAR_COLORS, **(colors or {})}
    table = []
    for partner in partners:
        g = recs[recs.partner == partner]
        for key in keys:
            st = bar_stats(_bar_rows(g, key), threshold)
            table.append({"partner": partner, "condition": lab[key], "key": key, **st})
    t = pd.DataFrame(table)
    note = corner_n(t, n_words)
    st = paired_stats(recs, partners, threshold, pool_presentations, labels)
    st["partner_i"] = st["partner"].map({p: i for i, p in enumerate(partners)})

    w, step = (0.36, 0.37) if pool_presentations else (0.25, 0.26)
    x0 = np.arange(len(partners), dtype=float)
    with plt.rc_context(_PUB_RC):
        fig = plt.figure(figsize=(7.2, 5.6))
        gs = fig.add_gridspec(2, 1, left=0.1, right=0.985, top=0.86, bottom=0.19, hspace=0.18)
        axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
        xpos, bracket_top = {}, []
        for ax, (col, err, ylab) in zip(axes, BAR_PANELS):
            ylab = ylab.format(threshold=threshold)
            _style_axis(ax)
            for j, key in enumerate(keys):
                d = t[t.key.astype(str) == str(key)]
                x = x0 + (j - (len(keys) - 1) / 2) * step
                xpos.update({(i, key): xi for i, xi in enumerate(x)})
                ax.bar(x, d[col], w, color=col_of[key], edgecolor="white",
                       linewidth=0.6, zorder=2)
                if err is not None:                       # 95 % CI, flies as the unit
                    elo = (d[col] - d["ci_lo"]).to_numpy(dtype=float)
                    ehi = (d["ci_hi"] - d[col]).to_numpy(dtype=float)
                    some = np.isfinite(elo) & np.isfinite(ehi) & ((elo > 0) | (ehi > 0))
                    ax.errorbar(x[some], d[col][some], yerr=[elo[some], ehi[some]], fmt="none",
                                ecolor=INK, elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=3)
                for xi in x[d[col].to_numpy(dtype=float) == 0]:   # a zero bar stays visible
                    ax.plot([xi - w / 2, xi + w / 2], [0, 0], color=col_of[key], lw=2.2,
                            solid_capstyle="butt", zorder=2.6)
                if note is None:
                    ends = d["ci_hi"] if err is not None else d[col]
                    for xi, v, e, n in zip(x, d[col], ends, d["n"]):
                        if np.isfinite(v):
                            ax.text(xi, e if np.isfinite(e) else v, f"{n}", ha="center",
                                    va="bottom", fontsize=6, color=INK2)
            ax.set_ylabel(ylab, color=INK, fontsize=8.5)
            ax.set_xlim(-0.6, len(partners) - 0.4)
            ax.set_xticks(x0)
            ax.axhline(0, color=RULE, linewidth=0.6, zorder=2.5)
        # A: score axis from the data (scores run -1..5); B: a share, 0-100 %.
        # Each panel then grows to fit its significance brackets.
        top = t.assign(a=t["ci_hi"].fillna(t["mean"]), b=t["pct"]).groupby("partner")
        lo = float(np.nanmin(t["ci_lo"].fillna(t["mean"]))) if len(t) else 0.0
        hi = float(np.nanmax(t["ci_hi"].fillna(t["mean"]))) if len(t) else 1.0
        span = max(hi - min(lo, 0.0), 0.5)
        for ax, panel, col, unit, base in ((axes[0], "score", "a", 0.06 * span, hi + 0.08 * span),
                                            (axes[1], "pct", "b", 6.0, 100.0)):
            tops = {i: max(float(np.nanmax(top.get_group(p)[col])), 0.0)
                    for i, p in enumerate(partners)}
            ymax = _brackets(ax, st[st["panel"] == panel], xpos, tops, unit)
            bracket_top.append(max(base, ymax))
        axes[0].set_ylim(min(lo, 0.0) - (0.06 * span if lo < 0 else 0), bracket_top[0])
        axes[0].yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        axes[0].spines["bottom"].set_visible(False)       # the zero rule is the baseline
        axes[0].tick_params(axis="x", length=0, labelbottom=False)
        axes[1].set_ylim(0, bracket_top[1])
        axes[1].yaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
        axes[1].set_xticklabels([p.replace(" ", "\n") for p in partners], color=INK, fontsize=8)
        if xlabel:
            axes[1].set_xlabel(xlabel, color=INK, fontsize=8.5)
        fig.align_ylabels(axes)
        for ax, letter in zip(axes, "AB"):
            ax.text(-0.085, 1.02, letter, transform=ax.transAxes, fontsize=11,
                    fontweight="bold", color=INK, va="bottom")
        fig.text(0.1, 0.975, title, fontsize=10, fontweight="bold", color=INK, va="top")
        fig.legend(handles=[Patch(facecolor=col_of[k], label=lab[k]) for k in keys],
                   loc="lower left", bbox_to_anchor=(0.1, 0.875), ncol=len(keys), frameon=False,
                   fontsize=8, handlelength=1.1, handleheight=0.9, columnspacing=1.6,
                   borderaxespad=0, labelcolor=INK)
        corner = (f"{note}  \u00b7  A: mean, 95 % CI (flies)" if note
                  else "n per bar above it  \u00b7  A: mean, 95 % CI (flies)")
        corner += f"  \u00b7  {started}" if started else ""
        fig.text(0.1, 0.937, corner, ha="left", va="top", fontsize=8, color=INK2)  # subtitle
        fig.text(0.1, 0.012, STATS_NOTE.format(n=len(partners) * len(COMPARISONS[
                     pool_presentations])), ha="left", va="bottom", fontsize=6.5, color=INK2,
                 wrap=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        pdf = path.with_suffix(".pdf")
        fig.savefig(path, dpi=300, facecolor="white")
        fig.savefig(pdf, facecolor="white")
        plt.close(fig)
    csv = path.with_suffix(".csv")
    t.drop(columns=["key", "pct_sem"]).rename(columns={
        "ci_lo": "score_ci95_lo_flies", "ci_hi": "score_ci95_hi_flies",
        "mean": "mean_score", "sem": "sem_score", "pct": "pct_responding",
        "pct_sem": "sem_pct"}).assign(threshold=threshold).to_csv(csv, index=False)
    stats_csv = path.with_name(f"{path.stem}_stats.csv")
    st.drop(columns=["key_a", "key_b", "partner_i"]).assign(
        stars=st["stars"].replace("", "ns"),
        diff_unit=np.where(st["panel"] == "pct", "percentage points", "score"),
        test="exact paired sign-flip, two-sided; Holm within panel").to_csv(stats_csv, index=False)
    return [path, pdf, csv, stats_csv]


# ── each odor's change minus the Paraffin Oil control's change (user 2026-11-18) ──
#: Paraffin Oil follows IAA exactly as every other odor does, so its change from alone
#: to in-pair is what time in the rig, starvation and IAA just before give any odor.
#: An odor's change minus that change, in the same fly, is what the odor itself adds.
CONTROL_ODOR = "Paraffin Oil"
CONTROL_KINDS = ("odor minus control", "control's own change")
CONTROL_HEAD = {FIRST: "IAA first: each odor's change after IAA, minus Paraffin Oil's change",
                SECOND: "IAA second: IAA's change after each odor, minus its change after "
                        "Paraffin Oil"}
#: What a solid bar is, and what the hatched control bar is, per group.
CONTROL_TEXT = {
    FIRST: ("each odor's change (alone → right after IAA) minus Paraffin Oil's change "
            "in the same flies", "Paraffin Oil's own change (control)"),
    SECOND: ("IAA's change (alone → right after the odor) minus IAA's change after "
             "Paraffin Oil in the same flies", "IAA's change after Paraffin Oil (control)")}
CONTROL_LABELS = {"pair": "1st + 2nd presentation pooled, minus control",
                  1: "1st presentation, minus control", 2: "2nd presentation, minus control"}
CONTROL_PANELS = (("score", "Change in mean PER score"),
                  ("pct", "Change in responding\ntrials (% points)"))
_PRESENTATION = {"pair": "1st + 2nd pooled", 1: "1st", 2: "2nd"}


def _trial_values(g: pd.DataFrame, key, threshold: float, panel: str) -> pd.Series:
    """A bar's scored trials (score, or responded 0/1), indexed by fly."""
    r = _bar_rows(g, key)
    s = pd.to_numeric(r["score"], errors="coerce")
    r, s = r[s.notna()], s[s.notna()]
    v = s if panel == "score" else (s >= threshold).astype(float)
    return pd.Series(v.to_numpy(dtype=float), index=pd.MultiIndex.from_arrays(
        [r["dataset"].to_numpy(), r["fly"].to_numpy(), r["fly_number"].to_numpy()]))


def _change(g: pd.DataFrame, key, threshold: float, panel: str, flies=None):
    """(pooled change, per-fly change) from alone to ``key``, over the flies with both
    (and in ``flies`` when given). Pooled: trial sum / trial count of each side, never
    the mean of per-fly means; per fly: the mean of its ``key`` trials minus alone."""
    post, pre = (_trial_values(g, k, threshold, panel) for k in (key, "pre"))
    both = post.index.unique().intersection(pre.index.unique())
    if flies is not None:
        both = both.intersection(flies)
    post, pre = post[post.index.isin(both)], pre[pre.index.isin(both)]
    levels = [0, 1, 2]
    per_fly = post.groupby(level=levels).mean() - pre.groupby(level=levels).mean()
    pooled = float(post.mean() - pre.mean()) if len(both) else np.nan
    return pooled, per_fly


def control_change_table(recs: pd.DataFrame, keys, threshold: float,
                         control: str = CONTROL_ODOR) -> pd.DataFrame:
    """Per panel ('score', 'pct') and presentation key: each odor's change minus the
    control's change over the flies with both, and the control's own change.

    ``value`` is the plotted bar (pooled trials; % panel in percentage points). Tests
    take one value per fly: ``p_batch`` (headline; every fly of a batch flipped
    together), ``ci_lo``/``ci_hi`` (95 % CI inverting it; NaN when too few batches can
    reach 0.05) and ``p`` (flies flipped one by one), with ``p_batch_holm`` / ``p_holm``
    Holm across every bar of the panel; ``stars`` from ``p_batch_holm``. Empty without
    the control odor."""
    ctrl = recs[recs.partner == control]
    if ctrl.empty:
        return pd.DataFrame()
    odors = [p for p in ORDER if p in set(recs.partner) and p != control]
    rows = []
    for panel in ("score", "pct"):
        scale = 100.0 if panel == "pct" else 1.0

        def row(key, partner, kind, d_fly, odor_c, ctrl_c, value):
            d = d_fly.to_numpy(dtype=float)
            n = len(d)
            sem = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else (0.0 if n else np.nan)
            batch = (bs.batch_sign_flip(d, d_fly.index.get_level_values(1), ci=True) if n
                     else {"p": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n_batches": 0})
            return {"panel": panel, "key": key, "partner": partner, "kind": kind,
                    "n_flies": n, "n_batches": batch["n_batches"],
                    "odor_change": scale * odor_c,
                    "control_change": scale * ctrl_c, "value": scale * value,
                    "sem": scale * sem if panel == "score" else np.nan,
                    "ci_lo": scale * batch["ci_lo"], "ci_hi": scale * batch["ci_hi"],
                    "mean_per_fly": scale * float(d.mean()) if n else np.nan,
                    "p_batch": batch["p"], "p": sign_flip_p(d) if n else np.nan}

        for key in keys:
            for partner in odors:
                g = recs[recs.partner == partner]
                _, o_fly = _change(g, key, threshold, panel)
                _, c_fly = _change(ctrl, key, threshold, panel)
                flies = o_fly.index.intersection(c_fly.index)
                o_pooled, o_fly = _change(g, key, threshold, panel, flies)
                c_pooled, c_fly = _change(ctrl, key, threshold, panel, flies)
                rows.append(row(key, partner, CONTROL_KINDS[0], o_fly - c_fly,
                                o_pooled, c_pooled, o_pooled - c_pooled))
            c_pooled, c_fly = _change(ctrl, key, threshold, panel)
            rows.append(row(key, control, CONTROL_KINDS[1], c_fly, np.nan,
                            c_pooled, c_pooled))
    t = pd.DataFrame(rows)
    t["p_holm"], t["p_batch_holm"] = np.nan, np.nan
    for panel in ("score", "pct"):
        m = t["panel"] == panel
        t.loc[m, "p_holm"] = holm(t.loc[m, "p"])
        t.loc[m, "p_batch_holm"] = holm(t.loc[m, "p_batch"])
    t["stars"] = t["p_batch_holm"].map(stars)
    return t


def control_bars(recs: pd.DataFrame, title: str, path: Path, *, threshold: float,
                 xlabel: str = "", pool_presentations: bool = False, started: str = "",
                 group: str = FIRST, colors: dict | None = None,
                 labels: dict | None = None) -> list[Path]:
    """(A) mean score, (B) % responding: per odor, its change from alone to in-pair minus
    the Paraffin Oil control's change in the same flies (solid), and the control's own
    change (hatched, set apart on the right). One solid bar per odor pooled, else one per
    presentation, each against the same presentation of the control. Writes the PNG
    (300 dpi), a vector PDF and a CSV of every bar with its test; [] without the control."""
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    keys = ("pair",) if pool_presentations else (1, 2)
    t = control_change_table(recs, keys, threshold)
    if t.empty:
        return []
    odors = [p for p in ORDER if p in set(t.partner) and p != CONTROL_ODOR]
    lab = {**CONTROL_LABELS, **(labels or {})}
    col_of = {**BAR_COLORS, **(colors or {})}
    solid_note, control_label = CONTROL_TEXT.get(group, CONTROL_TEXT[FIRST])
    w, step = (0.55, 0.0) if len(keys) == 1 else (0.36, 0.37)
    xc = {p: float(i) for i, p in enumerate(odors)}
    xc[CONTROL_ODOR] = len(odors) + 0.35                 # set apart from the odors
    ns = set(t["n_flies"].astype(int))
    tick_n = len(ns) != 1
    hatch = "////"
    with plt.rc_context({**_PUB_RC, "hatch.linewidth": 0.6}):
        fig = plt.figure(figsize=(7.2, 5.6))
        gs = fig.add_gridspec(2, 1, left=0.1, right=0.985, top=0.865, bottom=0.2, hspace=0.18)
        axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
        for ax, (panel, ylab) in zip(axes, CONTROL_PANELS):
            _style_axis(ax)
            d = t[t.panel == panel]
            lo, hi = 0.0, 0.0
            for j, key in enumerate(keys):
                off = (j - (len(keys) - 1) / 2) * step
                for r in d[d.key.astype(str) == str(key)].itertuples():
                    v = float(r.value)
                    if not np.isfinite(v):
                        continue
                    x = xc[r.partner] + off
                    ctl = r.kind == CONTROL_KINDS[1]
                    c = col_of[key]
                    ax.bar(x, v, w, color="white" if ctl else c, edgecolor=c if ctl else "white",
                           hatch=hatch if ctl else None, linewidth=0.8 if ctl else 0.6, zorder=2)
                    if v == 0:                                   # a zero bar stays visible
                        ax.plot([x - w / 2, x + w / 2], [0, 0], color=c, lw=2.2,
                                solid_capstyle="butt", zorder=2.6)
                    # A: 95 % CI inverting the batch-level test; B: bars only (user 2026-11-16)
                    has_ci = panel == "score" and np.isfinite(r.ci_lo) and np.isfinite(r.ci_hi)
                    if has_ci:
                        ax.errorbar([x], [v], yerr=[[max(v - r.ci_lo, 0)], [max(r.ci_hi - v, 0)]],
                                    fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.6,
                                    capthick=0.7, zorder=3)
                    end = (r.ci_hi if v >= 0 else r.ci_lo) if has_ci else v
                    lo, hi = min(lo, end), max(hi, end)
                    label = f"{r.stars}\n{_p_short(r.p_batch)}" if r.stars else _p_short(r.p_batch)
                    ax.annotate(label, (x, end), xytext=(0, 2 if v >= 0 else -2),
                                textcoords="offset points", ha="center",
                                va="bottom" if v >= 0 else "top", fontsize=5.5, color=INK,
                                zorder=4, linespacing=1.0,
                                fontweight="bold" if r.stars else "normal")
            span = max(hi - lo, 0.5 if panel == "score" else 10.0)
            ax.set_ylim(lo - (0.16 * span if lo < 0 else 0.04 * span), hi + 0.16 * span)
            n_tests = int(np.isfinite(d["p_batch"]).sum())
            ax.set_title(f"exploratory · Holm-adjusted over {n_tests} tests", loc="right",
                         fontsize=6.5, color=INK2, pad=2)
            ax.axhline(0, color=INK2, linewidth=0.7, zorder=2.5)
            ax.axvline((len(odors) - 1 + xc[CONTROL_ODOR]) / 2, color=RULE, linewidth=0.6,
                       linestyle=(0, (2, 2)), zorder=1)
            ax.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
            ax.set_ylabel(ylab, color=INK, fontsize=8.5)
            ax.set_xlim(-0.6, xc[CONTROL_ODOR] + 0.6)
            ax.set_xticks([xc[p] for p in odors + [CONTROL_ODOR]])
        axes[0].tick_params(axis="x", length=0, labelbottom=False)
        n_of = t[t.panel == "score"].groupby("partner")["n_flies"].max()
        ticks = [p.replace(" ", "\n") + (f"\nn={int(n_of[p])}" if tick_n else "") for p in odors]
        ticks.append(CONTROL_ODOR + "\n(control)"
                     + (f", n={int(n_of[CONTROL_ODOR])}" if tick_n else ""))
        axes[1].set_xticklabels(ticks, color=INK, fontsize=8)
        if xlabel:
            axes[1].set_xlabel(xlabel, color=INK, fontsize=8.5)
        fig.align_ylabels(axes)
        for ax, letter in zip(axes, "AB"):
            ax.text(-0.085, 1.02, letter, transform=ax.transAxes, fontsize=11,
                    fontweight="bold", color=INK, va="bottom")
        fig.text(0.1, 0.975, title, fontsize=10, fontweight="bold", color=INK, va="top")
        handles = [Patch(facecolor=col_of[k], label=lab[k]) for k in keys]
        handles.append(Patch(facecolor="white", edgecolor=INK2, hatch=hatch, label=control_label))
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.1, 0.88),
                   ncol=len(handles), frameon=False, fontsize=8, handlelength=1.1,
                   handleheight=0.9, columnspacing=1.6, borderaxespad=0, labelcolor=INK)
        corner = (f"n = {ns.pop()} flies" if not tick_n else "n flies under each odor")
        corner += "  ·  A: 95 % CI, whole batches flipped"
        corner += f"  ·  {started}" if started else ""
        fig.text(0.1, 0.94, corner, ha="left", va="top", fontsize=8, color=INK2)
        fig.text(0.1, 0.012,
                 f"Solid: {solid_note}; above 0: rose more than with the control. "
                 "Bars pool trials. Number at each bar: its raw p, sign-flip with every fly of "
                 "a batch flipped together; * p < 0.05, ** p < 0.01, *** p < 0.001 only after "
                 "Holm over every bar of the panel. A whiskers: 95 % CI inverting that test "
                 "(clears 0 exactly when the raw p < 0.05; none when too few batches). "
                 "Fly-level p in the CSV.", ha="left", va="bottom", fontsize=6.5, color=INK2,
                 wrap=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        pdf = path.with_suffix(".pdf")
        fig.savefig(path, dpi=300, facecolor="white")
        fig.savefig(pdf, facecolor="white")
        plt.close(fig)
    csv = path.with_suffix(".csv")
    t.assign(presentation=t["key"].map(lambda k: _PRESENTATION.get(k, k)),
             stars=t["stars"].replace("", "ns"),
             unit=np.where(t["panel"] == "pct", "percentage points", "score"),
             threshold=threshold, control=CONTROL_ODOR,
             test="exact sign-flip of per-fly values vs 0, two-sided; Holm within panel"
             ).drop(columns=["key"]).to_csv(csv, index=False)
    return [path, pdf, csv]


# ── pre-specified primary endpoint (user 2026-11-18, after the stats audit) ──────
#: Registered 2026-11-18, pinned by a test: changing any of it is a new registration.
#: The naive pre-block runs ~5-35 min into a session and the pair block ~65-140 min, so
#: any "alone vs in pair" change also holds time in the rig, starvation and re-testing.
#: The click partner sits in the same odor-2 slot at the same time, so subtracting its
#: change leaves what the odors add. Everything else in the folder is exploratory.
PRIMARY_ENDPOINT = {
    "registered": "2026-11-18",
    "metric": "responded (score >= 2), % of trials",
    "window": "production scorer window (10 s odor; model_predictions.csv)",
    "partners": "the 7 odor partners: IAA, Hexanol, Linalool, Benzaldehyde, "
                "Ethyl Butyrate, Citral, 3-Octanol",
    "control": "the Paraffin Oil partner: a master-valve click in the same odor-2 slot",
    "IAA-first": "per fly: mean over the 7 partners of (partner in pair - partner naive) "
                 "minus (click in pair - click naive)",
    "IAA-second": "per fly: mean over the 7 partners of (IAA after the partner - IAA "
                  "naive) minus (IAA after the click - IAA naive)",
    "contrasts": "naive -> round 1; naive -> round 2; round 1 -> round 2; "
                 "naive -> rounds 1 + 2 pooled",
    "test": "two-sided sign-flip, every fly of a batch flipped together (headline), and "
            "across flies; Holm over the 4 contrasts of each dataset",
    "ci": "95 % CI inverting the batch-level sign-flip",
}
PRIMARY_PARTNERS = ("IAA", "Hexanol", "Linalool", "Benzaldehyde", "Ethyl Butyrate",
                    "Citral", "3-Octanol")
PRIMARY_DIR = "00_Primary-Endpoint"
PRIMARY_CSV = "primary_endpoint.csv"
#: (name, from, to): "from"/"to" are rounds (0 = naive); "pooled" = rounds 1 + 2.
PRIMARY_CONTRASTS = (("naive -> round 1", 0, 1), ("naive -> round 2", 0, 2),
                     ("round 1 -> round 2", 1, 2), ("naive -> rounds 1 + 2 pooled", 0, "pooled"))
PRIMARY_MAIN, PRIMARY_SENS = "main", "odor-1 responders excluded"
#: Within-fly check: which partner plays the click, shuffled per fly (Monte Carlo).
LABEL_PERMS = 20_000
_PRIMARY_GROUP_WORDS = {FIRST: ("7 odor partners\n(right after IAA)", "Click\n(right after IAA)"),
                        SECOND: ("IAA right after\nthe 7 odor partners", "IAA right after\nthe click")}


def _primary_cells(recs: pd.DataFrame, threshold: float, odor1_scores=None,
                   drop_odor1_responders: bool = False) -> pd.DataFrame:
    """Responded (0/1) per (dataset, fly, fly_number, partner) and round (0 naive, 1, 2):
    the primary partners and the click only. ``drop_odor1_responders``: pair trials
    whose odor 1 crossed the threshold (``odor1_scores[(dataset, fly, fly_number,
    label)]``) are left out (the carry-over sensitivity analysis)."""
    r = recs[recs["partner"].isin(PRIMARY_PARTNERS + (CONTROL_ODOR,))]
    s = pd.to_numeric(r["score"], errors="coerce")
    r = r[s.notna()].assign(resp=(s[s.notna()] >= threshold).astype(float))
    if drop_odor1_responders:
        o1 = np.array([np.nan if (odor1_scores or {}).get(k) is None
                       else float((odor1_scores or {})[k])
                       for k in zip(r["dataset"], r["fly"], r["fly_number"], r["label"])])
        r = r[~((r["role"].to_numpy() == "post") & (o1 >= threshold))]
    if r.empty:
        return pd.DataFrame(columns=[0, 1, 2])
    cells = r.pivot_table(index=["dataset", "fly", "fly_number", "partner"], columns="round",
                          values="resp", aggfunc="mean")
    return cells.reindex(columns=[0, 1, 2])


def _sides(cells: pd.DataFrame, a, b) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(post trials, pre trials) of each cell for a contrast, NaN where missing."""
    post = cells[[1, 2]] if b == "pooled" else cells[[b]]
    return post, cells[[a]]


def _within_fly_label_p(per_fly: list[np.ndarray], obs: float) -> float:
    """Shuffle which partner plays the click within each fly (the last entry of each
    array is the click's contrast): two-sided Monte Carlo p of the summed per-fly
    value (odor mean - control)."""
    rng = np.random.default_rng(bs.SEED)
    total = np.zeros(LABEL_PERMS)
    for c in per_fly:
        m = c.size
        options = (c.sum() - c) / (m - 1) - c          # each partner as the control
        total += options[rng.integers(0, m, LABEL_PERMS)]
    hits = int(np.sum(np.abs(total) >= abs(obs) - 1e-9))
    return (hits + 1) / (LABEL_PERMS + 1)


def primary_table(recs: pd.DataFrame, threshold: float, odor1_scores=None,
                  drop_odor1_responders: bool = False) -> pd.DataFrame:
    """The registered contrasts (``PRIMARY_ENDPOINT``), one row per dataset x contrast.

    Per cell (fly x partner) the contrast's change (post mean - pre); per fly, the mean
    over its odor partners minus its click's. ``estimate`` / ``odor_change`` /
    ``click_change`` pool trials over the flies tested (sum / trials, x 100: % points);
    ``p_batch`` (headline) flips whole batches, ``p_fly`` flies; Holm over the dataset's
    4 contrasts. ``p_within_fly_perm`` (a check for the unequal trial counts of the
    pooled and IAA-second contrasts): which partner is the click, shuffled per fly."""
    cells = _primary_cells(recs, threshold, odor1_scores, drop_odor1_responders)
    analysis = PRIMARY_SENS if drop_odor1_responders else PRIMARY_MAIN
    rows = []
    if cells.empty:
        return pd.DataFrame(rows)
    for ds in [d for d in DATASETS if d in set(cells.index.get_level_values(0))]:
        c_ds = cells.xs(ds, level=0, drop_level=False)
        ds_rows = []
        for name, a, b in PRIMARY_CONTRASTS:
            post, pre = _sides(c_ds, a, b)
            change = post.mean(axis=1, skipna=True) - pre.iloc[:, 0]
            ok = change.notna() & post.notna().any(axis=1)
            change = change[ok]
            is_click = change.index.get_level_values("partner") == CONTROL_ODOR
            fly_key = change.index.droplevel("partner")
            click = change[is_click].droplevel("partner")
            odor = change[~is_click]
            odor_mean = odor.groupby(level=[0, 1, 2]).mean()
            flies = odor_mean.index.intersection(click.index)
            v = (odor_mean[flies] - click[flies]).to_numpy(dtype=float)
            in_flies = fly_key.isin(flies)
            odor_cells = ~is_click & in_flies
            click_cells = is_click & in_flies
            sub_post, sub_pre = post.loc[ok], pre.loc[ok]

            def pooled(mask):
                pv = sub_post[mask].to_numpy(dtype=float)
                qv = sub_pre[mask].to_numpy(dtype=float)
                pv, qv = pv[np.isfinite(pv)], qv[np.isfinite(qv)]
                return 100.0 * (pv.mean() - qv.mean()) if pv.size and qv.size else np.nan
            odor_c, click_c = pooled(odor_cells), pooled(click_cells)
            batches = [f for _, f, _ in flies]
            bt_r = bs.batch_sign_flip(v, batches, ci=True) if len(v) else {}
            per_fly = []
            for fk in flies:
                oc = odor[odor.index.droplevel("partner") == fk].to_numpy(dtype=float)
                per_fly.append(np.append(oc, click[fk]))
            ds_rows.append({
                "dataset": ds, "group": GROUP_SHORT[ds], "analysis": analysis,
                "contrast": name, "n_flies": int(len(v)),
                "n_batches": int(bt_r.get("n_batches", 0)),
                "odor_change": odor_c, "click_change": click_c,
                "estimate": odor_c - click_c,
                "ci_lo": 100.0 * bt_r.get("ci_lo", np.nan),
                "ci_hi": 100.0 * bt_r.get("ci_hi", np.nan),
                "p_batch": bt_r.get("p", np.nan), "batch_exact": bt_r.get("exact", False),
                "p_fly": sign_flip_p(v) if len(v) else np.nan,
                "p_within_fly_perm": (_within_fly_label_p(per_fly, float(v.sum()))
                                      if len(v) else np.nan)})
        for col in ("p_batch", "p_fly"):
            adj = holm([r[col] for r in ds_rows])
            for r, h in zip(ds_rows, adj):
                r[f"{col}_holm"] = h
        rows += ds_rows
    return pd.DataFrame(rows)


def _p_short(p: float) -> str:
    """'p=0.13', 'p=0.004', 'p<0.001' for a bar label."""
    if not np.isfinite(p):
        return ""
    return "p<0.001" if p < 0.001 else f"p={p:.3f}" if p < 0.01 else f"p={p:.2f}"


def _p_text(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def primary_figure(cells: pd.DataFrame, table: pd.DataFrame, ds: str, path: Path, *,
                   threshold: float, started: str = "", note: str = "") -> list[Path]:
    """(A) % responding naive / round 1 / round 2, odor partners pooled vs the click;
    (B) naive vs rounds 1 + 2 pooled; (C, D) the controlled changes (odors - click) with
    the batch-inversion 95 % CI and both p values (Holm over the 4 contrasts)."""
    from matplotlib.patches import Patch

    c_ds = cells.xs(ds, level=0, drop_level=False)
    part = c_ds.index.get_level_values("partner")
    with_click = {k[:3] for k in c_ds.index[part == CONTROL_ODOR]}   # flies the test uses
    keep = [k[:3] in with_click for k in c_ds.index]
    c_ds = c_ds[keep]
    is_click = c_ds.index.get_level_values("partner") == CONTROL_ODOR
    fly_ids = ["|".join(map(str, k[:3])) for k in c_ds.index]
    words = _PRIMARY_GROUP_WORDS[ds]
    col = GROUP_COLORS[ds]

    def level(mask, rounds):
        v = c_ds[mask][list(rounds)].to_numpy(dtype=float)
        ids = np.repeat(np.array(fly_ids, dtype=object)[mask], len(rounds))
        v = v.ravel()
        ok = np.isfinite(v)
        if not ok.any():
            return np.nan, np.nan, np.nan
        return fs.fly_cluster_rate_ci(v[ok] > 0.5, ids[ok])

    with plt.rc_context(_PUB_RC):
        fig = plt.figure(figsize=(7.2, 6.6))
        gs = fig.add_gridspec(2, 2, left=0.1, right=0.985, top=0.855, bottom=0.17,
                              hspace=0.42, wspace=0.28, width_ratios=[1.55, 1])
        axes = [fig.add_subplot(gs[i, j]) for i in (0, 1) for j in (0, 1)]
        # A / B: levels
        for ax, keys in ((axes[0], ("pre", 1, 2)), (axes[1], ("pre", "pair"))):
            _style_axis(ax)
            rounds_of = {"pre": (0,), 1: (1,), 2: (2,), "pair": (1, 2)}
            w = 0.8 / len(keys)
            hi = 0.0
            for gi, mask in enumerate((~is_click, is_click)):
                for j, key in enumerate(keys):
                    m, lo, up = level(mask, rounds_of[key])
                    x = gi + (j - (len(keys) - 1) / 2) * w
                    ax.bar(x, m, w * 0.92, color=BAR_COLORS[key], edgecolor="white",
                           linewidth=0.6, zorder=2)
                    if np.isfinite(lo):
                        ax.errorbar([x], [m], yerr=[[m - lo], [up - m]], fmt="none",
                                    ecolor=INK, elinewidth=0.7, capsize=1.6, capthick=0.7,
                                    zorder=3)
                        hi = max(hi, up)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(words, color=INK, fontsize=7.5)
            ax.set_ylim(0, max(10.0, hi * 1.15))
            ax.set_ylabel(f"Responding trials (%)\nscore ≥ {threshold:g}", color=INK,
                          fontsize=8)
        axes[0].legend(handles=[Patch(facecolor=BAR_COLORS[k], label=lab) for k, lab in
                                (("pre", "Naive"), (1, "Round 1"), (2, "Round 2"))],
                       loc="upper left", frameon=False, fontsize=7, ncol=3, handlelength=1,
                       columnspacing=1.0, borderaxespad=0.2)
        axes[1].legend(handles=[Patch(facecolor=BAR_COLORS[k], label=lab) for k, lab in
                                (("pre", "Naive"), ("pair", "Rounds 1 + 2"))],
                       loc="upper left", frameon=False, fontsize=7, ncol=2, handlelength=1,
                       columnspacing=1.0, borderaxespad=0.2)
        # C / D: controlled changes
        t = table[(table["dataset"] == ds)].set_index("contrast")
        for ax, names in ((axes[2], [c[0] for c in PRIMARY_CONTRASTS[:3]]),
                          (axes[3], [PRIMARY_CONTRASTS[3][0]])):
            _style_axis(ax)
            lo_all, hi_all = 0.0, 0.0
            for i, name in enumerate(names):
                if name not in t.index:
                    continue
                r = t.loc[name]
                ax.bar(i, r["estimate"], 0.55, color=col, edgecolor="white", linewidth=0.6,
                       zorder=2)
                ax.plot([i - 0.275, i + 0.275], [r["estimate"]] * 2, color=col, lw=2.2,
                        solid_capstyle="butt", zorder=2.6)      # a near-0 bar stays visible
                if np.isfinite(r["ci_lo"]) and np.isfinite(r["ci_hi"]):
                    ax.errorbar([i], [r["estimate"]],
                                yerr=[[r["estimate"] - r["ci_lo"]], [r["ci_hi"] - r["estimate"]]],
                                fmt="none", ecolor=INK, elinewidth=0.8, capsize=2,
                                capthick=0.8, zorder=3)
                    top, bot = r["ci_hi"], r["ci_lo"]
                else:
                    top = bot = r["estimate"]
                lo_all, hi_all = min(lo_all, bot), max(hi_all, top)
                txt = (f"{r['estimate']:+.1f} pp\nbatch p {_p_text(r['p_batch'])} "
                       f"(Holm {_p_text(r['p_batch_holm'])})\nfly p {_p_text(r['p_fly'])} "
                       f"(Holm {_p_text(r['p_fly_holm'])})")
                bold = r["p_batch_holm"] < 0.05
                ax.text(i, top, txt, ha="center", va="bottom", fontsize=6.3, color=INK,
                        fontweight="bold" if bold else "normal", zorder=4,
                        linespacing=1.15)
            span = max(hi_all - lo_all, 20.0)
            ax.set_ylim(lo_all - 0.12 * span, hi_all + 0.62 * span)
            ax.axhline(0, color=INK2, linewidth=0.7, zorder=2.5)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels([n.replace(" -> ", " →\n") for n in names], color=INK,
                               fontsize=7.5)
            ax.set_xlim(-0.6, len(names) - 0.4)
            ax.set_ylabel("Odor change − click change\n(% points)", color=INK, fontsize=8)
        for ax, letter in zip(axes, "ABCD"):
            ax.text(-0.16 if letter in "AC" else -0.22, 1.04, letter, transform=ax.transAxes,
                    fontsize=11, fontweight="bold", color=INK, va="bottom")
        n_f = int(t["n_flies"].max()) if len(t) else 0
        n_b = int(t["n_batches"].max()) if len(t) else 0
        fig.text(0.1, 0.975, f"{GROUP_SHORT[ds]}: primary endpoint (pre-specified "
                 f"{PRIMARY_ENDPOINT['registered']}){note}", fontsize=10, fontweight="bold",
                 color=INK, va="top")
        sub = (f"n = {n_f} flies / {n_b} batches  ·  % responding, odor partners pooled "
               f"minus the click in the same flies")
        sub += f"  ·  {started}" if started else ""
        fig.text(0.1, 0.94, sub, fontsize=8, color=INK2, va="top")
        defn = PRIMARY_ENDPOINT[GROUP_SHORT[ds]]
        foot = (f"C, D: {defn}. Bars pool trials; whiskers: 95 % CI inverting the batch-level "
                "sign-flip (raw; clears 0 exactly when batch p < 0.05). p: two-sided sign-flip "
                "with every fly of a batch flipped together (headline) and across flies; Holm "
                "over the 4 contrasts. A, B: 95 % CI with flies as the unit.")
        if ds == SECOND:
            foot += (" Odor 2 starts 11 s after odor 1 but is scored against the pre-odor-1 "
                     "baseline, so 'IAA after X' still carries X's own late response.")
        fig.text(0.1, 0.012, foot, fontsize=6.2, color=INK2, va="bottom", wrap=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        pdf = path.with_suffix(".pdf")
        fig.savefig(path, dpi=300, facecolor="white")
        fig.savefig(pdf, facecolor="white")
        plt.close(fig)
    return [path, pdf]


def write_primary_endpoint(out: Path, recs: pd.DataFrame, *, threshold: float,
                           odor1_scores: dict, started: str = "") -> list[Path]:
    """``00_Primary-Endpoint/``: the registered contrasts per dataset (figure + CSV), and
    the carry-over sensitivity analysis (pair trials whose odor 1 responded left out)
    for IAA-second. [] when no dataset has the click partner."""
    main = primary_table(recs, threshold)
    if main.empty:
        return []
    sens = primary_table(recs, threshold, odor1_scores, drop_odor1_responders=True)
    d = out / PRIMARY_DIR
    written = []
    cells = _primary_cells(recs, threshold)
    for ds in pd.unique(main["dataset"]):
        written += primary_figure(cells, main, ds, d / f"primary_endpoint_{GROUP_SHORT[ds]}.png",
                                  threshold=threshold, started=started)
    if SECOND in set(sens.get("dataset", [])):
        cells_s = _primary_cells(recs, threshold, odor1_scores, drop_odor1_responders=True)
        written += primary_figure(
            cells_s, sens, SECOND,
            d / f"primary_endpoint_{GROUP_SHORT[SECOND]}_odor1_responders_excluded.png",
            threshold=threshold, started=started,
            note=": carry-over check, pair trials whose odor 1 responded left out")
    table = pd.concat([main, sens], ignore_index=True)
    csv = d / PRIMARY_CSV
    table.assign(unit="percentage points", threshold=threshold,
                 registered=PRIMARY_ENDPOINT["registered"]).to_csv(csv, index=False)
    written.append(csv)
    return written


def primary_readme(table: pd.DataFrame) -> list[str]:
    """README.md's opening section: the registration verbatim, then the results."""
    lines = [f"# Primary endpoint (pre-specified {PRIMARY_ENDPOINT['registered']})", "",
             f"Figures and table: `{PRIMARY_DIR}/`. Registered in code "
             "(`PRIMARY_ENDPOINT`, pinned by a test); everything else in this folder is "
             "exploratory.", ""]
    lines += [f"- **{k}**: {v}" for k, v in PRIMARY_ENDPOINT.items()] + [""]
    main = table[table["analysis"] == PRIMARY_MAIN] if "analysis" in table else table
    for ds_short, g in main.groupby("group", sort=False):
        n_f, n_b = int(g["n_flies"].max()), int(g["n_batches"].max())
        lines += [f"## {ds_short} ({n_f} flies, {n_b} batches)", "",
                  "| contrast | odors' change | click's change | odors − click [95 % CI] "
                  "| batch p (Holm) | fly p (Holm) |", "|---|---|---|---|---|---|"]
        for r in g.itertuples():
            ci = (f"[{r.ci_lo:+.1f}, {r.ci_hi:+.1f}]" if np.isfinite(r.ci_lo)
                  and np.isfinite(r.ci_hi) else "[open: too few batches]")
            lines.append(f"| {r.contrast} | {r.odor_change:+.1f} | {r.click_change:+.1f} | "
                         f"**{r.estimate:+.1f}** {ci} | {_p_text(r.p_batch)} "
                         f"({_p_text(r.p_batch_holm)}) | {_p_text(r.p_fly)} "
                         f"({_p_text(r.p_fly_holm)}) |")
        sig = g[g["p_batch_holm"] < 0.05]
        lines += ["", ("No contrast clears p < 0.05 after Holm at batch level." if sig.empty
                       else "Clears p < 0.05 after Holm at batch level: "
                       + ", ".join(sig["contrast"]) + "."), ""]
    sens = table[table["analysis"] == PRIMARY_SENS] if "analysis" in table else table.iloc[0:0]
    sens = sens[sens["dataset"] == SECOND]
    if not sens.empty:
        lines += ["## Carry-over check (IAA-second)", "",
                  "Odor 2 starts 11 s after odor 1 but is scored against the baseline before "
                  "odor 1, so odor 1's late response and the onset dip land in odor 2's window. "
                  "In IAA-first the partner and the click both follow IAA, which largely "
                  "cancels it; in IAA-second 'IAA after X' still carries X's own response. "
                  "Here the pair trials whose odor 1 responded are left out:", "",
                  "| contrast | odors − click | batch p (Holm) | flies |", "|---|---|---|---|"]
        lines += [f"| {r.contrast} | {r.estimate:+.1f} | {_p_text(r.p_batch)} "
                  f"({_p_text(r.p_batch_holm)}) | {r.n_flies} |" for r in sens.itertuples()]
        lines.append("")
    lines += ["Units: percentage points of responding trials. The odors' and click's changes "
              "pool trials (never a mean of per-fly means); the tests take one value per fly. "
              "`p_within_fly_perm` in the CSV shuffles which partner is the click within each "
              "fly, a check for the unequal trial counts of the pooled and IAA-second "
              "contrasts.", ""]
    return lines


# ── odor 1 of each pair vs that odor naive (user 2026-11-16) ─────────────────
#: Odor 1 of a pair: "pre" is the same odor's naive pre-test trial.
ODOR1_LABELS = {"pre": "Naive (pre-test)", 1: "Test block, 1st presentation",
                2: "Test block, 2nd presentation",
                "pair": "Test block, 1st + 2nd presentation pooled"}
ODOR1_N_WORDS = {"pre": "naive", 1: "1st in test block", 2: "2nd in test block",
                 "pair": "in test block (2 per fly)"}
ODOR1_HEAD = {FIRST: "IAA first: IAA at the start of each pair vs naive IAA",
              SECOND: "IAA second: each odor at the start of its pair vs the same odor naive"}
ODOR1_XLABEL = {FIRST: "Pair (scored: IAA, presented right before this odor)",
                SECOND: "Scored odor (naive, or presented right before IAA)"}


# ── naive pre-test: IAA-first vs IAA-second flies (user 2026-11-16) ──────────
#: Violet / yellow, palette slots 7 and 4: CVD ΔE 37.9, normal 42.1 (validate_palette.js).
#: Yellow sits at 2.1:1 on white; the legend and the CSV twin carry identity and values.
GROUP_COLORS = {FIRST: "#6250d6", SECOND: "#eda100"}
GROUP_SHORT = {FIRST: "IAA-first", SECOND: "IAA-second"}
UNPAIRED_NOTE = ("Exploratory. * p < 0.05, ** p < 0.01, *** p < 0.001: IAA-first vs IAA-second, "
                 "whole batch folders swapped between the groups within each session (each "
                 "session keeps its number of IAA-second folders), Holm-adjusted over the "
                 "{n} odors of the panel; unmarked pairs are not significant. Fly-level p in "
                 "the stats CSV.")
EXACT_SPLITS = 100_000


def perm_p_unpaired(a, b, n_perm: int = 100_000) -> float:
    """Two-sided two-sample permutation p for a difference in means: the share of
    relabelings of the pooled values whose |mean difference| is at least the observed
    one. Exact when there are at most ``EXACT_SPLITS`` relabelings, else a seeded Monte
    Carlo estimate with the +1 correction. NaN when a group is empty."""
    import itertools
    import math
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not a.size or not b.size:
        return float("nan")
    x = np.concatenate([a, b])
    n, k = x.size, a.size
    obs = abs(a.mean() - b.mean()) - 1e-12
    total = x.sum()
    if math.comb(n, k) <= EXACT_SPLITS:
        sa = np.array([x[list(c)].sum() for c in itertools.combinations(range(n), k)])
        return float(np.mean(np.abs(sa / k - (total - sa) / (n - k)) >= obs))
    rng = np.random.default_rng(0)
    sa = np.array([x[rng.permutation(n)[:k]].sum() for _ in range(n_perm)])
    hits = int(np.sum(np.abs(sa / k - (total - sa) / (n - k)) >= obs))
    return float((hits + 1) / (n_perm + 1))


def _group_fly_values(rows: pd.DataFrame, threshold: float, panel: str) -> pd.Series:
    s = pd.to_numeric(rows["score"], errors="coerce")
    rows, s = rows[s.notna()], s[s.notna()]
    v = s if panel == "score" else (s >= threshold).astype(float)
    return v.groupby([rows["fly"], rows["fly_number"]]).mean()


def pretest_bars(pre: pd.DataFrame, path: Path, *, threshold: float, title: str,
                 started: str = "") -> list[Path]:
    """Naive pre-test responses to every odor, IAA-first vs IAA-second flies side by side:
    (A) mean score ± SEM over trials, (B) % responding. Each odor: a two-sample
    permutation test between the groups, Holm across odors within a panel. Writes the
    PNG, PDF, the plotted table and the stats as CSV."""
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, MaxNLocator

    groups = [g for g in DATASETS if (pre["dataset"] == g).any()]
    odors = [o for o in ORDER if o in set(pre["odor"])]
    table = []
    for odor in odors:
        for g in groups:
            rows = pre[(pre["odor"] == odor) & (pre["dataset"] == g)]
            table.append({"odor": odor, "group": GROUP_SHORT[g], "dataset": g,
                          **bar_stats(rows, threshold)})
    t = pd.DataFrame(table)

    st_rows = []
    if len(groups) == 2:
        ga, gb = groups
        for panel in ("score", "pct"):
            for i, odor in enumerate(odors):
                va, vb = (_group_fly_values(pre[(pre["odor"] == odor) & (pre["dataset"] == g)],
                                            threshold, panel) for g in (ga, gb))
                scale = 100.0 if panel == "pct" else 1.0
                # group is a batch property: whole folders swap groups within a session
                vals = pd.concat([va, vb])
                folders = [f for f, _ in vals.index]
                labels = {**{f: ga for f, _ in va.index}, **{f: gb for f, _ in vb.index}}
                batch = (bs.batch_label_perm(vals.to_numpy(dtype=float), folders, labels,
                                             {f: bs.session_of(f) for f in labels})
                         if len(va) and len(vb) else {"p": np.nan, "n_batches": 0})
                st_rows.append({"panel": panel, "odor": odor, "partner_i": i,
                                "a": GROUP_SHORT[ga], "b": GROUP_SHORT[gb], "key_a": ga, "key_b": gb,
                                "n_a": len(va), "n_b": len(vb), "n_batches": batch["n_batches"],
                                "mean_diff": scale * float(vb.mean() - va.mean())
                                if len(va) and len(vb) else np.nan,
                                "p_batch": batch["p"], "p_fly": perm_p_unpaired(va, vb)})
    st = pd.DataFrame(st_rows, columns=["panel", "odor", "partner_i", "a", "b", "key_a", "key_b",
                                        "n_a", "n_b", "n_batches", "mean_diff", "p_batch",
                                        "p_fly"])
    st["p_holm"] = np.nan
    for panel in ("score", "pct"):
        m = st["panel"] == panel
        if m.any():
            st.loc[m, "p_holm"] = holm(st.loc[m, "p_batch"])
    st["stars"] = st["p_holm"].map(stars) if len(st) else pd.Series(dtype=str)

    ns = {GROUP_SHORT[g]: sorted(set(t.loc[(t.dataset == g) & (t.n > 0), "n"].astype(int)))
          for g in groups}
    note = ("n = " + ", ".join(f"{v[0]} {k} flies" for k, v in ns.items())
            + " (1 naive trial per fly per odor)") if all(len(v) == 1 for v in ns.values()) else None

    w, step = (0.36, 0.37) if len(groups) == 2 else (0.5, 0.0)
    x0 = np.arange(len(odors), dtype=float)
    with plt.rc_context(_PUB_RC):
        fig = plt.figure(figsize=(7.2, 5.6))
        gs = fig.add_gridspec(2, 1, left=0.1, right=0.985, top=0.86, bottom=0.19, hspace=0.18)
        axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
        xpos = {}
        for ax, (col, err, ylab) in zip(axes, BAR_PANELS):
            _style_axis(ax)
            for j, g in enumerate(groups):
                d = t[t.dataset == g]
                x = x0 + (j - (len(groups) - 1) / 2) * step
                xpos.update({(i, g): xi for i, xi in enumerate(x)})
                ax.bar(x, d[col], w, color=GROUP_COLORS[g], edgecolor="white", linewidth=0.6,
                       zorder=2)
                if err is not None:                       # 95 % CI, flies as the unit
                    elo = (d[col] - d["ci_lo"]).to_numpy(dtype=float)
                    ehi = (d["ci_hi"] - d[col]).to_numpy(dtype=float)
                    some = np.isfinite(elo) & np.isfinite(ehi) & ((elo > 0) | (ehi > 0))
                    ax.errorbar(x[some], d[col][some], yerr=[elo[some], ehi[some]], fmt="none",
                                ecolor=INK, elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=3)
                for xi in x[d[col].to_numpy(dtype=float) == 0]:
                    ax.plot([xi - w / 2, xi + w / 2], [0, 0], color=GROUP_COLORS[g], lw=2.2,
                            solid_capstyle="butt", zorder=2.6)
                if note is None:
                    for xi, v, n in zip(x, d[col], d["n"]):
                        if np.isfinite(v):
                            ax.text(xi, v, f"{n}", ha="center", va="bottom", fontsize=6, color=INK2)
            ax.set_ylabel(ylab.format(threshold=threshold), color=INK, fontsize=8.5)
            ax.set_xlim(-0.6, len(odors) - 0.4)
            ax.set_xticks(x0)
            ax.axhline(0, color=RULE, linewidth=0.6, zorder=2.5)
        top = t.assign(a=t["ci_hi"].fillna(t["mean"]), b=t["pct"]).groupby("odor")
        lo = float(np.nanmin(t["ci_lo"].fillna(t["mean"]))) if len(t) else 0.0
        hi = float(np.nanmax(t["ci_hi"].fillna(t["mean"]))) if len(t) else 1.0
        span = max(hi - min(lo, 0.0), 0.5)
        limits = []
        for ax, panel, col, unit, base in ((axes[0], "score", "a", 0.06 * span, hi + 0.08 * span),
                                            (axes[1], "pct", "b", 6.0, 100.0)):
            tops = {i: max(float(np.nanmax(top.get_group(o)[col])), 0.0) for i, o in enumerate(odors)}
            limits.append(max(base, _brackets(ax, st[st["panel"] == panel], xpos, tops, unit)))
        axes[0].set_ylim(min(lo, 0.0) - (0.06 * span if lo < 0 else 0), limits[0])
        axes[0].yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        axes[0].spines["bottom"].set_visible(False)
        axes[0].tick_params(axis="x", length=0, labelbottom=False)
        axes[1].set_ylim(0, limits[1])
        axes[1].yaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
        axes[1].set_xticklabels([o.replace(" ", "\n") for o in odors], color=INK, fontsize=8)
        axes[1].set_xlabel("Odor, presented alone in the pre-test (before any pair)", color=INK,
                           fontsize=8.5)
        fig.align_ylabels(axes)
        for ax, letter in zip(axes, "AB"):
            ax.text(-0.085, 1.02, letter, transform=ax.transAxes, fontsize=11,
                    fontweight="bold", color=INK, va="bottom")
        fig.text(0.1, 0.975, title, fontsize=10, fontweight="bold", color=INK, va="top")
        fig.legend(handles=[Patch(facecolor=GROUP_COLORS[g], label=f"{GROUP_SHORT[g]} flies")
                            for g in groups],
                   loc="lower left", bbox_to_anchor=(0.1, 0.875), ncol=len(groups), frameon=False,
                   fontsize=8, handlelength=1.1, handleheight=0.9, columnspacing=1.6,
                   borderaxespad=0, labelcolor=INK)
        corner = (f"{note}  ·  A: mean, 95 % CI (flies)" if note
                  else "n per bar above it  ·  A: mean, 95 % CI (flies)")
        corner += f"  ·  {started}" if started else ""
        fig.text(0.1, 0.937, corner, ha="left", va="top", fontsize=8, color=INK2)  # subtitle
        fig.text(0.1, 0.012, UNPAIRED_NOTE.format(n=len(odors)), ha="left", va="bottom", fontsize=6.5, color=INK2,
                 wrap=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        pdf = path.with_suffix(".pdf")
        fig.savefig(path, dpi=300, facecolor="white")
        fig.savefig(pdf, facecolor="white")
        plt.close(fig)
    csv = path.with_suffix(".csv")
    t.drop(columns=["pct_sem"]).rename(columns={
        "mean": "mean_score", "sem": "sem_score", "pct": "pct_responding"}).assign(
        threshold=threshold).to_csv(csv, index=False)
    stats_csv = path.with_name(f"{path.stem}_stats.csv")
    st.drop(columns=["key_a", "key_b", "partner_i"]).assign(
        stars=st["stars"].replace("", "ns"),
        diff_unit=np.where(st["panel"] == "pct", "percentage points", "score"),
        test="p_batch (headline): whole folders swapped between the groups within each "
             "session, two-sided; Holm across odors within panel (p_holm). p_fly: flies "
             "swapped one by one (ignores batches)"
    ).to_csv(stats_csv, index=False)
    return [path, pdf, csv, stats_csv]


# ── odor 1 -> odor 2 reaction contingency ──────────────────────────────────
def pair_reactions(pred: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """One row per pair trial: odor-1 / odor-2 scores and reacted flags."""
    p = pred[pred["trial_label"].astype(str).str.contains("_then_", case=False)].copy()
    p["is2"] = p["trial_label"].str.endswith(op.ODOR2_SUFFIX)
    p["base"] = p["trial_label"].str.replace(op.ODOR2_SUFFIX, "", regex=False)
    key = ["dataset", "fly", "fly_number", "base"]
    s1 = p[~p["is2"]].set_index(key)["score"].rename("s1")
    s2 = p[p["is2"]].set_index(key)["score"].rename("s2")
    t = pd.concat([s1, s2], axis=1).dropna().reset_index().rename(columns={"base": "label"})
    parsed = t["label"].map(parse_label)
    t["number"] = parsed.map(lambda x: x[1])
    t["odor1"] = parsed.map(lambda x: x[2][0])
    t["odor2"] = parsed.map(lambda x: x[2][1])
    t["r1"] = t["s1"] >= threshold
    t["r2"] = t["s2"] >= threshold
    return t


def conditional_rates(t: pd.DataFrame) -> dict:
    r1, r2 = t["r1"].astype(bool), t["r2"].astype(bool)
    rate = lambda m: float(r2[m].mean()) if m.any() else float("nan")  # noqa: E731
    return {"n_r1": int(r1.sum()), "r2_given_r1": rate(r1),
            "n_not_r1": int((~r1).sum()), "r2_given_not_r1": rate(~r1),
            "n": int(len(t)), "r2": float(r2.mean()) if len(t) else float("nan"),
            "both": int((r1 & r2).sum()), "only1": int((r1 & ~r2).sum()),
            "only2": int((~r1 & r2).sum()), "neither": int((~r1 & ~r2).sum())}


_MONTHS = {m: i for i, m in enumerate(
    ("january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"), start=1)}
_FOLDER = re.compile(r"^([a-z]+)_(\d+)_batch_(\d+)(?:_rig_(\d+))?$", re.IGNORECASE)


def short_folder(fly: str) -> str:
    """'november_14_batch_1_rig_3' -> '11-14 b1 rig3'; a folder without a rig tail is rig 1."""
    m = _FOLDER.match(str(fly))
    if not m or m.group(1).lower() not in _MONTHS:
        return str(fly)
    return f"{_MONTHS[m.group(1).lower()]}-{int(m.group(2))} b{m.group(3)} rig{m.group(4) or 1}"


def odor1_vs_odor2_figure(t: pd.DataFrame, title: str, threshold: float, path: Path) -> Path:
    c = conditional_rates(t)
    n_flies = t.groupby(["dataset", "fly", "fly_number"]).ngroups
    per_fly_w = max(1.6, 0.13 * n_flies)          # room for each fly's two n labels
    fig = plt.figure(figsize=(17 + 3.8 * (per_fly_w - 1.6), 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.25, per_fly_w], wspace=0.45)

    ax = fig.add_subplot(gs[0])
    m = np.array([[c["both"], c["only1"]], [c["only2"], c["neither"]]], dtype=float)
    ax.imshow(m, cmap="Greys", vmin=0, vmax=max(m.max(), 1) * 1.6)
    for (i, j), v in np.ndenumerate(m):
        row_n = m[i].sum()
        pct = f"\n{100 * v / row_n:.0f}% of row" if row_n else ""
        ax.text(j, i, f"{int(v)}{pct}", ha="center", va="center", fontsize=10, weight="bold")
    ax.set_xticks([0, 1], ["reacts to\nodor 2", "no reaction\nto odor 2"])
    ax.set_yticks([0, 1], ["reacts to\nodor 1", "no reaction\nto odor 1"])
    ax.set_title("pair trials", fontsize=11, weight="bold")

    ax = fig.add_subplot(gs[1])
    vals = [c["r2_given_r1"], c["r2_given_not_r1"]]
    ns = [c["n_r1"], c["n_not_r1"]]
    ax.bar([0, 1], vals, color=["#1b9e77", "#999999"], width=0.6)
    for x, v, n in zip([0, 1], vals, ns):
        if np.isfinite(v):
            ax.text(x, v + 0.02, f"{100 * v:.0f}%\n(n={n})", ha="center", va="bottom", fontsize=9)
    ax.set_xticks([0, 1], ["after reacting\nto odor 1", "after no reaction\nto odor 1"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("fraction reacting to odor 2")
    ax.set_title("odor-2 reaction rate", fontsize=11, weight="bold")

    ax = fig.add_subplot(gs[2])
    levels = list(range(-1, 6))
    mat = np.zeros((len(levels), len(levels)))
    for a, bb in zip(t["s1"].astype(int), t["s2"].astype(int)):
        mat[levels.index(a), levels.index(bb)] += 1
    ax.imshow(mat, cmap="Blues", origin="lower")
    for (i, j), v in np.ndenumerate(mat):
        if v:
            ax.text(j, i, int(v), ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(levels)), levels)
    ax.set_yticks(range(len(levels)), levels)
    ax.set_xlabel("odor 2 score")
    ax.set_ylabel("odor 1 score")
    ax.axhline(levels.index(int(np.ceil(threshold))) - 0.5, color="#d95f02", lw=1)
    ax.axvline(levels.index(int(np.ceil(threshold))) - 0.5, color="#d95f02", lw=1)
    ax.set_title("score pairs (orange = reaction threshold)", fontsize=10, weight="bold")

    ax = fig.add_subplot(gs[3])
    flies = sorted(t.groupby(["dataset", "fly", "fly_number"]).groups)
    for k, (ds, fly, fn) in enumerate(flies):
        f = t[(t.dataset == ds) & (t.fly == fly) & (t.fly_number == fn)]
        cf = conditional_rates(f)
        for off, v, n, col in ((-0.18, cf["r2_given_r1"], cf["n_r1"], "#1b9e77"),
                               (0.18, cf["r2_given_not_r1"], cf["n_not_r1"], "#999999")):
            if n:
                ax.bar(k + off, v, 0.34, color=col)
                ax.text(k + off, v + 0.02, f"{n}", ha="center", va="bottom", fontsize=7,
                        rotation=90)
    short = [f"{'1st' if ds == FIRST else '2nd'} {short_folder(fly)} f{fn}"
             for ds, fly, fn in flies]
    ax.set_xticks(range(len(flies)), short, rotation=60, ha="right", fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("fraction reacting to odor 2")
    ax.set_title("per fly (green: after odor-1 reaction, grey: after none; number = n trials)",
                 fontsize=9, weight="bold")

    fig.suptitle(f"{title}\nreacts = model score >= {threshold:g}; one count per pair trial",
                 fontsize=12, y=1.08)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def read_config(config: Path) -> tuple[float, str]:
    """(reaction_prediction.binary_threshold, flagged_flies_csv or "")."""
    import yaml
    cfg = yaml.safe_load(config.read_text())
    return (float(cfg["reaction_prediction"]["binary_threshold"]),
            str(cfg.get("flagged_flies_csv") or ""))


def binary_threshold(config: Path) -> float:
    return read_config(config)[0]


def _read_wide(csv_dir: Path, stem: str) -> pd.DataFrame:
    """The two datasets' rows of one wide table: the parquet (seconds), else the CSV."""
    pq = csv_dir / f"{stem}.parquet"
    if pq.is_file():
        return pd.read_parquet(pq, filters=[("dataset", "in", list(DATASETS))])
    parts = [c[c["dataset"].isin(DATASETS)]
             for c in pd.read_csv(csv_dir / f"{stem}.csv", chunksize=2000, low_memory=False)]
    return pd.concat(parts, ignore_index=True)


def _not_frozen(df: pd.DataFrame) -> pd.DataFrame:
    if "frozen" not in df.columns:
        return df
    return df[~df["frozen"].astype(str).str.lower().eq("true")]


def _fly_keys(df: pd.DataFrame) -> set:
    return {(str(d), str(f), int(n)) for d, f, n in
            zip(df["dataset"], df["fly"], pd.to_numeric(df["fly_number"]))}


def _not_flagged(df: pd.DataFrame, excluded: set) -> pd.DataFrame:
    keys = zip(df["dataset"].astype(str).str.strip(), df["fly"].astype(str).str.strip(),
               df["fly_number"].map(canon_fly_number))
    return df[[k not in excluded for k in keys]]


def load_inputs(csv_dir: Path, predictions: Path, flagged_csv: str = ""):
    """(pretest rows, pair rows, predictions): the two datasets only, frozen folders
    and flies the flagged-flies CSV marks dead or bad (FLY-State != 1, the scorer's
    rule) left out of all three, so such a fly leaves every figure and count."""
    excluded = load_flagged_fly_exclusions(flagged_csv) if flagged_csv else set()
    pair_df = _not_flagged(_not_frozen(_read_wide(csv_dir, WIDE_TABLES[0])), excluded)
    pair_df = pair_df[pair_df["trial_label"].map(op.is_pair_label)]
    pre_df = _not_flagged(_not_frozen(_read_wide(csv_dir, WIDE_TABLES[1])), excluded)
    pred = pd.read_csv(predictions)
    pred = pred[pred["dataset"].isin(DATASETS)]
    keep = _fly_keys(pre_df) | _fly_keys(pair_df)
    pred = pred[[k in keep for k in zip(pred["dataset"].astype(str), pred["fly"].astype(str),
                                       pd.to_numeric(pred["fly_number"]).astype(int))]]
    return pre_df.reset_index(drop=True), pair_df.reset_index(drop=True), pred.reset_index(drop=True)


def _drop_empty_traces(rows: pd.DataFrame) -> pd.DataFrame:
    """Without the ``dir_val_*`` columns these rows leave empty: another dataset's longer
    trace widens the shared table without changing these rows."""
    cols = [c for c in rows.columns if str(c).startswith("dir_val_")]
    empty = rows[cols].isna().all()
    return rows.drop(columns=empty.index[empty.to_numpy(dtype=bool)])


def fingerprint(pre_df: pd.DataFrame, pair_df: pd.DataFrame, pred: pd.DataFrame,
                threshold: float) -> str:
    """Digest of everything the folder is drawn from: rows, scores, threshold, code."""
    return fc.combine(fc.code_digest([Path(__file__)]), f"{threshold:g}",
                      fc.frame_digest(_drop_empty_traces(pre_df)),
                      fc.frame_digest(_drop_empty_traces(pair_df)), fc.frame_digest(pred))


def _read_cache(out: Path) -> dict:
    try:
        got = json.loads((out / CACHE_FILE).read_text())
        return got if isinstance(got, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_cache(out: Path, fp: str, files: list[str]) -> None:
    tmp = out / f".{CACHE_FILE}.tmp"
    tmp.write_text(json.dumps({"fingerprint": fp, "files": sorted(files)}, indent=1))
    os.replace(tmp, out / CACHE_FILE)


def _lock(out: Path, timeout_s: float = LOCK_TIMEOUT_S):
    """One build of the folder at a time (a hand run and the pipeline); None on timeout.
    Keep the handle open: the kernel drops the lock when it closes."""
    handle = open(out / ".build.lock", "a")
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except OSError:
            if time.monotonic() >= deadline:
                handle.close()
                return None
            time.sleep(0.5)


# ── odor + 5 s after window, scores figures (user 2026-11-17) ──────────────────
PLUS_AFTER_S = 5.0
PLUS5_WINDOW = f"odor + {PLUS_AFTER_S:g} s after"
#: Green / magenta / dark magenta (palette slots 6 and 5, magenta darkened for the 2nd
#: presentation): all-pairs CVD ΔE >= 13.4, normal >= 27.0 (validate_palette.js, light).
#: No odor-window b2b figure uses them, so a +5 s figure cannot pass for one. Magenta
#: sits at 2.6:1 on white; the legend and the CSV twin carry identity.
PLUS5_COLORS = {"pre": "#008300", 1: "#e87ba4", 2: "#8a2a5c", "pair": "#e87ba4"}
#: Short: three must fit one legend row; the title names the window in full.
PLUS5_LABELS = {"pre": f"Alone (pre-block), +{PLUS_AFTER_S:g} s",
                1: f"In pair, 1st presentation, +{PLUS_AFTER_S:g} s",
                2: f"In pair, 2nd presentation, +{PLUS_AFTER_S:g} s",
                "pair": f"In pair, 1st + 2nd pooled, +{PLUS_AFTER_S:g} s"}
PLUS5_HEAD = {FIRST: f"IAA first, scored on {PLUS5_WINDOW}: each odor alone vs right after IAA",
              SECOND: f"IAA second, scored on {PLUS5_WINDOW}: IAA alone vs right after each odor"}
PLUS5_CONTROL_HEAD = {
    FIRST: f"IAA first, scored on {PLUS5_WINDOW}: each odor's change minus Paraffin Oil's",
    SECOND: f"IAA second, scored on {PLUS5_WINDOW}: IAA's change after each odor minus "
            "after Paraffin Oil"}
PLUS5_CONTROL_LABELS = {k: f"{v}, +{PLUS_AFTER_S:g} s" for k, v in
                        {"pair": "1st + 2nd pooled, minus control",
                         1: "1st, minus control", 2: "2nd, minus control"}.items()}


def plus_after_rows(pre_df: pd.DataFrame, pair_df: pd.DataFrame, odor2_window) -> pd.DataFrame:
    """The trials rescored on odor + 5 s: each naive pre-test trial as recorded and each
    pair trial's odor 2 as production scores it (``op.split_pair_rows``: the pre-odor
    block, then everything from odor 2's onset, timed as a normal trial). Odor 1 rows are
    left out: their 5 s after would be odor 2."""
    split, _ = op.split_pair_rows(pair_df, odor2_window)
    o2 = split[split["trial_label"].astype(str).str.endswith(op.ODOR2_SUFFIX)]
    return pd.concat([pre_df, o2], ignore_index=True)


def plus_after_model(config: Path) -> Path | None:
    """reaction_prediction.model_path of the config, None when unset or missing."""
    import yaml
    cfg = yaml.safe_load(Path(config).read_text()) or {}
    path = str((cfg.get("reaction_prediction") or {}).get("model_path") or "")
    return Path(path) if path and Path(path).is_file() else None


def make_plus_after_scorer(model: Path, config: Path):
    """``score(rows) -> int array``: the production model on odor + 5 s, scored as
    production scores (window_rescore: the model's delay, features, two-guard)."""
    if wr is None:
        raise RuntimeError(f"window_rescore unavailable: {_WR_ERROR}")
    spec = wr.window_spec("odor_plus_after", PLUS_AFTER_S)
    rule = wr.threshold_rule_from_config(config)
    return lambda rows: np.asarray(wr.rescore(rows, spec, model, rule=rule)).astype(int)


def plus_after_figure(out: Path, pre_df, pair_df, scores: dict, scorer, onset, odor2_window,
                      *, threshold: float, started: str) -> list[Path]:
    """<group>/<group>_scores(_pooled)_odor_plus_5s.* and <group>_odor_plus_5s_scores.csv
    for both groups; nothing for a group whose rescoring fails (the rest still builds)."""
    written = []
    for ds in DATASETS:
        pre = pre_df[pre_df["dataset"] == ds]
        pair = pair_df[pair_df["dataset"] == ds]
        if pair.empty:
            continue
        rows = plus_after_rows(pre, pair, odor2_window)
        try:
            new = np.asarray(scorer(rows)).astype(int)
        except Exception as exc:  # noqa: BLE001 - a scorer problem must not cost the folder
            print(f"[b2b][WARN] {ds}: odor + {PLUS_AFTER_S:g} s rescoring failed, figures "
                  f"skipped: {exc}")
            continue
        keys = list(zip(rows["dataset"], rows["fly"], rows["fly_number"].astype(int),
                        rows["trial_label"]))
        recs = build_records(pre, pair, dict(zip(keys, new.tolist())), onset)
        short = GROUP_SHORT[ds]
        for pooled, name in ((False, "scores"), (True, "scores_pooled")):
            written += score_bars(recs, PLUS5_HEAD[ds],
                                  out / short / f"{short}_{name}_odor_plus_5s.png",
                                  threshold=threshold, xlabel=BAR_XLABEL[ds],
                                  pool_presentations=pooled, labels=PLUS5_LABELS,
                                  colors=PLUS5_COLORS, started=started)
        for pooled, name in ((False, "change_minus_paraffin"),
                             (True, "change_minus_paraffin_pooled")):
            written += control_bars(recs, PLUS5_CONTROL_HEAD[ds],
                                    out / short / f"{short}_{name}_odor_plus_5s.png",
                                    threshold=threshold, xlabel=BAR_XLABEL[ds],
                                    pool_presentations=pooled, started=started, group=ds,
                                    colors=PLUS5_COLORS, labels=PLUS5_CONTROL_LABELS)
        csv = out / short / f"{short}_odor_plus_5s_scores.csv"
        pd.DataFrame(keys, columns=["dataset", "fly", "fly_number", "trial_label"]).assign(
            score=new, score_odor_window=[scores.get(k) for k in keys],
            score_window=PLUS5_WINDOW).to_csv(csv, index=False)
        written.append(csv)
    return written


# ── clock window: flies by their first pre-test start (user 2026-11-17) ────────
#: A raw pre-test trial file; its stamp is the Pi-clock start, read as
#: experiment_start_times.start_table reads it. Kept here so this script's code
#: fingerprint does not take in that module's imports.
_PRETEST_FILE = re.compile(r"^output_.*_pretest_\d+_.*_(\d{8}_\d{6})\.csv$")


def first_pretest_start(roots, dataset: str, folder: str) -> dt.datetime | None:
    """The earliest raw pre-test trial of a batch folder, from the first root holding any."""
    for root in roots:
        d = Path(root) / dataset / folder
        if not d.is_dir():
            continue
        times = [dt.datetime.strptime(m[1], "%Y%m%d_%H%M%S")
                 for f in d.iterdir() if (m := _PRETEST_FILE.match(f.name))]
        if times:
            return min(times)
    return None


def parse_clock(text: str) -> dt.time:
    """``"09:40"`` / ``"16:40"`` -> time of day."""
    return dt.datetime.strptime(str(text).strip(), "%H:%M").time()


def clock_text(t: dt.time) -> str:
    """``9:40am``, ``4pm``."""
    hour, half = t.hour % 12 or 12, "am" if t.hour < 12 else "pm"
    return f"{hour}{half}" if t.minute == 0 else f"{hour}:{t.minute:02d}{half}"


def window_text(lo: dt.time, hi: dt.time) -> str:
    return f"{clock_text(lo)}\u2013{clock_text(hi)}"


def folder_starts(frames, roots) -> pd.DataFrame:
    """One row per (dataset, batch folder) in ``frames``: its first pre-test start (NaT
    when no root holds a raw pre-test file)."""
    keys = sorted({(str(d), str(f)) for df in frames for d, f in zip(df["dataset"], df["fly"])})
    return pd.DataFrame([{"dataset": d, "folder": f, "first_pretest": first_pretest_start(roots, d, f)}
                         for d, f in keys], columns=["dataset", "folder", "first_pretest"])


def window_mask(starts: pd.DataFrame, lo: dt.time, hi: dt.time) -> pd.Series:
    """Folders whose first pre-test started in [lo, hi); an unknown start is in no window."""
    return pd.Series([t is not None and not pd.isna(t) and lo <= pd.Timestamp(t).time() < hi
                      for t in starts["first_pretest"]], index=starts.index, dtype=bool)


def keep_folders(df: pd.DataFrame, keep: set) -> pd.DataFrame:
    return df[[(str(d), str(f)) in keep for d, f in zip(df["dataset"], df["fly"])]].reset_index(drop=True)


def _start_lines(rows: pd.DataFrame) -> list[str]:
    lines = ["| dataset | batch folder | first pre-test (Pi clock) |", "|---|---|---|"]
    for r in rows.sort_values(["first_pretest", "dataset", "folder"], na_position="last").itertuples():
        when = "no raw pre-test file found" if pd.isna(r.first_pretest) else \
            pd.Timestamp(r.first_pretest).strftime("%Y-%m-%d %H:%M")
        lines.append(f"| {r.dataset} | {r.folder} | {when} |")
    return lines


def write_readme(out: Path, starts: pd.DataFrame, inside: pd.Series, window: str,
                 primary: pd.DataFrame | None = None, sections: list[str] | None = None) -> Path:
    """README.md: the primary endpoint first (when ``primary`` has rows), then the
    window, ``sections`` (figure guide, multiplicity, caveats), the folders drawn and the
    folders left out."""
    head = (f"# IAA odor-pair (b2b) figures: flies started {window}" if window
            else "# IAA odor-pair (b2b) figures: every start time")
    lines = []
    if primary is not None and not primary.empty:
        lines += primary_readme(primary) + ["---", ""]
    lines += [head, "",
             "Built by `scripts/analysis/b2b_pair_figures.py` (pipeline step "
             "`analysis.b2b_pair_figures`, config_new.yaml); redrawn when the two datasets' "
             "rows, their scores, the reaction threshold or the code change.", ""]
    if window:
        lines += [f"Batch folders whose first pre-test trial started {window} (Pi clock, from "
                  "the raw trial file name; the end is exclusive). Frozen folders and flies "
                  "flagged dead or bad are left out before the window is applied.", ""]
    lines += list(sections or [])
    drawn = starts[inside]
    if drawn.empty:
        lines += [f"**No batch started {window or 'yet'}: no figures.** The folder fills on "
                  "the first pipeline run after one does.", ""]
    else:
        lines += [f"## Drawn ({len(drawn)} batch folders)", "", *_start_lines(drawn), ""]
    if window and (~inside).any():
        lines += ["## Left out: started outside the window", "", *_start_lines(starts[~inside]), ""]
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text("\n".join(lines))
    return out / "README.md"


#: README: what each figure set is and how much it can carry (stats audit 2026-11-17).
FIGURE_GUIDE = [
    "## What is here", "",
    "| set | what it compares | status |", "|---|---|---|",
    f"| `{PRIMARY_DIR}/` | odor partners' change minus the click's, same flies (above) "
    "| **primary** (pre-specified) |",
    "| `<group>/<group>_change_minus_paraffin(_pooled)(_odor_plus_5s)` | each partner's change "
    "minus the click's, per partner; batch-level p on every bar | exploratory, controlled |",
    "| `Change-Minus-Click/` | the same contrast on log AUC during / before and the peak-z "
    "ratio, both datasets | exploratory, controlled |",
    "| `<group>/<group>_scores(_pooled)(_odor_plus_5s)`, `both/` | each odor alone (naive) vs "
    "in the pair | secondary: NOT controlled for time in the rig or re-testing |",
    "| `<group>/<group>_peakz_pooled` | the same alone vs in the pair on the PER-trace peak-z "
    "ratio (signed, unsigned); raw p on every odor | secondary: NOT controlled |",
    "| `<group>/<group>_odor1_vs_naive(_pooled)` | odor 1 of each pair vs the same odor naive "
    "| secondary: NOT controlled |",
    "| `<group>/<group>_dataset_mean.png`, `per_fly/` | mean and per-fly traces | descriptive |",
    "| `pretest/` | naive pre-test, IAA-first vs IAA-second (folders swapped within session) "
    "| exploratory |",
    "| `Pre-Exposure-Baseline/` | naive pre-test by group and rig, batch-level tests "
    "| exploratory |",
    "| `Time-of-Day_2h-bins_whole-dataset/` | every fly of both windows by start time, "
    "batch-level tests | exploratory |",
    "| `odor1_vs_odor2/` | does a fly that reacts to odor 1 also react to odor 2 (counts, "
    "no p) | descriptive |", "",
    "Timing: the naive pre-block runs about 5-35 min into a session, the pair block about "
    "65-140 min, so any alone-vs-pair difference also holds time in the rig, starvation and "
    "re-testing; the click (the Paraffin Oil partner, a master-valve click with no odor since "
    "2026-11-10) sits in the same slot at the same time and is the control.", "",
    "**Odor 2 is scored against the baseline before odor 1.** Odor 2 starts 11 s after odor 1, "
    "so odor 1's late response and the onset dip land in odor 2's window; carry-over alone "
    "reproduces the 'odor 1 predicts odor 2' association (`odor1_vs_odor2/`). The click "
    "contrast largely cancels it in IAA-first (partner and click both follow IAA), not in "
    "IAA-second (see the carry-over check above).", "",
    "Units: flies and batches. A batch folder is one rig's run of a session; start time, rig "
    "and group are batch properties, so their tests keep a batch's flies together. Bars "
    "pool trials (never a mean of per-fly means); intervals count flies.", "",
]
#: Known data issues, reported 2026-11-18; none changed by these scripts.
DATA_NOTES = [
    "## Known data issues", "",
    "- BehaviorLocust (`_rig_2`) Hexanol is the highest naive response there in every session "
    "(Benzaldehyde lower). The rig's software pin map (B GPIO 22, H 27) is unchanged since "
    "July and every Hexanol trial fired valve H, so a swap would be in the tubing or valve "
    "wiring: only a bench test on the rig can settle it. Unverified.",
    "- The three `november_14_batch_1*` folders' session_metadata gave the starvation start as "
    "Nov 15 (after the recording); corrected to Nov 14 on 2026-11-18, local and secured copies "
    "(originals: `/securedstorage/DATAsec/cole/metadata_corrections/`). These figures use the "
    "target hours, never that date.",
    "- `IAA-PairSecond-Control-1-1/november_13_batch_1_rig_3` pretest_1 (the click, 4 flies) "
    "had a one-frame odor window: the onset click fell between frame reads. Fixed in "
    "`fbpipe/utils/trial_metadata.py` (2026-11-18, window from the sensors log); it reaches "
    "these figures on the next full pipeline run.", "",
]
#: Raw p columns counted for the folder-wide BH q: from each tuple, the first present
#: (a table's batch-level p is its headline; the fly-level twin is not a second test).
P_COLUMNS = (("p_batch", "p"), ("p_rate_batch",), ("p_score_batch",), ("p_trend_batch",),
             ("p_bins_batch",))
BH_CSV = "folder_tests_bh.csv"


def folder_tests(out: Path, written) -> pd.DataFrame:
    """Every raw p this build's tables test, one row per test: family (the table), column,
    p, and the Benjamini-Hochberg q over the whole folder. The primary endpoint's
    carry-over rows are a sensitivity check, not further tests."""
    rows = []
    for path in sorted({Path(p) for p in written}):
        if path.suffix != ".csv" or not path.is_file() or path.name == BH_CSV:
            continue
        try:
            t = pd.read_csv(path)
        except Exception:  # noqa: BLE001 - an unreadable table is just not counted
            continue
        if "analysis" in t.columns:
            t = t[t["analysis"] == PRIMARY_MAIN]
        for cols in P_COLUMNS:
            col = next((c for c in cols if c in t.columns), None)
            if col is None:
                continue
            for v in pd.to_numeric(t[col], errors="coerce").dropna():
                rows.append({"family": str(path.relative_to(out)), "column": col,
                             "p": float(v)})
    df = pd.DataFrame(rows, columns=["family", "column", "p"])
    df["q_bh"] = fs.bh_adjust(df["p"].to_numpy(dtype=float)) if len(df) else []
    return df


def multiplicity_lines(tests: pd.DataFrame) -> list[str]:
    if tests.empty:
        return []
    q = tests["q_bh"]
    low = tests.nsmallest(3, "p")
    lines = ["## Multiplicity", "",
             f"{len(tests)} tests in {tests['family'].nunique()} tables in this folder. Every "
             "figure Holm-adjusts its own panel; across the whole folder, Benjamini-Hochberg "
             f"q: {int((q < 0.05).sum())} below 0.05 (smallest q {q.min():.2f}). Smallest raw "
             "p: " + "; ".join(f"{r.p:.3g} (`{r.family}`)" for r in low.itertuples())
             + f". Every test and its q: `{BH_CSV}`.", ""]
    return lines


def _prune_empty_dirs(out: Path) -> None:
    for d in sorted((p for p in out.rglob("*") if p.is_dir()), key=lambda p: len(p.parts),
                    reverse=True):
        if not any(d.iterdir()):
            d.rmdir()


def odor2_lookup(roots):
    """``odor2_window(row)`` -> odor 2's valve window (s) of a pair row from its rig frame
    log, each trial read once per run."""
    memo = {}

    def odor2_window(row):
        k = (str(row["dataset"]), str(row["fly"]), str(row["trial_label"]))
        if k not in memo:
            p = op.find_frames_csv(row, roots)
            memo[k] = None if p is None else op.second_odor_window_s(p, float(row["fps"]))
        return memo[k]
    return odor2_window


def build(out: Path, pre_df, pair_df, pred, thr: float, roots, window: str = "",
          plus5_scorer=None, odor2_window=None, config: Path | None = None) -> list[Path]:
    """Draw every figure and table; the paths written. ``plus5_scorer`` (rows -> scores
    on odor + 5 s) adds the odor + 5 s figures."""
    scores = {(r.dataset, r.fly, r.fly_number, r.trial_label): r.score for r in pred.itertuples()}
    odor2_window = odor2_window or odor2_lookup(roots)

    def onset(row):
        w = odor2_window(row)
        return None if w is None else w[0]

    started = f"flies started {window}" if window else ""
    tail = f" ({started})" if started else ""
    recs = build_records(pre_df, pair_df, scores, onset)
    print(f"[b2b] {len(recs)} presentations, {int(recs['score'].isna().sum())} without a score; "
          f"{recs.groupby(['dataset', 'fly', 'fly_number']).ngroups} flies")
    for short in ("IAA-first", "IAA-second"):        # a fly that left loses its figure
        for old in (out / short / "per_fly").glob("*.png"):
            old.unlink()
    written = []
    for ds, short in ((FIRST, "IAA-first"), (SECOND, "IAA-second")):
        sub = recs[recs.dataset == ds]
        if sub.empty:
            print(f"[b2b] {ds}: no presentations, no figures")
            continue
        written += per_fly(recs, ds, out / short / "per_fly")
        head = ("IAA first: each odor alone vs right after IAA" if ds == FIRST
                else "IAA second: IAA alone vs right after each odor")
        written.append(dataset_mean(sub, f"{head} — {ds}{tail}",
                                    out / short / f"{short}_dataset_mean.png"))
        for pooled, name in ((False, "scores"), (True, "scores_pooled")):
            written += score_bars(sub, head, out / short / f"{short}_{name}.png", threshold=thr,
                                  xlabel=BAR_XLABEL[ds], pool_presentations=pooled,
                                  started=started)
        for pooled, name in ((False, "change_minus_paraffin"),
                             (True, "change_minus_paraffin_pooled")):
            written += control_bars(sub, CONTROL_HEAD[ds], out / short / f"{short}_{name}.png",
                                    threshold=thr, xlabel=BAR_XLABEL[ds],
                                    pool_presentations=pooled, started=started, group=ds)
    if not recs.empty:
        both_head = ("Both: second odor alone vs in the pair, by partner "
                     "(IAA-first: the partner; IAA-second: IAA)")
        written.append(dataset_mean(recs, both_head + tail, out / "both" / "both_dataset_mean.png"))
        for pooled, name in ((False, "scores"), (True, "scores_pooled")):
            written += score_bars(recs, "Both datasets: the second odor alone vs in the pair",
                                  out / "both" / f"both_{name}.png", threshold=thr,
                                  xlabel=BAR_XLABEL["both"], pool_presentations=pooled,
                                  started=started)
    written += write_primary_endpoint(out, recs, threshold=thr, odor1_scores=scores,
                                      started=started)
    if config is not None:   # Change-Minus-Click/: log AUC and peak-z ratio minus the click
        from scripts.analysis import b2b_minus_click_measures as mc
        written += mc.write(out, pre_df, pair_df, odor2_window, config=config, started=started)
    if plus5_scorer is not None:
        written += plus_after_figure(out, pre_df, pair_df, scores, plus5_scorer, onset,
                                     odor2_window, threshold=thr, started=started)
    out.mkdir(parents=True, exist_ok=True)
    recs.drop(columns=["trace"]).to_csv(out / "b2b_presentations.csv", index=False)
    written.append(out / "b2b_presentations.csv")

    # odor 1 of each pair (nothing precedes it in its trial) vs that odor naive
    o1 = odor1_records(pre_df, pair_df, scores)
    for ds, short in ((FIRST, "IAA-first"), (SECOND, "IAA-second")):
        sub = o1[o1.dataset == ds]
        if sub.empty:
            continue
        for pooled, name in ((False, "odor1_vs_naive"), (True, "odor1_vs_naive_pooled")):
            written += score_bars(sub, ODOR1_HEAD[ds], out / short / f"{short}_{name}.png",
                                  threshold=thr, xlabel=ODOR1_XLABEL[ds],
                                  pool_presentations=pooled, labels=ODOR1_LABELS,
                                  n_words=ODOR1_N_WORDS, started=started)
    o1.drop(columns=["trace"]).to_csv(out / "b2b_odor1_presentations.csv", index=False)
    written.append(out / "b2b_odor1_presentations.csv")

    # the naive pre-test, every odor, IAA-first vs IAA-second flies
    pre = pretest_records(pre_df, scores)
    if not pre.empty:
        written += pretest_bars(pre, out / "pretest" / "pretest_naive_by_group.png", threshold=thr,
                                title="Pre-test (naive): every odor alone, IAA-first vs IAA-second flies",
                                started=started)
        pre.to_csv(out / "pretest" / "pretest_naive_trials.csv", index=False)
        written.append(out / "pretest" / "pretest_naive_trials.csv")

    pt = pair_reactions(pred, thr)
    od = out / "odor1_vs_odor2"
    for ds, short in ((FIRST, "IAA-first"), (SECOND, "IAA-second")):
        if (pt.dataset == ds).any():
            written.append(odor1_vs_odor2_figure(
                pt[pt.dataset == ds],
                f"{short}: does a fly that reacts to odor 1 also react to odor 2?{tail}",
                thr, od / f"{short}_odor1_vs_odor2.png"))
    if not pt.empty:
        written.append(odor1_vs_odor2_figure(
            pt, f"Both datasets: does a fly that reacts to odor 1 also react to odor 2?{tail}", thr,
            od / "both_odor1_vs_odor2.png"))
    od.mkdir(parents=True, exist_ok=True)
    pt.to_csv(od / "pair_trial_reactions.csv", index=False)
    written.append(od / "pair_trial_reactions.csv")
    return written


#: Time-of-Day test-block figure: what each group's comparison scores.
TOD_HEADS = {FIRST: "each odor alone vs right after IAA", SECOND: "IAA alone vs right after each odor"}
TOD_LABELS = {"naive": "Alone (naive pre-test)", "pair": "In pair, 1st + 2nd presentation pooled"}


def extra_sets(out: Path, pre_win, pre_all, pair_all, pred_all, starts, thr: float, roots,
               window: str, odor2_window) -> list[Path]:
    """``Pre-Exposure-Baseline/`` from this folder's flies and
    ``Time-of-Day_2h-bins_whole-dataset/`` from every fly (b2b_baseline_time_of_day)."""
    scores = {(r.dataset, r.fly, r.fly_number, r.trial_label): r.score
              for r in pred_all.itertuples()}
    written = bt.write_baseline(out, pretest_records(pre_win, scores), GROUP_SHORT,
                                threshold=thr, window=window)

    def onset(row):
        w = odor2_window(row)
        return None if w is None else w[0]

    recs = build_records(pre_all, pair_all, scores, onset)
    written += bt.write_time_of_day(
        out, pretest_records(pre_all, scores), recs, starts, roots, GROUP_SHORT, TOD_HEADS,
        threshold=thr, paired_p=sign_flip_p,
        colors={"naive": BAR_COLORS["pre"], "pair": BAR_COLORS["pair"]}, labels=TOD_LABELS)
    return written


def main(argv=None) -> str:
    """'built', 'kept' (nothing changed since the last build) or 'empty' (no batch started
    in the window: only the README is left); SystemExit on a held lock."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv-dir", type=Path, default=CSV_DIR)
    ap.add_argument("--predictions", type=Path, default=None,
                    help="default: <csv-dir>/model_predictions.csv")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "config_new.yaml")
    ap.add_argument("--data-root", type=Path, action="append", default=None,
                    help="where <dataset>/<folder>/ rig frame logs live; repeatable "
                         f"(default {DATA_ROOT}, then {SECURED_ROOT})")
    ap.add_argument("--flagged-flies-csv", default=None,
                    help="default: flagged_flies_csv of --config")
    ap.add_argument("--force", action="store_true", help="redraw even if nothing changed")
    ap.add_argument("--start", type=parse_clock, default=None, metavar="HH:MM",
                    help="only batch folders whose first pre-test started at or after this "
                         "(Pi clock); needs --end")
    ap.add_argument("--end", type=parse_clock, default=None, metavar="HH:MM",
                    help="... and before this (exclusive)")
    a = ap.parse_args(argv)
    if (a.start is None) != (a.end is None):
        ap.error("--start and --end come together")
    window = window_text(a.start, a.end) if a.start is not None else ""
    predictions = a.predictions or a.csv_dir / "model_predictions.csv"
    roots = a.data_root or [DATA_ROOT, SECURED_ROOT]
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    lock = _lock(out)
    if lock is None:
        raise SystemExit(f"[b2b] another build of {out} held the lock past "
                         f"{LOCK_TIMEOUT_S // 60} min; not built")
    try:
        thr, flagged = read_config(a.config)
        flagged = a.flagged_flies_csv if a.flagged_flies_csv is not None else flagged
        pre_all, pair_all, pred_all = load_inputs(a.csv_dir, predictions, flagged)
        starts = folder_starts((pre_all, pair_all), roots)
        inside = (window_mask(starts, a.start, a.end) if window
                  else pd.Series(True, index=starts.index, dtype=bool))
        keep = set(zip(starts.loc[inside, "dataset"], starts.loc[inside, "folder"]))
        pre_df, pair_df, pred = (keep_folders(df, keep) for df in (pre_all, pair_all, pred_all))
        # every fly's rows (Time-of-Day takes them all) and the start table
        model = plus_after_model(a.config)
        if model is None:
            print("[b2b] no reaction_prediction.model_path in the config: "
                  f"odor + {PLUS_AFTER_S:g} s figure skipped")
        fp = fc.combine(fingerprint(pre_all, pair_all, pred_all, thr), f"window={window}",
                        starts.astype(str).to_csv(index=False),
                        fc.file_digest(model) if model else "no model")
        cache = _read_cache(out)
        files = cache.get("files") or []
        if (not a.force and cache.get("fingerprint") == fp and files
                and all((out / f).is_file() for f in files)):
            print(f"[b2b] nothing changed since the last build; {len(files)} files kept in {out}")
            return "kept"
        (out / CACHE_FILE).unlink(missing_ok=True)       # a failed build is not current
        empty = pre_df.empty and pair_df.empty
        scorer = make_plus_after_scorer(model, a.config) if model and not empty else None
        o2w = odor2_lookup(roots)
        written = [] if empty else build(out, pre_df, pair_df, pred, thr, roots, window, scorer,
                                         o2w, config=a.config)
        written += extra_sets(out, pre_df, pre_all, pair_all, pred_all, starts, thr, roots,
                              window, o2w)
        pcsv = out / PRIMARY_DIR / PRIMARY_CSV
        primary = pd.read_csv(pcsv) if pcsv in {Path(w) for w in written} else None
        tests = folder_tests(out, written)
        tests.to_csv(out / BH_CSV, index=False)
        written.append(out / BH_CSV)
        sections = (FIGURE_GUIDE + multiplicity_lines(tests) + DATA_NOTES) if not empty else []
        written.append(write_readme(out, starts, inside, window, primary, sections))
        now = {str(p.relative_to(out)) for p in written}
        for f in set(files) - now:                        # drawn last time, not this time
            (out / f).unlink(missing_ok=True)
        _prune_empty_dirs(out)
        _write_cache(out, fp, sorted(now))
        if empty:
            print(f"[b2b] no batch started {window or 'at all'}: README.md and the "
                  f"whole-dataset sets only, in {out}")
            return "empty"
        print(f"[b2b] wrote {len(written)} files under {out}" + (f" (flies started {window})"
                                                                 if window else ""))
        return "built"
    finally:
        lock.close()


if __name__ == "__main__":
    main()
