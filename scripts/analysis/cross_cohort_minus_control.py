#!/usr/bin/env python3
"""Cross-cohort change minus the control cohort's change, per odor, with a controlled test.

For each exposed cohort and odor:

    (mean of the exposed flies' own post - pre) - (mean of the control flies' own post - pre)

on the same odor. Each fly gives one trial per odor per phase, so a mean over flies is
the trial-pooled mean (never a mean of means). The p-value swaps WHOLE BATCHES (one rig x
one session; Monte Carlo, ``n_perm`` random splits) between the exposed cohort and the
control (``fbpipe.analysis.fly_stats.controlled_change_test``), so flies of one batch
always move together. Batches run on the same day on different rigs still count as
separate units, so day-to-day drift is only partly covered (a day-level shuffle of IAA vs
the control, 2026-11-18 review, gave p 0.18-0.99 against 0.11-0.99 here). 95 % CI: the
shifts that same batch-shuffle test does not reject (test inversion), so a whisker clears
0 exactly when the raw p < 0.05. Every cohort x odor contrast of a panel is one
exploratory family: stars use the Holm-adjusted p; every bar is labelled with its raw p
(user 2026-11-18). ``p_fly_ignores_batches`` in the CSVs shuffles single flies: secondary,
never the result.

The control cohort's exposure protocol is counted from config/paraffin_off_protocol_trials.csv
(batches run before the 2026-11-10 valve fix, whose 'paraffin' trials shut the paraffin
line off for 30 s) and stated on every figure.

Measures, read from each cohort folder's own per-fly table (the numbers that cohort's
figures draw; a fly missing a phase has no change and is left out):

  peakz_during  PeakZ-Before-vs-During/peakz_per_fly_change.csv
                zpeak_during_signed / zpeak_during_unsigned (peak z in the odor window, in
                SDs of the before window)
  peakz_ratio   the same table: zpeak_ratio_signed (peak z during, sign kept, / |peak z
                before|) and zpeak_ratio_unsigned (|during| / |before|)
  log_auc       AUC-LogRatio-Before-vs-During/logratio_per_fly_change.csv
                thr_auc_log_ratio = ln((thresholded AUC during + eps) / (before + eps)),
                eps = the scorer's; the scoring model's AUC-During-Before-LogRatio

Outputs (``--out-dir``/Cross-Cohort unless ``out_dir`` is given):

  cross_cohort_peakz_during_change_minus_control_window.{png,csv}
  cross_cohort_peakz_ratio_change_minus_control_window.{png,csv}
  cross_cohort_log_auc_change_minus_control_window.{png,csv}
  cross_cohort_log_auc_change_window.{png,csv}   every cohort's own change (paired exact
                                                 Wilcoxon, 95 % t CI, Holm stars); unlike
                                                 the peak-z change figure in
                                                 sensitivity_window_results (+-1 SEM)
  <odor>_change_<cohort>_vs_<control>_window.{png,csv}   --focus-cohort: that cohort's own
                                                 sensitized odor, every measure, uncorrected

User request 2026-11-18: "make this [cross_cohort_peakz_during_change_window] but change
minus control and look for significance, also do it for log AUC during/before"; then
"want during/before for both log AUC and I also want peak z" (the peak-z ratio).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fbpipe.analysis import fly_stats as fs  # noqa: E402
from scripts.analysis.phase_terms import SENSITIZED_ODOR  # noqa: E402

# Same slots and inks as sensitivity_window_results (pinned by a test), copied so this
# module never imports that 3000-line script just for colours.
COHORT_PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BAND = "#f4f3f0"

#: measure -> source table, panels (column stem, change y label, minus-control y label),
#: value format, title phrase.
MEASURES: dict[str, dict] = {
    "peakz_during": {
        "source": Path("PeakZ-Before-vs-During") / "peakz_per_fly_change.csv",
        "panels": (
            ("zpeak_during_signed",
             "Change in signed peak z during the odor\n(post − pre, SD of the before window)",
             "Exposed − control change in signed\npeak z during the odor (SD of the\nbefore "
             "window)"),
            ("zpeak_during_unsigned",
             "Change in unsigned peak z during the odor\n(post − pre, SD of the before window)",
             "Exposed − control change in unsigned\npeak z during the odor (SD of the\nbefore "
             "window)"),
        ),
        "fmt": "{:+.2f}",
        "phrase": "the PER-trace peak z-score during the odor",
    },
    "peakz_ratio": {
        "source": Path("PeakZ-Before-vs-During") / "peakz_per_fly_change.csv",
        "panels": (
            ("zpeak_ratio_signed",
             "Change in signed peak-z ratio\n(during / |before|, post − pre)",
             "Exposed − control change in signed\npeak-z ratio (during / |before|)"),
            ("zpeak_ratio_unsigned",
             "Change in unsigned peak-z ratio\n(|during| / |before|, post − pre)",
             "Exposed − control change in unsigned\npeak-z ratio (|during| / |before|)"),
        ),
        "fmt": "{:+.2f}",
        "phrase": "the PER-trace peak z-score, during / before",
    },
    "log_auc": {
        "source": Path("AUC-LogRatio-Before-vs-During") / "logratio_per_fly_change.csv",
        "panels": (
            ("thr_auc_log_ratio",
             "Change in ln((AUC during + ε) / (AUC before + ε))\n(post − pre; thresholded "
             "AUC, the scorer's input)",
             "Exposed − control change in\nln((AUC during + ε) / (AUC before + ε))\n"
             "(thresholded AUC, the scorer's input)"),
        ),
        "fmt": "{:+.2f}",
        "phrase": "log AUC during / before",
    },
}

MINUS_CONTROL_MEASURES = ("peakz_during", "peakz_ratio", "log_auc")
#: Measures that also get every cohort's own change here (peak z's lives in
#: sensitivity_window_results: cross_cohort_peakz_during_change_window).
CHANGE_MEASURES = ("log_auc",)

N_PERM, N_BOOT, SEED = 20000, 10000, 0
PARAFFIN_OFF_MANIFEST = REPO_ROOT / "config" / "paraffin_off_protocol_trials.csv"


def paraffin_off_batches(path: Path = PARAFFIN_OFF_MANIFEST) -> dict[str, set[str]]:
    """dataset -> batch folders whose paraffin trials ran with paraffin shut off (the
    pre-2026-11-10 valve protocol). Empty when the manifest is missing."""
    path = Path(path)
    if not path.is_file():
        return {}
    t = pd.read_csv(path, comment="#")
    t = t[t["protocol"].astype(str) == "paraffin_OFF_30s"]
    return {str(ds): set(g["batch"].astype(str)) for ds, g in t.groupby("dataset")}


def protocol_note(flies: pd.DataFrame, control: str, off: Mapping[str, set]) -> str:
    """How many of the control flies drawn ran the paraffin-off exposure protocol."""
    bad = off.get(control) or set()
    if flies is None or flies.empty or not bad:
        return ""
    f = flies[["batch", "fly_number"]].drop_duplicates()
    batches = set(f["batch"].astype(str))
    off_b = batches & bad
    if not off_b:
        return ""
    off_f = f[f["batch"].astype(str).isin(off_b)]
    return (f"Control exposure: {len(off_b)} of {len(batches)} {control} batches "
            f"({len(off_f)} of {len(f)} flies) ran before the 2026-11-10 valve fix, when each "
            "'paraffin' trial shut the paraffin line off for 30 s; the rest had paraffin "
            "flowing. No odor either way.")


def trained_odor(cohort: str) -> str:
    """The cohort's sensitized odor, display name ("IAA-Sensitivity-1-1" -> "Isoamyl Acetate")."""
    from scripts.analysis.envelope_visuals import _trained_label

    return _trained_label(cohort) or cohort


def cohort_colours(cohorts: Sequence[str]) -> dict[str, str]:
    """Slot by position in the configured list (as sensitivity_window_results does)."""
    return {c: COHORT_PALETTE[i % len(COHORT_PALETTE)] for i, c in enumerate(cohorts)}


# ── tables ───────────────────────────────────────────────────────────────────
def per_fly_changes(out: Path, cohort: str, measure: str, stem: str) -> pd.DataFrame:
    """One row per fly x odor with both phases: cohort, batch, fly_number, odor, pre,
    post, change. Empty when the cohort has no such table (or not this measure)."""
    path = Path(out) / cohort / MEASURES[measure]["source"]
    cols = ["cohort", "batch", "fly_number", "odor", "pre", "post", "change"]
    if not path.is_file():
        return pd.DataFrame(columns=cols)
    t = pd.read_csv(path)
    need = {"fly", "fly_number", "odor", f"{stem}_pre", f"{stem}_post"}
    if not need <= set(t.columns):
        return pd.DataFrame(columns=cols)
    pre = pd.to_numeric(t[f"{stem}_pre"], errors="coerce")
    post = pd.to_numeric(t[f"{stem}_post"], errors="coerce")
    ok = np.isfinite(pre.to_numpy(float)) & np.isfinite(post.to_numpy(float))
    d = pd.DataFrame({"cohort": cohort, "batch": t["fly"].astype(str),
                      "fly_number": pd.to_numeric(t["fly_number"], errors="coerce"),
                      "odor": t["odor"].astype(str), "pre": pre, "post": post})[ok]
    d["change"] = d["post"] - d["pre"]
    return d.reset_index(drop=True)[cols]


def minus_control_table(out: Path, cohorts: Sequence[str], control: str, measure: str, *,
                        n_perm: int = N_PERM, n_boot: int = N_BOOT,
                        seed: int = SEED) -> pd.DataFrame:
    """Per panel x exposed cohort x odor: the controlled contrast and its tests.
    Holm is taken over every contrast of one panel (one exploratory family)."""
    rows = []
    for stem, *_ in MEASURES[measure]["panels"]:
        ctrl = per_fly_changes(out, control, measure, stem)
        if ctrl.empty:
            continue
        panel_rows = []
        for cohort in [c for c in cohorts if c != control]:
            x = per_fly_changes(out, cohort, measure, stem)
            if x.empty:
                continue
            sensitized = trained_odor(cohort)
            for odor in sorted(set(x["odor"]) & set(ctrl["odor"])):
                r = fs.controlled_change_test(x[x["odor"] == odor], ctrl[ctrl["odor"] == odor],
                                              value="change", batch="batch", n_perm=n_perm,
                                              n_boot=n_boot, seed=seed, ci="permutation")
                r = dict(r)
                r["p_fly_ignores_batches"] = r.pop("p_fly")
                panel_rows.append({"measure": measure, "panel": stem, "cohort": cohort,
                                   "control": control, "odor": odor,
                                   "sensitized_odor": odor == sensitized, **r,
                                   "n_perm": n_perm})
        if not panel_rows:
            continue
        holm = fs.holm_adjust([r["p_batch"] for r in panel_rows])
        for r, h in zip(panel_rows, holm):
            r["p_batch_holm"] = float(h)
            r["n_tests_in_family"] = len(panel_rows)
            r["stars"] = fs.stars(h).replace("n.s.", "")
        rows += panel_rows
    return pd.DataFrame(rows)


def change_table(out: Path, cohorts: Sequence[str], measure: str) -> pd.DataFrame:
    """Per panel x cohort x odor: the flies' own mean change, 95 % t CI over flies, exact
    Wilcoxon signed-rank (paired by fly), Holm over every cohort x odor of the panel."""
    rows = []
    for stem, *_ in MEASURES[measure]["panels"]:
        panel_rows = []
        for cohort in cohorts:
            x = per_fly_changes(out, cohort, measure, stem)
            if x.empty:
                continue
            sensitized = trained_odor(cohort)
            for odor, g in x.groupby("odor", sort=True):
                d = g["change"].to_numpy(float)
                m, lo, hi = fs.fly_cluster_mean_ci(d, np.arange(d.size))
                p = fs.wilcoxon_signed_rank_p(d)
                panel_rows.append({
                    "measure": measure, "panel": stem, "cohort": cohort, "odor": odor,
                    "sensitized_odor": odor == sensitized, "n_flies": int(d.size),
                    "n_batches": int(g["batch"].nunique()),
                    "mean_pre": float(g["pre"].mean()), "mean_post": float(g["post"].mean()),
                    "change": m, "ci_lo": lo, "ci_hi": hi,
                    "median_change": float(np.median(d)),
                    "p_wilcoxon": np.nan if p is None else float(p)})
        if not panel_rows:
            continue
        holm = fs.holm_adjust([r["p_wilcoxon"] for r in panel_rows])
        for r, h in zip(panel_rows, holm):
            r["p_wilcoxon_holm"] = float(h)
            r["n_tests_in_family"] = len(panel_rows)
            r["stars"] = fs.stars(h).replace("n.s.", "")
        rows += panel_rows
    return pd.DataFrame(rows)


# ── figure ───────────────────────────────────────────────────────────────────
_MONTHS = {m: i for i, m in enumerate(
    ("january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"), start=1)}


def batch_dates(batches: Sequence[str]) -> tuple[str, str] | None:
    """First and last run date from batch folder names (``november_05_batch_1_rig_3``)."""
    days = []
    for b in batches:
        m = re.match(r"^([a-z]+)_(\d{1,2})_batch", str(b).lower())
        if m and m.group(1) in _MONTHS:
            days.append((_MONTHS[m.group(1)], int(m.group(2))))
    if not days:
        return None
    # folder names carry no year: these cohorts all ran in 2026, sorted by (month, day)
    fmt = lambda md: f"{pd.Timestamp(2026, *md):%b} {md[1]}"  # noqa: E731
    return fmt(min(days)), fmt(max(days))


def odor_order(table: pd.DataFrame, cohorts: Sequence[str]) -> list[str]:
    """Each cohort's own exposed odor in cohort order, then the rest alphabetically."""
    first: list[str] = []
    for c in cohorts:
        hit = table.loc[(table["cohort"] == c) & table["sensitized_odor"].astype(bool), "odor"]
        for o in hit:                    # a table with two panels lists each odor twice
            if o not in first:
                first.append(o)
    return first + sorted(set(table["odor"]) - set(first))


def _overlaps(a: tuple[str, str], b: tuple[str, str]) -> bool:
    day = lambda s: pd.Timestamp(f"{s} 2026")  # noqa: E731
    return day(a[0]) <= day(b[1]) and day(b[0]) <= day(a[1])


def run_date_note(table: pd.DataFrame, cohorts: Sequence[str], control: str) -> str:
    """'Run dates: IAA Oct 9–Nov 1 (never overlaps the control); …; Paraffin control …'."""
    ctrl = batch_dates(_batches_of(table, control, control))
    names = short_names([*cohorts, control])
    parts = []
    for c in cohorts:
        d = batch_dates(_batches_of(table, c, control))
        if not d:
            continue
        flag = " (never overlaps the control)" if ctrl and not _overlaps(d, ctrl) else ""
        parts.append(f"{names[c]} {d[0]}–{d[1]}{flag}")
    if ctrl:
        parts.append(f"{names[control]} control {ctrl[0]}–{ctrl[1]}")
    return ("Run dates: " + "; ".join(parts) + ".") if parts else ""


def _p_short(p: float) -> str:
    """Three decimals below 0.1, so a p under 0.05 never prints as 'p=0.05'."""
    if p is None or not np.isfinite(p):
        return ""
    return "p<0.001" if p < 0.001 else f"p={p:.3f}" if p < 0.1 else f"p={p:.2f}"


def _style(ax, n_odors: int, odors: Sequence[str], ylabel: str) -> None:
    ax.axhline(0, color=INK_2, lw=1.0, zorder=1)
    ax.set_xlim(-0.5, n_odors - 0.5)
    ax.set_xticks(range(n_odors))
    ax.set_xticklabels(odors, fontsize=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=10, color=INK)
    lo_, hi_ = ax.get_ylim()
    pad = 0.10 * (hi_ - lo_)
    ax.set_ylim(lo_ - pad, hi_ + pad)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2)


def _bars(ax, table: pd.DataFrame, present: Sequence[str], odors: Sequence[str],
          colours: Mapping[str, str], *, value: str, lo: str, hi: str, raw_p: str,
          fmt: str) -> None:
    k = len(present)
    step = 0.84 / max(k, 1)
    by = table.set_index(["cohort", "odor"])
    for i in range(len(odors)):
        if i % 2:
            ax.axvspan(i - 0.5, i + 0.5, color=BAND, zorder=0, lw=0)
    for j, c in enumerate(present):
        for i, odor in enumerate(odors):
            if (c, odor) not in by.index:
                continue
            r = by.loc[(c, odor)]
            x = i + (j - (k - 1) / 2) * step
            y = float(r[value])
            sens = bool(r["sensitized_odor"])
            ax.bar(x, y, width=step * 0.86, color=colours[c], edgecolor=INK,
                   linewidth=1.6 if sens else 0, zorder=2)
            y_lo, y_hi = float(r[lo]), float(r[hi])
            if np.isfinite(y_lo) and np.isfinite(y_hi):
                ax.errorbar(x, y, yerr=[[max(0.0, y - y_lo)], [max(0.0, y_hi - y)]],
                            fmt="none", ecolor=INK_2, elinewidth=1.0, zorder=3)
            tip = (y_hi if np.isfinite(y_hi) else y) if y >= 0 else (y_lo if np.isfinite(y_lo)
                                                                    else y)
            # every bar carries its raw p (user 2026-11-18: "check p values all odors");
            # the stars are the Holm-adjusted ones; the exposed odor also gets its value
            stars = "" if pd.isna(r["stars"]) else str(r["stars"])
            p_txt = _p_short(float(r[raw_p]))
            label = f"{fmt.format(y)}{stars}\n{p_txt}" if sens else f"{p_txt}{stars}"
            if label.strip():
                ax.annotate(label, (x, tip), xytext=(0, 3 if y >= 0 else -3),
                            textcoords="offset points", ha="center",
                            va="bottom" if y >= 0 else "top", fontsize=8 if sens else 6.8,
                            color=INK, fontweight="bold" if sens else "normal",
                            linespacing=0.95)


def plot_minus_control(table: pd.DataFrame, cohorts: Sequence[str], control: str,
                       measure: str, title: str, path: Path, protocol: str = "") -> Path:
    """One panel per measure stem; a bar per exposed cohort at every odor."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    spec = MEASURES[measure]
    colours = cohort_colours(cohorts)
    present = [c for c in cohorts if c != control and c in set(table["cohort"])]
    odors = odor_order(table, present)
    panels = [(s, y) for s, _, y in spec["panels"] if s in set(table["panel"])]
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(max(8.0, 1.45 * len(odors) + 2.0), 1.6 + 4.3 * n),
                             squeeze=False, gridspec_kw={"hspace": 0.34})
    for ax, (stem, ylab) in zip(axes[:, 0], panels):
        t = table[table["panel"] == stem]
        _bars(ax, t, present, odors, colours, value="estimate", lo="ci_lo", hi="ci_hi",
              raw_p="p_batch", fmt=spec["fmt"])
        _style(ax, len(odors), odors, ylab)
        ax.set_title(f"{int(t['n_tests_in_family'].iloc[0])} contrasts; * = Holm-adjusted "
                     "p < 0.05", loc="right", fontsize=8.5, color=INK_2)
    first = table[table["panel"] == panels[0][0]]
    ctrl_n = first.groupby("cohort")[["n_flies_control", "n_batches_control"]].max()
    exp_n = first.groupby("cohort")[["n_flies_exposed", "n_batches_exposed"]].max()
    handles = [Patch(color=colours[c], label=(
        f"{c} (n={int(exp_n.loc[c, 'n_flies_exposed'])} flies / "
        f"{int(exp_n.loc[c, 'n_batches_exposed'])} batches, sensitized to {trained_odor(c)})"))
        for c in present]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.95 / fig.get_figheight()),
               ncol=min(2, len(present)), frameon=False, fontsize=9.5, labelcolor=INK)
    c_n = (int(ctrl_n["n_flies_control"].max()), int(ctrl_n["n_batches_control"].max()))
    fig.suptitle(title, fontsize=13, color=INK, y=1 - 0.12 / fig.get_figheight())
    perm = ("every batch split enumerated" if table["perm_exact"].astype(bool).all()
            else f"{int(table['n_perm'].iloc[0]):,} random batch splits")
    note = (f"Bar = exposed flies' mean post − pre minus {control} flies' mean post − pre on the "
            f"same odor (control n≤{c_n[0]} flies / {c_n[1]} batches). p: whole batches (rig × "
            f"session) shuffled between the two cohorts ({perm}); whiskers: 95% CI from "
            "inverting that test (clears 0 exactly when the raw p<0.05). Every bar: its raw p; "
            "* p<0.05, ** p<0.01, *** p<0.001 only after Holm over every contrast in the panel "
            f"(exploratory). Black outline = the cohort's own {SENSITIZED_ODOR} (value shown).")
    extra = run_date_note(table, present, control)
    fewer = []
    for c in present:
        g = first[first["cohort"] == c]
        top = int(g["n_flies_exposed"].max())
        for r in g[g["n_flies_exposed"] < top].itertuples():
            fewer.append(f"{short_names(cohorts)[c]} {r.odor} {int(r.n_flies_exposed)} flies / "
                         f"{int(r.n_batches_exposed)} batches")
    if fewer:
        extra += (" Fewer flies on: " + "; ".join(fewer) + " (that odor was only in the "
                  "panel of later batches).")
    if protocol:
        extra += " " + protocol
    if "Paraffin Oil" in odors:
        extra += (" The Paraffin Oil test trials of batches run before the fix also had "
                  "paraffin shut off, on both sides of that column.")
    fig.text(0.5, 1 - 0.42 / fig.get_figheight(), _wrap(note, fig), ha="center", va="top",
             fontsize=8.3, color=INK_2)
    fig.text(0.5, 0.012, _wrap(extra, fig), ha="center", va="bottom", fontsize=8.3,
             color=INK_2)
    fig.subplots_adjust(top=1 - 1.75 / fig.get_figheight(), bottom=0.95 / fig.get_figheight(),
                        left=0.09, right=0.98)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return Path(path)


def _wrap(text: str, fig) -> str:
    """Wrap a note to the figure's width (8.3 pt text: about 15 characters per inch)."""
    import textwrap

    return textwrap.fill(text, width=max(60, int(fig.get_figwidth() * 15)))


def _short(cohort: str) -> str:
    return cohort.split("-")[0]


def short_names(cohorts: Sequence[str]) -> dict[str, str]:
    """Cohort -> its prefix ("IAA"), or the full name when two cohorts share a prefix."""
    pref = [_short(c) for c in cohorts]
    return {c: (p if pref.count(p) == 1 else c) for c, p in zip(cohorts, pref)}


def _batches_of(table: pd.DataFrame, cohort: str, control: str) -> list[str]:
    """Batch folders behind ``cohort``'s rows (the control's: every row's control side)."""
    col = "batches_control" if cohort == control else "batches_exposed"
    if col not in table.columns:
        return []
    vals = table[col] if cohort == control else table.loc[table["cohort"] == cohort, col]
    out: set[str] = set()
    for v in vals.dropna():
        out |= {b for b in str(v).split(";") if b}
    return sorted(out)


def plot_change(table: pd.DataFrame, cohorts: Sequence[str], measure: str, title: str,
                path: Path) -> Path:
    """Every cohort's own mean change (the twin of the peak-z change figure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    spec = MEASURES[measure]
    colours = cohort_colours(cohorts)
    present = [c for c in cohorts if c in set(table["cohort"])]
    odors = odor_order(table, present)
    panels = [(s, y) for s, y, _ in spec["panels"] if s in set(table["panel"])]
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(max(8.0, 1.45 * len(odors) + 2.0), 1.6 + 4.3 * n),
                             squeeze=False, gridspec_kw={"hspace": 0.34})
    for ax, (stem, ylab) in zip(axes[:, 0], panels):
        t = table[table["panel"] == stem]
        _bars(ax, t, present, odors, colours, value="change", lo="ci_lo", hi="ci_hi",
              raw_p="p_wilcoxon", fmt=spec["fmt"])
        _style(ax, len(odors), odors, ylab)
        ax.set_title(f"{int(t['n_tests_in_family'].iloc[0])} tests; * = Holm-adjusted p < 0.05",
                     loc="right", fontsize=8.5, color=INK_2)
    nf = table.groupby("cohort")["n_flies"].max()
    handles = [Patch(color=colours[c], label=f"{c}  (n={int(nf[c])} flies, sensitized to "
                                             f"{trained_odor(c)})") for c in present]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1 - 0.95 / fig.get_figheight()),
               ncol=min(2, len(present)), frameon=False, fontsize=9.5, labelcolor=INK)
    fig.suptitle(title, fontsize=13, color=INK, y=1 - 0.12 / fig.get_figheight())
    note = ("Bars: mean of each fly's own post − pre (one trial per fly per odor per phase). "
            "Whiskers: 95% t CI over flies. Exact Wilcoxon signed-rank, paired by fly; "
            "every bar: its raw p; * p<0.05, ** p<0.01, *** p<0.001 only after Holm over every "
            f"bar in the panel (exploratory). Black outline = the cohort's own {SENSITIZED_ODOR} "
            "(value shown). Not controlled for time or re-testing: see the minus-control figure.")
    fig.text(0.5, 1 - 0.42 / fig.get_figheight(), _wrap(note, fig), ha="center", va="top",
             fontsize=8.3, color=INK_2)
    fig.subplots_adjust(top=1 - 1.75 / fig.get_figheight(), bottom=0.45 / fig.get_figheight(),
                        left=0.09, right=0.98)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return Path(path)


# ── one odor, exposed cohort vs control, every measure (uncorrected) ─────────
#: (panel key, source, column/stem, panel title, scale, value format). "window" reads
#: Data/window_per_fly_per_odor.csv (score / responded); the rest the cohort folders'
#: per-fly tables, as above.
FOCUS_PANELS = (
    ("responded", "window", "responded",
     "% of flies responding (score ≥ 2)\nchange, percentage points  [primary endpoint]",
     100.0, "{:+.1f}"),
    ("score", "window", "score", "Mean PER score\nchange (post − pre)", 1.0, "{:+.2f}"),
    ("zpeak_during_signed", "peakz_during", "zpeak_during_signed",
     "Signed peak z during the odor\nchange (SD of the before window)", 1.0, "{:+.2f}"),
    ("zpeak_during_unsigned", "peakz_during", "zpeak_during_unsigned",
     "Unsigned peak z during the odor\nchange (SD of the before window)", 1.0, "{:+.2f}"),
    ("zpeak_ratio_signed", "peakz_ratio", "zpeak_ratio_signed",
     "Signed peak-z ratio, during / |before|\nchange (post − pre)", 1.0, "{:+.2f}"),
    ("zpeak_ratio_unsigned", "peakz_ratio", "zpeak_ratio_unsigned",
     "Unsigned peak-z ratio, |during| / |before|\nchange (post − pre)", 1.0, "{:+.2f}"),
    ("thr_auc_log_ratio", "log_auc", "thr_auc_log_ratio",
     "ln((AUC during + ε) / (AUC before + ε))\nchange (post − pre; the scorer's input)", 1.0,
     "{:+.2f}"),
)
WINDOW_TABLE = Path("Data") / "window_per_fly_per_odor.csv"


def window_changes(out: Path, cohort: str, odor: str, kind: str) -> pd.DataFrame:
    """Per fly (window flies, one trial per phase) for ``kind`` "score" or "responded":
    batch, fly_number, pre, post, change."""
    path = Path(out) / WINDOW_TABLE
    cols = ["batch", "fly_number", "pre", "post", "change"]
    if not path.is_file():
        return pd.DataFrame(columns=cols)
    w = pd.read_csv(path)
    g = w[(w["cohort"] == cohort) & (w["odor"] == odor)]
    if kind == "responded":
        pre = g["responded_pre"].astype(str).str.lower().eq("true")
        post = g["responded_post"].astype(str).str.lower().eq("true")
    else:
        pre = pd.to_numeric(g["score_pre"], errors="coerce")
        post = pd.to_numeric(g["score_post"], errors="coerce")
    d = pd.DataFrame({"batch": g["batch_folder"].astype(str), "fly_number": g["fly_number"],
                      "pre": pre, "post": post})
    d = d.dropna(subset=["pre", "post"])
    d["change"] = d["post"].astype(float) - d["pre"].astype(float)
    return d.reset_index(drop=True)[cols]


def focus_table(out: Path, exposed: str, control: str, odor: str, *, n_perm: int = N_PERM,
                seed: int = SEED) -> pd.DataFrame:
    """One row per measure: each cohort's own change on ``odor`` (mean over flies, 95 % t
    CI over flies, its own paired p) and exposed minus control (whole batches shuffled,
    inversion CI, fly-level p). Every p is uncorrected: one pre-chosen comparison per
    panel, the user's request of 2026-11-18."""
    rows = []
    for key, source, stem, label, scale, fmt in FOCUS_PANELS:
        if source == "window":
            x = window_changes(out, exposed, odor, stem)
            c = window_changes(out, control, odor, stem)
        else:
            x = per_fly_changes(out, exposed, source, stem)
            c = per_fly_changes(out, control, source, stem)
            x, c = x[x["odor"] == odor], c[c["odor"] == odor]
        if x.empty or c.empty:
            continue
        xv = scale * x["change"].to_numpy(float)
        cv = scale * c["change"].to_numpy(float)
        mx, lox, hix = fs.fly_cluster_mean_ci(xv, np.arange(xv.size))
        mc, loc, hic = fs.fly_cluster_mean_ci(cv, np.arange(cv.size))
        if stem == "responded":
            px = fs.mcnemar_exact_p(x["pre"].astype(bool), x["post"].astype(bool))
            pc = fs.mcnemar_exact_p(c["pre"].astype(bool), c["post"].astype(bool))
            own = "exact McNemar"
        else:
            px = fs.wilcoxon_signed_rank_p(x["change"])
            pc = fs.wilcoxon_signed_rank_p(c["change"])
            own = "exact Wilcoxon signed-rank"
        r = fs.controlled_change_test(x.assign(v=xv), c.assign(v=cv), value="v", batch="batch",
                                      n_perm=n_perm, seed=seed, ci="permutation")
        rows.append({
            "panel": key, "label": label, "odor": odor, "exposed": exposed, "control": control,
            "n_exposed": int(xv.size), "n_control": int(cv.size),
            "n_batches_exposed": r["n_batches_exposed"], "n_batches_control": r["n_batches_control"],
            "exposed_pre": float(scale * x["pre"].astype(float).mean()),
            "exposed_post": float(scale * x["post"].astype(float).mean()),
            "exposed_change": mx, "exposed_lo": lox, "exposed_hi": hix,
            "p_exposed_own": np.nan if px is None else float(px),
            "control_pre": float(scale * c["pre"].astype(float).mean()),
            "control_post": float(scale * c["post"].astype(float).mean()),
            "control_change": mc, "control_lo": loc, "control_hi": hic,
            "p_control_own": np.nan if pc is None else float(pc), "own_test": own,
            "estimate": r["estimate"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
            "p_batch": r["p_batch"], "p_fly_ignores_batches": r["p_fly"],
            "perm_exact": r["perm_exact"], "n_perm": n_perm, "fmt": fmt})
    return pd.DataFrame(rows)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def focus_stem(exposed: str, control: str, odor: str) -> str:
    names = short_names([exposed, control])
    keep = lambda n: re.sub(r"[^A-Za-z0-9.]+", "_", n).strip("_")  # noqa: E731
    return f"{_slug(odor)}_change_{keep(names[exposed])}_vs_{keep(names[control])}_window"


def plot_focus(table: pd.DataFrame, cohorts: Sequence[str], exposed: str, control: str,
               odor: str, title: str, path: Path, note: str = "") -> Path:
    """A panel per measure: the two cohorts' own change on ``odor`` and the bracket with
    exposed minus control. Uncorrected p everywhere (one comparison per panel)."""
    import math as _m

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = cohort_colours(cohorts)
    n = len(table)
    ncols = 4
    nrows = _m.ceil((n + 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.3 * nrows + 1.3),
                             squeeze=False, gridspec_kw={"hspace": 0.62, "wspace": 0.42})
    flat = axes.ravel()
    for ax, r in zip(flat, table.itertuples()):
        fmt = r.fmt
        groups = ((exposed, r.exposed_change, r.exposed_lo, r.exposed_hi, r.p_exposed_own,
                   r.n_exposed, r.n_batches_exposed),
                  (control, r.control_change, r.control_lo, r.control_hi, r.p_control_own,
                   r.n_control, r.n_batches_control))
        tops, bots = [0.0], [0.0]
        for i, (c, y, lo, hi, p, nf, nb) in enumerate(groups):
            ax.bar(i, y, width=0.62, color=colours[c], zorder=2)
            ax.errorbar(i, y, yerr=[[max(0.0, y - lo)], [max(0.0, hi - y)]], fmt="none",
                        ecolor=INK_2, elinewidth=1.1, capsize=0, zorder=3)
            tip = hi if y >= 0 else lo
            ax.annotate(f"{fmt.format(y)}\n{_p_short(p)}", (i, tip),
                        xytext=(0, 3 if y >= 0 else -3), textcoords="offset points",
                        ha="center", va="bottom" if y >= 0 else "top", fontsize=8.5,
                        color=INK, linespacing=0.95)
            tops.append(hi)
            bots.append(lo)
        span = max(tops) - min(bots)
        span = span if span > 0 else 1.0
        y_br = max(tops) + 0.30 * span
        ax.plot([0, 0, 1, 1], [y_br - 0.04 * span, y_br, y_br, y_br - 0.04 * span],
                color=INK, lw=1.0, zorder=3)
        ax.text(0.5, y_br + 0.03 * span,
                f"Δ {fmt.format(r.estimate)}  [{fmt.format(r.ci_lo)}, {fmt.format(r.ci_hi)}]\n"
                f"{_p_short(r.p_batch)} (batches shuffled)",
                ha="center", va="bottom", fontsize=8.6, color=INK,
                fontweight="bold" if r.p_batch < 0.05 else "normal")
        ax.set_ylim(min(bots) - 0.22 * span, y_br + 0.42 * span)
        ax.axhline(0, color=INK_2, lw=1.0, zorder=1)
        ax.set_xlim(-0.6, 1.6)
        ax.set_xticks([0, 1])
        names = short_names([exposed, control])
        ax.set_xticklabels([f"{names[exposed]} cohort\nn={r.n_exposed} flies",
                            f"{names[control]} control\nn={r.n_control} flies"],
                           fontsize=8.6, color=INK)
        ax.set_title(r.label, fontsize=9.3, color=INK, loc="left")
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_2)
        ax.tick_params(colors=INK_2, labelsize=8.5)
    for ax in flat[n:]:
        ax.axis("off")
    text = (f"Every fly met {odor} once before and once after exposure. Bars: each group's "
            "mean change (post − pre); whiskers: 95% t CI over flies; under each value: that "
            "group's own pre-vs-post p (exact Wilcoxon signed-rank for the score and the trace "
            "measures, exact McNemar for % responding). Bracket: exposed − control difference, "
            "its 95% CI from inverting the test, and p with whole batches (rig × session) "
            f"shuffled between the two cohorts ({int(table['n_perm'].iloc[0]):,} random splits). "
            "All p uncorrected: one comparison per panel (% responding is the pre-specified "
            "primary endpoint). " + note)
    flat[n].text(0.0, 1.0, _wrap_width(text, 52), ha="left", va="top", fontsize=8.6,
                 color=INK_2, transform=flat[n].transAxes)
    fig.suptitle(title, fontsize=13, color=INK, y=0.995)
    fig.subplots_adjust(top=1 - 1.0 / fig.get_figheight(), bottom=0.07, left=0.06, right=0.985)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return Path(path)


def _wrap_width(text: str, width: int) -> str:
    import textwrap

    return textwrap.fill(text, width=width)


# ── driver ───────────────────────────────────────────────────────────────────
def write(out: Path, cohorts: Sequence[str], control: str | None, *, window: str,
          out_dir: Path | None = None, n_perm: int = N_PERM, n_boot: int = N_BOOT,
          seed: int = SEED, focus: Sequence[str] = ()) -> list[Path]:
    """Write every figure + CSV this module makes into ``out_dir`` (default
    ``out``/Cross-Cohort); returns the PNG paths."""
    out = Path(out)
    dest = Path(out_dir) if out_dir is not None else out / "Cross-Cohort"
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    off = paraffin_off_batches()
    if control:
        for measure in MINUS_CONTROL_MEASURES:
            t = minus_control_table(out, cohorts, control, measure, n_perm=n_perm,
                                    n_boot=n_boot, seed=seed)
            if t.empty:
                continue
            t = _with_batches(t, out, cohorts, control, measure)
            stem = f"cross_cohort_{measure}_change_minus_control_window"
            t.to_csv(dest / f"{stem}.csv", index=False)
            ctrl_flies = per_fly_changes(out, control, measure, t["panel"].iloc[0])
            written.append(plot_minus_control(
                t, cohorts, control, measure,
                f"Pre- to post-exposure change in {MEASURES[measure]['phrase']}, minus "
                f"{control}'s change, every odor: flies started {window}",
                dest / f"{stem}.png", protocol=protocol_note(ctrl_flies, control, off)))
    for measure in CHANGE_MEASURES:
        t = change_table(out, cohorts, measure)
        if t.empty:
            continue
        stem = f"cross_cohort_{measure}_change_window"
        t.to_csv(dest / f"{stem}.csv", index=False)
        written.append(plot_change(
            t, cohorts, measure,
            f"Pre- to post-exposure change in {MEASURES[measure]['phrase']}, every odor: "
            f"flies started {window}", dest / f"{stem}.png"))
    for exposed in (focus if control else ()):
        odor = trained_odor(exposed)
        t = focus_table(out, exposed, control, odor, n_perm=n_perm, seed=seed)
        if t.empty:
            continue
        stem = focus_stem(exposed, control, odor)
        t.drop(columns=["fmt"]).to_csv(dest / f"{stem}.csv", index=False)
        dates = run_date_note(_focus_batches(out, exposed, control, odor), [exposed], control)
        proto = protocol_note(window_changes(out, control, odor, "score"), control, off)
        written.append(plot_focus(
            t, cohorts, exposed, control, odor,
            f"{odor}: change in {exposed} flies vs {control} flies, pre- to post-exposure: "
            f"flies started {window} (uncorrected p)", dest / f"{stem}.png",
            note=" ".join(x for x in (dates, proto) if x)))
    return written


def _focus_batches(out: Path, exposed: str, control: str, odor: str) -> pd.DataFrame:
    """The batch folders behind the focus figure, shaped for run_date_note."""
    x = window_changes(out, exposed, odor, "score")
    c = window_changes(out, control, odor, "score")
    return pd.DataFrame({"cohort": [exposed],
                         "batches_exposed": [";".join(sorted(set(x["batch"])))],
                         "batches_control": [";".join(sorted(set(c["batch"])))]})


def _with_batches(t: pd.DataFrame, out: Path, cohorts: Sequence[str], control: str,
                  measure: str) -> pd.DataFrame:
    """Add the batch folders behind each row (for the run-date note and the CSV), taken
    from that row's own panel: a panel with other missing data gets its own lists."""
    def lists(cohort: str, stem: str) -> pd.Series:
        x = per_fly_changes(out, cohort, measure, stem)
        return x.groupby("odor")["batch"].agg(lambda s: ";".join(sorted(set(s))))

    cache: dict[tuple[str, str], pd.Series] = {}
    exp_col, ctrl_col = [], []
    for stem, c, o in zip(t["panel"], t["cohort"], t["odor"]):
        for key in ((c, stem), (control, stem)):
            if key not in cache:
                cache[key] = lists(*key)
        exp_col.append(cache[(c, stem)].get(o, ""))
        ctrl_col.append(cache[(control, stem)].get(o, ""))
    t = t.copy()
    t["batches_exposed"] = exp_col
    t["batches_control"] = ctrl_col
    return t


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out-dir", required=True, type=Path,
                   help="the sensitivity results folder (holds one folder per cohort)")
    p.add_argument("--cohort", action="append", required=True,
                   help="every cohort in the configured order, control included (repeatable)")
    p.add_argument("--control", default=None, help="the control cohort")
    p.add_argument("--window", default="09:40–16:20", help="start window, for the titles")
    p.add_argument("--dest", type=Path, default=None,
                   help="output folder (default: <out-dir>/Cross-Cohort)")
    p.add_argument("--n-perm", type=int, default=N_PERM)
    p.add_argument("--n-boot", type=int, default=N_BOOT,
                   help="unused by the figures (their CIs invert the permutation test)")
    p.add_argument("--focus-cohort", action="append", default=[],
                   help="also draw this cohort's own sensitized odor vs the control, every "
                        "measure, uncorrected p (repeatable)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = write(args.out_dir, args.cohort, args.control, window=args.window,
                  out_dir=args.dest, n_perm=args.n_perm, n_boot=args.n_boot,
                  focus=args.focus_cohort)
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
