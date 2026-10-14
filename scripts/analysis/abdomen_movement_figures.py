"""The ``Abdomen-Movement/`` figure set and the "Abdomen movement" section of ``report.md``.

Takes the two tables of ``mounting_posture.py`` -- the fly-trial table (``fly_trial_posture.csv``) and the fly table
(``fly_posture.csv``) -- and draws six figures, each titled with the QUESTION and answered in plain words computed from
the data (``Yes -- flies tilted more head-up rest more curled`` / ``No clear link``), the numbers in a smaller line:

1. ``abdomen_rest_bend_per_fly``          strip plot of the resting curl (bend) by rig;
2. ``mount_pitch_vs_abdomen_rest_bend``    per fly, two panels: rig-centred mounting angle vs resting curl / tip pitch;
3. ``abdomen_curl_events_by_window``       pooled curl / straightening events per minute just before vs at the start of the
                                          odor and of the light (equal-length windows);
4. ``abdomen_vs_spontaneous_per``          per fly: spontaneous curl rate vs spontaneous PER rate;
5. ``per_abdomen_coupling``                per-fly median proboscis-abdomen coupling by the fly's dominant proboscis state;
6. ``mount_pitch_vs_abdomen_activity``     per fly, two panels: rig-centred mounting angle vs bend SD / tip moving fraction.

The conventions are those of ``mounting_posture_figures`` (whose primitives this module imports). The fly is the unit of
inference: the per-fly scatters use the fly table, the rig-centred mounting angle (the tip pitch is rig-centred too:
camera roll shifts both) and a permutation p stratified by dataset x BATCH (flies of one batch share the mounting day, the
glue session and the starvation, so only they are interchangeable) that alone decides the verdict, the dataset x rig p
being printed beside it as a secondary number; with no batch holding two flies there is nothing to shuffle and the answer
is "Not enough data to tell". Event rates pool TRIALS (events over minutes of the same trials, never a mean of per-trial
rates).

Figure 3 is a placebo design with a leave-one-trial-out baseline. For each odor or light trial L is the shorter of the
stimulus and half the pre-stimulus window. The trial's resting bend is the median bend over the early baseline windows
(the part of the pre-stimulus window before its last L s) of the fly's OTHER trials, pooled over frames, so it depends
neither on the trial's own two windows nor on the frames next to them; curls and straightenings are counted against that
one resting bend in the last L s before the stimulus and in the first L s of it (``fbpipe.analysis.mounting.placebo_fly``,
columns written by ``mounting_posture.py``). The p is the Monte-Carlo window-label swap permutation of
``paired_window_permutation(scheme="swap")``. The design assumes WITHIN-TRIAL STATIONARITY: with no stimulus effect the
two windows are interchangeable. A baseline taken from the whole pre-stimulus window (the first design) or from the early
part of the same trial (the second) made a stimulus that changes nothing look significant (13 to 40 % of simulated runs at
40 pooled trials for the first, up to 64 % at 400 trials for the second); a bend that drifts steadily inside the trials
still does, and more so the more trials are pooled. The randomisation is by trial (trials, not flies, are shuffled). A fly
needs at least 3 trials with a known stimulus onset (a trial is scored only when at least 2 of the fly's other trials have
a usable baseline window); a fly whose identity is not consistent does not enter.

Only quality-ok windows of identity-consistent flies enter a number. Every p is uncorrected, and says so; a comparison
with fewer than three flies is "Not enough data to tell". Figure 5 is descriptive.

``compute_abdomen_stats`` does all the arithmetic once; ``build_abdomen_figures`` / ``render_abdomen_figures`` and
``abdomen_report_section`` only format it. An input without a usable abdomen window draws nothing (and removes the six
pictures of an earlier run); a figure that cannot be drawn for want of flies is a placeholder that says why; one that
fails to build leaves no picture behind and is named in the section.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fbpipe.analysis import mounting as M  # noqa: E402
from scripts.analysis import mounting_posture_figures as FIG  # noqa: E402

LOGGER = logging.getLogger("abdomen_movement_figures")

NAMES = ("abdomen_rest_bend_per_fly", "mount_pitch_vs_abdomen_rest_bend", "abdomen_curl_events_by_window",
         "abdomen_vs_spontaneous_per", "per_abdomen_coupling", "mount_pitch_vs_abdomen_activity")
TITLES = {
    "abdomen_rest_bend_per_fly": "How curled is the abdomen at rest on each rig?",
    "mount_pitch_vs_abdomen_rest_bend": "Does mounting angle change how the abdomen rests?",
    "abdomen_curl_events_by_window": "Does the abdomen curl or straighten when the stimulus starts?",
    "abdomen_vs_spontaneous_per": "Are flies restless in both the abdomen and the proboscis?",
    "per_abdomen_coupling": "Do the proboscis and the abdomen move together?",
    "mount_pitch_vs_abdomen_activity": "Does mounting angle change how active the abdomen is?",
}
KINDS = ("odor", "light")                       # the stimulus kinds with a figure-3 pair; odor + light is in neither
EVENTS = ("curl", "straighten")
EVENT_LABELS = {"curl": "Curls", "straighten": "Straightenings"}
EVENT_NOUN = {"curl": "Curl", "straighten": "Straightening"}
WINDOW_LABELS = {"odor": "the odor", "light": "the light"}
WINDOWS = ("placebo", "during")                 # the last L s before the stimulus, the first L s of it (one baseline)
WINDOW_NAMES = {"placebo": "Just before", "during": "Start of"}
MINUTES = "pb_window_min"                       # L in minutes: the length of both windows of a trial
MORE, LESS = "more often than just before the stimulus", "less often than just before the stimulus"
CENTRED_TIP = "tip_pitch_centred_deg"           # the fly table's tip pitch minus its rig's median (derived here)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _flag(df: pd.DataFrame, name: str) -> np.ndarray:
    """A quality flag of *df* as booleans (False for every row when the column is absent)."""
    return FIG.as_flag(df[name]).to_numpy(dtype=bool) if name in df else np.zeros(len(df), dtype=bool)


def _signed(v: float, digits: int = 2) -> str:
    return f"{v:+.{digits}f}" if np.isfinite(v) else "n/a"


def _signed_rate(v: float) -> str:
    """A difference of two rates (events per minute) with its sign and the digits of ``num_text``; never ``-0.00``."""
    if not np.isfinite(v):
        return "n/a"
    text = FIG.num_text(abs(v))
    return ("-" if v < 0 and float(text) > 0 else "+") + text


def _seconds_text(w: Mapping, *, median: bool = True) -> str:
    """``15 s`` / ``3.8–15 s (median 7.5 s)`` / ``n/a``: the window length L of one stimulus kind, to 0.1 s."""
    if not np.isfinite(w["median"]):
        return "n/a"
    low, mid, high = (round(float(w[k]), 1) for k in ("min", "median", "max"))
    if low == high:
        return f"{low:g} s"
    return f"{low:g}–{high:g} s" + (f" (median {mid:g} s)" if median else "")


def _centred_tip(fly: pd.DataFrame) -> np.ndarray:
    """The resting tip pitch of every fly minus its rig's median (camera roll shifts it like the mount pitch)."""
    tip = pd.Series(FIG.float_column(fly, "tip_pitch_rest_deg"), index=fly.index)
    return (tip - tip.groupby(fly["rig"].astype(str)).transform("median")).to_numpy()


def _scatter(fly: pd.DataFrame, column: str, *, x: str = "mount_pitch_centred_deg", n_perm: int, seed: int,
             up: str, down: str) -> dict:
    """``FIG.scatter_stats`` (permutation p stratified by dataset x batch, the dataset x rig p beside it; no batch with two
    flies to shuffle is "Not enough data to tell", never "No clear link")."""
    return FIG.scatter_stats(fly, column, x=x, n_perm=n_perm, seed=seed, up=up, down=down)


# ---------------------------------------------------------------------------
# Statistics (the one place any number is computed)
# ---------------------------------------------------------------------------

def _count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def _window_counts(trials: pd.DataFrame) -> dict:
    """How many windows of each kind exist, are usable and were dropped (from the quality flag and the NaN gates).

    ``pre`` is the whole pre-stimulus window (the per-fly figures), ``base`` the part of it before the last L s (the trial
    lends it to the other trials of its fly when it is tracked well enough: usable, else dropped), ``placebo`` the last
    L s before the stimulus and ``during`` the first L s of it. A ``placebo`` / ``during`` window that exists is usable
    (scored), dropped (its own tracking was too poor) or blank (``blank_by_base``: the fly had no resting bend for the
    trial, fewer than two other trials with a usable baseline).
    """
    has = {name: np.isfinite(FIG.float_column(trials, f"abd_tracked_frac_{name}"))
           for name in ("pre", "base", "placebo", "during")}
    ok, lends = _flag(trials, "abd_quality_ok"), _flag(trials, "abd_base_ok")
    rest = np.isfinite(FIG.float_column(trials, "abd_rest_bend_loo_deg"))
    out = {"pre": {"with_window": _count(has["pre"]), "usable": _count(has["pre"] & ok),
                   "dropped": _count(has["pre"] & ~ok)},
           "base": {"with_window": _count(has["base"]), "usable": _count(has["base"] & lends),
                    "dropped": _count(has["base"] & ~lends)}}
    for name in WINDOWS:
        scored = np.isfinite(FIG.float_column(trials, f"abd_curl_events_{name}"))
        out[name] = {"with_window": _count(has[name]), "usable": _count(has[name] & rest & scored),
                     "dropped": _count(has[name] & rest & ~scored), "blank_by_base": _count(has[name] & ~rest)}
    return out


def _pairs(trials: pd.DataFrame) -> pd.DataFrame:
    """The trials whose just-before AND start-of-stimulus windows were both scored and have a length: what a rate uses."""
    scored = np.zeros(len(trials), dtype=bool)
    for event in EVENTS:
        scored |= (np.isfinite(FIG.float_column(trials, f"abd_{event}_events_placebo"))
                   & np.isfinite(FIG.float_column(trials, f"abd_{event}_events_during")))
    return trials[(np.nan_to_num(FIG.float_column(trials, MINUTES), nan=0.0) > 0) & scored]


def _window(use: pd.DataFrame, events: str) -> dict:
    ev, mins = float(use[events].sum()), float(use[MINUTES].sum())
    return {"events": ev, "minutes": mins, "rate": FIG.pooled_rate(ev, mins)}


def _window_seconds(use: pd.DataFrame) -> dict:
    """The length L of the windows of *use* in seconds: smallest, median and largest (NaN without a trial)."""
    seconds = 60.0 * FIG.float_column(use, MINUTES)
    if not seconds.size:
        return {"min": np.nan, "median": np.nan, "max": np.nan}
    return {"min": float(seconds.min()), "median": float(np.median(seconds)), "max": float(seconds.max())}


def _event_cell(pairs: pd.DataFrame, kind: str, event: str, *, n_perm: int, seed: int) -> dict:
    """Pooled event rates in the two windows of the stimulus of *kind* (the same trials) and the window-swap p."""
    col_before, col_during = f"abd_{event}_events_placebo", f"abd_{event}_events_during"
    use = pairs[pairs["stim_kind"] == kind] if "stim_kind" in pairs else pairs.iloc[0:0]
    use = use[np.isfinite(FIG.float_column(use, col_before)) & np.isfinite(FIG.float_column(use, col_during))]
    diff, p = np.nan, np.nan
    if len(use):
        diff, p = M.paired_window_permutation(use, events_pre=col_before, min_pre=MINUTES, events_stim=col_during,
                                              min_stim=MINUTES, n=n_perm, seed=seed, scheme="swap")
    n_flies = int(use["fly_id"].nunique())
    answer = FIG.verdict(p if n_flies >= FIG.MIN_FLIES else np.nan, diff, up=MORE, down=LESS, none="No clear difference")
    return {"kind": kind, "event": event, "n_trials": int(len(use)), "n_flies": n_flies,
            "placebo": _window(use, col_before), "during": _window(use, col_during), "window_s": _window_seconds(use),
            "diff": diff, "p_perm": p, "verdict": answer}


def _event_stats(trials: pd.DataFrame, *, n_perm: int, seed: int) -> dict:
    pairs = _pairs(trials)
    out: dict = {event: {kind: _event_cell(pairs, kind, event, n_perm=n_perm, seed=seed) for kind in KINDS}
                 for event in EVENTS}
    kinds = pairs["stim_kind"] if "stim_kind" in pairs else pd.Series(np.nan, index=pairs.index, dtype=object)
    out["window_s"] = {kind: _window_seconds(pairs[(kinds == kind).to_numpy()]) for kind in KINDS}
    out["n_other"] = int(np.sum(~kinds.isin(KINDS)))
    return out


def _coupling_stats(fly: pd.DataFrame) -> dict:
    """Per-fly median coupling rho by the fly's dominant proboscis state (descriptive: quartiles, no test)."""
    if "proboscis_state_dominant" not in fly:
        fly = fly.assign(proboscis_state_dominant=None)
    rho = FIG.float_column(fly, "per_abdomen_coupling_rho")
    have = fly[np.isfinite(rho) & fly["proboscis_state_dominant"].notna().to_numpy()]
    present = [s for s in FIG.STATES if s != "held_forward" or (have["proboscis_state_dominant"] == s).any()]
    groups = []
    for s in present:
        g = have[have["proboscis_state_dominant"] == s]
        v = FIG.float_column(g, "per_abdomen_coupling_rho")
        q1, med, q3 = np.percentile(v, [25, 50, 75]) if v.size else (np.nan, np.nan, np.nan)
        groups.append({"state": s, "label": FIG.STATE_LABELS[s], "n_flies": int(len(g)),
                       "n_trials": int(np.nansum(FIG.float_column(g, "n_coupling_trials"))),
                       "median": float(med), "q1": float(q1), "q3": float(q3)})
    every = FIG.float_column(have, "per_abdomen_coupling_rho")
    return {"groups": groups, "n_flies": int(len(have)), "n_trials": sum(g["n_trials"] for g in groups),
            "median": float(np.median(every)) if every.size else float("nan")}


def _coupling_answer(c: Mapping) -> str:
    def part(g):
        return f"{g['label'].lower()} {_signed(g['median'])}" if g["n_flies"] else f"{g['label'].lower()} no flies"
    return (f"Overall median ρ={_signed(c['median'])} over {FIG.count_noun(c['n_flies'], 'fly', 'flies')}; "
            f"by proboscis state: {', '.join(part(g) for g in c['groups'])} (descriptive, no test)")


def compute_abdomen_stats(fly_trial: pd.DataFrame, fly: pd.DataFrame, *, n_perm: int, seed: int = 0,
                          min_tracked_frac: Optional[float] = None) -> dict:
    """Every number the figures and the section show. ``placeholders`` names the figures that cannot be drawn and why;
    ``no_abdomen_data`` is True when no identity-consistent fly-trial has a usable pre-stimulus window. The event cells
    (``events[event][kind]``) compare the ``placebo`` window (the last L s before the stimulus) with the ``during`` window
    (the first L s of it) of the same trials.

    *min_tracked_frac* is the ``abd_min_tracked_frac`` gate the windows were judged by (recorded for the section).
    """
    trials = FIG.consistent_trials(fly_trial)
    stats: dict = {"n_perm": int(n_perm), "seed": int(seed), "alpha": FIG.ALPHA, "min_tracked_frac": min_tracked_frac,
                   "n_fly_trials": int(len(fly_trial)), "n_trials": int(len(trials)), "n_flies": int(len(fly)),
                   "windows": _window_counts(trials), "placeholders": {}}
    stats["no_abdomen_data"] = not bool(_flag(trials, "abd_quality_ok").any())
    if stats["no_abdomen_data"]:
        return stats
    rigs = FIG.rig_summary(fly, "abd_rest_bend_deg")
    stats["rest_bend"] = {"rigs": rigs, "n_flies": sum(r["n_flies"] for r in rigs),
                          "answer": FIG.rig_answer(rigs, "resting bend", "Rig medians of the resting bend span")}
    fly = fly.assign(**{CENTRED_TIP: _centred_tip(fly)})
    kw = dict(n_perm=n_perm, seed=seed)
    stats["bend_vs_pitch"] = _scatter(fly, "abd_rest_bend_deg", up="flies tilted more head-up rest more curled",
                                      down="flies tilted more head-down rest more curled", **kw)
    stats["tip_vs_pitch"] = _scatter(fly, CENTRED_TIP, up="flies tilted more head-up have the tip pointing further down",
                                     down="flies tilted more head-down have the tip pointing further down", **kw)
    stats["events"] = _event_stats(trials, **kw)
    stats["restless"] = _scatter(fly, "spont_per_rate_per_min", x="abd_curl_rate_pre_per_min",
                                 up="flies that curl more often also show more spontaneous PER",
                                 down="flies that curl more often show less spontaneous PER", **kw)
    stats["coupling"] = _coupling_stats(fly)
    stats["coupling"]["answer"] = _coupling_answer(stats["coupling"])
    stats["activity"] = {
        "bend_sd": _scatter(fly, "abd_bend_sd_pre_deg", up="flies tilted more head-up have a more variable bend",
                            down="flies tilted more head-down have a more variable bend", **kw),
        "tip_moving": _scatter(fly, "tip_moving_frac_pre", up="flies tilted more head-up move the tip more often",
                               down="flies tilted more head-down move the tip more often", **kw)}
    _placeholders(stats)
    return stats


def _placeholders(stats: dict) -> None:
    holes, enough = stats["placeholders"], FIG.MIN_FLIES
    if not stats["rest_bend"]["n_flies"]:
        holes["abdomen_rest_bend_per_fly"] = {"reason": "no fly has a resting bend", "n": FIG.n_flies_label(0)}
    for name, panels, what in (
            ("mount_pitch_vs_abdomen_rest_bend", (stats["bend_vs_pitch"]["n_flies"], stats["tip_vs_pitch"]["n_flies"]),
             "resting curl, tip pitch"),
            ("mount_pitch_vs_abdomen_activity", (stats["activity"]["bend_sd"]["n_flies"],
                                                 stats["activity"]["tip_moving"]["n_flies"]),
             "bend SD, tip moving fraction")):
        if max(panels) < enough:
            holes[name] = {"reason": f"needs at least {enough} flies with both values ({what}: "
                                     f"{' and '.join(str(n) for n in panels)})", "n": FIG.n_flies_label(max(panels))}
    if not any(c["n_trials"] for event in EVENTS for c in stats["events"][event].values()):
        holes["abdomen_curl_events_by_window"] = {
            "reason": "no odor or light trial has a resting bend from its fly's other trials, a just-before window and "
                      "a start-of-stimulus window",
            "n": FIG.n_label(0, 0)}
    if stats["restless"]["n_flies"] < enough:
        holes["abdomen_vs_spontaneous_per"] = {
            "reason": f"needs at least {enough} flies with both rates (have {stats['restless']['n_flies']})",
            "n": FIG.n_flies_label(stats["restless"]["n_flies"])}
    if not stats["coupling"]["n_flies"]:
        holes["per_abdomen_coupling"] = {"reason": "no fly has both a coupling and a proboscis state",
                                         "n": FIG.n_flies_label(0)}


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _label_shared_points(ax, x: np.ndarray, y: np.ndarray) -> None:
    """Write ``×n`` beside every point that n >= 2 flies share (a pile of identical dots looks like one)."""
    shared = pd.Series(list(zip(np.round(x, 9), np.round(y, 9)))).value_counts()
    for (px, py), n in shared[shared > 1].items():
        ax.annotate(f"×{n}", xy=(px, py), xytext=(7, 5), textcoords="offset points", fontsize=8, color=FIG.INK_SECONDARY,
                    annotation_clip=False)


def _limits(values: np.ndarray, kind: str) -> Tuple[float, float]:
    """Axis limits for *values*: ``positive`` starts just under zero, ``signed`` pads both ends."""
    if not values.size:
        return (-0.04, 1.0) if kind == "positive" else (-1.0, 1.0)
    lo, hi = float(values.min()), float(values.max())
    if kind == "positive":
        top = hi * 1.15 if hi > 0 else 1.0
        return -0.04 * top, top
    pad = 0.1 * ((hi - lo) or 1.0)
    return lo - pad, hi + pad


def _y_axis(ax, y: np.ndarray, kind: str) -> None:
    if kind == "pct":
        top = FIG.headroom(y)
        FIG.pct_axis(ax, top)
        ax.set_ylim(-0.04 * top, top)
        return
    lo, hi = _limits(y, kind)
    ax.set_ylim(lo, hi)
    if lo < 0 < hi:
        ax.axhline(0, color=FIG.AXIS, linewidth=0.8, zorder=1)


def _draw_scatter(ax, panel: Mapping) -> None:
    """One per-fly scatter: blue dots with a surface ring, shared points counted, axes set for the panel's kind."""
    x, y = np.asarray(panel["x"], float), np.asarray(panel["y"], float)
    done = np.isfinite(x) & np.isfinite(y)
    x, y = x[done], y[done]
    ax.scatter(x, y, s=FIG.POINT_MARK["s"], facecolor=FIG.BLUE, edgecolor=FIG.SURFACE, linewidth=1.0, zorder=3)
    _label_shared_points(ax, x, y)
    if panel["x_kind"] == "pitch":
        FIG.pitch_axis(ax, x, "Mounting angle, rig-centred (°)")        # short: the hints share its row in a narrow panel
    else:
        ax.set_xlim(*_limits(x, "positive"))
        ax.set_xlabel(panel["xlabel"])
    _y_axis(ax, y, panel["y_kind"])
    ax.set_ylabel(panel["ylabel"])


def _scatter_panel(heading: str, stat: Mapping, x, y, ylabel: str, *, y_kind: str, x_kind: str = "pitch",
                   xlabel: str = "") -> dict:
    return {"heading": heading, "stat": stat, "x": x, "y": y, "ylabel": ylabel, "y_kind": y_kind, "x_kind": x_kind,
            "xlabel": xlabel}


def _two_panel_scatter(name: str, panels: Sequence[Mapping], numbers: str) -> Figure:
    """Two per-fly scatters side by side (separate axes, each with its own y axis): heading and numbers over each."""
    panel_w, left, gap, right, bottom, height = 4.0, 0.95, 1.05, 0.3, 0.8, 5.6
    width = left + 2 * panel_w + gap + right
    fig = FIG.new_figure(width, height)
    answer = "\n".join(f"{p['heading']}: {p['stat']['verdict']}" for p in panels)
    used = FIG.title_block(fig, TITLES[name], answer, numbers)
    notes = [FIG.wrap_text(fig, FIG.scatter_numbers(p["stat"]), 8, "normal", panel_w) for p in panels]
    axes_top = used + 0.28 + 0.14 * max(len(n) for n in notes) + 0.12
    corner = left
    for panel, note in zip(panels, notes):
        ax = FIG.inch_axes(fig, corner, bottom, panel_w, height - axes_top - bottom)
        _draw_scatter(ax, panel)
        fig.text(corner / width, 1 - used / height, panel["heading"], ha="left", va="top", fontsize=10, weight="bold",
                 color=FIG.INK)
        fig.text(corner / width, 1 - (used + 0.25) / height, "\n".join(note), ha="left", va="top", fontsize=8,
                 color=FIG.INK_SECONDARY, linespacing=1.4)
        corner += panel_w + gap
    return fig


def _fig_rest_bend(trials, fly, stats):
    rb = stats["rest_bend"]
    rigs = rb["rigs"]
    width, height = max(6.0, 1.5 * len(rigs) + 2.8), 5.2
    fig = FIG.new_figure(width, height)
    top = FIG.title_block(fig, TITLES["abdomen_rest_bend_per_fly"], rb["answer"],
                          f"0° = straight, positive = curled toward the proboscis; a fly's resting bend is its median bend "
                          f"before the stimulus; {FIG.n_flies_label(rb['n_flies'])}")
    ax = FIG.inch_axes(fig, 0.95, 0.8, width - 1.3, height - top - 0.8 - 0.1)
    groups = []
    for r in rigs:
        v = FIG.float_column(fly[fly["rig"].astype(str) == r["rig"]], "abd_rest_bend_deg")
        groups.append({"values": v[np.isfinite(v)], "median": r["median_deg"], "q1": r["q1_deg"], "q3": r["q3_deg"]})
    FIG.strip_groups(ax, groups, stats["seed"])
    ax.set_xticklabels([f"{r['rig']}\n{FIG.n_flies_label(r['n_flies'])}" for r in rigs])
    ax.set_ylabel("Resting abdomen bend (°; + = curled)")
    ax.set_xlabel("dot = one fly    bar = median    light line = interquartile range", color=FIG.MUTED, fontsize=8.5)
    return fig


def _fig_rest_vs_pitch(trials, fly, stats):
    x = FIG.float_column(fly, "mount_pitch_centred_deg")
    panels = [_scatter_panel("Resting curl", stats["bend_vs_pitch"], x, FIG.float_column(fly, "abd_rest_bend_deg"),
                             "Resting abdomen bend (°; + = curled)", y_kind="signed"),
              _scatter_panel("Resting tip pitch", stats["tip_vs_pitch"], x, _centred_tip(fly),
                             "Tip pitch at rest, rig-centred (°; + = pointing down)", y_kind="signed")]
    return _two_panel_scatter("mount_pitch_vs_abdomen_rest_bend", panels,
                              "Each dot is one fly; camera roll shifts the tip pitch like the mounting angle, so both are "
                              "relative to the fly's rig median; p-values are uncorrected")


def _fig_activity(trials, fly, stats):
    x = FIG.float_column(fly, "mount_pitch_centred_deg")
    panels = [_scatter_panel("Bend variability", stats["activity"]["bend_sd"], x,
                             FIG.float_column(fly, "abd_bend_sd_pre_deg"), "SD of the bend before the stimulus (°)",
                             y_kind="positive"),
              _scatter_panel("Tip movement", stats["activity"]["tip_moving"], x,
                             100.0 * FIG.float_column(fly, "tip_moving_frac_pre"), "Frames with the tip moving",
                             y_kind="pct")]
    return _two_panel_scatter("mount_pitch_vs_abdomen_activity", panels,
                              "Each dot is one fly, measured over its pre-stimulus windows; p-values are uncorrected")


def _fig_restless(trials, fly, stats):
    s = stats["restless"]
    width, height, bottom = 7.2, 5.6, 0.8
    fig = FIG.new_figure(width, height)
    top = FIG.title_block(fig, TITLES["abdomen_vs_spontaneous_per"], s["verdict"],
                          f"{FIG.scatter_numbers(s)}; each rate pools a fly's pre-stimulus windows (events over minutes)")
    ax = FIG.inch_axes(fig, 0.9, bottom, width - 1.2, height - top - bottom - 0.1)
    _draw_scatter(ax, _scatter_panel("", s, FIG.float_column(fly, "abd_curl_rate_pre_per_min"),
                                     FIG.float_column(fly, "spont_per_rate_per_min"), "Spontaneous PER events per minute",
                                     y_kind="positive", x_kind="rate",
                                     xlabel="Spontaneous abdomen curls per minute (before the stimulus)"))
    return fig


def _fig_coupling(trials, fly, stats):
    c = stats["coupling"]
    groups = c["groups"]
    width, height = max(6.0, 1.5 * len(groups) + 2.8), 5.4
    fig = FIG.new_figure(width, height)
    top = FIG.title_block(
        fig, TITLES["per_abdomen_coupling"], c["answer"],
        f"ρ is the within-trial Spearman correlation between how far out the proboscis is and the abdomen bend (ρ > 0: the "
        f"abdomen curls more when the proboscis is further out); a dot is a fly's median over its trials, flies are grouped "
        f"by their most common proboscis state before the stimulus; {FIG.n_flies_label(c['n_flies'])}, "
        f"{FIG.count_noun(c['n_trials'], 'trial', 'trials')}")
    ax = FIG.inch_axes(fig, 0.95, 0.8, width - 1.3, height - top - 0.8 - 0.1)
    strip = []
    for g in groups:
        v = FIG.float_column(fly[fly["proboscis_state_dominant"] == g["state"]], "per_abdomen_coupling_rho")
        strip.append({"values": v[np.isfinite(v)], "median": g["median"], "q1": g["q1"], "q3": g["q3"]})
    FIG.strip_groups(ax, strip, stats["seed"], fmt=_signed)
    reach = max([abs(v) for g in strip for v in g["values"]] + [0.0])
    lim = min(1.05, max(0.3, 1.15 * reach))
    ax.set_ylim(-lim, lim)
    ax.set_xticklabels([f"{g['label']}\n{FIG.n_flies_label(g['n_flies'])}" for g in groups])
    ax.set_ylabel("Coupling ρ (proboscis distance vs abdomen bend)")
    ax.set_xlabel("dot = one fly    bar = median    light line = interquartile range", color=FIG.MUTED, fontsize=8.5)
    return fig


def _pair_axes(fig, left: float, bottom: float, width: float, height: float, n_pairs: int):
    """Axes for *n_pairs* pairs of bars; returns it, the x positions of the bars and the bar width in data units."""
    ax = FIG.inch_axes(fig, left, bottom, width, height)
    positions = [2.4 * j + k for j in range(n_pairs) for k in (0, 1)]
    ax.set_xlim(-0.6, positions[-1] + 0.6)
    ax.set_xticks(positions)
    return ax, positions, FIG.BAR_IN * (positions[-1] + 1.2) / width


def _fig_events(trials, fly, stats):
    ev = stats["events"]
    kinds = [k for k in KINDS if any(ev[e][k]["n_trials"] for e in EVENTS)]
    slot, left, gap, right, bottom, height = 1.0, 0.95, 0.9, 0.3, 0.85, 5.4
    panel_w = slot * (2.4 * (len(kinds) - 1) + 2.2)
    width = left + 2 * panel_w + gap + right
    fig = FIG.new_figure(width, height)
    answer = "\n".join(f"{EVENT_LABELS[e]} at the start of {WINDOW_LABELS[k]}: {ev[e][k]['verdict']}"
                       for e in EVENTS for k in kinds)
    lengths = ", ".join(f"{_seconds_text(ev['window_s'][k], median=False)} for {WINDOW_LABELS[k]}" for k in kinds)
    used = FIG.title_block(
        fig, TITLES["abdomen_curl_events_by_window"], answer,
        "A curl is a sustained rise of the bend above the fly's resting bend, a straightening a sustained fall below it. "
        f"Each pair compares the last L s before the stimulus with the first L s of it (L = {lengths}), both against the "
        "fly's resting bend from its other trials. Bars pool trials (events over minutes of the same trials). Above each "
        "pair: during minus just before, and the p-value of a test that swaps the two window labels at random within each "
        "trial (trial-level: trials, not flies, are shuffled; it assumes the bend is stationary within a trial), "
        "uncorrected")
    rates = [ev[e][k][w]["rate"] for e in EVENTS for k in kinds for w in WINDOWS]
    ceiling = max(1.0, 1.4 * max([r for r in rates if np.isfinite(r)] + [0.0]))
    corner = left
    for event in EVENTS:
        ax, positions, bar_w = _pair_axes(fig, corner, bottom, panel_w, height - used - 0.3 - bottom, len(kinds))
        ax.set_ylim(0, ceiling)
        labels = []
        for j, kind in enumerate(kinds):
            cell = ev[event][kind]
            for k, window in enumerate(WINDOWS):
                rate, x = cell[window]["rate"], positions[2 * j + k]
                if np.isfinite(rate):
                    FIG.draw_bar(ax, x, rate, bar_w, held=window == "during", hatch="//" if kind == "light" else None)
                    FIG.tip_label(ax, x, rate, FIG.num_text(rate), ceiling)
                labels.append(f"{WINDOW_NAMES[window]}\n{WINDOW_LABELS[kind]}\n"
                              f"{FIG.n_lines(cell['n_trials'], cell['n_flies'])}")
            ax.text(positions[2 * j] + 0.5, 0.99 * ceiling, _pair_note(cell), ha="center", va="top", fontsize=8,
                    color=FIG.INK_SECONDARY, linespacing=1.3, zorder=5,
                    bbox=dict(facecolor=FIG.SURFACE, edgecolor="none", pad=1.5))      # a grid line does not strike through
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(f"{EVENT_NOUN[event]} events per minute")
        fig.text(corner / width, 1 - used / height, EVENT_LABELS[event], ha="left", va="top", fontsize=10, weight="bold",
                 color=FIG.INK)
        corner += panel_w + gap
    return fig


def _pair_note(cell: Mapping) -> str:
    """The difference and the permutation p written above a pair of bars."""
    if cell["n_flies"] < FIG.MIN_FLIES:
        return f"{_signed_rate(cell['diff'])} per min\nfewer than {FIG.MIN_FLIES} flies"
    return f"{_signed_rate(cell['diff'])} per min\n{FIG.p_text(cell['p_perm'])} (uncorrected)"


BUILDERS: Dict[str, Callable[[pd.DataFrame, pd.DataFrame, dict], Figure]] = {
    "abdomen_rest_bend_per_fly": _fig_rest_bend, "mount_pitch_vs_abdomen_rest_bend": _fig_rest_vs_pitch,
    "abdomen_curl_events_by_window": _fig_events, "abdomen_vs_spontaneous_per": _fig_restless,
    "per_abdomen_coupling": _fig_coupling, "mount_pitch_vs_abdomen_activity": _fig_activity}


def _placeholder(name: str, stats: dict) -> Figure:
    info = stats["placeholders"][name]
    return FIG.placeholder_figure(TITLES[name], info["reason"], info["n"])


def iter_abdomen_figures(fly_trial: pd.DataFrame, fly: pd.DataFrame, stats: dict) -> Iterator[Tuple[str, Figure]]:
    """The six figures one at a time (a failing one is logged and left out; see ``FIG.build_each``)."""
    return FIG.build_each(BUILDERS, _placeholder, FIG.consistent_trials(fly_trial), fly, stats, tag="ABDOMEN",
                          logger=LOGGER)


def build_abdomen_figures(fly_trial: pd.DataFrame, fly: pd.DataFrame, stats: dict) -> Dict[str, Figure]:
    """All six figures as matplotlib ``Figure`` objects (nothing is written); none without a usable abdomen window."""
    return {} if stats["no_abdomen_data"] else dict(iter_abdomen_figures(fly_trial, fly, stats))


def render_abdomen_figures(fly_trial: pd.DataFrame, fly: pd.DataFrame, out_dir, *, n_perm: int, seed: int = 0,
                           stats: Optional[dict] = None) -> Dict[str, Path]:
    """Write the six PNGs (300 dpi) into *out_dir*; returns ``{figure name: path}`` of those written.

    The six PNGs of an earlier run are removed first, so a figure that fails to build leaves no stale picture behind for
    the report to point at: after the call ``<name>.png`` exists exactly for the names returned. With no fly-trials, or
    none with a usable abdomen window, nothing is drawn or created and those earlier pictures are removed too. *stats* is
    ``compute_abdomen_stats``' result when the caller has it already (the section takes the same one).
    """
    out = Path(out_dir).expanduser()
    if len(fly_trial) == 0:
        FIG.remove_pictures(out, NAMES)
        return {}
    stats = stats if stats is not None else compute_abdomen_stats(fly_trial, fly, n_perm=n_perm, seed=seed)
    if stats["no_abdomen_data"]:
        FIG.remove_pictures(out, NAMES)
        return {}
    return FIG.write_pictures(iter_abdomen_figures(fly_trial, fly, stats), out, NAMES)


# ---------------------------------------------------------------------------
# The "Abdomen movement" section of report.md
# ---------------------------------------------------------------------------

def _gate_text(gate: Optional[float]) -> str:
    return (f"tracked in fewer than {100 * gate:g}% of their frames" if gate is not None
            else "tracked in too few of their frames")


def _blank_clause(n: int) -> str:
    """``; 2 more windows were blank because ...`` (nothing when none was), in the right number."""
    return (f"; {n} more {'window was' if n == 1 else 'windows were'} blank because the fly had fewer than 2 other trials "
            "with a usable baseline window") if n else ""


def _windows_lines(stats: Mapping) -> List[str]:
    w, gate = stats["windows"], _gate_text(stats["min_tracked_frac"])
    pre, base = w["pre"], w["base"]
    lines = [f"- Pre-stimulus windows: {pre['dropped']} of {pre['with_window']} pre-stimulus windows dropped ({gate}, "
             f"`abd_min_tracked_frac`); {pre['usable']} usable",
             f"- Baseline windows (the part of the pre-stimulus window before its last L s, which a trial lends to the "
             f"resting bend of the other trials of its fly): {base['dropped']} of {base['with_window']} baseline windows "
             f"dropped ({gate}); {base['usable']} usable"]
    for name, label, what in (("placebo", "Just-before windows (the last L s before the stimulus)", "just-before"),
                              ("during", "Start-of-stimulus windows (the first L s of the stimulus)",
                               "start-of-stimulus")):
        win = w[name]
        lines.append(f"- {label}: {win['dropped']} of {win['with_window']} {what} windows dropped ({gate})"
                     f"{_blank_clause(win['blank_by_base'])}; {win['usable']} usable")
    return lines


def _no_data_section(stats: Mapping, figures_note: str) -> str:
    n = stats["n_fly_trials"]
    if not n:
        why = ("No abdomen data in scope: there are no new-glue fly-trials (see the exclusions above for what was left "
               "out). " + figures_note)
        return "\n".join(["## Abdomen movement", "", why.strip()])
    why = (f"No abdomen data in scope: none of the {FIG.count_noun(n, 'fly-trial', 'fly-trials')} has a usable "
           "(tip-tracked) pre-stimulus window of an identity-consistent fly, so no abdomen figure was drawn. "
           + figures_note).strip()
    return "\n".join(["## Abdomen movement", "", why, "",
                      "Windows dropped for poor tracking (identity-consistent fly-trials):", "", *_windows_lines(stats)])


def _rate_cell(w: Mapping) -> str:
    return f"{w['events']:g} / {w['minutes']:.2f} min = {w['rate']:.2f}/min"


def _events_block(stats: Mapping) -> List[str]:
    ev = stats["events"]
    L = ["| Stimulus | Event | Just before the stimulus | Start of the stimulus | Difference | Trials / flies | "
         "Window-swap permutation p |", "|---|---|---|---|---|---|---|"]
    for kind in KINDS:
        for event in EVENTS:
            c = ev[event][kind]
            if not c["n_trials"]:
                L.append(f"| {kind} | {EVENT_LABELS[event]} | no trial | no trial | n/a | {FIG.n_label(0, 0)} | n/a |")
                continue
            L.append(f"| {kind} | {EVENT_LABELS[event]} | {_rate_cell(c['placebo'])} | {_rate_cell(c['during'])} | "
                     f"{_signed_rate(c['diff'])}/min | {FIG.n_label(c['n_trials'], c['n_flies'])} | "
                     f"{FIG.p_text(c['p_perm'], 4)} (uncorrected) |")
    L += [""]
    L.append("- Window length L: " + ", ".join(f"{kind} {_seconds_text(ev['window_s'][kind])}" for kind in KINDS))
    for kind in KINDS:
        for event in EVENTS:
            c = ev[event][kind]
            if c["n_trials"]:
                L.append(f"- {EVENT_LABELS[event]} at the start of {WINDOW_LABELS[kind]}: **{c['verdict']}**"
                         + (f" (fewer than {FIG.MIN_FLIES} flies)" if c["n_flies"] < FIG.MIN_FLIES else ""))
    if ev["n_other"]:
        L.append(f"- {FIG.count_noun(ev['n_other'], 'usable trial', 'usable trials')} with another stimulus (odor + light "
                 "training trials, or none) are in neither pair: their window mixes the stimuli or has none")
    return L


def _apart(lines: List[str]) -> List[str]:
    """*lines* followed by a blank line (markdown would read the next paragraph or table as part of a list item)."""
    return [*lines, ""] if lines else []


def _coupling_row(g: Mapping) -> str:
    if not g["n_flies"]:
        return f"| {g['label']} | {FIG.n_flies_label(0)} | 0 | n/a | n/a |"
    return (f"| {g['label']} | {FIG.n_flies_label(g['n_flies'])} | {g['n_trials']:,} | {_signed(g['median'])} | "
            f"{_signed(g['q1'])} to {_signed(g['q3'])} |")


def _panel_bullets(label: str, s: Mapping, shuffled: str) -> List[str]:
    return [f"**{label}:** {s['verdict']}", "", *FIG.scatter_bullets(s, shuffled)[1:], ""]


def abdomen_report_section(fly_trial: pd.DataFrame, fly: pd.DataFrame, stats: Mapping,
                           figures: Optional[Mapping[str, Path]] = None) -> str:
    """The ``## Abdomen movement`` section of ``report.md`` as markdown: pooled event rates per window with n trials /
    flies, rho / p of every per-fly scatter, the permutation p's and what they shuffle, and the windows the
    tracked-fraction gate dropped.

    *figures* is what ``render_abdomen_figures`` returned: a figure that is not in it (it failed to build) is named in
    the section. Without *figures* the section claims nothing about the pictures.
    """
    missing = [n for n in NAMES if n not in figures] if figures is not None else []
    note = ("**Figures missing from this folder:** " + ", ".join(f"`{n}.png`" for n in missing) +
            " (a figure that fails to build is logged and skipped; see the run log)." if missing else "")
    if stats["no_abdomen_data"]:
        return _no_data_section(stats, "The pictures of an earlier run were removed." if figures is not None else "")
    q, ph = TITLES, FIG.placeholder_line
    n_perm, seed = stats["n_perm"], stats["seed"]
    L = ["## Abdomen movement", ""]
    if note:
        L += [note, ""]
    L += ["Abdomen bend is 180° minus the eye–abdomen–tip interior angle: 0 = straight, positive = curled toward the "
          "proboscis (ventral). Tip pitch is the direction of the tip from the abdomen centre against the image horizontal "
          "(positive = the tip points down). Only windows judged usable of identity-consistent flies enter a number: "
          "`abd_quality_ok` for the pre-stimulus window, the same tracked-fraction gate for each window of the stimulus "
          "comparison and for every baseline that is lent to the other trials of its fly.", "",
          "How to read the statistics:", "",
          "- The fly is the unit of inference. Per-fly figures use the fly table; pooled rates use the trials of "
          "identity-consistent flies and divide pooled events by pooled minutes (never a mean of per-trial rates).",
          "- A curl is a sustained rise of the bend above the fly's resting bend, a straightening a sustained fall below "
          "it; thresholds `curl_rise_deg` and `curl_min_s` of the `mounting:` block. In the per-fly figures the resting "
          "bend is the fly's median bend before the stimulus; the stimulus comparison uses the one described next.",
          "- Stimulus comparison (placebo design): L is the shorter of the stimulus and half the pre-stimulus window "
          "(its value is given under the events table). A trial's resting bend is the median bend over the early baseline "
          "windows (the part of the pre-stimulus window before its last L s) of the fly's OTHER trials, pooled over frames: "
          "the trial itself and every trial's just-before and start-of-stimulus frames are left out, so it does not depend "
          "on the two windows being compared or on the frames next to them. Curls and straightenings are counted against "
          "it in the last L s before the stimulus (\"just before\") and in the first L s of the stimulus (\"start of\").",
          "- Assumption (within-trial stationarity): with no stimulus effect the two windows of a trial are "
          "interchangeable, which is what the swap test below needs. A fly needs at least 3 trials with a known stimulus "
          "onset: a trial is scored only when at least 2 of the fly's other trials have a usable baseline window (tracked "
          "in at least `abd_min_tracked_frac` of its frames); flies whose identity is not consistent are left out.",
          "- Mount pitch is rig-centred (fly value minus the median of its rig) because camera roll differs per rig; the "
          "tip pitch is rig-centred the same way, since the roll shifts it too.",
          f"- Stratified permutation p: the x variable is shuffled among flies of the same dataset × batch, two-sided on "
          f"|ρ|, {n_perm} permutations, seed {seed}. This p decides every \"Yes\" in a figure title. Flies of one batch "
          f"share the mounting day, the glue session and the starvation, so only flies of the same batch are "
          f"interchangeable (a batch is recorded on one rig); the same permutation within dataset × rig is printed "
          f"beside it as a secondary number.",
          f"- Window-swap permutation p: the labels of a trial's two windows (just before / start of the stimulus) are "
          f"swapped with probability 1/2, independently in each trial, and the pooled rate difference (start minus just "
          f"before) is recomputed; two-sided, {n_perm} permutations, seed {seed}: a Monte-Carlo permutation estimate of the "
          f"p-value. The randomisation is trial-level (trials, not flies, are shuffled): it treats trials as independent, "
          f"so a fly with many trials counts for more and a fly's trials are not kept together.",
          "- Limitation: a bend that drifts steadily inside the trial breaks the assumption above: the start of the "
          "stimulus is later, and farther from the baseline, than the window before it, so it can differ even when the "
          "stimulus does nothing, and the more trials are pooled the more a small drift shows as a difference. A small p "
          "needs a look at the traces when the posture visibly drifts.",
          f"- \"Yes\" in a figure title means the permutation p is below {stats['alpha']} (uncorrected) with at least "
          f"{FIG.MIN_FLIES} flies to compare; fewer is \"Not enough data to tell\". **All p-values are uncorrected.**", "",
          "### Windows dropped for poor tracking", "",
          "A window is usable only when the bend is tracked in at least `abd_min_tracked_frac` of its frames"
          + (f" (here {100 * stats['min_tracked_frac']:g}%)" if stats["min_tracked_frac"] is not None else "")
          + "; a dropped window has every abdomen metric left empty, never zero. Counted over the "
          f"{FIG.count_noun(stats['n_trials'], 'fly-trial', 'fly-trials')} of identity-consistent flies:", "",
          *_windows_lines(stats), "",
          f"### {q['abdomen_rest_bend_per_fly']}", "",
          "Figure `abdomen_rest_bend_per_fly.png`: the resting bend of every fly, by rig.", "",
          f"- {stats['rest_bend']['answer']}", *ph(stats, "abdomen_rest_bend_per_fly"), "",
          "| Rig | Flies | Median (°) | Q1 – Q3 (°) |", "|---|---|---|---|"]
    L += [f"| {r['rig']} | {FIG.n_flies_label(r['n_flies'])} | {r['median_deg']:+.1f} | "
          f"{r['q1_deg']:+.1f} to {r['q3_deg']:+.1f} |" for r in stats["rest_bend"]["rigs"]]
    L += ["", f"### {q['mount_pitch_vs_abdomen_rest_bend']}", "",
          "Figure `mount_pitch_vs_abdomen_rest_bend.png`: per fly, rig-centred mounting angle vs the resting bend (left) "
          "and vs the rig-centred resting tip pitch (right).", "",
          *_apart(ph(stats, "mount_pitch_vs_abdomen_rest_bend")),
          *_panel_bullets("Resting curl", stats["bend_vs_pitch"], "mounting angle"),
          *_panel_bullets("Resting tip pitch", stats["tip_vs_pitch"], "mounting angle"),
          f"### {q['abdomen_curl_events_by_window']}", "",
          "Figure `abdomen_curl_events_by_window.png`: curl and straightening events per minute, pooled over trials, in the "
          "last L s before vs the first L s of the odor and of the light (light-only trials; hatched bars), each pair from "
          "the same trials, both windows scored against the fly's resting bend from its other trials.", "",
          *_apart(ph(stats, "abdomen_curl_events_by_window")),
          *_events_block(stats), "",
          *FIG.scatter_lines("abdomen_vs_spontaneous_per", q["abdomen_vs_spontaneous_per"], stats["restless"],
                             "per fly, spontaneous curl rate vs spontaneous PER rate (events per minute, pooled over the "
                             "pre-stimulus windows)", "curl rate"),
          *ph(stats, "abdomen_vs_spontaneous_per"), "",
          f"### {q['per_abdomen_coupling']}", "",
          "Figure `per_abdomen_coupling.png`: per-fly median coupling (within-trial Spearman ρ between the proboscis "
          "distance and the abdomen bend), by the fly's dominant proboscis state. Descriptive: no test.", "",
          f"- {stats['coupling']['answer']}", *ph(stats, "per_abdomen_coupling"), "",
          "| Dominant proboscis state | Flies | Coupling trials | Median ρ | Q1 – Q3 |", "|---|---|---|---|---|"]
    L += [_coupling_row(g) for g in stats["coupling"]["groups"]]
    L += ["", f"### {q['mount_pitch_vs_abdomen_activity']}", "",
          "Figure `mount_pitch_vs_abdomen_activity.png`: per fly, rig-centred mounting angle vs the SD of the bend (left) "
          "and vs the fraction of frames with the tip moving (right), both before the stimulus.", "",
          *_apart(ph(stats, "mount_pitch_vs_abdomen_activity")),
          *_panel_bullets("Bend variability", stats["activity"]["bend_sd"], "mounting angle"),
          *_panel_bullets("Tip movement", stats["activity"]["tip_moving"], "mounting angle")]
    return "\n".join(L).rstrip("\n")


def _failed_section(err: Exception) -> str:
    return ("## Abdomen movement\n\nThe abdomen movement figures and statistics could not be computed: "
            f"{type(err).__name__}: {err} (see the run log). The pictures of an earlier run were removed.")


def abdomen_outputs(fly_trial: pd.DataFrame, fly: pd.DataFrame, out_dir, *, n_perm: int, seed: int = 0,
                    min_tracked_frac: Optional[float] = None) -> Tuple[str, Optional[Dict[str, Path]]]:
    """``(section, figures)``: the abdomen statistics, the six PNGs in *out_dir* (``figures`` is None without one) and
    the report section that describes exactly the pictures written.

    Never raises: a failure is logged, the pictures of an earlier run are removed and the section says so, so the
    mounting figures and report that sit beside it are not lost.
    """
    out = Path(out_dir).expanduser() if out_dir else None
    try:
        stats = compute_abdomen_stats(fly_trial, fly, n_perm=n_perm, seed=seed, min_tracked_frac=min_tracked_frac)
        figures = render_abdomen_figures(fly_trial, fly, out, n_perm=n_perm, seed=seed, stats=stats) if out else None
        return abdomen_report_section(fly_trial, fly, stats, figures=figures), figures
    except Exception as err:                                    # noqa: BLE001  (see the docstring)
        LOGGER.warning("[ABDOMEN] the abdomen analysis failed: %s: %s", type(err).__name__, err, exc_info=True)
        if out is not None:
            try:
                FIG.remove_pictures(out, NAMES)
            except OSError:
                pass
        return _failed_section(err), ({} if out is not None else None)
