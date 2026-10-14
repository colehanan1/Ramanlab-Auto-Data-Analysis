"""The ``Abdomen-Movement/`` figure set and the "Abdomen movement" section of ``report.md``.

Takes the two tables of ``mounting_posture.py`` -- the fly-trial table (``fly_trial_posture.csv``) and the fly table
(``fly_posture.csv``) -- and draws six figures, each titled with the QUESTION and answered in plain words computed from
the data (``Yes -- flies tilted more head-up rest more curled`` / ``No clear link``), the numbers in a smaller line:

1. ``abdomen_rest_bend_per_fly``          strip plot of the resting curl (bend) by rig;
2. ``mount_pitch_vs_abdomen_rest_bend``    per fly, two panels: rig-centred mounting angle vs resting curl / tip pitch;
3. ``abdomen_curl_events_by_window``       pooled curl / straightening events per minute before vs during the odor and light;
4. ``abdomen_vs_spontaneous_per``          per fly: spontaneous curl rate vs spontaneous PER rate;
5. ``per_abdomen_coupling``                per-fly median proboscis-abdomen coupling by the fly's dominant proboscis state;
6. ``mount_pitch_vs_abdomen_activity``     per fly, two panels: rig-centred mounting angle vs bend SD / tip moving fraction.

The conventions are those of ``mounting_posture_figures`` (whose primitives this module imports). The fly is the unit of
inference: the per-fly scatters use the fly table, the rig-centred mounting angle (the tip pitch is rig-centred too:
camera roll shifts both) and a permutation p stratified by dataset x rig. Event rates pool TRIALS (events over minutes of
the same trials, never a mean of per-trial rates). The pre-vs-stimulus p of figure 3 is the within-trial permutation of
``fbpipe.analysis.mounting.paired_window_permutation`` in its ``"split"`` scheme: each trial's events are re-assigned
between its two windows in proportion to their lengths. (Exchanging the two whole windows is exact only when they are
equally long; with 30 s before an odor against its 4 s it calls about one stimulus in five that changes nothing
"significant".) Only quality-ok windows (``abd_quality_ok``) of identity-consistent flies enter a number. Every p is
uncorrected, and says so; a comparison with fewer than three flies is "Not enough data to tell". Figure 5 is descriptive.

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
MORE, LESS = "more often than before the stimulus", "less often than before the stimulus"
CENTRED_TIP = "tip_pitch_centred_deg"           # the fly table's tip pitch minus its rig's median (derived here)
P_SCHEME = "split"                              # how paired_window_permutation re-draws a trial's split (see above)


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


def _centred_tip(fly: pd.DataFrame) -> np.ndarray:
    """The resting tip pitch of every fly minus its rig's median (camera roll shifts it like the mount pitch)."""
    tip = pd.Series(FIG.float_column(fly, "tip_pitch_rest_deg"), index=fly.index)
    return (tip - tip.groupby(fly["rig"].astype(str)).transform("median")).to_numpy()


def _scatter(fly: pd.DataFrame, column: str, *, x: str = "mount_pitch_centred_deg", n_perm: int, seed: int,
             up: str, down: str) -> dict:
    """``FIG.scatter_stats``, except that no dataset x rig with two flies (nothing was shuffled) is not "No clear link"."""
    s = FIG.scatter_stats(fly, column, x=x, n_perm=n_perm, seed=seed, up=up, down=down)
    if s["n_flies"] >= FIG.MIN_FLIES and s["n_strata_shuffled"] == 0:
        s["verdict"] = "Not enough data to tell (no dataset × rig has two flies to shuffle)"
    return s


# ---------------------------------------------------------------------------
# Statistics (the one place any number is computed)
# ---------------------------------------------------------------------------

def _window_counts(trials: pd.DataFrame) -> dict:
    """How many pre-stimulus / stimulus windows exist, are usable and were dropped (from the quality flags)."""
    ok, ok_stim = _flag(trials, "abd_quality_ok"), _flag(trials, "abd_quality_ok_stim")
    has_pre = np.isfinite(FIG.float_column(trials, "abd_tracked_frac_pre"))
    has_stim = np.isfinite(FIG.float_column(trials, "abd_tracked_frac_stim"))
    return {"pre": {"with_window": int(has_pre.sum()), "usable": int((has_pre & ok).sum()),
                    "dropped": int((has_pre & ~ok).sum())},
            "stim": {"with_window": int(has_stim.sum()), "usable": int((has_stim & ok_stim).sum()),
                     "dropped": int((has_stim & ok & ~ok_stim).sum()), "blank_by_pre": int((has_stim & ~ok).sum())}}


def _pairs(trials: pd.DataFrame) -> pd.DataFrame:
    """The trials whose pre-stimulus AND stimulus windows are both quality-ok and have a length: what a rate uses."""
    minutes = np.column_stack([FIG.float_column(trials, "pre_min"), FIG.float_column(trials, "stim_min")])
    has_length = (np.nan_to_num(minutes, nan=0.0) > 0).all(axis=1)
    return trials[_flag(trials, "abd_quality_ok") & _flag(trials, "abd_quality_ok_stim") & has_length]


def _window(use: pd.DataFrame, events: str, minutes: str) -> dict:
    ev, mins = float(use[events].sum()), float(use[minutes].sum())
    return {"events": ev, "minutes": mins, "rate": FIG.pooled_rate(ev, mins)}


def _event_cell(pairs: pd.DataFrame, kind: str, event: str, *, n_perm: int, seed: int) -> dict:
    """Pooled event rates before and during the stimulus of *kind* (the same trials) and the within-trial permutation p."""
    col_pre, col_stim = f"abd_{event}_events_pre", f"abd_{event}_events_stim"
    use = pairs[pairs["stim_kind"] == kind] if "stim_kind" in pairs else pairs.iloc[0:0]
    use = use[np.isfinite(FIG.float_column(use, col_pre)) & np.isfinite(FIG.float_column(use, col_stim))]
    diff, p = np.nan, np.nan
    if len(use):
        diff, p = M.paired_window_permutation(use, events_pre=col_pre, min_pre="pre_min", events_stim=col_stim,
                                              min_stim="stim_min", n=n_perm, seed=seed, scheme=P_SCHEME)
    n_flies = int(use["fly_id"].nunique())
    answer = FIG.verdict(p if n_flies >= FIG.MIN_FLIES else np.nan, diff, up=MORE, down=LESS, none="No clear difference")
    return {"kind": kind, "event": event, "n_trials": int(len(use)), "n_flies": n_flies,
            "pre": _window(use, col_pre, "pre_min"), "stim": _window(use, col_stim, "stim_min"),
            "diff": diff, "p_perm": p, "verdict": answer}


def _event_stats(trials: pd.DataFrame, *, n_perm: int, seed: int) -> dict:
    pairs = _pairs(trials)
    out: dict = {event: {kind: _event_cell(pairs, kind, event, n_perm=n_perm, seed=seed) for kind in KINDS}
                 for event in EVENTS}
    other = ~pairs["stim_kind"].isin(KINDS) if "stim_kind" in pairs else np.ones(len(pairs), dtype=bool)
    out["n_other"] = int(np.sum(other))
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
    ``no_abdomen_data`` is True when no identity-consistent fly-trial has a usable pre-stimulus window.

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
            "reason": "no odor or light trial has a usable abdomen window before and during the stimulus",
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
    answer = "\n".join(f"{EVENT_LABELS[e]} before vs during {WINDOW_LABELS[k]}: {ev[e][k]['verdict']}"
                       for e in EVENTS for k in kinds)
    used = FIG.title_block(
        fig, TITLES["abdomen_curl_events_by_window"], answer,
        "A curl is a sustained rise of the bend above the fly's resting bend, a straightening a sustained fall below it. "
        "Bars pool trials (events over minutes of the same trials before and during the stimulus). Above each pair: during "
        "minus before, and the p-value of a permutation within trials (each trial's events are re-assigned between its "
        "two windows in proportion to their lengths), uncorrected")
    rates = [ev[e][k][w]["rate"] for e in EVENTS for k in kinds for w in ("pre", "stim")]
    ceiling = max(1.0, 1.4 * max([r for r in rates if np.isfinite(r)] + [0.0]))
    corner = left
    for event in EVENTS:
        ax, positions, bar_w = _pair_axes(fig, corner, bottom, panel_w, height - used - 0.3 - bottom, len(kinds))
        ax.set_ylim(0, ceiling)
        labels = []
        for j, kind in enumerate(kinds):
            cell = ev[event][kind]
            for k, window in enumerate(("pre", "stim")):
                rate, x = cell[window]["rate"], positions[2 * j + k]
                if np.isfinite(rate):
                    FIG.draw_bar(ax, x, rate, bar_w, held=window == "stim", hatch="//" if kind == "light" else None)
                    FIG.tip_label(ax, x, rate, FIG.num_text(rate), ceiling)
                labels.append(f"{'Before' if window == 'pre' else 'During'} {WINDOW_LABELS[kind]}\n"
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


def _windows_lines(stats: Mapping) -> List[str]:
    w, gate = stats["windows"], _gate_text(stats["min_tracked_frac"])
    pre, stim = w["pre"], w["stim"]
    blank = stim["blank_by_pre"]
    return [f"- Pre-stimulus windows: {pre['dropped']} of {pre['with_window']} pre-stimulus windows dropped ({gate}, "
            f"`abd_min_tracked_frac`); {pre['usable']} usable",
            f"- Stimulus windows: {stim['dropped']} of {stim['with_window']} stimulus windows dropped ({gate}); "
            f"{blank} more {'window was' if blank == 1 else 'windows were'} blank because their pre-stimulus baseline "
            f"was dropped; {stim['usable']} usable"]


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
    L = ["| Stimulus | Event | Before the stimulus | During the stimulus | Difference | Trials / flies | "
         "Event-split permutation p |", "|---|---|---|---|---|---|---|"]
    for kind in KINDS:
        for event in EVENTS:
            c = ev[event][kind]
            if not c["n_trials"]:
                L.append(f"| {kind} | {EVENT_LABELS[event]} | no trial | no trial | n/a | {FIG.n_label(0, 0)} | n/a |")
                continue
            L.append(f"| {kind} | {EVENT_LABELS[event]} | {_rate_cell(c['pre'])} | {_rate_cell(c['stim'])} | "
                     f"{_signed_rate(c['diff'])}/min | {FIG.n_label(c['n_trials'], c['n_flies'])} | "
                     f"{FIG.p_text(c['p_perm'], 4)} (uncorrected) |")
    L += [""]
    for kind in KINDS:
        for event in EVENTS:
            c = ev[event][kind]
            if c["n_trials"]:
                L.append(f"- {EVENT_LABELS[event]} before vs during {WINDOW_LABELS[kind]}: **{c['verdict']}**"
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
          "(positive = the tip points down). Only windows judged usable (`abd_quality_ok`) of identity-consistent flies "
          "enter a number.", "",
          "How to read the statistics:", "",
          "- The fly is the unit of inference. Per-fly figures use the fly table; pooled rates use the trials of "
          "identity-consistent flies and divide pooled events by pooled minutes (never a mean of per-trial rates).",
          "- A curl is a sustained rise of the bend above the fly's resting bend (the median bend before the stimulus), a "
          "straightening a sustained fall below it; thresholds `curl_rise_deg` and `curl_min_s` of the `mounting:` block.",
          "- Mount pitch is rig-centred (fly value minus the median of its rig) because camera roll differs per rig; the "
          "tip pitch is rig-centred the same way, since the roll shifts it too.",
          f"- Stratified permutation p: the x variable is shuffled among flies of the same dataset × rig, two-sided on |ρ|, "
          f"{n_perm} permutations, seed {seed}.",
          f"- Event-split permutation p: the events of each trial are re-assigned between its two windows at random, each "
          f"to the stimulus window with probability proportional to that window's length (a trial's total, and what makes "
          f"it busy, are kept, and a short stimulus window is compared fairly with the longer one before it), and the "
          f"pooled rate difference (during minus before) is recomputed; two-sided, {n_perm} permutations, seed {seed}.",
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
          "Figure `abdomen_curl_events_by_window.png`: curl and straightening events per minute, pooled over trials, before "
          "vs during the odor and before vs during the light (light-only trials; hatched bars), each pair from the same "
          "trials with a usable abdomen window on both sides.", "", *_apart(ph(stats, "abdomen_curl_events_by_window")),
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
