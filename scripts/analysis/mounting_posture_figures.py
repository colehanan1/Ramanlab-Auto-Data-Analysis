"""The ``Mounting-Posture/`` figure set and ``report.md`` of the mounting analysis.

Takes the two tables of ``mounting_posture.py`` -- the fly-trial table (``fly_trial_posture.csv``) and the fly table
(``fly_posture.csv``) -- and draws six figures. Each is titled with the QUESTION, answered in plain words computed from
the data (``Yes -- flies tilted more head-up show more spontaneous PER`` / ``No clear link``), the numbers in a smaller
line under it:

1. ``mount_pitch_per_fly``               strip plot of the mounting angle by rig;
2. ``mount_pitch_vs_spontaneous_per``    per fly: rig-centred pitch vs % of pre-stimulus windows with a spontaneous PER;
3. ``mount_pitch_vs_held_out``           per fly: rig-centred pitch vs held-out fraction (marker fill = dominant sub-type);
4. ``response_by_mount_pitch_tercile``   pooled-trial response rate per tercile of the rig-centred pitch;
5. ``baseline_state_vs_response``        pooled bars: response by proboscis state and by spontaneous PER;
6. ``pullins_held_vs_retracted``         pull-ins per minute before / during the stimulus, held vs retracted (descriptive).

Inference uses the FLY as the unit: the per-fly scatters report Spearman rho / p and a permutation p stratified by
dataset x rig, the tercile p is the same per-fly test of response rate vs pitch, and the baseline-state p shuffles the
state inside each fly's trials of one trial type (so a pretest-vs-testing difference cannot make it small), and the
difference shown beside it is the Mantel-Haenszel one over those same fly x trial-type groups (the bars stay pooled).
The response analyses use only scored trials (pretest + testing trials that have a score). Bars pool TRIALS (never a
mean of per-fly means) and say ``n=... trials / ... flies``. Every p-value is uncorrected, and says so; a comparison
with fewer than three flies to compare is "Not enough data to tell". A figure that cannot be drawn (too few flies) is
a placeholder that says why; one that fails to build leaves no picture behind and is named in the report, and a run
with nothing in scope removes the pictures of an earlier one.

``compute_stats`` does all the arithmetic once; ``build_figures`` / ``render_figures`` and ``write_report`` only
format it. ``write_report`` assembles ``report.md`` from section strings (``report_sections``) and takes
``extra_sections``, so a further analysis can append its own.
"""
from __future__ import annotations

import logging
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fbpipe.analysis import mounting as M  # noqa: E402
from fbpipe.plot_style import apply_lab_style  # noqa: E402

LOGGER = logging.getLogger("mounting_posture_figures")

ALPHA = 0.05                  # "Yes" in a title only below this (uncorrected) p
MIN_FLIES = 3                 # a per-fly scatter / tercile figure / baseline comparison needs this many flies
SCORED_TYPES = ("pretest", "testing")        # the trial types that get a score: the only ones the response analyses use
DPI = 300
REPORT_NAME = "report.md"

# Colours: blue / orange are categorical slots 1 and 2 of the validated palette; held / spontaneous states are orange
# with a slash hatch (the texture, not the hue alone, carries "state"). Text never wears a series colour.
BLUE, ORANGE, HATCH = "#2a78d6", "#eb6834", "//"
SURFACE, INK, INK_SECONDARY, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BAR_IN = 0.22                 # bar thickness in inches (under the 24 px cap)

_RC = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8, "text.color": INK, "axes.labelcolor": INK_SECONDARY, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "hatch.linewidth": 1.0,
    "axes.unicode_minus": False,           # tick labels use the same "-" as the numbers written in the text
}

NAMES = ("mount_pitch_per_fly", "mount_pitch_vs_spontaneous_per", "mount_pitch_vs_held_out",
         "response_by_mount_pitch_tercile", "baseline_state_vs_response", "pullins_held_vs_retracted")
TITLES = {
    "mount_pitch_per_fly": "How are the flies mounted on each rig?",
    "mount_pitch_vs_spontaneous_per": "Does mounting angle change spontaneous PER?",
    "mount_pitch_vs_held_out": "Does mounting angle change how often the proboscis is held out?",
    "response_by_mount_pitch_tercile": "Do flies mounted at different angles respond differently?",
    "baseline_state_vs_response": "Does the state before the stimulus predict the response?",
    "pullins_held_vs_retracted": "Do held-out proboscises pull in when the stimulus starts?",
}
STATES = ("retracted", "held_down", "held_back", "held_forward")
STATE_LABELS = {"retracted": "Retracted", "held_down": "Held down", "held_back": "Held back", "held_forward": "Held forward",
                "no_spont": "No spontaneous PER", "spont": "Spontaneous PER"}
THIRDS = (("lowest", "Lowest third", "most head-down"), ("middle", "Middle third", ""),
          ("highest", "Highest third", "most head-up"))
# Marker per fly in the held-out scatter: fill = the fly's dominant held sub-type (filled = back, hollow = down). Areas are
# chosen so a filled and a hollow marker weigh the same; a fly that was never held out gets a cross.
SUBTYPES = ("held_back", "held_down", "held_forward", None)
SUBTYPE_LABELS = {"held_back": "mostly held back", "held_down": "mostly held down", "held_forward": "mostly held forward",
                  None: "no held sub-type"}
SUBTYPE_MARKS = {"held_back": dict(marker="o", s=60, face=BLUE, edge=SURFACE, lw=1.0),
                 "held_down": dict(marker="o", s=34, face=SURFACE, edge=BLUE, lw=1.3),
                 "held_forward": dict(marker="^", s=60, face=BLUE, edge=SURFACE, lw=1.0),
                 None: dict(marker="x", s=40, face=BLUE, edge=BLUE, lw=1.3)}
POINT_MARK = SUBTYPE_MARKS["held_back"]


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def p_text(p, digits: int = 3) -> str:
    """``p=0.61`` / ``p=0.012`` / ``p<0.001`` (figures, ``digits=3``); ``digits=4`` keeps four decimals for the report."""
    if p is None or not np.isfinite(p):
        return "p=n/a"
    floor = 10.0 ** -digits
    if p < floor:
        return f"p<{floor:.{digits}f}"
    if digits == 3:
        return f"p={p:.3f}" if p < 0.1 else f"p={p:.2f}"
    return f"p={p:.{digits}f}"


def _rho_text(rho, digits: int = 2) -> str:
    return f"ρ={rho:.{digits}f}" if rho is not None and np.isfinite(rho) else "ρ=n/a"


def _count(n: int, one: str, many: str) -> str:
    """``1 fly`` / ``2 flies``: a count with its noun in the right number (thousands separated)."""
    return f"{n:,} {one if n == 1 else many}"


def n_flies_label(flies: int) -> str:
    return f"n={_count(flies, 'fly', 'flies')}"


def n_label(trials: int, flies: int) -> str:
    """``n=120 trials / 14 flies``: the pooled-trial label of every bar."""
    return f"n={_count(trials, 'trial', 'trials')} / {_count(flies, 'fly', 'flies')}"


def _points(diff: float, digits: int = 0) -> str:
    """A difference of two proportions in percentage points (``+9 points``); ``n/a`` when there is none."""
    if not np.isfinite(diff):
        return "n/a"
    return f"{round(100 * diff, digits) + 0.0:+.{digits}f} points"                # + 0.0: a rounded -0.0 prints as +0


def _group_n(group: Mapping) -> str:
    """``n_label`` of a stats entry that carries ``n_trials`` and ``n_flies``."""
    return n_label(group["n_trials"], group["n_flies"])


def verdict(p, effect, *, up: str, down: str, none: str = "No clear link") -> str:
    """The plain-language answer: ``Yes -- <up|down>`` when ``p < ALPHA`` (by the sign of *effect*), else *none*."""
    if p is None or not np.isfinite(p):
        return "Not enough data to tell"
    if p < ALPHA:
        return f"Yes — {up if effect > 0 else down}"
    return none


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    """Column *name* as floats (NaN for missing / None); all NaN when the column is absent."""
    return df[name].to_numpy(dtype=float, na_value=np.nan) if name in df else np.full(len(df), np.nan)


def _flag(series: pd.Series) -> pd.Series:
    """A boolean column that may have been through a CSV (bool, 0/1 or "True"/"False" text)."""
    if series.dtype == bool:
        return series
    return series.map(lambda v: str(v).strip().lower() in ("true", "1", "1.0"))


def _consistent(fly_trial: pd.DataFrame) -> pd.DataFrame:
    """The fly-trials of identity-consistent flies: the only ones a pooled bar may count."""
    if fly_trial.empty or "identity_consistent" not in fly_trial:
        return fly_trial
    return fly_trial[_flag(fly_trial["identity_consistent"])]


def _scored(trials: pd.DataFrame) -> pd.DataFrame:
    """The trials the response analyses use: pretest + testing trials that have a score (training trials get none)."""
    if trials.empty:
        return trials
    return trials[trials["trial_type"].isin(SCORED_TYPES) & trials["score"].notna() & trials["responded"].notna()]


def _strata(fly: pd.DataFrame) -> np.ndarray:
    return (fly["dataset"].astype(str) + " | " + fly["rig"].astype(str)).to_numpy()


def _rate(events: float, minutes: float) -> float:
    return float(events / minutes) if minutes > 0 else float("nan")


def _num(v: float) -> str:
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.2f}" if v < 1 else f"{v:.1f}" if v < 10 else f"{v:.0f}"


# ---------------------------------------------------------------------------
# Statistics (the one place any number is computed)
# ---------------------------------------------------------------------------

def _rig_summary(fly: pd.DataFrame) -> List[dict]:
    frame = pd.DataFrame({"rig": fly["rig"].astype(str).to_numpy(), "pitch": _col(fly, "mount_pitch_deg")}).dropna()
    out = []
    for rig, g in frame.groupby("rig", sort=True):
        q1, med, q3 = np.percentile(g["pitch"].to_numpy(float), [25, 50, 75])
        out.append({"rig": str(rig), "n_flies": int(len(g)), "median_deg": float(med),
                    "q1_deg": float(q1), "q3_deg": float(q3)})
    return out


def _rig_answer(rigs: Sequence[dict]) -> str:
    if not rigs:
        return "No fly has a mounting angle"
    lo, hi = min(rigs, key=lambda r: r["median_deg"]), max(rigs, key=lambda r: r["median_deg"])
    if len(rigs) == 1:
        return f"One rig: the median mounting angle is {lo['median_deg']:+.1f}° ({lo['rig']})"
    return (f"Rig medians span {hi['median_deg'] - lo['median_deg']:.1f}° ({lo['rig']} {lo['median_deg']:+.1f}° to "
            f"{hi['rig']} {hi['median_deg']:+.1f}°)")


def _scatter_stats(fly: pd.DataFrame, column: str, *, n_perm: int, seed: int, up: str, down: str,
                   none: str = "No clear link") -> dict:
    """Per-fly rig-centred pitch vs *column*: Spearman rho / p, the dataset x rig stratified permutation p, n."""
    x, y, strata = _col(fly, "mount_pitch_centred_deg"), _col(fly, column), _strata(fly)
    done = np.isfinite(x) & np.isfinite(y)
    rho, p = M.spearman(x, y)
    p_perm = M.stratified_permutation_p(x, y, strata, n=n_perm, seed=seed)
    sizes = pd.Series(strata[done]).value_counts()
    answer = verdict(p_perm, rho, up=up, down=down, none=none)
    if not np.isfinite(p_perm) and done.sum() >= MIN_FLIES:
        answer += " (every fly has the same value of one of the two)"
    return {"n_flies": int(done.sum()), "rho": rho, "p": p, "p_perm": p_perm, "n_strata": int(len(sizes)),
            "n_strata_shuffled": int((sizes > 1).sum()), "verdict": answer}


def _tercile_stats(fly: pd.DataFrame, trials: pd.DataFrame, *, n_perm: int, seed: int) -> dict:
    """The per-fly test of response rate vs pitch, and the pooled-trial response rate in each third of the pitch."""
    out = _scatter_stats(fly, "response_rate", n_perm=n_perm, seed=seed, up="flies tilted more head-up respond more often",
                         down="flies tilted more head-down respond more often", none="No clear difference")
    out.update(bins=[], n_trials=0)
    pitch, rate = _col(fly, "mount_pitch_centred_deg"), _col(fly, "response_rate")
    use = fly[np.isfinite(pitch) & np.isfinite(rate)]
    if len(use) < MIN_FLIES:
        return out
    ids, centred = use["fly_id"].to_numpy(), _col(use, "mount_pitch_centred_deg")
    scored = _scored(trials)
    for (name, _, _), idx in zip(THIRDS, np.array_split(np.argsort(centred, kind="stable"), 3)):
        rows = scored[scored["fly_id"].isin(ids[idx])]
        k = float(rows["responded"].sum())
        out["bins"].append({"name": name, "lo_deg": float(centred[idx].min()), "hi_deg": float(centred[idx].max()),
                            "n_flies": int(len(idx)), "n_trials": int(len(rows)), "n_responded": int(k),
                            "rate": k / len(rows) if len(rows) else float("nan")})
    out["n_trials"] = sum(b["n_trials"] for b in out["bins"])
    return out


def _bars(df: pd.DataFrame, states: Sequence[Tuple[str, pd.Series]]) -> List[dict]:
    out = []
    for state, mask in states:
        rows = df[mask]
        k = float(rows["responded"].sum())
        out.append({"state": state, "n_trials": int(len(rows)), "n_flies": int(rows["fly_id"].nunique()),
                    "n_responded": int(k), "rate": k / len(rows) if len(rows) else float("nan")})
    return out


def _within_fly(df: pd.DataFrame, state: str, *, n_perm: int, seed: int, up: str, down: str) -> dict:
    """The within-fly permutation p of a state vs response and the difference that goes with it.

    The state is shuffled inside each (fly, trial type) group, so a difference between pretest and testing (where state
    and response may both shift) cannot make the p small; only groups seen in both states count, and fewer than
    ``MIN_FLIES`` flies among them is "Not enough data to tell" whatever the p. ``diff`` is the Mantel-Haenszel risk
    difference over those same groups, so a pretest-vs-testing shift cannot inflate it either; ``diff_pooled`` is the
    pooled difference of ``within_fly_permutation`` (what the bars show), kept for the report.
    """
    groups = df.assign(fly_phase=df["fly_id"].astype(str) + "|" + df["trial_type"].astype(str))
    diff_pooled, p = M.within_fly_permutation(groups, state=state, outcome="responded", fly="fly_phase", n=n_perm, seed=seed)
    diff = M.stratified_risk_difference(groups, state=state, outcome="responded", stratum="fly_phase")
    states = groups.groupby("fly_phase")[state].nunique()
    used = groups[groups["fly_phase"].isin(states[states == 2].index)]
    n_flies = int(used["fly_id"].nunique())
    answer = verdict(p if n_flies >= MIN_FLIES else float("nan"), diff, up=up, down=down, none="No clear difference")
    return {"diff": diff, "diff_pooled": diff_pooled, "p_perm": p, "n_flies_both": n_flies,
            "n_groups_both": int(used["fly_phase"].nunique()), "n_trials_both": int(len(used)), "verdict": answer}


def _baseline_stats(trials: pd.DataFrame, *, n_perm: int, seed: int) -> dict:
    scored = _scored(trials)
    known = scored[scored["held_subtype"].notna()]
    known = known.assign(held_any=(known["held_subtype"] != "retracted").astype(float))
    present = [s for s in STATES if s in ("retracted", "held_down", "held_back") or (known["held_subtype"] == s).any()]
    spont = scored[scored["spont_per_any"].notna()]
    return {
        "proboscis": {"bars": _bars(known, [(s, known["held_subtype"] == s) for s in present]),
                      **_within_fly(known, "held_any", n_perm=n_perm, seed=seed,
                                    up="trials with the proboscis held out respond more often",
                                    down="trials with the proboscis held out respond less often")},
        "spontaneous": {"bars": _bars(spont, [("no_spont", spont["spont_per_any"] == 0),
                                              ("spont", spont["spont_per_any"] == 1)]),
                        **_within_fly(spont, "spont_per_any", n_perm=n_perm, seed=seed,
                                      up="trials with a spontaneous PER respond more often",
                                      down="trials with a spontaneous PER respond less often")}}


def _pullin_stats(trials: pd.DataFrame) -> dict:
    """Pull-ins per minute before / during the stimulus: pooled events over pooled minutes, held vs retracted trials."""
    cols = ["pullin_events_pre", "pullin_events_stim", "pre_min", "stim_min"]
    use = trials[trials["held_subtype"].notna()]
    if len(use):
        values = np.column_stack([_col(use, c) for c in cols])
        use = use[np.isfinite(values).all(axis=1) & (values[:, 2] > 0) & (values[:, 3] > 0)]
    held = use["held_subtype"] != "retracted"
    groups = {}
    for name, rows in (("retracted", use[~held]), ("held", use[held])):
        ev_pre, ev_stim = float(rows["pullin_events_pre"].sum()), float(rows["pullin_events_stim"].sum())
        min_pre, min_stim = float(rows["pre_min"].sum()), float(rows["stim_min"].sum())
        groups[name] = {"n_trials": int(len(rows)), "n_flies": int(rows["fly_id"].nunique()), "events_pre": ev_pre,
                        "minutes_pre": min_pre, "rate_pre": _rate(ev_pre, min_pre), "events_stim": ev_stim,
                        "minutes_stim": min_stim, "rate_stim": _rate(ev_stim, min_stim)}
    return {"groups": groups, "answer": _pullin_answer(groups)}


def _pullin_answer(groups: Mapping[str, dict]) -> str:
    def part(label, g):
        return f"{label}: {_num(g['rate_pre'])} → {_num(g['rate_stim'])}" if g["n_trials"] else f"{label}: no trials"
    return f"{part('Held-out trials', groups['held'])}; {part('retracted', groups['retracted'])} (pull-ins per min)"


def compute_stats(fly_trial: pd.DataFrame, fly: pd.DataFrame, *, n_perm: int, seed: int = 0) -> dict:
    """Every number the figures and the report show. ``placeholders`` names the figures that cannot be drawn and why."""
    trials = _consistent(fly_trial)
    stats: dict = {"n_perm": int(n_perm), "seed": int(seed), "alpha": ALPHA, "n_fly_trials": int(len(fly_trial)),
                   "n_trials": int(len(trials)), "n_flies": int(len(fly)), "placeholders": {}}
    if fly_trial.empty:
        return stats
    stats["rigs"] = _rig_summary(fly)
    stats["rig_answer"] = _rig_answer(stats["rigs"])
    stats["spontaneous"] = _scatter_stats(
        fly, "spont_per_frac", n_perm=n_perm, seed=seed, up="flies tilted more head-up show more spontaneous PER",
        down="flies tilted more head-down show more spontaneous PER")
    stats["held_out"] = _scatter_stats(
        fly, "held_out_frac", n_perm=n_perm, seed=seed, up="flies tilted more head-up hold the proboscis out more often",
        down="flies tilted more head-down hold the proboscis out more often")
    stats["tercile"] = _tercile_stats(fly, trials, n_perm=n_perm, seed=seed)
    stats["baseline"] = _baseline_stats(trials, n_perm=n_perm, seed=seed)
    stats["pullins"] = _pullin_stats(trials)
    holes = stats["placeholders"]
    if not stats["rigs"]:
        holes["mount_pitch_per_fly"] = {"reason": "no fly has a mounting angle", "n": n_flies_label(0)}
    for name, key in (("mount_pitch_vs_spontaneous_per", "spontaneous"), ("mount_pitch_vs_held_out", "held_out"),
                      ("response_by_mount_pitch_tercile", "tercile")):
        if stats[key]["n_flies"] < MIN_FLIES:
            holes[name] = {"reason": f"needs at least {MIN_FLIES} flies with both values (have {stats[key]['n_flies']})",
                           "n": n_flies_label(stats[key]["n_flies"])}
    if not sum(b["n_trials"] for panel in stats["baseline"].values() for b in panel["bars"]):
        holes["baseline_state_vs_response"] = {
            "reason": "no scored trial has a known proboscis state or spontaneous-PER state", "n": n_label(0, 0)}
    if not sum(g["n_trials"] for g in stats["pullins"]["groups"].values()):
        holes["pullins_held_vs_retracted"] = {
            "reason": "no trial has a known proboscis state and pull-in counts", "n": n_label(0, 0)}
    return stats


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

@contextmanager
def _style() -> Iterator[None]:
    """The lab style, the figure rc and 300 dpi for the figures built inside; the global matplotlib state is restored."""
    with matplotlib.rc_context():
        apply_lab_style(dpi=DPI)
        matplotlib.rcParams.update(_RC)
        yield


def _new_figure(width: float, height: float) -> Figure:
    fig = Figure(figsize=(width, height), facecolor=SURFACE)
    FigureCanvasAgg(fig)
    return fig


_GLUE = re.compile(r"n=[\d,]+ (?:trials?|flies|fly)(?: / [\d,]+ (?:flies|fly))?|p[=<][\d.]+ \(uncorrected\)"
                   r"|[+-]\d+(?:\.\d+)? points")


def _wrap(fig: Figure, text: str, size: float, weight: str, max_in: float) -> List[str]:
    """*text* broken at spaces into lines no wider than *max_in* inches (measured with the real font).

    An ``n=...`` label, a ``p=... (uncorrected)`` and a ``+9 points`` stay on one line (their spaces are non-breaking).
    """
    renderer = fig.canvas.get_renderer()
    probe = fig.text(0, 0, "", fontsize=size, weight=weight)
    lines: List[str] = []
    for paragraph in _GLUE.sub(lambda m: m.group(0).replace(" ", "\u00a0"), text).split("\n"):
        line = ""
        for word in paragraph.split(" "):
            trial = f"{line} {word}".strip()
            probe.set_text(trial)
            if line and probe.get_window_extent(renderer).width / fig.dpi > max_in:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    probe.remove()
    return lines


def _title_block(fig: Figure, title: str, answer: str = "", numbers: str = "") -> float:
    """The question (bold), the computed answer and the numbers (small) at the top left; returns the inches used."""
    width, height = fig.get_size_inches()
    x, room, used = 0.3 / width, width - 0.6, 0.2
    for text, size, weight, color, line_in, gap in ((title, 13, "bold", INK, 0.208, 0.07),
                                                     (answer, 10.5, "normal", INK, 0.19, 0.05),
                                                     (numbers, 8.5, "normal", INK_SECONDARY, 0.165, 0.0)):
        if not text:
            continue
        lines = _wrap(fig, text, size, weight, room)
        fig.text(x, 1 - used / height, "\n".join(lines), ha="left", va="top", fontsize=size, weight=weight, color=color,
                 linespacing=1.15 if size == 13 else 1.3 if size > 9 else 1.4)
        used += line_in * len(lines) + gap
    return used + 0.12


def _axes(fig: Figure, left: float, bottom: float, width: float, height: float):
    """An axes placed in inches (left / bottom from the figure's lower left corner), recessive grid, two spines."""
    w, h = fig.get_size_inches()
    ax = fig.add_axes([left / w, bottom / h, width / w, height / h])
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6, linestyle="-")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction="out", length=3, colors=MUTED)
    ax.tick_params(axis="x", labelcolor=INK_SECONDARY)
    return ax


def _pct_axis(ax, top: float) -> None:
    """A 0-100 % axis that may run past 100 for headroom (ticks stop at 100)."""
    step = next(s for s in (5, 10, 20, 25) if min(top, 100) / s <= 5)
    ax.set_ylim(0, top)
    ax.set_yticks(np.arange(0, min(top, 100) + 1e-9, step))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))


def _headroom(rates_pct) -> float:
    finite = [r for r in rates_pct if np.isfinite(r)]
    return max(20.0, 1.2 * max(finite, default=0.0))


def _bar(ax, x: float, height: float, width: float, *, held: bool):
    return ax.bar(x, height, width=width, facecolor=ORANGE if held else BLUE, edgecolor="white" if held else "none",
                  hatch=HATCH if held else None, linewidth=0, zorder=3)[0]


def _tip(ax, x: float, height: float, text: str, top: float) -> None:
    ax.text(x, height + 0.015 * top, text, ha="center", va="bottom", fontsize=8.5, color=INK, zorder=5)


def _n_lines(trials: int, flies: int) -> str:
    return f"n={_count(trials, 'trial', 'trials')}\n/ {_count(flies, 'fly', 'flies')}"


def _bar_axes(fig: Figure, left: float, bottom: float, width: float, height: float, n_slots: int):
    """Axes with *n_slots* category slots; returns it and the bar width (data units) that makes a bar ``BAR_IN`` thick."""
    ax = _axes(fig, left, bottom, width, height)
    ax.set_xlim(-0.6, n_slots - 0.4)
    ax.set_xticks(range(n_slots))
    return ax, BAR_IN * (n_slots + 0.2) / width


def _placeholder(name: str, stats: dict) -> Figure:
    info = stats["placeholders"][name]
    fig = _new_figure(6.8, 4.4)
    _title_block(fig, TITLES[name], f"Not enough data — {info['reason']}", info["n"])
    fig.text(0.5, 0.4, "No figure: not enough data", ha="center", va="center", fontsize=11, color=MUTED)
    return fig


def _fig_pitch_per_fly(trials: pd.DataFrame, fly: pd.DataFrame, stats: dict) -> Figure:
    rigs = stats["rigs"]
    width, height = max(6.0, 1.5 * len(rigs) + 2.8), 5.2
    fig = _new_figure(width, height)
    top = _title_block(fig, TITLES["mount_pitch_per_fly"], stats["rig_answer"],
                       f"Camera roll differs per rig, so angles are compared within a rig; "
                       f"{n_flies_label(sum(r['n_flies'] for r in rigs))}")
    ax = _axes(fig, 0.95, 0.8, width - 1.3, height - top - 0.8 - 0.1)
    rng = np.random.default_rng(stats["seed"])
    for i, r in enumerate(rigs):
        v = _col(fly[fly["rig"].astype(str) == r["rig"]], "mount_pitch_deg")
        v = v[np.isfinite(v)]
        ax.vlines(i, r["q1_deg"], r["q3_deg"], color=BLUE, linewidth=2.0, alpha=0.30, zorder=2)
        ax.hlines(r["median_deg"], i - 0.28, i + 0.28, color=INK_SECONDARY, linewidth=1.6, zorder=4)
        ax.scatter(i + rng.uniform(-0.16, 0.16, v.size), v, s=POINT_MARK["s"], facecolor=BLUE, edgecolor=SURFACE,
                   linewidth=1.0, zorder=3)
        ax.text(i + 0.33, r["median_deg"], f"{r['median_deg']:+.1f}°", ha="left", va="center", fontsize=8.5,
                color=INK_SECONDARY)
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.set_xlim(-0.6, len(rigs) - 0.4)
    ax.set_xticks(range(len(rigs)))
    ax.set_xticklabels([f"{r['rig']}\n{n_flies_label(r['n_flies'])}" for r in rigs])
    ax.set_ylabel("Body pitch (°; + = head up)")
    ax.set_xlabel("dot = one fly    bar = median    light line = interquartile range", color=MUTED, fontsize=8.5)
    return fig


def _scatter_axes(ax, x: np.ndarray, y: np.ndarray, ylabel: str) -> None:
    top = _headroom(y)
    ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
    pad = 0.08 * (np.ptp(x) or 1.0)
    ax.set_xlim(x.min() - pad, x.max() + pad)
    _pct_axis(ax, top)
    ax.set_ylim(-0.04 * top, top)
    ax.set_xlabel("Mounting angle relative to the rig median (°)")
    for x_axes, align, hint in ((0.0, "left", "← head-down"), (1.0, "right", "head-up →")):
        ax.annotate(hint, xy=(x_axes, 0), xycoords="axes fraction", xytext=(0, -19), textcoords="offset points",
                    ha=align, va="top", fontsize=8.5, color=MUTED, annotation_clip=False)
    ax.set_ylabel(ylabel)


def _scatter_numbers(s: dict) -> str:
    return (f"{_rho_text(s['rho'])}, {p_text(s['p'])} (uncorrected); stratified permutation {p_text(s['p_perm'])} "
            f"(uncorrected); {n_flies_label(s['n_flies'])}")


def _subtype_scatter(fig: Figure, ax, fly: pd.DataFrame, x: np.ndarray, y: np.ndarray, done: np.ndarray) -> None:
    """One marker style per dominant held sub-type (filled = back, hollow = down) and the legend that names them."""
    dominant = [d if d in SUBTYPES else None for d in fly["held_subtype_dominant"]]
    handles = []
    for kind in SUBTYPES:
        mine = done & np.array([d == kind for d in dominant])
        if not mine.any():
            continue
        m = SUBTYPE_MARKS[kind]
        colours = dict(color=m["face"]) if m["marker"] == "x" else dict(facecolor=m["face"], edgecolor=m["edge"])
        ax.scatter(x[mine], y[mine], s=m["s"], marker=m["marker"], linewidths=m["lw"], zorder=3,
                   label=SUBTYPE_LABELS[kind], **colours)
        handles.append(Line2D([], [], linestyle="", marker=m["marker"], markersize=m["s"] ** 0.5,
                              markerfacecolor=m["face"], markeredgecolor=m["edge"], markeredgewidth=m["lw"],
                              label=SUBTYPE_LABELS[kind]))
    legend = fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
                        bbox_to_anchor=(0.5, 0.02), title="Dominant proboscis state when held out", title_fontsize=8.5,
                        handletextpad=0.3, columnspacing=1.6)
    for text in [legend.get_title(), *legend.get_texts()]:
        text.set_color(INK_SECONDARY)


def _fig_scatter(name: str, stats: dict, fly: pd.DataFrame, key: str, column: str, ylabel: str,
                 *, kinds: bool = False) -> Figure:
    s = stats[key]
    width, height, bottom = 6.8, 5.6, (1.25 if kinds else 0.8)
    fig = _new_figure(width, height)
    top = _title_block(fig, TITLES[name], s["verdict"], _scatter_numbers(s))
    ax = _axes(fig, 0.9, bottom, width - 1.2, height - top - bottom - 0.1)
    x, y = _col(fly, "mount_pitch_centred_deg"), 100.0 * _col(fly, column)
    done = np.isfinite(x) & np.isfinite(y)
    if kinds:
        _subtype_scatter(fig, ax, fly, x, y, done)
    else:
        ax.scatter(x[done], y[done], s=POINT_MARK["s"], facecolor=BLUE, edgecolor=SURFACE, linewidth=1.0, zorder=3)
    _scatter_axes(ax, x[done], y[done], ylabel)
    return fig


def _fig_spontaneous(trials, fly, stats):
    return _fig_scatter("mount_pitch_vs_spontaneous_per", stats, fly, "spontaneous", "spont_per_frac",
                        "Pre-stimulus windows with a spontaneous PER")


def _fig_held_out(trials, fly, stats):
    return _fig_scatter("mount_pitch_vs_held_out", stats, fly, "held_out", "held_out_frac",
                        "Trials with the proboscis held out", kinds=True)


def _fig_tercile(trials, fly, stats):
    t = stats["tercile"]
    width, height, bottom = 6.8, 5.5, 1.3
    fig = _new_figure(width, height)
    numbers = (f"per-fly {_rho_text(t['rho'])}, {p_text(t['p'])} (uncorrected); stratified permutation "
               f"{p_text(t['p_perm'])} (uncorrected); bars pool the pretest + testing trials, {_group_n(t)}")
    top = _title_block(fig, TITLES["response_by_mount_pitch_tercile"], t["verdict"], numbers)
    ax, bar_w = _bar_axes(fig, 0.95, bottom, width - 1.25, height - top - bottom - 0.1, len(t["bins"]))
    ceiling = _headroom([100 * b["rate"] for b in t["bins"]])
    _pct_axis(ax, ceiling)
    labels = []
    for i, (b, (_, name, note)) in enumerate(zip(t["bins"], THIRDS)):
        if np.isfinite(b["rate"]):
            _bar(ax, i, 100 * b["rate"], bar_w, held=False)
            _tip(ax, i, 100 * b["rate"], f"{100 * b['rate']:.0f}%", ceiling)
        labels.append(f"{name}{f' ({note})' if note else ''}\n{b['lo_deg']:+.1f}° to {b['hi_deg']:+.1f}°\n"
                      f"{_n_lines(b['n_trials'], b['n_flies'])}")
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Trials with a response")
    ax.set_xlabel("Third of the rig-centred mounting angle (pretest + testing trials pooled)")
    return fig


def _panel_note(panel: dict, what: str) -> str:
    n = panel["n_flies_both"]
    if not n:
        return f"{what}: no fly was seen in both states, so there is nothing to compare"
    have = f"{_count(n, 'fly', 'flies')} {'has' if n == 1 else 'have'} both states"
    if n < MIN_FLIES:
        have += f" (needs at least {MIN_FLIES})"
    return (f"{what}, within fly × phase (MH): {_points(panel['diff'])}; within-fly permutation "
            f"{p_text(panel['p_perm'])} (uncorrected); {have}")


def _fig_baseline(trials, fly, stats):
    pro, spo = stats["baseline"]["proboscis"], stats["baseline"]["spontaneous"]
    slot, left, gap, right, bottom, height = 1.45, 0.95, 0.9, 0.3, 1.05, 5.4
    widths = [slot * (len(pro["bars"]) + 0.2), slot * (len(spo["bars"]) + 0.2)]
    width = left + sum(widths) + gap + right
    fig = _new_figure(width, height)
    answer = f"Proboscis position: {pro['verdict']}\nSpontaneous PER: {spo['verdict']}"
    used = _title_block(fig, TITLES["baseline_state_vs_response"], answer,
                        "Bars: pooled pretest + testing trials. Differences (MH = Mantel–Haenszel) and p-values "
                        "(uncorrected): within fly × trial type, the state shuffled among a fly's trials of the same type")
    notes = [_wrap(fig, _panel_note(pro, "Held out minus retracted"), 8, "normal", widths[0]),
             _wrap(fig, _panel_note(spo, "Spontaneous PER minus none"), 8, "normal", widths[1])]
    axes_top = used + 0.28 + 0.14 * max(len(n) for n in notes) + 0.12
    ceiling = _headroom([100 * b["rate"] for p in (pro, spo) for b in p["bars"]])
    corner = left
    headings = ("Proboscis position at rest", "Spontaneous PER before the stimulus")
    for panel, heading, w, note in zip((pro, spo), headings, widths, notes):
        ax, bar_w = _bar_axes(fig, corner, bottom, w, height - axes_top - bottom, len(panel["bars"]))
        _pct_axis(ax, ceiling)
        labels = []
        for i, b in enumerate(panel["bars"]):
            if b["n_trials"]:
                _bar(ax, i, 100 * b["rate"], bar_w, held=b["state"] not in ("retracted", "no_spont"))
                _tip(ax, i, 100 * b["rate"], f"{100 * b['rate']:.0f}%", ceiling)
            labels.append(f"{STATE_LABELS[b['state']]}\n{_n_lines(b['n_trials'], b['n_flies'])}")
        ax.set_xticklabels(labels, fontsize=8)
        fig.text(corner / width, 1 - used / height, heading, ha="left", va="top", fontsize=10, weight="bold", color=INK)
        fig.text(corner / width, 1 - (used + 0.25) / height, "\n".join(note), ha="left", va="top", fontsize=8,
                 color=INK_SECONDARY, linespacing=1.4)
        corner += w + gap
    fig.axes[0].set_ylabel("Trials with a response")
    return fig


def _fig_pullins(trials, fly, stats):
    groups = stats["pullins"]["groups"]
    width, height, bottom = 6.8, 5.0, 1.05
    fig = _new_figure(width, height)
    top = _title_block(fig, TITLES["pullins_held_vs_retracted"], stats["pullins"]["answer"],
                       "Before → during the stimulus; pooled events over pooled minutes, all trial types; "
                       "descriptive, no test")
    ax, _ = _bar_axes(fig, 0.95, bottom, width - 1.25, height - top - bottom - 0.1, 2)
    bar_w = BAR_IN * 2.2 / (width - 1.25)
    rates = [v for g in groups.values() for v in (g["rate_pre"], g["rate_stim"]) if np.isfinite(v)]
    ceiling = max(1.0, 1.2 * max(rates, default=0.0))
    ax.set_ylim(0, ceiling)
    for slot, key in enumerate(("rate_pre", "rate_stim")):
        for name, shift in (("retracted", -0.5 * bar_w - 0.012), ("held", 0.5 * bar_w + 0.012)):
            if np.isfinite(groups[name][key]):
                _bar(ax, slot + shift, groups[name][key], bar_w, held=name == "held")
                _tip(ax, slot + shift, groups[name][key], _num(groups[name][key]), ceiling)
    ax.set_xticklabels(["Before the stimulus", "During the stimulus"])
    ax.set_ylabel("Pull-in events per minute")
    swatches = [Patch(facecolor=BLUE, linewidth=0, label=f"Retracted ({_group_n(groups['retracted'])})"),
                Patch(facecolor=ORANGE, edgecolor="white", hatch=HATCH, linewidth=0,
                      label=f"Held out ({_group_n(groups['held'])})")]
    legend = fig.legend(handles=swatches, loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, 0.02))
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    return fig


BUILDERS: Dict[str, Callable[[pd.DataFrame, pd.DataFrame, dict], Figure]] = {
    "mount_pitch_per_fly": _fig_pitch_per_fly, "mount_pitch_vs_spontaneous_per": _fig_spontaneous,
    "mount_pitch_vs_held_out": _fig_held_out, "response_by_mount_pitch_tercile": _fig_tercile,
    "baseline_state_vs_response": _fig_baseline, "pullins_held_vs_retracted": _fig_pullins}


def iter_figures(fly_trial: pd.DataFrame, fly: pd.DataFrame, stats: dict) -> Iterator[Tuple[str, Figure]]:
    """The figures one at a time, built under the lab style (which stays active while the caller holds the figure).

    A figure that raises is logged (with its traceback) and left out, so one bad figure never costs the rest nor the report.
    """
    trials = _consistent(fly_trial)
    with _style():
        for name, build in BUILDERS.items():
            try:
                fig = _placeholder(name, stats) if name in stats["placeholders"] else build(trials, fly, stats)
            except Exception as err:                       # noqa: BLE001  (see the docstring)
                LOGGER.warning("[MOUNTING] figure %s failed: %s: %s", name, type(err).__name__, err, exc_info=True)
                continue
            yield name, fig


def build_figures(fly_trial: pd.DataFrame, fly: pd.DataFrame, stats: dict) -> Dict[str, Figure]:
    """All six figures as matplotlib ``Figure`` objects (nothing is written)."""
    return dict(iter_figures(fly_trial, fly, stats))


def _remove_pictures(out: Path) -> None:
    """Delete the six figures of an earlier run from *out* (those names only; nothing else, and nothing is created)."""
    for name in NAMES:
        (out / f"{name}.png").unlink(missing_ok=True)


def render_figures(fly_trial: pd.DataFrame, fly: pd.DataFrame, out_dir, *, n_perm: int, seed: int = 0,
                   stats: Optional[dict] = None) -> Dict[str, Path]:
    """Write the six PNGs (300 dpi) into *out_dir*; returns ``{figure name: path}`` of those written.

    The six PNGs of an earlier run are removed first, so a figure that fails to build leaves no stale picture behind for
    the report to point at: after the call ``<name>.png`` exists exactly for the names returned. With no fly-trials
    nothing is drawn or created and those earlier pictures are removed too, so the folder never holds a figure the report
    (which then says nothing is in scope) does not describe. *stats* is ``compute_stats``' result when the caller has it
    already (the report takes the same one); it is computed here otherwise.
    """
    out = Path(out_dir).expanduser()
    if len(fly_trial) == 0:
        _remove_pictures(out)
        return {}
    stats = stats if stats is not None else compute_stats(fly_trial, fly, n_perm=n_perm, seed=seed)
    out.mkdir(parents=True, exist_ok=True)
    _remove_pictures(out)
    written: Dict[str, Path] = {}
    for name, fig in iter_figures(fly_trial, fly, stats):
        path = out / f"{name}.png"
        fig.savefig(path, dpi=DPI, facecolor=SURFACE)
        written[name] = path
    return written


# ---------------------------------------------------------------------------
# report.md
# ---------------------------------------------------------------------------

EXCLUSION_LABELS = {
    "n_batches": ("Batches walked", "live batches on or after the cut-off date"),
    "n_trials": ("Trials with a geometry table", "tracked trials read"),
    "n_fly_trials": ("Fly-trials in the table", "rows of `fly_trial_posture.csv`"),
    "eye_mismatch": ("Eye mismatch",
                     "geometry eye farther than `eye_match_px` from the production anchor: fly-trial dropped"),
    "non_square_frame": ("Non-square frames", "row kept; every eye-unit metric left empty"),
    "warmup_slots": ("Warm-up slots", "slot numbered by warm-up order, not production's fly N: fly-trial dropped"),
    "no_production_table": ("No production table", "no `<trial>_fly{N}_distances` table found: fly-trial dropped"),
    "unreadable_geometry": ("Unreadable geometry", "geometry table or meta that could not be read: dropped"),
    "flagged_flies": ("Flagged flies", "removed before anything was computed"),
    "flagged_fly_trials": ("Flagged fly-trials", "trials of the flagged flies"),
    "identity_drift": ("Identity drift", "production anchor wanders between trials: trial rows kept, fly left out of "
                                         "every per-fly statistic (flies listed below)"),
    "prediction_exact_duplicates": ("Exact duplicate score rows", "kept once"),
    "prediction_conflicts": ("Conflicting scores", "one trial key whose rows disagree on the score: no score"),
    "prediction_same_score_duplicates": ("Same-score duplicate rows", "collapsed to one"),
    "prediction_key_conflicts": ("Conflicting scores across labels", "two labels of one trial disagree: no score"),
    "predictions_csv": ("Scores file", "`model_predictions.csv` the scores were read from"),
    "predictions_missing": ("Scores file missing", "yes = the scores are empty"),
    "pending_trials": ("Pending trials", "trial videos of the live batches the abdomen stage has not tracked yet: "
                                         "absent from these tables"),
    "pending_trials_error": ("Pending-trial count failed", "why the walk of the video folders failed"),
}


def _cell(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "n/a"
    if isinstance(value, (list, tuple)):
        return str(len(value))
    if isinstance(value, float):
        return f"{value:g}"
    return str(value).replace("|", "\\|")


def _exclusions_section(exclusions: Mapping) -> str:
    lines = ["## Exclusions", "",
             "What was left out of the tables and why. Counts are fly-trials unless said otherwise.", "",
             "| What | Count | Meaning |", "|---|---|---|"]
    for key, value in exclusions.items():
        label, note = EXCLUSION_LABELS.get(key, (key.replace("_", " ").capitalize(), "not described by this report"))
        if key == "pending_trials" and value is None:
            note = "could not be counted (the error is in the row below)"
        lines.append(f"| {label} (`{key}`) | {_cell(value)} | {note} |")
    drift = exclusions.get("identity_drift") or []
    if drift:
        lines += ["", f"Flies with identity drift (`identity_drift`), {len(drift)}:", ""]
        lines += [f"- `{d.get('fly_id', '?')}`: anchors spread {d.get('spread_px', '?')} px over "
                  f"{d.get('n_trials', '?')} trial(s)" for d in drift]
    return "\n".join(lines)


def _header(fly_trial: pd.DataFrame, stats: Mapping) -> str:
    lines = ["# Mounting posture report", "",
             "**All p-values in this report are uncorrected** for multiple comparisons: read them as screening, not as "
             "confirmed effects.", ""]
    if len(fly_trial) == 0:
        lines += ["Nothing to report: no new-glue fly-trials in scope (see the exclusions below for what was left out)."]
        return "\n".join(lines)
    lines += [f"New-glue flies only: {_count(stats['n_fly_trials'], 'fly-trial', 'fly-trials')} in the table, "
              f"{stats['n_trials']} of them from identity-consistent flies; "
              f"{_count(stats['n_flies'], 'identity-consistent fly', 'identity-consistent flies')}.", "",
              "How to read the statistics:", "",
              "- The fly is the unit of inference. Per-fly figures use the fly table; pooled-trial bars use the trials of "
              "identity-consistent flies and divide pooled counts (never a mean of per-fly means).",
              "- Mount pitch is the body pitch in degrees, `+` = abdomen below the eye (head up). Camera roll differs per "
              "rig, so pitch is compared within rig: the scatters and terciles use the rig-centred pitch (fly value minus "
              "the median of its rig).",
              f"- Spearman ρ / p: per fly, asymptotic p. Stratified permutation p: the pitch is shuffled among flies of "
              f"the same dataset × rig, two-sided on |ρ|, {stats['n_perm']} permutations, seed {stats['seed']}.",
              "- The response analyses (tercile bars, baseline-state bars and their tests) use only scored trials: "
              "pretest + testing trials that have a score (training trials get none).",
              f"- Within-fly permutation p: the state label is shuffled among the trials of each fly of one trial type "
              f"(pretest or testing), so a pretest-vs-testing difference cannot create an association; only fly × "
              f"trial-type groups seen in both states count; {stats['n_perm']} permutations, seed {stats['seed']}.",
              "- The difference shown for each baseline-state comparison is the Mantel–Haenszel-weighted risk difference "
              "over those same groups, Σ w (p1 − p0) / Σ w with w = n1·n0 / (n1 + n0), where p1 / p0 are the response "
              "rates in the state / not in it: a pretest-vs-testing shift cannot inflate it. The bars are pooled rates "
              "and keep that shift.",
              f"- \"Yes\" in a figure title means the permutation p is below {stats['alpha']} (uncorrected) with at least "
              f"{MIN_FLIES} flies to compare; fewer is \"Not enough data to tell\"."]
    return "\n".join(lines)


def _scatter_lines(name: str, question: str, s: Mapping, what: str) -> List[str]:
    return [f"### {question}", "",
            f"Figure `{name}.png`: {what}.", "",
            f"- Answer: **{s['verdict']}**",
            f"- {n_flies_label(s['n_flies'])}; Spearman {_rho_text(s['rho'], 3)}, {p_text(s['p'], 4)} (uncorrected)",
            f"- Stratified permutation {p_text(s['p_perm'], 4)} (uncorrected): pitch shuffled within dataset × rig strata, "
            f"{_count(s['n_strata'], 'stratum', 'strata')} ({s['n_strata_shuffled']} with at least 2 flies)"]


def _placeholder_line(stats: Mapping, name: str) -> List[str]:
    info = stats["placeholders"].get(name)
    return [f"- **Placeholder, not drawn:** not enough data — {info['reason']}."] if info else []


def _mounting_section(stats: Mapping, missing: Sequence[str] = ()) -> str:
    q = TITLES
    L = ["## Mounting", ""]
    if missing:
        L += ["**Figures missing from this folder:** " + ", ".join(f"`{n}.png`" for n in missing) +
              " (a figure that fails to build is logged and skipped; see the run log).", ""]
    L += [f"### {q['mount_pitch_per_fly']}", "",
          "Figure `mount_pitch_per_fly.png`: the mounting angle of every fly, by rig.", "",
          f"- {stats['rig_answer']}", *_placeholder_line(stats, "mount_pitch_per_fly"), "",
          "| Rig | Flies | Median (°) | Q1 – Q3 (°) |", "|---|---|---|---|"]
    L += [f"| {r['rig']} | {n_flies_label(r['n_flies'])} | {r['median_deg']:+.1f} | "
          f"{r['q1_deg']:+.1f} to {r['q3_deg']:+.1f} |" for r in stats["rigs"]]
    L += [""] + _scatter_lines("mount_pitch_vs_spontaneous_per", q["mount_pitch_vs_spontaneous_per"], stats["spontaneous"],
                               "per fly, rig-centred pitch vs the % of its pre-stimulus windows with a spontaneous PER")
    L += _placeholder_line(stats, "mount_pitch_vs_spontaneous_per") + [""]
    L += _scatter_lines("mount_pitch_vs_held_out", q["mount_pitch_vs_held_out"], stats["held_out"],
                        "per fly, rig-centred pitch vs the fraction of its trials with the proboscis held out "
                        "(marker fill = the fly's dominant held sub-type)")
    L += _placeholder_line(stats, "mount_pitch_vs_held_out") + [""]
    t = stats["tercile"]
    L += _scatter_lines("response_by_mount_pitch_tercile", q["response_by_mount_pitch_tercile"], t,
                        "pooled-trial response rate (pretest + testing trials) in each third of the rig-centred pitch; the "
                        "p-values are the per-fly test of response rate vs pitch")
    L += _placeholder_line(stats, "response_by_mount_pitch_tercile")
    L += ["", "| Third | Pitch range (° from the rig median) | Trials / flies | Response rate |", "|---|---|---|---|"]
    L += [f"| {n} | {b['lo_deg']:+.1f} to {b['hi_deg']:+.1f} | {_group_n(b)} | {100 * b['rate']:.1f}% |"
          for b, (_, n, _) in zip(t["bins"], THIRDS)]
    L += ["", f"### {q['baseline_state_vs_response']}", "",
          "Figure `baseline_state_vs_response.png`: pooled response rate of the pretest + testing trials by the state "
          "before the stimulus.", ""]
    L += _placeholder_line(stats, "baseline_state_vs_response")
    for key, title, what in (("proboscis", "Proboscis state", "held out (any sub-type) minus retracted"),
                             ("spontaneous", "Spontaneous PER", "spontaneous PER minus none")):
        panel = stats["baseline"][key]
        if panel["n_groups_both"]:
            groups = _count(panel["n_groups_both"], "fly × trial-type group", "fly × trial-type groups")
            used = (f"{_points(panel['diff'], 1)} (Mantel–Haenszel over {groups} of "
                    f"{_count(panel['n_flies_both'], 'fly', 'flies')} seen in both states, "
                    f"{_count(panel['n_trials_both'], 'trial', 'trials')}); the pooled difference of the bars is "
                    f"{_points(panel['diff_pooled'], 1)}")
            if panel["n_flies_both"] < MIN_FLIES:
                used += f"; fewer than {MIN_FLIES} flies, so no answer is drawn"
        else:
            used = "no fly was seen in both states, so there is nothing to compare"
        L += [f"**{title}** — {panel['verdict']}", "",
              f"- {what[0].upper() + what[1:]}, within fly × trial type: {used}",
              f"- Within-fly permutation {p_text(panel['p_perm'], 4)} (uncorrected), shuffled inside each fly × trial-type "
              f"group; {stats['n_perm']} permutations, seed {stats['seed']}", "",
              "| State | Trials / flies | Response rate |", "|---|---|---|"]
        L += [f"| {STATE_LABELS[b['state']]} | {_group_n(b)} | {100 * b['rate']:.1f}% |"
              for b in panel["bars"] if b["n_trials"]]
        L += [""]
    g = stats["pullins"]["groups"]
    L += [f"### {q['pullins_held_vs_retracted']}", "",
          "Figure `pullins_held_vs_retracted.png`: pull-in events per minute (pooled events over pooled minutes, every "
          "trial type with a stimulus window), before vs during the stimulus. Descriptive: no test.", "",
          f"- {stats['pullins']['answer']}",
          *_placeholder_line(stats, "pullins_held_vs_retracted"),
          "", "| Group | Trials / flies | Pull-ins per min before | Pull-ins per min during |", "|---|---|---|---|"]
    L += [f"| {label} | {_group_n(g[k])} | {_num(g[k]['rate_pre'])} | {_num(g[k]['rate_stim'])} |"
          for label, k in (("Retracted", "retracted"), ("Held out", "held"))]
    return "\n".join(L)


def report_sections(fly_trial: pd.DataFrame, fly: pd.DataFrame, exclusions: Mapping, stats: Mapping,
                    *, figures_missing: Sequence[str] = ()) -> List[str]:
    """The report as a list of markdown sections: header, exclusions and (with data) the mounting section."""
    sections = [_header(fly_trial, stats), _exclusions_section(exclusions)]
    if len(fly_trial):
        sections.append(_mounting_section(stats, figures_missing))
    return sections


def write_report(fly_trial: pd.DataFrame, fly: pd.DataFrame, exclusions: Mapping, stats: Mapping, out_dir,
                 *, figures: Optional[Mapping[str, Path]] = None, extra_sections: Sequence[str] = ()) -> Path:
    """Write ``<out_dir>/report.md``: every exclusions key, rho / p of every per-fly scatter, every permutation p.

    *figures* is what ``render_figures`` returned: a figure that is not in it (it failed to build) is named in the mounting
    section, so the report never points at a picture that is not there. Without *figures* the report claims nothing
    about the pictures (the folder is not probed: a PNG found there may be left over from an earlier run).
    *extra_sections* (markdown strings, e.g. ``"## Abdomen movement ..."``) are appended after the mounting section.
    """
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    missing = [n for n in NAMES if n not in figures] if figures is not None and len(fly_trial) else []
    sections = report_sections(fly_trial, fly, exclusions, stats, figures_missing=missing) + list(extra_sections)
    path = out / REPORT_NAME
    path.write_text("\n\n".join(s.strip("\n") for s in sections) + "\n", encoding="utf-8")
    return path
