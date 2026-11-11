"""Per-fly PER traces: the naive pre-test and the post-training test, overlaid.

The ``*-Sensitivity-*`` (and Hex test) protocol presents the same odor panel to
the same fly twice — ``pretest_*`` before training, ``testing_*`` after — so the
most direct look at what training did to one fly is the two traces for an odor
drawn on the same axes:

    pre-test (naive)  -> orange
    post-training     -> blue

A ``*-Sensitivity-*`` cohort's exposure block pairs its odor with nothing, so
its panels read pre-/post-exposure and that odor "sensitized odor", not
CS+; the light-conditioned Hex test keeps the labels above.

One figure per fly, one row per odor. Rows run in alphanumeric (natural) order
of the odor name and are the cohort's odor UNION, so row k is the same odor on
every fly of a cohort, even a fly that missed a presentation (its row is left
empty and says so). The figure is drawn in the style of the raw per-fly trace
figures (Raw-*-PER-Traces) it sits beside, without their threshold line.

Odor order is randomised between the two panels, so pairing is on the ODOR
parsed from the trial label, never on the trial index. The rig writes that odor
as ``<name>-<concentration>`` in several spellings (``3-octonol-0.1``,
``3octanol-0.1``); both halves are normalised to one display name plus the
concentration (``3-Octanol 0.1``) before they are matched. A sensitivity
cohort's rows are labelled with the concentration as ``3-Octanol 0.1% (v/v)``.

Frozen experiment folders are excluded, like every other figure. The light-only
control trial (``testing_9_lightonly``) is not an odor and gets no row.

Usage
-----
    python3 scripts/analysis/pretest_vs_test_traces.py \
        --pretest-wide-csv .../all_envelope_rows_wide_combined_base_pretest.parquet \
        --testing-wide-csv .../all_envelope_rows_wide_combined_base.parquet \
        --out-dir .../Results/New-Opto-Fly-Figures/Pre-vs-Post-PER-Traces
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fbpipe.odor_constants import ODOR_CANON  # noqa: E402
from fbpipe.utils.figure_cache import FigureCache, code_digest  # noqa: E402
from fbpipe.utils.frozen_folders import drop_frozen  # noqa: E402
from fbpipe.utils.tables import read_table  # noqa: E402
from scripts.analysis import envelope_visuals as ev  # noqa: E402
from scripts.analysis import odor_conc_labels  # noqa: E402
from scripts.analysis import phase_terms  # noqa: E402

# The user's convention for this figure: before training orange, after blue.
PRE_COLOR = "tab:orange"
POST_COLOR = "tab:blue"
PRE_LABEL = "Pre-test (naive)"
POST_LABEL = "Post-training"

Y_LABEL = "Max Distance x Angle %"
INK_MUTED = "#6b6b66"

# The raw per-fly trace figures as the nightly draws them: EnvelopePlotConfig's
# defaults under run_workflows._PIPELINE_ENVELOPE_STYLE_DEFAULTS. This figure
# copies their look, so it is sized from the same scales (only the style fields
# are read; tests pin them to the pipeline's).
_RAW_STYLE = ev.EnvelopePlotConfig(
    matrix_npy=Path(), codes_json=Path(), out_dir=Path(), latency_sec=0.0,
    style_scale=2.0, trace_linewidth_scale=2.0, panel_title_scale=0.9,
    figure_title_scale=0.72, figure_subtitle_scale=0.5, legend_scale=0.675,
    plot_size_scale=1.2, fixed_y_max=105.0, legend_anchor_y=0.895,
    figure_title_y=0.925, figure_subtitle_y=0.905, panel_title_x=0.01,
    tight_h_pad=0.12,
)

_CONC_RE = re.compile(r"^(?P<name>.+?)-(?P<conc>\d+(?:\.\d+)?)$")
_LIGHT_ONLY = "Optogenetic Light Control"

# Rig spellings that _display_label_ci does not know with their punctuation
# stripped ("3octanol"); keyed on the alphanumeric core of every known name.
_COMPACT_LABELS: dict[str, str] = {}
for _key, _label in ev._DISPLAY_LABEL_LOWER.items():
    _COMPACT_LABELS.setdefault(re.sub(r"[^a-z0-9]", "", _key), _label)
for _key in ODOR_CANON:
    _resolved = ev._display_label_ci(str(_key))
    if _resolved != str(_key):
        _COMPACT_LABELS.setdefault(re.sub(r"[^a-z0-9]", "", str(_key).lower()), _resolved)
_COMPACT_LABELS.setdefault("3octanol", ev._display_label_ci("3-octanol"))


# ---------------------------------------------------------------------------
# Odor names and order
# ---------------------------------------------------------------------------


def _base_display(dataset_canon: str, token: str) -> str:
    label = ev._display_label_ci(token)
    if label == token:
        label = _COMPACT_LABELS.get(re.sub(r"[^a-z0-9]", "", token.lower()), token)
    return ev.apply_dataset_odor_remap(dataset_canon, label)


def odor_parts(dataset_canon: str, trial_label: str) -> tuple[str, str] | None:
    """``(display name, concentration)`` for an odor trial, else None."""
    token = ev._extract_odor_from_label(trial_label)
    if token == str(trial_label):
        return None  # no odor token in the label
    match = _CONC_RE.match(token)
    name, conc = (match["name"], match["conc"]) if match else (token, "")
    display = _base_display(dataset_canon, name)
    if display == _LIGHT_ONLY:
        return None
    return display, conc


def odor_name(dataset_canon: str, trial_label: str) -> str | None:
    """The row label for a trial: ``"Hexanol 0.1"``, or None for a non-odor trial."""
    parts = odor_parts(dataset_canon, trial_label)
    if parts is None:
        return None
    name, conc = parts
    return f"{name} {conc}" if conc else name


def _natural_key(text: str) -> tuple:
    out = []
    for i, chunk in enumerate(re.split(r"(\d+(?:\.\d+)?)", str(text).casefold())):
        out.append((0, float(chunk), "") if i % 2 else (1, 0.0, chunk))
    return tuple(out)


def order_odors(names: Iterable[str]) -> list[str]:
    """Alphanumeric, case-insensitive, numbers compared as numbers."""
    return sorted(set(names), key=_natural_key)


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


@dataclass
class Trace:
    env: np.ndarray
    fps: float
    odor_on_s: float | None
    odor_off_s: float | None
    trial_label: str
    score: int | None = None


@dataclass
class FlyPair:
    dataset: str
    fly: str
    fly_number: str
    odors: list[str]
    pre: dict[str, Trace] = field(default_factory=dict)
    post: dict[str, Trace] = field(default_factory=dict)
    trained_odor: str = ""
    # Each row's concentration as its name carries it ("Hexanol 0.1" -> "0.1",
    # "" for none), so the label can write it as % (v/v) without re-parsing.
    concentrations: dict[str, str] = field(default_factory=dict)


def _prepare(df: pd.DataFrame, phase: str) -> tuple[pd.DataFrame, list[str]]:
    df = drop_frozen(df.copy())
    if "trial_type" in df.columns:
        df = df.loc[df["trial_type"].astype(str).str.strip().str.lower() == phase]
    df = ev._normalise_fly_columns(df.copy())
    df["dataset_canon"] = df["dataset"].map(ev._canon_dataset)
    parts = [odor_parts(ds, label) for ds, label in zip(df["dataset_canon"], df["trial_label"])]
    df["odor_base"] = [pt[0] if pt else None for pt in parts]
    df["odor_conc"] = [pt[1] if pt else "" for pt in parts]
    df = df.loc[df["odor_base"].notna()].copy()
    env_cols = sorted(
        (c for c in df.columns if str(c).startswith("dir_val_")),
        key=lambda c: int(str(c).rsplit("_", 1)[-1]),
    )
    return df, env_cols


ScoreKey = tuple[str, str, str, str]


def load_scores(predictions_csv: Path | str) -> dict[ScoreKey, int]:
    """Model scores keyed ``(canonical dataset, fly, fly_number, trial_label)``.

    Same source as the "Score: N" on the Raw-*-PER-Traces figures; pretest and
    testing rows both live in it.
    """
    df = pd.read_csv(
        predictions_csv, usecols=["dataset", "fly", "fly_number", "trial_label", "score"]
    )
    df = ev._normalise_fly_columns(df)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.loc[df["score"].notna()]
    return {
        (ev._canon_dataset(str(ds)), str(fly), str(num), str(label)): int(score)
        for ds, fly, num, label, score in df[
            ["dataset", "fly", "fly_number", "trial_label", "score"]
        ].itertuples(index=False, name=None)
    }


def _fold_bare_concentrations(pre: pd.DataFrame, post: pd.DataFrame) -> None:
    """Set ``odor``, folding a bare name into the cohort's ONLY concentration.

    Batches recorded before ``~/odor_pins.txt`` (~2026-10-06) wrote
    ``hexanol``; later ones write ``hexanol-0.1`` for the same panel. Left
    alone, one cohort splits every odor into two half-empty rows. When a
    cohort carries exactly one labelled concentration of an odor the bare name
    is that concentration; with several (the Hex test's EB 0.1/1/10) it is
    ambiguous and keeps its own row.
    """
    both = pd.concat([pre[["dataset_canon", "odor_base", "odor_conc"]],
                      post[["dataset_canon", "odor_base", "odor_conc"]]])
    labelled = both.loc[both["odor_conc"] != ""].drop_duplicates()
    sole = labelled.groupby(["dataset_canon", "odor_base"])["odor_conc"].agg(
        lambda c: c.iloc[0] if len(c) == 1 else ""
    ).to_dict()
    for df in (pre, post):
        conc = [
            c or sole.get((ds, base), "")
            for ds, base, c in zip(df["dataset_canon"], df["odor_base"], df["odor_conc"])
        ]
        df["odor"] = [f"{b} {c}" if c else b for b, c in zip(df["odor_base"], conc)]


def _dedupe(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    """One presentation per (fly, odor) per phase — the protocol never repeats one.

    A duplicate (an aborted-and-restarted session) keeps the earliest trial,
    matching pretest_vs_test_comparison. Run AFTER pairing so the count only
    reports flies that are actually drawn.
    """
    df = df.assign(_trial_num=df["trial_label"].map(ev._trial_num))
    df = df.sort_values("_trial_num", kind="stable")
    key = ["dataset_canon", "fly", "fly_number", "odor"]
    dupes = int(df.duplicated(subset=key).sum())
    if dupes:
        print(f"[pre-vs-post-traces] {phase}: {dupes} duplicate (fly, odor) row(s) dropped")
    return df.drop_duplicates(subset=key, keep="first")


def _to_trace(
    row: pd.Series, env_cols: list[str], scores: dict[ScoreKey, int]
) -> Trace | None:
    env = ev._extract_env_row(
        row[env_cols].to_numpy(dtype=float), trace_len=row.get("trace_len")
    )
    if env.size == 0:
        return None
    fps = pd.to_numeric(row.get("fps"), errors="coerce")
    on = pd.to_numeric(row.get("trial_odor_on_s"), errors="coerce")
    off = pd.to_numeric(row.get("trial_odor_off_s"), errors="coerce")
    window_ok = math.isfinite(on) and math.isfinite(off) and off > on > 0
    return Trace(
        env=env,
        fps=float(fps) if math.isfinite(fps) and fps > 0 else float("nan"),
        odor_on_s=float(on) if window_ok else None,
        odor_off_s=float(off) if window_ok else None,
        trial_label=str(row["trial_label"]),
        score=scores.get((
            str(row["dataset_canon"]), str(row["fly"]),
            str(row["fly_number"]), str(row["trial_label"]),
        )),
    )


def pair_flies(
    pretest: pd.DataFrame,
    testing: pd.DataFrame,
    *,
    cohorts: Sequence[str] | None = None,
    scores: dict[ScoreKey, int] | None = None,
) -> list[FlyPair]:
    """Every fly with BOTH a pre-test and a post-training panel.

    ``scores`` (from :func:`load_scores`) attaches each trial's model score;
    without it no score is drawn.
    """
    scores = scores or {}
    pre, pre_cols = _prepare(pretest, "pretest")
    post, post_cols = _prepare(testing, "testing")
    if cohorts:
        wanted = {ev._canon_dataset(c) for c in cohorts}
        pre = pre.loc[pre["dataset_canon"].isin(wanted)]
        post = post.loc[post["dataset_canon"].isin(wanted)]

    pre, post = pre.copy(), post.copy()
    _fold_bare_concentrations(pre, post)

    fly_key = ["dataset_canon", "fly", "fly_number"]
    both = pre[fly_key].drop_duplicates().merge(post[fly_key].drop_duplicates())
    if both.empty:
        return []
    pre = _dedupe(pre.merge(both, on=fly_key), "pretest")
    post = _dedupe(post.merge(both, on=fly_key), "testing")

    cohort_odors = {
        ds: order_odors(pd.concat([pre.loc[pre["dataset_canon"] == ds, "odor"],
                                   post.loc[post["dataset_canon"] == ds, "odor"]]))
        for ds in both["dataset_canon"].unique()
    }
    # A row's name is "<odor_base> <concentration>" (_fold_bare_concentrations).
    concentrations = {
        odor: odor[len(base):].strip()
        for frame in (pre, post)
        for base, odor in zip(frame["odor_base"], frame["odor"])
    }

    pre_groups = dict(tuple(pre.groupby(fly_key, sort=False)))
    post_groups = dict(tuple(post.groupby(fly_key, sort=False)))
    flies: list[FlyPair] = []
    for ds, fly, num in sorted(
        both.itertuples(index=False, name=None),
        key=lambda k: (k[0], *ev._fly_sort_key(k[1], k[2])),
    ):
        pair = FlyPair(
            dataset=ds, fly=fly, fly_number=num, odors=cohort_odors[ds],
            trained_odor=str(ev._trained_label(ds)),
            concentrations={odor: concentrations[odor] for odor in cohort_odors[ds]},
        )
        for target, groups, cols in (
            (pair.pre, pre_groups, pre_cols),
            (pair.post, post_groups, post_cols),
        ):
            for _, row in groups[(ds, fly, num)].iterrows():
                trace = _to_trace(row, cols, scores)
                if trace is not None:
                    target[row["odor"]] = trace
        flies.append(pair)
    return flies


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


@dataclass
class TraceStyle:
    """Fallbacks only: the time axis is always the recording itself.

    ``odor_on_s``/``odor_off_s`` shade the window only for a trial whose rig
    sidecar carried no odor timing; ``fps_default`` covers a missing fps.
    """

    fps_default: float = 40.0
    odor_on_s: float = 30.0
    odor_off_s: float = 60.0
    # The raw figures' 0-100 % axis with headroom; the odor label hangs from it.
    y_max: float = _RAW_STYLE.fixed_y_max


def _duration_s(trace: Trace, style: TraceStyle) -> float:
    """How long the video is: its frame count over its frame rate."""
    fps = trace.fps if math.isfinite(trace.fps) else style.fps_default
    return trace.env.size / max(fps, 1e-9)


def _is_trained_row(pair: FlyPair, odor: str) -> bool:
    trace = pair.pre.get(odor) or pair.post.get(odor)
    if trace is None:
        return False
    parts = odor_parts(pair.dataset, trace.trial_label)
    return bool(parts) and parts[0].strip().lower() == pair.trained_odor.strip().lower()


def _row_label(pair: FlyPair, odor: str) -> tuple[str, str | None]:
    """A row's drawn odor label, and its two-line form when it carries a tag.

    A sensitivity cohort writes the concentration as ``0.1% (v/v)`` (none when
    its batches recorded none); any other cohort keeps the row name as it is.
    """
    sensitivity = phase_terms.is_sensitivity(pair.dataset)
    name = odor
    if sensitivity:
        conc = pair.concentrations.get(odor, "")
        if conc and odor.endswith(f" {conc}"):
            name = odor_conc_labels.with_conc(odor[: -len(conc) - 1], conc)
        else:
            name = odor_conc_labels.odor_conc_label(odor)
    if not _is_trained_row(pair, odor):
        return name, None
    tag = phase_terms.odor_tag(pair.dataset, "trained, CS+")
    # After "% (v/v)" the tag is one more parenthetical; others keep two spaces.
    sep = " " if sensitivity else "  "
    return f"{name}{sep}({tag})", f"{name}\n({tag})"


def _right_aligned_run(ax, items, *, x: float, y: float, gap_pt: float) -> list:
    """Draw ``(text, style)`` items right to left from ``x``, each left of the last.

    Each is anchored on the previous one's box, so texts of any width never overlap.
    """
    drawn: list = []
    for text, kw in items:
        if not drawn:
            artist = ax.text(x, y, text, transform=ax.get_yaxis_transform(),
                             ha="right", va="top", clip_on=False, **kw)
        else:
            artist = ax.annotate(text, xy=(0, 1), xycoords=drawn[-1],
                                 xytext=(-gap_pt, 0), textcoords="offset points",
                                 ha="right", va="top", **kw)
        drawn.append(artist)
    return drawn


def _clear_of(label, right: Sequence, renderer, gap_px: float) -> bool:
    """True when ``label`` ends at least ``gap_px`` left of every text in ``right``."""
    end = label.get_window_extent(renderer).x1 + gap_px
    return all(end <= text.get_window_extent(renderer).x0 for text in right)


def build_figure(pair: FlyPair, style: TraceStyle):
    """One row per cohort odor, pre (orange) and post (blue) overlaid. Unsaved.

    Drawn like the raw per-fly trace figures (``_RAW_STYLE`` through
    generate_envelope_plots' size formulas), without their threshold line.
    Rows share the time axis, which runs to the end of the fly's longest
    video, so only the bottom row carries time labels.
    """
    raw = _RAW_STYLE
    ui = ev._bounded_scale(raw.style_scale)
    tick_width = 0.8 * ui
    tick_length = 3.5 * ui
    base_font = 10 * ui
    xlabel_font = 11 * ui
    panel_title_font = 11 * ui * ev._bounded_scale(raw.panel_title_scale)
    legend_font = 9 * ui * ev._bounded_scale(raw.legend_scale)
    figure_title_font = 14 * ui * ev._bounded_scale(raw.figure_title_scale)
    figure_subtitle_font = 12 * ui * ev._bounded_scale(raw.figure_subtitle_scale)
    trace_lw = 1.2 * ev._bounded_scale(raw.trace_linewidth_scale)
    plot_size_scale = ev._bounded_scale(raw.plot_size_scale)
    rc = {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8 * ui,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": base_font,
        "xtick.major.width": tick_width,
        "ytick.major.width": tick_width,
        "xtick.minor.width": tick_width,
        "ytick.minor.width": tick_width,
        "xtick.major.size": tick_length,
        "ytick.major.size": tick_length,
        "xtick.minor.size": tick_length * 0.7,
        "ytick.minor.size": tick_length * 0.7,
    }

    show_scores = any(
        t.score is not None for t in [*pair.pre.values(), *pair.post.values()]
    )
    fly_end = max(
        (_duration_s(t, style) for t in [*pair.pre.values(), *pair.post.values()]),
        default=style.odor_off_s,
    )
    n_rows = max(len(pair.odors), 1)
    # A *-Sensitivity-* cohort pairs its exposed odor with nothing, so its panels
    # are pre-/post-exposure; any other cohort (the Hex test) keeps its words.
    pre_legend = phase_terms.pre_label(pair.dataset, PRE_LABEL)
    post_legend = phase_terms.post_label(pair.dataset, POST_LABEL)
    pre_name = phase_terms.pre_label(pair.dataset, "pre-test", case="lower")
    post_name = phase_terms.post_label(pair.dataset, "post-training", case="lower")
    pre_title = phase_terms.pre_label(pair.dataset, "Pre-Test", case="title")
    post_title = phase_terms.post_label(pair.dataset, "Post-Training", case="title")
    score_kw = {"fontsize": panel_title_font, "weight": "bold"}
    note_kw = {"fontsize": 0.85 * base_font, "style": "italic", "color": INK_MUTED,
               "gid": "note"}
    gap_pt = panel_title_font

    with plt.rc_context(rc):
        fig_h = max(3.0, n_rows * 1.6 * plot_size_scale + 1.5)
        fig, axes = plt.subplots(
            n_rows, 1, figsize=(10 * plot_size_scale, fig_h), sharex=True
        )
        axes = np.atleast_1d(axes)

        fits = []
        for ax, odor in zip(axes, pair.odors):
            windows = []
            row_traces = [t for t in (pair.pre.get(odor), pair.post.get(odor)) if t]
            row_end = max((_duration_s(t, style) for t in row_traces), default=fly_end)
            for phase, traces, colour, label in (
                ("pre", pair.pre, PRE_COLOR, pre_legend),
                ("post", pair.post, POST_COLOR, post_legend),
            ):
                trace = traces.get(odor)
                if trace is None:
                    continue
                fps = trace.fps if math.isfinite(trace.fps) else style.fps_default
                t = np.arange(trace.env.size, dtype=float) / max(fps, 1e-9)
                ax.plot(t, trace.env, color=colour, linewidth=trace_lw,
                        label=label, zorder=3 if phase == "post" else 2)
                if trace.odor_on_s is not None:
                    windows.append((trace.odor_on_s, trace.odor_off_s))
            if not windows and (odor in pair.pre or odor in pair.post):
                windows.append((style.odor_on_s, style.odor_off_s))
            if windows:
                on = min(w[0] for w in windows)
                off = max(w[1] for w in windows)
                ax.axvspan(on, min(off, row_end), alpha=ev.ODOR_PLUS_LIGHT_ALPHA,
                           color=ev.ODOR_PLUS_LIGHT_COLOR)
            ax.set_ylim(0, style.y_max)
            ax.tick_params(axis="both", which="both", labelsize=base_font,
                           width=tick_width, length=tick_length)

            text, wrapped = _row_label(pair, odor)
            label = ax.text(raw.panel_title_x, style.y_max, text,
                            transform=ax.get_yaxis_transform(), ha="left", va="top",
                            fontsize=panel_title_font, weight="bold", color="black",
                            clip_on=False, gid="odor-label")
            # Right to left: the post slot, then the pre slot (the legend's
            # order). A missing phase's slot says so instead of scoring.
            missing = [name for name, traces in ((pre_name, pair.pre),
                                                 (post_name, pair.post))
                       if odor not in traces]
            note = None
            if missing:
                note = "not presented" if len(missing) == 2 else f"no {missing[0]} trial"
            items = []
            for prefix, traces, colour in (("Post", pair.post, POST_COLOR),
                                           ("Pre", pair.pre, PRE_COLOR)):
                trace = traces.get(odor)
                if trace is None and note:
                    items.append((note, note_kw))
                    note = None
                elif trace is not None and show_scores:
                    value = "n/a" if trace.score is None else str(trace.score)
                    items.append((f"{prefix} score: {value}", {**score_kw, "color": colour}))
            right = _right_aligned_run(ax, items, x=1.0 - raw.panel_title_x,
                                       y=style.y_max, gap_pt=gap_pt)
            fits.append((label, wrapped, right))

        axes[0].set_xlim(0, fly_end)
        axes[0].xaxis.set_major_locator(MultipleLocator(ev.PER_TIME_TICK_STEP_S))
        axes[-1].set_xlabel("Time (s)", fontsize=xlabel_font)
        ylabel = fig.text(0.02, 0.5, Y_LABEL, va="center", rotation="vertical",
                          fontsize=base_font)
        title = fig.suptitle(
            f"Proboscis Distance, {pre_title} vs {post_title}",
            x=raw.figure_title_x, y=raw.figure_title_y, ha=raw.figure_title_ha,
            va="center", fontsize=figure_title_font, fontweight="bold",
        )
        number = "" if pair.fly_number.upper() == "UNKNOWN" else f" - Fly {pair.fly_number}"
        fig.text(raw.figure_subtitle_x, raw.figure_subtitle_y,
                 f"{pair.dataset} · {pair.fly}{number}", ha=raw.figure_subtitle_ha,
                 va="center", fontsize=figure_subtitle_font, weight="bold",
                 gid="subtitle")
        handles = [
            Line2D([0], [0], color=PRE_COLOR, linewidth=trace_lw, label=pre_legend),
            Line2D([0], [0], color=POST_COLOR, linewidth=trace_lw, label=post_legend),
            plt.Rectangle((0, 0), 1, 1, color=ev.ODOR_PLUS_LIGHT_COLOR,
                          alpha=ev.ODOR_PLUS_LIGHT_ALPHA, label="Odor"),
        ]
        legend = fig.legend(handles=handles, loc="upper right",
                            bbox_to_anchor=(raw.legend_anchor_x, raw.legend_anchor_y),
                            frameon=True, fontsize=legend_font)

        # The raw figure's legend hangs over its top row's score; this one
        # hangs from just under the title, and the rows start below it.
        renderer = fig.canvas.get_renderer()
        to_fig = fig.transFigure.inverted()
        pad = 0.5 * legend_font / 72 / fig_h
        title_box = to_fig.transform_bbox(title.get_window_extent(renderer))
        legend.set_bbox_to_anchor((raw.legend_anchor_x, title_box.y0 - pad),
                                  transform=fig.transFigure)
        legend_box = to_fig.transform_bbox(legend.get_window_extent(renderer))
        layout_top = max(0.84, 0.93 - 0.03 * max(ui - 1.0, 0.0))
        fig.tight_layout(rect=[0.04, 0, 1, min(layout_top, legend_box.y0 - pad)],
                         h_pad=raw.tight_h_pad)
        # Centred on the rows, which start lower than the raw figure's.
        ylabel.set_y((axes[0].get_position().y1 + axes[-1].get_position().y0) / 2)

        # A long label must not run into its row's scores: the tag moves to a
        # second line first, then the label shrinks.
        gap_px = gap_pt * fig.dpi / 72
        for label, wrapped, right in fits:
            if wrapped and not _clear_of(label, right, renderer, gap_px):
                label.set_text(wrapped)
            while (not _clear_of(label, right, renderer, gap_px)
                   and label.get_fontsize() > 0.6 * panel_title_font):
                label.set_fontsize(label.get_fontsize() * 0.9)
    return fig


CACHE_FILE = "_figure_cache.json"


def output_path(out_dir: Path, pair: FlyPair) -> Path:
    folder = ev.resolve_dataset_output_dir(Path(out_dir), [pair.dataset])
    suffix = "" if pair.fly_number.upper() == "UNKNOWN" else f"_fly{pair.fly_number}"
    return folder / f"{pair.fly}{suffix}_pre_vs_post_traces.png"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def pair_fingerprint(pair: FlyPair, style: TraceStyle, code: str) -> str:
    """Digest of everything one fly's figure is drawn from."""
    h = hashlib.sha256(repr((code, style, pair.dataset, pair.fly, pair.fly_number, pair.odors,
                             pair.trained_odor, sorted(pair.concentrations.items()))).encode())
    for name, phase in (("pre", pair.pre), ("post", pair.post)):
        for odor in sorted(phase):
            t = phase[odor]
            # With its phase: a trace moved from pre to post is drawn in the other colour.
            h.update(repr((name, odor, t.fps, t.odor_on_s, t.odor_off_s, t.trial_label,
                           t.score)).encode())
            h.update(np.asarray(t.env, dtype=float).tobytes())
    return h.hexdigest()


def render_changed(flies: Sequence[FlyPair], out_dir: Path, style: TraceStyle,
                   code: str) -> tuple[int, int]:
    """Draw only the flies that are new or changed. Returns ``(drawn, kept)``."""
    cache = FigureCache(Path(out_dir) / CACHE_FILE)
    drawn = 0
    try:
        for pair in flies:
            path = output_path(out_dir, pair)
            key = str(path.relative_to(out_dir))
            fingerprint = pair_fingerprint(pair, style, code)
            if path.exists() and cache.is_current(key, fingerprint):
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            fig = build_figure(pair, style)
            # Cropped like the raw trace figures: their layout leaves the top margin.
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            cache.record(key, fingerprint)
            drawn += 1
    finally:
        cache.save()
    return drawn, len(flies) - drawn


def _register_odor_remaps(config_path: Path) -> None:
    """Load ``dataset_overrides.*.odor_remap`` the way the driver does.

    This runs as a subprocess, so without it a remapped cohort would show the
    rig's label here and the corrected one in every other figure.
    """
    from fbpipe.config import load_settings

    settings = load_settings(config_path)
    ev.set_dataset_odor_remap({
        str(name): dict(ov.odor_remap)
        for name, ov in settings.dataset_overrides.items()
        if getattr(ov, "odor_remap", None)
    })


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pretest-wide-csv", required=True, type=Path)
    p.add_argument("--testing-wide-csv", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--predictions-csv", type=Path, default=None,
                   help="model_predictions.csv; draws each trial's model score.")
    p.add_argument("--config", type=Path, default=None,
                   help="Pipeline config, for per-dataset odor_remap display names.")
    p.add_argument("--cohorts", nargs="*", default=None,
                   help="Only these datasets (default: every fly with both phases).")
    p.add_argument("--fps-default", type=float, default=TraceStyle.fps_default)
    p.add_argument("--odor-on-s", type=float, default=TraceStyle.odor_on_s)
    p.add_argument("--odor-off-s", type=float, default=TraceStyle.odor_off_s)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    style = TraceStyle(
        fps_default=args.fps_default, odor_on_s=args.odor_on_s,
        odor_off_s=args.odor_off_s,
    )
    if args.config is not None:
        _register_odor_remaps(args.config)
    scores: dict[ScoreKey, int] = {}
    if args.predictions_csv is not None:
        if Path(args.predictions_csv).exists():
            scores = load_scores(args.predictions_csv)
        else:
            print(f"[pre-vs-post-traces] no predictions CSV at {args.predictions_csv}; "
                  "scores omitted")
    pretest = read_table(args.pretest_wide_csv)
    testing = read_table(args.testing_wide_csv)
    flies = pair_flies(pretest, testing, cohorts=args.cohorts, scores=scores)
    print(f"[pre-vs-post-traces] {len(flies)} fly(ies) with both phases")
    drawn, kept = render_changed(flies, args.out_dir, style, code_digest([Path(__file__)]))
    print(f"[pre-vs-post-traces] drew {drawn} new or changed figure(s), kept {kept} "
          f"unchanged, under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
