"""Methods figure: what each PER response score (-1..5) looks like.

One curated example trace per score value, drawn as the raw ``combined_pct``
signal ("PER %") -- or, in ``--baseline`` mode, as PER minus each trace's
pre-odor resting level. Two-column split: Reaction (score >= 2) on the left,
No-reaction (score <= 1) on the right.

Odor identities are deliberately omitted; the shaded window is simply "odor on".

Scores/colors mirror ``score_summary.SCORE_COLORS`` (the CVD-validated PRGn
purple->green ramp; pinned by ``tests/test_score_matrix.py``).

Run:
    python scripts/analysis/score_scale_figure.py
Outputs (both modes):
    figures/score_scale_PER_traces.png / .pdf              (raw PER %)
    figures/score_scale_PER_traces_baseline.png / .pdf     (baseline-subtracted)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

# --------------------------------------------------------------------------- #
# Data locations
# --------------------------------------------------------------------------- #
TRACE_ROOT = Path("/home/ramanlab/Documents/cole/Data/flys_New")
WIDE_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/"
    "all_envelope_rows_wide_combined_base.csv"
)
OUT_DIR = Path(__file__).resolve().parents[2] / "figures"

# --------------------------------------------------------------------------- #
# Fixed, CVD-validated score palette (mirror of score_summary.SCORE_COLORS).
# Do not re-order or recolour -- pinned by tests/test_score_matrix.py.
# --------------------------------------------------------------------------- #
SCORE_COLORS = {
    -1: "#762a83",  # strong purple  -- retraction
    0: "#e2d4e8",   # pale lavender  -- no response (modal)
    1: "#f2ebf5",   # palest lavender-- minimal
    2: "#a6dba0",   # light green    -- reaction boundary (score >= 2)
    3: "#5aae61",
    4: "#1b7837",
    5: "#00441b",   # darkest green  -- strong reaction
}

TRACE_INK = "#26262b"      # neutral dark line -- shape is the focus
ODOR_SHADE = "#ebe7df"     # warm neutral for the "odor on" window
BASELINE_INK = "#9a9a9a"   # faint resting reference

# --------------------------------------------------------------------------- #
# The seven curated exemplars (score -> trace).  Odours anonymised in the plot.
#   (score, dataset, fly, fly_number, trial_label, plain-language tag)
# --------------------------------------------------------------------------- #
EXEMPLARS = [
    (5, "RandomPanel-Training-24-10", "june_27_batch_1", 2,
     "training_1_linalool", "Strong, sustained"),
    (4, "RandomPanel-Training-24-10", "june_27_batch_1", 3,
     "training_3_ethylbutyrate", "Strong extension"),
    (3, "RandomPanel-24-1", "july_10_batch_1_rig_2", 2,
     "training_3_acv", "Clear extension"),
    (2, "RandomPanel-Training-24-10", "june_27_batch_1_rig_2", 3,
     "training_6_ethylbutyrate", "Weak / brief extension"),
    (1, "RandomPanel-24-1", "july_10_batch_1_rig_2", 2,
     "training_8_3-octonol", "Minimal movement"),
    (0, "RandomPanel-24-1", "july_10_batch_1_rig_2", 2,
     "training_14_citral", "No response"),
    (-1, "RandomPanel-24-1", "july_03_batch_1_rig_2", 2,
     "training_14_acv", "Retraction (dips below rest)"),
]

LEFT_COL = [5, 4, 3, 2]     # Reaction  (score >= 2)
RIGHT_COL = [1, 0, -1]      # No reaction (score <= 1)
X_MAX = 90                  # seconds

# axis geometry per mode: (y_lo, y_hi, ticks, label, zero_ref)
#   y_hi carries extra "sky" above the data for the badge/labels.
MODE_AXES = {
    "raw": (0, 125, [0, 50, 100], "PER %", None),
    "baseline": (-32, 105, [-25, 0, 25, 50], "ΔPER (%)", 0.0),
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _trace_path(dataset: str, fly: str, fnum: int, trial: str) -> Path:
    return (TRACE_ROOT / dataset / fly / "angle_distance_rms_envelope"
            / f"{trial}_fly{fnum}_distances_fly{fnum}_angle_distance_rms_envelope.csv")


def load_exemplars() -> dict:
    """Return {score: dict(time, per, on_s, off_s, base, tag)}."""
    meta_cols = ["dataset", "fly", "fly_number", "trial_label",
                 "trial_odor_on_s", "trial_odor_off_s"]
    wide = pd.read_csv(WIDE_CSV, usecols=meta_cols)

    out = {}
    for score, dataset, fly, fnum, trial, tag in EXEMPLARS:
        p = _trace_path(dataset, fly, fnum, trial)
        if not p.exists():
            raise FileNotFoundError(f"score {score}: missing trace {p}")
        t = pd.read_csv(p, usecols=["time_s", "combined_pct"])
        w = wide[(wide.dataset == dataset) & (wide.fly == fly)
                 & (wide.fly_number == fnum) & (wide.trial_label == trial)]
        if not w.empty and pd.notna(w.trial_odor_on_s.iloc[0]):
            on_s = float(w.trial_odor_on_s.iloc[0])
            off_s = float(w.trial_odor_off_s.iloc[0])
        else:
            on_s, off_s = 30.0, 60.0  # RandomPanel default schedule
        base = t.loc[t.time_s < on_s - 2, "combined_pct"].mean()
        out[score] = dict(time=t.time_s.to_numpy(), per=t.combined_pct.to_numpy(),
                          on_s=on_s, off_s=off_s, base=base, tag=tag)
    return out


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def _ideal_text(hex_color: str) -> str:
    """Black or white text for legibility on a filled badge."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r + 0.7152 * g + 0.4722 * b  # slight green bias
    return "#ffffff" if lum < 0.55 else "#1a1a1a"


def _draw_panel(ax, score: int, d: dict, mode: str, *,
                show_x: bool, show_odor_label: bool):
    y_lo, y_hi, ticks, ylabel, zero_ref = MODE_AXES[mode]
    y = d["per"] - d["base"] if mode == "baseline" else d["per"]
    ref = zero_ref if mode == "baseline" else d["base"]
    data_top = ticks[-1] + (25 if mode == "baseline" else 0)

    ax.axvspan(d["on_s"], d["off_s"], color=ODOR_SHADE, zorder=0, lw=0)
    ax.axhline(ref, color=BASELINE_INK, lw=0.8, ls=(0, (4, 3)), alpha=0.55, zorder=1)
    ax.plot(d["time"], y, color=TRACE_INK, lw=1.3, zorder=3, solid_capstyle="round")

    ax.set_xlim(0, X_MAX)
    ax.set_ylim(y_lo, y_hi)
    ax.set_yticks(ticks)
    ax.set_xticks([0, 30, 60, 90])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bcbcbc")
    ax.spines["left"].set_bounds(ticks[0], data_top)
    ax.tick_params(colors="#666666", labelsize=8.5, length=3)
    ax.set_ylabel(ylabel, fontsize=9.5, color="#444444")
    if show_x:
        ax.set_xlabel("Time (s)", fontsize=9.5, color="#444444")
    else:
        ax.set_xticklabels([])

    if show_odor_label:
        ax.text((d["on_s"] + d["off_s"]) / 2, 0.965, "odor on",
                transform=ax.get_xaxis_transform(), ha="center", va="center",
                fontsize=8.5, color="#8a8175", style="italic", zorder=4)

    # score badge (colour = identity) + plain-language tag, in the sky
    fill = SCORE_COLORS[score]
    edge = "#8a8a8a" if score in (0, 1) else fill  # ring the pale ones
    txt = "−1" if score == -1 else str(score)
    ax.add_patch(FancyBboxPatch(
        (0.028, 0.815), 0.125, 0.155, transform=ax.transAxes,
        boxstyle="round,pad=0.006,rounding_size=0.03",
        facecolor=fill, edgecolor=edge, lw=1.0, zorder=5, clip_on=False))
    ax.text(0.0905, 0.892, txt, transform=ax.transAxes, ha="center", va="center",
            fontsize=15, fontweight="bold", color=_ideal_text(fill),
            zorder=6, clip_on=False)
    # (plain-language descriptor intentionally omitted -- explained in the caption)


def _draw_scale_key(ax):
    """Full -1..5 colour scale (no binary-cut annotation)."""
    ax.set_axis_off()
    scores = list(range(-1, 6))
    n = len(scores)
    x0, x1, y0, h = 0.06, 0.94, 0.46, 0.26
    w = (x1 - x0) / n
    for i, s in enumerate(scores):
        fill = SCORE_COLORS[s]
        edge = "#8a8a8a" if s in (0, 1) else fill
        ax.add_patch(plt.Rectangle((x0 + i * w, y0), w, h, transform=ax.transAxes,
                                   facecolor=fill, edgecolor=edge, lw=0.8,
                                   clip_on=False))
        ax.text(x0 + (i + 0.5) * w, y0 + h / 2, f"{s}".replace("-1", "−1"),
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=_ideal_text(fill))
    ax.text(x0 + 1.5 * w, y0 - 0.10, "no reaction", transform=ax.transAxes,
            ha="center", va="top", fontsize=8.5, color="#6b4a73")
    ax.text(x0 + 5 * w, y0 - 0.10, "reaction", transform=ax.transAxes,
            ha="center", va="top", fontsize=8.5, color="#1b7837")
    ax.text(0.5, 0.96, "The Response Score Scale", transform=ax.transAxes,
            ha="center", va="top", fontsize=10.5, fontweight="bold",
            color="#2a2a2a")


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def build_figure(data: dict, mode: str) -> plt.Figure:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(9.4, 10.2))
    gs = fig.add_gridspec(
        4, 2, left=0.085, right=0.975, top=0.865, bottom=0.075,
        hspace=0.42, wspace=0.20)

    for row, score in enumerate(LEFT_COL):
        ax = fig.add_subplot(gs[row, 0])
        _draw_panel(ax, score, data[score], mode, show_x=(row == 3),
                    show_odor_label=(row == 0))
    for row, score in enumerate(RIGHT_COL):
        ax = fig.add_subplot(gs[row, 1])
        _draw_panel(ax, score, data[score], mode, show_x=(row == 2),
                    show_odor_label=(row == 0))
    _draw_scale_key(fig.add_subplot(gs[3, 1]))

    fig.text(0.30, 0.885, "REACTION   (score ≥ 2)", ha="center",
             fontsize=12.5, fontweight="bold", color="#1b7837")
    fig.text(0.755, 0.885, "NO REACTION   (score ≤ 1)", ha="center",
             fontsize=12.5, fontweight="bold", color="#762a83")

    title = ("Baseline-Subtracted PER Across the −1 to 5 Score Scale"
             if mode == "baseline"
             else "PER Responses Across the −1 to 5 Score Scale")
    fig.suptitle(title, x=0.5, y=0.965, fontsize=16, fontweight="bold",
                 color="#1a1a1a")
    return fig


def build_group(data: dict, scores: list, mode: str, title: str,
                title_color: str) -> plt.Figure:
    """One standalone single-column figure for a subset of scores."""
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    n = len(scores)
    fig = plt.figure(figsize=(7.4, 2.15 * n + 0.9))
    gs = fig.add_gridspec(n, 1, left=0.135, right=0.965,
                          top=1 - 0.62 / (2.15 * n + 0.9), bottom=0.5 / (2.15 * n + 0.9),
                          hspace=0.4)
    for row, score in enumerate(scores):
        ax = fig.add_subplot(gs[row, 0])
        _draw_panel(ax, score, data[score], mode, show_x=(row == n - 1),
                    show_odor_label=(row == 0))
    fig.suptitle(title, y=1 - 0.16 / (2.15 * n + 0.9), fontsize=15,
                 fontweight="bold", color=title_color)
    return fig


def build_scale() -> plt.Figure:
    """Standalone -1..5 colour-scale legend."""
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(7.6, 2.05))
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
    ax.set_axis_off()
    scores = list(range(-1, 6))
    n = len(scores)
    x0, x1, y0, h = 0.02, 0.98, 0.30, 0.34
    w = (x1 - x0) / n
    for i, s in enumerate(scores):
        fill = SCORE_COLORS[s]
        edge = "#8a8a8a" if s in (0, 1) else fill
        ax.add_patch(plt.Rectangle((x0 + i * w, y0), w, h, transform=ax.transAxes,
                                   facecolor=fill, edgecolor=edge, lw=0.8, clip_on=False))
        ax.text(x0 + (i + 0.5) * w, y0 + h / 2, f"{s}".replace("-1", "−1"),
                transform=ax.transAxes, ha="center", va="center",
                fontsize=15, fontweight="bold", color=_ideal_text(fill))
    ax.text(x0 + 1.5 * w, y0 - 0.10, "no reaction", transform=ax.transAxes,
            ha="center", va="top", fontsize=11, color="#6b4a73")
    ax.text(x0 + 5 * w, y0 - 0.10, "reaction", transform=ax.transAxes,
            ha="center", va="top", fontsize=11, color="#1b7837")
    ax.text(0.5, 0.99, "The Response Score Scale", transform=ax.transAxes,
            ha="center", va="top", fontsize=13.5, fontweight="bold", color="#1a1a1a")
    return fig


def _save(fig, stem: str) -> None:
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    data = load_exemplars()

    # combined two-column figures (raw + baseline)
    for mode, stem in [("raw", "score_scale_PER_traces"),
                       ("baseline", "score_scale_PER_traces_baseline")]:
        _save(build_figure(data, mode), stem)

    # split-by-class standalone figures, raw + baseline
    for mode, sfx in [("raw", "raw"), ("baseline", "baseline")]:
        pre = "Baseline-Subtracted " if mode == "baseline" else ""
        _save(build_group(data, LEFT_COL, mode,
                          f"{pre}PER Reactions (Score ≥ 2)", "#1b7837"),
              f"score_scale_reactions_{sfx}")
        _save(build_group(data, RIGHT_COL, mode,
                          f"{pre}PER: No Reaction & Retraction (Score ≤ 1)", "#762a83"),
              f"score_scale_noreaction_{sfx}")

    # standalone scale legend
    _save(build_scale(), "score_scale_legend")


if __name__ == "__main__":
    main()
