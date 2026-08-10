"""Methods figures for PID verification of odor delivery.

Two publication figures:

``pid_trace_<odor>.png/.pdf``
    One PID recording, start to finish: raw voltage across the whole session
    with the valve-open windows shaded (panel a), and the same exposures
    aligned to valve actuation as voltage above each exposure's own pre-odor
    baseline (panel b), with the 3-sigma onset criterion drawn on.

``pid_odor_arrival_latency.png/.pdf``
    Valve-to-fly arrival latency for every exposure: per odorant and enclosure
    (panel a) and pooled across both enclosures (panel b).

Two different provenances meet in these figures, and they are kept separate on
purpose:

* Panel-level *latencies* (figure 2) are the archived per-exposure exports the
  thesis text was written from -- ``*_latency_rms.csv`` for the optogenetic
  enclosure and ``aggregate_all_exposures.csv`` for the manual one.  This
  module re-plots those values; it does not re-detect them.  The script that
  produced them applied an RMS smooth, a 0.5 s floor and a 15 s ceiling.
* The single *trace* (figure 1) is re-detected here from raw voltage with the
  documented 3-sigma criterion below, because no archived onset exists for that
  session.  Its onset is therefore a like-for-like illustration of the
  criterion, not one of the 53 pooled measurements.

Two manual-enclosure exposures (5.0 s and 7.2 s) are excluded from the pooled
statistics; see ``EXCLUDED_EXPOSURES``.  They are still drawn in panel a as
open markers so the exclusion is visible rather than silent.

Run:
    python scripts/analysis/pid_odor_figures.py
    python scripts/analysis/pid_odor_figures.py --session 1752791786   # benzaldehyde
Outputs (.png / .pdf / .svg each):
    figures/pid_trace_ethyl_butyrate     (figure 1; --session picks the odorant)
    figures/pid_odor_arrival_latency     (figure 2)
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# --------------------------------------------------------------------------- #
# Data locations
# --------------------------------------------------------------------------- #
DATA_ROOT = Path("/home/ramanlab/Documents/cole/Data/Odor_data")
OPTO_DIR = DATA_ROOT / "data_odor opto"
MANUAL_DIR = DATA_ROOT / "odor data PID Manual"
OPTO_LATENCY_DIR = OPTO_DIR / "all_odors"
MANUAL_EXPOSURES_CSV = MANUAL_DIR / "more" / "all_odors" / "aggregate_all_exposures.csv"
MANUAL_RAW_DIR = MANUAL_DIR / "more" / "data_odor" / "data_odor"

BENZ_SESSION = MANUAL_DIR / "odor_test_1752791786.csv"
OUT_DIR = Path(__file__).resolve().parents[2] / "figures"

# Sessions that can front figure 1: session id -> (odor, enclosure, provenance note)
TRACE_SESSIONS = {
    "1752791786": (
        "Benzaldehyde",
        "manual",
        "manual enclosure, 5 Hz sampling, delivery tube removed",
    ),
    "1757376621": (
        "Ethyl Butyrate",
        "optogenetic",
        "optogenetic enclosure, PID at the fly position",
    ),
}

# --------------------------------------------------------------------------- #
# Exposures dropped from the pooled statistics: the two longest manual-enclosure
# detections (hexanol 5.01 s, linalool 7.21 s).  Hard-coded rather than derived
# by a "drop the two largest" rule so the exclusion cannot drift with the data.
# --------------------------------------------------------------------------- #
EXCLUDED_EXPOSURES: tuple[tuple[str, int], ...] = (
    ("1756844942", 0),  # hexanol,  manual, 5.008 s
    ("1756848566", 1),  # linalool, manual, 7.211 s
)

ODOR_NAMES = {
    "A": "Apple Cider Vinegar",
    "B": "Benzaldehyde",
    "C": "Citral",
    "E": "Ethyl Butyrate",
    "H": "Hexanol",
    "L": "Linalool",
    "O": "3-Octanol",
}
ODOR_ORDER = ["O", "A", "B", "C", "E", "H", "L"]
ODOR_TICKS = {
    "A": "Apple cider vinegar",
    "B": "Benzaldehyde",
    "C": "Citral",
    "E": "Ethyl butyrate",
    "H": "Hexanol",
    "L": "Linalool",
    "O": "3-Octanol",
}

# --------------------------------------------------------------------------- #
# Palette.  Two categorical slots (one per enclosure) from the CVD-validated
# default theme; verified with the dataviz validator on the light surface:
#   worst-pair CVD dE 24.7 (protan) / 32.7 (tritan), normal-vision dE 33.6,
#   both >= 3:1 contrast.  Enclosure colour is identical in both figures.
# --------------------------------------------------------------------------- #
RIG_COLORS = {"optogenetic": "#2a78d6", "manual": "#eb6834"}
RIG_LABELS = {"optogenetic": "Optogenetic enclosure", "manual": "Manual enclosure"}
RIG_SHORT = {"optogenetic": "Optogenetic", "manual": "Manual"}

SURFACE = "#ffffff"
INK = "#1b1b1b"
INK_SOFT = "#5c5c5c"
INK_FAINT = "#9a9a9a"
ODOR_SHADE = "#ebe7df"  # warm neutral for the valve-open window
GRID = "#e4e4e1"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK_SOFT,
        "axes.linewidth": 0.7,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "font.size": 8,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "pdf.fonttype": 42,  # editable text in Illustrator / Inkscape
        "ps.fonttype": 42,
        "svg.fonttype": "none",  # keep <text> as text, not paths, in the SVG
    }
)

FIG_WIDTH_IN = 6.5  # full thesis text width

# Onset criterion for the re-detected single trace.
PRE_WINDOW_SEC = 20.0
SIGMA = 3.0
SUSTAIN_SAMPLES = 3


# --------------------------------------------------------------------------- #
# Aggregate latency dataset
# --------------------------------------------------------------------------- #
def load_latency_table() -> pd.DataFrame:
    """Every attempted exposure from both enclosures, one row per exposure.

    Columns: rig, session, odor_code, odor, repeat, latency_s, note, included.
    ``latency_s`` is NaN where the archived export recorded no valid detection;
    ``included`` marks the exposures that enter the pooled statistics.
    """
    rows = []

    for path in sorted(OPTO_LATENCY_DIR.glob("odor_test_*_latency_rms.csv")):
        session = re.search(r"odor_test_(\d+)_", path.name).group(1)
        df = pd.read_csv(path)
        for row in df.itertuples():
            rows.append(
                {
                    "rig": "optogenetic",
                    "session": session,
                    "odor_code": str(row.Odor).strip(),
                    "repeat": int(row.Repeat),
                    "latency_s": float(row.Latency_s) if pd.notna(row.Latency_s) else np.nan,
                    "note": "" if pd.isna(row.Note) else str(row.Note),
                }
            )

    manual = pd.read_csv(MANUAL_EXPOSURES_CSV)
    for row in manual.itertuples():
        rows.append(
            {
                "rig": "manual",
                "session": re.search(r"(\d+)", str(row.file)).group(1),
                "odor_code": str(row.Odor).strip(),
                "repeat": int(row.Repeat),
                "latency_s": float(row.Latency_s) if pd.notna(row.Latency_s) else np.nan,
                "note": "" if pd.isna(row.Note) else str(row.Note),
            }
        )

    table = pd.DataFrame(rows)
    table["odor"] = table.odor_code.map(ODOR_NAMES)
    dropped = set(EXCLUDED_EXPOSURES)
    is_dropped = np.array(
        [(s, r) in dropped for s, r in zip(table.session, table.repeat)], dtype=bool
    )
    table["included"] = table.latency_s.notna() & ~is_dropped
    return table


def pooled_stats(latencies: Sequence[float] | pd.Series) -> tuple[int, float, float]:
    """(n, mean, SEM) over the non-missing latencies."""
    values = pd.Series(list(latencies), dtype=float).dropna()
    if values.empty:
        return 0, float("nan"), float("nan")
    return len(values), float(values.mean()), float(values.sem())


# --------------------------------------------------------------------------- #
# Single-session PID trace
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Window:
    """One valve-open exposure inside a session."""

    repeat: int
    t_open: float
    t_close: float


@dataclass(frozen=True)
class Onset:
    """Result of the 3-sigma onset criterion for one exposure."""

    latency_s: float | None
    baseline_v: float
    baseline_sd_v: float
    threshold_v: float
    peak_v: float


def load_pid_session(path: Path) -> pd.DataFrame:
    """Read a raw ``odor_test_*.csv`` PID log (Timestamp, Voltage, Phase, Repeat)."""
    df = pd.read_csv(path)
    missing = {"Timestamp", "Voltage", "Phase", "Repeat"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    df = df.copy()
    df["Timestamp"] = pd.to_numeric(df.Timestamp, errors="coerce")
    df["Voltage"] = pd.to_numeric(df.Voltage, errors="coerce")
    df["Phase"] = df.Phase.astype(str).str.strip().str.lower()
    df = df.dropna(subset=["Timestamp", "Voltage"]).reset_index(drop=True)
    df["Timestamp"] -= float(df.Timestamp.iloc[0])
    return df


def exposure_windows(df: pd.DataFrame) -> list[Window]:
    """Valve-open windows, in order, from the ``odor_on`` phase samples."""
    on = df[df.Phase == "odor_on"]
    if on.empty:
        raise ValueError("session contains no odor_on samples")
    windows = [
        Window(repeat=int(rep), t_open=float(g.Timestamp.min()), t_close=float(g.Timestamp.max()))
        for rep, g in on.groupby("Repeat", sort=True)
    ]
    return sorted(windows, key=lambda w: w.t_open)


def detect_onset(
    df: pd.DataFrame,
    window: Window,
    *,
    pre_sec: float = PRE_WINDOW_SEC,
    sigma: float = SIGMA,
    sustain: int = SUSTAIN_SAMPLES,
) -> Onset:
    """First sustained rise above baseline + ``sigma`` SD after valve actuation.

    Baseline is the ``pre_sec`` seconds of PID signal immediately before the
    valve opens; the onset is the first of ``sustain`` consecutive samples above
    threshold.  Returns ``latency_s=None`` when the signal never crosses.
    """
    pre = df[(df.Timestamp < window.t_open) & (df.Timestamp >= window.t_open - pre_sec)].Voltage
    if len(pre) < 4:
        raise ValueError(f"fewer than 4 pre-odor samples for repeat {window.repeat}")
    baseline = float(pre.mean())
    baseline_sd = float(pre.std(ddof=1))
    threshold = baseline + sigma * baseline_sd

    during = df[(df.Timestamp >= window.t_open) & (df.Timestamp <= window.t_close)]
    t = during.Timestamp.to_numpy() - window.t_open
    v = during.Voltage.to_numpy()

    latency: float | None = None
    for i in range(len(v) - sustain + 1):
        if np.all(v[i : i + sustain] > threshold):
            latency = float(t[i])
            break

    return Onset(
        latency_s=latency,
        baseline_v=baseline,
        baseline_sd_v=baseline_sd,
        threshold_v=threshold,
        peak_v=float(v.max()),
    )


PLATEAU_START_SEC = 5.0  # skip the rising edge before measuring the plateau


def plateau_stats(df: pd.DataFrame, window: Window, onset: Onset) -> tuple[float, float]:
    """(mean, SD) of the signal above baseline once the plume has fully arrived.

    Measured from ``PLATEAU_START_SEC`` after valve actuation to valve closure --
    the interval the figure uses to show that delivery is steady, not a spike.
    """
    seg = df[
        (df.Timestamp >= window.t_open + PLATEAU_START_SEC) & (df.Timestamp <= window.t_close)
    ]
    if seg.empty:
        raise ValueError(f"no plateau samples for repeat {window.repeat}")
    dv = seg.Voltage - onset.baseline_v
    return float(dv.mean()), float(dv.std(ddof=1))


# --------------------------------------------------------------------------- #
# Figure 1 -- one odorant, one recording
# --------------------------------------------------------------------------- #
def make_trace_figure(session_path: Path, outdir: Path) -> tuple[Path, ...]:
    session = re.search(r"odor_test_(\d+)", session_path.name).group(1)
    odor, rig, _note = TRACE_SESSIONS.get(session, ("Odorant", "manual", ""))
    df = load_pid_session(session_path)
    windows = exposure_windows(df)
    onsets = [detect_onset(df, w) for w in windows]

    fig, (ax_full, ax_zoom) = plt.subplots(
        1,
        2,
        figsize=(FIG_WIDTH_IN, 2.6),
        gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.28},
    )

    # -- panel a: the whole recording ------------------------------------- #
    for w in windows:
        ax_full.axvspan(w.t_open, w.t_close, color=ODOR_SHADE, lw=0, zorder=0)
    ax_full.plot(df.Timestamp, df.Voltage, color=RIG_COLORS[rig], lw=0.7, zorder=3)

    ax_full.set_xlabel("Time (s)")
    ax_full.set_ylabel("PID signal (V)")
    ax_full.set_xlim(0, float(df.Timestamp.max()))
    y_lo, y_hi = float(df.Voltage.min()), float(df.Voltage.max())
    pad = 0.12 * (y_hi - y_lo)
    ax_full.set_ylim(y_lo - pad, y_hi + 2.6 * pad)
    ax_full.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax_full.set_axisbelow(True)

    band_label_y = y_hi + 1.15 * pad
    for i, w in enumerate(windows, start=1):
        ax_full.text(
            (w.t_open + w.t_close) / 2,
            band_label_y,
            str(i),
            ha="center",
            va="center",
            fontsize=7,
            color=INK_SOFT,
        )
    ax_full.text(
        0.0,
        y_hi + 2.25 * pad,
        "shading: odor valve open",
        ha="left",
        va="center",
        fontsize=7,
        color=INK_SOFT,
    )

    # -- panel b: the first exposure, across the whole valve-open window ---- #
    first_w, first_onset = windows[0], onsets[0]
    duration = first_w.t_close - first_w.t_open
    tail = 0.13 * duration  # a little past valve close, to show the return
    x_lo, x_hi = -0.10 * duration, duration + tail

    seg = df[
        (df.Timestamp >= first_w.t_open + x_lo) & (df.Timestamp <= first_w.t_open + x_hi)
    ]
    t = seg.Timestamp.to_numpy() - first_w.t_open
    dv = seg.Voltage.to_numpy() - first_onset.baseline_v
    color = RIG_COLORS[rig]

    ax_zoom.axvspan(0.0, duration, color=ODOR_SHADE, lw=0, zorder=0)
    ax_zoom.plot(t, dv, color=color, lw=1.0, zorder=3)

    threshold_dv = first_onset.baseline_sd_v * SIGMA
    ax_zoom.axhline(threshold_dv, color=INK_FAINT, ls=(0, (2, 2)), lw=0.7, zorder=2)
    ax_zoom.axvline(0.0, color=INK_SOFT, ls=(0, (3, 2)), lw=0.7, zorder=2)

    y_hi_zoom = float(dv.max())
    ax_zoom.set_xlim(x_lo, x_hi)
    ax_zoom.set_ylim(-0.09 * y_hi_zoom, 1.34 * y_hi_zoom)
    ax_zoom.set_xlabel("Time from valve actuation (s)")
    ax_zoom.set_ylabel("PID above baseline (ΔV)")
    ax_zoom.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax_zoom.set_axisbelow(True)

    # The plateau: mean +/- SD once the plume has fully arrived, drawn across the
    # interval it was measured over so "steady for the whole presentation" is
    # something the reader can check rather than take on trust.
    plateau_mean, plateau_sd = plateau_stats(df, first_w, first_onset)
    ax_zoom.hlines(
        plateau_mean, PLATEAU_START_SEC, duration, color=INK, lw=0.9, zorder=4
    )
    ax_zoom.text(
        (PLATEAU_START_SEC + duration) / 2,
        plateau_mean + 0.055 * y_hi_zoom,
        f"plateau {plateau_mean:.2f} ± {plateau_sd:.2f} V",
        fontsize=7,
        color=INK,
        ha="center",
        va="bottom",
        zorder=6,
    ).set_path_effects([pe.Stroke(linewidth=2.0, foreground=SURFACE), pe.Normal()])

    if first_onset.latency_s is not None:
        idx = int(np.argmin(np.abs(t - first_onset.latency_s)))
        ax_zoom.plot(
            [first_onset.latency_s],
            [dv[idx]],
            marker="o",
            ms=4.0,
            mfc=color,
            mec=SURFACE,
            mew=0.9,
            zorder=5,
        )
        ax_zoom.text(
            0.02 * duration,
            1.10 * y_hi_zoom,
            f"onset {first_onset.latency_s:.1f} s",
            fontsize=7,
            color=INK,
            ha="left",
            va="center",
        )

    ax_zoom.text(
        0.02 * duration,
        1.22 * y_hi_zoom,
        "valve opens",
        fontsize=7,
        color=INK_SOFT,
        ha="left",
        va="center",
    )
    ax_zoom.text(
        duration - 0.02 * duration,
        1.22 * y_hi_zoom,
        "valve closes",
        fontsize=7,
        color=INK_SOFT,
        ha="right",
        va="center",
    )
    ax_zoom.text(
        x_hi,
        threshold_dv + 0.035 * y_hi_zoom,
        f"{SIGMA:.0f}σ threshold",
        fontsize=7,
        color=INK_SOFT,
        ha="right",
        va="bottom",
    )

    for ax, letter in ((ax_full, "a"), (ax_zoom, "b")):
        ax.text(
            -0.13 if ax is ax_full else -0.19,
            1.06,
            letter,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"pid_trace_{odor.lower().replace(' ', '_')}"
    return _save(fig, outdir, stem)


# --------------------------------------------------------------------------- #
# Figure 2 -- arrival latency across odorants and enclosures
# --------------------------------------------------------------------------- #
def make_latency_figure(outdir: Path) -> tuple[Path, ...]:
    table = load_latency_table()

    fig, (ax_odor, ax_pool) = plt.subplots(
        1,
        2,
        figsize=(FIG_WIDTH_IN, 2.9),
        gridspec_kw={"width_ratios": [1.62, 1.0], "wspace": 0.30},
    )

    # -- panel a: per odorant, per enclosure ------------------------------- #
    rigs = ["optogenetic", "manual"]
    offsets = {"optogenetic": -0.19, "manual": 0.19}
    rng = np.random.default_rng(7)  # jitter only; fixed seed keeps the figure reproducible

    for x, code in enumerate(ODOR_ORDER):
        for rig in rigs:
            sub = table[(table.rig == rig) & (table.odor_code == code)]
            if sub.empty:
                continue
            center = x + offsets[rig]
            color = RIG_COLORS[rig]
            detected = sub[sub.latency_s.notna()]
            if detected.empty:
                ax_odor.text(
                    center, 0.22, "n.d.", fontsize=6.5, color=color, ha="center", va="bottom"
                )
                continue

            kept = detected[detected.included]
            dropped = detected[~detected.included]
            jitter = rng.uniform(-0.075, 0.075, len(kept))
            ax_odor.scatter(
                center + jitter,
                kept.latency_s,
                s=11,
                color=color,
                alpha=0.75,
                lw=0.0,
                zorder=3,
            )
            if not dropped.empty:
                ax_odor.scatter(
                    center + rng.uniform(-0.075, 0.075, len(dropped)),
                    dropped.latency_s,
                    s=16,
                    facecolor=SURFACE,
                    edgecolor=color,
                    lw=0.8,
                    zorder=3,
                )
            if not kept.empty:
                _, mean, sem = pooled_stats(kept.latency_s)
                ax_odor.errorbar(
                    center,
                    mean,
                    yerr=0.0 if np.isnan(sem) else sem,
                    color=color,
                    lw=1.1,
                    capsize=2.2,
                    capthick=1.1,
                    zorder=4,
                )
                ax_odor.plot(
                    [center - 0.13, center + 0.13],
                    [mean, mean],
                    color=color,
                    lw=1.7,
                    solid_capstyle="butt",
                    zorder=5,
                )

    ax_odor.set_xticks(range(len(ODOR_ORDER)))
    ax_odor.set_xticklabels(
        [ODOR_TICKS[c] for c in ODOR_ORDER], fontsize=7, rotation=28, ha="right",
        rotation_mode="anchor",
    )
    ax_odor.set_xlim(-0.6, len(ODOR_ORDER) - 0.4)
    ax_odor.set_ylim(0, 8.2)
    ax_odor.set_ylabel("Arrival latency (s)")
    ax_odor.yaxis.grid(True, color=GRID, lw=0.6)
    ax_odor.set_axisbelow(True)

    handles = [
        Line2D([], [], marker="o", ls="none", ms=4, color=RIG_COLORS[r], label=RIG_SHORT[r])
        for r in rigs
    ]
    handles.append(
        Line2D(
            [], [], marker="o", ls="none", ms=4, mfc=SURFACE, mec=INK_SOFT, mew=0.8,
            label="Excluded",
        )
    )
    ax_odor.legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        ncol=3,
        fontsize=7,
        handletextpad=0.3,
        columnspacing=1.1,
        borderpad=0.0,
        borderaxespad=0.2,
    )

    # -- panel b: pooled distribution -------------------------------------- #
    pooled = table[table.included]
    n, mean, sem = pooled_stats(pooled.latency_s)
    bins = np.arange(0.0, 6.5, 0.5)
    ax_pool.hist(
        [pooled[pooled.rig == r].latency_s for r in rigs],
        bins=bins,
        stacked=True,
        color=[RIG_COLORS[r] for r in rigs],
        label=[RIG_LABELS[r] for r in rigs],
        edgecolor=SURFACE,
        linewidth=0.8,
        zorder=3,
    )

    counts, _ = np.histogram(pooled.latency_s, bins=bins)
    y_top = float(counts.max()) * 1.30
    ax_pool.axvspan(mean - sem, mean + sem, color=INK_FAINT, alpha=0.28, lw=0, zorder=2)
    ax_pool.axvline(mean, color=INK, lw=1.0, zorder=4)
    ax_pool.text(
        5.9,
        y_top * 0.98,
        f"pooled mean\n{mean:.1f} ± {sem:.1f} s\nn = {n}",
        fontsize=7.5,
        color=INK,
        ha="right",
        va="top",
        linespacing=1.35,
    )
    ax_pool.set_xlim(0, 6.0)
    ax_pool.set_ylim(0, y_top)
    ax_pool.set_xlabel("Arrival latency (s)")
    ax_pool.set_ylabel("Exposures")
    ax_pool.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax_pool.yaxis.grid(True, color=GRID, lw=0.6)
    ax_pool.set_axisbelow(True)

    for ax, letter, dx in ((ax_odor, "a", -0.105), (ax_pool, "b", -0.185)):
        ax.text(
            dx,
            1.06,
            letter,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    outdir.mkdir(parents=True, exist_ok=True)
    return _save(fig, outdir, "pid_odor_arrival_latency")


def _save(fig: plt.Figure, outdir: Path, stem: str) -> tuple[Path, ...]:
    paths = []
    for suffix in (".png", ".pdf", ".svg"):
        path = outdir / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        paths.append(path)
    plt.close(fig)
    return tuple(paths)


# --------------------------------------------------------------------------- #
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        default="1757376621",
        choices=sorted(TRACE_SESSIONS),
        help="Session id for the single-trace figure.",
    )
    parser.add_argument("--outdir", type=Path, default=OUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    session_path = (
        BENZ_SESSION
        if args.session == "1752791786"
        else OPTO_DIR / f"odor_test_{args.session}.csv"
    )

    for path in make_trace_figure(session_path, args.outdir):
        print(f"Wrote {path}")
    for path in make_latency_figure(args.outdir):
        print(f"Wrote {path}")

    table = load_latency_table()
    for rig in ("optogenetic", "manual"):
        n, mean, sem = pooled_stats(table[(table.rig == rig) & table.included].latency_s)
        print(f"{rig:>12}: n={n:2d}  {mean:.2f} +/- {sem:.2f} s")
    n, mean, sem = pooled_stats(table[table.included].latency_s)
    print(f"{'pooled':>12}: n={n:2d}  {mean:.2f} +/- {sem:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
