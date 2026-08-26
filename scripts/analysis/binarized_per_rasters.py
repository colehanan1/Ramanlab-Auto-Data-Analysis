"""Binarised PER rasters: every fly, every trial, sorted by conditioning vigor.

Takes the same envelope traces the per-fly figures under
``Raw-Training-PER-Traces/`` draw, thresholds each trial at the **same red line**
those figures show -- ``theta = median_before + k * MAD_before`` with k = 2, the
one-sided baseline threshold from ``envelope_visuals._baseline_theta`` -- and
collapses each trace to 0/1 per frame.

Flies are then ranked by the mean fraction of the **odor window** spent above
threshold, averaged over that fly's training trials. That single ordering is
reused for every panel of every figure, so a row is the same fly everywhere and
the eye can track it across training and testing.

Two figures per dataset:

``<dataset>_training_raster.png|svg``
    One panel per conditioning trial (Training 1 ... 6). Rows are flies, top =
    most responsive during conditioning.

``<dataset>_testing_raster.png|svg``
    One panel per **odor**, not per trial index -- in this protocol testing
    trials 2-7 are randomised per fly, so a column of "testing 3" would be a
    different odor on every row. An odor presented twice (hexanol, at testing 1
    and testing 8) gets one panel per presentation: "Hexanol 1", "Hexanol 2".
    Panels run left to right by median trial index, so the fixed first odor
    lands on the left and light-only on the right.

Usage::

    python scripts/analysis/binarized_per_rasters.py \\
        --wide-csv .../all_envelope_rows_wide_combined_base.parquet \\
        --training-wide-csv .../all_envelope_rows_wide_combined_base_training.parquet \\
        --predictions-csv .../model_predictions.csv \\
        --dataset Hex-Control-24-0.1 \\
        --out-dir .../New-Opto-Fly-Figures/Binarized-PER-Rasters

For the continuous envelope heatmap variant, add ``--mode heatmap`` (or
``--mode both`` to emit both binary rasters and heatmaps). Heatmap rows are
sorted by each fly's mean training ``AUC-During``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.analysis.envelope_visuals import _compute_theta as _ev_compute_theta  # noqa: E402
from scripts.analysis.envelope_visuals import _rolling_baseline as _ev_rolling_baseline  # noqa: E402
from scripts.analysis.score_scale_figure import SCORE_COLORS  # noqa: E402

LOGGER = logging.getLogger("binarized_per_rasters")

DPI = 400
K_DEFAULT = 2.0
MAD_TO_SIGMA = 1.4826

# --------------------------------------------------------------------------- #
# Palette. The raster is a two-state map, so it gets the two poles the rest of
# the project already uses for "reacted" -- the pinned PRGn green -- against a
# near-surface neutral for "did not". Never red/green (protanopia collapse), and
# never a value ramp: there are exactly two states.
# --------------------------------------------------------------------------- #
ON_COLOR = SCORE_COLORS[4]        # "#1b7837" -- above threshold
OFF_COLOR = "#eceae5"             # below threshold: recedes toward the surface
MISSING_COLOR = "#ffffff"         # fly not tested on this odor
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ODOR_INK = "#2a78d6"              # odor-window rules on the light binary raster
ODOR_RULE_LIGHT = "#ffffff"       # ...and on the dark end of a sequential ramp

_RC = {
    "figure.dpi": 150,
    "savefig.dpi": DPI,
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 8,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.labelcolor": INK_SECONDARY,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}

_TRIAL_RE = re.compile(r"^(?:training|testing)_(\d+)_(.+)$", re.IGNORECASE)

ODOR_PRETTY = {
    "hexanol": "Hexanol",
    "3-octonol": "3-Octanol",
    "3-octanol": "3-Octanol",
    "ethylbutyrate": "Ethyl Butyrate",
    "citral": "Citral",
    "linalool": "Linalool",
    # "Apple Cider Vinegar", not "ACV": the config's odor_remap is keyed on the
    # project's canonical display name (fbpipe.odor_constants.DISPLAY_LABEL), so
    # a short form here would silently miss the remap.
    "acv": "Apple Cider Vinegar",
    "apple cider vinegar": "Apple Cider Vinegar",
    "benzaldehyde": "Benzaldehyde",
    "isoamylacetate": "Isoamyl Acetate",
    "lightonly": "Light only",
}


#: Odors dropped from the testing figure by default. Light-only is the
#: optogenetic US, not an odor -- it belongs with the training panels, and on a
#: by-odor figure it is a near-saturated column that compresses everything else.
DEFAULT_EXCLUDE_ODORS = ("lightonly",)


def odor_pretty(token: str) -> str:
    """Plain display name for an odor token, before any per-dataset remap."""
    key = str(token).strip().lower()
    return ODOR_PRETTY.get(key, key.replace("_", " ").title())


def odor_display(token: str, dataset: str) -> str:
    """Display label for an odor, honouring the config's per-dataset ``odor_remap``.

    Several cohorts delivered a different liquid than the trial-label suffix
    claims -- Hex-Control-24-0.1's "ACV" channel was actually isoamyl acetate --
    and ``config_new.yaml`` records that per dataset. Resolving labels through
    the same registry every other figure uses means this script cannot drift
    from them, and picks up the delivered concentrations for free.
    """
    from fbpipe.odor_constants import canon_dataset
    from scripts.analysis.envelope_visuals import apply_dataset_odor_remap

    plain = odor_pretty(token)
    for key in (str(dataset), canon_dataset(str(dataset))):
        mapped = apply_dataset_odor_remap(key, plain)
        if mapped != plain:
            return mapped
    return plain


def load_config_remap(config_path: Path | str) -> int:
    """Register every dataset's ``odor_remap`` from the config; returns the count."""
    from fbpipe.config import load_settings

    from scripts.analysis.envelope_visuals import set_dataset_odor_remap

    settings = load_settings(str(config_path))
    remap = {
        str(ds): dict(ov.odor_remap)
        for ds, ov in settings.dataset_overrides.items()
        if getattr(ov, "odor_remap", None)
    }
    set_dataset_odor_remap(remap)
    if remap:
        LOGGER.info("Loaded odor_remap for %d datasets from %s", len(remap), config_path)
    return len(remap)


# --------------------------------------------------------------------------- #
# Threshold -- the red line from the per-fly trace figures
# --------------------------------------------------------------------------- #


def baseline_theta(window: Sequence[float], k: float, min_delta: float = 0.0) -> float:
    """``median + max(k * 1.4826 * MAD, min_delta)``, counting upward deviations only.

    A byte-for-byte mirror of ``envelope_visuals._baseline_theta`` (pinned by
    ``tests/test_binarized_per_rasters.py`` against that function). One-sided on
    purpose: a downward dip is the opposite of a proboscis extension, so letting
    it inflate the dispersion would raise the bar for a real response.
    """
    w = np.asarray(window, dtype=float)
    if w.size == 0:
        return float("nan")
    baseline = float(np.nanmedian(w))
    if not math.isfinite(baseline):
        return float("nan")
    dev = w - baseline
    up = dev[np.isfinite(dev) & (dev > 0.0)]
    mad_up = float(np.nanmedian(up)) if up.size else 0.0
    return float(baseline + max(k * MAD_TO_SIGMA * mad_up, float(min_delta)))


def trial_theta(
    trace: Sequence[float],
    *,
    fps: float,
    baseline_until_s: float,
    k: float = K_DEFAULT,
    min_delta: float = 0.0,
    anchor_s: Optional[float] = None,
) -> float:
    """Threshold for one trial, computed on the pre-odor baseline only.

    ``min_delta`` and ``anchor_s`` mirror ``envelope_visuals._compute_theta``; both
    default off so the raster keeps matching the red line in the trace figures.
    """
    t = np.asarray(trace, dtype=float)
    if t.size == 0 or not math.isfinite(fps) or fps <= 0:
        return float("nan")
    end = min(int(round(baseline_until_s * fps)), t.size)
    if end <= 0:
        return float("nan")
    if anchor_s is None:
        return baseline_theta(t[:end], k, min_delta)
    return float(
        _ev_compute_theta(
            t, fps, end / fps, k, min_delta=min_delta, anchor_s=float(anchor_s)
        )
    )


def baseline_location(
    trace: Sequence[float],
    *,
    fps: float,
    baseline_until_s: float,
    anchor_s: Optional[float] = None,
) -> float:
    """The resting position theta is measured *from*, for this trial.

    Under the legacy rule that is the median of the whole pre-odor baseline;
    under the anchored rule it is the median of the last ``anchor_s`` seconds.
    Kept separate from :func:`trial_theta` because the graded raster needs the
    location on its own, to express a response as "% above rest".
    """
    t = np.asarray(trace, dtype=float)
    if t.size == 0 or not math.isfinite(fps) or fps <= 0:
        return float("nan")
    end = min(int(round(baseline_until_s * fps)), t.size)
    if end <= 0:
        return float("nan")
    before = t[:end]
    if anchor_s is None:
        finite = before[np.isfinite(before)]
        return float(np.median(finite)) if finite.size else float("nan")
    loc, _ = _ev_rolling_baseline(before, fps, float(anchor_s))
    return float(loc)


def binarize(trace: Sequence[float], theta: float) -> np.ndarray:
    """1 where the envelope is strictly above theta.

    Strictly above, matching ``envelope_visuals``' ``during > theta``. A
    non-finite sample is 0 -- the tracker had nothing there, which is not
    evidence of a response (see the rig-3 light artifact).
    """
    t = np.asarray(trace, dtype=float)
    if not math.isfinite(theta):
        return np.zeros(t.size, dtype=bool)
    return np.asarray(np.isfinite(t) & (t > theta), dtype=bool)


def window_fraction(
    binary: Sequence[bool], *, fps: float, start_s: float, end_s: float
) -> float:
    """Fraction of the frames in ``[start_s, end_s)`` that are 1."""
    b = np.asarray(binary, dtype=bool)
    if b.size == 0 or not math.isfinite(fps) or fps <= 0:
        return float("nan")
    a = max(0, int(round(start_s * fps)))
    z = min(b.size, int(round(end_s * fps)))
    if z <= a:
        return float("nan")
    return float(b[a:z].mean())


# --------------------------------------------------------------------------- #
# Trial table
# --------------------------------------------------------------------------- #


def parse_trial(label: object) -> tuple[int, str] | None:
    m = _TRIAL_RE.match(str(label).strip())
    if not m:
        return None
    return int(m.group(1)), m.group(2).strip().lower()


def fly_id(fly: object, fly_number: object) -> str:
    return f"{fly}#{fly_number}"


def _dir_val_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("dir_val_")]
    return sorted(cols, key=lambda c: int(c.split("_")[-1]))


def build_trials(
    wide_df: pd.DataFrame,
    *,
    dataset: str,
    trial_type: str,
    keep: Optional[Iterable[tuple[str, int]]] = None,
    k: float = K_DEFAULT,
    min_delta: float = 0.0,
    anchor_s: Optional[float] = None,
) -> pd.DataFrame:
    """One row per trial: its threshold, its binary trace, and its odor-window score.

    ``min_delta`` and ``anchor_s`` select the baseline rule (see
    :func:`trial_theta`). Both default off, so an existing raster is unchanged.
    """
    want_ds = str(dataset).strip().casefold()
    df = wide_df[wide_df["dataset"].astype(str).str.strip().str.casefold() == want_ds]
    if "trial_type" in df.columns:
        df = df[df["trial_type"].astype(str).str.strip() == str(trial_type).strip()]
    if df.empty:
        return pd.DataFrame()

    if keep is not None:
        keep_set = {(str(f).strip(), int(n)) for f, n in keep}
        mask = [
            (str(f).strip(), int(n)) in keep_set
            for f, n in zip(df["fly"], df["fly_number"])
        ]
        df = df[pd.Series(mask, index=df.index)]
        if df.empty:
            return pd.DataFrame()

    dv = _dir_val_columns(df)
    values = df[dv].to_numpy(dtype=float)

    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        parsed = parse_trial(r["trial_label"])
        if parsed is None:
            continue
        index, odor = parsed
        fps = float(pd.to_numeric(r.get("fps"), errors="coerce") or 0.0)
        if not math.isfinite(fps) or fps <= 0:
            LOGGER.warning("Skipping %s %s: bad fps", r["fly"], r["trial_label"])
            continue
        n = pd.to_numeric(r.get("trace_len"), errors="coerce")
        n = int(n) if math.isfinite(float(n or float("nan"))) else values.shape[1]
        n = max(0, min(n, values.shape[1]))
        trace = values[i, :n]

        on_s = float(pd.to_numeric(r.get("trial_odor_on_s"), errors="coerce"))
        off_s = float(pd.to_numeric(r.get("trial_odor_off_s"), errors="coerce"))
        if not math.isfinite(on_s) or not math.isfinite(off_s):
            LOGGER.warning("Skipping %s %s: no odor window", r["fly"], r["trial_label"])
            continue

        theta = trial_theta(
            trace,
            fps=fps,
            baseline_until_s=on_s,
            k=k,
            min_delta=min_delta,
            anchor_s=anchor_s,
        )
        baseline_loc = baseline_location(
            trace, fps=fps, baseline_until_s=on_s, anchor_s=anchor_s
        )
        binary = binarize(trace, theta)
        # Magnitude above the resting position, kept only where the trace cleared
        # theta. Zero elsewhere so the graded raster can paint below-threshold
        # cells with a single flat colour (Mode.under_color) rather than a green
        # so pale it reads as "small response" when it is "no response".
        graded = np.where(
            binary & np.isfinite(trace),
            np.asarray(trace, dtype=float) - baseline_loc,
            0.0,
        )
        graded = np.clip(np.nan_to_num(graded, nan=0.0), 0.0, None)
        frac = window_fraction(binary, fps=fps, start_s=on_s, end_s=off_s)
        a = max(0, int(round(on_s * fps)))
        z = min(binary.size, int(round(off_s * fps)))

        rows.append({
            "dataset": dataset,
            "fly": r["fly"],
            "fly_number": int(r["fly_number"]),
            "fly_id": fly_id(r["fly"], int(r["fly_number"])),
            "trial_type": str(r.get("trial_type", trial_type)).strip(),
            "trial_index": index,
            "odor": odor,
            "fps": fps,
            "trace_len": int(n),
            "odor_on_s": on_s,
            "odor_off_s": off_s,
            "theta": theta,
            "baseline_loc": baseline_loc,
            "binary": binary,
            "graded": graded,
            "trace": trace,
            "auc_during": float(pd.to_numeric(r.get("AUC-During"), errors="coerce")),
            "n_ones_odor": int(binary[a:z].sum()) if z > a else 0,
            "n_frames_odor": int(z - a) if z > a else 0,
            "odor_fraction": frac,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["fly", "fly_number", "trial_index"]).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


#: Sort keys: how "most responsive during conditioning" is measured.
SORT_KEYS = {
    "binary": "odor_fraction",   # fraction of the odor window above theta
    "auc": "auc_during",         # AUC-During, straight from the wide table
}


def fly_scores(training_trials: pd.DataFrame, by: str = "binary") -> dict[str, float]:
    """Per fly: the mean of ``by`` across its training trials.

    Mean over that fly's own trials -- each trial contributes one value, and
    every fly has the same six, so this is not a mean-of-unbalanced-means.
    """
    if by not in SORT_KEYS:
        raise ValueError(f"unknown sort key {by!r}; expected one of {sorted(SORT_KEYS)}")
    if training_trials.empty:
        return {}
    col = SORT_KEYS[by]
    g = training_trials.groupby("fly_id")[col].mean()
    return {str(k): float(v) for k, v in g.items()}


def fly_order(training_trials: pd.DataFrame, by: str = "binary") -> list[str]:
    """Fly ids, most responsive during conditioning first.

    Ties break on the fly id so the row order is reproducible across runs and
    across the training/testing figures. A NaN score sorts last rather than
    silently leading.
    """
    scores = fly_scores(training_trials, by=by)
    return sorted(
        scores,
        key=lambda f: (-(scores[f] if math.isfinite(scores[f]) else -math.inf), f),
    )


# --------------------------------------------------------------------------- #
# Raster assembly
# --------------------------------------------------------------------------- #


def raster(
    trials: pd.DataFrame,
    order: Sequence[str],
    *,
    pre_s: float,
    post_s: float,
    bin_s: float,
    column: str = "binary",
) -> tuple[np.ndarray, np.ndarray]:
    """Flies x time grid, aligned on odor onset. NaN = fly not tested.

    ``column`` is "binary" (0/1 against theta) or "trace" (the raw envelope
    percentage), so both figure variants share one alignment path.

    Aligning on odor onset rather than frame 0 keeps the odor rule in the same
    column for every row even though ``trial_odor_on_s`` drifts by tens of
    milliseconds between trials.
    """
    n_bins = max(1, int(round((pre_s + post_s) / bin_s)))
    time_axis = -pre_s + bin_s * (np.arange(n_bins) + 0.5)
    grid = np.full((len(order), n_bins), np.nan, dtype=float)

    by_fly = {str(f): g for f, g in trials.groupby("fly_id")} if not trials.empty else {}
    for r, fid in enumerate(order):
        g = by_fly.get(str(fid))
        if g is None or g.empty:
            continue
        row = g.iloc[0]
        series = np.asarray(row[column], dtype=float)
        fps = float(row["fps"])
        on = float(row["odor_on_s"])
        # Sample the trace at each bin centre, in seconds from odor onset.
        idx = np.round((on + time_axis) * fps).astype(int)
        valid = (idx >= 0) & (idx < series.size)
        vals = np.full(n_bins, np.nan)
        vals[valid] = series[idx[valid]]
        grid[r] = vals
    return grid, time_axis


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Mode:
    """What a raster cell encodes, and how it is coloured."""

    key: str
    column: str
    cmap: str
    vmin: float
    vmax: float
    cbar_label: str
    #: Colour for values below ``vmin``. The graded mode uses it to paint
    #: below-threshold cells, which carry no magnitude.
    under_color: str = ""

    def with_vmax(self, vmax: float) -> "Mode":
        return Mode(self.key, self.column, self.cmap, self.vmin, float(vmax),
                    self.cbar_label, self.under_color)


#: Two states, so a two-colour categorical map and a patch legend.
BINARY_MODE = Mode(
    key="binary", column="binary", cmap="", vmin=0.0, vmax=1.0,
    cbar_label="above θ",
)

#: Continuous magnitude. viridis, not blue->red: it runs cool at zero to warm at
#: full extension exactly as asked, but it is a *perceptually uniform sequential*
#: ramp, so equal colour steps are equal signal steps and it survives protanopia
#: and deuteranopia. A blue->white->red "coolwarm" would imply a meaningful
#: midpoint that this scale does not have (0 is the floor, not the centre), and
#: a rainbow would invent contrast where the data has none.
ENVELOPE_MODE = Mode(
    key="envelope", column="trace", cmap="viridis", vmin=0.0, vmax=100.0,
    cbar_label="Proboscis extension (max distance × angle, %)",
)

def graded_cmap() -> LinearSegmentedColormap:
    """Light -> dark single-hue green, ending on the project's pinned ON colour.

    Sequential data gets ONE hue running light to dark: equal colour steps read as
    equal signal steps, and a single hue survives every form of colour-vision
    deficiency because the ordering is carried by lightness, not by hue. A
    multi-hue or rainbow ramp would invent category boundaries where the data has
    a continuum. The dark end is ``ON_COLOR`` so a saturated cell here is exactly
    the green the plain binary raster paints, which lets the two be read together.
    """
    return LinearSegmentedColormap.from_list(
        # The light end is deliberately a clear green, not a near-white: it has
        # to be told apart from OFF_COLOR (the warm grey below theta) at a
        # glance, or "barely responded" reads as "did not respond".
        "per_graded", ["#c2e2cc", "#8ecba1", "#57ad74", "#2c8a51", ON_COLOR]
    )


#: Above threshold, shaded by how far above the *baseline* the proboscis is.
#:
#: The magnitude is measured from the baseline location, not from theta. Theta
#: sits k*sigma above the resting position by construction, and by a different
#: margin on every trial, so measuring the excess from it would shrink every
#: response by a trial-specific amount and make rows incomparable.
#:
#: ``vmin`` is a hair above zero so that exactly-zero cells -- everything below
#: theta -- fall to ``under_color`` and stay flat, keeping the "is it a response"
#: read of the binary raster while adding "how big".
GRADED_MODE = Mode(
    key="graded", column="graded", cmap="", vmin=1e-9, vmax=100.0,
    cbar_label="Extension above baseline, where above θ (%)",
    under_color=OFF_COLOR,
)

MODES = {m.key: m for m in (BINARY_MODE, ENVELOPE_MODE, GRADED_MODE)}


def robust_vmax(
    trials: pd.DataFrame, percentile: float = 99.0, column: str = "trace"
) -> float:
    """A percentile of every sample in ``trials`` -- contrast without a lie.

    The envelope is genuinely 0-100 %, but its median is ~9 and its 99th
    percentile ~65, so a linear 0-100 map spends two thirds of the ramp on
    values that never occur and washes the structure out.
    """
    if trials.empty:
        return float("nan")
    stacked = np.concatenate([np.asarray(x, dtype=float) for x in trials[column]])
    stacked = stacked[np.isfinite(stacked)]
    if stacked.size == 0:
        return float("nan")
    return float(np.nanpercentile(stacked, percentile))


@dataclass(frozen=True)
class Panel:
    """One column of a raster figure: a label and the trials it draws."""

    label: str
    trials: pd.DataFrame
    median_index: float
    odor: str = ""
    rank: int = 1


def testing_panels(
    testing_trials: pd.DataFrame,
    *,
    exclude_odors: Sequence[str] = DEFAULT_EXCLUDE_ODORS,
    dataset: Optional[str] = None,
) -> list[Panel]:
    """One panel per odor, splitting an odor that is presented more than once.

    Keyed on odor rather than trial index because testing trials 2-7 are
    randomised per fly: a "testing 3" column would show a different odor on
    every row. Presentation rank is computed within each fly, so "Hexanol 2" is
    each fly's own second hexanol, whatever index it landed on.

    Panels are grouped by odor and the *groups* ordered by their earliest median
    trial index, so an odor's repeated presentations stay side by side (Hexanol
    1 next to Hexanol 2) instead of being flung to opposite ends by the eight
    trials that separate them.
    """
    if testing_trials.empty:
        return []
    t = testing_trials.copy()
    drop = {str(o).strip().lower() for o in (exclude_odors or ())}
    if drop:
        t = t[~t["odor"].astype(str).str.strip().str.lower().isin(drop)]
    if t.empty:
        return []

    ds = dataset if dataset is not None else str(t["dataset"].iloc[0])
    t["_rank"] = (
        t.sort_values("trial_index").groupby(["fly_id", "odor"]).cumcount() + 1
    )
    repeated = {
        odor for odor, g in t.groupby("odor") if int(g["_rank"].max()) > 1
    }

    panels: list[Panel] = []
    for (odor, rank), g in t.groupby(["odor", "_rank"]):
        label = odor_display(odor, ds)
        if odor in repeated:
            label = f"{label} {int(rank)}"
        panels.append(
            Panel(label=label, trials=g.drop(columns="_rank"),
                  median_index=float(g["trial_index"].median()), odor=str(odor),
                  rank=int(rank))
        )
    # Group key: the odor's earliest panel. Sorting on that keeps every
    # presentation of one odor contiguous.
    first = {}
    for p in panels:
        first[p.odor] = min(first.get(p.odor, p.median_index), p.median_index)
    panels.sort(key=lambda p: (first[p.odor], p.odor, p.rank))
    return panels


def training_panels(training_trials: pd.DataFrame) -> list[Panel]:
    """One panel per conditioning trial, in order."""
    if training_trials.empty:
        return []
    panels = [
        Panel(label=f"Training {int(idx)}", trials=g, median_index=float(idx))
        for idx, g in training_trials.groupby("trial_index")
    ]
    panels.sort(key=lambda p: p.median_index)
    return panels


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #


def _cmap() -> ListedColormap:
    cm = ListedColormap([OFF_COLOR, ON_COLOR])
    cm.set_bad(color=MISSING_COLOR)
    return cm


def _row_labels(order: Sequence[str]) -> list[str]:
    out = []
    for fid in order:
        fly, _, num = str(fid).rpartition("#")
        out.append(f"{fly.replace('_rig_', ' r')}  f{num}")
    return out


def _draw_panel(ax, grid, time_axis, *, label, odor_span, first_col, order, scores,
                mode=BINARY_MODE):
    if mode.key == "binary":
        cmap, vmin, vmax = _cmap(), 0.0, 1.0
    elif mode.key == "graded":
        cmap = graded_cmap().copy()
        cmap.set_bad(color=MISSING_COLOR)
        # Everything below theta is exactly 0 and lands here, flat, so a real but
        # tiny response is never confused with no response at all.
        cmap.set_under(color=mode.under_color or OFF_COLOR)
        vmin, vmax = mode.vmin, mode.vmax
    else:
        cmap = matplotlib.colormaps[mode.cmap].copy()
        cmap.set_bad(color=MISSING_COLOR)
        vmin, vmax = mode.vmin, mode.vmax
    # Extent is the outer bin EDGES, not the centres time_axis carries -- passing
    # centres shifts the image half a bin against the odor rules.
    half = (time_axis[1] - time_axis[0]) / 2 if time_axis.size > 1 else 0.0
    im = ax.imshow(
        np.ma.masked_invalid(grid), cmap=cmap, vmin=vmin, vmax=vmax,
        aspect="auto", interpolation="nearest",
        extent=(time_axis[0] - half, time_axis[-1] + half, len(order) - 0.5, -0.5),
    )
    ax._raster_im = im
    # Odor window: two hairline rules, no shading -- a wash over a dense raster
    # hides the marks it is meant to frame. White on the sequential ramp, whose
    # low end is dark enough to swallow the blue.
    rule = ODOR_INK if mode.key in {"binary", "graded"} else ODOR_RULE_LIGHT
    for x in (0.0, odor_span):
        ax.axvline(x, color=rule, linewidth=1.0, alpha=0.9, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(_row_labels(order) if first_col else [""] * len(order))
    ax.tick_params(axis="y", length=0, labelcolor=INK_SECONDARY)
    ax.tick_params(axis="x", direction="out", length=3, colors=MUTED)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(label, fontsize=9.5, color=INK, weight="bold", pad=6)
    ax.set_xlabel("Time from odor on (s)")


def _figure(
    panels: Sequence[Panel],
    order: Sequence[str],
    *,
    dataset: str,
    title: str,
    subtitle: str,
    scores: Optional[dict[str, float]] = None,
    pre_s: float = 10.0,
    post_s: float = 40.0,
    bin_s: float = 0.1,
    mode: Mode = BINARY_MODE,
) -> tuple[plt.Figure, dict]:
    n = len(panels)
    if n == 0:
        raise ValueError("no panels to draw")
    odor_span = float(np.nanmedian([
        p.trials["odor_off_s"].median() - p.trials["odor_on_s"].median() for p in panels
    ]))

    # Fixed inch budgets for the chrome, so the layout holds whether there are
    # 3 panels or 9 and 4 flies or 40.
    left_in = 2.30                          # row labels live in the left margin
    right_in = 1.05 if mode.key != "binary" else 0.30   # colorbar gutter
    top_in, bottom_in = 1.35, 1.00          # title + subtitle / xlabel + legend
    panel_in = 1.75
    width = left_in + panel_in * n + right_in
    height = max(3.4, 0.22 * len(order) + top_in + bottom_in)
    meta_panels = []

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            1, n, figsize=(width, height), squeeze=False,
        )
        for i, panel in enumerate(panels):
            grid, time_axis = raster(
                panel.trials, order, pre_s=pre_s, post_s=post_s, bin_s=bin_s,
                column=mode.column,
            )
            _draw_panel(
                axes[0][i], grid, time_axis, label=panel.label,
                odor_span=odor_span, first_col=(i == 0), order=order, scores=scores,
                mode=mode,
            )
            with np.errstate(invalid="ignore"):
                on_frac = np.nanmean(
                    np.where(time_axis[None, :] >= 0, grid, np.nan)[:, time_axis >= 0]
                )
            meta_panels.append({
                "label": panel.label,
                "n_trials": int(len(panel.trials)),
                "n_flies": int(panel.trials["fly_id"].nunique()),
                "median_trial_index": panel.median_index,
                "mean_odor_fraction": (
                    float(panel.trials["odor_fraction"].mean())
                    if len(panel.trials) else float("nan")
                ),
                "mean_on_fraction_after_odor_onset": (
                    float(on_frac) if np.isfinite(on_frac) else None
                ),
            })

        if mode.key == "binary":
            handles = [
                plt.Rectangle((0, 0), 1, 1, facecolor=ON_COLOR, edgecolor="none",
                              label="above θ (proboscis extended)"),
                plt.Rectangle((0, 0), 1, 1, facecolor=OFF_COLOR, edgecolor=GRID,
                              label="below θ"),
                plt.Line2D([], [], color=ODOR_INK, linewidth=1.0,
                           label="odor on / off"),
            ]
            fig.legend(
                handles=handles, loc="lower center", ncol=3, frameon=False,
                bbox_to_anchor=(0.5, 0.10 / height), fontsize=8,
                labelcolor=INK_SECONDARY,
            )
        else:
            fig.legend(
                handles=[plt.Line2D([], [], color=ODOR_RULE_LIGHT, linewidth=1.2,
                                    label="odor on / off")],
                loc="lower center", ncol=1, frameon=False,
                bbox_to_anchor=(0.5, 0.10 / height), fontsize=8,
                labelcolor=INK_SECONDARY,
            )

        fig.suptitle(title, fontsize=13, weight="bold", color=INK,
                     y=1 - 0.28 / height)
        fig.text(0.5, 1 - 0.58 / height, subtitle, ha="center", va="top",
                 fontsize=8.5, color=MUTED, wrap=True)
        # Explicit margins rather than tight_layout: the panel titles and the
        # left-margin row labels are sized in inches above, and tight_layout
        # renegotiates them into the subtitle.
        fig.subplots_adjust(
            left=left_in / width, right=1 - right_in / width,
            top=1 - top_in / height, bottom=bottom_in / height,
            wspace=0.10,
        )

        if mode.key != "binary":
            # A dedicated axes, never `colorbar(ax=...)`: passing an axes steals
            # width from that one panel and breaks the alignment across the row.
            top = 1 - top_in / height
            bot = bottom_in / height
            cax = fig.add_axes([
                1 - (right_in - 0.28) / width, bot + 0.12 * (top - bot),
                0.16 / width, 0.76 * (top - bot),
            ])
            cbar = fig.colorbar(axes[0][-1]._raster_im, cax=cax)
            cbar.set_label(mode.cbar_label, color=INK_SECONDARY, fontsize=8)
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(length=2, colors=MUTED, labelsize=7.5)

    return fig, {
        "dataset": dataset,
        "mode": mode.key,
        "n_flies": len(order),
        "fly_order": list(order),
        "fly_scores": {k: scores[k] for k in order} if scores else None,
        "panels": meta_panels,
        "scale": {"vmin": mode.vmin, "vmax": mode.vmax, "cmap": mode.cmap or "binary"},
        "window": {"pre_s": pre_s, "post_s": post_s, "bin_s": bin_s,
                   "odor_span_s": odor_span},
    }


_SORT_BLURB = {
    "binary": "the mean fraction of the odor window spent above θ",
    "auc": "mean AUC-During",
}


def figure_training(
    training_trials: pd.DataFrame, order: Sequence[str], *, dataset: str,
    k: float = K_DEFAULT, mode: Mode = BINARY_MODE, sort_by: str = "binary", **kw,
) -> tuple[plt.Figure, dict]:
    scores = fly_scores(training_trials, by=sort_by)
    if mode.key == "binary":
        what = "Binarised PER"
        rule = (f"  θ = median$_{{before}}$ + {k:g}·MAD$_{{before}}$, per trial.")
    elif mode.key == "graded":
        what = "PER above baseline"
        rule = (
            "  Flat grey below θ; above θ, colour is how far above that trial's "
            f"own resting baseline the proboscis is (pale ≈ 0, solid ≥ {mode.vmax:.0f} %)."
        )
    else:
        what = "Proboscis extension"
        rule = "  Colour is the raw envelope, cool = resting, warm = fully extended."
    return _figure(
        training_panels(training_trials), order, dataset=dataset, scores=scores,
        mode=mode,
        title=f"{what} across conditioning — {dataset}",
        subtitle=(
            f"Each row is one fly ({len(order)} flies), sorted top-to-bottom by "
            f"{_SORT_BLURB.get(sort_by, sort_by)} across all training trials.{rule}"
        ),
        **kw,
    )


def figure_testing(
    testing_trials: pd.DataFrame, order: Sequence[str], *, dataset: str,
    k: float = K_DEFAULT, scores: Optional[dict[str, float]] = None,
    exclude_odors: Sequence[str] = DEFAULT_EXCLUDE_ODORS,
    mode: Mode = BINARY_MODE, sort_by: str = "binary", **kw,
) -> tuple[plt.Figure, dict]:
    dropped = sorted({
        odor_display(o, dataset)
        for o in testing_trials["odor"].unique()
        if str(o).strip().lower() in {str(x).strip().lower() for x in (exclude_odors or ())}
    }) if not testing_trials.empty else []
    note = f"  {', '.join(dropped)} not shown." if dropped else ""
    what = {
        "binary": "Binarised PER",
        "graded": "PER above baseline",
    }.get(mode.key, "Proboscis extension")
    return _figure(
        testing_panels(testing_trials, exclude_odors=exclude_odors, dataset=dataset),
        order, dataset=dataset, scores=scores, mode=mode,
        title=f"{what} at test, by odor — {dataset}",
        subtitle=(
            "Same fly order as the training figure — row N is the same fly in both, "
            f"sorted by {_SORT_BLURB.get(sort_by, sort_by)} during training.  "
            "Panels are keyed on the odor, not the trial index: testing trials 2–7 are "
            f"randomised per fly.{note}"
        ),
        **kw,
    )


def _with_fly_row_gaps(grid: np.ndarray, row_gap_px: int) -> tuple[np.ndarray, list[float]]:
    """Insert NaN rows after each fly so imshow renders true white row separators."""
    gap = max(0, int(row_gap_px))
    if grid.size == 0 or gap == 0:
        return grid, [float(i) for i in range(grid.shape[0])]

    out_rows = grid.shape[0] + gap * max(0, grid.shape[0] - 1)
    out = np.full((out_rows, grid.shape[1]), np.nan, dtype=float)
    centers: list[float] = []
    src = 0
    for dst in range(0, out_rows, gap + 1):
        out[dst] = grid[src]
        centers.append(float(dst))
        src += 1
        if src >= grid.shape[0]:
            break
    return out, centers


def _concat_panels(
    panels: Sequence[Panel],
    order: Sequence[str],
    *,
    pre_s: float,
    post_s: float,
    bin_s: float,
    mode: Mode,
    row_gap_px: int = 2,
) -> tuple[np.ndarray, list[float], list[dict]]:
    """One heatmap matrix with every panel placed left-to-right for each fly."""
    pieces = []
    segments = []
    x0 = 0
    odor_span = float(np.nanmedian([
        p.trials["odor_off_s"].median() - p.trials["odor_on_s"].median() for p in panels
    ]))
    for panel in panels:
        grid, time_axis = raster(
            panel.trials, order, pre_s=pre_s, post_s=post_s, bin_s=bin_s,
            column=mode.column,
        )
        pieces.append(grid)
        n = int(grid.shape[1])
        segments.append({
            "label": panel.label,
            "x0": x0,
            "x1": x0 + n,
            "center": x0 + (n - 1) / 2,
            "odor_on_x": x0 + pre_s / bin_s,
            "odor_off_x": x0 + (pre_s + odor_span) / bin_s,
            "n_trials": int(len(panel.trials)),
            "n_flies": int(panel.trials["fly_id"].nunique()),
        })
        x0 += n
    full = np.concatenate(pieces, axis=1) if pieces else np.empty((len(order), 0))
    return (*_with_fly_row_gaps(full, row_gap_px), segments)


def _plot_order_key(fig: plt.Figure, labels: Sequence[str], *, y: float) -> None:
    """Small external key listing the left-to-right segment order."""
    if not labels:
        return
    entries = [f"{i}. {label}" for i, label in enumerate(labels, start=1)]
    line = "Order: " + "  |  ".join(entries)
    fig.text(0.5, y, line, ha="center", va="bottom", fontsize=7.5,
             color=INK_SECONDARY, wrap=True)


def _continuous_heatmap_figure(
    panels: Sequence[Panel],
    order: Sequence[str],
    *,
    dataset: str,
    title: str,
    subtitle: str,
    mode: Mode = ENVELOPE_MODE,
    pre_s: float = 10.0,
    post_s: float = 40.0,
    bin_s: float = 0.1,
    row_gap_px: int = 2,
) -> tuple[plt.Figure, dict]:
    """Draw all panels left-to-right in one heatmap row per fly."""
    if not panels:
        raise ValueError("no panels to draw")
    if mode.key == "binary":
        raise ValueError("continuous heatmap figure expects a continuous mode")

    grid, y_centers, segments = _concat_panels(
        panels, order, pre_s=pre_s, post_s=post_s, bin_s=bin_s,
        mode=mode, row_gap_px=row_gap_px,
    )
    cmap = matplotlib.colormaps[mode.cmap].copy()
    cmap.set_bad(color=MISSING_COLOR)

    left_in = 2.40
    right_in = 1.10
    top_in = 1.45
    bottom_in = 1.25
    width = max(8.0, left_in + 1.05 * len(panels) + right_in)
    height = max(3.8, 0.18 * grid.shape[0] + top_in + bottom_in)

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(1, 1, figsize=(width, height))
        im = ax.imshow(
            np.ma.masked_invalid(grid), cmap=cmap, vmin=mode.vmin, vmax=mode.vmax,
            aspect="auto", interpolation="nearest",
            extent=(-0.5, grid.shape[1] - 0.5, grid.shape[0] - 0.5, -0.5),
        )
        rule = ODOR_RULE_LIGHT
        for seg in segments:
            ax.axvline(seg["x0"] - 0.5, color=GRID, linewidth=0.9, zorder=3)
            ax.axvline(seg["odor_on_x"] - 0.5, color=rule, linewidth=0.9, alpha=0.9, zorder=4)
            ax.axvline(seg["odor_off_x"] - 0.5, color=rule, linewidth=0.9, alpha=0.9, zorder=4)
        ax.axvline(segments[-1]["x1"] - 0.5, color=GRID, linewidth=0.9, zorder=3)

        ax.set_xlim(-0.5, grid.shape[1] - 0.5)
        ax.set_ylim(grid.shape[0] - 0.5, -0.5)
        ax.set_yticks(y_centers)
        ax.set_yticklabels(_row_labels(order))
        ax.tick_params(axis="y", length=0, labelcolor=INK_SECONDARY)
        ax.set_xticks([s["center"] for s in segments])
        ax.set_xticklabels([s["label"] for s in segments], rotation=35, ha="right")
        ax.tick_params(axis="x", direction="out", length=0, colors=INK_SECONDARY,
                       labelsize=8, pad=4)
        ax.xaxis.tick_top()
        ax.set_xlabel("")
        ax.set_ylabel("")
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)

        fig.subplots_adjust(
            left=left_in / width, right=1 - right_in / width,
            top=1 - top_in / height, bottom=bottom_in / height,
        )
        fig.suptitle(title, fontsize=13, weight="bold", color=INK,
                     y=1 - 0.25 / height)
        fig.text(0.5, 1 - 0.58 / height, subtitle, ha="center", va="top",
                 fontsize=8.5, color=MUTED, wrap=True)
        _plot_order_key(fig, [s["label"] for s in segments], y=0.20 / height)

        cax = fig.add_axes([
            1 - (right_in - 0.28) / width, bottom_in / height + 0.10,
            0.16 / width, 1 - (top_in + bottom_in) / height - 0.12,
        ])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(mode.cbar_label, color=INK_SECONDARY, fontsize=8)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(length=2, colors=MUTED, labelsize=7.5)
        fig.legend(
            handles=[
                plt.Line2D([], [], color=ODOR_RULE_LIGHT, linewidth=1.2,
                           label="odor on / off"),
                plt.Line2D([], [], color=GRID, linewidth=1.0,
                           label="trial / odor boundary"),
            ],
            loc="lower center", ncol=2, frameon=False,
            bbox_to_anchor=(0.5, 0.55 / height), fontsize=8,
            labelcolor=INK_SECONDARY,
        )

    return fig, {
        "dataset": dataset,
        "mode": f"{mode.key}_continuous",
        "n_flies": len(order),
        "fly_order": list(order),
        "segments": segments,
        "row_gap_px": int(row_gap_px),
        "scale": {"vmin": mode.vmin, "vmax": mode.vmax, "cmap": mode.cmap},
        "window": {"pre_s": pre_s, "post_s": post_s, "bin_s": bin_s},
    }


def figure_training_continuous_heatmap(
    training_trials: pd.DataFrame, order: Sequence[str], *, dataset: str,
    mode: Mode = ENVELOPE_MODE, sort_by: str = "auc", row_gap_px: int = 2, **kw,
) -> tuple[plt.Figure, dict]:
    return _continuous_heatmap_figure(
        training_panels(training_trials), order, dataset=dataset, mode=mode,
        row_gap_px=row_gap_px,
        title=f"Training proboscis extension timeline — {dataset}",
        subtitle=(
            f"Each row is one fly, sorted top-to-bottom by "
            f"{_SORT_BLURB.get(sort_by, sort_by)}. Training trials are concatenated "
            "left-to-right with odor on/off rules repeated in every segment."
        ),
        **kw,
    )


def figure_testing_continuous_heatmap(
    testing_trials: pd.DataFrame, order: Sequence[str], *, dataset: str,
    exclude_odors: Sequence[str] = DEFAULT_EXCLUDE_ODORS,
    mode: Mode = ENVELOPE_MODE, sort_by: str = "auc", row_gap_px: int = 2, **kw,
) -> tuple[plt.Figure, dict]:
    dropped = sorted({
        odor_display(o, dataset)
        for o in testing_trials["odor"].unique()
        if str(o).strip().lower() in {str(x).strip().lower() for x in (exclude_odors or ())}
    }) if not testing_trials.empty else []
    note = f"  {', '.join(dropped)} not shown." if dropped else ""
    return _continuous_heatmap_figure(
        testing_panels(testing_trials, exclude_odors=exclude_odors, dataset=dataset),
        order, dataset=dataset, mode=mode, row_gap_px=row_gap_px,
        title=f"Testing proboscis extension timeline by odor — {dataset}",
        subtitle=(
            f"Each row is one fly, sorted by {_SORT_BLURB.get(sort_by, sort_by)} "
            "during training. Testing odors are placed in the same fixed left-to-right "
            f"order for every fly; repeated odors are split by presentation.{note}"
        ),
        **kw,
    )


def _stack_panels_by_fly(
    panels: Sequence[Panel],
    order: Sequence[str],
    *,
    pre_s: float,
    post_s: float,
    bin_s: float,
    mode: Mode,
    row_gap_px: int = 2,
) -> tuple[np.ndarray, np.ndarray, list[float], list[float], list[dict]]:
    """Rows are fly groups; within each fly, panel 1, panel 2, ... are stacked."""
    if not panels:
        raise ValueError("no panels to stack")
    panel_grids = []
    time_axis = None
    for panel in panels:
        grid, t = raster(
            panel.trials, order, pre_s=pre_s, post_s=post_s, bin_s=bin_s,
            column=mode.column,
        )
        panel_grids.append(grid)
        if time_axis is None:
            time_axis = t

    gap = max(0, int(row_gap_px))
    n_panels = len(panels)
    n_flies = len(order)
    n_bins = panel_grids[0].shape[1]
    out_rows = n_flies * n_panels + gap * max(0, n_flies - 1)
    out = np.full((out_rows, n_bins), np.nan, dtype=float)
    fly_centers: list[float] = []
    panel_centers: list[float] = []
    row_meta: list[dict] = []

    dst = 0
    for fly_i, fid in enumerate(order):
        start = dst
        for panel_i, panel in enumerate(panels):
            out[dst] = panel_grids[panel_i][fly_i]
            panel_centers.append(float(dst))
            row_meta.append({
                "fly_id": str(fid),
                "panel": panel.label,
                "panel_index": panel_i + 1,
                "row": int(dst),
            })
            dst += 1
        fly_centers.append(start + (n_panels - 1) / 2)
        if fly_i != n_flies - 1:
            dst += gap

    assert time_axis is not None
    return out, time_axis, fly_centers, panel_centers, row_meta


def _stacked_heatmap_figure(
    panels: Sequence[Panel],
    order: Sequence[str],
    *,
    dataset: str,
    title: str,
    subtitle: str,
    mode: Mode = ENVELOPE_MODE,
    pre_s: float = 10.0,
    post_s: float = 40.0,
    bin_s: float = 0.1,
    row_gap_px: int = 2,
) -> tuple[plt.Figure, dict]:
    """Draw one time-aligned row per fly x panel, with gaps between flies."""
    if mode.key == "binary":
        raise ValueError("stacked heatmap figure expects a continuous mode")
    grid, time_axis, fly_centers, panel_centers, row_meta = _stack_panels_by_fly(
        panels, order, pre_s=pre_s, post_s=post_s, bin_s=bin_s,
        mode=mode, row_gap_px=row_gap_px,
    )
    odor_span = float(np.nanmedian([
        p.trials["odor_off_s"].median() - p.trials["odor_on_s"].median() for p in panels
    ]))
    cmap = matplotlib.colormaps[mode.cmap].copy()
    cmap.set_bad(color=MISSING_COLOR)

    left_in = 2.35
    right_in = 1.25
    top_in = 1.35
    bottom_in = 1.25
    width = 8.2
    height = max(4.5, 0.105 * grid.shape[0] + top_in + bottom_in)

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(1, 1, figsize=(width, height))
        half = (time_axis[1] - time_axis[0]) / 2 if time_axis.size > 1 else 0.0
        im = ax.imshow(
            np.ma.masked_invalid(grid), cmap=cmap, vmin=mode.vmin, vmax=mode.vmax,
            aspect="auto", interpolation="nearest",
            extent=(time_axis[0] - half, time_axis[-1] + half,
                    grid.shape[0] - 0.5, -0.5),
        )
        for x in (0.0, odor_span):
            ax.axvline(x, color=ODOR_RULE_LIGHT, linewidth=1.0, alpha=0.95, zorder=4)

        ax.set_yticks(fly_centers)
        ax.set_yticklabels(_row_labels(order))
        ax.tick_params(axis="y", length=0, labelcolor=INK_SECONDARY)
        ax.tick_params(axis="x", direction="out", length=3, colors=MUTED)
        ax.set_xlabel("Time from odor on (s)")
        ax.set_ylabel("")
        ax.set_title("")
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)

        if len(panels) <= 9 and grid.shape[0] <= 180:
            ax_r = ax.twinx()
            ax_r.set_ylim(ax.get_ylim())
            ax_r.set_yticks(panel_centers)
            ax_r.set_yticklabels([
                str((i % len(panels)) + 1) for i in range(len(panel_centers))
            ])
            ax_r.tick_params(axis="y", length=0, colors=MUTED, labelsize=6.5)
            for side in ("top", "right", "left", "bottom"):
                ax_r.spines[side].set_visible(False)

        fig.subplots_adjust(
            left=left_in / width, right=1 - right_in / width,
            top=1 - top_in / height, bottom=bottom_in / height,
        )
        fig.suptitle(title, fontsize=13, weight="bold", color=INK,
                     y=1 - 0.25 / height)
        fig.text(0.5, 1 - 0.58 / height, subtitle, ha="center", va="top",
                 fontsize=8.5, color=MUTED, wrap=True)
        _plot_order_key(fig, [p.label for p in panels], y=0.20 / height)

        cax = fig.add_axes([
            1 - (right_in - 0.30) / width, bottom_in / height + 0.10,
            0.16 / width, 1 - (top_in + bottom_in) / height - 0.12,
        ])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(mode.cbar_label, color=INK_SECONDARY, fontsize=8)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(length=2, colors=MUTED, labelsize=7.5)
        fig.legend(
            handles=[plt.Line2D([], [], color=ODOR_RULE_LIGHT, linewidth=1.2,
                                label="odor on / off")],
            loc="lower center", ncol=1, frameon=False,
            bbox_to_anchor=(0.5, 0.55 / height), fontsize=8,
            labelcolor=INK_SECONDARY,
        )

    return fig, {
        "dataset": dataset,
        "mode": f"{mode.key}_stacked_by_fly",
        "n_flies": len(order),
        "n_panels_per_fly": len(panels),
        "fly_order": list(order),
        "panel_order": [p.label for p in panels],
        "rows": row_meta,
        "row_gap_px": int(row_gap_px),
        "scale": {"vmin": mode.vmin, "vmax": mode.vmax, "cmap": mode.cmap},
        "window": {"pre_s": pre_s, "post_s": post_s, "bin_s": bin_s,
                   "odor_span_s": odor_span},
    }


def figure_training_stacked_heatmap(
    training_trials: pd.DataFrame, order: Sequence[str], *, dataset: str,
    mode: Mode = ENVELOPE_MODE, sort_by: str = "auc", row_gap_px: int = 2, **kw,
) -> tuple[plt.Figure, dict]:
    return _stacked_heatmap_figure(
        training_panels(training_trials), order, dataset=dataset, mode=mode,
        row_gap_px=row_gap_px,
        title=f"Training proboscis extension stacked by fly — {dataset}",
        subtitle=(
            f"Flies are sorted by {_SORT_BLURB.get(sort_by, sort_by)}. Within each "
            "fly, rows are training trials 1-6 from top to bottom, then a white gap "
            "before the next fly."
        ),
        **kw,
    )


def figure_testing_stacked_heatmap(
    testing_trials: pd.DataFrame, order: Sequence[str], *, dataset: str,
    exclude_odors: Sequence[str] = DEFAULT_EXCLUDE_ODORS,
    mode: Mode = ENVELOPE_MODE, sort_by: str = "auc", row_gap_px: int = 2, **kw,
) -> tuple[plt.Figure, dict]:
    dropped = sorted({
        odor_display(o, dataset)
        for o in testing_trials["odor"].unique()
        if str(o).strip().lower() in {str(x).strip().lower() for x in (exclude_odors or ())}
    }) if not testing_trials.empty else []
    note = f"  {', '.join(dropped)} not shown." if dropped else ""
    return _stacked_heatmap_figure(
        testing_panels(testing_trials, exclude_odors=exclude_odors, dataset=dataset),
        order, dataset=dataset, mode=mode, row_gap_px=row_gap_px,
        title=f"Testing proboscis extension stacked by fly — {dataset}",
        subtitle=(
            f"Flies are sorted by {_SORT_BLURB.get(sort_by, sort_by)} during training. "
            "Within each fly, rows follow the fixed testing odor order shown in the key, "
            f"then a white gap before the next fly.{note}"
        ),
        **kw,
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def kept_flies(predictions_csv: Path, dataset: str) -> set[tuple[str, int]]:
    """The flies the behavioural figures keep, straight from the predictions CSV.

    ``model_predictions.csv`` already has flagged flies and frozen folders
    filtered out, so membership in it *is* the "flies we keep" rule -- no second
    copy of the exclusion logic to drift out of sync.
    """
    p = pd.read_csv(predictions_csv)
    want = str(dataset).strip().casefold()
    p = p[p["dataset"].astype(str).str.strip().str.casefold() == want]
    return {
        (str(f).strip(), int(n))
        for f, n in zip(p["fly"], p["fly_number"])
    }


def _save(fig: plt.Figure, out_dir: Path, stem: str, *, svg: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    LOGGER.info("Saved %s", png)
    if svg:
        s = out_dir / f"{stem}.svg"
        fig.savefig(s, bbox_inches="tight", facecolor=SURFACE)
        LOGGER.info("Saved %s", s)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--training-wide-csv", type=Path, required=True)
    p.add_argument("--wide-csv", type=Path, required=True,
                   help="Full wide table (testing trials).")
    p.add_argument("--predictions-csv", type=Path, default=None,
                   help="Restrict to the flies the behavioural figures keep.")
    p.add_argument("--dataset", action="append", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("-k", "--threshold-k", type=float, default=K_DEFAULT,
                   help="θ = median_before + k·MAD_before (default 2, matching the "
                        "red line in Raw-Training-PER-Traces).")
    p.add_argument("--sort-by", choices=tuple(SORT_KEYS), default="binary",
                   help="Row ordering for the binary raster, computed on the "
                        "training trials and reused for testing. 'binary' is the "
                        "mean fraction of the odor window above θ; 'auc' is mean "
                        "AUC-During, which is what the heatmap uses and is "
                        "independent of θ -- so the row order stops moving when the "
                        "threshold rule changes.")
    p.add_argument("--threshold-min-delta", type=float, default=0.0,
                   help="Floor on how far above the baseline θ may sit, in units of "
                        "that fly's own full extension range (dir_val is normalised "
                        "per fly). Guards the flat-baseline case where the MAD "
                        "collapses. Calibrated optimum 5. 0 disables.")
    p.add_argument("--threshold-anchor-s", type=float, default=None,
                   help="Anchor θ on the median of the last N seconds before odor "
                        "onset instead of the whole baseline, with the noise scale a "
                        "low percentile of rolling σ. Guards bursts and held steps "
                        "inside the baseline. Omit for the shipped estimate.")
    p.add_argument("--stem-suffix", type=str, default="",
                   help="Appended to every output filename stem, so two threshold "
                        "rules can be written side by side without overwriting.")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "config_new.yaml",
                   help="Config supplying per-dataset odor_remap (e.g. this cohort's "
                        "ACV channel actually delivered isoamyl acetate).")
    p.add_argument("--exclude-odor", action="append", default=None,
                   help=f"Odor token to drop from the testing figure. Repeatable. "
                        f"Default: {', '.join(DEFAULT_EXCLUDE_ODORS)}. "
                        f"Pass --exclude-odor '' to keep everything.")
    p.add_argument("--pre-s", type=float, default=10.0)
    p.add_argument("--post-s", type=float, default=40.0)
    p.add_argument("--bin-s", type=float, default=0.1)
    p.add_argument("--mode", choices=("binary", "heatmap", "graded", "both"),
                   default="binary",
                   help="Figure variant to write. 'heatmap' uses raw envelope "
                        "values; 'graded' keeps the binary above/below-θ read but "
                        "shades above-θ cells by how far above BASELINE they sit, "
                        "light green at the threshold to dark green at --graded-vmax; "
                        "'both' writes binary plus heatmap.")
    p.add_argument("--graded-vmax", default="p99",
                   help="Upper colour limit for --mode graded: a number in percent "
                        "above baseline, or 'p99' (default) for the 99th percentile "
                        "of the graded values actually present.")
    p.add_argument("--heatmap-vmax", type=float, default=100.0,
                   help="Upper color limit for --mode heatmap/both (default 100, "
                        "the full percent-extension scale).")
    p.add_argument("--fly-gap-px", type=int, default=2,
                   help="Blank pixel rows between flies in the concatenated heatmaps "
                        "(default 2).")
    p.add_argument("--no-svg", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    exclude = (
        DEFAULT_EXCLUDE_ODORS if args.exclude_odor is None
        else tuple(o for o in args.exclude_odor if str(o).strip())
    )
    if args.config and Path(args.config).is_file():
        load_config_remap(args.config)
    else:
        LOGGER.warning("No config at %s — odor labels stay unremapped", args.config)

    from fbpipe.analysis.traces import read_wide_table

    training_wide = read_wide_table(args.training_wide_csv)
    testing_wide = read_wide_table(args.wide_csv)
    win = {"pre_s": args.pre_s, "post_s": args.post_s, "bin_s": args.bin_s}

    for dataset in args.dataset:
        keep = (
            kept_flies(args.predictions_csv, dataset) if args.predictions_csv else None
        )
        train = build_trials(
            training_wide, dataset=dataset, trial_type="training",
            keep=keep, k=args.threshold_k,
            min_delta=args.threshold_min_delta, anchor_s=args.threshold_anchor_s,
        )
        if train.empty:
            LOGGER.warning("No training trials for %s — skipping", dataset)
            continue
        test = build_trials(
            testing_wide, dataset=dataset, trial_type="testing",
            keep=keep, k=args.threshold_k,
            min_delta=args.threshold_min_delta, anchor_s=args.threshold_anchor_s,
        )

        ds_dir = args.out_dir / dataset.replace("/", "_")
        stem = dataset.replace("/", "_") + str(args.stem_suffix)
        svg = not args.no_svg

        metadata: dict[str, object] = {
            "mode": args.mode,
            "sort_by": args.sort_by,
            "threshold_k": args.threshold_k,
            "threshold_min_delta": args.threshold_min_delta,
            "threshold_anchor_s": args.threshold_anchor_s,
            "threshold": "median_before + k * 1.4826 * MAD_before (upward only)",
        }

        if args.mode in {"binary", "both"}:
            order = fly_order(train, by=args.sort_by)
            scores = fly_scores(train, by=args.sort_by)
            LOGGER.info(
                "%s binary: %d flies, %d training / %d testing trials; top fly %s "
                "(%.2f), bottom %s (%.2f)",
                dataset, len(order), len(train), len(test),
                order[0], scores[order[0]], order[-1], scores[order[-1]],
            )
            fig, meta_train = figure_training(
                train, order, dataset=dataset, k=args.threshold_k,
                mode=BINARY_MODE, sort_by=args.sort_by, **win
            )
            _save(fig, ds_dir, f"{stem}_training_raster", svg=svg)

            meta_test = None
            if not test.empty:
                fig, meta_test = figure_testing(
                    test, order, dataset=dataset, k=args.threshold_k, scores=scores,
                    exclude_odors=exclude, mode=BINARY_MODE, sort_by=args.sort_by, **win
                )
                _save(fig, ds_dir, f"{stem}_testing_raster", svg=svg)
            else:
                LOGGER.warning("No testing trials for %s", dataset)

            rank = pd.DataFrame({
                "rank": range(1, len(order) + 1),
                "fly_id": order,
                "mean_training_odor_fraction": [scores[f] for f in order],
            })
            rank.to_csv(ds_dir / f"{stem}_fly_order.csv", index=False)
            # Backward-compatible keys for the original binary-only workflow.
            metadata["training"] = meta_train
            metadata["testing"] = meta_test
            metadata["binary"] = {"training": meta_train, "testing": meta_test}

        if args.mode == "graded":
            order = fly_order(train, by=args.sort_by)
            scores = fly_scores(train, by=args.sort_by)
            if str(args.graded_vmax).strip().lower().startswith("p"):
                pct = float(str(args.graded_vmax).strip()[1:] or 99.0)
                vmax = robust_vmax(train, percentile=pct, column="graded")
            else:
                vmax = float(args.graded_vmax)
            if not math.isfinite(vmax) or vmax <= 0:
                vmax = 100.0
            gmode = GRADED_MODE.with_vmax(vmax)
            LOGGER.info("%s graded: %d flies, colour scale 0-%.1f %% above baseline",
                        dataset, len(order), vmax)

            fig, meta_train = figure_training(
                train, order, dataset=dataset, k=args.threshold_k,
                mode=gmode, sort_by=args.sort_by, **win
            )
            _save(fig, ds_dir, f"{stem}_training_graded", svg=svg)

            meta_test = None
            if not test.empty:
                fig, meta_test = figure_testing(
                    test, order, dataset=dataset, k=args.threshold_k, scores=scores,
                    exclude_odors=exclude, mode=gmode, sort_by=args.sort_by, **win
                )
                _save(fig, ds_dir, f"{stem}_testing_graded", svg=svg)

            pd.DataFrame({
                "rank": range(1, len(order) + 1),
                "fly_id": order,
                "sort_score": [scores[f] for f in order],
            }).to_csv(ds_dir / f"{stem}_fly_order.csv", index=False)
            metadata["graded"] = {
                "training": meta_train, "testing": meta_test, "vmax": float(vmax),
            }

        if args.mode in {"heatmap", "both"}:
            heatmap_mode = ENVELOPE_MODE.with_vmax(args.heatmap_vmax)
            order = fly_order(train, by="auc")
            scores = fly_scores(train, by="auc")
            LOGGER.info(
                "%s heatmap: %d flies, %d training / %d testing trials; top fly %s "
                "(mean AUC %.2f), bottom %s (%.2f)",
                dataset, len(order), len(train), len(test),
                order[0], scores[order[0]], order[-1], scores[order[-1]],
            )
            fig, meta_train = figure_training(
                train, order, dataset=dataset, k=args.threshold_k,
                mode=heatmap_mode, sort_by="auc", **win
            )
            _save(fig, ds_dir, f"{stem}_training_heatmap", svg=svg)
            fig, meta_train_timeline = figure_training_continuous_heatmap(
                train, order, dataset=dataset, mode=heatmap_mode,
                sort_by="auc", row_gap_px=args.fly_gap_px, **win
            )
            _save(fig, ds_dir, f"{stem}_training_heatmap_timeline", svg=svg)
            fig, meta_train_stack = figure_training_stacked_heatmap(
                train, order, dataset=dataset, mode=heatmap_mode,
                sort_by="auc", row_gap_px=args.fly_gap_px, **win
            )
            _save(fig, ds_dir, f"{stem}_training_heatmap_stack", svg=svg)

            meta_test = None
            meta_test_timeline = None
            meta_test_stack = None
            if not test.empty:
                fig, meta_test = figure_testing(
                    test, order, dataset=dataset, k=args.threshold_k, scores=scores,
                    exclude_odors=exclude, mode=heatmap_mode, sort_by="auc", **win
                )
                _save(fig, ds_dir, f"{stem}_testing_heatmap", svg=svg)
                fig, meta_test_timeline = figure_testing_continuous_heatmap(
                    test, order, dataset=dataset, exclude_odors=exclude,
                    mode=heatmap_mode, sort_by="auc", row_gap_px=args.fly_gap_px,
                    **win,
                )
                _save(fig, ds_dir, f"{stem}_testing_heatmap_timeline", svg=svg)
                fig, meta_test_stack = figure_testing_stacked_heatmap(
                    test, order, dataset=dataset, exclude_odors=exclude,
                    mode=heatmap_mode, sort_by="auc", row_gap_px=args.fly_gap_px,
                    **win,
                )
                _save(fig, ds_dir, f"{stem}_testing_heatmap_stack", svg=svg)
            else:
                LOGGER.warning("No testing trials for %s", dataset)

            rank = pd.DataFrame({
                "rank": range(1, len(order) + 1),
                "fly_id": order,
                "mean_training_auc_during": [scores[f] for f in order],
            })
            rank.to_csv(ds_dir / f"{stem}_fly_order_by_training_auc.csv", index=False)
            if args.mode == "heatmap":
                metadata["training"] = meta_train
                metadata["testing"] = meta_test
            metadata["heatmap"] = {
                "training": meta_train,
                "testing": meta_test,
                "training_timeline": meta_train_timeline,
                "testing_timeline": meta_test_timeline,
                "training_stack": meta_train_stack,
                "testing_stack": meta_test_stack,
            }

        # Per-trial table, so the ordering and every 0/1 count is auditable.
        cols = [c for c in train.columns if c != "binary"]
        pd.concat([train[cols], test[cols]] if not test.empty else [train[cols]]) \
            .to_csv(ds_dir / f"{stem}_binarized_trials.csv", index=False)

        (ds_dir / f"{stem}_rasters.json").write_text(
            json.dumps(metadata, indent=2, default=float),
            encoding="utf-8",
        )
        LOGGER.info("Saved %s", ds_dir / f"{stem}_rasters.json")


if __name__ == "__main__":
    main()
