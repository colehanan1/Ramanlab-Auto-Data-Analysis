"""Mounting-posture, proboscis-state and abdomen-movement metrics -- pure maths.

NumPy / pandas / SciPy only: no file I/O and no pipeline imports, so the table
builder and the figure scripts share one definition of every quantity and a test
can pin each one. The same module owns the ``mounting:`` settings block
(:class:`MountingSettings`), so every threshold has one default and one parser.

Conventions
-----------
* Image coordinates: x to the right, y DOWN, in pixels. ``eu`` ("eye units") is
  pixels divided by the trial's ``eye_scale_px``.
* Angles are in degrees. The pitches take the ABSOLUTE x difference, so a rig that
  mirrors the image reads the same value.
* Head frame: *forward* is away from the anchor (the anchor is always on the pin
  side). theta is 0 straight down, + forward, - back toward the anchor.
* NaN means "not measurable" and is never replaced by a guess: a missing frame
  breaks a run, drops a pair, or is left out of a fraction.
* Rates, medians and fractions are pooled by the caller; nothing here averages an
  average. The permutation tests shuffle only within their stratum (or fly), so a
  stratum-level (fly-level) confound cannot produce a small p on its own; the
  stratified risk difference is the effect size that goes with them. The paired
  window test re-draws only how a trial is split between its two windows, so what
  makes a trial busy stays with it.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, fields
from datetime import date, datetime
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist

_PERM_TOL = 1e-12          # slack (x the outcome scale) on "permuted diff >= observed diff": subset sums round differently
_CHUNK_CELLS = 2_000_000   # permuted values held in memory at once (~16 MB of float64)
_SWAP_LENGTH_RATIO = 1.01  # the window swap needs the two windows of a trial within 1 % of each other's length
_PRED_KEY = ["dataset", "fly", "fly_number", "trial_label"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _f64(a) -> np.ndarray:
    """*a* as a float64 array; None / pd.NA / NaN all become NaN."""
    if isinstance(a, (pd.Series, pd.Index)):
        return a.to_numpy(dtype="float64", na_value=np.nan)
    return np.asarray(a, dtype="float64")


def _present(value) -> Optional[float]:
    """*value* as a float, or None when it is absent (None, NaN, inf, not a number)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truthy(value) -> bool:
    """True/1 -> True; False/0/None/NaN -> False (``bool(nan)`` is True, which would lie)."""
    number = _present(value)
    return number is not None and number != 0.0


def _finite_pairs(x, y) -> tuple[np.ndarray, np.ndarray]:
    """The pairs of *x* and *y* where both are finite, as two float arrays."""
    a, b = _f64(x), _f64(y)
    if a.shape != b.shape:
        raise ValueError(f"x and y must have the same length, got {a.shape} and {b.shape}")
    keep = np.isfinite(a) & np.isfinite(b)
    return a[keep], b[keep]


# ---------------------------------------------------------------------------
# Per-frame geometry
# ---------------------------------------------------------------------------

def head_frame_vector(eye_x, eye_y, prob_x, prob_y, anchor_x, *, frame_mid: float = 540.0):
    """Eye -> proboscis in the head frame: ``(hypot(dx, dy), degrees(atan2(forward * dx, dy)))``.

    ``dx, dy = proboscis - eye``; ``forward`` is +1 if ``anchor_x < frame_mid`` else -1 (away from
    the anchor), so theta is 0 straight down and + forward. An unknown anchor leaves theta NaN
    rather than guessing which way is forward.
    """
    dx = _f64(prob_x) - _f64(eye_x)
    dy = _f64(prob_y) - _f64(eye_y)
    anchor = _f64(anchor_x)
    forward = np.where(anchor < frame_mid, 1.0, -1.0)
    forward = np.where(np.isfinite(anchor), forward, np.nan)
    return np.hypot(dx, dy), np.degrees(np.arctan2(forward * dx, dy))


def body_pitch_deg(eye_x, eye_y, abd_x, abd_y) -> np.ndarray:
    """``degrees(atan2(abd_y - eye_y, |abd_x - eye_x|))``: + = abdomen below the eye (head up), - = head down."""
    return np.degrees(np.arctan2(_f64(abd_y) - _f64(eye_y), np.abs(_f64(abd_x) - _f64(eye_x))))


def tip_pitch_deg(abd_x, abd_y, tip_x, tip_y) -> np.ndarray:
    """``degrees(atan2(tip_y - abd_y, |tip_x - abd_x|))``: + = tip below the abdomen centre.

    Mirror-invariant like :func:`body_pitch_deg`, so the two are comparable.
    """
    return np.degrees(np.arctan2(_f64(tip_y) - _f64(abd_y), np.abs(_f64(tip_x) - _f64(abd_x))))


def angle_between_deg(u, v) -> float:
    """Unsigned angle between two 2-D vectors, ``degrees(atan2(|u x v|, u . v))``; NaN if either is zero or missing."""
    a, b = _f64(u).reshape(2), _f64(v).reshape(2)
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or not (a.any() and b.any()):
        return float("nan")
    return float(np.degrees(np.arctan2(abs(a[0] * b[1] - a[1] * b[0]), a @ b)))


# ---------------------------------------------------------------------------
# Stimulus window
# ---------------------------------------------------------------------------

def stimulus_frames(trial: Mapping, *, fps: float) -> tuple[Optional[int], Optional[int]]:
    """``(onset, offset)`` in frames: the earliest start and the latest end of the odor and the light.

    Odor frames count only when ``trial["has_odor"]`` is truthy (a LightOnly trial carries a filler
    ``odor_on_frame`` that must never be used); light times are ``round(seconds * fps)``. Anything
    absent or NaN is skipped, and either value is None when nothing is known.
    """
    starts: list[int] = []
    ends: list[int] = []
    if _truthy(trial.get("has_odor")):
        for key, bucket in (("odor_on_frame", starts), ("odor_off_frame", ends)):
            frame = _present(trial.get(key))
            if frame is not None:
                bucket.append(int(round(frame)))
    for key, bucket in (("light_on_s", starts), ("light_off_s", ends)):
        seconds = _present(trial.get(key))
        if seconds is not None:
            bucket.append(int(round(seconds * fps)))
    return (min(starts) if starts else None, max(ends) if ends else None)


def window_slice(n_frames: int, onset: Optional[int], offset: Optional[int]) -> tuple[slice, slice]:
    """``(pre, stim)`` = ``slice(0, onset)`` and ``slice(onset, offset)``.

    The stimulus window runs to the last frame when the end is unknown, and both windows are
    empty when the onset is.
    """
    if onset is None:
        return slice(0, 0), slice(0, 0)
    return slice(0, onset), slice(onset, n_frames if offset is None else offset)


# ---------------------------------------------------------------------------
# Proboscis rest, held out, retracted reference
# ---------------------------------------------------------------------------

def rest_level(x, *, q: float = 0.10, min_finite: int = 40) -> float:
    """The ``q`` quantile of the finite frames (a trial's rest level); NaN with fewer than ``min_finite`` of them."""
    finite = _f64(x)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, q)) if finite.size >= max(1, min_finite) else float("nan")


def held_out(dist_eu_pre, *, ref_eu: float, margin_eu: float, frac_min: float) -> bool:
    """True when >= ``frac_min`` of the FINITE pre-stimulus frames have ``dist >= ref_eu + margin_eu``.

    False when there is no finite frame (or no usable reference).
    """
    dist = _f64(dist_eu_pre)
    dist = dist[np.isfinite(dist)]
    cutoff = ref_eu + margin_eu
    if dist.size == 0 or not math.isfinite(cutoff):
        return False
    return bool(np.count_nonzero(dist >= cutoff) / dist.size >= frac_min)


def held_subtype(rest_theta_deg: float, *, back_deg: float = -10.0, forward_deg: float = 10.0) -> str:
    """``held_back`` if theta < ``back_deg``, ``held_forward`` if > ``forward_deg``, else ``held_down``.

    A NaN angle is neither back nor forward, so it comes out ``held_down``; only call this for a
    trial that is held out and has a rest angle.
    """
    if rest_theta_deg < back_deg:
        return "held_back"
    if rest_theta_deg > forward_deg:
        return "held_forward"
    return "held_down"


def retracted_reference(fly_min_rest_eu) -> float:
    """Median of the finite per-fly minimum rest distances (eu): the retracted reference; NaN if there are none."""
    finite = _f64(fly_min_rest_eu)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else float("nan")


def anchor_spread_px(xy) -> float:
    """Largest pairwise Euclidean distance between a fly's anchors (px); 0.0 for fewer than two finite points."""
    points = _f64(xy).reshape(-1, 2)
    points = points[np.isfinite(points).all(axis=1)]
    return float(pdist(points).max()) if len(points) > 1 else 0.0


# ---------------------------------------------------------------------------
# Runs and events
# ---------------------------------------------------------------------------

def _runs(hit: np.ndarray, *, min_frames: int, start: int, end: Optional[int]) -> list[tuple[int, int]]:
    """Half-open ``(i0, i1)`` runs of True inside ``hit[start:end]`` that last at least ``min_frames`` frames."""
    lo = max(0, int(start))
    hi = hit.size if end is None else min(hit.size, int(end))
    if hi <= lo:
        return []
    edges = np.diff(np.concatenate(([0], hit[lo:hi].astype(np.int8), [0])))
    begins, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    long_enough = (stops - begins) >= max(1, int(min_frames))
    return [(lo + int(a), lo + int(b)) for a, b in zip(begins[long_enough], stops[long_enough])]


def runs_above(x, level: float, *, min_frames: int, end: Optional[int] = None, start: int = 0) -> list[tuple[int, int]]:
    """Half-open ``(i0, i1)`` runs inside ``[start, end)`` where ``x > level`` for >= ``min_frames`` frames.

    NaN breaks a run, and a run is clipped to the window before its length is judged.
    """
    return _runs(_f64(x) > level, min_frames=min_frames, start=start, end=end)


def runs_below(x, level: float, *, min_frames: int, end: Optional[int] = None, start: int = 0) -> list[tuple[int, int]]:
    """Mirror of :func:`runs_above` for drops: runs where ``x < level``."""
    return _runs(_f64(x) < level, min_frames=min_frames, start=start, end=end)


def curl_events(bend_deg, rest_deg: float, *, rise_deg: float, min_frames: int,
                start: int = 0, end: Optional[int] = None) -> tuple[int, int]:
    """``(n_curl, n_straighten)``: runs of ``bend > rest + rise`` and of ``bend < rest - rise`` over ``[start, end)``.

    A missing ``rest_deg`` compares False everywhere and counts nothing, so judge the window's
    tracked fraction (and the resting bend) before calling.
    """
    bend = _f64(bend_deg)
    window = dict(min_frames=min_frames, start=start, end=end)
    return (len(runs_above(bend, rest_deg + rise_deg, **window)),
            len(runs_below(bend, rest_deg - rise_deg, **window)))


# The columns of the placebo design, in one place: the table builder and its tests take them from here.
PLACEBO_COLUMNS = ("pb_window_min", "abd_tracked_frac_base", "abd_tracked_frac_placebo", "abd_tracked_frac_during",
                   "abd_rest_bend_base_deg", "abd_curl_events_placebo", "abd_straighten_events_placebo",
                   "abd_curl_events_during", "abd_straighten_events_during")


def placebo_events(bend_deg, *, onset: Optional[int], offset: Optional[int], fps: float, rise_deg: float, min_s: float,
                   min_tracked_frac: float) -> dict[str, float]:
    """Curl / straightening events in an equal-length placebo window and in the stimulus window, both scored against a
    resting bend taken from EARLIER in the trial (the columns :data:`PLACEBO_COLUMNS`).

    With ``L = min(stimulus length, floor(pre-stimulus length / 2))`` frames the trial is cut into the baseline
    ``[0, onset - L)``, the placebo ``[onset - L, onset)`` and the during window ``[onset, onset + L)``; the windows come
    from :func:`window_slice`, so the stimulus runs to the last frame when its end is unknown and only frames that exist
    count. The resting bend is the median of the finite bend over the baseline; curls (``bend > rest + rise_deg``) and
    straightenings (``bend < rest - rise_deg``) lasting at least ``round(min_s * fps)`` frames are counted in the placebo
    window and in the during window against that SAME resting bend, so the two windows are equally long, scored the same
    way, and exchangeable when the stimulus does nothing. (Scoring the pre-stimulus window against its own median, as the
    per-fly columns do, makes it look quieter than any window the baseline was not fitted to, and a stimulus that changes
    nothing then reads as "more events".) The baseline is never shorter than L (L is at most half the pre-stimulus
    window) and needs a tracked fraction of at least *min_tracked_frac* (and at least one finite frame), else there is no
    resting bend and every count is NaN; the placebo and the during window each need the same of their own frames, else
    their counts are NaN. ``pb_window_min`` is L in minutes, the tracked fractions are those of the three windows, and
    everything is NaN when there is no stimulus, when the pre-stimulus window has fewer than two frames (no room for a
    baseline and a placebo window) or when the stimulus has no frame.

    **Limitation: the design assumes a stable resting posture over the trial.** It removes the bias of scoring a window
    against a baseline fitted to itself, but the during window is LATER in the trial than the placebo window, so a bend
    that wanders steadily away from the baseline (a posture that drifts) still makes the two windows differ when the
    stimulus does nothing. Simulated (40 trials of a stationary AR(1) bend, time constant 3 s, 40 fps, 15 s windows, the
    window-swap test, 1000 replicates) the false-positive rate at alpha 0.05 is 0.02 for an SD of 4 degrees and 0.04 for
    6 degrees, and with a random-walk drift added to the SD-4 bend it is 0.02-0.03, 0.06 and 0.15 for 0.3, 0.5 and 1.0
    degrees per square-root second (``tests/test_mounting_metrics.py`` pins the stationary and the 0.3 cases).
    """
    bend = _f64(bend_deg)
    out = dict.fromkeys(PLACEBO_COLUMNS, float("nan"))
    if not (math.isfinite(fps) and fps > 0):
        return out
    pre, stim = (slice(*w.indices(bend.size)[:2]) for w in window_slice(bend.size, onset, offset))
    start = pre.stop
    frames = min(max(0, stim.stop - stim.start), max(0, pre.stop - pre.start) // 2)
    if frames < 1:
        return out
    windows = {"base": slice(0, start - frames), "placebo": slice(start - frames, start),
               "during": slice(start, start + frames)}
    out["pb_window_min"] = frames / fps / 60.0
    tracked = {}
    for name, window in windows.items():
        tracked[name] = float(np.isfinite(bend[window]).mean())
        out[f"abd_tracked_frac_{name}"] = tracked[name]
    base = bend[windows["base"]]
    if tracked["base"] < min_tracked_frac or not np.isfinite(base).any():
        return out
    rest = float(np.median(base[np.isfinite(base)]))
    out["abd_rest_bend_base_deg"] = rest
    for name in ("placebo", "during"):
        if tracked[name] >= min_tracked_frac and tracked[name] > 0:
            curls, straightens = curl_events(bend, rest, rise_deg=rise_deg, min_frames=max(1, round(min_s * fps)),
                                             start=windows[name].start, end=windows[name].stop)
            out[f"abd_curl_events_{name}"], out[f"abd_straighten_events_{name}"] = float(curls), float(straightens)
    return out


# ---------------------------------------------------------------------------
# Point speed
# ---------------------------------------------------------------------------

def point_speed_eu_s(x, y, *, eye_scale_px: float, fps: float, smooth_frames: int = 5) -> np.ndarray:
    """Speed in eu/s: ``hypot(dx, dy) / eye_scale_px * fps`` between consecutive frames (``speed[0]`` is NaN).

    x and y are first smoothed by a centred ``smooth_frames`` rolling median (``min_periods=1``), then
    every frame that was missing in x or y is set missing again, so a gap blanks the speeds on both
    sides of it and is never interpolated. An unusable ``eye_scale_px`` or ``fps`` gives all NaN.
    """
    xs, ys = _f64(x), _f64(y)
    speed = np.full(xs.size, np.nan)
    if xs.size < 2 or not (math.isfinite(eye_scale_px) and eye_scale_px > 0 and math.isfinite(fps) and fps > 0):
        return speed
    missing = ~(np.isfinite(xs) & np.isfinite(ys))
    smoothed = []
    for axis in (xs, ys):
        rolled = pd.Series(np.where(missing, np.nan, axis)).rolling(max(1, int(smooth_frames)), center=True, min_periods=1)
        values = rolled.median().to_numpy(dtype="float64", copy=True)
        values[missing] = np.nan
        smoothed.append(values)
    speed[1:] = np.hypot(np.diff(smoothed[0]), np.diff(smoothed[1])) / eye_scale_px * fps
    return speed


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def spearman(x, y) -> tuple[float, float]:
    """Spearman ``(rho, p)`` over the pairs where both values are finite.

    ``(nan, nan)`` with fewer than 3 pairs or no variation in either series.
    """
    a, b = _finite_pairs(x, y)
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(a, b)
    return float(rho), float(p)


def coupling_rho(per_eu, bend_deg, *, min_frames: int = 200) -> float:
    """Spearman rho between proboscis distance (eu) and bend over the frames where both are finite.

    NaN when fewer than ``min_frames`` frames qualify.
    """
    a, b = _finite_pairs(per_eu, bend_deg)
    return spearman(a, b)[0] if a.size >= min_frames else float("nan")


def stratified_permutation_p(x, y, strata, *, n: int, seed: int = 0) -> float:
    """Two-sided permutation p of ``|Spearman rho(x, y)|`` with x shuffled WITHIN each stratum.

    ``p = (1 + #{|rho_perm| >= |rho_obs|}) / (n + 1)`` over ``n`` draws of ``default_rng(seed)``.
    Complete cases only (finite x and y, a stratum label); NaN with fewer than 3 of them or no
    variation. Spearman is Pearson on ranks, and with y's centred ranks fixed rho is proportional to
    ``rank_x . centred_y``, so the ranks are taken once and only the x ranks are permuted. Ranks are
    multiples of 0.5, so these sums are exact and a rearrangement that ties the observed value is
    detected exactly.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    a, b = _f64(x), _f64(y)
    codes = pd.factorize(pd.Series(list(strata), dtype=object))[0]       # -1 for a missing label
    if not a.shape == b.shape == codes.shape:
        raise ValueError("x, y and strata must have the same length")
    keep = np.isfinite(a) & np.isfinite(b) & (codes >= 0)
    a, b, codes = a[keep], b[keep], codes[keep]
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    rank_x, rank_y = stats.rankdata(a), stats.rankdata(b)
    centred_y = rank_y - rank_y.mean()
    observed = abs(rank_x @ centred_y)
    groups = [np.flatnonzero(codes == c) for c in np.unique(codes)]
    movable = [g for g in groups if g.size > 1]
    fixed = float(sum(rank_x[g] @ centred_y[g] for g in groups if g.size == 1))
    rng = np.random.default_rng(seed)
    step = max(1, min(n, _CHUNK_CELLS // max((g.size for g in movable), default=1)))
    hits = 0
    for done in range(0, n, step):
        draws = min(step, n - done)
        total = np.full(draws, fixed)
        for g in movable:
            block = np.tile(rank_x[g], (draws, 1))
            rng.permuted(block, axis=1, out=block)
            total += block @ centred_y[g]
        hits += int(np.count_nonzero(np.abs(total) >= observed))
    return (1 + hits) / (n + 1)


def within_fly_permutation(df: pd.DataFrame, *, state: str, outcome: str, fly: str,
                           n: int, seed: int = 0) -> tuple[float, float]:
    """``(diff, p)``: pooled ``mean(outcome | state) - mean(outcome | not state)`` and its permutation p.

    Only flies seen in BOTH states count, and trials missing state, outcome or fly are dropped
    first. The p shuffles the state labels within each fly (each fly keeps its counts):
    ``(1 + #{|diff_perm| >= |diff|}) / (n + 1)`` over ``n`` draws of ``default_rng(seed)``.
    ``(nan, nan)`` when no fly has both states.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    values, flag = _f64(df[outcome]), _f64(df[state])
    codes = pd.factorize(df[fly])[0]                                   # -1 for a missing fly
    keep = np.isfinite(values) & np.isfinite(flag) & (codes >= 0)
    values, on, codes = values[keep], flag[keep] != 0, codes[keep]
    flies = []                                                          # (outcomes, in-state mask)
    for c in np.unique(codes):
        mine = codes == c
        if 0 < int(on[mine].sum()) < int(mine.sum()):
            flies.append((values[mine], on[mine]))
    if not flies:
        return float("nan"), float("nan")
    n_on = sum(int(m.sum()) for _, m in flies)
    n_off = sum(m.size - int(m.sum()) for _, m in flies)
    total = float(sum(v.sum() for v, _ in flies))

    def pooled_diff(sum_on):
        return sum_on / n_on - (total - sum_on) / n_off

    observed = float(pooled_diff(float(sum(v[m].sum() for v, m in flies))))
    slack = _PERM_TOL * (max(float(np.abs(v).max()) for v, _ in flies) or 1.0)     # eligible flies only
    rng = np.random.default_rng(seed)
    step = max(1, min(n, _CHUNK_CELLS // max(v.size for v, _ in flies)))
    hits = 0
    for done in range(0, n, step):
        draws = min(step, n - done)
        sum_on = np.zeros(draws)
        for v, m in flies:                       # a random k-subset of the fly's trials gets the state
            block = np.tile(v, (draws, 1))
            rng.permuted(block, axis=1, out=block)
            sum_on += block[:, :int(m.sum())].sum(axis=1)
        hits += int(np.count_nonzero(np.abs(pooled_diff(sum_on)) >= abs(observed) - slack))
    return observed, (1 + hits) / (n + 1)


def stratified_risk_difference(df: pd.DataFrame, *, state: str, outcome: str, stratum: str) -> float:
    """Mantel-Haenszel-weighted risk difference of *outcome* between the trials in and out of *state*, over *stratum*.

    ``sum_k w_k (p1_k - p0_k) / sum_k w_k`` with ``w_k = n1_k n0_k / (n1_k + n0_k)``, k running over the strata that hold
    both states and ``p1_k`` / ``p0_k`` being that stratum's mean outcome among the trials in / not in the state. A
    confound that moves state and outcome together between strata (trial type, say) cancels here, as it does in a
    permutation test that shuffles the state inside each stratum; the pooled difference does not cancel it. Trials
    missing state, outcome or stratum are dropped first (as in :func:`within_fly_permutation`); NaN when no stratum has
    both states.
    """
    values, flag = _f64(df[outcome]), _f64(df[state])
    codes = pd.factorize(df[stratum])[0]                                   # -1 for a missing stratum
    keep = np.isfinite(values) & np.isfinite(flag) & (codes >= 0)
    values, on, codes = values[keep], flag[keep] != 0, codes[keep]
    size = int(codes.max()) + 1 if codes.size else 0
    n1, n0 = np.bincount(codes[on], minlength=size), np.bincount(codes[~on], minlength=size)
    sum1 = np.bincount(codes[on], weights=values[on], minlength=size)
    sum0 = np.bincount(codes[~on], weights=values[~on], minlength=size)
    both = (n1 > 0) & (n0 > 0)
    if not both.any():
        return float("nan")
    weight = n1[both] * n0[both] / (n1[both] + n0[both])
    return float(np.sum(weight * (sum1[both] / n1[both] - sum0[both] / n0[both])) / weight.sum())


def paired_window_permutation(df: pd.DataFrame, *, events_pre: str, min_pre: str, events_stim: str, min_stim: str,
                              n: int, seed: int = 0, scheme: str = "split") -> tuple[float, float]:
    """``(diff, p)``: the pooled event rate during the stimulus minus before it, and its within-trial permutation p.

    Every row is one trial with two windows: ``events_pre`` events in ``min_pre`` minutes before the stimulus and
    ``events_stim`` in ``min_stim`` minutes during it. ``diff`` is ``sum(events_stim) / sum(min_stim) - sum(events_pre) /
    sum(min_pre)`` in events per minute: pooled events over pooled minutes, never a mean of per-trial rates. The p keeps
    every trial intact (what makes a trial busy, or a fly restless, stays with it) and re-draws only how it is split
    between its two windows; it is ``(1 + #{|diff_perm| >= |diff|}) / (n + 1)`` over ``n`` draws of
    ``default_rng(seed)``, two-sided. Trials missing a window (a NaN count or length, or a length that is not positive)
    are left out first; ``(nan, nan)`` when no trial is left. *scheme* says how the split is re-drawn:

    * ``"split"`` (the default): the trial's events (whole numbers) are re-assigned to its windows at random, each to the
      stimulus window with probability ``min_stim / (min_pre + min_stim)``, the minutes staying with their windows. That
      is exact for any window lengths **only if the events of a trial are independent and come at a constant rate** when
      the stimulus changes nothing. Real events (curls of a restless fly, a drifting bend) come in bursts, so two windows
      of one trial resemble each other more than chance counts allow; the p then holds its level only as far as that
      assumption does, and it says nothing about whether the two windows were scored in the same way.
    * ``"swap"``: the trial's two windows, events AND minutes, are exchanged with probability 1/2. **Valid only when the
      two windows of every trial have the same length** (unequal windows are not exchangeable: the short one is far
      noisier than the long one, a swap does not preserve that, and about one stimulus in five that changes nothing
      reads p < 0.05 for 30 s against 4 s), so a trial whose two windows differ in length by more than 1 % is a
      ``ValueError``. The swap is exact when the two windows of a trial are exchangeable under "the stimulus changes
      nothing", which the caller must arrange (see :func:`placebo_events`).

    In both schemes the random draw is made per TRIAL: trials are treated as independent, so a fly with many trials
    weighs more, and the correlation between the trials of one fly is not modelled.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if scheme not in ("swap", "split"):
        raise ValueError(f"scheme must be 'swap' or 'split', got {scheme!r}")
    e0, m0, e1, m1 = (_f64(df[c]) for c in (events_pre, min_pre, events_stim, min_stim))
    keep = np.isfinite(e0) & np.isfinite(e1) & np.isfinite(m0) & np.isfinite(m1) & (m0 > 0) & (m1 > 0)
    e0, m0, e1, m1 = e0[keep], m0[keep], e1[keep], m1[keep]
    if not e0.size:
        return float("nan"), float("nan")
    if scheme == "swap" and (np.maximum(m0, m1) > _SWAP_LENGTH_RATIO * np.minimum(m0, m1)).any():
        raise ValueError("scheme='swap' needs every trial's two windows to have the same length (within 1 %); "
                         "use scheme='split' for windows of different lengths")
    total_e, total_m = float(e0.sum() + e1.sum()), float(m0.sum() + m1.sum())
    if scheme == "split":
        events = e0 + e1
        if (e0 < 0).any() or (e1 < 0).any() or not np.array_equal(events, np.round(events)):
            raise ValueError("scheme='split' needs event counts that are whole numbers of at least 0")
        events, share = events.astype(np.int64), m1 / (m0 + m1)
    gain_e, gain_m = e0 - e1, m0 - m1                  # what the stimulus window takes up when a trial's windows are swapped

    def pooled_diff(stim_e, stim_m):
        return stim_e / stim_m - (total_e - stim_e) / (total_m - stim_m)

    observed = float(pooled_diff(float(e1.sum()), float(m1.sum())))
    slack = _PERM_TOL * max(1.0, float(np.max(np.abs(np.r_[e0 / m0, e1 / m1]))))       # ties, whichever way they round
    rng = np.random.default_rng(seed)
    step = max(1, min(n, _CHUNK_CELLS // e0.size))
    hits = 0
    for done in range(0, n, step):
        draws = min(step, n - done)
        if scheme == "swap":
            swap = rng.random((draws, e0.size)) < 0.5
            stim_e, stim_m = e1.sum() + swap @ gain_e, m1.sum() + swap @ gain_m
        else:
            stim_e, stim_m = rng.binomial(events, share, size=(draws, e0.size)).sum(axis=1), m1.sum()
        hits += int(np.count_nonzero(np.abs(pooled_diff(stim_e, stim_m)) >= abs(observed) - slack))
    return observed, (1 + hits) / (n + 1)


# ---------------------------------------------------------------------------
# Score table
# ---------------------------------------------------------------------------

def dedupe_predictions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Make ``(dataset, fly, fly_number, trial_label)`` unique in a ``model_predictions`` table; returns ``(df, counts)``.

    Exact duplicate rows are kept once (``exact_duplicates`` = rows removed). A key whose rows
    disagree on ``score`` loses ALL its rows (``conflicts`` = number of such keys). Rows that agree
    on the score but differ elsewhere collapse to the first (``same_score_duplicates``).
    """
    missing = [c for c in [*_PRED_KEY, "score"] if c not in df.columns]
    if missing:
        raise KeyError(f"prediction table is missing column(s) {missing}")
    rows = df.drop_duplicates().reset_index(drop=True)
    counts = {"exact_duplicates": len(df) - len(rows)}
    group = rows.groupby(_PRED_KEY, dropna=False, sort=False).ngroup().to_numpy()
    scores = pd.DataFrame({"group": group, "score": rows["score"].to_numpy()}).drop_duplicates()
    disagreeing = scores.loc[scores["group"].duplicated(keep=False), "group"].unique()
    counts["conflicts"] = int(disagreeing.size)
    rows = rows.loc[~np.isin(group, disagreeing)]
    repeat = rows.duplicated(subset=_PRED_KEY, keep="first").to_numpy()
    counts["same_score_duplicates"] = int(repeat.sum())
    return rows.loc[~repeat].reset_index(drop=True), counts


# ---------------------------------------------------------------------------
# Settings: the ``mounting:`` block of an abdomen overlay
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MountingSettings:
    """The ``mounting:`` block; every default is the value the overlay template ships with."""
    enabled: bool = True
    csv_out_dir: str = ""
    out_dir: str = ""
    abdomen_out_dir: str = ""
    batch_date_from: Optional[date] = None
    rest_quantile: float = 0.10
    retracted_ref_eu: Optional[float] = None       # None = per rig, from the flies' most-retracted trials
    held_margin_eu: float = 0.25
    held_frac_min: float = 0.8
    held_back_deg: float = -10.0
    held_forward_deg: float = 10.0
    event_rise_eu: float = 0.30
    event_drop_eu: float = 0.30
    event_min_s: float = 0.3
    identity_tol_px: float = 25.0
    eye_match_px: float = 25.0
    binary_threshold: float = 2.0
    n_permutations: int = 10000
    curl_rise_deg: float = 10.0
    curl_min_s: float = 0.3
    tip_move_eu_s: float = 0.5
    speed_smooth_frames: int = 5
    abd_min_tracked_frac: float = 0.5


def _number(block: Mapping, key: str, *, lo=None, hi=None, lo_open=False, hi_open=False, integer=False):
    """``block[key]`` (or the dataclass default) as a finite float / int inside the given bounds; ValueError otherwise."""
    value = block.get(key, getattr(MountingSettings, key))
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"mounting.{key} must be a finite number, got {value!r}")
    if integer and float(value) != int(value):
        raise ValueError(f"mounting.{key} must be a whole number, got {value!r}")
    number = int(value) if integer else float(value)
    if lo is not None and not (number > lo if lo_open else number >= lo):
        raise ValueError(f"mounting.{key} must be {'>' if lo_open else '>='} {lo}, got {value!r}")
    if hi is not None and not (number < hi if hi_open else number <= hi):
        raise ValueError(f"mounting.{key} must be {'<' if hi_open else '<='} {hi}, got {value!r}")
    return number


def _date(raw, what: str) -> Optional[date]:
    """A date, a datetime or an ISO ``YYYY-MM-DD`` string; None stays None."""
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if isinstance(raw, str) and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", raw.strip()):
        try:
            return date.fromisoformat(raw.strip())
        except ValueError:
            pass                                    # right shape, not a calendar day
    raise ValueError(f"{what} must be a date (YYYY-MM-DD), got {raw!r}")


def _path(block: Mapping, key: str) -> str:
    raw = block.get(key)
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise ValueError(f"mounting.{key} must be a path, got {raw!r}")
    return str(Path(raw.strip()).expanduser()) if raw.strip() else ""


def parse_mounting_block(block: Mapping, *, default_batch_date_from: Optional[date] = None) -> MountingSettings:
    """The overlay's ``mounting:`` block as :class:`MountingSettings`; an unknown key or a value out of range is a ValueError.

    ``batch_date_from`` takes the abdomen block's cut-off (``default_batch_date_from``) when the block does not set it.
    """
    if not isinstance(block, Mapping):
        raise ValueError(f"mounting block must be a mapping, got {block!r}")
    known = [f.name for f in fields(MountingSettings)]
    unknown = sorted(str(k) for k in block if k not in known)
    if unknown:
        raise ValueError(f"mounting has unknown key(s): {unknown}; expected a subset of {sorted(known)}")
    enabled = block.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"mounting.enabled must be true or false, got {enabled!r}")
    ref = None if block.get("retracted_ref_eu") is None else _number(block, "retracted_ref_eu", lo=0.0)
    back = _number(block, "held_back_deg")
    forward = _number(block, "held_forward_deg", lo=back)
    num = lambda key, **bounds: _number(block, key, **bounds)        # noqa: E731  (one-line bound specs below)
    return MountingSettings(
        enabled=enabled, csv_out_dir=_path(block, "csv_out_dir"), out_dir=_path(block, "out_dir"),
        abdomen_out_dir=_path(block, "abdomen_out_dir"),
        batch_date_from=(_date(block.get("batch_date_from"), "mounting.batch_date_from")
                         or _date(default_batch_date_from, "default_batch_date_from")),
        rest_quantile=num("rest_quantile", lo=0.0, hi=1.0, lo_open=True, hi_open=True), retracted_ref_eu=ref,
        held_margin_eu=num("held_margin_eu", lo=0.0), held_frac_min=num("held_frac_min", lo=0.0, hi=1.0, lo_open=True),
        held_back_deg=back, held_forward_deg=forward,
        event_rise_eu=num("event_rise_eu", lo=0.0, lo_open=True), event_drop_eu=num("event_drop_eu", lo=0.0, lo_open=True),
        event_min_s=num("event_min_s", lo=0.0, lo_open=True),
        identity_tol_px=num("identity_tol_px", lo=0.0), eye_match_px=num("eye_match_px", lo=0.0),
        binary_threshold=num("binary_threshold"), n_permutations=num("n_permutations", lo=1, integer=True),
        curl_rise_deg=num("curl_rise_deg", lo=0.0, lo_open=True), curl_min_s=num("curl_min_s", lo=0.0, lo_open=True),
        tip_move_eu_s=num("tip_move_eu_s", lo=0.0), speed_smooth_frames=num("speed_smooth_frames", lo=1, integer=True),
        abd_min_tracked_frac=num("abd_min_tracked_frac", lo=0.0, hi=1.0))
