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
  stratum-level (fly-level) confound cannot produce a small p on its own.
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
