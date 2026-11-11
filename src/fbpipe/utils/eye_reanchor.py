"""Drift-following eye anchor: static for measuring, moved only by real moves.

``yolo_infer`` freezes each fly's eye at an anchor taken from the first frames
of a trial and measures every proboscis distance from it. It also records the
eye it detects in every frame (``eye_live_x/y``). Measuring from that raw live
eye adds detection jitter and, now and then, a 1-2 frame false match tens of
px away -- enough to reset a fly's global min and rescale all its traces.

This module keeps the anchor static and moves it only for evidence of a real
move:

* **Sustained drift** -- over a trailing ``sustain_s`` window (at least half
  of it detected, the eye-QC window) the eye sat still (x and y
  interquartile ranges within ``STILL_IQR_PX``) at a median position more
  than ``max_drift_px`` from the anchor, *and* the eye stays there through
  the next window (half of it detected, still, still past the threshold).
  The anchor then steps to where the eye stayed, from the frame the shift
  began, and stays there until the next sustained move. A fly that has moved
  once in the trial is followed at ``FOLLOW_PX`` from then on, so a loose eye
  that settles back a few px does not leave the anchor stuck beside it; a
  fly that never makes a first move is measured exactly as in production. The median *position* is used, not eye-QC's median
  *distance*, and the eye must sit still: rig vibration swings the eye
  +-25 px around where it was for a second at a time, which a median
  distance -- or the median of a two-spot window -- reads as drift.
* **Group move** -- every fly of the trial (at least ``min_frac`` of them seen,
  two or more) holds a shift in the same direction: the bar drops to
  ``group_min_px``. A brief jump of all flies at once is NOT followed: on
  1,107 such frames (600 trials) the fixed anchor kept the eye->proboscis
  distance steadier (mean error 0.45 px) than following the shared jump
  (0.74 px) or each fly's own eye (1.15 px) -- the eye detection swings with
  the vibration while the tracked proboscis barely does.
* A few-frame spike of one fly (a false eye match) never moves anything.
* **Light on** -- the stimulus light makes YOLO place the eye 5-10 px off
  while the head stays put; frames from light-on to just after light-off are
  no evidence of a move (``reanchor_trial(hold=...)``).
* **Warm-up miss** -- a trial whose own first ``scan_frames`` frames never
  saw the eye had its anchor borrowed from the batch template. It takes the
  anchor of the next trial (in recording order) that did see the eye, since
  the eye does not move between videos; the previous one when none follows.

Pure functions; nothing here reads or writes files.
"""
from __future__ import annotations

import math
from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EYE_X, EYE_Y = "x_class0", "y_class0"
BASE_X, BASE_Y = "x_class0_anchor", "y_class0_anchor"
PROB_X, PROB_Y = "x_class1", "y_class1"
LIVE_X, LIVE_Y = "eye_live_x", "eye_live_y"

#: Group-move defaults, from the per-frame eye jitter of 1.07 M frames (300
#: trials, four Sensitivity datasets): |live - trailing median| p99 1.3 px,
#: p99.9 2.75 px; a 0.5 s median of it is far tighter. Held group shifts seen
#: in 600 trials were 2-3 px (7 trials), never more.
GROUP_MIN_PX = 2.0
GROUP_AGREE_PX = 1.5
GROUP_MIN_FRAC = 0.75
#: After a fly's first move in a trial, later settled positions are followed
#: at this resolution (see step_offsets).
FOLLOW_PX = 2.0
#: "Sat still": interquartile range of the eye's x and of its y over the
#: window. Detection jitter gives ~0.5 px, a slow drift of 0.2 px/frame ~2 px,
#: a rig vibration 10-50 px.
STILL_IQR_PX = 4.0
SCAN_FRAMES = 5  # yolo_infer's warm-up scan


# --------------------------------------------------------------------------
# Sustained drift -> step anchor
# --------------------------------------------------------------------------


def _sustained_xy(
    dx: np.ndarray, dy: np.ndarray, window: int, max_iqr_px: float = STILL_IQR_PX
) -> Tuple[np.ndarray, np.ndarray]:
    """Where the eye has settled over the trailing ``window`` frames.

    The x and y medians of the detected frames, NaN unless at least half the
    window holds a detection (the eye-QC rule) *and* the eye sat still --
    both interquartile ranges within ``max_iqr_px``. A vibrating eye flips
    between two spots, and the median of a two-spot window jumps to one of
    them; the spread is what tells the shake from a held shift. Nothing
    before the window first fills.
    """
    sx, sy = pd.Series(dx, dtype=float), pd.Series(dy, dtype=float)
    seen = sx.notna() & sy.notna()
    need = max(1, math.ceil(window / 2))
    enough = seen.rolling(window, min_periods=1).sum().to_numpy() >= need
    stats = []
    for s in (sx.where(seen), sy.where(seen)):
        r = s.rolling(window, min_periods=need)
        stats.append((r.median().to_numpy(dtype=float),
                      (r.quantile(0.75) - r.quantile(0.25)).to_numpy(dtype=float)))
    (mx, iqx), (my, iqy) = stats
    ok = enough & (iqx <= max_iqr_px) & (iqy <= max_iqr_px)
    mx = np.where(ok, mx, np.nan)
    my = np.where(ok, my, np.nan)
    mx[: window - 1] = np.nan
    my[: window - 1] = np.nan
    return mx, my


def _stays(dev_x: np.ndarray, dev_y: np.ndarray, b: int, window: int,
           max_iqr_px: float = STILL_IQR_PX, blocked: Optional[np.ndarray] = None
           ) -> Optional[Tuple[float, float]]:
    """Where the eye sat over frames [b, b + window): its median, or None
    unless at least half of those frames hold a detection and both
    interquartile ranges are within ``max_iqr_px`` -- and none of them is
    ``blocked`` (light on: no evidence either way)."""
    x, y = dev_x[b : b + window], dev_y[b : b + window]
    if x.size == 0:
        return None
    if blocked is not None and blocked[b : b + window].any():
        return None
    seen = np.isfinite(x) & np.isfinite(y)
    if seen.sum() < max(1, math.ceil(x.size / 2)):
        return None
    xs, ys = x[seen], y[seen]
    if np.subtract(*np.percentile(xs, [75, 25])) > max_iqr_px or np.subtract(*np.percentile(ys, [75, 25])) > max_iqr_px:
        return None
    return float(np.median(xs)), float(np.median(ys))


def step_offsets(
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    *,
    window: int,
    max_drift_px,
    initial: Tuple[float, float] = (0.0, 0.0),
    follow_px: Optional[float] = None,
    blocked: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Piecewise-constant anchor offset for one fly in one trial.

    ``dev_x/dev_y`` is where the eye was seen relative to the frozen anchor
    (NaN when unmatched). ``max_drift_px`` is a number or one threshold per
    frame (lower where the flies moved as a group). Once the anchor has moved
    in this trial, ``follow_px`` (when given) caps the threshold: the fly has
    shown it moves, so the anchor follows where the eye settles next instead
    of sticking up to ``max_drift_px`` away from it. Returns the offset to add
    to the anchor in every frame and one event dict per move. ``blocked``
    frames (light on) are no evidence: a move whose confirming half-second
    touches one waits until after them. Each event has ``frame`` =
    onset, ``dx``/``dy`` = the new offset, ``step_px`` = how far it moved).
    """
    dev_x = np.asarray(dev_x, dtype=float)
    dev_y = np.asarray(dev_y, dtype=float)
    n = dev_x.size
    thr0 = np.broadcast_to(np.asarray(max_drift_px, dtype=float), (n,))
    thr_moved = thr0 if follow_px is None else np.minimum(thr0, float(follow_px))
    window = max(1, int(window))
    ox = np.empty(n)
    oy = np.empty(n)
    cur = (float(initial[0]), float(initial[1]))
    start = 0
    events: List[dict] = []
    while start < n:
        thr = thr_moved if events else thr0
        rx, ry = dev_x[start:] - cur[0], dev_y[start:] - cur[1]
        mx, my = _sustained_xy(rx, ry, window)
        event = None
        for t in np.flatnonzero(np.hypot(mx, my) > thr[start:]):   # segment-relative, earliest first
            # ... and it must STAY: the next window holds the eye, still, at a
            # spot past the threshold. Light-onset junk (a false eye that sits
            # still for a few frames before the eye is lost) fails here.
            new = _stays(dev_x, dev_y, start + t + 1, window, blocked=blocked)
            if new is None or math.hypot(new[0] - cur[0], new[1] - cur[1]) <= thr[start + t]:
                continue
            lo = max(0, t - window + 1)
            # onset: first frame of the trigger window already on the far side
            # (past the threshold, in the direction of the median shift)
            d = rx[lo : t + 1] * mx[t] + ry[lo : t + 1] * my[t]
            above = np.flatnonzero(d > thr[start + t] * math.hypot(mx[t], my[t]))
            onset = lo + (int(above[0]) if above.size else 0)
            if events and onset == 0:
                onset = 1                             # always make progress
            event = (start + onset, new, float(thr[start + t]))
            break
        if event is None:
            break
        a, new, used_thr = event
        ox[start:a] = cur[0]
        oy[start:a] = cur[1]
        events.append({
            "frame": a,
            "dx": new[0],
            "dy": new[1],
            "step_px": float(math.hypot(new[0] - cur[0], new[1] - cur[1])),
            "threshold_px": used_thr,
        })
        cur = new
        start = a
    ox[start:] = cur[0]
    oy[start:] = cur[1]
    return ox, oy, events


# --------------------------------------------------------------------------
# Group moves (every fly holds the same shift)
# --------------------------------------------------------------------------


def group_moves(
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    *,
    window: int,
    min_px: float = GROUP_MIN_PX,
    agree_px: float = GROUP_AGREE_PX,
    min_frac: float = GROUP_MIN_FRAC,
) -> np.ndarray:
    """Frames where every fly of a trial holds the same shift.

    ``dev_x/dev_y`` are (n_frames, n_flies): each fly's eye relative to its
    starting anchor. Per fly, the sustained (trailing-median) shift as in
    ``step_offsets``. A frame is a group move when at least
    ``max(2, ceil(min_frac * n_flies))`` flies have a sustained value, every
    one of them is shifted at least ``min_px``, and all lie within
    ``agree_px`` of their median shift.
    """
    dx = np.asarray(dev_x, dtype=float)
    dy = np.asarray(dev_y, dtype=float)
    if dx.ndim == 1:
        dx, dy = dx[:, None], dy[:, None]
    n, k = dx.shape
    if k < 2:
        return np.zeros(n, dtype=bool)
    MX = np.full((n, k), np.nan)
    MY = np.full((n, k), np.nan)
    for j in range(k):
        MX[:, j], MY[:, j] = _sustained_xy(dx[:, j], dy[:, j], int(window))
    ok = np.isfinite(MX) & np.isfinite(MY)
    need = max(2, math.ceil(min_frac * k))
    big = np.where(ok, np.hypot(MX, MY) >= min_px, True)
    cand = (ok.sum(1) >= need) & big.all(1)
    out = np.zeros(n, dtype=bool)
    for f in np.flatnonzero(cand):
        m = ok[f]
        cx, cy = np.median(MX[f, m]), np.median(MY[f, m])
        out[f] = bool(np.max(np.hypot(MX[f, m] - cx, MY[f, m] - cy)) <= agree_px)
    return out


# --------------------------------------------------------------------------
# Warm-up miss -> next video's anchor
# --------------------------------------------------------------------------


def warmup_found(live_x, live_y, scan_frames: int = SCAN_FRAMES) -> bool:
    """Whether the eye was matched in any of the trial's first ``scan_frames`` frames."""
    lx = np.asarray(live_x, dtype=float)[:scan_frames]
    ly = np.asarray(live_y, dtype=float)[:scan_frames]
    return bool((np.isfinite(lx) & np.isfinite(ly)).any())


def warmup_offsets(
    order: Sequence[Hashable],
    anchors: Mapping[Hashable, Tuple[float, float]],
    found: Mapping[Hashable, bool],
) -> Dict[Hashable, Tuple[float, float]]:
    """Starting offset per trial for one fly.

    ``order`` lists the fly's trials in recording order. A trial whose warm-up
    saw the eye keeps its own anchor (offset 0). One that did not takes the
    anchor of the next trial that did -- the previous one if none follows --
    as an offset from its own (template) anchor. Never found anywhere: 0.
    """
    out: Dict[Hashable, Tuple[float, float]] = {}
    seen = [t for t in order if found.get(t)]
    for i, t in enumerate(order):
        if found.get(t) or not seen:
            out[t] = (0.0, 0.0)
            continue
        later = [s for s in order[i + 1 :] if found.get(s)]
        earlier = [s for s in order[:i] if found.get(s)]
        donor = later[0] if later else earlier[-1]
        out[t] = (
            float(anchors[donor][0] - anchors[t][0]),
            float(anchors[donor][1] - anchors[t][1]),
        )
    return out


# --------------------------------------------------------------------------
# One trial's per-fly tables
# --------------------------------------------------------------------------


def _base(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """The frozen yolo_infer anchor: kept in *_anchor by an earlier rewrite."""
    if BASE_X in df.columns and BASE_Y in df.columns:
        return df[BASE_X].to_numpy(dtype=float), df[BASE_Y].to_numpy(dtype=float)
    return df[EYE_X].to_numpy(dtype=float), df[EYE_Y].to_numpy(dtype=float)


def _live(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if LIVE_X in df.columns and LIVE_Y in df.columns:
        return (pd.to_numeric(df[LIVE_X], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(df[LIVE_Y], errors="coerce").to_numpy(dtype=float))
    nan = np.full(len(df), np.nan)
    return nan, nan.copy()


def remeasure(df: pd.DataFrame, ex: np.ndarray, ey: np.ndarray, rows: np.ndarray) -> None:
    """In place: eye -> (ex, ey) and distance/angle re-measured on ``rows``.

    Same geometry as yolo_infer: distance = eye->proboscis, angle = unsigned
    angle between eye->proboscis and eye->odor-anchor.
    """
    if not rows.any():
        return
    px = df[PROB_X].to_numpy(dtype=float)[rows]
    py = df[PROB_Y].to_numpy(dtype=float)[rows]
    x, y = ex[rows], ey[rows]
    df.loc[rows, EYE_X] = x
    df.loc[rows, EYE_Y] = y
    df.loc[rows, "distance_0_1"] = np.hypot(px - x, py - y)
    if "x_anchor" in df.columns and "y_anchor" in df.columns:
        ax = df["x_anchor"].to_numpy(dtype=float)[rows]
        ay = df["y_anchor"].to_numpy(dtype=float)[rows]
        df.loc[rows, "distance_0_anchor"] = np.hypot(x - ax, y - ay)
        v1x, v1y, v2x, v2y = px - x, py - y, ax - x, ay - y
        df.loc[rows, "angle_deg_c0_c1_vs_anchor"] = np.degrees(
            np.arctan2(np.abs(v1x * v2y - v1y * v2x), v1x * v2x + v1y * v2y)
        )


def reanchor_trial(
    tables: Mapping[Hashable, pd.DataFrame],
    *,
    window: int,
    max_drift_px: float,
    initial_offsets: Optional[Mapping[Hashable, Tuple[float, float]]] = None,
    group_min_px: float = GROUP_MIN_PX,
    group_agree_px: float = GROUP_AGREE_PX,
    group_min_frac: float = GROUP_MIN_FRAC,
    follow_px: float = FOLLOW_PX,
    hold: Optional[Tuple[int, int]] = None,
) -> Tuple[Dict[Hashable, pd.DataFrame], List[dict]]:
    """Re-measure every fly of one trial from its drift-following anchor.

    ``tables`` maps fly -> its per-fly YOLO table (``frame`` column). Returns
    new tables (inputs untouched) and one summary dict per fly. Added columns:
    ``x/y_class0_anchor`` (the frozen anchor), ``eye_step_dx/dy`` (the offset
    from it), ``eye_group_move`` (the flies held a shared shift this frame) and
    ``eye_moved`` (eye differs from the frozen anchor in this frame). Rows
    whose eye ends up on the frozen anchor keep their values exactly.

    ``hold`` = (first, last) frame, inclusive, while the stimulus light is on:
    the light makes YOLO place the eye 5-10 px off while the head stays put
    (it slides back as the light goes off), so those frames are no evidence
    of a move -- they count as unmatched. A real slip during the light is
    taken up from the first frame after it.
    """
    initial_offsets = initial_offsets or {}
    flies = list(tables)
    frames = sorted(set().union(*(set(tables[f]["frame"].astype(int)) for f in flies))) if flies else []
    pos = {fr: i for i, fr in enumerate(frames)}
    n = len(frames)

    def _seen(df):
        lx, ly = _live(df)
        if hold is not None:
            fr = df["frame"].to_numpy()
            off = (fr >= hold[0]) & (fr <= hold[1])
            lx, ly = np.where(off, np.nan, lx), np.where(off, np.nan, ly)
        return lx, ly

    DX = np.full((n, len(flies)), np.nan)
    DY = np.full((n, len(flies)), np.nan)
    for j, f in enumerate(flies):
        idx = tables[f]["frame"].astype(int).map(pos).to_numpy()
        bx, by = _base(tables[f])
        lx, ly = _seen(tables[f])
        ix, iy = initial_offsets.get(f, (0.0, 0.0))
        DX[idx, j], DY[idx, j] = lx - bx - ix, ly - by - iy
    group = group_moves(DX, DY, window=window, min_px=group_min_px,
                        agree_px=group_agree_px, min_frac=group_min_frac)
    thr_all = np.where(group, min(group_min_px, max_drift_px), max_drift_px)

    out: Dict[Hashable, pd.DataFrame] = {}
    summary: List[dict] = []
    for j, f in enumerate(flies):
        df = tables[f].copy()
        idx = df["frame"].astype(int).map(pos).to_numpy()
        bx, by = _base(df)
        lx, ly = _seen(df)
        init = initial_offsets.get(f, (0.0, 0.0))
        held = None
        if hold is not None:
            fr = df["frame"].to_numpy()
            held = (fr >= hold[0]) & (fr <= hold[1])
        ox, oy, events = step_offsets(lx - bx, ly - by, window=window,
                                      max_drift_px=thr_all[idx], initial=init, follow_px=follow_px,
                                      blocked=held)
        ex, ey = bx + ox, by + oy
        cur_x = df[EYE_X].to_numpy(dtype=float)
        cur_y = df[EYE_Y].to_numpy(dtype=float)
        moved = (ex != bx) | (ey != by)
        # also restore rows an earlier rewrite moved, so a second pass is a no-op
        rows = moved | (cur_x != bx) | (cur_y != by)
        rows &= np.isfinite(ex) & np.isfinite(ey)
        df[BASE_X], df[BASE_Y] = bx, by
        remeasure(df, ex, ey, rows)
        df["eye_step_dx"], df["eye_step_dy"] = ox, oy
        df["eye_group_move"] = group[idx]
        df["eye_moved"] = moved
        out[f] = df
        steps = np.hypot(ox, oy)
        summary.append({
            "fly": f,
            "n_frames": len(df),
            "initial_dx": float(init[0]),
            "initial_dy": float(init[1]),
            "n_steps": len(events),
            "first_move_group": bool(events) and events[0]["threshold_px"] < max_drift_px,
            "n_follow_steps": max(0, len(events) - 1),
            "first_step_frame": events[0]["frame"] if events else None,
            "max_offset_px": float(np.nanmax(steps)) if len(steps) else 0.0,
            "final_offset_px": float(steps[-1]) if len(steps) else 0.0,
            "group_frames": int(group[idx].sum()),
            "moved_frac": float(moved.mean()) if len(df) else 0.0,
            "events": events,
        })
    return out, summary
