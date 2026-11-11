"""Measure each proboscis distance from the eye detected in *that* frame.

``yolo_infer`` freezes every fly's eye at an anchor from the trial's first
frames and measures every distance and angle from that fixed point; it also
records, per frame, where the eye was actually detected (``eye_live_x/y``).
These helpers re-measure a per-fly table from the per-frame eye (the frozen
anchor where no eye was matched), keep the frozen anchor in
``x_class0_anchor``/``y_class0_anchor`` so a table can always be put back, and
optionally hold the eye still through a window -- the stimulus light, under
which YOLO places the eye 1-8 px off while the head stays put.

Same geometry as ``yolo_infer``: distance = eye->proboscis, angle = unsigned
angle between eye->proboscis and eye->odor-anchor.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .eye_reanchor import remeasure

EYE_X, EYE_Y = "x_class0", "y_class0"
PROB_X, PROB_Y = "x_class1", "y_class1"
LIVE_X, LIVE_Y = "eye_live_x", "eye_live_y"
BASE_X, BASE_Y = "x_class0_anchor", "y_class0_anchor"
HELPER_COLUMNS = (BASE_X, BASE_Y, "eye_from_live", "eye_held")


def has_live_eye(df: pd.DataFrame) -> bool:
    return LIVE_X in df.columns and LIVE_Y in df.columns


def apply_live_eye(df: pd.DataFrame) -> pd.DataFrame:
    """A copy of ``df`` measured from the per-frame eye (anchor where none).

    Adds ``x/y_class0_anchor`` (the frozen anchor) and ``eye_from_live``.
    A table already rewritten re-measures from its kept anchor, so applying
    twice equals applying once.
    """
    out = df.copy()
    already = BASE_X in out.columns and BASE_Y in out.columns
    anchor_x = out[BASE_X if already else EYE_X].astype(float)
    anchor_y = out[BASE_Y if already else EYE_Y].astype(float)
    out[BASE_X] = anchor_x
    out[BASE_Y] = anchor_y

    if has_live_eye(out):
        live_x = pd.to_numeric(out[LIVE_X], errors="coerce")
        live_y = pd.to_numeric(out[LIVE_Y], errors="coerce")
        use_live = live_x.notna() & live_y.notna() & anchor_x.notna() & anchor_y.notna()
    else:
        live_x = live_y = pd.Series(np.nan, index=out.index)
        use_live = pd.Series(False, index=out.index)

    ex = anchor_x.where(~use_live, live_x)
    ey = anchor_y.where(~use_live, live_y)
    out["eye_from_live"] = use_live.astype(bool)
    if not use_live.any() and not already:
        return out

    out[EYE_X] = ex
    out[EYE_Y] = ey
    px = out[PROB_X].astype(float)
    py = out[PROB_Y].astype(float)
    ax = out["x_anchor"].astype(float)
    ay = out["y_anchor"].astype(float)
    out["distance_0_1"] = np.hypot(px - ex, py - ey)
    out["distance_0_anchor"] = np.hypot(ex - ax, ey - ay)
    v1x, v1y = px - ex, py - ey
    v2x, v2y = ax - ex, ay - ey
    out["angle_deg_c0_c1_vs_anchor"] = np.degrees(np.arctan2(np.abs(v1x * v2y - v1y * v2x), v1x * v2x + v1y * v2y))
    return out


def stimulus_hold(meta: dict, odor: bool = True, before_s: float = 0.5,
                  after_s: float = 1.0) -> Optional[Tuple[int, int]]:
    """(first, last) frame of the light window -- and the odor window when
    ``odor`` -- widened by ``before_s``/``after_s``; None if neither applies.

    The light window ends at ``light_last_s`` (the whole pulse train), or at
    ``light_off_s`` in a dict without it."""
    fps = float(meta.get("fps") or 40.0)
    spans = []
    on, off = meta.get("odor_on_frame"), meta.get("odor_off_frame")
    if odor and on is not None and off is not None:
        spans.append((int(on) - int(round(before_s * fps)), int(off) + int(round(after_s * fps))))
    lon = meta.get("light_on_s")
    loff = meta.get("light_last_s")
    if loff is None:
        loff = meta.get("light_off_s")
    if lon is not None:
        loff = loff if loff is not None else float(meta.get("n_frames") or 0) / fps
        spans.append((int(round((float(lon) - before_s) * fps)), int(round((float(loff) + after_s) * fps))))
    if not spans:
        return None
    return max(0, min(a for a, _ in spans)), max(b for _, b in spans)


def apply_live_eye_held(df: pd.DataFrame, hold, pre_frames: int) -> pd.DataFrame:
    """``apply_live_eye``, except that frames in ``hold`` (first, last) keep the
    eye where it sat over the ``pre_frames`` before it (median per-frame eye;
    the frozen anchor if none was seen). Adds ``eye_held``."""
    out = apply_live_eye(df)
    out["eye_held"] = False
    if hold is None:
        return out
    fr = out["frame"].to_numpy()
    inside = (fr >= hold[0]) & (fr <= hold[1])
    if not inside.any():
        return out
    lx = pd.to_numeric(out.get(LIVE_X), errors="coerce").to_numpy(dtype=float)
    ly = pd.to_numeric(out.get(LIVE_Y), errors="coerce").to_numpy(dtype=float)
    pre = (fr >= hold[0] - pre_frames) & (fr < hold[0]) & np.isfinite(lx) & np.isfinite(ly)
    if pre.sum() >= max(1, pre_frames // 4):
        hx, hy = float(np.median(lx[pre])), float(np.median(ly[pre]))
    else:
        hx = float(np.nanmedian(out[BASE_X].to_numpy(dtype=float)[inside]))
        hy = float(np.nanmedian(out[BASE_Y].to_numpy(dtype=float)[inside]))
    ex = out[EYE_X].to_numpy(dtype=float).copy()
    ey = out[EYE_Y].to_numpy(dtype=float).copy()
    ex[inside], ey[inside] = hx, hy
    remeasure(out, ex, ey, inside)
    out.loc[inside, "eye_from_live"] = False
    out.loc[inside, "eye_held"] = True
    return out


def restore_fixed_eye(df: pd.DataFrame) -> pd.DataFrame:
    """Undo ``apply_live_eye*``: eye back on the frozen anchor, distances and
    angles re-measured, helper columns dropped. Unchanged when never applied."""
    if BASE_X not in df.columns or BASE_Y not in df.columns:
        return df
    out = df.copy()
    bx = out[BASE_X].to_numpy(dtype=float)
    by = out[BASE_Y].to_numpy(dtype=float)
    cx = out[EYE_X].to_numpy(dtype=float)
    cy = out[EYE_Y].to_numpy(dtype=float)
    rows = ~(np.isclose(cx, bx, equal_nan=True) & np.isclose(cy, by, equal_nan=True))
    remeasure(out, bx, by, rows)
    return out.drop(columns=[c for c in HELPER_COLUMNS if c in out.columns])
