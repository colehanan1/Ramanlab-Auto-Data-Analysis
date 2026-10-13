"""Pure geometry for the abdomen pipeline. No I/O, no YOLO, no pandas.

Coordinates are image pixels (x right, y down). Every function accepts plain
sequences or numpy arrays and returns floats / numpy arrays; NaN means "not
measurable" and is never replaced by a guess.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np


def map_corners(corners: Sequence, sx: float, sy: float) -> np.ndarray:
    """Scale box corners from model space to another pixel space, x and y independently."""
    pts = np.asarray(corners, dtype=float).reshape(-1, 2).copy()
    pts[:, 0] *= float(sx)
    pts[:, 1] *= float(sy)
    return pts


def obb_length_width(corners: Sequence) -> Tuple[float, float]:
    """(long side, short side) of a 4-corner box given in cyclic order.

    Opposite edges are averaged, so a slightly non-rectangular quad (after a
    non-uniform mapping) still yields one length and one width.
    """
    p = np.asarray(corners, dtype=float).reshape(4, 2)
    e = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)   # e01, e12, e23, e30
    a = 0.5 * (e[0] + e[2])
    b = 0.5 * (e[1] + e[3])
    return (float(max(a, b)), float(min(a, b)))


def interior_angle(eye_xy: Sequence, abd_xy: Sequence, tip_xy: Sequence) -> float:
    """Angle at the abdomen centre between the eye and the tip, in degrees.

    180 = eye, abdomen and tip on a straight line; smaller = the tip curls
    back toward the eye. NaN when a vector is degenerate.
    """
    u = np.asarray(eye_xy, dtype=float) - np.asarray(abd_xy, dtype=float)
    v = np.asarray(tip_xy, dtype=float) - np.asarray(abd_xy, dtype=float)
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu <= 0 or nv <= 0:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(float(u @ v) / (nu * nv), -1.0, 1.0))))


def ventral_sign(eye_xy: Sequence, abd_xy: Sequence, pro_xy: Sequence) -> float:
    """Which side of the eye->abdomen axis the proboscis lies on: +1, -1, or 0."""
    a = np.asarray(abd_xy, dtype=float) - np.asarray(eye_xy, dtype=float)
    p = np.asarray(pro_xy, dtype=float) - np.asarray(eye_xy, dtype=float)
    return float(np.sign(a[0] * p[1] - a[1] * p[0]))


def signed_bend(eye_xy: Sequence, abd_xy: Sequence, tip_xy: Sequence, side_sign: float) -> float:
    """180 minus the interior angle, signed so that + means the tip deflects toward
    ``side_sign`` (the ventral / proboscis side from :func:`ventral_sign`).
    With ``side_sign == 0`` the bend is returned unsigned."""
    ang = interior_angle(eye_xy, abd_xy, tip_xy)
    if not np.isfinite(ang):
        return float("nan")
    bend = 180.0 - ang
    if not side_sign:
        return float(bend)
    a = np.asarray(abd_xy, dtype=float) - np.asarray(eye_xy, dtype=float)
    t = np.asarray(tip_xy, dtype=float) - np.asarray(abd_xy, dtype=float)
    s = float(np.sign(a[0] * t[1] - a[1] * t[0]))
    return float(bend * s * float(np.sign(side_sign)))


class OnlineVelocityGate:
    """Causal jump gate: a point farther than ``max_jump_px`` from the last ACCEPTED
    point is rejected and does not become the reference (the same walk as
    ``distance_sanity.sanitize_proboscis_velocity_dataframe``), so one bad frame
    cannot drag the reference along. ``max_jump_px <= 0`` disables the gate."""

    def __init__(self, max_jump_px: float) -> None:
        self.max_jump = float(max_jump_px)
        self.last: Optional[Tuple[float, float]] = None

    def rejects(self, x: float, y: float) -> bool:
        if not (np.isfinite(x) and np.isfinite(y)):
            return False
        if self.max_jump > 0 and self.last is not None and math.hypot(x - self.last[0], y - self.last[1]) > self.max_jump:
            return True
        self.last = (float(x), float(y))
        return False


def velocity_gate(xy: Sequence, max_jump_px: float) -> np.ndarray:
    """Array form of :class:`OnlineVelocityGate`: True where a point is rejected."""
    pts = np.asarray(xy, dtype=float).reshape(-1, 2)
    gate = OnlineVelocityGate(max_jump_px)
    return np.array([gate.rejects(x, y) for x, y in pts], dtype=bool)


def proboscis_pct(d: Sequence, gmin: float, gmax: float, floor: float) -> np.ndarray:
    """Proboscis percentage: 0 at ``gmin``, 100 at ``max(gmax, floor)``.

    Deliberately diverges from ``fbpipe.steps.distance_normalize`` in ONE respect:
    production NaNs anything outside ``[gmin, gmax]``, but ``gmin`` here is a trimmed
    percentile, so ~``per_trim_pct`` % of frames sit below it by construction and
    NaN-ing them would blank the very resting baseline the trim exists to stabilise.
    Out-of-range values therefore CLAMP to 0 / 100 (user spec 2026-09-10). A missing
    detection stays NaN -- absent is not the same as resting.
    """
    d = np.asarray(d, dtype=float)
    eff = max(float(gmax), float(floor))
    out = np.full(d.shape, np.nan, dtype=float)
    fin = np.isfinite(d)
    if eff != gmin:
        out[fin] = np.clip(100.0 * (d[fin] - gmin) / (eff - gmin), 0.0, 100.0)
    else:
        out[fin] = 0.0
    return out


def angle_pct(angle: Sequence, angle_min: float, angle_max: float,
              min_range_deg: float = 0.0) -> np.ndarray:
    """Per-fly abdomen angle percentage: 0 at the fly's straightest observed angle
    (``angle_max``), 100 at its most curled (``angle_min``). ``min_range_deg``
    floors the denominator so a fly that barely moves cannot reach 100 on noise."""
    a = np.asarray(angle, dtype=float)
    rng = max(float(angle_max) - float(angle_min), float(min_range_deg))
    if not (np.isfinite(rng) and rng > 0):
        return np.full(a.shape, np.nan, dtype=float)
    return np.clip(100.0 * (float(angle_max) - a) / rng, 0.0, 100.0)


def anchor_angle_deg(eye_xy: Sequence, pro_xy: Sequence, anchor_xy: Sequence) -> float:
    """Unsigned angle between eye->proboscis and eye->rig-anchor (production parity)."""
    e = np.asarray(eye_xy, dtype=float)
    v1 = np.asarray(pro_xy, dtype=float) - e
    v2 = np.asarray(anchor_xy, dtype=float) - e
    dot = float(v1 @ v2)
    cross = float(v1[0] * v2[1] - v1[1] * v2[0])
    return float(np.degrees(np.arctan2(abs(cross), dot)))
