"""Geometric plausibility rules for eye / proboscis / abdomen / abdomenTip.

Every threshold below was measured from the 900 correctly-paired flies in the
`labelledAbdo` label set. Lengths are expressed in units of

    eye_scale = sqrt(eye_area_px)          # ~48 px in the current rig

so the rules follow the fly rather than the camera. That matters: the existing
absolute ``max_eye_prob_distance_px`` gate is scale-dependent, and the largest
true eye->proboscis extension in the labels is 179 px, past both the 150 px
dataclass default and the 160 px in config_new.yaml. A fly imaged larger is
clipped by a fixed radius while an identical fly imaged smaller sails through.

Three findings drive the rules, and each contradicts an intuition worth naming:

* The abdomen is NOT opposite the proboscis. The median separation is 79 deg
  (range 43..126), so a "180 deg from the proboscis" test rejects real flies.
* The abdomen tip lies on the FAR side of the abdomen from the eye, in 100% of
  labels. This is the only reliable way to break the tie when one fly's abdomen
  reaches past a neighbour's head, which plain nearest-neighbour matching gets
  wrong often enough to invent a fake rig orientation.
* Absolute orientation and handedness are properties of the RIG, not the fly:
  the head->abdomen direction is ~20 deg in 23 sessions but ~163 deg in the
  mirrored flybehavior2 configuration (post 2026-07-25), and the proboscis->
  abdomen turn flips sign with it (100% clean split). Both are therefore
  estimated per frame and never hardcoded. See ``utils/rig_anchor.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

EYE = "eye"
PROBOSCIS = "proboscis"
ABDOMEN = "abdomen"
ABDOMEN_TIP = "abdomenTip"


@dataclass(frozen=True)
class FlyGeometryLimits:
    """Accept/reject bounds. Comments give the observed range in the labels."""

    # -- eye: sets the scale for every other rule, so it is gated first --------
    eye_scale_min: float = 30.0          # px            [33.2 .. 66.3]
    eye_scale_max: float = 75.0
    eye_aspect_max: float = 2.0          #               [0.98 .. 1.69]

    # -- proboscis ------------------------------------------------------------
    d_eye_pro_min: float = 0.6           # x eye_scale   [0.79 .. 2.82]
    d_eye_pro_max: float = 3.2
    pro_size_min: float = 0.15           # x eye_scale   [0.22 .. 0.84]
    pro_size_max: float = 1.0

    # -- abdomen --------------------------------------------------------------
    d_eye_abd_min: float = 3.4           # x eye_scale   [3.56 .. 7.19]
    d_eye_abd_max: float = 7.4
    abd_size_min: float = 1.7            # x eye_scale   [1.82 .. 4.23]
    abd_size_max: float = 4.5

    # -- abdomen tip ----------------------------------------------------------
    d_eye_tip_min: float = 4.4           # x eye_scale   [4.59 .. 9.71]
    d_eye_tip_max: float = 10.0
    d_abd_tip_min: float = 0.9           # x eye_scale   [0.94 .. 2.64]
    d_abd_tip_max: float = 2.8
    tip_ratio_min: float = 1.03          # d_tip/d_abd   [1.08 .. 1.50]
    tip_ratio_max: float = 1.55
    tip_size_min: float = 0.35           # x eye_scale   [0.40 .. 1.67]
    tip_size_max: float = 1.8
    tip_axis_tol_deg: float = 30.0       # deg           [0.13 .. 24.05]

    # -- relations between parts ---------------------------------------------
    ang_pro_abd_min: float = 35.0        # deg           [43.3 .. 125.9]
    ang_pro_abd_max: float = 135.0
    dir_tol_deg: float = 50.0            # vs the frame's posterior axis

    # -- visibility -----------------------------------------------------------
    posterior_reach: float = 6.37        # x eye_scale, median eye->tip
    edge_margin: float = 0.0             # x eye_scale of required clearance

    # Absolute fallbacks, used only when there is no eye to scale against.
    abs_d_eye_pro_max_px: float = 280.0


DEFAULT_LIMITS = FlyGeometryLimits()
_MEDIAN_D_EYE_ABD = 4.96      # eye->abdomen distance of the median labelled fly, eye_scale units


# --------------------------------------------------------------------------- #
# small geometry helpers
# --------------------------------------------------------------------------- #
def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else np.zeros(2)


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Unsigned angle between two vectors, in degrees."""
    return float(np.degrees(np.arccos(np.clip(_unit(a) @ _unit(b), -1.0, 1.0))))


def polygon_centroid(pts: np.ndarray) -> np.ndarray:
    """Area-weighted centroid; falls back to the vertex mean when degenerate."""
    p = np.asarray(pts, dtype=float)
    x, y = p[:, 0], p[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y2 - x2 * y
    area2 = cross.sum()
    if abs(area2) < 1e-9:
        return p.mean(axis=0)
    return np.array([((x + x2) * cross).sum() / (3.0 * area2),
                     ((y + y2) * cross).sum() / (3.0 * area2)])


def polygon_area(pts: np.ndarray) -> float:
    p = np.asarray(pts, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return float(abs((x * np.roll(y, -1) - np.roll(x, -1) * y).sum() / 2.0))


@dataclass
class Part:
    """One detection: a class label plus polygon or rotated-box vertices."""

    label: str
    pts: np.ndarray
    score: float = float("nan")

    def __post_init__(self) -> None:
        self.pts = np.asarray(self.pts, dtype=float).reshape(-1, 2)

    @property
    def centroid(self) -> np.ndarray:
        return polygon_centroid(self.pts)

    @property
    def area(self) -> float:
        return polygon_area(self.pts)

    @property
    def extents(self) -> Tuple[float, float]:
        """(major, minor) oriented extents, via PCA on the vertices."""
        d = self.pts - self.pts.mean(axis=0)
        if len(self.pts) < 2:
            return 0.0, 0.0
        _, _, vt = np.linalg.svd(d, full_matrices=False)
        proj = d @ vt.T
        e = proj.max(axis=0) - proj.min(axis=0)
        return float(e[0]), float(e[1] if len(e) > 1 else 0.0)

    def contains(self, point: np.ndarray) -> bool:
        """Bounding-box containment; the tip lands inside its abdomen 99.8% of the time."""
        p = self.pts
        return bool(p[:, 0].min() <= point[0] <= p[:, 0].max()
                    and p[:, 1].min() <= point[1] <= p[:, 1].max())


def touches_border(part: Part, width: int, height: int, pad: float = 1.5) -> bool:
    """Whether a detection runs into the image border, i.e. is cut off.

    A clipped abdomen is the visible half of a fly whose tip is off-frame. The
    predicted-tip margin alone misses this: it extrapolates along the frame's
    average posterior axis, and a fly angled more steeply than that average
    reaches the border sooner than predicted.
    """
    p = part.pts
    return bool(p[:, 0].min() <= pad or p[:, 1].min() <= pad
                or p[:, 0].max() >= width - pad or p[:, 1].max() >= height - pad)


def eye_scale(eye: Part) -> float:
    """The length unit every other rule is expressed in."""
    return float(np.sqrt(eye.area))


@dataclass
class Fly:
    """One assembled fly. Parts that failed a gate are left as ``None``."""

    eye: Part
    scale: float
    proboscis: Optional[Part] = None
    abdomen: Optional[Part] = None
    tip: Optional[Part] = None
    tip_margin: Optional[float] = None      # predicted-tip clearance, eye_scale units
    eye_ok: bool = True                     # False once the eye itself fails a gate
    rejected: List[str] = field(default_factory=list)
    abdomen_only: bool = False              # accepted by the tip-less fallback; angle not measurable

    @property
    def has_abdomen(self) -> bool:
        return self.abdomen is not None

    @property
    def is_fly(self) -> bool:
        """Whether this eye has any body attached to it.

        A real fly always shows a proboscis about 1.3 eye-widths out (mutual
        nearest-neighbour holds for 98.4% of labelled eyes). An eye with neither
        a proboscis nor an abdomen is a false positive, not a fly -- typically a
        low-confidence detection on clutter at the frame edge. Worth checking
        before a slot is frozen for it, since a frozen slot persists for the
        whole video.
        """
        return self.proboscis is not None or self.abdomen is not None

    @property
    def trackable(self) -> bool:
        """Abdomen usable downstream: present, tipped, and fully in frame."""
        return self.abdomen is not None and self.tip is not None


@dataclass(frozen=True)
class Verdict:
    """Outcome of a single check, carrying why it failed."""

    ok: bool
    reason: str = ""
    value: float = float("nan")

    def __bool__(self) -> bool:
        return self.ok


# --------------------------------------------------------------------------- #
# individual checks -- each usable on its own
# --------------------------------------------------------------------------- #
def check_eye(eye: Part, limits: FlyGeometryLimits = DEFAULT_LIMITS) -> Verdict:
    s = eye_scale(eye)
    if not (limits.eye_scale_min <= s <= limits.eye_scale_max):
        return Verdict(False, f"eye_scale {s:.1f}px outside "
                              f"[{limits.eye_scale_min:.0f},{limits.eye_scale_max:.0f}]", s)
    major, minor = eye.extents
    if minor > 0 and major / minor > limits.eye_aspect_max:
        return Verdict(False, f"eye aspect {major / minor:.2f} > {limits.eye_aspect_max}", s)
    return Verdict(True, "", s)


def check_proboscis(eye: Part, proboscis: Part,
                    limits: FlyGeometryLimits = DEFAULT_LIMITS,
                    scale: Optional[float] = None) -> Verdict:
    """Eye-relative proboscis gate.

    This is the scale-aware counterpart to the absolute
    ``proboscis_filter.max_eye_prob_distance_px`` radius: it accepts a long
    extension from a large fly and rejects the same pixel distance from a small
    one, which a fixed radius cannot do.
    """
    s = float(scale if scale is not None else eye_scale(eye))
    d_px = float(np.linalg.norm(proboscis.centroid - eye.centroid))
    if not np.isfinite(d_px):
        return Verdict(False, "non-finite eye->proboscis distance", d_px)
    if s <= 0:
        ok = d_px <= limits.abs_d_eye_pro_max_px
        return Verdict(ok, "" if ok else f"eye->proboscis {d_px:.0f}px exceeds absolute cap", d_px)
    d = d_px / s
    if d < limits.d_eye_pro_min:
        return Verdict(False, f"eye->proboscis {d:.2f} < {limits.d_eye_pro_min} eye-widths", d)
    if d > limits.d_eye_pro_max:
        return Verdict(False, f"eye->proboscis {d:.2f} > {limits.d_eye_pro_max} eye-widths", d)
    size = np.sqrt(proboscis.area) / s
    if not (limits.pro_size_min <= size <= limits.pro_size_max):
        return Verdict(False, f"proboscis size {size:.2f} outside "
                              f"[{limits.pro_size_min},{limits.pro_size_max}] eye-widths", d)
    return Verdict(True, "", d)


def estimate_posterior_axis(pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
                            prior: Optional[np.ndarray] = None,
                            limits: FlyGeometryLimits = DEFAULT_LIMITS) -> Optional[np.ndarray]:
    """Frame-level head->abdomen direction from candidate (eye_c, abdomen_c) pairs.

    Flies in one frame are mounted alike, so their eye->abdomen vectors agree
    to within ~27 deg (p95). The absolute direction belongs to the rig, so it is
    measured here rather than assumed; pass ``prior`` to seed single-fly frames.
    """
    pairs = list(pairs)
    if not pairs:
        return None if prior is None else _unit(np.asarray(prior, dtype=float))
    U = np.array([_unit(np.asarray(a) - np.asarray(e)) for e, a in pairs])
    u = _unit(np.asarray(prior, dtype=float)) if prior is not None else _unit(U.mean(axis=0))
    for _ in range(3):                       # trim outliers, re-average
        dev = np.degrees(np.arccos(np.clip(U @ u, -1.0, 1.0)))
        keep = U[dev <= limits.dir_tol_deg]
        if not len(keep):
            break
        u = _unit(keep.mean(axis=0))
    return u


def estimate_chirality(triples: Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> float:
    """Sign of cross(eye->proboscis, eye->abdomen), by majority vote.

    Negative in all 23 standard sessions and positive in the mirrored
    flybehavior2 session -- a clean 0%/100% split -- so this doubles as a
    mirrored-rig detector. Returns 0.0 when there is nothing to vote on.
    """
    signs = []
    for eye_c, pro_c, abd_c in triples:
        up = _unit(np.asarray(pro_c) - np.asarray(eye_c))
        ua = _unit(np.asarray(abd_c) - np.asarray(eye_c))
        if not up.any() or not ua.any():
            continue
        signs.append(np.sign(np.cross(up, ua)))
    if not signs:
        return 0.0
    arr = np.asarray(signs, dtype=float)
    return float(np.sign(arr.sum()))


# --------------------------------------------------------------------------- #
# whole-frame assembly
# --------------------------------------------------------------------------- #
def _abdomen_gate(fly: Fly, abd: Part, axis: Optional[np.ndarray], chirality: float,
                  limits: FlyGeometryLimits) -> Optional[Tuple[float, float, np.ndarray]]:
    """Distance / direction / size / proboscis-angle / handedness gates for one
    (fly, abdomen) pair, shared by the tip-paired and the tip-less paths.
    Returns ``(d, ang_axis, v)`` in eye-scale units / degrees / px, or None."""
    v = abd.centroid - fly.eye.centroid
    n_v = float(np.linalg.norm(v))
    if n_v <= 0:
        return None
    d = n_v / fly.scale
    if not (limits.d_eye_abd_min <= d <= limits.d_eye_abd_max):
        return None
    ang_axis = angle_between(v, axis) if axis is not None else 0.0
    if ang_axis > limits.dir_tol_deg:
        return None
    if not (limits.abd_size_min <= np.sqrt(abd.area) / fly.scale <= limits.abd_size_max):
        return None
    if fly.proboscis is not None:
        vp = fly.proboscis.centroid - fly.eye.centroid
        ang_pa = angle_between(vp, v)
        if not (limits.ang_pro_abd_min <= ang_pa <= limits.ang_pro_abd_max):
            return None
        if chirality and np.sign(np.cross(_unit(vp), _unit(v))) != chirality:
            return None      # wrong-handed: the rig does not mount flies this way
    return d, ang_axis, v


def _candidate_triples(flies: List[Fly], abdomens: List[Part], tips: List[Part],
                       axis: Optional[np.ndarray], chirality: float,
                       limits: FlyGeometryLimits) -> List[Tuple[float, int, int, int]]:
    """Every (fly, abdomen, tip) combination that passes all geometric gates."""
    out: List[Tuple[float, int, int, int]] = []
    for i, fly in enumerate(flies):
        if not fly.eye_ok or fly.scale <= 0:
            continue
        for j, abd in enumerate(abdomens):
            gate = _abdomen_gate(fly, abd, axis, chirality, limits)
            if gate is None:
                continue
            d, ang_axis, v = gate
            n_v = float(np.linalg.norm(v))
            for k, tip in enumerate(tips):
                v_t = tip.centroid - fly.eye.centroid
                n_t = float(np.linalg.norm(v_t))
                if not (limits.d_eye_tip_min <= n_t / fly.scale <= limits.d_eye_tip_max):
                    continue
                if not (limits.tip_ratio_min <= n_t / n_v <= limits.tip_ratio_max):
                    continue
                d_at = float(np.linalg.norm(tip.centroid - abd.centroid)) / fly.scale
                if not (limits.d_abd_tip_min <= d_at <= limits.d_abd_tip_max):
                    continue
                if not (limits.tip_size_min <= np.sqrt(tip.area) / fly.scale <= limits.tip_size_max):
                    continue
                # decisive test: the tip continues AWAY from the eye
                if float(np.dot(tip.centroid - abd.centroid, v / n_v)) <= 0:
                    continue
                if angle_between(v_t, v) > limits.tip_axis_tol_deg:
                    continue
                cost = ((0.0 if abd.contains(tip.centroid) else 1.0)
                        + abs(d - _MEDIAN_D_EYE_ABD) + ang_axis / 90.0)
                out.append((cost, i, j, k))
    out.sort()
    return out


def assign_abdomen_only(flies: List[Fly], abdomens: List[Part], width: int, height: int,
                        axis: Optional[np.ndarray], chirality: float,
                        limits: FlyGeometryLimits = DEFAULT_LIMITS) -> int:
    """Tip-less fallback: give each fly still without an abdomen the best unclaimed
    abdomen that passes every abdomen gate and is not clipped by the frame border.
    The angle stays unmeasurable (``Fly.tip`` is None); ``Fly.abdomen_only`` marks
    the row so downstream code can tell the two provenances apart. Returns the
    number of flies that received an abdomen."""
    claimed = {id(f.abdomen) for f in flies if f.abdomen is not None}
    cands: List[Tuple[float, int, int]] = []
    for i, fly in enumerate(flies):
        if fly.abdomen is not None or not fly.eye_ok or fly.scale <= 0:
            continue
        for j, abd in enumerate(abdomens):
            if id(abd) in claimed or touches_border(abd, width, height):
                continue
            gate = _abdomen_gate(fly, abd, axis, chirality, limits)
            if gate is None:
                continue
            d, ang_axis, _ = gate
            cands.append((abs(d - _MEDIAN_D_EYE_ABD) + ang_axis / 90.0, i, j))
    cands.sort()
    used_f: set = set()
    used_a: set = set()
    n = 0
    for _, i, j in cands:
        if i in used_f or j in used_a:
            continue
        used_f.add(i)
        used_a.add(j)
        flies[i].abdomen = abdomens[j]
        flies[i].abdomen_only = True
        n += 1
    return n


def assemble(parts: Iterable[Part], width: int, height: int,
             limits: FlyGeometryLimits = DEFAULT_LIMITS,
             prior_axis: Optional[np.ndarray] = None,
             prior_chirality: float = 0.0,
             abdomen_only_fallback: bool = False) -> List[Fly]:
    """Group detections into flies, withholding parts that fail the geometry.

    Works with whatever classes are present: given only eyes and proboscises
    (the current two-class pipeline) it applies the eye and proboscis gates and
    leaves ``abdomen``/``tip`` as ``None``. Nothing is fabricated or moved --
    a part that fails is dropped, and ``Fly.rejected`` says why.

    ``prior_chirality`` seeds the handedness gate when a frame cannot vote on
    it (a single fly); ``abdomen_only_fallback`` records a gated, border-clear
    abdomen box for flies whose tip was not found (``Fly.abdomen_only``).
    """
    parts = list(parts)
    eyes = [p for p in parts if p.label == EYE]
    pros = [p for p in parts if p.label == PROBOSCIS]
    abds = [p for p in parts if p.label == ABDOMEN]
    tips = [p for p in parts if p.label == ABDOMEN_TIP]

    flies: List[Fly] = []
    for eye in eyes:
        verdict = check_eye(eye, limits)
        fly = Fly(eye=eye, scale=verdict.value if np.isfinite(verdict.value) else 0.0)
        if not verdict.ok:
            fly.eye_ok = False
            fly.rejected.append(verdict.reason)
        flies.append(fly)

    # proboscis: nearest eye wins, then the eye-relative gate decides
    for fly in flies:
        if fly.scale <= 0:
            continue
        best: Optional[Tuple[float, Part]] = None
        for pro in pros:
            d = float(np.linalg.norm(pro.centroid - fly.eye.centroid))
            if best is None or d < best[0]:
                best = (d, pro)
        if best is None:
            continue
        verdict = check_proboscis(fly.eye, best[1], limits, scale=fly.scale)
        if verdict.ok:
            fly.proboscis = best[1]
        else:
            fly.rejected.append(verdict.reason)

    if not abds and not tips:
        # Two-class flow (the current pipeline model emits no abdomen classes):
        # stay quiet rather than blaming every fly for a part that was never
        # predicted. Abdomens present WITHOUT tips still fall through below, so
        # a tipless abdomen is withheld with a reason.
        return flies

    # Pass 1 runs without a direction gate -- the tip rules alone pair flies
    # reliably, and a distance-only axis seed is worst in exactly the frames
    # that matter. Pass 2 applies the gate using an axis measured from what
    # pass 1 accepted, which is self-consistent.
    axis = None if prior_axis is None else _unit(np.asarray(prior_axis, dtype=float))
    chirality = float(prior_chirality)
    for pass_no in range(2):
        for fly in flies:
            fly.abdomen = fly.tip = None
        used_e, used_a, used_t = set(), set(), set()
        for _, i, j, k in _candidate_triples(flies, abds, tips, axis, chirality, limits):
            if i in used_e or j in used_a or k in used_t:
                continue
            used_e.add(i); used_a.add(j); used_t.add(k)
            flies[i].abdomen, flies[i].tip = abds[j], tips[k]
        if pass_no == 0:
            accepted = [(f.eye.centroid, f.abdomen.centroid) for f in flies if f.abdomen]
            if not accepted:
                break                      # nothing to learn from; keep pass-1 result
            axis = estimate_posterior_axis(accepted, prior=prior_axis, limits=limits)
            vote = estimate_chirality(
                (f.eye.centroid, f.proboscis.centroid, f.abdomen.centroid)
                for f in flies if f.abdomen is not None and f.proboscis is not None)
            chirality = vote if vote else float(prior_chirality)

    # Visibility: could this fly's tip be in frame at all? A fly whose predicted
    # tip falls outside the image is a half-fly and must not be tracked.
    for fly in flies:
        if axis is not None and fly.scale > 0:
            p = fly.eye.centroid + axis * limits.posterior_reach * fly.scale
            fly.tip_margin = float(
                min(p[0], width - p[0], p[1], height - p[1]) / fly.scale)
        if fly.eye_ok and not fly.is_fly:
            fly.rejected.append(
                "eye has neither a proboscis nor an abdomen - not a fly")
        if fly.abdomen is None and fly.eye_ok:
            # Always say why, and prefer the specific diagnosis. An in-range
            # abdomen that is clipped by the border is a half-visible fly, which
            # is a different failure from geometry that simply did not match.
            clipped = any(
                touches_border(abd, width, height)
                and limits.d_eye_abd_min
                <= float(np.linalg.norm(abd.centroid - fly.eye.centroid)) / fly.scale
                <= limits.d_eye_abd_max
                for abd in abds) if fly.scale > 0 else False
            if clipped:
                fly.rejected.append(
                    "abdomen is clipped by the frame edge - its tip is out of "
                    "frame, so the fly is not trackable")
            elif fly.tip_margin is not None and fly.tip_margin < limits.edge_margin:
                fly.rejected.append(
                    f"predicted tip {fly.tip_margin:.1f} eye-widths outside frame - "
                    "abdomen not fully visible")
            else:
                fly.rejected.append("no (abdomen, tip) pair passed the geometry gates")
    if abdomen_only_fallback and abds:
        assign_abdomen_only(flies, abds, width, height, axis, chirality, limits)
    return flies


def summarise(flies: Sequence[Fly]) -> Dict[str, int]:
    """Counts for logging: how many flies kept each part."""
    return {
        "flies": len(flies),
        "eye_ok": sum(1 for f in flies if f.eye_ok),
        "with_proboscis": sum(1 for f in flies if f.proboscis is not None),
        "with_abdomen": sum(1 for f in flies if f.abdomen is not None),
        "trackable": sum(1 for f in flies if f.trackable),
    }
