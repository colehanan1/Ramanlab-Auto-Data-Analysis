"""Mount QC: is every fly framed well enough to be scored?

Pure analysis shared by the live rig-monitor mount check (src/rigmon/mountcheck)
and the pipeline's dataset setup-QC figures. Input: eye/proboscis detections of
a set of frames from one rig. Output: per-fly-slot stats and plain-language
flags. No I/O, no GPU.

Slots are numbered top to bottom (fly 1 = top). A proboscis is paired with an
eye using the production gate shape: the production quarter-ellipse (``anisotropic_semi_axes``): semi-axis
``gate_px`` left, right and down, ``gate_px / up_divisor`` upward.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np

from .distance_sanity import anisotropic_semi_axes

EYE, PROBOSCIS = "eye", "proboscis"
SIDES = ("top", "bottom", "left", "right")


@dataclass(frozen=True)
class Detection:
    cls: str
    x: float
    y: float
    w: float
    h: float
    r: float = 0.0
    conf: float = 1.0

    @classmethod
    def from_dict(cls, d: dict) -> "Detection":
        return cls(str(d["cls"]), float(d["x"]), float(d["y"]), float(d["w"]),
                   float(d["h"]), float(d.get("r", 0.0)), float(d.get("conf", 1.0)))

    def extent(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounds (x0, y0, x1, y1) of the rotated box."""
        c, s = abs(math.cos(self.r)), abs(math.sin(self.r))
        hw = (self.w * c + self.h * s) / 2
        hh = (self.w * s + self.h * c) / 2
        return self.x - hw, self.y - hh, self.x + hw, self.y + hh


@dataclass(frozen=True)
class Thresholds:
    expected_flies: int = 4
    missing_eye_frac: float = 0.20
    proboscis_min_frac: float = 0.50
    edge_margin_px: float = 40.0
    bottom_room_px: float = 120.0
    wobble_px: float = 15.0
    min_eye_conf: float = 0.6
    sharpness_ratio: float = 0.6
    slot_gap_px: float = 60.0
    min_slot_support: float = 0.05


@dataclass
class SlotStats:
    slot: int
    eye_frac: float
    proboscis_frac: float
    eye_xy: tuple[float, float]
    proboscis_xy: Optional[tuple[float, float]]
    eye_margins: dict
    proboscis_margins: Optional[dict]
    eye_proboscis_px: Optional[float]
    eye_proboscis_spread_px: Optional[float]
    wobble_px: float
    eye_conf: float
    sharpness: Optional[float] = None


@dataclass(frozen=True)
class Flag:
    code: str
    slot: Optional[int]
    part: Optional[str]
    value: Optional[float]
    message: str


@dataclass
class MountReport:
    width: int
    height: int
    frames: int
    gate_px: float
    slots: list
    flags: list
    eye_conf_mean: Optional[float]

    @property
    def ok(self) -> bool:
        return not self.flags

    def to_dict(self) -> dict:
        return asdict(self)


def in_gate(eye: Detection, prob: Detection, gate_px: float, up_divisor: float) -> bool:
    dx, dy = prob.x - eye.x, prob.y - eye.y
    a, b = anisotropic_semi_axes(dx, dy, gate_px, up_divisor)
    return (dx / float(a)) ** 2 + (dy / float(b)) ** 2 <= 1.0


def find_slots(frames: Sequence[Sequence[Detection]], th: Thresholds) -> list[float]:
    """Eye y-centres of the fly slots, top to bottom (at most ``expected_flies``)."""
    ys = sorted(d.y for f in frames for d in f if d.cls == EYE)
    if not ys:
        return []
    groups = [[ys[0]]]
    for y in ys[1:]:
        if y - groups[-1][-1] > th.slot_gap_px:
            groups.append([y])
        else:
            groups[-1].append(y)
    n = max(1, len(frames))
    groups = [g for g in groups if len(g) / n >= th.min_slot_support]
    groups = sorted(groups, key=len, reverse=True)[: th.expected_flies]
    return sorted(float(np.median(g)) for g in groups)


def _assign_eyes(frame, centers, th) -> dict:
    out: dict[int, Detection] = {}
    for d in frame:
        if d.cls != EYE or not centers:
            continue
        k = min(range(len(centers)), key=lambda i: abs(d.y - centers[i]))
        if abs(d.y - centers[k]) > th.slot_gap_px:
            continue
        if k not in out or d.conf > out[k].conf:
            out[k] = d
    return out


def _pair(frame, eyes: dict, gate_px: float, up_divisor: float) -> dict:
    probs = [d for d in frame if d.cls == PROBOSCIS]
    cands = sorted(
        (math.hypot(p.x - e.x, p.y - e.y), k, i)
        for k, e in eyes.items() for i, p in enumerate(probs)
        if in_gate(e, p, gate_px, up_divisor))
    out: dict[int, Detection] = {}
    used: set[int] = set()
    for _, k, i in cands:
        if k in out or i in used:
            continue
        out[k] = probs[i]
        used.add(i)
    return out


def _margins(d: Detection, width: int, height: int) -> dict:
    x0, y0, x1, y1 = d.extent()
    return {"top": max(0.0, y0), "bottom": max(0.0, height - y1),
            "left": max(0.0, x0), "right": max(0.0, width - x1)}


def _robust_min(values) -> float:
    return float(np.percentile(values, 10))


def region_sharpness(gray: np.ndarray, x: float, y: float, half: int = 48) -> float:
    """Variance of the Laplacian in a window around (x, y) — higher is sharper."""
    import cv2

    h, w = gray.shape[:2]
    x0, x1 = max(0, int(x) - half), min(w, int(x) + half)
    y0, y1 = max(0, int(y) - half), min(h, int(y) + half)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return 0.0
    return float(cv2.Laplacian(gray[y0:y1, x0:x1], cv2.CV_64F).var())


def analyze(frames, width: int, height: int, *, gate_px: float, up_divisor: float = 4.0,
            th: Thresholds = Thresholds(), sharpness: Optional[dict] = None,
            sharpness_reference: Optional[float] = None) -> MountReport:
    n = len(frames)
    if n == 0:
        return MountReport(width, height, 0, gate_px, [],
                           [Flag("camera", None, None, None, "camera stream not reachable (no frames)")],
                           None)
    centers = find_slots(frames, th)
    eyes_by: dict[int, list] = {k: [] for k in range(len(centers))}
    pairs_by: dict[int, list] = {k: [] for k in range(len(centers))}
    short = 0
    for f in frames:
        eyes = _assign_eyes(f, centers, th)
        if len(eyes) < th.expected_flies:
            short += 1
        for k, p in _pair(f, eyes, gate_px, up_divisor).items():
            pairs_by[k].append((eyes[k], p))
        for k, e in eyes.items():
            eyes_by[k].append(e)

    slots: list[SlotStats] = []
    for k in range(len(centers)):
        es, ps = eyes_by[k], pairs_by[k]
        ex, ey = float(np.median([e.x for e in es])), float(np.median([e.y for e in es]))
        # median of a 2-D Gaussian's radial distance is 1.1774 sigma (Rayleigh) -> per-axis sigma
        wob = float(np.median([math.hypot(e.x - ex, e.y - ey) for e in es])) / 1.1774
        em = {s: _robust_min([_margins(e, width, height)[s] for e in es]) for s in SIDES}
        if ps:
            pm = {s: _robust_min([_margins(p, width, height)[s] for _, p in ps]) for s in SIDES}
            dists = [math.hypot(p.x - e.x, p.y - e.y) for e, p in ps]
            pxy = (float(np.median([p.x for _, p in ps])), float(np.median([p.y for _, p in ps])))
            dmed = float(np.median(dists))
            spread = float(np.percentile(dists, 75) - np.percentile(dists, 25))
        else:
            pm, pxy, dmed, spread = None, None, None, None
        slots.append(SlotStats(k + 1, len(es) / n, len(ps) / n, (ex, ey), pxy, em, pm, dmed, spread,
                               wob, float(np.mean([e.conf for e in es])),
                               (sharpness or {}).get(k + 1)))

    confs = [e.conf for es in eyes_by.values() for e in es]
    conf_mean = float(np.mean(confs)) if confs else None
    return MountReport(width, height, n, gate_px, slots,
                       _flags(slots, len(centers), short / n, conf_mean, th, sharpness,
                              sharpness_reference),
                       conf_mean)


def _flags(slots, n_slots, short_frac, conf_mean, th, sharpness, reference) -> list:
    flags: list[Flag] = []
    if n_slots < th.expected_flies:
        flags.append(Flag("flies_visible", None, EYE, float(n_slots),
                          f"only {n_slots} flies visible (expected {th.expected_flies})"))
    elif short_frac > th.missing_eye_frac:
        worst = min(slots, key=lambda s: s.eye_frac)
        flags.append(Flag("flies_visible", worst.slot, EYE, short_frac,
                          f"fewer than {th.expected_flies} eyes in {short_frac * 100:.0f}% of frames "
                          f"(fly {worst.slot} missing in {(1 - worst.eye_frac) * 100:.0f}% of frames)"))
    for s in slots:
        if s.proboscis_frac < th.proboscis_min_frac:
            flags.append(Flag("proboscis_missing", s.slot, PROBOSCIS, s.proboscis_frac,
                              f"fly {s.slot}: proboscis not detected "
                              f"({s.proboscis_frac * 100:.0f}% of frames)"))
        room_flagged = False
        if s.proboscis_margins and s.proboscis_margins["bottom"] < th.bottom_room_px:
            v = s.proboscis_margins["bottom"]
            flags.append(Flag("bottom_room", s.slot, PROBOSCIS, v,
                              f"fly {s.slot}: proboscis {v:.0f} px from bottom — PER may leave frame"))
            room_flagged = True
        for part, m in ((EYE, s.eye_margins), (PROBOSCIS, s.proboscis_margins)):
            if not m:
                continue
            for side in SIDES:
                if part == PROBOSCIS and side == "bottom" and room_flagged:
                    continue
                if m[side] < th.edge_margin_px:
                    flags.append(Flag("edge_margin", s.slot, part, m[side],
                                      f"fly {s.slot}: {part} {m[side]:.0f} px from {side} edge"))
        if s.wobble_px > th.wobble_px:
            flags.append(Flag("wobble", s.slot, EYE, s.wobble_px,
                              f"fly {s.slot}: head moving a lot ({s.wobble_px:.0f} px) — loose mount?"))
    if conf_mean is not None and conf_mean < th.min_eye_conf:
        flags.append(Flag("image_quality", None, None, conf_mean,
                          f"image looks blurry or dark (eye confidence {conf_mean:.2f})"))
    elif reference and sharpness:
        overall = float(np.median(list(sharpness.values())))
        if overall < th.sharpness_ratio * reference:
            flags.append(Flag("image_quality", None, None, overall,
                              f"image looks blurry or dark (sharpness {overall:.0f} vs usual {reference:.0f})"))
    return flags
