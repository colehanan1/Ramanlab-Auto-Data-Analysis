"""Batch-level eye anchor template: one fly slot set for all a batch's trials.

``yolo_infer`` sizes each trial from its own first 5 frames (``_scan_initial_
fly_count``). A fly whose eye sits at the frame edge is detected in some
trials and missed in others, and because slots are numbered top-to-bottom the
miss does not just drop that fly -- it renumbers every fly below it. In
``IAA-Sensitivity-6-1/october_09_batch_3`` the top fly's eye centre is 4 px
from the top edge: found in 3 of 21 trials, so ``fly1`` means the edge fly in
three trials and the fly below it in the other eighteen.

The template settles that once per batch. Every trial's first frames vote; a
slot joins the template when at least ``min_trials`` trials saw it (so one
spurious detection cannot invent a fly, and one lucky detection cannot
renumber a batch). Applying it, a trial keeps its own anchor wherever it
detected one -- anchors settle ~10 px over a session, and freezing them at
the template position would read as a standing ``eye_drift_px`` on every fly
-- and borrows the template's only for the slots it missed.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .multi_fly import match_live_eyes
from .vision import xyxy_to_cxcywh

#: Cache file written in the batch directory. Inspectable, and the per-batch
#: pre-pass is shared by every worker slicing that batch.
TEMPLATE_FILENAME = "eye_anchor_template.json"

#: Cache format. Bumped when the payload's meaning changes so stale files are
#: rebuilt rather than misread.
TEMPLATE_VERSION = 1

#: Detections within this many px are the same eye even when the flies are
#: mounted as close as they ever get. Comfortably above the ~10 px an anchor
#: settles over a session, far below the closest mount spacing seen (176 px).
DEFAULT_CLUSTER_TOL_MIN_PX = 25.0

#: Once provisional clusters exist, the tolerance grows to this fraction of
#: the closest slot-to-slot spacing. Same 0.45 guard ``match_live_eyes`` uses:
#: under half the spacing, so no detection can reach two separate slots.
DEFAULT_CLUSTER_TOL_FRAC = 0.45


@dataclass(frozen=True)
class TemplateSlot:
    """One fly slot: where its eye sits and which trials saw it there."""

    box: Tuple[float, float, float, float]
    n_trials_seen: int
    trials: Tuple[str, ...]
    #: Trials that paired a proboscis inside this eye's gate. An eye alone is
    #: not a measurable fly: june_03_batch_2's bottom-edge eye was detected in
    #: 2 of 15 trials but its proboscis never entered the frame.
    n_trials_paired: int = 0

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2.0

    @property
    def center(self) -> Tuple[float, float]:
        return (self.cx, self.cy)

    def box_array(self) -> np.ndarray:
        return np.asarray(self.box, dtype=np.float32)


def _centre(box) -> Tuple[float, float]:
    cx, cy, _, _ = xyxy_to_cxcywh(np.asarray(box, dtype=np.float32))
    return (float(cx), float(cy))


def _cluster(points: Sequence[Tuple[str, np.ndarray, bool]], tol_px: float) -> List[List[int]]:
    """Leader-cluster ``points`` against running centroids.

    Points arrive sorted top-to-bottom, and a point joins a cluster only when
    it is within ``tol_px`` of that cluster's *centroid* -- never of its
    nearest member -- so a stray detection between two flies cannot chain them
    into one slot.
    """
    clusters: List[List[int]] = []
    centroids: List[Tuple[float, float]] = []
    for idx, point in enumerate(points):
        box = point[1]
        cx, cy = _centre(box)
        best = -1
        best_dist = tol_px
        for c_idx, (gx, gy) in enumerate(centroids):
            dist = math.hypot(cx - gx, cy - gy)
            if dist <= best_dist:
                best, best_dist = c_idx, dist
        if best < 0:
            clusters.append([idx])
            centroids.append((cx, cy))
            continue
        clusters[best].append(idx)
        members = [_centre(points[i][1]) for i in clusters[best]]
        centroids[best] = (
            float(np.mean([m[0] for m in members])),
            float(np.mean([m[1] for m in members])),
        )
    return clusters


def _min_centroid_spacing(points, clusters: List[List[int]]) -> Optional[float]:
    if len(clusters) < 2:
        return None
    centres = [
        (
            float(np.mean([_centre(points[i][1])[0] for i in c])),
            float(np.mean([_centre(points[i][1])[1] for i in c])),
        )
        for c in clusters
    ]
    return min(
        math.hypot(bx - ax, by - ay)
        for i, (ax, ay) in enumerate(centres)
        for bx, by in centres[i + 1 :]
    )


def build_template(
    trial_boxes: Mapping[str, np.ndarray],
    *,
    min_trials: int = 2,
    cluster_tol_min_px: float = DEFAULT_CLUSTER_TOL_MIN_PX,
    cluster_tol_frac: float = DEFAULT_CLUSTER_TOL_FRAC,
    trial_paired: Optional[Mapping[str, Sequence[bool]]] = None,
    require_proboscis: bool = False,
) -> List[TemplateSlot]:
    """Build a batch's slot set from each trial's first-frame eye boxes.

    ``trial_boxes`` maps a trial name to the ``(N, 4)`` eye boxes its warm-up
    scan found. The result is ordered top-to-bottom by the same
    ``np.lexsort((xs, ys))`` convention ``EyeAnchorManager`` uses, so slot *i*
    is fly *i+1* in every trial of the batch.

    ``trial_paired`` says, per box, whether a proboscis sat inside that eye's
    gate. With ``require_proboscis``, a slot that never paired one in any trial
    is dropped: its eye is clipped by the frame edge and its proboscis is out
    of shot, so it could only ever contribute an empty trace. A caller that
    omits ``trial_paired`` keeps every slot, so an old cache or a stand-in
    cannot silently lose a batch's flies.
    """
    points: List[Tuple[str, np.ndarray, bool]] = []
    for trial in sorted(trial_boxes):
        boxes = np.asarray(trial_boxes[trial], dtype=np.float32).reshape(-1, 4)
        flags = list((trial_paired or {}).get(trial, []))
        for idx, box in enumerate(boxes):
            paired = bool(flags[idx]) if idx < len(flags) else False
            points.append((trial, box, paired))
    if not points:
        return []

    # Deterministic order: the template must not depend on which worker, or
    # which slice of the batch, happened to build it.
    points.sort(key=lambda p: (_centre(p[1])[1], _centre(p[1])[0], p[0]))

    provisional = _cluster(points, cluster_tol_min_px)
    spacing = _min_centroid_spacing(points, provisional)
    tol = cluster_tol_min_px
    if spacing is not None:
        tol = max(cluster_tol_min_px, cluster_tol_frac * spacing)
    clusters = _cluster(points, tol)

    have_pairing = trial_paired is not None
    slots: List[TemplateSlot] = []
    for cluster in clusters:
        trials = sorted({points[i][0] for i in cluster})
        if len(trials) < min_trials:
            continue
        paired_trials = sorted({points[i][0] for i in cluster if points[i][2]})
        if require_proboscis and have_pairing and not paired_trials:
            continue
        box = np.median(
            np.stack([points[i][1] for i in cluster], axis=0).astype(np.float64), axis=0
        )
        slots.append(
            TemplateSlot(
                box=tuple(float(v) for v in box),
                n_trials_seen=len(trials),
                trials=tuple(trials),
                n_trials_paired=len(paired_trials),
            )
        )

    if not slots:
        return []
    order = np.lexsort(([s.cx for s in slots], [s.cy for s in slots]))
    return [slots[i] for i in order]


def seed_anchor_boxes(
    slots: Sequence[TemplateSlot],
    det_xyxy: np.ndarray,
    match_radius_px: float,
) -> Tuple[List[np.ndarray], List[bool]]:
    """Fill every template slot for one trial.

    Returns the anchor box per slot and, alongside it, whether that box was
    borrowed from the template. A slot takes the trial's own detection when
    one lies in range -- assignment is one-to-one and neighbour-clamped by
    ``match_live_eyes`` -- and the template's box only when none does. A
    detection matching no slot is dropped: the template is authoritative, so a
    stray eye cannot add a fly partway through a batch.
    """
    boxes = np.asarray(det_xyxy, dtype=np.float32).reshape(-1, 4)
    matched = match_live_eyes([s.center for s in slots], boxes, match_radius_px)

    by_centre: Dict[Tuple[float, float], int] = {}
    for idx, box in enumerate(boxes):
        by_centre.setdefault(_centre(box), idx)

    seeded: List[np.ndarray] = []
    borrowed: List[bool] = []
    for slot, centre in zip(slots, matched):
        det_idx = by_centre.get(centre) if centre is not None else None
        if det_idx is None:
            seeded.append(slot.box_array())
            borrowed.append(True)
        else:
            seeded.append(boxes[det_idx].copy())
            borrowed.append(False)
    return seeded, borrowed


# ── Cache ────────────────────────────────────────────────────────────────────


def template_path(batch_dir: Path) -> Path:
    return Path(batch_dir) / TEMPLATE_FILENAME


def save_template(path: Path, slots: Sequence[TemplateSlot], min_trials: int) -> None:
    """Write the template atomically so concurrent workers never read a
    half-written file."""
    payload = {
        "version": TEMPLATE_VERSION,
        "min_trials": int(min_trials),
        "slots": [
            {
                "box": list(s.box),
                "cx": s.cx,
                "cy": s.cy,
                "n_trials_seen": s.n_trials_seen,
                "n_trials_paired": s.n_trials_paired,
                "trials": list(s.trials),
            }
            for s in slots
        ],
    }
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_template(
    path: Path, min_trials: Optional[int] = None
) -> Optional[List[TemplateSlot]]:
    """Read a cached template, or ``None`` when there is nothing usable.

    A missing, corrupt, or differently-parameterised cache reads as ``None``
    so the caller rebuilds rather than seeding from a stale slot set.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != TEMPLATE_VERSION:
        return None
    if min_trials is not None and payload.get("min_trials") != int(min_trials):
        return None
    try:
        return [
            TemplateSlot(
                box=tuple(float(v) for v in s["box"]),
                n_trials_seen=int(s["n_trials_seen"]),
                n_trials_paired=int(s.get("n_trials_paired", 0)),
                trials=tuple(str(t) for t in s["trials"]),
            )
            for s in payload["slots"]
        ]
    except (KeyError, TypeError, ValueError):
        return None
