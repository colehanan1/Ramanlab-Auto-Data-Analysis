from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .vision import iou, xyxy_to_cxcywh

ZERO_IOU_EPS = 1e-8


def _pairwise_zero_iou(new_box: np.ndarray, existing: List[np.ndarray], eps: float) -> bool:
    if not existing:
        return True
    existing_stack = np.stack(existing, axis=0)
    ious = iou(existing_stack, new_box.reshape(1, 4))[:, 0]
    return bool(np.all(ious <= eps))


def enforce_zero_iou_and_topk(
    boxes: np.ndarray,
    scores: np.ndarray,
    k: int,
    eps: float = ZERO_IOU_EPS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return up to ``k`` boxes with pairwise IoU <= ``eps``."""

    if boxes.shape[0] == 0 or k <= 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)

    order = np.argsort(-scores)
    kept_boxes: List[np.ndarray] = []
    kept_scores: List[float] = []
    for idx in order:
        if len(kept_boxes) >= k:
            break
        cand = boxes[idx]
        if _pairwise_zero_iou(cand, kept_boxes, eps):
            kept_boxes.append(cand.copy())
            kept_scores.append(float(scores[idx]))

    if not kept_boxes:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)

    return np.stack(kept_boxes, axis=0).astype(np.float32), np.array(kept_scores, dtype=np.float32)


class EyeAnchorManager:
    """Lock the positions of up to ``max_eyes`` eye detections."""

    def __init__(self, max_eyes: int = 4, zero_iou_eps: float = ZERO_IOU_EPS):
        self.max_eyes = max_eyes
        self.zero_iou_eps = zero_iou_eps
        self.anchors_xyxy: List[np.ndarray] = []
        self.anchor_ids: List[int] = []
        self._next_eye_id = 1001
        self.confirmed = False

    def try_update_from_dets(self, det_xyxy: np.ndarray, det_scores: np.ndarray) -> None:
        if self.confirmed or det_xyxy.shape[0] == 0:
            return

        order = np.argsort(-det_scores)
        added = 0
        for idx in order:
            if len(self.anchors_xyxy) >= self.max_eyes:
                break
            cand = det_xyxy[idx]
            if _pairwise_zero_iou(cand, self.anchors_xyxy, self.zero_iou_eps):
                self.anchors_xyxy.append(cand.copy())
                self.anchor_ids.append(self._next_eye_id)
                self._next_eye_id += 1
                added += 1

        if len(self.anchors_xyxy) >= self.max_eyes:
            centers = [xyxy_to_cxcywh(b)[:2] for b in self.anchors_xyxy]
            # Flies are stacked vertically: number slots top-to-bottom (by y),
            # with x as a deterministic tie-break when two eyes share a row.
            xs = [c[0] for c in centers]
            ys = [c[1] for c in centers]
            order = np.lexsort((xs, ys))
            self.anchors_xyxy = [self.anchors_xyxy[i] for i in order]
            self.anchor_ids = [self.anchor_ids[i] for i in order]
            self.confirmed = True

    def seed_from_boxes(self, boxes: Sequence[np.ndarray]) -> None:
        """Freeze the anchors at ``boxes``, in the order given.

        Used by the batch-level eye template: the slot set and its order come
        from every trial in the batch, not from whichever eyes this one video
        happened to show first. The manager is confirmed on the spot, so a
        trial that never detects one of its flies still carries that slot and
        no later frame can renumber the rest.
        """
        self.anchors_xyxy = [np.asarray(b, dtype=np.float32).reshape(4).copy() for b in boxes]
        self.anchor_ids = [self._next_eye_id + i for i in range(len(self.anchors_xyxy))]
        self._next_eye_id += len(self.anchors_xyxy)
        self.confirmed = True

    def get_centers(self) -> List[Tuple[float, float]]:
        centers: List[Tuple[float, float]] = []
        for b in self.anchors_xyxy:
            cx, cy, _, _ = xyxy_to_cxcywh(b)
            centers.append((float(cx), float(cy)))
        return centers

    def draw(self, frame) -> None:
        try:
            import cv2  # local import to keep module importable without OpenCV
        except Exception:
            return  # silently skip drawing if OpenCV is not installed

        centers = self.get_centers()
        for idx, b in enumerate(self.anchors_xyxy):
            x1, y1, x2, y2 = b.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cx, cy = map(int, centers[idx])
            cv2.putText(
                frame,
                f"eye{idx} id={self.anchor_ids[idx]}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)


def match_live_eyes(
    anchor_centers: Sequence[Tuple[float, float]],
    det_xyxy: np.ndarray,
    max_radius_px: float,
) -> List[Optional[Tuple[float, float]]]:
    """Find each frozen eye anchor's eye detection in the current frame.

    Returns one entry per anchor: the centre of the detection matched to it, or
    ``None`` when none is in range. QC only -- the anchors themselves never move.
    An anchor searches ``max_radius_px``, clamped to 0.45x the distance to its
    nearest other anchor, so no detection is in range of two separate anchors
    and a slot whose own eye is missing cannot pick up its neighbour's. Pairs
    are assigned globally, nearest first, one-to-one.
    """
    matched: List[Optional[Tuple[float, float]]] = [None] * len(anchor_centers)
    boxes = np.asarray(det_xyxy, dtype=np.float32).reshape(-1, 4)
    if not matched or boxes.shape[0] == 0:
        return matched

    anchors = [(float(x), float(y)) for x, y in anchor_centers]
    radii: List[float] = []
    for i, (ax, ay) in enumerate(anchors):
        radius = float(max_radius_px)
        for j, (bx, by) in enumerate(anchors):
            if j != i:
                radius = min(radius, 0.45 * math.hypot(bx - ax, by - ay))
        radii.append(radius)

    centers = [(float(c[0]), float(c[1])) for c in (xyxy_to_cxcywh(b) for b in boxes)]
    pairs: List[Tuple[float, int, int]] = []
    for i, (ax, ay) in enumerate(anchors):
        for j, (cx, cy) in enumerate(centers):
            dist = math.hypot(cx - ax, cy - ay)
            if dist <= radii[i]:
                pairs.append((dist, i, j))
    pairs.sort()  # nearest first; ties go to the lower anchor, then detection, index

    used: set = set()
    for _, i, j in pairs:
        if matched[i] is None and j not in used:
            matched[i] = centers[j]
            used.add(j)
    return matched


class StablePairing:
    """Maintain one-to-one pairings between frozen eyes and proboscis tracks."""

    #: Cost-mode hysteresis: a track an eye already holds costs this much less
    #: for that eye, so a proboscis near the boundary between two close flies
    #: does not flicker between them -- but a clearly better owner still wins.
    #: Not much less than this: an unpaired eye costs 1 and an abandoned track
    #: nothing, so at 0.1 a lower eye took the upper fly's deep PER as its tip
    #: came within ~30 px above it (178 px mount), and two flies both extended
    #: ~250 px swapped one PER for none. Well under ~0.6: a lower fly's
    #: proboscis sits about spacing / 280 deeper in the upper eye's gate
    #: (0.58-0.63 at the closest mounts seen, 176-178 px), and a binding that
    #: wrong must still be handed over (october_07_batch_2_pretest_5).
    STICKY_BONUS = 0.4
    #: Cost mode: tracks whose centres sit this close are one physical proboscis
    #: the tracker briefly followed twice; only one of them may be paired.
    DUPLICATE_PX = 15.0

    def __init__(self, max_pairs: int = 4, rebind_max_dist_px: float = 220.0):
        self.eye_to_cls8: Dict[int, Optional[int]] = {}
        self.max_pairs = max_pairs
        self.rebind_max_dist_px = float(rebind_max_dist_px)

    def step(
        self,
        eye_ids: List[int],
        eye_centers: List[Tuple[float, float]],
        cls8_tracks,
        accept: Optional[Callable[[Tuple[float, float], Tuple[float, float]], bool]] = None,
        cost: Optional[Callable[[Tuple[float, float], Tuple[float, float]], float]] = None,
    ) -> None:
        """Advance the pairing one frame.

        ``accept(eye_center, track_center)`` is the eye->proboscis gate. When
        given, a binding the gate now rejects is released *before* assignment, so
        the eye that actually owns the track can claim it this same frame, and no
        eye ever binds a track outside its gate.

        ``cost(eye_center, track_center)`` switches to joint assignment: how deep
        the track sits in that eye's gate (0 at the eye, 1 at the edge, inf
        outside). Every frame all eyes are paired at once, minimising the total
        cost, with an unpaired eye costing 1. When flies are mounted close their
        gates overlap, and visiting eyes in order let the upper eye keep the
        lower fly's proboscis; jointly, each track goes to the eye it sits
        clearly deepest in, holding a track being worth ``STICKY_BONUS``.
        """
        if not eye_ids:
            return

        live_cls8_ids = {t.id for t in cls8_tracks}
        track_centers = {t.id: xyxy_to_cxcywh(t.box_xyxy)[:2] for t in cls8_tracks}
        center_by_eye = dict(zip(eye_ids, eye_centers))

        if cost is not None:
            self._assign_jointly(eye_ids, eye_centers, track_centers, cost)
            return

        for eye_id in list(self.eye_to_cls8.keys()):
            cls8_id = self.eye_to_cls8[eye_id]
            if cls8_id is not None and cls8_id not in live_cls8_ids:
                self.eye_to_cls8[eye_id] = None
            elif (
                cls8_id is not None
                and accept is not None
                and eye_id in center_by_eye
                and not accept(center_by_eye[eye_id], tuple(track_centers[cls8_id]))
            ):
                self.eye_to_cls8[eye_id] = None

        claimed_live_ids = {
            cls8_id
            for cls8_id in self.eye_to_cls8.values()
            if cls8_id is not None and cls8_id in live_cls8_ids
        }

        used_this_step = set(claimed_live_ids)

        for idx, eye_id in enumerate(eye_ids):
            cls8_id = self.eye_to_cls8.get(eye_id, None)
            if cls8_id is not None and cls8_id in live_cls8_ids:
                used_this_step.add(cls8_id)
                continue

            if eye_id not in self.eye_to_cls8:
                self.eye_to_cls8[eye_id] = None

            ex, ey = eye_centers[idx]
            best_id: Optional[int] = None
            best_dist: Optional[float] = None
            for track_id, (cx, cy) in track_centers.items():
                if track_id in used_this_step:
                    continue
                if accept is not None and not accept((ex, ey), (cx, cy)):
                    continue
                dist = math.hypot(cx - ex, cy - ey)
                if best_dist is None or dist < best_dist:
                    best_id, best_dist = track_id, dist

            if best_id is not None and best_dist is not None and best_dist <= self.rebind_max_dist_px:
                self.eye_to_cls8[eye_id] = best_id
                used_this_step.add(best_id)

    def _assign_jointly(self, eye_ids, eye_centers, track_centers, cost) -> None:
        from scipy.optimize import linear_sum_assignment

        track_ids = list(track_centers)
        n_eyes, n_tracks = len(eye_ids), len(track_ids)
        blocked = 1e6
        # Columns: the tracks, then one "unpaired" column per eye at cost 1 (the
        # gate's edge), so an eye takes a track only when that beats going empty.
        matrix = np.full((n_eyes, n_tracks + n_eyes), blocked)
        for i, (eye_id, (ex, ey)) in enumerate(zip(eye_ids, eye_centers)):
            matrix[i, n_tracks + i] = 1.0
            held = self.eye_to_cls8.get(eye_id)
            for j, track_id in enumerate(track_ids):
                cx, cy = track_centers[track_id]
                c = float(cost((float(ex), float(ey)), (float(cx), float(cy))))
                if not math.isfinite(c) or c > 1.0:
                    continue
                if track_id == held:
                    c -= self.STICKY_BONUS
                elif math.hypot(cx - ex, cy - ey) > self.rebind_max_dist_px:
                    continue
                # An exact tie goes to the earlier (upper) slot, as visiting the
                # eyes in order always did; no real cost difference is this small.
                matrix[i, j] = c + i * 1e-9

        # One proboscis, two tracks: keep the copy that fits its best eye best and
        # block the rest, or a neighbouring eye can claim the duplicate.
        best = matrix[:, :n_tracks].min(axis=0) if n_tracks else np.zeros(0)
        kept: List[int] = []
        for j in sorted(range(n_tracks), key=lambda j: (best[j], j)):
            cj = track_centers[track_ids[j]]
            if any(
                math.hypot(cj[0] - track_centers[track_ids[k]][0], cj[1] - track_centers[track_ids[k]][1])
                <= self.DUPLICATE_PX
                for k in kept
            ):
                matrix[:, j] = blocked
            else:
                kept.append(j)

        rows, cols = linear_sum_assignment(matrix)
        for eye_id in eye_ids:
            self.eye_to_cls8[eye_id] = None
        for i, j in zip(rows, cols):
            if j < n_tracks and matrix[i, j] < blocked:
                self.eye_to_cls8[eye_ids[i]] = track_ids[j]
