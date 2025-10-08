from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

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
    """Lock the positions of up to ``max_eyes`` class-2 detections."""

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
            order = np.argsort([c[0] for c in centers])
            self.anchors_xyxy = [self.anchors_xyxy[i] for i in order]
            self.anchor_ids = [self.anchor_ids[i] for i in order]
            self.confirmed = True

    def get_centers(self) -> List[Tuple[float, float]]:
        centers: List[Tuple[float, float]] = []
        for b in self.anchors_xyxy:
            cx, cy, _, _ = xyxy_to_cxcywh(b)
            centers.append((float(cx), float(cy)))
        return centers

    def draw(self, frame) -> None:
        for idx, b in enumerate(self.anchors_xyxy):
            x1, y1, x2, y2 = b.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cx, cy = map(int, self.get_centers()[idx])
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


class StablePairing:
    """Maintain one-to-one pairings between frozen eyes and proboscis tracks."""

    def __init__(self, max_pairs: int = 4, rebind_max_dist_px: float = 220.0):
        self.eye_to_cls8: Dict[int, Optional[int]] = {}
        self.max_pairs = max_pairs
        self.rebind_max_dist_px = float(rebind_max_dist_px)

    def step(
        self,
        eye_ids: List[int],
        eye_centers: List[Tuple[float, float]],
        cls8_tracks,
    ) -> None:
        if not eye_ids:
            return

        live_cls8_ids = {t.id for t in cls8_tracks}

        for eye_id in list(self.eye_to_cls8.keys()):
            cls8_id = self.eye_to_cls8[eye_id]
            if cls8_id is not None and cls8_id not in live_cls8_ids:
                self.eye_to_cls8[eye_id] = None

        claimed_live_ids = {
            cls8_id
            for cls8_id in self.eye_to_cls8.values()
            if cls8_id is not None and cls8_id in live_cls8_ids
        }

        used_this_step = set(claimed_live_ids)
        track_centers = {t.id: xyxy_to_cxcywh(t.box_xyxy)[:2] for t in cls8_tracks}

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
                dist = math.hypot(cx - ex, cy - ey)
                if best_dist is None or dist < best_dist:
                    best_id, best_dist = track_id, dist

            if best_id is not None and best_dist is not None and best_dist <= self.rebind_max_dist_px:
                self.eye_to_cls8[eye_id] = best_id
                used_this_step.add(best_id)

