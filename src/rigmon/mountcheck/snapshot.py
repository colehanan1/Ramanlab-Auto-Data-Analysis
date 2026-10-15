"""Annotated snapshot: boxes, fly slot numbers, flagged items in red, issue list."""

from __future__ import annotations

import math

import cv2
import numpy as np

from fbpipe.utils.mount_qc import EYE, Detection, MountReport

BLUE, YELLOW, RED, WHITE = (255, 140, 0), (0, 210, 255), (0, 0, 255), (255, 255, 255)


def _slot_of(d: Detection, report: MountReport) -> int | None:
    best, best_d = None, float("inf")
    for s in report.slots:
        ref = s.eye_xy if d.cls == EYE or s.proboscis_xy is None else s.proboscis_xy
        dist = math.hypot(d.x - ref[0], d.y - ref[1])
        if dist < best_d:
            best, best_d = s.slot, dist
    return best


def render(img: np.ndarray, dets: list, report: MountReport) -> np.ndarray:
    out = img.copy()
    flagged = {(f.slot, f.part) for f in report.flags if f.slot is not None}
    for d in dets:
        slot = _slot_of(d, report)
        color = RED if (slot, d.cls) in flagged else (BLUE if d.cls == EYE else YELLOW)
        box = cv2.boxPoints(((d.x, d.y), (d.w, d.h), math.degrees(d.r))).astype(np.int32)
        cv2.polylines(out, [box], True, color, 3)
    for s in report.slots:
        cv2.putText(out, f"fly {s.slot}", (int(s.eye_xy[0]) + 35, int(s.eye_xy[1]) + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
    for i, f in enumerate(report.flags[:6]):
        cv2.putText(out, f.message, (12, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)
    return out
