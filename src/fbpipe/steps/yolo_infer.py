from __future__ import annotations

import logging
import math
import os
import time

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from ultralytics import YOLO
import torch
import gc

from ..config import EyeQCSettings, Settings, get_main_directories
from ..utils import eye_template
from ..utils.frozen_folders import iter_live_batch_dirs
from ..utils.tables import read_table, write_table
from ..utils.timestamps import pick_timestamp_column, pick_frame_column, to_seconds_series
from ..utils.vision import xyxy_to_cxcywh
from ..utils.yolo_results import collect_detections
from ..utils.video_writer import FFmpegFrameWriter
from ..utils.distance_sanity import anisotropic_boundary_offsets, anisotropic_semi_axes
from ..utils.rig_anchor import resolve_anchor
from ..utils.rig_gates import apply_rig_gate_overrides
from ..utils.track import MultiObjectTracker, SingleClassTracker
from ..utils.multi_fly import (
    EyeAnchorManager,
    StablePairing,
    enforce_zero_iou_and_topk,
    match_live_eyes,
)
from ..utils.columns import (
    EYE_CLASS,
    PROBOSCIS_CLASS,
    PROBOSCIS_CORNERS_COL,
    PROBOSCIS_DISTANCE_COL,
    PROBOSCIS_TRACK_COL,
    PROBOSCIS_X_COL,
    PROBOSCIS_Y_COL,
)

log = logging.getLogger("fbpipe.yolo")
logging.getLogger("ultralytics").setLevel(logging.WARNING)

SINGLE_TRACK_CLASSES = tuple(c for c in (1,) if c not in (EYE_CLASS, PROBOSCIS_CLASS))
ALL_TRACKED_CLASSES = SINGLE_TRACK_CLASSES + (EYE_CLASS, PROBOSCIS_CLASS)

# Live-eye QC marker on the annotated video: a thin cyan ring. Never red or
# magenta (check_light_stimulus measures the frame's redness on this mp4), and
# no text or drift-dependent styling (human scorers watch this video blinded).
LIVE_EYE_MARKER_BGR = (255, 255, 0)
LIVE_EYE_MARKER_RADIUS_PX = 6

def _flow_nudge(prev_gray, gray, box_xyxy, flow_skip_edge: int, flow_params: dict):
    if prev_gray is None: return box_xyxy
    x1,y1,x2,y2 = box_xyxy.astype(int)
    x1 = max(flow_skip_edge, x1); y1 = max(flow_skip_edge, y1)
    x2 = min(gray.shape[1]-flow_skip_edge, x2); y2 = min(gray.shape[0]-flow_skip_edge, y2)
    if x2<=x1 or y2<=y1: return box_xyxy
    flow = cv2.calcOpticalFlowFarneback(prev_gray[y1:y2, x1:x2], gray[y1:y2, x1:x2], None, **flow_params)
    dx = np.median(flow[...,0]); dy = np.median(flow[...,1])
    nudged = box_xyxy.copy().astype(np.float32)
    nudged[0::2] += dx; nudged[1::2] += dy
    return nudged

def _angle_deg_between(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return float(np.degrees(np.arctan2(abs(cross), dot)))


def _prepare_single_class_input(
    boxes: np.ndarray, scores: np.ndarray, corners: List[Optional[List[float]]]
) -> Tuple[np.ndarray, np.ndarray, Optional[List[float]]]:
    if boxes.shape[0] == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), None
    best_idx = int(np.argmax(scores))
    selected_corners = corners[best_idx] if corners and best_idx < len(corners) else None
    return boxes[[best_idx]], scores[[best_idx]], selected_corners


def _scan_initial_eye_boxes(
    cap,
    max_frames: int,
    target_size: Tuple[int, int],
    settings: Settings,
    predict_fn,
) -> Tuple[np.ndarray, List[bool]]:
    """Return the eye boxes locked from the first ``max_frames`` frames, and
    whether each one had a proboscis inside its gate.

    The batch eye template needs the positions, not just the count: it votes
    them across the batch's trials and lends a trial the ones it missed. It
    needs the pairing too, because an eye alone is not a measurable fly -- a
    fly clipped by the frame edge can show an eye while its proboscis never
    enters the shot.
    """

    eye_mgr = EyeAnchorManager(max_eyes=settings.max_flies, zero_iou_eps=settings.zero_iou_epsilon)
    frames_checked = 0
    target_w, target_h = target_size
    prob_centres: List[Tuple[float, float]] = []

    while frames_checked < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        if (w, h) != (target_w, target_h):
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        result = predict_fn(frame, settings.conf_thres)[0]
        dets = collect_detections(result, ALL_TRACKED_CLASSES)
        cls2_boxes = dets.get(EYE_CLASS, {}).get("boxes", np.zeros((0, 4), np.float32))
        cls2_scores = dets.get(EYE_CLASS, {}).get("scores", np.zeros((0,), np.float32))
        cls2_boxes, cls2_scores = enforce_zero_iou_and_topk(
            cls2_boxes,
            cls2_scores,
            settings.max_flies,
            eps=settings.zero_iou_epsilon,
        )

        eye_mgr.try_update_from_dets(cls2_boxes, cls2_scores)

        # Every proboscis seen during the scan, matched against the final
        # anchors below. Collected rather than matched per frame because the
        # manager re-sorts its anchors top-to-bottom when it confirms.
        for box in dets.get(PROBOSCIS_CLASS, {}).get("boxes", np.zeros((0, 4), np.float32)):
            pcx, pcy, _, _ = xyxy_to_cxcywh(np.asarray(box, dtype=np.float32))
            prob_centres.append((float(pcx), float(pcy)))

        frames_checked += 1

        if eye_mgr.confirmed:
            break

    if not eye_mgr.anchors_xyxy:
        return np.zeros((0, 4), np.float32), []

    boxes = np.stack(eye_mgr.anchors_xyxy, axis=0).astype(np.float32)
    n_eyes = boxes.shape[0]
    paired = [
        any(
            _eye_prob_pair_is_valid(eye_xy, prob_xy, settings, n_eyes)
            for prob_xy in prob_centres
        )
        for eye_xy in eye_mgr.get_centers()
    ]
    return boxes, paired


def _scan_initial_fly_count(cap, max_frames, target_size, settings, predict_fn) -> int:
    """How many flies the first ``max_frames`` frames show."""

    boxes, _ = _scan_initial_eye_boxes(cap, max_frames, target_size, settings, predict_fn)
    return int(boxes.shape[0])


def load_or_build_eye_template(
    batch_dir: Path,
    video_paths: Sequence[Path],
    settings: Settings,
    scan_boxes: Callable[[Path], np.ndarray],
) -> Tuple[Optional[List[eye_template.TemplateSlot]], Dict[Path, np.ndarray]]:
    """Resolve the batch's fly slot set before any of its videos is processed.

    Returns the slots and whatever this call scanned, so the main loop reuses
    its own videos' scans instead of paying for them twice. A cached template
    costs no scans at all -- which matters because every worker slicing the
    batch would otherwise re-scan all of its videos.

    ``None`` means "no usable template": the block is off, the batch is empty,
    or no slot cleared ``min_trials`` (a one-trial batch can never corroborate
    anything). The caller then sizes the trial from its own frames, exactly as
    before this existed.
    """
    tpl_cfg = getattr(settings, "eye_template", None)
    if tpl_cfg is None or not tpl_cfg.enabled or not video_paths:
        return None, {}

    path = eye_template.template_path(batch_dir)
    cached = eye_template.load_template(path, min_trials=tpl_cfg.min_trials)
    if cached is not None:
        return (cached or None), {}

    scanned: Dict[Path, np.ndarray] = {}
    paired: Dict[str, List[bool]] = {}
    for video_path in sorted(video_paths, key=str):
        boxes, flags = scan_boxes(video_path)
        scanned[video_path] = boxes
        paired[video_path.name] = list(flags)

    slots = eye_template.build_template(
        {p.name: boxes for p, boxes in scanned.items()},
        min_trials=tpl_cfg.min_trials,
        cluster_tol_min_px=tpl_cfg.cluster_tol_min_px,
        cluster_tol_frac=tpl_cfg.cluster_tol_frac,
        trial_paired=paired,
        require_proboscis=tpl_cfg.require_proboscis,
    )
    try:
        eye_template.save_template(path, slots, tpl_cfg.min_trials)
    except OSError as exc:  # a read-only mirror must not abort the run
        print(f"[YOLO] could not cache the eye template in {batch_dir}: {exc}")
    return (slots or None), scanned


def _template_slot_count(
    slots: Optional[Sequence[eye_template.TemplateSlot]], settings: Settings
) -> Optional[int]:
    """How many fly slots the template gives a trial, capped by ``max_flies``."""

    if not slots:
        return None
    return min(len(slots), int(settings.max_flies))


def _active_proboscis_track_limit(active_max_flies: int) -> int:
    """Keep proboscis capacity aligned with the eye slots inferred for the video."""

    return max(0, int(active_max_flies))


def _limit_proboscis_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    active_max_flies: int,
) -> Tuple[np.ndarray, np.ndarray]:
    limit = _active_proboscis_track_limit(active_max_flies)
    if boxes.shape[0] == 0 or limit <= 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)

    order = np.argsort(-scores)[:limit]
    return boxes[order], scores[order]


def _build_proboscis_tracker(settings: Settings, active_max_flies: int) -> MultiObjectTracker:
    # The proboscis is a small, fast object: its box often has little or no
    # overlap with its own previous-frame box, so IoU association fails and
    # spawns spurious new tracks. Match by center distance instead.
    return MultiObjectTracker(
        iou_thres=settings.iou_match_thres,
        max_age=settings.max_age,
        ema_alpha=settings.ema_alpha,
        max_tracks=_active_proboscis_track_limit(active_max_flies),
        match_mode="center",
        max_match_dist=settings.proboscis_match_max_dist_px,
    )


def _max_valid_eye_prob_distance_px(settings: Settings, active_max_flies: int) -> float:
    """Apply an extra eye→proboscis distance sanity cap for crowded videos.

    Originally gated on exactly 3 flies; now applies whenever there are 3 or
    more fly slots (including 4-fly recordings), since the mispairing risk only
    grows as the arena gets more crowded.
    """

    if active_max_flies >= 3:
        return float(settings.three_fly_max_eye_prob_distance_px)
    return math.inf


def _eye_prob_distance_is_valid(
    distance_px: float,
    settings: Settings,
    active_max_flies: int,
) -> bool:
    if not np.isfinite(distance_px):
        return False
    return distance_px <= _max_valid_eye_prob_distance_px(settings, active_max_flies)


def _eye_prob_pair_is_valid(
    eye_xy: Tuple[float, float],
    prob_xy: Tuple[float, float],
    settings: Settings,
    active_max_flies: int,
) -> bool:
    """May this proboscis be paired with this eye?

    The circular >=3-fly cap AND the anisotropic ``proboscis_filter`` gate that
    reject_bad_proboscis enforces downstream. Checking the gate only downstream
    is too late: by then a proboscis sitting outside one eye's gate has already
    been locked to that eye, and its real owner never gets it.
    """
    dx = float(prob_xy[0]) - float(eye_xy[0])
    dy = float(prob_xy[1]) - float(eye_xy[1])
    if not _eye_prob_distance_is_valid(float(np.hypot(dx, dy)), settings, active_max_flies):
        return False
    pf = getattr(settings, "proboscis_filter", None)
    if pf is None or not getattr(pf, "enabled", False):
        return True
    max_px = float(getattr(pf, "max_eye_prob_distance_px", 0) or 0)
    if max_px <= 0:
        return True
    a, b = anisotropic_semi_axes(dx, dy, max_px, float(getattr(pf, "up_divisor", 4.0)))
    return (dx / a) ** 2 + (dy / float(b)) ** 2 <= 1.0


def _proboscis_gate_active(settings: Settings) -> bool:
    pf = getattr(settings, "proboscis_filter", None)
    return bool(
        pf is not None
        and getattr(pf, "enabled", False)
        and float(getattr(pf, "max_eye_prob_distance_px", 0) or 0) > 0
    )


def _eye_prob_gate_depth(
    eye_xy: Tuple[float, float],
    prob_xy: Tuple[float, float],
    settings: Settings,
    active_max_flies: int,
) -> float:
    """How deep a proboscis sits in this eye's anisotropic gate: 0 at the eye,
    1 at the gate's edge, inf when the pair is not allowed at all.

    The pairing cost. Comparing depths, not raw distances, is what separates
    close flies: 55 px under the lower eye is shallow for it (~0.2) but deep for
    an eye 178 px higher (~0.8), while anything above an eye is costly because
    the gate's upward reach is only max_px / up_divisor.
    """
    if not _eye_prob_pair_is_valid(eye_xy, prob_xy, settings, active_max_flies):
        return math.inf
    pf = settings.proboscis_filter
    dx = float(prob_xy[0]) - float(eye_xy[0])
    dy = float(prob_xy[1]) - float(eye_xy[1])
    a, b = anisotropic_semi_axes(
        dx, dy, float(pf.max_eye_prob_distance_px), float(getattr(pf, "up_divisor", 4.0))
    )
    return float(math.sqrt((dx / a) ** 2 + (dy / float(b)) ** 2))


def _clear_proboscis_match(row: Dict[str, float], idx: int) -> None:
    row[f"cls8_{idx}_track_id"] = np.nan
    row[f"cls8_{idx}_x"] = np.nan
    row[f"cls8_{idx}_y"] = np.nan
    row[f"dist_eye_{idx}_cls8_{idx}"] = np.nan
    row[f"angle_eye_{idx}_cls8_vs_anchor"] = np.nan


def _live_eye_match_radius(eye_qc) -> float:
    """``eye_qc.match_radius_px`` when it is a finite positive number, else the default.

    load_settings passes odd numbers through for the QC step to range-check, but
    here a radius that matches nothing would record every eye as missing -- for
    good, since YOLO never re-runs a processed trial.
    """
    try:
        radius = float(getattr(eye_qc, "match_radius_px", math.nan))
    except (TypeError, ValueError):
        radius = math.nan
    if math.isfinite(radius) and radius > 0:
        return radius
    return EyeQCSettings().match_radius_px


def _set_live_eye(
    row: Dict[str, float],
    idx: int,
    live_xy: Optional[Tuple[float, float]],
    anchor_xy: Optional[Tuple[float, float]] = None,
) -> None:
    """Slot ``idx``'s per-frame (live) eye centre and its drift from the frozen
    anchor ``anchor_xy``; NaN when the slot has no anchor or its eye went
    unmatched. QC only: every distance, angle and gate keeps the frozen anchor."""
    if live_xy is None or anchor_xy is None:
        row[f"eye_{idx}_live_x"] = np.nan
        row[f"eye_{idx}_live_y"] = np.nan
        row[f"eye_{idx}_drift_px"] = np.nan
        return
    lx, ly = live_xy
    row[f"eye_{idx}_live_x"] = lx
    row[f"eye_{idx}_live_y"] = ly
    row[f"eye_{idx}_drift_px"] = float(np.hypot(lx - anchor_xy[0], ly - anchor_xy[1]))


def _process_frame(
    frame,
    frame_number,
    current_timestamp,
    single_trackers: Dict[int, SingleClassTracker],
    prev_gray,
    anchor,
    settings: Settings,
    result,
    eye_mgr: EyeAnchorManager,
    cls8_tracker: MultiObjectTracker,
    pairer: StablePairing,
    active_max_flies: int,
):
    flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    dets = collect_detections(result, ALL_TRACKED_CLASSES)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    single_outputs: Dict[int, Dict[str, float]] = {}
    for cls_id in SINGLE_TRACK_CLASSES:
        boxes = dets.get(cls_id, {}).get("boxes", np.zeros((0, 4), np.float32))
        scores = dets.get(cls_id, {}).get("scores", np.zeros((0,), np.float32))
        corners_list = dets.get(cls_id, {}).get("corners", [])
        boxes, scores, selected_corners = _prepare_single_class_input(boxes, scores, corners_list)
        tracker = single_trackers[cls_id]
        track = tracker.step(boxes, scores)
        if track is None:
            single_outputs[cls_id] = dict(track_id=np.nan, x=np.nan, y=np.nan, corners=np.nan)
            continue
        if settings.use_optical_flow and track.time_since_update > 0:
            track.box_xyxy = _flow_nudge(prev_gray, gray, track.box_xyxy, settings.flow_skip_edge, flow_params)
        cx, cy, _, _ = xyxy_to_cxcywh(track.box_xyxy)
        single_outputs[cls_id] = dict(
            track_id=track.id,
            x=float(cx),
            y=float(cy),
            corners=selected_corners if selected_corners is not None else np.nan,
        )

    cls2_boxes = dets.get(EYE_CLASS, {}).get("boxes", np.zeros((0, 4), np.float32))
    cls2_scores = dets.get(EYE_CLASS, {}).get("scores", np.zeros((0,), np.float32))
    # Every eye detection, kept for the live-eye QC before top-k thins them: a
    # confident false positive can push a real eye out of the top k.
    raw_eye_boxes = cls2_boxes
    cls2_boxes, cls2_scores = enforce_zero_iou_and_topk(
        cls2_boxes,
        cls2_scores,
        active_max_flies,
        eps=settings.zero_iou_epsilon,
    )
    eye_mgr.try_update_from_dets(cls2_boxes, cls2_scores)
    eye_centers = eye_mgr.get_centers()
    eye_ids = eye_mgr.anchor_ids

    # Per-frame (live) eye position, QC only: where each anchor's eye is
    # detected this frame. Matched against whatever anchors exist, confirmed or
    # not, so the frames before confirmation are recorded too.
    eye_qc = getattr(settings, "eye_qc", None)
    track_live_eye = bool(getattr(eye_qc, "enabled", False))
    live_eyes = (
        match_live_eyes(eye_centers, raw_eye_boxes, _live_eye_match_radius(eye_qc))
        if track_live_eye
        else []
    )

    cls8_boxes = dets.get(PROBOSCIS_CLASS, {}).get("boxes", np.zeros((0, 4), np.float32))
    cls8_scores = dets.get(PROBOSCIS_CLASS, {}).get("scores", np.zeros((0,), np.float32))
    cls8_boxes, cls8_scores = _limit_proboscis_detections(cls8_boxes, cls8_scores, active_max_flies)

    tracks8 = cls8_tracker.step(cls8_boxes, cls8_scores)

    if settings.use_optical_flow:
        for track in tracks8:
            if track.time_since_update > 0:
                track.box_xyxy = _flow_nudge(prev_gray, gray, track.box_xyxy, settings.flow_skip_edge, flow_params)

    if _proboscis_gate_active(settings):
        # Pair all eyes jointly by gate depth each frame: flies mounted close
        # have overlapping gates, and pairing eye by eye let the upper eye keep
        # the lower fly's proboscis (october_07_batch_2_pretest_5).
        pairer.step(
            eye_ids,
            eye_centers,
            tracks8,
            cost=lambda eye_xy, prob_xy: _eye_prob_gate_depth(
                eye_xy, prob_xy, settings, active_max_flies
            ),
        )
    else:
        pairer.step(
            eye_ids,
            eye_centers,
            tracks8,
            accept=lambda eye_xy, prob_xy: _eye_prob_pair_is_valid(
                eye_xy, prob_xy, settings, active_max_flies
            ),
        )

    eye_mgr.draw(frame)
    # Draw the proboscis geometry gate around each eye: a proboscis outside
    # this anisotropic blob (generous left/right/down, tight up) is never paired
    # with that eye, and reject_bad_proboscis enforces it again downstream.
    pf = getattr(settings, "proboscis_filter", None)
    gate_max_px = float(getattr(pf, "max_eye_prob_distance_px", 0) or 0)
    if gate_max_px > 0:
        up_divisor = float(getattr(pf, "up_divisor", 1.0) or 1.0)
        offsets = anisotropic_boundary_offsets(gate_max_px, up_divisor)
        for ex_c, ey_c in eye_centers:
            poly = np.array(
                [[int(ex_c + ox), int(ey_c + oy)] for ox, oy in offsets], dtype=np.int32
            )
            cv2.polylines(frame, [poly], isClosed=True, color=(0, 200, 0), thickness=2)
    for track in tracks8:
        x1, y1, x2, y2 = track.box_xyxy.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cx, cy, _, _ = xyxy_to_cxcywh(track.box_xyxy)
        cv2.putText(
            frame,
            f"c{PROBOSCIS_CLASS} id={track.id}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )
        cv2.circle(frame, (int(cx), int(cy)), 4, (255, 0, 0), -1)

    cls8_by_id = {t.id: t for t in tracks8}

    row: Dict[str, float] = {
        "frame": frame_number,
        "timestamp": current_timestamp,
        "x_anchor": anchor[0],
        "y_anchor": anchor[1],
    }

    single_track_class = SINGLE_TRACK_CLASSES[0] if SINGLE_TRACK_CLASSES else None
    det_single = {"track_id": np.nan, "x": np.nan, "y": np.nan, "corners": np.nan}
    if single_track_class is not None:
        det_single = single_outputs.get(single_track_class, det_single)
        row.update(
            {
                f"track_id_class{single_track_class}": det_single["track_id"],
                f"x_class{single_track_class}": det_single["x"],
                f"y_class{single_track_class}": det_single["y"],
                f"corners_class{single_track_class}": str(det_single["corners"]),
            }
        )

    AX, AY = anchor
    for idx in range(settings.max_flies):
        if idx < len(eye_centers):
            ex, ey = eye_centers[idx]
            eye_id = eye_ids[idx]
            row[f"eye_{idx}_track_id"] = eye_id
            row[f"eye_{idx}_x"] = ex
            row[f"eye_{idx}_y"] = ey
            if track_live_eye:
                _set_live_eye(row, idx, live_eyes[idx], (ex, ey))
            row[f"dist_eye_{idx}_anchor"] = float(np.hypot(ex - AX, ey - AY))

            c8_id = pairer.eye_to_cls8.get(eye_id)
            row[f"cls8_{idx}_track_id"] = np.nan if c8_id is None else c8_id
            if c8_id is not None and int(c8_id) in cls8_by_id:
                track = cls8_by_id[int(c8_id)]
                cx, cy, _, _ = xyxy_to_cxcywh(track.box_xyxy)
                distance_px = float(np.hypot(cx - ex, cy - ey))
                if _eye_prob_pair_is_valid((ex, ey), (cx, cy), settings, active_max_flies):
                    row[f"cls8_{idx}_x"] = float(cx)
                    row[f"cls8_{idx}_y"] = float(cy)
                    row[f"dist_eye_{idx}_cls8_{idx}"] = distance_px
                    cv2.line(frame, (int(ex), int(ey)), (int(cx), int(cy)), (0, 255, 0), 4)
                    v_eye_prob = (cx - ex, cy - ey)
                    v_eye_anchor = (AX - ex, AY - ey)
                    row[f"angle_eye_{idx}_cls8_vs_anchor"] = _angle_deg_between(v_eye_prob, v_eye_anchor)
                else:
                    pairer.eye_to_cls8[eye_id] = None
                    _clear_proboscis_match(row, idx)
            else:
                _clear_proboscis_match(row, idx)

            if single_track_class is not None:
                if not np.isnan(det_single["x"]):
                    row[f"distance_class{single_track_class}_eye_{idx}"] = float(
                        np.hypot(det_single["x"] - ex, det_single["y"] - ey)
                    )
                else:
                    row[f"distance_class{single_track_class}_eye_{idx}"] = np.nan

            cv2.line(frame, (int(ex), int(ey)), (int(AX), int(AY)), (0, 165, 255), 2)
        else:
            row[f"eye_{idx}_track_id"] = np.nan
            row[f"eye_{idx}_x"] = np.nan
            row[f"eye_{idx}_y"] = np.nan
            if track_live_eye:
                _set_live_eye(row, idx, None)
            _clear_proboscis_match(row, idx)
            row[f"dist_eye_{idx}_anchor"] = np.nan
            if single_track_class is not None:
                row[f"distance_class{single_track_class}_eye_{idx}"] = np.nan

    # Live-eye marker last, so nothing covers it: the same thin ring at any
    # drift (see LIVE_EYE_MARKER_BGR for why it is cyan and wordless).
    if track_live_eye and getattr(eye_qc, "draw_live_eye", False):
        for live in live_eyes:
            if live is not None:
                cv2.circle(
                    frame,
                    (int(live[0]), int(live[1])),
                    LIVE_EYE_MARKER_RADIUS_PX,
                    LIVE_EYE_MARKER_BGR,
                    1,
                )

    return frame, row, gray


def _run_chunked_inference(cap, max_frame, target_wh, writer, timestamps, fps, anchor,
                           settings, batched_predict_fn, single_trackers, prev_gray,
                           eye_mgr, cls8_tracker, pairer, active_max_flies, batch_size):
    AX, AY = anchor
    target_w, target_h = target_wh
    rows, frame_idx, B = [], 0, max(1, batch_size)   # caller already clamped B per engine guard
    while cap.isOpened():
        batch_frames, batch_meta = [], []
        while len(batch_frames) < B and frame_idx <= max_frame:   # per-FRAME max_frame guard
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:        # EOF / corrupt-frame guard
                break
            h, w = frame.shape[:2]
            frame = (cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                     if (w, h) != (target_w, target_h) else frame.copy())  # break cv2 buffer aliasing
            batch_meta.append((frame_idx, timestamps.get(frame_idx, frame_idx / fps)))
            batch_frames.append(frame)
            frame_idx += 1
        if not batch_frames:                                      # never predict([])
            break
        results = batched_predict_fn(batch_frames, settings.conf_thres)
        assert len(results) == len(batch_frames)                  # fail loud on engine truncation
        for frame, (fidx, ts), result in zip(batch_frames, batch_meta, results):
            frame, row, prev_gray = _process_frame(
                frame, fidx, ts, single_trackers, prev_gray, (AX, AY),
                settings, result, eye_mgr, cls8_tracker, pairer, active_max_flies)
            writer.write(frame)
            rows.append(row)
    return rows


def _discard_empty_trial_outputs(out_dir: Path, out_mp4: Path, video_name: str) -> None:
    """Remove the outputs of a zero-frame inference run so the trial is retried.

    A source video that opens but decodes no frames (corrupt bitstream) would
    otherwise leave a header-only annotated mp4 and an empty table behind, and
    the ``out_dir.exists()`` skip-guard would treat the trial as processed on
    every future run. The directory is only removed when nothing else is in it.
    """
    log.error(
        "%s decoded 0 frames (corrupt video?); discarding empty outputs so the "
        "trial is retried on the next run", video_name,
    )
    print(f"[YOLO] {video_name}: decoded 0 frames (corrupt video?); leaving trial unprocessed")
    out_mp4.unlink(missing_ok=True)
    try:
        out_dir.rmdir()
    except OSError:
        pass  # other files present — keep them


def _export_per_fly_csvs(
    df: pd.DataFrame,
    out_dir: Path,
    folder_name: str,
    cfg: Settings,
    active_max_flies: int,
) -> List[Path]:
    per_fly_paths: List[Path] = []
    for idx in range(active_max_flies):
        eye_x_col = f"eye_{idx}_x"
        if eye_x_col not in df.columns or df[eye_x_col].notna().sum() == 0:
            continue

        single_track_class = SINGLE_TRACK_CLASSES[0] if SINGLE_TRACK_CLASSES else None
        eye_track_col = f"track_id_class{EYE_CLASS}"
        eye_x_out_col = f"x_class{EYE_CLASS}"
        eye_y_out_col = f"y_class{EYE_CLASS}"
        eye_corners_col = f"corners_class{EYE_CLASS}"
        distance_eye_anchor_col = f"distance_{EYE_CLASS}_anchor"
        angle_eye_prob_anchor_col = f"angle_deg_c{EYE_CLASS}_c{PROBOSCIS_CLASS}_vs_anchor"

        slot_data = {
            "frame": df["frame"],
            "timestamp": df["timestamp"],
            eye_track_col: df[f"eye_{idx}_track_id"],
            eye_x_out_col: df[eye_x_col],
            eye_y_out_col: df[f"eye_{idx}_y"],
            eye_corners_col: np.nan,
            PROBOSCIS_TRACK_COL: df[f"cls8_{idx}_track_id"],
            PROBOSCIS_X_COL: df[f"cls8_{idx}_x"],
            PROBOSCIS_Y_COL: df[f"cls8_{idx}_y"],
            PROBOSCIS_CORNERS_COL: np.nan,
            "x_anchor": df["x_anchor"],
            "y_anchor": df["y_anchor"],
            PROBOSCIS_DISTANCE_COL: df[f"dist_eye_{idx}_cls8_{idx}"],
            distance_eye_anchor_col: df[f"dist_eye_{idx}_anchor"],
            angle_eye_prob_anchor_col: df[f"angle_eye_{idx}_cls8_vs_anchor"],
            # Per-frame (live) eye centre and its drift from the frozen anchor
            # above: QC only, nothing measures from them. NaN floats when the
            # merged table has none (eye_qc disabled).
            "eye_live_x": df.get(f"eye_{idx}_live_x", np.nan),
            "eye_live_y": df.get(f"eye_{idx}_live_y", np.nan),
            "eye_drift_px": df.get(f"eye_{idx}_drift_px", np.nan),
        }

        if single_track_class is not None:
            slot_data.update(
                {
                    f"track_id_class{single_track_class}": df.get(f"track_id_class{single_track_class}", np.nan),
                    f"x_class{single_track_class}": df.get(f"x_class{single_track_class}", np.nan),
                    f"y_class{single_track_class}": df.get(f"y_class{single_track_class}", np.nan),
                    f"corners_class{single_track_class}": df.get(
                        f"corners_class{single_track_class}", np.nan
                    ),
                }
            )
            distance_single_eye_out = f"distance_{single_track_class}_{EYE_CLASS}"
            distance_single_eye_in = f"distance_class{single_track_class}_eye_{idx}"
            slot_data[distance_single_eye_out] = df.get(distance_single_eye_in, np.nan)

        slot_df = pd.DataFrame(slot_data)

        if slot_df[[eye_track_col, eye_x_out_col, eye_y_out_col]].notna().sum().sum() == 0:
            continue

        csv_path = out_dir / f"{folder_name}_fly{idx + 1}_distances.csv"
        written_path = write_table(slot_df, csv_path)
        per_fly_paths.append(written_path)

    return per_fly_paths

def _is_cuda_failure(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(tok in msg for tok in ("cuda", "cudnn", "cublas", "expandable_segments", "device-side assert", "hip"))


def _make_batched_predict_fn(model, get_device, set_device, allow_cpu, *, cuda_empty_cache=None,
                             is_engine=False):
    """Build a batched YOLO predict fn with CUDA OOM sub-batch backoff and optional CPU fallback.

    get_device() -> current device string ('cuda'/'cpu'); set_device(target) switches it.
    On a CUDA failure: free VRAM, halve the sub-batch and retry the SAME frames; at batch=1,
    fall back to CPU if allow_cpu, else re-raise. Returns one Results per input image.

    TensorRT engines get no ``half`` flag: ultralytics >= 8.4 rebuilds the predictor
    (re-deserializing the engine and a fresh ~3.7 GB execution context) on every call
    that carries it, and an exported engine is already FP16.
    """
    precision = {} if is_engine else {"half": True}

    def batched_predict_fn(images, conf_thres):
        bs = len(images)
        while True:
            try:
                out = []
                for i in range(0, len(images), bs):
                    out.extend(model.predict(images[i:i+bs], conf=conf_thres,
                                             verbose=False, device=get_device(), **precision))
                return out
            except RuntimeError as exc:
                if get_device() == "cuda" and _is_cuda_failure(exc):
                    if cuda_empty_cache is not None:
                        cuda_empty_cache()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if bs > 1:
                        bs = max(1, bs // 2)
                        log.warning("CUDA OOM/failure at batch; retrying at sub-batch %d", bs)
                        continue
                    if allow_cpu:
                        log.warning("CUDA failed at batch=1 (%s); switching to CPU.", exc)
                        set_device("cpu")
                        return model.predict(images, conf=conf_thres, verbose=False,
                                             device=get_device(), half=False)
                raise
    return batched_predict_fn


def main(cfg: Settings):
    cuda_available = torch.cuda.is_available()
    use_cuda = cuda_available and not cfg.allow_cpu

    torch.backends.cudnn.benchmark = use_cuda
    if cuda_available:
        torch.backends.cuda.matmul.allow_tf32 = cfg.cuda_allow_tf32
        torch.backends.cudnn.allow_tf32 = cfg.cuda_allow_tf32

    if not cuda_available and not cfg.allow_cpu:
        raise RuntimeError("CUDA is not available. Set allow_cpu: true only for smoke tests.")

    if cfg.allow_cpu and cuda_available and not use_cuda:
        log.warning("allow_cpu is enabled; forcing YOLO inference to run on CPU.")

    # Check for TensorRT engine first, fall back to .pt file
    engine_path = str(Path(cfg.model_path).with_suffix('.engine'))
    if Path(engine_path).exists():
        log.info(f"Loading TensorRT engine: {engine_path}")
        model = YOLO(engine_path)
        is_engine = True
    else:
        log.info(f"Loading PyTorch model: {cfg.model_path} (export to .engine for 2-5x speedup)")
        model = YOLO(cfg.model_path)
        is_engine = False
    device_in_use: Optional[str] = None

    def _set_device(target: str):
        nonlocal device_in_use
        if device_in_use == target:
            return
        if target == "cpu":
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # TensorRT engines are bound to the GPU they were built for and reject
        # .to(); the device is supplied per-call in predict() instead.
        if not is_engine:
            model.to(target)
        device_in_use = target

    target_device = "cuda" if use_cuda else "cpu"
    try:
        _set_device(target_device)
    except RuntimeError as exc:
        if target_device == "cuda":
            if cfg.allow_cpu and _is_cuda_failure(exc):
                log.warning("CUDA initialisation failed (%s); falling back to CPU.", exc)
                _set_device("cpu")
            else:
                raise
        else:
            raise

    if device_in_use is None:
        raise RuntimeError("Failed to initialise YOLO device")

    batched_predict_fn = _make_batched_predict_fn(
        model,
        get_device=lambda: device_in_use,
        set_device=_set_device,
        allow_cpu=cfg.allow_cpu,
        is_engine=is_engine,
    )

    def predict_fn(image, conf_thres):   # single-frame adapter for the warm-up scan
        return batched_predict_fn([image], conf_thres)

    # Optional video-level parallelism: N worker processes each handle a disjoint
    # slice of the deterministically-ordered video list via global_index % N.
    # Defaults (1/0) preserve the original single-process behaviour exactly.
    num_workers = max(1, int(os.getenv("NUM_WORKERS", "1")))
    worker_index = int(os.getenv("WORKER_INDEX", "0")) % num_workers
    if num_workers > 1:
        # Cap OpenCV/torch CPU threads so N workers don't oversubscribe the cores
        # (BLAS/OpenMP pools are capped via env in the driver, before import).
        _t = max(1, (os.cpu_count() or num_workers) // num_workers)
        try:
            cv2.setNumThreads(_t)
        except Exception:
            pass
        try:
            torch.set_num_threads(_t)
        except Exception:
            pass
        print(f"[YOLO] worker {worker_index+1}/{num_workers}: every {num_workers}th video, {_t} CPU threads")

    live_eye_radius = _live_eye_match_radius(cfg.eye_qc)
    if cfg.eye_qc.enabled and live_eye_radius != cfg.eye_qc.match_radius_px:
        print(
            f"[YOLO] eye_qc.match_radius_px={cfg.eye_qc.match_radius_px!r} is not a positive "
            f"number; matching per-frame eyes within {live_eye_radius:g} px instead."
        )

    roots = get_main_directories(cfg)
    _job_idx = -1
    for root in sorted(roots, key=str):
        if not root.is_dir():
            print(f"[YOLO] main_directories entry does not exist: {root}")
            continue

        # Frozen experiment folders are skipped entirely: no GPU, no I/O. Their
        # existing per-trial CSVs stay put so build_wide_csv still emits them.
        for fly in iter_live_batch_dirs(cfg, root):
            # Per-batch gate widening (drawn gate polygon + ≥3-fly pairing cap).
            fly_cfg = apply_rig_gate_overrides(cfg, fly)
            video_files = sorted(
                (f for f in fly.iterdir() if f.suffix.lower() in (".mp4", ".avi")), key=str
            )

            def _scan_video_eyes(path: Path, _cfg=cfg) -> np.ndarray:
                """The warm-up scan for the batch template, on its own capture."""
                probe = cv2.VideoCapture(str(path))
                if not probe.isOpened():
                    print(f"[YOLO] Cannot open {path} for the batch eye template")
                    return np.zeros((0, 4), np.float32), []
                try:
                    pw = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1080
                    ph = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
                    size = (1080, 1080) if (pw, ph) != (1080, 1080) else (pw, ph)
                    return _scan_initial_eye_boxes(probe, 5, size, _cfg, predict_fn)
                finally:
                    probe.release()

            # Built on the first video this worker actually processes in the
            # batch, so a batch it owns nothing in costs nothing.
            template_state: Optional[Tuple[Optional[List[eye_template.TemplateSlot]], Dict[Path, np.ndarray]]] = None
            for video_path in video_files:
                _job_idx += 1
                if _job_idx % num_workers != worker_index:
                    continue
                base = video_path.stem
                # rig_3 is physically mirrored, so its anchor is on the other
                # side. Resolved per video because one run can span both rigs.
                AX, AY = resolve_anchor(video_path)
                csv_file_path = fly / f"{base.replace('_preprocessed','')}.csv"
                parts = base.split("_")
                folder_name = "_".join(parts[1:7]) if len(parts)>=7 else base
                out_dir = fly / folder_name
                if out_dir.exists():
                    print(f"[YOLO] Skipping {video_path.name}: detected processed folder {out_dir.name}")
                    continue
                out_dir.mkdir(exist_ok=True)
                out_mp4 = out_dir / f"{folder_name}_distance_annotated.mp4"

                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    print(f"[YOLO] Cannot open {video_path}")
                    continue

                original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1080
                original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
                target_w, target_h = (1080, 1080) if (original_w, original_h) != (1080, 1080) else (original_w, original_h)

                scan_frames = 5
                if template_state is None:
                    template_state = load_or_build_eye_template(
                        fly, video_files, cfg, _scan_video_eyes
                    )
                    if template_state[0]:
                        print(
                            f"[YOLO] {fly.name}: batch eye template has "
                            f"{len(template_state[0])} fly slot(s) "
                            f"(seen in {[s.n_trials_seen for s in template_state[0]]} trials, "
                            f"proboscis paired in {[s.n_trials_paired for s in template_state[0]]})."
                        )
                template_slots, template_scans = template_state

                eye_boxes = template_scans.pop(video_path, None)
                if eye_boxes is None:
                    eye_boxes, _ = _scan_initial_eye_boxes(
                        cap, scan_frames, (target_w, target_h), cfg, predict_fn
                    )
                detected_flies = int(eye_boxes.shape[0])
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                    cap.release()
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        print(f"[YOLO] Cannot reopen {video_path} after warm-up scan")
                        continue

                # The batch template, where there is one, decides the slot set:
                # a fly this trial's first frames missed still gets its slot, at
                # its number, so fly1 means the same fly in every trial.
                seeded_anchor_boxes = None
                template_count = _template_slot_count(template_slots, cfg)
                if template_count is not None:
                    active_max_flies = template_count
                    seeded_anchor_boxes, borrowed = eye_template.seed_anchor_boxes(
                        template_slots[:active_max_flies],
                        eye_boxes,
                        cfg.eye_template.match_radius_px,
                    )
                    if any(borrowed):
                        missing = [i + 1 for i, b in enumerate(borrowed) if b]
                        print(
                            f"[YOLO] {video_path.name}: {active_max_flies} fly slots from the "
                            f"batch template; fly {missing} not detected in the first "
                            f"{scan_frames} frames, anchor borrowed from the template."
                        )
                    else:
                        print(
                            f"[YOLO] {video_path.name}: {active_max_flies} fly slots from the "
                            f"batch template, all detected in this trial."
                        )
                elif detected_flies == 0:
                    print(
                        f"[YOLO] {video_path.name}: no eyes detected in first {scan_frames} frames; "
                        f"using configured max ({cfg.max_flies})."
                    )
                    active_max_flies = cfg.max_flies
                else:
                    active_max_flies = min(detected_flies, cfg.max_flies)
                    print(
                        f"[YOLO] {video_path.name}: detected {active_max_flies} fly/fly slots from first {scan_frames} frames."
                    )

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                max_frame = total_frames-1 if total_frames>0 else 10**9
                timestamps = {}
                calculated_fps = cap.get(cv2.CAP_PROP_FPS) or cfg.fps_default

                df_timestamps = None
                if csv_file_path.exists():
                    try:
                        df_timestamps = read_table(csv_file_path)
                    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
                        log.warning(
                            "Timestamp CSV %s is empty/unparseable (%s); using frame-index timestamps.",
                            csv_file_path.name, exc,
                        )
                if df_timestamps is not None:
                    frame_col = pick_frame_column(df_timestamps)
                    if frame_col is not None:
                        ts_col = pick_timestamp_column(df_timestamps)
                        if ts_col is not None:
                            secs = to_seconds_series(df_timestamps, ts_col)
                            tmp = pd.DataFrame({
                                "_frame": pd.to_numeric(df_timestamps[frame_col], errors="coerce"),
                                "seconds": secs
                            }).dropna(subset=["_frame", "seconds"])
                            tmp["_frame"] = tmp["_frame"].astype(int)
                            timestamps = tmp.set_index("_frame")["seconds"].to_dict()
                            if not tmp["_frame"].empty:
                                max_frame = int(tmp["_frame"].max())
                            fps_from_csv = (tmp.shape[0] / (tmp["seconds"].iloc[-1] - tmp["seconds"].iloc[0])) if tmp.shape[0] >= 2 else None
                            if fps_from_csv and np.isfinite(fps_from_csv) and fps_from_csv > 0:
                                calculated_fps = fps_from_csv

                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1080
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
                # Frames are resized to 1080x1080 in the loop when the source
                # differs, so the writer must use the *output* dimensions.
                out_w, out_h = (1080, 1080) if (w, h) != (1080, 1080) else (w, h)

                # The annotated video is for human review only; its container
                # framerate must be a sane playback rate. calculated_fps is
                # derived from the CSV timestamps and can be garbage (e.g. when
                # the source timestamps are mis-scaled), producing a video with
                # an absurd fps that players refuse to open. Clamp to a sane
                # range and fall back to the configured default otherwise.
                video_fps = calculated_fps
                if not (1.0 <= video_fps <= 240.0):
                    log.warning(
                        "Computed fps %.3f out of range for %s; using fps_default=%.1f for the annotated video",
                        video_fps, out_mp4.name, cfg.fps_default,
                    )
                    video_fps = cfg.fps_default
                fps = video_fps

                # Encode H.264 in a single pass by piping annotated frames
                # straight into ffmpeg (libx264, yuv420p, +faststart). This
                # replaces the old write-then-re-encode flow: one encode instead
                # of two (~half the CPU), and yuv420p/faststart keep the output
                # broadly playable. Fall back to OpenCV mp4v if ffmpeg is absent.
                writer = FFmpegFrameWriter(out_mp4, fps, out_w, out_h, crf=30, preset="medium")
                if not writer.ok:
                    log.warning("ffmpeg pipe unavailable; falling back to OpenCV mp4v writer")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (out_w, out_h))
                    if not writer.isOpened():
                        log.error(f"Failed to open VideoWriter for {out_mp4}, skipping video")
                        continue

                single_trackers = {
                    cls: SingleClassTracker(
                        iou_thres=cfg.iou_match_thres, max_age=cfg.max_age, ema_alpha=cfg.ema_alpha
                    )
                    for cls in SINGLE_TRACK_CLASSES
                }
                eye_mgr = EyeAnchorManager(max_eyes=active_max_flies, zero_iou_eps=cfg.zero_iou_epsilon)
                if seeded_anchor_boxes is not None:
                    eye_mgr.seed_from_boxes(seeded_anchor_boxes)
                cls8_tracker = _build_proboscis_tracker(cfg, active_max_flies)
                pairer = StablePairing(max_pairs=active_max_flies)

                target_w, target_h = (1080, 1080) if (w, h) != (1080, 1080) else (w, h)
                pairer.rebind_max_dist_px = cfg.pair_rebind_ratio * math.hypot(target_w, target_h)

                t0 = time.time()
                B = max(1, cfg.inference_batch_size) if (not is_engine or cfg.engine_supports_batch) else 1
                _path_tag = "engine" if is_engine else "pt"
                _batch_tag = " (engine batch-capable)" if is_engine and cfg.engine_supports_batch else ""
                log.info("YOLO inference: %s path, effective batch size %d%s", _path_tag, B, _batch_tag)
                print(f"[YOLO] {video_path.name}: {_path_tag} path, effective batch size {B}{_batch_tag}")
                rows = _run_chunked_inference(
                    cap, max_frame, (target_w, target_h), writer, timestamps, fps, (AX, AY),
                    fly_cfg, batched_predict_fn, single_trackers, None,
                    eye_mgr, cls8_tracker, pairer, active_max_flies, B)
                cap.release(); writer.release()

                if not rows:
                    _discard_empty_trial_outputs(out_dir, out_mp4, video_path.name)
                    continue

                if out_mp4.exists():
                    final_size = out_mp4.stat().st_size / (1024**2)
                    log.info(f"  Final output: {final_size:.1f} MB ({fps:.1f} fps)")

                df_rows = pd.DataFrame(rows)
                merged_csv_path = out_dir / f"{folder_name}_distances_merged.csv"
                write_table(df_rows, merged_csv_path)

                per_fly_paths = _export_per_fly_csvs(df_rows, out_dir, folder_name, cfg, active_max_flies)

                extra_msg = ""
                if per_fly_paths:
                    extra_msg = " (per-fly CSVs: " + ", ".join(p.name for p in per_fly_paths) + ")"
                print(f"[YOLO] {video_path.name} → {out_dir} in {time.time()-t0:.1f}s{extra_msg}")
