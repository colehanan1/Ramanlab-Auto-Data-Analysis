from __future__ import annotations

import logging
import math
import os
import time

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO
import torch
import gc  # Add this import

from ..config import Settings, get_main_directories
from ..utils.tables import read_table, write_table, table_path
from ..utils.timestamps import pick_timestamp_column, pick_frame_column, to_seconds_series
from ..utils.vision import xyxy_to_cxcywh
from ..utils.yolo_results import collect_detections
from ..utils.video_writer import FFmpegFrameWriter
from ..utils.distance_sanity import anisotropic_boundary_offsets
from ..utils.track import MultiObjectTracker, SingleClassTracker
from ..utils.multi_fly import EyeAnchorManager, StablePairing, enforce_zero_iou_and_topk
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


def _scan_initial_fly_count(
    cap,
    max_frames: int,
    target_size: Tuple[int, int],
    settings: Settings,
    predict_fn,
):
    """Infer the number of flies present using the first ``max_frames`` frames."""

    eye_mgr = EyeAnchorManager(max_eyes=settings.max_flies, zero_iou_eps=settings.zero_iou_epsilon)
    frames_checked = 0
    target_w, target_h = target_size

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
        frames_checked += 1

        if eye_mgr.confirmed:
            break

    return len(eye_mgr.anchor_ids)


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


def _clear_proboscis_match(row: Dict[str, float], idx: int) -> None:
    row[f"cls8_{idx}_track_id"] = np.nan
    row[f"cls8_{idx}_x"] = np.nan
    row[f"cls8_{idx}_y"] = np.nan
    row[f"dist_eye_{idx}_cls8_{idx}"] = np.nan
    row[f"angle_eye_{idx}_cls8_vs_anchor"] = np.nan


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
    cls2_boxes, cls2_scores = enforce_zero_iou_and_topk(
        cls2_boxes,
        cls2_scores,
        active_max_flies,
        eps=settings.zero_iou_epsilon,
    )
    eye_mgr.try_update_from_dets(cls2_boxes, cls2_scores)
    eye_centers = eye_mgr.get_centers()
    eye_ids = eye_mgr.anchor_ids

    cls8_boxes = dets.get(PROBOSCIS_CLASS, {}).get("boxes", np.zeros((0, 4), np.float32))
    cls8_scores = dets.get(PROBOSCIS_CLASS, {}).get("scores", np.zeros((0,), np.float32))
    cls8_boxes, cls8_scores = _limit_proboscis_detections(cls8_boxes, cls8_scores, active_max_flies)

    tracks8 = cls8_tracker.step(cls8_boxes, cls8_scores)

    if settings.use_optical_flow:
        for track in tracks8:
            if track.time_since_update > 0:
                track.box_xyxy = _flow_nudge(prev_gray, gray, track.box_xyxy, settings.flow_skip_edge, flow_params)

    pairer.step(eye_ids, eye_centers, tracks8)

    eye_mgr.draw(frame)
    # Draw the proboscis geometry gate around each eye: any proboscis outside
    # this anisotropic blob (generous left/down, tight right/up) is rejected
    # downstream (see ProboscisFilterSettings).
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
            row[f"dist_eye_{idx}_anchor"] = float(np.hypot(ex - AX, ey - AY))

            c8_id = pairer.eye_to_cls8.get(eye_id)
            row[f"cls8_{idx}_track_id"] = np.nan if c8_id is None else c8_id
            if c8_id is not None and int(c8_id) in cls8_by_id:
                track = cls8_by_id[int(c8_id)]
                cx, cy, _, _ = xyxy_to_cxcywh(track.box_xyxy)
                distance_px = float(np.hypot(cx - ex, cy - ey))
                if _eye_prob_distance_is_valid(distance_px, settings, active_max_flies):
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
            _clear_proboscis_match(row, idx)
            row[f"dist_eye_{idx}_anchor"] = np.nan
            if single_track_class is not None:
                row[f"distance_class{single_track_class}_eye_{idx}"] = np.nan

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


def _make_batched_predict_fn(model, get_device, set_device, allow_cpu, *, cuda_empty_cache=None):
    """Build a batched YOLO predict fn with CUDA OOM sub-batch backoff and optional CPU fallback.

    get_device() -> current device string ('cuda'/'cpu'); set_device(target) switches it.
    On a CUDA failure: free VRAM, halve the sub-batch and retry the SAME frames; at batch=1,
    fall back to CPU if allow_cpu, else re-raise. Returns one Results per input image.
    """
    def batched_predict_fn(images, conf_thres):
        bs = len(images)
        while True:
            try:
                out = []
                for i in range(0, len(images), bs):
                    out.extend(model.predict(images[i:i+bs], conf=conf_thres,
                                             verbose=False, device=get_device(), half=True))
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
    )

    def predict_fn(image, conf_thres):   # single-frame adapter for the warm-up scan
        return batched_predict_fn([image], conf_thres)

    AX, AY = cfg.anchor_x, cfg.anchor_y

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

    roots = get_main_directories(cfg)
    _job_idx = -1
    for root in sorted(roots, key=str):
        if not root.is_dir():
            print(f"[YOLO] main_directories entry does not exist: {root}")
            continue

        for fly in sorted((p for p in root.iterdir() if p.is_dir()), key=str):
            video_files = sorted(
                (f for f in fly.iterdir() if f.suffix.lower() in (".mp4", ".avi")), key=str
            )
            for video_path in video_files:
                _job_idx += 1
                if _job_idx % num_workers != worker_index:
                    continue
                base = video_path.stem
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
                detected_flies = _scan_initial_fly_count(cap, scan_frames, (target_w, target_h), cfg, predict_fn)
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                    cap.release()
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        print(f"[YOLO] Cannot reopen {video_path} after warm-up scan")
                        continue

                if detected_flies == 0:
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
                    cfg, batched_predict_fn, single_trackers, None,
                    eye_mgr, cls8_tracker, pairer, active_max_flies, B)
                cap.release(); writer.release()

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
