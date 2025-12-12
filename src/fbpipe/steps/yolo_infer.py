from __future__ import annotations

import logging
import math
import time

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO
import torch
import gc  # Add this import

from ..config import Settings
from ..utils.timestamps import pick_timestamp_column, pick_frame_column, to_seconds_series
from ..utils.vision import order_corners, xyxy_to_cxcywh
from ..utils.track import MultiObjectTracker, SingleClassTracker
from ..utils.multi_fly import EyeAnchorManager, StablePairing, enforce_zero_iou_and_topk
from ..utils.columns import (
    PROBOSCIS_CLASS,
    PROBOSCIS_CORNERS_COL,
    PROBOSCIS_DISTANCE_COL,
    PROBOSCIS_TRACK_COL,
    PROBOSCIS_X_COL,
    PROBOSCIS_Y_COL,
)

log = logging.getLogger("fbpipe.yolo")
logging.getLogger("ultralytics").setLevel(logging.WARNING)

EYE_CLASS = 2
SINGLE_TRACK_CLASSES = (1,)
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

def _collect_detections(result) -> Dict[int, Dict[str, np.ndarray]]:
    out: Dict[int, Dict[str, List[np.ndarray]]] = {
        cls: {"boxes": [], "scores": [], "corners": []} for cls in ALL_TRACKED_CLASSES
    }

    if hasattr(result, "obb") and result.obb is not None:
        xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()
        cls_arr = result.obb.cls.cpu().numpy().astype(int)
        if hasattr(result.obb, "conf") and result.obb.conf is not None:
            conf_arr = result.obb.conf.cpu().numpy()
        else:
            conf_arr = np.ones_like(cls_arr, dtype=np.float32)
        for idx, (cls_id, conf) in enumerate(zip(cls_arr, conf_arr)):
            if cls_id not in out:
                continue
            corners = xyxyxyxy[idx].reshape(4, 2)
            x1, y1 = corners[:, 0].min(), corners[:, 1].min()
            x2, y2 = corners[:, 0].max(), corners[:, 1].max()
            out[cls_id]["boxes"].append(np.array([x1, y1, x2, y2], dtype=np.float32))
            out[cls_id]["scores"].append(float(conf))
            out[cls_id]["corners"].append(order_corners(corners))
    else:
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            cls_arr = result.boxes.cls.cpu().numpy().astype(int)
            conf_arr = result.boxes.conf.cpu().numpy().astype(np.float32)
            for box, cls_id, conf in zip(xyxy, cls_arr, conf_arr):
                if cls_id not in out:
                    continue
                out[cls_id]["boxes"].append(box)
                out[cls_id]["scores"].append(float(conf))
                out[cls_id]["corners"].append(None)

    parsed: Dict[int, Dict[str, np.ndarray]] = {}
    for cls_id, data in out.items():
        boxes = np.stack(data["boxes"], axis=0) if data["boxes"] else np.zeros((0, 4), np.float32)
        scores = np.array(data["scores"], dtype=np.float32) if data["scores"] else np.zeros((0,), np.float32)
        parsed[cls_id] = {"boxes": boxes, "scores": scores, "corners": data["corners"]}
    return parsed


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
        dets = _collect_detections(result)
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


def _process_frame(
    frame,
    frame_number,
    current_timestamp,
    single_trackers: Dict[int, SingleClassTracker],
    prev_gray,
    anchor,
    settings: Settings,
    predict_fn,
    eye_mgr: EyeAnchorManager,
    cls8_tracker: MultiObjectTracker,
    pairer: StablePairing,
    active_max_flies: int,
):
    conf_thres = settings.conf_thres
    flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    result = predict_fn(frame, conf_thres)[0]
    dets = _collect_detections(result)

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
    if cls8_boxes.shape[0] > 0:
        order = np.argsort(-cls8_scores)[: settings.max_proboscis_tracks]
        cls8_boxes = cls8_boxes[order]
        cls8_scores = cls8_scores[order]

    tracks8 = cls8_tracker.step(cls8_boxes, cls8_scores)

    if settings.use_optical_flow:
        for track in tracks8:
            if track.time_since_update > 0:
                track.box_xyxy = _flow_nudge(prev_gray, gray, track.box_xyxy, settings.flow_skip_edge, flow_params)

    pairer.step(eye_ids, eye_centers, tracks8)

    eye_mgr.draw(frame)
    for track in tracks8:
        x1, y1, x2, y2 = track.box_xyxy.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cx, cy, _, _ = xyxy_to_cxcywh(track.box_xyxy)
        cv2.putText(
            frame,
            f"c8 id={track.id}",
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

    det_class1 = single_outputs.get(1, {"track_id": np.nan, "x": np.nan, "y": np.nan, "corners": np.nan})
    row.update(
        {
            "track_id_class1": det_class1["track_id"],
            "x_class1": det_class1["x"],
            "y_class1": det_class1["y"],
            "corners_class1": str(det_class1["corners"]),
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
                row[f"cls8_{idx}_x"] = float(cx)
                row[f"cls8_{idx}_y"] = float(cy)
                row[f"dist_eye_{idx}_cls8_{idx}"] = float(np.hypot(cx - ex, cy - ey))
                cv2.line(frame, (int(ex), int(ey)), (int(cx), int(cy)), (0, 255, 0), 4)
                v_eye_prob = (cx - ex, cy - ey)
                v_eye_anchor = (AX - ex, AY - ey)
                row[f"angle_eye_{idx}_cls8_vs_anchor"] = _angle_deg_between(v_eye_prob, v_eye_anchor)
            else:
                row[f"cls8_{idx}_x"] = np.nan
                row[f"cls8_{idx}_y"] = np.nan
                row[f"dist_eye_{idx}_cls8_{idx}"] = np.nan
                row[f"angle_eye_{idx}_cls8_vs_anchor"] = np.nan

            if not np.isnan(det_class1["x"]):
                row[f"distance_class1_eye_{idx}"] = float(
                    np.hypot(det_class1["x"] - ex, det_class1["y"] - ey)
                )
            else:
                row[f"distance_class1_eye_{idx}"] = np.nan

            cv2.line(frame, (int(ex), int(ey)), (int(AX), int(AY)), (0, 165, 255), 2)
        else:
            row[f"eye_{idx}_track_id"] = np.nan
            row[f"eye_{idx}_x"] = np.nan
            row[f"eye_{idx}_y"] = np.nan
            row[f"cls8_{idx}_track_id"] = np.nan
            row[f"cls8_{idx}_x"] = np.nan
            row[f"cls8_{idx}_y"] = np.nan
            row[f"dist_eye_{idx}_cls8_{idx}"] = np.nan
            row[f"angle_eye_{idx}_cls8_vs_anchor"] = np.nan
            row[f"dist_eye_{idx}_anchor"] = np.nan
            row[f"distance_class1_eye_{idx}"] = np.nan

    return frame, row, gray


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

        slot_df = pd.DataFrame(
            {
                "frame": df["frame"],
                "timestamp": df["timestamp"],
                "track_id_class1": df.get("track_id_class1", np.nan),
                "x_class1": df.get("x_class1", np.nan),
                "y_class1": df.get("y_class1", np.nan),
                "corners_class1": df.get("corners_class1", np.nan),
                "track_id_class2": df[f"eye_{idx}_track_id"],
                "x_class2": df[eye_x_col],
                "y_class2": df[f"eye_{idx}_y"],
                "corners_class2": np.nan,
                PROBOSCIS_TRACK_COL: df[f"cls8_{idx}_track_id"],
                PROBOSCIS_X_COL: df[f"cls8_{idx}_x"],
                PROBOSCIS_Y_COL: df[f"cls8_{idx}_y"],
                PROBOSCIS_CORNERS_COL: np.nan,
                "x_anchor": df["x_anchor"],
                "y_anchor": df["y_anchor"],
                "distance_1_2": df[f"distance_class1_eye_{idx}"],
                PROBOSCIS_DISTANCE_COL: df[f"dist_eye_{idx}_cls8_{idx}"],
                "distance_2_anchor": df[f"dist_eye_{idx}_anchor"],
                "angle_deg_c2_26_vs_anchor": df[f"angle_eye_{idx}_cls8_vs_anchor"],
            }
        )

        if slot_df[["track_id_class2", "x_class2", "y_class2"]].notna().sum().sum() == 0:
            continue

        csv_path = out_dir / f"{folder_name}_fly{idx + 1}_distances.csv"
        slot_df.to_csv(csv_path, index=False)
        per_fly_paths.append(csv_path)

    return per_fly_paths

def _is_cuda_failure(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(tok in msg for tok in ("cuda", "cudnn", "cublas", "expandable_segments", "device-side assert", "hip"))


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
    else:
        log.info(f"Loading PyTorch model: {cfg.model_path} (export to .engine for 2-5x speedup)")
        model = YOLO(cfg.model_path)
    device_in_use: Optional[str] = None

    def _set_device(target: str):
        nonlocal device_in_use
        if device_in_use == target:
            return
        if target == "cpu":
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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

    def predict_fn(image, conf_thres):
        nonlocal device_in_use
        try:
            return model.predict(image, conf=conf_thres, verbose=False, device=device_in_use, half=True)
        except RuntimeError as exc:
            if device_in_use == "cuda" and cfg.allow_cpu and _is_cuda_failure(exc):
                log.warning("CUDA inference failed (%s); switching to CPU for the rest of the run.", exc)
                _set_device("cpu")
                return model.predict(image, conf=conf_thres, verbose=False, device=device_in_use, half=True)
            raise

    AX, AY = cfg.anchor_x, cfg.anchor_y

    # Handle main_directory as either string or list
    directories = cfg.main_directory if isinstance(cfg.main_directory, list) else [cfg.main_directory]

    for main_dir in directories:
        root = Path(main_dir).expanduser().resolve()
        if not root.is_dir():
            print(f"[YOLO] main_directory does not exist: {root}")
            continue

        for fly in [p for p in root.iterdir() if p.is_dir()]:
            video_files = [f for f in fly.iterdir() if f.suffix.lower() in (".mp4",".avi")]
            for video_path in video_files:
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

                if csv_file_path.exists():
                    df_timestamps = pd.read_csv(csv_file_path)
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

                fps = calculated_fps
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1080
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
                # Use H.264 codec (avc1) for better compression - output videos will be similar size to input
                writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))

                single_trackers = {
                    cls: SingleClassTracker(
                        iou_thres=cfg.iou_match_thres, max_age=cfg.max_age, ema_alpha=cfg.ema_alpha
                    )
                    for cls in SINGLE_TRACK_CLASSES
                }
                eye_mgr = EyeAnchorManager(max_eyes=active_max_flies, zero_iou_eps=cfg.zero_iou_epsilon)
                cls8_tracker = MultiObjectTracker(
                    iou_thres=cfg.iou_match_thres,
                    max_age=cfg.max_age,
                    ema_alpha=cfg.ema_alpha,
                    max_tracks=cfg.max_proboscis_tracks,
                )
                pairer = StablePairing(max_pairs=active_max_flies)

                target_w, target_h = (1080, 1080) if (w, h) != (1080, 1080) else (w, h)
                pairer.rebind_max_dist_px = cfg.pair_rebind_ratio * math.hypot(target_w, target_h)

                rows = []
                prev_gray = None
                frame_idx = 0
                t0 = time.time()
                while cap.isOpened() and frame_idx <= max_frame:
                    ok, frame = cap.read()
                    if not ok: break
                    if (w, h) != (1080, 1080):
                        frame = cv2.resize(frame, (1080,1080), interpolation=cv2.INTER_LINEAR)
                    ts = timestamps.get(frame_idx, frame_idx / fps)
                    frame, row, prev_gray = _process_frame(
                        frame,
                        frame_idx,
                        ts,
                        single_trackers,
                        prev_gray,
                        (AX, AY),
                        cfg,
                        predict_fn,
                        eye_mgr,
                        cls8_tracker,
                        pairer,
                        active_max_flies,
                    )
                    writer.write(frame)
                    rows.append(row)
                    frame_idx += 1
                cap.release(); writer.release()
                df_rows = pd.DataFrame(rows)
                merged_csv_path = out_dir / f"{folder_name}_distances_merged.csv"
                df_rows.to_csv(merged_csv_path, index=False)

                per_fly_paths = _export_per_fly_csvs(df_rows, out_dir, folder_name, cfg, active_max_flies)

                extra_msg = ""
                if per_fly_paths:
                    extra_msg = " (per-fly CSVs: " + ", ".join(p.name for p in per_fly_paths) + ")"
                print(f"[YOLO] {video_path.name} → {out_dir} in {time.time()-t0:.1f}s{extra_msg}")
