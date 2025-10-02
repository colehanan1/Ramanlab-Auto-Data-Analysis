
from __future__ import annotations
import os, time, logging, cv2, numpy as np, pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO
import torch

from ..config import Settings
from ..utils.timestamps import pick_timestamp_column, pick_frame_column, to_seconds_series
from ..utils.vision import order_corners, xyxy_to_cxcywh
from ..utils.track import SingleClassTracker

log = logging.getLogger("fbpipe.yolo")
logging.getLogger("ultralytics").setLevel(logging.WARNING)

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

def _process_frame(frame, frame_number, current_timestamp, trackers: Dict[int, SingleClassTracker],
                   prev_gray, anchor, settings: Settings, model: YOLO):
    CONF_THRES = settings.conf_thres
    FLOW_PARAMS = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    r = model.predict(source=frame, conf=CONF_THRES, verbose=False)[0]

    dets_by_class: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
        1:(np.zeros((0,4),np.float32), np.zeros((0,),np.float32)),
        2:(np.zeros((0,4),np.float32), np.zeros((0,),np.float32)),
        6:(np.zeros((0,4),np.float32), np.zeros((0,),np.float32)),
    }
    obb_corners_by_class: Dict[int, Optional[List[List[float]]]] = {1: None, 2: None, 6: None}

    if hasattr(r, 'obb') and r.obb is not None:
        xyxyxyxy = r.obb.xyxyxyxy.cpu().numpy()
        cls_arr  = r.obb.cls.cpu().numpy().astype(int)
        conf_arr = (r.obb.conf.cpu().numpy() if hasattr(r.obb, 'conf') and r.obb.conf is not None
                    else np.ones_like(cls_arr, dtype=np.float32))
        for i, (c, s) in enumerate(zip(cls_arr, conf_arr)):
            if c not in dets_by_class: continue
            corners = xyxyxyxy[i].reshape(4,2)
            x1, y1 = corners[:,0].min(), corners[:,1].min()
            x2, y2 = corners[:,0].max(), corners[:,1].max()
            prev_boxes, prev_scores = dets_by_class[c]
            if len(prev_scores)==0 or s > prev_scores[0]:
                dets_by_class[c] = (np.array([[x1,y1,x2,y2]], dtype=np.float32), np.array([s], dtype=np.float32))
                obb_corners_by_class[c] = order_corners(corners)
    else:
        xyxy = r.boxes.xyxy.cpu().numpy() if (r.boxes is not None and len(r.boxes)>0) else np.zeros((0,4), np.float32)
        cls_arr  = r.boxes.cls.cpu().numpy().astype(int) if (r.boxes is not None and len(r.boxes)>0) else np.zeros((0,), int)
        conf_arr = r.boxes.conf.cpu().numpy().astype(np.float32) if (r.boxes is not None and len(r.boxes)>0) else np.zeros((0,), np.float32)
        for b, c, s in zip(xyxy, cls_arr, conf_arr):
            if c not in dets_by_class: continue
            prev_boxes, prev_scores = dets_by_class[c]
            if len(prev_scores)==0 or s > prev_scores[0]:
                dets_by_class[c] = (np.array([b], dtype=np.float32), np.array([s], np.float32))
                obb_corners_by_class[c] = None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    out_by_class = {}
    for cls_id in [1,2,6]:
        boxes, scores = dets_by_class[cls_id]
        best_track = trackers[cls_id].step(boxes, scores)
        corners = obb_corners_by_class[cls_id]
        if best_track is None:
            out_by_class[cls_id] = dict(track_id=np.nan, x=np.nan, y=np.nan, corners=np.nan)
            continue
        box = best_track.box_xyxy
        if settings.use_optical_flow and best_track.time_since_update > 0:
            box = _process_frame.flow_nudge(prev_gray, gray, box) if False else _flow_nudge(prev_gray, gray, box, settings.flow_skip_edge, FLOW_PARAMS)
            best_track.box_xyxy = box
        cx, cy, w, h = xyxy_to_cxcywh(box)
        out_by_class[cls_id] = dict(track_id=best_track.id, x=float(cx), y=float(cy), corners=(corners if corners is not None else np.nan))

    # distances
    def compute_distance(a, b):
        return float(np.hypot(a["x"] - b["x"], a["y"] - b["y"]))

    det1 = out_by_class.get(1, {"track_id":np.nan,"x":np.nan,"y":np.nan,"corners":np.nan})
    det2 = out_by_class.get(2, {"track_id":np.nan,"x":np.nan,"y":np.nan,"corners":np.nan})
    det6 = out_by_class.get(6, {"track_id":np.nan,"x":np.nan,"y":np.nan,"corners":np.nan})
    AX, AY = anchor

    d12 = d26 = d2A = np.nan
    if not (np.isnan(det1["x"]) or np.isnan(det2["x"])):
        d12 = compute_distance(det1, det2); cv2.line(frame, (int(det1["x"]), int(det1["y"])), (int(det2["x"]), int(det2["y"])), (0,255,0), 6)
    if not (np.isnan(det2["x"]) or np.isnan(det6["x"])):
        d26 = compute_distance(det2, det6); cv2.line(frame, (int(det2["x"]), int(det2["y"])), (int(det6["x"]), int(det6["y"])), (255,0,0), 6)
    if not np.isnan(det2["x"]):
        d2A = float(np.hypot(det2["x"] - AX, det2["y"] - AY))
        cv2.line(frame, (int(det2["x"]), int(det2["y"])), (int(AX), int(AY)), (0,165,255), 6)

    # angle @ class2 between (2→6) and (2→ANCHOR)
    def angle_deg_between(v1, v2):
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        return float(np.degrees(np.arctan2(abs(cross), dot)))
    angle_deg = np.nan
    if not (np.isnan(det2["x"]) or np.isnan(det6["x"])):
        v_26 = (det6["x"] - det2["x"], det6["y"] - det2["y"])
        v_2A = (AX - det2["x"], AY - det2["y"])
        angle_deg = angle_deg_between(v_26, v_2A)

    row = {
        "frame": frame_number, "timestamp": current_timestamp,
        "track_id_class1": det1["track_id"], "x_class1": det1["x"], "y_class1": det1["y"], "corners_class1": str(det1["corners"]),
        "track_id_class2": det2["track_id"], "x_class2": det2["x"], "y_class2": det2["y"], "corners_class2": str(det2["corners"]),
        "track_id_class6": det6["track_id"], "x_class6": det6["x"], "y_class6": det6["y"], "corners_class6": str(det6["corners"]),
        "x_anchor": AX, "y_anchor": AY,
        "distance_1_2": d12, "distance_2_6": d26, "distance_2_anchor": d2A,
        "angle_deg_c2_26_vs_anchor": angle_deg
    }
    return frame, row, gray

def main(cfg: Settings):
    allocator_was_set = False
    if torch.cuda.is_available() and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments=True"
        allocator_was_set = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = cfg.cuda_allow_tf32
    torch.backends.cudnn.allow_tf32 = cfg.cuda_allow_tf32
    if not torch.cuda.is_available() and not cfg.allow_cpu:
        raise RuntimeError("CUDA is not available. Set allow_cpu: true only for smoke tests.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(cfg.model_path)
    try:
        model.to(device)
    except RuntimeError as exc:
        if "expandable_segments" in str(exc) and allocator_was_set:
            log.warning("CUDA allocator does not support expandable_segments; retrying without it.")
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            model.to(device)
        else:
            raise

    AX, AY = cfg.anchor_x, cfg.anchor_y
    root = Path(cfg.main_directory).expanduser().resolve()
    if not root.is_dir():
        print(f"[YOLO] main_directory does not exist: {root}")
        return

    for fly in [p for p in root.iterdir() if p.is_dir()]:
        video_files = [f for f in fly.iterdir() if f.suffix.lower() in (".mp4",".avi")]
        for video_path in video_files:
            base = video_path.stem
            csv_file_path = fly / f"{base.replace('_preprocessed','')}.csv"
            parts = base.split("_")
            folder_name = "_".join(parts[1:7]) if len(parts)>=7 else base
            out_dir = fly / folder_name
            out_dir.mkdir(exist_ok=True)
            out_mp4 = out_dir / f"{folder_name}_distance_annotated.mp4"
            if out_mp4.exists():
                print(f"[YOLO] Skipping (exists): {out_mp4}")
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"[YOLO] Cannot open {video_path}")
                continue

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
            writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            trackers = {
                1: SingleClassTracker(iou_thres=cfg.iou_match_thres, max_age=cfg.max_age, ema_alpha=cfg.ema_alpha),
                2: SingleClassTracker(iou_thres=cfg.iou_match_thres, max_age=cfg.max_age, ema_alpha=cfg.ema_alpha),
                6: SingleClassTracker(iou_thres=cfg.iou_match_thres, max_age=cfg.max_age, ema_alpha=cfg.ema_alpha),
            }

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
                frame, row, prev_gray = _process_frame(frame, frame_idx, ts, trackers, prev_gray, (AX,AY), cfg, model)
                writer.write(frame)
                rows.append(row)
                frame_idx += 1
            cap.release(); writer.release()
            pd.DataFrame(rows).to_csv(out_dir / f"{folder_name}_distances_merged.csv", index=False)
            print(f"[YOLO] {video_path.name} → {out_dir} in {time.time()-t0:.1f}s")
