#!/usr/bin/env python
"""
Standalone starter script: deploy the 4-class OBB model (eye, proboscis, thorax,
abdomenTip) on a folder of fly videos and produce, per video:

  * <video>_tracked.mp4       - overlay video (all 4 classes) + a live plot panel
  * <video>_flyN_metrics.png  - static per-fly plot (PER + abdomen dist/angle)
  * <video>_tracking.csv       - per-frame, per-fly metrics

It mirrors how the fbpipe pipeline runs YOLO tracking WITHOUT importing or
modifying the pipeline:

  - eye + thorax NEVER move (fly is glued): they are found in the first few
    frames only and frozen as per-fly anchors, numbered top->bottom by y
    (fly 1 = topmost), exactly like the pipeline's EyeAnchorManager.
  - proboscis and abdomenTip DO move: they are tracked frame-to-frame with a
    small memory (each frame is matched to the previous frame's position), and
    short gaps (missed detections) are interpolated from the previous/next
    detected position.
  - proboscis is paired to its eye anchor  -> PER distance + angle,
    abdomenTip is paired to its thorax anchor (top abdomen with top thorax)
    -> abdomen distance + angle (curl/extension).

Nothing here is imported by the pipeline; it is a self-contained deploy demo.

Usage:
    conda activate yolo-env
    python scripts/analysis/abdomen_per_tracking.py \
        --folder /home/ramanlab/Documents/cole/Data/test-data/abdomenTrackingTest/Batch1 \
        [--video output_..._testing_3_...mp4]   # limit to one video (smoke test)
        [--max-frames N]                          # cap frames (debug)
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Config / constants
# ----------------------------------------------------------------------------
DEFAULT_MODEL = (
    "/home/ramanlab/Documents/cole/model/PER+Abdomen-BehaviorTracking/"
    "runs/obb/train/weights/best.pt"
)
DEFAULT_FOLDER = (
    "/home/ramanlab/Documents/cole/Data/test-data/abdomenTrackingTest/Batch1"
)
OUT_SUBDIR = "abdomen_tracking_out"

# Class ids as trained (verified): {0: abdomenTip, 1: eye, 2: proboscis, 3: thorax}
CLS_ABDOMENTIP, CLS_EYE, CLS_PROBOSCIS, CLS_THORAX = 0, 1, 2, 3

CONF_THRES = 0.40          # matches pipeline yolo.conf_thres (per-frame tracking)
WARMUP_CONF = 0.15         # lower conf when locking anchors so low-conf eyes appear
IMGSZ = 1088               # matches training imgsz
FRAME_SIZE = 1080          # pipeline resizes frames to 1080x1080
BATCH = 16                 # frames per predict batch
WARMUP_FRAMES = 5          # eye+thorax never move: find them in the first 5 frames
MAX_FLIES = 3              # SOP: 3 flies per batch
EYE_TO_THORAX_MATCH = 250.0  # an eye anchor is within this many px of its thorax
ANCHOR_X, ANCHOR_Y = 1079.0, 540.0   # global angle reference (pipeline anchor)

# --- percentage cross-trial plots (--percent-plots) ---
PLOT_MAX_SECONDS = 90.0     # show only the first 90 s (30 baseline + 30 odor + 30 post)
ODOR_ON_S = 30.0            # odor on  (black line)
ODOR_OFF_S = 60.0          # odor off (black line)
# distance signals on the left axis, as signed % deviation from resting (0 = avg);
# (label, csv_column, color)
DIST_SIGNALS = [
    ("proboscis dist (% from rest)", "per_distance", "#1f77b4"),
    ("abdomen dist (% from rest)", "abdomen_distance", "#d62728"),
]
# abdomen angle is shown separately on the right axis, signed vs resting (0 = avg)
ANGLE_COL = "abdomen_angle"
ANGLE_COLOR = "#2ca02c"

DIAG = math.hypot(FRAME_SIZE, FRAME_SIZE)   # ~1527 px
# Frame-to-frame "memory" jump gates: a moving part must stay near its last
# known position, else it is treated as missing (interpolation bridges the gap).
PROB_MAX_JUMP = 120.0
ABD_MAX_JUMP = 250.0
# Init/sanity gates when there is no memory yet (first frame or after a long gap).
PROB_ANCHOR_GATE = 0.2 * DIAG   # ~305 px (proboscis stays near its eye)
# Hard cap on the thorax -> abdomenTip distance: an abdomenTip is never paired to
# a thorax farther than this (enforced on every frame, memory-tracked included).
ABD_ANCHOR_GATE = 350.0
# A fly's abdomen is always at/below its own thorax; it never sits more than this
# many px ABOVE the thorax (smaller y). Forbids pairing an abdomenTip to a thorax
# that is well below it (i.e. it belongs to a fly higher up).
ABD_MAX_ABOVE_ANCHOR = 25.0

# Overlay colors (BGR)
COL = {
    "eye": (0, 255, 255),        # yellow
    "proboscis": (255, 128, 0),  # blue-ish
    "thorax": (0, 200, 0),       # green
    "abdomenTip": (255, 0, 255), # magenta
    "per_line": (0, 255, 0),     # green eye->proboscis
    "abd_line": (255, 0, 255),   # magenta thorax->abdomenTip
}
FLY_COLORS = [(60, 76, 231), (219, 152, 52), (96, 174, 39)]  # per-fly (BGR)


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def angle_between(v1, v2):
    """Unsigned angle (deg) between two vectors, like the pipeline's helper."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return float(np.degrees(np.arctan2(abs(cross), dot)))


def vector_angle_deg(dx, dy):
    """Signed direction of a vector in image coords, deg in [-180, 180].
    0 = pointing +x (right), +90 = pointing down (+y)."""
    return float(np.degrees(np.arctan2(dy, dx)))


# ----------------------------------------------------------------------------
# Detection extraction
# ----------------------------------------------------------------------------
def detections_by_class(result):
    """From an ultralytics OBB result -> dict {cls_id: list of (cx,cy,conf,corners)}."""
    out = {c: [] for c in (CLS_ABDOMENTIP, CLS_EYE, CLS_PROBOSCIS, CLS_THORAX)}
    obb = getattr(result, "obb", None)
    if obb is None or obb.xywhr is None or len(obb) == 0:
        return out
    xywhr = obb.xywhr.cpu().numpy()
    corners = obb.xyxyxyxy.cpu().numpy()  # (N,4,2)
    cls = obb.cls.cpu().numpy().astype(int)
    conf = obb.conf.cpu().numpy()
    for i in range(len(cls)):
        c = int(cls[i])
        if c in out:
            out[c].append((float(xywhr[i, 0]), float(xywhr[i, 1]),
                           float(conf[i]), corners[i]))
    return out


def top_k_by_conf(dets, k):
    return sorted(dets, key=lambda d: d[2], reverse=True)[:k]


def _aabb(corners):
    """Axis-aligned bbox (x1,y1,x2,y2) from OBB corners, like the pipeline does."""
    xs, ys = corners[:, 0], corners[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def dedup_zero_iou(dets, k, iou_eps=1e-6):
    """Keep the top-k detections by confidence such that no two kept boxes
    overlap (pairwise IoU <= iou_eps). Mirrors the pipeline's
    enforce_zero_iou_and_topk: prevents two overlapping detections (e.g. two
    abdomen tips, or a duplicate box) from being tracked as one object."""
    ordered = sorted(dets, key=lambda d: d[2], reverse=True)
    kept, kept_boxes = [], []
    for d in ordered:
        b = _aabb(d[3])
        if all(_iou(b, kb) <= iou_eps for kb in kept_boxes):
            kept.append(d)
            kept_boxes.append(b)
        if len(kept) >= k:
            break
    return kept


# ----------------------------------------------------------------------------
# Odor window from sibling frame CSV (ActiveOFM column)
# ----------------------------------------------------------------------------
def load_active_ofm(video_path: Path):
    """Return list[bool] of odor-on state per frame from the sibling
    output_*.csv (ActiveOFM). None if not found."""
    csv_path = video_path.with_suffix(".csv")
    if not csv_path.exists():
        return None
    states = []
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if "ActiveOFM" not in (reader.fieldnames or []):
                return None
            for r in reader:
                # ActiveOFM is 'off' when no odor, or an odor code (OFM_H, OFM_B,
                # ...) while the olfactory machine is delivering odor.
                v = str(r.get("ActiveOFM", "off")).strip().lower()
                states.append(v not in ("off", "", "none", "nan"))
    except Exception:
        return None
    return states


def ofm_on_span(states):
    """First contiguous odor-on span (start_idx, end_idx), else None."""
    if not states:
        return None
    on = [i for i, s in enumerate(states) if s]
    if not on:
        return None
    return on[0], on[-1]


# ----------------------------------------------------------------------------
# Warm-up: eye + thorax never move -> find in first N frames, freeze, number
# flies top->bottom. Also capture a representative box (corners) for drawing.
# ----------------------------------------------------------------------------
def build_anchors(model, cap, n_warmup):
    """Lock eye + thorax anchors from the first `n_warmup` frames (they never
    move). Fly count comes from the THORAX class (reliable ~0.99 conf); eyes are
    detected at a lower conf and each eye is matched to its thorax band, because
    the top fly's eye is often low-confidence."""
    eye_dets, thorax_dets, thorax_counts = [], [], []
    frames_read = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    batch = []
    while frames_read < n_warmup:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[0] != FRAME_SIZE or frame.shape[1] != FRAME_SIZE:
            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        batch.append(frame)
        frames_read += 1
        if len(batch) == BATCH or frames_read >= n_warmup:
            results = model.predict(batch, conf=WARMUP_CONF, imgsz=IMGSZ,
                                    verbose=False, half=True)
            for res in results:
                d = detections_by_class(res)
                # count flies from reliable (>= CONF_THRES) thorax detections
                thx_reliable = dedup_zero_iou(
                    [t for t in d[CLS_THORAX] if t[2] >= CONF_THRES], MAX_FLIES)
                thorax_counts.append(len(thx_reliable))
                thorax_dets.extend(thx_reliable)
                eye_dets.extend(dedup_zero_iou(d[CLS_EYE], MAX_FLIES + 1))
            batch = []

    if not thorax_counts or max(thorax_counts) == 0:
        return []
    # median thorax count -> robust to an occasional missed/extra thorax
    n_flies = int(np.clip(round(float(np.median(thorax_counts))), 1, MAX_FLIES))

    def cluster_topk(dets, n):
        """Cluster detections into n vertical bands; return per-band median
        center + the corners of the detection nearest that center."""
        if not dets:
            return []
        order = np.argsort([d[1] for d in dets])
        dets_sorted = [dets[i] for i in order]
        bands = np.array_split(np.arange(len(dets_sorted)), n)
        centers = []
        for band in bands:
            if len(band) == 0:
                continue
            band_dets = [dets_sorted[i] for i in band]
            cx = float(np.median([d[0] for d in band_dets]))
            cy = float(np.median([d[1] for d in band_dets]))
            nearest = min(band_dets, key=lambda d: math.hypot(d[0] - cx, d[1] - cy))
            centers.append(((cx, cy), nearest[3]))
        centers.sort(key=lambda c: c[0][1])  # top->bottom
        return centers

    thoraxes = cluster_topk(thorax_dets, n_flies)  # [((cx,cy), corners)], top->bottom

    # match each eye detection to its nearest thorax band; average per fly
    eye_groups = [[] for _ in range(len(thoraxes))]
    tcenters = [c[0] for c in thoraxes]
    for e in eye_dets:
        di = int(np.argmin([math.hypot(e[0] - tc[0], e[1] - tc[1]) for tc in tcenters]))
        if math.hypot(e[0] - tcenters[di][0], e[1] - tcenters[di][1]) <= EYE_TO_THORAX_MATCH:
            eye_groups[di].append(e)

    flies = []
    for i, (tcenter, tcorners) in enumerate(thoraxes):
        grp = eye_groups[i]
        if grp:
            ecenter = (float(np.median([e[0] for e in grp])),
                       float(np.median([e[1] for e in grp])))
            ecorners = min(grp, key=lambda e: math.hypot(e[0] - ecenter[0],
                                                         e[1] - ecenter[1]))[3]
        else:
            # no eye found near this thorax -> fall back to thorax as eye anchor
            ecenter, ecorners = tcenter, None
        flies.append({"fly": i + 1, "eye": ecenter, "thorax": tcenter,
                      "eye_corners": ecorners, "thorax_corners": tcorners})
    return flies


# ----------------------------------------------------------------------------
# Tracking with memory + gap interpolation for a moving part (proboscis / abdomen)
# ----------------------------------------------------------------------------
def track_part(dets_per_frame, part_key, anchors, init_mode, max_jump, anchor_gate,
               max_above=None, hard_anchor_cap=None):
    """Assign one detection per fly per frame using frame-to-frame memory.

    Per frame:
      1. Flies with a remembered position match the nearest detection within
         max_jump (one-to-one, closest first).
      2. Any remaining flies/detections are initialised by `init_mode`:
         'rank'   -> top-to-bottom order (top abdomen with top thorax),
         'nearest'-> nearest anchor (proboscis sits on its eye),
         gated by anchor_gate.
    `max_above` (if set) forbids pairing a detection to a fly when the detection
    sits more than max_above px ABOVE that fly's anchor (smaller y) -- used so an
    abdomenTip is never bound to a thorax well below it (a higher fly's abdomen).
    `hard_anchor_cap` (if set) is a HARD max distance from the fly's anchor,
    enforced on EVERY frame (memory-tracked included): a detection farther than
    this from the anchor is never that fly's part (e.g. abdomenTip > 350px from
    its thorax is the wrong abdomen).
    Then internal gaps (missed frames) are linearly interpolated from the
    previous/next known position. Leading/trailing gaps are left empty.

    Returns series[fly] = list over frames of dict:
        {"x","y","corners"(np or None),"src": "det"|"interp"} or None.
    """
    n_flies = len(anchors)
    n = len(dets_per_frame)
    series = [[None] * n for _ in range(n_flies)]
    prev = [None] * n_flies  # last known (x, y) per fly

    def eligible(di, fly, dets):
        """A detection may be this fly's part only if it is (a) not far above the
        fly's anchor and (b) within the hard distance cap of the anchor."""
        if max_above is not None and dets[di][1] < anchors[fly][1] - max_above:
            return False
        if hard_anchor_cap is not None:
            if math.hypot(dets[di][0] - anchors[fly][0],
                          dets[di][1] - anchors[fly][1]) > hard_anchor_cap:
                return False
        return True

    for fr in range(n):
        dets = dets_per_frame[fr].get(part_key, [])
        assigned = {}   # fly -> det index
        used = set()

        # (1) memory match
        mem = []
        for fly in range(n_flies):
            if prev[fly] is None:
                continue
            for di, d in enumerate(dets):
                if not eligible(di, fly, dets):
                    continue
                c = math.hypot(d[0] - prev[fly][0], d[1] - prev[fly][1])
                if c <= max_jump:
                    mem.append((c, fly, di))
        mem.sort(key=lambda t: t[0])
        for c, fly, di in mem:
            if fly in assigned or di in used:
                continue
            assigned[fly] = di
            used.add(di)

        # (2) init the rest
        rem_flies = [f for f in range(n_flies) if f not in assigned]
        rem_dets = [di for di in range(len(dets)) if di not in used]
        if rem_flies and rem_dets:
            if init_mode == "rank":
                rd = sorted(rem_dets, key=lambda di: dets[di][1])  # by y
                rf = sorted(rem_flies, key=lambda f: anchors[f][1])
                for f, di in zip(rf, rd):
                    if not eligible(di, f, dets):
                        continue
                    if math.hypot(dets[di][0] - anchors[f][0],
                                  dets[di][1] - anchors[f][1]) <= anchor_gate:
                        assigned[f] = di
                        used.add(di)
            else:  # nearest anchor
                cc = []
                for f in rem_flies:
                    for di in rem_dets:
                        if not eligible(di, f, dets):
                            continue
                        c = math.hypot(dets[di][0] - anchors[f][0],
                                       dets[di][1] - anchors[f][1])
                        if c <= anchor_gate:
                            cc.append((c, f, di))
                cc.sort(key=lambda t: t[0])
                for c, f, di in cc:
                    if f in assigned or di in used:
                        continue
                    assigned[f] = di
                    used.add(di)

        # record
        for fly in range(n_flies):
            if fly in assigned:
                d = dets[assigned[fly]]
                series[fly][fr] = {"x": d[0], "y": d[1], "corners": d[3], "src": "det"}
                prev[fly] = (d[0], d[1])
            # else: leave None, keep stale prev so memory survives short gaps

    # interpolate internal gaps
    for fly in range(n_flies):
        seq = series[fly]
        known = [i for i, v in enumerate(seq) if v is not None]
        for k in range(len(known) - 1):
            i, j = known[k], known[k + 1]
            if j - i > 1:
                xi, yi = seq[i]["x"], seq[i]["y"]
                xj, yj = seq[j]["x"], seq[j]["y"]
                for m in range(i + 1, j):
                    t = (m - i) / (j - i)
                    seq[m] = {"x": xi + (xj - xi) * t, "y": yi + (yj - yi) * t,
                              "corners": None, "src": "interp"}
    return series


# ----------------------------------------------------------------------------
# Metrics from tracked positions
# ----------------------------------------------------------------------------
def compute_metrics(flies, prob_series, abd_series, n_frames):
    """metrics[frame] = list of per-fly dicts."""
    metrics = []
    for fr in range(n_frames):
        rows = []
        for fi, f in enumerate(flies):
            ex, ey = f["eye"]
            tx, ty = f["thorax"]
            row = {"fly": f["fly"], "eye_x": ex, "eye_y": ey,
                   "thorax_x": tx, "thorax_y": ty,
                   "proboscis_x": np.nan, "proboscis_y": np.nan,
                   "per_distance": np.nan, "per_angle": np.nan, "per_src": "",
                   "abdomenTip_x": np.nan, "abdomenTip_y": np.nan,
                   "abdomen_distance": np.nan, "abdomen_angle": np.nan,
                   "abdomen_src": ""}
            p = prob_series[fi][fr]
            if p is not None:
                row["proboscis_x"], row["proboscis_y"] = p["x"], p["y"]
                row["per_distance"] = float(math.hypot(p["x"] - ex, p["y"] - ey))
                v_ep = (p["x"] - ex, p["y"] - ey)
                v_ea = (ANCHOR_X - ex, ANCHOR_Y - ey)
                row["per_angle"] = angle_between(v_ep, v_ea)
                row["per_src"] = p["src"]
            a = abd_series[fi][fr]
            if a is not None:
                row["abdomenTip_x"], row["abdomenTip_y"] = a["x"], a["y"]
                row["abdomen_distance"] = float(math.hypot(a["x"] - tx, a["y"] - ty))
                row["abdomen_angle"] = vector_angle_deg(a["x"] - tx, a["y"] - ty)
                row["abdomen_src"] = a["src"]
            rows.append(row)
        metrics.append(rows)
    return metrics


# ----------------------------------------------------------------------------
# Drawing overlay onto a frame using the tracked positions
# ----------------------------------------------------------------------------
def draw_overlay(frame, fr, flies, prob_series, abd_series, metrics):
    for fi, f in enumerate(flies):
        ex, ey = f["eye"]
        tx, ty = f["thorax"]
        fc = FLY_COLORS[fi % len(FLY_COLORS)]
        row = metrics[fr][fi]

        # static eye + thorax (never move)
        if f["eye_corners"] is not None:
            cv2.polylines(frame, [f["eye_corners"].astype(np.int32)], True, COL["eye"], 2)
        cv2.circle(frame, (int(ex), int(ey)), 5, COL["eye"], -1)
        if f["thorax_corners"] is not None:
            cv2.polylines(frame, [f["thorax_corners"].astype(np.int32)], True, COL["thorax"], 2)
        cv2.circle(frame, (int(tx), int(ty)), 5, COL["thorax"], -1)
        cv2.putText(frame, f"fly{f['fly']}", (int(ex) - 10, int(ey) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fc, 2)

        # proboscis (tracked) + PER line
        p = prob_series[fi][fr]
        if p is not None:
            px, py = int(p["x"]), int(p["y"])
            if p["corners"] is not None:
                cv2.polylines(frame, [p["corners"].astype(np.int32)], True, COL["proboscis"], 2)
            else:  # interpolated -> marker
                cv2.circle(frame, (px, py), 7, COL["proboscis"], 2)
            cv2.line(frame, (int(ex), int(ey)), (px, py), COL["per_line"], 3)
            cv2.putText(frame, f"PER {row['per_distance']:.0f}px", (px + 6, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL["per_line"], 2)

        # abdomenTip (tracked) + abdomen line
        a = abd_series[fi][fr]
        if a is not None:
            ax, ay = int(a["x"]), int(a["y"])
            if a["corners"] is not None:
                cv2.polylines(frame, [a["corners"].astype(np.int32)], True, COL["abdomenTip"], 2)
            else:
                cv2.circle(frame, (ax, ay), 7, COL["abdomenTip"], 2)
            cv2.line(frame, (int(tx), int(ty)), (ax, ay), COL["abd_line"], 3)
            cv2.putText(frame,
                        f"abd {row['abdomen_distance']:.0f}px {row['abdomen_angle']:.0f}deg",
                        (ax + 6, ay), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL["abd_line"], 2)


# ----------------------------------------------------------------------------
# Plot panel (pre-rendered once, moving cursor composited per frame)
# ----------------------------------------------------------------------------
def render_panel(metrics, n_frames, fps, flies, ofm_span, width, panel_h):
    t = np.arange(n_frames) / fps
    fig, axes = plt.subplots(3, 1, figsize=(width / 100, panel_h / 100),
                             dpi=100, sharex=True)
    specs = [("PER distance [px]", "per_distance"),
             ("Abdomen dist [px]", "abdomen_distance"),
             ("Abdomen angle [deg]", "abdomen_angle")]
    for ax, (title, key) in zip(axes, specs):
        for fi, f in enumerate(flies):
            series = np.array([metrics[fr][fi][key] if fi < len(metrics[fr]) else np.nan
                               for fr in range(n_frames)], dtype=float)
            c = tuple(reversed([v / 255 for v in FLY_COLORS[fi % len(FLY_COLORS)]]))
            ax.plot(t, series, color=c, lw=1.0, label=f"fly {f['fly']}")
        if ofm_span is not None:
            ax.axvspan(ofm_span[0] / fps, ofm_span[1] / fps, color="orange", alpha=0.15)
        ax.set_ylabel(title, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=6, loc="upper right", ncol=len(flies))
    axes[-1].set_xlabel("time [s]", fontsize=7)
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    panel = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    panel = cv2.resize(panel, (width, panel_h))
    x0_ax, x1_ax = axes[0].get_position().x0, axes[0].get_position().x1
    plt.close(fig)
    t_max = t[-1] if len(t) else 1.0

    def x_of_frame(fr):
        frac = (fr / fps) / t_max if t_max > 0 else 0
        return int((x0_ax + frac * (x1_ax - x0_ax)) * width)

    return panel, x_of_frame


def save_static_plots(metrics, n_frames, fps, flies, ofm_span, out_dir, stem):
    t = np.arange(n_frames) / fps
    for fi, f in enumerate(flies):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        specs = [("Proboscis extension (PER) distance [px]", "per_distance"),
                 ("Abdomen tip distance from thorax [px]", "abdomen_distance"),
                 ("Abdomen tip angle (thorax->tip) [deg]", "abdomen_angle")]
        for ax, (title, key) in zip(axes, specs):
            series = np.array([metrics[fr][fi][key] if fi < len(metrics[fr]) else np.nan
                               for fr in range(n_frames)], dtype=float)
            ax.plot(t, series, color="#c0392b", lw=1.2)
            if ofm_span is not None:
                ax.axvspan(ofm_span[0] / fps, ofm_span[1] / fps, color="orange",
                           alpha=0.2, label="odor on")
            ax.set_ylabel(title, fontsize=9)
            ax.grid(alpha=0.3)
        if ofm_span is not None:
            axes[0].legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("time [s]")
        fig.suptitle(f"{stem}  -  fly {f['fly']}", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}_fly{f['fly']}_metrics.png", dpi=120)
        plt.close(fig)


# ----------------------------------------------------------------------------
# Process a single video
# ----------------------------------------------------------------------------
def process_video(model, video_path: Path, out_root: Path, max_frames=None):
    stem = video_path.stem
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {stem} ===")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ! could not open {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 40.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(n_total, max_frames) if max_frames else n_total

    # 1) freeze eye + thorax anchors from the first few frames
    flies = build_anchors(model, cap, WARMUP_FRAMES)
    if not flies:
        print("  ! no flies detected in warm-up; skipping")
        cap.release()
        return
    print("  flies (top->bottom): "
          + ", ".join(f"fly{f['fly']}@eye({f['eye'][0]:.0f},{f['eye'][1]:.0f})"
                      for f in flies))

    # 2) odor window
    states = load_active_ofm(video_path)
    ofm_span = ofm_on_span(states) if states else None
    if ofm_span:
        print(f"  odor-on frames {ofm_span[0]}..{ofm_span[1]} "
              f"({ofm_span[0]/fps:.1f}-{ofm_span[1]/fps:.1f}s)")

    # 3) PASS 1 (detect): store per-frame proboscis + abdomenTip detections
    dets_per_frame = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fidx = 0
    batch = []
    while fidx < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[0] != FRAME_SIZE or frame.shape[1] != FRAME_SIZE:
            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        batch.append(frame)
        fidx += 1
        if len(batch) == BATCH or fidx >= n_frames:
            results = model.predict(batch, conf=CONF_THRES, imgsz=IMGSZ,
                                    verbose=False, half=True)
            for res in results:
                d = detections_by_class(res)
                dets_per_frame.append({
                    # zero-IoU dedup so overlapping boxes aren't tracked as one
                    "proboscis": dedup_zero_iou(d[CLS_PROBOSCIS], MAX_FLIES),
                    "abdomenTip": dedup_zero_iou(d[CLS_ABDOMENTIP], MAX_FLIES),
                })
            batch = []
        if fidx % 1000 == 0:
            print(f"    [detect] frame {fidx}/{n_frames}")
    n_done = len(dets_per_frame)

    # 4) track moving parts with memory + gap interpolation
    eye_anchors = [f["eye"] for f in flies]
    thorax_anchors = [f["thorax"] for f in flies]
    prob_series = track_part(dets_per_frame, "proboscis", eye_anchors,
                             init_mode="nearest", max_jump=PROB_MAX_JUMP,
                             anchor_gate=PROB_ANCHOR_GATE)
    abd_series = track_part(dets_per_frame, "abdomenTip", thorax_anchors,
                            init_mode="rank", max_jump=ABD_MAX_JUMP,
                            anchor_gate=ABD_ANCHOR_GATE,
                            max_above=ABD_MAX_ABOVE_ANCHOR,
                            hard_anchor_cap=ABD_ANCHOR_GATE)

    # 5) metrics + panel + static plots + CSV
    metrics = compute_metrics(flies, prob_series, abd_series, n_done)
    panel_h = int(FRAME_SIZE * 0.45)
    panel, x_of_frame = render_panel(metrics, n_done, fps, flies, ofm_span,
                                     FRAME_SIZE, panel_h)
    save_static_plots(metrics, n_done, fps, flies, ofm_span, out_dir, stem)
    write_csv(out_dir / f"{stem}_tracking.csv", metrics, states, fps, n_done)

    # 6) PASS 2 (draw): re-read frames, overlay from tracks, stack panel + cursor
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_composite = out_dir / f"{stem}_composite_tmp.mp4"
    writer = cv2.VideoWriter(str(tmp_composite), fourcc, fps,
                             (FRAME_SIZE, FRAME_SIZE + panel_h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for fr in range(n_done):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[0] != FRAME_SIZE or frame.shape[1] != FRAME_SIZE:
            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        draw_overlay(frame, fr, flies, prob_series, abd_series, metrics)
        composite = np.vstack([frame, panel.copy()])
        cx = x_of_frame(fr)
        cv2.line(composite, (cx, FRAME_SIZE), (cx, FRAME_SIZE + panel_h),
                 (0, 0, 255), 2)
        writer.write(composite)
        if fr % 1000 == 0 and fr:
            print(f"    [draw] frame {fr}/{n_done}")
    writer.release()
    cap.release()

    # 7) transcode to h264 (arg list, no shell -> no command-injection)
    final_video = out_dir / f"{stem}_tracked.mp4"
    rc = 1
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_composite),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "25",
             str(final_video)],
            capture_output=True,
        )
        rc = proc.returncode
    except FileNotFoundError:
        rc = 1
    if rc == 0 and final_video.exists():
        tmp_composite.unlink(missing_ok=True)
    else:
        tmp_composite.rename(final_video)

    print(f"  wrote {final_video.name}, {len(flies)} fly plot(s), "
          f"{stem}_tracking.csv ({n_done} frames)")


def write_csv(csv_path, metrics, states, fps, n_frames):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "active_ofm", "fly",
                    "eye_x", "eye_y", "thorax_x", "thorax_y",
                    "proboscis_x", "proboscis_y", "per_distance", "per_angle", "per_src",
                    "abdomenTip_x", "abdomenTip_y", "abdomen_distance", "abdomen_angle",
                    "abdomen_src"])
        for fr in range(n_frames):
            ofm = states[fr] if states and fr < len(states) else ""
            for r in metrics[fr]:
                w.writerow([fr, round(fr / fps, 4), ofm, r["fly"],
                            f"{r['eye_x']:.1f}", f"{r['eye_y']:.1f}",
                            f"{r['thorax_x']:.1f}", f"{r['thorax_y']:.1f}",
                            _fmt(r["proboscis_x"]), _fmt(r["proboscis_y"]),
                            _fmt(r["per_distance"]), _fmt(r["per_angle"]), r["per_src"],
                            _fmt(r["abdomenTip_x"]), _fmt(r["abdomenTip_y"]),
                            _fmt(r["abdomen_distance"]), _fmt(r["abdomen_angle"]),
                            r["abdomen_src"]])


def _fmt(v):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.2f}"


# ----------------------------------------------------------------------------
# Percentage cross-trial plots (one figure per fly; trials stacked top->bottom)
# ----------------------------------------------------------------------------
def percent_scale(values, lo_p=1.0, hi_p=99.0):
    """Return (lo, hi) reference for scaling to %: robust min/max of `values`
    using the 1st/99th percentiles (so one tracking glitch doesn't define
    100%). `values` is pooled across ALL trials of a fly so the scale is shared
    and the largest value seen across all videos maps to 100%."""
    valid = np.asarray(values, dtype=float)
    valid = valid[~np.isnan(valid)]
    if valid.size < 3:
        return None
    lo, hi = float(np.percentile(valid, lo_p)), float(np.percentile(valid, hi_p))
    return (lo, hi) if hi > lo else None


def apply_percent(arr, scale):
    """Scale a signal to 0-100% using a shared (lo, hi). NaN stays NaN."""
    arr = np.asarray(arr, dtype=float)
    if scale is None:
        return np.full_like(arr, np.nan)
    lo, hi = scale
    return np.clip(100.0 * (arr - lo) / (hi - lo), 0.0, 100.0)


def signed_percent_ref(values, p=99.0):
    """Reference (resting_median, max_abs_dev) for a signed % that is CENTERED on
    the resting/average value: 0% = median (resting), +/-100% = the p-th
    percentile of |value - median| pooled across all trials (robust to glitches).
    `values` is pooled across ALL trials of a fly so the reference is shared."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size < 3:
        return None
    rest = float(np.median(v))
    m = float(np.percentile(np.abs(v - rest), p))
    return (rest, m) if m > 0 else None


def apply_signed_percent(arr, ref):
    """Signed % deviation from resting: 0 = resting, clipped to [-100, +100]."""
    arr = np.asarray(arr, dtype=float)
    if ref is None:
        return np.full_like(arr, np.nan)
    rest, m = ref
    return np.clip(100.0 * (arr - rest) / m, -100.0, 100.0)


def _trial_key(stem):
    """(category, number, odor) for ordering trials 1 (top) .. N (bottom)."""
    m = re.search(r"(testing|training)_(\d+)_([A-Za-z0-9\-]+)", stem)
    if m:
        return (m.group(1), int(m.group(2)), m.group(3))
    return ("zzz", 999, stem)


def _load_fly_rows(csv_path, fly):
    rows = [r for r in csv.DictReader(open(csv_path)) if r["fly"] == str(fly)]
    rows.sort(key=lambda r: int(r["frame"]))
    return rows


def make_percent_plots(out_root, fly_ids=(1, 2, 3),
                       max_time=PLOT_MAX_SECONDS,
                       odor_on=ODOR_ON_S, odor_off=ODOR_OFF_S):
    """One figure per fly. Each trial is a subplot (trial 1 top -> N bottom).
    Left axis: proboscis + abdomen distance as % (100% = max across this fly's
    trials). Right axis: abdomen angle in degrees, signed vs the fly's AVERAGE
    (resting) angle -> 0 = resting, + / - show which way the abdomen swings.
    First `max_time` s; black vertical lines mark odor on/off."""
    csvs = sorted(out_root.glob("*/*_tracking.csv"))
    trials = sorted(((_trial_key(c.stem), c) for c in csvs), key=lambda t: t[0])
    if not trials:
        print(f"no *_tracking.csv found under {out_root}")
        return
    print(f"{len(trials)} trial(s), flies {list(fly_ids)} -> percent plots")

    for fly in fly_ids:
        # Pass 1: load each trial's series (within the shown window) and pool
        # signals ACROSS ALL TRIALS so scales/reference are shared per fly.
        per_trial = []  # (label, t, {dist_key: raw}, angle_raw)
        pooled = {key: [] for _, key, _ in DIST_SIGNALS}
        pooled_angle = []
        for (cat, num, odor), csv_path in trials:
            label = f"{cat[:4]}_{num}\n{odor}"
            rows = _load_fly_rows(csv_path, fly)
            if not rows:
                per_trial.append((label, None, None, None))
                continue
            def col(key):
                return np.array([float(r[key]) if r[key] not in ("", "nan") else np.nan
                                 for r in rows])[mask]

            t = np.array([float(r["time_s"]) for r in rows])
            mask = t <= max_time
            t = t[mask]
            sig = {key: col(key) for _, key, _ in DIST_SIGNALS}
            angle = col(ANGLE_COL)
            # Same 350px rule as tracking: an abdomenTip more than ABD_ANCHOR_GATE
            # from its thorax is the wrong abdomen -> drop its distance AND angle.
            bad_abd = sig["abdomen_distance"] > ABD_ANCHOR_GATE
            sig["abdomen_distance"][bad_abd] = np.nan
            angle[bad_abd] = np.nan
            for _, key, _ in DIST_SIGNALS:
                pooled[key].append(sig[key])
            pooled_angle.append(angle)
            per_trial.append((label, t, sig, angle))

        # signed % reference per distance signal -> 0 = resting avg, +/-100% = max
        # deviation across trials (proboscis and abdomen distance both centered)
        scales = {key: signed_percent_ref(np.concatenate(vals)) if vals else None
                  for key, vals in pooled.items()}
        # resting angle = average (median) across all trials; symmetric right-axis limit
        all_angle = np.concatenate(pooled_angle) if pooled_angle else np.array([])
        all_angle = all_angle[~np.isnan(all_angle)]
        rest_angle = float(np.median(all_angle)) if all_angle.size else 0.0
        dev = np.abs(all_angle - rest_angle)
        alim = float(np.percentile(dev, 99)) if dev.size else 1.0
        alim = max(alim, 1.0)

        # Pass 2: plot
        n = len(trials)
        fig, axes = plt.subplots(n, 1, figsize=(11, 1.7 * n + 1.2), sharex=True)
        if n == 1:
            axes = [axes]
        angle_handle = None
        for ax, (label, t, sig, angle) in zip(axes, per_trial):
            ax2 = ax.twinx()  # right axis: signed abdomen angle (deg from rest)
            if t is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
            else:
                for lab, key, color in DIST_SIGNALS:
                    ax.plot(t, apply_signed_percent(sig[key], scales[key]),
                            color=color, lw=1.0, label=lab)
                ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
                h, = ax2.plot(t, angle - rest_angle, color=ANGLE_COLOR, lw=1.0,
                              label="abdomen angle (deg from rest)")
                angle_handle = h
                ax.axvline(odor_on, color="black", lw=1.6)
                ax.axvline(odor_off, color="black", lw=1.6)
            ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
            ax.set_ylim(-105, 105)
            ax.set_yticks([-100, -50, 0, 50, 100])
            ax.set_xlim(0, max_time)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2)
            ax2.set_ylim(-alim, alim)
            ax2.tick_params(labelsize=6, colors=ANGLE_COLOR)
        # combined legend (left-axis distance handles + right-axis angle handle)
        handles, labels = axes[0].get_legend_handles_labels()
        if angle_handle is not None:
            handles.append(angle_handle)
            labels.append("abdomen angle (deg from rest)")
        axes[0].legend(handles, labels, fontsize=7, loc="upper right", ncol=3)
        axes[-1].set_xlabel("time [s]  (black lines = odor on 30s / off 60s)")
        fig.suptitle(f"Fly {fly}  -  distance % from resting (left, 0=avg, +/-100)  "
                     f"&  abdomen angle deg from resting {rest_angle:.0f}deg "
                     f"(right, 0=avg)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.99))
        out_png = out_root / f"fly{fly}_all_trials_percent.png"
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        print(f"  wrote {out_png.name}  (resting angle = {rest_angle:.1f} deg, "
              f"+/-{alim:.0f} deg range)")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=DEFAULT_FOLDER)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--video", default=None, help="process only this video basename")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--percent-plots", action="store_true",
                    help="skip tracking; just build per-fly cross-trial %% plots "
                         "from existing *_tracking.csv files")
    args = ap.parse_args()

    folder = Path(args.folder)
    out_root = folder / OUT_SUBDIR
    out_root.mkdir(parents=True, exist_ok=True)

    # Plot-only mode: no model needed, just read the CSVs already produced.
    if args.percent_plots:
        make_percent_plots(out_root)
        return

    from ultralytics import YOLO
    model = YOLO(args.model)
    print("model:", args.model)
    print("classes:", model.names)

    videos = sorted(folder.glob("output_*.mp4"))
    if args.video:
        videos = [v for v in videos if args.video in v.name]
    if not videos:
        print("no videos found")
        return
    print(f"{len(videos)} video(s) to process -> {out_root}")

    for v in videos:
        try:
            process_video(model, v, out_root, max_frames=args.max_frames)
        except Exception as e:
            import traceback
            print(f"  ! error on {v.name}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
