"""Per-video tracking for the abdomen geometry pipeline.

Frames are decoded at native size, resized to the 1080x1080 model frame for
inference and for the geometry rules (the space the labels and
``FlyGeometryLimits`` were measured in), and every reported metric is computed
from the accepted parts' corners mapped back to native pixels
(``geometry.measure_in: native``; ``model`` reports the 1080x1080 space).

Fly slots are frozen from the warm-up frames like production and never move:
the frozen eye is the eye ``Part`` in EVERY frame, so ``flies[i]`` is slot ``i``
by construction and a dropped eye detection cannot hand a proboscis to a
neighbour. The proboscis velocity gate runs online so the annotated video shows
exactly what the table contains.
"""
from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

from ..config import Settings
from ..utils import fly_geometry as FG
from ..utils.fly_geometry import Fly, FlyGeometryLimits, Part
from ..utils.multi_fly import EyeAnchorManager
from ..utils.rig_anchor import resolve_anchor
from ..utils.video_writer import FFmpegFrameWriter
from . import columns as C
from . import metrics as M
from .config import GeometrySettings
from .detect import parts_from_result
from .trial_meta import TrialInfo, frame_alignment

MODEL_SIDE = 1080
NAN = float("nan")
# Per-slot colours (BGR) = scripts/analysis/light_trial_traces.FLY_COLORS + orange for a 5th slot.
SLOT_COLORS_BGR = {1: (0, 0, 255), 2: (255, 0, 0), 3: (0, 128, 0), 4: (173, 68, 142), 5: (0, 165, 255)}
PART_COLORS_BGR = {FG.EYE: (0, 255, 255), FG.PROBOSCIS: (255, 208, 0),
                   FG.ABDOMEN: (210, 77, 255), FG.ABDOMEN_TIP: (0, 252, 124)}
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PRO_REASON_PREFIXES = ("eye->proboscis", "proboscis size", "non-finite eye->proboscis")


def model_frame_size(width: int, height: int) -> Tuple[int, int]:
    """Production resizes any non-1080x1080 source to 1080x1080 before predict."""
    if (int(width), int(height)) != (MODEL_SIDE, MODEL_SIDE):
        return (MODEL_SIDE, MODEL_SIDE)
    return (int(width), int(height))


@dataclass
class VideoGeometry:
    width: int
    height: int
    model_w: int
    model_h: int
    measure_native: bool = True

    @property
    def sx(self) -> float:
        return self.width / self.model_w

    @property
    def sy(self) -> float:
        return self.height / self.model_h

    def to_native(self, pts) -> np.ndarray:
        return M.map_corners(pts, self.sx, self.sy)

    def to_report(self, pts) -> np.ndarray:
        """Corners in the space the tables report (native, or model when configured)."""
        return self.to_native(pts) if self.measure_native else np.asarray(pts, dtype=float).reshape(-1, 2).copy()

    def to_model_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if (w, h) != (self.model_w, self.model_h):
            return cv2.resize(frame, (self.model_w, self.model_h), interpolation=cv2.INTER_LINEAR)
        return frame.copy()


@dataclass
class FrozenFly:
    slot: int                 # 1-based, numbered top -> bottom
    eye_model: Part           # frozen eye in model space (best warm-up detection)
    eye_report: Part          # the same corners in reporting space
    scale: float              # eye_scale in model space: the unit of every rule
    presence: float           # fraction of warm-up frames in which this eye was seen
    conf: float


@dataclass
class TrackResult:
    tables: Dict[int, pd.DataFrame]
    meta: Dict[str, object]
    frozen: List[FrozenFly]


def _xyxy(pts) -> np.ndarray:
    p = np.asarray(pts, dtype=float)
    return np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()], dtype=np.float32)


def _score(p: Part) -> float:
    return float(p.score) if np.isfinite(p.score) else 0.0


def _feed(mgr: EyeAnchorManager, eyes: Sequence[Part]) -> None:
    if eyes:
        mgr.try_update_from_dets(np.stack([_xyxy(e.pts) for e in eyes]).astype(np.float32),
                                 np.array([_score(e) for e in eyes], dtype=np.float32))


def freeze_slots(warmup_parts: Sequence[Sequence[Part]], vg: VideoGeometry, limits: FlyGeometryLimits, *,
                 max_flies: int, presence_min: float, zero_iou_eps: float = 1e-8
                 ) -> Tuple[List[FrozenFly], Optional[np.ndarray], float]:
    """Freeze the fly slots plus the posterior-axis and chirality priors.

    Pass 1 counts eyes with a manager capped at ``max_flies`` (production's
    ``_scan_initial_fly_count``); pass 2 re-runs a manager capped at that count
    so it confirms and numbers the slots top -> bottom. Only eyes that
    ``fly_geometry`` accepts as a fly (a proboscis or an abdomen attached) are
    eligible, and an anchor seen in fewer than ``presence_min`` of the warm-up
    frames is dropped. The best-confidence detection of each anchor becomes its
    frozen eye, so ``eye_scale`` is constant for the whole video.
    """
    per_frame_eyes: List[List[Part]] = []
    axis_pairs, chir_triples = [], []
    for parts in warmup_parts:
        flies = FG.assemble(list(parts), vg.model_w, vg.model_h, limits)
        per_frame_eyes.append([f.eye for f in flies if f.eye_ok and f.is_fly and f.scale > 0])
        axis_pairs += [(f.eye.centroid, f.abdomen.centroid) for f in flies if f.abdomen is not None]
        chir_triples += [(f.eye.centroid, f.proboscis.centroid, f.abdomen.centroid)
                         for f in flies if f.abdomen is not None and f.proboscis is not None]
    axis = FG.estimate_posterior_axis(axis_pairs, limits=limits) if axis_pairs else None
    chirality = FG.estimate_chirality(chir_triples) if chir_triples else 0.0
    n_frames = len(per_frame_eyes)
    counter = EyeAnchorManager(max_eyes=max(1, int(max_flies)), zero_iou_eps=zero_iou_eps)
    for eyes in per_frame_eyes:
        _feed(counter, eyes)
    n = len(counter.anchor_ids)
    if n == 0:
        return [], axis, chirality
    mgr = EyeAnchorManager(max_eyes=n, zero_iou_eps=zero_iou_eps)
    for eyes in per_frame_eyes:
        _feed(mgr, eyes)
    kept = []
    for center in mgr.get_centers():
        c = np.asarray(center, dtype=float)
        matched: List[Part] = []
        for eyes in per_frame_eyes:
            near = [e for e in eyes if np.linalg.norm(e.centroid - c) <= 0.5 * FG.eye_scale(e)]
            if near:
                matched.append(min(near, key=lambda e: float(np.linalg.norm(e.centroid - c))))
        presence = len(matched) / max(1, n_frames)
        if not matched or presence < presence_min:
            continue
        best = max(matched, key=_score)
        kept.append((c, best, presence))
    kept.sort(key=lambda t: (t[0][1], t[0][0]))      # top -> bottom, x as tie-break
    frozen = [
        FrozenFly(slot=i + 1, eye_model=Part(FG.EYE, p.pts, p.score),
                  eye_report=Part(FG.EYE, vg.to_report(p.pts), p.score),
                  scale=FG.eye_scale(p), presence=pr, conf=_score(p))
        for i, (_, p, pr) in enumerate(kept)
    ]
    return frozen, axis, chirality


def assemble_frame(detected: Sequence[Part], frozen: Sequence[FrozenFly], vg: VideoGeometry,
                   limits: FlyGeometryLimits, axis: Optional[np.ndarray], chirality: float) -> List[Fly]:
    """Run the rules with the FROZEN eyes; ``flies[i]`` is ``frozen[i]``."""
    parts = [fz.eye_model for fz in frozen] + [p for p in detected if p.label != FG.EYE]
    return FG.assemble(parts, vg.model_w, vg.model_h, limits, prior_axis=axis,
                       prior_chirality=chirality, abdomen_only_fallback=True)


def _split_reasons(rejected: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    pro = next((r for r in rejected if r.startswith(_PRO_REASON_PREFIXES)), None)
    abd = next((r for r in rejected if not r.startswith(_PRO_REASON_PREFIXES)), None)
    return pro, abd


def fly_row(fly: Fly, fz: FrozenFly, vg: VideoGeometry, anchor_report: np.ndarray, n_eyes: int,
            chirality: float) -> dict:
    """One per-frame row for one slot, in reporting space. NaN = withheld, never a guess."""
    eye = FG.polygon_centroid(fz.eye_report.pts)
    row = {c: NAN for c in C.PER_FRAME_COLUMNS}
    row.update({C.EYE_X: float(eye[0]), C.EYE_Y: float(eye[1]), C.EYE_SCALE: FG.eye_scale(fz.eye_report),
                C.N_EYES: int(n_eyes), C.ODOR_ON: False})
    pro_reason, abd_reason = _split_reasons(fly.rejected)
    pc = ac = None
    if fly.proboscis is not None:
        pro = vg.to_report(fly.proboscis.pts)
        pc = FG.polygon_centroid(pro)
        row.update({C.PRO_X: float(pc[0]), C.PRO_Y: float(pc[1]), C.PRO_CONF: _score(fly.proboscis),
                    C.PRO_DIST: float(np.linalg.norm(pc - eye)),
                    C.PRO_ANGLE_ANCHOR: M.anchor_angle_deg(eye, pc, anchor_report), C.PRO_STATUS: C.PRO_OK})
    else:
        row[C.PRO_STATUS] = pro_reason or C.PRO_MISSING
    if fly.abdomen is not None:
        abd = vg.to_report(fly.abdomen.pts)
        ac = FG.polygon_centroid(abd)
        length, width = M.obb_length_width(abd)
        row.update({C.ABD_X: float(ac[0]), C.ABD_Y: float(ac[1]), C.ABD_CONF: _score(fly.abdomen),
                    C.ABD_LEN: length, C.ABD_WID: width, C.EYE_ABD_DIST: float(np.linalg.norm(ac - eye))})
    if fly.tip is not None and ac is not None:
        tip = vg.to_report(fly.tip.pts)
        tc = FG.polygon_centroid(tip)
        side = M.ventral_sign(eye, ac, pc) if pc is not None else -float(chirality)
        row.update({C.TIP_X: float(tc[0]), C.TIP_Y: float(tc[1]), C.TIP_CONF: _score(fly.tip),
                    C.ABD_TIP_DIST: float(np.linalg.norm(tc - ac)),
                    C.ABD_ANGLE: M.interior_angle(eye, ac, tc), C.ABD_BEND: M.signed_bend(eye, ac, tc, side)})
    if fly.abdomen_only:
        row[C.STATUS] = C.STATUS_ABDOMEN_ONLY
    elif fly.abdomen is not None and fly.tip is not None:
        row[C.STATUS] = C.STATUS_OK
    else:
        row[C.STATUS] = abd_reason or C.STATUS_NO_ABDOMEN
    return row


def finalize_slot_table(rows: Sequence[dict], timestamps: np.ndarray, odor_on: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(list(rows), columns=C.PER_FRAME_COLUMNS)
    n = len(df)
    df[C.TIMESTAMP] = np.asarray(timestamps, dtype=float)[:n]
    df[C.ODOR_ON] = np.asarray(odor_on, dtype=bool)[:n]
    df[C.FRAME] = df[C.FRAME].astype(int)
    df[C.N_EYES] = df[C.N_EYES].astype(int)
    return df[C.PER_FRAME_COLUMNS]


def _poly(frame, pts, color, thickness=2):
    cv2.polylines(frame, [np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)], True, color, thickness, cv2.LINE_AA)


def draw_frame(frame: np.ndarray, flies: Sequence[Fly], frozen: Sequence[FrozenFly], vg: VideoGeometry,
               rows: Sequence[dict]) -> np.ndarray:
    """Overlay boxes, the eye->proboscis line, the eye->abdomen->tip polyline and the
    status text on the NATIVE frame. What is drawn is exactly what the row says."""
    for fly, fz, row in zip(flies, frozen, rows):
        col = SLOT_COLORS_BGR.get(fz.slot, (200, 200, 200))
        eye_pts = vg.to_native(fz.eye_model.pts)
        ec = FG.polygon_centroid(eye_pts)
        _poly(frame, eye_pts, col, 2)
        cv2.putText(frame, f"fly{fz.slot}", (int(ec[0]) - 14, int(ec[1]) - int(0.8 * fz.scale * vg.sy)),
                    _FONT, 0.6, col, 2, cv2.LINE_AA)
        if fly.proboscis is not None and row[C.PRO_STATUS] == C.PRO_OK:
            pro = vg.to_native(fly.proboscis.pts)
            pc = FG.polygon_centroid(pro)
            _poly(frame, pro, PART_COLORS_BGR[FG.PROBOSCIS], 2)
            cv2.line(frame, (int(ec[0]), int(ec[1])), (int(pc[0]), int(pc[1])), col, 2, cv2.LINE_AA)
            cv2.putText(frame, f"PER {row[C.PRO_DIST]:.0f}px", (int(pc[0]) + 6, int(pc[1])), _FONT, 0.5, col, 1, cv2.LINE_AA)
        elif fly.proboscis is not None:
            _poly(frame, vg.to_native(fly.proboscis.pts), (60, 60, 255), 1)
        if fly.abdomen is not None:
            abd = vg.to_native(fly.abdomen.pts)
            ac = FG.polygon_centroid(abd)
            _poly(frame, abd, PART_COLORS_BGR[FG.ABDOMEN], 2)
            cv2.line(frame, (int(ec[0]), int(ec[1])), (int(ac[0]), int(ac[1])), col, 2, cv2.LINE_AA)
            if fly.tip is not None:
                tip = vg.to_native(fly.tip.pts)
                tc = FG.polygon_centroid(tip)
                _poly(frame, tip, PART_COLORS_BGR[FG.ABDOMEN_TIP], 2)
                cv2.line(frame, (int(ac[0]), int(ac[1])), (int(tc[0]), int(tc[1])), col, 2, cv2.LINE_AA)
                cv2.putText(frame, f"{row[C.ABD_ANGLE]:.0f} deg", (int(ac[0]) + 6, int(ac[1]) - 8), _FONT, 0.5, col, 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "box only (no tip)", (int(ac[0]) + 6, int(ac[1]) - 8), _FONT, 0.45, col, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, str(row[C.STATUS])[:48], (int(ec[0]) - 40, int(ec[1]) + int(1.2 * fz.scale * vg.sy)),
                        _FONT, 0.42, (60, 60, 255), 1, cv2.LINE_AA)
    return frame


def _open(video_path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    return cap


def track_video(video_path: Path, info: TrialInfo, *, predict_fn: Callable[[list, float], list],
                id_to_label: Mapping[int, str], settings: Settings, gcfg: GeometrySettings,
                max_frames: Optional[int] = None, annotated_path: Optional[Path] = None,
                batch_size: Optional[int] = None) -> TrackResult:
    """Track one trial video: warm-up -> frozen slots -> per-frame rows per slot."""
    t0 = time.time()
    video_path = Path(video_path)
    cap = _open(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or MODEL_SIDE
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or MODEL_SIDE
    fps_video = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    mw, mh = model_frame_size(width, height)
    vg = VideoGeometry(width, height, mw, mh, measure_native=(gcfg.measure_in == "native"))
    limits = gcfg.limits

    batch_size = max(1, int(batch_size if batch_size is not None else settings.inference_batch_size))
    warm: List[np.ndarray] = []
    while len(warm) < gcfg.warmup_frames:
        ok, fr = cap.read()
        if not ok or fr is None or fr.size == 0:
            break
        warm.append(vg.to_model_frame(fr))
    cap.release()
    # Chunked like the main loop: a dynamic TensorRT engine rejects any batch above its profile.
    warm_results: List[object] = []
    for i in range(0, len(warm), batch_size):
        warm_results.extend(predict_fn(warm[i:i + batch_size], settings.conf_thres))
    warm_parts = [parts_from_result(r, id_to_label) for r in warm_results]
    frozen, axis, chirality = freeze_slots(warm_parts, vg, limits, max_flies=settings.max_flies,
                                           presence_min=gcfg.warmup_presence_min,
                                           zero_iou_eps=settings.zero_iou_epsilon)

    fps = float(info.fps) if info.fps and info.fps > 0 else (fps_video if 1.0 <= fps_video <= 240.0 else settings.fps_default)
    AX, AY = resolve_anchor(video_path)
    anchor_report = np.array([AX * vg.sx, AY * vg.sy]) if vg.measure_native else np.array([AX, AY], dtype=float)
    meta: Dict[str, object] = {
        "pipeline": "fbpipe.geometry", "schema_version": 1,
        "trial": info.to_dict(),
        "frame": {"width": width, "height": height, "model_w": mw, "model_h": mh, "sx": vg.sx, "sy": vg.sy,
                  "measure_in": gcfg.measure_in},
        "fps": fps, "fps_video": fps_video, "n_frames_video": n_video, "n_frames_processed": 0,
        "warmup": {"frames_requested": gcfg.warmup_frames, "frames_used": len(warm),
                   "presence_min": gcfg.warmup_presence_min},
        "slots": [{"slot": fz.slot, "eye_x": float(FG.polygon_centroid(fz.eye_report.pts)[0]),
                   "eye_y": float(FG.polygon_centroid(fz.eye_report.pts)[1]), "eye_scale_px": fz.scale,
                   "presence": fz.presence, "conf": fz.conf} for fz in frozen],
        "posterior_axis_deg": None if axis is None else float(math.degrees(math.atan2(axis[1], axis[0])) % 360.0),
        "chirality": float(chirality),
        "rig_anchor_model_px": [float(AX), float(AY)],
        "limits": dataclasses.asdict(limits),
        "conf_thres": settings.conf_thres, "imgsz": gcfg.imgsz, "iou": gcfg.iou,
        "proboscis_max_jump_px": gcfg.proboscis_max_jump_px,
        "tracked_fraction": {}, "seconds": 0.0,
    }
    if not frozen:
        meta["seconds"] = round(time.time() - t0, 2)
        return TrackResult({}, meta, [])

    writer = None
    if annotated_path is not None:
        video_fps = fps if 1.0 <= fps <= 240.0 else settings.fps_default
        writer = FFmpegFrameWriter(annotated_path, video_fps, width, height, crf=30, preset="medium")
        if not writer.ok:
            writer = cv2.VideoWriter(str(annotated_path), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (width, height))

    gates = {fz.slot: M.OnlineVelocityGate(gcfg.proboscis_max_jump_px) for fz in frozen}
    rows: Dict[int, List[dict]] = {fz.slot: [] for fz in frozen}
    counts = {fz.slot: {"proboscis": 0, "abdomen": 0, "abdomen_tip": 0} for fz in frozen}
    cap = _open(video_path)
    idx = 0
    while True:
        batch: List[np.ndarray] = []
        while len(batch) < batch_size and (max_frames is None or idx + len(batch) < max_frames):
            ok, fr = cap.read()
            if not ok or fr is None or fr.size == 0:
                break
            batch.append(fr)
        if not batch:
            break
        results = predict_fn([vg.to_model_frame(f) for f in batch], settings.conf_thres)
        if len(results) != len(batch):
            raise RuntimeError("predict returned a different number of results than frames")
        for fr, res in zip(batch, results):
            parts = parts_from_result(res, id_to_label)
            n_eyes = sum(1 for p in parts if p.label == FG.EYE)
            flies = assemble_frame(parts, frozen, vg, limits, axis, chirality)
            frame_rows = []
            for fly, fz in zip(flies, frozen):
                row = fly_row(fly, fz, vg, anchor_report, n_eyes, chirality)
                row[C.FRAME] = idx
                if row[C.PRO_STATUS] == C.PRO_OK and gates[fz.slot].rejects(row[C.PRO_X], row[C.PRO_Y]):
                    for col in C.PROBOSCIS_VALUE_COLUMNS:
                        row[col] = NAN
                    row[C.PRO_STATUS] = C.PRO_VELOCITY
                cnt = counts[fz.slot]
                cnt["proboscis"] += int(row[C.PRO_STATUS] == C.PRO_OK)
                cnt["abdomen"] += int(fly.abdomen is not None)
                cnt["abdomen_tip"] += int(row[C.STATUS] == C.STATUS_OK)
                rows[fz.slot].append(row)
                frame_rows.append(row)
            if writer is not None:
                writer.write(draw_frame(fr, flies, frozen, vg, frame_rows))
            idx += 1
    cap.release()
    if writer is not None:
        writer.release()

    ts, odor_on = frame_alignment(Path(info.sidecar_csv) if info.sidecar_csv else None, idx, fps)
    tables = {slot: finalize_slot_table(r, ts, odor_on) for slot, r in rows.items()}
    meta["n_frames_processed"] = idx
    meta["tracked_fraction"] = {str(slot): {k: (v / idx if idx else 0.0) for k, v in c.items()}
                                for slot, c in counts.items()}
    meta["seconds"] = round(time.time() - t0, 2)
    return TrackResult(tables, meta, frozen)
