#!/usr/bin/env python3
"""Proboscis-extension (PER) distance from hand-drawn polygon labels.

The lab's normal PER pipeline consumes YOLO *box* outputs. For this trial the PER
video was instead hand-labeled in **polygon** format (X-AnyLabeling JSON), one label
set every 10th frame. This script turns those polygons into the same kind of trace the
YOLO pipeline produces — an eye/proboscis center pair plus their Euclidean distance —
and plots proboscis (``Labellum``) distance from the ``Eye`` over *real* time, with the
odor window shaded.

Two data quirks this script corrects (see the plan doc / trial.json + timestamps.csv):

1. **The PER video FPS is wrong.** ``PER.mp4`` is muxed at a 20 fps timebase (so it
   *plays* for ~74 s), but the trial actually ran ~34.7 s (TRIAL_START -> TRIAL_COMPLETE
   in ``timestamps.csv``; corroborated by the AL camera at ~10 fps for 340 frames). The
   Grasshopper3 free-ran faster than its 20 fps target while the H264 sink stamped 20
   fps. There is no per-frame PER timestamp file, so we reconstruct the true rate as
   ``n_per_frames / trial_duration_s`` (~42.6 fps) and anchor PER frame 0 to TRIAL_START.

2. **Odor timing** comes from the ``PHASE_START``/``ODOR_CMD`` rows in ``timestamps.csv``
   (wall-clock relative to TRIAL_START), falling back to the trial.json phase schedule.

PER <-> AL sync
---------------
Both cameras are launched together at TRIAL_START, and the AL camera writes reliable
per-frame wall-clock timestamps (``timestamps.csv`` FRAME_CAPTURED rows; matching
``ElapsedTime-ms`` in ``AL/AL_MMStack_metadata.txt``). A PER frame maps to real time via
``per_time(i) = i / true_per_fps`` and then to the nearest AL frame by timestamp. The AL
stack page index equals the AL frame number, so a PER frame resolves directly to an
``AL_MMStack.ome.tif`` page. This script emits that mapping as ``PER_AL_frame_map.csv``.

Two label sources, identical downstream:
    * hand-drawn polygons in ``PER/`` (default), one set per 10 frames; or
    * ``--yolo`` runs the trained OBB detector (classes 0=eye, 1=proboscis) over every
      frame for a dense trace.

``--trim-frames N`` (default 20) drops the leading frames recorded before the AL
acquisition started and re-zeros the plot time axis at frame N (t=0 == AL start).

Outputs (<src> = "polygon" or "yolo"):
    <trial>/PER_distance_<src>.png/.svg - distance % + angle panels
    <trial>/PER_distance_<src>.csv      - per-frame centers, distance/angle, combined %
    <trial>/PER_AL_frame_map.csv        - every PER frame -> wall-clock -> nearest AL frame
    <experiment>/analysis/<tag>_combined_distance_angle_<src>.png/.svg
                                        - combined Distance x Angle metric (actual values)

Usage:
    python scripts/analysis/per_polygon_distance.py [TRIAL_DIR]            # hand polygons
    python scripts/analysis/per_polygon_distance.py [TRIAL_DIR] --yolo     # YOLO OBB model
    python scripts/analysis/per_polygon_distance.py [TRIAL_DIR] --yolo --model /path/best.pt
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# Make ``fbpipe`` and the ``scripts.analysis`` package importable when run as a plain
# script (mirrors envelope_visuals.py). _ROOT lets us reuse the lab's combine helpers.
_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fbpipe.plot_style import apply_lab_style  # noqa: E402
from fbpipe.figure_export import install_svg_sidecar  # noqa: E402

# Reuse the lab's exact combine math (angle->multiplier, RMS, Hilbert envelope) so the
# combined Distance x Angle metric matches the Raw-Testing-PER-Traces figures.
from scripts.analysis.envelope_combined import (  # noqa: E402
    _angle_multiplier,
    _rolling_rms,
    _hilbert_envelope,
    WEIGHTED_EFFECTIVE_MAX_FLOOR,
)

DEFAULT_TRIAL_DIR = Path(
    "/home/ramanlab/Documents/cole/Data/Imaging/BEHAV_ALrecords/"
    "26-05-27-badsurgery-newodors/"
    "ALLX2_Gh146xChrim-Gr5anoRet_F2d_20260528_153222PERDuringodormore/trial_002_OFM_H"
)

# Combined Distance x Angle figure goes in the experiment's analysis/ folder by default
# (``<experiment>/analysis/``, i.e. the trial dir's parent). Override with --combined-fig-dir.

EYE_LABEL = "Eye"
PROBOSCIS_LABEL = "Labellum"
LABEL_STRIDE = 10  # one label set per 10 video frames
NOMINAL_PER_FPS = 20.0

# Newly trained OBB detector for the PER microscope video. classes: 0=eye, 1=proboscis.
DEFAULT_YOLO_MODEL = Path(
    "/home/ramanlab/Documents/cole/model/PER-Micro-Scope/runs/obb/train/weights/best.pt"
)
EYE_CLASS_ID = 0
PROBOSCIS_CLASS_ID = 1
# First frames precede the AL acquisition start; drop them and re-zero the time axis.
DEFAULT_TRIM_FRAMES = 20


# --------------------------------------------------------------------------- geometry
def polygon_centroid(points: list) -> tuple[float, float]:
    """Area-weighted centroid of a polygon via the shoelace formula.

    Falls back to the mean of the vertices when the polygon is degenerate
    (near-zero signed area), which keeps thin/self-touching hand traces stable.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())
    x = pts[:, 0]
    y = pts[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-9:
        return float(x.mean()), float(y.mean())
    cx = ((x + x_next) * cross).sum() / (6.0 * area)
    cy = ((y + y_next) * cross).sum() / (6.0 * area)
    return float(cx), float(cy)


def combined_distance_angle(distance_px: np.ndarray, theta_deg: np.ndarray,
                            window: int) -> dict:
    """Combine distance + angle exactly like the lab's Raw-Testing-PER-Traces pipeline.

    Replicates ``envelope_combined.py``:
    1. Reference angle = angle at the **minimum-distance frame** (proboscis fully
       retracted -> 0% extension). ``angle_centered_pct`` is then 0% at that frame.
    2. ``angle_centered_pct = (theta - reference) / max|theta - reference| * 100``
       (range -100..+100).
    3. ``multiplier = _angle_multiplier(angle_centered_pct)``  (<=0% -> 1.0x, positive
       grows log to 2.0x at +100%).
    4. ``d_px_weighted = distance_px * multiplier``  (combine in the pixel domain).
    5. ``combined_pct = 100*(d_px_weighted - gmin)/(effective_max - gmin)`` anchored at
       the (weighted) minimum, with the lab's 150 px effective-max floor.
    6. rolling-RMS then Hilbert envelope (the published "Max Distance x Angle %" trace).
    """
    d = np.asarray(distance_px, dtype=float)
    theta = np.asarray(theta_deg, dtype=float)

    # (1) reference angle at the min-distance frame
    if np.isfinite(d).any():
        ref_idx = int(np.nanargmin(d))
        reference_angle = float(theta[ref_idx]) if np.isfinite(theta[ref_idx]) else 0.0
    else:
        ref_idx, reference_angle = -1, 0.0

    # (2) centered angle -> percentage (min-distance frame == 0%)
    centered_deg = theta - reference_angle
    fly_max = float(np.nanmax(np.abs(centered_deg))) if np.isfinite(centered_deg).any() else 0.0
    if fly_max > 0:
        angle_centered_pct = (centered_deg / fly_max) * 100.0
    else:
        angle_centered_pct = np.zeros_like(centered_deg)
    angle_centered_pct = np.clip(angle_centered_pct, -100.0, 100.0)

    # (3) multiplier and (4) pixel-domain weighting. np.where in _angle_multiplier
    # evaluates the log1p branch for negative pct too (then discards it) -> harmless
    # invalid-value warning; silence it.
    with np.errstate(invalid="ignore"):
        multiplier = _angle_multiplier(angle_centered_pct)
    d_px_weighted = d * multiplier

    # (5) anchor combined % at the weighted minimum (min-distance-ish frame -> 0%)
    finite_w = d_px_weighted[np.isfinite(d_px_weighted)]
    gmin = float(np.min(finite_w)) if finite_w.size else 0.0
    gmax = float(np.max(finite_w)) if finite_w.size else 0.0
    effective_max = max(gmax, WEIGHTED_EFFECTIVE_MAX_FLOOR)
    if effective_max != gmin:
        combined_pct = 100.0 * (d_px_weighted - gmin) / (effective_max - gmin)
    else:
        combined_pct = np.where(np.isfinite(d_px_weighted), 0.0, np.nan)

    # (6) rolling RMS + Hilbert envelope (smoothed published trace)
    combined_rms = _rolling_rms(combined_pct, window)
    envelope = _hilbert_envelope(combined_rms, window)

    return {
        "reference_angle": reference_angle,
        "ref_idx": ref_idx,
        "angle_centered_pct": angle_centered_pct,
        "multiplier": multiplier,
        "d_px_weighted": d_px_weighted,
        "combined_pct": combined_pct,
        "combined_rms": combined_rms,
        "envelope": envelope,
    }


# ---------------------------------------------------------------------------- clock
def _parse_iso(ts: str) -> _dt.datetime:
    return _dt.datetime.fromisoformat(ts)


def load_trial_clock(trial_dir: Path) -> dict:
    """Build the real-time clock + odor window + AL frame timestamps from a trial.

    Returns a dict with: ``trial_duration_s``, ``odor_on_s``, ``odor_off_s``,
    ``al_frames`` (sorted list of (al_frame:int, t_rel_s:float)), and ``odor_source``.
    Times are seconds relative to TRIAL_START.
    """
    ts_path = trial_dir / "timestamps.csv"
    rows = list(csv.DictReader(ts_path.open()))
    if not rows:
        raise SystemExit(f"empty timestamps.csv: {ts_path}")

    t0 = _parse_iso(rows[0]["timestamp"])

    def rel(ts: str) -> float:
        return (_parse_iso(ts) - t0).total_seconds()

    odor_on_s = odor_off_s = None
    trial_duration_s = None
    al_frames: list[tuple[int, float]] = []
    for r in rows:
        ev, phase = r["event"], (r["phase"] or "").upper()
        if ev == "FRAME_CAPTURED":
            try:
                al_frames.append((int(r["frame"]), rel(r["timestamp"])))
            except (ValueError, KeyError):
                pass
        elif ev == "PHASE_START" and phase == "ODOR":
            odor_on_s = rel(r["timestamp"])
        elif ev == "PHASE_START" and phase == "POST-ODOR":
            odor_off_s = rel(r["timestamp"])
        elif ev == "TRIAL_COMPLETE":
            trial_duration_s = rel(r["timestamp"])

    al_frames.sort()
    odor_source = "timestamps.csv PHASE_START rows"

    # Fallbacks from trial.json phase schedule (BASELINE / ODOR / POST durations).
    if odor_on_s is None or odor_off_s is None:
        sched = _load_phase_schedule(trial_dir)
        if sched is not None:
            baseline, odor, _post = sched
            odor_on_s = baseline if odor_on_s is None else odor_on_s
            odor_off_s = baseline + odor if odor_off_s is None else odor_off_s
            odor_source = "trial.json phase schedule (fallback)"

    if trial_duration_s is None:
        # last AL frame time, or sum of phase schedule
        if al_frames:
            trial_duration_s = al_frames[-1][1]
        else:
            sched = _load_phase_schedule(trial_dir)
            trial_duration_s = sum(sched) if sched else 0.0

    return {
        "trial_duration_s": float(trial_duration_s),
        "odor_on_s": None if odor_on_s is None else float(odor_on_s),
        "odor_off_s": None if odor_off_s is None else float(odor_off_s),
        "al_frames": al_frames,
        "odor_source": odor_source,
    }


def _load_phase_schedule(trial_dir: Path) -> tuple[float, float, float] | None:
    """Return (baseline_s, odor_s, post_s) from trial.json, if present."""
    tj = trial_dir / "trial.json"
    if not tj.exists():
        return None
    data = json.loads(tj.read_text())
    found: dict[str, float] = {}

    # Canonical structure: ``phases: [{name: BASELINE, seconds: 15}, ...]``.
    phases = data.get("phases") if isinstance(data, dict) else None
    if isinstance(phases, list):
        for ph in phases:
            if isinstance(ph, dict) and "name" in ph and "seconds" in ph:
                found[str(ph["name"]).upper().replace("POST-ODOR", "POST")] = float(ph["seconds"])

    # Fallback: BASELINE/ODOR/POST as plain numeric keys anywhere in the tree.
    if not ({"BASELINE", "ODOR"} <= set(found)):
        def walk(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    ku = str(k).upper()
                    if ku in ("BASELINE", "ODOR", "POST", "POST-ODOR") and isinstance(v, (int, float)):
                        found.setdefault(ku.replace("POST-ODOR", "POST"), float(v))
                    walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    walk(v)

        walk(data)

    if {"BASELINE", "ODOR"} <= set(found):
        return found["BASELINE"], found["ODOR"], found.get("POST", 0.0)
    return None


def per_frame_count(trial_dir: Path) -> int:
    """PER video frame count: trial.json cameras.PER.frames_written, else ffprobe."""
    tj = trial_dir / "trial.json"
    if tj.exists():
        data = json.loads(tj.read_text())
        cams = data.get("cameras", {}) if isinstance(data, dict) else {}
        per = cams.get("PER", {}) if isinstance(cams, dict) else {}
        n = per.get("frames_written")
        if isinstance(n, int) and n > 0:
            return n
    # Fallback: ffprobe nb_frames
    mp4 = trial_dir / "PER.mp4"
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(mp4),
        ],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(out)


def nearest_al_frame(t_s: float, al_frames: list[tuple[int, float]]) -> tuple[int, float, float]:
    """Nearest AL frame to a real time. Returns (al_frame, al_time_s, dt_ms)."""
    if not al_frames:
        return -1, float("nan"), float("nan")
    times = np.array([t for _f, t in al_frames])
    idx = int(np.abs(times - t_s).argmin())
    al_frame, al_t = al_frames[idx]
    return al_frame, al_t, (t_s - al_t) * 1000.0


# ---------------------------------------------------------------------------- labels
def load_polygon_distances(per_dir: Path) -> list[dict]:
    """Parse every label JSON; return per-frame eye/proboscis centers + distance px."""
    out: list[dict] = []
    for jf in sorted(per_dir.glob("*.json")):
        try:
            label_idx = int(jf.stem)
        except ValueError:
            continue
        frame = label_idx * LABEL_STRIDE
        data = json.loads(jf.read_text())
        eye_pts = prob_pts = None
        for shape in data.get("shapes", []):
            lbl = shape.get("label")
            if lbl == EYE_LABEL and eye_pts is None:
                eye_pts = shape.get("points")
            elif lbl == PROBOSCIS_LABEL and prob_pts is None:
                prob_pts = shape.get("points")
        row = {"frame": frame, "label_file": jf.name}
        if eye_pts and prob_pts:
            ex, ey = polygon_centroid(eye_pts)
            px, py = polygon_centroid(prob_pts)
            # Geometry mirrors geom_features.py: dx/dy from eye->proboscis, r =
            # Euclidean distance, direction as a (cos,sin) unit vector + angle.
            dx, dy = px - ex, py - ey
            dist = float(np.hypot(dx, dy))
            r_safe = dist + 1e-6
            row.update(
                eye_x=ex, eye_y=ey, prob_x=px, prob_y=py,
                dx=dx, dy=dy, distance_px=dist,
                cos_theta=dx / r_safe, sin_theta=dy / r_safe,
                theta_deg=float(np.degrees(np.arctan2(dy, dx))),
            )
        else:
            missing = []
            if not eye_pts:
                missing.append(EYE_LABEL)
            if not prob_pts:
                missing.append(PROBOSCIS_LABEL)
            row.update(
                eye_x=np.nan, eye_y=np.nan, prob_x=np.nan, prob_y=np.nan,
                dx=np.nan, dy=np.nan, distance_px=np.nan,
                cos_theta=np.nan, sin_theta=np.nan, theta_deg=np.nan,
            )
            print(f"  [warn] {jf.name}: missing {', '.join(missing)} polygon", file=sys.stderr)
        out.append(row)
    out.sort(key=lambda r: r["frame"])
    return out


def _best_center(det: dict, class_id: int) -> tuple[float, float] | None:
    """Highest-confidence detection's box center for *class_id* (or None)."""
    from fbpipe.utils.vision import xyxy_to_cxcywh

    info = det.get(class_id)
    if not info:
        return None
    boxes = info["boxes"]
    scores = info["scores"]
    if boxes is None or len(boxes) == 0:
        return None
    idx = int(np.argmax(scores)) if len(scores) else 0
    cx, cy, _w, _h = xyxy_to_cxcywh(boxes[idx])
    return float(cx), float(cy)


def load_yolo_distances(video_path: Path, model_path: Path, conf: float = 0.25,
                        trim: int = 0, save_trimmed: Path | None = None,
                        save_fps: float | None = None) -> list[dict]:
    """Run the OBB detector over every video frame; return rows in the polygon schema.

    Mirrors ``load_polygon_distances`` output (frame, eye/prob centers, dx/dy,
    distance_px, cos/sin/theta) so the whole downstream (%, angle multiplier,
    combined metric, plots) is identical whether labels come from hand polygons or
    the YOLO model. Centers are OBB box centers (same convention as yolo_infer.py).

    If ``save_trimmed`` is given, the frames from index ``trim`` onward are re-encoded
    to that path **with the YOLO detections drawn on them** (via ``result.plot()``), at
    ``save_fps`` if provided (else the true rate), in the same decode pass.
    """
    from ultralytics import YOLO
    from fbpipe.utils.yolo_results import collect_detections

    model = YOLO(str(model_path))
    rows: list[dict] = []
    classes = (EYE_CLASS_ID, PROBOSCIS_CLASS_ID)
    writer = None  # lazily created once we know the frame size
    for frame_idx, result in enumerate(
        model.predict(source=str(video_path), stream=True, conf=conf, verbose=False)
    ):
        if save_trimmed is not None and frame_idx >= trim:
            import cv2

            annotated = result.plot()  # BGR frame with OBB boxes + class labels drawn
            if writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(save_trimmed), fourcc,
                                         float(save_fps or 20.0), (w, h))
            writer.write(annotated)
        det = collect_detections(result, classes)
        eye = _best_center(det, EYE_CLASS_ID)
        prob = _best_center(det, PROBOSCIS_CLASS_ID)
        row = {"frame": frame_idx, "label_file": ""}
        if eye and prob:
            ex, ey = eye
            px, py = prob
            dx, dy = px - ex, py - ey
            dist = float(np.hypot(dx, dy))
            r_safe = dist + 1e-6
            row.update(
                eye_x=ex, eye_y=ey, prob_x=px, prob_y=py,
                dx=dx, dy=dy, distance_px=dist,
                cos_theta=dx / r_safe, sin_theta=dy / r_safe,
                theta_deg=float(np.degrees(np.arctan2(dy, dx))),
            )
        else:
            row.update(
                eye_x=np.nan, eye_y=np.nan, prob_x=np.nan, prob_y=np.nan,
                dx=np.nan, dy=np.nan, distance_px=np.nan,
                cos_theta=np.nan, sin_theta=np.nan, theta_deg=np.nan,
            )
        rows.append(row)
    if writer is not None:
        writer.release()
        print(f"Wrote {save_trimmed} (labeled, frames {trim}.., {save_fps:.2f} fps)")
    n_missing = sum(1 for r in rows if not np.isfinite(r["distance_px"]))
    if n_missing:
        print(f"  [warn] {n_missing}/{len(rows)} frames missing eye and/or proboscis "
              f"detection (conf>={conf})", file=sys.stderr)
    return rows


def dump_frames_to_png(video_path: Path, out_dir: Path) -> int:
    """Extract every frame of *video_path* to ``out_dir`` as zero-padded PNGs.

    Uses ffmpeg (fast, no Python decode loop). Returns the frame count written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path),
         "-start_number", "0", str(out_dir / "frame_%05d.png")],
        check=True,
    )
    return len(list(out_dir.glob("frame_*.png")))


# ------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trial_dir", nargs="?", default=str(DEFAULT_TRIAL_DIR),
                    help="Trial directory containing PER/, PER.mp4, timestamps.csv, trial.json")
    ap.add_argument("--combined-fig-dir", default=None,
                    help="Where to write the combined Distance x Angle figure "
                         "(default: <experiment>/analysis/)")
    ap.add_argument("--yolo", action="store_true",
                    help="Detect eye/proboscis with the YOLO OBB model instead of "
                         "reading the hand-drawn PER/ polygon labels (dense, every frame)")
    ap.add_argument("--model", default=str(DEFAULT_YOLO_MODEL),
                    help="YOLO OBB weights (used with --yolo)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="YOLO confidence threshold (used with --yolo)")
    ap.add_argument("--trim-frames", type=int, default=DEFAULT_TRIM_FRAMES,
                    help="Drop the first N PER frames (before AL recording started) "
                         "and re-zero the plot time axis at frame N")
    ap.add_argument("--per-duration-s", type=float, default=None,
                    help="Real PER video duration in seconds (sets true fps = "
                         "n_frames / duration). Default: TRIAL_START->TRIAL_COMPLETE "
                         "from timestamps.csv")
    ap.add_argument("--no-save-trimmed-video", action="store_true",
                    help="With --yolo, do NOT write the labeled trimmed video "
                         "(default: write <trial>/PER_trimmed_labeled.mp4)")
    ap.add_argument("--dump-frames", action="store_true",
                    help="Also extract every PER.mp4 frame as a PNG into <trial>/PER_frames/")
    ap.add_argument("--odor-on", type=float, default=None,
                    help="Override odor-ON time (s) on the plot axis (default: protocol "
                         "BASELINE duration, e.g. 15)")
    ap.add_argument("--odor-off", type=float, default=None,
                    help="Override odor-OFF time (s) on the plot axis (default: "
                         "BASELINE+ODOR, e.g. 19)")
    ap.add_argument("--title", default=None,
                    help="Figure title (default: '<odor_code> PER Tracking')")
    ap.add_argument("--odor-label", default=None,
                    help="Label for the shaded odor window (default: odor_code)")
    ap.add_argument("--odor-color", default="tab:purple",
                    help="Color of the shaded odor window")
    args = ap.parse_args()

    trial_dir = Path(args.trial_dir).expanduser().resolve()
    per_dir = trial_dir / "PER"
    if not args.yolo and not per_dir.is_dir():
        raise SystemExit(f"no PER/ label folder in {trial_dir}")
    combined_dir = (Path(args.combined_fig_dir).expanduser()
                    if args.combined_fig_dir else trial_dir.parent / "analysis")
    # Tag combines the experiment folder + trial folder so figures are identifiable.
    trial_tag = f"{trial_dir.parent.name}_{trial_dir.name}"

    # Title / odor label: explicit override > trial.json odor_code default.
    odor_code = trial_dir.name
    tj = trial_dir / "trial.json"
    if tj.exists():
        try:
            odor_code = json.loads(tj.read_text()).get("odor_code", odor_code)
        except Exception:  # noqa: BLE001
            pass
    fig_title = args.title or f"{odor_code} PER Tracking"
    odor_label = args.odor_label or str(odor_code)

    apply_lab_style()
    install_svg_sidecar()

    # --- clock + fps reconstruction -------------------------------------------------
    clock = load_trial_clock(trial_dir)
    n_per = per_frame_count(trial_dir)
    trial_dur = clock["trial_duration_s"]
    # Real PER duration: user override (e.g. 35 s, with a ~1 s pre-AL lead) or the
    # TRIAL_START->TRIAL_COMPLETE span. This sets the true frame rate.
    per_duration = args.per_duration_s if args.per_duration_s else trial_dur
    true_fps = n_per / per_duration if per_duration > 0 else float("nan")
    odor_on_s, odor_off_s = clock["odor_on_s"], clock["odor_off_s"]
    al_frames = clock["al_frames"]

    print(f"Trial dir         : {trial_dir}")
    print(f"PER frames        : {n_per}")
    print(f"Trial duration    : {trial_dur:.3f} s (TRIAL_START -> TRIAL_COMPLETE)")
    print(f"PER fps (nominal) : {NOMINAL_PER_FPS:.1f}  ->  plays for {n_per / NOMINAL_PER_FPS:.2f} s (WRONG)")
    print(f"PER fps (true)    : {true_fps:.3f}  (= {n_per} frames / {per_duration:.3f} s)")
    print(f"Odor window       : {odor_on_s:.3f} - {odor_off_s:.3f} s  [{clock['odor_source']}]")
    print(f"AL frames logged  : {len(al_frames)}")

    video_path = trial_dir / "PER.mp4"
    trim = max(0, int(args.trim_frames))

    # --- optional: dump every frame to PNG ------------------------------------------
    if args.dump_frames:
        frames_dir = trial_dir / "PER_frames"
        print(f"Dumping frames -> {frames_dir} ...")
        n_png = dump_frames_to_png(video_path, frames_dir)
        print(f"Wrote {n_png} PNG frames to {frames_dir}")

    # --- detections (YOLO OBB model OR hand-drawn polygons) -------------------------
    if args.yolo:
        model_path = Path(args.model).expanduser()
        print(f"Source            : YOLO OBB model {model_path}")
        save_trimmed = None if args.no_save_trimmed_video else trial_dir / "PER_trimmed_labeled.mp4"
        rows = load_yolo_distances(video_path, model_path, conf=args.conf,
                                   trim=trim, save_trimmed=save_trimmed, save_fps=true_fps)
        source_label = "yolo"
    else:
        print(f"Source            : hand-drawn polygons in {per_dir}")
        rows = load_polygon_distances(per_dir)
        source_label = "polygon"
    if not rows:
        raise SystemExit("no detections/labels produced")

    # Trim leading frames recorded before the AL acquisition started, then re-zero the
    # time axis at the first retained frame so t=0 == AL recording start.
    rows = [r for r in rows if r["frame"] >= trim]
    if not rows:
        raise SystemExit(f"--trim-frames {trim} removed all rows")
    t_offset = trim / true_fps
    # Odor window on the (re-zeroed) plot axis. Priority: explicit override > protocol
    # phase schedule (t=0 == recording start, so odor = [baseline, baseline+odor]) >
    # jittery wall-clock PHASE_START shifted by the trim offset.
    sched = _load_phase_schedule(trial_dir)
    if args.odor_on is not None and args.odor_off is not None:
        odor_on_plot, odor_off_plot = args.odor_on, args.odor_off
        odor_src_plot = "override"
    elif sched is not None:
        baseline_s, odor_dur_s, _post = sched
        odor_on_plot, odor_off_plot = baseline_s, baseline_s + odor_dur_s
        odor_src_plot = "protocol schedule"
    else:
        odor_on_plot = None if odor_on_s is None else odor_on_s - t_offset
        odor_off_plot = None if odor_off_s is None else odor_off_s - t_offset
        odor_src_plot = "wall-clock (shifted)"

    # Mirror geom_features.py r_pct_minmax: 100*(v - min)/((max - min) + 1e-6), clipped.
    def pct_minmax(values: np.ndarray) -> tuple[np.ndarray, float, float]:
        a = np.asarray(values, dtype=float)
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        return np.clip(100.0 * (a - lo) / ((hi - lo) + 1e-6), 0.0, 100.0), lo, hi

    d = np.array([r["distance_px"] for r in rows], dtype=float)
    theta = np.array([r["theta_deg"] for r in rows], dtype=float)
    dist_pct, d_min, d_max = pct_minmax(d)

    # Combine distance + angle the lab way: angle centered on the min-distance frame
    # (0%), turned into a 1.0-2.0x multiplier, applied in the pixel domain, then
    # normalized + RMS + Hilbert-enveloped. Window ~= 1 s of samples at the row rate.
    frame_steps = np.diff([r["frame"] for r in rows])
    step = float(np.median(frame_steps)) if frame_steps.size else float(LABEL_STRIDE)
    combine_window = max(3, int(round(true_fps / max(step, 1.0))))  # ~1 s of samples
    comb = combined_distance_angle(d, theta, combine_window)

    for i, r in enumerate(rows):
        r["time_s"] = r["frame"] / true_fps - t_offset
        r["distance_pct"] = float(dist_pct[i]) if np.isfinite(r["distance_px"]) else np.nan
        r["angle_centered_pct"] = float(comb["angle_centered_pct"][i])
        r["angle_multiplier"] = float(comb["multiplier"][i])
        r["d_px_weighted"] = float(comb["d_px_weighted"][i])
        r["combined_pct"] = float(comb["combined_pct"][i])
        r["combined_envelope"] = float(comb["envelope"][i])
        # AL mapping uses absolute PER time (from TRIAL_START), not the re-zeroed axis.
        alf, alt, dt_ms = nearest_al_frame(r["frame"] / true_fps, al_frames)
        r["nearest_al_frame"] = alf

    print(f"Rows ({source_label})       : {len(rows)} (trimmed first {trim} frames)")
    print(f"Distance px range : {d_min:.1f} - {d_max:.1f}")
    print(f"Ref angle (min-d) : {comb['reference_angle']:.2f} deg "
          f"(frame idx {comb['ref_idx']}, the 0% extension anchor)")
    print(f"Combined window   : {combine_window} samples  (~{combine_window * step / true_fps:.2f} s)")
    if odor_on_plot is not None:
        print(f"Odor (plot axis)  : {odor_on_plot:.3f} - {odor_off_plot:.3f} s  [{odor_src_plot}]")

    # --- write PER_distance.csv -----------------------------------------------------
    dist_csv = trial_dir / f"PER_distance_{source_label}.csv"
    cols = ["frame", "time_s", "label_file", "eye_x", "eye_y", "prob_x", "prob_y",
            "dx", "dy", "distance_px", "distance_pct",
            "cos_theta", "sin_theta", "theta_deg", "angle_centered_pct",
            "angle_multiplier", "d_px_weighted", "combined_pct", "combined_envelope",
            "nearest_al_frame"]
    with dist_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"Wrote {dist_csv}")

    # --- write PER_AL_frame_map.csv (every PER frame) -------------------------------
    labeled_frames = {r["frame"] for r in rows}
    map_csv = trial_dir / "PER_AL_frame_map.csv"
    with map_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["per_frame", "per_time_s", "al_frame", "al_time_s", "dt_ms", "is_labeled"])
        for i in range(n_per):
            t = i / true_fps
            alf, alt, dt_ms = nearest_al_frame(t, al_frames)
            w.writerow([i, f"{t:.4f}", alf, f"{alt:.4f}", f"{dt_ms:.2f}",
                        int(i in labeled_frames)])
    print(f"Wrote {map_csv}")

    # --- figure ---------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_arr = np.array([r["time_s"] for r in rows])
    x_min = float(t_arr.min())
    x_max = float(t_arr.max())
    odor_color = args.odor_color
    # Dense YOLO traces: drop per-point markers; sparse polygon traces: keep them.
    mstyle = dict(marker="o", markersize=2.5) if len(rows) <= 300 else dict(marker=None)

    def shade_odor(ax):
        if odor_on_plot is not None and odor_off_plot is not None:
            ax.axvspan(odor_on_plot, odor_off_plot, alpha=0.25, color=odor_color, label=odor_label)
            ax.axvline(odor_on_plot, linestyle="--", linewidth=0.9, color=odor_color)
            ax.axvline(odor_off_plot, linestyle="--", linewidth=0.9, color=odor_color)

    # Two stacked panels sharing the time axis: distance % (top), angle (bottom).
    fig, (ax_d, ax_a) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)

    shade_odor(ax_d)
    ax_d.plot(t_arr, dist_pct, color="black", linewidth=1.2, **mstyle)
    ax_d.set_ylabel("PER Distance %")
    ax_d.set_ylim(-2, 102)
    ax_d.legend(loc="upper right", frameon=False)

    shade_odor(ax_a)
    ax_a.plot(t_arr, theta, color="tab:blue", linewidth=1.2, **mstyle)
    ax_a.set_ylabel("Proboscis angle (deg)")
    ax_a.set_xlabel("Time (s)")
    ax_a.set_xlim(x_min, x_max)

    fig.suptitle(fig_title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    png = trial_dir / f"PER_distance_{source_label}.png"
    fig.savefig(png)  # SVG sidecar emitted automatically
    plt.close(fig)
    print(f"Wrote {png} (+ .svg)")

    # --- combined Distance x Angle figure (actual per-label values, no RMS) ----------
    combined = comb["combined_pct"]
    fig2, ax_c = plt.subplots(figsize=(9, 4))
    shade_odor(ax_c)
    ax_c.plot(t_arr, combined, color="black", linewidth=1.2, **mstyle)
    ax_c.set_ylabel("Max Distance × Angle %")
    ax_c.set_xlabel("Time (s)")
    ax_c.set_xlim(x_min, x_max)
    ax_c.set_ylim(bottom=0)
    ax_c.margins(x=0)
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(loc="upper right", frameon=False, fontsize=8)
    ax_c.set_title(fig_title, fontsize=12)

    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_png = combined_dir / f"{trial_tag}_combined_distance_angle_{source_label}.png"
    fig2.tight_layout()
    fig2.savefig(combined_png)  # SVG sidecar emitted automatically
    plt.close(fig2)
    print(f"Wrote {combined_png} (+ .svg)")

    print(
        "\nSync note: per_time = per_frame / true_fps (PER frame 0 = TRIAL_START); "
        "nearest AL frame == AL_MMStack.ome.tif page index. Use PER_AL_frame_map.csv "
        "to cross-reference any PER frame to an AL imaging frame."
    )


if __name__ == "__main__":
    main()
