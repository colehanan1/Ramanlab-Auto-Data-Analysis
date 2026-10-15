"""Video detector worker for trial traces (own process; niced; one at a time via rigmon.gpu).

    python -m rigmon.traces.detect_video --model M --video V --out OUT.json --stride 2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from rigmon.mountcheck.detect import WorkerError, dets_from_arrays

SRC = Path(__file__).resolve().parents[2]


def run_video_worker(model, video, out_json, *, timeout: float, stride: int = 2, conf: float = 0.25,
                     python_argv: list | None = None) -> dict:
    argv = (python_argv or [sys.executable, "-m", "rigmon.traces.detect_video"]) + [
        "--model", str(model), "--video", str(video), "--out", str(out_json),
        "--stride", str(stride), "--conf", str(conf)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (str(SRC), os.environ.get("PYTHONPATH", "")) if p)}
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env,
                           preexec_fn=lambda: os.nice(10))
    except subprocess.TimeoutExpired:
        raise WorkerError(f"trace detector timed out after {timeout:.0f} s")
    if p.returncode != 0:
        raise WorkerError((p.stderr or p.stdout or f"exit {p.returncode}")[-600:])
    try:
        return json.loads(Path(out_json).read_text())
    except (OSError, ValueError) as e:
        raise WorkerError(f"trace detector wrote no usable output: {e}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="rigmon.traces.detect_video")
    ap.add_argument("--model", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args(argv)
    if not Path(a.model).exists():
        sys.exit(f"model not found: {a.model}")
    import cv2
    from ultralytics import YOLO

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f"cannot open video: {a.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    model = YOLO(a.model, task="obb")
    frames, i = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % max(1, a.stride) == 0:
            r = model.predict(img, conf=a.conf, verbose=False)[0]
            frames.append({"i": i, "dets": dets_from_arrays(
                r.names, r.obb.cls.tolist(), r.obb.xywhr.tolist(), r.obb.conf.tolist())})
        i += 1
    cap.release()
    Path(a.out).write_text(json.dumps({"fps": fps, "frame_count": i, "width": w, "height": h,
                                       "frames": frames}))


if __name__ == "__main__":
    main()
