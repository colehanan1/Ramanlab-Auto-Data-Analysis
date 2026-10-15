"""Detector worker. Runs in its own process so the GPU context (~4 GB with the
TensorRT engine) is freed when it exits and a crash can't take the hub down.

    python -m rigmon.mountcheck.detect --model M --frames DIR --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]


class WorkerError(RuntimeError):
    pass


def dets_from_arrays(names: dict, cls: list, xywhr: list, conf: list) -> list[dict]:
    return [{"cls": names[int(c)], "x": round(b[0], 1), "y": round(b[1], 1), "w": round(b[2], 1),
             "h": round(b[3], 1), "r": round(b[4], 4), "conf": round(cf, 3)}
            for c, b, cf in zip(cls, xywhr, conf)]


def run_worker(model, frames_dir, out_json, *, timeout: float, conf: float = 0.25,
               python_argv: list | None = None) -> dict:
    argv = (python_argv or [sys.executable, "-m", "rigmon.mountcheck.detect"]) + [
        "--model", str(model), "--frames", str(frames_dir), "--out", str(out_json), "--conf", str(conf)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (str(SRC), os.environ.get("PYTHONPATH", "")) if p)}
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        raise WorkerError(f"detector timed out after {timeout:.0f} s")
    if p.returncode != 0:
        raise WorkerError((p.stderr or p.stdout or f"exit {p.returncode}")[-600:])
    return json.loads(Path(out_json).read_text())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="rigmon.mountcheck.detect")
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args(argv)
    if not Path(a.model).exists():
        sys.exit(f"model not found: {a.model}")
    from ultralytics import YOLO  # heavy import only inside the worker

    from .frames import decode

    model = YOLO(a.model, task="obb")
    frames, size = [], (0, 0)
    for f in sorted(Path(a.frames).glob("*.jpg")):
        img = decode(f.read_bytes())
        size = (img.shape[1], img.shape[0])
        r = model.predict(img, conf=a.conf, verbose=False)[0]
        frames.append({"name": f.name, "dets": dets_from_arrays(
            r.names, r.obb.cls.tolist(), r.obb.xywhr.tolist(), r.obb.conf.tolist())})
    Path(a.out).write_text(json.dumps({"width": size[0], "height": size[1],
                                       "names": {int(k): v for k, v in model.names.items()},
                                       "frames": frames}))


if __name__ == "__main__":
    main()
