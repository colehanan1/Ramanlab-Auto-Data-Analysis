#!/usr/bin/env python3
"""Write brightened copies of a batch's trial videos for a YOLO re-run.

One fixed lookup table for the whole batch (not per frame), so brightness never
flickers between frames or trials: grey -> stretch the batch's 1st..99.5th
percentile to 0..255 -> gamma 0.6. Frame count, frame rate and file names are
kept, so the frame CSV / sidecar files next to the copies still line up.

Never writes next to the source: --out must be a different directory.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

FFMPEG = "/usr/bin/ffmpeg"
FFPROBE = "/usr/bin/ffprobe"


def batch_lut(videos, samples_per_video=6, gamma=0.6):
    vals = []
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        for idx in np.linspace(5, max(total - 5, 6), samples_per_video).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                vals.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[::4, ::4].ravel())
        cap.release()
    pix = np.concatenate(vals)
    lo, hi = float(np.percentile(pix, 1)), float(np.percentile(pix, 99.5))
    x = np.clip((np.arange(256, dtype=np.float32) - lo) / max(hi - lo, 1.0), 0, 1)
    return (255 * x**gamma).astype(np.uint8), {"p1": lo, "p99_5": hi, "gamma": gamma}


def brighten(src: Path, dst: Path, lut: np.ndarray) -> int:
    rate = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=avg_frame_rate", "-of", "csv=p=0", str(src)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    cap = cv2.VideoCapture(str(src))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    proc = subprocess.Popen(
        [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}",
         "-framerate", rate, "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "12",
         "-pix_fmt", "yuv420p", str(dst)],
        stdin=subprocess.PIPE,
    )
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        proc.stdin.write(cv2.LUT(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), lut).tobytes())
        n += 1
    cap.release()
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on {src}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.resolve() == args.batch_dir.resolve():
        raise SystemExit("--out must differ from the source batch directory")
    args.out.mkdir(parents=True, exist_ok=True)
    videos = sorted(args.batch_dir.glob("output_*.mp4"))
    lut, params = batch_lut(videos)
    print(f"[brighten] {len(videos)} videos, LUT {params}", flush=True)
    report = {"source": str(args.batch_dir), "lut": params, "videos": {}}
    for src in videos:
        cap = cv2.VideoCapture(str(src))
        want = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        n = brighten(src, args.out / src.name, lut)
        cap = cv2.VideoCapture(str(args.out / src.name))
        got = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        report["videos"][src.name] = {"frames_in": want, "frames_written": n, "frames_out": got}
        print(f"[brighten] {src.name}: {n} frames written, container says {got} (source {want})", flush=True)
    (args.out / "brighten_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
