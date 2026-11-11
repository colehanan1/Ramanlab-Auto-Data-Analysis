"""Alignment check: frames where the burned-in odor label box is lit vs the rig CSV's ActiveOFM rows.

remux_from_csv.py draws a white box at x=10..110, y=10..60 while ActiveOFM != off, timed from the
CSV's SensorNs. If frame i of a video lines up with CSV row i, the box's on/off frame numbers
match the CSV's ActiveOFM transitions.
"""
import csv
import subprocess
import sys
from pathlib import Path

FFMPEG = "/usr/bin/ffmpeg"
W, H = 80, 30  # inside the box (x 20..100, y 20..50)


def box_lit(video: Path) -> list[bool]:
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", str(video), "-map", "0:v:0",
                          "-vf", f"crop={W}:{H}:20:20,format=gray", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    n = len(raw) // (W * H)
    means = [sum(raw[k * W * H:(k + 1) * W * H]) / (W * H) for k in range(n)]
    return [m > 200 for m in means]


def transitions(flags):
    return [i for i in range(1, len(flags)) if flags[i] != flags[i - 1]]


def csv_flags(csv_path: Path):
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if (r.get("SensorNs") or "").strip()]
    return [(r.get("ActiveOFM") or "off").strip().lower() != "off" for r in rows]


if __name__ == "__main__":
    video, rig_csv = Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None
    lit = box_lit(video)
    print(f"{video.name[:70]}: {len(lit)} frames, box on/off at frames {transitions(lit)}")
    if rig_csv:
        fl = csv_flags(rig_csv)
        print(f"  CSV {rig_csv.name[:60]}: {len(fl)} rows, ActiveOFM on/off at rows {transitions(fl)}")
