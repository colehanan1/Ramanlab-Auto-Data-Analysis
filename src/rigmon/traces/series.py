"""Pure maths for a trial trace: detections → per-fly eye–proboscis px over time,
and odor-on intervals from the trial's frame CSV."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from fbpipe.utils.mount_qc import Detection, Thresholds, assign_eyes, find_slots, pair_proboscis


@dataclass
class TraceSeries:
    t_s: list
    flies: dict
    n_slots: int


def build_series(frames: list, fps: float, *, gate_px: float, up_divisor: float = 4.0,
                 th: Thresholds = Thresholds()) -> TraceSeries:
    dets = [[Detection.from_dict(d) for d in f["dets"]] for f in frames]
    centers = find_slots(dets, th)
    flies = {k + 1: [] for k in range(len(centers))}
    t_s = []
    for f, fd in zip(frames, dets):
        t_s.append(float(f["i"]) / fps if fps else 0.0)
        eyes = assign_eyes(fd, centers, th)
        probs = pair_proboscis(fd, eyes, gate_px, up_divisor)
        for k in range(len(centers)):
            if k in eyes and k in probs:
                e, p = eyes[k], probs[k]
                flies[k + 1].append(math.hypot(p.x - e.x, p.y - e.y))
            else:
                flies[k + 1].append(float("nan"))
    return TraceSeries(t_s, flies, len(centers))


def odor_segments(csv_path) -> list:
    """[(on_s, off_s, label)] — same rule as remux_from_csv.parse_ofm_segments_from_frame_csv."""
    path = Path(csv_path)
    if not path.is_file():
        return []
    segments, first, last_t, on_t, cur = [], None, 0.0, None, None
    try:
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                s = (row.get("SensorNs") or "").strip()
                if not s:
                    continue
                try:
                    ns = int(s)
                except ValueError:
                    continue
                first = ns if first is None else first
                t = (ns - first) / 1e9
                last_t = t
                label = (row.get("ActiveOFM") or "off").strip() or "off"
                if label != "off" and on_t is None:
                    on_t, cur = t, label
                elif on_t is not None and label != cur:
                    segments.append((on_t, t, cur))
                    on_t, cur = (t, label) if label != "off" else (None, None)
    except (OSError, csv.Error):
        return []
    if on_t is not None:
        segments.append((on_t, last_t, cur))
    return segments
