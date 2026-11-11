#!/usr/bin/env python3
"""Video: production's fixed eye anchor vs the drift-following anchor.

Each clip is one fly in one trial, played at half speed:

* left -- the fly, zoomed 3x around its eye and proboscis. Blue ring = the
  fixed anchor production measures from; orange cross = the drift-following
  anchor; small white dot = the eye YOLO detected in that frame; the two
  lines are the eye->proboscis distance each one measures.
* right -- how far the detected eye sits from the fixed anchor (grey) and
  where the drift-following anchor is (orange), over the clip; and the fly's
  distance trace (% of its range) for the whole trial, production (blue) vs
  drift-following (orange), cursor at the current frame.

No fly numbers or folder names are drawn on the frames (montage text rule);
``clips.csv`` beside the video says which fly each clip is.

Reads the comparison trees and the raw videos on the secured mirror; writes
only under ``Results/Figures/LiveEye-Comparison/Drift-Anchor/video``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

COLE = Path("/home/ramanlab/Documents/cole")
DATA = COLE / "Data"
SECURED = Path("/securedstorage/DATAsec/cole/Data-secured-New")
ARMS = {"anchor": DATA / "flys_AnchorEye", "drift": DATA / "flys_DriftEye"}
OUT = COLE / "Results" / "Figures" / "LiveEye-Comparison" / "Drift-Anchor" / "video"
FFMPEG = "/usr/bin/ffmpeg"

BLUE = "#2a78d6"      # fixed anchor (production)
ORANGE = "#eb6834"    # drift-following anchor
GREY = "#8a8a86"
W, H = 1920, 1080
PANEL = 1080          # left video panel, square
CROP = 360            # source px around the fly -> 3x zoom
OUT_FPS = 20.0        # source is 40 fps: half speed


def _bgr(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


@dataclass
class Clip:
    dataset: str
    batch: str
    trial: str
    fly: int
    start_s: float
    end_s: float
    title: str
    note: str
    dataset_fly: str = ""


def _table(arm: str, c: Clip) -> pd.DataFrame:
    p = ARMS[arm] / c.dataset / c.batch / c.trial / f"{c.trial}_fly{c.fly}_distances.parquet"
    return pd.read_parquet(p).sort_values("frame").set_index("frame")


def _raw_video(c: Clip) -> Path:
    stem = c.trial  # output_<trial>_<odor>_<stamp>.mp4
    hits = sorted((SECURED / c.dataset / c.batch).glob(f"output_{stem}_*.mp4"))
    hits = [h for h in hits if "_preprocessed" not in h.name]
    if not hits:
        raise FileNotFoundError(f"no raw video for {c.dataset}/{c.batch}/{c.trial}")
    return hits[-1]


def _odor_window(c: Clip) -> tuple:
    try:
        meta = json.loads((ARMS["anchor"] / c.dataset / c.batch / c.trial / "_trial_meta.json").read_text())
        fps = float(meta.get("fps") or 40.0)
        return meta["odor_on_frame"] / fps, meta["odor_off_frame"] / fps
    except (OSError, KeyError, TypeError, ValueError):
        return 30.0, 60.0


# --------------------------------------------------------------------------
# Right-hand panel (static per clip) + cursor geometry
# --------------------------------------------------------------------------


def _panel(c: Clip, old: pd.DataFrame, new: pd.DataFrame, fps: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pw, ph = W - PANEL, H
    dpi = 100
    fig = plt.figure(figsize=(pw / dpi, ph / dpi), dpi=dpi, facecolor="white")
    plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
    title = textwrap.wrap(c.title, 44)
    note = textwrap.wrap(c.note, 72)
    fig.text(0.06, 0.965, "\n".join(title), fontsize=18, weight="bold", va="top", color="#1d1d1b",
             linespacing=1.25)
    fig.text(0.06, 0.965 - 0.034 * len(title) - 0.012, "\n".join(note), fontsize=12.5, va="top",
             color="#4a4a47", linespacing=1.3)

    t = new.index.to_numpy() / fps
    sel = (t >= c.start_s) & (t <= c.end_s)

    ax1 = fig.add_axes([0.13, 0.50, 0.82, 0.30])
    base_x, base_y = new["x_class0_anchor"].to_numpy(), new["y_class0_anchor"].to_numpy()
    live = np.hypot(new["eye_live_x"].to_numpy() - base_x, new["eye_live_y"].to_numpy() - base_y)
    step = np.hypot(new["eye_step_dx"].to_numpy(), new["eye_step_dy"].to_numpy())
    # dots as well as the line: a single detected frame between two missed
    # ones (a false match) has no neighbour to draw a line to
    ax1.plot(t[sel], live[sel], color=GREY, lw=1.0, marker=".", markersize=3,
             label="eye detected in each frame")
    ax1.plot(t[sel], np.zeros(sel.sum()), color=BLUE, lw=2.5, label="fixed anchor (production)")
    ax1.plot(t[sel], step[sel], color=ORANGE, lw=2.5, drawstyle="steps-post", label="drift-following anchor")
    top = max(8.0, float(np.nanmax(live[sel])) * 1.1 if np.isfinite(live[sel]).any() else 8.0)
    ax1.set_ylim(-0.5, min(top, 60.0))
    ax1.set_xlim(c.start_s, c.end_s)
    ax1.set_ylabel("distance from the\nfixed anchor (px)")
    ax1.set_xlabel("time in trial (s)")
    ax1.legend(frameon=False, loc="upper left", fontsize=11.5, ncol=1)

    ax2 = fig.add_axes([0.13, 0.08, 0.82, 0.30])
    on, off = _odor_window(c)
    ax2.axvspan(on, off, color="#efede7", zorder=0, lw=0)
    ax2.text((on + off) / 2, 1.0, "odor on", transform=ax2.get_xaxis_transform(), ha="center",
             va="bottom", fontsize=11, color="#6b6a66")
    to = old.index.to_numpy() / fps
    ax2.plot(to, old["distance_percentage"].to_numpy(dtype=float), color=BLUE, lw=1.3,
             label="production (fixed anchor)")
    ax2.plot(t, new["distance_percentage"].to_numpy(dtype=float), color=ORANGE, lw=1.3,
             label="drift-following anchor")
    ax2.axvspan(c.start_s, c.end_s, ymin=0, ymax=0.03, color="#1d1d1b", lw=0)
    ax2.set_xlim(0, float(max(to.max(), t.max())))
    ax2.set_ylabel("proboscis distance\n(% of fly's range)")
    ax2.set_xlabel("time in trial (s); black bar = this clip")
    ax2.legend(frameon=False, loc="upper left", fontsize=11.5)

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3][:, :, ::-1].copy()

    def to_px(ax, x):
        bb = ax.get_window_extent()
        x0, x1 = ax.get_xlim()
        return int(round(bb.x0 + (x - x0) / (x1 - x0) * bb.width))

    def yspan(ax):
        bb = ax.get_window_extent()
        return int(round(ph - bb.y1)), int(round(ph - bb.y0))

    geo = {"ax1": (lambda x: to_px(ax1, x), yspan(ax1)), "ax2": (lambda x: to_px(ax2, x), yspan(ax2))}
    plt.close(fig)
    return img, geo


def _card(text: str, sub: str) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="white")
    fig.text(0.5, 0.57, "\n".join(textwrap.wrap(text, 52)), ha="center", va="center", fontsize=30,
             weight="bold", color="#1d1d1b", linespacing=1.3)
    fig.text(0.5, 0.40, "\n".join(textwrap.wrap(sub, 95)), ha="center", va="center", fontsize=18,
             color="#4a4a47", linespacing=1.4)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3][:, :, ::-1].copy()
    plt.close(fig)
    return img


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------


def _marker_ring(img, x, y, color, r=13, th=3):
    cv2.circle(img, (int(round(x)), int(round(y))), r, color, th, cv2.LINE_AA)


def _marker_cross(img, x, y, color, r=16, th=4):
    x, y = int(round(x)), int(round(y))
    cv2.line(img, (x - r, y), (x + r, y), color, th, cv2.LINE_AA)
    cv2.line(img, (x, y - r), (x, y + r), color, th, cv2.LINE_AA)


def render_clip(c: Clip, writer, legend_img: np.ndarray) -> None:
    old, new = _table("anchor", c), _table("drift", c)
    video = _raw_video(c)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 40.0
    panel, geo = _panel(c, old, new, fps)
    f0, f1 = int(c.start_s * fps), int(c.end_s * fps)
    # crop centred on the median eye/proboscis midpoint over the clip
    seg = new.loc[f0:f1]
    ex, ey = np.nanmedian(seg["x_class0_anchor"]), np.nanmedian(seg["y_class0_anchor"])
    px = np.nanmedian(seg["x_class1"]) if seg["x_class1"].notna().any() else ex
    py = np.nanmedian(seg["y_class1"]) if seg["y_class1"].notna().any() else ey
    cx, cy = (ex + px) / 2, (ey + py) / 2
    x0 = int(np.clip(cx - CROP / 2, 0, 1080 - CROP))
    y0 = int(np.clip(cy - CROP / 2, 0, 1080 - CROP))
    k = PANEL / CROP
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    blue, orange, white = _bgr(BLUE), _bgr(ORANGE), (255, 255, 255)
    for f in range(f0, f1 + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (1080, 1080):
            frame = cv2.resize(frame, (1080, 1080), interpolation=cv2.INTER_LINEAR)
        view = cv2.resize(frame[y0 : y0 + CROP, x0 : x0 + CROP], (PANEL, PANEL), interpolation=cv2.INTER_CUBIC)
        canvas = np.empty((H, W, 3), np.uint8)
        canvas[:, :PANEL] = view
        canvas[:, PANEL:] = panel
        if f in new.index:
            r = new.loc[f]
            ro = old.loc[f] if f in old.index else r

            def P(x, y):
                return ((x - x0) * k, (y - y0) * k)

            ox, oy = P(r["x_class0_anchor"], r["y_class0_anchor"])
            nx, ny = P(r["x_class0"], r["y_class0"])
            if np.isfinite(r["x_class1"]) and np.isfinite(r["y_class1"]):
                px, py = P(r["x_class1"], r["y_class1"])
                cv2.line(canvas, (int(ox), int(oy)), (int(px), int(py)), blue, 3, cv2.LINE_AA)
                cv2.line(canvas, (int(nx), int(ny)), (int(px), int(py)), orange, 3, cv2.LINE_AA)
                cv2.circle(canvas, (int(px), int(py)), 7, white, -1, cv2.LINE_AA)
                cv2.circle(canvas, (int(px), int(py)), 7, (30, 30, 30), 2, cv2.LINE_AA)
            _marker_ring(canvas, ox, oy, blue)
            _marker_cross(canvas, nx, ny, orange)
            if np.isfinite(r["eye_live_x"]) and np.isfinite(r["eye_live_y"]):
                lx, ly = P(r["eye_live_x"], r["eye_live_y"])
                cv2.circle(canvas, (int(lx), int(ly)), 5, white, -1, cv2.LINE_AA)
                cv2.circle(canvas, (int(lx), int(ly)), 5, (30, 30, 30), 1, cv2.LINE_AA)
        canvas[H - legend_img.shape[0] :, :legend_img.shape[1]] = legend_img
        tsec = f / fps
        for key in ("ax1", "ax2"):
            to_px, (ya, yb) = geo[key]
            xp = PANEL + to_px(tsec)
            if PANEL <= xp < W:
                cv2.line(canvas, (xp, ya), (xp, yb), (29, 29, 29), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{tsec:5.1f} s  (half speed)", (PANEL - 400, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)
        writer.stdin.write(canvas.tobytes())
    cap.release()


def _legend() -> np.ndarray:
    img = np.full((132, 560, 3), 255, np.uint8)
    blue, orange = _bgr(BLUE), _bgr(ORANGE)
    _marker_ring(img, 34, 28, blue)
    _marker_cross(img, 34, 66, orange)
    cv2.circle(img, (34, 104), 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (34, 104), 5, (30, 30, 30), 1, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for y, text in ((36, "fixed anchor (production)"), (74, "drift-following anchor"),
                    (112, "eye detected in this frame")):
        cv2.putText(img, text, (66, y), font, 0.85, (29, 29, 29), 2, cv2.LINE_AA)
    return img


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clips_json", type=Path, help="list of Clip dicts")
    ap.add_argument("--out", type=Path, default=OUT / "eye_anchor_fixed_vs_drift_following.mp4")
    args = ap.parse_args(argv)
    clips = [Clip(**d) for d in json.loads(args.clips_json.read_text())]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if OUT not in args.out.resolve().parents:
        raise SystemExit(f"refusing to write {args.out}: outside {OUT}")
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", str(OUT_FPS), "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out)]
    writer = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    legend = _legend()
    intro = _card("Fixed eye anchor vs drift-following anchor",
                  "Blue = where production measures from (fixed for the whole trial).   "
                  "Orange = the new anchor: it moves only when the eye moves and stays.")
    for _ in range(int(OUT_FPS * 4)):
        writer.stdin.write(intro.tobytes())
    for i, c in enumerate(clips, 1):
        card = _card(f"{i}/{len(clips)}  {c.title}", c.note)
        for _ in range(int(OUT_FPS * 3)):
            writer.stdin.write(card.tobytes())
        render_clip(c, writer, legend)
    writer.stdin.close()
    writer.wait()
    pd.DataFrame([asdict(c) for c in clips]).to_csv(args.out.with_name("clips.csv"), index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
