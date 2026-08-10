#!/usr/bin/env python3
"""Composite presentation video: cropped annotated fly clip + live PER trace.

Top    : the YOLO-annotated recording, cropped tight to the fly so the empty
         arena space is gone.
Bottom : PER (``combined_pct``) against time for the clip window, drawn in as
         the clip plays, with the odor window shaded. Nothing else.

Colours follow the house style used by ``make_pubfig_october_02_fly_1_testing_2``
(black trace, ``#6cc070`` odor span). The playhead uses the PRGn purple that the
score palette already relies on; the green/purple pair is CVD-validated
(deutan dE 33.4, normal dE 40.7).

Usage
-----
    python scripts/analysis/make_per_trace_video.py            # defaults below
    python scripts/analysis/make_per_trace_video.py --start 25 --end 65
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Rectangle

# --------------------------------------------------------------------------- #
# Defaults for the october_02_fly_1 testing_2 (Hexanol) clip
# --------------------------------------------------------------------------- #
FLY_DIR = Path(
    "/securedstorage/DATAsec/cole/Data-secured/Hex-Training/october_02_fly_1"
)
DEFAULT_VIDEO = (
    FLY_DIR
    / "videos_with_rms/testing"
    / "october_02_fly_1_testing_2_distance_annotated_dup1.mp4"
)
DEFAULT_TRACE = (
    FLY_DIR
    / "angle_distance_rms_envelope"
    / "testing_2_fly1_angle_distance_rms_envelope.csv"
)
DEFAULT_STATE_CSV = (
    FLY_DIR
    / "RMS_calculations"
    / "updated_october_02_fly_1_testing_2_fly1_distances.csv"
)
DEFAULT_OUT = Path("/home/ramanlab/Documents/cole/Data/thesis_videos") / (
    "october_02_fly_1_testing_2_hexanol_per_25-65s.mp4"
)

VALUE_COL = "combined_pct"
ODOR_NAME = "Hexanol"

# Tight crop on the fly (x, y, w, h in source pixels). Derived from the
# max-intensity projection of the clip window: bright pixels span x 262-1079,
# y 96-621 in the 1080x1080 source, so everything below/left of that is arena.
CROP = (220, 80, 860, 574)

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
INK = "#111111"          # trace + primary text (single series, not categorical)
INK_MUTED = "#6b6b66"    # axis labels, captions
SURFACE = "#ffffff"
RULE = "#d8d8d3"
ODOR_FILL = "#6cc070"    # house Hexanol green
ODOR_EDGE = "#4a9d52"    # darker step, passes 3:1 for chip text/border
PLAYHEAD = "#7b3294"     # PRGn purple, CVD-validated against the green
GHOST_ALPHA = 0.14       # un-played remainder of the trace

CANVAS_W, CANVAS_H = 1920, 1080
DPI = 100


@dataclass
class Layout:
    """Pixel rectangles (x0, y0_from_top, w, h) turned into figure fractions."""

    @staticmethod
    def rect(x: float, y_top: float, w: float, h: float) -> list[float]:
        return [x / CANVAS_W, (CANVAS_H - y_top - h) / CANVAS_H, w / CANVAS_W, h / CANVAS_H]


def _use_arial() -> None:
    for name in (
        "Arial.ttf",
        "Arial_Bold.ttf",
        "arialbd.ttf",
    ):
        path = Path("/usr/share/fonts/truetype/msttcorefonts") / name
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = "Arial" if "Arial" in installed else "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.sans-serif": [family],
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "axes.edgecolor": INK_MUTED,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


def _odor_window(state_csv: Path) -> tuple[float, float]:
    """Odor on/off seconds from the OFM_State column."""
    df = pd.read_csv(state_csv, usecols=["timestamp", "OFM_State"])
    during = df.loc[df["OFM_State"].astype(str).str.lower() == "during", "timestamp"]
    if during.empty:
        raise RuntimeError(f"No 'during' rows in {state_csv}")
    return float(during.min()), float(during.max())


def _load_trace(trace_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(trace_csv)
    if VALUE_COL not in df.columns:
        raise RuntimeError(f"Column '{VALUE_COL}' missing from {trace_csv}")
    t = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(df[VALUE_COL], errors="coerce").to_numpy(dtype=float)
    return t, v


def build_figure(
    crop_shape: tuple[int, int],
    t_win: np.ndarray,
    v_win: np.ndarray,
    start: float,
    end: float,
    odor_on: float,
    odor_off: float,
):
    """Assemble the static canvas and return the artists that change per frame."""
    fig = plt.figure(figsize=(CANVAS_W / DPI, CANVAS_H / DPI), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)

    # -- video panel (centred, as large as the trace panel allows) --------- #
    crop_h, crop_w = crop_shape
    vid_w = 1120
    vid_h = int(round(vid_w * crop_h / crop_w))
    ax_vid = fig.add_axes(Layout.rect((CANVAS_W - vid_w) / 2, 14, vid_w, vid_h))
    ax_vid.set_xticks([])
    ax_vid.set_yticks([])
    for spine in ax_vid.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(RULE)
        spine.set_linewidth(1.0)
    im = ax_vid.imshow(np.zeros((crop_h, crop_w, 3), dtype=np.uint8))

    # -- trace panel ------------------------------------------------------- #
    ax = fig.add_axes(Layout.rect(112, 800, CANVAS_W - 112 - 50, 185))
    ax.set_facecolor(SURFACE)
    ax.set_xlim(start, end)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(np.arange(start, end + 0.01, 5))
    ax.tick_params(labelsize=17, length=4, width=0.9)
    ax.set_xlabel("Time (s)", fontsize=19, color=INK_MUTED, labelpad=6)
    ax.set_ylabel("PER (%)", fontsize=19, color=INK_MUTED, labelpad=8)
    ax.grid(axis="y", color=RULE, lw=0.8, alpha=0.7)
    ax.set_axisbelow(True)

    # odor window: shaded span + direct label (the label is the required relief
    # for the fill's sub-3:1 contrast)
    ax.add_patch(
        Rectangle(
            (odor_on, 0),
            odor_off - odor_on,
            100,
            facecolor=ODOR_FILL,
            alpha=0.22,
            edgecolor="none",
            zorder=0,
        )
    )
    for edge in (odor_on, odor_off):
        ax.axvline(edge, color=ODOR_EDGE, lw=1.4, ls=(0, (5, 4)), alpha=0.9, zorder=1)
    ax.text(
        odor_on + 0.5,
        95,
        f"{ODOR_NAME} on",
        ha="left",
        va="top",
        fontsize=17,
        color=ODOR_EDGE,
        weight="bold",
        zorder=4,
    )

    # ghost of the full window, then the revealed portion on top
    ax.plot(t_win, v_win, color=INK, lw=2.0, alpha=GHOST_ALPHA, zorder=2)
    (drawn,) = ax.plot([], [], color=INK, lw=2.6, solid_capstyle="round", zorder=5)
    head_line = ax.axvline(start, color=PLAYHEAD, lw=2.0, alpha=0.85, zorder=6)
    (head_dot,) = ax.plot(
        [],
        [],
        marker="o",
        markersize=11,
        markerfacecolor=PLAYHEAD,
        markeredgecolor=SURFACE,
        markeredgewidth=2.0,
        linestyle="none",
        zorder=7,
    )

    return fig, dict(im=im, drawn=drawn, head_line=head_line, head_dot=head_dot)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    ap.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--start", type=float, default=25.0)
    ap.add_argument("--end", type=float, default=65.0)
    ap.add_argument("--limit", type=int, default=0, help="render only N frames (preview)")
    args = ap.parse_args()

    _use_arial()

    t_all, v_all = _load_trace(args.trace)
    odor_on, odor_off = _odor_window(args.state)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    f0 = int(round(args.start * fps))
    f1 = int(round(args.end * fps))
    f1 = min(f1, n_src - 1, len(t_all) - 1)
    n_out = f1 - f0 + 1
    if args.limit:
        n_out = min(n_out, args.limit)

    t_win = t_all[f0 : f1 + 1]
    v_win = v_all[f0 : f1 + 1]

    x, y, w, h = CROP
    fig, art = build_figure(
        (h, w), t_win, v_win, args.start, args.end, odor_on, odor_off
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgba",
        "-s", f"{CANVAS_W}x{CANVAS_H}", "-r", f"{fps:.6f}",
        "-i", "pipe:0",
        "-an",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(args.out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    canvas = fig.canvas
    try:
        for i in range(n_out):
            ok, frame = cap.read()
            if not ok:
                break
            crop = frame[y : y + h, x : x + w]
            art["im"].set_data(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            t_now = float(t_win[i])
            v_now = float(v_win[i])
            art["drawn"].set_data(t_win[: i + 1], v_win[: i + 1])
            art["head_line"].set_xdata([t_now, t_now])
            art["head_dot"].set_data([t_now], [v_now])

            canvas.draw()
            proc.stdin.write(canvas.buffer_rgba())

            if i % 200 == 0:
                print(f"  frame {i}/{n_out}  t={t_now:.2f}s", flush=True)
    finally:
        cap.release()
        if proc.stdin:
            proc.stdin.close()
        proc.wait()
        plt.close(fig)

    print(f"[OK] Wrote {args.out}  ({n_out} frames @ {fps:.2f} fps)")


if __name__ == "__main__":
    main()
