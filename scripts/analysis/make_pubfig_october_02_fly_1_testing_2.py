#!/usr/bin/env python3
"""Standalone publication figure for october_02_fly_1 testing_2 with frame annotations.

This script does NOT modify the pipeline outputs. It creates a single composite
figure: top row shows 5 selected frames, bottom plot shows the combined-base
PER distance trace with frame-time markers.
"""

from __future__ import annotations

from pathlib import Path

import io
import json
import os
import shutil
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path("/home/ramanlab/Documents/cole")

FRAME_DIR = ROOT / "Data/imagesForFigures/imagesForOptoHEXTesting"
FRAME_FILES = [
    FRAME_DIR / "frame_0000800.png",
    FRAME_DIR / "frame_0001500.png",
    FRAME_DIR / "frame_0001800.png",
    FRAME_DIR / "frame_0002200.png",
]
FRAME_NUMS = [800, 1500, 1800, 2200]

DIST_CSV = (
    ROOT
    / "Data/flys/opto_hex/october_10_batch_1/RMS_calculations/"
    "updated_october_10_batch_1_testing_4_fly2_distances.csv"
)

MATRIX_PATH = ROOT / "Data/Opto/Combined/matrix/combined_base/envelope_matrix_float16.npy"
CODES_PATH = ROOT / "Data/Opto/Combined/matrix/combined_base/code_maps.json"

OUT_PATH = FRAME_DIR / "october_10_batch_1_testing_4_fly2_pubfig.png"
SVG_RASTER_WIDTH = 1080

def _resolve_frame_path(path: Path) -> Path:
    if path.exists():
        return path
    if path.suffix.lower() == ".png":
        alt = path.with_suffix(".svg")
        if alt.exists():
            return alt
    if path.suffix.lower() == ".svg":
        alt = path.with_suffix(".png")
        if alt.exists():
            return alt
    return path


def _load_image(path: Path) -> Image.Image:
    path = _resolve_frame_path(path)
    if path.suffix.lower() != ".svg":
        return Image.open(path)

    try:
        import cairosvg  # type: ignore
        png_bytes = cairosvg.svg2png(url=str(path), output_width=SVG_RASTER_WIDTH)
        return Image.open(io.BytesIO(png_bytes))
    except Exception:
        pass

    rsvg = shutil.which("rsvg-convert")
    if rsvg:
        png_bytes = subprocess.check_output(
            [rsvg, "-w", str(SVG_RASTER_WIDTH), str(path)]
        )
        return Image.open(io.BytesIO(png_bytes))

    inkscape = shutil.which("inkscape")
    if inkscape:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            subprocess.run(
                [
                    inkscape,
                    str(path),
                    "--export-type=png",
                    "--export-filename",
                    tmp_path,
                    "--export-width",
                    str(SVG_RASTER_WIDTH),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            img = Image.open(tmp_path)
            img.load()
            return img
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    raise RuntimeError(
        f"SVG support requires cairosvg, inkscape, or rsvg-convert: {path}"
    )


def _load_env_trace() -> tuple[np.ndarray, float, str]:
    with CODES_PATH.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    matrix = np.load(MATRIX_PATH, allow_pickle=False)
    df = pd.DataFrame(matrix, columns=meta["column_order"])

    code_maps = meta["code_maps"]
    rev_maps = {
        col: {int(code): label for label, code in mapping.items()}
        for col, mapping in code_maps.items()
    }
    for col in ["dataset", "fly", "fly_number", "trial_type", "trial_label"]:
        if col in df.columns:
            vals = np.rint(df[col].to_numpy(np.float32, copy=False)).astype(np.int32, copy=False)
            df[col] = pd.Series(vals).map(rev_maps.get(col, {})).fillna("UNKNOWN")

    if "fps" in df.columns:
        fps_codes = np.rint(df["fps"].to_numpy(np.float32, copy=False)).astype(np.int32, copy=False)
        if "fps" in rev_maps:
            fps_strings = pd.Series(fps_codes).map(rev_maps["fps"]).astype(str)
            df["fps"] = pd.to_numeric(fps_strings, errors="coerce")
        else:
            df["fps"] = pd.to_numeric(df["fps"], errors="coerce")

    row = df[
        (df["fly"] == "october_10_batch_1")
        & (df["trial_label"] == "testing_4")
        & (df["trial_type"] == "testing")
        & (df["fly_number"] == "2")
    ]
    if row.empty:
        raise RuntimeError("No combined-base row found for october_02_fly_1 testing_2.")

    row = row.iloc[0]
    fps = float(row.get("fps", 40.0)) if np.isfinite(row.get("fps", 40.0)) else 40.0
    dataset = str(row.get("dataset", "UNKNOWN"))

    env_cols = [c for c in meta["env_columns"] if str(c).startswith("dir_val_")]
    env = row[env_cols].to_numpy(float, copy=False)
    env = env[np.isfinite(env) & (env > 0)]

    return env, fps, dataset


def _load_frame_times() -> dict[int, float]:
    df = pd.read_csv(DIST_CSV)
    if "frame" not in df.columns or "timestamp" not in df.columns:
        raise RuntimeError("Distance CSV missing required columns: frame, timestamp.")
    lookup = df.set_index("frame")["timestamp"].to_dict()
    missing = [f for f in FRAME_NUMS if f not in lookup]
    if missing:
        raise RuntimeError(f"Frames not found in distance CSV: {missing}")
    return {f: float(lookup[f]) for f in FRAME_NUMS}


def main() -> None:
    env, fps, dataset = _load_env_trace()
    frame_times = _load_frame_times()

    arial_candidates = [
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/ariali.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arialbi.ttf",
    ]
    for font_path in arial_candidates:
        path = Path(font_path)
        if path.exists():
            font_manager.fontManager.addfont(str(path))

    odor_on_s = 32.15
    odor_off_s = 62.15
    after_show_s = 30.0
    x_max = odor_off_s + after_show_s

    env = env / 2.0
    t = np.arange(env.size, dtype=float) / max(fps, 1e-9)
    mask = t <= x_max + 1e-9
    t = t[mask]
    env = env[mask]

    colors = [
        "#d62728",
        "#ff7f0e",
        "#2ca02c",
        "#1f77b4",
        "#9467bd",
    ]

    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 14,
        }
    )

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.9, 1.0], hspace=0.06, wspace=0.03)

    image_axes = []
    for idx, (frame_path, frame_num, color) in enumerate(zip(FRAME_FILES, FRAME_NUMS, colors)):
        ax = fig.add_subplot(gs[0, idx])
        img = _load_image(frame_path)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3.0)
            spine.set_edgecolor(color)
        time_s = frame_times[frame_num]
        ax.set_title(f"{int(round(time_s))} s", fontsize=14, color=color, pad=6)
        image_axes.append((frame_num, ax))

    ax = fig.add_subplot(gs[1, :])
    ax.plot(t, env, linewidth=2.0, color="black")
    ax.axvspan(odor_on_s, odor_off_s, alpha=0.20, color="#9e9e9e")
    y_top = 100.0
    for frame_num, color in zip(FRAME_NUMS, colors):
        time_s = frame_times[frame_num]
        ax.vlines(time_s, 0, y_top, color=color, linewidth=1.8, alpha=0.95)

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_top)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Max Distance x Angle (%)", fontsize=12)

    ax.legend(
        handles=[Patch(facecolor="#9e9e9e", edgecolor="none", alpha=0.20, label="Odor")],
        loc="upper right",
        frameon=True,
        fontsize=14,
    )

    fig.suptitle(
        "Conditioned Fruit Fly — Hexanol Response After Training (Distance x Angle %)",
        fontsize=14,
        weight="bold",
        y=0.91,
    )

    if image_axes:
        plot_top = ax.get_position().y1
        for (frame_num, ax_img), color in zip(image_axes, colors):
            time_s = frame_times[frame_num]
            x_disp, y_disp = ax.transData.transform((time_s, y_top))
            x_fig, y_fig = fig.transFigure.inverted().transform((x_disp, y_disp))

            pos = ax_img.get_position()
            img_center_x = 0.5 * (pos.x0 + pos.x1)
            img_bottom_y = pos.y0

            fig.add_artist(
                plt.Line2D(
                    [x_fig, img_center_x],
                    [plot_top, img_bottom_y],
                    transform=fig.transFigure,
                    color=color,
                    linewidth=1.2,
                    alpha=0.95,
                )
            )

    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
