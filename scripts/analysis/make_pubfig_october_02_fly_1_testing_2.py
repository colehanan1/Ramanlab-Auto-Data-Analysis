#!/usr/bin/env python3
"""Standalone publication figure with two testing traces and frame annotations.

Top half: testing_2 (Hexanol) frames + trace.
Bottom half: testing_1 (Apple Cider Vinegar) frames + trace (values x2).
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
from scipy.signal import hilbert
from PIL import Image


ROOT = Path("/home/ramanlab/Documents/cole")

FRAME_DIR_TOP = ROOT / "Data/imagesForFigures/imagesForOptoHEXTesting3"
FRAME_DIR_BOTTOM = ROOT / "Data/imagesForFigures/imagesForOptoHEXTesting4"
FRAME_FILES = [
    "frame_0000800.png",
    "frame_0001500.png",
    "frame_0001800.png",
    "frame_0002200.png",
]
FRAME_NUMS = [800, 1500, 1800, 2200]

DIST_CSV_TOP = (
    ROOT
    / "Data/flys/opto_hex/october_08_batch_2/RMS_calculations/"
    "updated_october_08_batch_2_testing_2_fly1_distances.csv"
)
DIST_CSV_BOTTOM = (
    ROOT
    / "Data/flys/opto_hex/october_08_batch_2/RMS_calculations/"
    "updated_october_08_batch_2_testing_1_fly1_distances.csv"
)

MATRIX_PATH = ROOT / "Data/Opto/Combined/matrix/combined_base/envelope_matrix_float16.npy"
CODES_PATH = ROOT / "Data/Opto/Combined/matrix/combined_base/code_maps.json"
if not MATRIX_PATH.exists():
    MATRIX_PATH = ROOT / "Data/csvs/single_matrix_opto/envelope_matrix_float16.npy"
if not CODES_PATH.exists():
    CODES_PATH = ROOT / "Data/csvs/single_matrix_opto/code_maps.json"

OUT_PATH = FRAME_DIR_TOP / "october_08_batch_2_testing_2_and_1_fly1_pubfig.png"
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


def _rolling_rms(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(values, dtype=float)
    return (
        series.pow(2)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .pow(0.5)
        .to_numpy()
    )


def _hilbert_envelope(values: np.ndarray, window: int) -> np.ndarray:
    env = np.abs(hilbert(np.nan_to_num(values, nan=0.0)))
    return (
        pd.Series(env, dtype=float)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def _angle_multiplier(angle_pct: np.ndarray) -> np.ndarray:
    pct = np.asarray(angle_pct, dtype=float)
    pct = np.clip(pct, -100.0, 100.0)
    return np.where(
        pct <= 0.0,
        1.0,
        1.0 + np.log1p((pct / 100.0) * (np.e - 1.0)),  # log scale 1.0 → 2.0
    )


def _env_from_distance_csv(csv_path: Path, fps_default: float = 40.0) -> tuple[np.ndarray, float, str]:
    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        time_s = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
        diffs = np.diff(time_s)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        fps = float(1.0 / np.median(diffs)) if diffs.size else fps_default
    else:
        fps = fps_default

    dist_col = None
    for candidate in ("distance_percentage_0_1", "distance_percentage", "distance_pct"):
        if candidate in df.columns:
            dist_col = candidate
            break

    # Read raw pixel distance for pixel-domain combining
    raw_dist_col = None
    for candidate in ("distance_0_1", "distance_proboscis", "distance"):
        if candidate in df.columns:
            raw_dist_col = candidate
            break

    angle_col = None
    for candidate in ("angle_centered_pct", "angle_centered_percentage", "angle_pct"):
        if candidate in df.columns:
            angle_col = candidate
            break
    if angle_col is None:
        raise RuntimeError(f"Angle column not found in {csv_path.name}")

    angle_pct = pd.to_numeric(df[angle_col], errors="coerce").to_numpy(dtype=float)
    multiplier = _angle_multiplier(angle_pct)

    if raw_dist_col is not None:
        # Pixel-domain combining: d_px * multiplier, then normalize with weighted bounds
        d_px = pd.to_numeric(df[raw_dist_col], errors="coerce").to_numpy(dtype=float)
        d_px_weighted = d_px * multiplier

        # Load per-fly normalization bounds from sidecar JSON written by pipeline
        norm_meta_path = csv_path.parent.parent / "angle_distance_rms_envelope" / "fly_norm_metadata.json"
        if norm_meta_path.exists():
            with norm_meta_path.open() as _f:
                meta = json.load(_f)
            # Determine fly number (this script targets fly1)
            fn_key = "1"
            weighted_gmin = float(
                meta.get("fly_weighted_min_by_number", {}).get(fn_key, 0.0)
            )
            effective_max = float(
                meta.get("fly_effective_max_by_number", {}).get(fn_key, 150.0)
            )
        else:
            weighted_gmin = 0.0
            effective_max = 150.0
            print(f"[WARN] No fly_norm_metadata.json at {norm_meta_path}; using fallback bounds.")

        if effective_max != weighted_gmin:
            combined = 100.0 * (d_px_weighted - weighted_gmin) / (effective_max - weighted_gmin)
        else:
            combined = np.where(np.isfinite(d_px_weighted), 0.0, np.nan)
    else:
        # Fallback: use distance_pct if raw pixel column not available
        if dist_col is None:
            raise RuntimeError(f"Neither raw distance nor distance_pct found in {csv_path.name}")
        dist_pct = (
            pd.to_numeric(df[dist_col], errors="coerce")
            .to_numpy(dtype=float)
        )
        combined = dist_pct * multiplier

    window_frames = max(int(round(0.25 * fps)), 1)
    rms = _rolling_rms(combined, window_frames)
    env = _hilbert_envelope(rms, window_frames)
    return env, fps, "distance_csv"


def _load_env_trace(trial_label: str, csv_path: Path) -> tuple[np.ndarray, float, str]:
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

    mask = (
        (df["fly"] == "october_08_batch_2")
        & (df["trial_label"] == trial_label)
        & (df["trial_type"] == "testing")
    )
    if "fly_number" in df.columns:
        mask &= df["fly_number"] == "1"

    row = df[mask]
    if row.empty:
        return _env_from_distance_csv(csv_path)

    row = row.iloc[0]
    fps = float(row.get("fps", 40.0)) if np.isfinite(row.get("fps", 40.0)) else 40.0
    dataset = str(row.get("dataset", "UNKNOWN"))

    env_cols = [c for c in meta["env_columns"] if str(c).startswith("dir_val_")]
    env = row[env_cols].to_numpy(float, copy=False)
    env = env[np.isfinite(env) & (env > 0)]

    return env, fps, dataset


def _load_frame_times(csv_path: Path) -> dict[int, float]:
    df = pd.read_csv(csv_path)
    if "frame" not in df.columns or "timestamp" not in df.columns:
        raise RuntimeError("Distance CSV missing required columns: frame, timestamp.")
    lookup = df.set_index("frame")["timestamp"].to_dict()
    missing = [f for f in FRAME_NUMS if f not in lookup]
    if missing:
        raise RuntimeError(f"Frames not found in distance CSV: {missing}")
    return {f: float(lookup[f]) for f in FRAME_NUMS}


def _connect_frames(fig: plt.Figure, image_axes: list[tuple[int, plt.Axes]], plot_ax: plt.Axes,
                    frame_times: dict[int, float], colors: list[str], y_max: float) -> None:
    plot_top = plot_ax.get_position().y1
    for (frame_num, ax_img), color in zip(image_axes, colors):
        time_s = frame_times[frame_num]
        x_disp, y_disp = plot_ax.transData.transform((time_s, y_max))
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


def main() -> None:
    env_top, fps_top, _ = _load_env_trace("testing_2", DIST_CSV_TOP)
    env_bottom, fps_bottom, _ = _load_env_trace("testing_1", DIST_CSV_BOTTOM)
    frame_times_top = _load_frame_times(DIST_CSV_TOP)
    frame_times_bottom = _load_frame_times(DIST_CSV_BOTTOM)

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

    scale_top = 1.0
    scale_bottom = 2.0

    env_top = env_top * scale_top
    t_top = np.arange(env_top.size, dtype=float) / max(fps_top, 1e-9)
    mask_top = t_top <= x_max + 1e-9
    t_top = t_top[mask_top]
    env_top = env_top[mask_top]

    env_bottom = env_bottom * scale_bottom
    t_bottom = np.arange(env_bottom.size, dtype=float) / max(fps_bottom, 1e-9)
    mask_bottom = t_bottom <= x_max + 1e-9
    t_bottom = t_bottom[mask_bottom]
    env_bottom = env_bottom[mask_bottom]

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
            "svg.fonttype": "none",
        }
    )

    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(
        5, 4,
        height_ratios=[1.6, 0.85, 0.08, 1.6, 0.85],
        hspace=0.16,
        wspace=0.03,
    )

    image_axes_top = []
    for idx, (frame_name, frame_num, color) in enumerate(zip(FRAME_FILES, FRAME_NUMS, colors)):
        ax = fig.add_subplot(gs[0, idx])
        img = _load_image(FRAME_DIR_TOP / frame_name)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3.0)
            spine.set_edgecolor(color)
        time_s = frame_times_top[frame_num]
        ax.set_title(f"{int(round(time_s))} s", fontsize=14, color=color, pad=6)
        image_axes_top.append((frame_num, ax))

    ax_top = fig.add_subplot(gs[1, :])
    ax_top.plot(t_top, env_top, linewidth=2.0, color="black")
    hex_odor_color = "#6cc070"
    acv_odor_color = "#ffc685"

    ax_top.axvspan(odor_on_s, odor_off_s, alpha=0.25, color=hex_odor_color)
    y_top = 100.0
    for frame_num, color in zip(FRAME_NUMS, colors):
        time_s = frame_times_top[frame_num]
        ax_top.vlines(time_s, 0, y_top, color=color, linewidth=1.8, alpha=0.95)

    ax_top.set_xlim(0, x_max)
    ax_top.set_ylim(0, y_top)
    ax_top.set_xlabel("Time (s)", fontsize=12)
    ax_top.set_ylabel("Max Distance x Angle (%)", fontsize=12)
    ax_top.set_title("")
    ax_top.legend(
        handles=[Patch(facecolor=hex_odor_color, edgecolor="none", alpha=0.25, label="Hexanol")],
        loc="upper right",
        frameon=True,
        fontsize=14,
    )

    image_axes_bottom = []
    for idx, (frame_name, frame_num, color) in enumerate(zip(FRAME_FILES, FRAME_NUMS, colors)):
        ax = fig.add_subplot(gs[3, idx])
        img = _load_image(FRAME_DIR_BOTTOM / frame_name)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3.0)
            spine.set_edgecolor(color)
        time_s = frame_times_bottom[frame_num]
        ax.set_title(f"{int(round(time_s))} s", fontsize=14, color=color, pad=6)
        image_axes_bottom.append((frame_num, ax))

    ax_bottom = fig.add_subplot(gs[4, :])
    ax_bottom.plot(t_bottom, env_bottom, linewidth=2.0, color="black")
    ax_bottom.axvspan(odor_on_s, odor_off_s, alpha=0.25, color=acv_odor_color)
    y_bottom = 100.0
    for frame_num, color in zip(FRAME_NUMS, colors):
        time_s = frame_times_bottom[frame_num]
        ax_bottom.vlines(time_s, 0, y_bottom, color=color, linewidth=1.8, alpha=0.95)

    ax_bottom.set_xlim(0, x_max)
    ax_bottom.set_ylim(0, y_bottom)
    ax_bottom.set_xlabel("Time (s)", fontsize=12)
    ax_bottom.set_ylabel("Max Distance x Angle (%)", fontsize=12)
    ax_bottom.set_title("")
    ax_bottom.legend(
        handles=[Patch(facecolor=acv_odor_color, edgecolor="none", alpha=0.25, label="Apple Cider Vinegar")],
        loc="upper right",
        frameon=True,
        fontsize=14,
    )

    fig.suptitle(
        "Conditioned Fruit Fly — Hexanol vs Apple Cider Vinegar",
        fontsize=14,
        weight="bold",
        y=0.98,
    )

    _connect_frames(fig, image_axes_top, ax_top, frame_times_top, colors, y_top)
    _connect_frames(fig, image_axes_bottom, ax_bottom, frame_times_bottom, colors, y_bottom)

    _save_all(fig, OUT_PATH)
    plt.close(fig)

    print(f"[OK] Wrote {OUT_PATH}")


def _save_all(fig: plt.Figure, out_path: Path) -> None:
    out_path = Path(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    svg_path = out_path.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
