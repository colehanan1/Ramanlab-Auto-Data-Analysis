from __future__ import annotations

import io
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from moviepy.editor import CompositeVideoClip, VideoClip, VideoFileClip

# Environment variable flags for debugging and fallback control
_DEBUG_COMPOSE = os.environ.get("COMPOSE_RMS_DEBUG", "").lower() in ("1", "true", "yes")
_USE_FFMPEG_FALLBACK = os.environ.get("COMPOSE_RMS_USE_FFMPEG", "").lower() in ("1", "true", "yes")
_DEBUG_PRINTED = False  # Track if version info was printed

from ..config import Settings, get_main_directories
from ..utils.columns import (
    EYE_CLASS,
    find_proboscis_distance_percentage_column,
    find_proboscis_xy_columns,
)
from ..utils.fly_files import iter_fly_distance_csvs

# Display defaults mirror the notebook example
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2,
        "axes.linewidth": 1.25,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "legend.fontsize": 11,
    }
)

VIDEO_INPUT_DIR = "videos_with_rms"  # where copied trial videos live
VIDEO_OUTPUT_SUBDIR = "with_line_plots"  # subfolder under VIDEO_INPUT_DIR
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mpg", ".mpeg", ".m4v"}
PANEL_HEIGHT_FRACTION = 0.24
YLIM = (-100, 100)
RMS_WINDOW_S = 1.0
THRESH_K = 4.0
ANCHOR_X, ANCHOR_Y = 1080.0, 540.0
PCT_COL_ROBUST = f"distance_class1_class{EYE_CLASS}_pct"
DIST_COL_ROBUST = f"distance_class1_class{EYE_CLASS}"
# Modification #2: Increased from 0.05 (5%) to 0.10 (10%) for more aggressive outlier filtering
TRIM_FRAC = 0.10
VIDEO_EXTS_CHECK = VIDEO_EXTS  # alias for clarity

MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
)


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
            print(f"    Deleted source video: {path.name}")
    except Exception as exc:
        print(f"    [warn] Could not delete {path}: {exc}")


def _maybe_rmdir_empty(dir_path: Path) -> None:
    try:
        if dir_path.exists() and not any(dir_path.iterdir()):
            dir_path.rmdir()
            print(f"    Removed empty folder: {dir_path}")
    except Exception as exc:
        print(f"    [warn] Could not remove folder {dir_path}: {exc}")


def _is_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS_CHECK


def timestamp_to_seconds(value) -> float:
    if pd.isna(value):
        return float("nan")
    try:
        return float(value)
    except Exception:
        text = str(value).strip()
        parts = text.split(":")
        try:
            if len(parts) == 4:
                hh, mm, ss, ms = parts
                return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0
            if len(parts) == 3:
                hh, mm, ss = parts
                return int(hh) * 3600 + int(mm) * 60 + float(ss)
            if len(parts) == 2:
                mm, ss = parts
                return int(mm) * 60 + float(ss)
            if len(parts) == 1:
                return float(parts[0])
        except Exception:
            return float("nan")
    return float("nan")


def _normalise_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lookup = {_normalise_column_name(col): col for col in df.columns}
    for cand in candidates:
        key = _normalise_column_name(cand)
        if key in lookup:
            return lookup[key]
    return None


def derive_fps(df: pd.DataFrame, default: float) -> float:
    fps_col = find_col(df, ["fps", "FPS", "frame_rate", "frameRate"])
    if fps_col is not None:
        fps_val = pd.to_numeric(df[fps_col], errors="coerce").median()
        if np.isfinite(fps_val) and fps_val > 0:
            return float(fps_val)
    return default


def ensure_time_series(
    df: pd.DataFrame, frame_col: Optional[str], ts_col: Optional[str], fps_default: float
) -> Tuple[pd.Series, Dict[str, float]]:
    if ts_col is not None and ts_col in df.columns:
        if ts_col in {"time_seconds", "relative_time", "time_s"}:
            series = pd.to_numeric(df[ts_col], errors="coerce")
        else:
            series = df[ts_col].apply(timestamp_to_seconds)
        if series.notna().sum() >= 2 and np.nanmax(np.diff(series.dropna().to_numpy())) > 0:
            return series, {"used": "timestamp"}
    if frame_col is not None and frame_col in df.columns:
        frames = pd.to_numeric(df[frame_col], errors="coerce")
        if frames.notna().sum() >= 2:
            fps = derive_fps(df, fps_default)
            f0 = int(np.nanmin(frames.values))
            return (frames - f0) / fps, {"used": "frame_fallback", "fps": fps}
    fps = derive_fps(df, fps_default)
    idx = pd.Series(np.arange(len(df), dtype=float) / fps, index=df.index)
    return idx, {"used": "index_fallback", "fps": fps}


def extract_trial_index(name: str, category: str) -> Optional[int]:
    match = re.search(rf"{category}_(\d+)", name.lower())
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return None
    fallback = re.findall(r"\d+", name)
    if fallback:
        try:
            return int(fallback[-1])
        except Exception:
            return None
    return None


def odor_window_from_ofm(df: pd.DataFrame, time_col: str) -> Optional[Tuple[float, float]]:
    col = find_col(df, ["OFM State", "OFM_State", "ofm_state", "ofm"])
    if col is None or time_col not in df.columns:
        return None
    try:
        mask = df[col].astype(str).str.lower().eq("during")
        if mask.any():
            idx = np.flatnonzero(mask.to_numpy())
            start = float(df[time_col].iloc[idx[0]])
            end = float(df[time_col].iloc[idx[-1]])
            return start, end
    except Exception:
        return None
    return None


def compute_angle_deg_at_point2(df: pd.DataFrame) -> pd.Series:
    x2_col = find_col(
        df,
        [
            f"x_class{EYE_CLASS}",
            f"x_class_{EYE_CLASS}",
            f"class{EYE_CLASS}_x",
            "x_class2",
            "x_class_2",
            "class2_x",
        ],
    )
    y2_col = find_col(
        df,
        [
            f"y_class{EYE_CLASS}",
            f"y_class_{EYE_CLASS}",
            f"class{EYE_CLASS}_y",
            "y_class2",
            "y_class_2",
            "class2_y",
        ],
    )
    x_prob, y_prob = find_proboscis_xy_columns(df)
    if not all((x2_col, y2_col, x_prob, y_prob)):
        raise ValueError("Missing columns for angle computation")
    p2x = pd.to_numeric(df[x2_col], errors="coerce").to_numpy()
    p2y = pd.to_numeric(df[y2_col], errors="coerce").to_numpy()
    p3x = pd.to_numeric(df[x_prob], errors="coerce").to_numpy()
    p3y = pd.to_numeric(df[y_prob], errors="coerce").to_numpy()
    ux, uy = (ANCHOR_X - p2x), (ANCHOR_Y - p2y)
    vx, vy = (p3x - p2x), (p3y - p2y)
    dot = ux * vx + uy * vy
    cross = ux * vy - uy * vx
    n1 = np.hypot(ux, uy)
    n2 = np.hypot(vx, vy)
    valid = (n1 > 0) & (n2 > 0) & np.isfinite(dot) & np.isfinite(cross)
    angles = np.full(len(p2x), np.nan)
    angles[valid] = np.degrees(np.arctan2(np.abs(cross[valid]), dot[valid]))
    return pd.Series(angles, index=df.index, name="angle_ARB_deg")


def compute_angle_multiplier_continuous(angle_deg: float) -> float:
    """
    Convert angle in degrees to continuous multiplier for distance percentage.

    Modification #3: Continuous exponential (piecewise linear) angle scaling
    replaces bin-based approach for smooth transitions.

    Mapping:
        -100° → 0.5× (proboscis retracted toward body)
        0° → 1.0× (neutral, baseline)
        +100° → 2.0× (proboscis extended)

    Uses piecewise linear interpolation for smooth transitions.

    Args:
        angle_deg: Angle in degrees (will be clamped to [-100, 100])

    Returns:
        Multiplier value in range [0.5, 2.0]

    Examples:
        >>> compute_angle_multiplier_continuous(-100)
        0.5
        >>> compute_angle_multiplier_continuous(-50)
        0.75
        >>> compute_angle_multiplier_continuous(0)
        1.0
        >>> compute_angle_multiplier_continuous(50)
        1.5
        >>> compute_angle_multiplier_continuous(100)
        2.0
    """
    # Clamp angle to valid range
    clamped = np.clip(angle_deg, -100.0, 100.0)

    if clamped < 0:
        # Linear interpolation: -100° → 0.5, 0° → 1.0
        # Formula: 0.5 + 0.5 * (1.0 + normalized)
        normalized = clamped / 100.0  # Range: -1.0 to 0.0
        multiplier = 0.5 + 0.5 * (1.0 + normalized)
    else:
        # Linear interpolation: 0° → 1.0, +100° → 2.0
        # Formula: 1.0 + normalized
        normalized = clamped / 100.0  # Range: 0.0 to 1.0
        multiplier = 1.0 + normalized

    return float(multiplier)


def compute_angle_multiplier_series(angle_series: pd.Series) -> pd.Series:
    """
    Vectorized version of compute_angle_multiplier_continuous for pandas Series.

    Args:
        angle_series: Series of angle values in degrees

    Returns:
        Series of multiplier values in range [0.5, 2.0]
    """
    angles = angle_series.to_numpy(dtype=float)
    clamped = np.clip(angles, -100.0, 100.0)

    multipliers = np.where(
        clamped < 0,
        0.5 + 0.5 * (1.0 + clamped / 100.0),  # Negative angles
        1.0 + clamped / 100.0  # Positive angles
    )

    return pd.Series(multipliers, index=angle_series.index, name="angle_multiplier")


def find_fly_reference_angle(csvs_raw: List[Path], trimmed_min: Optional[float] = None) -> float:
    """
    Find the fly's reference angle (0°) for baseline measurements.

    Modification #5: When trimmed_min is provided, finds the angle at frames
    nearest to trimmed_min (more stable than using global minimum which may be outliers).

    Args:
        csvs_raw: List of CSV file paths to analyze
        trimmed_min: Optional trimmed minimum distance value (from _ead_compute_trim_min_max)

    Returns:
        Reference angle in degrees, or NaN if not found
    """
    best: Optional[Tuple[int, float, float]] = None
    for path in csvs_raw:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        try:
            angle = compute_angle_deg_at_point2(df)
        except Exception:
            continue
        # Use flexible column finder instead of hardcoded names
        dist_col = find_proboscis_distance_percentage_column(df)
        if dist_col is None:
            # Fallback to find_col with common names
            dist_col = find_col(
                df,
                [
                    PCT_COL_ROBUST,
                    "distance_percentage_2_8",
                    "distance_percentage",
                    "distance_percent",
                    "distance_pct",
                    "distance_class1_class2_pct",
                ],
            )
        if dist_col is None:
            continue
        dist = pd.to_numeric(df[dist_col], errors="coerce").to_numpy()

        # Modification #5: Use trimmed_min for more stable reference angle
        if trimmed_min is not None:
            # Find frame nearest to trimmed_min (more stable than global min)
            distance_from_trim = np.abs(dist - trimmed_min)
            if not np.isfinite(distance_from_trim).any():
                continue
            idx = int(np.nanargmin(distance_from_trim))
            angle_here = float(angle.iloc[idx]) if np.isfinite(angle.iloc[idx]) else np.nan
            candidate = (0, float(distance_from_trim[idx]), angle_here)
        else:
            # Fallback to old logic if trimmed_min not provided
            exact = np.flatnonzero(dist == 0)
            if exact.size > 0:
                idx = int(exact[0])
                angle_here = float(angle.iloc[idx]) if np.isfinite(angle.iloc[idx]) else np.nan
                candidate = (0, 0.0, angle_here)
            else:
                with np.errstate(invalid="ignore"):
                    absd = np.abs(dist)
                if not np.isfinite(absd).any():
                    continue
                idx = int(np.nanargmin(absd))
                angle_here = float(angle.iloc[idx]) if np.isfinite(angle.iloc[idx]) else np.nan
                candidate = (1, float(absd[idx]), angle_here)

        if best is None or candidate < best:
            best = candidate
    return best[2] if best is not None else float("nan")


def compute_fly_max_abs_centered(csvs_raw: List[Path], ref_angle: float) -> float:
    fly_max = 0.0
    for path in csvs_raw:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        try:
            angle = compute_angle_deg_at_point2(df)
        except Exception:
            continue
        centered = angle - ref_angle if np.isfinite(ref_angle) else angle * 0.0
        local = np.nanmax(np.abs(centered.to_numpy(dtype=float)))
        if np.isfinite(local):
            fly_max = max(fly_max, float(local))
    return fly_max if np.isfinite(fly_max) and fly_max > 0 else float("nan")


def _infer_category_from_path(path: Path) -> Optional[str]:
    tokens = " ".join([*path.parts, path.stem]).lower()
    if any(tok in tokens for tok in ("training", "train", "trn")):
        return "training"
    if any(tok in tokens for tok in ("testing", "test", "tst")):
        return "testing"
    return None


def _collect_trial_csvs(
    fly_dir: Path, category: str
) -> Dict[int, List[Tuple[str, int, Path]]]:
    base = fly_dir / "RMS_calculations"
    results: Dict[int, Dict[str, Tuple[int, Path]]] = {}
    if not base.is_dir():
        return {}

    category_lower = category.lower()
    for path, token, slot_idx in iter_fly_distance_csvs(base, recursive=True):
        name_lower = path.name.lower()
        if category_lower not in name_lower:
            continue
        trial = extract_trial_index(path.stem, category_lower)
        if trial is None:
            continue

        entries = results.setdefault(trial, {})
        existing = entries.get(token)
        prefer_new = name_lower.startswith("updated_")
        if existing is None or (prefer_new and not existing[1].name.lower().startswith("updated_")):
            entries[token] = (slot_idx, path)

    ordered: Dict[int, List[Tuple[str, int, Path]]] = {}
    for trial, entry_map in results.items():
        ordered[trial] = sorted(
            [(token, slot_idx, path) for token, (slot_idx, path) in entry_map.items()],
            key=lambda item: item[1],
        )
    return ordered


def _find_video_for_trial(fly_dir: Path, category: str, tri: int) -> Optional[Path]:
    video_dir = fly_dir / VIDEO_INPUT_DIR / category
    if not video_dir.is_dir():
        return None
    videos = [p for p in video_dir.iterdir() if _is_video(p)]
    token = f"{category}_{tri}"
    exact = [p for p in videos if token in p.stem.lower()]
    if exact:
        return exact[0]
    loose = [p for p in videos if re.search(rf"[_\-]{tri}(\D|$)", p.stem)]
    if loose:
        return loose[0]
    if len(videos) == 1:
        return videos[0]
    return None


def _derive_fly_label(token: str, slot_idx: int) -> str:
    """
    Derive human-readable label from token.
    Used for legend in multi-fly RMS plots.

    Examples:
        "fly_0_slot1_distances" → "Fly 0 Slot 1"
        "slot2_distances" → "Slot 2"
        "unknown_token" → "Series 0"
    """
    # Remove "_distances" suffix
    clean = token.replace("_distances", "")

    # Extract fly and slot numbers if present
    parts = []
    if "fly_" in clean or "fly" in clean.lower():
        # Extract fly number
        match = re.search(r'fly[_\s]*(\d+)', clean, re.IGNORECASE)
        if match:
            parts.append(f"Fly {match.group(1)}")

    if "slot" in clean.lower():
        match = re.search(r'slot[_\s]*(\d+)', clean, re.IGNORECASE)
        if match:
            parts.append(f"Slot {match.group(1)}")

    if not parts:
        # Fallback: use slot_idx
        parts.append(f"Series {slot_idx}")

    return " ".join(parts)


def _render_line_panel_png(
    series_list: List[dict],
    width_px: int,
    height_px: int,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    odor_on: float | None,
    odor_off: float | None,
    threshold: float | None,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(width_px / 100, height_px / 100), dpi=100)

    for series in series_list:
        # Use explicit color if provided, otherwise let matplotlib auto-cycle through default colors
        if "color" in series:
            ax.plot(series["t"], series["y"], label=series.get("label", "RMS"), linewidth=1.2, color=series["color"])
        else:
            ax.plot(series["t"], series["y"], label=series.get("label", "RMS"), linewidth=1.2)

    if threshold is not None and np.isfinite(threshold):
        ax.axhline(threshold, color="red", linewidth=1.2, label="Threshold")

    if (
        odor_on is not None
        and odor_off is not None
        and np.isfinite(odor_on)
        and np.isfinite(odor_off)
    ):
        ax.axvline(odor_on, color="red", linewidth=1.0)
        ax.axvline(odor_off, color="red", linewidth=1.0)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(loc="upper right", frameon=True, framealpha=0.9)
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_linewidth(0.8)

    ax.axis("off")
    plt.tight_layout(pad=0)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buffer.seek(0)
    array = np.array(Image.open(buffer).convert("RGB"))
    return np.array(Image.fromarray(array).resize((width_px, height_px), resample=Image.BILINEAR))


def _validate_fps(fps_value: float | None, fallback: float = 40.0) -> float:
    """Validate fps is a finite positive number, return fallback if not."""
    if fps_value is None:
        return fallback
    try:
        fps_float = float(fps_value)
        if not math.isfinite(fps_float) or fps_float <= 0:
            return fallback
        return fps_float
    except (TypeError, ValueError):
        return fallback


def _print_debug_info(output_fps: float, clip: VideoFileClip, panel_clip: VideoClip, composite: CompositeVideoClip) -> None:
    """Print debug information about fps values (gated by COMPOSE_RMS_DEBUG env var)."""
    global _DEBUG_PRINTED
    if not _DEBUG_COMPOSE:
        return

    # Print version info once per run
    if not _DEBUG_PRINTED:
        import moviepy
        import decorator
        print("[COMPOSE_RMS_DEBUG] === Version Info ===")
        print(f"  moviepy version: {moviepy.__version__}")
        print(f"  decorator version: {decorator.__version__}")
        _DEBUG_PRINTED = True

    print("[COMPOSE_RMS_DEBUG] === FPS Debug Block ===")
    print(f"  output_fps: {output_fps} (type: {type(output_fps).__name__})")
    print(f"  source clip.fps: {clip.fps} (type: {type(clip.fps).__name__ if clip.fps is not None else 'NoneType'})")
    print(f"  panel_clip.fps: {panel_clip.fps} (type: {type(panel_clip.fps).__name__ if panel_clip.fps is not None else 'NoneType'})")
    print(f"  composite.fps: {composite.fps} (type: {type(composite.fps).__name__ if composite.fps is not None else 'NoneType'})")
    print("[COMPOSE_RMS_DEBUG] ======================")


def _compose_via_ffmpeg(
    video_path: Path,
    panel_png_path: Path,
    out_mp4: Path,
    output_fps: float,
    duration: float,
    video_height: int,
    panel_height: int,
) -> bool:
    """
    Compose video using direct ffmpeg instead of MoviePy.
    This is a fallback for when MoviePy's decorator chain loses fps.
    """
    try:
        # Build ffmpeg command to vstack original video with panel image
        # The panel is converted to a video stream at the same fps and duration
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-loop", "1", "-t", str(duration), "-framerate", str(output_fps),
            "-i", str(panel_png_path),
            "-filter_complex",
            f"[0:v]fps={output_fps}[v0];[1:v]scale=-1:{panel_height}[v1];[v0][v1]vstack=inputs=2",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-an",  # No audio
            "-r", str(output_fps),
            str(out_mp4),
        ]

        if _DEBUG_COMPOSE:
            print(f"[COMPOSE_RMS_DEBUG] ffmpeg command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[COMPOSE_RMS] ffmpeg failed: {result.stderr}")
            return False

        return out_mp4.exists() and out_mp4.stat().st_size > 0
    except Exception as e:
        print(f"[COMPOSE_RMS] ffmpeg fallback error: {e}")
        return False


def _compose_lineplot_video(
    video_path: Path,
    series_list: List[dict],
    xlim: Tuple[float, float],
    odor_on: float | None,
    odor_off: float | None,
    out_mp4: Path,
    *,
    panel_height_fraction: float,
    ylim: Tuple[float, float],
    threshold: float | None,
) -> bool:
    clip = VideoFileClip(str(video_path))
    vw, vh = clip.size
    panel_height = max(1, int(vh * panel_height_fraction))
    background = _render_line_panel_png(series_list, vw, panel_height, xlim, ylim, odor_on, odor_off, threshold)

    # Determine output fps with defensive validation
    raw_fps = clip.fps
    output_fps = _validate_fps(raw_fps, fallback=40.0)

    if _DEBUG_COMPOSE:
        print(f"[COMPOSE_RMS_DEBUG] raw clip.fps={raw_fps}, validated output_fps={output_fps}")

    # Use ffmpeg fallback if enabled
    if _USE_FFMPEG_FALLBACK:
        clip.close()
        out_mp4.parent.mkdir(parents=True, exist_ok=True)

        # Save panel as temporary PNG for ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            panel_png_path = Path(tmp.name)
            Image.fromarray(background).save(panel_png_path)

        try:
            # Get duration before closing clip
            clip = VideoFileClip(str(video_path))
            duration = clip.duration
            clip.close()

            return _compose_via_ffmpeg(
                video_path, panel_png_path, out_mp4,
                output_fps, duration, vh, panel_height
            )
        finally:
            try:
                panel_png_path.unlink()
            except Exception:
                pass

    # MoviePy path with enforced fps via set_fps() API
    def _add_cursor(img: np.ndarray, current_time: float, x_range: Tuple[float, float]) -> np.ndarray:
        output = img.copy()
        start, end = x_range
        if not (np.isfinite(start) and np.isfinite(end)) or end <= start:
            return output
        frac = float(np.clip((current_time - start) / (end - start), 0.0, 0.9999))
        x_pos = int(frac * (output.shape[1] - 1))
        output[:, x_pos : x_pos + 2, 0] = 255
        output[:, x_pos : x_pos + 2, 1:] = 0
        return output

    def panel_frame(t_cur: float) -> np.ndarray:
        return _add_cursor(background, t_cur, xlim)

    # Apply fps using MoviePy's set_fps() API (returns new clip object)
    clip = clip.set_fps(output_fps)

    panel_clip = VideoClip(panel_frame, duration=clip.duration)
    panel_clip = panel_clip.set_fps(output_fps)

    composite = CompositeVideoClip(
        [clip.set_position(("center", 0)), panel_clip.set_position(("center", vh))],
        size=(vw, vh + panel_height),
    )
    composite = composite.set_fps(output_fps)

    # Print debug info before writing
    _print_debug_info(output_fps, clip, panel_clip, composite)

    # Final defensive check: abort if fps is still None
    if composite.fps is None or not math.isfinite(composite.fps) or composite.fps <= 0:
        print(f"[COMPOSE_RMS] ERROR: composite.fps is invalid ({composite.fps}) despite set_fps(). Aborting write.")
        for obj in (clip, panel_clip, composite):
            try:
                obj.close()
            except Exception:
                pass
        return False

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    composite.write_videofile(str(out_mp4), fps=output_fps, codec="libx264", audio=False, preset="ultrafast")

    for obj in (clip, panel_clip, composite):
        try:
            obj.close()
        except Exception:
            pass

    try:
        return out_mp4.exists() and out_mp4.stat().st_size > 0
    except Exception:
        return False


def _ead_compute_trim_min_max(fly_dir: Path) -> Optional[Tuple[float, float]]:
    base = fly_dir / "RMS_calculations"
    if not base.is_dir():
        return None
    values: List[np.ndarray] = []
    for path, _, _ in iter_fly_distance_csvs(base, recursive=True):
        try:
            df = pd.read_csv(path)
            # Use flexible column finder instead of hardcoded column name
            dist_col = find_proboscis_distance_percentage_column(df)
            if dist_col is None:
                # Fallback to old column name
                dist_col = find_col(
                    df,
                    [
                        DIST_COL_ROBUST,
                        PCT_COL_ROBUST,
                        "distance_percentage_2_8",
                        "distance_percentage",
                        "distance_pct",
                        "distance_class1_class2_pct",
                    ],
                )
            if dist_col is None:
                continue
            arr = pd.to_numeric(df[dist_col], errors="coerce").to_numpy()
        except Exception:
            continue
        values.append(arr)
    if not values:
        return None
    all_values = np.concatenate(values)
    all_values = all_values[np.isfinite(all_values)]
    if all_values.size == 0:
        return None
    global_max = float(np.max(all_values))
    percentile = float(np.percentile(all_values, 100 * TRIM_FRAC, method="linear"))
    kept = all_values[all_values >= percentile]
    trimmed_min = float(np.min(kept)) if kept.size else float(np.min(all_values))
    return trimmed_min, global_max


def _discover_month_folders(root: Path) -> List[Path]:
    return [path for path in root.iterdir() if path.is_dir() and any(path.name.lower().startswith(m) for m in MONTHS)]


def _process_fly_angles(fly_dir: Path) -> None:
    """
    Process fly directory to add angle columns to RMS_calculations CSVs.

    Modification #3: Adds angle_centered_deg and angle_multiplier columns.
    Modification #5: Uses trimmed_min for stable reference angle calculation.

    Args:
        fly_dir: Path to fly directory containing RMS_calculations/
    """
    rms_dir = fly_dir / "RMS_calculations"
    if not rms_dir.is_dir():
        return

    # Get trimmed min for stable reference angle (Modification #5)
    trimmed_stats = _ead_compute_trim_min_max(fly_dir)
    trimmed_min = trimmed_stats[0] if trimmed_stats else None

    # Find all CSVs in RMS_calculations
    csv_paths = list(rms_dir.glob("*.csv"))
    if not csv_paths:
        return

    # Calculate reference angle using trimmed_min (Modification #5)
    reference_angle = find_fly_reference_angle(csv_paths, trimmed_min=trimmed_min)
    if not np.isfinite(reference_angle):
        print(f"[ANGLES] {fly_dir.name}: Could not determine reference angle, using 0.0")
        reference_angle = 0.0
    else:
        print(f"[ANGLES] {fly_dir.name}: Reference angle = {reference_angle:.2f}°")

    # Process each CSV to add angle columns
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)

            # Skip if already processed
            if "angle_multiplier" in df.columns:
                continue

            # Compute raw angles
            try:
                angles = compute_angle_deg_at_point2(df)
            except Exception as e:
                print(f"[ANGLES] {csv_path.name}: Could not compute angles: {e}")
                continue

            # Center angles relative to reference (Modification #5)
            centered_angles = angles - reference_angle

            # Compute continuous multipliers (Modification #3)
            multipliers = compute_angle_multiplier_series(centered_angles)

            # Add new columns
            df["angle_ARB_deg"] = angles
            df["angle_centered_deg"] = centered_angles
            df["angle_multiplier"] = multipliers

            # Save back to CSV
            df.to_csv(csv_path, index=False)
            print(f"[ANGLES] {csv_path.name}: Added angle columns (ref={reference_angle:.2f}°)")

        except Exception as e:
            print(f"[ANGLES] {csv_path.name}: Error processing: {e}")
            continue


def main(cfg: Settings) -> None:
    # Kill-switch: skip entire step if DISABLE_COMPOSE_RMS is set
    if os.environ.get("DISABLE_COMPOSE_RMS", "").lower() in ("1", "true", "yes"):
        print("[RMS] compose_videos_rms step SKIPPED (DISABLE_COMPOSE_RMS=1)")
        return

    roots = get_main_directories(cfg)
    print(f"[RMS] Starting RMS composition in {len(roots)} directories")

    pre_sec = float(cfg.odor_on_s)
    odor_duration = max(float(cfg.odor_off_s) - float(cfg.odor_on_s), 0.0)
    post_sec = 90.0  # matches notebook behaviour
    odor_latency = max(float(cfg.odor_latency_s), 0.0)

    delete_source_after = bool(cfg.delete_source_after_render)

    for root in roots:
        if not root.is_dir():
            print(f"[RMS] Warning: Directory does not exist: {root}")
            continue

        print(f"[RMS] Processing root directory: {root}")
        for fly_dir in sorted(_discover_month_folders(root)):
            fly_name = fly_dir.name
            print(f"\n=== Fly: {fly_name} ===")

            # Modification #3: Process angles and add multiplier columns (uses Mod #5 for reference angle)
            _process_fly_angles(fly_dir)

            for category in ("training", "testing"):
                trial_map = _collect_trial_csvs(fly_dir, category)
                if not trial_map:
                    print(f"  [{category}] No trials discovered.")
                    continue

                out_root = fly_dir / VIDEO_INPUT_DIR / VIDEO_OUTPUT_SUBDIR
                out_root.mkdir(parents=True, exist_ok=True)

                for tri, slot_entries in sorted(trial_map.items()):
                    video_path = _find_video_for_trial(fly_dir, category, tri)
                    if not video_path:
                        print(f"  [{category} {tri}] ⤫ No matching video in {VIDEO_INPUT_DIR}/{category}/")
                        continue

                    # Build series_list for ALL flies/slots in this trial (multi-fly plotting)
                    series_list = []
                    global_xlim = [float('inf'), float('-inf')]
                    first_odor_on, first_odor_off, first_threshold = None, None, None

                    for token, slot_idx, csv_path in slot_entries:
                        series = _series_rms_from_rmscalc(
                            csv_path,
                            cfg.fps_default,
                            pre_sec,
                            odor_duration,
                            post_sec,
                            odor_latency,
                        )
                        if series is None:
                            slot_label = token.replace("_distances", "")
                            print(f"  [{category} {tri} {slot_label}] ⤫ No RMS series; skipping.")
                            continue

                        t, rms, odor_on, odor_off, threshold = series

                        # Derive label for legend (e.g., "Fly 0 Slot 1")
                        label = _derive_fly_label(token, slot_idx)
                        series_list.append({"t": t, "y": rms, "label": label})

                        # Update global xlim (across all series)
                        global_xlim[0] = min(global_xlim[0], float(np.nanmin(t)))
                        global_xlim[1] = max(global_xlim[1], float(np.nanmax(t)))

                        # Capture first non-None overlays
                        if first_odor_on is None and odor_on is not None and np.isfinite(odor_on):
                            first_odor_on = odor_on
                        if first_odor_off is None and odor_off is not None and np.isfinite(odor_off):
                            first_odor_off = odor_off
                        if first_threshold is None and threshold is not None and np.isfinite(threshold):
                            first_threshold = threshold

                    # Skip trial if no valid series found
                    if not series_list:
                        print(f"  [{category} {tri}] ⤫ No valid RMS series found; skipping trial.")
                        continue

                    # Single output mp4 per trial with all flies' RMS lines
                    out_mp4 = out_root / f"{fly_name}_{category}_{tri}_ALLFLIES_rms.mp4"
                    if out_mp4.exists():
                        print(f"  [{category} {tri}] ⤫ Exists, skipping: {out_mp4.name}")
                        if delete_source_after:
                            _safe_unlink(video_path)
                        continue

                    print(
                        f"  [{category} {tri}] ✓ Multi-fly Video: {video_path.name} ({len(series_list)} flies) → {out_mp4.name}"
                    )
                    ok = _compose_lineplot_video(
                        video_path,
                        series_list,  # All series for this trial
                        tuple(global_xlim),
                        first_odor_on,
                        first_odor_off,
                        out_mp4,
                        panel_height_fraction=PANEL_HEIGHT_FRACTION,
                        ylim=YLIM,
                        threshold=first_threshold,
                    )

                    if ok:
                        print(f"  [{category} {tri}] [SAVED] {out_mp4.name}")
                        if delete_source_after:
                            _safe_unlink(video_path)
                    else:
                        print(f"  [{category} {tri}] ⤫ Render failed; source retained.")

                if delete_source_after:
                    category_dir = fly_dir / VIDEO_INPUT_DIR / category
                    if category_dir.exists():
                        remaining = [p for p in category_dir.iterdir() if _is_video(p)]
                        if not remaining:
                            _maybe_rmdir_empty(category_dir)

        if delete_source_after:
            for fly_dir in sorted(_discover_month_folders(root)):
                cat_root = fly_dir / VIDEO_INPUT_DIR
                if cat_root.exists() and not any(cat_root.iterdir()):
                    _maybe_rmdir_empty(cat_root)


_cached_series: Dict[Path, Tuple[np.ndarray, np.ndarray, float, float, float]] = {}


def _series_rms_from_rmscalc(
        csv_path: Path,
        fps_default: float,
        pre_sec: float,
        odor_duration: float,
        post_sec: float,
        odor_latency: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, float, float, float]]:
        real_path = csv_path.resolve()
        if real_path in _cached_series:
            return _cached_series[real_path]

        if not csv_path.exists():
            return None

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        frame_col = find_col(df, ["frame", "Frame", "frame_num", "frame_index"])
        ts_col = find_col(
            df,
            ["timestamp", "Timestamp", "time", "Time", "time_seconds", "relative_time", "time_s"],
        )
        ts, _meta = ensure_time_series(df, frame_col, ts_col, fps_default)
        if ts.notna().sum() < 2:
            return None

        rel_time = pd.to_numeric(ts, errors="coerce")
        rel_time = rel_time - rel_time.min()
        df["relative_time"] = rel_time

        odor_window = odor_window_from_ofm(df, "relative_time")
        if odor_window is None:
            start = pre_sec + odor_latency
            odor_window = (start, start + odor_duration)
        else:
            odor_window = (float(odor_window[0]) + odor_latency, float(odor_window[1]) + odor_latency)

        odor_on, odor_off = odor_window
        total_duration = pre_sec + (odor_off - odor_on) + post_sec

        pct_col = find_proboscis_distance_percentage_column(df)
        if pct_col is None:
            pct_col = find_col(df, [PCT_COL_ROBUST, "distance_class1_class2_pct"])
        if pct_col is None:
            return None

        values = pd.to_numeric(df[pct_col], errors="coerce").to_numpy(dtype=float)
        fps = derive_fps(df, fps_default)
        window = max(1, int(round(fps * RMS_WINDOW_S)))
        series = pd.Series(values)
        rms = series.rolling(window, min_periods=max(1, window // 2), center=True).apply(
            lambda x: float(np.sqrt(np.nanmean(np.square(x)))), raw=False
        ).to_numpy()

        t_axis = np.linspace(0.0, total_duration, len(rms))

        pre_mask = rel_time.to_numpy(dtype=float) < (odor_on if np.isfinite(odor_on) else pre_sec)
        if pre_mask.size != rms.size:
            approx_len = max(1, int(pre_sec * fps))
            pre_vals = rms[:approx_len]
        else:
            pre_vals = rms[pre_mask]
        mu = float(np.nanmean(pre_vals)) if np.isfinite(pre_vals).any() else float("nan")
        sd = float(np.nanstd(pre_vals)) if np.isfinite(pre_vals).any() else float("nan")
        threshold = mu + THRESH_K * sd if np.isfinite(mu) and np.isfinite(sd) else float("nan")

        payload = (t_axis, rms.astype(float), float(odor_on), float(odor_off), threshold)
        _cached_series[real_path] = payload
        return payload
