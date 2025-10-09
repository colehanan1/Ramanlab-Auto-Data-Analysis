from __future__ import annotations

import io
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from moviepy.editor import CompositeVideoClip, VideoClip, VideoFileClip

from ..config import Settings
from ..utils.columns import (
    find_proboscis_distance_percentage_column,
    find_proboscis_xy_columns,
)
from ..utils.fly_files import iter_fly_distance_csvs

# Display defaults mirror the notebook example
plt.rcParams.update(
    {
        "font.family": "serif",
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
PCT_COL_ROBUST = "distance_class1_class2_pct"
DIST_COL_ROBUST = "distance_class1_class2"
TRIM_FRAC = 0.05
VIDEO_EXTS_CHECK = VIDEO_EXTS  # alias for clarity
LINE_COLOR_CYCLE = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

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


def _format_series_label(label: str) -> str:
    cleaned = label.replace("_", " ").strip()
    cleaned = re.sub(r"(?<=\D)(\d+)", r" \1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.title() if cleaned else label


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
    x2_col = find_col(df, ["x_class2", "x_class_2", "class2_x"])
    y2_col = find_col(df, ["y_class2", "y_class_2", "class2_y"])
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


def find_fly_reference_angle(csvs_raw: List[Path]) -> float:
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
        dist_col = find_col(
            df,
            [
                "distance_percentage",
                "distance_percent",
                "distance_pct",
                "distance_class1_class2_pct",
            ],
        )
        if dist_col is None:
            continue
        dist = pd.to_numeric(df[dist_col], errors="coerce").to_numpy()
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
        ax.plot(series["t"], series["y"], label=series.get("label", "RMS"), linewidth=1.2, color=series.get("color", "blue"))

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

    panel_clip = VideoClip(panel_frame, duration=clip.duration)
    composite = CompositeVideoClip(
        [clip.set_position(("center", 0)), panel_clip.set_position(("center", vh))],
        size=(vw, vh + panel_height),
    )

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    composite.write_videofile(str(out_mp4), fps=clip.fps, codec="libx264", audio=False, preset="ultrafast")

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
            arr = pd.to_numeric(
                pd.read_csv(path, usecols=[DIST_COL_ROBUST])[DIST_COL_ROBUST],
                errors="coerce",
            ).to_numpy()
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


def main(cfg: Settings) -> None:
    root = Path(cfg.main_directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"main_directory does not exist: {root}")

    pre_sec = float(cfg.odor_on_s)
    odor_duration = max(float(cfg.odor_off_s) - float(cfg.odor_on_s), 0.0)
    post_sec = 90.0  # matches notebook behaviour
    odor_latency = max(float(cfg.odor_latency_s), 0.0)

    delete_source_after = bool(cfg.delete_source_after_render)

    for fly_dir in sorted(_discover_month_folders(root)):
        fly_name = fly_dir.name
        print(f"\n=== Fly: {fly_name} ===")
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

                slot_success = False
                series_payloads: List[Dict[str, object]] = []
                for token, _slot_idx, csv_path in slot_entries:
                    slot_label = token.replace("_distances", "")
                    out_mp4 = out_root / f"{fly_name}_{slot_label}_{category}_{tri}_LINES_rms.mp4"
                    series = _series_rms_from_rmscalc(
                        csv_path,
                        cfg.fps_default,
                        pre_sec,
                        odor_duration,
                        post_sec,
                        odor_latency,
                    )
                    if series is None:
                        print(f"  [{category} {tri} {slot_label}] ⤫ No RMS series; skipping.")
                        continue

                    color_idx = len(series_payloads) % len(LINE_COLOR_CYCLE)
                    series_payloads.append(
                        {
                            "slot_label": slot_label,
                            "series": series,
                            "color": LINE_COLOR_CYCLE[color_idx],
                            "out_mp4": out_mp4,
                        }
                    )

                if not series_payloads:
                    continue

                label_summary = ", ".join(
                    _format_series_label(payload["slot_label"]) for payload in series_payloads
                )
                primary_entry = series_payloads[0]
                primary_output = cast(Path, primary_entry["out_mp4"])

                action = "↻ Overwriting" if primary_output.exists() else "✓ Video"
                print(
                    f"  [{category} {tri}] {action}: {video_path.name} → {primary_output.name} "
                    f"(lines: {label_summary})"
                )

                xmins: List[float] = []
                xmaxs: List[float] = []
                odor_ons: List[float] = []
                odor_offs: List[float] = []
                thresholds: List[float] = []

                for payload in series_payloads:
                    t, rms, odor_on, odor_off, threshold = cast(
                        Tuple[np.ndarray, np.ndarray, float, float, float], payload["series"]
                    )
                    t = np.asarray(t, dtype=float)
                    finite_t = t[np.isfinite(t)]
                    if finite_t.size:
                        xmins.append(float(finite_t.min()))
                        xmaxs.append(float(finite_t.max()))
                    if np.isfinite(odor_on):
                        odor_ons.append(float(odor_on))
                    if np.isfinite(odor_off):
                        odor_offs.append(float(odor_off))
                    if np.isfinite(threshold):
                        thresholds.append(float(threshold))

                if xmins and xmaxs:
                    xlim = (min(xmins), max(xmaxs))
                else:
                    max_len = max(
                        len(cast(Tuple[np.ndarray, np.ndarray, float, float, float], payload["series"])[0])
                        for payload in series_payloads
                    )
                    xlim = (0.0, float(max_len))

                odor_on_val = float(min(odor_ons)) if odor_ons else None
                odor_off_val = float(max(odor_offs)) if odor_offs else None

                threshold_val: Optional[float]
                if thresholds:
                    base = thresholds[0]
                    if any(not np.isclose(base, other, rtol=1e-3, atol=1e-3) for other in thresholds[1:]):
                        threshold_val = None
                    else:
                        threshold_val = base
                else:
                    threshold_val = None

                series_list = []
                for payload in series_payloads:
                    t, rms, _odor_on, _odor_off, _threshold = cast(
                        Tuple[np.ndarray, np.ndarray, float, float, float], payload["series"]
                    )
                    series_list.append(
                        {
                            "t": np.asarray(t, dtype=float),
                            "y": np.asarray(rms, dtype=float),
                            "label": _format_series_label(str(payload["slot_label"])),
                            "color": payload["color"],
                        }
                    )

                ok = _compose_lineplot_video(
                    video_path,
                    series_list,
                    xlim,
                    odor_on_val,
                    odor_off_val,
                    primary_output,
                    panel_height_fraction=PANEL_HEIGHT_FRACTION,
                    ylim=YLIM,
                    threshold=threshold_val,
                )

                if not ok:
                    print(
                        f"  [{category} {tri} {primary_entry['slot_label']}] ⤫ Render failed; source retained."
                    )
                    continue

                slot_success = True
                print(
                    f"  [{category} {tri} {primary_entry['slot_label']}] [SAVED] {primary_output.name}"
                )

                for payload in series_payloads[1:]:
                    dest = cast(Path, payload["out_mp4"])
                    existed = dest.exists()
                    try:
                        shutil.copy2(primary_output, dest)
                        tag = "[UPDATED]" if existed else "[SAVED]"
                        print(
                            f"  [{category} {tri} {payload['slot_label']}] {tag} {dest.name}"
                        )
                        slot_success = True
                    except Exception as exc:
                        print(
                            f"  [{category} {tri} {payload['slot_label']}] ⤫ Copy failed: {exc}"
                        )

                if delete_source_after and slot_success:
                    _safe_unlink(video_path)

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
        pct_col = find_col(df, ["distance_class1_class2_pct"])
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
