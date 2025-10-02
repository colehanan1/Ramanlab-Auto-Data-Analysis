
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import re, io, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from moviepy.editor import VideoFileClip, VideoClip, CompositeVideoClip
from ..config import Settings

VIDEO_INPUT_DIR = "videos_with_rms"
VIDEO_OUTPUT_SUBDIR = "with_line_plots"   # cleaned to avoid nested same-name dir
VIDEO_EXTS = {".mp4",".mov",".avi",".mkv",".mpg",".mpeg",".m4v"}
PANEL_HEIGHT_FRACTION = 0.24
YLIM = (0, 100) # RMS normalized % is 0..100; threshold overlaid
RMS_WINDOW_S = 1.0
THRESH_K = 4.0

def _is_video(p: Path): return p.is_file() and p.suffix.lower() in VIDEO_EXTS

def _extract_trial_index(name: str, category: str) -> Optional[int]:
    m = re.search(rf"{category}_(\d+)", name.lower())
    if m:
        try: return int(m.group(1))
        except: return None
    nums = re.findall(r"\d+", name)
    if nums:
        try: return int(nums[-1])
        except: return None
    return None

def _discover_trials(fly_dir: Path, category: str) -> List[int]:
    trials=set()
    rdir = fly_dir / "RMS_calculations"
    if rdir.is_dir():
        for p in rdir.glob("*merged.csv"):
            if category in p.name.lower():
                ti = _extract_trial_index(p.stem, category)
                if ti is not None: trials.add(ti)
    return sorted(trials)

def _find_video_for_trial(fly_dir: Path, category: str, tri: int) -> Optional[Path]:
    vid_dir = fly_dir / VIDEO_INPUT_DIR / category
    if not vid_dir.is_dir(): return None
    vids = [p for p in vid_dir.iterdir() if _is_video(p)]
    token = f"{category}_{tri}"
    cand = [v for v in vids if token in v.stem.lower()]
    if cand: return cand[0]
    cand = [v for v in vids if re.search(rf"[_\-]{tri}(\D|$)", v.stem)]
    if cand: return cand[0]
    return vids[0] if len(vids)==1 else None

def _series_rms_from_rmscalc(fly_dir: Path, category: str, trial_index: int, fps_default: float) -> Optional[Tuple[np.ndarray,np.ndarray,float,float,float]]:
    base = fly_dir / "RMS_calculations"
    if not base.is_dir(): return None
    cands = sorted([p for p in base.glob("*merged.csv") if category in p.name.lower() and _extract_trial_index(p.stem, category)==trial_index])
    if not cands: return None
    p = cands[0]
    df = pd.read_csv(p)
    df.columns = df.columns.str.strip()
    # time
    if "time_seconds" in df.columns:
        ts = pd.to_numeric(df["time_seconds"], errors="coerce")
    elif "timestamp" in df.columns:
        # fallback: assume seconds float-like
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
    else:
        ts = pd.Series(np.arange(len(df))/fps_default)
    rel = ts - ts.min()
    df["relative_time"] = rel
    # odor
    if "OFM_State" in df.columns:
        mask = df["OFM_State"].astype(str).str.lower().eq("during").to_numpy()
        if mask.any():
            idx = np.flatnonzero(mask); odor_on = float(rel.iloc[idx[0]]); odor_off = float(rel.iloc[idx[-1]])
        else:
            odor_on, odor_off = 30.0, 60.0
    else:
        odor_on, odor_off = 30.0, 60.0
    # pick distance-%
    for c in ["distance_percentage_2_6","distance_percentage","distance_class1_class2_pct"]:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float); break
    else:
        return None
    fps = fps_default
    win = max(1, int(round(fps*RMS_WINDOW_S)))
    s = pd.Series(vals)
    rms = s.rolling(win, min_periods=max(1, win//2), center=True).apply(lambda x: float(np.sqrt(np.nanmean(np.square(x)))), raw=False).to_numpy()
    # threshold (pre-odor)
    pre = rms[rel.to_numpy(dtype=float) < (odor_on if np.isfinite(odor_on) else 30.0)]
    mu = float(np.nanmean(pre)) if pre.size else np.nan
    sd = float(np.nanstd(pre)) if pre.size else np.nan
    thr = mu + THRESH_K * sd if np.isfinite(mu) and np.isfinite(sd) else np.nan
    t = np.linspace(0.0, (30.0 + (odor_off-odor_on) + 90.0), len(rms))
    return t, rms.astype(float), float(odor_on), float(odor_off), thr

def _render_line_panel_png(series_list: List[dict], width_px: int, height_px: int,
                           xlim: Tuple[float,float], ylim: Tuple[float,float],
                           odor_on: float | None, odor_off: float | None,
                           threshold: float | None) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(width_px/100, height_px/100), dpi=100)
    if series_list:
        s = series_list[0]
        ax.plot(s["t"], s["y"], label="RMS", linewidth=1.2)
    if threshold is not None and np.isfinite(threshold):
        ax.axhline(threshold, linewidth=1.2, label="Threshold")
    if odor_on is not None and odor_off is not None and np.isfinite(odor_on) and np.isfinite(odor_off):
        ax.axvline(odor_on, linewidth=1.0)
        ax.axvline(odor_off, linewidth=1.0)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax.axis("off")
    import io
    buf = io.BytesIO(); plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0); plt.close(fig); buf.seek(0)
    arr = np.array(Image.open(buf).convert("RGB"))
    return np.array(Image.fromarray(arr).resize((width_px, height_px), resample=Image.BILINEAR))

def _compose(video_path: Path, panel_img_fn, xlim, out_mp4: Path) -> bool:
    clip = VideoFileClip(str(video_path))
    vw, vh = clip.size
    panel_clip = VideoClip(lambda t: panel_img_fn(t), duration=clip.duration)
    comp = CompositeVideoClip([clip.set_position(("center", 0)),
                               panel_clip.set_position(("center", vh))],
                              size=(vw, vh + panel_clip.size[1]))
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    comp.write_videofile(str(out_mp4), fps=clip.fps, codec="libx264", audio=False, preset="ultrafast")
    for c in (clip, panel_clip, comp):
        try: c.close()
        except: pass
    return out_mp4.exists() and out_mp4.stat().st_size>0

def main(cfg: Settings):
    root = Path(cfg.main_directory).expanduser().resolve()
    for fly in [p for p in root.iterdir() if p.is_dir()]:
        for category in ("training","testing"):
            trials = _discover_trials(fly, category)
            if not trials: 
                continue
            out_root = fly / VIDEO_INPUT_DIR / VIDEO_OUTPUT_SUBDIR
            out_root.mkdir(parents=True, exist_ok=True)
            for tri in trials:
                vid = _find_video_for_trial(fly, category, tri)
                if not vid:
                    print(f"[RMSVID] No video for {fly.name} {category} {tri}")
                    continue
                series = _series_rms_from_rmscalc(fly, category, tri, cfg.fps_default)
                if series is None:
                    print(f"[RMSVID] No RMS for {fly.name} {category} {tri}")
                    continue
                t, y, on, off, thr = series
                xlim=(float(np.nanmin(t)), float(np.nanmax(t)))
                def panel_img_fn(cur_t: float, vw=[None], ph=[None]):
                    W, H = (1920, 1080)  # approximate; moviepy provides size via video clip
                    # We'll regenerate BG once to avoid heavy redraw cost
                    static_bg = _render_line_panel_png([{"t":t,"y":y}], width_px=W, height_px=int(1080*PANEL_HEIGHT_FRACTION),
                                                       xlim=xlim, ylim=YLIM, odor_on=on, odor_off=off, threshold=thr)
                    # draw time cursor
                    img = static_bg.copy()
                    x = int((cur_t - xlim[0]) / (xlim[1]-xlim[0]) * (img.shape[1]-1))
                    x = max(0, min(img.shape[1]-2, x))
                    img[:, x:x+2, 0] = 255; img[:, x:x+2, 1:] = 0
                    return img
                out_mp4 = out_root / f"{fly.name}_{category}_{tri}_LINES_rms.mp4"
                if out_mp4.exists():
                    print(f"[RMSVID] Exists: {out_mp4.name} — skipping")
                    continue
                ok = _compose(vid, panel_img_fn, xlim, out_mp4)
                if ok and cfg.delete_source_after_render:
                    try: vid.unlink()
                    except Exception: pass
                print(f"[RMSVID] {'OK' if ok else 'FAIL'} {out_mp4}")
