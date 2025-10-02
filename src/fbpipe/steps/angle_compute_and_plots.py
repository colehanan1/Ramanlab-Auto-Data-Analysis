
from __future__ import annotations
from pathlib import Path
import glob, re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from ..config import Settings

MONTHS = ("january","february","march","april","may","june","july","august","september","october","november","december")

def compute_angle_deg(df: pd.DataFrame, anchor_x: float, anchor_y: float) -> pd.Series:
    req = ["x_class2","y_class2","x_class6","y_class6"]
    if any(c not in df.columns for c in req):
        return pd.Series(np.nan, index=df.index)
    p2x = df["x_class2"].astype(float); p2y = df["y_class2"].astype(float)
    p3x = df["x_class6"].astype(float); p3y = df["y_class6"].astype(float)
    ux, uy = (anchor_x - p2x), (anchor_y - p2y)
    vx, vy = (p3x - p2x), (p3y - p2y)
    dot = ux*vx + uy*vy; cross = ux*vy - uy*vx
    n1 = np.hypot(ux, uy); n2 = np.hypot(vx, vy)
    valid = (n1>0) & (n2>0) & np.isfinite(dot) & np.isfinite(cross)
    ang = np.full(len(df), np.nan)
    ang[valid.to_numpy()] = np.degrees(np.arctan2(np.abs(cross[valid]), dot[valid]))
    return pd.Series(ang, index=df.index, name="angle_ARB_deg")

def require_distance_col(df: pd.DataFrame) -> str | None:
    for cand in ["distance_percentage","distance_percentage_2_6","distance_pct"]:
        if cand in df.columns: return cand
    return None

def find_reference_angle(csv_paths, anchor_x, anchor_y) -> float:
    best = None
    for p in csv_paths:
        df = pd.read_csv(p)
        dist_col = require_distance_col(df)
        if dist_col is None: continue
        ang = compute_angle_deg(df, anchor_x, anchor_y)
        dist = pd.to_numeric(df[dist_col], errors="coerce").to_numpy()
        exact = np.flatnonzero(dist==0)
        if exact.size>0:
            idx = int(exact[0]); angle_here = float(ang.iloc[idx]) if np.isfinite(ang.iloc[idx]) else np.nan
            cand = (0, 0.0, angle_here)
        else:
            with np.errstate(invalid="ignore"):
                absd = np.abs(dist)
            if absd.size==0 or not np.isfinite(absd).any(): continue
            idx = int(np.nanargmin(absd))
            angle_here = float(ang.iloc[idx]) if np.isfinite(ang.iloc[idx]) else np.nan
            cand = (1, float(absd[idx]), angle_here)
        if best is None or cand < best: best = cand
    return best[2] if best else np.nan

def main(cfg: Settings):
    root = Path(cfg.main_directory).expanduser().resolve()
    for fly in [p for p in root.iterdir() if p.is_dir()]:
        # collect csvs under month folders
        month_folders = [sub for sub in fly.rglob("*") if sub.is_dir() and sub.name.lower().startswith(MONTHS)]
        csvs = []
        for mf in month_folders:
            csvs += [Path(x) for x in glob.iglob(str(mf / "**" / "*merged.csv"), recursive=True)]
        if not csvs: 
            continue
        ref = find_reference_angle(csvs, cfg.anchor_x, cfg.anchor_y)
        unc_dir = fly / "angle_plots"
        ctr_dir = unc_dir / "angle_plots_centered"
        unc_dir.mkdir(exist_ok=True); ctr_dir.mkdir(exist_ok=True)
        processed=0
        for p in csvs:
            df = pd.read_csv(p)
            ang = compute_angle_deg(df, cfg.anchor_x, cfg.anchor_y)
            df["angle_deg_c2_26_vs_anchor"] = ang
            df["angle_centered_deg"] = ang - ref if np.isfinite(ref) else np.nan
            df.to_csv(p, index=False); processed += 1
            # time axis
            t = (pd.to_numeric(df["frame"], errors="coerce")/cfg.fps_default if "frame" in df.columns 
                 else pd.Series(np.arange(len(df))/cfg.fps_default))
            # safe plot name
            safe = "_".join(p.relative_to(fly).with_suffix("").parts)
            # uncentered
            png_u = unc_dir / f"{safe}_angle_ARB.png"
            plt.figure(figsize=(10,4)); plt.plot(t, ang.to_numpy(), linewidth=1.5)
            plt.xlabel("Time (s)"); plt.ylabel("Angle (deg)"); plt.title(p.stem); plt.grid(True, alpha=0.25); plt.tight_layout()
            plt.savefig(png_u, dpi=150); plt.close()
            # centered
            png_c = ctr_dir / f"{safe}_angle_ARB_centered.png"
            plt.figure(figsize=(10,4)); plt.plot(t, (ang-ref).to_numpy(), linewidth=1.5)
            plt.axhline(0, linestyle="--", linewidth=1.0, alpha=0.6)
            plt.xlabel("Time (s)"); plt.ylabel("Centered angle (deg)"); plt.title(f"{p.stem} — ref={ref:.2f}°")
            plt.grid(True, alpha=0.25); plt.tight_layout()
            plt.savefig(png_c, dpi=150); plt.close()
        print(f"[ANGLE] {fly.name}: CSVs updated={processed}, plots at {unc_dir}")
