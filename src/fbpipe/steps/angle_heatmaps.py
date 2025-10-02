
from __future__ import annotations
from pathlib import Path
import glob, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from ..config import Settings

def _infer_category(p: Path):
    s = ("/".join(p.parts) + p.stem).lower()
    if "testing" in s: return "testing"
    if "training" in s: return "training"
    return None

def _time_axis(df: pd.DataFrame, fps: float):
    if "time_s" in df.columns: return df["time_s"].astype(float).to_numpy()
    if "frame"  in df.columns: return (pd.to_numeric(df["frame"], errors="coerce").to_numpy(dtype=float) / fps)
    return np.arange(len(df))/fps

def main(cfg: Settings):
    root = Path(cfg.main_directory).expanduser().resolve()
    for fly in [p for p in root.iterdir() if p.is_dir()]:
        out_dir = fly / "angle_plots" / "angle_plots_centered_percentage_heatmaps"
        out_dir.mkdir(parents=True, exist_ok=True)
        csvs = [Path(x) for x in glob.iglob(str(fly / "**" / "*merged.csv"), recursive=True)]
        groups = {"training": [], "testing": []}
        for p in csvs:
            cat = _infer_category(p) or "testing"
            groups[cat].append(p)
        for cat, paths in groups.items():
            trials = []
            for p in sorted(paths):
                df = pd.read_csv(p)
                if "angle_centered_pct" in df.columns:
                    t = _time_axis(df, cfg.fps_default)
                    y = pd.to_numeric(df["angle_centered_pct"], errors="coerce").to_numpy(dtype=float)
                else:
                    # compute from deg using per-fly max |centered| (approximate per file)
                    if "angle_centered_deg" not in df.columns: 
                        continue
                    t = _time_axis(df, cfg.fps_default)
                    deg = pd.to_numeric(df["angle_centered_deg"], errors="coerce").to_numpy(dtype=float)
                    denom = np.nanmax(np.abs(deg)) if np.isfinite(np.nanmax(np.abs(deg))) and np.nanmax(np.abs(deg))>0 else 1.0
                    y = (deg/denom)*100.0
                trials.append((p.stem, t, y))
            if not trials: 
                continue
            n = len(trials); fig, axs = plt.subplots(n,1, figsize=(18, max(2,2*n)), sharex=False)
            if n==1: axs=[axs]
            cmap = mpl.colormaps["coolwarm"].copy(); cmap.set_bad("dimgray")
            norm = Normalize(vmin=-100, vmax=100)
            for ax, (label,t,y) in zip(axs, trials):
                if len(t)<2 or len(y)!=len(t):
                    ax.text(0.5, 0.5, f"Skipped: {label}", transform=ax.transAxes, ha="center"); ax.set_yticks([]); continue
                edges = np.linspace(t[0], t[-1], len(t)+1)
                X,Ym = np.meshgrid(edges, [0,1])
                row = np.asarray(y, dtype=float).reshape(1,-1)
                ax.pcolormesh(X, Ym, row, cmap=cmap, shading="auto", norm=norm)
                ax.set_yticks([]); ax.set_xlim(t[0], t[-1]); ax.set_title(label, loc="left")
            axs[-1].set_xlabel("Time (s)")
            fig.suptitle(f"{fly.name} – {cat.capitalize()} Trials (Centered Angle %)", fontsize=20)
            fig.tight_layout(rect=[0,0,1,0.96])
            out_png = out_dir / f"{fly.name}_{cat}_angle_centered_pct_heatmap.png"
            plt.savefig(out_png, bbox_inches="tight"); plt.close()
            print(f"[HEAT] {out_png}")
