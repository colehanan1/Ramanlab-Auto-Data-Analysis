
from __future__ import annotations
from pathlib import Path
import re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.signal import hilbert
from ..config import Settings

TESTING_REGEX = re.compile(r"testing_(\d+)")

def _resolve_measure_column(df: pd.DataFrame):
    for c in ["distance_percentage_2_6","distance_percentage"]:
        if c in df.columns: return c
    return None

def _compute_envelope(series: pd.Series, win_frames: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.between(0,100, inclusive="both")
    if not valid.any(): return pd.Series(np.nan, index=s.index)
    s_interp = s.where(valid).interpolate(limit_direction="both")
    if s_interp.isna().all(): return pd.Series(np.nan, index=s.index)
    analytic = hilbert(s_interp.to_numpy())
    env = np.abs(analytic)
    env_series = pd.Series(env, index=s.index).mask(~valid)
    return env_series.rolling(window=win_frames, center=True, min_periods=1).mean()

def _rolling_rms_valid(series: pd.Series, win_frames: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.between(0,100, inclusive="both")
    s2 = s.where(valid).pow(2)
    rms = s2.rolling(window=win_frames, center=True, min_periods=1).mean()
    return rms.pow(0.5)

def main(cfg: Settings):
    root = Path(cfg.main_directory).expanduser().resolve()
    win_frames = max(int(cfg.window_sec * cfg.fps_default), 1)
    for fly in [p for p in root.iterdir() if p.is_dir()]:
        rdir = fly / "RMS_calculations"
        if not rdir.is_dir(): continue
        csvs = sorted(rdir.glob("*testing*.csv"))
        trials = {}
        gmax = 0.0
        for p in csvs:
            df = pd.read_csv(p)
            meas = _resolve_measure_column(df)
            if meas is None: continue
            time = df["time_seconds"].to_numpy(dtype=float) if "time_seconds" in df.columns else np.arange(len(df))/cfg.fps_default
            env = _compute_envelope(df[meas], win_frames).to_numpy()
            df[f"{meas}_rms_win{win_frames}"] = _rolling_rms_valid(df[meas], win_frames)
            df.to_csv(p, index=False)

            pre_mask = time < cfg.odor_on_s
            pre_vals = env[pre_mask] if np.any(pre_mask) else env[:max(1,int(cfg.odor_on_s*cfg.fps_default))]
            mu = float(np.nanmean(pre_vals)) if pre_vals.size else 0.0
            sd = float(np.nanstd(pre_vals)) if pre_vals.size else 0.0
            thr = mu + 4.0 * sd
            local_max = float(np.nanmax(env)) if env.size else 0.0
            gmax = max(gmax, local_max if np.isfinite(local_max) else 0.0)
            label = TESTING_REGEX.search(p.stem).group(0) if TESTING_REGEX.search(p.stem) else p.stem
            trials[label] = (time, env, thr)

        if not trials: 
            continue
        # plot
        out_dir = fly / "RMS_calculations" / "envelope_over_time_plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"{fly.name}_envelope_over_time_subplots.png"

        n = len(trials); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(n, 1, figsize=(10, 2.5*n), sharex=True)
        if n==1: axes=[axes]
        ymax = gmax*1.02 if gmax>0 else 1.0
        for ax, (label,(t, env, thr)) in zip(axes, trials.items()):
            ax.plot(t, env, linewidth=1)
            if np.any(np.isfinite(env)):
                idx = np.nanargmax(env); ax.plot(t[idx], env[idx], "o", ms=6)
            ax.axhline(thr, linestyle="-", linewidth=1)
            ax.axvline(cfg.odor_on_s, linestyle="--", linewidth=1)
            ax.axvline(cfg.odor_off_s, linestyle="--", linewidth=1)
            ax.axvspan(cfg.odor_on_s, cfg.odor_off_s, alpha=0.15)
            ax.set_ylim(0, ymax); ax.set_ylabel("Envelope"); ax.set_title(label); ax.grid(True)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"{fly.name}: Envelope (win={cfg.window_sec}s; odor {cfg.odor_on_s:.0f}–{cfg.odor_off_s:.0f}s)", y=0.98)
        fig.tight_layout(rect=[0,0,1,0.95])
        fig.savefig(out_png, dpi=300); plt.close(fig)
        print(f"[ENV] {out_png}")
