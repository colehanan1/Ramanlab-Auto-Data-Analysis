
from __future__ import annotations
from pathlib import Path
import re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from ..config import Settings

PHASE_ALIASES = {
    "before":"before","pre":"before","baseline":"before",
    "during":"during","on":"during","odor_on":"during",
    "after":"after","post":"after","odor_off":"after",
}
MEASURE_COLS = ["distance_percentage_2_6","distance_percentage"]
THRESHOLD = 1.1
FIGSIZE = (9,5)

def _resolve_ofm_column(df: pd.DataFrame) -> str:
    for c in ["OFM State","OFM_State"]:
        if c in df.columns: return c
    raise KeyError("OFM State column not found")

def _resolve_measure_column(df: pd.DataFrame) -> str:
    for c in MEASURE_COLS:
        if c in df.columns: return c
    raise KeyError("Measure col not found")

def _normalize_state(val: str) -> str:
    key = str(val).strip().lower()
    if key not in PHASE_ALIASES: return "unknown"
    return PHASE_ALIASES[key]

def _extract_trial_label(stem: str) -> str:
    m = re.search(r"testing_\d+", stem)
    return m.group(0) if m else stem

def compute_segment_rms(series: pd.Series) -> float:
    arr = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if arr.size == 0: return float("nan")
    return float(np.sqrt(np.mean(arr**2)))

def plot_grouped_bar_chart(fly: str, rows, out_path: Path):
    labels = [t for t,_ in rows]
    during = [d["during"] for _,d in rows]
    after  = [d["after"] for _,d in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(x-0.35, during, width=0.35, edgecolor="black", color=["red" if v<THRESHOLD else "green" for v in during])
    ax.bar(x,      after,  width=0.35, edgecolor="black", color=["red" if v<THRESHOLD else "green" for v in after])
    for bar in ax.patches:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{bar.get_height():.2f}", ha="center", va="bottom")
    ax.axhline(THRESHOLD, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean ratio (segment / baseline)"); ax.set_title(f"Fly: {fly} — During & After")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300); plt.close(fig)

def main(cfg: Settings):
    root = Path(cfg.main_directory).expanduser().resolve()
    for fly in [p for p in root.iterdir() if p.is_dir()]:
        rdir = fly / "RMS_calculations"
        csvs = sorted(rdir.glob("*testing*.csv"))
        if not csvs: continue
        rows = []
        for p in csvs:
            df = pd.read_csv(p)
            try:
                meas = _resolve_measure_column(df); state = _resolve_ofm_column(df)
            except Exception:
                continue
            clean = df[[meas, state]].copy()
            clean[meas] = pd.to_numeric(clean[meas], errors="coerce")
            clean = clean[clean[meas].between(0,100, inclusive="both")]
            if clean.empty: continue
            clean["phase"] = clean[state].apply(_normalize_state)
            base = clean[clean["phase"]=="before"][meas]
            if base.empty: continue
            mb = base.mean()
            clean["ratio"] = clean[meas] / mb
            rows.append((_extract_trial_label(p.stem), {
                "during": float(clean[clean["phase"]=="during"]["ratio"].mean()),
                "after":  float(clean[clean["phase"]=="after"]["ratio"].mean())
            }))
        if rows:
            out = fly / "RMS_calculations" / "histograms" / "odor_response_bar.png"
            plot_grouped_bar_chart(fly.name, rows, out)
            print(f"[HIST] {out}")
