"""Statistics over the geometry pipeline's outputs.

One row per (batch, fly slot, trial) with proboscis-extension (PER %) and abdomen
metrics measured in the trial's stimulus window against its pre-stimulus baseline,
joined to the production model scores, the blinded human scores (with their
"abdomen" comments) and the flagged-fly states. Every statistic pools trials
(n = fly-trials); nothing here averages per-fly averages.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..config import Settings
from ..utils.frozen_folders import FROZEN_COLUMN
from . import columns as C
from .config import GeometrySettings
from .steps import iter_batches, load_trial_entries
from .trial_meta import TrialInfo

FLY_STATUS = {1: "alive", 0: "weak", -1: "dead"}
_TRIAL_RE = re.compile(r"(pretest|training|testing)_(\d+)", re.IGNORECASE)
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

ID_COLUMNS = ["dataset", "fly", "fly_number", "fly_type", "trial_type", "trial_index", "odor", "odor_norm",
              "trial_label", "trial_label_lc", "has_odor", "window_on_s", "window_off_s", "n_frames", FROZEN_COLUMN]
METRIC_COLUMNS = [
    "per_tracked_frac", "abdomen_tracked_frac", "angle_tracked_frac",
    "per_base", "per_win_mean", "per_win_max", "per_delta", "per_auc", "per_px_win_max",
    "angle_base", "angle_win_mean", "angle_win_max", "angle_delta", "angle_auc",
    "angle_activity_base", "angle_activity_win", "angle_range_win",
    "length_base", "length_win_mean", "length_delta", "width_base", "width_win_mean", "width_delta",
    "rho_length_angle", "rho_width_angle", "n_frames_box_angle", "dominant_status",
]
ANGLE_METRICS = ("angle_delta", "angle_win_max", "angle_activity_win", "angle_range_win")


def norm_odor(odor: Optional[str]) -> str:
    return str(odor or "").strip().casefold()


def _spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 3 or np.nanstd(x[m]) == 0 or np.nanstd(y[m]) == 0:
        return float("nan"), float("nan"), n
    rho, p = sps.spearmanr(x[m], y[m])
    return float(rho), float(p), n


def _window_stats(t: np.ndarray, y: np.ndarray, on: Optional[float], off: Optional[float]) -> dict:
    out = {"base": np.nan, "win_mean": np.nan, "win_max": np.nan, "delta": np.nan, "auc": np.nan,
           "act_base": np.nan, "act_win": np.nan, "range_win": np.nan}
    if on is None or off is None:
        return out
    fin = np.isfinite(y)
    base = fin & (t < on)
    win = fin & (t >= on) & (t < off)          # [on, off): the off frame is already odor-off
    if base.any():
        out["base"] = float(np.mean(y[base]))
        out["act_base"] = float(np.std(y[base]))
    if win.any():
        out["win_mean"] = float(np.mean(y[win]))
        out["win_max"] = float(np.max(y[win]))
        out["act_win"] = float(np.std(y[win]))
        out["range_win"] = float(np.max(y[win]) - np.min(y[win]))
        out["auc"] = float(_trapezoid(y[win], t[win])) if win.sum() > 1 else 0.0
    if base.any() and win.any():
        out["delta"] = out["win_mean"] - out["base"]
    return out


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    return df[name].to_numpy(dtype=float) if name in df.columns else np.full(len(df), np.nan)


def _dominant_status(df: pd.DataFrame) -> str:
    if C.STATUS not in df.columns:
        return ""
    s = df[C.STATUS].astype(str)
    bad = s[s != C.STATUS_OK]
    return str(bad.mode().iloc[0]) if len(bad) else C.STATUS_OK


def fly_trial_metrics_for_batch(batch_dir: Path, dataset: str, frozen: bool, gcfg: GeometrySettings,
                                settings: Settings) -> pd.DataFrame:
    rows = []
    for _, info, tables in load_trial_entries(batch_dir, gcfg):
        win = info.stimulus_window_s
        on, off = (win if win is not None else (None, None))
        for slot, path in sorted(tables.items()):
            df = C.read_table_any(path)
            t = _col(df, C.TIMESTAMP)
            per, per_px = _col(df, C.PRO_PCT), _col(df, C.PRO_DIST)
            ang, length, width = _col(df, C.ABD_ANGLE_PCT), _col(df, C.ABD_LEN_PCT), _col(df, C.ABD_WID_PCT)
            p, a, l, w = (_window_stats(t, y, on, off) for y in (per, ang, length, width))
            rho_l, _, n_la = _spearman(length, ang)
            rho_w, _, _ = _spearman(width, ang)
            wmask = np.isfinite(per_px) & (t >= on) & (t < off) if on is not None else np.zeros(len(t), bool)
            n = int(len(df))
            rows.append({
                "dataset": dataset or info.dataset, "fly": Path(batch_dir).name, "fly_number": int(slot),
                "fly_type": info.fly_type, "trial_type": info.trial_type, "trial_index": int(info.trial_index),
                "odor": info.odor, "odor_norm": norm_odor(info.odor), "trial_label": info.label,
                "trial_label_lc": info.label.casefold(), "has_odor": bool(info.has_odor),
                "window_on_s": on, "window_off_s": off, "n_frames": n, FROZEN_COLUMN: bool(frozen),
                "per_tracked_frac": float(np.isfinite(per).mean()) if n else 0.0,
                "abdomen_tracked_frac": float(np.isfinite(length).mean()) if n else 0.0,
                "angle_tracked_frac": float(np.isfinite(ang).mean()) if n else 0.0,
                "per_base": p["base"], "per_win_mean": p["win_mean"], "per_win_max": p["win_max"],
                "per_delta": p["delta"], "per_auc": p["auc"],
                "per_px_win_max": float(np.max(per_px[wmask])) if wmask.any() else np.nan,
                "angle_base": a["base"], "angle_win_mean": a["win_mean"], "angle_win_max": a["win_max"],
                "angle_delta": a["delta"], "angle_auc": a["auc"],
                "angle_activity_base": a["act_base"], "angle_activity_win": a["act_win"], "angle_range_win": a["range_win"],
                "length_base": l["base"], "length_win_mean": l["win_mean"], "length_delta": l["delta"],
                "width_base": w["base"], "width_win_mean": w["win_mean"], "width_delta": w["delta"],
                "rho_length_angle": rho_l, "rho_width_angle": rho_w, "n_frames_box_angle": n_la,
                "dominant_status": _dominant_status(df),
            })
    return pd.DataFrame(rows, columns=ID_COLUMNS + METRIC_COLUMNS)


def collect_fly_trial_metrics(settings: Settings, gcfg: GeometrySettings, *, include_frozen: bool = False) -> pd.DataFrame:
    frames = []
    for batch_dir, dataset, frozen in iter_batches(settings, live_only=False):
        if frozen and not include_frozen:
            continue
        part = fly_trial_metrics_for_batch(batch_dir, dataset, frozen, gcfg, settings)
        if len(part):
            frames.append(part)
    if not frames:
        return pd.DataFrame(columns=ID_COLUMNS + METRIC_COLUMNS)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
def _read_flagged(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    state_col = next((c for c in raw.columns if "fly-state" in str(c).lower() or "fly_state" in str(c).lower()), None)
    if state_col is None:
        return pd.DataFrame(columns=["dataset", "fly", "fly_number", "fly_state"])
    out = raw[["dataset", "fly", "fly_number", state_col]].copy()
    out.columns = ["dataset", "fly", "fly_number", "fly_state"]
    out["fly_number"] = pd.to_numeric(out["fly_number"], errors="coerce").astype("Int64")
    out["fly_state"] = pd.to_numeric(out["fly_state"], errors="coerce")
    return out.dropna(subset=["fly_number", "fly_state"]).drop_duplicates(["dataset", "fly", "fly_number"], keep="last")


def attach_labels(df: pd.DataFrame, *, predictions_csv: Optional[Path] = None, blinded_csv: Optional[Path] = None,
                  flagged_csv: Optional[Path] = None) -> pd.DataFrame:
    out = df.copy()
    out["fly_number"] = out["fly_number"].astype(int)
    out["model_score"], out["model_prediction"] = np.nan, np.nan
    out["user_score"], out["comment"], out["abdomen_comment"] = np.nan, None, False
    out["fly_state"], out["fly_status"] = 1, "alive"
    if predictions_csv is not None and Path(predictions_csv).is_file():
        p = pd.read_csv(predictions_csv)
        p = p.assign(fly_number=pd.to_numeric(p["fly_number"], errors="coerce"),
                     trial_label_lc=p["trial_label"].astype(str).str.casefold())
        p = p.dropna(subset=["fly_number"]).drop_duplicates(["dataset", "fly", "fly_number", "trial_label_lc"], keep="last")
        p["fly_number"] = p["fly_number"].astype(int)
        keep = [c for c in ("score", "prediction") if c in p.columns]
        merged = out.merge(p[["dataset", "fly", "fly_number", "trial_label_lc"] + keep], how="left",
                           on=["dataset", "fly", "fly_number", "trial_label_lc"], suffixes=("", "_pred"))
        if "score" in keep:
            out["model_score"] = merged["score"].to_numpy()
        if "prediction" in keep:
            out["model_prediction"] = merged["prediction"].to_numpy()
    if blinded_csv is not None and Path(blinded_csv).is_file():
        b = pd.read_csv(blinded_csv)
        parsed = b["trial_label"].astype(str).str.extract(_TRIAL_RE.pattern, flags=re.IGNORECASE)
        b = b.assign(trial_type=parsed[0].str.lower(), trial_index=pd.to_numeric(parsed[1], errors="coerce"),
                     fly_number=pd.to_numeric(b["fly_number"], errors="coerce"))
        b = b.dropna(subset=["trial_type", "trial_index", "fly_number"])
        b["trial_index"] = b["trial_index"].astype(int)
        b["fly_number"] = b["fly_number"].astype(int)
        b = b.drop_duplicates(["dataset", "fly", "fly_number", "trial_type", "trial_index"], keep="last")
        merged = out.merge(b[["dataset", "fly", "fly_number", "trial_type", "trial_index", "user_score", "comment"]],
                           how="left", on=["dataset", "fly", "fly_number", "trial_type", "trial_index"], suffixes=("", "_blind"))
        out["user_score"] = pd.to_numeric(merged["user_score_blind"] if "user_score_blind" in merged else merged["user_score"], errors="coerce").to_numpy()
        comments = merged["comment_blind"] if "comment_blind" in merged else merged["comment"]
        out["comment"] = comments.to_numpy()
        out["abdomen_comment"] = comments.astype(str).str.contains("abdo", case=False, na=False).to_numpy()
    if flagged_csv is not None and Path(flagged_csv).is_file():
        f = _read_flagged(flagged_csv)
        f["fly_number"] = f["fly_number"].astype(int)
        merged = out.merge(f, how="left", on=["dataset", "fly", "fly_number"], suffixes=("", "_flag"))
        state = merged["fly_state_flag"] if "fly_state_flag" in merged else merged["fly_state"]
        out["fly_state"] = state.fillna(1).astype(int).to_numpy()
        out["fly_status"] = [FLY_STATUS.get(int(s), "alive") for s in out["fly_state"]]
    out["abdomen_comment"] = out["abdomen_comment"].astype(bool)
    return out


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def bh_adjust(p: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaN stays NaN)."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    vals = p[ok]
    n = vals.size
    order = np.argsort(vals)
    ranked = vals[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


def auc_from_groups(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Probability that a random positive exceeds a random negative (ties count 1/2)."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not pos.size or not neg.size:
        return float("nan")
    u = sps.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (pos.size * neg.size))


def spearman_table(df: pd.DataFrame, x: str, y: str, group_cols: Sequence[str] = (), min_n: int = 8) -> pd.DataFrame:
    group_cols = list(group_cols)
    rows = []
    groups = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    for key, sub in groups:
        rho, p, n = _spearman(sub[x].to_numpy(float), sub[y].to_numpy(float))
        if n < min_n:
            continue
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(group_cols, key)), "x": x, "y": y, "n": n, "rho": rho, "p": p})
    out = pd.DataFrame(rows, columns=group_cols + ["x", "y", "n", "rho", "p"])
    out["p_bh"] = bh_adjust(out["p"].to_numpy()) if len(out) else np.array([], dtype=float)
    return out


def status_comparison(df: pd.DataFrame, metrics: Sequence[str] = ("per_delta", "per_win_max", *ANGLE_METRICS),
                      min_n: int = 5) -> pd.DataFrame:
    rows = []
    if "fly_status" not in df.columns:
        return pd.DataFrame(columns=["metric", "fly_status", "n_trials", "n_flies", "median", "mean", "kruskal_p"])
    for metric in metrics:
        groups = {s: sub[metric].to_numpy(float) for s, sub in df.groupby("fly_status")}
        groups = {s: v[np.isfinite(v)] for s, v in groups.items()}
        big = [v for v in groups.values() if v.size >= min_n]
        kp = float(sps.kruskal(*big).pvalue) if len(big) >= 2 else float("nan")
        for s, v in groups.items():
            sub = df[df["fly_status"] == s]
            rows.append({"metric": metric, "fly_status": s, "n_trials": int(v.size),
                         "n_flies": int(sub[["dataset", "fly", "fly_number"]].drop_duplicates().shape[0]),
                         "median": float(np.median(v)) if v.size else np.nan,
                         "mean": float(np.mean(v)) if v.size else np.nan, "kruskal_p": kp})
    return pd.DataFrame(rows)


def blinded_comment_check(df: pd.DataFrame, metrics: Sequence[str] = ANGLE_METRICS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Trials the blinded scorer annotated with 'abdomen' vs the other scored trials."""
    scored = df[df["user_score"].notna()] if "user_score" in df.columns else df.iloc[0:0]
    commented = scored[scored["abdomen_comment"]]
    others = scored[~scored["abdomen_comment"]]
    rows = []
    for m in metrics:
        pos, neg = commented[m].to_numpy(float), others[m].to_numpy(float)
        pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
        p = float(sps.mannwhitneyu(pos, neg, alternative="two-sided").pvalue) if pos.size and neg.size else np.nan
        rows.append({"metric": m, "n_commented": int(pos.size), "n_other_scored": int(neg.size),
                     "median_commented": float(np.median(pos)) if pos.size else np.nan,
                     "median_other": float(np.median(neg)) if neg.size else np.nan,
                     "auc_commented_vs_other": auc_from_groups(pos, neg), "mannwhitney_p": p})
    cols = ["dataset", "fly", "fly_number", "trial_label", "user_score", "comment", "model_score",
            "per_delta", "per_win_max", *metrics, "angle_tracked_frac", "dominant_status"]
    table = commented[[c for c in cols if c in commented.columns]].sort_values(["dataset", "fly", "fly_number"])
    return table.reset_index(drop=True), pd.DataFrame(rows)


def odor_summary(df: pd.DataFrame, min_n: int = 8) -> pd.DataFrame:
    sub = df[df["has_odor"] & df["per_delta"].notna()]
    rows = []
    for odor, g in sub.groupby("odor_norm"):
        if len(g) < min_n:
            continue
        rho, p, n = _spearman(g["per_delta"].to_numpy(float), g["angle_delta"].to_numpy(float))
        pd_, ad = g["per_delta"].to_numpy(float), g["angle_delta"].to_numpy(float)
        both = np.isfinite(pd_) & np.isfinite(ad)
        rows.append({"odor": odor, "n_trials": int(len(g)), "n_flies": int(g[["dataset", "fly", "fly_number"]].drop_duplicates().shape[0]),
                     "per_delta_mean": float(np.nanmean(pd_)), "per_delta_sem": float(np.nanstd(pd_, ddof=1) / math.sqrt(max(1, np.isfinite(pd_).sum()))),
                     "angle_delta_mean": float(np.nanmean(ad)) if both.any() else np.nan,
                     "angle_delta_sem": float(np.nanstd(ad, ddof=1) / math.sqrt(max(1, np.isfinite(ad).sum()))) if both.any() else np.nan,
                     "frac_per_response": float(np.nanmean(pd_ >= 20.0)), "frac_angle_response": float(np.nanmean(ad[both] >= 20.0)) if both.any() else np.nan,
                     "frac_both": float(np.mean((pd_[both] >= 20.0) & (ad[both] >= 20.0))) if both.any() else np.nan,
                     "rho_per_vs_angle": rho, "p": p, "n_both": n})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_bh"] = bh_adjust(out["p"].to_numpy())
    return out.sort_values("n_trials", ascending=False) if len(out) else out


def box_vs_angle(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in ("rho_length_angle", "rho_width_angle"):
        v = df[name].to_numpy(float)
        v = v[np.isfinite(v)]
        rows.append({"per_trial_rho": name, "n_trials": int(v.size), "median_rho": float(np.median(v)) if v.size else np.nan,
                     "q25": float(np.percentile(v, 25)) if v.size else np.nan, "q75": float(np.percentile(v, 75)) if v.size else np.nan,
                     "frac_negative": float(np.mean(v < 0)) if v.size else np.nan, "frac_below_-0.5": float(np.mean(v < -0.5)) if v.size else np.nan,
                     "frac_above_0.5": float(np.mean(v > 0.5)) if v.size else np.nan})
    for xcol, ycol in (("length_delta", "angle_delta"), ("width_delta", "angle_delta"), ("length_delta", "per_delta")):
        rho, p, n = _spearman(df[xcol].to_numpy(float), df[ycol].to_numpy(float))
        rows.append({"per_trial_rho": f"across trials: {xcol} vs {ycol}", "n_trials": n, "median_rho": rho, "q25": np.nan, "q75": np.nan,
                     "frac_negative": np.nan, "frac_below_-0.5": np.nan, "frac_above_0.5": p})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df is None or not len(df):
        return "_(no rows)_\n"
    d = df.head(max_rows).copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else (f"{v:.3g}" if abs(v) < 0.01 else f"{v:.3f}"))
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    if len(df) > max_rows:
        lines.append(f"\n_({len(df) - max_rows} more rows in the CSV)_")
    return "\n".join(lines) + "\n"


def _figures(df: pd.DataFrame, out_dir: Path, corr_odor: pd.DataFrame, blinded_summary: pd.DataFrame) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..plot_style import apply_lab_style
    from .figures import PER_COLOR, ANGLE_COLOR, TEXT_SECONDARY
    apply_lab_style()
    figs: List[Path] = []
    odor_trials = df[df["has_odor"]]
    x, y = odor_trials["per_delta"].to_numpy(float), odor_trials["angle_delta"].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() >= 3:
        rho, p, n = _spearman(x, y)
        fig, ax = plt.subplots(figsize=(6, 5))
        hb = ax.hexbin(x[m], y[m], gridsize=35, cmap="Blues", mincnt=1, linewidths=0.2)
        fig.colorbar(hb, ax=ax, label="fly-trials")
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8, ls="--"); ax.axvline(0, color=TEXT_SECONDARY, lw=0.8, ls="--")
        ax.set_xlabel("PER: odor-window mean minus pre-odor mean (% of fly range)")
        ax.set_ylabel("Abdomen angle: odor-window mean minus pre-odor mean (%)")
        ax.set_title(f"PER vs abdomen angle per odor trial (Spearman rho {rho:.2f}, p {p:.1e}, n={n})", fontsize=10, loc="left")
        out = out_dir / "per_vs_angle_scatter.png"; fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig); figs.append(out)
    if len(corr_odor):
        oc = corr_odor.sort_values("n_trials", ascending=False)
        fig, axes = plt.subplots(1, 2, figsize=(11, 0.35 * len(oc) + 2.2), sharey=True)
        ypos = np.arange(len(oc))
        axes[0].barh(ypos, oc["per_delta_mean"], xerr=oc["per_delta_sem"], color=PER_COLOR, height=0.6)
        axes[0].set_title("PER delta, mean +/- SEM over trials", fontsize=9, loc="left")
        axes[1].barh(ypos, oc["angle_delta_mean"], xerr=oc["angle_delta_sem"], color=ANGLE_COLOR, height=0.6)
        axes[1].set_title("abdomen angle delta, mean +/- SEM over trials", fontsize=9, loc="left")
        for ax in axes:
            ax.axvline(0, color=TEXT_SECONDARY, lw=0.8); ax.set_yticks(ypos)
        axes[0].set_yticklabels([f"{o} (n={n})" for o, n in zip(oc["odor"], oc["n_trials"])], fontsize=8)
        for yy, (rho, n) in enumerate(zip(oc["rho_per_vs_angle"], oc["n_both"])):
            axes[1].text(1.01, yy, f"rho {rho:.2f}" if np.isfinite(rho) else "", transform=axes[1].get_yaxis_transform(), fontsize=7, va="center", color=TEXT_SECONDARY)
        axes[0].invert_yaxis()
        out = out_dir / "odor_deltas.png"; fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig); figs.append(out)
    if "fly_status" in df.columns and df["fly_status"].nunique() > 1:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, metric, color in ((axes[0], "per_delta", PER_COLOR), (axes[1], "angle_activity_win", ANGLE_COLOR)):
            groups = [(s, df.loc[df["fly_status"] == s, metric].dropna().to_numpy()) for s in ("alive", "weak", "dead")]
            groups = [(s, v) for s, v in groups if v.size]
            ax.boxplot([v for _, v in groups], tick_labels=[f"{s}\n(n={v.size})" for s, v in groups], showfliers=False,
                       medianprops={"color": color, "lw": 2})
            ax.set_title(metric.replace("_", " "), fontsize=9, loc="left")
        out = out_dir / "status_comparison.png"; fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig); figs.append(out)
    rl, rw = df["rho_length_angle"].dropna().to_numpy(), df["rho_width_angle"].dropna().to_numpy()
    if rl.size or rw.size:
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.linspace(-1, 1, 41)
        if rl.size:
            ax.hist(rl, bins=bins, alpha=0.7, color=PER_COLOR, label=f"box length vs angle (n={rl.size}, median {np.median(rl):.2f})")
        if rw.size:
            ax.hist(rw, bins=bins, alpha=0.7, color=ANGLE_COLOR, label=f"box width vs angle (n={rw.size}, median {np.median(rw):.2f})")
        ax.axvline(0, color=TEXT_SECONDARY, lw=0.8); ax.set_xlabel("per-trial Spearman rho (frames)"); ax.set_ylabel("fly-trials")
        ax.legend(fontsize=8, frameon=False); ax.set_title("Abdomen box size vs abdomen angle %, within trials", fontsize=10, loc="left")
        out = out_dir / "box_vs_angle_rho.png"; fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig); figs.append(out)
    scored = df[df["user_score"].notna()] if "user_score" in df.columns else df.iloc[0:0]
    if len(scored) and scored["abdomen_comment"].any():
        fig, ax = plt.subplots(figsize=(7, 4))
        rng = np.random.default_rng(0)
        for flag, color, label in ((False, PER_COLOR, "scored, no abdomen comment"), (True, ANGLE_COLOR, "scored with 'abdomen' comment")):
            sub = scored[scored["abdomen_comment"] == flag]
            ax.scatter(sub["user_score"] + rng.uniform(-0.18, 0.18, len(sub)), sub["angle_delta"], s=14, alpha=0.6, color=color, label=f"{label} (n={len(sub)})")
        ax.set_xlabel("blinded human score"); ax.set_ylabel("abdomen angle delta (%)"); ax.legend(fontsize=8, frameon=False)
        ax.set_title("Human 'abdomen' comments vs tracked abdomen movement", fontsize=10, loc="left")
        out = out_dir / "blinded_comments.png"; fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig); figs.append(out)
    return figs


def run_analysis(df: pd.DataFrame, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df.to_csv(out_dir / "fly_trial_metrics.csv", index=False)
    try:
        df.to_parquet(out_dir / "fly_trial_metrics.parquet", index=False)
    except Exception:
        pass
    odor_trials = df[df["has_odor"]]
    corr = pd.concat([
        spearman_table(odor_trials, "per_delta", "angle_delta").assign(scope="all odor trials"),
        spearman_table(odor_trials, "per_win_max", "angle_win_max").assign(scope="all odor trials"),
        spearman_table(odor_trials, "per_delta", "angle_activity_win").assign(scope="all odor trials"),
        spearman_table(odor_trials, "per_delta", "angle_delta", group_cols=("trial_type",)).assign(scope="by trial type"),
        spearman_table(odor_trials, "per_delta", "angle_delta", group_cols=("odor_norm",)).assign(scope="by odor"),
        spearman_table(odor_trials, "per_delta", "angle_delta", group_cols=("dataset",)).assign(scope="by dataset"),
        spearman_table(odor_trials, "per_delta", "angle_delta", group_cols=("fly_status",) if "fly_status" in df.columns else ()).assign(scope="by fly status"),
        spearman_table(odor_trials, "per_delta", "angle_delta", group_cols=("dataset", "fly", "fly_number"), min_n=10).assign(scope="by fly"),
    ], ignore_index=True)
    corr.to_csv(out_dir / "correlations.csv", index=False)
    status = status_comparison(df)
    status.to_csv(out_dir / "status_comparison.csv", index=False)
    commented, blinded_summary = blinded_comment_check(df)
    commented.to_csv(out_dir / "blinded_abdomen_comments.csv", index=False)
    blinded_summary.to_csv(out_dir / "blinded_summary.csv", index=False)
    box = box_vs_angle(df)
    box.to_csv(out_dir / "box_vs_angle.csv", index=False)
    odors = odor_summary(df)
    odors.to_csv(out_dir / "odor_summary.csv", index=False)
    figs = _figures(df, out_dir, odors, blinded_summary)

    by_fly = corr[corr["scope"] == "by fly"]
    pooled = corr[(corr["scope"] == "all odor trials") & (corr["x"] == "per_delta") & (corr["y"] == "angle_delta")]
    n_flies = int(df[["dataset", "fly", "fly_number"]].drop_duplicates().shape[0])
    lines = [
        "# Abdomen geometry statistics", "",
        f"Fly-trials: {len(df)} (odor trials {int(df['has_odor'].sum())}) from {n_flies} flies in {df['dataset'].nunique()} datasets. "
        "Metrics: `*_delta` = stimulus-window mean minus pre-stimulus mean (% of each fly's own range), `*_win_max` = window maximum, "
        "`angle_activity_win` = SD of abdomen angle % inside the window. n always counts fly-trials; nothing averages per-fly averages.", "",
        "## 1. Does PER correlate with abdomen movement?", "",
        "Spearman correlations (BH-adjusted p across each scope):", "", _md_table(corr[corr["scope"] != "by fly"].drop(columns=["scope"]).assign(scope=corr[corr["scope"] != "by fly"]["scope"]) if len(corr) else corr),
    ]
    if len(by_fly):
        lines += ["", f"Per fly (n >= 10 odor trials): {len(by_fly)} flies; rho median {by_fly['rho'].median():.2f}, "
                  f"{(by_fly['rho'] > 0.5).mean():.0%} above +0.5, {(by_fly['rho'] < -0.5).mean():.0%} below -0.5. Strongest positive / negative:", "",
                  _md_table(by_fly.sort_values("rho", ascending=False).head(8).drop(columns=["scope", "x", "y"])),
                  _md_table(by_fly.sort_values("rho").head(8).drop(columns=["scope", "x", "y"]))]
    lines += ["", "## 2. By odor", "", _md_table(odors), "",
              "## 3. Alive / weak / dead flies (flagged-flys-truth: 0 = weak, -1 = dead, absent = alive)", "", _md_table(status), "",
              "## 4. Blinded scoring comments mentioning the abdomen vs the tracked abdomen", "",
              _md_table(blinded_summary), "", "Each commented trial with its tracked metrics:", "", _md_table(commented, max_rows=60), "",
              "## 5. Abdomen box size vs abdomen angle", "",
              "Within-trial Spearman rho over frames (box length % / width % of rest vs angle %; negative = the box gets shorter as the fly curls), "
              "then across-trial correlations of the window deltas (the last column of those rows is the p-value):", "", _md_table(box), "",
              "## Figures", ""] + [f"- `{f.name}`" for f in figs]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"n_fly_trials": int(len(df)), "n_flies": n_flies, "figures": figs, "out_dir": out_dir,
            "pooled_rho": float(pooled["rho"].iloc[0]) if len(pooled) else float("nan")}
