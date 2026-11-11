#!/usr/bin/env python3
"""Compare the anchor-eye and live-eye arms built by ``live_eye_shadow_run.py``.

Writes ``Results/Figures/LiveEye-Comparison/Comparison/``: trial-level score
changes, trace-level metric changes, the sensitivity-results tables side by
side, and a few figures. Fly identity across arms is decided by eye position
(``match_flies_by_eye``), never by fly number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
COLE = Path("/home/ramanlab/Documents/cole")
DATA = COLE / "Data"
ARMS = {"anchor": DATA / "flys_AnchorEye", "live": DATA / "flys_LiveEye"}
CSV_ROOT = DATA / "CSVs-LiveEye"
RESULTS_ROOT = COLE / "Results" / "Figures" / "LiveEye-Comparison"
OUT = RESULTS_ROOT / "Comparison"
EYE_QC = COLE / "Results" / "New-Opto-Fly-Figures" / "Eye-Tracking-QC" / "eye_drift_qc.csv"
KEY = ["dataset", "fly", "trial_label"]
MATCH_PX = 60.0


def match_flies_by_eye(anchor: dict, live: dict, max_px: float = MATCH_PX) -> dict:
    """Map each live fly number to the anchor fly at the same eye (or None)."""
    pairs = sorted(
        (float(np.hypot(lx - ax, ly - ay)), ln, an)
        for ln, (lx, ly) in live.items()
        for an, (ax, ay) in anchor.items()
    )
    out, used = {ln: None for ln in live}, set()
    for dist, ln, an in pairs:
        if dist > max_px or out[ln] is not None or an in used:
            continue
        out[ln] = an
        used.add(an)
    return out


def _batch_eyes(arm_root: Path, dataset: str, batch: str) -> dict:
    """Fly number -> frozen eye anchor, read from one trial table per fly."""
    eyes = {}
    for path in sorted((arm_root / dataset / batch).glob("*/*_fly[0-9]_distances.parquet")):
        if path.parent.name in ("RMS_calculations", "angle_distance_rms_envelope"):
            continue
        n = int(path.stem.split("_fly")[-1].split("_")[0])
        if n in eyes:
            continue
        cols = pd.read_parquet(path).columns
        xcol, ycol = ("x_class0_anchor", "y_class0_anchor") if "x_class0_anchor" in cols else ("x_class0", "y_class0")
        df = pd.read_parquet(path, columns=[xcol, ycol]).dropna()
        if len(df):
            eyes[n] = (float(df[xcol].median()), float(df[ycol].median()))
    return eyes


def _predictions(arm: str) -> pd.DataFrame:
    frames = []
    for name in ("model_predictions.csv", "model_predictions_training.csv"):
        path = CSV_ROOT / arm / name
        if path.exists():
            frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    # The training sidecar carries no trial_type column; its labels say it.
    label_type = df["trial_label"].astype(str).str.split("_").str[0]
    df["trial_type"] = df.get("trial_type", pd.Series(np.nan, index=df.index)).fillna(label_type)
    return df[["dataset", "fly", "fly_number", "trial_label", "trial_type", "prediction", "score"]]


def _wide(arm: str) -> pd.DataFrame:
    stem = "all_envelope_rows_wide_combined_base"
    cols = ["dataset", "fly", "fly_number", "trial_label", "trial_type", "Peak-Value",
            "AUC-During", "AUC-Before", "global_min", "global_max", "tracking_pct_missing"]
    frames = []
    for suffix in ("", "_pretest", "_training"):
        path = CSV_ROOT / arm / f"{stem}{suffix}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            frames.append(df[[c for c in cols if c in df.columns]])
    return pd.concat(frames, ignore_index=True)


def _number_map(anchor_df: pd.DataFrame) -> pd.DataFrame:
    """(dataset, fly, live fly_number) -> anchor fly_number, by eye position."""
    rows = []
    for (ds, batch), _ in anchor_df.groupby(["dataset", "fly"]):
        a = _batch_eyes(ARMS["anchor"], ds, batch)
        live = _batch_eyes(ARMS["live"], ds, batch)
        for ln, an in match_flies_by_eye(a, live).items():
            rows.append({"dataset": ds, "fly": batch, "fly_number_live": ln, "fly_number": an,
                         "eye_x": live[ln][0], "eye_y": live[ln][1]})
    return pd.DataFrame(rows)


def _join(anchor: pd.DataFrame, live: pd.DataFrame, numbers: pd.DataFrame) -> pd.DataFrame:
    live = live.rename(columns={"fly_number": "fly_number_live"}).merge(
        numbers[["dataset", "fly", "fly_number_live", "fly_number"]],
        on=["dataset", "fly", "fly_number_live"], how="left",
    )
    return anchor.merge(live, on=[*KEY, "fly_number"], how="outer", suffixes=("_anchor", "_live"))


def _eye_qc() -> pd.DataFrame:
    if not EYE_QC.exists():
        return pd.DataFrame()
    qc = pd.read_csv(EYE_QC)
    qc["fly"] = qc["trial_dir"].map(lambda p: Path(str(p)).parent.name)
    qc["trial_label"] = qc["trial_dir"].map(lambda p: Path(str(p)).name)
    return qc[["dataset", "fly", "fly_number", "trial_label", "eye_qc_flag", "eye_qc_reason",
               "drift_p95_px", "drift_max_px", "eye_missing_frac"]]


def main() -> None:
    sys.path[:0] = [str(REPO), str(REPO / "src")]
    OUT.mkdir(parents=True, exist_ok=True)
    pa, pl = _predictions("anchor"), _predictions("live")
    numbers = _number_map(pa)
    numbers.to_csv(OUT / "fly_identity_by_eye.csv", index=False)

    scores = _join(pa, pl, numbers)
    for col in ("trial_type",):
        scores[col] = scores[f"{col}_anchor"].fillna(scores[f"{col}_live"])
    scores["score_change"] = scores["score_live"] - scores["score_anchor"]
    scores["responder_flip"] = np.select(
        [(scores.prediction_anchor == 0) & (scores.prediction_live == 1),
         (scores.prediction_anchor == 1) & (scores.prediction_live == 0)],
        ["became responder", "stopped responding"], default="",
    )
    qc = _eye_qc()
    if len(qc):
        qc["trial_key"] = qc["trial_label"].str.split("_").str[-2:].str.join("_")
        scores["trial_key"] = scores["trial_label"].str.split("_").str[:2].str.join("_")
        scores = scores.merge(qc.drop(columns="trial_label"), left_on=["dataset", "fly", "fly_number", "trial_key"],
                              right_on=["dataset", "fly", "fly_number", "trial_key"], how="left")
    scores.to_csv(OUT / "trial_scores_anchor_vs_live.csv", index=False)

    both = scores.dropna(subset=["score_anchor", "score_live"])
    summ = (
        both.groupby(["dataset", "trial_type"])
        .apply(lambda g: pd.Series({
            "n_trials": len(g),
            "pct_same_score": 100 * (g.score_change == 0).mean(),
            "n_score_up": int((g.score_change > 0).sum()),
            "n_score_down": int((g.score_change < 0).sum()),
            "n_change_ge2": int((g.score_change.abs() >= 2).sum()),
            "n_became_responder": int((g.responder_flip == "became responder").sum()),
            "n_stopped_responding": int((g.responder_flip == "stopped responding").sum()),
            "pct_responding_anchor": 100 * g.prediction_anchor.mean(),
            "pct_responding_live": 100 * g.prediction_live.mean(),
        }), include_groups=False)
        .reset_index()
    )
    summ.to_csv(OUT / "score_change_summary.csv", index=False)

    wa, wl = _wide("anchor"), _wide("live")
    traces = _join(wa, wl, numbers)
    for m in ("Peak-Value", "AUC-During", "global_max", "global_min"):
        traces[f"{m}_change"] = traces[f"{m}_live"] - traces[f"{m}_anchor"]
    traces.to_csv(OUT / "trial_trace_metrics_anchor_vs_live.csv", index=False)

    for name in ("window_summary_per_odor.csv", "whole-cohort_summary_per_odor.csv"):
        a = pd.read_csv(RESULTS_ROOT / "anchor" / "Sensitivity-Results" / "Data" / name)
        l = pd.read_csv(RESULTS_ROOT / "live" / "Sensitivity-Results" / "Data" / name)
        side = a.merge(l, on=["cohort", "odor", "sensitized_odor"], how="outer", suffixes=("_anchor", "_live"))
        side.to_csv(OUT / f"sensitivity_{name}", index=False)

    _figures(scores, numbers)
    print(summ.to_string(index=False))


# Validated pair (dataviz validator, light surface): blue = anchor, orange = live.
ARM_COLORS = {"anchor": "#2a78d6", "live": "#eb6834"}
ARM_NAMES = {"anchor": "fixed eye anchor (production)", "live": "per-frame eye"}


def _figures(scores: pd.DataFrame, numbers: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    both = scores.dropna(subset=["score_anchor", "score_live"])

    # 1. score transitions, one panel per trial type
    types = [t for t in ("pretest", "testing", "training") if t in set(both.trial_type)]
    levels = sorted(set(both.score_anchor.astype(int)) | set(both.score_live.astype(int)))
    fig, axes = plt.subplots(1, len(types), figsize=(4.2 * len(types), 4.2), constrained_layout=True)
    for ax, t in zip(np.atleast_1d(axes), types):
        g = both[both.trial_type == t]
        m = pd.crosstab(g.score_anchor.astype(int), g.score_live.astype(int)).reindex(
            index=levels, columns=levels, fill_value=0)
        ax.imshow(np.log1p(m.values), cmap="Blues", origin="lower")
        for i, a in enumerate(levels):
            for j, b in enumerate(levels):
                v = m.values[i, j]
                if v:
                    ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                            color="white" if np.log1p(v) > 0.6 * np.log1p(m.values.max()) else "#222")
        ax.set_xticks(range(len(levels)), levels)
        ax.set_yticks(range(len(levels)), levels)
        ax.set_xlabel("score, per-frame eye")
        ax.set_ylabel("score, fixed anchor (production)")
        same = 100 * (g.score_anchor == g.score_live).mean()
        ax.set_title(f"{t}: {same:.1f}% unchanged\n(n={len(g)} trials)")
    fig.suptitle("Model score per trial: fixed eye anchor vs per-frame eye (diagonal = unchanged)")
    fig.savefig(OUT / "01_score_transitions.png", dpi=200)
    plt.close(fig)

    # 2. % responding per cohort x odor, pre and post, anchor vs live
    side = pd.read_csv(OUT / "sensitivity_window_summary_per_odor.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    for ax, phase in zip(axes, ("pre", "post")):
        x, y = side[f"pct_responding_{phase}_anchor"], side[f"pct_responding_{phase}_live"]
        ax.plot([0, 100], [0, 100], color="#9a9a96", lw=1, zorder=0)
        sens = side.sensitized_odor.astype(bool)
        ax.scatter(x[~sens], y[~sens], s=36, color=ARM_COLORS["anchor"], edgecolor="white", lw=1.5,
                   label="other odors")
        ax.scatter(x[sens], y[sens], s=64, marker="D", color=ARM_COLORS["live"], edgecolor="white", lw=1.5,
                   label="sensitized odor")
        for _, r in side[sens].iterrows():
            ax.annotate(r.cohort.split("-Sensitivity")[0], (r[f"pct_responding_{phase}_anchor"],
                        r[f"pct_responding_{phase}_live"]), xytext=(5, -10), textcoords="offset points",
                        fontsize=8, color="#444")
        lim = max(5.0, float(np.nanmax([x.max(), y.max()])) + 5)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("% responding, fixed anchor (production)")
        ax.set_ylabel("% responding, per-frame eye")
        phase_name = "pre-exposure" if phase == "pre" else "post-exposure"
        ax.set_title(f"{phase_name} (window flies)")
        ax.legend(frameon=False, loc="upper left")
    fig.suptitle("% of flies responding per cohort × odor; on the line = no change")
    fig.savefig(OUT / "02_pct_responding_anchor_vs_live.png", dpi=200)
    plt.close(fig)

    # 3. traces for the largest score changes
    changed = both.assign(absd=both.score_change.abs()).query("absd > 0")
    changed = changed.sort_values(["absd", "dataset"], ascending=[False, True]).head(6)
    if len(changed):
        wide = {arm: _wide_traces(arm, changed) for arm in ("anchor", "live")}
        fig, axes = plt.subplots(len(changed), 1, figsize=(9, 1.9 * len(changed)), sharex=True,
                                 constrained_layout=True)
        for ax, (_, r) in zip(np.atleast_1d(axes), changed.iterrows()):
            for arm in ("anchor", "live"):
                num = r.fly_number if arm == "anchor" else r.fly_number_live
                tr = wide[arm].get((r.dataset, r.fly, int(num), r.trial_label))
                if tr is not None:
                    t = np.arange(len(tr)) / 40.0
                    ax.plot(t, tr, lw=1.6, color=ARM_COLORS[arm],
                            label=f"{ARM_NAMES[arm]}: score {int(r[f'score_{arm}'])}")
            ax.axvspan(30, 60, color="#e9e7e1", zorder=0)
            ax.set_title(f"{r.dataset} · {r.fly} fly {int(r.fly_number)} · {r.trial_label}", fontsize=9, loc="left")
            ax.legend(frameon=False, fontsize=8, loc="upper right")
        np.atleast_1d(axes)[-1].set_xlabel("time (s); shaded = odor on")
        fig.suptitle("Largest score changes: pipeline trace, fixed anchor vs per-frame eye")
        fig.savefig(OUT / "03_largest_changes_traces.png", dpi=200)
        plt.close(fig)


def _wide_traces(arm: str, rows: pd.DataFrame) -> dict:
    stem = "all_envelope_rows_wide_combined_base"
    out = {}
    for suffix in ("", "_pretest", "_training"):
        path = CSV_ROOT / arm / f"{stem}{suffix}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        vals = [c for c in df.columns if c.startswith("dir_val_")]
        for _, w in df[df.fly.isin(rows.fly) & df.trial_label.isin(rows.trial_label)].iterrows():
            tr = w[vals].to_numpy(dtype=float)
            out[(w.dataset, w.fly, int(w.fly_number), w.trial_label)] = tr[~np.isnan(tr)] if np.isfinite(tr).any() else tr
    return out


if __name__ == "__main__":
    main()
