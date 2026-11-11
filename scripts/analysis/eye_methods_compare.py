#!/usr/bin/env python3
"""Every eye-position method side by side against production (the anchor arm).

Methods (arms built by ``live_eye_shadow_run.py``):
* ``live``     -- the eye detected in every frame, all trial long (2026-11-09);
* ``livelight`` -- the eye detected in every frame, except while the stimulus
  light is on (YOLO misplaces the eye under it): then it stays where it sat
  over the second before;
* ``livehold`` -- the same, held during the odor as well as the light;
* ``drift``    -- static anchor, moved only when the eye moves and stays.

Scores include the hand-flagged flies (``*_allflies.csv``). The Nov 7 batch is
left out (its re-run used brightened video; it is flagged bad-pin-out), so a
fly's slot number names the same fly in every arm.

Writes ``Results/Figures/LiveEye-Comparison/All-Methods/``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "analysis" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dec = _load("drift_eye_compare")
lec = dec.lec
OUT = lec.RESULTS_ROOT / "All-Methods"
METHODS = {
    "live": "every frame, all trial",
    "livelight": "every frame, held while light on",
    "livehold": "every frame, held during odor/light",
    "drift": "drift-following anchor",
}
KEY = ["dataset", "fly", "fly_number", "trial_label"]
EXCLUDE = ("3Oct-Sensitivity-1-0.1", "november_07_batch_1")


def _scores(arm: str) -> pd.DataFrame:
    df = dec._predictions_all(arm)
    df = df[~((df.dataset == EXCLUDE[0]) & (df.fly == EXCLUDE[1]))]
    return df[df.dataset.isin(dec.DATASETS)]


def _paired(method: str, qc: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    a, m = _scores("anchor"), _scores(method)
    s = a.merge(m[KEY + ["prediction", "score"]], on=KEY, suffixes=("_anchor", "_new"))
    s["method"] = method
    s["score_change"] = s.score_new - s.score_anchor
    s["flip"] = s.prediction_anchor != s.prediction_new
    s["trial_key"] = s.trial_label.str.split("_").str[:2].str.join("_")
    s = s.merge(qc[["dataset", "fly", "fly_number", "trial_key", "qc_group"]],
                on=["dataset", "fly", "fly_number", "trial_key"], how="left")
    s["qc_group"] = s.qc_group.fillna("no eye-QC row")
    s = s.merge(flags, on=["dataset", "fly", "fly_number"], how="left")
    s["hand_flag_group"] = s.hand_flag.map(dec._hand_group)
    return s


def _summ(s: pd.DataFrame, by: list) -> pd.DataFrame:
    def f(g):
        return pd.Series({
            "n_trials": len(g),
            "pct_score_changed": round(100 * (g.score_change != 0).mean(), 1),
            "pct_changed_by_2plus": round(100 * (g.score_change.abs() >= 2).mean(), 1),
            "n_responder_flips": int(g.flip.sum()),
            "pct_responding_production": round(100 * g.prediction_anchor.mean(), 1),
            "pct_responding_method": round(100 * g.prediction_new.mean(), 1),
        })
    return s.groupby(by).apply(f, include_groups=False).reset_index()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    qc = dec._qc()
    flags = dec._hand_flags()
    paired = pd.concat([_paired(m, qc, flags) for m in METHODS], ignore_index=True)
    paired["method_name"] = paired.method.map(METHODS)
    paired.to_csv(OUT / "trial_scores_all_methods.csv", index=False)

    tables = {
        "by_method": _summ(paired, ["method_name"]),
        "by_method_and_trial_type": _summ(paired, ["method_name", "trial_type"]),
        "by_method_and_eye_qc": _summ(paired, ["method_name", "qc_group"]),
        "by_method_and_hand_flag": _summ(paired, ["method_name", "hand_flag_group"]),
        "by_method_and_dataset": _summ(paired, ["method_name", "dataset"]),
    }
    for name, t in tables.items():
        t.to_csv(OUT / f"score_change_{name}.csv", index=False)

    for name in ("window_summary_per_odor.csv", "whole-cohort_summary_per_odor.csv"):
        parts = []
        for arm in ("anchor", *METHODS):
            p = lec.RESULTS_ROOT / arm / "Sensitivity-Results" / "Data" / name
            parts.append(pd.read_csv(p).set_index(["cohort", "odor", "sensitized_odor"]).add_suffix(f"_{arm}"))
        side = pd.concat(parts, axis=1).reset_index()
        side.to_csv(OUT / f"sensitivity_{name}", index=False)

    _figures(paired, tables)
    for name in ("by_method", "by_method_and_trial_type", "by_method_and_eye_qc", "by_method_and_hand_flag"):
        print(f"\n== {name}\n" + tables[name].to_string(index=False))


def _figures(paired: pd.DataFrame, tables: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    ink, mark = "#1d1d1b", "#2a78d6"

    # 1. % of trials whose score changed, per method, in each group (position, not colour)
    groups = [
        ("all trials", paired.assign(g="all")),
        ("passes eye QC", paired[paired.qc_group.eq("passes eye QC")]),
        ("fails eye QC", paired[paired.qc_group.str.startswith("fails")]),
        ("hand-flagged flies", paired[paired.hand_flag_group.ne("not hand-flagged")]),
    ]
    names = list(METHODS.values())
    fig, axes = plt.subplots(1, len(groups), figsize=(15, 3.6), sharey=True)
    for ax, (title, g) in zip(axes, groups):
        vals = [100 * (g[g.method_name == n].score_change != 0).mean() for n in names]
        ns = [int((g.method_name == n).sum()) for n in names]
        y = np.arange(len(names))
        ax.hlines(y, 0, vals, color="#c9c7c1", lw=2)
        ax.plot(vals, y, "o", ms=9, color=mark, mec="white", mew=2)
        for yi, v in zip(y, vals):
            ax.text(v + 1, yi, f"{v:.1f}%", va="center", fontsize=9, color=ink)
        ax.set_xlim(0, max(30, max(vals) + 8))
        ax.set_title(f"{title}\n(n={ns[0]} trials)", fontsize=10)
        ax.set_xlabel("% of trials whose score changed")
        ax.grid(axis="x", color="#ecebe7", lw=0.8)
    axes[0].set_yticks(np.arange(len(names)), names)
    axes[0].invert_yaxis()
    fig.suptitle("How often each eye method changes a trial's score vs production (fixed anchor)", y=1.02)
    fig.savefig(OUT / "01_pct_scores_changed_by_method.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. largest score changes for each held per-frame method, traces vs production
    for arm, fname in (("livelight", "02_largest_changes_livelight.png"), ("livehold", "03_largest_changes_livehold.png")):
        m = paired[(paired.method == arm) & (paired.score_change != 0)]
        m = m.assign(absd=m.score_change.abs()).sort_values(["absd", "dataset"], ascending=[False, True]).head(6)
        if not len(m):
            continue
        tr = {a: lec._wide_traces(a, m) for a in ("anchor", arm)}
        colors = {"anchor": "#2a78d6", arm: "#eb6834"}
        label = {"anchor": "production (fixed anchor)", arm: METHODS[arm]}
        fig, axes = plt.subplots(len(m), 1, figsize=(9, 1.9 * len(m)), sharex=True, constrained_layout=True)
        for ax, (_, r) in zip(np.atleast_1d(axes), m.iterrows()):
            for a in ("anchor", arm):
                t = tr[a].get((r.dataset, r.fly, int(r.fly_number), r.trial_label))
                if t is not None:
                    score = int(r.score_anchor if a == "anchor" else r.score_new)
                    ax.plot(np.arange(len(t)) / 40.0, t, lw=1.6, color=colors[a], label=f"{label[a]}: score {score}")
            ax.axvspan(30, 60, color="#e9e7e1", zorder=0)
            ax.set_title(f"{r.dataset} · {r.fly} fly {int(r.fly_number)} · {r.trial_label}", fontsize=8.5, loc="left")
            ax.legend(frameon=False, fontsize=8, loc="upper right")
        np.atleast_1d(axes)[-1].set_xlabel("time (s); shaded = odor on")
        fig.suptitle(f"Largest score changes: production vs {METHODS[arm]}")
        fig.savefig(OUT / fname, dpi=200)
        plt.close(fig)

if __name__ == "__main__":
    main()
