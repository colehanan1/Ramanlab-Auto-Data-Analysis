#!/usr/bin/env python3
"""IAA: which quality-gate flies' traces change most when light frames are
left out of the fly's 0 % (min-only light exclusion), and what changes.

A fly is a quality-gate fly when any of its trials fails the eye-drift QC
(production report) or it is on the hand-flag list. The scored trace is
``combined_pct`` -- distance x angle multiplier, scaled by the fly's weighted
min / effective max (``angle_distance_rms_envelope/fly_norm_metadata.json``)
-- so that is what is compared, production copy vs ``nolightmin`` copy.

Writes ``Results/Figures/LiveEye-Comparison/IAA-NoLightMin/``:
``qc_gate_flies_trace_bounds.csv`` (every fly, gate status, bounds before /
after, scores changed) and ``06_qc_gate_flies_most_changed.png``.
"""
from __future__ import annotations

import importlib.util
import json
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


cmp_ = _load("iaa_lighthold_compare")
dec, lec = cmp_.dec, cmp_.lec
DS = "IAA-Sensitivity-1-1"
ARM = "nolightmin"
OUT = lec.RESULTS_ROOT / "IAA-NoLightMin"
BLUE, ORANGE = cmp_.BLUE, cmp_.ORANGE
TOP = 6


def _bounds(tree: Path) -> pd.DataFrame:
    rows = []
    for meta in sorted((tree / DS).glob("*/angle_distance_rms_envelope/fly_norm_metadata.json")):
        m = json.loads(meta.read_text())
        for n, lo in m["fly_weighted_min_by_number"].items():
            if not str(n).isdigit():
                continue
            rows.append({"fly": meta.parent.parent.name, "fly_number": int(n), "w_min": lo,
                         "w_max": m["fly_weighted_max_by_number"][n],
                         "w_eff_max": m["fly_effective_max_by_number"][n]})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a = _bounds(lec.ARMS["anchor"])
    b = _bounds(lec.ARMS[ARM])
    m = a.merge(b, on=["fly", "fly_number"], suffixes=("_production", "_minonly"))
    m["range_production"] = m.w_eff_max_production - m.w_min_production
    m["range_minonly"] = m.w_eff_max_minonly - m.w_min_minonly
    m["min_change"] = m.w_min_minonly - m.w_min_production
    m["range_change_pct"] = 100 * (m.range_minonly / m.range_production - 1)

    qc = dec._qc()
    qc = qc[(qc.dataset == DS) & qc.eye_qc_flag]
    qc_n = qc.groupby(["fly", "fly_number"]).size().rename("eye_qc_failed_trials")
    qc_kind = qc.groupby(["fly", "fly_number"]).qc_group.agg(lambda s: "; ".join(sorted(set(s)))).rename("eye_qc_reasons")
    flags = dec._hand_flags()
    flags = flags[flags.dataset == DS][["fly", "fly_number", "hand_flag"]]
    m = m.join(qc_n, on=["fly", "fly_number"]).join(qc_kind, on=["fly", "fly_number"])
    m = m.merge(flags, on=["fly", "fly_number"], how="left")
    m["eye_qc_failed_trials"] = m.eye_qc_failed_trials.fillna(0).astype(int)
    m["qc_gate_fly"] = (m.eye_qc_failed_trials > 0) | m.hand_flag.notna()

    s = pd.read_csv(OUT / "trial_scores.csv")
    ch = s.assign(changed=s.score_change.fillna(0) != 0, flipped=s.responder_flip.fillna("") != "")
    agg = ch.groupby(["fly", "fly_number"]).agg(trials_scored=("changed", "size"),
                                                scores_changed=("changed", "sum"),
                                                responder_flips=("flipped", "sum"),
                                                dataset_fly=("dataset_fly", "first"))
    m = m.join(agg, on=["fly", "fly_number"])
    m = m.sort_values("range_change_pct", key=lambda x: x.abs(), ascending=False)
    m.round(2).to_csv(OUT / "qc_gate_flies_trace_bounds.csv", index=False)

    pd.set_option("display.width", 230)
    print(f"flies: {len(m)}; quality-gate flies: {int(m.qc_gate_fly.sum())}; scale changed (min moved >= 1 px): "
          f"{int((m.min_change.abs() >= 1).sum())} (gate flies {int(((m.min_change.abs() >= 1) & m.qc_gate_fly).sum())})")
    cols = ["dataset_fly", "fly", "fly_number", "w_min_production", "w_min_minonly", "w_eff_max_production",
            "range_change_pct", "eye_qc_failed_trials", "hand_flag", "scores_changed", "responder_flips"]
    print(m[(m.min_change.abs() >= 1)][cols].round(1).to_string(index=False))

    _figure(m[m.qc_gate_fly & (m.min_change.abs() >= 1)].head(TOP), s)


def _figure(top: pd.DataFrame, s: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if top.empty:
        print("no quality-gate fly's scale changed")
        return
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    want = [("pretest", "isoamylacetate"), ("testing", "isoamylacetate"), ("testing", "lightonly")]
    titles = ["IAA pre-exposure", "IAA post-exposure", "light-only trial"]
    fig, axes = plt.subplots(len(top), 3, figsize=(15, 2.3 * len(top) + 1.2), sharex=True, squeeze=False)
    for i, (_, f) in enumerate(top.iterrows()):
        rows = s[(s.fly == f.fly) & (s.fly_number == f.fly_number)]
        for j, ((tt, odor), title) in enumerate(zip(want, titles)):
            ax = axes[i, j]
            r = rows[(rows.trial_type == tt) & rows.trial_label.str.contains(odor)]
            ax.axvspan(30, 60, color="#ecebe7", zorder=0, lw=0)
            if r.empty:
                ax.text(0.5, 0.5, "no trial", transform=ax.transAxes, ha="center", color="#8a8a86")
                continue
            r = r.iloc[0]
            tr = {k: lec._wide_traces(k, r.to_frame().T) for k in ("anchor", ARM)}
            for k, col, sc, lab in (("anchor", BLUE, r.score_production, "production"),
                                    (ARM, ORANGE, r.score_new, "light left out of min")):
                t = tr[k].get((r.dataset, r.fly, int(r.fly_number), r.trial_label))
                if t is not None:
                    ax.plot(np.arange(len(t)) / 40.0, t, lw=1.4, color=col,
                            label=f"{lab}: score {int(sc) if pd.notna(sc) else '–'}")
            ax.set_ylim(0, 105)
            ax.legend(frameon=False, fontsize=7.5, loc="upper right")
            if i == 0:
                ax.set_title(title, fontsize=10)
            if j == 0:
                why = []
                if f.eye_qc_failed_trials:
                    why.append(f"eye QC fails {int(f.eye_qc_failed_trials)} trials")
                if isinstance(f.hand_flag, str):
                    why.append(f"hand-flagged: {f.hand_flag}")
                ax.set_ylabel(f"Fly {int(f.dataset_fly)}\n{f.fly} slot {int(f.fly_number)}\n"
                              f"0 % {f.w_min_production:.0f}→{f.w_min_minonly:.0f} px "
                              f"({f.range_change_pct:+.0f}% range)\n" + "; ".join(why), fontsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel("time (s); shaded = odor / light on")
    fig.suptitle("IAA quality-gate flies whose scale changes most: production vs light frames left out of the 0 % "
                 "(% of fly's range)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "06_qc_gate_flies_most_changed.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
