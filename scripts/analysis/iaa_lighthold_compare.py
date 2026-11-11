#!/usr/bin/env python3
"""IAA-Sensitivity-1-1: production's fixed eye vs the per-frame eye held while
the light is on (``live_eye_shadow.py --hold-light``, arm ``livelight``).

Both arms are copies of today's production data run through the same steps,
code, config and flag list; the anchor arm is the control. Writes
``Results/Figures/LiveEye-Comparison/IAA-LightHold/``:

* ``per_fly_min_max.csv`` -- each fly's global min/max (the 0 % / 100 % of its
  traces) in both arms, and the frame that sets the per-frame arm's min;
* ``trial_scores.csv`` + ``score_change_summary.csv`` -- every scored trial,
  hand-flagged flies included;
* ``sensitivity_window_summary_per_odor.csv`` -- the IAA results side by side,
  plus production's own table as a reproduction check;
* figures 01-04.
"""
from __future__ import annotations

import argparse
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


dec = _load("drift_eye_compare")
lec = dec.lec
from fbpipe.utils import fly_registry as fr  # noqa: E402

DS = "IAA-Sensitivity-1-1"
#: arm -> (tree, output folder, label)
VARIANTS = {
    "livelight": ("flys_LiveEyeLightHold", "IAA-LightHold", "per-frame eye, held while light on"),
    "nolightscale": ("flys_NoLightMinMax", "IAA-NoLightMinMax", "fixed eye, light frames left out of min/max"),
    "nolightmin": ("flys_NoLightMin", "IAA-NoLightMin", "light frames left out of the 0 % only"),
    "final": ("flys_FinalMethod", "IAA-FinalMethod", "per-frame eye (held in light) + light out of 0 %"),
}
ARM = "livelight"
for _a, (_tree, _out, _label) in VARIANTS.items():
    lec.ARMS[_a] = lec.DATA / _tree
OUT = lec.RESULTS_ROOT / VARIANTS[ARM][1]
PROD_RESULTS = Path("/home/ramanlab/Documents/cole/Results/Sensitiviy-Results(11-15)")
KEY = ["dataset", "fly", "fly_number", "trial_label"]
BLUE, ORANGE = "#2a78d6", "#eb6834"   # validated pair (dataviz validator, light surface)
NAME = {"anchor": "production (fixed eye)", **{a: v[2] for a, v in VARIANTS.items()}}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def _wide(arm: str) -> pd.DataFrame:
    frames = []
    for x in ("", "_pretest", "_training"):
        p = lec.CSV_ROOT / arm / f"all_envelope_rows_wide_combined_base{x}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p, columns=["dataset", "fly", "fly_number", "trial_label",
                                                      "global_min", "global_max"]))
    w = pd.concat(frames, ignore_index=True)
    return w[w.dataset == DS]


def _scores(arm: str) -> pd.DataFrame:
    df = dec._predictions_all(arm)
    return df[df.dataset == DS]


def _min_frame(batch: str, n: int) -> dict:
    """The frame that sets the per-frame arm's closest proboscis distance."""
    best = None
    for p in sorted((lec.ARMS[ARM] / DS / batch).glob(f"*/*_fly{n}_distances.parquet")):
        if p.parent.name in ("RMS_calculations", "angle_distance_rms_envelope"):
            continue
        cols = ["frame", "distance_0_1", "eye_live_x", "eye_live_y", "x_class0_anchor", "y_class0_anchor", "eye_held"]
        have = set(pd.read_parquet(p).columns)
        if "x_class0_anchor" not in have:
            have |= {"x_class0_anchor", "y_class0_anchor"}
        d = pd.read_parquet(p, columns=[c for c in cols if c in pd.read_parquet(p).columns])
        if "x_class0_anchor" not in d.columns:
            d["x_class0_anchor"], d["y_class0_anchor"] = d["eye_live_x"], d["eye_live_y"]
        d = d[d.distance_0_1 >= 10]
        if d.empty:
            continue
        i = d.distance_0_1.idxmin()
        cand = (float(d.distance_0_1[i]), p.parent.name, int(d.frame[i]),
                float(np.hypot(d.eye_live_x[i] - d.x_class0_anchor[i], d.eye_live_y[i] - d.y_class0_anchor[i])))
        if best is None or cand[0] < best[0]:
            best = cand
    if best is None:
        return {}
    return {"min_set_by_trial": best[1].split("_", 3)[-1], "min_frame_s": round(best[2] / 40.0, 2),
            "eye_offset_at_min_px": round(best[3], 1)}


def _registry() -> dict:
    flags = dec._hand_flags()
    h = flags[flags.dataset == DS]
    reason = {(b, int(n)): r for b, n, r in zip(h.fly, h.fly_number, h.hand_flag)}
    reg = fr.update_registry(dec._flies_now("anchor", DS, set(reason)), None)
    OUT.mkdir(parents=True, exist_ok=True)
    reg.assign(flag_reason=[reason.get((b, int(n)), "") for b, n in zip(reg.batch, reg.fly_number)]).to_csv(
        OUT / "fly_registry_IAA.csv", index=False)
    return {(b, int(n)): int(k) for b, n, k in zip(reg.batch, reg.fly_number, reg.dataset_fly)}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv=None) -> None:
    global ARM, OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=sorted(VARIANTS), default="livelight")
    ap.add_argument("--fly", action="append", default=["october_09_batch_2_rig_3:1"],
                    help="batch:fly for the per-fly pre-vs-post figure")
    args = ap.parse_args(argv)
    ARM = args.arm
    OUT = lec.RESULTS_ROOT / VARIANTS[ARM][1]
    OUT.mkdir(parents=True, exist_ok=True)
    num = _registry()
    flags = dec._hand_flags()

    # 1. min / max per fly
    a = _wide("anchor").groupby(["fly", "fly_number"])[["global_min", "global_max"]].first()
    l = _wide(ARM).groupby(["fly", "fly_number"])[["global_min", "global_max"]].first()
    mm = a.join(l, lsuffix="_production", rsuffix="_new", how="outer").reset_index()
    mm["min_change_px"] = mm.global_min_new - mm.global_min_production
    mm["max_change_px"] = mm.global_max_new - mm.global_max_production
    mm["range_change_pct"] = 100 * ((mm.global_max_new - mm.global_min_new)
                                   / (mm.global_max_production - mm.global_min_production) - 1)
    mm["dataset_fly"] = [num.get((b, int(n))) for b, n in zip(mm.fly, mm.fly_number)]
    # where the per-frame eye's new min came from (per-frame-eye arms only)
    extra = [(_min_frame(b, int(n)) if ARM == "livelight" and abs(c) >= 1 else {})
             for b, n, c in zip(mm.fly, mm.fly_number, mm.min_change_px.fillna(0))]
    mm = pd.concat([mm, pd.DataFrame(extra, index=mm.index)], axis=1)
    mm = mm.merge(flags[flags.dataset == DS][["fly", "fly_number", "hand_flag"]], on=["fly", "fly_number"], how="left")
    mm["biggest_change_px"] = mm[["min_change_px", "max_change_px"]].abs().max(axis=1)
    mm = mm.sort_values("biggest_change_px", ascending=False)
    mm.round(2).to_csv(OUT / "per_fly_min_max.csv", index=False)

    # 2. scores
    s = _scores("anchor").merge(_scores(ARM)[KEY + ["prediction", "score"]], on=KEY, how="outer",
                                suffixes=("_production", "_new"))
    s["score_change"] = s.score_new - s.score_production
    s["responder_flip"] = np.select(
        [(s.prediction_production == 0) & (s.prediction_new == 1),
         (s.prediction_production == 1) & (s.prediction_new == 0)],
        ["became responder", "stopped responding"], default="")
    s = s.merge(flags[flags.dataset == DS][["fly", "fly_number", "hand_flag"]], on=["fly", "fly_number"], how="left")
    s["hand_flagged"] = s.hand_flag.notna()
    s["dataset_fly"] = [num.get((b, int(n))) for b, n in zip(s.fly, s.fly_number)]
    s = s.merge(mm[["fly", "fly_number", "min_change_px", "max_change_px"]], on=["fly", "fly_number"], how="left")
    s.to_csv(OUT / "trial_scores.csv", index=False)

    both = s.dropna(subset=["score_production", "score_new"])
    moved = both.min_change_px.abs().ge(1) | both.max_change_px.abs().ge(1)

    def summ(g):
        return pd.Series({
            "n_trials": len(g), "n_flies": g.groupby(["fly", "fly_number"]).ngroups,
            "pct_same_score": round(100 * (g.score_change == 0).mean(), 1),
            "n_up": int((g.score_change > 0).sum()), "n_down": int((g.score_change < 0).sum()),
            "n_became_responder": int((g.responder_flip == "became responder").sum()),
            "n_stopped_responding": int((g.responder_flip == "stopped responding").sum()),
            "pct_responding_production": round(100 * g.prediction_production.mean(), 1),
            "pct_responding_new": round(100 * g.prediction_new.mean(), 1),
        })

    table = pd.concat([
        summ(both).to_frame("all trials").T,
        both.groupby("trial_type").apply(summ, include_groups=False),
        summ(both[~both.hand_flagged]).to_frame("flies production scores (not hand-flagged)").T,
        summ(both[both.hand_flagged]).to_frame("hand-flagged flies").T,
        summ(both[moved]).to_frame("flies whose min or max moved >= 1 px").T,
        summ(both[~moved]).to_frame("flies whose min and max held").T,
    ])
    table.index.name = "group"
    table.to_csv(OUT / "score_change_summary.csv")

    # 3. sensitivity results
    parts = []
    for arm, root in (("production_rerun", lec.RESULTS_ROOT / "anchor"), ("new", lec.RESULTS_ROOT / ARM)):
        t = pd.read_csv(root / "Sensitivity-Results" / "Data" / "window_summary_per_odor.csv")
        parts.append(t[t.cohort == DS].set_index(["cohort", "odor", "sensitized_odor"]).add_suffix(f"_{arm}"))
    prod = pd.read_csv(PROD_RESULTS / "Data" / "window_summary_per_odor.csv")
    parts.append(prod[prod.cohort == DS].set_index(["cohort", "odor", "sensitized_odor"]).add_suffix("_production_1435"))
    side = pd.concat(parts, axis=1).reset_index()
    side.to_csv(OUT / "sensitivity_window_summary_per_odor.csv", index=False)

    _figures(mm, both, s, args.fly)
    pd.set_option("display.width", 220)
    print(table.to_string())
    cols = [c for c in side.columns if c.split("_production")[0].split("_new")[0] in
            ("pct_responding_pre", "pct_responding_post", "p_rate_mcnemar_paired", "p_score_wilcoxon", "n_flies")]
    print(side[["odor", "sensitized_odor"] + cols].round(3).to_string(index=False))
    show = [c for c in ["fly", "fly_number", "dataset_fly", "global_min_production", "global_min_new",
                        "global_max_production", "global_max_new", "range_change_pct", "min_set_by_trial",
                        "min_frame_s", "eye_offset_at_min_px", "hand_flag"] if c in mm.columns]
    print(mm.head(12)[show].round(1).to_string(index=False))
    print(f"flies whose min moved >=1 px: {int((mm.min_change_px.abs() >= 1).sum())}, max moved >=1 px: "
          f"{int((mm.max_change_px.abs() >= 1).sum())} of {len(mm)}; range change median "
          f"{mm.range_change_pct[mm.biggest_change_px >= 1].median():.1f}%")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _figures(mm, both, s, fly_args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # 1. each fly's min and max, production -> new method (dumbbells), largest changes first
    top = mm.dropna(subset=["global_min_production", "global_min_new"]).head(30).iloc[::-1]
    labels = [f"Fly {k}" + (" (flagged)" if isinstance(h, str) else "") for k, h in zip(top.dataset_fly, top.hand_flag)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.3 * len(top) + 1.8), sharey=True)
    for ax, what in zip(axes, ("min", "max")):
        p, q = top[f"global_{what}_production"], top[f"global_{what}_new"]
        y = np.arange(len(top))
        ax.hlines(y, p, q, color="#c9c7c1", lw=2)
        ax.plot(p, y, "o", color=BLUE, ms=7, mec="white", mew=1.5, label=NAME["anchor"])
        ax.plot(q, y, "o", color=ORANGE, ms=7, mec="white", mew=1.5, label=NAME[ARM])
        ax.set_xlabel(f"fly's global {what}: the {'0' if what == 'min' else '100'} % of its traces (px)")
        ax.grid(axis="x", color="#ecebe7", lw=0.8)
    axes[0].set_yticks(np.arange(len(top)), labels, fontsize=8)
    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    fig.suptitle(f"IAA-Sensitivity-1-1: each fly's distance range, production vs {NAME[ARM]} (largest changes)")
    fig.tight_layout()
    fig.savefig(OUT / "01_min_max_per_fly.png", dpi=200)
    plt.close(fig)

    # 2. score transitions by trial type
    types = [t for t in ("pretest", "testing", "training") if t in set(both.trial_type)]
    levels = list(range(-1, 6))
    fig, axes = plt.subplots(1, len(types), figsize=(4.6 * len(types), 4.6))
    for ax, t in zip(np.atleast_1d(axes), types):
        g = both[both.trial_type == t]
        m = pd.crosstab(g.score_production.astype(int), g.score_new.astype(int)).reindex(
            index=levels, columns=levels, fill_value=0)
        ax.imshow(np.log1p(m.values), cmap="Blues", origin="lower")
        top_v = np.log1p(m.values.max()) or 1
        for i in range(len(levels)):
            for j in range(len(levels)):
                v = m.values[i, j]
                if v:
                    ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                            color="white" if np.log1p(v) > 0.6 * top_v else "#222")
        ax.set_xticks(range(len(levels)), levels)
        ax.set_yticks(range(len(levels)), levels)
        ax.set_xlabel(f"score, {NAME[ARM]}")
        ax.set_ylabel("score, production")
        ax.set_title(f"{t}: {100 * (g.score_change == 0).mean():.1f}% unchanged (n={len(g)})")
    fig.suptitle("IAA-Sensitivity-1-1: model score per trial × fly (diagonal = unchanged; flagged flies included)")
    fig.tight_layout()
    fig.savefig(OUT / "02_score_transitions.png", dpi=200)
    plt.close(fig)

    # 3. largest score changes, traces
    ch = both[both.score_change != 0].assign(absd=lambda d: d.score_change.abs())
    ch = ch.sort_values(["absd", "fly"], ascending=[False, True]).head(6)
    if len(ch):
        tr = {a: lec._wide_traces(a, ch) for a in ("anchor", ARM)}
        fig, axes = plt.subplots(len(ch), 1, figsize=(9, 1.9 * len(ch)), sharex=True, constrained_layout=True)
        for ax, (_, r) in zip(np.atleast_1d(axes), ch.iterrows()):
            for a, col, sc in (("anchor", BLUE, r.score_production), (ARM, ORANGE, r.score_new)):
                t = tr[a].get((r.dataset, r.fly, int(r.fly_number), r.trial_label))
                if t is not None:
                    ax.plot(np.arange(len(t)) / 40.0, t, lw=1.6, color=col, label=f"{NAME[a]}: score {int(sc)}")
            ax.axvspan(30, 60, color="#e9e7e1", zorder=0)
            ax.set_title(f"Fly {r.dataset_fly} ({r.fly} slot {int(r.fly_number)}) · {r.trial_label} · "
                         f"min {r.min_change_px:+.1f} px, max {r.max_change_px:+.1f} px", fontsize=8.5, loc="left")
            ax.legend(frameon=False, fontsize=8, loc="upper right")
        np.atleast_1d(axes)[-1].set_xlabel("time (s); shaded = odor on")
        fig.suptitle(f"IAA-Sensitivity-1-1: largest score changes, production vs {NAME[ARM]}")
        fig.savefig(OUT / "03_largest_changes_traces.png", dpi=200)
        plt.close(fig)

    # 5. every trial that flipped responder status, production vs new; 07 = became responder only
    flips_all = both[both.responder_flip != ""].sort_values(["responder_flip", "trial_type", "dataset_fly", "trial_label"])
    for fl, fname, what in ((flips_all, "05_flipped_trials_traces.png", "flipped responder status"),
                            (flips_all[flips_all.responder_flip == "became responder"],
                             "07_became_responder_traces.png", "went from non-responder to responder")):
        if not len(fl):
            continue
        tr = {a: lec._wide_traces(a, fl) for a in ("anchor", ARM)}
        ncol = 2
        nrow = int(np.ceil(len(fl) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.1 * nrow + 1.0), sharex=True, squeeze=False)
        for ax, (_, r) in zip(axes.flat, fl.iterrows()):
            ax.axvspan(30, 60, color="#ecebe7", zorder=0, lw=0)
            for a, col, sc in (("anchor", BLUE, r.score_production), (ARM, ORANGE, r.score_new)):
                t = tr[a].get((r.dataset, r.fly, int(r.fly_number), r.trial_label))
                if t is not None:
                    ax.plot(np.arange(len(t)) / 40.0, t, lw=1.4, color=col, label=f"{NAME[a]}: score {int(sc)}")
            ax.set_ylim(0, 105)
            flag = f" · flagged: {r.hand_flag}" if isinstance(r.hand_flag, str) else ""
            ax.set_title(f"Fly {r.dataset_fly} ({r.fly} slot {int(r.fly_number)}) · {r.trial_label} · "
                         f"{r.responder_flip}\nfly min {r.min_change_px:+.1f} px, max {r.max_change_px:+.1f} px{flag}",
                         fontsize=8.5, loc="left")
            ax.legend(frameon=False, fontsize=7.5, loc="upper right")
        for ax in axes.flat[len(fl):]:
            ax.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("time (s); shaded = odor on")
        for ax in axes[:, 0]:
            ax.set_ylabel("% of fly's range")
        fig.suptitle(f"IAA-Sensitivity-1-1: all {len(fl)} trials that {what} "
                     f"(production vs {NAME[ARM]}; responder = score ≥ 2)", y=1.0)
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=170)
        plt.close(fig)

    # 4. per-fly pre vs post, both methods side by side
    for spec in fly_args:
        batch, n = spec.split(":")
        rows = s[(s.fly == batch) & (s.fly_number == int(n)) & s.trial_type.isin(["pretest", "testing"])].copy()
        if rows.empty:
            continue
        rows["odor"] = rows.trial_label.str.split("_", n=2).str[2]
        odors = sorted(o for o in rows.odor.unique() if "light" not in o)
        tr = {a: lec._wide_traces(a, rows) for a in ("anchor", ARM)}
        fig, axes = plt.subplots(len(odors), 2, figsize=(13, 1.6 * len(odors) + 1.2), sharex=True, sharey=True)
        for i, odor in enumerate(odors):
            for j, (a, col_name) in enumerate((("anchor", "score_production"), (ARM, "score_new"))):
                ax = axes[i, j]
                ax.axvspan(30, 60, color="#ecebe7", zorder=0, lw=0)
                txt = []
                for phase, col in (("pretest", ORANGE), ("testing", BLUE)):
                    r = rows[(rows.odor == odor) & (rows.trial_type == phase)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    t = tr[a].get((r.dataset, r.fly, int(r.fly_number), r.trial_label))
                    if t is not None:
                        ax.plot(np.arange(len(t)) / 40.0, t, lw=1.4, color=col)
                    sc = r[col_name]
                    txt.append(f"{'pre' if phase == 'pretest' else 'post'} {int(sc) if pd.notna(sc) else '–'}")
                ax.set_ylim(0, 100)
                ax.text(0.01, 0.92, odor, transform=ax.transAxes, fontsize=9, weight="bold", va="top")
                ax.text(0.99, 0.92, "  ".join(txt), transform=ax.transAxes, fontsize=9, va="top", ha="right")
                if i == 0:
                    ax.set_title(NAME[a], fontsize=11)
        for ax in axes[-1]:
            ax.set_xlabel("time (s); shaded = odor on")
        k = num_label = rows.dataset_fly.iloc[0]
        fig.suptitle(f"IAA-Sensitivity-1-1 · {batch} slot {n} (Fly {num_label}): pre-exposure (orange) vs "
                     f"post-exposure (blue), % of the fly's range", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / f"04_{batch}_fly{n}_pre_vs_post_both_methods.png", dpi=170)
        plt.close(fig)


if __name__ == "__main__":
    main()
