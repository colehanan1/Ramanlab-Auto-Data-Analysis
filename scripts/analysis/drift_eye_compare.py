#!/usr/bin/env python3
"""Compare the drift-following eye anchor against production (the anchor arm).

Arms (built by ``live_eye_shadow_run.py``):
* ``anchor`` -- production tables re-run as they are (control);
* ``drift``  -- ``drift_eye_shadow.py``: anchor static, moved only by a held
  drift (eye-QC rule on the eye's settled position) or a held group move;
  warm-up misses take the next video's anchor; november_07_batch_1 re-run on
  brightened video (4 flies).
* ``live``   -- the raw per-frame eye of 2026-11-09, for reference.

Fly identity across arms comes from the dataset fly registry
(``fbpipe.utils.fly_registry``): built from production first, then updated
with the drift arm, so a fly a re-run found is numbered on top.

Writes ``Results/Figures/LiveEye-Comparison/Drift-Anchor/``.
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
    spec.loader.exec_module(mod)
    return mod


lec = _load("live_eye_compare")
des = _load("drift_eye_shadow")
from fbpipe.utils import fly_registry as fr  # noqa: E402

lec.ARMS["drift"] = lec.DATA / "flys_DriftEye"
OUT = lec.RESULTS_ROOT / "Drift-Anchor"
DATASETS = ("3Oct-Sensitivity-1-0.1", "Hex-Sensitivity-1-0.1", "IAA-Sensitivity-1-1", "Paraffin-Sensitivity-1-100")
# Validated pair (dataviz validator, light surface).
COLORS = {"anchor": "#2a78d6", "drift": "#eb6834"}
NAMES = {"anchor": "fixed anchor (production)", "drift": "drift-following anchor"}


# --------------------------------------------------------------------------
# Fly registry
# --------------------------------------------------------------------------


def _batch_started(arm_root: Path, dataset: str, batch: str) -> str:
    trials = des.trial_dirs(arm_root / dataset / batch)
    return min((des.recorded_at(t) for t in trials), default="~")


def _flies_now(arm: str, dataset: str, flagged: set) -> pd.DataFrame:
    root = lec.ARMS[arm]
    rows = []
    for batch in sorted(p.name for p in (root / dataset).iterdir() if p.is_dir()):
        start = _batch_started(root, dataset, batch)
        for n, (x, y) in lec._batch_eyes(root, dataset, batch).items():
            rows.append({"batch": batch, "fly_number": n, "eye_x": x, "eye_y": y,
                         "first_recorded": start, "flagged": (batch, n) in flagged})
    return pd.DataFrame(rows)


def build_registries(qc: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: number production's flies, then update with the drift arm.

    ``flagged`` is the hand-flag list (flagged-flys-truth.csv) -- a flagged
    fly keeps its number; ``flag_reason`` says why, ``eye_qc_failed_trials``
    counts its trials the eye-drift QC flags. Flags are keyed by production's
    slot numbers, so they are carried to the drift arm by registry number.
    """
    out = []
    reg_dir = OUT / "fly_registry"
    reg_dir.mkdir(parents=True, exist_ok=True)
    hand = _hand_flags()
    for ds in DATASETS:
        h = hand[hand.dataset == ds]
        reason = {(b, int(n)): r for b, n, r in zip(h.fly, h.fly_number, h.hand_flag)}
        q = qc[(qc.dataset == ds) & qc.eye_qc_flag]
        qc_count = q.groupby(["fly", "fly_number"]).size().to_dict()
        first = fr.update_registry(_flies_now("anchor", ds, set(reason)), None)
        reg = fr.update_registry(_flies_now("drift", ds, set()), first)
        by_number = first.set_index("dataset_fly")
        prod_key = {k: (r.batch, int(r.fly_number)) for k, r in by_number.iterrows()}
        reg["flagged"] = reg.dataset_fly.map(lambda k: prod_key.get(k) in reason)
        reg["flag_reason"] = reg.dataset_fly.map(lambda k: reason.get(prod_key.get(k), ""))
        reg["eye_qc_failed_trials"] = reg.dataset_fly.map(lambda k: int(qc_count.get(prod_key.get(k), 0)))
        reg["label"] = reg.dataset_fly.map(fr.label)
        first.to_csv(reg_dir / f"{ds}_production.csv", index=False)
        reg.to_csv(reg_dir / f"{ds}.csv", index=False)
        for arm, now in (("anchor", first), ("drift", reg[reg.present])):
            out.append(now.assign(dataset=ds, arm=arm)[["dataset", "arm", "batch", "fly_number", "dataset_fly"]])
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def _with_ids(df: pd.DataFrame, ids: pd.DataFrame, arm: str) -> pd.DataFrame:
    m = ids[ids.arm == arm].rename(columns={"batch": "fly"}).drop(columns="arm")
    return df.merge(m, on=["dataset", "fly", "fly_number"], how="left")


def _drift_summary() -> pd.DataFrame:
    frames = []
    for ds in DATASETS:
        p = lec.ARMS["drift"] / ds / "_drift_eye_summary.csv"
        frames.append(pd.read_csv(p).assign(dataset=ds))
    s = pd.concat(frames, ignore_index=True).rename(columns={"batch": "fly", "fly": "fly_number"})
    s["trial_key"] = s["trial"].str.split("_").str[-2:].str.join("_")
    s["anchor_moved"] = (s.n_steps > 0) | (s.initial_dx != 0) | (s.initial_dy != 0)
    return s[["dataset", "fly", "fly_number", "trial_key", "anchor_moved", "n_steps", "first_move_group", "n_follow_steps",
              "max_offset_px", "warmup_found", "initial_dx", "initial_dy"]]


def _qc() -> pd.DataFrame:
    qc = lec._eye_qc()
    qc = qc[qc.dataset.isin(DATASETS)].copy()
    qc["eye_qc_flag"] = qc.eye_qc_flag.astype(str).str.lower().eq("true")
    qc["trial_key"] = qc["trial_label"].str.split("_").str[-2:].str.join("_")
    kind = qc.eye_qc_reason.fillna("").str.extract(r"^(drift|eye unmatched|eye never)")[0]
    qc["qc_group"] = np.where(~qc.eye_qc_flag, "passes eye QC",
                              np.where(kind.eq("drift"), "fails eye QC: drift", "fails eye QC: eye unmatched"))
    return qc


def _predictions_all(arm: str) -> pd.DataFrame:
    """Scores for every fly, the hand-flagged ones included (``score_allflies`` stage)."""
    frames = [pd.read_csv(lec.CSV_ROOT / arm / name)
              for name in ("model_predictions_allflies.csv", "model_predictions_training_allflies.csv")]
    df = pd.concat(frames, ignore_index=True)
    label_type = df["trial_label"].astype(str).str.split("_").str[0]
    df["trial_type"] = df.get("trial_type", pd.Series(np.nan, index=df.index)).fillna(label_type)
    return df[["dataset", "fly", "fly_number", "trial_label", "trial_type", "prediction", "score"]]


def _hand_flags() -> pd.DataFrame:
    """flagged-flys-truth.csv (config_new's flagged_flies_csv): why a fly is excluded."""
    from fbpipe.config import load_settings

    path = Path(load_settings(lec.REPO / "config" / "config_new.yaml").flagged_flies_csv)
    f = pd.read_csv(path)
    f = f[f.dataset.isin(DATASETS)].copy()
    f["fly_number"] = pd.to_numeric(f.fly_number, errors="coerce")
    f["comment"] = f["comment"].fillna("").astype(str)
    reasons = f.groupby(["dataset", "fly", "fly_number"])["comment"].agg(lambda c: "+".join(sorted(set(c))))
    return reasons.rename("hand_flag").reset_index()


def _hand_group(reason) -> str:
    if not isinstance(reason, str) or not reason:
        return "not hand-flagged"
    if "bad-pin-out" in reason:
        return "hand-flagged: bad-pin-out (Pi 2 Nov 5-9)"
    if "moved" in reason:
        return "hand-flagged: moved"
    return "hand-flagged: other reason"


def _score_pair(ids: pd.DataFrame) -> pd.DataFrame:
    pa = _with_ids(_predictions_all("anchor"), ids, "anchor")
    pdr = _with_ids(_predictions_all("drift"), ids, "drift")
    key = ["dataset", "fly", "trial_label", "dataset_fly"]
    s = pa.merge(pdr, on=key, how="outer", suffixes=("_anchor", "_drift"))
    s["trial_type"] = s.trial_type_anchor.fillna(s.trial_type_drift)
    s["fly_number"] = s.fly_number_anchor.fillna(s.fly_number_drift)
    s["score_change"] = s.score_drift - s.score_anchor
    s["responder_flip"] = np.select(
        [(s.prediction_anchor == 0) & (s.prediction_drift == 1), (s.prediction_anchor == 1) & (s.prediction_drift == 0)],
        ["became responder", "stopped responding"], default="")
    s["trial_key"] = s["trial_label"].str.split("_").str[:2].str.join("_")
    return s


def _group_table(both: pd.DataFrame, by: list) -> pd.DataFrame:
    def stats(g):
        return pd.Series({
            "n_trials": len(g),
            "n_flies": g["fly_key"].nunique(),
            "pct_same_score": round(100 * (g.score_change == 0).mean(), 1),
            "n_score_up": int((g.score_change > 0).sum()),
            "n_score_down": int((g.score_change < 0).sum()),
            "n_became_responder": int((g.responder_flip == "became responder").sum()),
            "n_stopped_responding": int((g.responder_flip == "stopped responding").sum()),
            "pct_responding_anchor": round(100 * g.prediction_anchor.mean(), 1),
            "pct_responding_drift": round(100 * g.prediction_drift.mean(), 1),
        })
    both = both.assign(fly_key=both["dataset"].astype(str) + "|" + both["fly"].astype(str) + "|"
                       + both["dataset_fly"].astype(str))
    return both.groupby(by).apply(stats, include_groups=False).reset_index()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    qc = _qc()
    ids = build_registries(qc)
    s = _score_pair(ids)
    s = s.merge(_drift_summary(), left_on=["dataset", "fly", "fly_number_drift", "trial_key"],
                right_on=["dataset", "fly", "fly_number", "trial_key"], how="left", suffixes=("", "_ds"))
    s = s.merge(qc[["dataset", "fly", "fly_number", "trial_key", "eye_qc_flag", "qc_group", "eye_qc_reason"]],
                left_on=["dataset", "fly", "fly_number_anchor", "trial_key"],
                right_on=["dataset", "fly", "fly_number", "trial_key"], how="left", suffixes=("", "_qc"))
    s = s.drop(columns=[c for c in s.columns if c.endswith(("_ds", "_qc"))])
    s["anchor_moved"] = s.anchor_moved.fillna(False).astype(bool)
    flags = _hand_flags()
    s = s.merge(flags.rename(columns={"fly_number": "fly_number_anchor"}),
                on=["dataset", "fly", "fly_number_anchor"], how="left")
    s["hand_flag_group"] = s.hand_flag.map(_hand_group)
    moved_flies = set(map(tuple, s.loc[s.anchor_moved, ["dataset", "fly", "dataset_fly"]].to_numpy()))
    flagged_flies = set(map(tuple, s.loc[s.eye_qc_flag.fillna(False).astype(bool),
                                         ["dataset", "fly", "dataset_fly"]].to_numpy()))
    trio = list(map(tuple, s[["dataset", "fly", "dataset_fly"]].to_numpy()))
    s["trial_group"] = np.select(
        [s.score_anchor.isna(), s.anchor_moved, [t in moved_flies for t in trio]],
        ["new fly (Nov 7 re-run)", "anchor moved in this trial", "anchor moved in another trial of this fly"],
        default="anchor never moved for this fly")
    s["qc_group"] = s.qc_group.fillna("no eye-QC row")
    s["fly_group"] = np.where([t in flagged_flies for t in trio], "fly fails eye QC in ≥1 trial",
                              "fly passes eye QC in every trial")
    s["dataset_fly_label"] = s.dataset_fly.map(lambda v: fr.label(v) if pd.notna(v) else "")
    s.to_csv(OUT / "trial_scores_anchor_vs_drift.csv", index=False)

    both = s.dropna(subset=["score_anchor", "score_drift"])
    nov7 = both.fly.eq("november_07_batch_1") & both.dataset.eq("3Oct-Sensitivity-1-0.1")
    tables = {
        "by_trial_group": _group_table(both[~nov7], ["trial_group"]),
        "by_qc_group": _group_table(both[~nov7], ["qc_group"]),
        "by_qc_group_and_trial_type": _group_table(both[~nov7], ["qc_group", "trial_type"]),
        "by_fly_group": _group_table(both[~nov7], ["fly_group"]),
        "by_hand_flag_group": _group_table(both[~nov7], ["hand_flag_group"]),
        "by_hand_flag_and_trial_group": _group_table(both[~nov7], ["hand_flag_group", "trial_group"]),
        "production_scored_only": _group_table(both[~nov7 & both.hand_flag_group.eq("not hand-flagged")],
                                               ["trial_group"]),
        "by_dataset": _group_table(both[~nov7], ["dataset", "trial_type"]),
        "nov07_batch_1": _group_table(both[nov7], ["trial_group"]),
    }
    with pd.ExcelWriter(OUT / "score_change_summary.xlsx") as xw:
        for name, t in tables.items():
            t.to_csv(OUT / f"score_change_{name}.csv", index=False)
            t.to_excel(xw, sheet_name=name[:31], index=False)

    for name in ("window_summary_per_odor.csv", "whole-cohort_summary_per_odor.csv"):
        parts = []
        for arm in ("anchor", "drift", "live"):
            p = lec.RESULTS_ROOT / arm / "Sensitivity-Results" / "Data" / name
            if p.exists():
                parts.append(pd.read_csv(p).set_index(["cohort", "odor", "sensitized_odor"]).add_suffix(f"_{arm}"))
        side = pd.concat(parts, axis=1).reset_index()
        side.to_csv(OUT / f"sensitivity_{name}", index=False)

    _figures(both[~nov7], s)
    for name in ("by_trial_group", "by_qc_group", "by_hand_flag_group", "by_hand_flag_and_trial_group",
                 "production_scored_only"):
        print(f"\n== {name}\n" + tables[name].to_string(index=False))


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _transition(ax, g, title):
    levels = list(range(0, 6))
    m = pd.crosstab(g.score_anchor.astype(int), g.score_drift.astype(int)).reindex(
        index=levels, columns=levels, fill_value=0)
    ax.imshow(np.log1p(m.values), cmap="Blues", origin="lower")
    top = np.log1p(m.values.max()) if m.values.max() else 1
    for i in range(len(levels)):
        for j in range(len(levels)):
            v = m.values[i, j]
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                        color="white" if np.log1p(v) > 0.6 * top else "#222")
    ax.set_xticks(range(len(levels)), levels)
    ax.set_yticks(range(len(levels)), levels)
    ax.set_xlabel("score, drift-following anchor")
    ax.set_ylabel("score, fixed anchor (production)")
    same = 100 * (g.score_anchor == g.score_drift).mean() if len(g) else float("nan")
    ax.set_title(f"{title}\n{same:.1f}% unchanged (n={len(g)} trials)", fontsize=10)


def _figures(both: pd.DataFrame, s: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # 1. transitions: the QC-failing test set vs everything else
    groups = [
        ("fails eye QC: drift", both.qc_group.eq("fails eye QC: drift")),
        ("hand-flagged 'moved'", both.hand_flag_group.eq("hand-flagged: moved")),
        ("any fly, anchor moved in this trial", both.anchor_moved),
        ("fly whose anchor never moved", both.trial_group.eq("anchor never moved for this fly")),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.6))
    for ax, (title, mask) in zip(axes, groups):
        _transition(ax, both[mask], title)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.80, bottom=0.13, wspace=0.32)
    fig.suptitle("Model score per trial × fly: production fixed anchor vs drift-following anchor\n"
                 "diagonal = unchanged · hand-flagged flies included · Nov 7 batch excluded", y=0.98)
    fig.savefig(OUT / "01_score_transitions_by_eye_qc.png", dpi=200)
    plt.close(fig)

    # 2. % responding per cohort x odor, production vs drift
    side = pd.read_csv(OUT / "sensitivity_window_summary_per_odor.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    for ax, phase in zip(axes, ("pre", "post")):
        x, y = side[f"pct_responding_{phase}_anchor"], side[f"pct_responding_{phase}_drift"]
        ax.plot([0, 100], [0, 100], color="#9a9a96", lw=1, zorder=0)
        sens = side.sensitized_odor.astype(bool)
        ax.scatter(x[~sens], y[~sens], s=36, color=COLORS["anchor"], edgecolor="white", lw=1.5, label="other odors")
        ax.scatter(x[sens], y[sens], s=64, marker="D", color=COLORS["drift"], edgecolor="white", lw=1.5,
                   label="sensitized odor")
        for _, r in side[sens].iterrows():
            ax.annotate(r.cohort.split("-Sensitivity")[0],
                        (r[f"pct_responding_{phase}_anchor"], r[f"pct_responding_{phase}_drift"]),
                        xytext=(5, -10), textcoords="offset points", fontsize=8, color="#444")
        lim = max(5.0, float(np.nanmax([x.max(), y.max()])) + 5)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("% responding, fixed anchor (production)")
        ax.set_ylabel("% responding, drift-following anchor")
        ax.set_title(("pre-exposure" if phase == "pre" else "post-exposure") + " (window flies)")
        ax.legend(frameon=False, loc="upper left")
    fig.suptitle("% of flies responding per cohort × odor; on the line = no change")
    fig.savefig(OUT / "02_pct_responding_anchor_vs_drift.png", dpi=200)
    plt.close(fig)

    # 3. traces of the largest score changes where the anchor moved
    changed = both[both.anchor_moved & both.score_change.ne(0)]
    changed = changed.assign(absd=changed.score_change.abs()).sort_values(
        ["absd", "max_offset_px"], ascending=[False, False]).head(6)
    if len(changed):
        wide = {arm: lec._wide_traces(arm, changed) for arm in ("anchor", "drift")}
        fig, axes = plt.subplots(len(changed), 1, figsize=(9, 1.9 * len(changed)), sharex=True,
                                 constrained_layout=True)
        for ax, (_, r) in zip(np.atleast_1d(axes), changed.iterrows()):
            for arm in ("anchor", "drift"):
                num = r[f"fly_number_{arm}"]
                tr = wide[arm].get((r.dataset, r.fly, int(num), r.trial_label))
                if tr is not None:
                    ax.plot(np.arange(len(tr)) / 40.0, tr, lw=1.6, color=COLORS[arm],
                            label=f"{NAMES[arm]}: score {int(r[f'score_{arm}'])}")
            ax.axvspan(30, 60, color="#e9e7e1", zorder=0)
            ax.set_title(f"{r.dataset} · {r.fly} · {r.dataset_fly_label} (slot {int(r.fly_number_anchor)}) · "
                         f"{r.trial_label} · anchor moved {r.max_offset_px:.1f} px", fontsize=8.5, loc="left")
            ax.legend(frameon=False, fontsize=8, loc="upper right")
        np.atleast_1d(axes)[-1].set_xlabel("time (s); shaded = odor on")
        fig.suptitle("Largest score changes where the anchor moved: fixed vs drift-following")
        fig.savefig(OUT / "03_largest_changes_traces.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
