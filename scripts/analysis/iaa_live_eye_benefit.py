#!/usr/bin/env python3
"""IAA: where does the per-frame eye help, and where does it only add noise?

For every IAA trial x fly, from production's own tables (which hold both the
fixed eye ``x/y_class0`` and the per-frame eye ``eye_live_x/y``):

* ``held_offset_px`` -- median distance of the per-frame eye from the fixed eye
  over the trial (light frames excluded). Large = the eye really sat somewhere
  else, so every fixed-eye distance in that trial is off by about that much.
* ``jitter_px`` -- median frame-to-frame step of the per-frame eye (detection
  wobble).
* ``score_change`` -- per-frame eye held while light on (``livelight`` arm)
  minus production, from ``IAA-LightHold/trial_scores.csv``.

``helps``: held offset >= 3 px (a real move the fixed eye misses).
``no_benefit``: held offset < 1 px (the eye stayed put) but the score changed
anyway -- that change is wobble, not correction.

Writes ``Results/Figures/LiveEye-Comparison/IAA-LiveEye-Benefit/``.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "analysis" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cmp_ = _load("iaa_lighthold_compare")
lec = cmp_.lec
from fbpipe.utils.light_frames import light_frame_mask  # noqa: E402

DS = "IAA-Sensitivity-1-1"
PROD_TREE = lec.ARMS["anchor"]
SCORES = lec.RESULTS_ROOT / "IAA-LightHold" / "trial_scores.csv"
OUT = lec.RESULTS_ROOT / "IAA-LiveEye-Benefit"
BLUE, ORANGE, GREY = cmp_.BLUE, cmp_.ORANGE, "#8a8a86"
HELPS_PX, STILL_PX, N_SHOW = 3.0, 1.0, 6


def _trial_dir(batch: str, label: str) -> Path | None:
    m = re.match(r"(pretest|training|testing)_(\d+)", label)
    hits = sorted((PROD_TREE / DS / batch).glob(f"*_{m.group(1)}_{m.group(2)}")) if m else []
    return hits[0] if hits else None


def _metrics(batch: str, n: int, label: str) -> dict:
    t = _trial_dir(batch, label)
    if t is None:
        return {}
    p = t / f"{t.name}_fly{n}_distances.parquet"
    if not p.exists():
        return {}
    d = pd.read_parquet(p, columns=["frame", "x_class0", "y_class0", "eye_live_x", "eye_live_y"])
    try:
        meta = json.loads((t / "_trial_meta.json").read_text())
    except (OSError, ValueError):
        meta = {}
    lit = light_frame_mask(d.frame.to_numpy(), meta, 0.5, 1.0)
    off = np.hypot(d.eye_live_x - d.x_class0, d.eye_live_y - d.y_class0).to_numpy()
    off_dark = off[~lit & np.isfinite(off)]
    step = np.hypot(d.eye_live_x.diff(), d.eye_live_y.diff()).to_numpy()
    step = step[~lit & np.isfinite(step)]
    return {
        "trial_dir": t.name,
        "held_offset_px": float(np.median(off_dark)) if off_dark.size else np.nan,
        "offset_p90_px": float(np.percentile(off_dark, 90)) if off_dark.size else np.nan,
        "jitter_px": float(np.median(step)) if step.size else np.nan,
        "eye_seen_frac": float(np.isfinite(off).mean()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(SCORES)
    s = s[s.dataset == DS].dropna(subset=["score_production", "score_new"])
    rows = []
    for r in s.itertuples(index=False):
        rows.append({**r._asdict(), **_metrics(r.fly, int(r.fly_number), r.trial_label)})
    m = pd.DataFrame(rows)
    m["abs_score_change"] = (m.score_new - m.score_production).abs()
    m["group"] = np.select(
        [m.held_offset_px >= HELPS_PX, (m.held_offset_px < STILL_PX) & (m.abs_score_change > 0)],
        ["helps: eye really moved", "no benefit: eye still, score changed"], default="")
    cols = ["group", "dataset_fly", "fly", "fly_number", "trial_label", "held_offset_px", "offset_p90_px",
            "jitter_px", "eye_seen_frac", "score_production", "score_new", "responder_flip", "hand_flag"]
    m[[c for c in cols if c in m.columns]].sort_values(["group", "held_offset_px"], ascending=[True, False]).round(2) \
        .to_csv(OUT / "live_eye_benefit_by_trial.csv", index=False)

    # summary
    bins = pd.cut(m.held_offset_px, [-0.01, 1, 2, 3, 5, 1e9], labels=["<1 px", "1-2 px", "2-3 px", "3-5 px", ">5 px"])
    summ = m.groupby(bins, observed=False).agg(
        trials=("abs_score_change", "size"),
        pct_score_changed=("abs_score_change", lambda x: round(100 * (x > 0).mean(), 1)),
        flips=("responder_flip", lambda x: int((x.fillna("") != "").sum())),
        median_jitter_px=("jitter_px", "median"),
    ).round(2)
    summ.index.name = "eye held off the fixed point by"
    summ.to_csv(OUT / "summary_by_eye_offset.csv")
    print(summ.to_string())

    helps = m[m.group.str.startswith("helps")].sort_values("held_offset_px", ascending=False).head(N_SHOW)
    noben = m[m.group.str.startswith("no benefit")].sort_values(["abs_score_change", "jitter_px"],
                                                                 ascending=[False, False]).head(N_SHOW)
    print(f"\nhelps (held offset >= {HELPS_PX} px): {int((m.held_offset_px >= HELPS_PX).sum())} trials; "
          f"no benefit (still eye, score changed): {int(m.group.str.startswith('no benefit').sum())} trials")
    _figure(helps, noben)


def _figure(helps: pd.DataFrame, noben: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    n = max(len(helps), len(noben))
    fig = plt.figure(figsize=(16, 3.0 * n + 1.2))
    gs = fig.add_gridspec(n * 2, 2, height_ratios=[2.2, 1.0] * n, hspace=0.55, wspace=0.12,
                          top=1 - 1.0 / (3.0 * n + 1.2), bottom=0.03)
    for col, (df, head) in enumerate(((helps, "per-frame eye HELPS: the eye really moved"),
                                      (noben, "per-frame eye does NOT help: eye stayed put, score changed anyway"))):
        tr = {a: lec._wide_traces(a, df) for a in ("anchor", "livelight")} if len(df) else {}
        for i, (_, r) in enumerate(df.iterrows()):
            ax = fig.add_subplot(gs[2 * i, col])
            ax2 = fig.add_subplot(gs[2 * i + 1, col], sharex=ax)
            for a, c, sc, lab in (("anchor", BLUE, r.score_production, "fixed eye (production)"),
                                  ("livelight", ORANGE, r.score_new, "per-frame eye")):
                t = tr[a].get((r.dataset, r.fly, int(r.fly_number), r.trial_label))
                if t is not None:
                    ax.plot(np.arange(len(t)) / 40.0, t, color=c, lw=1.3, label=f"{lab}: score {int(sc)}")
            ax.axvspan(30, 60, color="#ecebe7", zorder=0, lw=0)
            ax.set_ylim(0, 105)
            ax.legend(frameon=False, fontsize=7.5, loc="upper right")
            ax.set_title(f"Fly {int(r.dataset_fly)} ({r.fly} slot {int(r.fly_number)}) · {r.trial_label} · "
                         f"eye held {r.held_offset_px:.1f} px off · jitter {r.jitter_px:.2f} px", loc="left", fontsize=8.5)
            if i == 0:
                fig.text(0.29 if col == 0 else 0.71, 1 - 0.55 / (3.0 * n + 1.2), head, ha="center",
                         fontsize=11, weight="bold")
            p = _trial_dir(r.fly, r.trial_label) / f"{_trial_dir(r.fly, r.trial_label).name}_fly{int(r.fly_number)}_distances.parquet"
            d = pd.read_parquet(p, columns=["frame", "x_class0", "y_class0", "eye_live_x", "eye_live_y"])
            off = np.hypot(d.eye_live_x - d.x_class0, d.eye_live_y - d.y_class0)
            ax2.plot(d.frame / 40.0, off, color=GREY, lw=0.8)
            ax2.axhline(0, color=BLUE, lw=1.5)
            ax2.set_ylabel("eye off\nfixed pt (px)", fontsize=7.5)
            ax2.set_ylim(0, max(6.0, float(np.nanpercentile(off, 99)) * 1.2))
            plt.setp(ax.get_xticklabels(), visible=False)
        if len(df):
            ax2.set_xlabel("time (s); shaded = odor on")
    fig.suptitle("IAA: per-frame eye vs fixed eye — trace % of fly's range (top) and how far the detected eye sat "
                 "from the fixed point (bottom)", y=1.0, fontsize=11)
    fig.savefig(OUT / "live_eye_helps_vs_not.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
