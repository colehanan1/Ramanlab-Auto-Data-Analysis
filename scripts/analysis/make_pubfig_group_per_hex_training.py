#!/usr/bin/env python3
"""Publication figure: one 4-fly group on one unrewarded CS+ test trial.

Group selection (see ``find_groups``) over the ``config_new`` wide table and
``model_predictions.csv``:

* trained (non-control) dataset, testing trial, no light, not frozen;
* folder holds >= 4 flies and none is in ``flagged-flys-truth.csv`` (weak/dead)
  or the non-reactive flagged list;
* exactly 2 of the 4 flies respond (score >= 2) and at least one scores 4-5.

Only Hex-Training-24-0.1 / august_13_batch_2_rig_2 passes (2026-09-23), on
testing_1_hexanol (the CS+) and testing_3_citral; the CS+ trial is drawn.

Frames come from the yolo_infer distance-annotated video, so the tracked eye
and proboscis boxes and the distance line being scored are visible.

Top: full arena at fly 3's peak, plus pre-odor vs peak crops of both
responders. Bottom: all four flies' PER traces on one axis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config/config_new.yaml"
CSV_DIR = Path("/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys")
WIDE = CSV_DIR / "all_envelope_rows_wide_combined_base.parquet"
PREDICTIONS = CSV_DIR / "model_predictions.csv"
FLY_TRUTH = CSV_DIR / "flagged-flys-truth.csv"
NON_REACTIVE = CSV_DIR / "all_envelope_rows_wide_combined_base_flagged_flies.txt"

DATASET = "Hex-Training-24-0.1"
FOLDER = "august_13_batch_2_rig_2"
TRIAL = "testing_1_hexanol"
_BATCH_DIR = Path("/securedstorage/DATAsec/cole/Data-secured-New/Hex-Training-24-0.1/august_13_batch_2_rig_2")
# Both are rendered, one figure each. yolo = yolo_infer overlay (eye box,
# proboscis box, eye-proboscis distance line, gate arc); raw = camera video in grayscale.
VIDEOS = {
    "yolo": _BATCH_DIR / "august_13_batch_2_testing_1/august_13_batch_2_testing_1_distance_annotated.mp4",
    "raw": _BATCH_DIR / "output_august_13_batch_2_testing_1_Hexanol_20260813_170818.mp4",
}
OUT_DIR = Path("/home/ramanlab/Documents/cole/Results/Figures/pubfig_group_per_hex_training")

REST_FRAME = 1000  # 25 s, before odor onset
# fly -> (peak frame, crop box x0, y0, size) in video pixels. Peak frames are
# the argmax of each fly's trace inside the odor window.
RESPONDERS = {3: (2038, 470, 530, 260), 2: (2386, 395, 300, 260)}
ARENA_CROP = (235, 0, 925, 1080)  # x0, y0, x1, y1: drops the vial, burned-in label and anchor end

# Okabe-Ito: responders saturated, non-responders neutral.
FLY_COLORS = {3: "#D55E00", 2: "#0072B2", 1: "#8C8C8C", 4: "#4D4D4D"}
ODOR_COLOR = "#009E73"


def load_alive_exclusions() -> set[tuple[str, str, int]]:
    truth = pd.read_csv(FLY_TRUTH)
    truth = truth.iloc[:, :3]
    truth.columns = ["dataset", "fly", "fly_number"]
    bad = set(zip(truth.dataset, truth.fly, truth.fly_number.astype(int)))
    nr = pd.read_csv(NON_REACTIVE, comment="#", names=["dataset", "fly", "fly_number", "lo", "hi"])
    bad |= set(zip(nr.dataset, nr.fly, nr.fly_number.astype(int)))
    return bad


def find_groups() -> pd.DataFrame:
    cols = ["dataset", "fly", "fly_number", "trial_type", "trial_label", "trial_light_on_s", "frozen"]
    wide = pd.read_parquet(WIDE, columns=cols)
    preds = pd.read_csv(PREDICTIONS)[["dataset", "fly", "fly_number", "trial_label", "score"]]
    bad = load_alive_exclusions()
    flies = wide.groupby(["dataset", "fly"]).fly_number.unique()

    m = wide.merge(preds, on=["dataset", "fly", "fly_number", "trial_label"])
    m = m[
        m.dataset.str.contains("Training")
        & (m.trial_type == "testing")
        & m.trial_light_on_s.isna()
        & ~m.frozen
        & ~m.trial_label.str.contains("light")
    ]
    rows = []
    for (ds, fly, label), d in m.groupby(["dataset", "fly", "trial_label"]):
        members = flies[(ds, fly)]
        if len(members) < 4 or d.fly_number.nunique() < 4:
            continue
        if any((ds, fly, int(n)) in bad for n in members):
            continue
        scores = d.score.to_numpy()
        if (scores >= 2).sum() == 2 and scores.max() >= 4:
            rows.append({"dataset": ds, "fly": fly, "trial_label": label,
                         "scores": dict(zip(d.fly_number.astype(int), scores.astype(int)))})
    return pd.DataFrame(rows)


def load_traces() -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    wide = pd.read_parquet(
        WIDE, filters=[("dataset", "==", DATASET), ("fly", "==", FOLDER), ("trial_label", "==", TRIAL)]
    )
    traces = {}
    for _, r in wide.iterrows():
        n = int(r.trace_len)
        traces[int(r.fly_number)] = r[[f"dir_val_{i}" for i in range(n)]].to_numpy(float)
    return traces, wide.set_index("fly_number")


def read_frames(style: str, indices: list[int]) -> dict[int, np.ndarray]:
    """RGB frames for the yolo style; contrast-stretched grayscale in [0, 1] for raw."""
    cap = cv2.VideoCapture(str(VIDEOS[style]))
    frames = {}
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {idx} from {VIDEOS[style]}")
        frames[idx] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB if style == "yolo" else cv2.COLOR_BGR2GRAY)
    cap.release()
    if style == "raw":
        lo, hi = np.percentile(frames[indices[0]], [1, 99.7])
        frames = {k: np.clip((v.astype(float) - lo) / (hi - lo), 0, 1) for k, v in frames.items()}
    return frames


def show(ax, img: np.ndarray) -> None:
    if img.ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(img)


def style_image_axes(ax, color=None, lw=2.0):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(color is not None)
        if color is not None:
            s.set_edgecolor(color)
            s.set_linewidth(lw)


def setup_fonts():
    for p in Path("/usr/share/fonts/truetype/msttcorefonts").glob("[Aa]rial*.ttf"):
        font_manager.fontManager.addfont(str(p))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.dpi": 600,
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list-groups", action="store_true", help="print every qualifying group and exit")
    args = ap.parse_args()

    groups = find_groups()
    if args.list_groups:
        print(groups.to_string())
        return
    chosen = groups[(groups.fly == FOLDER) & (groups.trial_label == TRIAL) & (groups.dataset == DATASET)]
    if chosen.empty:
        raise SystemExit(f"{DATASET}/{FOLDER}/{TRIAL} no longer passes the selection:\n{groups}")
    scores = chosen.iloc[0].scores

    cfg = yaml.safe_load(CONFIG.read_text())
    latency = float(cfg["analysis"]["combined"]["combine"]["odor_latency_s"])

    traces, meta = load_traces()
    fps = float(meta.fps.iloc[0])
    odor_on = float(meta.trial_odor_on_s.iloc[0]) + latency
    odor_off = float(meta.trial_odor_off_s.iloc[0]) + latency

    setup_fonts()
    for style in VIDEOS:
        render(style, scores, traces, fps, odor_on, odor_off)


def render(style, scores, traces, fps, odor_on, odor_off) -> None:
    frames = read_frames(style, [REST_FRAME] + [spec[0] for spec in RESPONDERS.values()])
    rest = frames[REST_FRAME]
    peaks = {fly: frames[spec[0]] for fly, spec in RESPONDERS.items()}

    fig = plt.figure(figsize=(7.2, 6.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.28)
    # A and the crop grid get their own gap so the Fly/Score row labels sit clear of A.
    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.6], wspace=0.32)
    crops = top[1].subgridspec(2, 2, wspace=0.06, hspace=0.1)

    # A: arena at fly 3's peak
    ax_a = fig.add_subplot(top[0])
    x0, y0, x1, y1 = ARENA_CROP
    first_peak_fly = 3
    show(ax_a, peaks[first_peak_fly][y0:y1, x0:x1])
    style_image_axes(ax_a)
    for fly, (_, cx, cy, size) in RESPONDERS.items():
        # The overlay is already busy with colour, so the yolo boxes are dashed white.
        ax_a.add_patch(Rectangle((cx - x0, cy - y0), size, size, fill=False,
                                 **(dict(edgecolor="white", linewidth=1.0, linestyle=(0, (4, 2)))
                                    if style == "yolo" else dict(edgecolor=FLY_COLORS[fly], linewidth=1.3))))
    head_dir = (Path("/home/ramanlab/Documents/cole/Data/flys_New") / DATASET / FOLDER
                / "august_13_batch_2_testing_1")
    for n in sorted(traces):
        xy = pd.read_parquet(head_dir / f"august_13_batch_2_testing_1_fly{n}_distances.parquet",
                             columns=["x_class0", "y_class0"]).to_numpy()
        hy = float(np.nanmedian(xy[:, 1]))
        ax_a.text(24, hy - y0, str(n), color="white", fontsize=8, fontweight="bold",
                  ha="center", va="center", zorder=5,
                  bbox=dict(boxstyle="circle,pad=0.3", fc=FLY_COLORS[n], ec="white", lw=0.6))
    ax_a.text(0.03, 0.02, f"t = {RESPONDERS[first_peak_fly][0] / fps:.1f} s", transform=ax_a.transAxes,
              color="white", fontsize=7, ha="left", va="bottom")

    # B-E: rest vs peak crops per responder
    for row, (fly, (pk, cx, cy, size)) in enumerate(RESPONDERS.items()):
        for col, (img, label) in enumerate([(rest, f"Pre-odor  {REST_FRAME / fps:.0f} s"),
                                            (peaks[fly], f"Peak  {pk / fps:.1f} s")]):
            ax = fig.add_subplot(crops[row, col])
            show(ax, img[cy:cy + size, cx:cx + size])
            style_image_axes(ax, FLY_COLORS[fly] if col == 1 else "#BBBBBB", lw=1.6 if col == 1 else 0.8)
            ax.text(0.04, 0.04, label, transform=ax.transAxes, color="white", fontsize=7,
                    ha="left", va="bottom")
            if col == 0:
                ax.text(-0.06, 0.5, f"Fly {fly}\nScore {scores[fly]}", transform=ax.transAxes,
                        color=FLY_COLORS[fly], fontsize=8, fontweight="bold",
                        ha="right", va="center", rotation=0)

    # F: all four traces
    ax_t = fig.add_subplot(outer[1])
    ax_t.axvspan(odor_on, odor_off, color=ODOR_COLOR, alpha=0.15, lw=0)
    order = sorted(traces, key=lambda n: (scores[n] >= 2, scores[n]))  # responders drawn on top
    for n in order:
        y = traces[n]
        t = np.arange(y.size) / fps
        resp = scores[n] >= 2
        ax_t.plot(t, y, color=FLY_COLORS[n], lw=1.3 if resp else 0.9, alpha=1.0 if resp else 0.85,
                  label=f"Fly {n} (Score {scores[n]})", zorder=3 if resp else 2)
    ymax = max(np.nanmax(v) for v in traces.values())
    for fly, (pk, *_rest) in RESPONDERS.items():
        ax_t.plot(pk / fps, traces[fly][pk] + 0.04 * ymax, marker="v", ms=5,
                  color=FLY_COLORS[fly], zorder=4, clip_on=False)
    ax_t.axvline(REST_FRAME / fps, color="#BBBBBB", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax_t.set_xlim(0, t[-1])
    ax_t.set_ylim(0, 105)
    ax_t.set_xlabel("Time (s)")
    ax_t.set_ylabel("Proboscis extension\n(distance × angle, % of fly max)")
    handles, labels = ax_t.get_legend_handles_labels()
    idx = sorted(range(len(labels)), key=lambda i: labels[i])
    handles = [handles[i] for i in idx] + [Patch(fc=ODOR_COLOR, alpha=0.15, label="Hexanol")]
    ax_t.legend(handles=handles, loc="upper left", frameon=False, fontsize=7, ncol=1,
                handlelength=1.6, borderaxespad=0.2)

    for ax, letter in [(ax_a, "A"), (fig.axes[1], "B"), (ax_t, "C")]:
        ax.text(-0.02 if ax is not ax_t else -0.085, 1.0 if ax is not ax_t else 1.04, letter,
                transform=ax.transAxes, fontsize=11, fontweight="bold", ha="right", va="bottom")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"{DATASET}_{FOLDER}_{TRIAL}_group_per_{style}"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {stem}.png  scores={scores}")


if __name__ == "__main__":
    main()
