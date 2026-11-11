#!/usr/bin/env python3
"""Measure each proboscis distance from the eye detected in *that* frame.

Production (``yolo_infer``) freezes every fly's eye at an anchor chosen from
the first frames of the batch and measures every proboscis distance and angle
from that fixed point. It also records, per frame, where the eye was actually
detected (``eye_live_x/y``, the eye-QC columns) -- recorded, never measured
from.

This module rewrites a per-fly YOLO table so the eye coordinate follows the
live detection, falling back to the frozen anchor on frames where the eye was
not found. It only ever writes into a separate comparison tree; production
tables are read, never modified.

Same geometry as ``yolo_infer``: distance = Euclidean eye->proboscis, angle =
the unsigned angle between eye->proboscis and eye->odor-anchor.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fbpipe.utils.eye_reanchor import remeasure  # noqa: E402,F401
from fbpipe.utils.live_eye import apply_live_eye_held, stimulus_hold as _stimulus_hold  # noqa: E402

from fbpipe.utils.live_eye import (  # noqa: E402  (the pipeline step uses the same code)
    EYE_X, EYE_Y, LIVE_X, LIVE_Y, PROB_X, PROB_Y, apply_live_eye,
)


#: The only tree this module may rewrite. Production lives in flys_New.
SHADOW_DIR_NAME = "flys_LiveEye"


def _trial_tables(root: Path):
    """Per-fly YOLO tables in trial folders (not the RMS / envelope copies)."""
    for path in sorted(root.glob("*/*/*_fly[0-9]_distances.parquet")):
        if path.name.startswith("updated_"):
            continue
        if path.parent.name in ("RMS_calculations", "angle_distance_rms_envelope"):
            continue
        yield path


def apply_tree(dataset_root: Path) -> list:
    """Rewrite every trial table under one dataset of the comparison tree."""
    dataset_root = Path(dataset_root).resolve()
    if SHADOW_DIR_NAME not in dataset_root.parts:
        raise SystemExit(f"refusing to rewrite {dataset_root}: not under a {SHADOW_DIR_NAME} tree")
    summary = []
    for path in _trial_tables(dataset_root):
        df = pd.read_parquet(path)
        out = apply_live_eye(df)
        before = df["distance_0_1"].astype(float)
        after = out["distance_0_1"].astype(float)
        both = before.notna() & after.notna()
        out.to_parquet(path, index=False)
        summary.append({
            "file": str(path.relative_to(dataset_root)),
            "n_frames": len(out),
            "live_eye_frac": float(out["eye_from_live"].mean()) if len(out) else np.nan,
            "abs_change_median_px": float((after[both] - before[both]).abs().median()) if both.any() else np.nan,
            "abs_change_max_px": float((after[both] - before[both]).abs().max()) if both.any() else np.nan,
        })
    return summary


# --------------------------------------------------------------------------
# Per-frame eye, held while the odor (or light) is on
# --------------------------------------------------------------------------

#: The trees the held variants may rewrite (odor+light hold, light-only hold).
HOLD_DIR_NAME = "flys_LiveEyeOdorHold"
LIGHT_HOLD_DIR_NAME = "flys_LiveEyeLightHold"
HOLD_BEFORE_S = 0.5   # the logged light-on trailed the video by 0.14 s
HOLD_AFTER_S = 1.0    # the eye point is back within 0.5 s of light-off
PRE_HOLD_S = 1.0      # where the eye sat: median over this long before the hold


def stimulus_hold(meta: dict, odor: bool = True):
    """(first, last) frame of the odor window -- and the light window, when
    there is one -- widened by HOLD_BEFORE_S / HOLD_AFTER_S; None if neither.
    ``odor=False``: the light window only."""
    return _stimulus_hold(meta, odor=odor, before_s=HOLD_BEFORE_S, after_s=HOLD_AFTER_S)


def apply_tree_held(dataset_root: Path, odor: bool = True) -> list:
    """Rewrite every trial table under one dataset of a held-variant tree."""
    dataset_root = Path(dataset_root).resolve()
    tree = HOLD_DIR_NAME if odor else LIGHT_HOLD_DIR_NAME
    if tree not in dataset_root.parts:
        raise SystemExit(f"refusing to rewrite {dataset_root}: not under a {tree} tree")
    summary = []
    for path in _trial_tables(dataset_root):
        try:
            meta = json.loads((path.parent / "_trial_meta.json").read_text())
        except (OSError, ValueError):
            meta = {}
        hold = stimulus_hold(meta, odor=odor)
        pre = int(round(PRE_HOLD_S * float(meta.get("fps") or 40.0)))
        df = pd.read_parquet(path)
        out = apply_live_eye_held(df, hold, pre)
        before = df["distance_0_1"].astype(float)
        after = out["distance_0_1"].astype(float)
        both = before.notna() & after.notna()
        out.to_parquet(path, index=False)
        summary.append({
            "file": str(path.relative_to(dataset_root)),
            "hold": None if hold is None else f"{hold[0]}-{hold[1]}",
            "held_frac": float(out["eye_held"].mean()) if len(out) else np.nan,
            "abs_change_median_px": float((after[both] - before[both]).abs().median()) if both.any() else np.nan,
        })
    return summary


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_roots", nargs="+", type=Path)
    ap.add_argument("--hold-stimulus", action="store_true",
                    help="keep the eye where it sat before the odor/light while it is on")
    ap.add_argument("--hold-light", action="store_true",
                    help="keep the eye where it sat before the light while it is on (odor: per frame)")
    args = ap.parse_args(argv)
    if args.hold_stimulus or args.hold_light:
        for root in args.dataset_roots:
            summary = pd.DataFrame(apply_tree_held(root, odor=not args.hold_light))
            summary.to_csv(root / "_live_eye_held_summary.csv", index=False)
            print(f"[live-eye-held] {root.name}: {len(summary)} tables, held "
                  f"{summary.held_frac.mean():.1%} of frames, |Δdistance| median "
                  f"{summary.abs_change_median_px.median():.2f} px")
        return
    for root in args.dataset_roots:
        summary = pd.DataFrame(apply_tree(root))
        summary.to_csv(root / "_live_eye_transform_summary.csv", index=False)
        print(
            f"[live-eye] {root.name}: {len(summary)} tables, live eye in "
            f"{summary.live_eye_frac.mean():.1%} of frames, |Δdistance| median "
            f"{summary.abs_change_median_px.median():.2f} px, max {summary.abs_change_max_px.max():.1f} px"
        )


if __name__ == "__main__":
    main()
