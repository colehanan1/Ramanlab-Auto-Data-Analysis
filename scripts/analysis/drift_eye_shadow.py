#!/usr/bin/env python3
"""Rewrite a comparison tree so every distance is measured from the
drift-following eye anchor (``fbpipe.utils.eye_reanchor``).

Per batch: trials are put in recording order (the sidecar timestamp), each
fly's warm-up misses borrow the next trial's anchor, and every trial is
re-measured with sustained-drift steps and common-mode (camera) shifts.
Thresholds come from ``eye_qc`` in config_new.yaml -- the same rule the
eye-drift QC flags with -- and the window from each trial's own fps.

Only ever writes under a ``flys_DriftEye`` tree. Writes, per dataset root:
``_drift_eye_summary.csv`` (one row per trial x fly) and
``_drift_eye_events.csv`` (one row per anchor move).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src")]

from fbpipe.utils import eye_reanchor as er  # noqa: E402

SHADOW_DIR_NAME = "flys_DriftEye"
CONFIG = REPO / "config" / "config_new.yaml"
TRIAL_RE = re.compile(r"(pretest|training|testing)_(\d+)$")
FLY_RE = re.compile(r"_fly(\d+)_distances\.parquet$")
STAMP_RE = re.compile(r"_(\d{8})_(\d{6})\.(txt|csv)$")


def eye_qc_thresholds(config: Path = CONFIG) -> tuple[float, float, float]:
    """(max_drift_px, sustain_s, fps_default) as the pipeline reads them."""
    from fbpipe.config import load_settings

    s = load_settings(config)
    return float(s.eye_qc.max_drift_px), float(s.eye_qc.sustain_s), float(s.fps_default)


def trial_dirs(batch: Path) -> list[Path]:
    return [p for p in batch.iterdir() if p.is_dir() and TRIAL_RE.search(p.name)]


def trial_meta(trial: Path) -> dict:
    try:
        return json.loads((trial / "_trial_meta.json").read_text())
    except (OSError, ValueError):
        return {}


def recorded_at(trial: Path) -> str:
    """Sortable recording time: the sidecar's _YYYYMMDD_HHMMSS stamp."""
    m = STAMP_RE.search(str(trial_meta(trial).get("sidecar_txt") or ""))
    if m:
        return m.group(1) + m.group(2)
    return f"~{trial.stat().st_mtime:020.6f}"   # sorts after every stamped trial


#: Under the stimulus light YOLO places the eye off where it is (median 2.1 /
#: 0.9 / 8.2 px on rigs 1 / 2 / 3, 224 fly x light-only trials) while the head
#: stays put; the eye point is back within 0.5 s of light-off. The anchor keeps
#: the eye position it had before the light from light-on to 1 s after light-off.
#: The logged light-on can trail the video (3Oct october_11_batch_2 testing_9:
#: frames lit from 29.86 s, logged 30.00 s), so the hold starts 0.5 s early.
LIGHT_HOLD_AFTER_S = 1.0
LIGHT_HOLD_BEFORE_S = 0.5


def light_hold(meta: dict, fps: float):
    """(first, last) frame the stimulus light is on, from the trial metadata
    (fbpipe.utils.trial_metadata parses it from the sensors CSV); None when
    the trial had no light."""
    on, off = meta.get("light_on_s"), meta.get("light_off_s")
    if on is None:
        return None
    if off is None:
        off = float(meta.get("n_frames") or 0) / fps
    first = max(0, int((float(on) - LIGHT_HOLD_BEFORE_S) * fps))
    return first, int(np.ceil((float(off) + LIGHT_HOLD_AFTER_S) * fps))


def fly_tables(trial: Path) -> dict[int, Path]:
    out = {}
    for p in trial.glob(f"{trial.name}_fly*_distances.parquet"):
        m = FLY_RE.search(p.name)
        if m:
            out[int(m.group(1))] = p
    return dict(sorted(out.items()))


def process_batch(batch: Path, *, max_drift_px: float, sustain_s: float, fps_default: float):
    trials = sorted(trial_dirs(batch), key=recorded_at)
    loaded: dict[Path, dict[int, pd.DataFrame]] = {}
    for t in trials:
        paths = fly_tables(t)
        if paths:
            loaded[t] = {f: pd.read_parquet(p).sort_values("frame", kind="stable").reset_index(drop=True)
                         for f, p in paths.items()}
    order = [t for t in trials if t in loaded]

    # Warm-up misses: per fly, which trials saw the eye in their first frames.
    initial: dict[Path, dict[int, tuple[float, float]]] = {t: {} for t in order}
    found_map: dict[tuple[Path, int], bool] = {}
    flies = sorted({f for t in order for f in loaded[t]})
    for f in flies:
        ts = [t for t in order if f in loaded[t]]
        anchors, found = {}, {}
        for t in ts:
            df = loaded[t][f]
            bx, by = er._base(df)
            lx, ly = er._live(df)
            anchors[t] = (float(bx[0]), float(by[0]))
            found[t] = er.warmup_found(lx, ly, er.SCAN_FRAMES)
            found_map[(t, f)] = found[t]
        for t, off in er.warmup_offsets(ts, anchors, found).items():
            initial[t][f] = off

    rows, events = [], []
    for t in order:
        meta = trial_meta(t)
        fps = float(meta.get("fps") or fps_default)
        window = max(1, int(round(sustain_s * fps)))
        hold = light_hold(meta, fps)
        new, summary = er.reanchor_trial(loaded[t], window=window, max_drift_px=max_drift_px,
                                         initial_offsets=initial[t], hold=hold)
        for f, df in new.items():
            df.to_parquet(fly_tables(t)[f], index=False)
        for s in summary:
            evs = s.pop("events")
            rows.append({"batch": batch.name, "trial": t.name, "recorded_at": recorded_at(t),
                         "trial_type": meta.get("trial_type"), "odor": meta.get("odor"),
                         "warmup_found": found_map.get((t, s["fly"])), "window_frames": window,
                         "light_hold": None if hold is None else f"{hold[0]}-{hold[1]}", **s})
            for e in evs:
                events.append({"batch": batch.name, "trial": t.name, "fly": s["fly"],
                               "time_s": e["frame"] / fps, **e})
    return rows, events


def apply_tree(dataset_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_root = Path(dataset_root).resolve()
    if SHADOW_DIR_NAME not in dataset_root.parts:
        raise SystemExit(f"refusing to rewrite {dataset_root}: not under a {SHADOW_DIR_NAME} tree")
    max_drift_px, sustain_s, fps_default = eye_qc_thresholds()
    rows, events = [], []
    for batch in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        r, e = process_batch(batch, max_drift_px=max_drift_px, sustain_s=sustain_s, fps_default=fps_default)
        rows += r
        events += e
    summary = pd.DataFrame(rows)
    ev = pd.DataFrame(events)
    summary.to_csv(dataset_root / "_drift_eye_summary.csv", index=False)
    ev.to_csv(dataset_root / "_drift_eye_events.csv", index=False)
    return summary, ev


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_roots", nargs="+", type=Path)
    args = ap.parse_args(argv)
    for root in args.dataset_roots:
        s, e = apply_tree(root)
        if s.empty:
            print(f"[drift-eye] {root.name}: no trial tables")
            continue
        print(
            f"[drift-eye] {root.name}: {len(s)} trial-flies; warm-up misses {int((~s.warmup_found.astype(bool)).sum())}; "
            f"anchor moved in {int((s.n_steps > 0).sum())} trial-flies ({len(e)} moves, "
            f"median {e.step_px.median() if len(e) else 0:.1f} px); first move a group move in {int(s.first_move_group.sum())} trial-flies; follow-up moves {int(s.n_follow_steps.sum())}"
        )


if __name__ == "__main__":
    main()
