"""Skipping the RMS copies leaves every per-trial output as it was, over two nights.

reject_bad_proboscis, distance_stats, distance_normalize* and
detect_dropped_frames no longer read every ``RMS_calculations/updated_*`` copy
(see tests/test_fly_files_exclude_dirs.py). This replays the per-trial steps
those changes touch, in ``fbpipe.pipeline.ORDERED_STEPS`` order up to
rms_copy_filter, on twin trees: one with the implementation as it was before,
one with the current modules. Night 1 is forced; between the nights the gates
TIGHTEN (the case where a skipped copy could keep a stale gate or stale stats),
and night 2 runs forced or not. Every file must be byte-identical, except the
reports detect_dropped_frames used to write next to each RMS copy (no reader,
no longer written).

The tree has what makes the skip risky on real data: in-sync copies carrying
the OFM_State / angle columns later steps add, an orphan copy (its source is
gone), a nested batch under ``flagged/`` with its own .csv copies and the
``updated_updated_*`` copies of those, a legacy .csv copy, 3-fly trials (the
3-fly distance rule) and a Flybehavior2 batch after 2026-07-25 (rig gate
override).

The "before" side: ``_base_reject_process_fly_dir`` and
``_base_stats_process_fly_dir`` are verbatim copies of the old per-batch
bodies, and distance_normalize* / detect_dropped_frames (whose only change is
the ``exclude_dirs`` argument) run with the scan forced back to the default.
Every write is followed by a short pause on both sides, so the non-force mtime
comparisons (rms_copy_filter, detect_dropped_frames) never see a tie.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from fbpipe.config import (
    ForceSettings,
    ParallelSettings,
    ProboscisFilterSettings,
    RigGateOverride,
    Settings,
)
from fbpipe.steps import detect_dropped_frames, distance_stats, reject_bad_proboscis, rms_copy_filter
from fbpipe.utils.columns import EYE_CLASS, PROBOSCIS_DISTANCE_COL, find_proboscis_distance_column
from fbpipe.utils.distance_sanity import (
    csv_requires_three_fly_distance_sanitization,
    sanitize_proboscis_dataframe,
    sanitize_three_fly_distance_dataframe,
)
from fbpipe.utils.fly_files import iter_fly_distance_csvs
from fbpipe.utils.frozen_folders import iter_live_batch_dirs
from fbpipe.utils.rig_gates import apply_rig_gate_overrides
from fbpipe.utils.tables import read_schema_columns, read_table
from fbpipe.utils.tables import write_table as _write_table_now

log = logging.getLogger("test.per_trial_match_base")

OLD = 1_700_000_000  # every file's mtime between the nights


def write_table(df, path, **kwargs):
    """write_table, then a pause longer than the filesystem's mtime granularity."""
    out = _write_table_now(df, path, **kwargs)
    time.sleep(0.006)
    return out


# ---------------------------------------------------------------------------
# The implementation before the copies were skipped (verbatim)
# ---------------------------------------------------------------------------


def _base_reject_process_fly_dir(fly_dir: Path, cfg: Settings) -> Tuple[int, int, int]:
    cfg = apply_rig_gate_overrides(cfg, fly_dir)
    pf = cfg.proboscis_filter
    max_dist = float(pf.max_eye_prob_distance_px)
    max_jump = float(pf.max_jump_px)
    up_divisor = float(getattr(pf, "up_divisor", 4.0))

    files_modified = 0
    geo_total = 0
    vel_total = 0
    for csv_path, token, _ in iter_fly_distance_csvs(fly_dir, recursive=True):
        try:
            df = read_table(csv_path)
        except Exception as exc:
            log.warning("[PROB-FILTER] Failed to read %s: %s", csv_path, exc)
            continue

        cleaned, geo_count, vel_count = sanitize_proboscis_dataframe(
            df,
            max_distance_px=max_dist,
            max_jump_px=max_jump,
            up_divisor=up_divisor,
        )

        if geo_count or vel_count:
            write_table(cleaned, csv_path)
            files_modified += 1
            geo_total += geo_count
            vel_total += vel_count
            log.info(
                "[PROB-FILTER] %s: dropped %d out-of-radius, %d impossible-jump",
                csv_path.name,
                geo_count,
                vel_count,
            )
    return files_modified, geo_total, vel_total


def _stats_path_for_token(fly_dir: Path, token: str) -> Path:
    slot_label = token.replace("_distances", "")
    return fly_dir / f"{slot_label}_global_distance_stats_class_{EYE_CLASS}.json"


def _needs_stats_refresh(csv_path: Path, *, force_recompute: bool) -> bool:
    if force_recompute:
        return True
    try:
        columns = set(read_schema_columns(csv_path))
    except Exception:
        return True
    return "distance_percentage" not in columns


def _base_stats_process_fly_dir(fly_dir: Path, cfg: Settings) -> None:
    cfg = apply_rig_gate_overrides(cfg, fly_dir)
    force_recompute = bool(getattr(getattr(cfg, "force", None), "pipeline", False))

    print(f"[DIST] Inspecting fly directory: {fly_dir.name}")
    entries = list(iter_fly_distance_csvs(fly_dir, recursive=True))
    if not entries:
        return

    tokens_to_refresh: set[str] = set()
    if force_recompute:
        tokens_to_refresh = {token for _, token, _ in entries}
    else:
        for csv_path, token, _ in entries:
            stats_path = _stats_path_for_token(fly_dir, token)
            if not stats_path.exists():
                tokens_to_refresh.add(token)
                continue
            if csv_requires_three_fly_distance_sanitization(
                csv_path,
                cfg.three_fly_max_eye_prob_distance_px,
            ):
                tokens_to_refresh.add(token)
                continue
            if _needs_stats_refresh(csv_path, force_recompute=force_recompute):
                tokens_to_refresh.add(token)

    if not tokens_to_refresh:
        print(f"[DIST] {fly_dir.name}: all slot stats are up-to-date; skipping.")
        return

    slot_ranges: Dict[str, Tuple[float, float]] = {}
    for csv_path, token, _ in entries:
        if token not in tokens_to_refresh:
            continue
        print(
            f"[DIST] Reading {csv_path.name} for slot '{token}' in {fly_dir.name}"
        )
        try:
            df = read_table(csv_path)
        except Exception as exc:
            print(
                f"[DIST] Failed to read {csv_path.name} (slot {token}): {exc}"
            )
            continue

        df, sanitized_count = sanitize_three_fly_distance_dataframe(
            df,
            csv_path,
            cfg.three_fly_max_eye_prob_distance_px,
        )
        if sanitized_count:
            write_table(df, csv_path)
            print(
                f"[DIST] Sanitized {sanitized_count} over-limit 3-fly rows in {csv_path.name} "
                f"(>{cfg.three_fly_max_eye_prob_distance_px}px)."
            )

        dist_col = find_proboscis_distance_column(df)
        if dist_col is None:
            print(
                f"[DIST] No proboscis distance column found in {csv_path.name};"
                f" expected aliases such as '{PROBOSCIS_DISTANCE_COL}' or 'proboscis_distance'."
            )
            continue

        distances = pd.to_numeric(df[dist_col], errors="coerce")
        mask = distances.between(cfg.class2_min, cfg.class2_max, inclusive="both")
        vals = distances[mask]
        if vals.empty:
            print(
                f"[DIST] Slot {token} in {csv_path.name} had no values within "
                f"[{cfg.class2_min}, {cfg.class2_max}]; skipping."
            )
            continue

        local_min = float(vals.min())
        local_max = float(vals.max())
        print(
            f"[DIST] Slot {token}: local min={local_min:.3f}, max={local_max:.3f}"
        )
        current = slot_ranges.get(token)
        if current is None:
            slot_ranges[token] = (local_min, local_max)
        else:
            slot_ranges[token] = (min(current[0], local_min), max(current[1], local_max))

    if not slot_ranges:
        print(f"[DIST] No in-range distances for {fly_dir.name}")
        return

    for token, (gmin, gmax) in sorted(slot_ranges.items(), key=lambda item: item[0]):
        slot_label = token.replace("_distances", "")
        # Modification #1: Add fly_max_distance and effective_max_threshold
        stats = {
            "global_min": gmin,
            "global_max": gmax,
            "fly_max_distance": gmax,  # Actual max before 95px floor
            "effective_max_threshold": 95.0  # Minimum threshold for normalization
        }
        stats_path = fly_dir / f"{slot_label}_global_distance_stats_class_{EYE_CLASS}.json"
        with open(stats_path, "w", encoding="utf-8") as fp:
            json.dump(stats, fp)
        print(f"[DIST] {fly_dir.name}/{slot_label}: min={gmin:.3f} max={gmax:.3f} fly_max={gmax:.3f}")
        print(f"[DIST] Wrote stats JSON → {stats_path}")


def _default_scan(base, *, recursive=True, exclude_dirs=()):
    """iter_fly_distance_csvs as the old call sites used it (no exclusion)."""
    return iter_fly_distance_csvs(base, recursive=recursive)


# ---------------------------------------------------------------------------
# The tree
# ---------------------------------------------------------------------------


def _trial_df(seed: int, *, reach: float, n: int = 160, gaps: bool = False) -> pd.DataFrame:
    """A per-fly trial table as yolo_infer writes it, with points that only the
    night-2 gates reject: two extensions (one up to ``reach`` px), short upward
    flicks, single-frame jumps, and missing detections."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    r = 55.0 + rng.normal(0.0, 1.5, n)
    r += (reach - 55.0) * np.exp(-0.5 * ((t - 50.0) / 6.0) ** 2)
    r += (0.6 * reach - 33.0) * np.exp(-0.5 * ((t - 115.0) / 5.0) ** 2)
    theta = np.deg2rad(70.0 + 15.0 * np.sin(t / 11.0))  # below the eye (dy > 0)
    dx, dy = r * np.cos(theta), r * np.sin(theta)
    for i in (20, 21, 90):  # upward flick: inside 160/4 px, outside 120/4 px
        dx[i], dy[i] = 5.0, -35.0
    for i in (30, 75, 140):  # single-frame jump of ~65 px: inside 80, outside 50
        dx[i] += 65.0
    ex = 640.0 + rng.normal(0.0, 0.3, n)
    ey = 300.0 + rng.normal(0.0, 0.3, n)
    px, py = ex + dx, ey + dy
    missing = rng.choice(n, 6, replace=False)
    px[missing] = np.nan
    py[missing] = np.nan
    frames = np.arange(n)
    df = pd.DataFrame(
        {
            "frame": frames,
            "timestamp": frames / 40.0,
            "track_id_class0": np.full(n, 1001),
            "x_class0": ex,
            "y_class0": ey,
            "corners_class0": np.full(n, np.nan),
            "track_id_class1": np.full(n, 1),
            "x_class1": px,
            "y_class1": py,
            "corners_class1": np.full(n, np.nan),
            "x_anchor": np.full(n, 1080.0),
            "y_anchor": np.full(n, 540.0),
            "distance_0_1": np.hypot(px - ex, py - ey),
            "distance_0_anchor": np.hypot(1080.0 - ex, 540.0 - ey),
            "angle_deg_c0_c1_vs_anchor": np.degrees(theta),
        }
    )
    if gaps:  # dropped frames
        df = df.drop(index=list(range(60, 72))).reset_index(drop=True)
    return df


def _batch(root: Path, name: str, trials: Dict[str, Tuple[int, float]], *, seed: int, prefix: str) -> Path:
    batch = root / name
    for k, (trial, (flies, reach)) in enumerate(trials.items()):
        tdir = batch / f"{prefix}_{trial}"
        for slot in range(1, flies + 1):
            df = _trial_df(seed + 97 * k + slot, reach=reach - 6.0 * slot, gaps=(k == 0 and slot == 1))
            write_table(df, tdir / f"{tdir.name}_fly{slot}_distances.parquet")
    return batch


def _build_tree(root: Path) -> None:
    # A 3-fly batch on a rig without overrides.
    a = _batch(
        root, "october_05_batch_2",
        {"testing_1": (3, 150.0), "training_1": (3, 115.0)}, seed=1, prefix="october_05_batch_2",
    )
    # Flybehavior2 after 2026-07-25: the rig_gate_overrides entry applies.
    b = _batch(
        root, "august_09_batch_1_rig_2",
        {"testing_1": (4, 250.0), "testing_2": (4, 150.0)}, seed=2, prefix="august_09_batch_1",
    )
    (b / "session_metadata.txt").write_text("Session Metadata\nHost: Flybehavior2 | OS: Linux\n")
    (b / "output_august_09_batch_1_testing_1_ACV_20260809_142012.csv").write_text("Frame,ActiveOFM\n0,off\n")
    # Hex-Control-24-0.01's flagged/ folder: a whole batch one level down,
    # with that batch's own RMS copies as .csv (as on disk).
    nested = _batch(
        root / "flagged", "may_20_batch_2",
        {"testing_8": (2, 140.0)}, seed=3, prefix="may_20_batch_1",
    )
    for src in sorted(nested.rglob("*_distances.parquet")):
        df = read_table(src)
        copy = nested / "RMS_calculations" / f"updated_{src.stem}.csv"
        copy.parent.mkdir(parents=True, exist_ok=True)
        df[["frame", "timestamp", "x_class0", "y_class0", "x_class1", "y_class1"]].to_csv(copy, index=False)
    # An orphan copy (its trial folder is gone) and a legacy .csv copy whose
    # source exists.
    orphan_src = _trial_df(11, reach=150.0)
    orphan = orphan_src[["frame", "timestamp", "x_class0", "y_class0", "x_class1", "y_class1"]].copy()
    orphan["distance_percentage_0_1"] = 20.0
    write_table(orphan, a / "RMS_calculations" / "updated_october_05_batch_2_testing_9_fly1_distances.parquet")
    legacy = _trial_df(12, reach=200.0)[["frame", "timestamp", "x_class0", "y_class0", "x_class1", "y_class1"]]
    legacy.to_csv(a / "RMS_calculations" / "updated_october_05_batch_2_training_1_fly3_distances.csv", index=False)


def _after_night_one(root: Path) -> None:
    """What the steps after rms_copy_filter leave behind (update_ofm_state, then
    combine's angle columns), then every mtime set to the same old stamp."""
    for copy in sorted(root.rglob("RMS_calculations/updated_*.parquet")):
        df = read_table(copy)
        df["OFM_State"] = np.where(df["frame"] < 40, "before", "during")
        df["angle_ARB_deg"] = np.arange(len(df), dtype=float)
        _write_table_now(df, copy)
    for p in root.rglob("*"):
        os.utime(p, (OLD, OLD))


def _cfg(root: Path, *, force: bool, tight: bool) -> Settings:
    # The spatial gate and the 3-fly gate are distinct keys; they get different
    # values on both nights so a step reading the wrong one cannot pass. The
    # 3-fly gate is the tighter one, so the 3-fly rule in distance_stats and
    # distance_normalize has rows to blank on both nights.
    gate, three_fly, jump, ov = (
        (120.0, 112.0, 50.0, (200.0, 205.0, 190.0)) if tight else (160.0, 140.0, 80.0, (280.0, 285.0, 270.0))
    )
    return Settings(
        model_path="",
        main_directories=[str(root)],
        auto_discover_flagged=False,
        allow_cpu=True,
        force=ForceSettings(pipeline=force),
        parallel=ParallelSettings(enabled=False),
        class2_min=10.0,
        class2_max=gate,
        proboscis_filter=ProboscisFilterSettings(
            enabled=True, max_eye_prob_distance_px=gate, up_divisor=4.0, max_jump_px=jump
        ),
        three_fly_max_eye_prob_distance_px=three_fly,
        rig_gate_overrides=(
            RigGateOverride(
                host="Flybehavior2",
                after=dt.date(2026, 7, 25),
                max_eye_prob_distance_px=ov[0],
                class2_max=ov[1],
                three_fly_max_eye_prob_distance_px=ov[2],
            ),
        ),
    )


_STEP_MODULES = (reject_bad_proboscis, distance_stats, detect_dropped_frames, rms_copy_filter)


def _night(root: Path, cfg: Settings, normalize, *, before: bool, monkeypatch) -> None:
    with monkeypatch.context() as mp:
        for mod in (*_STEP_MODULES, normalize):
            if hasattr(mod, "write_table"):
                mp.setattr(mod, "write_table", write_table)
        if before:
            for fly_dir in iter_live_batch_dirs(cfg, root):
                _base_reject_process_fly_dir(fly_dir, cfg)
            for fly_dir in iter_live_batch_dirs(cfg, root):
                _base_stats_process_fly_dir(fly_dir, cfg)
            mp.setattr(normalize, "iter_fly_distance_csvs", _default_scan)
            mp.setattr(detect_dropped_frames, "iter_fly_distance_csvs", _default_scan)
        else:
            reject_bad_proboscis.main(cfg)
            distance_stats.main(cfg)
        normalize.main(cfg)
        detect_dropped_frames.main(cfg)
        rms_copy_filter.main(cfg)


def _is_rms_side_report(rel: Path) -> bool:
    return "RMS_calculations" in rel.parts[:-1] and rel.name.endswith("_dropped_frames.txt")


def _digests(root: Path) -> Dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.md5(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and not _is_rms_side_report(p.relative_to(root))
    }


@pytest.mark.parametrize(
    "normalize_name",
    ["distance_normalize", "distance_normalize_gpu", "distance_normalize_ultra"],
)
@pytest.mark.parametrize("night_two_forced", [True, False], ids=["forced", "not-forced"])
def test_two_nights_with_a_gate_tightening_match_the_old_steps(
    tmp_path, monkeypatch, normalize_name, night_two_forced
):
    normalize = importlib.import_module(f"fbpipe.steps.{normalize_name}")
    trees = {}
    for side in ("before", "now"):
        root = tmp_path / side / "Hex-Control-24-0.01"
        _build_tree(root)
        _night(root, _cfg(root, force=True, tight=False), normalize, before=(side == "before"), monkeypatch=monkeypatch)
        _after_night_one(root)
        _night(
            root, _cfg(root, force=night_two_forced, tight=True), normalize,
            before=(side == "before"), monkeypatch=monkeypatch,
        )
        trees[side] = root

    before, now = _digests(trees["before"]), _digests(trees["now"])
    assert sorted(before) == sorted(now)
    differing = sorted(rel for rel in before if before[rel] != now[rel])
    assert differing == []

    # The scenario is not vacuous: night 2 re-gated the orphan and the nested
    # .csv copies (rewritten as .parquet), re-derived the stats, and the
    # tighter gate blanked points of the trial tables.
    root = trees["now"]
    a = root / "october_05_batch_2"
    orphan = read_table(a / "RMS_calculations" / "updated_october_05_batch_2_testing_9_fly1_distances.parquet")
    assert orphan["x_class1"].isna().sum() > 6
    nested_rms = root / "flagged" / "may_20_batch_2" / "RMS_calculations"
    assert sorted(p.suffix for p in nested_rms.glob("updated_*_distances.*")) == [".parquet", ".parquet"]
    stats = json.loads((a / "fly1_global_distance_stats_class_0.json").read_text())
    assert stats["global_max"] <= 120.0
    assert read_table(a / "october_05_batch_2_testing_1" / "october_05_batch_2_testing_1_fly1_distances.parquet")[
        "distance_0_1"
    ].max() <= 120.0
