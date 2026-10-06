"""Synthetic combine inputs shaped like a real flys_New batch.

Shared by the combine read-memo and parallel-combine tests. The layout copies
``Hex-Sensitivity-24-0.1/august_28_batch_1_rig_3`` as it sat on disk on
2026-10-05::

    <batch>/<prefix>_<phase>_<n>/<prefix>_<phase>_<n>_fly<k>_distances.parquet
    <batch>/<prefix>_<phase>_<n>/<prefix>_<phase>_<n>_distances_merged.parquet
    <batch>/RMS_calculations/updated_<prefix>_<phase>_<n>_fly<k>_distances.parquet
    <batch>/output_<prefix>_<phase>_<n>_<Odor>_<YYYYMMDD>_<HHMMSS>.csv
    <batch>/session_metadata.txt

``prefix`` is the batch name without its ``_rig_N`` suffix, as on the rigs.
The ``updated_*`` copies carry no angle columns (rms_copy_filter rewrites them
without, every night), so combine has to write those back -- the path the
read memo must invalidate on.
"""

from __future__ import annotations

import hashlib
import re
import zlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

ODORS = ("EthylButyrate", "Hexanol", "Linalool", "3-Octonol", "Citral", "Benzaldehyde")
DEFAULT_PHASES: tuple[tuple[str, int], ...] = (("pretest", 2), ("training", 2), ("testing", 2))


def _prefix(batch: str) -> str:
    return re.sub(r"_rig_\d+$", "", batch)


def _rng(*parts: object) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def _fly_columns(rng: np.random.Generator, frames: int) -> dict[str, np.ndarray]:
    t = np.arange(frames, dtype=float)
    eye_x = rng.uniform(300.0, 800.0)
    eye_y = rng.uniform(80.0, 600.0)
    # Resting ~40 px, one extension bump mid-trial, a little jitter.
    dist = 40.0 + 45.0 * np.exp(-(((t - frames * 0.55) / (frames * 0.08)) ** 2))
    dist = dist + rng.normal(0.0, 1.5, frames)
    theta = rng.uniform(0.0, 2.0 * np.pi) + 0.4 * np.sin(t / 23.0)
    px = eye_x + dist * np.cos(theta)
    py = eye_y + dist * np.sin(theta)
    dropped = rng.random(frames) < 0.03
    px[dropped] = np.nan
    py[dropped] = np.nan
    d01 = np.hypot(px - eye_x, py - eye_y)
    lo, hi = np.nanmin(d01), np.nanmax(d01)
    pct = ((d01 - lo) / (hi - lo) * 100.0).astype(np.float32)
    return {
        "frame": np.arange(frames, dtype=np.int64),
        "timestamp": t / 40.0,
        "x_class0": np.full(frames, eye_x),
        "y_class0": np.full(frames, eye_y),
        "x_class1": px,
        "y_class1": py,
        "distance_0_1": d01,
        "distance_percentage_0_1": pct,
    }


def write_batch(
    root: Path,
    batch: str,
    *,
    flies: Sequence[int] = (1, 2, 3),
    phases: Iterable[tuple[str, int]] = DEFAULT_PHASES,
    frames: int = 240,
    host: str | None = None,
    rec_date: str = "20260828",
) -> Path:
    """Write one synthetic batch under ``root`` and return its directory."""

    batch_dir = root / batch
    rms = batch_dir / "RMS_calculations"
    rms.mkdir(parents=True, exist_ok=True)
    prefix = _prefix(batch)
    clock = 110000
    for phase, count in phases:
        for idx in range(1, count + 1):
            odor = ODORS[(idx - 1 + len(phase)) % len(ODORS)]
            trial = f"{prefix}_{phase}_{idx}"
            trial_dir = batch_dir / trial
            trial_dir.mkdir(parents=True, exist_ok=True)
            merged: dict[str, np.ndarray] = {}
            for fly in flies:
                cols = _fly_columns(_rng(root.name, batch, trial, fly), frames)
                full = pd.DataFrame(cols)
                full["distance_percentage"] = full["distance_percentage_0_1"]
                full.to_parquet(trial_dir / f"{trial}_fly{fly}_distances.parquet", index=False)
                ofm = np.where(np.arange(frames) < frames // 3, "before", "during")
                rms_df = full[
                    ["frame", "timestamp", "x_class0", "y_class0",
                     "x_class1", "y_class1", "distance_percentage_0_1"]
                ].copy()
                rms_df["OFM_State"] = ofm.astype(object)
                rms_df.to_parquet(rms / f"updated_{trial}_fly{fly}_distances.parquet", index=False)
                merged[f"eye_{fly - 1}_x"] = cols["x_class0"]
                merged[f"eye_{fly - 1}_y"] = cols["y_class0"]
                merged[f"cls8_{fly - 1}_x"] = cols["x_class1"]
                merged[f"cls8_{fly - 1}_y"] = cols["y_class1"]
            pd.DataFrame(
                {"frame": np.arange(frames, dtype=np.int64), **merged}
            ).to_parquet(trial_dir / f"{trial}_distances_merged.parquet", index=False)
            stamp = f"{rec_date}_{clock:06d}"
            clock += 530
            (batch_dir / f"output_{trial}_{odor}_{stamp}.csv").write_text(
                "Timestamp,ActiveOFM\n", encoding="utf-8"
            )
    meta = ["Session Metadata", f"Born (approx): {rec_date[:4]}-{rec_date[4:6]}-01"]
    if host:
        meta.append(f"Host: {host} | OS: Linux 6.12.34+rpt-rpi-2712")
    (batch_dir / "session_metadata.txt").write_text("\n".join(meta) + "\n", encoding="utf-8")
    return batch_dir


def tree_digest(root: Path) -> dict[str, str]:
    """md5 of every file under ``root``, keyed by its path relative to ``root``."""

    return {
        str(p.relative_to(root)): hashlib.md5(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
