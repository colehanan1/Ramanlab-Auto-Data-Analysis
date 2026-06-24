"""Determinism test for the parallel refactor of distance_stats.

Verifies that running main() with parallel=enabled produces byte-identical
JSON sidecar outputs to running it serially on an identical fixture.

Run with:
    pytest tests/test_distance_stats_parallel.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fbpipe.config import ForceSettings, ParallelSettings, Settings
from fbpipe.steps.distance_stats import main
from fbpipe.utils.tables import write_table


# ---------------------------------------------------------------------------
# Realistic fixture columns (per assignment spec)
# ---------------------------------------------------------------------------

_FIXTURE_COLS = [
    "frame",
    "timestamp",
    "x_class0",
    "y_class0",
    "x_class1",
    "y_class1",
    "corners_class1",
    "distance_0_1",
    "angle_deg_c0_c1_vs_anchor",
]


def _make_fly_df(n: int = 80, seed: int = 0) -> pd.DataFrame:
    """Return a minimal DataFrame that mimics a per-fly distance file.

    Values for distance_0_1 are chosen to fall inside [70, 250] (cfg defaults)
    so stats are actually computed.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "frame": np.arange(n),
            "timestamp": np.linspace(0.0, n / 40.0, n),
            "x_class0": rng.uniform(0, 1280, n),
            "y_class0": rng.uniform(0, 720, n),
            "x_class1": rng.uniform(0, 1280, n),
            "y_class1": rng.uniform(0, 720, n),
            "corners_class1": [f"[[{i},{i+1}]]" for i in range(n)],
            # Keep in [70, 250] so the stage always produces a stats JSON
            "distance_0_1": rng.uniform(80.0, 200.0, n),
            "angle_deg_c0_c1_vs_anchor": rng.uniform(-180, 180, n),
        }
    )


def _build_fixture(root: Path, n_flies: int = 3) -> list[Path]:
    """Create n_flies fly directories under root, each with one distance file."""
    fly_dirs = []
    for i in range(n_flies):
        fly_dir = root / f"fly{i + 1}"
        fly_dir.mkdir(parents=True)
        df = _make_fly_df(n=80, seed=i)
        # Write as parquet — iter_fly_distance_csvs accepts both .csv and .parquet
        write_table(df, fly_dir / "fly1_distances.parquet")
        fly_dirs.append(fly_dir)
    return fly_dirs


def _make_cfg(root: Path, *, parallel_enabled: bool, n_jobs: int = 2) -> Settings:
    return Settings(
        model_path="",
        main_directories=[str(root)],
        force=ForceSettings(pipeline=True),  # force=True so stats are always recomputed
        parallel=ParallelSettings(enabled=parallel_enabled, n_jobs=n_jobs),
        auto_discover_flagged=False,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_json_outputs(root: Path) -> dict[str, dict]:
    """Return {relative_path_str: parsed_json} for every .json under root."""
    result = {}
    for p in sorted(root.rglob("*.json")):
        key = str(p.relative_to(root))
        with open(p, encoding="utf-8") as fh:
            result[key] = json.load(fh)
    return result


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------

def test_parallel_matches_serial(tmp_path: Path) -> None:
    """Serial and parallel runs produce identical JSON sidecar outputs."""
    serial_root = tmp_path / "serial"
    parallel_root = tmp_path / "parallel"

    # Build two identical fixture trees
    _build_fixture(serial_root)
    _build_fixture(parallel_root)

    # Serial run
    main(_make_cfg(serial_root, parallel_enabled=False))

    # Parallel run (n_jobs=2 to exercise the multiprocess path even with 3 items)
    main(_make_cfg(parallel_root, parallel_enabled=True, n_jobs=2))

    serial_out = _collect_json_outputs(serial_root)
    parallel_out = _collect_json_outputs(parallel_root)

    assert set(serial_out.keys()) == set(parallel_out.keys()), (
        f"Different JSON files produced.\n"
        f"Serial: {sorted(serial_out)}\n"
        f"Parallel: {sorted(parallel_out)}"
    )

    for rel_path in sorted(serial_out):
        s = serial_out[rel_path]
        p = parallel_out[rel_path]
        assert s.keys() == p.keys(), (
            f"{rel_path}: key mismatch — serial={set(s)}, parallel={set(p)}"
        )
        for key in s:
            assert s[key] == pytest.approx(p[key], rel=1e-9), (
                f"{rel_path}[{key!r}]: serial={s[key]} vs parallel={p[key]}"
            )


def test_stats_json_content(tmp_path: Path) -> None:
    """Each stats JSON must contain the four expected keys."""
    root = tmp_path / "root"
    _build_fixture(root, n_flies=2)
    main(_make_cfg(root, parallel_enabled=False))

    jsons = list(root.rglob("*.json"))
    assert jsons, "Expected at least one stats JSON to be written"
    for p in jsons:
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
        for key in ("global_min", "global_max", "fly_max_distance", "effective_max_threshold"):
            assert key in data, f"Missing key '{key}' in {p.name}"


def test_skip_up_to_date(tmp_path: Path) -> None:
    """When force=False and JSON already exists with no refresh needed, no rewrite occurs."""
    root = tmp_path / "root"
    fly_dir = root / "fly1"
    fly_dir.mkdir(parents=True)

    df = _make_fly_df(n=20, seed=7)
    # Add distance_percentage so _needs_stats_refresh returns False
    df["distance_percentage"] = df["distance_0_1"] / 250.0
    write_table(df, fly_dir / "fly1_distances.parquet")

    # Pre-write the stats JSON so the skip-logic path fires
    stats_path = fly_dir / "fly1_global_distance_stats_class_0.json"
    pre_stats = {"global_min": 99.0, "global_max": 199.0, "fly_max_distance": 199.0, "effective_max_threshold": 95.0}
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(pre_stats, fh)
    mtime_before = stats_path.stat().st_mtime

    cfg = Settings(
        model_path="",
        main_directories=[str(root)],
        force=ForceSettings(pipeline=False),
        parallel=ParallelSettings(enabled=False),
        auto_discover_flagged=False,
    )
    main(cfg)

    mtime_after = stats_path.stat().st_mtime
    assert mtime_before == mtime_after, "Stats JSON was unexpectedly rewritten despite being up-to-date"
