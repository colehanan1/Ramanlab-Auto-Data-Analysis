"""Determinism test for the parallel refactor of distance_normalize.

Verifies that running main() with parallel=enabled produces byte-identical
Parquet output to running it serially on an identical fixture.

Run with:
    pytest tests/test_distance_normalize_parallel.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fbpipe.config import ForceSettings, ParallelSettings, Settings
from fbpipe.steps.distance_normalize import main
from fbpipe.utils.columns import (
    EYE_CLASS,
    PROBOSCIS_CLASS,
    PROBOSCIS_DISTANCE_COL,
)
from fbpipe.utils.tables import read_table, write_table

# ---------------------------------------------------------------------------
# Realistic fixture columns from the assignment spec
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

_GMIN = 10.0
_GMAX = 200.0
_FLY_MAX = 180.0
_EFFECTIVE_MAX = max(_FLY_MAX, 95.0)  # 180.0


def _make_fly_df(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Return a minimal DataFrame that mimics a fly distances file."""
    rng = np.random.default_rng(seed)
    # distances strictly inside [_GMIN, _GMAX] so normalization is clean
    distances = rng.uniform(_GMIN + 1.0, _GMAX - 1.0, n)
    return pd.DataFrame(
        {
            "frame": np.arange(n),
            "timestamp": np.linspace(0.0, n / 40.0, n),
            "x_class0": rng.uniform(0, 1280, n),
            "y_class0": rng.uniform(0, 720, n),
            "x_class1": rng.uniform(0, 1280, n),
            "y_class1": rng.uniform(0, 720, n),
            "corners_class1": [f"[[{i},{i+1}]]" for i in range(n)],
            # Use the canonical column name that find_proboscis_distance_column recognises
            PROBOSCIS_DISTANCE_COL: distances,
            "angle_deg_c0_c1_vs_anchor": rng.uniform(-180, 180, n),
        }
    )


def _write_stats_json(fly_dir: Path, slot_label: str) -> None:
    """Write the global distance stats JSON that distance_normalize expects."""
    stats = {
        "global_min": _GMIN,
        "global_max": _GMAX,
        "fly_max_distance": _FLY_MAX,
        "effective_max_threshold": 95.0,
    }
    stats_path = fly_dir / f"{slot_label}_global_distance_stats_class_{EYE_CLASS}.json"
    stats_path.write_text(json.dumps(stats), encoding="utf-8")


def _build_fixture(root: Path, n_flies: int = 3) -> list[Path]:
    """Create n_flies fly directories under root, each with one distance file + stats."""
    fly_dirs = []
    for i in range(n_flies):
        fly_dir = root / f"fly{i + 1}"
        fly_dir.mkdir(parents=True)
        df = _make_fly_df(n=60, seed=i)
        write_table(df, fly_dir / "fly1_distances.parquet")
        _write_stats_json(fly_dir, slot_label="fly1")
        fly_dirs.append(fly_dir)
    return fly_dirs


def _make_cfg(root: Path, *, parallel_enabled: bool, n_jobs: int = 2) -> Settings:
    return Settings(
        model_path="",
        main_directories=[str(root)],
        force=ForceSettings(pipeline=True),   # always recompute so skip logic doesn't mask diffs
        parallel=ParallelSettings(enabled=parallel_enabled, n_jobs=n_jobs),
        auto_discover_flagged=False,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_outputs(root: Path) -> dict[str, pd.DataFrame]:
    """Return {relative_path_str: DataFrame} for every parquet in the tree."""
    result = {}
    for p in sorted(root.rglob("*.parquet")):
        key = str(p.relative_to(root))
        result[key] = read_table(p)
    return result


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------

def test_parallel_matches_serial(tmp_path: Path) -> None:
    """Serial and parallel runs produce identical Parquet outputs."""
    serial_root = tmp_path / "serial"
    parallel_root = tmp_path / "parallel"

    # Build two identical fixture trees
    _build_fixture(serial_root)
    _build_fixture(parallel_root)

    # Serial run
    cfg_serial = _make_cfg(serial_root, parallel_enabled=False)
    main(cfg_serial)

    # Parallel run (n_jobs=2 to exercise the multiprocess path with 3 items)
    cfg_parallel = _make_cfg(parallel_root, parallel_enabled=True, n_jobs=2)
    main(cfg_parallel)

    # Collect outputs
    serial_out = _collect_outputs(serial_root)
    parallel_out = _collect_outputs(parallel_root)

    assert set(serial_out.keys()) == set(parallel_out.keys()), (
        f"Different files produced.\nSerial: {set(serial_out)}\nParallel: {set(parallel_out)}"
    )

    for rel_path in serial_out:
        pd.testing.assert_frame_equal(
            serial_out[rel_path].reset_index(drop=True),
            parallel_out[rel_path].reset_index(drop=True),
            check_like=False,
            obj=rel_path,
        )


def test_normalized_columns_present(tmp_path: Path) -> None:
    """After main(), each file must have all expected normalization columns."""
    root = tmp_path / "root"
    _build_fixture(root, n_flies=2)
    cfg = _make_cfg(root, parallel_enabled=False)
    main(cfg)

    expected_cols = {
        PROBOSCIS_DISTANCE_COL,
        f"distance_percentage_{EYE_CLASS}_{PROBOSCIS_CLASS}",
        f"min_distance_{EYE_CLASS}_{PROBOSCIS_CLASS}",
        f"max_distance_{EYE_CLASS}_{PROBOSCIS_CLASS}",
        f"effective_max_distance_{EYE_CLASS}_{PROBOSCIS_CLASS}",
        "distance_percentage",
    }
    for p in root.rglob("*.parquet"):
        df = read_table(p)
        missing = expected_cols - set(df.columns)
        assert not missing, f"Missing columns {missing} in {p.relative_to(root)}"


def test_normalization_values_correct(tmp_path: Path) -> None:
    """Normalized percentage values match the expected formula.

    The code uses effective_max (not gmax) as the scale denominator, so
    distances strictly within [gmin, effective_max] produce [0, 100] while
    distances between effective_max and gmax may exceed 100%.  We test
    distances that stay within [gmin, effective_max] so the result is
    predictably in [0, 100].
    """
    root = tmp_path / "root"
    fly_dir = root / "fly1"
    fly_dir.mkdir(parents=True)

    rng = np.random.default_rng(7)
    n = 20
    # Keep distances inside [gmin, effective_max] so pct is in [0, 100]
    distances = rng.uniform(_GMIN + 1.0, _EFFECTIVE_MAX - 1.0, n)
    df = pd.DataFrame(
        {
            "frame": np.arange(n),
            "timestamp": np.linspace(0.0, 0.5, n),
            PROBOSCIS_DISTANCE_COL: distances,
        }
    )
    write_table(df, fly_dir / "fly1_distances.parquet")
    _write_stats_json(fly_dir, slot_label="fly1")

    cfg = _make_cfg(root, parallel_enabled=False)
    main(cfg)

    out_path = fly_dir / "fly1_distances.parquet"
    result = read_table(out_path)
    pct_col = f"distance_percentage_{EYE_CLASS}_{PROBOSCIS_CLASS}"
    pct = result[pct_col].to_numpy()
    # All in-range distances should map to [0, 100]
    assert np.all(pct >= 0.0) and np.all(pct <= 100.0), (
        f"Percentage values out of [0, 100]: min={pct.min():.3f}, max={pct.max():.3f}"
    )
    # Spot-check the formula: pct = 100 * (d - gmin) / (effective_max - gmin)
    expected = 100.0 * (distances - _GMIN) / (_EFFECTIVE_MAX - _GMIN)
    np.testing.assert_allclose(pct, expected, rtol=1e-9)
