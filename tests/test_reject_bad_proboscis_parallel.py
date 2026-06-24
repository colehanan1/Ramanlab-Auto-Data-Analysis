"""Determinism test: reject_bad_proboscis produces byte-identical output
whether the inner fly_dir loop runs serially or with parallel_map (joblib).

The test builds two identical fixture trees (one for the serial run, one for
the parallel run), calls main() on each, then reads back the produced Parquet
files and asserts they are equal via pandas.testing.assert_frame_equal.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fbpipe.config import ForceSettings, ParallelSettings, ProboscisFilterSettings, Settings
from fbpipe.steps.reject_bad_proboscis import main
from fbpipe.utils.tables import read_table, write_table, resolve_existing


# ---------------------------------------------------------------------------
# Shared columns matching the assignment brief
# ---------------------------------------------------------------------------

_COLS = [
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


def _make_fly_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Return a tiny synthetic per-fly distance DataFrame.

    Includes one deliberately out-of-radius proboscis and one impossible jump
    so the filter actually does something (non-trivial write).
    """
    rng = np.random.default_rng(seed)

    # Eye positions (class 0)
    x_eye = rng.uniform(400, 600, n)
    y_eye = rng.uniform(400, 600, n)

    # Proboscis (class 1) — mostly nearby the eye
    x_prob = x_eye + rng.uniform(-40, 40, n)
    y_prob = y_eye + rng.uniform(0, 40, n)  # downward (within gate)

    # Row 5: proboscis jumps 200px above the eye -> OUTSIDE geometry gate (up_divisor=4 -> 37.5px up limit)
    x_prob[5] = x_eye[5]
    y_prob[5] = y_eye[5] - 200.0  # far above eye

    # Row 10: impossible velocity jump
    x_prob[10] = x_prob[9] + 500.0  # far away in x

    distances = np.sqrt((x_prob - x_eye) ** 2 + (y_prob - y_eye) ** 2)

    return pd.DataFrame({
        "frame": list(range(n)),
        "timestamp": [float(i) / 40.0 for i in range(n)],
        "x_class0": x_eye.tolist(),
        "y_class0": y_eye.tolist(),
        "x_class1": x_prob.tolist(),
        "y_class1": y_prob.tolist(),
        "corners_class1": ["[[0,0],[1,1]]"] * n,
        "distance_0_1": distances.tolist(),
        "angle_deg_c0_c1_vs_anchor": rng.uniform(0, 360, n).tolist(),
    })


def _build_fixture(root: Path, n_fly_dirs: int = 3) -> None:
    """Create n_fly_dirs under root, each with one fly1_distances.parquet."""
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n_fly_dirs):
        fly_dir = root / f"fly_dir_{i}"
        fly_dir.mkdir()
        df = _make_fly_df(seed=i)
        # Write using write_table so the Parquet is produced in the same way
        # that the pipeline would produce it.
        write_table(df, fly_dir / "fly1_distances.csv")


def _make_cfg(root: Path, *, parallel_enabled: bool, n_jobs: int = 2) -> Settings:
    return Settings(
        model_path="",
        main_directories=[str(root)],
        force=ForceSettings(pipeline=False),
        parallel=ParallelSettings(enabled=parallel_enabled, n_jobs=n_jobs),
        proboscis_filter=ProboscisFilterSettings(
            enabled=True,
            max_eye_prob_distance_px=150.0,
            up_divisor=4.0,
            max_jump_px=80.0,
        ),
    )


def _collect_outputs(root: Path) -> dict[str, pd.DataFrame]:
    """Return {relative_str_path: DataFrame} for every Parquet file under root."""
    result: dict[str, pd.DataFrame] = {}
    for p in sorted(root.rglob("*.parquet")):
        rel = str(p.relative_to(root))
        result[rel] = read_table(p)
    return result


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------

def test_serial_parallel_outputs_identical(tmp_path):
    """Serial and parallel runs produce identical Parquet files."""
    serial_root = tmp_path / "serial"
    parallel_root = tmp_path / "parallel"

    _build_fixture(serial_root, n_fly_dirs=3)
    _build_fixture(parallel_root, n_fly_dirs=3)

    cfg_serial = _make_cfg(serial_root, parallel_enabled=False)
    cfg_parallel = _make_cfg(parallel_root, parallel_enabled=True, n_jobs=2)

    main(cfg_serial)
    main(cfg_parallel)

    serial_outputs = _collect_outputs(serial_root)
    parallel_outputs = _collect_outputs(parallel_root)

    assert set(serial_outputs.keys()) == set(parallel_outputs.keys()), (
        f"File sets differ.\nSerial:   {sorted(serial_outputs.keys())}\n"
        f"Parallel: {sorted(parallel_outputs.keys())}"
    )

    for rel_path in sorted(serial_outputs.keys()):
        pd.testing.assert_frame_equal(
            serial_outputs[rel_path].reset_index(drop=True),
            parallel_outputs[rel_path].reset_index(drop=True),
            check_like=False,
            obj=f"file={rel_path}",
        )


def test_filter_actually_modifies_files(tmp_path):
    """Sanity-check: the filter removes at least one point in each fly_dir."""
    root = tmp_path / "root"
    _build_fixture(root, n_fly_dirs=2)

    # Capture pre-run values so we can confirm change.
    pre_run: dict[str, pd.DataFrame] = {}
    for p in sorted(root.rglob("*.parquet")):
        pre_run[str(p.relative_to(root))] = read_table(p)

    cfg = _make_cfg(root, parallel_enabled=False)
    main(cfg)

    for rel, pre_df in pre_run.items():
        post_df = read_table(root / rel)
        # At least one NaN should have been introduced (geometry or velocity gate)
        pre_nan = pre_df.isnull().sum().sum()
        post_nan = post_df.isnull().sum().sum()
        assert post_nan > pre_nan, (
            f"{rel}: expected NaN count to increase after filter "
            f"(was {pre_nan}, still {post_nan})"
        )


def test_disabled_filter_leaves_files_unchanged(tmp_path):
    """When proboscis_filter.enabled=False main() returns early and writes nothing."""
    root = tmp_path / "root"
    _build_fixture(root, n_fly_dirs=2)

    # Capture mtimes before.
    mtimes_before = {p: p.stat().st_mtime_ns for p in root.rglob("*.parquet")}

    cfg = Settings(
        model_path="",
        main_directories=[str(root)],
        force=ForceSettings(pipeline=False),
        parallel=ParallelSettings(enabled=False),
        proboscis_filter=ProboscisFilterSettings(enabled=False),
    )
    main(cfg)

    for p, mtime_before in mtimes_before.items():
        assert p.stat().st_mtime_ns == mtime_before, (
            f"{p} was modified even though proboscis_filter.enabled=False"
        )
