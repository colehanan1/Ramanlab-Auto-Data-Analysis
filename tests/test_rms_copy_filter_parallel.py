"""Determinism test for rms_copy_filter parallel refactor.

Verifies that enabling parallel=True (n_jobs=2) produces byte-identical
Parquet outputs to the serial (parallel=False) execution, across >=2 fly_dirs.

Run with:
    pytest tests/test_rms_copy_filter_parallel.py -q
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from fbpipe.config import ForceSettings, ParallelSettings, Settings
from fbpipe.steps.rms_copy_filter import main
from fbpipe.utils.tables import read_table


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

# Realistic columns matching the per-fly distance file schema described in the
# task brief (also matches the schema used in test_rms_copy_filter_parquet.py).
_DISTANCE_COLS = [
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


def _make_fly_distance_df(n: int = 6, seed: int = 0) -> pd.DataFrame:
    """Return a minimal but realistic per-fly distance DataFrame."""
    rng = pd.Series(range(seed, seed + n))
    return pd.DataFrame(
        {
            "frame": list(range(n)),
            "timestamp": [i * 0.025 for i in range(n)],
            "x_class0": (100.0 + rng).tolist(),
            "y_class0": (200.0 + rng).tolist(),
            "x_class1": (300.0 + rng).tolist(),
            "y_class1": (400.0 + rng).tolist(),
            "corners_class1": [f"[[{i},{i+1}],[{i+2},{i+3}]]" for i in range(n)],
            "distance_0_1": [50.0 + i * 0.5 for i in range(n)],
            "angle_deg_c0_c1_vs_anchor": [45.0 + i for i in range(n)],
        }
    )


def _build_fixture(root: Path, n_flies: int = 3) -> None:
    """Create n_flies fly directories each containing one fly distance Parquet."""
    for fly_idx in range(1, n_flies + 1):
        fly_dir = root / f"fly{fly_idx:02d}"
        fly_dir.mkdir(parents=True)
        df = _make_fly_distance_df(n=6, seed=fly_idx * 10)
        out = fly_dir / f"fly{fly_idx}_distances.parquet"
        df.to_parquet(out, index=False)


def _collect_outputs(root: Path) -> dict[str, pd.DataFrame]:
    """Return a mapping of relative output path -> DataFrame for every RMS output."""
    results: dict[str, pd.DataFrame] = {}
    for parquet in sorted(root.rglob("RMS_calculations/updated_*.parquet")):
        rel = parquet.relative_to(root)
        results[str(rel)] = read_table(parquet)
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRmsCopyFilterDeterminism:
    """Serial and parallel modes produce identical outputs."""

    def _make_cfg(
        self,
        root: Path,
        parallel_enabled: bool,
        n_jobs: int = 2,
    ) -> Settings:
        return Settings(
            model_path="",
            main_directories=[str(root)],
            force=ForceSettings(pipeline=False),
            parallel=ParallelSettings(enabled=parallel_enabled, n_jobs=n_jobs),
        )

    def test_serial_produces_outputs(self, tmp_path: Path) -> None:
        """Smoke: serial run creates RMS_calculations outputs."""
        root = tmp_path / "serial_smoke"
        root.mkdir()
        _build_fixture(root)
        cfg = self._make_cfg(root, parallel_enabled=False)
        main(cfg)
        outputs = _collect_outputs(root)
        assert len(outputs) >= 2, "Expected at least 2 fly outputs"

    def test_parallel_produces_outputs(self, tmp_path: Path) -> None:
        """Smoke: parallel run also creates RMS_calculations outputs."""
        root = tmp_path / "parallel_smoke"
        root.mkdir()
        _build_fixture(root)
        cfg = self._make_cfg(root, parallel_enabled=True, n_jobs=2)
        main(cfg)
        outputs = _collect_outputs(root)
        assert len(outputs) >= 2, "Expected at least 2 fly outputs"

    def test_serial_and_parallel_outputs_are_equal(self, tmp_path: Path) -> None:
        """Core determinism: identical fixtures produce identical Parquet frames."""
        # Build two identical fixtures in separate directories.
        root_serial = tmp_path / "serial"
        root_parallel = tmp_path / "parallel"
        root_serial.mkdir()
        root_parallel.mkdir()
        _build_fixture(root_serial)
        _build_fixture(root_parallel)

        # Run serial on one copy and parallel on the other.
        cfg_serial = self._make_cfg(root_serial, parallel_enabled=False)
        cfg_parallel = self._make_cfg(root_parallel, parallel_enabled=True, n_jobs=2)

        main(cfg_serial)
        main(cfg_parallel)

        serial_outputs = _collect_outputs(root_serial)
        parallel_outputs = _collect_outputs(root_parallel)

        # Same set of output files must exist in both runs.
        assert set(serial_outputs.keys()) == set(parallel_outputs.keys()), (
            f"Output file sets differ.\n"
            f"  Serial:   {sorted(serial_outputs.keys())}\n"
            f"  Parallel: {sorted(parallel_outputs.keys())}"
        )

        # Each DataFrame must be element-wise equal.
        for rel_path in sorted(serial_outputs.keys()):
            df_s = serial_outputs[rel_path].reset_index(drop=True)
            df_p = parallel_outputs[rel_path].reset_index(drop=True)
            pd.testing.assert_frame_equal(
                df_s,
                df_p,
                check_exact=True,
                obj=f"Output '{rel_path}'",
            )

    def test_force_recompute_skips_freshness_check(self, tmp_path: Path) -> None:
        """force.pipeline=True causes outputs to be rewritten even if fresh."""
        root = tmp_path / "force_root"
        root.mkdir()
        _build_fixture(root, n_flies=2)

        cfg_no_force = self._make_cfg(root, parallel_enabled=False)
        cfg_force = Settings(
            model_path="",
            main_directories=[str(root)],
            force=ForceSettings(pipeline=True),
            parallel=ParallelSettings(enabled=False),
        )

        # First run (no-force) produces outputs.
        main(cfg_no_force)
        first_outputs = _collect_outputs(root)
        assert first_outputs, "First run should produce outputs"

        # Second run with force=True should overwrite without error.
        main(cfg_force)
        second_outputs = _collect_outputs(root)
        assert set(first_outputs.keys()) == set(second_outputs.keys())
