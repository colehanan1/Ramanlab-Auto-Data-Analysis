"""calculate_acceleration decides from the schema before reading a table.

The step needs ``angle_multiplier``, which only compose_videos_rms writes, and
compose_videos_rms is not in the pipeline. So every night it used to read every
RMS table in full only to print "Missing angle_multiplier" and write nothing.
The schema read it already did for the "already processed" skip now also skips
tables without ``angle_multiplier``. Tables that have it are processed exactly
as before, and the step stays in ``ORDERED_STEPS`` so a hand-run of
compose_videos_rms followed by the pipeline still works.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fbpipe.config import ForceSettings, ParallelSettings, Settings
from fbpipe.utils.tables import read_schema_columns, read_table, write_table

BATCH = "october_05_batch_2"


def _rms_df(n: int = 40, *, angle: bool, pct: bool = True, accel: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    data = {
        "frame": np.arange(n),
        "timestamp": np.arange(n) / 40.0,
        "x_class0": rng.uniform(0, 1280, n),
        "y_class0": rng.uniform(0, 720, n),
        "x_class1": rng.uniform(0, 1280, n),
        "y_class1": rng.uniform(0, 720, n),
    }
    if pct:
        data["distance_percentage_0_1"] = rng.uniform(0, 100, n)
    if angle:
        data["angle_multiplier"] = rng.uniform(0.5, 1.5, n)
    if accel:
        data["acceleration_pct_per_frame"] = np.zeros(n)
    return pd.DataFrame(data)


def _build(root: Path) -> dict[str, Path]:
    rdir = root / BATCH / "RMS_calculations"
    rdir.mkdir(parents=True)
    files = {
        "no_angle": write_table(_rms_df(angle=False), rdir / f"updated_{BATCH}_testing_1_fly1_distances.parquet"),
        "angle": write_table(_rms_df(angle=True), rdir / f"updated_{BATCH}_testing_1_fly2_distances.parquet"),
        "done": write_table(_rms_df(angle=True, accel=True), rdir / f"updated_{BATCH}_testing_2_fly1_distances.parquet"),
        "angle_no_pct": write_table(_rms_df(angle=True, pct=False), rdir / f"updated_{BATCH}_testing_2_fly2_distances.parquet"),
    }
    corrupt = rdir / f"updated_{BATCH}_testing_3_fly1_distances.parquet"
    corrupt.write_bytes(b"not a parquet file")
    files["corrupt"] = corrupt
    return files


def _cfg(root: Path) -> Settings:
    return Settings(
        model_path="",
        main_directories=[str(root)],
        auto_discover_flagged=False,
        allow_cpu=True,
        force=ForceSettings(pipeline=True),
        parallel=ParallelSettings(enabled=False),
    )


_MODULES = ["fbpipe.steps.calculate_acceleration", "fbpipe.steps.calculate_acceleration_gpu"]


def _spy(monkeypatch, module) -> list[Path]:
    reads: list[Path] = []
    real = module.read_table

    def spy(path, *args, **kwargs):
        reads.append(Path(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(module, "read_table", spy)
    return reads


@pytest.mark.parametrize("module_name", _MODULES)
def test_tables_without_angle_multiplier_are_not_read(module_name, tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module(module_name)
    root = tmp_path / "Hex-HexTest-Training-24-1"
    files = _build(root)
    before = {k: p.read_bytes() for k, p in files.items()}
    reads = _spy(monkeypatch, module)

    module.main(_cfg(root))

    read_names = {p.name for p in reads}
    assert files["no_angle"].name not in read_names
    assert files["done"].name not in read_names
    # Everything that could still be processed is read as before.
    assert files["angle"].name in read_names
    assert files["angle_no_pct"].name in read_names
    assert files["corrupt"].name in read_names  # schema unreadable -> falls through
    for key in ("no_angle", "done", "angle_no_pct", "corrupt"):
        assert files[key].read_bytes() == before[key], key


@pytest.mark.parametrize("module_name", _MODULES)
def test_tables_with_angle_multiplier_are_processed_as_before(module_name, tmp_path):
    import importlib

    module = importlib.import_module(module_name)
    root = tmp_path / "Hex-HexTest-Training-24-1"
    files = _build(root)
    src = read_table(files["angle"])

    module.main(_cfg(root))

    out = read_table(files["angle"])
    combined = src["distance_percentage_0_1"].to_numpy() * src["angle_multiplier"].to_numpy()
    expected = np.full(len(combined), np.nan)
    expected[1:] = np.diff(combined)
    np.testing.assert_allclose(out["combined_distance_x_angle"], combined, rtol=1e-6)
    np.testing.assert_allclose(out["acceleration_pct_per_frame"], expected, rtol=1e-5, atol=1e-4, equal_nan=True)
    assert out["acceleration_flag"].tolist() == (np.abs(expected) > module.ACCELERATION_THRESHOLD).tolist()
    assert "acceleration_pct_per_frame" in read_schema_columns(files["angle"])


def test_step_stays_in_the_pipeline():
    from fbpipe.pipeline import ORDERED_STEPS

    names = [step.name for step in ORDERED_STEPS]
    assert names[-1] == "calculate_acceleration"
    assert names.index("calculate_acceleration") > names.index("update_ofm_state")
