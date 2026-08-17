"""A figure the driver no longer builds must not survive in the folder.

Re-labelling Hex-24-0.1 to its august panel renamed every figure in that
cohort's folder: ``Hexanol_1_...png`` became ``Hexanol_0.1%_1_...png`` and
``Sour_Dough_Yeast_25%_...png`` stopped existing altogether. The re-run wrote
the new names beside the old ones, leaving a folder in which half the figures
described a panel the dataset no longer claims to have run.

Each driver prunes its own outputs -- matched by the filename tail it writes --
and nothing else in the folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analysis import envelope_visuals as ev  # noqa: E402
from scripts.analysis.dataset_mean_traces_tvc import (  # noqa: E402
    prune_stale_figures,
)

N_FRAMES = 120
FPS = 40.0
ODOR_ON_S = 0.5
BASELINE_FRAMES = int(round(ODOR_ON_S * FPS))


@pytest.fixture(autouse=True)
def _protocol():
    saved_protocol = ev.get_protocol()
    saved_remap = {ds: dict(m) for ds, m in ev._DATASET_ODOR_REMAP.items()}
    ev.set_protocol("v2")
    ev.set_dataset_odor_remap({})
    try:
        yield
    finally:
        ev.set_protocol(saved_protocol)
        ev.set_dataset_odor_remap(saved_remap)


def _wide(datasets, labels):
    rows = []
    for dataset in datasets:
        for fly in ("july_20_batch_1", "july_20_batch_2"):
            for fly_number in (1, 2):
                for i, label in enumerate(labels):
                    row = {
                        "dataset": dataset,
                        "fly": fly,
                        "fly_number": fly_number,
                        "trial_type": "testing",
                        "trial_label": label,
                    }
                    row.update({
                        f"dir_val_{j}": (i + 1) * max(0, j - BASELINE_FRAMES)
                        for j in range(N_FRAMES)
                    })
                    rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------


def test_a_file_the_run_did_not_write_is_removed(tmp_path):
    stale = tmp_path / "Sour_Dough_Yeast_25%_training_vs_control.png"
    fresh = tmp_path / "Citral_1%_training_vs_control.png"
    for path in (stale, fresh):
        path.write_bytes(b"x")
    removed = prune_stale_figures(tmp_path, [fresh], pattern="*_training_vs_control.png")
    assert removed == [stale]
    assert fresh.exists() and not stale.exists()


def test_files_from_another_driver_are_left_alone(tmp_path):
    """Two figure families share a folder in some layouts; one must not delete
    the other's work."""
    other = tmp_path / "Hexanol_conc_series.png"
    sidecar = tmp_path / "training_vs_control.json"
    for path in (other, sidecar):
        path.write_bytes(b"x")
    prune_stale_figures(tmp_path, [], pattern="*_training_vs_control.png")
    assert other.exists() and sidecar.exists()


def test_a_missing_directory_is_not_an_error(tmp_path):
    assert prune_stale_figures(tmp_path / "nope", [], pattern="*.png") == []


# ---------------------------------------------------------------------------
# Both drivers use it
# ---------------------------------------------------------------------------


def test_the_trained_vs_control_driver_prunes_a_renamed_odor(tmp_path):
    from scripts.analysis.dataset_mean_traces_tvc import main

    wide = tmp_path / "wide.parquet"
    _wide(("A-Training-1", "A-Control-1"), ["testing_1_hexanol"]).to_parquet(wide)
    out_dir = tmp_path / "figs"
    out_dir.mkdir()
    stale = out_dir / "Sour_Dough_Yeast_25%_training_vs_control.png"
    stale.write_bytes(b"x")

    main([
        "--wide-csv", str(wide),
        "--train-dataset", "A-Training-1",
        "--control-dataset", "A-Control-1",
        "--out-dir", str(out_dir),
        "--fps", str(FPS), "--odor-on-s", str(ODOR_ON_S), "--odor-off-s", "1.0",
    ])
    assert not stale.exists()
    assert (out_dir / "Hexanol_training_vs_control.png").exists()


def test_the_conc_series_driver_prunes_too(tmp_path):
    from scripts.analysis.randompanel_conc_traces import main

    wide = tmp_path / "wide.parquet"
    _wide(
        ("RandomPanel-24-1", "RandomPanel-24-0.1"),
        ["testing_1_hexanol", "testing_2_hexanol"],
    ).to_parquet(wide)
    out_dir = tmp_path / "figs"
    out_dir.mkdir()
    stale = out_dir / "Linalool_conc_series.png"
    stale.write_bytes(b"x")

    main([
        "--wide-csv", str(wide),
        "--out-dir", str(out_dir),
        "--dataset", "RandomPanel-24-1=1",
        "--dataset", "RandomPanel-24-0.1=0.1",
        "--fps", str(FPS), "--odor-on-s", str(ODOR_ON_S), "--odor-off-s", "1.0",
    ])
    assert not stale.exists()
    assert (out_dir / "Hexanol_conc_series.png").exists()
