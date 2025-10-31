"""Unit tests for angle-distance combination helpers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from scripts import envelope_combined as ec  # noqa: E402
from scripts import run_workflows as rw  # noqa: E402


def test_angle_multiplier_never_below_unity():
    """Ensure dir_val multipliers never attenuate the distance percentage."""

    angles = np.array([-80, -30, -12, 0, 15, 35, 55, 90], dtype=float)
    multipliers = ec._angle_multiplier(angles)

    assert np.all(multipliers[:4] == 1.0)
    assert multipliers[4] == 1.25
    assert multipliers[5] == 1.50
    assert multipliers[6] == 1.75
    assert multipliers[7] == 2.00


def test_angle_multiplier_handles_scalars():
    """Scalar inputs should still respect the unity floor."""

    assert ec._angle_multiplier(np.array([-100.0])).item() == 1.0
    assert ec._angle_multiplier(np.array([5.0])).item() == 1.0


def test_eb_control_matches_opto_ordering():
    """EB_control should mirror opto_EB odor labels for late trials."""

    expected = {
        6: "Apple Cider Vinegar",
        7: "3-Octonol",
        8: "Benzaldehyde",
        9: "Citral",
        10: "Linalool",
    }

    for trial, odor in expected.items():
        label = f"testing_{trial}"
        assert ec._display_odor("opto_EB", label) == odor
        assert ec._display_odor("EB_control", label) == odor


def test_build_wide_csv_exports_training_subset(tmp_path):
    """The wide CSV builder should emit the training-only subset when requested."""

    dataset_root = tmp_path / "hex_control"
    fly_dir = dataset_root / "october_01_fly1"
    out_dir = fly_dir / "angle_distance_rms_envelope"
    out_dir.mkdir(parents=True, exist_ok=True)

    for trial_type in ("testing", "training"):
        values = np.linspace(0, 100, 10, dtype=float)
        df = pd.DataFrame({"envelope_of_rms": values})
        stem = f"october_01_fly1_{trial_type}_1_angle_distance_rms_envelope"
        df.to_csv(out_dir / f"{stem}.csv", index=False)

    output_csv = tmp_path / "wide.csv"
    training_csv = tmp_path / "wide_training.csv"
    ec.build_wide_csv(
        [str(dataset_root)],
        str(output_csv),
        measure_cols=["envelope_of_rms"],
        extra_trial_exports={"training": str(training_csv)},
    )

    wide_df = pd.read_csv(output_csv)
    training_df = pd.read_csv(training_csv)

    assert set(wide_df["trial_type"].str.lower()) == {"testing", "training"}
    assert set(training_df["trial_type"].str.lower()) == {"training"}
    assert len(training_df) < len(wide_df)


def test_run_combined_creates_training_matrix(tmp_path):
    """_run_combined should materialise matrices for training exports."""

    dataset_root = tmp_path / "hex_control"
    fly_dir = dataset_root / "october_01_fly1"
    csv_dir = fly_dir / "angle_distance_rms_envelope"
    csv_dir.mkdir(parents=True, exist_ok=True)

    values = np.linspace(0, 100, 4000, dtype=float)
    for trial_type in ("testing", "training"):
        stem = f"october_01_fly1_{trial_type}_1_angle_distance_rms_envelope"
        pd.DataFrame({"envelope_of_rms": values}).to_csv(
            csv_dir / f"{stem}.csv", index=False
        )

    training_csv = tmp_path / "wide_training.csv"
    matrix_dir = tmp_path / "training_matrix"

    rw._run_combined(
        {
            "wide": {
                "roots": [str(dataset_root)],
                "output_csv": str(tmp_path / "wide.csv"),
                "measure_cols": ["envelope_of_rms"],
                "trial_type_exports": [
                    {
                        "trial_type": "training",
                        "output_csv": str(training_csv),
                        "matrix_out_dir": str(matrix_dir),
                    }
                ],
            }
        },
        settings=None,
    )

    matrix_path = matrix_dir / "envelope_matrix_float16.npy"
    assert matrix_path.exists()
    matrix = np.load(matrix_path)
    assert matrix.shape[0] == 1
