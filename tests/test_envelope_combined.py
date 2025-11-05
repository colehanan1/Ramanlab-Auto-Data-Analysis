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
from scripts import envelope_visuals as ev  # noqa: E402
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


def test_has_training_trials_detects_training_entries(tmp_path):
    """_has_training_trials should return True whenever any list includes training."""

    testing_entry = [("testing_1", tmp_path / "a.csv", "testing")]
    training_entry = [("training_1", tmp_path / "b.csv", "training")]

    assert not ec._has_training_trials(testing_entry)
    assert ec._has_training_trials(training_entry)
    assert ec._has_training_trials(testing_entry, training_entry)


def test_locate_trials_discovers_training_without_month_dirs(tmp_path):
    """Legacy layouts without month folders should still surface training CSVs."""

    fly_dir = tmp_path / "legacy_fly"
    conditioning_dir = fly_dir / "legacy" / "conditioning_day"
    conditioning_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = conditioning_dir / "fly1_conditioning_trial1_distances.csv"
    pd.DataFrame({"distance_percentage": [0.0, 1.0]}).to_csv(raw_csv, index=False)

    entries = ec._locate_trials(fly_dir, ("*fly*_distances.csv",), ec.DIST_COLS)

    assert entries, "Expected to discover conditioning training trial CSV."
    labels = {label for label, _, _ in entries}
    categories = {category for _, _, category in entries}
    assert "training_1" in labels
    assert "training" in categories


def test_locate_trials_skips_derived_envelope_outputs(tmp_path):
    """Derived outputs inside angle_distance_rms_envelope should not be reprocessed."""

    fly_dir = tmp_path / "october_01_fly1"
    month_dir = fly_dir / "october"
    month_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = month_dir / "october_fly1_testing_1_distances.csv"
    pd.DataFrame({"distance_percentage": [5.0, 10.0]}).to_csv(raw_csv, index=False)

    derived_dir = fly_dir / "angle_distance_rms_envelope"
    derived_dir.mkdir(parents=True, exist_ok=True)
    derived_csv = derived_dir / "fly1_testing_1_distances.csv"
    pd.DataFrame({"distance_percentage": [1.0, 2.0]}).to_csv(derived_csv, index=False)

    entries = ec._locate_trials(fly_dir, ("*fly*_distances.csv",), ec.DIST_COLS)
    paths = {path for _, path, _ in entries}

    assert raw_csv in paths
    assert derived_csv not in paths


def test_hex_control_testing_trials_map_to_acv():
    """Hexanol datasets should label testing 1 and 3 as Apple Cider Vinegar."""

    assert ec._display_odor("hex_control", "testing_1") == "Apple Cider Vinegar"
    assert ec._display_odor("hex_control", "testing_3") == "Apple Cider Vinegar"
    assert ec._display_odor("opto_hex", "testing_1") == "Apple Cider Vinegar"
    assert ec._display_odor("opto_hex", "testing_2") == "Hexanol"


def test_resolve_dataset_output_dir_separates_eb_variants(tmp_path):
    """EB control and opto_EB datasets should land in distinct result folders."""

    base = tmp_path
    eb_control_dir = ev.resolve_dataset_output_dir(base, ["EB_control"])
    opto_eb_dir = ev.resolve_dataset_output_dir(base, ["opto_EB"])

    assert eb_control_dir != opto_eb_dir
    assert eb_control_dir.name != opto_eb_dir.name
    assert "EB_control" in eb_control_dir.name
    assert "opto_EB" in opto_eb_dir.name


def test_testing_aliases_follow_control_ordering():
    """Opto datasets should share the same testing labels as their controls."""

    schedules = {
        "EB_control": {
            6: "Apple Cider Vinegar",
            7: "3-Octonol",
            8: "Benzaldehyde",
            9: "Citral",
            10: "Linalool",
        },
        "hex_control": {
            6: "Benzaldehyde",
            7: "3-Octonol",
            8: "Ethyl Butyrate",
            9: "Citral",
            10: "Linalool",
        },
        "benz_control": {
            6: "Apple Cider Vinegar",
            7: "3-Octonol",
            8: "Ethyl Butyrate",
            9: "Citral",
            10: "Linalool",
        },
    }

    aliases = {
        "EB_control": ["opto_EB"],
        "hex_control": ["opto_hex"],
        "benz_control": ["opto_benz", "opto_benz_1"],
    }

    for control, mapping in schedules.items():
        for trial, odor in mapping.items():
            label = f"testing_{trial}"
            assert ec._display_odor(control, label) == odor
            for alias in aliases[control]:
                assert ec._display_odor(alias, label) == odor


def test_training_schedule_matches_spec():
    """Training trials map to dataset-specific odors for every canonical dataset."""

    expectations = {
        "EB_control": {
            1: "Ethyl Butyrate",
            2: "Ethyl Butyrate",
            3: "Ethyl Butyrate",
            4: "Ethyl Butyrate",
            5: "Hexanol",
            6: "Ethyl Butyrate",
            7: "Hexanol",
            8: "Ethyl Butyrate",
        },
        "opto_EB": {
            1: "Ethyl Butyrate",
            2: "Ethyl Butyrate",
            3: "Ethyl Butyrate",
            4: "Ethyl Butyrate",
            5: "Hexanol",
            6: "Ethyl Butyrate",
            7: "Hexanol",
            8: "Ethyl Butyrate",
        },
        "hex_control": {
            1: "Hexanol",
            2: "Hexanol",
            3: "Hexanol",
            4: "Hexanol",
            5: "Apple Cider Vinegar",
            6: "Hexanol",
            7: "Apple Cider Vinegar",
            8: "Hexanol",
        },
        "opto_hex": {
            1: "Hexanol",
            2: "Hexanol",
            3: "Hexanol",
            4: "Hexanol",
            5: "Apple Cider Vinegar",
            6: "Hexanol",
            7: "Apple Cider Vinegar",
            8: "Hexanol",
        },
        "benz_control": {
            1: "Benzaldehyde",
            2: "Benzaldehyde",
            3: "Benzaldehyde",
            4: "Benzaldehyde",
            5: "Hexanol",
            6: "Benzaldehyde",
            7: "Hexanol",
            8: "Benzaldehyde",
        },
        "opto_benz": {
            1: "Benzaldehyde",
            2: "Benzaldehyde",
            3: "Benzaldehyde",
            4: "Benzaldehyde",
            5: "Hexanol",
            6: "Benzaldehyde",
            7: "Hexanol",
            8: "Benzaldehyde",
        },
        "opto_benz_1": {
            1: "Benzaldehyde",
            2: "Benzaldehyde",
            3: "Benzaldehyde",
            4: "Benzaldehyde",
            5: "Hexanol",
            6: "Benzaldehyde",
            7: "Hexanol",
            8: "Benzaldehyde",
        },
    }

    for dataset, mapping in expectations.items():
        for trial_num, expected in mapping.items():
            label = f"training_{trial_num}"
            assert ec._display_odor(dataset, label) == expected


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

    assert set(wide_df["trial_type"].str.lower()) == {"testing"}
    assert wide_df["trial_type"].str.lower().tolist() == sorted(
        wide_df["trial_type"].str.lower().tolist()
    )
    assert set(training_df["trial_type"].str.lower()) == {"training"}
    assert len(wide_df) == 1
    assert len(training_df) == 1


def test_fly_max_centered_skips_empty_csv(tmp_path):
    """Empty per-trial CSVs should be ignored when computing the max angle delta."""

    empty_csv = tmp_path / "empty.csv"
    pd.DataFrame(columns=["x_class2", "y_class2", "x_class8", "y_class8"]).to_csv(
        empty_csv, index=False
    )

    valid_csv = tmp_path / "valid.csv"
    valid_df = pd.DataFrame(
        {
            "x_class2": [1000.0, 1000.0],
            "y_class2": [500.0, 500.0],
            "x_class8": [1100.0, 1120.0],
            "y_class8": [500.0, 540.0],
        }
    )
    valid_df.to_csv(valid_csv, index=False)

    angles = ec._compute_angle_deg(valid_df).to_numpy(dtype=float)
    reference = float(angles[0])
    expected = np.nanmax(np.abs(angles - reference))

    result = ec._fly_max_centered([empty_csv, valid_csv], reference)

    assert np.isclose(result, expected, equal_nan=False)


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


def test_auto_sync_wide_roots_copies_training_outputs(tmp_path):
    """Auto-sync should mirror combine outputs into wide roots when needed."""

    combine_root = tmp_path / "local" / "hex_control"
    secure_root = tmp_path / "secure" / "hex_control"
    combine_csv_dir = combine_root / "angle_distance_rms_envelope"
    combine_csv_dir.mkdir(parents=True, exist_ok=True)
    secure_root.mkdir(parents=True, exist_ok=True)

    training_file = combine_csv_dir / "october_01_fly1_training_1_angle_distance_rms_envelope.csv"
    pd.DataFrame({"envelope_of_rms": [1.0, 2.0, 3.0]}).to_csv(training_file, index=False)

    rw._auto_sync_wide_roots({"hex_control": combine_root.resolve()}, [secure_root.resolve()])

    mirrored = secure_root / "angle_distance_rms_envelope" / training_file.name
    assert mirrored.exists()
