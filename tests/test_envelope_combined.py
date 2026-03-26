"""Unit tests for angle-distance combination helpers."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from scripts.analysis import envelope_combined as ec  # noqa: E402
from scripts.analysis import envelope_visuals as ev  # noqa: E402
from scripts.pipeline import run_workflows as rw  # noqa: E402
from fbpipe.utils.columns import EYE_CLASS, PROBOSCIS_CLASS  # noqa: E402


def test_angle_multiplier_never_below_unity():
    """Ensure multipliers are >= 1.0 for all angles and match log formula."""

    angles = np.array([-100, -50, -1, 0, 50, 100], dtype=float)
    multipliers = ec._angle_multiplier(angles)

    # Negative/zero angles always produce 1.0
    assert np.all(multipliers[:4] == 1.0)
    # Positive angles produce > 1.0
    assert np.all(multipliers[4:] > 1.0)
    # At +100%, multiplier should be exactly 2.0
    np.testing.assert_allclose(multipliers[5], 2.0)
    # All values >= 1.0 (never attenuates)
    assert np.all(multipliers >= 1.0)


def test_angle_multiplier_handles_scalars():
    """Scalar inputs: negative → 1.0, positive → > 1.0."""

    assert ec._angle_multiplier(np.array([-100.0])).item() == 1.0
    assert ec._angle_multiplier(np.array([5.0])).item() > 1.0


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

    assert ec._display_odor("Hex-Control", "testing_1") == "Apple Cider Vinegar"
    assert ec._display_odor("Hex-Control", "testing_3") == "Apple Cider Vinegar"
    assert ec._display_odor("Hex-Training", "testing_1") == "Apple Cider Vinegar"
    assert ec._display_odor("Hex-Training", "testing_2") == "Hexanol"
    # Legacy names still resolve through ODOR_CANON aliases
    assert ec._display_odor("hex_control", "testing_1") == "Apple Cider Vinegar"
    assert ec._display_odor("opto_hex", "testing_1") == "Apple Cider Vinegar"


def test_resolve_dataset_output_dir_separates_eb_variants(tmp_path):
    """EB-Control and EB-Training datasets should land in distinct result folders."""

    base = tmp_path
    eb_control_dir = ev.resolve_dataset_output_dir(base, ["EB-Control"])
    eb_training_dir = ev.resolve_dataset_output_dir(base, ["EB-Training"])

    assert eb_control_dir != eb_training_dir
    assert eb_control_dir.name != eb_training_dir.name
    assert "EB-Control" in eb_control_dir.name
    assert "EB-Training" in eb_training_dir.name


def test_testing_aliases_follow_control_ordering():
    """Training datasets should share the same testing labels as their controls."""

    schedules = {
        "EB-Control": {
            6: "Apple Cider Vinegar",
            7: "3-Octonol",
            8: "Benzaldehyde",
            9: "Citral",
            10: "Linalool",
        },
        "Hex-Control": {
            6: "Benzaldehyde",
            7: "3-Octonol",
            8: "Ethyl Butyrate",
            9: "Citral",
            10: "Linalool",
        },
        "Benz-Control": {
            6: "Apple Cider Vinegar",
            7: "3-Octonol",
            8: "Ethyl Butyrate",
            9: "Citral",
            10: "Linalool",
        },
    }

    aliases = {
        "EB-Control": ["EB-Training"],
        "Hex-Control": ["Hex-Training", "Hex-Training-24", "Hex-Training-36"],
        "Benz-Control": ["Benz-Training", "Benz-Training-24"],
    }

    for control, mapping in schedules.items():
        for trial, odor in mapping.items():
            label = f"testing_{trial}"
            assert ec._display_odor(control, label) == odor
            for alias in aliases[control]:
                assert ec._display_odor(alias, label) == odor

    for trial, odor in schedules["Hex-Control"].items():
        label = f"testing_{trial}"
        assert ec._display_odor("Hex-Control-24", label) == odor
        assert ec._display_odor("Hex-Control-36", label) == odor


def test_training_schedule_matches_spec():
    """Training trials map to dataset-specific odors for every canonical dataset."""

    expectations = {
        "EB-Control": {
            1: "Ethyl Butyrate",
            2: "Ethyl Butyrate",
            3: "Ethyl Butyrate",
            4: "Ethyl Butyrate",
            5: "Hexanol",
            6: "Ethyl Butyrate",
            7: "Hexanol",
            8: "Ethyl Butyrate",
        },
        "EB-Training": {
            1: "Ethyl Butyrate",
            2: "Ethyl Butyrate",
            3: "Ethyl Butyrate",
            4: "Ethyl Butyrate",
            5: "Hexanol",
            6: "Ethyl Butyrate",
            7: "Hexanol",
            8: "Ethyl Butyrate",
        },
        "Hex-Control": {
            1: "Hexanol",
            2: "Hexanol",
            3: "Hexanol",
            4: "Hexanol",
            5: "Apple Cider Vinegar",
            6: "Hexanol",
            7: "Apple Cider Vinegar",
            8: "Hexanol",
        },
        "Hex-Control-24": {
            1: "Hexanol",
            2: "Hexanol",
            3: "Hexanol",
            4: "Hexanol",
            5: "Apple Cider Vinegar",
            6: "Hexanol",
            7: "Apple Cider Vinegar",
            8: "Hexanol",
        },
        "Hex-Control-36": {
            1: "Hexanol",
            2: "Hexanol",
            3: "Hexanol",
            4: "Hexanol",
            5: "Apple Cider Vinegar",
            6: "Hexanol",
            7: "Apple Cider Vinegar",
            8: "Hexanol",
        },
        "Hex-Training": {
            1: "Hexanol",
            2: "Hexanol",
            3: "Hexanol",
            4: "Hexanol",
            5: "Apple Cider Vinegar",
            6: "Hexanol",
            7: "Apple Cider Vinegar",
            8: "Hexanol",
        },
        "Hex-Training-24": {
            1: "Hexanol",
            2: "Hexanol",
            3: "Hexanol",
            4: "Hexanol",
            5: "Apple Cider Vinegar",
            6: "Hexanol",
            7: "Apple Cider Vinegar",
            8: "Hexanol",
        },
        "Hex-Training-36": {
            1: "Hexanol",
            2: "Hexanol",
            3: "Hexanol",
            4: "Hexanol",
            5: "Apple Cider Vinegar",
            6: "Hexanol",
            7: "Apple Cider Vinegar",
            8: "Hexanol",
        },
        "Benz-Control": {
            1: "Benzaldehyde",
            2: "Benzaldehyde",
            3: "Benzaldehyde",
            4: "Benzaldehyde",
            5: "Hexanol",
            6: "Benzaldehyde",
            7: "Hexanol",
            8: "Benzaldehyde",
        },
        "Benz-Training": {
            1: "Benzaldehyde",
            2: "Benzaldehyde",
            3: "Benzaldehyde",
            4: "Benzaldehyde",
            5: "Hexanol",
            6: "Benzaldehyde",
            7: "Hexanol",
            8: "Benzaldehyde",
        },
        "Benz-Training-24": {
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


def test_build_wide_csv_accepts_renamed_combined_pct_column(tmp_path):
    """Legacy combined_base requests should still resolve renamed combined_pct data."""

    dataset_root = tmp_path / "acv_training"
    fly_dir = dataset_root / "march_05_batch_1"
    out_dir = fly_dir / "angle_distance_rms_envelope"
    out_dir.mkdir(parents=True, exist_ok=True)

    values = np.array([0.0, 25.0, 50.0, 100.0], dtype=float)
    stem = "march_05_batch_1_testing_1_angle_distance_rms_envelope"
    pd.DataFrame({"combined_pct": values}).to_csv(out_dir / f"{stem}.csv", index=False)

    output_csv = tmp_path / "wide_combined_base.csv"
    ec.build_wide_csv(
        [str(dataset_root)],
        str(output_csv),
        measure_cols=["combined_base"],
    )

    wide_df = pd.read_csv(output_csv)

    assert len(wide_df) == 1
    np.testing.assert_allclose(
        wide_df.loc[0, ["dir_val_0", "dir_val_1", "dir_val_2", "dir_val_3"]].to_numpy(dtype=float),
        values,
    )


def test_build_wide_csv_prefers_distance_labelled_trial_alias(tmp_path):
    """Duplicate trial aliases should keep only the distances-labelled combined CSV."""

    dataset_root = tmp_path / "acv_training"
    fly_dir = dataset_root / "march_05_batch_1"
    out_dir = fly_dir / "angle_distance_rms_envelope"
    out_dir.mkdir(parents=True, exist_ok=True)

    short_values = np.array([10.0, 20.0, 30.0], dtype=float)
    long_values = np.array([40.0, 50.0, 60.0], dtype=float)

    pd.DataFrame({"combined_pct": short_values}).to_csv(
        out_dir / "testing_1_fly1_angle_distance_rms_envelope.csv",
        index=False,
    )
    pd.DataFrame({"combined_pct": long_values}).to_csv(
        out_dir / "testing_1_fly1_distances_fly1_angle_distance_rms_envelope.csv",
        index=False,
    )

    output_csv = tmp_path / "wide_combined_base.csv"
    ec.build_wide_csv(
        [str(dataset_root)],
        str(output_csv),
        measure_cols=["combined_pct"],
    )

    wide_df = pd.read_csv(output_csv)

    assert len(wide_df) == 1
    assert (
        wide_df.loc[0, "trial_label"]
        == "testing_1_fly1_distances_fly1_angle_distance_rms_envelope"
    )
    np.testing.assert_allclose(
        wide_df.loc[0, ["dir_val_0", "dir_val_1", "dir_val_2"]].to_numpy(dtype=float),
        long_values,
    )


def test_extract_env_row_preserves_internal_nan_gaps():
    """Trace extraction should preserve timebase gaps instead of compacting them away."""

    env = ev._extract_env_row(
        np.array([10.0, np.nan, 20.0, 0.0, np.nan], dtype=float),
        trace_len=4,
    )

    assert env.shape == (4,)
    assert np.isnan(env[1])
    assert env[3] == 0.0


def test_wide_to_matrix_preserves_nan_gaps_and_metrics(tmp_path):
    """Matrix export should keep NaN gaps in dir_val columns and exclude tracking stats from env."""

    wide_csv = tmp_path / "wide_combined_base.csv"
    pd.DataFrame(
        {
            "dataset": ["ACV-Training"],
            "fly": ["march_05_batch_1"],
            "fly_number": ["1"],
            "trial_type": ["testing"],
            "trial_label": ["testing_1"],
            "fps": [40.0],
            "global_min": [1.0],
            "global_max": [5.0],
            "trimmed_global_min": [1.0],
            "trimmed_global_max": [5.0],
            "tracking_missing_frames": [12],
            "tracking_pct_missing": [25.0],
            "tracking_flagged": [1.0],
            "trace_len": [4],
            "dir_val_0": [10.0],
            "dir_val_1": [np.nan],
            "dir_val_2": [20.0],
            "dir_val_3": [0.0],
        }
    ).to_csv(wide_csv, index=False)

    out_dir = tmp_path / "matrix"
    ec.wide_to_matrix(str(wide_csv), str(out_dir))

    meta = json.loads((out_dir / "code_maps.json").read_text())
    assert meta["env_columns"] == ["dir_val_0", "dir_val_1", "dir_val_2", "dir_val_3"]
    assert "tracking_missing_frames" in meta["metric_columns"]

    loaded_df, env_cols = ev._load_matrix(
        out_dir / "envelope_matrix_float16.npy",
        out_dir / "code_maps.json",
    )

    assert env_cols == ["dir_val_0", "dir_val_1", "dir_val_2", "dir_val_3"]
    assert np.isnan(loaded_df.loc[0, "dir_val_1"])
    assert loaded_df.loc[0, "tracking_missing_frames"] == 12.0

    env = ev._extract_env_row(
        loaded_df.loc[0, env_cols].to_numpy(dtype=float),
        trace_len=loaded_df.loc[0, "trace_len"],
    )
    assert env.shape == (4,)
    assert np.isnan(env[1])
    assert env[3] == 0.0


def test_generate_envelope_plots_filters_to_requested_fly(tmp_path):
    """Envelope plotting should honor explicit fly filters without affecting defaults."""

    wide_csv = tmp_path / "wide_testing.csv"
    pd.DataFrame(
        {
            "dataset": ["Hex-Training", "Hex-Training", "Hex-Training"],
            "fly": ["october_10_batch_1", "october_10_batch_1", "october_10_batch_2"],
            "fly_number": ["2", "2", "1"],
            "trial_type": ["testing", "testing", "testing"],
            "trial_label": ["testing_1", "testing_2", "testing_1"],
            "fps": [40.0, 40.0, 40.0],
            "global_min": [1.0, 1.0, 1.0],
            "global_max": [20.0, 25.0, 20.0],
            "trimmed_global_min": [1.0, 1.0, 1.0],
            "trimmed_global_max": [20.0, 25.0, 20.0],
            "trace_len": [4, 4, 4],
            "dir_val_0": [0.0, 2.0, 1.0],
            "dir_val_1": [8.0, 4.0, 3.0],
            "dir_val_2": [16.0, 6.0, 5.0],
            "dir_val_3": [4.0, 8.0, 2.0],
        }
    ).to_csv(wide_csv, index=False)

    matrix_dir = tmp_path / "matrix"
    ec.wide_to_matrix(str(wide_csv), str(matrix_dir))

    cfg = ev.EnvelopePlotConfig(
        matrix_npy=matrix_dir / "envelope_matrix_float16.npy",
        codes_json=matrix_dir / "code_maps.json",
        out_dir=tmp_path / "plots",
        latency_sec=0.0,
        odor_latency_s=0.0,
        trial_type="testing",
        overwrite=True,
        fly_filter="october_10_batch_1",
        fly_number_filter="2",
        style_scale=2.0,
        trace_linewidth_scale=2.0,
        panel_title_scale=0.8,
        figure_title_scale=0.5,
        figure_subtitle_scale=0.5,
        legend_scale=0.8,
        plot_size_scale=1.1,
        single_ylabel_trial_num=1,
        fixed_y_max=105.0,
        y_label_override="Max Distance x Angle %",
    )
    ev.generate_envelope_plots(cfg)

    target_png = (
        tmp_path
        / "plots"
        / "Hex-Training"
        / "october_10_batch_1_fly2_testing_envelope_trials_by_odor_30_shifted.png"
    )
    other_png = (
        tmp_path
        / "plots"
        / "Hex-Training"
        / "october_10_batch_2_fly1_testing_envelope_trials_by_odor_30_shifted.png"
    )

    assert target_png.exists()
    assert not other_png.exists()


def test_select_trial_rows_prefers_distance_label_alias():
    """Duplicate raw aliases should collapse to the canonical distance-labelled trial row."""

    df = pd.DataFrame(
        {
            "trial_label": [
                "testing_1_fly1_angle_distance_rms_envelope",
                "testing_1_fly1_distances_fly1_angle_distance_rms_envelope",
                "testing_2_fly1_distances_fly1_angle_distance_rms_envelope",
            ],
            "value": [1, 2, 3],
        },
        index=[10, 11, 12],
    )

    selected = ev._select_trial_rows(df)

    assert list(selected.index) == [11, 12]
    assert selected.iloc[0]["trial_label"] == "testing_1_fly1_distances_fly1_angle_distance_rms_envelope"


def test_pipeline_envelope_plot_config_applies_style_defaults(tmp_path):
    """Pipeline envelope configs should inherit the approved trace styling unless overridden."""

    cfg, _ = rw._envelope_plot_config(
        {
            "matrix_npy": tmp_path / "matrix.npy",
            "codes_json": tmp_path / "codes.json",
            "out_dir": tmp_path / "plots",
            "latency_sec": 0.0,
        }
    )

    assert cfg.style_scale == 2.0
    assert cfg.trace_linewidth_scale == 2.0
    assert cfg.fixed_y_max == 105.0
    assert cfg.y_label_override == "Max Distance x Angle %"
    assert cfg.figure_title_y == 0.925
    assert cfg.figure_subtitle_y == 0.905
    assert cfg.panel_title_x == 0.01
    assert cfg.panel_title_y == 105.0
    assert cfg.panel_title_va == "top"
    assert cfg.panel_title_use_data_y is True
    assert cfg.legend_anchor_y == 0.895
    assert cfg.tight_h_pad == 0.12

    overridden, _ = rw._envelope_plot_config(
        {
            "matrix_npy": tmp_path / "matrix.npy",
            "codes_json": tmp_path / "codes.json",
            "out_dir": tmp_path / "plots",
            "latency_sec": 0.0,
            "style_scale": 1.5,
            "fixed_y_max": 110.0,
        }
    )

    assert overridden.style_scale == 1.5
    assert overridden.fixed_y_max == 110.0


def test_fly_max_centered_skips_empty_csv(tmp_path):
    """Empty per-trial CSVs should be ignored when computing the max angle delta."""

    empty_csv = tmp_path / "empty.csv"
    eye_x = f"x_class{EYE_CLASS}"
    eye_y = f"y_class{EYE_CLASS}"
    prob_x = f"x_class{PROBOSCIS_CLASS}"
    prob_y = f"y_class{PROBOSCIS_CLASS}"
    pd.DataFrame(columns=[eye_x, eye_y, prob_x, prob_y]).to_csv(empty_csv, index=False)

    valid_csv = tmp_path / "valid.csv"
    valid_df = pd.DataFrame(
        {
            eye_x: [1000.0, 1000.0],
            eye_y: [500.0, 500.0],
            prob_x: [1100.0, 1120.0],
            prob_y: [500.0, 540.0],
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


def test_combine_nan_propagation():
    """NaN in d_px or angle must produce NaN in d_px_weighted."""

    d_px = np.array([10.0, np.nan, 20.0, 30.0])
    angle_pct = np.array([50.0, 50.0, np.nan, 0.0])
    multiplier = ec._angle_multiplier(angle_pct)
    weighted = d_px * multiplier

    assert np.isfinite(weighted[0]), "Valid d_px + valid angle should be finite"
    assert np.isnan(weighted[1]), "NaN d_px should propagate to weighted"
    assert np.isnan(weighted[2]), "NaN angle should produce NaN multiplier and weighted"
    assert np.isfinite(weighted[3]), "Valid d_px + zero angle should be finite"


def test_sentinel_replaced_with_nan():
    """Distance normalization should use NaN for out-of-range, not sentinels (-1/101)."""

    try:
        from fbpipe.utils.gpu_accelerated import GPUBatchProcessor
    except ImportError:
        import pytest
        pytest.skip("gpu_accelerated not importable (missing torch)")

    proc = GPUBatchProcessor(device="cpu")
    d = np.array([5.0, 50.0, 100.0, 300.0], dtype=np.float32)
    result = proc.normalize_distances_batch(d, gmin=10.0, gmax=250.0, effective_max=250.0)

    # Under-range (5.0 < gmin=10.0) -> NaN, not -1
    assert np.isnan(result[0]), "Under-range should be NaN, not -1"
    # In-range -> finite percentage
    assert np.isfinite(result[1])
    assert np.isfinite(result[2])
    # Over-range (300.0 > gmax=250.0) -> NaN, not 101
    assert np.isnan(result[3]), "Over-range should be NaN, not 101"
    # Verify no sentinel values remain
    assert not np.any(result == -1.0), "No -1 sentinels should exist"
    assert not np.any(result == 101.0), "No 101 sentinels should exist"


def test_pixel_domain_then_normalize():
    """Combined metric: multiply in pixel domain, normalize with per-fly weighted bounds."""

    # Synthetic fly data
    d_px = np.array([80.0, 100.0, 120.0, np.nan], dtype=float)
    angle_pct = np.array([0.0, 50.0, 100.0, 50.0], dtype=float)

    multiplier = ec._angle_multiplier(angle_pct)
    d_px_weighted = d_px * multiplier

    # mult at 0% = 1.0, mult at 100% = 2.0
    np.testing.assert_allclose(multiplier[0], 1.0)
    np.testing.assert_allclose(multiplier[2], 2.0, atol=0.01)

    # Weighted values (approximate)
    assert np.isclose(d_px_weighted[0], 80.0)  # 80 * 1.0
    assert d_px_weighted[1] > 100.0  # 100 * ~1.62
    assert np.isclose(d_px_weighted[2], 240.0)  # 120 * 2.0
    assert np.isnan(d_px_weighted[3])  # NaN d_px

    # Compute per-fly weighted bounds
    finite_dpw = d_px_weighted[np.isfinite(d_px_weighted)]
    weighted_gmin = float(np.min(finite_dpw))
    weighted_gmax = float(np.max(finite_dpw))
    effective_max_weighted = max(weighted_gmax, 150.0)

    # Normalize
    combined_pct = 100.0 * (d_px_weighted - weighted_gmin) / (
        effective_max_weighted - weighted_gmin
    )

    # The minimum weighted value should map to 0%
    assert np.isclose(combined_pct[0], 0.0)
    # The maximum weighted value should map to 100% (since effective_max == weighted_gmax here)
    assert np.isclose(combined_pct[2], 100.0)
    # Mid-range value should be between 0 and 100
    assert 0.0 < combined_pct[1] < 100.0
    # NaN should propagate
    assert np.isnan(combined_pct[3])
