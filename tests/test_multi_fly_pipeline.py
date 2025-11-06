import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from envelope_combined import (
    AUC_COLUMNS,
    AFTER_FRAMES,
    BEFORE_FRAMES,
    DURING_FRAMES,
    CombineConfig,
    build_wide_csv,
    combine_distance_angle,
)
from envelope_exports import CollectConfig, ConvertConfig, collect_envelopes, convert_wide_csv
from fbpipe.config import Settings
from fbpipe.steps import detect_dropped_frames, distance_normalize, distance_stats
from fbpipe.utils.fly_files import iter_fly_distance_csvs


def _write_dummy_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "frame": [0, 1, 2, 3],
            "proboscis_distance": [75.0, 100.0, np.nan, 140.0],
            "distance_percentage": [0.0, 0.0, 0.0, 0.0],
            "min_distance_2_8": [0.0, 0.0, 0.0, 0.0],
            "max_distance_2_8": [0.0, 0.0, 0.0, 0.0],
        }
    )
    df.to_csv(path, index=False)


def test_iter_fly_distance_csvs_detects_slot(tmp_path):
    fly_dir = tmp_path / "october_07_fly_1"
    csv_path = fly_dir / "october_07_fly_1_testing_2_fly1_distances.csv"
    _write_dummy_csv(csv_path)

    results = list(iter_fly_distance_csvs(fly_dir, recursive=True))
    assert len(results) == 1
    path, token, slot_idx = results[0]
    assert path == csv_path
    assert token == "fly1_distances"
    assert slot_idx == 1


def test_distance_pipeline_creates_stats_and_normalizes(tmp_path):
    root = tmp_path / "experiment"
    fly_dir = root / "october_07_session"
    csv_path = fly_dir / "RMS_calculations" / "october_07_fly_1_testing_2_fly1_distances.csv"
    _write_dummy_csv(csv_path)

    settings = Settings(
        model_path="",
        main_directory=str(root),
        class2_min=70.0,
        class2_max=150.0,
    )

    distance_stats.main(settings)

    stats_path = fly_dir / "fly1_global_distance_stats_class_2.json"
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text())
    assert stats["global_min"] == 75.0
    assert stats["global_max"] == 140.0

    distance_normalize.main(settings)

    df = pd.read_csv(csv_path)
    assert "distance_percentage_2_8" in df.columns
    assert "distance_2_8" in df.columns
    assert df["min_distance_2_8"].iloc[0] == 75.0
    assert df["max_distance_2_8"].iloc[0] == 140.0

    expected = 100.0 * (df["proboscis_distance"] - 75.0) / (140.0 - 75.0)
    np.testing.assert_allclose(
        df.loc[[0, 1, 3], "distance_percentage_2_8"],
        expected.loc[[0, 1, 3]],
        atol=1e-6,
        equal_nan=True,
    )

    detect_dropped_frames.main(settings)

    report_file = Path(str(csv_path.with_suffix("")) + "_dropped_frames.txt")
    assert report_file.exists()
    report_text = report_file.read_text()
    assert "2" in report_text  # frame 2 contains NaN distance
    assert "Total dropped frames" in report_text 

def test_collect_and_convert_capture_fly_number(tmp_path):
    root = tmp_path / "experiment"
    fly_dir = root / "october_07_session"
    csv_path = fly_dir / "RMS_calculations" / "october_07_fly_1_testing_2_fly3_distances.csv"
    _write_dummy_csv(csv_path)

    out_csv = tmp_path / "all_envelope_rows_wide.csv"
    collect_cfg = CollectConfig(
        roots=[root],
        measure_cols=["distance_percentage"],
        fps_default=40.0,
        window_sec=0.25,
        fallback_fps=40.0,
        out_csv=out_csv,
    )

    collect_envelopes(collect_cfg)

    df = pd.read_csv(out_csv)
    assert "fly_number" in df.columns
    assert str(df.loc[0, "fly_number"]) == "3"
    assert df.loc[0, "fly"] == "october_07_session_fly3"
    for col in (
        "AUC-Before",
        "AUC-During",
        "AUC-After",
        "AUC-During-Before-Ratio",
        "AUC-After-Before-Ratio",
        "TimeToPeak-During",
        "Peak-Value",
    ):
        assert col in df.columns

    convert_cfg = ConvertConfig(
        input_csv=out_csv,
        out_dir=tmp_path / "artifacts",
        matrix_npy=None,
        code_key=None,
        codes_json=None,
    )

    convert_wide_csv(convert_cfg)

    code_maps = json.loads((convert_cfg.out_dir / "code_maps.json").read_text())
    assert "fly_number" in code_maps["code_maps"]
    assert code_maps["code_maps"]["fly_number"].get("3") == 1
    assert code_maps["column_order"][2] == "fly_number"


def test_combine_distance_angle_includes_fly_number_column(tmp_path):
    root = tmp_path / "experiment"
    fly_dir = root / "october_07_session"
    month_dir = fly_dir / "october"

    for idx, distance in enumerate((
        [10.0, 20.0, 30.0, 40.0],
        [15.0, 25.0, 35.0, 45.0],
    ), start=1):
        csv_name = f"october_07_fly_1_testing_1_fly{idx}_distances.csv"
        csv_path = month_dir / "RMS_calculations" / csv_name
        angle_series = [0.0, 5.0 * idx, 10.0 * idx, 15.0 * idx]
        _write_distance_angle_csv(
            csv_path,
            distance_pct=distance,
            angle_pct=angle_series,
        )

    combine_cfg = CombineConfig(root=root, fps_default=40.0, window_sec=0.25)
    combine_distance_angle(combine_cfg)

    out_csv_dir = fly_dir / "angle_distance_rms_envelope"
    csv_outputs = sorted(out_csv_dir.glob("*angle_distance_rms_envelope.csv"))
    assert len(csv_outputs) == 2

    observed_numbers = set()
    for csv_path in csv_outputs:
        df = pd.read_csv(csv_path)
        assert "fly_number" in df.columns
        unique_numbers = set(df["fly_number"].astype(str))
        assert len(unique_numbers) == 1
        observed_numbers.update(unique_numbers)

    assert observed_numbers == {"1", "2"}

    plot_outputs = sorted((out_csv_dir / "plots").glob("*.png"))
    assert len(plot_outputs) == 2
    plot_names = {path.name for path in plot_outputs}
    assert any("_fly1_" in name for name in plot_names)
    assert any("_fly2_" in name for name in plot_names)


def test_build_wide_csv_adds_auc_columns(tmp_path):
    dataset_root = tmp_path / "secured_dataset"
    fly_dir = dataset_root / "session_a"
    csv_dir = fly_dir / "angle_distance_rms_envelope"
    csv_dir.mkdir(parents=True, exist_ok=True)

    csv_path = csv_dir / "session_a_testing_1_fly1_distances.csv"
    before = np.full(BEFORE_FRAMES, 1.0)
    during = np.linspace(1.0, 5.0, DURING_FRAMES)
    after = np.full(AFTER_FRAMES, 1.0)
    values = np.concatenate([before, during, after])
    pd.DataFrame({"envelope_of_rms": values}).to_csv(csv_path, index=False)

    out_csv = tmp_path / "all_envelope_rows_wide.csv"
    build_wide_csv(
        [str(dataset_root)],
        str(out_csv),
        measure_cols=["envelope_of_rms"],
        fps_fallback=40.0,
        distance_limits=(0.0, 300.0),
    )

    df = pd.read_csv(out_csv)
    for column in AUC_COLUMNS:
        assert column in df.columns
    assert np.isfinite(df.loc[0, "AUC-During"])
    assert abs(df.loc[0, "Peak-Value"] - 5.0) < 1e-6
    assert "global_min" in df.columns
    assert "global_max" in df.columns
    assert "trimmed_global_min" in df.columns
    assert "trimmed_global_max" in df.columns
    assert "local_min" in df.columns
    assert "local_max" in df.columns
    assert "local_min_during" in df.columns
    assert "local_max_during" in df.columns
    assert "local_max_over_global_min" in df.columns
    assert "local_max_during_over_global_min" in df.columns
    assert "non_reactive_flag" in df.columns
    expected_trimmed_min = float(np.nanpercentile(values, 2.5))
    expected_trimmed_max = float(np.nanpercentile(values, 97.5))
    assert math.isclose(df.loc[0, "global_min"], 1.0)
    assert math.isclose(df.loc[0, "global_max"], 5.0)
    assert math.isclose(df.loc[0, "trimmed_global_min"], expected_trimmed_min)
    assert math.isclose(df.loc[0, "trimmed_global_max"], expected_trimmed_max)
    assert math.isclose(df.loc[0, "local_min"], 1.0)
    assert math.isclose(df.loc[0, "local_max"], 5.0)
    assert math.isclose(df.loc[0, "local_min_during"], 1.0)
    assert math.isclose(df.loc[0, "local_max_during"], 5.0)
    assert math.isclose(df.loc[0, "local_max_over_global_min"], 5.0)
    assert math.isclose(df.loc[0, "local_max_during_over_global_min"], 5.0)
    assert df.loc[0, "non_reactive_flag"] == 1.0

    flagged_file = out_csv.with_name(out_csv.stem + "_flagged_flies.txt")
    assert flagged_file.exists()
    flagged_lines = [line for line in flagged_file.read_text().splitlines() if line and not line.startswith("#")]
    assert flagged_lines, "expected at least one flagged entry"
    dataset, fly, fly_number, trimmed_min, trimmed_max = flagged_lines[0].split(",")
    assert dataset == "session_a"
    assert math.isclose(float(trimmed_min), expected_trimmed_min, rel_tol=1e-6)
    assert math.isclose(float(trimmed_max), expected_trimmed_max, rel_tol=1e-6)


def test_local_extrema_respect_distance_limits(tmp_path):
    dataset_root = tmp_path / "secured_dataset"
    fly_dir = dataset_root / "session_a"
    csv_dir = fly_dir / "angle_distance_rms_envelope"
    csv_dir.mkdir(parents=True, exist_ok=True)

    csv_path = csv_dir / "session_a_testing_1_fly1_distances.csv"
    values = np.array([5.0, 12.0, 16.0, 300.0], dtype=float)
    pd.DataFrame({"envelope_of_rms": values}).to_csv(csv_path, index=False)

    out_csv = tmp_path / "all_envelope_rows_wide.csv"
    build_wide_csv(
        [str(dataset_root)],
        str(out_csv),
        measure_cols=["envelope_of_rms"],
        fps_fallback=40.0,
        distance_limits=(10.0, 250.0),
    )

    df = pd.read_csv(out_csv)
    assert math.isclose(df.loc[0, "global_min"], 12.0)
    assert math.isclose(df.loc[0, "global_max"], 16.0)
    assert math.isclose(df.loc[0, "local_min"], 12.0)
    assert math.isclose(df.loc[0, "local_max"], 16.0)
    assert math.isclose(df.loc[0, "local_max_over_global_min"], 16.0 / 12.0)
    assert np.isnan(df.loc[0, "local_min_during"])
    assert np.isnan(df.loc[0, "local_max_during"])
    assert np.isnan(df.loc[0, "local_max_during_over_global_min"])


def test_global_extrema_aggregate_across_trials(tmp_path):
    dataset_root = tmp_path / "secured_dataset"
    fly_dir = dataset_root / "session_b"
    csv_dir = fly_dir / "angle_distance_rms_envelope"
    csv_dir.mkdir(parents=True, exist_ok=True)

    csv_one = csv_dir / "session_b_testing_1_fly2_distances.csv"
    csv_two = csv_dir / "session_b_testing_2_fly2_distances.csv"
    pd.DataFrame({"envelope_of_rms": [15.0, 20.0]}).to_csv(csv_one, index=False)
    pd.DataFrame({"envelope_of_rms": [8.0, 30.0]}).to_csv(csv_two, index=False)

    out_csv = tmp_path / "all_envelope_rows_wide.csv"
    build_wide_csv(
        [str(dataset_root)],
        str(out_csv),
        measure_cols=["envelope_of_rms"],
        fps_fallback=40.0,
        distance_limits=(0.0, 40.0),
    )

    df = pd.read_csv(out_csv)
    assert len(df) == 2
    assert np.allclose(df["global_min"], [8.0, 8.0])
    assert np.allclose(df["global_max"], [30.0, 30.0])
    assert math.isclose(df.loc[df["trial_label"] == "testing_1"].iloc[0]["local_min"], 15.0)
    assert math.isclose(df.loc[df["trial_label"] == "testing_1"].iloc[0]["local_max"], 20.0)
    assert math.isclose(
        df.loc[df["trial_label"] == "testing_1"].iloc[0]["local_max_over_global_min"],
        20.0 / 8.0,
    )
    assert np.isnan(
        df.loc[df["trial_label"] == "testing_1"].iloc[0]["local_min_during"]
    )
    assert np.isnan(
        df.loc[df["trial_label"] == "testing_1"].iloc[0]["local_max_during"]
    )
    assert np.isnan(
        df.loc[df["trial_label"] == "testing_1"].iloc[0]["local_max_during_over_global_min"]
    )
    assert math.isclose(df.loc[df["trial_label"] == "testing_2"].iloc[0]["local_min"], 8.0)
    assert math.isclose(df.loc[df["trial_label"] == "testing_2"].iloc[0]["local_max"], 30.0)
    assert math.isclose(
        df.loc[df["trial_label"] == "testing_2"].iloc[0]["local_max_over_global_min"],
        30.0 / 8.0,
    )
    assert np.isnan(
        df.loc[df["trial_label"] == "testing_2"].iloc[0]["local_min_during"]
    )
    assert np.isnan(
        df.loc[df["trial_label"] == "testing_2"].iloc[0]["local_max_during"]
    )
    assert np.isnan(
        df.loc[df["trial_label"] == "testing_2"].iloc[0]["local_max_during_over_global_min"]
    )
