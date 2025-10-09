import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "scripts"))

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
