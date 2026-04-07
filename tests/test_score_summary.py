import importlib.util
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "scripts", PROJECT_ROOT / "src"):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

MODULE_PATH = PROJECT_ROOT / "scripts" / "analysis" / "score_summary.py"

spec = importlib.util.spec_from_file_location("score_summary", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _make_score_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    train_patterns = [
        [0, 5, 4, 4, 5],
        [1, 4, 5, 5, 4],
        [0, 5, 5, 4, 5],
        [1, 4, 4, 5, 4],
        [0, 5, 4, 5, 5],
    ]
    ctrl_patterns = [
        [0, 0, 0, 1, 0],
        [1, 1, 0, 0, 1],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
    ]

    for idx, scores in enumerate(train_patterns, start=1):
        fly = f"train_fly_{idx}"
        for trial_label, score in zip(
            ("testing_1", "testing_2", "testing_3", "testing_4", "testing_5"), scores
        ):
            rows.append(
                {
                    "dataset": "EB-Training",
                    "fly": fly,
                    "fly_number": str(idx),
                    "trial_label": trial_label,
                    "score": score,
                    "trial_type": "testing",
                }
            )

    for idx, scores in enumerate(ctrl_patterns, start=1):
        fly = f"ctrl_fly_{idx}"
        for trial_label, score in zip(
            ("testing_1", "testing_2", "testing_3", "testing_4", "testing_5"), scores
        ):
            rows.append(
                {
                    "dataset": "EB-Control",
                    "fly": fly,
                    "fly_number": str(idx),
                    "trial_label": trial_label,
                    "score": score,
                    "trial_type": "testing",
                }
            )

    return rows


def test_compute_training_vs_control_summary_uses_score_distributions(tmp_path):
    csv_path = tmp_path / "scores.csv"
    pd.DataFrame(_make_score_rows()).to_csv(csv_path, index=False)

    df = module._load_scores(csv_path)
    summary = module._compute_training_vs_control_summary(df)

    eb_rows = summary[
        (summary["training_dataset"] == "EB-Training")
        & (summary["control_dataset"] == "EB-Control")
        & (summary["odor"] == "Ethyl Butyrate")
    ].sort_values("trial_num")
    hex_rows = summary[
        (summary["training_dataset"] == "EB-Training")
        & (summary["control_dataset"] == "EB-Control")
        & (summary["odor"] == "Hexanol")
    ].sort_values("trial_num")

    assert eb_rows["trial_num"].tolist() == [2, 4, 5]
    assert hex_rows["trial_num"].tolist() == [1, 3]

    eb_row = eb_rows.iloc[0]
    assert eb_row["n_flies_train"] == 5
    assert eb_row["n_flies_ctrl"] == 5
    assert eb_row["mean_score_train"] == pytest.approx(4.6, rel=1e-6)
    assert eb_row["mean_score_ctrl"] == pytest.approx(0.4, rel=1e-6)
    assert eb_row["score_p_value"] < 0.02
    assert eb_row["significance"] in {"*", "**"}
    assert bool(eb_row["is_trained"]) is True


def test_generate_score_summary_writes_train_vs_control_outputs_from_scores_only(tmp_path):
    csv_path = tmp_path / "scores.csv"
    out_dir = tmp_path / "out"
    pd.DataFrame(_make_score_rows()).to_csv(csv_path, index=False)

    module.generate_score_summary(csv_path=csv_path, out_dir=out_dir, overwrite=True)

    train_ctrl_csv = out_dir / "score_summary_train_vs_control.csv"
    train_ctrl_plot = out_dir / "mean_score_train_vs_ctrl_EB-Training.png"

    assert (out_dir / "score_summary_by_odor_testing.csv").exists()
    assert (out_dir / "mean_score_heatmap.png").exists()
    assert train_ctrl_csv.exists()
    assert train_ctrl_plot.exists()

    exported = pd.read_csv(train_ctrl_csv)
    assert "score_p_value" in exported.columns
    assert "n_flies_train" in exported.columns
    eb_export = exported[
        (exported["training_dataset"] == "EB-Training")
        & (exported["control_dataset"] == "EB-Control")
    ].sort_values("trial_num")
    assert eb_export["trial_num"].tolist() == [1, 2, 3, 4, 5]
    assert eb_export["odor"].tolist().count("Hexanol") == 2
    assert eb_export["odor"].tolist().count("Ethyl Butyrate") == 3
