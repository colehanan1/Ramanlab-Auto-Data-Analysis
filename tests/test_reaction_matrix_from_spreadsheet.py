import importlib.util
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "src"):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

MODULE_PATH = PROJECT_ROOT / "scripts" / "reaction_matrix_from_spreadsheet.py"

spec = importlib.util.spec_from_file_location("reaction_matrix_from_spreadsheet", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_filter_trial_types_removes_training_rows():
    df = pd.DataFrame(
        {
            "dataset": ["set1", "set1", "set2"],
            "fly": ["f1", "f2", "f3"],
            "fly_number": ["1", "2", "3"],
            "trial_label": ["Trial1", "Trial2", "Trial3"],
            "prediction": [1, 0, 1],
            "trial_type": ["testing", "training", "Testing"],
        }
    )

    filtered = module._filter_trial_types(df, allowed=("testing",))

    assert len(filtered) == 2
    assert all(filtered["trial_type"].str.lower() == "testing")


def test_load_predictions_raises_when_no_testing(tmp_path):
    csv_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "dataset": ["set1"],
            "fly": ["f1"],
            "fly_number": ["1"],
            "trial_label": ["Trial1"],
            "prediction": [1],
            "trial_type": ["training"],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(RuntimeError):
        module._load_predictions(csv_path)
