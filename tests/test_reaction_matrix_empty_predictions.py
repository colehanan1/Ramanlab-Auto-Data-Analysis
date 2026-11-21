"""The reaction-matrix figure step skips (never aborts the run) when the
predictions spreadsheet holds no rows -- e.g. every fly of a small dataset was
flagged non-reactive, as the first Caleb manual-odor fly was on 2026-11-19."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts.pipeline import run_workflows as rw  # noqa: E402


def test_header_only_csv_has_no_rows(tmp_path):
    p = tmp_path / "model_predictions.csv"
    p.write_text("dataset,fly,trial_type,prediction\n")
    assert rw._csv_has_data_rows(p) is False


def test_csv_with_a_row_has_rows(tmp_path):
    p = tmp_path / "model_predictions.csv"
    p.write_text("dataset,fly,trial_type,prediction\nX,f1,testing,0\n")
    assert rw._csv_has_data_rows(p) is True


def test_blank_trailing_lines_do_not_count(tmp_path):
    p = tmp_path / "model_predictions.csv"
    p.write_text("dataset,fly\n\n\n")
    assert rw._csv_has_data_rows(p) is False
