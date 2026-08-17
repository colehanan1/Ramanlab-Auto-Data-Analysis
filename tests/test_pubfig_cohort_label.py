"""``--cohort-label`` renames what the default title is *about*.

The pipeline renders the same cohort under two metrics, whose titles differ by
their y-axis label ("Mean PER Score" vs "% of Flies Responding"). Passing a
fully-formed ``--title`` per metric would make the caller reconstruct that
label; ``--cohort-label`` instead substitutes only the subject, so both titles
stay correct without the pipeline knowing either metric's wording.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analysis import pubfig_score_train_vs_control as pub  # noqa: E402


def _predictions(tmp_path: Path) -> Path:
    rows = []
    for dataset in ("Hex-Training-24-0.01", "Hex-Control-24-0.01"):
        for fly_number in (1, 2):
            for trial in ("testing_2_acv", "testing_3_benzaldehyde"):
                rows.append(
                    {
                        "dataset": dataset,
                        "fly": "august_05_batch_1_rig_2",
                        "fly_number": fly_number,
                        "trial_label": trial,
                        "score": 4.0,
                        "trial_type": "testing",
                        "fly_type": "GR5a-Old",
                    }
                )
    out = tmp_path / "model_predictions.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _title_of(png: Path) -> str:
    """Read the rendered title back out of the sibling SVG (text is kept)."""
    return (png.with_suffix(".svg")).read_text(errors="ignore")


def _run(tmp_path: Path, extra: list[str]) -> Path:
    out_dir = tmp_path / "figs"
    pub.main(
        [
            "dataset",
            "--train-dataset", "Hex-Training-24-0.01",
            "--predictions-csv", str(_predictions(tmp_path)),
            "--figures-dir", str(out_dir),
            "--out-stem", "t",
            *extra,
        ]
    )
    return out_dir / "t.png"


def test_cohort_label_replaces_the_dataset_in_the_score_title(tmp_path):
    svg = _title_of(_run(tmp_path, ["--cohort-label", "Hex-24-0.01, August"]))
    assert "Mean PER Score" in svg
    assert "Hex-24-0.01, August" in svg


def test_cohort_label_replaces_the_dataset_in_the_percent_title(tmp_path):
    svg = _title_of(
        _run(
            tmp_path,
            ["--cohort-label", "Hex-24-0.01, August", "--metric", "percent-responding"],
        )
    )
    assert "Flies Responding" in svg
    assert "Hex-24-0.01, August" in svg


def test_without_a_cohort_label_the_dataset_name_is_still_used(tmp_path):
    svg = _title_of(_run(tmp_path, []))
    assert "Hex-Training-24-0.01" in svg


def test_an_explicit_title_still_wins(tmp_path):
    """--title is the escape hatch and must not be overridden by the label."""
    svg = _title_of(
        _run(tmp_path, ["--cohort-label", "ignored", "--title", "My Exact Title"])
    )
    assert "My Exact Title" in svg
    assert "ignored" not in svg
