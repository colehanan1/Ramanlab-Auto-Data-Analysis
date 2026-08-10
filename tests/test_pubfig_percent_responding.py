"""Tests for the "% of flies responding" companion metric.

The score figure answers "how hard did they respond"; this one answers "how
many of them responded at all". Both are drawn by
``pubfig_score_train_vs_control`` off the same predictions CSV, the same month
filter and the same flagged-fly exclusions, so the two figures always describe
the *same* flies. ``test_both_metrics_agree_on_odor_order_and_n`` is the test
that keeps that promise honest — a reader putting the two panels side by side
is entitled to assume column 3 is the same odor in both.

Why the response threshold is 2 and not 1: the ordinal PER model emits only
{-1, 0, 2, 3, 4, 5} — it never outputs 1 — so ">= 1" and ">= 2" select exactly
the same trials on real data. 2 is used because it is the lowest score the
model can actually produce that means "the proboscis extended". -1 is a
non-reactive trial, which is a non-response, not a missing value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analysis import pubfig_score_train_vs_control as pub  # noqa: E402


# ---------------------------------------------------------------------------
# The responder threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("score", [2.0, 3.0, 4.0, 5.0])
def test_a_positive_score_is_a_response(score):
    assert pub.responded(score) is True


@pytest.mark.parametrize("score", [0.0, -1.0])
def test_zero_and_non_reactive_are_not_responses(score):
    assert pub.responded(score) is False


def test_threshold_of_one_would_select_the_same_trials():
    """The model never emits 1, so >=1 and >=2 must agree on every real score."""
    for score in (-1.0, 0.0, 2.0, 3.0, 4.0, 5.0):
        assert (score >= 1.0) == pub.responded(score)


# ---------------------------------------------------------------------------
# Percentage / error / significance arithmetic
# ---------------------------------------------------------------------------


def _predictions(tmp_path: Path, spec) -> Path:
    """Build a predictions CSV from {(dataset, fly_number): score} per odor."""
    rows = []
    for (dataset, fly_number), score in spec.items():
        for trial in ("testing_2_acv", "testing_3_benzaldehyde"):
            rows.append(
                {
                    "dataset": dataset,
                    "fly": "july_31_batch_1_rig_2",
                    "fly_number": fly_number,
                    "trial_label": trial,
                    "score": score,
                    "trial_type": "testing",
                    "fly_type": "GR5a-Old",
                }
            )
    out = tmp_path / "model_predictions.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _half_responding(tmp_path: Path) -> Path:
    return _predictions(
        tmp_path,
        {
            ("Hex-Training-24-0.01", 1): 4.0,
            ("Hex-Training-24-0.01", 2): 0.0,
            ("Hex-Control-24-0.01", 1): 0.0,
            ("Hex-Control-24-0.01", 2): 0.0,
        },
    )


def test_percentage_counts_flies_not_trials(tmp_path):
    """One of two trained flies responds -> 50%, not 25% or 100%.

    Each fly contributes two trials here; pooling trials instead of flies would
    still give 50% in this case, so the fly that responds does so on BOTH its
    trials to keep the two definitions distinguishable elsewhere.
    """
    rows = pub.rows_from_percent_responding(
        _half_responding(tmp_path), "Hex-Training-24-0.01", config=None
    )
    assert set(rows["n_train"]) == {2}
    assert rows["mean_train"].tolist() == pytest.approx([50.0, 50.0])
    assert rows["mean_ctrl"].tolist() == pytest.approx([0.0, 0.0])


def test_error_bar_is_the_binomial_standard_error(tmp_path):
    """50% of 2 flies -> 100*sqrt(.5*.5/2) = 35.36 points."""
    rows = pub.rows_from_percent_responding(
        _half_responding(tmp_path), "Hex-Training-24-0.01", config=None
    )
    assert rows["sem_train"].tolist() == pytest.approx([35.355339] * 2, abs=1e-4)
    # A cohort where nobody responds has no spread at all.
    assert rows["sem_ctrl"].tolist() == pytest.approx([0.0, 0.0])


def test_everyone_responds_versus_nobody_is_significant(tmp_path):
    """Fisher's exact on 5 v 5 with a clean split is p < 0.05."""
    spec = {}
    for i in range(1, 6):
        spec[("Hex-Training-24-0.01", i)] = 4.0
        spec[("Hex-Control-24-0.01", i)] = 0.0
    rows = pub.rows_from_percent_responding(
        _predictions(tmp_path, spec), "Hex-Training-24-0.01", config=None
    )
    assert rows["mean_train"].tolist() == pytest.approx([100.0, 100.0])
    assert rows["mean_ctrl"].tolist() == pytest.approx([0.0, 0.0])
    assert all(p < 0.05 for p in rows["p_value"]), rows["p_value"].tolist()


def test_identical_cohorts_are_not_significant(tmp_path):
    spec = {}
    for i in range(1, 6):
        spec[("Hex-Training-24-0.01", i)] = 4.0
        spec[("Hex-Control-24-0.01", i)] = 4.0
    rows = pub.rows_from_percent_responding(
        _predictions(tmp_path, spec), "Hex-Training-24-0.01", config=None
    )
    assert all(p == pytest.approx(1.0) for p in rows["p_value"])


def test_percentages_never_leave_the_zero_to_hundred_range(tmp_path):
    rows = pub.rows_from_percent_responding(
        _half_responding(tmp_path), "Hex-Training-24-0.01", config=None
    )
    for col in ("mean_train", "mean_ctrl"):
        assert rows[col].between(0.0, 100.0).all()


# ---------------------------------------------------------------------------
# The two metrics must describe the same flies
# ---------------------------------------------------------------------------


def _two_era_predictions(tmp_path: Path) -> Path:
    rows = []
    for dataset, trained in (
        ("Hex-Training-24-0.01", True),
        ("Hex-Control-24-0.01", False),
    ):
        for fly, resp in (("april_22_batch_1", 0.0), ("july_31_batch_1_rig_2", 4.0)):
            for fly_number in (1, 2):
                for trial in ("testing_2_acv", "testing_3_benzaldehyde"):
                    rows.append(
                        {
                            "dataset": dataset,
                            "fly": fly,
                            "fly_number": fly_number,
                            "trial_label": trial,
                            "score": resp if trained else 0.0,
                            "trial_type": "testing",
                            "fly_type": "GR5a-Old",
                        }
                    )
    out = tmp_path / "model_predictions.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def test_both_metrics_agree_on_odor_order_and_n(tmp_path):
    """Companion panels must line up column for column, with the same cohort n."""
    preds = _two_era_predictions(tmp_path)
    kw = dict(config=None, fly_months=("july",))
    score = pub.rows_from_score_summary(preds, "Hex-Training-24-0.01", **kw)
    pct = pub.rows_from_percent_responding(preds, "Hex-Training-24-0.01", **kw)
    assert list(pct["odor"]) == list(score["odor"])
    assert list(pct["n_train"]) == list(score["n_train"])
    assert list(pct["n_ctrl"]) == list(score["n_ctrl"])
    assert list(pct["is_trained"]) == list(score["is_trained"])


def test_month_filter_applies_to_the_percentage_metric(tmp_path):
    """July-only: every trained fly responds, so 100% rather than the pooled 50%."""
    preds = _two_era_predictions(tmp_path)
    pooled = pub.rows_from_percent_responding(preds, "Hex-Training-24-0.01", config=None)
    july = pub.rows_from_percent_responding(
        preds, "Hex-Training-24-0.01", config=None, fly_months=("july",)
    )
    assert pooled["mean_train"].tolist() == pytest.approx([50.0, 50.0])
    assert july["mean_train"].tolist() == pytest.approx([100.0, 100.0])
    assert set(july["n_train"]) == {2}


def test_flagged_exclusions_apply_to_the_percentage_metric(tmp_path):
    """Dropping the one responding fly must take the cohort to 0%."""
    preds = _half_responding(tmp_path)
    flagged = tmp_path / "flagged.csv"
    pd.DataFrame(
        [
            {
                "dataset": "Hex-Training-24-0.01",
                "fly": "july_31_batch_1_rig_2",
                "fly_number": 1,
                "FLY-State(1, 0, -1)": -1,
                "comment": "dead",
            }
        ]
    ).to_csv(flagged, index=False)
    rows = pub.rows_from_percent_responding(
        preds, "Hex-Training-24-0.01", config=None, flagged_flies_csv=str(flagged)
    )
    assert set(rows["n_train"]) == {1}
    assert rows["mean_train"].tolist() == pytest.approx([0.0, 0.0])


# ---------------------------------------------------------------------------
# Plot scale — the percentage panel must not reuse the score axis
# ---------------------------------------------------------------------------


def test_metric_specs_have_the_right_axes():
    assert pub.SCORE_METRIC.y_label == "Mean PER Score"
    assert (pub.SCORE_METRIC.y_min, pub.SCORE_METRIC.y_max) == (-1.5, 5.0)
    assert pub.PERCENT_METRIC.y_label == "% of Flies Responding"
    assert (pub.PERCENT_METRIC.y_min, pub.PERCENT_METRIC.y_max) == (0.0, 100.0)


def test_percentage_panel_uses_the_percentage_axis(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = pub.rows_from_percent_responding(
        _half_responding(tmp_path), "Hex-Training-24-0.01", config=None
    )
    fig, ax = plt.subplots()
    pub.plot_train_vs_control(ax, rows, title="t", metric=pub.PERCENT_METRIC)
    assert ax.get_ylabel() == "% of Flies Responding"
    assert ax.get_ylim() == (0.0, 100.0)
    plt.close(fig)


def test_score_panel_axis_is_unchanged_by_the_new_metric(tmp_path):
    """Regression: adding the metric must not move the score figure's axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = pub.rows_from_score_summary(
        _half_responding(tmp_path), "Hex-Training-24-0.01", config=None
    )
    fig, ax = plt.subplots()
    pub.plot_train_vs_control(ax, rows, title="t")  # no metric -> score default
    assert ax.get_ylabel() == "Mean PER Score"
    assert ax.get_ylim() == (-1.5, 5.0)
    plt.close(fig)


def test_cli_accepts_the_metric_flag(tmp_path, capsys):
    out_dir = tmp_path / "figs"
    pub.main(
        [
            "dataset",
            "--train-dataset", "Hex-Training-24-0.01",
            "--predictions-csv", str(_half_responding(tmp_path)),
            "--figures-dir", str(out_dir),
            "--metric", "percent-responding",
            "--out-stem", "pct",
            "--title", "pct",
        ]
    )
    assert (out_dir / "pct.png").exists()
    assert "%" in capsys.readouterr().out
