"""Pre-test vs post-training mean traces in ``dataset_mean_comparisons``.

``dataset_mean_traces_tvc.py`` overlays two arms per odor presentation. Both
arms have always been two *datasets* — a trained cohort and its control, i.e.
different flies.

The ``*-Sensitivity-*`` cohorts need the other axis: two *phases of one
dataset*, the same fly before and after training. That is a within-subject
comparison, so it needs no control cohort and no concentration matching.

The two arms already come from separate frames inside ``main`` (``_prepare``
takes the frame as an argument), so the phase mode feeds the pre-test table as
the second arm rather than a second dataset.
"""

from pathlib import Path

import pytest

import scripts.analysis.dataset_mean_traces_tvc as tvc


def _args(argv):
    return tvc.build_parser(argv)


BASE = [
    "--wide-csv", "/x/testing.parquet",
    "--train-dataset", "Hex-Sensitivity-24-0.1",
    "--out-dir", "/x/out",
]


# ── the CLI ───────────────────────────────────────────────────────────────


def test_the_pretest_table_can_be_supplied():
    a = _args(BASE + ["--control-dataset", "Hex-Sensitivity-24-0.1",
                      "--pretest-wide-csv", "/x/pretest.parquet"])
    assert str(a.pretest_wide_csv) == "/x/pretest.parquet"


def test_the_pretest_table_is_optional():
    """Every trained-vs-control invocation must keep working untouched."""
    a = _args(["--wide-csv", "/x/w.parquet", "--train-dataset", "A",
               "--control-dataset", "B", "--out-dir", "/x/out"])
    assert a.pretest_wide_csv is None


def test_a_control_dataset_is_not_required_in_phase_mode():
    """There is no control cohort — the fly's own pre-test IS the control."""
    a = _args(BASE + ["--pretest-wide-csv", "/x/pretest.parquet"])
    assert a.control_dataset in (None, "", "Hex-Sensitivity-24-0.1")


def test_a_control_dataset_is_still_required_without_the_pretest_table():
    with pytest.raises(SystemExit):
        _args(["--wide-csv", "/x/w.parquet", "--train-dataset", "A",
               "--out-dir", "/x/out"])


# ── the arm labels ────────────────────────────────────────────────────────


def test_phase_mode_labels_the_arms_by_phase():
    """"Trained"/"Control" would misdescribe them: both arms are the same
    flies, and neither is a control cohort."""
    assert tvc.phase_arm_labels() == ("Post-training", "Pre-test")


def test_phase_mode_has_its_own_filename_tag():
    """It must not overwrite the training_vs_control figures in the same tree."""
    assert tvc.PHASE_TAG == "post_vs_pretest"
    assert tvc.PHASE_TAG != "training_vs_control"


def test_the_trained_vs_control_labels_are_unchanged():
    assert tvc.DEFAULT_ARM_LABELS == ("Trained", "Control")


# ── _prepare must be able to select a phase ───────────────────────────────


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _wide(trial_type, prefix):
    rows = []
    for fn in (1, 2):
        for i, odor in enumerate(["hexanol", "citral"], start=1):
            row = {
                "dataset": "Hex-Sensitivity-24-0.1", "fly": "b1", "fly_number": fn,
                "trial_type": trial_type, "trial_label": f"{prefix}_{i}_{odor}",
                "fps": 40.0, "trace_len": 200,
            }
            row.update({f"dir_val_{j}": 1.0 for j in range(200)})
            rows.append(row)
    return pd.DataFrame(rows)


def test_prepare_still_defaults_to_testing():
    """Every trained-vs-control caller relies on this."""
    both = pd.concat([_wide("testing", "testing"), _wide("pretest", "pretest")],
                     ignore_index=True)
    _out, _rows, sub = tvc._prepare(
        both, "Hex-Sensitivity-24-0.1", fps=40.0, odor_on_s=30.0
    )
    assert set(sub["trial_type"]) == {"testing"}


def test_prepare_can_select_the_pretest_phase():
    """It filtered to trial_type == "testing" unconditionally, so the pre-test
    arm came back empty: "No usable pre-test traces"."""
    both = pd.concat([_wide("testing", "testing"), _wide("pretest", "pretest")],
                     ignore_index=True)
    _out, _rows, sub = tvc._prepare(
        both, "Hex-Sensitivity-24-0.1", fps=40.0, odor_on_s=30.0,
        trial_type="pretest",
    )
    assert set(sub["trial_type"]) == {"pretest"}
    assert len(sub) == 4


# ── the figure title must name the arms actually drawn ────────────────────


def test_the_default_title_follows_the_arm_labels():
    """It hardcoded "Trained vs Control", so a pre-test figure was captioned as
    a trained-vs-control one — the caption is what a reader trusts."""
    from scripts.analysis.dataset_means_specific_flies import default_arm_title

    assert default_arm_title("Hexanol", "Post-training", "Pre-test") == (
        "Hexanol - Post-training vs Pre-test"
    )


def test_the_trained_vs_control_title_is_unchanged():
    from scripts.analysis.dataset_means_specific_flies import default_arm_title

    assert default_arm_title("Hexanol", "Trained", "Control") == (
        "Hexanol - Trained vs Control"
    )


def test_a_three_arm_title_still_names_the_naive_arm():
    from scripts.analysis.dataset_means_specific_flies import default_arm_title

    assert default_arm_title("Hexanol", "Trained", "Control", with_naive=True) == (
        "Hexanol - Trained vs Control vs Naive"
    )
