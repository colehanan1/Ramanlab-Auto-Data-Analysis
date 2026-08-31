"""The scorer has to see the pretest table, or the comparison has no data.

``combine`` partitions the wide output by trial type: ``main_trial_allow`` is
``{"testing"}``, so pretest rows land ONLY in the ``trial_type_exports`` table
for pretest. ``reaction_prediction`` reads one ``data_csv`` and filters to
``("testing",)``, so without this wiring ``model_predictions.csv`` would carry
zero pretest rows and every pre-vs-post figure would come out empty.

Two additions, both additive:
  * ``extra_data_csvs`` — more wide tables to score alongside ``data_csv``;
  * ``trial_types`` — the trial types to keep, still ``("testing",)`` by default
    so every existing config scores exactly what it scored before.
"""

from pathlib import Path

import pandas as pd
import pytest

from fbpipe.config import ReactionPredictionSettings
from fbpipe.steps.predict_reactions import _filter_trial_types, _read_input_tables


def _table(trial_types, dataset="Hex-Sensitivity-24-0.1"):
    return pd.DataFrame(
        {
            "dataset": [dataset] * len(trial_types),
            "fly": ["august_28_batch_1"] * len(trial_types),
            "fly_number": [1] * len(trial_types),
            "trial_label": [f"{t}_{i}_hexanol" for i, t in enumerate(trial_types, 1)],
            "trial_type": list(trial_types),
        }
    )


# ── the trial-type filter ─────────────────────────────────────────────────


def test_the_filter_still_defaults_to_testing_only():
    """Every existing config relies on this default; it must not move."""
    df = _table(["testing", "pretest", "training"])
    kept = _filter_trial_types(df)
    assert list(kept["trial_type"]) == ["testing"]


def test_the_filter_can_keep_pretest_alongside_testing():
    df = _table(["testing", "pretest", "training"])
    kept = _filter_trial_types(df, allowed=("testing", "pretest"))
    assert sorted(kept["trial_type"]) == ["pretest", "testing"]


def test_the_filter_never_lets_training_through():
    df = _table(["testing", "pretest", "training"])
    kept = _filter_trial_types(df, allowed=("testing", "pretest"))
    assert "training" not in set(kept["trial_type"])


# ── settings ──────────────────────────────────────────────────────────────


def test_settings_default_to_the_current_behaviour():
    s = ReactionPredictionSettings()
    assert s.extra_data_csvs == ()
    assert s.trial_types == ("testing",)


def test_settings_accept_a_pretest_table_and_type():
    s = ReactionPredictionSettings(
        data_csv="/x/wide.csv",
        extra_data_csvs=("/x/wide_pretest.csv",),
        trial_types=("testing", "pretest"),
    )
    assert s.extra_data_csvs == ("/x/wide_pretest.csv",)
    assert s.trial_types == ("testing", "pretest")


# ── reading the tables ────────────────────────────────────────────────────


def test_reading_one_table_returns_it_unchanged(tmp_path):
    main = tmp_path / "wide.csv"
    _table(["testing", "testing"]).to_csv(main, index=False)
    df = _read_input_tables(main, [])
    assert len(df) == 2


def test_reading_concatenates_the_pretest_table(tmp_path):
    main = tmp_path / "wide.csv"
    extra = tmp_path / "wide_pretest.csv"
    _table(["testing", "testing"]).to_csv(main, index=False)
    _table(["pretest", "pretest", "pretest"]).to_csv(extra, index=False)

    df = _read_input_tables(main, [extra])
    assert len(df) == 5
    assert df["trial_type"].value_counts().to_dict() == {"testing": 2, "pretest": 3}


def test_a_missing_extra_table_is_skipped_not_fatal(tmp_path):
    """A cohort with no naive panel simply has no pretest table yet.

    The pretest table does not exist until a run produces one, and every
    non-sensitivity cohort will never have pretest rows. Aborting the whole
    scoring step over that would take the pipeline down.
    """
    main = tmp_path / "wide.csv"
    _table(["testing"]).to_csv(main, index=False)
    df = _read_input_tables(main, [tmp_path / "does_not_exist.csv"])
    assert len(df) == 1


def test_an_empty_extra_table_is_harmless(tmp_path):
    main = tmp_path / "wide.csv"
    extra = tmp_path / "wide_pretest.csv"
    _table(["testing"]).to_csv(main, index=False)
    _table([]).to_csv(extra, index=False)
    assert len(_read_input_tables(main, [extra])) == 1


def test_the_main_table_is_still_required(tmp_path):
    with pytest.raises(FileNotFoundError):
        _read_input_tables(tmp_path / "missing.csv", [])


def test_rows_from_both_tables_keep_their_identity(tmp_path):
    """Pairing pre with post depends on dataset/fly/fly_number surviving intact."""
    main = tmp_path / "wide.csv"
    extra = tmp_path / "wide_pretest.csv"
    _table(["testing"]).to_csv(main, index=False)
    _table(["pretest"]).to_csv(extra, index=False)

    df = _read_input_tables(main, [extra])
    for col in ("dataset", "fly", "fly_number", "trial_label", "trial_type"):
        assert col in df.columns
    assert df["fly"].nunique() == 1
    assert set(df["trial_type"]) == {"testing", "pretest"}


# ── containment: pretest rows must not leak into the existing figures ─────


def _predictions_csv(tmp_path):
    """A predictions CSV in the real shape, carrying both phases for one fly."""
    rows = []
    for phase in ("pretest", "testing"):
        for i, odor in enumerate(["hexanol", "citral", "linalool"], start=1):
            rows.append({
                "dataset": "Hex-Sensitivity-24-0.1",
                "fly": "august_28_batch_1",
                "fly_number": 1,
                "trial_label": f"{phase}_{i}_{odor}",
                "score": 3,
                "trial_type": phase,
            })
    path = tmp_path / "model_predictions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_score_summary_drops_pretest_rows(tmp_path):
    """Every existing figure reads model_predictions.csv through _load_scores.

    Adding pretest rows to that file is only safe because _load_scores filters
    to trial_type == "testing". If that filter ever goes, every train-vs-control
    figure silently doubles its trial count with naive trials.
    """
    from scripts.analysis.score_summary import _load_scores

    df = _load_scores(_predictions_csv(tmp_path))
    assert set(df["trial_type"]) == {"testing"}
    assert len(df) == 3


def test_score_summary_keeps_every_testing_trial(tmp_path):
    from scripts.analysis.score_summary import _load_scores

    df = _load_scores(_predictions_csv(tmp_path))
    assert sorted(df["trial"]) == [
        "testing_1_hexanol", "testing_2_citral", "testing_3_linalool",
    ]
