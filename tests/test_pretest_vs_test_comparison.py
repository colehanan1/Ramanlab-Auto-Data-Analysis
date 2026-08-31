"""Pre-test vs test comparison for the *-Sensitivity-* cohorts.

The protocol runs the SAME 7 odors on the SAME fly before and after training
(``pretest_1..7`` -> ``training_1..6`` -> ``testing_1..7``), so the naive panel
is a within-subject control. That pairing is the whole point: the tests here are
paired (McNemar on responded/not, Wilcoxon signed-rank on the ordinal score),
and n counts FLIES, because the fly is the unit that is paired.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import pretest_vs_test_comparison as pvt
from scripts.analysis.score_summary import _load_scores

ODORS = ["hexanol", "citral", "linalool"]


def _predictions(tmp_path, per_fly_scores):
    """per_fly_scores: {fly_number: {"pretest": [...], "testing": [...]}}."""
    rows = []
    for fly_number, phases in per_fly_scores.items():
        for phase, scores in phases.items():
            for i, (odor, score) in enumerate(zip(ODORS, scores), start=1):
                if score is None:
                    continue
                rows.append({
                    "dataset": "Hex-Sensitivity-24-0.1",
                    "fly": "august_28_batch_1",
                    "fly_number": fly_number,
                    "trial_label": f"{phase}_{i}_{odor}",
                    "score": score,
                    "trial_type": phase,
                })
    path = tmp_path / "model_predictions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _simple(tmp_path, n_flies=4, pre=1, post=4):
    return _predictions(tmp_path, {
        fn: {"pretest": [pre] * 3, "testing": [post] * 3}
        for fn in range(1, n_flies + 1)
    })


# ── loading each phase separately ─────────────────────────────────────────


def test_load_scores_still_defaults_to_testing(tmp_path):
    df = _load_scores(_simple(tmp_path))
    assert set(df["trial_type"]) == {"testing"}


def test_load_scores_can_be_asked_for_the_pretest_phase(tmp_path):
    df = _load_scores(_simple(tmp_path), trial_types=("pretest",))
    assert set(df["trial_type"]) == {"pretest"}
    assert len(df) == 4 * 3


def test_load_scores_raises_when_the_requested_phase_is_absent(tmp_path):
    path = _predictions(tmp_path, {1: {"testing": [3, 3, 3]}})
    with pytest.raises(RuntimeError):
        _load_scores(path, trial_types=("pretest",))


# ── pairing ───────────────────────────────────────────────────────────────


def test_pairing_produces_one_row_per_fly_and_odor(tmp_path):
    paired = pvt.load_paired_scores(_simple(tmp_path))
    assert len(paired) == 4 * 3
    assert set(paired.columns) >= {
        "dataset_canon", "fly", "fly_number", "odor", "score_pre", "score_post",
    }


def test_pairing_keeps_the_pre_and_post_score_on_one_row(tmp_path):
    paired = pvt.load_paired_scores(_simple(tmp_path, pre=1, post=4))
    assert set(paired["score_pre"]) == {1}
    assert set(paired["score_post"]) == {4}


def test_pairing_matches_the_same_odor_not_the_same_trial_number(tmp_path):
    """Odor order is randomised between the panels — pretest_1 is not testing_1.

    Here fly 1 sees hexanol first in the pretest and LAST in the post-test. A
    pairing keyed on trial index would compare hexanol against linalool.
    """
    rows = []
    pre_order = ["hexanol", "citral", "linalool"]
    post_order = ["linalool", "citral", "hexanol"]
    scores = {"hexanol": (0, 5), "citral": (1, 1), "linalool": (2, 2)}
    for i, odor in enumerate(pre_order, start=1):
        rows.append({"dataset": "Hex-Sensitivity-24-0.1", "fly": "b1", "fly_number": 1,
                     "trial_label": f"pretest_{i}_{odor}", "score": scores[odor][0],
                     "trial_type": "pretest"})
    for i, odor in enumerate(post_order, start=1):
        rows.append({"dataset": "Hex-Sensitivity-24-0.1", "fly": "b1", "fly_number": 1,
                     "trial_label": f"testing_{i}_{odor}", "score": scores[odor][1],
                     "trial_type": "testing"})
    path = tmp_path / "p.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    paired = pvt.load_paired_scores(path).set_index("odor")
    hexanol = [o for o in paired.index if "hexanol" in o.lower()][0]
    assert paired.loc[hexanol, "score_pre"] == 0
    assert paired.loc[hexanol, "score_post"] == 5


def test_a_fly_missing_one_half_of_a_pair_is_dropped(tmp_path):
    """An unpaired trial has no within-subject comparison; it must not sneak in."""
    path = _predictions(tmp_path, {
        1: {"pretest": [1, 1, 1], "testing": [4, 4, 4]},
        2: {"pretest": [1, 1, None], "testing": [4, 4, 4]},  # no pre for linalool
    })
    paired = pvt.load_paired_scores(path)
    assert len(paired) == 5
    fly2 = paired[paired["fly_number"].astype(str) == "2"]
    assert not any("linalool" in o.lower() for o in fly2["odor"])


def test_dropped_pairs_are_reported_not_silently_absorbed(tmp_path):
    path = _predictions(tmp_path, {
        1: {"pretest": [1, 1, 1], "testing": [4, 4, 4]},
        2: {"pretest": [1, 1, None], "testing": [4, 4, 4]},
    })
    paired = pvt.load_paired_scores(path)
    assert paired.attrs["n_unpaired"] == 1


# ── the paired statistics ─────────────────────────────────────────────────


def test_mcnemar_detects_a_clean_all_flies_gained_response():
    pre = np.zeros(10, dtype=bool)
    post = np.ones(10, dtype=bool)
    assert pvt.mcnemar_p(pre, post) < 0.01


def test_mcnemar_returns_none_when_nothing_changed():
    """b + c == 0: no discordant pairs, so the test is undefined, not p=1."""
    pre = np.array([True, False, True, False])
    assert pvt.mcnemar_p(pre, pre.copy()) is None


def test_mcnemar_is_symmetric_in_magnitude():
    pre = np.array([False] * 8 + [True] * 2)
    post = np.array([True] * 8 + [False] * 2)
    assert pvt.mcnemar_p(pre, post) == pytest.approx(pvt.mcnemar_p(post, pre))


def test_mcnemar_ignores_concordant_pairs():
    """Only the flies that CHANGED carry information in a paired binary test."""
    disc_pre = np.array([False, False, False])
    disc_post = np.array([True, True, True])
    p_small = pvt.mcnemar_p(disc_pre, disc_post)

    padded_pre = np.concatenate([disc_pre, np.ones(50, dtype=bool)])
    padded_post = np.concatenate([disc_post, np.ones(50, dtype=bool)])
    assert pvt.mcnemar_p(padded_pre, padded_post) == pytest.approx(p_small)


def test_wilcoxon_detects_a_consistent_score_increase():
    pre = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    post = np.array([4, 5, 4, 5, 4, 5, 4, 5])
    assert pvt.wilcoxon_p(pre, post) < 0.05


def test_wilcoxon_returns_none_when_every_difference_is_zero():
    pre = np.array([2, 2, 2, 2, 2, 2])
    assert pvt.wilcoxon_p(pre, pre.copy()) is None


def test_wilcoxon_returns_none_below_the_minimum_pair_count():
    """Two pairs can never reach p < 0.05; reporting a number would mislead."""
    assert pvt.wilcoxon_p(np.array([0, 1]), np.array([4, 5])) is None


def test_the_tests_return_none_on_empty_input():
    empty = np.array([], dtype=float)
    assert pvt.wilcoxon_p(empty, empty) is None
    assert pvt.mcnemar_p(empty.astype(bool), empty.astype(bool)) is None


# ── per-odor summary ──────────────────────────────────────────────────────


def test_summary_has_one_row_per_odor(tmp_path):
    summary = pvt.per_odor_summary(pvt.load_paired_scores(_simple(tmp_path)))
    assert len(summary) == 3
    assert set(summary.columns) >= {
        "odor", "n_flies", "mean_pre", "mean_post", "pct_pre", "pct_post",
        "p_score", "p_rate",
    }


def test_summary_n_counts_flies_not_trials(tmp_path):
    """The fly is the paired unit, so n is flies — 4 flies, not 12 trials."""
    summary = pvt.per_odor_summary(pvt.load_paired_scores(_simple(tmp_path, n_flies=4)))
    assert set(summary["n_flies"]) == {4}


def test_summary_response_percentage_uses_the_binary_threshold(tmp_path):
    """binary_threshold=2: a score of 1 is not a response, 2 is."""
    path = _predictions(tmp_path, {
        fn: {"pretest": [1, 1, 1], "testing": [2, 2, 2]} for fn in range(1, 5)
    })
    summary = pvt.per_odor_summary(pvt.load_paired_scores(path), binary_threshold=2)
    assert set(summary["pct_pre"]) == {0.0}
    assert set(summary["pct_post"]) == {100.0}


def test_summary_means_are_pooled_over_flies(tmp_path):
    path = _predictions(tmp_path, {
        1: {"pretest": [0, 0, 0], "testing": [4, 4, 4]},
        2: {"pretest": [2, 2, 2], "testing": [4, 4, 4]},
    })
    summary = pvt.per_odor_summary(pvt.load_paired_scores(path))
    assert set(summary["mean_pre"]) == {1.0}
    assert set(summary["mean_post"]) == {4.0}


def test_summary_is_ordered_deterministically(tmp_path):
    a = pvt.per_odor_summary(pvt.load_paired_scores(_simple(tmp_path)))
    b = pvt.per_odor_summary(pvt.load_paired_scores(_simple(tmp_path)))
    assert list(a["odor"]) == list(b["odor"])


def test_summary_carries_a_p_value_for_a_real_effect(tmp_path):
    path = _predictions(tmp_path, {
        fn: {"pretest": [0, 0, 0], "testing": [5, 5, 5]} for fn in range(1, 9)
    })
    summary = pvt.per_odor_summary(pvt.load_paired_scores(path))
    assert (summary["p_rate"] < 0.05).all()
    assert (summary["p_score"] < 0.05).all()


def test_summary_tolerates_an_odor_with_no_variation(tmp_path):
    """No discordant pairs for an odor → p is None for it, not a crash."""
    path = _predictions(tmp_path, {
        fn: {"pretest": [3, 3, 3], "testing": [3, 3, 3]} for fn in range(1, 5)
    })
    summary = pvt.per_odor_summary(pvt.load_paired_scores(path))
    assert summary["p_score"].isna().all()
    assert summary["p_rate"].isna().all()


# ── figures ───────────────────────────────────────────────────────────────


def test_the_four_figures_are_written(tmp_path):
    out = tmp_path / "figs"
    written = pvt.render_cohort(
        pvt.load_paired_scores(_simple(tmp_path)),
        cohort="Hex-Sensitivity-24-0.1",
        out_dir=out,
    )
    names = sorted(p.name for p in written)
    assert len(names) == len(set(names))
    assert all(p.exists() and p.stat().st_size > 0 for p in written)


def test_the_figure_set_covers_every_requested_view(tmp_path):
    written = pvt.render_cohort(
        pvt.load_paired_scores(_simple(tmp_path)),
        cohort="Hex-Sensitivity-24-0.1",
        out_dir=tmp_path / "figs",
    )
    stems = " ".join(p.stem for p in written)
    for expected in ("score_bars", "rate_bars", "trained_odor", "slopes", "heatmap"):
        assert expected in stems, expected


def test_rendering_an_empty_frame_writes_nothing(tmp_path):
    empty = pvt.load_paired_scores(_simple(tmp_path)).iloc[0:0]
    written = pvt.render_cohort(empty, cohort="X", out_dir=tmp_path / "figs")
    assert written == []


def test_the_score_figures_reuse_the_fixed_prgn_ramp():
    """SCORE_COLORS is CVD-validated and pinned; a second ramp must not appear."""
    from scripts.analysis.score_summary import _score_cmap

    assert pvt._score_cmap is _score_cmap


def test_the_y_axis_labels_come_from_the_shared_module():
    from scripts.analysis.per_axis_labels import PERCENT_Y_LABEL, SCORE_Y_LABEL

    assert pvt.SCORE_Y_LABEL == SCORE_Y_LABEL
    assert pvt.PERCENT_Y_LABEL == PERCENT_Y_LABEL


# ── the CS+ odor of a sensitivity cohort ──────────────────────────────────


@pytest.mark.parametrize("cohort,expected", [
    ("Hex-Sensitivity-24-0.1", "Hexanol"),
    ("IAA-Sensitivity-24-1", "Isoamyl Acetate"),
    ("EB-Sensitivity-24-1", "Ethyl Butyrate"),
    ("3Oct-Sensitivity-24-0.1", "3-Octanol"),
])
def test_trained_odor_resolves_for_the_sensitivity_cohorts(cohort, expected):
    """_trained_label matched only -Training-/-Control-, so these fell through
    to the dataset NAME: no CS+ odor could be identified, no tick was bolded,
    and the spotlight/slope figures silently did not render."""
    from scripts.analysis.envelope_visuals import _trained_label

    assert _trained_label(cohort) == expected


def test_trained_odor_still_resolves_for_the_existing_cohorts():
    from scripts.analysis.envelope_visuals import _trained_label

    assert _trained_label("Hex-Training-24-0.1") == "Hexanol"
    assert _trained_label("EB-Control-24-1") == "Ethyl Butyrate"


# ── graceful behaviour before the pre-test data exists ────────────────────


def test_a_predictions_csv_with_no_pretest_rows_yields_an_empty_frame(tmp_path):
    """Until a full re-run produces pretest rows, model_predictions.csv has none.

    Every non-sensitivity cohort will never have them at all. Raising here made
    the pipeline step exit non-zero on every run, so this is an empty result
    with a note instead.
    """
    path = _predictions(tmp_path, {1: {"testing": [3, 3, 3]}})
    paired = pvt.load_paired_scores(path)
    assert paired.empty
    assert list(paired.columns) != []


def test_an_empty_result_still_carries_the_unpaired_count(tmp_path):
    path = _predictions(tmp_path, {1: {"testing": [3, 3, 3]}})
    assert pvt.load_paired_scores(path).attrs["n_unpaired"] == 3


def test_a_predictions_csv_with_no_testing_rows_yields_an_empty_frame(tmp_path):
    path = _predictions(tmp_path, {1: {"pretest": [3, 3, 3]}})
    assert pvt.load_paired_scores(path).empty


def test_main_exits_cleanly_when_there_is_nothing_to_pair(tmp_path, capsys):
    path = _predictions(tmp_path, {1: {"testing": [3, 3, 3]}})
    pvt.main(["--predictions-csv", str(path), "--out-root", str(tmp_path / "out")])
    assert "no paired" in capsys.readouterr().out.lower()


# ── the IAA cohort's odor label ───────────────────────────────────────────


def test_the_rig_spelling_of_isoamyl_acetate_gets_a_display_label():
    """Rig files write "IsoamylAcetate"; _DISPLAY_LABEL_LOWER only carries the
    canon keys, so the token fell through as the raw string. On its own that is
    an ugly tick — but it also means the odor column never matched
    _trained_label("IAA-Sensitivity-24-1") == "Isoamyl Acetate", so the CS+
    spotlight and slope figures silently did not render for that cohort."""
    from scripts.analysis.envelope_visuals import _display_label_ci

    assert _display_label_ci("isoamylacetate") == "Isoamyl Acetate"
    assert _display_label_ci("IsoamylAcetate") == "Isoamyl Acetate"


def test_the_already_mapped_odors_are_unchanged():
    """The fallback must only fire for tokens that had no label at all."""
    from scripts.analysis.envelope_visuals import _display_label_ci

    assert _display_label_ci("hexanol") == "Hexanol"
    assert _display_label_ci("3-octonol") == "3-Octanol"
    assert _display_label_ci("ethylbutyrate") == "Ethyl Butyrate"
    assert _display_label_ci("acv") == "Apple Cider Vinegar"


def test_an_unknown_token_still_comes_back_unchanged():
    from scripts.analysis.envelope_visuals import _display_label_ci

    assert _display_label_ci("not-an-odor") == "not-an-odor"


def test_the_iaa_cohort_renders_its_trained_odor_figures(tmp_path):
    """End to end: the CS+ odor of IAA-Sensitivity must match and render."""
    rows = []
    for fn in range(1, 9):
        for phase, score in (("pretest", 0), ("testing", 5)):
            for i, odor in enumerate(["isoamylacetate", "hexanol", "citral"], start=1):
                rows.append({
                    "dataset": "IAA-Sensitivity-24-1", "fly": "b1", "fly_number": fn,
                    "trial_label": f"{phase}_{i}_{odor}", "score": score,
                    "trial_type": phase,
                })
    path = tmp_path / "p.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    written = pvt.render_cohort(
        pvt.load_paired_scores(path),
        cohort="IAA-Sensitivity-24-1",
        out_dir=tmp_path / "figs",
    )
    stems = " ".join(p.stem for p in written)
    assert "trained_odor" in stems
    assert "slopes" in stems


# ── heatmap row order ─────────────────────────────────────────────────────


def test_heatmap_rows_are_ordered_numerically_by_fly(tmp_path):
    """String sorting put fly #10 between #1 and #2, which reads as a data error."""
    path = _predictions(tmp_path, {
        fn: {"pretest": [1, 1, 1], "testing": [4, 4, 4]} for fn in range(1, 13)
    })
    order = pvt.heatmap_fly_order(pvt.load_paired_scores(path))
    assert [int(str(f).rsplit("#", 1)[1]) for f in order] == list(range(1, 13))


def test_heatmap_fly_order_is_stable_across_calls(tmp_path):
    path = _predictions(tmp_path, {
        fn: {"pretest": [1, 1, 1], "testing": [4, 4, 4]} for fn in range(1, 6)
    })
    paired = pvt.load_paired_scores(path)
    assert pvt.heatmap_fly_order(paired) == pvt.heatmap_fly_order(paired)


def test_heatmap_fly_order_handles_a_non_numeric_fly_number(tmp_path):
    """fly_number is UNKNOWN for sidecar-less batches; it must not crash."""
    rows = []
    for fn in ("2", "UNKNOWN", "1"):
        for phase, score in (("pretest", 1), ("testing", 4)):
            rows.append({"dataset": "Hex-Sensitivity-24-0.1", "fly": "b1",
                         "fly_number": fn, "trial_label": f"{phase}_1_hexanol",
                         "score": score, "trial_type": phase})
    path = tmp_path / "p.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    order = pvt.heatmap_fly_order(pvt.load_paired_scores(path))
    assert len(order) == 3


# ── slope plot overplotting ───────────────────────────────────────────────


def test_slope_jitter_separates_flies_that_share_a_trajectory():
    """PER scores are integers, so many flies land on identical (pre, post)
    pairs and collapse into one visible line — contradicting the "one line per
    fly" caption. A small deterministic x offset keeps every fly visible."""
    offsets = pvt.slope_jitter(12)
    assert len(set(np.round(offsets, 6))) == 12


def test_slope_jitter_is_deterministic():
    """Re-running the pipeline must not reshuffle the figure."""
    assert np.allclose(pvt.slope_jitter(8), pvt.slope_jitter(8))


def test_slope_jitter_never_touches_the_score_axis():
    """The whole point: a measured 0 must be DRAWN at 0.

    Jittering y would render a score of 0 at -0.16, which on an axis where -1 is
    a real model output reads as a genuinely negative score.
    """
    offsets = pvt.slope_jitter(20)
    assert offsets.ndim == 1
    assert np.max(np.abs(offsets)) <= 0.2  # x units, well inside the category


def test_slope_jitter_is_centred_on_the_category():
    """Offsets must be symmetric so the cloud is not shifted off its tick."""
    assert pvt.slope_jitter(9).sum() == pytest.approx(0.0)


def test_slope_jitter_handles_the_degenerate_counts():
    assert len(pvt.slope_jitter(0)) == 0
    assert pvt.slope_jitter(1).tolist() == [0.0]


# ── flagged flies must be excluded, like every other figure ───────────────


def _flagged_csv(tmp_path, flagged_fly_numbers):
    """The flagged-flies truth table: FLY-State != 1 means exclude."""
    rows = [
        {"dataset": "Hex-Sensitivity-24-0.1", "fly": "august_28_batch_1",
         "fly_number": fn, "FLY-State(1, 0, -1)": (0 if fn in flagged_fly_numbers else 1)}
        for fn in range(1, 5)
    ]
    path = tmp_path / "flagged-flys-truth.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_flagged_flies_are_dropped_from_the_paired_set(tmp_path):
    """A flagged fly must not reach the figures.

    _load_scores only applies the flagged-flies table when `threshold` is not
    None, so passing the CSV alone silently did nothing and the new figures
    would have kept flies every other figure drops.
    """
    preds = _simple(tmp_path, n_flies=4)
    paired = pvt.load_paired_scores(
        preds, flagged_flies_csv=_flagged_csv(tmp_path, {2, 3})
    )
    assert set(paired["fly_number"].astype(int)) == {1, 4}


def test_no_flagged_csv_keeps_every_fly(tmp_path):
    paired = pvt.load_paired_scores(_simple(tmp_path, n_flies=4))
    assert set(paired["fly_number"].astype(int)) == {1, 2, 3, 4}


def test_a_missing_flagged_csv_path_is_not_fatal(tmp_path):
    paired = pvt.load_paired_scores(
        _simple(tmp_path, n_flies=4), flagged_flies_csv=str(tmp_path / "nope.csv")
    )
    assert len(paired) == 4 * 3


def test_flagging_lowers_the_reported_n(tmp_path):
    """n is flies, so dropping flies must move it — silently keeping them would
    overstate the sample."""
    preds = _simple(tmp_path, n_flies=4)
    full = pvt.per_odor_summary(pvt.load_paired_scores(preds))
    trimmed = pvt.per_odor_summary(
        pvt.load_paired_scores(preds, flagged_flies_csv=_flagged_csv(tmp_path, {2, 3}))
    )
    assert set(full["n_flies"]) == {4}
    assert set(trimmed["n_flies"]) == {2}
