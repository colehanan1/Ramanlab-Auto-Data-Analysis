"""Pre-test vs test comparison for the *-Sensitivity-* cohorts.

The protocol runs the SAME 7 odors on the SAME fly before and after training
(``pretest_1..7`` -> ``training_1..6`` -> ``testing_1..7``), so the naive panel
is a within-subject control. That pairing is the whole point: the tests here are
paired (McNemar on responded/not, Wilcoxon signed-rank on the ordinal score),
and n counts FLIES, because the fly is the unit that is paired.
"""


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


# ── response-rate test: McNemar by default, Fisher's exact on request ──────


def _rate_preds(tmp_path):
    # 6 flies; pre: 1 responder per odor, post: 5 responders per odor.
    return _predictions(tmp_path, {
        fn: {"pretest": [3 if fn == 1 else 0] * 3,
             "testing": [0 if fn == 1 else 3] * 3}
        for fn in range(1, 7)
    })


def test_rate_test_defaults_to_mcnemar(tmp_path):
    """The pipeline never passes rate_test, so its figures keep McNemar."""
    paired = pvt.load_paired_scores(_rate_preds(tmp_path))
    summary = pvt.per_odor_summary(paired, binary_threshold=2)
    odor = paired[paired["odor"] == summary["odor"].iloc[0]]
    expected = pvt.mcnemar_p(odor["score_pre"] >= 2, odor["score_post"] >= 2)
    assert summary["p_rate"].iloc[0] == pytest.approx(expected)


def test_fisher_rate_test_compares_the_two_phases_as_a_2x2_table(tmp_path):
    from scipy.stats import fisher_exact

    paired = pvt.load_paired_scores(_rate_preds(tmp_path))
    summary = pvt.per_odor_summary(paired, binary_threshold=2, rate_test="fisher")
    # pre 1/6 responders, post 5/6.
    expected = fisher_exact([[1, 5], [5, 1]]).pvalue
    assert summary["p_rate"].tolist() == pytest.approx([expected] * 3)


def test_fisher_p_is_none_on_empty_input():
    assert pvt.fisher_p([], []) is None


def test_unknown_rate_test_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        pvt.per_odor_summary(pvt.load_paired_scores(_simple(tmp_path)), rate_test="chi2")


def test_rate_test_flag_defaults_to_mcnemar_and_accepts_fisher():
    base = ["--predictions-csv", "p.csv", "--out-root", "o"]
    assert pvt._parse_args(base).rate_test == "mcnemar"
    assert pvt._parse_args(base + ["--rate-test", "fisher"]).rate_test == "fisher"


def test_render_cohort_with_fisher_writes_the_same_set(tmp_path):
    written = pvt.render_cohort(
        pvt.load_paired_scores(_rate_preds(tmp_path)),
        cohort="Hex-Sensitivity-24-0.1", out_dir=tmp_path / "figs", rate_test="fisher",
    )
    stems = " ".join(p.stem for p in written)
    for expected in ("score_bars", "rate_bars", "trained_odor"):
        assert expected in stems, expected


# ── figure wording: sensitization vs conditioning ─────────────────────────


def _paired_for(cohort, n_flies=8, odors=("Hexanol", "Citral", "Linalool")):
    """Paired scores whose panel includes both cohorts' exposed odor, Hexanol."""
    return pd.DataFrame([
        {"dataset_canon": cohort, "fly": "b1", "fly_number": fn, "odor": odor,
         "score_pre": fn % 3, "score_post": (fn + 2) % 6}
        for fn in range(1, n_flies + 1)
        for odor in odors
    ])


@pytest.fixture
def figure_text(monkeypatch):
    """Every string drawn on each figure render_cohort saves, by file stem."""
    from pathlib import Path

    import matplotlib.figure
    import matplotlib.text

    seen = {}
    real_savefig = matplotlib.figure.Figure.savefig

    def spy(fig, fname, *args, **kwargs):
        real_savefig(fig, fname, *args, **kwargs)  # drawn, so tick text is final
        seen[Path(fname).stem] = " | ".join(
            t.get_text() for t in fig.findobj(matplotlib.text.Text) if t.get_text()
        )

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", spy)
    return seen


#: Figure -> (wording on a *-Sensitivity-* cohort, wording on a conditioned one).
_WORDING = {
    "score_bars": (["Pre-exposure", "Post-exposure",
                    "pre-exposure vs post-exposure"],
                   ["Pre-test (naive)", "Post-training", "pre-test vs post-training"]),
    "rate_bars": (["Pre-exposure", "Post-exposure",
                   "pre-exposure vs post-exposure"],
                  ["Pre-test (naive)", "Post-training", "pre-test vs post-training"]),
    "trained_odor": (["Hexanol (sensitized odor)", "Pre-exposure", "Post-exposure"],
                     ["Hexanol (CS+)", "Pre-test", "Post-training"]),
    "slopes": (["Hexanol (sensitized odor)", "Pre-exposure", "Post-exposure"],
               ["Hexanol (CS+)", "Pre-test", "Post-training"]),
    "heatmap": (["Pre-exposure", "Post-exposure"],
                ["Pre-test (naive)", "Post-training"]),
}


def test_a_sensitivity_cohort_says_pre_and_post_sensitization(tmp_path, figure_text):
    """Nothing is paired with the odor in a *-Sensitivity-* cohort, so its panels
    are pre-/post-exposure and the exposed odor is the sensitized odor."""
    cohort = "Hex-Sensitivity-24-0.1"
    pvt.render_cohort(_paired_for(cohort), cohort=cohort, out_dir=tmp_path)
    assert {s.split(f"{cohort}_")[-1] for s in figure_text} == set(_WORDING)
    for stem, text in figure_text.items():
        text = text.replace("-\n", "-")  # the two-bar ticks wrap after the hyphen
        for wanted in _WORDING[stem.split(f"{cohort}_")[-1]][0]:
            assert wanted in text, (stem, wanted)
        for old in ("Pre-test", "pre-test", "Post-training", "post-training", "CS+", "naive"):
            assert old not in text, (stem, old)


def test_a_conditioned_cohort_with_a_pretest_keeps_its_wording(tmp_path, figure_text):
    """Hex-HexTest-Training-* pairs the odor with light AND runs a pre-test, so it
    goes through these same figures and must still say pre-test / CS+."""
    cohort = "Hex-HexTest-Training-32-1"
    pvt.render_cohort(_paired_for(cohort), cohort=cohort, out_dir=tmp_path)
    assert {s.split(f"{cohort}_")[-1] for s in figure_text} == set(_WORDING)
    for stem, text in figure_text.items():
        for wanted in _WORDING[stem.split(f"{cohort}_")[-1]][1]:
            assert wanted in text, (stem, wanted)
        assert "ensitiz" not in text, stem


def test_the_shared_phase_labels_keep_the_conditioning_wording():
    """starvation_*, time_of_day_trend and others import these; the sensitivity
    wording is picked per figure, never by changing the constants."""
    assert pvt.PRE_LABEL == "Pre-test (naive)"
    assert pvt.POST_LABEL == "Post-training"


# ── odor labels: "<Odor> <c>% (v/v)" on a sensitivity figure ──────────────

#: Raw ~/odor_pins.txt labels, as batches recorded after ~2026-10-06 load.
_PINS = ("3-octonol-0.1", "hexanol-0.1", "isoamylacetate-1")
_PINS_TEXT = ("3-Octanol 0.1% (v/v)", "Hexanol 0.1% (v/v)", "Isoamyl Acetate 1% (v/v)")


def _stems(figure_text, cohort):
    return {stem.split(f"{cohort}_")[-1]: text for stem, text in figure_text.items()}


def test_a_sensitivity_cohort_draws_each_odor_with_its_concentration(tmp_path, figure_text):
    cohort = "IAA-Sensitivity-1-1"
    pvt.render_cohort(_paired_for(cohort, odors=_PINS), cohort=cohort, out_dir=tmp_path)
    drawn = _stems(figure_text, cohort)
    assert set(drawn) == set(_WORDING)
    for stem in ("score_bars", "rate_bars", "heatmap"):
        for wanted in _PINS_TEXT:
            assert wanted in drawn[stem], (stem, wanted)
    for stem in ("trained_odor", "slopes"):
        assert "Isoamyl Acetate 1% (v/v) (sensitized odor)" in drawn[stem], stem
    for stem, text in drawn.items():
        for raw in _PINS:
            assert raw not in text, (stem, raw)


def test_a_label_with_no_concentration_gets_no_percent(tmp_path, figure_text):
    """Batches before the pins labels recorded no concentration; none is invented."""
    cohort = "IAA-Sensitivity-40-1"
    odors = ("Isoamyl Acetate", "Hexanol", "3-Octanol")
    pvt.render_cohort(_paired_for(cohort, odors=odors), cohort=cohort, out_dir=tmp_path)
    drawn = _stems(figure_text, cohort)
    assert set(drawn) == set(_WORDING)
    for stem, text in drawn.items():
        assert "(v/v)" not in text, stem
    for stem in ("score_bars", "heatmap"):
        for odor in odors:
            assert odor in drawn[stem], (stem, odor)
    assert "Isoamyl Acetate (sensitized odor)" in drawn["slopes"]


def test_a_conditioned_cohort_draws_its_odor_labels_as_loaded(tmp_path, figure_text):
    """Hex-HexTest-Training-* is not a sensitivity cohort, so every odor label,
    pins token or not, is drawn exactly as loaded."""
    cohort = "Hex-HexTest-Training-32-1"
    odors = ("ethylbutyrate-0.1", "hexanol-1", "isoamylacetate-1")
    pvt.render_cohort(_paired_for(cohort, odors=odors), cohort=cohort, out_dir=tmp_path)
    drawn = _stems(figure_text, cohort)
    assert set(drawn) == set(_WORDING)
    for stem in ("score_bars", "rate_bars", "heatmap"):
        for odor in odors:
            assert odor in drawn[stem], (stem, odor)
    for stem in ("trained_odor", "slopes"):
        assert f"{cohort} — hexanol-1 (CS+)" in drawn[stem], stem
    for stem, text in drawn.items():
        assert "(v/v)" not in text, stem


def test_odor_text_formats_only_sensitivity_cohorts():
    assert pvt._odor_text("isoamylacetate-1", "IAA-Sensitivity-1-1") == "Isoamyl Acetate 1% (v/v)"
    assert pvt._odor_text("3-Octanol", "3Oct-Sensitivity-24-0.1") == "3-Octanol"
    assert pvt._odor_text("isoamylacetate-1", "Hex-HexTest-Training-32-1") == "isoamylacetate-1"


def test_the_summary_csv_keeps_the_loaded_labels(tmp_path):
    """Display only: the CSV other scripts read keeps the grouping key."""
    cohort = "IAA-Sensitivity-1-1"
    pvt.render_cohort(_paired_for(cohort, odors=_PINS), cohort=cohort, out_dir=tmp_path)
    summary = pd.read_csv(tmp_path / f"{cohort}_summary.csv")
    assert sorted(summary["odor"]) == sorted(_PINS)


# ── the sensitized odor, matched by name ──────────────────────────────────


@pytest.mark.parametrize("cohort,token", [
    ("IAA-Sensitivity-1-1", "isoamylacetate-1"),
    ("IAA-Sensitivity5s-1-1", "isoamylacetate-1"),
    ("3Oct-Sensitivity-1-0.1", "3-octonol-0.1"),
])
def test_a_pins_labelled_cohort_gets_its_sensitized_odor_figures(tmp_path, cohort, token):
    """The prefix match compared "isoamylacetate-1" with "Isoamyl Acetate", so
    these cohorts never drew trained_odor / slopes and their tick was never bold."""
    rows = [
        {"dataset": cohort, "fly": "october_08_batch_2", "fly_number": fn,
         "trial_label": f"{phase}_{i}_{odor}", "score": score, "trial_type": phase}
        for fn in range(1, 9)
        for phase, score in (("pretest", 0), ("testing", 4))
        for i, odor in enumerate((token, "citral-1", "linalool-1"), start=1)
    ]
    path = tmp_path / "p.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    written = pvt.render_cohort(pvt.load_paired_scores(path), cohort=cohort,
                                out_dir=tmp_path / "figs")
    stems = " ".join(p.stem for p in written)
    assert "trained_odor" in stems
    assert "slopes" in stems


def _bold_ticks(summary, cohort):
    import matplotlib.pyplot as plt

    fig, ax = pvt.build_grouped_bars(summary, cohort=cohort, kind="score", p_key="p_score")
    bold = {t.get_text() for t in ax.get_xticklabels() if t.get_fontweight() == "bold"}
    plt.close(fig)
    return bold


def test_the_sensitized_odor_tick_is_bold_for_a_pins_label():
    cohort = "IAA-Sensitivity-1-1"
    summary = pvt.per_odor_summary(_paired_for(cohort, odors=_PINS))
    assert _bold_ticks(summary, cohort) == {"Isoamyl Acetate 1% (v/v)"}


def test_mixed_labels_keep_the_old_label_row(tmp_path, figure_text):
    """IAA-Sensitivity-24-1 holds both spellings as separate odor groups. Its
    old-label "Isoamyl Acetate" row was the one the prefix match found, so the
    trained-odor and slope figures keep drawing it; both of its ticks are bold."""
    cohort = "IAA-Sensitivity-24-1"
    paired = _paired_for(cohort, odors=("Isoamyl Acetate", "isoamylacetate-1", "Hexanol"))
    summary = pvt.per_odor_summary(paired)

    assert pvt._trained_odor_row(summary, cohort)["odor"] == "Isoamyl Acetate"
    # Whatever the order: a row the original prefix rule matches wins.
    flipped = summary.iloc[::-1].reset_index(drop=True)
    assert flipped["odor"].iloc[0] == "isoamylacetate-1"
    assert pvt._trained_odor_row(flipped, cohort)["odor"] == "Isoamyl Acetate"

    assert _bold_ticks(summary, cohort) == {"Isoamyl Acetate", "Isoamyl Acetate 1% (v/v)"}

    pvt.render_cohort(paired, cohort=cohort, out_dir=tmp_path)
    drawn = _stems(figure_text, cohort)
    for stem in ("trained_odor", "slopes"):
        assert "Isoamyl Acetate (sensitized odor)" in drawn[stem], stem
        assert "Isoamyl Acetate 1% (v/v) (sensitized odor)" not in drawn[stem], stem


def test_name_matching_applies_to_a_conditioned_cohort_too():
    cohort = "IAA-Odor10s-Training-24-1"
    summary = pvt.per_odor_summary(_paired_for(cohort, odors=("citral-1", "isoamylacetate-1")))
    assert pvt._trained_odor_row(summary, cohort)["odor"] == "isoamylacetate-1"
    assert _bold_ticks(summary, cohort) == {"isoamylacetate-1"}
