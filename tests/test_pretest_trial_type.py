"""``pretest`` is a first-class trial type, not a flavour of ``testing``.

Before this change ``TESTING_HINT_PREFIXES`` listed ``"pretest"``, so the naive
pre-training panel of the ``*-Sensitivity-*`` cohorts was classified as
``testing``.  Two things went wrong silently:

  * ``Raw-Testing-PER-Traces/<cohort>/`` rendered 14 panels per fly
    (``pretest_1..7`` + ``testing_1..7``) instead of the 7 post-training ones;
  * every reaction-percentage / mean-score figure pooled naive trials with
    post-training trials, roughly halving any learning effect.

The pre-test protocol (see ``test_pretest_protocol.py``) is
``pretest_1..7 -> training_1..6 -> testing_1..7``, i.e. the *same* fly sees the
*same* 7 odors before and after training.  That is a within-subject control, so
the two halves have to stay separable all the way through the pipeline.
"""

from pathlib import Path

import pytest

from scripts.analysis.envelope_combined import (
    PRETEST_REGEX,
    TESTING_REGEX,
    TRAINING_REGEX,
    _build_trial_configs,
    _infer_category,
    _pretest_hints,
    _testing_hints,
    _training_hints,
)
from scripts.analysis import envelope_exports


def _csv(stem: str) -> Path:
    return Path("/data/Hex-Sensitivity-24-0.1/aug_28_batch_1") / f"{stem}.csv"


# ── categorisation ────────────────────────────────────────────────────────


def test_pretest_is_its_own_category():
    assert _infer_category(
        _csv("aug_28_fly1_pretest_3_Hexanol_fly1_distances_20260828_101010")
    ) == "pretest"


def test_testing_and_training_are_unmoved():
    """The other two arms must not shift — every finished cohort depends on them."""
    assert _infer_category(
        _csv("aug_28_fly1_testing_3_Hexanol_fly1_distances_20260828_101010")
    ) == "testing"
    assert _infer_category(
        _csv("aug_28_fly1_training_2_Hexanol_fly1_distances_20260828_101010")
    ) == "training"


def test_posttest_still_reads_as_testing():
    """``posttest`` shares a prefix with nothing; it must not follow pretest out."""
    assert _infer_category(_csv("aug_28_fly1_posttest_1_distances")) == "testing"


def test_pretest_dropped_from_the_testing_hints():
    assert "pretest" not in _testing_hints()
    assert "pretest" in _pretest_hints()


def test_pretest_does_not_leak_into_the_training_hints():
    """``pretrain`` is a training hint and ``pretest`` must not collide with it."""
    assert "pretest" not in _training_hints()


def test_category_falls_back_to_the_folder_when_the_filename_is_bare():
    path = Path(
        "/data/Hex-Sensitivity-24-0.1/aug_28_batch_1/"
        "aug_28_batch_1_pretest_3/rms_envelope.csv"
    )
    assert _infer_category(path) == "pretest"


def test_unclassifiable_paths_still_default_to_testing():
    assert _infer_category(Path("/data/whatever/rms_envelope.csv")) == "testing"


# ── regexes ───────────────────────────────────────────────────────────────


def test_pretest_regex_matches_the_label_form():
    m = PRETEST_REGEX.search("pretest_3_hexanol")
    assert m is not None
    assert m.group(1) == "3"
    assert m.group(2) == "hexanol"


def test_the_three_regexes_do_not_match_each_other():
    """`.search` is used, so a substring collision would mis-key a whole arm."""
    assert TESTING_REGEX.search("pretest_3_hexanol") is None
    assert TRAINING_REGEX.search("pretest_3_hexanol") is None
    assert PRETEST_REGEX.search("testing_3_hexanol") is None
    assert PRETEST_REGEX.search("posttest_3_hexanol") is None


# ── the third arm ─────────────────────────────────────────────────────────


def _entry(label, category):
    return (label, _csv(label), category)


def test_trial_configs_add_a_pretest_arm_when_pretest_trials_exist():
    entries = [_entry("pretest_1_hexanol", "pretest"), _entry("testing_1_hexanol", "testing")]
    assert [name for name, _ in _build_trial_configs(entries, [])] == ["testing", "pretest"]


def test_trial_configs_stay_at_one_arm_without_pretest_or_training():
    entries = [_entry("testing_1_hexanol", "testing")]
    assert [name for name, _ in _build_trial_configs(entries, [])] == ["testing"]


def test_trial_configs_keep_training_for_the_existing_cohorts():
    entries = [_entry("testing_1_hexanol", "testing"), _entry("training_1_hexanol", "training")]
    assert [name for name, _ in _build_trial_configs(entries, [])] == ["testing", "training"]


def test_trial_configs_carry_all_three_arms_for_a_sensitivity_fly():
    """The sensitivity protocol runs pretest, training and testing on one fly."""
    entries = [
        _entry("pretest_1_hexanol", "pretest"),
        _entry("training_1_hexanol", "training"),
        _entry("testing_1_hexanol", "testing"),
    ]
    names = [name for name, _ in _build_trial_configs(entries, [])]
    assert sorted(names) == ["pretest", "testing", "training"]


def test_trial_configs_pair_each_arm_with_its_own_regex():
    entries = [
        _entry("pretest_1_hexanol", "pretest"),
        _entry("training_1_hexanol", "training"),
    ]
    by_name = dict(_build_trial_configs(entries, []))
    assert by_name["pretest"] is PRETEST_REGEX
    assert by_name["training"] is TRAINING_REGEX
    assert by_name["testing"] is TESTING_REGEX


def test_trial_configs_look_at_both_entry_groups():
    """Angle and distance entries are passed separately; either may carry the hint."""
    names = [
        name
        for name, _ in _build_trial_configs([], [_entry("pretest_1_hexanol", "pretest")])
    ]
    assert "pretest" in names


# ── exports ───────────────────────────────────────────────────────────────


def test_export_trial_type_inference_knows_pretest():
    assert envelope_exports._infer_trial_type(
        _csv("aug_28_fly1_pretest_3_Hexanol_fly1_distances_20260828_101010")
    ) == "pretest"


def test_baseline_pool_still_includes_pretest_trials():
    """Pre-reclassification, pretest fed the per-fly baseline pool as 'testing'.

    Dropping it would shift every sensitivity fly's normalisation, so the pool
    has to name both types explicitly.
    """
    assert set(envelope_exports.BASELINE_POOL_TRIAL_TYPES) == {"testing", "pretest"}


def test_baseline_pool_excludes_training():
    assert "training" not in envelope_exports.BASELINE_POOL_TRIAL_TYPES


# ── the figure renderer ───────────────────────────────────────────────────


def test_envelope_plots_accept_pretest_as_a_trial_type():
    """generate_envelope_plots hard-allowlists trial types; pretest must be in it."""
    from scripts.analysis import envelope_visuals

    assert set(envelope_visuals.SUPPORTED_TRIAL_TYPES) == {
        "testing", "training", "pretest",
    }


def test_envelope_plots_still_reject_an_unknown_trial_type():
    """The gate must stay a closed allowlist — a typo'd trial_type is a config bug.

    (Asserted on the set rather than by calling generate_envelope_plots, which
    loads the matrix before it reaches the check.)
    """
    from scripts.analysis import envelope_visuals

    assert "nonsense" not in envelope_visuals.SUPPORTED_TRIAL_TYPES
    assert "posttest" not in envelope_visuals.SUPPORTED_TRIAL_TYPES


def test_pretest_figures_are_titled_pretest_not_testing():
    """The naive panel and the post-training panel must not share a title.

    `phase_label = "Training" if trial_type == "training" else "Testing"`
    collapsed every non-training phase into "Testing", so
    Raw-Pre-Testing-PER-Traces figures were stamped "Proboscis Distance Across
    Testing Trials" — identical to the post-training panel in the sibling
    folder. That is exactly the pre/post confusion the phase split exists to
    prevent.
    """
    from scripts.analysis.envelope_visuals import _phase_label

    assert _phase_label("pretest") == "Pre-Test"
    assert _phase_label("training") == "Training"
    assert _phase_label("testing") == "Testing"


def test_phase_label_falls_back_to_testing_for_anything_unknown():
    from scripts.analysis.envelope_visuals import _phase_label

    assert _phase_label("nonsense") == "Testing"
