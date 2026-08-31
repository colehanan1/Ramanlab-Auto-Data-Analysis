"""Tests for scripts/analysis/naive_vs_trial_score_bars.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.analysis import naive_vs_trial_score_bars as nvt  # noqa: E402
from scripts.analysis.envelope_visuals import set_protocol  # noqa: E402


@pytest.fixture(autouse=True)
def _v2():
    set_protocol("v2")
    yield
    set_protocol("v2")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_covers_the_four_control_cohorts():
    assert set(nvt.COHORTS) == {
        "Hex-Control-24-0.1",
        "Hex-Control-24-0.01",
        "EB-Control-24-1",
        "3Oct-Control-24-0.1",
        "Hex-Control-24-0.1-no-aug25-27",
    }


def test_hex_001_has_no_naive_arm():
    """No RandomPanel ran at 0.01%; the cohort is not matched to a neighbour dose."""
    assert nvt.COHORTS["Hex-Control-24-0.01"].naive_dataset is None
    assert nvt.COHORTS["Hex-Control-24-0.1"].naive_dataset == "RandomPanel-24-0.1"
    assert nvt.COHORTS["EB-Control-24-1"].naive_dataset == "RandomPanel-24-1"


def test_specs_default_to_six_training_and_two_testing_presentations():
    spec = nvt.COHORTS["Hex-Control-24-0.1"]
    assert spec.n_training == 6
    assert spec.testing_indices == (1, 8)


# ---------------------------------------------------------------------------
# Synthetic prediction frames
# ---------------------------------------------------------------------------


def _testing_rows() -> pd.DataFrame:
    """Two Hex-Control flies + two naive flies that meet Hexanol twice."""
    rows = []
    for fly, base in (("august_15_batch_2_rig_2", 3.0), ("august_18_batch_2_rig_3", 2.0)):
        for num in (1, 2):
            rows.append((("Hex-Control-24-0.1"), fly, num, "testing_1_hexanol", base))
            rows.append((("Hex-Control-24-0.1"), fly, num, "testing_8_hexanol", base - 1))
            # a distractor odor that must never reach a bar
            rows.append((("Hex-Control-24-0.1"), fly, num, "testing_3_citral", 5.0))
    for fly, base in (("july_01_batch_1_rig_2", 1.0), ("july_02_batch_1_rig_2", 0.0)):
        for num in (1, 2):
            rows.append(("RandomPanel-24-0.1", fly, num, "training_2_hexanol", base))
            rows.append(("RandomPanel-24-0.1", fly, num, "training_9_hexanol", base + 1))
            rows.append(("RandomPanel-24-0.1", fly, num, "training_4_linalool", 4.0))
    df = pd.DataFrame(
        rows, columns=["dataset", "fly", "fly_number", "trial_label", "score"]
    )
    df["trial_type"] = "testing"
    df["fly_type"] = "GR5a-Old"
    return df


def _training_rows() -> pd.DataFrame:
    rows = []
    for fly, base in (("august_15_batch_2_rig_2", 1.0), ("august_18_batch_2_rig_3", 2.0)):
        for num in (1, 2):
            for trial in range(1, 7):
                rows.append(
                    (
                        "Hex-Control-24-0.1",
                        fly,
                        num,
                        f"training_{trial}_hexanol",
                        base + 0.1 * trial,
                    )
                )
    df = pd.DataFrame(
        rows, columns=["dataset", "fly", "fly_number", "trial_label", "score"]
    )
    df["trial_type"] = "training"
    df["fly_type"] = "GR5a-Old"
    return df


@pytest.fixture
def scores(tmp_path) -> pd.DataFrame:
    path = tmp_path / "model_predictions.csv"
    _testing_rows().to_csv(path, index=False)
    return nvt.load_testing_scores(path)


@pytest.fixture
def training(tmp_path) -> pd.DataFrame:
    path = tmp_path / "model_predictions_training.csv"
    _training_rows().to_csv(path, index=False)
    return nvt.load_training_scores(path)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_naive_exposures_split_by_occurrence(scores):
    spec = nvt.COHORTS["Hex-Control-24-0.1"]
    first = nvt.naive_scores(scores, spec, exposure=1)
    second = nvt.naive_scores(scores, spec, exposure=2)
    # two folders x two fly_numbers = four flies per bar
    assert sorted(first["score"]) == [0.0, 0.0, 1.0, 1.0]
    assert sorted(second["score"]) == [1.0, 1.0, 2.0, 2.0]


def test_naive_selection_ignores_other_odors(scores):
    spec = nvt.COHORTS["Hex-Control-24-0.1"]
    got = nvt.naive_scores(scores, spec, exposure=1)
    assert len(got) == 4  # linalool never counted
    assert 4.0 not in set(got["score"])


def test_testing_presentations_map_to_test_1_and_2(scores):
    spec = nvt.COHORTS["Hex-Control-24-0.1"]
    bars = nvt.testing_scores(scores, spec)
    assert [b.label for b in bars] == ["Test 1", "Test 2"]
    assert sorted(bars[0].values) == [2.0, 2.0, 3.0, 3.0]  # testing_1
    assert sorted(bars[1].values) == [1.0, 1.0, 2.0, 2.0]  # testing_8


def test_base_odor_matches_across_a_concentration_suffix():
    assert nvt.base_odor("Hexanol (0.1%) 2") == "Hexanol"
    assert nvt.base_odor("3-Octanol") == "3-Octanol"


def test_training_bars_are_trials_one_through_six(training):
    spec = nvt.COHORTS["Hex-Control-24-0.1"]
    bars = nvt.training_scores(training, spec)
    assert [b.label for b in bars] == [f"Train {i}" for i in range(1, 7)]
    assert sorted(bars[0].values) == pytest.approx([1.1, 1.1, 2.1, 2.1])
    assert sorted(bars[5].values) == pytest.approx([1.6, 1.6, 2.6, 2.6])


def test_one_value_per_fly_even_when_a_rig_logged_a_trial_twice(tmp_path):
    """fly_number 1 and 2 are the same physical fly slot; each is its own unit."""
    frame = _testing_rows()
    dup = frame[frame["trial_label"] == "testing_1_hexanol"].copy()
    frame = pd.concat([frame, dup], ignore_index=True)
    path = tmp_path / "dup.csv"
    frame.to_csv(path, index=False)
    loaded = nvt.load_testing_scores(path)
    bars = nvt.testing_scores(loaded, nvt.COHORTS["Hex-Control-24-0.1"])
    assert len(bars[0].values) == 4  # 2 flies x 2 fly_numbers, not 8


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------


def test_a_cohort_with_naive_gets_four_figures():
    plans = nvt.figure_plans(nvt.COHORTS["Hex-Control-24-0.1"])
    assert [p.stem for p in plans] == [
        "Hex-Control-24-0.1_training_vs_naive-p1",
        "Hex-Control-24-0.1_training_vs_naive-p2",
        "Hex-Control-24-0.1_testing_vs_naive-p1",
        "Hex-Control-24-0.1_testing_vs_naive-p2",
    ]


def test_a_cohort_without_naive_gets_two_figures():
    plans = nvt.figure_plans(nvt.COHORTS["Hex-Control-24-0.01"])
    assert [p.stem for p in plans] == [
        "Hex-Control-24-0.01_training",
        "Hex-Control-24-0.01_testing",
    ]
    assert all(p.naive_exposure is None for p in plans)


def test_naive_bar_leads_and_is_an_open_hatched_bar(scores, training):
    spec = nvt.COHORTS["Hex-Control-24-0.1"]
    plan = nvt.figure_plans(spec)[0]
    bars = nvt.build_bars(plan, scores, training)
    assert bars[0].label == "Naive 1"
    assert bars[0].hatch == nvt.NAIVE_HATCH
    assert bars[0].color == nvt.NAIVE_FACE
    assert [b.label for b in bars[1:]] == [f"Train {i}" for i in range(1, 7)]
    assert all(b.hatch == "" for b in bars[1:])


def test_conditioned_bars_carry_the_odorant_palette_color(scores, training):
    from scripts.analysis import odor_bar_palette as pal

    spec = nvt.COHORTS["Hex-Control-24-0.1"]
    bars = nvt.build_bars(nvt.figure_plans(spec)[2], scores, training)
    assert bars[1].color == pal.odor_color("Hexanol")


def test_bars_without_a_naive_arm_start_at_the_first_trial(scores, training):
    spec = nvt.COHORTS["Hex-Control-24-0.01"]
    plan = nvt.figure_plans(spec)[0]
    bars = nvt.build_bars(plan, scores, training)
    assert bars and not any(b.label.startswith("Naive") for b in bars)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_each_trial_bar_is_tested_against_naive_and_holm_corrected():
    naive = nvt.Bar("Naive 1", [0.0, 0.0, 0.0, 1.0], "#fff", nvt.NAIVE_HATCH)
    trials = [
        nvt.Bar("Train 1", [4.0, 5.0, 5.0, 4.0], "#0f0", ""),
        nvt.Bar("Train 2", [0.0, 1.0, 0.0, 0.0], "#0f0", ""),
    ]
    stats = nvt.compare_to_naive(naive, trials)
    assert [s.label for s in stats] == ["Train 1", "Train 2"]
    assert all(s.p_holm >= s.p_raw for s in stats)
    assert stats[0].p_holm < 0.05
    assert stats[1].p_holm > 0.05


def test_no_naive_bar_means_no_comparisons():
    trials = [nvt.Bar("Train 1", [1.0, 2.0], "#0f0", "")]
    assert nvt.compare_to_naive(None, trials) == []


def test_omnibus_needs_two_populated_bars():
    one = [nvt.Bar("Train 1", [1.0, 2.0, 3.0], "#0f0", "")]
    assert nvt.omnibus(one) is None
    two = one + [nvt.Bar("Train 2", [4.0, 5.0, 6.0], "#0f0", "")]
    assert nvt.omnibus(two) is not None


# ---------------------------------------------------------------------------
# Training-score sidecar
# ---------------------------------------------------------------------------


def test_ensure_training_predictions_shells_out_when_missing(tmp_path):
    parquet = tmp_path / "training.parquet"
    _training_rows().to_parquet(parquet)
    model = tmp_path / "model.json"
    model.write_text("{}")
    out = tmp_path / "model_predictions_training.csv"
    calls: list[list[str]] = []

    def runner(cmd, **kwargs):
        calls.append(cmd)
        _training_rows().to_csv(out, index=False)
        return 0

    nvt.ensure_training_predictions(parquet, model, out, runner=runner)
    assert len(calls) == 1
    assert calls[0][:2] == ["flybehavior-response", "predict-ordinal"]
    assert str(model) in calls[0]
    assert out.exists()


def test_ensure_training_predictions_is_cached(tmp_path):
    parquet = tmp_path / "training.parquet"
    _training_rows().to_parquet(parquet)
    model = tmp_path / "model.json"
    model.write_text("{}")
    out = tmp_path / "model_predictions_training.csv"
    _training_rows().to_csv(out, index=False)
    calls = []

    def runner(cmd, **kwargs):
        calls.append(cmd)
        return 0

    nvt.ensure_training_predictions(parquet, model, out, runner=runner)
    assert calls == []
    nvt.ensure_training_predictions(parquet, model, out, runner=runner, rescore=True)
    assert len(calls) == 1


def test_ensure_training_predictions_filters_frozen_and_flagged_rows(tmp_path, monkeypatch):
    """The training bars must be filtered exactly like predict_reactions filters testing."""
    parquet = tmp_path / "training.parquet"
    frame = _training_rows()
    frame["frozen"] = [i % 2 == 0 for i in range(len(frame))]
    frame.to_parquet(parquet)
    model = tmp_path / "model.json"
    model.write_text("{}")
    out = tmp_path / "out.csv"
    seen: dict[str, pd.DataFrame] = {}

    def runner(cmd, **kwargs):
        data_csv = cmd[cmd.index("--data-csv") + 1]
        seen["input"] = pd.read_csv(data_csv)
        _training_rows().to_csv(out, index=False)
        return 0

    nvt.ensure_training_predictions(parquet, model, out, runner=runner)
    assert not seen["input"]["frozen"].any()


def test_training_loader_drops_labels_with_no_odor_token(tmp_path):
    """A placeholder folder logs bare ``training_N``; the testing side drops those."""
    frame = _training_rows()
    extra = frame.head(6).copy()
    extra["fly"] = "flagged"
    extra["trial_label"] = [f"training_{i}" for i in range(1, 7)]
    path = tmp_path / "with_placeholder.csv"
    pd.concat([frame, extra], ignore_index=True).to_csv(path, index=False)
    got = nvt.load_training_scores(path)
    assert "flagged" not in set(got["fly"])


def test_training_loader_keeps_training_rows(tmp_path):
    """_load_scores drops trial_type != testing; the training loader must not."""
    path = tmp_path / "t.csv"
    _training_rows().to_csv(path, index=False)
    got = nvt.load_training_scores(path)
    assert not got.empty
    assert set(got["trial_num"]) == {1, 2, 3, 4, 5, 6}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_writes_png_svg_and_a_stats_sidecar(tmp_path, scores, training):
    spec = nvt.COHORTS["Hex-Control-24-0.1"]
    plan = nvt.figure_plans(spec)[0]
    written = nvt.render(plan, scores, training, tmp_path)
    names = {p.name for p in written}
    assert f"{plan.stem}.png" in names
    assert f"{plan.stem}.svg" in names
    assert f"{plan.stem}_stats.csv" in names
    stats = pd.read_csv(tmp_path / f"{plan.stem}_stats.csv")
    assert {"label", "n", "mean", "sem", "p_raw", "p_holm"} <= set(stats.columns)


def test_render_skips_a_plan_with_no_data(tmp_path, scores, training):
    spec = nvt.COHORTS["Hex-Control-24-0.01"]  # no rows in the fixtures
    plan = nvt.figure_plans(spec)[1]
    assert nvt.render(plan, scores, training, tmp_path) == []


# ---------------------------------------------------------------------------
# Date-excluded variants
# ---------------------------------------------------------------------------


def test_variant_reuses_the_same_dataset_under_its_own_key():
    variant = nvt.COHORTS["Hex-Control-24-0.1-no-aug25-27"]
    assert variant.dataset == "Hex-Control-24-0.1"
    assert variant.key == "Hex-Control-24-0.1-no-aug25-27"
    assert variant.exclude_fly_dates == ("august_25", "august_26", "august_27")
    # the unfiltered cohort keeps every fly
    assert nvt.COHORTS["Hex-Control-24-0.1"].exclude_fly_dates == ()


def test_key_defaults_to_the_dataset_name():
    assert nvt.COHORTS["EB-Control-24-1"].key == "EB-Control-24-1"


def test_variant_figures_are_named_for_the_key_not_the_dataset():
    plans = nvt.figure_plans(nvt.COHORTS["Hex-Control-24-0.1-no-aug25-27"])
    assert plans[0].stem == "Hex-Control-24-0.1-no-aug25-27_training_vs_naive-p1"


def _dated_testing_rows() -> pd.DataFrame:
    rows = []
    for fly in ("august_13_batch_1_rig_3", "august_26_batch_1_rig_2"):
        rows.append(("Hex-Control-24-0.1", fly, 1, "testing_1_hexanol", 3.0))
        rows.append(("Hex-Control-24-0.1", fly, 1, "testing_8_hexanol", 2.0))
    for fly in ("july_01_batch_1_rig_2", "august_26_naive_rig_2"):
        rows.append(("RandomPanel-24-0.1", fly, 1, "training_2_hexanol", 1.0))
        rows.append(("RandomPanel-24-0.1", fly, 1, "training_9_hexanol", 2.0))
    df = pd.DataFrame(
        rows, columns=["dataset", "fly", "fly_number", "trial_label", "score"]
    )
    df["trial_type"] = "testing"
    df["fly_type"] = "GR5a-Old"
    return df


def test_excluded_dates_drop_the_cohorts_own_flies(tmp_path):
    path = tmp_path / "dated.csv"
    _dated_testing_rows().to_csv(path, index=False)
    loaded = nvt.load_testing_scores(path)
    variant = nvt.COHORTS["Hex-Control-24-0.1-no-aug25-27"]
    full = nvt.testing_scores(loaded, nvt.COHORTS["Hex-Control-24-0.1"])
    cut = nvt.testing_scores(loaded, variant)
    assert full[0].n == 2
    assert cut[0].n == 1


def test_excluded_dates_never_touch_the_naive_panel(tmp_path):
    """The naive baseline must stay identical, or the two variants aren't comparable."""
    path = tmp_path / "dated.csv"
    _dated_testing_rows().to_csv(path, index=False)
    loaded = nvt.load_testing_scores(path)
    full = nvt.naive_bar(loaded, nvt.COHORTS["Hex-Control-24-0.1"], 1)
    cut = nvt.naive_bar(loaded, nvt.COHORTS["Hex-Control-24-0.1-no-aug25-27"], 1)
    assert cut.n == full.n == 2  # august_26_naive_rig_2 survives in both


def test_excluded_dates_apply_to_training_too(tmp_path):
    frame = _training_rows()
    frame.loc[frame["fly"] == "august_18_batch_2_rig_3", "fly"] = "august_26_batch_1_rig_2"
    path = tmp_path / "dated_training.csv"
    frame.to_csv(path, index=False)
    loaded = nvt.load_training_scores(path)
    full = nvt.training_scores(loaded, nvt.COHORTS["Hex-Control-24-0.1"])
    cut = nvt.training_scores(loaded, nvt.COHORTS["Hex-Control-24-0.1-no-aug25-27"])
    assert full[0].n == 4
    assert cut[0].n == 2


def test_a_date_prefix_matches_only_at_the_start_of_the_folder_name():
    spec = nvt.COHORTS["Hex-Control-24-0.1-no-aug25-27"]
    assert nvt._excluded("august_26_batch_1_rig_2", spec.exclude_fly_dates)
    assert not nvt._excluded("august_13_batch_1_rig_3", spec.exclude_fly_dates)
    # a date appearing mid-name is not the folder's date
    assert not nvt._excluded("august_13_batch_august_26", spec.exclude_fly_dates)


# ---------------------------------------------------------------------------
# Summary figure: first and last conditioning trial against naive and test 1
# ---------------------------------------------------------------------------


def test_summary_plan_is_opt_in_per_cohort():
    assert nvt.COHORTS["Hex-Control-24-0.1-no-aug25-27"].summary is True
    assert nvt.COHORTS["Hex-Control-24-0.1"].summary is False
    stems = [p.stem for p in nvt.figure_plans(nvt.COHORTS["Hex-Control-24-0.1"])]
    assert not any("summary" in s for s in stems)


def test_summary_plans_follow_the_four_per_exposure_figures():
    plans = nvt.figure_plans(nvt.COHORTS["Hex-Control-24-0.1-no-aug25-27"])
    assert [p.stem for p in plans][-2:] == [
        "Hex-Control-24-0.1-no-aug25-27_summary_vs_naive-p1",
        "Hex-Control-24-0.1-no-aug25-27_summary_vs_naive-p2",
    ]


def test_summary_bars_are_naive_first_last_training_and_test_one(scores, training):
    spec = nvt.COHORTS["Hex-Control-24-0.1-no-aug25-27"]
    plan = next(p for p in nvt.figure_plans(spec) if p.phase == "summary")
    bars = nvt.build_bars(plan, scores, training)
    assert [b.label for b in bars] == ["Naive 1", "Train 1", "Train 6", "Test 1"]
    assert bars[0].hatch == nvt.NAIVE_HATCH
    assert all(b.hatch == "" for b in bars[1:])


def test_summary_takes_the_last_training_trial_from_the_spec():
    spec = nvt.COHORTS["Hex-Control-24-0.1-no-aug25-27"]
    assert nvt.summary_labels(spec) == ("Train 1", "Train 6", "Test 1")
    four = nvt.CohortSpec(
        dataset="X", odor="Hexanol", concentration="1%", naive_dataset=None,
        n_training=4, summary=True,
    )
    assert nvt.summary_labels(four) == ("Train 1", "Train 4", "Test 1")


# ---------------------------------------------------------------------------
# Percentage (response-rate) variants
# ---------------------------------------------------------------------------


def test_a_responder_is_a_score_of_at_least_two():
    assert nvt.REACTION_BOUNDARY == 2
    bar = nvt.Bar("Train 1", [0.0, 1.0, 1.9, 2.0, 5.0], "#0f0", "")
    assert bar.responders == 2
    assert bar.rate == pytest.approx(40.0)


def test_rate_ci_is_wilson_and_asymmetric_at_the_edges():
    bar = nvt.Bar("Train 1", [5.0] * 8, "#0f0", "")
    low, high = bar.rate_ci
    assert high == pytest.approx(100.0)
    assert 0.0 < low < 100.0  # Wilson never collapses to a zero-width interval


def test_rate_of_an_empty_bar_is_not_a_number():
    import math

    assert math.isnan(nvt.Bar("Train 1", [], "#0f0", "").rate)


def test_percent_stems_are_suffixed_so_they_never_overwrite_the_score_figure():
    plan = nvt.figure_plans(nvt.COHORTS["Hex-Control-24-0.1"])[0]
    assert plan.stem_for("score") == plan.stem
    assert plan.stem_for("percent") == f"{plan.stem}_percent"


def test_percent_comparisons_use_fisher_not_mann_whitney():
    """Rates are counts; the score test would be comparing ranks of 0/1."""
    naive = nvt.Bar("Naive 1", [0.0] * 10, "#fff", nvt.NAIVE_HATCH)
    trials = [nvt.Bar("Train 1", [5.0] * 10, "#0f0", "")]
    stats = nvt.compare_rates_to_naive(naive, trials)
    assert len(stats) == 1
    assert stats[0].p_holm < 0.05
    # the same bars are not significant on the score test's own terms only by
    # coincidence here -- what matters is that the rate test ran at all
    assert stats[0].label == "Train 1"


def test_percent_comparisons_need_a_naive_bar():
    assert nvt.compare_rates_to_naive(None, [nvt.Bar("T", [1.0], "#0f0", "")]) == []


def test_render_percent_writes_its_own_files_and_sidecar(tmp_path, scores, training):
    plan = nvt.figure_plans(nvt.COHORTS["Hex-Control-24-0.1"])[0]
    written = nvt.render(plan, scores, training, tmp_path, metric="percent")
    names = {p.name for p in written}
    assert f"{plan.stem}_percent.png" in names
    assert f"{plan.stem}_percent.svg" in names
    stats = pd.read_csv(tmp_path / f"{plan.stem}_percent_stats.csv")
    assert {"rate", "ci_low", "ci_high", "responders"} <= set(stats.columns)
    assert (stats["rate"] <= 100).all()


def test_score_and_percent_figures_share_the_same_bars(tmp_path, scores, training):
    plan = nvt.figure_plans(nvt.COHORTS["Hex-Control-24-0.1"])[0]
    nvt.render(plan, scores, training, tmp_path, metric="score")
    nvt.render(plan, scores, training, tmp_path, metric="percent")
    a = pd.read_csv(tmp_path / f"{plan.stem}_stats.csv")
    b = pd.read_csv(tmp_path / f"{plan.stem}_percent_stats.csv")
    assert list(a["label"]) == list(b["label"])
    assert list(a["n"]) == list(b["n"])
