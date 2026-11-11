"""Pre-test baseline variant of the naive-vs-trial score bars.

``naive_vs_trial_score_bars.py`` puts a *naive* bar beside each conditioning and
testing trial, so "how far off naive is this trial?" is readable trial by trial.
Its naive baseline is a SEPARATE concentration-matched RandomPanel cohort —
different flies.

The ``*-Sensitivity-*`` cohorts carry their own naive panel: the same fly saw
the same odor before training. That is a strictly better baseline (within
subject, no concentration matching needed, no cross-cohort confound), so these
cohorts use ``pretest_baseline=True`` and their figures land in their own
folder rather than mixing with the naive-baselined ones.

One presentation per odor in the pre-test, so there is exactly ONE baseline
exposure — not the naive panel's two.
"""


import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

import scripts.analysis.naive_vs_trial_score_bars as nvt

DATASET = "Hex-Sensitivity-24-0.1"
ODOR = "Hexanol"


def _spec(**kw):
    base = dict(dataset=DATASET, odor=ODOR, concentration="0.1%",
                naive_dataset=None, pretest_baseline=True)
    base.update(kw)
    return nvt.CohortSpec(**base)


def _scores(phases=("testing", "pretest")):
    """A scored frame in _load_scores' shape, carrying both phases."""
    rows = []
    for phase in phases:
        # Testing presents the CS+ at the indices the spec names; the pre-test
        # presents it once. Both carry trial_num so testing_scores can key on it.
        indices = (1, 8) if phase == "testing" else (1,)
        for fn in (1, 2, 3):
            for idx in indices:
                rows.append({
                    "dataset": DATASET, "dataset_canon": DATASET,
                    "fly": "b1", "fly_number": fn, "trial_num": idx,
                    "odor_display": ODOR, "trial_type": phase,
                    "score": 4 if phase == "testing" else 1,
                    "occurrence": 1,
                })
    return pd.DataFrame(rows)


# ── the spec ──────────────────────────────────────────────────────────────


def test_a_cohort_can_declare_a_pretest_baseline():
    assert _spec().pretest_baseline is True


def test_the_default_cohort_still_uses_a_naive_dataset():
    """Every existing cohort must be untouched."""
    spec = nvt.CohortSpec(dataset="Hex-Control-24-0.1", odor=ODOR,
                          concentration="0.1%", naive_dataset="RandomPanel-24-0.1")
    assert spec.pretest_baseline is False


# ── the baseline comes from the fly's OWN pre-test ────────────────────────


def test_the_baseline_reads_the_pretest_rows_of_the_same_dataset(_=None):
    got = nvt.naive_scores(_scores(), _spec(), exposure=1)
    assert len(got) == 3
    assert set(got["score"]) == {1.0}


def test_the_baseline_never_picks_up_testing_rows():
    """"pretest" contains "test" — the two phases must not blur here either."""
    got = nvt.naive_scores(_scores(), _spec(), exposure=1)
    assert 4.0 not in set(got["score"])


def test_no_baseline_when_the_cohort_has_no_pretest_rows():
    got = nvt.naive_scores(_scores(phases=("testing",)), _spec(), exposure=1)
    assert got.empty


def test_the_trial_bars_still_read_only_testing_rows():
    """The pre-test rows now share the frame; they must not become trial bars."""
    bars = nvt.testing_scores(_scores(), _spec())
    for bar in bars:
        assert all(v == 4.0 for v in bar.values), bar.label


# ── the bar itself ────────────────────────────────────────────────────────


def test_the_baseline_bar_is_labelled_pre_test_not_naive():
    """These figures sit beside naive-baselined ones; the label is the only
    thing distinguishing which baseline a reader is looking at."""
    bar = nvt.naive_bar(_scores(), _spec(), 1)
    assert "pre-test" in bar.label.casefold()
    assert "naive" not in bar.label.casefold()


def test_the_naive_cohorts_bar_still_says_naive():
    spec = nvt.CohortSpec(dataset="Hex-Control-24-0.1", odor=ODOR,
                          concentration="0.1%", naive_dataset="RandomPanel-24-0.1")
    bar = nvt.naive_bar(_scores(), spec, 1)
    assert "naive" in bar.label.casefold()


# ── figure plans: one baseline, not two ───────────────────────────────────


def test_a_pretest_cohort_gets_two_figures_not_four():
    """The naive panel presents each odor twice; the pre-test presents it once,
    so a second exposure would be an empty bar."""
    plans = nvt.figure_plans(_spec())
    assert len(plans) == 2
    assert {p.phase for p in plans} == {"training", "testing"}


def test_every_pretest_plan_uses_the_single_baseline_exposure():
    for plan in nvt.figure_plans(_spec()):
        assert plan.naive_exposure == 1


def test_the_naive_cohorts_still_get_four_figures():
    spec = nvt.CohortSpec(dataset="Hex-Control-24-0.1", odor=ODOR,
                          concentration="0.1%", naive_dataset="RandomPanel-24-0.1")
    assert len(nvt.figure_plans(spec)) == 4


def test_a_cohort_with_no_baseline_at_all_still_gets_two():
    spec = nvt.CohortSpec(dataset="Hex-Control-24-0.01", odor=ODOR,
                          concentration="0.01%", naive_dataset=None)
    assert len(nvt.figure_plans(spec)) == 2


# ── the registry and the new folder ───────────────────────────────────────


def test_the_sensitivity_cohorts_are_registered():
    keys = set(nvt.PRETEST_COHORTS)
    assert {"Hex-Sensitivity-24-0.1", "IAA-Sensitivity-24-1",
            "3Oct-Sensitivity-24-0.1", "IAA-Sensitivity-40-1"} <= keys


@pytest.mark.parametrize("hours", [1, 6, 12, 18, 30])
def test_the_iaa_starvation_series_is_registered(hours):
    spec = nvt.PRETEST_COHORTS[f"IAA-Sensitivity-{hours}-1"]
    assert spec.odor == "Isoamyl Acetate"
    assert spec.concentration == "1%"
    assert spec.pretest_baseline is True


def test_the_1h_3oct_cohort_is_registered():
    """config_new lists it under pretest_score_bars.datasets; unregistered,
    its `--dataset` flag exits the step with code 2 for every cohort."""
    spec = nvt.PRETEST_COHORTS["3Oct-Sensitivity-1-0.1"]
    assert spec.odor == "3-Octanol"
    assert spec.concentration == "0.1%"
    assert spec.pretest_baseline is True


def test_the_5s_iaa_cohort_is_registered():
    """Listed in config_new from 2026-10-16 but not here, so argparse exited
    the step with code 2 every night and no cohort's bars were redrawn."""
    spec = nvt.PRETEST_COHORTS["IAA-Sensitivity5s-1-1"]
    assert (spec.odor, spec.concentration) == ("Isoamyl Acetate", "1%")
    assert spec.pretest_baseline is True and spec.naive_dataset is None
    # Six odor-only exposures, as in the 30 s protocol.
    assert spec.n_training == 6
    assert {p.stem for p in nvt.figure_plans(spec)} == {
        "IAA-Sensitivity5s-1-1_training_vs_pretest",
        "IAA-Sensitivity5s-1-1_testing_vs_pretest",
    }


def test_the_cli_accepts_the_5s_cohort(monkeypatch):
    seen = {}
    monkeypatch.setattr(nvt, "run", lambda datasets, **kw: seen.update(datasets=datasets))
    assert nvt.main(["--pretest-baseline", "--dataset", "IAA-Sensitivity5s-1-1"]) == 0
    assert seen["datasets"] == ["IAA-Sensitivity5s-1-1"]


def test_the_5s_cohort_never_pools_the_30s_cohorts_rows():
    """Same odor, dose and starvation as IAA-Sensitivity-1-1; only the dataset
    name keeps the two protocols apart."""
    rows = _pins_scores(["isoamylacetate-1"])
    other = rows.assign(dataset="IAA-Sensitivity-1-1", dataset_canon="IAA-Sensitivity-1-1")
    five = rows.assign(dataset="IAA-Sensitivity5s-1-1", dataset_canon="IAA-Sensitivity5s-1-1",
                       fly_number=2)
    scores = pd.concat([other, five], ignore_index=True)
    spec = nvt.PRETEST_COHORTS["IAA-Sensitivity5s-1-1"]
    assert nvt.naive_bar(scores, spec, 1).n == 1
    assert [b.n for b in nvt.testing_scores(scores, spec)] == [1]


# ── odor_pins labels ("isoamylacetate-1") ─────────────────────────────────


def _pins_scores(displays):
    """One fly per display spelling, each with a pre-test, a test and a
    conditioning row -- the shape every bar builder reads."""
    rows = []
    for fn, display in enumerate(displays, start=1):
        for phase, idx in (("pretest", 4), ("testing", 2)):
            rows.append({
                "dataset": "IAA-Sensitivity-12-1",
                "dataset_canon": "IAA-Sensitivity-12-1",
                "fly": "oct_9", "fly_number": fn, "trial_num": idx,
                "odor_display": display, "trial_type": phase,
                "score": 3, "occurrence": 1,
            })
    return pd.DataFrame(rows)


def _iaa_spec(**kw):
    base = dict(dataset="IAA-Sensitivity-12-1", odor="Isoamyl Acetate",
                concentration="1%", naive_dataset=None, pretest_baseline=True)
    base.update(kw)
    return nvt.CohortSpec(**base)


def test_odor_pins_labels_reach_every_bar():
    """From 2026-10-06 the label carries the vial: ``isoamylacetate-1`` never
    mapped to "Isoamyl Acetate", so IAA-1/12/18/30 drew no figure at all."""
    scores = _pins_scores(["Isoamyl Acetate", "isoamylacetate-1"])
    spec = _iaa_spec()
    assert nvt.naive_bar(scores, spec, 1).n == 2
    assert [b.n for b in nvt.testing_scores(scores, spec)] == [2]


def test_a_different_vial_of_the_same_odor_is_not_merged():
    """The hex-test panel carries two hexanol vials; stripping the suffix
    alone would pool them. The cohort's own concentration decides."""
    scores = _pins_scores(["isoamylacetate-1", "isoamylacetate-0.1"])
    assert [b.n for b in nvt.testing_scores(scores, _iaa_spec())] == [1]
    assert [b.n for b in nvt.testing_scores(
        scores, _iaa_spec(concentration="0.1%"))] == [1]


def test_every_sensitivity_cohort_uses_its_own_pretest():
    for spec in nvt.PRETEST_COHORTS.values():
        assert spec.pretest_baseline is True
        assert spec.naive_dataset is None


def test_each_sensitivity_cohort_names_its_cs_plus_odor():
    expected = {
        "Hex-Sensitivity-24-0.1": "Hexanol",
        "IAA-Sensitivity-24-1": "Isoamyl Acetate",
        "3Oct-Sensitivity-24-0.1": "3-Octanol",
        "IAA-Sensitivity-40-1": "Isoamyl Acetate",
    }
    for key, odor in expected.items():
        assert nvt.PRETEST_COHORTS[key].odor == odor


def test_the_new_figures_go_to_their_own_folder():
    """They must not mix with the naive-baselined set — a reader could not tell
    which baseline a given figure used."""
    assert str(nvt.PRETEST_OUT_DIR).endswith("Pre-Test-vs-Trial-Score-Bars")
    assert str(nvt.PRETEST_OUT_DIR) != str(nvt.OUT_DIR)


def test_the_original_cohorts_are_untouched():
    assert "Hex-Control-24-0.1" in nvt.COHORTS
    assert "Hex-Sensitivity-24-0.1" not in nvt.COHORTS


# ── file naming ───────────────────────────────────────────────────────────


def test_the_filename_says_pretest_not_naive():
    """A file called ..._vs_naive-p1 in the pre-test folder would misdescribe
    its own baseline — the filename is what ends up in a figure caption."""
    plan = nvt.figure_plans(_spec())[0]
    assert plan.stem.endswith("_vs_pretest")
    assert "naive" not in plan.stem


def test_there_is_no_exposure_suffix_on_a_pretest_stem():
    """One presentation, so "-p1" would imply a second that does not exist."""
    for plan in nvt.figure_plans(_spec()):
        assert "-p" not in plan.stem.rsplit("_vs_", 1)[-1]


def test_the_naive_stems_are_unchanged():
    spec = nvt.CohortSpec(dataset="Hex-Control-24-0.1", odor=ODOR,
                          concentration="0.1%", naive_dataset="RandomPanel-24-0.1")
    stems = {p.stem for p in nvt.figure_plans(spec)}
    assert any(s.endswith("_vs_naive-p1") for s in stems)
    assert any(s.endswith("_vs_naive-p2") for s in stems)


# ── odor order is randomised, so trial INDEX cannot key the bars ──────────


def _scores_randomised():
    """The CS+ lands at a different testing index for each fly, as it really does."""
    rows = []
    for fn, idx in ((1, 1), (2, 4), (3, 7)):
        rows.append({
            "dataset": DATASET, "dataset_canon": DATASET, "fly": "b1",
            "fly_number": fn, "trial_num": idx, "odor_display": ODOR,
            "trial_type": "testing", "score": 4, "occurrence": 1,
        })
        rows.append({
            "dataset": DATASET, "dataset_canon": DATASET, "fly": "b1",
            "fly_number": fn, "trial_num": 1, "odor_display": ODOR,
            "trial_type": "pretest", "score": 1, "occurrence": 1,
        })
    return pd.DataFrame(rows)


def test_every_fly_reaches_the_test_bar_despite_randomised_order():
    """Keying on trial index silently dropped flies whose CS+ was not at
    index 1 — the pre-test bar showed n=7 beside a test bar of n=2."""
    bars = nvt.testing_scores(_scores_randomised(), _spec())
    total = sum(bar.n for bar in bars)
    assert total == 3, [(b.label, b.n) for b in bars]


def test_a_pretest_cohort_gets_one_test_bar_not_two():
    """The panel presents each odor once, so a second bar is always n=0."""
    bars = nvt.testing_scores(_scores_randomised(), _spec())
    assert len(bars) == 1
    assert bars[0].n == 3


def test_the_naive_cohorts_keep_their_two_indexed_bars():
    spec = nvt.CohortSpec(dataset="Hex-Control-24-0.1", odor=ODOR,
                          concentration="0.1%", naive_dataset="RandomPanel-24-0.1")
    assert spec.testing_indices == (1, 8)


def test_the_title_names_the_pretest_baseline():
    """"vs naive exposure 1" on a pre-test figure misdescribes the baseline."""
    plan = [p for p in nvt.figure_plans(_spec()) if p.phase == "testing"][0]
    # A sensitivity cohort's pre-test panel is its pre-exposure panel.
    assert "pre-exposure" in plan.title.casefold()
    assert "naive" not in plan.title.casefold()


# ── sensitization wording ─────────────────────────────────────────────────
#
# Nothing is paired with the odor in a *-Sensitivity-* cohort, so its panels
# are pre-/post-exposure. A light-conditioned cohort that also runs a
# pre-test really is conditioning, so it keeps "Pre-test" / "Test".

CONDITIONED = "Hex-HexTest-Training-32-1"


def _conditioned_spec():
    return _spec(dataset=CONDITIONED)


def _conditioned_scores():
    frame = _scores()
    frame["dataset"] = frame["dataset_canon"] = CONDITIONED
    return frame


def _drawn(monkeypatch, tmp_path, spec, scores):
    """The testing figure's title and tick texts, read off the rendered axes."""
    figs = []
    close = nvt.plt.close

    def _keep(fig=None):
        figs.append(fig)
        close(fig)

    monkeypatch.setattr(nvt.plt, "close", _keep)
    plan = next(p for p in nvt.figure_plans(spec) if p.phase == "testing")
    nvt.render(plan, scores, pd.DataFrame(), tmp_path)
    ax = figs[0].axes[0]
    return ax.get_title(), [t.get_text().split("\n")[0] for t in ax.get_xticklabels()]


def test_a_sensitivity_cohorts_titles_say_pre_and_post_sensitization():
    """The two-bar testing figure wraps before "vs": on one line the longer
    words overhang its axes by up to an inch."""
    titles = {p.phase: p.title for p in nvt.figure_plans(_spec())}
    assert titles["testing"] == (
        "Hexanol 0.1% (v/v) — post-exposure trials\nvs pre-exposure"
    )
    assert titles["training"] == (
        "Hexanol 0.1% (v/v) — conditioning trials vs pre-exposure"
    )


def test_a_sensitivity_cohorts_bars_keep_their_stats_csv_labels():
    """``label`` keys the stats CSV; only the drawn tick is reworded."""
    scores = _scores()
    baseline = nvt.naive_bar(scores, _spec(), 1)
    tests = nvt.testing_scores(scores, _spec())
    assert (baseline.label, baseline.display) == ("Pre-test", "Pre-exposure")
    assert [(b.label, b.display) for b in tests] == [("Test", "Post-exposure")]


def test_the_rendered_sensitivity_figure_says_pre_and_post_sensitization(
    monkeypatch, tmp_path
):
    title, ticks = _drawn(monkeypatch, tmp_path, _spec(), _scores())
    assert title.startswith(
        "Hexanol 0.1% (v/v) — post-exposure trials\nvs pre-exposure"
    )
    assert ticks == ["Pre-exposure", "Post-exposure"]
    stats = pd.read_csv(next(tmp_path.glob("*_stats.csv")))
    assert list(stats["label"]) == ["Pre-test", "Test"]


def test_a_conditioned_cohort_with_a_pretest_keeps_its_wording():
    titles = {p.phase: p.title for p in nvt.figure_plans(_conditioned_spec())}
    assert titles["testing"] == "Hexanol (0.1%) — testing trials vs pre-test"
    assert titles["training"] == "Hexanol (0.1%) — conditioning trials vs pre-test"


def test_the_rendered_conditioned_figure_keeps_pre_test(monkeypatch, tmp_path):
    title, ticks = _drawn(
        monkeypatch, tmp_path, _conditioned_spec(), _conditioned_scores()
    )
    assert title.startswith("Hexanol (0.1%) — testing trials vs pre-test")
    assert ticks == ["Pre-test", "Test"]


# ── the dose as "% (v/v)" ─────────────────────────────────────────────────
#
# A sensitivity cohort's title writes its dose "1% (v/v)". Display only: the
# registry's odor and concentration still match rows and fill the stats CSV,
# and the conditioned cohort above keeps "Hexanol (0.1%)".


@pytest.mark.parametrize("key, shown", [
    ("Hex-Sensitivity-24-0.1", "Hexanol 0.1% (v/v)"),
    ("IAA-Sensitivity-1-1", "Isoamyl Acetate 1% (v/v)"),
    ("IAA-Sensitivity5s-1-1", "Isoamyl Acetate 1% (v/v)"),
    ("3Oct-Sensitivity-24-0.1", "3-Octanol 0.1% (v/v)"),
    ("EB-Sensitivity-24-1", "Ethyl Butyrate 1% (v/v)"),
])
def test_a_sensitivity_title_writes_its_dose_as_v_v(key, shown):
    for plan in nvt.figure_plans(nvt.PRETEST_COHORTS[key]):
        assert plan.title.startswith(f"{shown} — "), plan.title


def test_no_sensitivity_title_keeps_the_parenthesised_dose():
    for spec in nvt.PRETEST_COHORTS.values():
        for plan in nvt.figure_plans(spec):
            assert "% (v/v) — " in plan.title, plan.title
            assert f"({spec.concentration})" not in plan.title


def test_a_sensitivity_spec_without_a_dose_draws_the_bare_name():
    """No concentration is invented: an empty dose shows the odor alone."""
    plan = next(p for p in nvt.figure_plans(_spec(concentration="")) if p.phase == "testing")
    assert plan.title.startswith("Hexanol — post-exposure trials")
    assert "%" not in plan.title


def test_the_rendered_pins_cohort_title_writes_the_dose_as_v_v(monkeypatch, tmp_path):
    title, ticks = _drawn(monkeypatch, tmp_path, _iaa_spec(),
                          _pins_scores(["isoamylacetate-1", "Isoamyl Acetate"]))
    assert title.startswith(
        "Isoamyl Acetate 1% (v/v) — post-exposure trials\nvs pre-exposure"
    )
    assert ticks == ["Pre-exposure", "Post-exposure"]


def test_the_stats_csv_keeps_the_registry_odor_and_dose(monkeypatch, tmp_path):
    _drawn(monkeypatch, tmp_path, _spec(), _scores())
    stats = pd.read_csv(next(tmp_path.glob("*_stats.csv")))
    assert set(stats["odor"]) == {"Hexanol"}
    assert set(stats["concentration"]) == {"0.1%"}
