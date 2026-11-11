"""Tests for :mod:`scripts.analysis.training_auc_vs_control_response`.

The question the module answers -- "do control flies that extend more during
odor-only conditioning respond more at test?" -- is one join and three
statistics. Each is checked against a hand-computable case or an independent
implementation (statsmodels, scipy), never against another function here.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from scripts.analysis import training_auc_vs_control_response as mod  # noqa: E402

statsmodels = pytest.importorskip("statsmodels")


# ---------------------------------------------------------------------------
# Fixtures: a miniature cohort with a known answer
# ---------------------------------------------------------------------------

DATASET = "Hex-Control-24-0.1"

#: fly -> (AUC per training trial, score on presentation 1, score on 8)
_FLIES = {
    ("day_1", 1): ([0.0, 0.0, 0.0, 0.0], 0, 0),
    ("day_1", 2): ([10.0, 20.0, 30.0, 40.0], 0, 1),
    ("day_2", 1): ([100.0, 100.0, 100.0, 100.0], 4, 0),
    ("day_2", 2): ([200.0, 180.0, 160.0, 140.0], 5, 4),
}


def _training_frame() -> pd.DataFrame:
    rows = []
    for (fly, num), (aucs, _s1, _s8) in _FLIES.items():
        for trial, auc in enumerate(aucs, start=1):
            rows.append({
                "dataset": DATASET,
                "fly": fly,
                "fly_number": num,
                "trial_type": "training",
                "trial_label": f"training_{trial}_hexanol",
                "AUC-During": auc,
            })
    # A second dataset that must never leak into the Hex-Control table.
    rows.append({
        "dataset": "Hex-Training-24-0.1", "fly": "day_9", "fly_number": 1,
        "trial_type": "training", "trial_label": "training_1_hexanol",
        "AUC-During": 999.0,
    })
    return pd.DataFrame(rows)


def _testing_frame() -> pd.DataFrame:
    rows = []
    for (fly, num), (_aucs, s1, s8) in _FLIES.items():
        for trial, score in ((1, s1), (8, s8)):
            rows.append({
                "dataset": DATASET, "fly": fly, "fly_number": num,
                "trial_type": "testing",
                "trial_label": f"testing_{trial}_hexanol",
                "score": score, "prediction": int(score >= 2),
            })
        # A distractor odor: must not be mistaken for a hexanol presentation.
        rows.append({
            "dataset": DATASET, "fly": fly, "fly_number": num,
            "trial_type": "testing", "trial_label": "testing_4_citral",
            "score": 5, "prediction": 1,
        })
    return pd.DataFrame(rows)


def _testing_auc_frame() -> pd.DataFrame:
    """Testing-side envelope rows: the AUC the fly actually produced at test."""
    rows = []
    for (fly, num), (aucs, s1, s8) in _FLIES.items():
        for trial, score in ((1, s1), (8, s8)):
            rows.append({
                "dataset": DATASET, "fly": fly, "fly_number": num,
                "trial_type": "testing",
                "trial_label": f"testing_{trial}_hexanol",
                "AUC-During": float(score) * 40.0 + aucs[0],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trial-label parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,expected",
    [
        ("training_3_hexanol", (3, "hexanol")),
        ("testing_10_3-octonol", (10, "3-octonol")),
        ("testing_9_lightonly", (9, "lightonly")),
        ("TESTING_2_Hexanol", (2, "hexanol")),
        ("  training_1_hexanol  ", (1, "hexanol")),
        ("nonsense", None),
        ("training_x_hexanol", None),
        # ~/odor_pins.txt appends the vial concentration to every label from
        # 2026-10-06 on. The odor is the same odor; only the spelling moved.
        ("testing_5_isoamylacetate-1", (5, "isoamylacetate")),
        ("pretest_2_EthylButyrate-0.1", (2, "ethylbutyrate")),
        ("testing_1_3-octonol-0.1", (1, "3-octonol")),
        ("training_3_Hexanol-10", (3, "hexanol")),
    ],
)
def test_parse_trial(label, expected):
    assert mod.parse_trial(label) == expected


def test_training_odor_token_pools_concentration_suffixed_labels():
    """IAA-Sensitivity-24-1 carries both spellings; the modal pick must not
    split the cohort into an "isoamylacetate" and an "isoamylacetate-1" half."""
    train = pd.DataFrame({
        "trial_label": ["training_1_isoamylacetate"] * 2
        + ["training_1_isoamylacetate-1"] * 3,
    })
    assert mod.training_odor_token(train) == "isoamylacetate"


def test_training_odor_token_is_the_modal_token():
    """One stray mislabelled trial must not rename the conditioning odor."""
    df = _training_frame()
    df.loc[len(df)] = {
        "dataset": DATASET, "fly": "day_1", "fly_number": 1,
        "trial_type": "training", "trial_label": "training_5_citral",
        "AUC-During": 1.0,
    }
    assert mod.training_odor_token(df[df["dataset"] == DATASET]) == "hexanol"


def test_presentation_trials_finds_only_the_named_odor():
    test = _testing_frame()
    assert mod.presentation_trials(test, "hexanol") == [1, 8]
    assert mod.presentation_trials(test, "citral") == [4]
    assert mod.presentation_trials(test, "linalool") == []


# ---------------------------------------------------------------------------
# The join
# ---------------------------------------------------------------------------


def test_build_fly_table_shape_and_values():
    table = mod.build_fly_table(
        _training_frame(), _testing_frame(), dataset=DATASET
    )
    assert list(table.attrs["presentations"]) == [1, 8]
    assert table.attrs["odor_token"] == "hexanol"
    assert len(table) == 4

    row = table.set_index(["fly", "fly_number"]).loc[("day_2", 2)]
    assert row["mean_auc"] == pytest.approx(170.0)          # (200+180+160+140)/4
    assert row["auc_slope"] == pytest.approx(-20.0)          # -20 per trial
    assert row["score_1"] == 5
    assert row["score_8"] == 4
    assert row["per_1"] == 1
    assert row["mean_score"] == pytest.approx(4.5)

    zero = table.set_index(["fly", "fly_number"]).loc[("day_1", 1)]
    assert zero["mean_auc"] == pytest.approx(0.0)
    assert zero["auc_slope"] == pytest.approx(0.0)
    assert zero["per_any"] == 0


def test_build_fly_table_per_any_is_a_logical_or_over_presentations():
    """A fly that reacts on either hexanol presentation counts as a responder."""
    test = _testing_frame()
    # day_1/2 scored 0 then 1 -> no PER on either presentation.
    test.loc[
        (test["fly"] == "day_1") & (test["fly_number"] == 2)
        & (test["trial_label"] == "testing_8_hexanol"),
        ["score", "prediction"],
    ] = [3, 1]
    table = mod.build_fly_table(_training_frame(), test, dataset=DATASET)
    row = table.set_index(["fly", "fly_number"]).loc[("day_1", 2)]
    assert row["per_1"] == 0
    assert row["per_8"] == 1
    assert row["per_any"] == 1


def test_build_fly_table_drops_flies_missing_either_side():
    """A fly with conditioning but no test trials is dropped, not carried NaN."""
    train = _training_frame()
    train.loc[len(train)] = {
        "dataset": DATASET, "fly": "day_3", "fly_number": 1,
        "trial_type": "training", "trial_label": "training_1_hexanol",
        "AUC-During": 50.0,
    }
    table = mod.build_fly_table(train, _testing_frame(), dataset=DATASET)
    assert ("day_3", 1) not in set(zip(table["fly"], table["fly_number"]))
    assert len(table) == 4


def test_build_fly_table_ignores_other_datasets():
    table = mod.build_fly_table(
        _training_frame(), _testing_frame(), dataset=DATASET
    )
    assert "day_9" not in set(table["fly"])


def test_build_fly_table_honours_an_explicit_odor_token():
    """`--odor-token citral` scores the novel odor, not the conditioning one."""
    table = mod.build_fly_table(
        _training_frame(), _testing_frame(), dataset=DATASET, odor_token="citral"
    )
    assert table.attrs["odor_token"] == "citral"
    assert list(table.attrs["presentations"]) == [4]
    assert (table["score_4"] == 5).all()


def test_build_fly_table_empty_for_an_unknown_dataset():
    table = mod.build_fly_table(
        _training_frame(), _testing_frame(), dataset="Nope-24-1"
    )
    assert table.empty


def test_build_fly_table_fixed_protocol_is_keyed_by_trial_number():
    """Every fly meets the odor on the same trials: keys stay trial numbers,
    so the control cohorts' columns and file names do not move."""
    table = mod.build_fly_table(
        _training_frame(), _testing_frame(), dataset=DATASET
    )
    assert table.attrs["presentation_mode"] == "fixed"
    assert table.attrs["presentation_slots"] == {1: [1], 8: [8]}


def test_build_fly_table_fixed_protocol_drops_a_fly_missing_a_presentation():
    """A fly that never reached trial 8 is dropped, exactly as before, rather
    than tipping the whole cohort into per-fly keying."""
    test = _testing_frame()
    test = test[~((test["fly"] == "day_1") & (test["fly_number"] == 1)
                  & (test["trial_label"] == "testing_8_hexanol"))]
    table = mod.build_fly_table(_training_frame(), test, dataset=DATASET)
    assert table.attrs["presentation_mode"] == "fixed"
    assert list(table.attrs["presentations"]) == [1, 8]
    assert ("day_1", 1) not in set(zip(table["fly"], table["fly_number"]))
    assert len(table) == 3


def test_build_fly_table_pools_concentration_suffixed_test_labels():
    """Flies recorded after the odor-pins switch spell it ``hexanol-0.1``."""
    test = _testing_frame()
    late = test["fly"] == "day_2"
    test.loc[late, "trial_label"] = test.loc[late, "trial_label"] + "-0.1"
    table = mod.build_fly_table(_training_frame(), test, dataset=DATASET)
    assert list(table.attrs["presentations"]) == [1, 8]
    assert len(table) == 4


# A *-Sensitivity-* post-test panel: the same 3 odors for every fly, in a
# different random order per fly, so the conditioning odor sits on a different
# trial for each. fly -> {trial: (odor label, score)}
_RANDOM_PANEL = {
    ("oct_1", 1): {1: ("isoamylacetate-1", 5), 2: ("citral-1", 0), 3: ("hexanol-0.1", 0)},
    ("oct_1", 2): {1: ("citral-1", 4), 2: ("isoamylacetate-1", 0), 3: ("hexanol-0.1", 4)},
    ("oct_2", 1): {1: ("hexanol-0.1", 5), 2: ("citral-1", 5), 3: ("isoamylacetate-1", 1)},
    ("oct_2", 2): {1: ("isoamylacetate-1", 3), 2: ("hexanol-0.1", 0), 3: ("citral-1", 0)},
}
_RANDOM_DATASET = "IAA-Sensitivity-12-1"


def _random_panel_frames():
    train, test, test_auc = [], [], []
    for (fly, num), trials in _RANDOM_PANEL.items():
        for t in (1, 2):
            train.append({
                "dataset": _RANDOM_DATASET, "fly": fly, "fly_number": num,
                "trial_type": "training",
                "trial_label": f"training_{t}_isoamylacetate-1",
                "AUC-During": float(10 * t + num),
            })
        for trial, (odor, score) in trials.items():
            label = f"testing_{trial}_{odor}"
            test.append({
                "dataset": _RANDOM_DATASET, "fly": fly, "fly_number": num,
                "trial_type": "testing", "trial_label": label,
                "score": score, "prediction": int(score >= 2),
            })
            test_auc.append({
                "dataset": _RANDOM_DATASET, "fly": fly, "fly_number": num,
                "trial_type": "testing", "trial_label": label,
                "AUC-During": float(score) * 100.0,
            })
    return pd.DataFrame(train), pd.DataFrame(test), pd.DataFrame(test_auc)


def test_build_fly_table_randomized_panel_scores_each_flys_own_presentation():
    """Keyed by trial number, "IAA 1..3" would be every fly's response to
    whatever odor happened to sit on trials 1..3 -- mostly not IAA."""
    train, test, test_auc = _random_panel_frames()
    table = mod.build_fly_table(
        train, test, dataset=_RANDOM_DATASET, testing_auc_df=test_auc
    )
    assert table.attrs["odor_token"] == "isoamylacetate"
    assert table.attrs["presentation_mode"] == "per_fly"
    assert list(table.attrs["presentations"]) == [1]
    assert table.attrs["presentation_slots"] == {1: [1, 2, 3]}
    assert len(table) == 4

    by_fly = table.set_index(["fly", "fly_number"])
    expected = {fly: next(s for o, s in trials.values() if o.startswith("isoamyl"))
                for fly, trials in _RANDOM_PANEL.items()}
    for fly, score in expected.items():
        assert by_fly.loc[fly, "score_1"] == score
        assert by_fly.loc[fly, "per_1"] == int(score >= 2)
        assert by_fly.loc[fly, "test_auc_1"] == pytest.approx(score * 100.0)
    assert by_fly["mean_score"].tolist() == by_fly["score_1"].tolist()


def test_trial_profile_renders_for_a_randomized_panel():
    """The figure that raised KeyError: 'IAA 5' on IAA-Sensitivity-24-1."""
    train, test, test_auc = _random_panel_frames()
    table = mod.build_fly_table(
        train, test, dataset=_RANDOM_DATASET, testing_auc_df=test_auc
    )
    fig, payload = mod.figure_trial_profile(
        table, dataset=_RANDOM_DATASET, odor_label="Isoamyl Acetate"
    )
    assert payload
    import matplotlib.pyplot as plt
    plt.close(fig)


def _drawn_text(dataset: str, suffix: str = "") -> str:
    """Every string drawn on every figure the CLI writes, for the fixture
    cohort filed under ``dataset``; ``suffix`` ("-0.1") spells its hexanol
    labels the way the odor pins do."""
    import matplotlib.pyplot as plt
    from matplotlib.text import Text

    def renamed(df):
        df = df.assign(dataset=df["dataset"].replace(DATASET, dataset))
        return df.assign(trial_label=df["trial_label"].str.replace(
            r"_hexanol$", f"_hexanol{suffix}", regex=True
        ))

    table = mod.build_fly_table(
        renamed(_training_frame()), renamed(_testing_frame()), dataset=dataset,
        testing_auc_df=renamed(_testing_auc_frame()),
    )
    kw = {"dataset": dataset, "odor_label": "Hexanol"}
    figs = [
        mod.figure_simple_overview(table, n_bins=2, **kw)[0],
        mod.figure_trial_predictors(table, presentation_index=0, **kw)[0],
        mod.figure_per(table, **kw)[0],
        mod.figure_score(table, **kw)[0],
        mod.figure_trial_matrix(table, **kw)[0],
        mod.figure_trial_profile(table, **kw)[0],
    ]
    try:
        return "\n".join(t.get_text() for fig in figs for t in fig.findobj(Text))
    finally:
        for fig in figs:
            plt.close(fig)


def test_a_sensitivity_cohorts_test_panel_is_post_sensitization():
    """Nothing is paired with a *-Sensitivity-* cohort's odor, so its second
    panel is post-exposure -- no figure may call it the test."""
    text = _drawn_text("Hex-Sensitivity-24-0.1")
    for drawn in (
        # Two lines: the longer phase name outruns the overview's short axes.
        "Flies that reacted\nat post-exposure (%)",
        "Reaction score\nat post-exposure (−1 … 5)",
        "Do flies that extend more during training react more at post-exposure?",
        "Predicts the size of the post-exposure response",
        # The odor in full (odor_conc_labels); the fixture records no
        # concentration, so none is drawn.
        "Which training trial predicts the Hexanol #1 post-exposure response?",
        "P(PER at post-exposure)",
        "no PER at post-exposure",
        "Post-exposure score (−1 … 5)",
        "Which conditioning trial predicts the post-exposure response?",
        "Post-exposure PER (0/1)",
        "each post-exposure outcome",
    ):
        assert drawn in text
    assert re.search(r"\b(?:test|testing)\b", text, re.IGNORECASE) is None


@pytest.mark.parametrize("dataset", ["Hex-HexTest-Training-32-1", DATASET])
def test_other_cohorts_keep_the_test_wording(dataset):
    """Hex-HexTest-Training-* also runs a pre-test, but it is light-conditioned,
    so its second panel really is the test. Controls keep it too."""
    text = _drawn_text(dataset)
    for drawn in (
        "Flies that reacted at test (%)",
        "Reaction score at test (−1 … 5)",
        "Do flies that extend more during training react more at test?",
        "Predicts the size of the test response",
        "Which training trial predicts the Hex 1 test?",
        "P(PER at test)",
        "no PER at test",
        "Testing score (−1 … 5)",
        "Which conditioning trial predicts the test response?",
        "Testing PER (0/1)",
        "each testing outcome",
    ):
        assert drawn in text
    assert "sensitiz" not in text.lower() and "exposure" not in text.lower()


# ---------------------------------------------------------------------------
# Odor names: a *-Sensitivity-* cohort draws "<Odor> <c>% (v/v)"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("labels,token,expected", [
    (["training_1_isoamylacetate-1", "testing_3_isoamylacetate-1"], "isoamylacetate", "1"),
    # Other odors' concentrations are not this odor's.
    (["training_1_3-octonol-0.1", "testing_2_hexanol-10"], "3-octonol", "0.1"),
    # IAA-Sensitivity-24-1 mixes pre-pins and pins batches: part of it
    # recorded no concentration, so none is claimed for the whole.
    (["training_1_isoamylacetate", "training_2_isoamylacetate-1"], "isoamylacetate", None),
    (["training_1_isoamylacetate-1", "training_2_isoamylacetate-10"], "isoamylacetate", None),
    (["training_1_isoamylacetate"], "isoamylacetate", None),
    ([], "isoamylacetate", None),
])
def test_recorded_conc(labels, token, expected):
    assert mod.recorded_conc(labels, token) == expected


def test_build_fly_table_carries_the_recorded_concentration():
    train, test, test_auc = _random_panel_frames()
    table = mod.build_fly_table(train, test, dataset=_RANDOM_DATASET)
    assert table.attrs["odor_conc"] == "1"
    plain = mod.build_fly_table(_training_frame(), _testing_frame(), dataset=DATASET)
    assert plain.attrs["odor_conc"] is None


def _random_panel_figures():
    """Every figure for the randomized IAA-Sensitivity-12-1 panel, which
    records "isoamylacetate-1": (all drawn text, the payloads, the figures)."""
    from matplotlib.text import Text

    train, test, test_auc = _random_panel_frames()
    table = mod.build_fly_table(
        train, test, dataset=_RANDOM_DATASET, testing_auc_df=test_auc
    )
    kw = {"dataset": _RANDOM_DATASET, "odor_label": "Isoamyl Acetate"}
    made = [
        mod.figure_simple_overview(table, n_bins=2, **kw),
        mod.figure_trial_predictors(table, presentation_index=0, **kw),
        mod.figure_per(table, **kw),
        mod.figure_score(table, **kw),
        mod.figure_trial_matrix(table, **kw),
        mod.figure_trial_profile(table, **kw),
    ]
    figs = [fig for fig, _ in made]
    text = "\n".join(t.get_text() for fig in figs for t in fig.findobj(Text))
    return text, [payload for _, payload in made], figs


def test_a_sensitivity_cohort_names_its_odor_with_the_concentration():
    """Where every figure said "IAA 1" -- which read as IAA at 1 % -- it now
    names the odor in full: "Isoamyl Acetate 1% (v/v)"."""
    import matplotlib.pyplot as plt

    text, payloads, figs = _random_panel_figures()
    try:
        for drawn in (
            "Isoamyl Acetate 1% (v/v) throughout.",
            "Which training trial predicts the Isoamyl Acetate 1% (v/v) "
            "post-exposure response?",
            "against the Isoamyl Acetate 1% (v/v) post-exposure outcome",
            "Control cohort: Isoamyl Acetate 1% (v/v) presented during conditioning",
            "same flies as Isoamyl Acetate 1% (v/v) — every other responder",
            "Isoamyl Acetate 1% (v/v) mean",
        ):
            assert drawn in text
        # Panel titles, the matrix header and the legend entry.
        assert text.split("\n").count("Isoamyl Acetate 1% (v/v)") >= 5
        assert re.search(r"\bIAA \d", text) is None
        # The legend names the lone series; a direct label beside the mean
        # diamond would run into the next panel.
        assert all(not ax.texts for ax in figs[-1].axes)
    finally:
        for fig in figs:
            plt.close(fig)
    # Display only: the payloads keep the labels they are keyed by.
    _overview, predictors, per, score, matrix, profile = payloads
    assert predictors["presentation"] == "IAA 1"
    assert predictors["odor"] == "Isoamyl Acetate"
    assert [s["column"] for s in per] == ["IAA 1", "IAA 1"]
    assert [s["column"] for s in score] == ["IAA 1", "IAA 1 mean"]
    assert {c["presentation"] for c in matrix["cells"]} == {"IAA 1"}
    assert set(profile[0]["series"]) == {"IAA 1"}


def test_a_sensitivity_cohort_without_a_recorded_concentration_claims_none():
    """Pre-pins batches recorded no concentration: the odor is named in full,
    with no percent, and repeat presentations are numbered."""
    text = _drawn_text("Hex-Sensitivity-24-0.1")
    for drawn in ("Hexanol #1", "Hexanol #2", "Hexanol #1 or #2",
                  "Hexanol #1 & #2 mean", "Hexanol throughout."):
        assert drawn in text
    assert "%" not in text.replace("95% CI", "").replace("95% Wilson", "")
    assert re.search(r"\bHex \d", text) is None


def test_a_sensitivity_cohort_numbers_repeats_after_the_concentration():
    text = _drawn_text("Hex-Sensitivity-24-0.1", suffix="-0.1")
    for drawn in ("Hexanol 0.1% (v/v) #1", "Hexanol 0.1% (v/v) #1 or #2",
                  "Hexanol 0.1% (v/v) #1 & #2 mean", "Hexanol 0.1% (v/v) throughout."):
        assert drawn in text


@pytest.mark.parametrize("dataset", ["Hex-HexTest-Training-32-1", DATASET])
def test_other_cohorts_keep_their_odor_labels(dataset):
    """A recorded concentration changes nothing outside the sensitivity cohorts."""
    for suffix in ("", "-0.1"):
        text = _drawn_text(dataset, suffix=suffix)
        for drawn in ("Hex 1", "Hex 2", "Hex 1 or 2", "Hex 1 & 2 mean",
                      "Hexanol throughout.", "Which training trial predicts the Hex 1 test?"):
            assert drawn in text
        assert "(v/v)" not in text
        assert "#1" not in text


def test_the_stats_keep_the_bare_odor_and_presentation_keys(tmp_path, monkeypatch):
    train, test, test_auc = _random_panel_frames()
    monkeypatch.setattr(mod, "_load_inputs", lambda args: (train, test, test_auc))
    mod.main([
        "--training-wide-csv", "unused", "--predictions-csv", "unused",
        "--dataset", _RANDOM_DATASET, "--out-dir", str(tmp_path), "--no-svg",
    ])
    entry = json.loads(
        (tmp_path / _RANDOM_DATASET / f"{_RANDOM_DATASET}_stats.json").read_text()
    )
    assert entry["odor"] == "Isoamyl Acetate"
    assert entry["odor_token"] == "isoamylacetate"
    assert [o["presentation"] for o in entry["overview"]] == ["IAA 1"]
    assert "(v/v)" not in json.dumps(entry)


def test_main_renders_later_cohorts_when_one_fails(tmp_path, monkeypatch):
    """One cohort raising must not cost every cohort after it its figures --
    that is how IAA-Sensitivity-40-1 never got a Training-AUC folder. The run
    still exits non-zero so the pipeline logs the failure."""
    train, test, test_auc = _random_panel_frames()
    other = "IAA-Sensitivity-30-1"
    train = pd.concat([train, train.assign(dataset=other)], ignore_index=True)
    test = pd.concat([test, test.assign(dataset=other)], ignore_index=True)
    monkeypatch.setattr(mod, "_load_inputs", lambda args: (train, test, None))

    real = mod.figure_trial_profile

    def flaky(table, *, dataset, odor_label):
        if dataset == _RANDOM_DATASET:
            raise RuntimeError("boom")
        return real(table, dataset=dataset, odor_label=odor_label)

    monkeypatch.setattr(mod, "figure_trial_profile", flaky)
    with pytest.raises(SystemExit) as exc:
        mod.main([
            "--training-wide-csv", "unused", "--predictions-csv", "unused",
            "--dataset", _RANDOM_DATASET, "--dataset", other,
            "--out-dir", str(tmp_path), "--no-svg",
        ])
    assert exc.value.code not in (0, None)
    assert (tmp_path / other / f"{other}_stats.json").exists()
    sidecar = json.loads(
        (tmp_path / "training_auc_vs_control_response.json").read_text()
    )
    assert list(sidecar) == [other]


# ---------------------------------------------------------------------------
# Slope
# ---------------------------------------------------------------------------


def test_auc_slope_matches_numpy_polyfit():
    series = pd.Series([5.0, 9.0, 11.0, 20.0], index=[1, 2, 3, 4])
    expected = np.polyfit(series.index.to_numpy(float), series.to_numpy(), 1)[0]
    assert mod.auc_slope(series) == pytest.approx(expected)


def test_auc_slope_needs_two_points():
    assert np.isnan(mod.auc_slope(pd.Series([3.0], index=[1])))
    assert np.isnan(mod.auc_slope(pd.Series([], dtype=float)))


# ---------------------------------------------------------------------------
# Logistic fit -- checked against statsmodels directly
# ---------------------------------------------------------------------------


def test_fit_logistic_matches_statsmodels():
    import statsmodels.api as sm

    rng = np.random.default_rng(7)
    x = rng.uniform(0, 300, 60)
    y = (rng.uniform(size=60) < 1 / (1 + np.exp(-(x - 150) / 60))).astype(int)

    fit = mod.fit_logistic(x, y)
    ref = sm.Logit(y, sm.add_constant(x)).fit(disp=0)

    assert fit.converged
    assert fit.intercept == pytest.approx(ref.params[0], rel=1e-6)
    assert fit.slope == pytest.approx(ref.params[1], rel=1e-6)
    assert fit.p_lr == pytest.approx(ref.llr_pvalue, rel=1e-6)
    assert fit.odds_ratio_100 == pytest.approx(np.exp(ref.params[1] * 100), rel=1e-6)


def test_fit_logistic_curve_is_monotone_and_bounded():
    x = np.array([0.0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    fit = mod.fit_logistic(x, y)
    assert fit.p_grid.min() >= 0.0 and fit.p_grid.max() <= 1.0
    assert np.all(np.diff(fit.p_grid) >= -1e-12)      # rising with AUC
    assert fit.slope > 0


def test_fit_logistic_degenerate_outcomes_do_not_raise():
    x = np.arange(10, dtype=float)
    for y in (np.zeros(10, dtype=int), np.ones(10, dtype=int)):
        fit = mod.fit_logistic(x, y)
        assert not fit.converged
        assert np.isnan(fit.p_lr)
        assert fit.p_grid.size == 0


def test_fit_logistic_ignores_nan_rows():
    x = np.array([0.0, 10, np.nan, 30, 40, 50, 60, 70, 80, 90])
    y = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1, 1])
    fit = mod.fit_logistic(x, y)
    assert fit.n == 9


# ---------------------------------------------------------------------------
# Equal-count bins and binomial rates
# ---------------------------------------------------------------------------


def test_equal_count_bins_splits_evenly_and_orders_by_value():
    values = np.array([50.0, 10, 30, 60, 20, 40])
    bins = mod.equal_count_bins(values, 3)
    assert sorted(np.bincount(bins).tolist()) == [2, 2, 2]
    # The two smallest values share bin 0, the two largest share bin 2.
    assert bins[np.argsort(values)[0]] == bins[np.argsort(values)[1]] == 0
    assert bins[np.argsort(values)[-1]] == bins[np.argsort(values)[-2]] == 2


def test_equal_count_bins_marks_nan_as_minus_one():
    bins = mod.equal_count_bins(np.array([1.0, np.nan, 3.0, 2.0]), 3)
    assert bins[1] == -1
    assert set(bins[[0, 2, 3]]) == {0, 1, 2}


def test_rate_by_bin_matches_hand_counts_and_wilson():
    from scripts.analysis.corrected_stats import wilson_ci

    values = np.arange(1.0, 7.0)                       # 1..6 -> bins 0,0,1,1,2,2
    outcome = np.array([0, 0, 0, 1, 1, 1])
    rates = mod.rate_by_bin(values, outcome, 3)
    assert [(r.k, r.n) for r in rates] == [(0, 2), (1, 2), (2, 2)]
    assert [r.rate for r in rates] == [0.0, 0.5, 1.0]
    assert rates[1].lo == pytest.approx(wilson_ci(1, 2)[0])
    assert rates[1].hi == pytest.approx(wilson_ci(1, 2)[1])
    # x is the bin's median AUC, so the marker sits over its own data.
    assert [r.x_center for r in rates] == [1.5, 3.5, 5.5]


def test_rate_by_bin_drops_empty_bins_rather_than_plotting_nan():
    rates = mod.rate_by_bin(np.array([1.0, 2.0]), np.array([0, 1]), 3)
    assert all(r.n > 0 for r in rates)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


@pytest.fixture()
def table():
    return mod.build_fly_table(_training_frame(), _testing_frame(), dataset=DATASET)


def test_per_figure_has_a_panel_per_presentation_plus_pooled(table):
    fig, stats = mod.figure_per(table, dataset=DATASET, odor_label="Hexanol")
    try:
        # 2 rows (group comparison, dose-response) x 3 columns (Hex 1, Hex 2, either)
        assert len(fig.axes) == 6
        assert [s["column"] for s in stats] == ["Hex 1", "Hex 2", "Hex 1 or 2"]
        assert all("mannwhitney_p" in s and "n_per" in s for s in stats)
    finally:
        matplotlib.pyplot.close(fig)


def test_score_figure_has_a_panel_per_presentation_plus_mean(table):
    fig, stats = mod.figure_score(table, dataset=DATASET, odor_label="Hexanol")
    try:
        assert len(fig.axes) == 6
        assert [s["column"] for s in stats] == ["Hex 1", "Hex 2", "Hex 1 & 2 mean"]
        assert all("spearman_rho" in s and "spearman_p" in s for s in stats)
    finally:
        matplotlib.pyplot.close(fig)


def test_figure_stats_match_scipy_on_the_fixture(table):
    from scipy.stats import mannwhitneyu, spearmanr

    _fig, per_stats = mod.figure_per(table, dataset=DATASET, odor_label="Hexanol")
    matplotlib.pyplot.close("all")
    auc = table["mean_auc"].to_numpy(float)
    per1 = table["per_1"].to_numpy(float)
    expected = mannwhitneyu(auc[per1 < 0.5], auc[per1 >= 0.5],
                            alternative="two-sided").pvalue
    assert per_stats[0]["mannwhitney_p"] == pytest.approx(expected)

    _fig, score_stats = mod.figure_score(table, dataset=DATASET, odor_label="Hexanol")
    matplotlib.pyplot.close("all")
    rho, p = spearmanr(auc, table["score_1"].to_numpy(float))
    assert score_stats[0]["spearman_rho"] == pytest.approx(rho)
    assert score_stats[0]["spearman_p"] == pytest.approx(p)


def test_palette_is_the_project_score_palette():
    """Score points reuse the pinned CVD-validated PRGn ramp, not new hues."""
    from scripts.analysis.score_scale_figure import SCORE_COLORS

    assert mod.SCORE_COLORS == SCORE_COLORS
    # The PER split reuses the same purple/green poles (validated dE 16.7).
    assert mod.NO_PER_COLOR == SCORE_COLORS[-1]
    assert mod.PER_COLOR == SCORE_COLORS[4]


def test_holm_adjustment_is_applied_across_the_three_columns(table):
    from scripts.analysis.corrected_stats import holm_adjust

    _fig, stats = mod.figure_per(table, dataset=DATASET, odor_label="Hexanol")
    matplotlib.pyplot.close("all")
    raw = [s["mannwhitney_p"] for s in stats]
    expected = holm_adjust(raw)
    got = [s["mannwhitney_p_holm"] for s in stats]
    np.testing.assert_allclose(got, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Per-training-trial analysis: does trial 1 alone predict the test response?
# ---------------------------------------------------------------------------


@pytest.fixture()
def full_table():
    return mod.build_fly_table(
        _training_frame(), _testing_frame(), dataset=DATASET,
        testing_auc_df=_testing_auc_frame(),
    )


def test_build_fly_table_keeps_one_auc_column_per_training_trial(full_table):
    assert list(full_table.attrs["training_trials"]) == [1, 2, 3, 4]
    row = full_table.set_index(["fly", "fly_number"]).loc[("day_2", 2)]
    assert [row[f"auc_t{t}"] for t in (1, 2, 3, 4)] == [200.0, 180.0, 160.0, 140.0]
    # The pooled column stays the mean of the per-trial columns.
    assert row["mean_auc"] == pytest.approx(np.mean([200.0, 180.0, 160.0, 140.0]))


def test_build_fly_table_carries_testing_auc_when_supplied(full_table):
    row = full_table.set_index(["fly", "fly_number"]).loc[("day_2", 2)]
    assert row["test_auc_1"] == pytest.approx(5 * 40.0 + 200.0)
    assert row["test_auc_8"] == pytest.approx(4 * 40.0 + 200.0)


def test_testing_auc_columns_absent_when_not_supplied(table):
    assert not [c for c in table.columns if c.startswith("test_auc_")]


def test_trial_correlation_matrix_shape_and_labels(full_table):
    mat = mod.trial_correlation_matrix(full_table)
    # One row per training trial plus the pooled mean.
    assert list(mat.rho.index) == ["Train 1", "Train 2", "Train 3", "Train 4", "Mean"]
    # Two presentations x three outcome kinds.
    assert list(mat.rho.columns) == [
        ("Hex 1", "AUC"), ("Hex 1", "Score"), ("Hex 1", "PER"),
        ("Hex 2", "AUC"), ("Hex 2", "Score"), ("Hex 2", "PER"),
    ]
    assert mat.rho.shape == mat.p_raw.shape == mat.p_bh.shape


def test_trial_correlation_matrix_rho_matches_scipy(full_table):
    from scipy.stats import spearmanr

    mat = mod.trial_correlation_matrix(full_table)
    for trial, label in ((1, "Train 1"), (3, "Train 3")):
        rho, p = spearmanr(full_table[f"auc_t{trial}"], full_table["score_1"])
        assert mat.rho.loc[label, ("Hex 1", "Score")] == pytest.approx(rho)
        assert mat.p_raw.loc[label, ("Hex 1", "Score")] == pytest.approx(p)
    rho, _p = spearmanr(full_table["mean_auc"], full_table["per_8"])
    assert mat.rho.loc["Mean", ("Hex 2", "PER")] == pytest.approx(rho)


def test_trial_correlation_matrix_bh_matches_statsmodels(full_table):
    from scripts.analysis.corrected_stats import bh_adjust

    mat = mod.trial_correlation_matrix(full_table)
    flat = mat.p_raw.to_numpy(float).ravel()
    expected = bh_adjust(flat)
    np.testing.assert_allclose(
        mat.p_bh.to_numpy(float).ravel(), expected, rtol=1e-12, equal_nan=True
    )


def test_trial_correlation_matrix_omits_auc_columns_without_testing_auc(table):
    mat = mod.trial_correlation_matrix(table)
    assert list(mat.rho.columns) == [
        ("Hex 1", "Score"), ("Hex 1", "PER"),
        ("Hex 2", "Score"), ("Hex 2", "PER"),
    ]


def test_trial_matrix_figure_renders(full_table):
    fig, payload = mod.figure_trial_matrix(
        full_table, dataset=DATASET, odor_label="Hexanol"
    )
    try:
        assert payload["n_flies"] == 4
        assert payload["n_tests"] == 5 * 6
        assert len(payload["cells"]) == 5 * 6
    finally:
        matplotlib.pyplot.close(fig)


def test_trial_profile_figure_has_one_panel_per_outcome_kind(full_table):
    fig, payload = mod.figure_trial_profile(
        full_table, dataset=DATASET, odor_label="Hexanol"
    )
    try:
        assert [s["outcome"] for s in payload] == ["AUC", "Score", "PER"]
        # Each panel carries a rho-per-training-trial series per presentation.
        for panel in payload:
            assert list(panel["series"]) == ["Hex 1", "Hex 2"]
            assert len(panel["series"]["Hex 1"]) == 4
    finally:
        matplotlib.pyplot.close(fig)


def test_trial_profile_falls_back_to_two_panels_without_testing_auc(table):
    fig, payload = mod.figure_trial_profile(
        table, dataset=DATASET, odor_label="Hexanol"
    )
    try:
        assert [s["outcome"] for s in payload] == ["Score", "PER"]
    finally:
        matplotlib.pyplot.close(fig)


# ---------------------------------------------------------------------------
# Plain-reading figures: bars + explicit significance marks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p,expected",
    [
        (0.0004, "***"),
        (0.001, "**"),
        (0.004, "**"),
        (0.01, "*"),
        (0.049, "*"),
        (0.05, "n.s."),
        (0.4, "n.s."),
        (float("nan"), "n.s."),
    ],
)
def test_stars(p, expected):
    assert mod.stars(p) == expected


def test_stars_boundaries_are_strict():
    """0.05 is not significant; 0.0499 is. No rounding a result into a star."""
    assert mod.stars(0.0499) == "*"
    assert mod.stars(0.05) == "n.s."
    assert mod.stars(0.0099) == "**"


def test_overview_figure_is_two_by_two(full_table):
    fig, stats = mod.figure_simple_overview(
        full_table, dataset=DATASET, odor_label="Hexanol", n_bins=2
    )
    try:
        assert len(fig.axes) == 4                      # PER / score x Hex 1 / Hex 2
        assert [s["presentation"] for s in stats] == ["Hex 1", "Hex 2"]
        for s in stats:
            assert {"per_bins", "score_bins", "per_low_vs_high", "score_low_vs_high",
                    "spearman_rho", "spearman_p"} <= set(s)
    finally:
        matplotlib.pyplot.close(fig)


def test_overview_low_vs_high_matches_scipy(full_table):
    from scipy.stats import fisher_exact, mannwhitneyu

    fig, stats = mod.figure_simple_overview(
        full_table, dataset=DATASET, odor_label="Hexanol", n_bins=2
    )
    matplotlib.pyplot.close(fig)

    auc = full_table["mean_auc"].to_numpy(float)
    bins = mod.equal_count_bins(auc, 2)
    low, high = bins == 0, bins == 1

    per = full_table["per_1"].to_numpy(float) >= 0.5
    k_hi, n_hi = int(per[high].sum()), int(high.sum())
    k_lo, n_lo = int(per[low].sum()), int(low.sum())
    expected = fisher_exact([[k_hi, n_hi - k_hi], [k_lo, n_lo - k_lo]])[1]
    assert stats[0]["per_low_vs_high"]["p_fisher"] == pytest.approx(expected)

    score = full_table["score_1"].to_numpy(float)
    expected = mannwhitneyu(score[high], score[low], alternative="two-sided").pvalue
    assert stats[0]["score_low_vs_high"]["p"] == pytest.approx(expected)


def test_overview_bins_are_ordered_low_to_high(full_table):
    _fig, stats = mod.figure_simple_overview(
        full_table, dataset=DATASET, odor_label="Hexanol", n_bins=2
    )
    matplotlib.pyplot.close("all")
    aucs = [b["auc_median"] for b in stats[0]["per_bins"]]
    assert aucs == sorted(aucs)


def test_trial_predictor_figure_one_panel_per_outcome(full_table):
    fig, payload = mod.figure_trial_predictors(
        full_table, presentation_index=0, dataset=DATASET, odor_label="Hexanol"
    )
    try:
        assert payload["presentation"] == "Hex 1"
        assert [p["outcome"] for p in payload["panels"]] == ["PER", "Score", "AUC"]
        for panel in payload["panels"]:
            # Six conditioning trials in the fixture is four, plus the mean bar.
            assert [b["label"] for b in panel["bars"]] == [
                "1", "2", "3", "4", "Mean"
            ]
            assert all("stars" in b for b in panel["bars"])
    finally:
        matplotlib.pyplot.close(fig)


def test_trial_predictor_figure_second_presentation(full_table):
    fig, payload = mod.figure_trial_predictors(
        full_table, presentation_index=1, dataset=DATASET, odor_label="Hexanol"
    )
    try:
        assert payload["presentation"] == "Hex 2"
        assert payload["trial"] == 8
    finally:
        matplotlib.pyplot.close(fig)


def test_trial_predictor_bars_match_the_matrix(full_table):
    mat = mod.trial_correlation_matrix(full_table)
    _fig, payload = mod.figure_trial_predictors(
        full_table, presentation_index=0, dataset=DATASET, odor_label="Hexanol"
    )
    matplotlib.pyplot.close("all")
    score_panel = next(p for p in payload["panels"] if p["outcome"] == "Score")
    expected = [
        float(mat.rho.loc[r, ("Hex 1", "Score")])
        for r in ["Train 1", "Train 2", "Train 3", "Train 4", "Mean"]
    ]
    got = [b["rho"] for b in score_panel["bars"]]
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_trial_predictor_drops_auc_panel_without_testing_auc(table):
    fig, payload = mod.figure_trial_predictors(
        table, presentation_index=0, dataset=DATASET, odor_label="Hexanol"
    )
    try:
        assert [p["outcome"] for p in payload["panels"]] == ["PER", "Score"]
    finally:
        matplotlib.pyplot.close(fig)


@pytest.mark.parametrize("k,n", [(0, 7), (0, 18), (10, 10), (13, 13)])
def test_asym_err_never_returns_a_negative_offset(k, n):
    """Wilson bounds miss an exact 0 or 1 by an ulp; raw yerr goes negative.

    matplotlib rejects the whole errorbar on a single negative offset, so an
    all-or-nothing bin took the figure down rather than drawing flat.
    """
    from scripts.analysis.corrected_stats import wilson_ci

    lo, hi = wilson_ci(k, n)
    rate = k / n
    assert min(rate - lo, hi - rate) < 0              # the trap this guards
    err = mod.asym_err([rate], [lo], [hi])
    assert err.shape == (2, 1)
    assert (err >= 0).all()


def test_asym_err_survives_every_small_wilson_interval():
    from scripts.analysis.corrected_stats import wilson_ci

    for n in range(1, 30):
        for k in range(n + 1):
            lo, hi = wilson_ci(k, n)
            assert (mod.asym_err([k / n], [lo], [hi]) >= 0).all()


def test_asym_err_scales_and_keeps_real_widths():
    err = mod.asym_err([0.5], [0.2], [0.9], scale=100.0)
    np.testing.assert_allclose(err[:, 0], [30.0, 40.0])


def test_overview_renders_with_an_all_or_nothing_bin(full_table):
    """A bin where every fly reacted must not blow up the error bars."""
    table = full_table.copy()
    table.attrs = dict(full_table.attrs)
    table["per_1"] = [0, 0, 1, 1]
    fig, _stats = mod.figure_simple_overview(
        table, dataset=DATASET, odor_label="Hexanol", n_bins=2
    )
    matplotlib.pyplot.close(fig)


def test_resolve_dataset_is_case_insensitive():
    train = _training_frame()
    assert mod.resolve_dataset(train, "hex-CONTROL-24-0.1") == DATASET
    assert mod.resolve_dataset(train, "  Hex-Control-24-0.1 ") == DATASET
    # Unknown names come back untouched, so the caller can log a clean warning.
    assert mod.resolve_dataset(train, "Nope-24-1") == "Nope-24-1"


def test_build_fly_table_matches_a_differently_cased_dataset():
    table = mod.build_fly_table(
        _training_frame(), _testing_frame(), dataset="HEX-control-24-0.1"
    )
    assert len(table) == 4


def test_bar_label_sits_outside_the_bar_on_both_sides():
    """A negative bar's label must hang below it, not sit inside it."""
    y, va = mod.bar_label_position(0.55)
    assert y > 0.55 and va == "bottom"
    y, va = mod.bar_label_position(-0.06)
    assert y < -0.06 and va == "top"
    # Exactly zero reads as a positive bar; the label goes above the baseline.
    y, va = mod.bar_label_position(0.0)
    assert y > 0 and va == "bottom"


def test_trial_predictor_axis_clears_the_lowest_label(full_table):
    """The y floor has to leave room for a label hanging under a negative bar."""
    table = full_table.copy()
    table.attrs = dict(full_table.attrs)
    # Make trial 2 anti-correlate with the Hex 1 score.
    table["auc_t2"] = -table["score_1"].to_numpy(float) * 10.0
    fig, payload = mod.figure_trial_predictors(
        table, presentation_index=0, dataset=DATASET, odor_label="Hexanol"
    )
    try:
        panel = next(p for p in payload["panels"] if p["outcome"] == "Score")
        lowest = min(b["rho"] for b in panel["bars"] if np.isfinite(b["rho"]))
        assert lowest < 0                                   # the case being tested
        label_y = mod.bar_label_position(lowest)[0]
        for ax in fig.axes:
            assert ax.get_ylim()[0] <= label_y
    finally:
        matplotlib.pyplot.close(fig)


def test_bin_ramp_is_the_validated_ordinal_scale():
    """low/mid/high is an ORDINAL bin, so it takes one hue, light->dark."""
    assert mod.BIN_RAMP == ("#86b6ef", "#3987e5", "#1c5cab")
    assert mod.bin_colors(2) == ["#86b6ef", "#1c5cab"]
    assert mod.bin_colors(3) == list(mod.BIN_RAMP)
