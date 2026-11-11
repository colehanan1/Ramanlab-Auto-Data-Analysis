"""Per-fly pre-test vs post-training PER traces, overlaid per odor.

One figure per fly, one row per odor. The naive (pretest) trace is orange and
the post-training (testing) trace is blue on the same axes. Rows run in
alphanumeric order of the odor name and are the SAME rows on every fly of a
cohort, so two figures can be compared row for row.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import scripts.analysis.pretest_vs_test_traces as pvt


# ── odor names ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "label, expected",
    [
        ("pretest_3_hexanol-0.1", "Hexanol 0.1"),
        ("testing_6_hexanol-0.1", "Hexanol 0.1"),
        # Every rig spelling of 3-octanol lands on one row.
        ("pretest_1_3-octonol-0.1", "3-Octanol 0.1"),
        ("testing_7_3octanol-0.1", "3-Octanol 0.1"),
        ("testing_2_3-octanol-0.1", "3-Octanol 0.1"),
        ("pretest_6_isoamylacetate-1", "Isoamyl Acetate 1"),
        ("testing_6_ethylbutyrate-10", "Ethyl Butyrate 10"),
        # No concentration token: the bare display name.
        ("testing_6_3-octonol", "3-Octanol"),
        ("pretest_2_Benzaldehyde", "Benzaldehyde"),
    ],
)
def test_odor_names_are_normalised(label, expected):
    assert pvt.odor_name("IAA-Sensitivity-24-1", label) == expected


@pytest.mark.parametrize(
    "label",
    [
        "testing_6",            # no odor token at all
        "testing_9_lightonly",  # the light-only control is not an odor
        "testing_9",
    ],
)
def test_non_odor_trials_have_no_name(label):
    assert pvt.odor_name("IAA-Sensitivity-24-1", label) is None


def test_odor_order_is_alphanumeric_and_natural():
    names = ["Linalool 1", "Hexanol 1", "Hexanol 0.1", "3-Octanol 0.1",
             "Ethyl Butyrate 10", "Ethyl Butyrate 1", "benzaldehyde"]
    assert pvt.order_odors(names) == [
        "3-Octanol 0.1", "benzaldehyde", "Ethyl Butyrate 1",
        "Ethyl Butyrate 10", "Hexanol 0.1", "Hexanol 1", "Linalool 1",
    ]


# ── pairing ────────────────────────────────────────────────────────────────


def _wide(rows, n=8):
    """Tiny wide table: dir_val_0..n-1 hold a constant per row."""
    records = []
    for ds, fly, num, label, value in rows:
        rec = {"dataset": ds, "fly": fly, "fly_number": num,
               "trial_type": label.split("_")[0], "trial_label": label,
               "fps": 1.0, "trace_len": n}
        rec.update({f"dir_val_{i}": float(value) for i in range(n)})
        records.append(rec)
    return pd.DataFrame.from_records(records)


DS = "IAA-Sensitivity-24-1"


def _tables():
    pre = _wide([
        (DS, "oct_1_batch_1", 1, "pretest_1_linalool-1", 10),
        (DS, "oct_1_batch_1", 1, "pretest_2_hexanol-0.1", 11),
        (DS, "oct_1_batch_1", 2, "pretest_1_hexanol-0.1", 12),
        # pre-test only fly: no within-fly comparison, so no figure
        (DS, "oct_1_batch_1", 3, "pretest_1_hexanol-0.1", 13),
    ])
    post = _wide([
        (DS, "oct_1_batch_1", 1, "testing_1_hexanol-0.1", 21),
        (DS, "oct_1_batch_1", 1, "testing_2_linalool-1", 20),
        (DS, "oct_1_batch_1", 1, "testing_9_lightonly", 99),
        (DS, "oct_1_batch_1", 2, "testing_1_citral-1", 22),
    ])
    return pre, post


def test_only_flies_with_both_phases_are_paired():
    flies = pvt.pair_flies(*_tables())
    assert sorted((f.fly, f.fly_number) for f in flies) == [
        ("oct_1_batch_1", "1"), ("oct_1_batch_1", "2"),
    ]


def test_pairing_is_on_the_odor_not_the_trial_number():
    fly1 = next(f for f in pvt.pair_flies(*_tables()) if f.fly_number == "1")
    assert float(fly1.pre["Hexanol 0.1"].env[0]) == 11
    assert float(fly1.post["Hexanol 0.1"].env[0]) == 21
    assert float(fly1.pre["Linalool 1"].env[0]) == 10
    assert float(fly1.post["Linalool 1"].env[0]) == 20


def test_every_fly_of_a_cohort_gets_the_same_rows():
    """Rows are the cohort's odor union, so row k is the same odor on every fly."""
    flies = pvt.pair_flies(*_tables())
    assert {tuple(f.odors) for f in flies} == {
        ("Citral 1", "Hexanol 0.1", "Linalool 1")
    }


def test_the_light_only_trial_gets_no_row():
    fly1 = next(f for f in pvt.pair_flies(*_tables()) if f.fly_number == "1")
    assert all("Light" not in o for o in fly1.odors)


def test_frozen_rows_are_excluded():
    pre, post = _tables()
    pre["frozen"] = [False, False, True, False]
    post["frozen"] = False
    flies = pvt.pair_flies(pre, post)
    assert [f.fly_number for f in flies] == ["1"]


# ── the figure ─────────────────────────────────────────────────────────────


def test_pre_is_orange_and_post_is_blue():
    fly1 = next(f for f in pvt.pair_flies(*_tables()) if f.fly_number == "1")
    fig = pvt.build_figure(fly1, pvt.TraceStyle())
    axes = fig.axes[: len(fly1.odors)]
    hexanol = axes[fly1.odors.index("Hexanol 0.1")]
    hexes = matplotlib.colors.to_hex
    colours = {round(float(line.get_ydata()[0])): hexes(line.get_color()) for line in hexanol.lines}
    assert colours[11] == hexes("tab:orange")
    assert colours[21] == hexes("tab:blue")
    matplotlib.pyplot.close(fig)


def test_one_row_per_cohort_odor_in_order_even_when_a_phase_is_missing():
    fly2 = next(f for f in pvt.pair_flies(*_tables()) if f.fly_number == "2")
    fig = pvt.build_figure(fly2, pvt.TraceStyle())
    rows = fig.axes[: len(fly2.odors)]
    titles = [ax.get_title(loc="left") for ax in rows]
    assert titles == ["Citral 1", "Hexanol 0.1", "Linalool 1"]
    # Linalool was never presented to fly 2: an empty, labelled row.
    assert not rows[2].lines
    matplotlib.pyplot.close(fig)


def test_main_writes_one_png_per_fly(tmp_path):
    pre, post = _tables()
    pre_path, post_path = tmp_path / "pre.parquet", tmp_path / "post.parquet"
    pre.to_parquet(pre_path)
    post.to_parquet(post_path)
    out = tmp_path / "out"
    pvt.main([
        "--pretest-wide-csv", str(pre_path),
        "--testing-wide-csv", str(post_path),
        "--out-dir", str(out),
    ])
    pngs = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.png"))
    assert pngs == [
        f"{DS}/oct_1_batch_1_fly1_pre_vs_post_traces.png",
        f"{DS}/oct_1_batch_1_fly2_pre_vs_post_traces.png",
    ]


def test_cohorts_filter(tmp_path):
    pre, post = _tables()
    pre_path, post_path = tmp_path / "pre.parquet", tmp_path / "post.parquet"
    pre.to_parquet(pre_path)
    post.to_parquet(post_path)
    out = tmp_path / "out"
    pvt.main([
        "--pretest-wide-csv", str(pre_path),
        "--testing-wide-csv", str(post_path),
        "--out-dir", str(out),
        "--cohorts", "Some-Other-Cohort",
    ])
    assert not list(out.rglob("*.png"))


# ── unlabelled vs labelled concentration (the Oct-6 odor-pins change) ─────


def test_a_bare_odor_folds_into_the_cohorts_only_concentration():
    """Batches before ~/odor_pins.txt wrote ``hexanol``, later ones
    ``hexanol-0.1``. Same panel, so one row — not two half-empty ones."""
    pre = _wide([
        (DS, "sep_1_batch_1", 1, "pretest_1_hexanol", 1),
        (DS, "oct_9_batch_1", 1, "pretest_1_hexanol-0.1", 2),
    ])
    post = _wide([
        (DS, "sep_1_batch_1", 1, "testing_1_hexanol", 3),
        (DS, "oct_9_batch_1", 1, "testing_1_hexanol-0.1", 4),
    ])
    flies = pvt.pair_flies(pre, post)
    assert {tuple(f.odors) for f in flies} == {("Hexanol 0.1",)}
    old = next(f for f in flies if f.fly == "sep_1_batch_1")
    assert float(old.pre["Hexanol 0.1"].env[0]) == 1


def test_a_bare_odor_stays_separate_when_the_concentration_is_ambiguous():
    pre = _wide([
        (DS, "a", 1, "pretest_1_ethylbutyrate", 1),
        (DS, "b", 1, "pretest_1_ethylbutyrate-1", 2),
        (DS, "b", 1, "pretest_2_ethylbutyrate-10", 3),
    ])
    post = _wide([
        (DS, "a", 1, "testing_1_ethylbutyrate", 4),
        (DS, "b", 1, "testing_1_ethylbutyrate-1", 5),
    ])
    flies = pvt.pair_flies(pre, post)
    assert {tuple(f.odors) for f in flies} == {
        ("Ethyl Butyrate", "Ethyl Butyrate 1", "Ethyl Butyrate 10")
    }


# ── the time axis is the recording, not a fixed window ────────────────────


def test_each_row_runs_to_the_end_of_its_longest_video():
    """No hardcoded 90 s: a row's x-axis is the full recording, frames / fps."""
    pre = _wide([(DS, "f", 1, "pretest_1_hexanol-0.1", 1)], n=120)   # 120 s @ 1 fps
    post = _wide([(DS, "f", 1, "testing_1_hexanol-0.1", 2),
                  (DS, "f", 1, "testing_2_linalool-1", 3)], n=150)   # 150 s
    pre.loc[0, [f"dir_val_{i}" for i in range(100, 120)]] = float("nan")
    pre.loc[0, "trace_len"] = 100                                    # 100 s video
    fly = pvt.pair_flies(pre, post)[0]
    fig = pvt.build_figure(fly, pvt.TraceStyle())
    hexanol, linalool = fig.axes[:2]
    assert hexanol.get_xlim() == (0.0, 150.0)
    assert linalool.get_xlim() == (0.0, 150.0)
    pre_line = next(l for l in hexanol.lines if l.get_ydata()[0] == 1)
    assert pre_line.get_xdata()[-1] == 99.0      # whole 100-frame video drawn
    matplotlib.pyplot.close(fig)


def test_no_fixed_window_flag_remains():
    args = pvt.build_parser().parse_args(
        ["--pretest-wide-csv", "a", "--testing-wide-csv", "b", "--out-dir", "c"]
    )
    assert not hasattr(args, "after_show_sec")


# ── model scores, both phases ─────────────────────────────────────────────


def _scores_csv(tmp_path):
    path = tmp_path / "model_predictions.csv"
    pd.DataFrame([
        (DS, "oct_1_batch_1", 1, "pretest_2_hexanol-0.1", 0),
        (DS, "oct_1_batch_1", 1, "testing_1_hexanol-0.1", 4),
        (DS, "oct_1_batch_1", 1, "testing_2_linalool-1", 2),
    ], columns=["dataset", "fly", "fly_number", "trial_label", "score"]).to_csv(path, index=False)
    return path


def test_scores_attach_to_their_own_phase_and_trial(tmp_path):
    scores = pvt.load_scores(_scores_csv(tmp_path))
    fly1 = next(f for f in pvt.pair_flies(*_tables(), scores=scores) if f.fly_number == "1")
    assert fly1.pre["Hexanol 0.1"].score == 0
    assert fly1.post["Hexanol 0.1"].score == 4
    assert fly1.post["Linalool 1"].score == 2
    assert fly1.pre["Linalool 1"].score is None      # not in the CSV


def test_both_scores_are_drawn_in_their_phase_colour(tmp_path):
    scores = pvt.load_scores(_scores_csv(tmp_path))
    fly1 = next(f for f in pvt.pair_flies(*_tables(), scores=scores) if f.fly_number == "1")
    fig = pvt.build_figure(fly1, pvt.TraceStyle())
    hexanol = fig.axes[fly1.odors.index("Hexanol 0.1")]
    texts = {t.get_text(): matplotlib.colors.to_hex(t.get_color()) for t in hexanol.texts}
    assert texts["Pre score: 0"] == matplotlib.colors.to_hex("tab:orange")
    assert texts["Post score: 4"] == matplotlib.colors.to_hex("tab:blue")
    linalool = fig.axes[fly1.odors.index("Linalool 1")]
    assert "Pre score: n/a" in {t.get_text() for t in linalool.texts}
    matplotlib.pyplot.close(fig)


def test_no_score_text_without_a_predictions_csv():
    fly1 = next(f for f in pvt.pair_flies(*_tables()) if f.fly_number == "1")
    fig = pvt.build_figure(fly1, pvt.TraceStyle())
    assert not any("score" in t.get_text() for ax in fig.axes for t in ax.texts)
    matplotlib.pyplot.close(fig)


def test_main_reads_the_predictions_csv(tmp_path):
    pre, post = _tables()
    pre.to_parquet(tmp_path / "pre.parquet")
    post.to_parquet(tmp_path / "post.parquet")
    assert pvt.main([
        "--pretest-wide-csv", str(tmp_path / "pre.parquet"),
        "--testing-wide-csv", str(tmp_path / "post.parquet"),
        "--out-dir", str(tmp_path / "out"),
        "--predictions-csv", str(_scores_csv(tmp_path)),
    ]) == 0


# ── wording: sensitization is not conditioning ────────────────────────────


def _exposed_odor_fly(ds, exposed):
    """The cohort's exposed odor in both panels, plus a pre-only and a post-only odor."""
    pre = _wide([(ds, "f", 1, f"pretest_1_{exposed}", 1),
                 (ds, "f", 1, "pretest_2_linalool-1", 2)])
    post = _wide([(ds, "f", 1, f"testing_1_{exposed}", 3),
                  (ds, "f", 1, "testing_2_citral-1", 4)])
    return pvt.pair_flies(pre, post)[0]


def _figure_wording(fly):
    fig = pvt.build_figure(fly, pvt.TraceStyle())
    rows = fig.axes[: len(fly.odors)]
    wording = {
        "suptitle": fig.get_suptitle().splitlines()[-1],
        "legend": [t.get_text() for t in fig.legends[0].get_texts()],
        "titles": [ax.get_title(loc="left") for ax in rows],
        "notes": [t.get_text() for ax in rows for t in ax.texts],
    }
    matplotlib.pyplot.close(fig)
    return wording


def test_a_sensitivity_cohort_reads_pre_and_post_sensitization():
    """Its exposure block pairs the odor with nothing: no pre-test, no CS+."""
    wording = _figure_wording(_exposed_odor_fly("IAA-Sensitivity-1-1", "isoamylacetate-1"))
    assert wording == {
        "suptitle": "PER traces, pre-exposure vs post-exposure",
        "legend": ["Pre-exposure", "Post-exposure", "Odor"],
        "titles": ["Citral 1", "Isoamyl Acetate 1  (sensitized odor)", "Linalool 1"],
        "notes": ["no pre-exposure trial", "no post-exposure trial"],
    }


def test_the_light_conditioned_hex_test_keeps_its_conditioning_wording():
    """Same pre/post protocol, but the light really is paired with hexanol."""
    wording = _figure_wording(_exposed_odor_fly("Hex-HexTest-Training-32-1", "hexanol-1"))
    assert wording == {
        "suptitle": "PER traces, pre-test vs post-training",
        "legend": ["Pre-test (naive)", "Post-training", "Odor"],
        "titles": ["Citral 1", "Hexanol 1  (trained, CS+)", "Linalool 1"],
        "notes": ["no pre-test trial", "no post-training trial"],
    }


def test_the_shared_phase_labels_keep_their_conditioning_text():
    """score_window_comparison imports these; the sensitivity wording is per figure."""
    assert (pvt.PRE_LABEL, pvt.POST_LABEL) == ("Pre-test (naive)", "Post-training")
