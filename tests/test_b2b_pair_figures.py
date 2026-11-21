"""scripts/analysis/b2b_pair_figures.py: which presentations are compared, and their scores."""
import sys
from pathlib import Path

import numpy as np
import pytest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analysis import b2b_pair_figures as b  # noqa: E402

N = 100


def _row(ds, label, on=30.0):
    r = {"dataset": ds, "fly": "b1", "fly_number": 1, "trial_label": label, "fps": 1.0,
         "trial_odor_on_s": on, "trial_odor_off_s": on + 10.0}
    r.update({f"dir_val_{i}": float(i) for i in range(N)})
    return r


def test_first_compares_partner_alone_vs_after_iaa():
    pre = pd.DataFrame([_row(b.FIRST, "pretest_2_linalool-1"), _row(b.FIRST, "pretest_8_isoamylacetate-1")])
    pairs = pd.DataFrame([_row(b.FIRST, "testing_3_isoamylacetate-1_then_linalool-1"),
                          _row(b.FIRST, "testing_10_isoamylacetate-1_then_linalool-1")])
    scores = {(b.FIRST, "b1", 1, "pretest_2_linalool-1"): 0,
              (b.FIRST, "b1", 1, "testing_3_isoamylacetate-1_then_linalool-1__odor2"): 2,
              (b.FIRST, "b1", 1, "testing_10_isoamylacetate-1_then_linalool-1__odor2"): 4,
              (b.FIRST, "b1", 1, "testing_3_isoamylacetate-1_then_linalool-1"): 5}   # odor 1: unused
    recs = b.build_records(pre, pairs, scores, lambda r: 41.0)
    got = recs.sort_values(["round"])[["role", "round", "odor", "partner", "score", "onset_s"]].values.tolist()
    assert got == [["pre", 0, "Linalool", "Linalool", 0, 30.0],
                   ["post", 1, "Linalool", "Linalool", 2, 41.0],
                   ["post", 2, "Linalool", "Linalool", 4, 41.0]]


def test_second_compares_iaa_alone_vs_after_partner():
    pre = pd.DataFrame([_row(b.SECOND, "pretest_8_isoamylacetate-1"), _row(b.SECOND, "pretest_2_citral-1")])
    pairs = pd.DataFrame([_row(b.SECOND, "testing_5_citral-1_then_isoamylacetate-1"),
                          _row(b.SECOND, "testing_12_citral-1_then_isoamylacetate-1"),
                          _row(b.SECOND, "testing_7_hexanol-0.1_then_isoamylacetate-1")])
    recs = b.build_records(pre, pairs, {}, lambda r: 41.0)
    cit = recs[recs.partner == "Citral"].sort_values("round")
    assert cit[["role", "round", "odor", "label"]].values.tolist() == [
        ["pre", 0, "IAA", "pretest_8_isoamylacetate-1"],
        ["post", 1, "IAA", "testing_5_citral-1_then_isoamylacetate-1"],
        ["post", 2, "IAA", "testing_12_citral-1_then_isoamylacetate-1"]]
    assert set(recs[recs.partner == "Hexanol"].role) == {"pre", "post"}


def test_aligned_window_and_padding():
    rec = {"fps": 1.0, "onset_s": 5.0, "trace": np.arange(N, dtype=float)}
    t, y = b.aligned(rec)
    assert t[0] == b.WINDOW_S[0] and len(t) == int(b.WINDOW_S[1] - b.WINDOW_S[0])
    assert np.isnan(y[0]) and y[list(t).index(0.0)] == 5.0


def test_mean_score_pools_trials():
    rows = pd.DataFrame({"score": [0, 4, 4, None]})
    assert b._mean_score(rows) == (8 / 3, 3)


# ── does a fly that reacts to odor 1 also react to odor 2? ─────────────────────
def test_pair_reactions_table_and_conditionals():
    P = "testing_{}_isoamylacetate-1_then_linalool-1"
    pred = pd.DataFrame([
        # (odor1, odor2) scores for 4 pair trials of one fly
        *[dict(dataset=b.FIRST, fly="b1", fly_number=1, trial_label=P.format(i), score=s1)
          for i, s1 in ((1, 3), (2, 4), (3, 0), (4, 1))],
        *[dict(dataset=b.FIRST, fly="b1", fly_number=1, trial_label=P.format(i) + "__odor2", score=s2)
          for i, s2 in ((1, 2), (2, 0), (3, 5), (4, 0))],
        dict(dataset=b.FIRST, fly="b1", fly_number=1, trial_label="pretest_1_linalool-1", score=5),
    ])
    t = b.pair_reactions(pred, threshold=2)
    assert len(t) == 4
    assert t.sort_values("number")[["r1", "r2"]].values.tolist() == [[True, True], [True, False],
                                                                      [False, True], [False, False]]
    c = b.conditional_rates(t)
    assert c == {"n_r1": 2, "r2_given_r1": 0.5, "n_not_r1": 2, "r2_given_not_r1": 0.5,
                 "n": 4, "r2": 0.5, "both": 1, "only1": 1, "only2": 1, "neither": 1}


# ── rebuilt by the pipeline, only when its inputs changed (user 2026-11-16) ─────
import yaml  # noqa: E402

from scripts.pipeline.run_workflows import (  # noqa: E402
    _b2b_pair_figures_command, _b2b_pair_figures_commands)

PAIR = "testing_3_isoamylacetate-1_then_linalool-1"


def _fixture(tmp_path, *, flies=("b1",), frozen=(), dead=(), starts=None):
    """Wide tables, predictions, config and rig frame logs for FIRST-dataset flies.
    ``starts``: fly -> HHMMSS of a raw pre-test trial file (the first pre-test start)."""
    csv_dir, data, out = tmp_path / "csv", tmp_path / "data", tmp_path / "out"
    csv_dir.mkdir(parents=True)
    pre, pair, pred = [], [], []
    for fly in flies:
        for label, rows in (("pretest_2_linalool-1", pre), (PAIR, pair)):
            r = _row(b.FIRST, label)
            r.update(fly=fly, frozen=fly in frozen)
            rows.append(r)
        for label, s in (("pretest_2_linalool-1", 0), (PAIR, 3), (PAIR + "__odor2", 4)):
            pred.append(dict(dataset=b.FIRST, fly=fly, fly_number=1, trial_label=label, score=s))
        d = data / b.FIRST / fly
        d.mkdir(parents=True)
        on = ["off"] * 30 + ["ofm_1"] * 10 + ["off"] * 1 + ["ofm_2"] * 10 + ["off"] * 49
        pd.DataFrame({"FrameNumber": range(N), "ActiveOFM": on}).to_csv(
            d / f"output_{fly}_{PAIR}_20261113_120000.csv", index=False)
        if fly in (starts or {}):
            pd.DataFrame({"FrameNumber": range(3)}).to_csv(
                d / f"output_{fly}_pretest_2_linalool-1_20261113_{starts[fly]}.csv", index=False)
    pd.DataFrame(pair).to_parquet(csv_dir / "all_envelope_rows_wide_combined_base.parquet")
    pd.DataFrame(pre).to_parquet(csv_dir / "all_envelope_rows_wide_combined_base_pretest.parquet")
    pd.DataFrame(pred).to_csv(csv_dir / "model_predictions.csv", index=False)
    cfg = tmp_path / "config.yaml"
    flagged = tmp_path / "flagged-flys-truth.csv"
    pd.DataFrame([dict(dataset=b.FIRST, fly=f, fly_number=1.0, **{"FLY-State(1, 0, -1)": -1})
                  for f in dead] + [dict(dataset=b.FIRST, fly="alive", fly_number=1,
                                         **{"FLY-State(1, 0, -1)": 1})]).to_csv(flagged, index=False)
    cfg.write_text(yaml.safe_dump({"reaction_prediction": {"binary_threshold": 2},
                                   "flagged_flies_csv": str(flagged)}))
    argv = ["--csv-dir", str(csv_dir), "--predictions", str(csv_dir / "model_predictions.csv"),
            "--out", str(out), "--config", str(cfg), "--data-root", str(data)]
    return argv, out, csv_dir


def test_rebuilds_only_when_inputs_change(tmp_path):
    argv, out, csv_dir = _fixture(tmp_path)
    assert b.main(argv) == "built"
    assert (out / "IAA-first" / "per_fly" / "b1_fly1.png").is_file()
    assert (out / "b2b_presentations.csv").is_file()
    assert b.main(argv) == "kept"
    # a new score -> rebuilt
    pred = pd.read_csv(csv_dir / "model_predictions.csv")
    pred.loc[pred.trial_label == PAIR + "__odor2", "score"] = 1
    pred.to_csv(csv_dir / "model_predictions.csv", index=False)
    assert b.main(argv) == "built"
    # a figure lost since the build -> rebuilt
    (out / "both" / "both_scores.png").unlink()
    assert b.main(argv) == "built"
    assert b.main(argv + ["--force"]) == "built"


def test_other_datasets_rows_do_not_trigger_a_rebuild(tmp_path):
    argv, out, csv_dir = _fixture(tmp_path)
    assert b.main(argv) == "built"
    path = csv_dir / "all_envelope_rows_wide_combined_base.parquet"
    t = pd.read_parquet(path)
    other = t.iloc[[0]].assign(dataset="EB-Control-24-1", fly="x")
    other["dir_val_200"] = 1.0           # a longer trace widens the shared table
    pd.concat([t, other], ignore_index=True).to_parquet(path)
    assert b.main(argv) == "kept"


def test_a_fly_that_leaves_loses_its_per_fly_figure(tmp_path):
    argv, out, _ = _fixture(tmp_path, flies=("b1", "b2"))
    b.main(argv)
    assert (out / "IAA-first" / "per_fly" / "b2_fly1.png").is_file()
    argv, out, _ = _fixture(tmp_path / "again", flies=("b1", "b2"), frozen=("b2",))
    argv[argv.index("--out") + 1] = str(tmp_path / "out")
    assert b.main(argv) == "built"
    assert not (tmp_path / "out" / "IAA-first" / "per_fly" / "b2_fly1.png").exists()


def test_frozen_folders_leave_the_odor1_vs_odor2_counts(tmp_path):
    argv, out, _ = _fixture(tmp_path, flies=("b1", "b2"), frozen=("b2",))
    b.main(argv)
    t = pd.read_csv(out / "odor1_vs_odor2" / "pair_trial_reactions.csv")
    assert set(t.fly) == {"b1"}


def test_pipeline_command_only_when_configured():
    class R:
        output_csv, python = "/csv/model_predictions.csv", ""

    class S:
        reaction_prediction = R()

    assert _b2b_pair_figures_command({}, S(), python_exec="py") is None
    assert _b2b_pair_figures_command({"b2b_pair_figures": {"enabled": False, "out_dir": "/r"}},
                                     S(), python_exec="py") is None
    cmd = _b2b_pair_figures_command({"b2b_pair_figures": {"out_dir": "/res/b2b"}}, S(),
                                    python_exec="py", config_path=Path("/cfg/c.yaml"))
    assert cmd[0] == "py" and cmd[1].endswith("scripts/analysis/b2b_pair_figures.py")
    assert cmd[cmd.index("--out") + 1] == "/res/b2b"
    assert cmd[cmd.index("--csv-dir") + 1] == "/csv"
    assert cmd[cmd.index("--predictions") + 1] == "/csv/model_predictions.csv"
    assert cmd[cmd.index("--config") + 1] == "/cfg/c.yaml"


def test_flagged_dead_flies_are_left_out_like_the_scorer_leaves_them(tmp_path):
    argv, out, _ = _fixture(tmp_path, flies=("b1", "b2"), dead=("b2",))
    b.main(argv)
    assert not (out / "IAA-first" / "per_fly" / "b2_fly1.png").exists()
    assert set(pd.read_csv(out / "b2b_presentations.csv").fly) == {"b1"}
    assert set(pd.read_csv(out / "odor1_vs_odor2" / "pair_trial_reactions.csv").fly) == {"b1"}


def test_short_folder_names_the_rig():
    assert b.short_folder("november_14_batch_1_rig_3") == "11-14 b1 rig3"
    assert b.short_folder("november_13_batch_1") == "11-13 b1 rig1"
    assert b.short_folder("odd") == "odd"


# ── score bars: mean ± SEM and % responding ± SEM, one n in the corner (user 2026-11-16) ──
def test_bar_stats_pool_trials_with_sem_over_trials():
    rows = pd.DataFrame({"score": [0, 4, 4, None]})
    st = b.bar_stats(rows, threshold=2)
    assert st["n"] == 3
    assert st["mean"] == pytest.approx(8 / 3)
    assert st["sem"] == pytest.approx(np.std([0, 4, 4], ddof=1) / np.sqrt(3))
    assert st["pct"] == pytest.approx(100 * 2 / 3)
    assert st["pct_sem"] == pytest.approx(100 * np.std([0, 1, 1], ddof=1) / np.sqrt(3))


def test_one_n_goes_in_the_corner_only_when_every_bar_shares_it():
    assert b.n_note([12, 12, 12]) == "n = 12 trials per bar"
    assert b.n_note([12, 11, 12]) is None


def _bar_recs(n_flies=3):
    recs = []
    for fly in range(n_flies):
        for partner in ("Linalool", "Hexanol"):
            recs.append(dict(dataset=b.FIRST, fly=f"f{fly}", fly_number=1, role="pre", round=0,
                             partner=partner, score=fly))
            for rnd in (1, 2):
                recs.append(dict(dataset=b.FIRST, fly=f"f{fly}", fly_number=1, role="post",
                                 round=rnd, partner=partner, score=fly + rnd))
    return pd.DataFrame(recs)


def test_score_bars_write_png_pdf_and_the_plotted_table(tmp_path):
    paths = b.score_bars(_bar_recs(), "t", tmp_path / "x_scores.png", threshold=2)
    assert {p.name for p in paths} == {"x_scores.png", "x_scores.pdf", "x_scores.csv",
                                       "x_scores_stats.csv"}
    t = pd.read_csv(tmp_path / "x_scores.csv")
    assert len(t) == 6 and set(t["n"]) == {3}
    lin2 = t[(t.partner == "Linalool") & (t.condition == b.BAR_LABELS[2])].iloc[0]
    assert lin2["mean_score"] == pytest.approx(3.0)            # scores 2, 3, 4
    assert lin2["pct_responding"] == pytest.approx(100.0)      # all >= 2


def test_percent_panel_has_no_error_bars(tmp_path):
    """User 2026-11-16: SEM on the score panel only; the % panel is bars alone."""
    b.score_bars(_bar_recs(), "t", tmp_path / "x_scores.png", threshold=2)
    assert "sem_pct" not in pd.read_csv(tmp_path / "x_scores.csv").columns
    assert b.BAR_PANELS[1][1] is None


# ── 1st and 2nd presentation in the pair pooled into one bar (user 2026-11-16) ──
def test_pooled_bars_pool_both_presentations_trials(tmp_path):
    paths = b.score_bars(_bar_recs(), "t", tmp_path / "x_scores_pooled.png", threshold=2,
                         pool_presentations=True)
    assert {p.name for p in paths} == {"x_scores_pooled.png", "x_scores_pooled.pdf",
                                       "x_scores_pooled.csv", "x_scores_pooled_stats.csv"}
    t = pd.read_csv(tmp_path / "x_scores_pooled.csv")
    assert list(t.condition.unique()) == [b.BAR_LABELS["pre"], b.BAR_LABELS["pair"]]
    lin = t[(t.partner == "Linalool") & (t.condition == b.BAR_LABELS["pair"])].iloc[0]
    # flies 0..2: 1st presentation scores 1,2,3; 2nd 2,3,4 -> 6 trials pooled
    assert lin["n"] == 6
    assert lin["mean_score"] == pytest.approx(15 / 6)
    assert lin["pct_responding"] == pytest.approx(100 * 5 / 6)


def test_corner_note_names_each_conditions_n_when_they_differ():
    t = pd.DataFrame({"key": ["pre", "pair", "pre", "pair"], "n": [12, 24, 12, 24]})
    assert b.corner_n(t) == "n = 12 trials alone, 24 in pair (2 per fly)"
    assert b.corner_n(t.assign(n=12)) == "n = 12 trials per bar"
    assert b.corner_n(t.assign(n=[12, 24, 11, 24])) is None


# ── significance: paired sign-flip test across flies, Holm within a panel (user 2026-11-16) ──
def test_sign_flip_p_is_exact():
    assert b.sign_flip_p([1, 1, 1, 1]) == pytest.approx(2 / 16)
    assert b.sign_flip_p([1] * 6) == pytest.approx(2 / 64)
    assert b.sign_flip_p([1, 1, 1, 0]) == pytest.approx(2 / 8)   # a tie carries no sign
    assert b.sign_flip_p([0, 0]) == 1.0
    assert b.sign_flip_p([0.5, 1.5, -0.5]) == pytest.approx(6 / 8)   # |sum| >= 1.5


def test_holm_and_stars():
    assert b.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert [b.stars(p) for p in (0.0009, 0.009, 0.04, 0.06, float("nan"))] == ["***", "**", "*", "", ""]


def test_stats_pair_each_fly_with_itself(tmp_path):
    recs = _bar_recs(n_flies=6)          # 2nd presentation = alone + 2 for every fly
    b.score_bars(recs, "t", tmp_path / "x_scores.png", threshold=2)
    st = pd.read_csv(tmp_path / "x_scores_stats.csv")
    row = st[(st.panel == "score") & (st.partner == "Linalool") & (st.a == b.BAR_LABELS["pre"])
             & (st.b == b.BAR_LABELS[2])].iloc[0]
    assert row["n_flies"] == 6 and row["mean_diff"] == pytest.approx(2.0)
    assert row["p"] == pytest.approx(2 / 64)
    assert len(st[st.panel == "score"]) == 2 * 3          # 2 odors x 3 comparisons
    assert row["p_holm"] == pytest.approx(min(1.0, row["p"] * 6), rel=0, abs=0) or row["p_holm"] >= row["p"]
    pooled = b.score_bars(recs, "t", tmp_path / "y.png", threshold=2, pool_presentations=True)
    sp = pd.read_csv(tmp_path / "y_stats.csv")
    assert set(zip(sp.a, sp.b)) == {(b.BAR_LABELS["pre"], b.BAR_LABELS["pair"])}
    # pooled per fly: mean of the 2 pair trials (alone+1, alone+2) - alone = 1.5
    assert sp[(sp.panel == "score")].mean_diff.tolist() == pytest.approx([1.5, 1.5])


# ── odor 1 of each pair vs the same odor naive; the naive pre-test by group (user 2026-11-16) ──
def test_odor1_records_first_score_iaa_at_the_start_of_each_pair_vs_naive_iaa():
    pre = pd.DataFrame([_row(b.FIRST, "pretest_2_linalool-1"), _row(b.FIRST, "pretest_8_isoamylacetate-1")])
    pairs = pd.DataFrame([_row(b.FIRST, "testing_3_isoamylacetate-1_then_linalool-1"),
                          _row(b.FIRST, "testing_10_isoamylacetate-1_then_linalool-1")])
    scores = {(b.FIRST, "b1", 1, "pretest_8_isoamylacetate-1"): 1,
              (b.FIRST, "b1", 1, "pretest_2_linalool-1"): 0,                 # not odor 1: unused
              (b.FIRST, "b1", 1, "testing_3_isoamylacetate-1_then_linalool-1"): 3,
              (b.FIRST, "b1", 1, "testing_10_isoamylacetate-1_then_linalool-1"): 4,
              (b.FIRST, "b1", 1, "testing_3_isoamylacetate-1_then_linalool-1__odor2"): 5}
    recs = b.odor1_records(pre, pairs, scores)
    got = recs.sort_values("round")[["role", "round", "odor", "partner", "score"]].values.tolist()
    assert got == [["pre", 0, "IAA", "Linalool", 1], ["post", 1, "IAA", "Linalool", 3],
                   ["post", 2, "IAA", "Linalool", 4]]


def test_odor1_records_second_score_the_other_odor_vs_its_own_naive_trial():
    pre = pd.DataFrame([_row(b.SECOND, "pretest_8_isoamylacetate-1"), _row(b.SECOND, "pretest_2_citral-1")])
    pairs = pd.DataFrame([_row(b.SECOND, "testing_5_citral-1_then_isoamylacetate-1"),
                          _row(b.SECOND, "testing_12_citral-1_then_isoamylacetate-1")])
    scores = {(b.SECOND, "b1", 1, "pretest_2_citral-1"): 2,
              (b.SECOND, "b1", 1, "pretest_8_isoamylacetate-1"): 0,
              (b.SECOND, "b1", 1, "testing_5_citral-1_then_isoamylacetate-1"): 1,
              (b.SECOND, "b1", 1, "testing_12_citral-1_then_isoamylacetate-1"): 0}
    recs = b.odor1_records(pre, pairs, scores)
    got = recs.sort_values("round")[["role", "round", "odor", "partner", "label", "score"]].values.tolist()
    assert got == [["pre", 0, "Citral", "Citral", "pretest_2_citral-1", 2],
                   ["post", 1, "Citral", "Citral", "testing_5_citral-1_then_isoamylacetate-1", 1],
                   ["post", 2, "Citral", "Citral", "testing_12_citral-1_then_isoamylacetate-1", 0]]


def test_pretest_records_one_per_fly_and_odor():
    pre = pd.DataFrame([_row(b.FIRST, "pretest_2_linalool-1"), _row(b.FIRST, "pretest_8_isoamylacetate-1"),
                        {**_row(b.SECOND, "pretest_2_linalool-1"), "fly": "b2"}])
    scores = {(b.FIRST, "b1", 1, "pretest_2_linalool-1"): 3, (b.SECOND, "b2", 1, "pretest_2_linalool-1"): 0}
    t = b.pretest_records(pre, scores)
    assert sorted(t[["dataset", "fly", "odor"]].values.tolist()) == [
        [b.FIRST, "b1", "IAA"], [b.FIRST, "b1", "Linalool"], [b.SECOND, "b2", "Linalool"]]
    assert t.set_index(["dataset", "odor"]).loc[(b.SECOND, "Linalool"), "score"] == 0


def test_unpaired_permutation_p_is_exact_for_small_groups():
    import itertools
    assert b.perm_p_unpaired([0, 0], [1, 1]) == pytest.approx(2 / 6)
    assert b.perm_p_unpaired([1, 2], [2, 1]) == 1.0
    assert np.isnan(b.perm_p_unpaired([], [1, 2]))
    a, c = [0, 1, 3], [2, 4, 5, 5]
    x, obs = np.array(a + c, float), abs(np.mean(a) - np.mean(c))
    splits = list(itertools.combinations(range(7), 3))
    want = np.mean([abs(x[list(s)].mean() - np.delete(x, list(s)).mean()) >= obs - 1e-12 for s in splits])
    assert b.perm_p_unpaired(a, c) == pytest.approx(want)


def test_pretest_bars_compare_the_two_groups_per_odor(tmp_path):
    recs = pd.DataFrame(
        [dict(dataset=b.FIRST, fly=f"f{i}", fly_number=1, odor="Linalool", score=0) for i in range(4)]
        + [dict(dataset=b.SECOND, fly=f"s{i}", fly_number=1, odor="Linalool", score=4) for i in range(4)]
        + [dict(dataset=ds, fly=f"{p}{i}", fly_number=1, odor="IAA", score=i)
           for ds, p in ((b.FIRST, "f"), (b.SECOND, "s")) for i in range(4)])
    paths = b.pretest_bars(recs, tmp_path / "pre.png", threshold=2, title="t")
    assert {p.name for p in paths} == {"pre.png", "pre.pdf", "pre.csv", "pre_stats.csv"}
    t = pd.read_csv(tmp_path / "pre.csv")
    assert len(t) == 4 and set(t["n"]) == {4}
    sec = t[(t.odor == "Linalool") & (t.dataset == b.SECOND)].iloc[0]
    assert sec["mean_score"] == 4 and sec["pct_responding"] == 100
    st = pd.read_csv(tmp_path / "pre_stats.csv")
    lin = st[(st.panel == "score") & (st.odor == "Linalool")].iloc[0]
    # fly level (secondary column): 2 of the C(8, 4) splits are as extreme
    assert lin["p_fly"] == pytest.approx(2 / 70) and lin["mean_diff"] == pytest.approx(4.0)
    assert st[(st.panel == "score") & (st.odor == "IAA")].iloc[0]["p_fly"] == 1.0
    # every fly is its own session here, so no folder can change group: batch p = 1
    assert lin["p_batch"] == 1.0


def test_pretest_groups_are_compared_by_permuting_folders_within_each_session(tmp_path):
    """Audit 2026-11-17: group is a batch property; a session's three rigs start together,
    so folders swap groups only within a session, keeping its number of PairSecond ones."""
    rows = []
    for day in range(4):                              # 4 sessions x (1 first + 2 second)
        for rig, ds, score in (("", b.FIRST, 0), ("_rig_2", b.SECOND, 4), ("_rig_3", b.SECOND, 4)):
            for fly in (1, 2):
                rows.append(dict(dataset=ds, fly=f"nov_{day}_batch_1{rig}", fly_number=fly,
                                 odor="Linalool", score=score))
    b.pretest_bars(pd.DataFrame(rows), tmp_path / "pre.png", threshold=2, title="t")
    st = pd.read_csv(tmp_path / "pre_stats.csv")
    lin = st[(st.panel == "score") & (st.odor == "Linalool")].iloc[0]
    assert lin["p_batch"] == pytest.approx(1 / 81)        # 3^4 arrangements, 1 as extreme
    assert lin["p_holm"] == pytest.approx(b.holm(st[st.panel == "score"]["p_batch"])[0])


def test_build_draws_the_odor1_and_pretest_figures(tmp_path):
    argv, out, _ = _fixture(tmp_path)
    assert b.main(argv) == "built"
    for name in ("IAA-first/IAA-first_odor1_vs_naive.png", "IAA-first/IAA-first_odor1_vs_naive_pooled.png",
                 "pretest/pretest_naive_by_group.png", "b2b_odor1_presentations.csv"):
        assert (out / name).is_file(), name
    t = pd.read_csv(out / "IAA-first" / "IAA-first_odor1_vs_naive.csv")
    got = t[t.condition == b.ODOR1_LABELS[1]][["partner", "mean_score"]].values.tolist()
    assert got == [["Linalool", 3.0]]          # the fixture's odor-1 (IAA) score


def test_odor1_figures_name_naive_and_the_test_block_in_the_n_note():
    t = pd.DataFrame({"key": ["pre", "pair"], "n": [20, 40]})
    assert b.corner_n(t, b.ODOR1_N_WORDS) == "n = 20 trials naive, 40 in test block (2 per fly)"


# ── clock windows: flies by their first pre-test start (user 2026-11-17) ───────
def _windowed(tmp_path):
    """b1 starts at 9:40, b2 at 16:20 (the day window's end, excluded), b3 at 17:15."""
    argv, out, _ = _fixture(tmp_path, flies=("b1", "b2", "b3"),
                            starts={"b1": "094000", "b2": "162000", "b3": "171500"})
    return argv, out


def _out(argv, path):
    argv = list(argv)
    argv[argv.index("--out") + 1] = str(path)
    return argv


def test_a_clock_window_keeps_the_folders_whose_first_pretest_started_inside(tmp_path):
    argv, out = _windowed(tmp_path)
    assert b.main(argv + ["--start", "09:40", "--end", "16:20"]) == "built"
    assert set(pd.read_csv(out / "b2b_presentations.csv").fly) == {"b1"}
    assert set(pd.read_csv(out / "odor1_vs_odor2" / "pair_trial_reactions.csv").fly) == {"b1"}
    eve = tmp_path / "eve"
    assert b.main(_out(argv, eve) + ["--start", "16:40", "--end", "21:20"]) == "built"
    assert set(pd.read_csv(eve / "b2b_presentations.csv").fly) == {"b3"}
    assert not (eve / "IAA-first" / "per_fly" / "b1_fly1.png").exists()
    readme = (eve / "README.md").read_text()
    assert "4:40pm" in readme and "9:20pm" in readme
    assert "b3" in readme and "17:15" in readme and "b1" in readme     # in, and left out


def test_without_a_window_every_folder_is_drawn(tmp_path):
    argv, out = _windowed(tmp_path)
    assert b.main(argv) == "built"
    assert set(pd.read_csv(out / "b2b_presentations.csv").fly) == {"b1", "b2", "b3"}


def test_a_folder_without_a_pretest_file_is_in_no_window(tmp_path):
    argv, out, _ = _fixture(tmp_path, flies=("b1", "b2"), starts={"b1": "120000"})
    assert b.main(argv + ["--start", "09:40", "--end", "16:20"]) == "built"
    assert set(pd.read_csv(out / "b2b_presentations.csv").fly) == {"b1"}
    assert "b2" in (out / "README.md").read_text()


def test_a_new_window_redraws_and_an_empty_one_clears_the_folder(tmp_path):
    argv, out = _windowed(tmp_path)
    day = argv + ["--start", "09:40", "--end", "16:20"]
    assert b.main(day) == "built"
    assert b.main(day) == "kept"
    assert b.main(argv + ["--start", "09:30", "--end", "16:20"]) == "built"   # same flies
    late = argv + ["--start", "21:30", "--end", "23:00"]
    assert b.main(late) == "empty"
    assert not (out / "both" / "both_scores.png").exists()     # the last window's figures go
    assert not (out / "b2b_presentations.csv").exists()
    assert not (out / "IAA-first").exists()
    assert "no batch started" in (out / "README.md").read_text().lower()
    assert b.main(late) == "kept"


def test_start_and_end_come_together(tmp_path):
    argv, _ = _windowed(tmp_path)
    with pytest.raises(SystemExit):
        b.main(argv + ["--start", "09:40"])


class _Settings:
    class reaction_prediction:
        output_csv, python = "/csv/model_predictions.csv", ""


def test_pipeline_runs_one_command_per_clock_window():
    cfg = {"b2b_pair_figures": {"windows": [
        {"out_dir": "/res/b2b", "start": "09:40", "end": "16:20"},
        {"out_dir": "/res/b2b-Evening", "start": 1000, "end": "21:20"}]}}   # YAML 16:40 -> 1000
    cmds = _b2b_pair_figures_commands(cfg, _Settings(), python_exec="py")
    got = [tuple(c[c.index(k) + 1] for k in ("--out", "--start", "--end")) for c in cmds]
    assert got == [("/res/b2b", "09:40", "16:20"), ("/res/b2b-Evening", "16:40", "21:20")]
    assert all(c[c.index("--csv-dir") + 1] == "/csv" for c in cmds)
    one = _b2b_pair_figures_commands({"b2b_pair_figures": {"out_dir": "/r"}}, _Settings(),
                                     python_exec="py")
    assert len(one) == 1 and "--start" not in one[0]
    assert _b2b_pair_figures_commands({"b2b_pair_figures": {"enabled": False, "windows": [
        {"out_dir": "/r", "start": "09:40", "end": "16:20"}]}}, _Settings(), python_exec="py") == []


def test_a_new_folder_outside_an_empty_window_still_updates_its_readme(tmp_path):
    argv, out, _ = _fixture(tmp_path, flies=("b1",), starts={"b1": "094000"})
    late = ["--start", "16:40", "--end", "21:20"]
    assert b.main(argv + late) == "empty"
    argv2, _, _ = _fixture(tmp_path / "more", flies=("b1", "b2"),
                           starts={"b1": "094000", "b2": "100000"})
    assert b.main(_out(argv2, out) + late) == "empty"        # not "kept": the README changed
    assert "b2" in (out / "README.md").read_text()


# ── odor + 5 s after window, IAA-first scores pooled (user 2026-11-17) ─────────
def test_plus_after_rows_are_the_pretest_trials_and_each_pairs_second_odor():
    pre = pd.DataFrame([_row(b.FIRST, "pretest_2_linalool-1")])
    pairs = pd.DataFrame([_row(b.FIRST, PAIR)])
    rows = b.plus_after_rows(pre, pairs, lambda r: (41.0, 51.0))
    assert sorted(rows.trial_label) == sorted(["pretest_2_linalool-1", PAIR + "__odor2"])
    o2 = rows[rows.trial_label == PAIR + "__odor2"].iloc[0]
    # odor 2 timed as a normal trial: on at the pre-odor block's end, 10 s long
    assert (o2["trial_odor_on_s"], o2["trial_odor_off_s"]) == (30.0, 40.0)
    assert o2["dir_val_30"] == 41.0                      # the trace resumes at odor 2's onset


def test_the_scorer_rescores_on_odor_plus_5s_with_the_config_model(monkeypatch, tmp_path):
    seen = {}

    def fake_rescore(df, spec, model_path, *, rule=None):
        seen.update(spec=spec, model=model_path, n=len(df))
        return np.zeros(len(df), dtype=int)

    monkeypatch.setattr(b.wr, "rescore", fake_rescore)
    monkeypatch.setattr(b.wr, "threshold_rule_from_config", lambda p: None)
    score = b.make_plus_after_scorer(Path("/m/model.json"), tmp_path / "c.yaml")
    assert list(score(pd.DataFrame([_row(b.FIRST, "pretest_2_linalool-1")]))) == [0]
    assert seen["model"] == Path("/m/model.json") and seen["n"] == 1
    assert (seen["spec"].name, seen["spec"].end_ref, seen["spec"].end_offset_s) == (
        "odor_plus_after", "off", 5.0)


def _with_fake_scorer(monkeypatch, tmp_path, scorer):
    model = tmp_path / "model.json"
    model.write_text("{}")
    monkeypatch.setattr(b, "plus_after_model", lambda config: model)
    monkeypatch.setattr(b, "make_plus_after_scorer", lambda m, config: scorer)
    return model


def test_the_plus_5s_figure_uses_the_rescored_scores_and_its_own_colors(monkeypatch, tmp_path):
    got = []

    def scorer(df):
        got.append(df)
        return np.where(df["trial_label"].str.endswith("__odor2"), 5, 1)

    _with_fake_scorer(monkeypatch, tmp_path, scorer)
    argv, out, _ = _fixture(tmp_path)
    assert b.main(argv) == "built"
    fig = out / "IAA-first" / "IAA-first_scores_pooled_odor_plus_5s"
    assert fig.with_suffix(".png").is_file() and fig.with_suffix(".pdf").is_file()
    t = pd.read_csv(fig.with_suffix(".csv"))
    assert t.set_index("condition")["mean_score"].to_dict() == {
        b.PLUS5_LABELS["pre"]: 1.0, b.PLUS5_LABELS["pair"]: 5.0}
    rows = pd.concat(got)
    o2 = rows[rows.trial_label == PAIR + "__odor2"].iloc[0]
    assert o2["trial_odor_on_s"] == 30.0 and o2["trial_odor_off_s"] == 40.0
    assert not rows.trial_label.eq(PAIR).any()            # odor 1 is never rescored
    s = pd.read_csv(out / "IAA-first" / "IAA-first_odor_plus_5s_scores.csv")
    assert set(s.trial_label) == {"pretest_2_linalool-1", PAIR + "__odor2"}
    assert set(s.score_window) == {"odor + 5 s after"}
    others = set(b.BAR_COLORS.values()) | set(b.GROUP_COLORS.values())
    assert set(b.PLUS5_COLORS.values()).isdisjoint(others)


def test_a_new_model_redraws_the_folder(monkeypatch, tmp_path):
    model = _with_fake_scorer(monkeypatch, tmp_path, lambda df: np.ones(len(df), dtype=int))
    argv, _, _ = _fixture(tmp_path)
    assert b.main(argv) == "built"
    assert b.main(argv) == "kept"
    model.write_text('{"new": 1}')
    assert b.main(argv) == "built"


def test_no_model_or_a_failing_scorer_skips_only_the_plus_5s_figure(monkeypatch, tmp_path):
    argv, out, _ = _fixture(tmp_path)                     # the fixture config has no model
    assert b.main(argv) == "built"
    assert not (out / "IAA-first" / "IAA-first_scores_pooled_odor_plus_5s.png").exists()

    def broken(df):
        raise RuntimeError("no scorer")

    (tmp_path / "m").mkdir()
    _with_fake_scorer(monkeypatch, tmp_path / "m", broken)
    assert b.main(argv + ["--force"]) == "built"
    assert (out / "IAA-first" / "IAA-first_scores_pooled.png").is_file()
    assert not (out / "IAA-first" / "IAA-first_scores_pooled_odor_plus_5s.png").exists()


# ── Pre-Exposure-Baseline and Time-of-Day sets (user 2026-11-17) ──────────────
from scripts.analysis import b2b_baseline_time_of_day as bt  # noqa: E402

RIGS = ("november_13_batch_1", "november_13_batch_1_rig_2", "november_13_batch_1_rig_3")


def _rigs_fixture(tmp_path):
    """Rig 1 starts 9:45, rig 2 12:10, rig 3 17:15; every batch starved 1 h."""
    argv, out, _ = _fixture(tmp_path, flies=RIGS,
                            starts=dict(zip(RIGS, ("094500", "121000", "171500"))))
    for fly in RIGS:
        (tmp_path / "data" / b.FIRST / fly / "session_metadata.txt").write_text(
            "Target starvation duration (hours): 1.0\n")
    return argv, out


def test_rig_of_a_batch_folder():
    assert [bt.rig_of(f) for f in RIGS] == [1, 2, 3]


def test_pre_exposure_baseline_is_the_window_flies_naive_pretest_by_group_and_rig(tmp_path):
    argv, out = _rigs_fixture(tmp_path)
    assert b.main(argv + ["--start", "09:40", "--end", "16:20"]) == "built"
    d = out / "Pre-Exposure-Baseline"
    assert (d / "pre_exposure_baseline.png").is_file()
    assert set(pd.read_csv(d / "pre_exposure_by_rig.csv").rig) == {1, 2}   # rig 3: 17:15
    coh = pd.read_csv(d / "pre_exposure_by_cohort.csv")
    assert set(coh.cohort) == {"IAA-first"}
    lin = coh[coh.odor == "Linalool"].iloc[0]
    assert (lin.n_flies, lin.mean_score_pre, lin.pct_responding) == (2, 0.0, 0.0)
    assert (d / "pre_exposure_stats.csv").is_file()


def test_time_of_day_takes_every_fly_whatever_the_window(tmp_path):
    argv, _ = _rigs_fixture(tmp_path)
    eve = tmp_path / "eve"
    assert b.main(_out(argv, eve) + ["--start", "21:30", "--end", "23:00"]) == "empty"
    d = eve / "Time-of-Day_2h-bins_whole-dataset" / b.FIRST
    assert (d / "time_of_day_pretest.png").is_file()
    assert (d / "time_of_day_test_block.png").is_file()
    pooled = pd.read_csv(d / "pretest_pooled_by_bin.csv")
    assert list(pooled.bin_label) == ["8:00–10:00", "12:00–14:00", "16:00–18:00"]
    tb = pd.read_csv(d / "test_block_by_bin.csv")
    assert list(tb[tb.role == "pair"].mean_score) == [4.0, 4.0, 4.0]
    assert list(tb[tb.role == "naive"].mean_score) == [0.0, 0.0, 0.0]
    assert list(tb[tb.role == "pair"].n_flies) == [1, 1, 1]
    comp = pd.read_csv(d / "flies_per_bin_by_starvation.csv")
    assert "1.0" in comp.columns                          # from session_metadata.txt
    assert (d / "test_block_stats.csv").is_file() and (d / "pretest_stats.csv").is_file()
    assert not (eve / "Pre-Exposure-Baseline").exists()   # no fly in this window
    assert b.main(_out(argv, eve) + ["--start", "21:30", "--end", "23:00"]) == "kept"


def test_test_block_table_pools_trials_and_takes_sem_over_flies():
    recs = pd.DataFrame([
        dict(dataset=b.FIRST, fly="f", fly_number=n, role=role, score=s)
        for n, role, s in ((1, "pre", 0), (1, "post", 4), (1, "post", 2),
                           (2, "pre", 1), (2, "post", 0))])
    start = {(b.FIRST, "f"): pd.Timestamp("2026-11-13 09:45")}
    t = bt.test_block_table(recs, start, width=2.0, threshold=2)
    pair = t[t.role == "pair"].iloc[0]
    assert pair.n_trials == 3 and pair.n_flies == 2
    assert pair.mean_score == pytest.approx(2.0)                # 6 / 3 trials, not (3 + 0) / 2
    assert pair.pct_responding == pytest.approx(200 / 3)
    assert pair.sem_score_flies == pytest.approx(np.std([3.0, 0.0], ddof=1) / np.sqrt(2))


def test_plus_5s_figures_for_both_groups_pooled_and_not(tmp_path):
    pre = pd.DataFrame([_row(b.SECOND, "pretest_8_isoamylacetate-1"),
                        _row(b.FIRST, "pretest_2_linalool-1")])
    pairs = pd.DataFrame([_row(b.SECOND, "testing_5_citral-1_then_isoamylacetate-1"),
                          _row(b.SECOND, "testing_12_citral-1_then_isoamylacetate-1"),
                          _row(b.FIRST, PAIR), _row(b.FIRST, PAIR.replace("_3_", "_11_"))])

    def scorer(df):   # 1st presentation 2, 2nd 4, alone 0
        lab = df["trial_label"].astype(str)
        return np.where(lab.str.contains("_12_|_11_"), 4, np.where(lab.str.endswith("__odor2"), 2, 0))

    written = b.plus_after_figure(tmp_path, pre, pairs, {}, scorer, lambda r: 41.0,
                                  lambda r: (41.0, 51.0), threshold=2, started="")
    names = {p.name for p in written}
    for short in ("IAA-first", "IAA-second"):
        for kind in ("scores", "scores_pooled"):
            assert f"{short}_{kind}_odor_plus_5s.png" in names
        assert f"{short}_odor_plus_5s_scores.csv" in names
    t = pd.read_csv(tmp_path / "IAA-second" / "IAA-second_scores_odor_plus_5s.csv")
    assert t.set_index("condition")["mean_score"].to_dict() == {
        b.PLUS5_LABELS["pre"]: 0.0, b.PLUS5_LABELS[1]: 2.0, b.PLUS5_LABELS[2]: 4.0}
    assert set(t.partner) == {"Citral"}
    assert len(set(b.PLUS5_COLORS.values())) == 3       # 1st and pooled share magenta


def test_pre_exposure_baseline_has_a_mean_score_version(tmp_path):
    argv, out = _rigs_fixture(tmp_path)
    assert b.main(argv + ["--start", "09:40", "--end", "16:20"]) == "built"
    assert (out / "Pre-Exposure-Baseline" / "pre_exposure_baseline_score.png").is_file()


def test_baseline_score_grid_is_the_mean_score_per_group_and_rig():
    long = pd.DataFrame({"cohort": ["IAA-first"] * 3, "odor": ["Linalool"] * 3,
                         "rig": [1, 1, 2], "fly_id": ["a", "b", "c"],
                         "score_pre": [0.0, 2.0, 5.0], "responded_pre": [False, True, True]})
    vals, notes, cols = bt.score_grid(long, ["IAA-first"])
    assert cols == ["Rig 1", "Rig 2", "All rigs"]
    assert vals.tolist() == [[1.0, 5.0, pytest.approx(7 / 3)]]
    assert notes[0][0] == "1.00\n2 flies"


# ── each odor's change minus the Paraffin Oil control's change (user 2026-11-18) ──
def _control_recs(missing_second=False):
    """4 flies. Linalool: alone 0, in pair 2 then 3 (change 2.5). Paraffin Oil: alone 1,
    in pair 1 then 2 (change 0.5). ``missing_second``: fly f3 lost its 2nd Linalool trial."""
    recs = []
    for fly in range(4):
        for partner, pre, post in (("Linalool", 0, (2, 3)), ("Paraffin Oil", 1, (1, 2))):
            recs.append(dict(dataset=b.FIRST, fly=f"f{fly}", fly_number=1, role="pre", round=0,
                             partner=partner, score=pre))
            for rnd, s in zip((1, 2), post):
                if missing_second and fly == 3 and partner == "Linalool" and rnd == 2:
                    s = None
                recs.append(dict(dataset=b.FIRST, fly=f"f{fly}", fly_number=1, role="post",
                                 round=rnd, partner=partner, score=s))
    return pd.DataFrame(recs)


def _ctl_row(t, panel, key, partner):
    return t[(t.panel == panel) & (t.key.astype(str) == str(key)) & (t.partner == partner)].iloc[0]


def test_change_minus_control_is_per_fly_for_the_test_and_pooled_trials_for_the_bar():
    t = b.control_change_table(_control_recs(missing_second=True), ("pair",), threshold=2)
    lin = _ctl_row(t, "score", "pair", "Linalool")
    # bar: trial sums / trial counts; Linalool's 7 pair trials = 2,3,2,3,2,3,2
    assert lin["odor_change"] == pytest.approx(17 / 7 - 0)
    assert lin["control_change"] == pytest.approx(12 / 8 - 1)
    assert lin["value"] == pytest.approx(17 / 7 - 0.5)
    # test: per fly (2.5 - 0.5) = 2 for f0-f2, (2 - 0.5) = 1.5 for f3
    assert lin["n_flies"] == 4 and lin["mean_per_fly"] == pytest.approx(7.5 / 4)
    assert lin["p"] == pytest.approx(b.sign_flip_p([2, 2, 2, 1.5]))
    assert lin["kind"] == "odor minus control"
    # % panel, threshold 2: Linalool 0 -> 100 %, Paraffin Oil 0 -> 50 %: +50 points
    assert _ctl_row(t, "pct", "pair", "Linalool")["value"] == pytest.approx(50.0)


def test_paraffin_oil_is_the_control_not_a_compared_odor():
    t = b.control_change_table(_control_recs(), ("pair",), threshold=2)
    ctl = t[t.partner == b.CONTROL_ODOR]
    assert set(ctl["kind"]) == {"control's own change"} and len(ctl) == 2      # one per panel
    row = _ctl_row(t, "score", "pair", b.CONTROL_ODOR)
    assert row["value"] == pytest.approx(0.5) and row["p"] == pytest.approx(2 / 16)
    # Holm across every bar in a panel: Linalool and the control
    sc = t[t.panel == "score"]
    assert sc["p_holm"].tolist() == pytest.approx(b.holm(sc["p"].tolist()))


def test_each_presentation_is_compared_with_the_same_presentation_of_paraffin_oil():
    t = b.control_change_table(_control_recs(), (1, 2), threshold=2)
    assert _ctl_row(t, "score", 1, "Linalool")["value"] == pytest.approx((2 - 0) - (1 - 1))
    assert _ctl_row(t, "score", 2, "Linalool")["value"] == pytest.approx((3 - 0) - (2 - 1))
    assert len(t[t.panel == "score"]) == 4                 # 2 presentations x (odor + control)


def test_iaa_second_compares_iaa_after_each_odor_with_iaa_after_paraffin_oil():
    pre = pd.DataFrame([_row(b.SECOND, "pretest_8_isoamylacetate-1")])
    pairs = pd.DataFrame([_row(b.SECOND, "testing_5_citral-1_then_isoamylacetate-1"),
                          _row(b.SECOND, "testing_9_paraffinoil-1_then_isoamylacetate-1")])
    scores = {(b.SECOND, "b1", 1, "pretest_8_isoamylacetate-1"): 1,
              (b.SECOND, "b1", 1, "testing_5_citral-1_then_isoamylacetate-1__odor2"): 4,
              (b.SECOND, "b1", 1, "testing_9_paraffinoil-1_then_isoamylacetate-1__odor2"): 2}
    recs = b.build_records(pre, pairs, scores, lambda r: 41.0)
    t = b.control_change_table(recs, ("pair",), threshold=2)
    cit = _ctl_row(t, "score", "pair", "Citral")
    # (IAA after Citral - IAA alone) - (IAA after Paraffin Oil - IAA alone) = 4 - 2
    assert (cit["odor_change"], cit["control_change"], cit["value"]) == (3.0, 1.0, 2.0)


def test_control_figure_writes_png_pdf_csv_and_skips_without_paraffin_oil(tmp_path):
    paths = b.control_bars(_control_recs(), "t", tmp_path / "x_change_minus_paraffin.png",
                           threshold=2, pool_presentations=True)
    assert {p.name for p in paths} == {"x_change_minus_paraffin.png", "x_change_minus_paraffin.pdf",
                                       "x_change_minus_paraffin.csv"}
    t = pd.read_csv(tmp_path / "x_change_minus_paraffin.csv")
    assert {"value", "odor_change", "control_change", "p", "p_holm", "stars"} <= set(t.columns)
    recs = _control_recs()
    assert b.control_bars(recs[recs.partner != b.CONTROL_ODOR], "t", tmp_path / "y.png",
                          threshold=2, pool_presentations=True) == []


def test_build_draws_the_minus_paraffin_figures_for_both_windows_and_presentations(tmp_path):
    pre = pd.DataFrame([_row(b.FIRST, "pretest_2_linalool-1"), _row(b.FIRST, "pretest_5_paraffinoil-1")])
    pairs = pd.DataFrame([_row(b.FIRST, PAIR), _row(b.FIRST, PAIR.replace("_3_", "_11_")),
                          _row(b.FIRST, "testing_4_isoamylacetate-1_then_paraffinoil-1"),
                          _row(b.FIRST, "testing_12_isoamylacetate-1_then_paraffinoil-1")])
    pred = pd.DataFrame([dict(dataset=b.FIRST, fly="b1", fly_number=1, trial_label=lab, score=s)
                         for lab, s in (("pretest_2_linalool-1", 0), ("pretest_5_paraffinoil-1", 1),
                                        (PAIR + "__odor2", 3), (PAIR.replace("_3_", "_11_") + "__odor2", 2),
                                        ("testing_4_isoamylacetate-1_then_paraffinoil-1__odor2", 1),
                                        ("testing_12_isoamylacetate-1_then_paraffinoil-1__odor2", 2),
                                        *((lab, 1) for lab in pairs.trial_label))])   # odor 1
    written = b.build(tmp_path, pre, pairs, pred, 2.0, [], odor2_window=lambda r: (41.0, 51.0),
                      plus5_scorer=lambda df: np.zeros(len(df), dtype=int))
    names = {p.name for p in written}
    for tail in ("", "_odor_plus_5s"):
        for kind in ("change_minus_paraffin", "change_minus_paraffin_pooled"):
            assert f"IAA-first_{kind}{tail}.png" in names, (kind, tail)


# ── pre-specified primary endpoint (user 2026-11-18, after the stats audit) ──────
def test_primary_endpoint_is_pinned():
    """Registered 2026-11-18. Changing any of it is a new registration, not an edit."""
    assert b.PRIMARY_ENDPOINT == {
        "registered": "2026-11-18",
        "metric": "responded (score >= 2), % of trials",
        "window": "production scorer window (10 s odor; model_predictions.csv)",
        "partners": "the 7 odor partners: IAA, Hexanol, Linalool, Benzaldehyde, "
                    "Ethyl Butyrate, Citral, 3-Octanol",
        "control": "the Paraffin Oil partner: a master-valve click in the same odor-2 slot",
        "IAA-first": "per fly: mean over the 7 partners of (partner in pair - partner naive) "
                     "minus (click in pair - click naive)",
        "IAA-second": "per fly: mean over the 7 partners of (IAA after the partner - IAA "
                      "naive) minus (IAA after the click - IAA naive)",
        "contrasts": "naive -> round 1; naive -> round 2; round 1 -> round 2; "
                     "naive -> rounds 1 + 2 pooled",
        "test": "two-sided sign-flip, every fly of a batch flipped together (headline), and "
                "across flies; Holm over the 4 contrasts of each dataset",
        "ci": "95 % CI inverting the batch-level sign-flip",
    }
    assert b.PRIMARY_PARTNERS == ("IAA", "Hexanol", "Linalool", "Benzaldehyde",
                                  "Ethyl Butyrate", "Citral", "3-Octanol")


def _primary_recs(ds=None, n_flies=4, odor=(0, 1, 1), click=(0, 0, 1)):
    """``n_flies`` flies, one per batch. Every odor partner: naive, round 1, round 2 =
    ``odor`` (scores; threshold 2 -> responded when 2+); the click: ``click``."""
    ds = ds or b.FIRST
    recs = []
    for f in range(n_flies):
        for partner in ("Linalool", "Hexanol", "Paraffin Oil"):
            vals = click if partner == "Paraffin Oil" else odor
            for rnd, v in zip((0, 1, 2), vals):
                recs.append(dict(dataset=ds, fly=f"nov_{f}_batch_1", fly_number=1,
                                 role="pre" if rnd == 0 else "post", round=rnd, partner=partner,
                                 odor=partner, label=f"t{rnd}_{partner}", score=3 * v))
    return pd.DataFrame(recs)


def test_primary_contrasts_are_partner_change_minus_click_change():
    t = b.primary_table(_primary_recs(), threshold=2).set_index("contrast")
    # odors: naive 0, r1 1, r2 1; click: naive 0, r1 0, r2 1 (responded, x 100)
    assert t.loc["naive -> round 1", "estimate"] == pytest.approx(100.0)
    assert t.loc["naive -> round 2", "estimate"] == pytest.approx(0.0)
    assert t.loc["round 1 -> round 2", "estimate"] == pytest.approx(-100.0)
    assert t.loc["naive -> rounds 1 + 2 pooled", "estimate"] == pytest.approx(50.0)
    r = t.loc["naive -> round 1"]
    assert (r["odor_change"], r["click_change"]) == (100.0, 0.0)
    assert r["n_flies"] == 4 and r["n_batches"] == 4
    assert r["p_fly"] == pytest.approx(2 / 16) and r["p_batch"] == pytest.approx(2 / 16)
    # Holm over the dataset's 4 contrasts
    assert t["p_batch_holm"].tolist() == pytest.approx(b.holm(t["p_batch"].tolist()))


def test_primary_pairsecond_naive_iaa_cancels():
    recs = _primary_recs(b.SECOND, odor=(1, 1, 0), click=(1, 0, 0))
    t = b.primary_table(recs, threshold=2).set_index("contrast")
    # IAA after odors (r1 1, r2 0) minus IAA after the click (r1 0, r2 0)
    assert t.loc["naive -> round 1", "estimate"] == pytest.approx(100.0)
    assert t.loc["naive -> round 2", "estimate"] == pytest.approx(0.0)


def test_primary_bar_pools_trials_and_the_test_counts_flies():
    recs = _primary_recs(n_flies=4)
    drop = (recs.fly == "nov_3_batch_1") & (recs.partner == "Hexanol") & (recs["round"] == 1)
    t = b.primary_table(recs[~drop], threshold=2).set_index("contrast")
    r = t.loc["naive -> round 1"]
    # 7 of 8 odor cells left: pooled odor change = 7/7 - 0; per-fly values all 1 still
    assert r["odor_change"] == pytest.approx(100.0) and r["n_flies"] == 4


def test_carryover_sensitivity_drops_pair_trials_whose_odor1_responded():
    recs = _primary_recs(b.SECOND, odor=(0, 1, 1), click=(0, 0, 0))
    odor1 = {(r.dataset, r.fly, r.fly_number, r.label): (3 if r.partner == "Linalool" else 0)
             for r in recs.itertuples() if r.role == "post"}
    t = b.primary_table(recs, threshold=2, odor1_scores=odor1, drop_odor1_responders=True)
    t = t.set_index("contrast")
    # Linalool's pair trials go (its odor 1 responded); Hexanol's stay
    assert t.loc["naive -> round 1", "estimate"] == pytest.approx(100.0)
    assert set(t["analysis"]) == {"odor-1 responders excluded"}


def test_build_writes_the_primary_endpoint_folder(tmp_path):
    recs = _primary_recs()
    paths = b.write_primary_endpoint(tmp_path, recs, threshold=2, odor1_scores={}, started="")
    names = {p.name for p in paths}
    assert {"primary_endpoint.csv", "primary_endpoint_IAA-first.png"} <= names
    readme = b.primary_readme(pd.read_csv(tmp_path / b.PRIMARY_DIR / "primary_endpoint.csv"))
    assert readme[0].startswith("# Primary endpoint (pre-specified 2026-11-18)")
    for value in b.PRIMARY_ENDPOINT.values():
        assert any(value in line for line in readme)


def test_controlled_bars_carry_a_batch_level_p_and_ci():
    """Audit 2026-11-17: every controlled bar gets the batch-level p (headline, stars via
    Holm over the panel) and the CI inverting it; the fly-level p stays in the CSV."""
    t = b.control_change_table(_control_recs(), ("pair",), threshold=2)
    lin = _ctl_row(t, "score", "pair", "Linalool")
    assert lin["p_batch"] == pytest.approx(2 / 16) and lin["n_batches"] == 4
    assert np.isnan(lin["ci_lo"])                 # 4 batches: min p 0.125, the CI stays open
    sc = t[t.panel == "score"]
    assert sc["p_batch_holm"].tolist() == pytest.approx(b.holm(sc["p_batch"].tolist()))
    assert (sc["stars"] == sc["p_batch_holm"].map(b.stars)).all()


def test_folder_bh_counts_each_table_s_headline_p_once(tmp_path):
    a = tmp_path / "a_stats.csv"
    pd.DataFrame({"p_batch": [0.01, 0.5], "p": [0.001, 0.4]}).to_csv(a, index=False)
    c = tmp_path / "c.csv"
    pd.DataFrame({"analysis": [b.PRIMARY_MAIN, b.PRIMARY_SENS], "p_batch": [0.02, 0.03]}).to_csv(
        c, index=False)
    d = tmp_path / "per_trial.csv"
    pd.DataFrame({"score": [1, 2]}).to_csv(d, index=False)
    t = b.folder_tests(tmp_path, [a, c, d])
    assert sorted(t["p"]) == [0.01, 0.02, 0.5]          # batch p, not the fly twin; no sens rows
    assert t["q_bh"].tolist() == pytest.approx(list(b.fs.bh_adjust(t["p"].to_numpy())))
    assert "3 tests in 2 tables" in "\n".join(b.multiplicity_lines(t))


def test_readme_opens_with_the_primary_endpoint_and_guides_every_set(tmp_path):
    primary = b.primary_table(_primary_recs(), threshold=2)
    starts = pd.DataFrame({"dataset": [b.FIRST], "folder": ["nov_0_batch_1"],
                           "first_pretest": [pd.Timestamp("2026-11-13 11:48")]})
    path = b.write_readme(tmp_path, starts, pd.Series([True]), "9:40am-4:20pm", primary,
                          b.FIGURE_GUIDE + b.DATA_NOTES)
    text = path.read_text()
    assert text.startswith("# Primary endpoint (pre-specified 2026-11-18)")
    for needle in ("## What is here", "NOT controlled for time in the rig",
                   "Odor 2 is scored against the baseline before odor 1",
                   "## Known data issues", "Change-Minus-Click/"):
        assert needle in text, needle
