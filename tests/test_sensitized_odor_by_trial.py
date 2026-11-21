"""The sensitized odor at every trial of the session (user 2026-11-17).

Pre-exposure, the six exposure trials and post-exposure, for every window fly of
one cohort: model score and % responding, the AUC during / before and the peak
z during / before, signed and unsigned, each with every pair of trials tested.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import scripts.analysis.auc_before_during as abd
import scripts.analysis.sensitized_odor_by_trial as sob
import scripts.analysis.start_window_report as swr

FPS = 40.0
DS = "IAA-Sensitivity-1-1"
FOLDER = "november_01_batch_1"


def _trace(seed: int, during: float) -> np.ndarray:
    """90 s: noise around 0, plus ``during`` for 30-60 s (fly f, exposure n: seed 10f + 1 + n)."""
    rng = np.random.default_rng(seed)
    t = rng.normal(0.0, 1.0, int(90 * FPS))
    t[int(30 * FPS):int(60 * FPS)] += during
    return t


def _rows(phase: str, labels_values):
    recs = []
    for fly, label, during, seed in labels_values:
        tr = _trace(seed, during)
        rec = {"dataset": DS, "fly": FOLDER, "fly_number": fly, "trial_type": phase,
               "trial_label": label, "fps": FPS, "trace_len": tr.size,
               "trial_odor_on_s": 30.0, "trial_odor_off_s": 60.0}
        rec.update({f"dir_val_{i}": v for i, v in enumerate(tr)})
        recs.append(rec)
    return pd.DataFrame.from_records(recs)


def _tables():
    """Flies 1-9; fly 9 has exposure trial 2 twice (an aborted session). Eight clean
    flies: the Wilcoxon helper needs at least 6 pairs."""
    pre, train, test, preds = [], [], [], []
    for fly in range(1, 10):
        base = 10 * fly
        pre.append((fly, "pretest_1_isoamylacetate", 2.0, base))
        pre.append((fly, "pretest_2_hexanol", 50.0, base + 1))         # not the sensitized odor
        for n in range(1, 7):
            train.append((fly, f"training_{n}_isoamylacetate", 2.0 + n, base + 1 + n))
        test.append((fly, "testing_1_isoamylacetate", 9.0, base + 9))
        preds += [(DS, FOLDER, fly, "pretest_1_isoamylacetate", 0),
                  (DS, FOLDER, fly, "pretest_2_hexanol", 5),
                  (DS, FOLDER, fly, "testing_1_isoamylacetate", 2 + (fly == 1))]
        preds += [(DS, FOLDER, fly, f"training_{n}_isoamylacetate", min(5, n))
                  for n in range(1, 7)]
    train.append((9, "training_2_isoamylacetate", 4.0, 999))
    cols = ["dataset", "fly", "fly_number", "trial_label", "score"]
    return (_rows("pretest", pre), _rows("training", train), _rows("testing", test),
            pd.DataFrame(preds, columns=cols))


def test_every_trial_of_the_sensitized_odor_gets_score_auc_and_peak_z():
    pre, train, test, preds = _tables()
    t = sob.per_trial_table(pre, train, test, preds)
    assert sob.sensitized_odor(train) == "Isoamyl Acetate"
    assert set(t["fly_id"]) == {f"{FOLDER}_fly{n}" for n in range(1, 9)}  # fly 9 dropped
    one = t[t["fly_id"] == f"{FOLDER}_fly1"].sort_values("trial_index")
    assert one["trial"].tolist() == sob.TRIALS
    assert one["score"].tolist() == [0, 1, 2, 3, 4, 5, 5, 3]
    assert one["responded"].tolist() == [False, False, True, True, True, True, True, True]
    # the same definitions as the AUC / peak-z folders, on the same trace
    want = abd.trial_auc(_trace(14, 5.0), fps=FPS, odor_on_s=30.0, odor_off_s=60.0)
    e3 = one[one["trial"] == "Exposure 3"].iloc[0]
    assert e3["ratio_unsigned"] == pytest.approx(want["ratio_unsigned"])
    assert e3["ratio_raw"] == pytest.approx(want["ratio_raw"])
    z = abd.trial_peak_z(_trace(14, 5.0), fps=FPS, odor_on_s=30.0, odor_off_s=60.0)
    assert e3["zpeak_ratio_signed"] == pytest.approx(z["zpeak_ratio_signed"])
    assert e3["zpeak_ratio_unsigned"] == pytest.approx(z["zpeak_ratio_unsigned"])
    assert (t["odor"] == "Isoamyl Acetate").all()                       # hexanol left out


def test_series_keep_only_flies_with_every_trial_and_test_every_pair():
    pre, train, test, preds = _tables()
    t = sob.per_trial_table(pre, train, test, preds)
    t.loc[(t["fly_id"] == f"{FOLDER}_fly2") & (t["trial"] == "Exposure 4"),
          "ratio_unsigned"] = np.nan
    w = sob.series(t, "ratio_unsigned")
    assert list(w.columns) == sob.TRIALS and f"{FOLDER}_fly2" not in w.index
    assert len(w) == 7
    w = sob.series(t, "zpeak_ratio_unsigned")
    s = sob.series_summary(w)
    assert s["trial"].tolist() == sob.TRIALS and (s["n_flies"] == 8).all()
    assert s.loc[0, "median"] == pytest.approx(float(np.median(w.iloc[:, 0])), abs=1e-4)
    tests = sob.series_tests(w)
    assert (tests["trial_a"] == "all").sum() == 1                          # Friedman
    pairs = tests[tests["trial_a"] != "all"]
    assert len(pairs) == 28 and pairs["p"].notna().all()
    assert (pairs["p_holm"] >= pairs["p"] - 1e-12).all()


def test_main_writes_every_figure_and_table(tmp_path):
    pre, train, test, preds = _tables()
    paths = {}
    for name, df in (("pretest", pre), ("training", train), ("testing", test)):
        paths[name] = tmp_path / f"wide_{name}.parquet"
        df.to_parquet(paths[name], index=False)
    split = preds["trial_label"].str.startswith("training")
    preds[~split].to_csv(tmp_path / "predictions.csv", index=False)
    preds[split].to_csv(tmp_path / "training_predictions.csv", index=False)
    out = tmp_path / "out"
    (out).mkdir()
    (out / "stale.png").write_text("x")
    code = sob.main(["--pretest-wide-csv", str(paths["pretest"]),
                     "--training-wide-csv", str(paths["training"]),
                     "--testing-wide-csv", str(paths["testing"]),
                     "--predictions-csv", str(tmp_path / "predictions.csv"),
                     "--training-predictions-csv", str(tmp_path / "training_predictions.csv"),
                     "--dataset", DS, "--label", "start 9:40am-4:20pm", "--out-dir", str(out)])
    assert code == 0
    names = {p.name for p in out.iterdir()}
    for stem in ("score", "auc_ratio", "peakz_ratio"):
        assert f"{DS}_{stem}_by_trial.png" in names
    assert {f"{DS}_by_trial_per_fly.csv", f"{DS}_by_trial_summary.csv",
            f"{DS}_by_trial_stats.csv"} <= names
    assert "stale.png" not in names                                        # folder is redrawn
    summary = pd.read_csv(out / f"{DS}_by_trial_summary.csv")
    assert set(summary["measure"]) == {"score", "ratio_raw", "ratio_unsigned",
                                       "zpeak_ratio_signed", "zpeak_ratio_unsigned"}


def test_the_cohort_report_runs_it_on_the_window_inputs(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(swr, "_run", lambda cmd, log: seen.setdefault(Path(str(cmd[1])).name, cmd) and 0)
    p = {k: tmp_path / k for k in ("predictions", "testing", "pretest", "training",
                                   "training_predictions")}
    swr.run_figures(tmp_path, p, DS, config=tmp_path / "c.yaml", label="start x", log=None)
    cmd = [str(c) for c in seen["sensitized_odor_by_trial.py"]]
    assert cmd[cmd.index("--training-wide-csv") + 1] == str(p["training"])
    assert cmd[cmd.index("--training-predictions-csv") + 1] == str(p["training_predictions"])
    assert cmd[cmd.index("--out-dir") + 1] == str(tmp_path / sob.OUT_DIR)


def test_only_the_unbiased_measure_keeps_a_null_line():
    nulls = {m: v[2] for m, v in sob.MEASURES.items()}
    assert nulls == {"ratio_raw": 1.0, "ratio_unsigned": None,
                     "zpeak_ratio_signed": None, "zpeak_ratio_unsigned": None}


def test_pairwise_tests_use_the_exact_tie_aware_wilcoxon():
    from fbpipe.analysis import fly_stats as fs
    base = np.array([5, 5, 5, 5, 4, 4, 2], float)
    wide = pd.DataFrame({t: base for t in sob.TRIALS})
    wide[sob.TRIALS[1]] = base - np.array([-2, -2, -2, -2, -1, -1, 1], float)
    tests = sob.series_tests(wide)
    row = tests[(tests["trial_a"] == sob.TRIALS[0]) & (tests["trial_b"] == sob.TRIALS[1])]
    assert float(row["p"].iloc[0]) == pytest.approx(round(0.0625, 6))
    fr = tests[tests["trial_a"] == "all"].iloc[0]
    assert fr["p"] == fr["p_holm"]


def test_figure_caption_marks_omnibus_exploratory_and_names_the_sham_bias(tmp_path):
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    pre, train, test, preds = _tables()
    t = sob.per_trial_table(pre, train, test, preds)
    ms = ("ratio_raw", "ratio_unsigned")
    wides = {m: sob.series(t, m) for m in ms}
    sums = {m: sob.series_summary(w) for m, w in wides.items()}
    tests = {m: sob.series_tests(w) for m, w in wides.items()}
    captured = {}
    orig = plt.close
    plt.close = lambda *a, **k: captured.setdefault("fig", plt.gcf())
    try:
        sob.plot_measures(wides, sums, tests, ms, "t", "note.", tmp_path / "x.png")
    finally:
        plt.close = orig
    text = " ".join(x.get_text() for x in captured["fig"].findobj(Text))
    assert "No fixed no-change value" in text and "exploratory" in text
    # a reference line (axhline: x spans 0..1 in axes coords) only where the null holds
    bar_raw, bar_unsigned = captured["fig"].axes[0], captured["fig"].axes[2]
    lines = [[set(l.get_ydata()) for l in ax.lines if list(l.get_xdata()) == [0, 1]]
             for ax in (bar_raw, bar_unsigned)]
    assert lines == [[{1.0}], []]
    plt.close("all")


def test_pairwise_tests_need_six_flies_like_the_score_panel():
    """The score panel of the same folder (sensitivity_window_results.trial_series_tests,
    pretest_vs_test_comparison.wilcoxon_p) leaves a pair untested below 6 non-zero
    differences; the ratio panels must not test (and Holm-count) more pairs than it."""
    rng = np.random.default_rng(0)
    five = pd.DataFrame(rng.normal(size=(5, 8)), columns=sob.TRIALS)
    pairs = sob.series_tests(five)
    pairs = pairs[pairs["trial_a"] != "all"]
    assert pairs["p"].isna().all() and pairs["p_holm"].isna().all()
    six = pd.DataFrame(rng.normal(size=(6, 8)), columns=sob.TRIALS)
    pairs = sob.series_tests(six)
    assert pairs.loc[pairs["trial_a"] != "all", "p"].notna().all()
