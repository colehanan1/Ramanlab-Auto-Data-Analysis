"""Pre vs post model scores on three windows: odor, the time after it, both.

Built for IAA-Sensitivity5s-1-1 (5 s odor): the odor's 5 s, the 5 s after
odor-off, and the 10 s combined. Every window is rescored through
window_rescore, so the three are compared on one feature path.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import scripts.analysis.score_window_comparison as swc

DS = "IAA-Sensitivity5s-1-1"
HEX = "Hex-HexTest-Training-32-1"  # light-conditioned, and also pre-tested
WINDOWS = ["odor", "after", "odor_plus_after"]


def _template(ds=DS, odors=("isoamylacetate-1", "hexanol-0.1")):
    rows = []
    for fly in range(6):
        for phase in ("pretest", "testing"):
            for i, odor in enumerate(odors):
                rows.append({"dataset": ds, "fly": f"oct_15_batch_{fly}", "fly_number": 1,
                             "trial_label": f"{phase}_{i + 1}_{odor}", "trial_type": phase,
                             "score": 0, "prediction": 0})
    return pd.DataFrame(rows)


def _fake_rescore(template, wide_paths, model_path, spec, *, binary_threshold, rule):
    """Post IAA scores 4 on the 'after' window only; everything else 0."""
    out = template.copy()
    hit = (out["trial_type"] == "testing") & out["trial_label"].str.contains("isoamyl")
    out["score"] = np.where(hit & (spec.name != "odor"), 4, 0)
    out["prediction"] = (out["score"] >= binary_threshold).astype(int)
    out["score_window"] = spec.label
    return out


@pytest.fixture
def paired(monkeypatch, tmp_path):
    monkeypatch.setattr(swc.wr, "rescore_predictions", _fake_rescore)
    return swc.paired_by_window(
        _template(), wide_paths=[], model_path="m", rule=None, after_s=5.0,
        binary_threshold=2,
    )


def test_one_paired_table_per_window(paired):
    assert list(paired) == WINDOWS
    for frame in paired.values():
        assert set(frame["odor"]) == {"Isoamyl Acetate 1", "Hexanol 0.1"}
        assert len(frame) == 12      # 6 flies x 2 odors, paired


def test_windows_carry_their_own_scores(paired):
    iaa = {w: f.loc[f["odor"] == "Isoamyl Acetate 1", "score_post"].mean()
           for w, f in paired.items()}
    assert iaa == {"odor": 0, "after": 4, "odor_plus_after": 4}


def test_summary_is_long_by_window(paired):
    summary = swc.summarise(paired, binary_threshold=2)
    assert list(summary["window"].unique()) == WINDOWS
    row = summary.query("window == 'after' and odor == 'Isoamyl Acetate 1'").iloc[0]
    assert row["pct_pre"] == 0 and row["pct_post"] == 100 and row["n_flies"] == 6


def test_window_titles_use_the_cohorts_odor_length():
    titles = swc.window_titles(odor_s=5.0, after_s=5.0)
    assert titles == {"odor": "Odor (5 s)", "after": "5 s after odor-off",
                      "odor_plus_after": "Odor + 5 s after (10 s)"}


def test_grid_figure_has_score_and_rate_rows_for_each_window(paired):
    summary = swc.summarise(paired, binary_threshold=2)
    fig = swc.build_grid(summary, cohort=DS, titles=swc.window_titles(5.0, 5.0))
    axes = [ax for ax in fig.axes if ax.patches]
    assert len(axes) == 6
    tops = [ax.get_title(loc="left") for ax in fig.axes[:3]]
    assert tops == ["Odor (5 s)", "5 s after odor-off", "Odor + 5 s after (10 s)"]
    # Fixed axes on every panel, so windows compare by bar height alone.
    assert {fig.axes[i].get_ylim() for i in range(3)} == {(-1.0, 5.0)}
    assert {fig.axes[i].get_ylim() for i in range(3, 6)} == {(0.0, 100.0)}
    matplotlib.pyplot.close(fig)


def test_bars_are_pre_orange_post_blue(paired):
    summary = swc.summarise(paired, binary_threshold=2)
    fig = swc.build_grid(summary, cohort=DS, titles=swc.window_titles(5.0, 5.0))
    colours = {matplotlib.colors.to_hex(p.get_facecolor()) for p in fig.axes[0].patches}
    assert colours == {matplotlib.colors.to_hex("tab:orange"),
                       matplotlib.colors.to_hex("tab:blue")}
    matplotlib.pyplot.close(fig)


def test_cs_plus_figure_compares_the_windows_on_the_trained_odor(paired):
    summary = swc.summarise(paired, binary_threshold=2)
    fig = swc.build_cs_plus(summary, cohort=DS, titles=swc.window_titles(5.0, 5.0))
    assert fig is not None
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert labels == ["Odor (5 s)", "5 s after odor-off", "Odor + 5 s after (10 s)"]
    matplotlib.pyplot.close(fig)


@pytest.mark.parametrize("ds, legend, versus, cs_odor", [
    (DS, ["Pre-exposure", "Post-exposure"],
     "pre-exposure vs post-exposure", "Isoamyl Acetate 1% (v/v) (sensitized odor)"),
    (HEX, ["Pre-test (naive)", "Post-training"],
     "pre-test vs post-training", "Hexanol 0.1 (CS+)"),
])
def test_phase_and_odor_words_follow_the_cohort(monkeypatch, ds, legend, versus, cs_odor):
    """Nothing is paired with a sensitivity cohort's odor: no pre-test or CS+.

    Hex-HexTest-* is light-conditioned but runs a pre-test too, so it reaches
    these same figures and must keep the conditioning words.
    """
    monkeypatch.setattr(swc.wr, "rescore_predictions", _fake_rescore)
    paired = swc.paired_by_window(_template(ds), wide_paths=[], model_path="m", rule=None,
                                  after_s=5.0, binary_threshold=2)
    summary = swc.summarise(paired, binary_threshold=2)
    titles = swc.window_titles(5.0, 5.0)

    grid = swc.build_grid(summary, cohort=ds, titles=titles)
    assert [t.get_text() for t in grid.legends[0].get_texts()] == legend
    assert grid.get_suptitle().splitlines()[0] == f"{ds} — model score by window, {versus}"

    cs = swc.build_cs_plus(summary, cohort=ds, titles=titles)
    assert [t.get_text() for t in cs.axes[0].get_legend().get_texts()] == legend
    assert cs.get_suptitle().splitlines()[0] == f"{ds} — {cs_odor}, score by window"
    matplotlib.pyplot.close(grid)
    matplotlib.pyplot.close(cs)


@pytest.mark.parametrize("ds, odors, ticks, cs_odor", [
    # Pins tokens carry the concentration: drawn as "% (v/v)".
    (DS, ("isoamylacetate-1", "hexanol-0.1"),
     ["Hexanol 0.1% (v/v)", "Isoamyl Acetate 1% (v/v)"],
     "Isoamyl Acetate 1% (v/v) (sensitized odor)"),
    # Batches from before the pins recorded none: the name alone, no invented %.
    ("IAA-Sensitivity-40-1", ("isoamylacetate", "hexanol"),
     ["Hexanol", "Isoamyl Acetate"], "Isoamyl Acetate (sensitized odor)"),
    # Every other cohort's odor names are drawn exactly as before.
    (HEX, ("isoamylacetate-1", "hexanol-0.1"),
     ["Hexanol 0.1", "Isoamyl Acetate 1"], "Hexanol 0.1 (CS+)"),
])
def test_odor_names_write_a_sensitivity_cohorts_concentration_as_v_v(
    monkeypatch, ds, odors, ticks, cs_odor
):
    monkeypatch.setattr(swc.wr, "rescore_predictions", _fake_rescore)
    paired = swc.paired_by_window(_template(ds, odors), wide_paths=[], model_path="m",
                                  rule=None, after_s=5.0, binary_threshold=2)
    summary = swc.summarise(paired, binary_threshold=2)
    titles = swc.window_titles(5.0, 5.0)

    grid = swc.build_grid(summary, cohort=ds, titles=titles)
    bar_axes = [ax for ax in grid.axes if ax.patches]
    assert len(bar_axes) == 6
    for ax in bar_axes:
        assert [t.get_text() for t in ax.get_xticklabels()] == ticks
    cs = swc.build_cs_plus(summary, cohort=ds, titles=titles)
    assert cs.get_suptitle().splitlines()[0] == f"{ds} — {cs_odor}, score by window"
    # Display only: the summary keys (and so the CSVs) keep the old names.
    assert sorted(summary["odor"].unique()) == sorted(
        t.replace("% (v/v)", "") for t in ticks)
    matplotlib.pyplot.close(grid)
    matplotlib.pyplot.close(cs)


def test_trials_table_has_one_column_per_window(paired):
    trials = swc.trials_table(paired)
    for w in WINDOWS:
        assert {f"score_pre_{w}", f"score_post_{w}"} <= set(trials.columns)
    assert len(trials) == 12
