"""src/per_analysis/plots.py: shared plot helpers that enforce the figure standards."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from per_analysis import plots as pl  # noqa: E402
from per_analysis import stats as st  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_registry_and_figures():
    pl.reset_ylims()
    yield
    pl.reset_ylims()
    plt.close("all")


@pytest.fixture
def trials():
    rng = np.random.default_rng(0)
    rows = []
    for f in range(10):
        for odor in ("Hexanol", "IAA"):                  # deliberately not in ODOR_ORDER order
            for cond, base in (("alone", 0.3), ("pair", 0.7)):
                for _ in range(3):
                    r = rng.random() < base
                    rows.append({"fly_id": f"f{f}", "batch": f"b{f // 3}", "odor": odor,
                                 "condition": cond, "responded": r,
                                 "score": int(rng.integers(2, 6) if r else rng.integers(-1, 2))})
    return pd.DataFrame(rows)


COLORS = {"alone": "#52514e", "pair": "#2a78d6"}
ORDER = ["IAA", "Hexanol", "3-Octanol"]


def _bars(ax, trials, metric="rate", **kw):
    table = st.group_summary(trials, by=["odor", "condition"], metric=metric)
    pf = st.per_fly(trials, "responded" if metric == "rate" else "score",
                    by=["odor", "condition"])
    return pl.paired_bars(ax, table, conditions=["alone", "pair"], metric=metric,
                          per_fly=pf, colors=COLORS, order=ORDER, **kw)


# ── save_figure ──────────────────────────────────────────────────────────────
def test_save_figure_writes_png_svg_csv(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    plotted = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    paths = pl.save_figure(fig, plotted, tmp_path / "new" / "deeper" / "fig_a")
    for ext in ("png", "svg", "csv"):
        p = tmp_path / "new" / "deeper" / f"fig_a.{ext}"
        assert p.exists() and p.stat().st_size > 0 and p in paths.values()
    back = pd.read_csv(tmp_path / "new" / "deeper" / "fig_a.csv")
    pd.testing.assert_frame_equal(back, plotted)
    from PIL import Image
    with Image.open(tmp_path / "new" / "deeper" / "fig_a.png") as im:
        assert round(im.info["dpi"][0]) == 300


# ── paired_bars ──────────────────────────────────────────────────────────────
def test_paired_bars_rate_plots_pooled_values_and_fly_lines(trials):
    fig, ax = plt.subplots()
    plotted = _bars(ax, trials)
    bars = plotted[plotted.layer == "bar"]
    assert len(bars) == 4
    # groups follow the configured odor order, not the table order
    assert bars.sort_values("x")["odor"].tolist()[::2] == ["IAA", "Hexanol"]
    iaa_pair = trials[(trials.odor == "IAA") & (trials.condition == "pair")]
    row = bars[(bars.odor == "IAA") & (bars.condition == "pair")].iloc[0]
    assert row["value"] == pytest.approx(100 * iaa_pair.responded.mean())   # pooled trials
    assert row["err_lo"] <= row["value"] <= row["err_hi"]
    flies = plotted[plotted.layer == "fly"]
    assert len(flies) == 10 * 2 * 2 and flies["value"].max() <= 100          # per-fly in %
    assert ax.get_legend() is not None                                       # 2 series
    assert ax.get_ylim()[0] == 0


def test_paired_bars_score_uses_sem_and_registered_limits(trials):
    pl.register_ylim("score", "exp9", (-1, 4))
    fig, ax = plt.subplots()
    plotted = _bars(ax, trials, metric="score", experiment="exp9")
    bars = plotted[plotted.layer == "bar"]
    assert (bars["interval"].str.contains("SEM")).all()
    assert ax.get_ylim()[0] == -1 and ax.get_ylim()[1] >= 4


def test_single_condition_has_no_legend(trials):
    fig, ax = plt.subplots()
    table = st.group_summary(trials[trials.condition == "pair"], by=["odor", "condition"],
                             metric="rate")
    pl.paired_bars(ax, table, conditions=["pair"], metric="rate", colors=COLORS, order=ORDER)
    assert ax.get_legend() is None


def test_text_wears_ink_never_series_colour(trials):
    fig, ax = plt.subplots()
    plotted = _bars(ax, trials)
    pl.title_block(fig, "Pairing raises PER", n_flies=10, n_trials=120,
                   test="paired sign-flip", correction="Holm")
    pl.caption(fig)
    texts = list(fig.texts) + list(ax.texts) + ax.get_xticklabels() + ax.get_yticklabels()
    legend = ax.get_legend()
    texts += legend.get_texts()
    series = {mcolors.to_hex(c) for c in COLORS.values()} - {pl.INK_SECONDARY}
    for t in texts:
        assert mcolors.to_hex(t.get_color()) not in series, t.get_text()
    assert len(plotted)


# ── stars ────────────────────────────────────────────────────────────────────
def test_stars_come_only_from_holm(trials):
    fig, ax = plt.subplots()
    plotted = _bars(ax, trials)
    with pytest.raises(ValueError):
        pl.add_holm_stars(ax, pd.DataFrame({"x1": [0], "x2": [1], "p": [0.001]}))
    comps = pd.DataFrame({"group": ["IAA", "Hexanol"], "cond_a": ["alone", "alone"],
                          "cond_b": ["pair", "pair"], "p": [0.0001, 0.0001],
                          "p_holm": [0.004, 0.2]})
    drawn = pl.add_holm_stars(ax, comps, plotted=plotted)
    assert drawn["stars"].tolist() == ["**", ""]
    star_texts = [t.get_text() for t in ax.texts]
    assert "**" in star_texts and "***" not in star_texts and "n.s." not in star_texts


# ── title, caption, ylim registry ────────────────────────────────────────────
def test_title_block_subtitle_states_n_test_correction():
    fig, ax = plt.subplots(layout="constrained")
    pl.title_block(fig, "Title", n_flies=12, n_trials=96, test="paired sign-flip",
                   correction="Holm")
    sub = [t.get_text() for t in fig.texts]
    assert "Title" in sub
    joined = " ".join(sub)
    for token in ("12 flies", "96 trials", "paired sign-flip", "Holm"):
        assert token in joined
    with pytest.warns(UserWarning):
        pl.title_block(plt.figure(), "T", "no counts here")


def test_caption_default_mentions_holm():
    fig = plt.figure()
    pl.caption(fig)
    assert any("Holm-adjusted p < 0.05" in t.get_text() for t in fig.texts)


def test_shared_ylim_registry():
    assert pl.shared_ylim("rate", "exp1") == (0.0, 100.0)
    assert pl.shared_ylim("score", "exp1") == (-1.0, 5.0)
    lims = pl.register_ylim("auc", "exp1", data=[0.2, 3.7, 1.1])
    assert lims[0] <= 0 and lims[1] >= 3.7
    assert pl.shared_ylim("auc", "exp1") == lims
    assert pl.register_ylim("auc", "exp1", data=[0.2, 3.7, 1.1]) == lims    # same: fine
    with pytest.raises(ValueError):
        pl.register_ylim("auc", "exp1", (0, 10))                            # conflicting
    with pytest.raises(KeyError):
        pl.shared_ylim("auc", "exp2")


def test_odor_order_and_colours_from_config_or_argument():
    assert pl.odor_order(["Zeta", "Hexanol", "IAA"], order=ORDER) == ["IAA", "Hexanol", "Zeta"]
    colours = pl.odor_colors()
    assert all(mcolors.is_color_like(c) for c in colours.values())


# ── forest and power ─────────────────────────────────────────────────────────
def test_forest_plot_sorted_by_effect():
    fig, ax = plt.subplots()
    t = pd.DataFrame({"name": ["a", "b", "c"], "est": [0.5, -0.2, 1.1],
                      "lo": [0.1, -0.6, 0.4], "hi": [0.9, 0.2, 1.8]})
    plotted = pl.forest_plot(ax, t, "est", "lo", "hi", "name")
    assert plotted.sort_values("y")["est"].tolist() == [-0.2, 0.5, 1.1]


def test_power_plot_legend_for_several_deltas():
    d = np.random.default_rng(0).normal(0.5, 1.0, 30)
    tbl = st.power_curve(d, deltas=[0.3, 0.6], ns=[5, 10, 20], n_sim=100, seed=0)
    fig, ax = plt.subplots()
    plotted = pl.power_plot(ax, tbl)
    assert len(plotted) == 6 and ax.get_legend() is not None
    with pytest.raises(ValueError):                              # no generated 9th hue
        pl.power_plot(ax, pd.DataFrame({"delta": np.repeat(np.arange(9), 2),
                                        "n": np.tile([5, 10], 9), "power": 0.5}))


def test_docstring_records_palette_validation():
    doc = pl.__doc__
    assert "validate_palette" in doc and "FAIL" in doc and "PASS" in doc
