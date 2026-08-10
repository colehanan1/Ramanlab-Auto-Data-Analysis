"""Significance brackets on the RandomPanel concentration comparison figure.

Only *significant* pairwise comparisons get a bracket; non-significant pairs are
dropped entirely (their p-values live in the stats CSV). Surviving brackets pack
downward so there is never a floating gap where an "ns" bracket used to be.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scripts.analysis.randompanel_conc_comparison import (
    ALPHA,
    _compute_stats,
    _draw_omnibus_labels,
    _draw_significance_brackets,
    _fmt_omnibus,
    _plot_grouped,
    holm_adjust,
    significant_brackets,
)


def _row(**p):
    base = {
        "odor": "Test",
        "H_score": 1.0,
        "q_score_kw": 1.0,
        "q_score_1.0v10.0": 1.0,
        "q_score_0.1v1.0": 1.0,
        "q_score_0.1v10.0": 1.0,
    }
    base.update(p)
    return base


class TestSignificantBrackets:
    def test_drops_non_significant_pairs(self):
        row = _row(**{"q_score_1.0v10.0": 0.002, "q_score_0.1v1.0": 0.31})
        out = significant_brackets(row, "score")
        assert [(a, b) for a, b, _p, _lv in out] == [(10.0, 1.0)]

    def test_keeps_nothing_when_all_ns(self):
        assert significant_brackets(_row(), "score") == []

    def test_levels_pack_without_gaps(self):
        # Adjacent-left is ns; the two survivors must sit at levels 0 and 1,
        # not at their fixed 1/2 slots (which would leave an empty row).
        row = _row(**{"q_score_0.1v1.0": 0.01, "q_score_0.1v10.0": 0.0004})
        assert [lv for *_x, lv in significant_brackets(row, "score")] == [0, 1]

    def test_outer_pair_stacks_highest(self):
        row = _row(
            **{
                "q_score_1.0v10.0": 0.01,
                "q_score_0.1v1.0": 0.02,
                "q_score_0.1v10.0": 0.03,
            }
        )
        out = significant_brackets(row, "score")
        assert out[-1][:2] == (10.0, 0.1)
        assert [lv for *_x, lv in out] == [0, 1, 2]

    def test_alpha_boundary_is_strict(self):
        row = _row(**{"q_score_1.0v10.0": ALPHA})
        assert significant_brackets(row, "score") == []

    def test_missing_and_nan_p_values_are_skipped(self):
        row = {"odor": "Test", "q_score_0.1v1.0": np.nan}
        assert significant_brackets(row, "score") == []

    def test_measure_selects_the_right_columns(self):
        row = _row(**{"q_score_1.0v10.0": 0.9})
        row["q_reaction_1.0v10.0"] = 0.001
        row["q_reaction_0.1v1.0"] = 0.9
        row["q_reaction_0.1v10.0"] = 0.9
        assert significant_brackets(row, "score") == []
        assert len(significant_brackets(row, "reaction")) == 1


class TestDrawBrackets:
    def _ax(self):
        fig, ax = plt.subplots()
        return fig, ax

    def test_only_significant_labels_are_drawn(self):
        fig, ax = self._ax()
        stats_by_odor = {
            "A": _row(**{"q_score_1.0v10.0": 0.002}),
            "B": _row(),
        }
        _draw_significance_brackets(
            ax,
            odors=["A", "B"],
            stats_by_odor=stats_by_odor,
            group_tops={"A": 1.0, "B": 1.0},
            bar_x={
                "A": {10.0: 0.0, 1.0: 0.25, 0.1: 0.5},
                "B": {10.0: 1.0, 1.0: 1.25, 0.1: 1.5},
            },
            measure="score",
            ref=3.0,
        )
        labels = [t.get_text() for t in ax.texts]
        assert labels == ["** p=0.002"]
        assert not any("ns" in lbl for lbl in labels)
        plt.close(fig)

    def test_returns_top_of_the_tallest_bracket_stack(self):
        fig, ax = self._ax()
        stats_by_odor = {
            "A": _row(**{"q_score_1.0v10.0": 0.002, "q_score_0.1v10.0": 0.004}),
        }
        top = _draw_significance_brackets(
            ax,
            odors=["A"],
            stats_by_odor=stats_by_odor,
            group_tops={"A": 1.0},
            bar_x={"A": {10.0: 0.0, 1.0: 0.25, 0.1: 0.5}},
            measure="score",
            ref=3.0,
        )
        assert top > 1.0
        plt.close(fig)

    def test_no_significant_pairs_leaves_axes_clean(self):
        fig, ax = self._ax()
        top = _draw_significance_brackets(
            ax,
            odors=["A"],
            stats_by_odor={"A": _row()},
            group_tops={"A": 1.0},
            bar_x={"A": {10.0: 0.0, 1.0: 0.25, 0.1: 0.5}},
            measure="score",
            ref=3.0,
        )
        assert list(ax.texts) == []
        assert list(ax.lines) == []
        assert top == pytest.approx(1.0)
        plt.close(fig)


class TestHolm:
    def test_matches_step_down_definition(self):
        # p = .01, .02, .03 with m=3 -> .03, .04, .04 (monotone, capped at 1)
        assert holm_adjust([0.01, 0.02, 0.03]) == pytest.approx([0.03, 0.04, 0.04])

    def test_is_monotone_and_capped(self):
        out = holm_adjust([0.5, 0.6, 0.7])
        assert list(out) == sorted(out)
        assert max(out) <= 1.0

    def test_nans_pass_through_and_do_not_count_toward_m(self):
        out = holm_adjust([0.01, np.nan, 0.02])
        assert np.isnan(out[1])
        assert out[0] == pytest.approx(0.02)  # m = 2, not 3


def _panel(effects: dict[str, tuple[float, float, float]], n: int = 20) -> pd.DataFrame:
    """Per-fly frame: ``effects[odor] = (rate@0.1, rate@1, rate@10)``."""
    recs = []
    for odor, rates in effects.items():
        for conc, rate in zip([0.1, 1.0, 10.0], rates):
            k = int(round(rate * n))
            for i in range(n):
                reacted = 1.0 if i < k else 0.0
                recs.append(
                    {
                        "odor": odor,
                        "conc": conc,
                        "fly": f"fly{i}",
                        "fly_number": i,
                        "score": 3.0 if reacted else 0.0,
                        "reacted": reacted,
                        "n_trials": 2,
                    }
                )
    return pd.DataFrame(recs)


class TestComputeStats:
    @pytest.fixture(scope="class")
    def stats(self) -> pd.DataFrame:
        # One strong dose effect, one flat odorant.
        panel = _panel({"Strong": (0.1, 0.1, 0.9), "Flat": (0.5, 0.5, 0.55)})
        return _compute_stats(panel, n_iter=2_000, seed=0)

    def test_reports_kruskal_h_and_holm_adjusted_omnibus(self, stats):
        from scipy.stats import kruskal

        panel = _panel({"Strong": (0.1, 0.1, 0.9), "Flat": (0.5, 0.5, 0.55)})
        sub = panel[panel["odor"] == "Strong"]
        expected = kruskal(
            *[sub[sub["conc"] == c]["reacted"].to_numpy() for c in (0.1, 1.0, 10.0)]
        )
        row = stats[stats["odor"] == "Strong"].iloc[0]
        assert row["H_reaction"] == pytest.approx(expected.statistic)
        assert row["p_reaction_kw"] == pytest.approx(expected.pvalue)
        # Holm over the 2-odorant family: smallest p is doubled.
        assert row["q_reaction_kw"] >= row["p_reaction_kw"]

    def test_posthoc_runs_only_when_the_omnibus_survives(self, stats):
        strong = stats[stats["odor"] == "Strong"].iloc[0]
        flat = stats[stats["odor"] == "Flat"].iloc[0]
        assert flat["q_reaction_kw"] >= ALPHA
        assert np.isnan(flat["q_reaction_0.1v10.0"])  # unprotected -> not tested
        assert np.isfinite(strong["q_reaction_0.1v10.0"])

    def test_unprotected_odorant_gets_no_brackets(self, stats):
        flat = stats[stats["odor"] == "Flat"].iloc[0]
        assert significant_brackets(flat, "reaction") == []

    def test_fisher_cross_check_columns_are_kept(self, stats):
        for col in ("p_score_omnibus", "p_reaction_omnibus", "p_reaction_0.1v10.0"):
            assert col in stats.columns


class TestOmnibusLabels:
    def test_label_reports_h_and_holm_p(self):
        row = _row(**{"H_score": 13.84, "q_score_kw": 0.0068})
        assert _fmt_omnibus(row, "score") == "H(2)=13.8, p=0.007"

    def test_significant_omnibus_is_inked_dark_and_bold(self):
        fig, ax = plt.subplots()
        _draw_omnibus_labels(
            ax,
            odors=["A", "B"],
            stats_by_odor={
                "A": {**_row(), "H_score": 13.8, "q_score_kw": 0.007},
                "B": {**_row(), "H_score": 0.2, "q_score_kw": 1.0},
            },
            x=np.array([0.0, 1.0]),
            measure="score",
        )
        sig, ns = ax.texts[0], ax.texts[1]
        assert sig.get_fontweight() == "bold"
        assert ns.get_fontweight() == "normal"
        assert sig.get_color() != ns.get_color()
        plt.close(fig)


@pytest.fixture
def summary() -> pd.DataFrame:
    recs = []
    for odor, vals in {"Alpha": (0.5, 1.0, 2.9), "Beta": (1.0, 1.1, 1.2)}.items():
        for conc, mean in zip([0.1, 1.0, 10.0], vals):
            recs.append(
                {
                    "odor": odor,
                    "conc": conc,
                    "n": 20,
                    "mean_score": mean,
                    "sem_score": 0.3,
                    "pct_react": mean / 5.0,
                    "ci_lo": max(0.0, mean / 5.0 - 0.1),
                    "ci_hi": min(1.0, mean / 5.0 + 0.1),
                }
            )
    return pd.DataFrame(recs)


@pytest.fixture
def stats() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(**{"q_score_1.0v10.0": 0.0004, "q_score_0.1v10.0": 0.02}),
            {**_row(), "odor": "Beta"},
        ]
    ).assign(odor=["Alpha", "Beta"])


class TestFigure:
    def test_figure_shows_only_significant_brackets(self, tmp_path, summary, stats):
        png = tmp_path / "fig.png"
        fig = _plot_grouped(
            summary,
            stats,
            value="mean_score",
            err="sem_score",
            measure="score",
            ylabel="Mean Ordinal Score",
            title="Test",
            png_path=png,
            return_fig=True,
        )
        ax = fig.axes[0]
        bracket_labels = [
            t.get_text() for t in ax.texts if t.get_text().startswith(("*", "p="))
        ]
        assert sorted(bracket_labels) == ["* p=0.020", "*** p<0.001"]
        assert png.exists()
        plt.close(fig)

    def test_legend_sits_inside_the_axes_and_carries_percent_units(
        self, tmp_path, summary, stats
    ):
        fig = _plot_grouped(
            summary, stats,
            value="mean_score", err="sem_score", measure="score",
            ylabel="Mean Ordinal Score", title="Test",
            png_path=tmp_path / "fig.png", return_fig=True,
        )
        ax = fig.axes[0]
        leg = ax.get_legend()
        assert [t.get_text() for t in leg.get_texts()] == ["10%", "1%", "0.1%"]
        # Anchored to the axes box, not floated above it.
        assert leg.get_bbox_to_anchor().bounds == ax.get_window_extent().bounds
        plt.close(fig)

    def test_no_reaction_threshold_line(self, tmp_path, summary, stats):
        fig = _plot_grouped(
            summary, stats,
            value="mean_score", err="sem_score", measure="score",
            ylabel="Mean Ordinal Score", title="Test",
            png_path=tmp_path / "fig.png", return_fig=True,
        )
        ax = fig.axes[0]
        assert not any("threshold" in t.get_text() for t in ax.texts)
        assert ax.get_xlabel() == ""
        plt.close(fig)

    def test_vector_formats_are_written_for_publication(self, tmp_path, summary, stats):
        png = tmp_path / "fig.png"
        _plot_grouped(
            summary,
            stats,
            value="mean_score",
            err="sem_score",
            measure="score",
            ylabel="Mean Ordinal Score",
            title="Test",
            png_path=png,
        )
        assert png.with_suffix(".pdf").exists()
        assert png.with_suffix(".svg").exists()

    def test_svg_text_is_editable(self, tmp_path, summary, stats):
        png = tmp_path / "fig.png"
        _plot_grouped(
            summary,
            stats,
            value="mean_score",
            err="sem_score",
            measure="score",
            ylabel="Mean Ordinal Score",
            title="Test",
            png_path=png,
        )
        svg = png.with_suffix(".svg").read_text()
        # Editable text means real <text> glyphs, not vectorised paths.
        assert "Mean Ordinal Score" in svg
