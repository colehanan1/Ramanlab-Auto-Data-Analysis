"""Pre/post PER across starvation groups (scripts/analysis/starvation_pre_post_comparison.py).

The rigs label trials from ~/odor_pins.txt as ``<Odor>-<conc>`` since
2026-10-01; the figure code only knows the bare rig spellings, so a 12 h and a
24 h fly would land in different odor columns unless the suffix is stripped
first. These tests pin that normalisation and the per-group statistics.
"""

from __future__ import annotations

import pandas as pd
import pytest
from scipy.stats import fisher_exact, mannwhitneyu

import scripts.analysis.starvation_pre_post_comparison as spc


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("testing_1_isoamylacetate-1", "testing_1_isoamylacetate"),
        ("pretest_4_3-Octonol-0.1", "pretest_4_3-octonol"),
        ("testing_4_3octanol-0.1", "testing_4_3-octonol"),
        ("pretest_2_Benzaldehyde-0.1", "pretest_2_benzaldehyde"),
        ("training_3_IsoamylAcetate-1", "training_3_isoamylacetate"),
        ("testing_3_ethylbutyrate-10", "testing_3_ethylbutyrate"),
        # Already in the legacy form: untouched apart from case.
        ("testing_2_linalool", "testing_2_linalool"),
        ("testing_6_3-octonol", "testing_6_3-octonol"),
        # The light probe carries no odor and no concentration.
        ("testing_9_lightonly", "testing_9_lightonly"),
        ("testing_9_LightOnly", "testing_9_lightonly"),
    ],
)
def test_normalize_trial_label(raw, expected):
    assert spc.normalize_trial_label(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("testing_1_isoamylacetate-1", "1"),
        ("pretest_4_3-Octonol-0.1", "0.1"),
        ("testing_2_linalool", None),
        ("testing_9_lightonly", None),
    ],
)
def test_label_concentration(raw, expected):
    assert spc.label_concentration(raw) == expected


def test_parse_group_spec_with_and_without_fly_prefix():
    g = spc.parse_group("24 h=IAA-Sensitivity-24-1:october_06")
    assert (g.label, g.dataset, g.fly_prefix) == ("24 h", "IAA-Sensitivity-24-1", "october_06")
    g = spc.parse_group("12 h=IAA-Sensitivity-12-1")
    assert (g.label, g.dataset, g.fly_prefix) == ("12 h", "IAA-Sensitivity-12-1", "")


def test_parse_group_rejects_missing_label():
    with pytest.raises(ValueError):
        spc.parse_group("IAA-Sensitivity-12-1")


def test_select_group_rows_matches_dataset_and_fly_prefix():
    df = pd.DataFrame({
        "dataset": ["IAA-Sensitivity-24-1"] * 3 + ["IAA-Sensitivity-12-1"],
        "fly": ["october_06_batch_2", "september_29_batch_1", "october_06_batch_2_rig_3",
                "october_06_batch_1"],
    })
    g = spc.parse_group("24 h=IAA-Sensitivity-24-1:october_06")
    out = spc.select_group_rows(df, g)
    assert list(out["fly"]) == ["october_06_batch_2", "october_06_batch_2_rig_3"]


def test_drop_flagged_excludes_state_0_and_minus_1_only(tmp_path):
    flagged = tmp_path / "flagged.csv"
    flagged.write_text(
        'dataset,fly,fly_number,"FLY-State(1, 0, -1)",comment,,,,\n'
        "IAA-Sensitivity-24-1,october_06_batch_2,1,0,bad-track,,,,\n"
        "IAA-Sensitivity-24-1,october_06_batch_2,4,-1,dead,,,,\n"
        "IAA-Sensitivity-24-1,october_06_batch_2,2,1,fine,,,,\n"
    )
    df = pd.DataFrame({
        "dataset": ["IAA-Sensitivity-24-1"] * 5,
        "fly": ["october_06_batch_2"] * 4 + ["october_06_batch_2_rig_2"],
        # Float spelling, as a parquet round-trip can produce.
        "fly_number": [1.0, 2.0, 3.0, 4.0, 1.0],
    })
    out = spc.drop_flagged(df, flagged)
    assert sorted(zip(out["fly"], out["fly_number"])) == [
        ("october_06_batch_2", 2.0), ("october_06_batch_2", 3.0),
        ("october_06_batch_2_rig_2", 1.0)]


def test_drop_flagged_without_a_table_keeps_everything(tmp_path):
    df = pd.DataFrame({"dataset": ["X"], "fly": ["f"], "fly_number": [1]})
    assert len(spc.drop_flagged(df, "")) == 1
    assert len(spc.drop_flagged(df, tmp_path / "missing.csv")) == 1


def _paired(group, scores_pre, scores_post, odor="Hexanol"):
    n = len(scores_pre)
    return pd.DataFrame({
        "group": [group] * n,
        "dataset_canon": ["X"] * n,
        "fly": [f"{group}_f{i}" for i in range(n)],
        "fly_number": [1] * n,
        "odor": [odor] * n,
        "score_pre": scores_pre,
        "score_post": scores_post,
    })


def test_group_summary_counts_flies_per_group_and_never_pools():
    paired = pd.concat([
        _paired("12 h", [0, 0, 3, 4], [2, 3, 3, 4]),
        _paired("24 h", [0, 0], [0, 5]),
    ])
    summary = spc.group_summary(paired, ["12 h", "24 h"], binary_threshold=2)
    a = summary[summary.group == "12 h"].iloc[0]
    b = summary[summary.group == "24 h"].iloc[0]
    assert a.n_flies == 4 and b.n_flies == 2
    assert a.mean_pre == pytest.approx(7 / 4)
    assert a.pct_post == pytest.approx(100.0)
    assert b.pct_pre == pytest.approx(0.0)
    assert b.pct_post == pytest.approx(50.0)


def test_between_group_stats_match_scipy():
    a = _paired("12 h", [0, 1, 3, 4, 2], [2, 3, 3, 4, 0])
    b = _paired("24 h", [0, 0, 0, 1], [0, 5, 0, 1])
    stats = spc.between_group_stats(pd.concat([a, b]), "12 h", "24 h", binary_threshold=2)
    row = stats.iloc[0]
    assert row.n_a == 5 and row.n_b == 4
    assert row.p_pre_score == pytest.approx(
        mannwhitneyu(a.score_pre, b.score_pre, alternative="two-sided").pvalue)
    assert row.p_post_score == pytest.approx(
        mannwhitneyu(a.score_post, b.score_post, alternative="two-sided").pvalue)
    delta_a = a.score_post - a.score_pre
    delta_b = b.score_post - b.score_pre
    assert row.p_delta_score == pytest.approx(
        mannwhitneyu(delta_a, delta_b, alternative="two-sided").pvalue)
    resp_pre_a = int((a.score_pre >= 2).sum())
    resp_pre_b = int((b.score_pre >= 2).sum())
    table = [[resp_pre_a, 5 - resp_pre_a], [resp_pre_b, 4 - resp_pre_b]]
    assert row.p_pre_rate == pytest.approx(fisher_exact(table).pvalue)


def test_between_group_stats_none_when_a_group_is_missing_the_odor():
    a = _paired("12 h", [0, 1], [2, 3], odor="Citral")
    b = _paired("24 h", [0, 0], [0, 5], odor="Hexanol")
    stats = spc.between_group_stats(pd.concat([a, b]), "12 h", "24 h", binary_threshold=2)
    assert stats.empty or stats["p_pre_score"].isna().all()


def test_figures_render(tmp_path):
    paired = pd.concat([
        _paired("12 h", [0, 1, 3, 4, 2, 0], [2, 3, 3, 4, 0, 1]),
        _paired("24 h", [0, 0, 0, 1, 2, 0], [0, 5, 0, 1, 3, 3]),
        _paired("12 h", [0, 1, 0, 0, 2, 0], [2, 3, 3, 4, 0, 1], odor="Isoamyl Acetate"),
        _paired("24 h", [0, 0, 0, 1, 2, 0], [0, 5, 0, 1, 3, 3], odor="Isoamyl Acetate"),
    ])
    groups = ["12 h", "24 h"]
    summary = spc.group_summary(paired, groups, binary_threshold=2)
    between = spc.between_group_stats(paired, *groups, binary_threshold=2)
    written = spc.render_bar_figures(
        paired, summary, between, groups=groups, trained_odor="Isoamyl Acetate",
        title="test", out_dir=tmp_path,
    )
    assert written and all(p.exists() and p.stat().st_size > 0 for p in written)


def test_group_summary_tests_response_rate_with_fisher_exact():
    """Pre vs post response %, per group: Fisher's exact on the 2x2 table."""
    paired = _paired("12 h", [0, 0, 0, 0, 3], [3, 3, 3, 0, 3])
    summary = spc.group_summary(paired, ["12 h"], binary_threshold=2)
    # pre 1/5 responders, post 4/5.
    assert summary["p_rate"].iloc[0] == pytest.approx(fisher_exact([[1, 4], [4, 1]]).pvalue)


def _two_group_inputs():
    paired = pd.concat([
        _paired("Oct 6", [0, 1, 3, 4, 2, 0], [2, 3, 3, 4, 0, 1]),
        _paired("Oct 7", [0, 0, 0, 1, 2, 0], [0, 5, 0, 1, 3, 3]),
    ])
    groups = ["Oct 6", "Oct 7"]
    return (paired, spc.group_summary(paired, groups, binary_threshold=2),
            spc.between_group_stats(paired, *groups, binary_threshold=2), groups)


def test_default_file_names_and_titles_are_unchanged(tmp_path):
    paired, summary, between, groups = _two_group_inputs()
    names = {p.name for p in spc.render_bar_figures(
        paired, summary, between, groups=groups, trained_odor="", title="t", out_dir=tmp_path)}
    assert {"pre_vs_post_by_starvation_score.png", "starvation_compared_within_phase_rate.png",
            "change_post_minus_pre_score.png"} <= names
    fig = spc.build_pre_post_bars(summary, groups=groups, kind="score", trained_odor="", title="t")
    assert "Oct 6 vs Oct 7 starved" in fig.axes[0].get_title(loc="left")


def test_comparison_text_and_file_tag_replace_the_starvation_wording(tmp_path):
    paired, summary, between, groups = _two_group_inputs()
    names = {p.name for p in spc.render_bar_figures(
        paired, summary, between, groups=groups, trained_odor="", title="t", out_dir=tmp_path,
        comparison="Oct 6 vs Oct 7 (24 h starved)", tag="day")}
    assert {"pre_vs_post_by_day_score.png", "day_compared_within_phase_rate.png"} <= names
    assert not any("starvation" in n for n in names)
    fig = spc.build_pre_post_bars(summary, groups=groups, kind="rate", trained_odor="", title="t",
                                  comparison="Oct 6 vs Oct 7 (24 h starved)")
    title = fig.axes[0].get_title(loc="left")
    assert "Oct 6 vs Oct 7 (24 h starved)" in title and "Oct 7 starved" not in title


def test_cli_comparison_and_file_tag_flags():
    args = spc._parse_args(["--predictions-csv", "p", "--group", "a=X", "--group", "b=Y",
                            "--out-dir", "o"])
    assert args.comparison == "" and args.file_tag == "starvation"
    args = spc._parse_args(["--predictions-csv", "p", "--group", "a=X", "--group", "b=Y",
                            "--out-dir", "o", "--comparison", "a vs b", "--file-tag", "day"])
    assert (args.comparison, args.file_tag) == ("a vs b", "day")


# ── three groups (12 h / 18 h / 24 h) ──────────────────────────────────────

THREE = ["12 h", "18 h", "24 h"]


def _three():
    return pd.concat([
        _paired("12 h", [0, 1, 0, 0, 2, 0], [3, 3, 3, 4, 2, 3], odor="Isoamyl Acetate"),
        _paired("18 h", [0, 0, 1, 1, 2, 0], [0, 1, 0, 1, 3, 3], odor="Isoamyl Acetate"),
        _paired("24 h", [2, 0, 0, 1, 2, 0], [0, 2, 0, 1, 3, 0], odor="Isoamyl Acetate"),
    ])


def test_pairwise_between_stats_covers_every_pair_in_order():
    stats = spc.pairwise_between_stats(_three(), THREE, binary_threshold=2)
    assert list(zip(stats.group_a, stats.group_b)) == [
        ("12 h", "18 h"), ("12 h", "24 h"), ("18 h", "24 h")]


def test_pairwise_between_stats_matches_the_two_group_function():
    paired = _three()
    stats = spc.pairwise_between_stats(paired, THREE, binary_threshold=2)
    direct = spc.between_group_stats(paired, "12 h", "24 h", binary_threshold=2).iloc[0]
    row = stats[(stats.group_a == "12 h") & (stats.group_b == "24 h")].iloc[0]
    assert row.p_delta_score == pytest.approx(direct.p_delta_score)


def test_omnibus_kruskal_matches_scipy():
    from scipy.stats import kruskal

    paired = _three()
    row = spc.omnibus_stats(paired, THREE, binary_threshold=2).iloc[0]
    by = [paired[paired.group == g] for g in THREE]
    assert row.p_kw_pre_score == pytest.approx(kruskal(*[b.score_pre for b in by]).pvalue)
    assert row.p_kw_delta_score == pytest.approx(
        kruskal(*[b.score_post - b.score_pre for b in by]).pvalue)
    assert 0.0 <= row.p_ffh_post_rate <= 1.0


def test_group_colors_step_light_to_dark_and_keep_the_two_group_pair():
    import matplotlib.colors as mcolors

    lum = lambda h: sum(mcolors.to_rgb(h))  # noqa: E731
    cols = spc.group_colors(3)
    for phase in (0, 1):
        assert lum(cols[0][phase]) > lum(cols[1][phase]) > lum(cols[2][phase])
    assert spc.group_colors(2) == spc.GROUP_COLORS


def test_three_group_figures_render(tmp_path):
    paired = _three()
    summary = spc.group_summary(paired, THREE, binary_threshold=2)
    between = spc.pairwise_between_stats(paired, THREE, binary_threshold=2)
    written = spc.render_bar_figures(paired, summary, between, groups=THREE,
                                     trained_odor="Isoamyl Acetate", title="t", out_dir=tmp_path)
    assert len(written) == 5 and all(p.stat().st_size > 0 for p in written)
    fig = spc.build_pre_post_bars(summary, groups=THREE, kind="score", trained_odor="", title="t")
    assert len(fig.axes[0].patches) == 6  # 3 groups x (pre, post) x 1 odor
    assert "12 h vs 18 h vs 24 h starved" in fig.axes[0].get_title(loc="left")


def test_group_count_must_be_at_least_two():
    with pytest.raises(SystemExit):
        spc.check_group_count([spc.parse_group("a=X")])
    spc.check_group_count([spc.parse_group(f"{i}=X") for i in range(3)])


# ── up to six groups, several fly prefixes per group ──────────────────────


def test_group_with_several_fly_prefixes_selects_each():
    df = pd.DataFrame({"dataset": ["D"] * 3,
                       "fly": ["october_06_b", "october_07_b", "september_29_b"]})
    g = spc.parse_group("24 h=D:october_06,october_07")
    assert list(spc.select_group_rows(df, g)["fly"]) == ["october_06_b", "october_07_b"]
    paired = df.rename(columns={"dataset": "dataset_canon"})
    assert list(spc.assign_groups(paired, [g])["fly"]) == ["october_06_b", "october_07_b"]


def test_five_groups_render_with_light_to_dark_ramps(tmp_path):
    import matplotlib.colors as mcolors

    labels = ["1 h", "12 h", "18 h", "24 h", "30 h"]
    paired = pd.concat([_paired(g, [0, 1, 2, 0], [2, 3, 1, 4], odor="Isoamyl Acetate")
                        for g in labels])
    summary = spc.group_summary(paired, labels, binary_threshold=2)
    between = spc.pairwise_between_stats(paired, labels, binary_threshold=2)
    assert len(between) == 10
    written = spc.render_bar_figures(paired, summary, between, groups=labels,
                                     trained_odor="Isoamyl Acetate", title="t", out_dir=tmp_path)
    assert len(written) == 5
    lum = [sum(mcolors.to_rgb(c[0])) for c in spc.group_colors(5)]
    assert lum == sorted(lum, reverse=True)
    spc.check_group_count([spc.parse_group(f"{i}=X") for i in range(5)])


def test_six_groups_render_and_seven_are_refused(tmp_path):
    import matplotlib.colors as mcolors

    labels = ["1 h", "6 h", "12 h", "18 h", "24 h", "30 h"]
    paired = pd.concat([_paired(g, [0, 1, 2, 0], [2, 3, 1, 4], odor="Isoamyl Acetate")
                        for g in labels])
    summary = spc.group_summary(paired, labels, binary_threshold=2)
    between = spc.pairwise_between_stats(paired, labels, binary_threshold=2)
    assert len(between) == 15
    written = spc.render_bar_figures(paired, summary, between, groups=labels,
                                     trained_odor="Isoamyl Acetate", title="t", out_dir=tmp_path)
    assert len(written) == 5
    for ramp in (spc.group_colors(6), [(c, c) for c in spc.delta_colors(6)]):
        assert len(set(ramp)) == 6
        for phase in (0, 1):
            lum = [sum(mcolors.to_rgb(c[phase])) for c in ramp]
            assert lum == sorted(lum, reverse=True)
    # adding the 6-step ramp must not repaint existing 4/5-group figures
    assert spc.group_colors(5)[0] == ("#86b6ef", "#f0915f")
    spc.check_group_count([spc.parse_group(f"{i}=X") for i in range(6)])
    with pytest.raises(SystemExit):
        spc.check_group_count([spc.parse_group(f"{i}=X") for i in range(7)])
