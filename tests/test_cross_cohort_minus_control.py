"""cross_cohort_minus_control: each cohort's per-fly change minus the control's, per odor."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from fbpipe.analysis import fly_stats as fs  # noqa: E402
from scripts.analysis import cross_cohort_minus_control as ccm  # noqa: E402

COHORTS = ["IAA-Sensitivity-1-1", "Hex-Sensitivity-1-0.1", "Paraffin-Sensitivity-1-100"]
CONTROL = "Paraffin-Sensitivity-1-100"
ODORS = ["Isoamyl Acetate", "Hexanol", "Citral"]


def _write_cohort(root: Path, cohort: str, shift: dict[str, float], n_batches: int = 3,
                  flies_per_batch: int = 3, seed: int = 0, missing_phase: bool = False) -> None:
    rng = np.random.default_rng(seed)
    rows_pz, rows_lr = [], []
    month = {"IAA": "october", "Hex": "november", "Paraffin": "november"}[cohort.split("-")[0]]
    for b in range(n_batches):
        batch = f"{month}_{10 + b:02d}_batch_1"
        for f in range(1, flies_per_batch + 1):
            for odor in ODORS:
                pre = rng.normal(0, 1)
                post = pre + shift.get(odor, 0.0) + rng.normal(0, 0.3)
                rows_pz.append({"dataset": cohort, "fly": batch, "fly_number": f, "odor": odor,
                                "zpeak_during_signed_pre": pre, "zpeak_during_signed_post": post,
                                "zpeak_during_unsigned_pre": abs(pre),
                                "zpeak_during_unsigned_post": abs(post),
                                "zpeak_ratio_signed_pre": pre / 3, "zpeak_ratio_signed_post": post / 3,
                                "zpeak_ratio_unsigned_pre": abs(pre) / 3,
                                "zpeak_ratio_unsigned_post": abs(post) / 3})
                rows_lr.append({"dataset": cohort, "fly": batch, "fly_number": f, "odor": odor,
                                "thr_auc_log_ratio_pre": pre / 2,
                                "thr_auc_log_ratio_post": post / 2})
    if missing_phase:                       # a fly slot with no post-test at all
        for rows in (rows_pz, rows_lr):
            for odor in ODORS:
                rows.append({**{k: (np.nan if k.endswith("_post") else v)
                                for k, v in rows[0].items()},
                             "fly": f"{month}_30_batch_1", "fly_number": 4, "odor": odor})
    win = []
    for r in rows_pz:
        if not np.isfinite(r.get("zpeak_during_signed_post", np.nan)):
            continue
        sp = int(np.clip(round(2 + r["zpeak_during_signed_pre"]), -1, 5))
        so = int(np.clip(round(2 + r["zpeak_during_signed_post"]), -1, 5))
        win.append({"cohort": cohort, "batch_folder": r["fly"], "fly_number": r["fly_number"],
                    "fly_id": f"{cohort}|{r['fly']}|{r['fly_number']}", "odor": r["odor"],
                    "score_pre": sp, "score_post": so, "score_change": so - sp,
                    "responded_pre": sp >= 2, "responded_post": so >= 2})
    data = root / "Data"
    data.mkdir(exist_ok=True)
    path = data / "window_per_fly_per_odor.csv"
    old = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    pd.concat([old, pd.DataFrame(win)]).to_csv(path, index=False)
    pz = root / cohort / "PeakZ-Before-vs-During"
    lr = root / cohort / "AUC-LogRatio-Before-vs-During"
    pz.mkdir(parents=True)
    lr.mkdir(parents=True)
    pd.DataFrame(rows_pz).to_csv(pz / "peakz_per_fly_change.csv", index=False)
    pd.DataFrame(rows_lr).to_csv(lr / "logratio_per_fly_change.csv", index=False)


@pytest.fixture()
def folder(tmp_path: Path) -> Path:
    _write_cohort(tmp_path, "IAA-Sensitivity-1-1", {"Isoamyl Acetate": 2.0}, seed=1,
                  missing_phase=True)
    _write_cohort(tmp_path, "Hex-Sensitivity-1-0.1", {"Hexanol": 0.5}, n_batches=4, seed=2)
    _write_cohort(tmp_path, CONTROL, {}, n_batches=4, seed=3)
    return tmp_path


def test_per_fly_changes_keep_only_flies_with_both_phases(folder):
    ch = ccm.per_fly_changes(folder, "IAA-Sensitivity-1-1", "peakz_during",
                             "zpeak_during_signed")
    assert len(ch) == 3 * 3 * len(ODORS)            # the fly with no post-test is gone
    assert set(ch.columns) >= {"cohort", "batch", "fly_number", "odor", "pre", "post", "change"}
    assert np.allclose(ch["change"], ch["post"] - ch["pre"])


def test_minus_control_rows_equal_the_controlled_test(folder):
    t = ccm.minus_control_table(folder, COHORTS, CONTROL, "peakz_during", n_perm=500,
                                n_boot=500)
    signed = t[t["panel"] == "zpeak_during_signed"]
    assert set(signed["cohort"]) == {"IAA-Sensitivity-1-1", "Hex-Sensitivity-1-0.1"}
    assert len(signed) == 2 * len(ODORS)
    row = signed[(signed["cohort"] == "IAA-Sensitivity-1-1")
                 & (signed["odor"] == "Isoamyl Acetate")].iloc[0]
    x = ccm.per_fly_changes(folder, "IAA-Sensitivity-1-1", "peakz_during", "zpeak_during_signed")
    c = ccm.per_fly_changes(folder, CONTROL, "peakz_during", "zpeak_during_signed")
    ref = fs.controlled_change_test(x[x["odor"] == "Isoamyl Acetate"],
                                    c[c["odor"] == "Isoamyl Acetate"], value="change",
                                    batch="batch", n_perm=500, n_boot=500, seed=0,
                                    ci="permutation")
    for key in ("estimate", "ci_lo", "ci_hi", "p_batch", "exposed_change", "control_change"):
        assert row[key] == pytest.approx(ref[key])
    # the fly-level p ignores batches: named so nobody reads it as the result
    assert row["p_fly_ignores_batches"] == pytest.approx(ref["p_fly"])
    assert "p_fly" not in row.index
    assert row["estimate"] == pytest.approx(x[x["odor"] == "Isoamyl Acetate"]["change"].mean()
                                            - c[c["odor"] == "Isoamyl Acetate"]["change"].mean())
    assert bool(row["sensitized_odor"]) is True
    assert row["ci_method"] == "permutation"
    assert (row["ci_lo"] > 0 or row["ci_hi"] < 0) == (row["p_batch"] < 0.05)
    assert row["n_flies_exposed"] == 9 and row["n_batches_exposed"] == 3
    assert row["n_flies_control"] == 12 and row["n_batches_control"] == 4


def test_holm_is_taken_over_every_contrast_in_a_panel(folder):
    t = ccm.minus_control_table(folder, COHORTS, CONTROL, "peakz_during", n_perm=500,
                                n_boot=500)
    for panel, g in t.groupby("panel"):
        assert np.allclose(g["p_batch_holm"].to_numpy(),
                           fs.holm_adjust(g["p_batch"].to_numpy()), equal_nan=True)
        assert list(g["stars"]) == [fs.stars(p).replace("n.s.", "") for p in g["p_batch_holm"]]


def test_log_auc_reads_the_scorers_log_ratio(folder):
    t = ccm.minus_control_table(folder, COHORTS, CONTROL, "log_auc", n_perm=300, n_boot=300)
    assert set(t["panel"]) == {"thr_auc_log_ratio"}
    pz = ccm.minus_control_table(folder, COHORTS, CONTROL, "peakz_during", n_perm=300,
                                 n_boot=300)
    a = t.set_index(["cohort", "odor"])["estimate"]
    b = pz[pz["panel"] == "zpeak_during_signed"].set_index(["cohort", "odor"])["estimate"]
    assert np.allclose(a.sort_index(), b.sort_index() / 2)    # the fixture's log ratio = z / 2


def test_peakz_ratio_is_during_over_before_signed_and_unsigned(folder):
    t = ccm.minus_control_table(folder, COHORTS, CONTROL, "peakz_ratio", n_perm=300,
                                n_boot=300)
    assert set(t["panel"]) == {"zpeak_ratio_signed", "zpeak_ratio_unsigned"}
    pz = ccm.minus_control_table(folder, COHORTS, CONTROL, "peakz_during", n_perm=300,
                                 n_boot=300)
    a = t[t["panel"] == "zpeak_ratio_signed"].set_index(["cohort", "odor"])["estimate"]
    b = pz[pz["panel"] == "zpeak_during_signed"].set_index(["cohort", "odor"])["estimate"]
    assert np.allclose(a.sort_index(), b.sort_index() / 3)    # the fixture's ratio = z / 3
    assert "during / before" in ccm.MEASURES["peakz_ratio"]["phrase"]


def test_change_table_is_paired_per_fly_with_exact_wilcoxon(folder):
    t = ccm.change_table(folder, COHORTS, "log_auc")
    assert set(t["cohort"]) == set(COHORTS)                    # the control is drawn too
    row = t[(t["cohort"] == "IAA-Sensitivity-1-1") & (t["odor"] == "Isoamyl Acetate")].iloc[0]
    x = ccm.per_fly_changes(folder, "IAA-Sensitivity-1-1", "log_auc", "thr_auc_log_ratio")
    d = x[x["odor"] == "Isoamyl Acetate"]["change"].to_numpy()
    m, lo, hi = fs.fly_cluster_mean_ci(d, np.arange(d.size))
    assert row["change"] == pytest.approx(d.mean())
    assert (row["ci_lo"], row["ci_hi"]) == pytest.approx((lo, hi))
    assert row["p_wilcoxon"] == pytest.approx(fs.wilcoxon_signed_rank_p(d))
    assert np.allclose(t["p_wilcoxon_holm"], fs.holm_adjust(t["p_wilcoxon"]), equal_nan=True)


def test_no_row_without_the_odor_in_both_cohorts(folder):
    df = pd.read_csv(folder / CONTROL / "PeakZ-Before-vs-During" / "peakz_per_fly_change.csv")
    df[df["odor"] != "Citral"].to_csv(
        folder / CONTROL / "PeakZ-Before-vs-During" / "peakz_per_fly_change.csv", index=False)
    t = ccm.minus_control_table(folder, COHORTS, CONTROL, "peakz_during", n_perm=200,
                                n_boot=200)
    assert "Citral" not in set(t["odor"])


def test_write_figures_names_and_contents(folder):
    paths = ccm.write(folder, COHORTS, CONTROL, window="09:40–16:20",
                      out_dir=folder / "Cross-Cohort", n_perm=300, n_boot=300)
    names = sorted(p.name for p in paths)
    assert names == sorted([
        "cross_cohort_peakz_during_change_minus_control_window.png",
        "cross_cohort_peakz_ratio_change_minus_control_window.png",
        "cross_cohort_log_auc_change_minus_control_window.png",
        "cross_cohort_log_auc_change_window.png",
    ])
    for p in paths:
        assert p.is_file() and p.stat().st_size > 10_000
        assert p.with_suffix(".csv").is_file()


def test_without_control_nothing_minus_control_is_written(folder):
    paths = ccm.write(folder, COHORTS, None, window="09:40–16:20",
                      out_dir=folder / "Cross-Cohort", n_perm=200, n_boot=200)
    assert [p.name for p in paths] == ["cross_cohort_log_auc_change_window.png"]


def test_batch_dates_and_odor_order():
    assert ccm.batch_dates(["november_05_batch_1_rig_3", "october_09_batch_2",
                            "november_12_batch_1"]) == ("Oct 9", "Nov 12")
    assert ccm.batch_dates(["weird"]) is None
    t = pd.DataFrame({"cohort": ["A", "A", "B", "B"], "odor": ["Citral", "Hexanol", "Citral",
                                                             "Linalool"],
                      "sensitized_odor": [False, True, False, True]})
    assert ccm.odor_order(t, ["B", "A"]) == ["Linalool", "Hexanol", "Citral"]
    two_panels = pd.concat([t.assign(panel="signed"), t.assign(panel="unsigned")])
    assert ccm.odor_order(two_panels, ["B", "A"]) == ["Linalool", "Hexanol", "Citral"]


def test_run_date_note_flags_only_cohorts_that_never_overlap_the_control():
    t = pd.DataFrame({"cohort": ["IAA-x", "Hex-x"],
                      "batches_exposed": ["october_09_batch_1;november_01_batch_1",
                                          "november_01_batch_1;november_12_batch_1"],
                      "batches_control": ["november_05_batch_1;november_12_batch_1"] * 2})
    note = ccm.run_date_note(t, ["IAA-x", "Hex-x"], "Paraffin-x")
    assert "IAA Oct 9–Nov 1 (never overlaps the control)" in note
    assert "Hex Nov 1–Nov 12;" in note and "Hex Nov 1–Nov 12 (" not in note
    assert "Paraffin control Nov 5–Nov 12" in note


def test_palette_and_inks_match_the_sensitivity_script():
    from scripts.analysis import sensitivity_window_results as swres
    assert ccm.COHORT_PALETTE == swres.COHORT_PALETTE
    assert (ccm.INK, ccm.INK_2, ccm.GRID) == (swres.INK, swres.INK_2, swres.GRID)


def test_cli(folder):
    rc = ccm.main(["--out-dir", str(folder), "--cohort", COHORTS[0], "--cohort", COHORTS[1],
                   "--cohort", COHORTS[2], "--control", CONTROL, "--window", "09:40–16:20",
                   "--n-perm", "200", "--n-boot", "200"])
    assert rc == 0
    assert (folder / "Cross-Cohort"
            / "cross_cohort_peakz_during_change_minus_control_window.csv").is_file()


# ── the one-odor focus figure (user 2026-11-18: IAA change, IAA-exposed vs Paraffin) ──
def test_focus_table_has_every_measure_with_uncorrected_tests(folder):
    t = ccm.focus_table(folder, "IAA-Sensitivity-1-1", CONTROL, "Isoamyl Acetate",
                        n_perm=400)
    assert list(t["panel"]) == [p[0] for p in ccm.FOCUS_PANELS]
    w = pd.read_csv(folder / "Data" / "window_per_fly_per_odor.csv")
    x = w[(w["cohort"] == "IAA-Sensitivity-1-1") & (w["odor"] == "Isoamyl Acetate")]
    c = w[(w["cohort"] == CONTROL) & (w["odor"] == "Isoamyl Acetate")]
    r = t.set_index("panel").loc["responded"]
    fx = x["responded_post"].astype(int) - x["responded_pre"].astype(int)
    fc = c["responded_post"].astype(int) - c["responded_pre"].astype(int)
    assert r["exposed_change"] == pytest.approx(100 * fx.mean())
    assert r["control_change"] == pytest.approx(100 * fc.mean())
    assert r["estimate"] == pytest.approx(100 * (fx.mean() - fc.mean()))
    assert r["p_exposed_own"] == pytest.approx(
        fs.mcnemar_exact_p(x["responded_pre"], x["responded_post"]) or np.nan, nan_ok=True)
    ref = fs.controlled_change_test(x.assign(v=100 * fx), c.assign(v=100 * fc), value="v",
                                    batch="batch_folder", n_perm=400, seed=0, ci="permutation")
    assert r["p_batch"] == pytest.approx(ref["p_batch"])
    assert (r["ci_lo"], r["ci_hi"]) == pytest.approx((ref["ci_lo"], ref["ci_hi"]))
    s = t.set_index("panel").loc["score"]
    assert s["p_exposed_own"] == pytest.approx(fs.wilcoxon_signed_rank_p(x["score_change"]))
    pz = t.set_index("panel").loc["zpeak_during_signed"]
    ch = ccm.per_fly_changes(folder, "IAA-Sensitivity-1-1", "peakz_during", "zpeak_during_signed")
    ch = ch[ch["odor"] == "Isoamyl Acetate"]["change"]
    assert pz["exposed_change"] == pytest.approx(ch.mean())
    assert pz["p_exposed_own"] == pytest.approx(fs.wilcoxon_signed_rank_p(ch))
    assert (t["n_exposed"] == 9).all() and (t["n_control"] == 12).all()


def test_focus_figure_is_written_by_write(folder):
    paths = ccm.write(folder, COHORTS, CONTROL, window="09:40–16:20",
                      out_dir=folder / "Cross-Cohort", n_perm=200, n_boot=200,
                      focus=["IAA-Sensitivity-1-1"])
    names = [p.name for p in paths]
    assert "isoamyl_acetate_change_IAA_vs_Paraffin_window.png" in names
    assert (folder / "Cross-Cohort" / "isoamyl_acetate_change_IAA_vs_Paraffin_window.csv").is_file()


def test_p_labels_never_round_a_p_below_005_up_to_005():
    assert ccm._p_short(0.0477) == "p=0.048"
    assert ccm._p_short(0.0607) == "p=0.061"
    assert ccm._p_short(0.2) == "p=0.20"
    assert ccm._p_short(0.0004) == "p<0.001"
    assert ccm._p_short(float("nan")) == ""


def test_short_names_fall_back_to_full_names_when_prefixes_collide():
    names = ccm.short_names(["IAA-Sensitivity-1-1", "IAA-Sensitivity-1-0.1",
                             "Paraffin-Sensitivity-1-100"])
    assert names["Paraffin-Sensitivity-1-100"] == "Paraffin"
    assert names["IAA-Sensitivity-1-1"] == "IAA-Sensitivity-1-1"
    assert ccm.batch_dates(["october_09_batch_2"]) == ("Oct 9", "Oct 9")


def test_protocol_note_counts_paraffin_off_batches_and_flies(tmp_path):
    manifest = tmp_path / "paraffin_off_protocol_trials.csv"
    manifest.write_text("# comment\n"
                        "dataset,batch,trial,protocol\n"
                        "P-Sensitivity-1-100,b1,training_1,paraffin_OFF_30s\n"
                        "P-Sensitivity-1-100,b2,training_1,paraffin_OFF_30s\n"
                        "H-Sensitivity-1-0.1,b9,pretest_1,paraffin_OFF_30s\n")
    off = ccm.paraffin_off_batches(manifest)
    assert off == {"P-Sensitivity-1-100": {"b1", "b2"}, "H-Sensitivity-1-0.1": {"b9"}}
    flies = pd.DataFrame({"batch": ["b1", "b1", "b2", "b3", "b3", "b3"],
                          "fly_number": [1, 2, 1, 1, 2, 3]})
    note = ccm.protocol_note(flies, "P-Sensitivity-1-100", off)
    assert "2 of 3 P-Sensitivity-1-100 batches (3 of 6 flies)" in note
    assert "No odor either way" in note
    assert ccm.protocol_note(flies, "P-Sensitivity-1-100", {}) == ""


def test_batch_lists_follow_each_panel(folder):
    # a fly with the signed peak z missing but the unsigned present must be listed on the
    # unsigned rows only
    path = folder / "IAA-Sensitivity-1-1" / "PeakZ-Before-vs-During" / "peakz_per_fly_change.csv"
    df = pd.read_csv(path)
    extra = df.iloc[:len(ODORS)].copy()
    extra["fly"] = "october_31_batch_9"
    extra["zpeak_during_signed_post"] = np.nan
    pd.concat([df, extra]).to_csv(path, index=False)
    t = ccm.minus_control_table(folder, COHORTS, CONTROL, "peakz_during", n_perm=200, n_boot=200)
    t = ccm._with_batches(t, folder, COHORTS, CONTROL, "peakz_during")
    iaa = t[t["cohort"] == "IAA-Sensitivity-1-1"]
    assert iaa[iaa["panel"] == "zpeak_during_unsigned"]["batches_exposed"].str.contains(
        "october_31_batch_9").all()
    assert not iaa[iaa["panel"] == "zpeak_during_signed"]["batches_exposed"].str.contains(
        "october_31_batch_9").any()
