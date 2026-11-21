"""run_all.py + per_analysis.report on a tiny synthetic trial table, every analysis module
stubbed (the real exp1/exp2/... are imported lazily by run_all, so a stub placed in
sys.modules under the real name is what run_all imports)."""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_all
from per_analysis import io, report

THR = 2
MODULES = ("exp1", "exp2", "exp3", "cross", "power", "daynight")
FOLDERS = {"exp1": "02_exp1_exposure_cohorts", "exp2": "03_exp2_iaa_first",
           "exp3": "04_exp3_iaa_second", "cross": "05_cross_experiment",
           "power": "06_power", "daynight": "08_day_vs_evening"}
SUB = {"exp1": "a_pre_vs_post", "exp2": "b_presentation_1_vs_2",
       "exp3": "c_vs_paraffin_reference", "cross": "d_paraffin_noise_floor"}
SPEC_DIRS = [
    "01_data_qc",
    *[f"02_exp1_exposure_cohorts/{s}" for s in (
        "a_pre_vs_post", "b_did_vs_paraffin", "c_global_drift", "d_time_since_exposure",
        "e_odor_specific_suppression", "models")],
    *[f"{e}/{s}" for e in ("03_exp2_iaa_first", "04_exp3_iaa_second") for s in (
        "a_alone_vs_pair_pooled", "b_presentation_1_vs_2", "c_vs_paraffin_reference",
        "d_shared_baseline_diagnostic", "models")],
    *[f"05_cross_experiment/{s}" for s in (
        "a_iaa_baseline_by_cohort", "b_iaa_iaa_across_experiments", "c_eb_iaa_bidirectional",
        "d_paraffin_noise_floor", "models")],
    "06_power", "07_sensitivity_with_exclusions", "08_day_vs_evening", "tables",
]


# ── synthetic trials ──────────────────────────────────────────────────────────
def _fly(fly_id, rows, *, experiment, cohort, batch):
    out = []
    for block, label, odor, score, *extra in rows:
        p = io.parse_trial_label(label)
        r = dict(experiment=experiment, cohort=cohort, dataset=cohort, fly_id=fly_id,
                 batch=batch, session=batch, rig=1, host="Flybehavior2", window="day",
                 block=block, trial_label=label, trial_index=p.trial_index, odor=odor,
                 prime=None, partner=odor, presentation=np.nan,
                 is_light_probe=p.is_light_probe, is_click=False,
                 is_odor2=len(p.odors) == 2, odor_window_s=10.0, score=score,
                 responded=score >= THR, odor1_score=np.nan)
        if extra:
            r.update(extra[0])
        out.append(r)
    return out


def _exp1_fly(n, scores):
    pre, expo, post = scores
    return _fly(f"IAA-Sensitivity-1-1|b1|{n}", [
        ("pre", "pretest_1_isoamylacetate-1", "IAA", pre),
        ("exposure", "training_1_isoamylacetate-1", "IAA", expo),
        ("post", "testing_1_isoamylacetate-1", "IAA", post)],
        experiment="exp1", cohort="IAA-Sensitivity-1-1", batch="b1")


def _exp2_fly(n, scores):
    alone, pair = scores
    return _fly(f"IAA-PairFirst-Control-1-1|b2|{n}", [
        ("alone", "pretest_1_hexanol-1", "Hexanol", alone),
        ("pair", "testing_1_isoamylacetate-1_then_hexanol-1", "Hexanol", pair,
         dict(prime="IAA", presentation=1, odor1_score=3))],
        experiment="exp2", cohort="IAA-PairFirst-Control-1-1", batch="b2")


@pytest.fixture
def trials():
    flies = [_exp1_fly(1, (0, 3, 4)), _exp1_fly(2, (2, 1, 3)),
             _exp1_fly(3, (0, 0, 0)),                      # never-responder
             _exp2_fly(1, (0, 3)), _exp2_fly(2, (1, 2))]
    return pd.DataFrame([r for f in flies for r in f])


N_FLIES = 5


# ── stub modules ──────────────────────────────────────────────────────────────
def _png(path: Path):
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.plot([0, 1], [0, 1])
    fig.savefig(path, dpi=20)
    plt.close(fig)


def _stub(name, *, fail=False):
    mod = types.ModuleType(f"per_analysis.{name}")
    mod.calls = []

    def run(trials, out, *, tables):
        out, tables = Path(out), Path(tables)
        mod.calls.append(dict(n_flies=trials["fly_id"].nunique(), out=out, tables=tables))
        if fail:
            raise RuntimeError(f"{name} exploded on purpose")
        folder = out / FOLDERS[name] / SUB.get(name, "")
        _png(folder / f"{name}_figure.png")
        (folder / f"{name}_figure.csv").write_text("x,y\n0,0\n")
        tables.mkdir(parents=True, exist_ok=True)
        (tables / f"{name}_table.csv").write_text("a,b\n1,2\n")
        findings = [dict(section=FOLDERS[name], analysis=f"{name}_secondary", primary=False,
                         text=f"{name} secondary finding text.", numbers={"n_flies": 3})]
        if name == "exp1":
            findings.append(dict(section=FOLDERS[name], analysis="did_vs_paraffin",
                                 primary=True, text="EXP1 PRIMARY RESULT text.",
                                 numbers={"p": 0.0123, "p_batch": 0.04}))
            findings.append(dict(section=FOLDERS[name], analysis="design_contradiction",
                                 primary=False, text="Module-found contradiction XYZ.",
                                 numbers={}))
        if name == "power":
            findings.append(dict(section=FOLDERS[name], analysis="skipped", primary=False,
                                 text="power curve for cohort Q skipped: too few flies.",
                                 numbers={}))
        return findings

    mod.run = run
    return mod


@pytest.fixture
def stubs(monkeypatch, trials):
    mods = {name: _stub(name) for name in MODULES}
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, f"per_analysis.{name}", mod)
    monkeypatch.setattr(run_all, "load_inputs",
                        lambda cfg: dict(trials=trials, raw=None, facts=None, b2b=None))
    return mods


def _run(tmp_path, *extra):
    return run_all.main(["--results", str(tmp_path / "results"), "--log-level", "WARNING",
                         *extra])


# ── run_all ───────────────────────────────────────────────────────────────────
def test_full_tree_readmes_index_and_summary(tmp_path, stubs):
    assert _run(tmp_path) == 0
    res = tmp_path / "results"
    for rel in SPEC_DIRS:
        d = res / rel
        assert d.is_dir(), rel
        readme = d / "README.md"
        assert readme.is_file(), rel
        lines = [ln for ln in readme.read_text().splitlines() if ln.strip()]
        assert 2 <= len(lines) <= 4, (rel, lines)
    for rel in ("02_exp1_exposure_cohorts/b_did_vs_paraffin",
                "04_exp3_iaa_second/c_vs_paraffin_reference"):
        assert "PRIMARY" in (res / rel / "README.md").read_text()
    assert "PRIMARY" not in (res / "02_exp1_exposure_cohorts/a_pre_vs_post/README.md").read_text()
    assert (res / "index.html").is_file()
    assert (res / "SUMMARY.md").is_file()
    assert (res / "run_all.log").is_file()
    assert (res / "01_data_qc" / "qc_report.md").is_file()
    assert (res / "01_data_qc" / "column_mapping.yaml").is_file()
    for name in MODULES:
        assert len(stubs[name].calls) == (2 if name in ("exp1", "exp2", "exp3", "cross") else 1)
        assert stubs[name].calls[0]["out"] == res
        assert stubs[name].calls[0]["tables"] == res / "tables"


def test_failing_module_is_recorded_and_the_others_still_run(tmp_path, stubs, monkeypatch):
    bad = _stub("exp2", fail=True)
    monkeypatch.setitem(sys.modules, "per_analysis.exp2", bad)
    assert _run(tmp_path) == 1
    assert len(bad.calls) == 2                       # main run and the exclusion run
    for name in ("exp1", "exp3", "cross", "power", "daynight"):
        assert stubs[name].calls, name
    summary = (tmp_path / "results" / "SUMMARY.md").read_text()
    assert "exp2 exploded on purpose" in summary
    assert "RuntimeError" in summary
    assert "Traceback" in (tmp_path / "results" / "run_all.log").read_text()


def test_unimportable_module_is_recorded(tmp_path, stubs, monkeypatch):
    monkeypatch.setitem(sys.modules, "per_analysis.daynight", None)   # import -> ImportError
    assert _run(tmp_path, "--only", "daynight", "--only", "exp1") == 1
    assert stubs["exp1"].calls
    summary = (tmp_path / "results" / "SUMMARY.md").read_text()
    assert "daynight" in summary and re.search(r"(ModuleNotFound|Import)Error", summary)


def test_index_links_every_png(tmp_path, stubs):
    assert _run(tmp_path) == 0
    res = tmp_path / "results"
    html = (res / "index.html").read_text()
    links = set(re.findall(r'(?:href|src)="([^"]+)"', html))
    assert not [u for u in links if re.match(r"^([a-z]+:|/)", u)], "absolute or external link"
    pngs = {p.relative_to(res).as_posix() for p in res.rglob("*.png")}
    assert len(pngs) == 6 + 4                       # 6 modules + 4 re-run under 07
    assert pngs <= links
    for u in links:
        if not u.startswith("#"):
            assert (res / u).exists(), u
    assert {"SUMMARY.md", "01_data_qc/qc_report.md", "tables/exp1_table.csv",
            "tables/power_table.csv"} <= links
    assert "07_sensitivity_with_exclusions/tables/exp1_table.csv" in links
    assert "prefers-color-scheme: dark" in html
    assert 'name="viewport"' in html
    # thumbnails grouped by folder in folder order, captioned by the README's first line
    order = [m for m in re.findall(r'<img[^>]+src="([^"]+)"', html)]
    folders = [u.rsplit("/", 1)[0] for u in order]
    assert folders == sorted(folders, key=lambda f: f.split("/"))
    caption = report.readme_caption(res / "02_exp1_exposure_cohorts" / "a_pre_vs_post")
    assert caption and caption in html.replace("&#x27;", "'")


def test_only_limits_both_runs(tmp_path, stubs):
    assert _run(tmp_path, "--only", "exp1", "--only", "power") == 0
    assert len(stubs["exp1"].calls) == 2
    assert len(stubs["power"].calls) == 1
    for name in ("exp2", "exp3", "cross", "daynight"):
        assert not stubs[name].calls, name
    res = tmp_path / "results"
    for rel in SPEC_DIRS:                             # the tree is complete regardless
        assert (res / rel / "README.md").is_file(), rel


def test_exclusion_run_writes_under_07(tmp_path, stubs, caplog):
    caplog.set_level("INFO")
    assert _run(tmp_path) == 0
    res = tmp_path / "results"
    s07 = res / "07_sensitivity_with_exclusions"
    main, excl = stubs["exp1"].calls
    assert main["n_flies"] == N_FLIES
    assert excl["n_flies"] < N_FLIES                  # the never-responder is dropped
    assert excl["out"] == s07 and excl["tables"] == s07 / "tables"
    assert (s07 / "02_exp1_exposure_cohorts" / "a_pre_vs_post" / "exp1_figure.png").is_file()
    assert (s07 / "tables" / "exp1_table.csv").is_file()
    assert (s07 / "excluded_flies.csv").is_file()
    for mod in ("02_exp1_exposure_cohorts", "03_exp2_iaa_first", "04_exp3_iaa_second",
                "05_cross_experiment"):                # mirrored structure, with READMEs
        assert (s07 / mod / "README.md").is_file(), mod
    assert not (s07 / "06_power").exists()
    log = (res / "run_all.log").read_text()
    assert re.search(r"rule=never_responders\s+flies=1\b", log), log[-2000:]
    summary = (res / "SUMMARY.md").read_text()
    assert "with exclusions" in summary.lower()


def test_report_only_rebuilds_the_report_without_running_modules(tmp_path, stubs):
    assert _run(tmp_path) == 0
    res = tmp_path / "results"
    calls = {n: len(stubs[n].calls) for n in MODULES}
    first_log = (res / "run_all.log").read_text()
    (res / "SUMMARY.md").unlink()
    (res / "index.html").unlink()
    assert _run(tmp_path, "--report-only") == 0
    assert {n: len(stubs[n].calls) for n in MODULES} == calls       # nothing re-ran
    summary = (res / "SUMMARY.md").read_text()
    assert "EXP1 PRIMARY RESULT text." in summary
    assert "exp1 ok / 07 ok" in summary
    assert "1 flies" in summary or "drops 1 fl" in summary        # QC summary kept
    assert (res / "index.html").is_file()
    assert (res / "run_all.log").read_text().startswith(first_log)  # appended, not wiped


def test_report_only_without_a_manifest_reads_status_from_the_log(tmp_path, stubs):
    assert _run(tmp_path) == 0
    res = tmp_path / "results"
    (res / "run_manifest.json").unlink()
    assert _run(tmp_path, "--report-only") == 0
    summary = (res / "SUMMARY.md").read_text()
    assert "exp1 ok / 07 ok" in summary and "daynight ok" in summary
    assert "drops 1 fl" in summary                                 # QC recomputed


def test_report_only_needs_findings(tmp_path, stubs):
    (tmp_path / "results").mkdir()
    assert _run(tmp_path, "--report-only") == 2


def test_exclude_qc_drops_flagged_flies_from_the_main_run_too(tmp_path, stubs):
    assert _run(tmp_path, "--exclude-qc", "--only", "exp1") == 0
    main, excl = stubs["exp1"].calls
    assert main["n_flies"] == excl["n_flies"] < N_FLIES


# ── report ────────────────────────────────────────────────────────────────────
def _qc_dir(res: Path):
    d = res / "01_data_qc"
    d.mkdir(parents=True)
    (d / "column_mapping.yaml").write_text(
        "score:\n  source: model_predictions.csv\n  description: ordinal -1..5\n"
        "  contract: true\nfly:\n  source: trial tables\n  description: batch\n"
        "  contract: false\n")
    (d / "qc_report.md").write_text("# Data QC\n\n123 scored trials, 9 flies, 2 cohorts.\n\n"
                                    "## Flies per cohort x window\n\n| cohort | day |\n"
                                    "|---|---|\n| C1 | 9 |\n\n## Missing trials\n\nnone\n")


def test_summary_puts_primary_first_and_lists_contradictions(tmp_path):
    res = tmp_path / "results"
    _qc_dir(res)
    findings = [
        dict(section="02_exp1_exposure_cohorts", analysis="drift", primary=False,
             text="SECONDARY drift text.", numbers={"p": 0.5}),
        dict(section="03_exp2_iaa_first", analysis="pooled", primary=False,
             text="SECONDARY pooled text.", numbers={}),
        dict(section="04_exp3_iaa_second", analysis="vs_paraffin", primary=True,
             text="PRIMARY exp3 text.", numbers={"p": 0.001, "n_flies": 20}),
        dict(section="03_exp2_iaa_first", analysis="design_contradiction", primary=False,
             text="MODULE contradiction text.", numbers={}),
        dict(section="06_power", analysis="skipped", primary=False,
             text="SKIPPED power text.", numbers={}),
        dict(section="08_day_vs_evening", analysis="failed", primary=False,
             text="daynight raised ValueError: nope", numbers={}, module="daynight"),
        dict(section="04_exp3_iaa_second", analysis="vs_paraffin", primary=True,
             text="PRIMARY exp3 text, exclusions.", numbers={"p": 0.01}, run="exclusions"),
    ]
    path = report.write_summary(res, findings, {"n_trials": 123, "n_flies": 9})
    text = path.read_text()
    heads = [ln for ln in text.splitlines() if ln.startswith("## ")]
    names = [h.lower() for h in heads]
    i_data = next(i for i, h in enumerate(names) if "data and assumptions" in h)
    i_prim = next(i for i, h in enumerate(names) if "primary" in h)
    i_by = next(i for i, h in enumerate(names) if "by analysis" in h)
    i_design = next(i for i, h in enumerate(names) if "data vs described design" in h)
    i_not = next(i for i, h in enumerate(names) if "not analysed" in h)
    assert i_data < i_prim < i_by < i_design < i_not
    assert text.index("PRIMARY exp3 text.") < text.index("SECONDARY drift text.")
    assert "PRIMARY exp3 text, exclusions." in text
    assert "score" in text and "model_predictions.csv" in text        # column mapping
    assert "123 scored trials" in text                                  # QC summary
    design = text[text.index(heads[i_design]):text.index(heads[i_not])]
    assert "MODULE contradiction text." in design
    for known in ("all 8 odors", "master-valve click", "never tested on paraffin",
                  "concurrently", "paraffin-off", "light-only", "0.025 s",
                  "2026-11-18", "carry-over", "single session",
                  # exp1 design contradictions
                  "first pre-test and the last post-test", "0.9999998", "collinear",
                  "26 evening flies",
                  # methods caveats
                  "GEE", "model of record", "variational", "pymer4", "sign-flip"):
        assert known in design, known
    assert "methods caveats" in heads[i_design].lower()
    tail = text[text.index(heads[i_not]):]
    assert "SKIPPED power text." in tail and "daynight raised ValueError: nope" in tail
    assert "MODULE contradiction text." not in text[text.index(heads[i_by]):
                                                   text.index(heads[i_design])]


def test_summary_pairs_each_primary_with_its_own_exclusion_twin(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    key = dict(section="exp1", analysis="b_did_vs_paraffin", primary=True, numbers={})
    findings = [dict(key, text="MAIN IAA."), dict(key, text="MAIN Hex."),
                dict(key, text="EXCL IAA.", run="exclusions"),
                dict(key, text="EXCL Hex.", run="exclusions"),
                dict(key, text="EXCL extra.", run="exclusions")]
    text = report.write_summary(res, findings, None).read_text()
    prim = text[text.index("## 2."):text.index("## 3.")]
    assert prim.count("EXCL IAA.") == 1 and prim.count("EXCL Hex.") == 1
    assert prim.index("MAIN IAA.") < prim.index("EXCL IAA.") < prim.index("MAIN Hex.") \
        < prim.index("EXCL Hex.") < prim.index("EXCL extra.")
    assert "with exclusions only" in prim[prim.index("EXCL Hex."):]


def test_summary_without_findings_or_qc_files(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    text = report.write_summary(res, [], None).read_text()
    assert "Data vs described design" in text and "all 8 odors" in text


def test_write_readmes_never_overwrites_and_marks_primary(tmp_path):
    res = tmp_path / "results"
    mine = res / "02_exp1_exposure_cohorts" / "a_pre_vs_post" / "README.md"
    mine.parent.mkdir(parents=True)
    mine.write_text("Module README line one.\nLine two.\n")
    prim = res / "04_exp3_iaa_second" / "c_vs_paraffin_reference" / "README.md"
    prim.parent.mkdir(parents=True)
    prim.write_text("Module-written IAA vs paraffin README.\nSecond line.\n")
    report.write_readmes(res)
    report.write_readmes(res)                          # idempotent
    assert mine.read_text() == "Module README line one.\nLine two.\n"
    ptxt = prim.read_text()
    assert ptxt.startswith("Module-written IAA vs paraffin README.\nSecond line.\n")
    assert ptxt.count("PRIMARY") == 1
    for rel in SPEC_DIRS:
        assert (res / rel / "README.md").is_file(), rel
    assert report.readme_caption(mine.parent) == "Module README line one."


def test_readme_caption_skips_a_heading_naming_the_folder(tmp_path):
    d = tmp_path / "01_data_qc"
    d.mkdir()
    (d / "README.md").write_text("# 01_data_qc\n`tidy_trials.parquet`: one row per trial.\n")
    assert report.readme_caption(d) == "`tidy_trials.parquet`: one row per trial."
    assert report.readme_caption(tmp_path / "missing") == ""


def test_normalize_findings_fills_missing_keys():
    out = report.normalize_findings([{"text": "t"}], module="exp1", run="main")
    assert out[0]["section"] == "02_exp1_exposure_cohorts"
    assert out[0]["primary"] is False and out[0]["numbers"] == {}
    assert out[0]["module"] == "exp1" and out[0]["run"] == "main"


def _window_table(res: Path, probe_s: float, click_s: float) -> None:
    import pandas as pd
    rows = ([dict(experiment="exp2", cohort="IAA-PairFirst-Control-1-1", is_light_probe=True,
                  odor_window_s=probe_s)] * 9
            + [dict(experiment="exp1", cohort="IAA-Sensitivity-1-1", is_light_probe=True,
                    odor_window_s=30.0)] * 3
            + [dict(experiment="exp3", cohort="IAA-PairSecond-Control-1-1", is_light_probe=False,
                    odor_window_s=click_s)] * 4
            + [dict(experiment="exp3", cohort="IAA-PairSecond-Control-1-1", is_light_probe=False,
                    odor_window_s=10.0)] * 5)
    pd.DataFrame(rows).to_parquet(res / "01_data_qc" / "tidy_trials.parquet")


def _design(text: str) -> str:
    i = text.index("## 4. Data vs described design")
    return text[i:text.index("\n## ", i + 5)]


def test_summary_counts_short_windows_from_the_trial_table(tmp_path):
    res = tmp_path / "results"
    _qc_dir(res)
    _window_table(res, probe_s=0.025, click_s=0.025)
    design = _design(report.write_summary(res, [], {"n_trials": 1}).read_text())
    assert "9 of 12 light probes" in design and "0.025 s" in design
    assert "4 odor trials" in design and "next full pipeline run" in design


def test_summary_says_the_windows_are_valid_once_rebuilt(tmp_path):
    res = tmp_path / "results"
    _qc_dir(res)
    _window_table(res, probe_s=10.0, click_s=10.0)
    design = _design(report.write_summary(res, [], {"n_trials": 1}).read_text())
    assert "All 12 light probes" in design and "sensors log" in design
    assert "0.025 s" not in design
