"""The score-window comparison on a model that carries an odor-arrival delay.

The pipeline's ``score_window_comparison`` stage exited 1 on every run once the
deployed scorer gained ``fbr_odor_delay_s`` (2026-11-09): window_rescore
refused delay models. End to end here, on a tiny guarded 1 s-delay model: the
stage exits 0 and writes its four outputs, its 'odor' window is the production
score of the same trials, and its figures say that every window is shifted by
the model's delay (and say nothing for a delay-free model).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

import _window_rescore_fixtures as fx  # noqa: E402
import scripts.analysis.score_window_comparison as swc  # noqa: E402
from flybehavior_response import xgb_ordinal as xo  # noqa: E402

DS = "IAA-Sensitivity5s-1-1"
ODORS = ("isoamylacetate-1", "hexanol-0.1")
KEY = ["dataset", "fly", "fly_number", "trial_label"]
NOTE = "Windows shifted +1 s from the valve times (the model's odor-arrival delay)"


def _wide(n_flies: int = 4) -> pd.DataFrame:
    """Pre and post trials of two odors; half the flies respond to post IAA late."""
    meta, traces = [], []
    for fly in range(n_flies):
        for phase in ("pretest", "testing"):
            for i, odor in enumerate(ODORS):
                late = phase == "testing" and i == 0 and fly % 2 == 0
                events = [(36.0, 39.0, 60.0)] if late else [(31.5, 33.0, 60.0 * (fly == 1))]
                meta.append({
                    "dataset": DS, "fly": f"oct_15_batch_{fly}", "fly_number": 1,
                    "trial_label": f"{phase}_{i + 1}_{odor}", "trial_type": phase,
                    "fps": 40.0, "trial_odor_on_s": 30.0, "trial_odor_off_s": 35.0,
                    "trial_duration_s": 65.0, "global_max": 80.0, "trimmed_global_min": 9.0,
                    "frozen": False,
                })
                traces.append(fx.trace(len(traces), 65.0, 40.0, events))
    trace_df = pd.DataFrame(np.vstack(traces),
                            columns=[f"dir_val_{j}" for j in range(fx.N_COLS)])
    return pd.concat([pd.DataFrame(meta), trace_df], axis=1)


@pytest.fixture
def inputs(tmp_path):
    wide = _wide()
    pre_p, post_p = tmp_path / "pre.parquet", tmp_path / "post.parquet"
    wide.loc[wide["trial_type"] == "pretest"].to_parquet(pre_p, index=False)
    wide.loc[wide["trial_type"] == "testing"].to_parquet(post_p, index=False)
    predictions = tmp_path / "model_predictions.csv"
    wide[KEY + ["trial_type"]].assign(score=0, prediction=0).to_csv(predictions, index=False)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(
        {"analysis": {"combined": {"combined_base": {"wide": fx.PIPELINE_WIDE_KEYS}}}}))
    return {"wide": wide, "paths": [pre_p, post_p], "predictions": predictions,
            "config": config}


def _argv(inputs, model, out_dir):
    return [
        "--predictions-csv", str(inputs["predictions"]),
        "--wide-csv", *map(str, inputs["paths"]),
        "--model-path", str(model), "--config", str(inputs["config"]),
        "--after-s", "5.0", "--binary-threshold", "2",
        "--out-dir", str(out_dir), "--cohorts", DS,
    ]


def test_the_stage_exits_0_on_a_delay_model_and_writes_its_outputs(inputs, tmp_path, monkeypatch):
    model = fx.build_tiny_model(tmp_path / "tiny")
    seen = {}
    for name in ("build_grid", "build_cs_plus"):
        real = getattr(swc, name)

        def spy(summary, *, _real=real, _name=name, **kw):
            seen[_name] = kw.get("delay_s")
            return _real(summary, **kw)

        monkeypatch.setattr(swc, name, spy)

    out_dir = tmp_path / "out"
    assert swc.main(_argv(inputs, model, out_dir)) == 0

    stem = swc.ev._safe_dirname(DS)
    folder = out_dir / stem
    for suffix in ("_score_windows.png", "_score_windows_cs_plus.png",
                   "_score_windows_summary.csv", "_score_windows_trials.csv"):
        assert (folder / f"{stem}{suffix}").is_file(), suffix
    assert seen == {"build_grid": 1.0, "build_cs_plus": 1.0}

    # The 'odor' window is the production score of the same trials.
    wide = inputs["wide"]
    booster = xo.load_xgb_model(model)
    production = xo.predict_ordinal_scores(
        booster, xo.features_for_model(wide, xo.scoring_config(booster)))
    by_trial = dict(zip(zip(wide["fly"], wide["trial_label"]), production.tolist()))
    trials = pd.read_csv(folder / f"{stem}_score_windows_trials.csv")
    assert len(trials) == 8                      # 4 flies x 2 odors, paired
    index = {"Isoamyl Acetate 1": 1, "Hexanol 0.1": 2}
    for row in trials.itertuples():
        i = index[row.odor]
        assert row.score_pre_odor == by_trial[(row.fly, f"pretest_{i}_{ODORS[i - 1]}")]
        assert row.score_post_odor == by_trial[(row.fly, f"testing_{i}_{ODORS[i - 1]}")]
    # The late responders (36-39 s) sit in the 'after' window [36, 41) s.
    late = trials.loc[(trials["odor"] == "Isoamyl Acetate 1")
                      & trials["fly"].isin(["oct_15_batch_0", "oct_15_batch_2"])]
    assert (late["score_post_after"] == 2).all() and (late["score_post_odor"] == 1).all()


def test_figures_name_the_models_delay_only_when_there_is_one(inputs, tmp_path):
    model = fx.build_tiny_model(tmp_path / "tiny")
    paired = swc.paired_by_window(
        pd.read_csv(inputs["predictions"]), wide_paths=inputs["paths"], model_path=model,
        rule=None, after_s=5.0, binary_threshold=2)
    summary = swc.summarise(paired, binary_threshold=2)
    titles = swc.window_titles(5.0, 5.0)
    for build in (swc.build_grid, swc.build_cs_plus):
        plain = build(summary, cohort=DS, titles=titles).get_suptitle().splitlines()
        for delay in (None, 0.0):
            assert build(summary, cohort=DS, titles=titles,
                         delay_s=delay).get_suptitle().splitlines() == plain
        shifted = build(summary, cohort=DS, titles=titles, delay_s=1.0).get_suptitle().splitlines()
        assert shifted == plain + [NOTE]
        plt.close("all")
