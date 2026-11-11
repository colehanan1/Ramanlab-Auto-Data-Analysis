"""Window rescoring of a model that carries an odor-arrival delay.

The production scorer (FlyBehaviorScoring ``predict-ordinal`` ->
``xgb_ordinal.features_for_model`` -> ``preprocess_features``) scores a model
saved with ``fbr_odor_delay_s = d`` from the trace alone
(``window_feature_frame``), on the layout of ``window_features.trial_windows``:

    baseline  [0, on)            valve-on; never moves
    transit   [on, on + d)       left out of every value, whole-trial extremes too
    odor      [on + d, off + d)
    post      [off + d, ...)     AUC-After over its first 30 s

with the scorer's own AUC threshold (baseline median + 3 MAD sigma; the wide
table's AUC-* columns are never read) and only ``global_max`` /
``trimmed_global_min`` taken from the table. ``predict_ordinal_scores`` then
applies the model's stored two-guard (a 2 the ensemble does not back -> 1).

Pinned here, with every expectation built by the scorer's own functions:
  * the 'odor' window reproduces that path exactly: windows, raw features,
    model features and scores, through the real ``predict-ordinal`` CLI too;
  * the moved windows shift by d the way production shifts the odor window:
    both valve references move by d, the baseline does not, the transit stays
    out, and the whole-trial values stay production's on every window;
  * the guard is applied exactly as production applies it, on every window;
  * delay-free models are rescored exactly as before (goldens captured from
    the pre-fix module: tests/golden/window_rescore_delay_free/).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

import _window_rescore_fixtures as fx  # noqa: E402
import scripts.analysis.window_rescore as wr  # noqa: E402
from flybehavior_response import xgb_ordinal as xo  # noqa: E402
from flybehavior_response.window_features import (  # noqa: E402
    compute_window_features,
    windows_for_ctx,
)

AFTER_S = 5.0
WINDOWS = ("odor", "after", "odor_plus_after")
GOLDEN = ROOT / "tests" / "golden" / "window_rescore_delay_free" / "golden.json"
SCORER = Path("/home/ramanlab/Documents/cole/VSCode/FlyBehaviorScoring/outputs/ordinal_scorer")
V2_MODEL = SCORER / "model_ordinal_xgb_v2_both.json"   # delay-free
CONFIG_NEW = ROOT / "config" / "config_new.yaml"

#: The deployed recipe's features (v2-auc-logratio-delay1s), as stored on it.
DEPLOYED_FEATURES = ([name for name, _ in xo.AUC_LOG_RATIOS.values()]
                     + xo.ENGINEERED_FEATURES[2:] + xo.SIGNAL_FEATURES)
WHOLE_TRIAL = ["local_min", "local_max", "local_max_over_global_min"]
BASELINE_ONLY = ["local_min_before", "local_max_before", "AUC-Before"]
PER_FLY = ["global_max", "trimmed_global_min"]


def _deployed_model() -> Path | None:
    """The model config_new.yaml deploys, else the 2026-11-09 guard60 model."""
    candidates = []
    if CONFIG_NEW.exists():
        cfg = yaml.safe_load(CONFIG_NEW.read_text()) or {}
        path = (cfg.get("reaction_prediction") or {}).get("model_path")
        if path:
            candidates.append(Path(path))
    candidates.append(SCORER / "model_ordinal_xgb_v2d1_both_20261109_guard60.json")
    return next((p for p in candidates if p.exists()), None)


DEPLOYED = _deployed_model()


def _spec(name: str) -> wr.WindowSpec:
    return wr.window_spec(name, AFTER_S)


def _cfg(delay: float | None = 1.0) -> xo.ScoringConfig:
    return xo.ScoringConfig(
        recipe="v2-auc-logratio-delay1s", features=list(DEPLOYED_FEATURES),
        fill_values={"global_max": 120.0, "local_max_over_global_min": 3.0},
        auc_eps=2.854, odor_delay_s=delay,
    )


@pytest.fixture(scope="module")
def rows() -> pd.DataFrame:
    return fx.wide_rows()


@pytest.fixture(scope="module")
def traces(rows) -> np.ndarray:
    return rows[[c for c in rows.columns if c.startswith("dir_val_")]].to_numpy(dtype=float)


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN.read_text())


@pytest.fixture
def tiny_model(tmp_path) -> Path:
    """Scores 2 everywhere; a 1 s delay; a 60 % two-guard on peak_z."""
    return fx.build_tiny_model(tmp_path / "tiny")


def _unguarded(booster, X: pd.DataFrame) -> np.ndarray:
    margins = booster.predict(xgb.DMatrix(X), output_margin=True).reshape(-1, xo.N_CLASSES)
    return xo.ORIGINAL_CLASSES[margins.argmax(axis=1)]


# ── windows ────────────────────────────────────────────────────────────────

EDGE_CTX = [
    {"trial_odor_on_s": 30.0, "trial_odor_off_s": 60.0, "trial_duration_s": 90.0, "fps": 40.0},
    {"trial_odor_on_s": 30.0, "trial_odor_off_s": 35.0, "trial_duration_s": 65.0, "fps": 40.0},
    {"trial_odor_on_s": np.nan, "trial_odor_off_s": None, "trial_duration_s": np.nan, "fps": np.nan},
    # ends inside the odor: the later bounds clamp to the recording
    {"trial_odor_on_s": 30.0, "trial_odor_off_s": 35.0, "trial_duration_s": 35.5, "fps": 40.0},
    {"trial_odor_on_s": 10.0, "trial_odor_off_s": 20.0, "trial_duration_s": 40.0, "fps": 20.0},
    # off-grid times and fps: rounding must match too
    {"trial_odor_on_s": 30.02, "trial_odor_off_s": 35.013, "trial_duration_s": 65.0, "fps": 39.97},
    {"trial_odor_on_s": 0.0, "trial_odor_off_s": 0.01, "trial_duration_s": 0.2, "fps": 40.0},
]


@pytest.mark.parametrize("delay", [1.0, 0.5, 0.0])
@pytest.mark.parametrize("ctx", EDGE_CTX)
def test_the_odor_window_of_a_delay_model_is_productions(ctx, delay):
    assert wr.trial_windows(ctx, 3600, _spec("odor"), delay_s=delay) == \
        windows_for_ctx(ctx, 3600, delay)


@pytest.mark.parametrize("delay, expected", [
    (1.0, {"odor": (1200, 1240, 1440), "after": (1200, 1440, 1640),
           "odor_plus_after": (1200, 1240, 1640)}),
    (0.5, {"odor": (1200, 1220, 1420), "after": (1200, 1420, 1620),
           "odor_plus_after": (1200, 1220, 1620)}),
])
def test_both_valve_references_move_by_the_delay_and_the_baseline_does_not(delay, expected):
    """odor [on+d, off+d), after [off+d, off+d+a), odor+after [on+d, off+d+a)."""
    ctx = {"trial_odor_on_s": 30.0, "trial_odor_off_s": 35.0,
           "trial_duration_s": 65.0, "fps": 40.0}
    for name, bounds in expected.items():
        w, fps = wr.trial_windows(ctx, 2600, _spec(name), delay_s=delay)
        assert (w.baseline_end, w.odor_start, w.odor_end) == bounds
        assert w.n_frames == 2600 and fps == 40.0


@pytest.mark.parametrize("name", WINDOWS)
def test_delay_free_windows_are_unchanged(rows, traces, golden, name):
    for i in range(len(rows)):
        for kwargs in ({}, {"delay_s": None}):
            w, fps = wr.trial_windows(rows.iloc[i], traces.shape[1], _spec(name), **kwargs)
            assert [w.baseline_end, w.odor_start, w.odor_end, w.n_frames, fps] == \
                golden["windows"][name][i]


# ── the 'odor' window is the production score ──────────────────────────────


def test_delay_odor_raw_features_are_productions(rows):
    expected = xo.window_feature_frame(rows, odor_delay_s=1.0)
    got = wr.raw_features(rows, _spec("odor"), rule=fx.pipeline_rule(), odor_delay_s=1.0)
    pd.testing.assert_frame_equal(got, expected, check_exact=True)


@pytest.mark.parametrize("delay", [1.0, 0.5, 0.0])
def test_delay_model_features_are_productions(rows, delay):
    """``fbr_odor_delay_s = 0`` is still the trace-only path in production."""
    cfg = _cfg(delay)
    got = wr.model_features(rows, _spec("odor"), cfg, rule=fx.pipeline_rule())
    pd.testing.assert_frame_equal(got, xo.features_for_model(rows, cfg), check_exact=True)


def test_odor_window_scores_match_the_predict_ordinal_cli(rows, tiny_model, tmp_path):
    """The pipeline's own scoring call, on the very table it would parse."""
    from flybehavior_response import cli

    data_csv, out_csv = tmp_path / "wide.csv", tmp_path / "predictions.csv"
    rows.to_csv(data_csv, index=False)
    cli.main(["predict-ordinal", "--data-csv", str(data_csv), "--model-path", str(tiny_model),
              "--output-csv", str(out_csv)])
    production = pd.read_csv(out_csv)["score"].tolist()
    got = wr.rescore(pd.read_csv(data_csv), _spec("odor"), tiny_model, rule=fx.pipeline_rule())
    assert got.tolist() == production
    assert set(production) == {1, 2}      # the guard kept some 2s and demoted others


@pytest.mark.skipif(DEPLOYED is None, reason="deployed scorer not on this machine")
def test_the_deployed_models_odor_window_is_its_production_score(rows):
    booster = xo.load_xgb_model(DEPLOYED)
    cfg = xo.scoring_config(booster)
    if cfg.odor_delay_s is None:
        pytest.skip("the deployed model carries no odor delay")
    production_features = xo.features_for_model(rows, cfg)
    pd.testing.assert_frame_equal(
        wr.model_features(rows, _spec("odor"), cfg, rule=fx.pipeline_rule()),
        production_features, check_exact=True)
    expected = xo.predict_ordinal_scores(booster, production_features)
    got = wr.rescore(rows, _spec("odor"), DEPLOYED, rule=fx.pipeline_rule())
    assert got.tolist() == expected.tolist()
    for name in ("after", "odor_plus_after"):
        moved = wr.rescore(rows, _spec(name), DEPLOYED, rule=fx.pipeline_rule())
        assert moved.dtype.kind == "i" and set(moved) <= set(range(-1, 6))


# ── the moved windows ──────────────────────────────────────────────────────


@pytest.mark.parametrize("name", WINDOWS)
def test_whole_trial_baseline_and_per_fly_values_are_productions_on_every_window(rows, name):
    keep = PER_FLY + WHOLE_TRIAL + BASELINE_ONLY
    production = xo.window_feature_frame(rows, odor_delay_s=1.0)[keep]
    got = wr.raw_features(rows, _spec(name), odor_delay_s=1.0)[keep]
    pd.testing.assert_frame_equal(got, production, check_exact=True)


def test_the_transit_second_stays_out_and_the_odor_stays_in(rows, traces):
    after = wr.raw_features(rows, _spec("after"), odor_delay_s=1.0)
    spike = fx.ROW_NAMES.index("transit_spike")          # +80 px in [30.2, 30.8) s
    assert np.nanmax(traces[spike]) > 85
    assert after.loc[spike, "local_max"] < 20
    response = fx.ROW_NAMES.index("standard_odor_response")   # inside the odor only
    assert after.loc[response, "local_max"] == np.nanmax(traces[response])


def test_the_second_after_valve_off_is_odor_for_a_delay_model(rows):
    """A response in [off, off + d) is odor for the delay model, 'after' without one."""
    i = fx.ROW_NAMES.index("valve_off_second")
    odor = wr.raw_features(rows, _spec("odor"), odor_delay_s=1.0)
    after = wr.raw_features(rows, _spec("after"), odor_delay_s=1.0)
    assert odor.loc[i, "AUC-During"] > 30 and after.loc[i, "AUC-During"] < 1
    odor0 = wr.raw_features(rows, _spec("odor"), rule=fx.pipeline_rule())
    after0 = wr.raw_features(rows, _spec("after"), rule=fx.pipeline_rule())
    assert odor0.loc[i, "AUC-During"] < 1 and after0.loc[i, "AUC-During"] > 30


@pytest.mark.parametrize("name", ["after", "odor_plus_after"])
def test_moved_window_values_are_the_scorers_own_on_the_moved_slot(rows, traces, name):
    got = wr.raw_features(rows, _spec(name), odor_delay_s=1.0)
    moved = ["local_min_during", "local_max_during_over_global_min", "AUC-During", "AUC-After"]
    for i in range(len(rows)):
        w, fps = wr.trial_windows(rows.iloc[i], traces.shape[1], _spec(name), delay_s=1.0)
        tmin = float(rows.loc[i, "trimmed_global_min"])
        expected = compute_window_features(traces[i], fps, w, trimmed_global_min=tmin)
        expected.update(xo.compute_signal_features(
            traces[i], fps=fps, odor_on_idx=w.odor_start, odor_off_idx=w.odor_end,
            n_frames=w.n_frames, baseline_end_idx=w.baseline_end))
        for key in moved + xo.SIGNAL_FEATURES:
            np.testing.assert_array_equal(got.loc[i, key], expected[key], err_msg=f"{i} {key}")


@pytest.mark.parametrize("name", WINDOWS)
def test_a_delay_model_keeps_its_own_auc_threshold(rows, name):
    """Production never reads the table's AUC columns for a delay model, so the
    pipeline's threshold rule (the CLI's --config) must not change anything."""
    from fbpipe.analysis.threshold import ThresholdRule

    own = wr.raw_features(rows, _spec(name), odor_delay_s=1.0)
    for rule in (fx.pipeline_rule(), ThresholdRule(std_mult=3.0, min_delta=40.0, anchor_s=5.0)):
        pd.testing.assert_frame_equal(
            wr.raw_features(rows, _spec(name), rule=rule, odor_delay_s=1.0), own,
            check_exact=True)


# ── the two-guard ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", WINDOWS)
def test_the_guard_is_applied_on_every_window_as_production_applies_it(rows, tiny_model, name):
    booster = xo.load_xgb_model(tiny_model)
    cfg = xo.scoring_config(booster)
    X = wr.model_features(rows, _spec(name), cfg)[booster.feature_names]
    unguarded = _unguarded(booster, X)
    expected = xo._apply_two_guard(unguarded, X, xo.two_guard(booster))
    got = wr.rescore(rows, _spec(name), tiny_model, rule=fx.pipeline_rule())
    assert got.tolist() == expected.tolist()
    assert set(unguarded) == {2} and set(got) == {1, 2}


def test_a_delay_free_guarded_model_is_guarded_too(rows, tmp_path):
    model = fx.build_tiny_model(tmp_path / "m", odor_delay_s=None)
    booster = xo.load_xgb_model(model)
    cfg = xo.scoring_config(booster)
    X = xo._finish(wr.raw_features(rows, _spec("odor"), rule=fx.pipeline_rule()),
                   cfg.features, cfg.fill_values, cfg.auc_eps)[booster.feature_names]
    expected = xo._apply_two_guard(_unguarded(booster, X), X, xo.two_guard(booster))
    got = wr.rescore(rows, _spec("odor"), model, rule=fx.pipeline_rule())
    assert got.tolist() == expected.tolist() and set(got) == {1, 2}


# ── delay-free models: exactly as before ───────────────────────────────────


@pytest.mark.parametrize("name", WINDOWS)
@pytest.mark.parametrize("tag", ["none", "pipeline"])
def test_delay_free_raw_features_are_bit_identical_to_before(rows, golden, name, tag):
    rule = None if tag == "none" else fx.pipeline_rule()
    expected = np.array(golden["raw"][f"{name}|{tag}"], dtype=float)
    for kwargs in ({}, {"odor_delay_s": None}):
        got = wr.raw_features(rows, _spec(name), rule=rule, **kwargs)
        assert list(got.columns) == golden["columns"]
        np.testing.assert_array_equal(got.to_numpy(dtype=float), expected)


@pytest.mark.skipif(not V2_MODEL.exists(), reason="delay-free production model not on this machine")
@pytest.mark.parametrize("name", WINDOWS)
def test_delay_free_model_scores_are_unchanged(rows, golden, name):
    if golden["scores_v2"] is None:
        pytest.skip("golden captured without the delay-free model")
    got = wr.rescore(rows, _spec(name), V2_MODEL, rule=fx.pipeline_rule())
    assert got.tolist() == golden["scores_v2"][name]


# ── labels, tables and the CLI ─────────────────────────────────────────────


def test_labels_name_the_delay_only_when_there_is_one():
    spec = _spec("odor_plus_after")
    assert wr.window_label(spec, None) == "odor + 5 s after" == spec.label
    assert wr.window_label(spec, 0.0) == "odor + 5 s after"
    assert wr.window_label(spec, 1.0) == "odor + 5 s after (+1 s arrival delay)"
    assert wr.window_label(_spec("after"), 0.5) == "5 s after odor (+0.5 s arrival delay)"
    assert wr.window_label(_spec("odor"), 1.0) == "odor (+1 s arrival delay)"


def test_the_models_delay_is_read_from_the_model_file(tmp_path):
    assert wr.model_odor_delay_s(fx.build_tiny_model(tmp_path / "a", odor_delay_s=1.0)) == 1.0
    assert wr.model_odor_delay_s(fx.build_tiny_model(tmp_path / "b", odor_delay_s=None)) is None


def _tables(tmp_path: Path, dataset: str = "DS-A"):
    pre, post = fx.wide_rows(dataset, "pretest"), fx.wide_rows(dataset, "testing")
    pre_p, post_p = tmp_path / "pre.parquet", tmp_path / "post.parquet"
    pre.to_parquet(pre_p, index=False)
    post.to_parquet(post_p, index=False)
    wide = pd.concat([pre, post], ignore_index=True)
    template = wide[wr.KEY + ["trial_type"]].assign(score=0, prediction=0)
    return wide, [pre_p, post_p], template


def test_rescore_predictions_scores_a_delay_model_as_production_does(tiny_model, tmp_path):
    wide, paths, template = _tables(tmp_path)
    out = wr.rescore_predictions(template, paths, tiny_model, _spec("odor"),
                                 binary_threshold=2, rule=fx.pipeline_rule())
    booster = xo.load_xgb_model(tiny_model)
    expected = xo.predict_ordinal_scores(booster, xo.features_for_model(wide, xo.scoring_config(booster)))
    assert out["trial_label"].tolist() == template["trial_label"].tolist()
    assert out["score"].tolist() == expected.tolist()
    assert out["prediction"].tolist() == (expected >= 2).astype(int).tolist()
    assert set(out["score_window"]) == {"odor (+1 s arrival delay)"}


def test_the_cli_exits_0_on_a_delay_model(tiny_model, tmp_path):
    wide, paths, template = _tables(tmp_path)
    predictions, out_csv, config = (tmp_path / "model_predictions.csv",
                                    tmp_path / "rescored.csv", tmp_path / "config.yaml")
    template.to_csv(predictions, index=False)
    config.write_text(yaml.safe_dump(
        {"analysis": {"combined": {"combined_base": {"wide": fx.PIPELINE_WIDE_KEYS}}}}))
    rc = wr.main([
        "--predictions-csv", str(predictions), "--wide-csv", *map(str, paths),
        "--model-path", str(tiny_model), "--config", str(config),
        "--window", "odor_plus_after", "--after-s", "5", "--binary-threshold", "2",
        "--paired-only", "--output-csv", str(out_csv),
    ])
    assert rc == 0
    out = pd.read_csv(out_csv)
    booster = xo.load_xgb_model(tiny_model)
    X = wr.model_features(wide, _spec("odor_plus_after"), xo.scoring_config(booster))
    assert out["score"].tolist() == xo.predict_ordinal_scores(booster, X).tolist()
    assert set(out["score_window"]) == {"odor + 5 s after (+1 s arrival delay)"}
