"""Rescoring trials with the production ordinal model on a different window.

The model scores a trial from features over fixed windows: baseline [0, on),
"during" [on, off) and "after" (30 s from the end of "during"). These tests pin
how a named window moves the "during" slot:

    odor             [on, off)              the production layout
    after            [off, off + after_s)   only the time after odor-off
    odor_plus_after  [on, off + after_s)    odor and the time after it
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.analysis.window_rescore as wr

FPS = 40.0


def _ctx(on=30.0, off=35.0, dur=65.0, fps=FPS):
    return {"trial_odor_on_s": on, "trial_odor_off_s": off,
            "trial_duration_s": dur, "fps": fps}


# ── window bounds ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("name, expected", [
    ("odor", (1200, 1200, 1400)),
    ("after", (1200, 1400, 1600)),
    ("odor_plus_after", (1200, 1200, 1600)),
])
def test_named_windows_move_only_the_during_slot(name, expected):
    w, fps = wr.trial_windows(_ctx(), n_cols=2600, spec=wr.window_spec(name, after_s=5.0))
    assert (w.baseline_end, w.odor_start, w.odor_end) == expected
    assert w.n_frames == 2600 and fps == FPS


def test_after_s_is_a_parameter():
    w, _ = wr.trial_windows(_ctx(on=30, off=60, dur=90), n_cols=3600,
                            spec=wr.window_spec("odor_plus_after", after_s=10.0))
    assert (w.odor_start, w.odor_end) == (1200, 2800)


def test_missing_timing_falls_back_to_the_30_60_protocol():
    ctx = {"trial_odor_on_s": np.nan, "trial_odor_off_s": None, "fps": np.nan}
    w, fps = wr.trial_windows(ctx, n_cols=3600, spec=wr.window_spec("odor"))
    assert (w.baseline_end, w.odor_start, w.odor_end) == (1200, 1200, 2400)
    assert fps == FPS


def test_windows_are_clipped_to_the_recording():
    w, _ = wr.trial_windows(_ctx(on=30, off=35, dur=37), n_cols=2600,
                            spec=wr.window_spec("after", after_s=5.0))
    assert w.n_frames == 1480 and w.odor_end == 1480


def test_unknown_window_name_is_an_error():
    with pytest.raises(ValueError):
        wr.window_spec("sideways")


# ── features ───────────────────────────────────────────────────────────────


def _bump_trace(start_s, stop_s, n=2600, seed=0):
    rng = np.random.default_rng(seed)
    x = 10 + rng.normal(0, 0.5, n)
    x[int(start_s * FPS):int(stop_s * FPS)] += 60
    return x


def _row(trace):
    row = {f"dir_val_{i}": v for i, v in enumerate(trace)}
    row.update(_ctx())
    row.update({"global_max": 80.0, "trimmed_global_min": 9.0})
    return row


def test_a_response_after_odor_off_lands_in_the_after_window_only():
    raw = wr.raw_features(pd.DataFrame([_row(_bump_trace(36, 39))]),
                          wr.window_spec("odor"))
    after = wr.raw_features(pd.DataFrame([_row(_bump_trace(36, 39))]),
                            wr.window_spec("after"))
    assert raw.loc[0, "AUC-During"] == pytest.approx(0.0, abs=1e-6)
    assert after.loc[0, "AUC-During"] > 100


def test_whole_trial_extremes_cover_the_odor_even_in_the_after_window():
    """local_max is a whole-trial feature; skipping the odor would change it."""
    trace = _bump_trace(31, 33)          # response DURING the odor only
    after = wr.raw_features(pd.DataFrame([_row(trace)]), wr.window_spec("after"))
    assert after.loc[0, "local_max"] == pytest.approx(trace.max())


# ── scoring with the real model (skipped where it is absent) ──────────────

MODEL = Path(
    "/home/ramanlab/Documents/cole/VSCode/FlyBehaviorScoring/outputs/"
    "ordinal_scorer/model_ordinal_xgb_v2_both.json"
)


@pytest.mark.skipif(not MODEL.exists(), reason="production model not on this machine")
def test_rescore_returns_ordinal_scores():
    df = pd.DataFrame([_row(_bump_trace(31, 34)), _row(_bump_trace(0, 0))])
    scores = wr.rescore(df, wr.window_spec("odor"), MODEL)
    assert scores.dtype.kind == "i"
    assert set(scores) <= set(range(-1, 6))


# ── the CLI: a predictions CSV with the score swapped ──────────────────────


def test_apply_scores_replaces_score_and_prediction_and_drops_unmatched():
    template = pd.DataFrame({
        "dataset": ["A", "A", "A"], "fly": ["f", "f", "f"], "fly_number": [1, 1, 1],
        "trial_label": ["pretest_1_x", "testing_1_x", "testing_2_y"],
        "score": [0, 0, 0], "prediction": [0, 0, 0], "trial_type": ["pretest", "testing", "testing"],
    })
    rescored = pd.DataFrame({
        "dataset": ["A", "A"], "fly": ["f", "f"], "fly_number": ["1", "1"],
        "trial_label": ["pretest_1_x", "testing_1_x"], "score": [3, 1],
    })
    out = wr.apply_scores(template, rescored, binary_threshold=2, window_label="odor+5s")
    assert out["trial_label"].tolist() == ["pretest_1_x", "testing_1_x"]
    assert out["score"].tolist() == [3, 1]
    assert out["prediction"].tolist() == [1, 0]
    assert set(out["score_window"]) == {"odor+5s"}


# ── the AUC threshold is the pipeline's, from the config ──────────────────


def test_auc_uses_the_threshold_rule_it_is_given():
    """Production AUCs come from fbpipe's ThresholdRule (config_new.yaml), not
    the scorer repo's MAD rule; a higher floor must shrink every AUC."""
    from fbpipe.analysis.threshold import ThresholdRule

    df = pd.DataFrame([_row(_bump_trace(31, 34))])
    low = wr.raw_features(df, wr.window_spec("odor"),
                          rule=ThresholdRule(std_mult=3.0, min_delta=1.0, anchor_s=5.0))
    high = wr.raw_features(df, wr.window_spec("odor"),
                           rule=ThresholdRule(std_mult=3.0, min_delta=40.0, anchor_s=5.0))
    assert high.loc[0, "AUC-During"] < low.loc[0, "AUC-During"]


def test_the_rule_is_read_from_the_wide_block_of_the_config(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "analysis:\n  combined:\n    combined_base:\n      wide:\n"
        "        threshold_std_mult: 4.0\n        threshold_min_delta: 7.0\n"
        "        threshold_anchor_s: 3.0\n"
    )
    rule = wr.threshold_rule_from_config(cfg)
    assert (rule.std_mult, rule.min_delta, rule.anchor_s) == (4.0, 7.0, 3.0)
