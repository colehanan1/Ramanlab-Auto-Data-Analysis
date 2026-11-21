"""Window rescoring on only the first N s of the odor (user 2026-11-17).

The model reads a trial through baseline, odor and the 30 s after the odor. For
``odor_first`` the odor is cut to its first N s: the rest of the odor comes out of
the trace and the post-odor period follows straight on, so the model scores the
trial as production would score one whose odor lasted N s, made of these frames.
Its "after" values stay the production post-odor window: moving it to follow the
shortened odor would put the cut odor back in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

import _window_rescore_fixtures as fx  # noqa: E402
import scripts.analysis.window_rescore as wr  # noqa: E402
from flybehavior_response import xgb_ordinal as xo  # noqa: E402

FIRST = wr.window_spec("odor_first", odor_s=10.0)
STD = {"trial_odor_on_s": 30.0, "trial_odor_off_s": 60.0, "trial_duration_s": 90.0,
       "fps": 40.0}


def test_the_window_is_named_by_its_length():
    assert FIRST.label == "first 10 s of odor"
    assert wr.window_spec("odor_first", odor_s=7.5).label == "first 7.5 s of odor"
    assert wr.window_label(FIRST, 1.0) == "first 10 s of odor (+1 s arrival delay)"
    assert FIRST.cut_odor and not wr.window_spec("odor").cut_odor


@pytest.mark.parametrize("delay, keep, resume", [(1.0, 1640, 2440), (None, 1600, 2400)])
def test_the_rest_of_the_odor_is_cut_and_the_post_odor_follows(delay, keep, resume):
    x = np.arange(3600, dtype=float)
    y, ctx = wr.first_odor_trial(x, STD, 10.0, delay_s=delay)
    assert np.array_equal(y, np.concatenate([x[:keep], x[resume:]]))
    assert ctx["trial_odor_on_s"] == 30.0 and ctx["trial_odor_off_s"] == 40.0
    w, _ = wr.trial_windows(ctx, len(y), wr.window_spec("odor"), delay_s=delay)
    assert (w.baseline_end, w.odor_end, w.n_frames) == (1200, keep, len(y))


def test_padding_past_the_recording_is_not_kept():
    x = np.concatenate([np.arange(2600, dtype=float), np.full(1000, np.nan)])
    ctx = {**STD, "trial_duration_s": 65.0}
    y, _ = wr.first_odor_trial(x, ctx, 10.0, delay_s=1.0)
    assert np.array_equal(y, np.concatenate([x[:1640], x[2440:2600]]))


def test_an_odor_no_longer_than_n_is_left_whole():
    x = np.arange(2600, dtype=float)
    ctx = {**STD, "trial_odor_off_s": 35.0, "trial_duration_s": 65.0}
    y, new = wr.first_odor_trial(x, ctx, 10.0, delay_s=1.0)
    assert np.array_equal(y, x)
    assert new["trial_odor_off_s"] == 35.0


def _late_and_early(where: tuple[float, float]) -> pd.DataFrame:
    """Two copies of the standard trial; the second has a bump at ``where`` s."""
    rows = fx.wide_rows().iloc[[0, 0]].reset_index(drop=True)
    cols = [c for c in rows.columns if c.startswith("dir_val_")]
    a, b = (int(round(t * 40.0)) for t in where)
    rows.loc[1, cols[a:b]] = rows.loc[1, cols[a:b]].to_numpy(dtype=float) + 70.0
    return rows


@pytest.mark.parametrize("delay", [1.0, None])
def test_a_response_after_the_first_n_s_of_odor_is_not_seen(delay):
    rows = _late_and_early((45.0, 55.0))           # odor seconds 15-25
    rule = None if delay is not None else fx.pipeline_rule()
    f = wr.raw_features(rows, FIRST, rule=rule, odor_delay_s=delay)
    pd.testing.assert_series_equal(f.iloc[0], f.iloc[1], check_names=False)
    full = wr.raw_features(rows, wr.window_spec("odor"), rule=rule, odor_delay_s=delay)
    assert full.loc[1, "peak_z"] > full.loc[0, "peak_z"] + 10


@pytest.mark.parametrize("delay", [1.0, None])
def test_a_response_in_the_first_n_s_of_odor_is_seen(delay):
    rows = _late_and_early((33.0, 38.0))
    rule = None if delay is not None else fx.pipeline_rule()
    f = wr.raw_features(rows, FIRST, rule=rule, odor_delay_s=delay)
    assert f.loc[1, "peak_z"] > f.loc[0, "peak_z"] + 10
    assert f.loc[1, "AUC-During"] > f.loc[0, "AUC-During"]


@pytest.mark.parametrize("delay", [1.0, None])
def test_the_post_odor_window_is_productions(delay):
    rows = _late_and_early((63.0, 70.0))           # after valve-off (+ delay)
    rule = None if delay is not None else fx.pipeline_rule()
    f = wr.raw_features(rows, FIRST, rule=rule, odor_delay_s=delay)
    full = wr.raw_features(rows, wr.window_spec("odor"), rule=rule, odor_delay_s=delay)
    assert f["AUC-After"].tolist() == pytest.approx(full["AUC-After"].tolist())
    assert f.loc[1, "AUC-After"] > f.loc[0, "AUC-After"]


def test_every_value_is_productions_on_the_cut_trial():
    rows = fx.wide_rows()
    f = wr.raw_features(rows, FIRST, odor_delay_s=1.0)
    cols = [c for c in rows.columns if c.startswith("dir_val_")]
    ctx_cols = [c for c in rows.columns if not c.startswith("dir_val_")]
    for i in range(len(rows)):
        y, ctx = wr.first_odor_trial(rows.loc[i, cols].to_numpy(dtype=float),
                                     rows.loc[i, ctx_cols], 10.0, delay_s=1.0)
        want = xo._delayed_raw(y, ctx, 40.0, 1.0)
        got = f.iloc[i]
        for key, value in want.items():
            assert got[key] == pytest.approx(value, nan_ok=True), (rows.loc[i, "fly"], key)


def test_the_cli_scores_the_first_n_s(tmp_path):
    model = fx.build_tiny_model(tmp_path / "tiny")
    pre, post = fx.wide_rows("DS-A", "pretest"), fx.wide_rows("DS-A", "testing")
    paths = [tmp_path / "pre.parquet", tmp_path / "post.parquet"]
    pre.to_parquet(paths[0], index=False)
    post.to_parquet(paths[1], index=False)
    wide = pd.concat([pre, post], ignore_index=True)
    template = wide[wr.KEY + ["trial_type"]].assign(score=0, prediction=0)
    predictions, out_csv, config = (tmp_path / "p.csv", tmp_path / "r.csv", tmp_path / "c.yaml")
    template.to_csv(predictions, index=False)
    config.write_text(yaml.safe_dump(
        {"analysis": {"combined": {"combined_base": {"wide": fx.PIPELINE_WIDE_KEYS}}}}))
    rc = wr.main(["--predictions-csv", str(predictions), "--wide-csv", *map(str, paths),
                  "--model-path", str(model), "--config", str(config),
                  "--window", "odor_first", "--odor-s", "10", "--binary-threshold", "2",
                  "--paired-only", "--output-csv", str(out_csv)])
    assert rc == 0
    out = pd.read_csv(out_csv)
    booster = xo.load_xgb_model(model)
    X = wr.model_features(wide, FIRST, xo.scoring_config(booster))
    assert out["score"].tolist() == xo.predict_ordinal_scores(booster, X).tolist()
    assert set(out["score_window"]) == {"first 10 s of odor (+1 s arrival delay)"}
