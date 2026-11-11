"""The config's AUC rule must reach the feature builder for delay-free models.

A delay-free model is scored on the pipeline's wide-table AUC threshold rule
(``analysis.combined.combined_base.wide`` in the config): the scorer repo's own
MAD rule reproduces production on only 92.5 % of scores (window_rescore
docstring). The 2026-11-10 delay fix rewrote the rescore_predictions -> _scores
-> model_features -> raw_features chain, and a mutant that dropped ``rule`` on
the way passed every other test: the tiny test model's scores do not depend on
the AUC values. These tests spy on ``raw_features`` instead of comparing scores.
A delay model keeps its own threshold, so it must receive the delay, not use the
rule (test_window_rescore_delay pins what it computes).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

import _window_rescore_fixtures as fx  # noqa: E402
import scripts.analysis.score_window_comparison as swc  # noqa: E402
import scripts.analysis.window_rescore as wr  # noqa: E402


@pytest.fixture
def calls(monkeypatch) -> list[dict]:
    """Every raw_features call's ``rule`` and ``odor_delay_s``."""
    seen: list[dict] = []
    real = wr.raw_features

    def spy(df, spec, *, rule=None, odor_delay_s=None):
        seen.append({"rule": rule, "odor_delay_s": odor_delay_s, "window": spec.name})
        return real(df, spec, rule=rule, odor_delay_s=odor_delay_s)

    monkeypatch.setattr(wr, "raw_features", spy)
    return seen


@pytest.fixture
def tables(tmp_path):
    rows = fx.wide_rows()
    wide = tmp_path / "wide.csv"
    rows.to_csv(wide, index=False)
    template = rows[wr.KEY + ["trial_type"]].assign(score=0, prediction=0)
    return rows, wide, template


def _model(tmp_path, delay):
    return fx.build_tiny_model(tmp_path / f"model_{delay}", odor_delay_s=delay, guard_min_agree=None)


def _config(tmp_path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(
        {"analysis": {"combined": {"combined_base": {"wide": dict(fx.PIPELINE_WIDE_KEYS)}}}}))
    return cfg


@pytest.mark.parametrize("window", ["odor", "after", "odor_plus_after"])
def test_rescore_predictions_hands_the_rule_to_a_delay_free_model(tmp_path, calls, tables, window):
    rows, wide, template = tables
    rule = fx.pipeline_rule()
    out = wr.rescore_predictions(template, [wide], _model(tmp_path, None),
                                 wr.window_spec(window, 5.0), binary_threshold=2, rule=rule)
    assert len(out) == len(rows)
    assert calls, "raw_features was never called"
    assert all(c["rule"] is rule and c["odor_delay_s"] is None for c in calls)


def test_the_cli_builds_the_rule_from_the_config_and_hands_it_over(tmp_path, calls, tables):
    rows, wide, template = tables
    preds, out_csv, cfg = tmp_path / "preds.csv", tmp_path / "out.csv", _config(tmp_path)
    template.to_csv(preds, index=False)
    rc = wr.main(["--predictions-csv", str(preds), "--wide-csv", str(wide),
                  "--model-path", str(_model(tmp_path, None)), "--config", str(cfg),
                  "--window", "odor_plus_after", "--after-s", "5", "--output-csv", str(out_csv)])
    assert rc == 0 and len(pd.read_csv(out_csv)) == len(rows)
    expected = wr.threshold_rule_from_config(cfg)
    assert expected == fx.pipeline_rule()
    assert calls and all(c["rule"] == expected for c in calls)


def test_the_window_comparison_hands_the_rule_to_every_window(tmp_path, calls, tables):
    _, wide, template = tables
    rule = fx.pipeline_rule()
    swc.paired_by_window(template, wide_paths=[wide], model_path=_model(tmp_path, None),
                         rule=rule, after_s=5.0, binary_threshold=2)
    assert {c["window"] for c in calls} == set(swc.WINDOWS)
    assert all(c["rule"] is rule and c["odor_delay_s"] is None for c in calls)


def test_a_delay_model_gets_its_delay_and_keeps_its_own_threshold(tmp_path, calls, tables):
    rows, wide, template = tables
    out = wr.rescore_predictions(template, [wide], _model(tmp_path, 1.0),
                                 wr.window_spec("odor", 5.0), binary_threshold=2,
                                 rule=fx.pipeline_rule())
    assert len(out) == len(rows)
    assert calls and all(c["odor_delay_s"] == 1.0 for c in calls)
