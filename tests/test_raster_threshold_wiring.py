"""The rasters must threshold at the rule the rest of the pipeline uses.

``binarized_per_rasters`` shipped with its own defaults -- k = 2, no floor, no
anchor -- while ``config_new.yaml``'s ``combined.combined_base.wide`` block (the
block that drives the AUC-* columns, the red line on the trace figures and the
scoring filters) carries the anchored rule: k = 3, floor 5, anchor 5 s. A raster
built on the defaults binarised at a *different* threshold than the figure it
sits beside, silently.

So the rule is resolved from that same config block, and explicit CLI flags
override it. When neither is available the script REFUSES rather than falling
back to its old defaults -- a silent fallback is the exact failure being fixed.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from fbpipe.analysis.threshold import ThresholdRule  # noqa: E402
from scripts.analysis import binarized_per_rasters as mod  # noqa: E402
from test_binarized_per_rasters import DATASET, _training_frame  # noqa: E402


PIPELINE_BLOCK = """\
analysis:
  combined:
    combined_base:
      wide:
        threshold_std_mult: 3.0
        threshold_min_delta: 5.0
        threshold_anchor_s: 5.0
        threshold_noise_block_s: 5.0
        threshold_noise_pctl: 25.0
"""


def write_cfg(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


# ── resolving the rule from config ────────────────────────────────────────


def test_rule_comes_from_the_pipeline_wide_block(tmp_path):
    rule = mod.resolve_threshold_rule(write_cfg(tmp_path, PIPELINE_BLOCK))
    assert rule == ThresholdRule(
        std_mult=3.0, min_delta=5.0, anchor_s=5.0,
        noise_block_s=5.0, noise_pctl=25.0,
    )


def test_resolved_rule_is_not_the_old_raster_default(tmp_path):
    """The regression this exists to prevent: k=2 / no floor / no anchor."""
    rule = mod.resolve_threshold_rule(write_cfg(tmp_path, PIPELINE_BLOCK))
    assert (rule.std_mult, rule.min_delta, rule.anchor_s) != (2.0, 0.0, None)


def test_the_real_config_new_resolves_to_the_pipeline_rule():
    """Pinned against the operative config, not a fixture: if someone retunes
    the threshold there, the rasters must follow without a code change."""
    cfg = ROOT / "config" / "config_new.yaml"
    if not cfg.is_file():
        pytest.skip("config_new.yaml not present")
    rule = mod.resolve_threshold_rule(cfg)
    assert rule.std_mult == 3.0
    assert rule.min_delta == 5.0
    assert rule.anchor_s == 5.0


def test_noise_keys_are_carried_not_dropped(tmp_path):
    """They were silently ignored before -- compute_theta got its own defaults."""
    cfg = write_cfg(tmp_path, PIPELINE_BLOCK.replace(
        "threshold_noise_pctl: 25.0", "threshold_noise_pctl: 60.0"
    ))
    assert mod.resolve_threshold_rule(cfg).noise_pctl == 60.0


# ── explicit overrides ────────────────────────────────────────────────────


def test_explicit_k_overrides_the_config(tmp_path):
    rule = mod.resolve_threshold_rule(write_cfg(tmp_path, PIPELINE_BLOCK), k=6.0)
    assert rule.std_mult == 6.0
    assert rule.min_delta == 5.0  # untouched keys still come from config


def test_explicit_anchor_can_select_the_legacy_estimator(tmp_path):
    """``--threshold-anchor-s 0`` means "legacy whole-window", not "0 seconds"."""
    rule = mod.resolve_threshold_rule(
        write_cfg(tmp_path, PIPELINE_BLOCK), anchor_s=0.0
    )
    assert rule.anchor_s is None


def test_missing_config_without_overrides_refuses(tmp_path):
    """A silent fallback to the old k=2 default is the bug, not the fix."""
    with pytest.raises(SystemExit, match="threshold"):
        mod.resolve_threshold_rule(tmp_path / "nope.yaml")


def test_config_without_a_threshold_block_refuses(tmp_path):
    with pytest.raises(SystemExit, match="threshold"):
        mod.resolve_threshold_rule(write_cfg(tmp_path, "analysis:\n  combined: {}\n"))


def test_full_overrides_work_without_any_config(tmp_path):
    """Standalone use stays possible -- but only by saying the rule out loud."""
    rule = mod.resolve_threshold_rule(
        tmp_path / "nope.yaml", k=2.0, min_delta=0.0, anchor_s=0.0
    )
    assert rule == ThresholdRule(std_mult=2.0, min_delta=0.0, anchor_s=None)


# ── the rule actually reaches the binarisation ────────────────────────────


def _trace(fps: float = 40.0) -> np.ndarray:
    """30 s of quiet baseline then a modest 8-unit step."""
    rng = np.random.default_rng(0)
    before = rng.normal(10.0, 0.5, int(30 * fps))
    after = rng.normal(18.0, 0.5, int(30 * fps))
    return np.concatenate([before, after])


def test_min_delta_floor_reaches_trial_theta():
    """A quiet baseline collapses k*sigma toward 0; the floor must decide."""
    trace = _trace()
    loose = mod.trial_theta(trace, fps=40.0, baseline_until_s=30.0, k=3.0)
    floored = mod.trial_theta(
        trace, fps=40.0, baseline_until_s=30.0, k=3.0, min_delta=5.0
    )
    assert floored > loose
    assert floored == pytest.approx(
        ThresholdRule(std_mult=3.0, min_delta=5.0).theta(
            trace, fps=40.0, baseline_until_s=30.0
        )
    )


def test_build_trials_uses_the_supplied_rule():
    """The pipeline rule must change the binarisation, not just the metadata."""
    legacy = ThresholdRule(std_mult=2.0)
    pipeline = ThresholdRule(std_mult=3.0, min_delta=5.0, anchor_s=5.0)
    frame = _training_frame()
    a = mod.build_trials(frame, dataset=DATASET, trial_type="training", rule=legacy)
    b = mod.build_trials(frame, dataset=DATASET, trial_type="training", rule=pipeline)
    assert not a.empty and not b.empty
    assert list(a["theta"]) != list(b["theta"])


def test_build_trials_records_the_rule_it_used():
    pipeline = ThresholdRule(std_mult=3.0, min_delta=5.0, anchor_s=5.0)
    t = mod.build_trials(
        _training_frame(), dataset=DATASET, trial_type="training", rule=pipeline
    )
    expected = pipeline.theta(
        np.asarray(t.iloc[0]["trace"], dtype=float),
        fps=float(t.iloc[0]["fps"]),
        baseline_until_s=float(t.iloc[0]["odor_on_s"]),
    )
    assert float(t.iloc[0]["theta"]) == pytest.approx(expected)
