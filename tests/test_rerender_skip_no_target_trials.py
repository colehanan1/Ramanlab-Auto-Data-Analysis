"""Tests for graceful skipping of training envelope blocks on testing-only data.

RandomPanel datasets are testing-only (every trial overridden to ``testing``),
so the training-targeted envelope matrices are empty. The score-annotation
re-render must skip those blocks instead of aborting the whole pipeline, while
still rendering the testing blocks.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.pipeline.run_workflows as rw  # noqa: E402
from scripts.analysis.envelope_visuals import NoTargetTrialsError  # noqa: E402


def test_no_target_trials_error_is_runtime_error():
    assert issubclass(NoTargetTrialsError, RuntimeError)


def _patch_config(monkeypatch):
    """Make _envelope_plot_config return a namespace echoing the entry's trial_type."""

    def fake_config(forced):
        return (
            SimpleNamespace(
                trial_type=forced.get("trial_type", "testing"),
                out_dir=Path("/tmp/out"),
            ),
            None,
        )

    monkeypatch.setattr(rw, "_envelope_plot_config", fake_config)


def test_rerender_skips_block_with_no_target_trials(monkeypatch):
    _patch_config(monkeypatch)
    rendered = []

    def fake_generate(config):
        if config.trial_type == "training":
            raise NoTargetTrialsError("No training trials found in matrix; ...")
        rendered.append(config.trial_type)

    monkeypatch.setattr(rw, "generate_envelope_plots", fake_generate)

    block = [{"trial_type": "training"}, {"trial_type": "testing"}]
    # Must NOT raise — training entry is skipped, testing entry renders.
    rw._rerender_envelope_block_with_scores(block, "test block")
    assert rendered == ["testing"]


def test_rerender_propagates_other_runtime_errors(monkeypatch):
    _patch_config(monkeypatch)

    def fake_generate(config):
        raise RuntimeError("some unrelated failure")

    monkeypatch.setattr(rw, "generate_envelope_plots", fake_generate)

    with pytest.raises(RuntimeError, match="some unrelated failure"):
        rw._rerender_envelope_block_with_scores([{"trial_type": "testing"}], "test block")
