"""The conditioning-trial scores must be rebuilt when their input changes.

``predict_reactions`` scores only ``testing`` and ``pretest`` trials, so the
CONDITIONING trials are scored by ``ensure_training_predictions`` into a
sidecar, ``model_predictions_training.csv``. Two things were wrong with it:

1. **Existence was the only cache check.** ``if output_csv.exists() and not
   rescore: return output_csv`` — no comparison against the wide table it was
   derived from. The sidecar predated the ``*-Sensitivity-*`` cohorts, so it
   held zero rows for them and every conditioning bar rendered ``n=0`` while
   the pre-test bar beside it showed n=13. Wrong numbers, no error.

2. **No pipeline step produced it.** It existed only as a side-effect of
   hand-running the figure script, so nothing was responsible for refreshing it
   as new cohorts landed.

A missing cohort is the dangerous failure here: it renders as a confident zero
rather than a crash, so freshness has to be checked, not assumed.
"""

from pathlib import Path

import pandas as pd
import pytest

import scripts.analysis.naive_vs_trial_score_bars as nvt


def _wide(tmp_path, name="training_wide.parquet"):
    rows = []
    for fn in (1, 2):
        for i in range(1, 4):
            row = {
                "dataset": "Hex-Sensitivity-24-0.1", "fly": "b1", "fly_number": fn,
                "trial_type": "training", "trial_label": f"training_{i}_hexanol",
                "fps": 40.0, "trace_len": 50,
            }
            row.update({f"dir_val_{j}": 1.0 for j in range(50)})
            rows.append(row)
    path = tmp_path / name
    pd.DataFrame(rows).to_parquet(path)
    return path


def _sidecar(tmp_path, name="model_predictions_training.csv"):
    path = tmp_path / name
    pd.DataFrame([{
        "dataset": "Hex-Control-24-0.1", "fly": "b1", "fly_number": 1,
        "trial_label": "training_1_hexanol", "score": 2, "trial_type": "training",
    }]).to_csv(path, index=False)
    return path


def _touch_after(path: Path, reference: Path) -> None:
    """Make *path* strictly newer than *reference*."""
    ref = reference.stat().st_mtime
    import os
    os.utime(path, (ref + 10, ref + 10))


# ── staleness ─────────────────────────────────────────────────────────────


def test_a_sidecar_older_than_its_input_is_rebuilt(tmp_path):
    """The whole bug: a cache built before new cohorts landed kept being served."""
    wide = _wide(tmp_path)
    out = _sidecar(tmp_path)
    _touch_after(wide, out)          # input is NEWER than the cache

    assert nvt.training_predictions_are_stale(out, wide) is True


def test_a_sidecar_newer_than_its_input_is_reused(tmp_path):
    """Rescoring on every run would be wasteful; only staleness triggers it."""
    wide = _wide(tmp_path)
    out = _sidecar(tmp_path)
    _touch_after(out, wide)          # cache is NEWER than the input

    assert nvt.training_predictions_are_stale(out, wide) is False


def test_a_missing_sidecar_is_stale(tmp_path):
    wide = _wide(tmp_path)
    assert nvt.training_predictions_are_stale(tmp_path / "nope.csv", wide) is True


def test_a_missing_input_is_not_treated_as_stale(tmp_path):
    """Nothing to rebuild FROM; the caller raises its own clearer error."""
    out = _sidecar(tmp_path)
    assert nvt.training_predictions_are_stale(out, tmp_path / "nope.parquet") is False


def test_ensure_rebuilds_when_the_input_is_newer(tmp_path):
    wide = _wide(tmp_path)
    out = _sidecar(tmp_path)
    _touch_after(wide, out)

    called = []
    nvt.ensure_training_predictions(
        wide, Path(__file__), out, runner=lambda *a, **k: called.append(a)
    )
    assert called, "stale sidecar was served instead of being rebuilt"


def test_ensure_reuses_a_fresh_sidecar(tmp_path):
    wide = _wide(tmp_path)
    out = _sidecar(tmp_path)
    _touch_after(out, wide)

    called = []
    nvt.ensure_training_predictions(
        wide, Path(__file__), out, runner=lambda *a, **k: called.append(a)
    )
    assert not called, "a fresh sidecar was needlessly rescored"


def test_rescore_still_forces_a_rebuild(tmp_path):
    wide = _wide(tmp_path)
    out = _sidecar(tmp_path)
    _touch_after(out, wide)

    called = []
    nvt.ensure_training_predictions(
        wide, Path(__file__), out, rescore=True,
        runner=lambda *a, **k: called.append(a),
    )
    assert called


# ── the pipeline owns the artefact ────────────────────────────────────────


def test_the_pipeline_builds_the_training_scores():
    """It was produced only as a side-effect of hand-running the figure script,
    so nothing refreshed it when new cohorts landed."""
    from scripts.pipeline.run_workflows import _training_scores_command

    class _S:
        class _R:
            output_csv = "/x/model_predictions.csv"
            python = ""
        reaction_prediction = _R()

    cfg = {"training_scores": {
        "enabled": True,
        "training_wide_csv": "/x/train.parquet",
        "output_csv": "/x/model_predictions_training.csv",
        "model_path": "/x/model.json",
    }}
    cmd = _training_scores_command(cfg, _S(), python_exec="python3", config_path=None)
    assert cmd is not None
    assert "/x/model_predictions_training.csv" in cmd


def test_no_training_scores_command_when_unconfigured():
    from scripts.pipeline.run_workflows import _training_scores_command

    class _S:
        reaction_prediction = None

    assert _training_scores_command({}, _S(), python_exec="python3",
                                    config_path=None) is None


def test_config_declares_the_training_scores_step():
    import yaml

    path = Path(__file__).resolve().parent.parent / "config" / "config_new.yaml"
    if not path.exists():
        pytest.skip("config/ is gitignored")
    cfg = yaml.safe_load(path.read_text())["analysis"]
    assert "training_scores" in cfg
    assert str(cfg["training_scores"]["output_csv"]).endswith(
        "model_predictions_training.csv"
    )
