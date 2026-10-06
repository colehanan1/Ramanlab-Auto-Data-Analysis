"""main()'s analysis tail on the figure pool: what runs when.

The steps from the mean traces to the cohort figures share one
``_FigurePool``. What must hold, whatever ``analysis.figure_workers`` says:

* 1 worker is today's run exactly: every command inline, one at a time, in
  the order mean traces -> naive_vs_trained -> training_scores -> pretest
  score bars -> pre-vs-post -> cohort figures.
* Pooled, no figure command starts before training_scores has finished: it
  rewrites model_predictions_training.csv, which the pretest score bars read.
* Every figure command has finished before the deferred local->secured copy
  (which may delete local data) and the SMB copy of the outputs.

main() runs for real on a minimal config; the steps around the pool are
stubbed to markers, and every command is a stand-in script that logs when it
starts and ends (CLOCK_MONOTONIC is shared across processes).
"""

from __future__ import annotations

import sys
import textwrap
import time
from pathlib import Path

import pytest
import yaml

from scripts.pipeline import run_workflows as rw

PY = sys.executable


def _script(path: Path, *, log: Path, sleep: float) -> Path:
    path.write_text(textwrap.dedent(f"""
        import sys, time
        def mark(what):
            with open({str(log)!r}, "a") as fh:
                fh.write(f"{{time.monotonic():.6f}} {{sys.argv[1]}} {{what}}\\n")
        mark("start")
        time.sleep({sleep})
        mark("end")
    """))
    return path


def _mark(log: Path, name: str) -> None:
    with open(log, "a") as fh:
        fh.write(f"{time.monotonic():.6f} {name} at\n")


def _events(log: Path) -> list[tuple[float, str, str]]:
    rows = []
    for line in log.read_text().splitlines():
        t, name, what = line.split(" ")
        rows.append((float(t), name, what))
    return rows


def _run_main(tmp_path: Path, monkeypatch, workers: int) -> list[tuple[float, str, str]]:
    log = tmp_path / "events.txt"
    script = _script(tmp_path / "figure.py", log=log, sleep=0.3)
    predictions = tmp_path / "model_predictions.csv"
    predictions.write_text("dataset,fly\n")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "cache_dir": str(tmp_path / "cache"),
        "reaction_prediction": {"output_csv": str(predictions)},
        "analysis": {
            "figure_workers": workers,
            # Truthy, so main() takes the combine path that hands back the
            # deferred secure cleanup (stubbed below).
            "combined": {"stub": True},
        },
    }))

    def cmd(name: str, *extra: str) -> list[str]:
        return [PY, str(script), name, *extra]

    pred = ("--predictions-csv", str(predictions))
    monkeypatch.setattr(rw, "_dataset_mean_traces_commands", lambda *a, **k: [
        cmd("mean_traces_1", "--out-dir", str(tmp_path / "mt1")),
        cmd("mean_traces_2", "--out-dir", str(tmp_path / "mt2")),
    ])
    monkeypatch.setattr(rw, "_naive_vs_trained_command",
                        lambda *a, **k: cmd("naive_vs_trained", *pred))
    monkeypatch.setattr(rw, "_training_scores_command",
                        lambda *a, **k: cmd("training_scores"))
    monkeypatch.setattr(rw, "_pretest_score_bars_command",
                        lambda *a, **k: cmd("pretest_score_bars"))
    monkeypatch.setattr(rw, "_pretest_vs_test_command",
                        lambda *a, **k: cmd("pretest_vs_test", *pred))
    monkeypatch.setattr(rw, "_cohort_figure_commands", lambda *a, **k: [
        cmd("cohort_binary", "--out-dir", str(tmp_path / "bin")),
        cmd("cohort_auc", "--out-dir", str(tmp_path / "auc")),
    ])

    # Everything outside the pooled section is a marker or a no-op.
    monkeypatch.setattr(rw, "_run_pipeline", lambda *a, **k: None)
    monkeypatch.setattr(rw, "_should_skip", lambda *a, **k: False)
    monkeypatch.setattr(rw, "_should_skip_with_manifest", lambda *a, **k: False)
    monkeypatch.setattr(
        rw, "_run_combined",
        lambda *a, **k: (lambda: _mark(log, "secure_cleanup")),
    )
    for name in ("_load_light_checks_for_envelopes", "_load_model_scores_for_envelopes",
                 "_run_envelope_visuals", "_run_training", "_run_light_only_traces",
                 "_run_dataset_means", "_run_dataset_means_training"):
        monkeypatch.setattr(rw, name, lambda *a, **k: None)
    monkeypatch.setattr(rw, "_run_reactions", lambda *a, **k: _mark(log, "reactions"))
    monkeypatch.setattr(rw, "_batch_copy_to_smb", lambda *a, **k: _mark(log, "smb_copy"))
    monkeypatch.setattr(rw, "_run_data_secured_sync", lambda *a, **k: None)
    monkeypatch.setattr(rw, "ntfy_notify", lambda *a, **k: pytest.fail("no notifications"))

    rw.main(["--config", str(config)])
    return _events(log)


POOLED = ["mean_traces_1", "mean_traces_2", "naive_vs_trained", "pretest_score_bars",
          "pretest_vs_test", "cohort_binary", "cohort_auc"]


def _times(events, name, what):
    return [t for t, n, w in events if n == name and w == what]


def test_one_worker_is_todays_serial_order(tmp_path, monkeypatch):
    events = _run_main(tmp_path, monkeypatch, workers=1)
    commands = [(n, w) for _t, n, w in events if w in ("start", "end")]
    order = ["mean_traces_1", "mean_traces_2", "naive_vs_trained", "training_scores",
             "pretest_score_bars", "pretest_vs_test", "cohort_binary", "cohort_auc"]
    # Strictly one at a time, in today's order.
    assert commands == [(n, w) for n in order for w in ("start", "end")]
    [last_end] = _times(events, "cohort_auc", "end")
    [cleanup] = _times(events, "secure_cleanup", "at")
    [smb] = _times(events, "smb_copy", "at")
    assert last_end < cleanup < smb


def test_pooled_starts_nothing_before_training_scores(tmp_path, monkeypatch):
    events = _run_main(tmp_path, monkeypatch, workers=3)
    [scores_end] = _times(events, "training_scores", "end")
    starts = {n: _times(events, n, "start")[0] for n in POOLED}
    ends = {n: _times(events, n, "end")[0] for n in POOLED}
    assert min(starts.values()) > scores_end
    # ...and they did overlap (three at a time).
    first_three = sorted(starts.values())[:3]
    assert first_three[-1] < min(ends.values())
    # Every figure is done before local data may be deleted or copied out.
    [cleanup] = _times(events, "secure_cleanup", "at")
    [smb] = _times(events, "smb_copy", "at")
    assert max(ends.values()) < cleanup < smb


def test_pooled_writes_each_step_state_once_its_commands_are_done(tmp_path, monkeypatch):
    _run_main(tmp_path, monkeypatch, workers=3)
    cache = tmp_path / "cache"
    for category in ("dataset_mean_traces", "naive_vs_trained",
                     "pretest_vs_test", "cohort_figures"):
        assert (cache / category / "analysis" / "state.json").exists(), category
