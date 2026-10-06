"""The figure-subprocess pool in run_workflows.

The analysis stage ran ~20 figure scripts strictly one after another (about
170 s a night) while 13 of 16 cores sat idle. ``_FigurePool`` runs them up to
``analysis.figure_workers`` (default 3: each peaks at 4.9-6.1 GB RSS) at a
time. What must NOT change:

* 1 worker is the old code path exactly: each command runs inline at submit,
  through ``subprocess.run``, with live output.
* Output is replayed per command, in submission order, never interleaved.
* Exit handling is the caller's, unchanged: a warning-style failure lets the
  others run; a ``check`` failure raises ``CalledProcessError`` like
  ``subprocess.run(check=True)`` did, and no further command starts.
* Ordering constraints hold: ``after=`` chains a command behind another (two
  commands writing one file), ``call()`` callbacks (a step's state write) run
  only once every earlier command has been reported, and a pooled run starts
  nothing before ``join()`` -- which is what lets the driver queue mean traces
  before training_scores yet run them after it.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.pipeline import run_workflows as rw

PY = sys.executable


def _py(code: str) -> list[str]:
    return [PY, "-c", textwrap.dedent(code)]


def _touch(path: Path) -> list[str]:
    return _py(f"open({str(path)!r}, 'a').close()")


# --------------------------------------------------------------------------
# The knob
# --------------------------------------------------------------------------


def test_figure_workers_defaults_to_three():
    assert rw._figure_workers({}) == 3
    assert rw._figure_workers(None) == 3


def test_figure_workers_reads_the_config():
    assert rw._figure_workers({"figure_workers": 1}) == 1
    assert rw._figure_workers({"figure_workers": "5"}) == 5


def test_figure_workers_never_below_one():
    assert rw._figure_workers({"figure_workers": 0}) == 1


def test_figure_workers_falls_back_on_junk(caplog):
    with caplog.at_level(logging.WARNING):
        assert rw._figure_workers({"figure_workers": "lots"}) == 3
    assert "figure_workers" in caplog.text


# --------------------------------------------------------------------------
# One worker == the old serial code path
# --------------------------------------------------------------------------


def test_serial_pool_runs_each_command_inline_at_submit(tmp_path):
    pool = rw._FigurePool(1)
    marker = tmp_path / "ran"
    seen = []
    pool.submit(_touch(marker), announce=lambda: seen.append("announce"),
                on_exit=seen.append)
    # Already ran: no join needed, exactly like the old for-loop.
    assert marker.exists()
    assert seen == ["announce", 0]
    pool.call(lambda: seen.append("callback"))
    assert seen[-1] == "callback"
    pool.join()  # nothing left to do


def test_serial_pool_uses_subprocess_run_with_inherited_output(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(rw.subprocess, "run", fake_run)
    monkeypatch.setattr(
        rw.subprocess, "Popen",
        lambda *a, **k: pytest.fail("serial mode must not use Popen"),
    )
    env = {"X": "1"}
    rw._FigurePool(1).submit(["prog", "--flag"], env=env)
    assert calls == [(["prog", "--flag"], {"env": env})]


def test_serial_check_failure_raises_at_submit():
    pool = rw._FigurePool(1)
    with pytest.raises(subprocess.CalledProcessError) as info:
        pool.submit(_py("raise SystemExit(4)"), check=True)
    assert info.value.returncode == 4


# --------------------------------------------------------------------------
# Pooled
# --------------------------------------------------------------------------


def test_pooled_pool_starts_nothing_before_join(tmp_path):
    pool = rw._FigurePool(3)
    marker = tmp_path / "ran"
    pool.submit(_touch(marker))
    time.sleep(0.3)
    assert not marker.exists()
    pool.join()
    assert marker.exists()


def test_pooled_commands_run_concurrently(tmp_path):
    """Each command waits for all three markers: serially, the first would
    time out and exit 1."""
    markers = [tmp_path / f"m{i}" for i in range(3)]
    rcs = []
    pool = rw._FigurePool(3)
    for m in markers:
        pool.submit(_py(f"""
            import pathlib, sys, time
            pathlib.Path({str(m)!r}).touch()
            names = {[str(x) for x in markers]!r}
            deadline = time.time() + 20
            while not all(pathlib.Path(n).exists() for n in names):
                if time.time() > deadline:
                    sys.exit(1)
                time.sleep(0.02)
        """), on_exit=rcs.append)
    pool.join()
    assert rcs == [0, 0, 0]


def test_pooled_pool_never_exceeds_max_workers(tmp_path):
    running = tmp_path / "running"
    running.mkdir()
    seen = tmp_path / "seen.txt"
    pool = rw._FigurePool(2)
    for i in range(5):
        pool.submit(_py(f"""
            import pathlib, time
            me = pathlib.Path({str(running)!r}) / "{i}"
            me.touch()
            for _ in range(6):
                n = len(list(pathlib.Path({str(running)!r}).iterdir()))
                with open({str(seen)!r}, "a") as fh:
                    fh.write(f"{{n}}\\n")
                time.sleep(0.05)
            me.unlink()
        """))
    pool.join()
    counts = [int(x) for x in seen.read_text().split()]
    assert counts and max(counts) <= 2


def test_pooled_output_is_replayed_whole_and_in_submission_order(capfd):
    pool = rw._FigurePool(3)
    # The first command finishes LAST; its block must still come first.
    pool.submit(_py("""
        import sys, time
        time.sleep(0.6)
        print("first-line-1"); print("first-line-2", file=sys.stderr)
        print("first-line-3")
    """), announce=lambda: print("ANNOUNCE-1"))
    pool.submit(_py('print("second-line")'), announce=lambda: print("ANNOUNCE-2"))
    pool.join()
    out = capfd.readouterr().out
    # Within one command, stdout vs stderr order is the child's own business
    # (its stdout is block-buffered, as it always was in the nightly log); the
    # pool's job is that the whole block sits between its header and the next.
    ann1, ann2 = out.index("ANNOUNCE-1"), out.index("ANNOUNCE-2")
    for line in ("first-line-1", "first-line-2", "first-line-3"):
        assert ann1 < out.index(line) < ann2, out
    assert ann2 < out.index("second-line"), out


def test_after_chains_a_command_behind_another(tmp_path):
    flag = tmp_path / "written-by-a"
    rcs = []
    pool = rw._FigurePool(3)
    a = pool.submit(_py(f"""
        import pathlib, time
        time.sleep(0.4)
        pathlib.Path({str(flag)!r}).write_text("a")
    """))
    pool.submit(
        _py(f"import pathlib, sys; sys.exit(0 if pathlib.Path({str(flag)!r}).exists() else 1)"),
        after=a, on_exit=rcs.append,
    )
    pool.join()
    assert rcs == [0]


def test_a_warning_style_failure_does_not_stop_the_others(tmp_path):
    rcs = []
    marker = tmp_path / "third"
    pool = rw._FigurePool(2)
    pool.submit(_py("pass"), on_exit=rcs.append)
    pool.submit(_py("raise SystemExit(3)"), on_exit=rcs.append)
    pool.submit(_touch(marker), on_exit=rcs.append)
    pool.join()
    assert rcs == [0, 3, 0]
    assert marker.exists()


def test_a_check_failure_raises_and_starts_nothing_new(tmp_path):
    finished = tmp_path / "b-finished"
    never = tmp_path / "c-ran"
    pool = rw._FigurePool(2)
    pool.submit(_py("raise SystemExit(5)"), check=True)
    pool.submit(_py(f"""
        import pathlib, time
        time.sleep(0.5)
        pathlib.Path({str(finished)!r}).touch()
    """), check=True)
    pool.submit(_touch(never), check=True)
    with pytest.raises(subprocess.CalledProcessError) as info:
        pool.join()
    assert info.value.returncode == 5
    # The command already running was waited for (no orphan figure process)...
    assert finished.exists()
    # ...and the queued one never started, as under subprocess.run(check=True).
    assert not never.exists()


def test_a_dependent_of_a_failed_check_never_starts(tmp_path):
    never = tmp_path / "dependent-ran"
    pool = rw._FigurePool(3)
    first = pool.submit(_py("raise SystemExit(2)"), check=True)
    pool.submit(_touch(never), after=first, check=True)
    with pytest.raises(subprocess.CalledProcessError):
        pool.join()
    assert not never.exists()


def test_callbacks_wait_for_every_earlier_command(tmp_path):
    log = tmp_path / "log.txt"
    snapshots = []
    pool = rw._FigurePool(3)
    pool.submit(_py(f"""
        import time
        time.sleep(0.4)
        open({str(log)!r}, "a").write("A\\n")
    """))
    pool.call(lambda: snapshots.append(log.read_text().split()))
    pool.submit(_py(f'open({str(log)!r}, "a").write("B\\n")'))
    pool.call(lambda: snapshots.append(sorted(log.read_text().split())))
    pool.join()
    # The first callback ran only after A (slow, submitted first) was done;
    # the second only after both.
    assert "A" in snapshots[0]
    assert snapshots[1] == ["A", "B"]


def test_join_with_only_callbacks_runs_them_in_order():
    seen = []
    pool = rw._FigurePool(3)
    pool.call(lambda: seen.append(1))
    pool.call(lambda: seen.append(2))
    pool.join()
    assert seen == [1, 2]


def test_join_is_reusable(tmp_path):
    pool = rw._FigurePool(2)
    for name in ("one", "two"):
        marker = tmp_path / name
        pool.submit(_touch(marker))
        pool.join()
        assert marker.exists()


# --------------------------------------------------------------------------
# Runner wiring
# --------------------------------------------------------------------------


def _fake_script(path: Path, name: str, *, log: Path, sleep: float = 0.0,
                 rc: int = 0) -> Path:
    """A stand-in figure script that logs when it starts and ends."""
    path.mkdir(parents=True, exist_ok=True)
    script = path / name
    script.write_text(textwrap.dedent(f"""
        import sys, time
        def mark(what):
            with open({str(log)!r}, "a") as fh:
                fh.write(f"{{time.monotonic():.6f}} {name} {{what}} {{' '.join(sys.argv[1:])}}\\n")
        mark("start")
        time.sleep({sleep})
        mark("end")
        sys.exit({rc})
    """))
    return script


def _events(log: Path) -> list[tuple[float, str, str, str]]:
    rows = []
    for line in log.read_text().splitlines():
        t, name, what, *rest = line.split(" ")
        rows.append((float(t), name, what, " ".join(rest)))
    return rows


def _when(events, name, what, args=""):
    return [t for t, n, w, a in events if n == name and w == what and args in a]


def test_cohort_figures_keep_the_two_auc_runs_in_order(tmp_path, monkeypatch):
    """Both AUC runs write Training-AUC-vs-Testing-Response/
    training_auc_vs_control_response.json; last writer wins, so the trained
    set must still finish after the control set."""
    log = tmp_path / "log.txt"
    raster = _fake_script(tmp_path / "a", "binarized_per_rasters.py", log=log, sleep=0.2)
    auc = _fake_script(tmp_path / "b", "training_auc_vs_control_response.py",
                       log=log, sleep=0.5)
    cmds = [
        [PY, str(raster), "control-binary"],
        [PY, str(raster), "control-graded"],
        [PY, str(auc), "control"],
        [PY, str(raster), "trained-binary"],
        [PY, str(raster), "trained-graded"],
        [PY, str(auc), "trained"],
    ]
    monkeypatch.setattr(rw, "_cohort_figure_commands", lambda *a, **k: cmds)
    pool = rw._FigurePool(3)
    rw._run_cohort_figures({}, SimpleNamespace(), config_path=None, pool=pool)
    pool.join()
    ev = _events(log)
    [ctrl_end] = _when(ev, "training_auc_vs_control_response.py", "end", "control")
    [trained_start] = _when(ev, "training_auc_vs_control_response.py", "start", "trained")
    assert trained_start >= ctrl_end
    # And it did run in parallel: a raster overlapped the control AUC run.
    [ctrl_start] = _when(ev, "training_auc_vs_control_response.py", "start", "control")
    raster_ends = _when(ev, "binarized_per_rasters.py", "end")
    assert any(ctrl_start < t < ctrl_end for t in raster_ends)


def test_cohort_figures_failure_warns_and_the_rest_still_run(tmp_path, monkeypatch, caplog):
    log = tmp_path / "log.txt"
    bad = _fake_script(tmp_path / "a", "binarized_per_rasters.py", log=log, rc=2)
    good = _fake_script(tmp_path / "b", "training_auc_vs_control_response.py", log=log)
    cmds = [[PY, str(bad), "x"], [PY, str(good), "y"]]
    monkeypatch.setattr(rw, "_cohort_figure_commands", lambda *a, **k: cmds)
    pool = rw._FigurePool(3)
    with caplog.at_level(logging.INFO):
        rw._run_cohort_figures({}, SimpleNamespace(), config_path=None, pool=pool)
        pool.join()
    assert "cohort_figures command exited with code 2: " + " ".join(cmds[0]) in caplog.text
    assert _when(_events(log), "training_auc_vs_control_response.py", "end")
    # "complete." is reported after both commands.
    assert caplog.text.index("exited with code 2") < caplog.text.index("cohort_figures complete.")


def test_mean_traces_failure_names_its_out_dir(tmp_path, monkeypatch, caplog):
    log = tmp_path / "log.txt"
    bad = _fake_script(tmp_path / "a", "dataset_mean_traces_tvc.py", log=log, rc=1)
    cmds = [[PY, str(bad), "--out-dir", "/figs/EB-24-1"]]
    monkeypatch.setattr(rw, "_dataset_mean_traces_commands", lambda *a, **k: cmds)
    pool = rw._FigurePool(3)
    with caplog.at_level(logging.INFO):
        rw._run_dataset_mean_traces({}, SimpleNamespace(), config_path=None, pool=pool)
        pool.join()
    assert "dataset_mean_traces for /figs/EB-24-1 exited with code 1" in caplog.text
    assert caplog.text.index("exited with code 1") < caplog.text.index(
        "dataset_mean_traces complete."
    )


@pytest.mark.parametrize("runner,builder,label", [
    (rw._run_naive_vs_trained, "_naive_vs_trained_command", "naive_vs_trained"),
    (rw._run_pretest_vs_test, "_pretest_vs_test_command", "pretest_vs_test"),
    (rw._run_pretest_score_bars, "_pretest_score_bars_command", "pretest_score_bars"),
])
def test_single_command_runners_keep_their_exit_messages(
    runner, builder, label, tmp_path, monkeypatch, caplog
):
    log = tmp_path / "log.txt"
    bad = _fake_script(tmp_path, "s.py", log=log, rc=7)
    monkeypatch.setattr(rw, builder, lambda *a, **k: [PY, str(bad)])
    pool = rw._FigurePool(3)
    with caplog.at_level(logging.INFO):
        runner({}, SimpleNamespace(), config_path=None, pool=pool)
        assert not log.exists()  # queued, not started
        pool.join()
    assert f"[analysis] {label} exited with code 7" in caplog.text


# ---- the reactions figure group ------------------------------------------


def _reaction_settings(tmp_path: Path) -> SimpleNamespace:
    csv = tmp_path / "model_predictions.csv"
    csv.write_text("dataset,fly\n")
    matrix = SimpleNamespace(
        out_dir=str(tmp_path / "matrix"), trial_orders=["trained-first"],
        latency_sec=0.0, after_window_sec=30.0, row_gap=0.6,
        height_per_gap_in=3.0, bottom_shift_in=0.5, include_hexanol=True,
        overwrite=True,
    )
    rp = SimpleNamespace(
        run_prediction=False, data_csv="", model_path="", output_csv=str(csv),
        threshold=None, matrix=matrix, python=PY, model_type="ordinal",
        publication_figures=None,
    )
    return SimpleNamespace(
        reaction_prediction=rp, non_reactive_span_px=5.0, flagged_flies_csv="",
        force=SimpleNamespace(reaction_matrix=True), protocol="v2",
        cache_dir=str(tmp_path / "cache"), dataset_overrides={},
    )


def _reaction_scripts(root: Path, log: Path, *, matrix_rc: int = 0) -> None:
    analysis = root / "scripts" / "analysis"
    _fake_script(analysis, "reaction_matrix_from_spreadsheet.py", log=log,
                 sleep=0.4, rc=matrix_rc)
    _fake_script(analysis, "reaction_matrix_training_vs_control.py", log=log)
    _fake_script(analysis, "score_summary.py", log=log, sleep=0.2)
    _fake_script(analysis, "randompanel_conc_comparison.py", log=log)


def _state_file(tmp_path: Path) -> Path:
    return tmp_path / "cache" / "reaction_matrix" / "reaction_prediction" / "state.json"


@pytest.mark.parametrize("workers", [1, 3])
def test_reactions_train_vs_ctrl_runs_after_the_matrix(tmp_path, monkeypatch, workers):
    """train_vs_ctrl reads the binary_reactions_*_unordered.csv the matrix
    command writes, and the matrix's own 'all/' collection rglobs the
    *unordered.png files train_vs_ctrl writes -- so it stays behind it."""
    log = tmp_path / "log.txt"
    _reaction_scripts(tmp_path, log)
    monkeypatch.setattr(rw, "REPO_ROOT", tmp_path)
    rw._run_reactions(_reaction_settings(tmp_path), figure_workers=workers)
    ev = _events(log)
    [matrix_end] = _when(ev, "reaction_matrix_from_spreadsheet.py", "end")
    [tvc_start] = _when(ev, "reaction_matrix_training_vs_control.py", "start")
    assert tvc_start >= matrix_end
    assert _when(ev, "score_summary.py", "end")
    assert _when(ev, "randompanel_conc_comparison.py", "end")
    assert _state_file(tmp_path).exists()


@pytest.mark.parametrize("workers", [1, 3])
def test_reactions_matrix_failure_raises_like_check_true(tmp_path, monkeypatch, workers):
    log = tmp_path / "log.txt"
    _reaction_scripts(tmp_path, log, matrix_rc=1)
    monkeypatch.setattr(rw, "REPO_ROOT", tmp_path)
    with pytest.raises(subprocess.CalledProcessError) as info:
        rw._run_reactions(_reaction_settings(tmp_path), figure_workers=workers)
    assert info.value.returncode == 1
    ev = _events(log)
    assert not _when(ev, "reaction_matrix_training_vs_control.py", "start")
    # The matrix state is only written once its figures succeeded.
    assert not _state_file(tmp_path).exists()
