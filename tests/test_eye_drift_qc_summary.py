"""The end-of-run eye-drift QC line run_workflows prints and sends with "Pipeline Complete".

The ``eye_drift_qc`` step upserts every trial x fly into one cumulative
``eye_drift_qc.csv``. At the very end of a run -- after the data-secured sync,
and under ``--figures-only`` too -- run_workflows reads that report back and
sums it up in one line: how many assessed flies are flagged, and how many
trials could not be assessed because their tables predate per-frame eye
tracking. Folders frozen right now are left out, as in every figure. The line
is a courtesy: no report, a broken report or any error gives "" and never
changes the run's outcome -- which is also why main() returns None and leaves
the line in the module's ``LAST_EYE_QC_SUMMARY`` for the ``__main__`` block.
"""
from __future__ import annotations

import sys
import ast
import inspect
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fbpipe.config as fbpipe_config
from fbpipe.config import DatasetOverride, EyeQCSettings, Settings
from fbpipe.steps import eye_drift_qc as qc
from scripts.pipeline import run_workflows as rw

DATASET = "E2E-EyeQC"
LIVE_BATCH = "october_07_batch_1_rig_2"
RETIRED_BATCH = "may_22_batch_1"
# When _write_report's report was last written, as every full-run line ends.
WRITTEN = datetime(2026, 10, 8, 3, 12)
LAST_WRITTEN = " (report last written 2026-10-08 03:12)"


@pytest.fixture(autouse=True)
def _no_real_side_effects(monkeypatch, tmp_path):
    """No test may read the real report dir or reach the phone."""
    monkeypatch.setattr(fbpipe_config, "DEFAULT_EYE_QC_REPORT_DIR", tmp_path / "default-report-dir")
    monkeypatch.setenv("FBPIPE_DISABLE_NTFY", "1")


def _settings(tmp_path: Path, **overrides) -> Settings:
    kwargs = dict(
        model_path="model.pt",
        main_directories=[str(tmp_path / "data" / DATASET)],
        datasets=(DATASET,),
        eye_qc=EyeQCSettings(report_dir=str(tmp_path / "qc")),
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


def _trial(tmp_path: Path, name: str, batch: str = LIVE_BATCH) -> Path:
    prefix = batch.removesuffix("_rig_2")
    return tmp_path / "data" / DATASET / batch / f"{prefix}_{name}"


def _flies(trial: Path, n: int, flagged: tuple[int, ...] = ()) -> list[dict]:
    """One assessed row per fly, as the step writes them."""
    return [
        {
            "dataset": DATASET,
            "fly": trial.parent.name,
            "fly_number": k,
            "slot_idx": k - 1,
            "eye_track_id": 1000 + k,
            "trial_dir": str(trial),
            "status": "assessed",
            "eye_qc_flag": k in flagged,
            "eye_qc_reason": "drift > 5.0 px (median over 0.5 s) from 30.1 s (max 7.9 px)" if k in flagged else "",
        }
        for k in range(1, n + 1)
    ]


def _trial_row(trial: Path, status: str) -> dict:
    """A trial-level row (fly_number blank): no_live_eye_columns, read_error, ..."""
    return {"dataset": DATASET, "fly": trial.parent.name, "trial_dir": str(trial), "status": status}


def _write_report(tmp_path: Path, rows: list[dict]) -> Path:
    """The report as the step leaves it on disk: its column order and dtypes,
    last written at WRITTEN."""
    path = tmp_path / "qc" / qc.REPORT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    qc._normalised(pd.DataFrame(rows).assign(qc_version=qc.QC_VERSION)).to_csv(path, index=False)
    os.utime(path, (WRITTEN.timestamp(), WRITTEN.timestamp()))
    return path


def _flagged_csv(tmp_path: Path) -> Path:
    return tmp_path / "qc" / qc.FLAGGED_FILENAME


# ---------------------------------------------------------------------------
# What the line says
# ---------------------------------------------------------------------------


def test_counts_flagged_flies_among_the_assessed_and_trials_without_live_columns(tmp_path):
    _write_report(
        tmp_path,
        _flies(_trial(tmp_path, "pretest_1"), 4, flagged=(2,))
        + _flies(_trial(tmp_path, "testing_1"), 4, flagged=(1,))
        + [_trial_row(_trial(tmp_path, f"testing_{i}"), "no_live_eye_columns") for i in (2, 3, 4)],
    )

    assert rw._eye_qc_summary(_settings(tmp_path)) == (
        "EYE-QC: 2 of 8 assessed trial-flies flagged (fly came loose?); "
        "3 trials not assessable (tables predate per-frame eye tracking) "
        f"-> {_flagged_csv(tmp_path)}{LAST_WRITTEN}"
    )


def test_large_counts_carry_thousands_separators(tmp_path):
    rows = []
    for i in range(300):
        rows += _flies(_trial(tmp_path, f"testing_{i}"), 4, flagged=(3,) if i < 3 else ())
    rows += [_trial_row(_trial(tmp_path, f"training_{i}"), "no_live_eye_columns") for i in range(2500)]
    _write_report(tmp_path, rows)

    summary = rw._eye_qc_summary(_settings(tmp_path))

    assert summary.startswith("EYE-QC: 3 of 1,200 assessed trial-flies flagged (fly came loose?); ")
    assert "; 2,500 trials not assessable (tables predate per-frame eye tracking) -> " in summary


def test_nothing_flagged_drops_the_loose_fly_hint_and_zero_counts(tmp_path):
    _write_report(tmp_path, _flies(_trial(tmp_path, "pretest_1"), 4))

    assert rw._eye_qc_summary(_settings(tmp_path)) == (
        f"EYE-QC: 0 of 4 assessed trial-flies flagged -> {_flagged_csv(tmp_path)}{LAST_WRITTEN}"
    )


def test_the_other_trial_level_statuses_are_counted_on_their_own(tmp_path):
    """A trial that could not be read is not one whose table predates eye tracking."""
    _write_report(
        tmp_path,
        _flies(_trial(tmp_path, "pretest_1"), 2)
        + [
            _trial_row(_trial(tmp_path, "pretest_2"), "no_live_eye_columns"),
            _trial_row(_trial(tmp_path, "pretest_3"), "read_error"),
            _trial_row(_trial(tmp_path, "testing_1"), "no_eye_anchor"),
            _trial_row(_trial(tmp_path, "testing_2"), "empty"),
            _trial_row(_trial(tmp_path, "testing_3"), "empty"),
        ],
    )

    assert rw._eye_qc_summary(_settings(tmp_path)) == (
        "EYE-QC: 0 of 2 assessed trial-flies flagged; "
        "1 trial not assessable (tables predate per-frame eye tracking); "
        "1 trial without an eye anchor; 2 trials with an empty table; 1 trial unreadable "
        f"-> {_flagged_csv(tmp_path)}{LAST_WRITTEN}"
    )


# ---------------------------------------------------------------------------
# Frozen folders
# ---------------------------------------------------------------------------


def _live_and_retired_report(tmp_path: Path) -> None:
    for batch in (LIVE_BATCH, RETIRED_BATCH):
        (tmp_path / "data" / DATASET / batch).mkdir(parents=True)
    _write_report(
        tmp_path,
        _flies(_trial(tmp_path, "pretest_1"), 4, flagged=(2,))
        + [_trial_row(_trial(tmp_path, "pretest_2"), "no_live_eye_columns")]
        # A retired batch: two flagged flies and five legacy trials that no
        # figure shows any more.
        + _flies(_trial(tmp_path, "testing_1", RETIRED_BATCH), 2, flagged=(1, 2))
        + [_trial_row(_trial(tmp_path, f"testing_{i}", RETIRED_BATCH), "no_live_eye_columns") for i in range(2, 7)],
    )


def _retiring(tmp_path: Path, *patterns: str) -> Settings:
    return _settings(tmp_path, dataset_overrides={DATASET: DatasetOverride(freeze_folders=patterns)})


def test_rows_of_folders_frozen_right_now_are_left_out(tmp_path):
    _live_and_retired_report(tmp_path)

    summary = rw._eye_qc_summary(_retiring(tmp_path, "may_22*"))

    assert summary.startswith(
        "EYE-QC: 1 of 4 assessed trial-flies flagged (fly came loose?); 1 trial not assessable "
    )


@pytest.mark.parametrize("thaw", ["named", "all"])
def test_a_thawed_dataset_keeps_its_folder_retirements(tmp_path, thaw):
    """Auto-freeze rule (2026-10-06): an explicit unfreeze (--thaw / --thaw-all)
    rebuilds a dataset with its folder retirements still applied, so the
    summary counts exactly what it counts without the thaw."""
    _live_and_retired_report(tmp_path)
    expected = rw._eye_qc_summary(_retiring(tmp_path, "may_22*"))
    settings = _retiring(tmp_path, "may_22*")
    # main() stashes --thaw / --thaw-all on settings before the summary runs.
    if thaw == "named":
        settings._thawed = (DATASET,)
    else:
        settings._thaw_all = True

    assert expected.startswith("EYE-QC: 1 of 4 assessed trial-flies flagged")
    assert rw._eye_qc_summary(settings) == expected


def test_a_report_whose_folders_are_all_frozen_says_nothing(tmp_path):
    _live_and_retired_report(tmp_path)
    assert rw._eye_qc_summary(_retiring(tmp_path, "*")) == ""


# ---------------------------------------------------------------------------
# --figures-only
# ---------------------------------------------------------------------------


def test_figures_only_names_the_existing_report_and_when_it_was_written(tmp_path):
    path = _write_report(tmp_path, _flies(_trial(tmp_path, "pretest_1"), 4, flagged=(2,)))
    written = datetime(2026, 10, 6, 3, 12).timestamp()
    os.utime(path, (written, written))

    summary = rw._eye_qc_summary(_settings(tmp_path), figures_only=True)

    assert summary == (
        "EYE-QC (--figures-only: from the existing eye_drift_qc.csv, last written 2026-10-06 03:12): "
        f"1 of 4 assessed trial-flies flagged (fly came loose?) -> {_flagged_csv(tmp_path)}"
    )
    # A full run's line keeps its usual start and ends with the same time.
    assert rw._eye_qc_summary(_settings(tmp_path)) == (
        f"EYE-QC: 1 of 4 assessed trial-flies flagged (fly came loose?) -> {_flagged_csv(tmp_path)}"
        " (report last written 2026-10-06 03:12)"
    )


def test_a_full_run_line_says_when_the_report_was_last_written(tmp_path):
    """The step may have been unable to update the report (a report dir left
    root-owned by a sudo run, a full disk); its counts are then an earlier
    run's. The time lets the "Pipeline Complete" ping show that, instead of
    passing them off as tonight's."""
    path = _write_report(tmp_path, _flies(_trial(tmp_path, "pretest_1"), 4))
    written = datetime(2026, 10, 1, 2, 5).timestamp()
    os.utime(path, (written, written))

    summary = rw._eye_qc_summary(_settings(tmp_path))

    assert summary.startswith("EYE-QC: 0 of 4 assessed trial-flies flagged -> ")
    assert summary.endswith(" (report last written 2026-10-01 02:05)")


# ---------------------------------------------------------------------------
# Never raises, never invents a line
# ---------------------------------------------------------------------------


def test_no_report_gives_an_empty_string(tmp_path):
    assert rw._eye_qc_summary(_settings(tmp_path)) == ""


def test_a_report_with_no_rows_gives_an_empty_string(tmp_path, caplog):
    path = tmp_path / "qc" / qc.REPORT_FILENAME
    path.parent.mkdir(parents=True)
    path.write_text(",".join(qc.REPORT_COLUMNS) + "\n", encoding="utf-8")

    assert rw._eye_qc_summary(_settings(tmp_path)) == ""
    assert "EYE-QC" not in caplog.text  # an empty report is not a broken one


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"this is not,an eye-drift report\n1,2\n",
        b"trial_dir,status\n/x/y/z_testing_1,assessed\n",  # no eye_qc_flag column
        # A stray comma mid-file: the columns after it would shift.
        b"trial_dir,status,eye_qc_flag\n/x/y/z_testing_1,assessed,True\n/x/y/z_testing_2,assessed,False,7,8\n",
        bytes(range(256)) * 8,
    ],
    ids=["empty-file", "another-csv", "missing-column", "ragged", "binary"],
)
def test_a_malformed_report_gives_an_empty_string_without_raising(tmp_path, caplog, content):
    path = tmp_path / "qc" / qc.REPORT_FILENAME
    path.parent.mkdir(parents=True)
    path.write_bytes(content)

    assert rw._eye_qc_summary(_settings(tmp_path)) == ""
    assert "EYE-QC" in caplog.text  # says why the line is missing


def test_a_directory_where_the_report_should_be_gives_an_empty_string(tmp_path):
    (tmp_path / "qc" / qc.REPORT_FILENAME).mkdir(parents=True)
    assert rw._eye_qc_summary(_settings(tmp_path)) == ""


def test_disabled_eye_qc_gives_an_empty_string(tmp_path):
    """The step skips itself when eye_qc is off; an old report must not speak for it."""
    _write_report(tmp_path, _flies(_trial(tmp_path, "pretest_1"), 4, flagged=(2,)))
    settings = _settings(tmp_path, eye_qc=EyeQCSettings(enabled=False, report_dir=str(tmp_path / "qc")))
    assert rw._eye_qc_summary(settings) == ""


class _ExplodingSettings:
    def __getattr__(self, name):
        raise RuntimeError(f"settings.{name} exploded")


def test_the_summary_never_raises(caplog):
    assert rw._eye_qc_summary(_ExplodingSettings()) == ""
    assert rw._eye_qc_summary(_ExplodingSettings(), figures_only=True) == ""
    assert "exploded" in caplog.text


def test_a_freeze_rule_the_frozen_check_rejects_costs_only_the_line(tmp_path, caplog):
    """frozen_folders raises on a freeze entry it cannot read; at the end of a
    run that must not turn a finished pipeline into a failed one."""
    _write_report(tmp_path, _flies(_trial(tmp_path, "pretest_1"), 4, flagged=(2,)))
    settings = _settings(
        tmp_path, dataset_overrides={DATASET: DatasetOverride(freeze_folders=({"match": "may_*"},))}
    )

    assert rw._eye_qc_summary(settings) == ""
    assert "freeze_folders" in caplog.text


# ---------------------------------------------------------------------------
# End to end: the report the step writes is the report the summary reads
# ---------------------------------------------------------------------------


def _merged_table(drifts, *, live: bool = True, n: int = 400) -> pd.DataFrame:
    data = {"frame": np.arange(n), "timestamp": np.arange(n) / 40.0}
    for i, drift in enumerate(drifts):
        ex, ey = np.full(n, 510.0), np.full(n, 140.0 + 290.0 * i)
        data[f"eye_{i}_track_id"] = np.full(n, 1001.0 + i)
        data[f"eye_{i}_x"], data[f"eye_{i}_y"] = ex, ey
        if live:
            data[f"eye_{i}_live_x"], data[f"eye_{i}_live_y"] = ex + drift, ey
            data[f"eye_{i}_drift_px"] = drift
    return pd.DataFrame(data)


def test_the_summary_reads_the_report_the_step_writes(tmp_path, monkeypatch):
    steady = np.full(400, 0.4)
    loose = steady.copy()
    loose[100:] = 9.0  # the eye moves 9 px off its anchor 2.5 s in and stays there
    for name, table in (
        ("pretest_1", _merged_table([steady, loose])),
        ("pretest_2", _merged_table([steady, steady], live=False)),  # inferred before eye tracking
    ):
        trial = _trial(tmp_path, name)
        trial.mkdir(parents=True)
        table.to_parquet(trial / f"{trial.name}_distances_merged.parquet", index=False)
    monkeypatch.setattr(qc, "ntfy_notify", lambda *args, **kwargs: True)
    settings = _settings(tmp_path)

    qc.main(settings)

    assert _flagged_csv(tmp_path).is_file()
    report_path = tmp_path / "qc" / qc.REPORT_FILENAME
    written = datetime.fromtimestamp(report_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    assert rw._eye_qc_summary(settings) == (
        "EYE-QC: 1 of 2 assessed trial-flies flagged (fly came loose?); "
        "1 trial not assessable (tables predate per-frame eye tracking) "
        f"-> {_flagged_csv(tmp_path)} (report last written {written})"
    )


def test_a_batch_moved_on_disk_is_counted_once(tmp_path, monkeypatch):
    """The step drops the rows of trials gone from disk, so a re-filed batch is
    counted under its new name only -- not twice, with its flags counted again."""
    steady = np.full(400, 0.4)
    loose = steady.copy()
    loose[100:] = 9.0
    trial = _trial(tmp_path, "pretest_1")
    trial.mkdir(parents=True)
    _merged_table([steady, loose]).to_parquet(trial / f"{trial.name}_distances_merged.parquet", index=False)
    monkeypatch.setattr(qc, "ntfy_notify", lambda *args, **kwargs: True)
    settings = _settings(tmp_path)
    qc.main(settings)

    trial.parent.rename(trial.parent.with_name("october_07_batch_1_rig_3"))
    qc.main(settings)

    assert rw._eye_qc_summary(settings).startswith(
        "EYE-QC: 1 of 2 assessed trial-flies flagged (fly came loose?) -> "
    )


# ---------------------------------------------------------------------------
# Wiring: the last thing main() does, then the "Pipeline Complete" ntfy
# ---------------------------------------------------------------------------


def _module_tree() -> ast.Module:
    return ast.parse(Path(rw.__file__).read_text(encoding="utf-8"))


def _main_def() -> ast.FunctionDef:
    """The function that runs a night: _main_locked(), which main() calls under
    the run lock (auto-freeze), or main() itself where there is no lock."""
    defs = {n.name: n for n in _module_tree().body if isinstance(n, ast.FunctionDef)}
    return defs.get("_main_locked") or defs["main"]


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
    ]


def _own_returns(fn: ast.FunctionDef) -> list[ast.Return]:
    """main()'s own return statements, not those of anything defined inside it."""
    found: list[ast.Return] = []
    stack: list[ast.AST] = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(node, ast.Return):
            found.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return found


def test_main_sums_up_the_report_last_after_the_data_secured_sync():
    body = _main_def().body
    sync = [i for i, stmt in enumerate(body) if _calls(stmt, "_run_data_secured_sync")]
    at = [i for i, stmt in enumerate(body) if _calls(stmt, "_eye_qc_summary")]
    assert len(sync) == 1 and len(at) == 1
    assert at[0] > sync[0]
    # Every mode reaches it, --figures-only included, and it knows which one ran.
    (call,) = _calls(body[at[0]], "_eye_qc_summary")
    assert ast.unparse(call) == "_eye_qc_summary(settings, figures_only=figures_only)"
    # The line lands in the module-level LAST_EYE_QC_SUMMARY -- declared global
    # first, or main() would only bind a local the __main__ block never sees --
    # and is printed.
    (target,) = body[at[0]].targets
    assert target.id == "LAST_EYE_QC_SUMMARY"
    assert any(isinstance(stmt, ast.Global) and target.id in stmt.names for stmt in body[: at[0]])
    assert any(ast.unparse(c) == f"print({target.id})" for stmt in body[at[0]:] for c in _calls(stmt, "print"))


def test_main_returns_none_so_the_legacy_shim_exits_zero():
    """scripts/run_workflows.py ends in ``raise SystemExit(main())``, and a
    SystemExit carrying any str -- even "" -- exits with status 1: a main()
    that handed its summary back would report every finished run as failed.
    So main() returns None and the module keeps the line instead."""
    assert inspect.get_annotations(rw.main, eval_str=True)["return"] is None
    assert [ret for ret in _own_returns(_main_def()) if ret.value is not None] == []
    assert isinstance(rw.LAST_EYE_QC_SUMMARY, str)
    (declared,) = [
        n for n in _module_tree().body
        if isinstance(n, ast.AnnAssign) and ast.unparse(n.target) == "LAST_EYE_QC_SUMMARY"
    ]
    assert ast.unparse(declared) == "LAST_EYE_QC_SUMMARY: str = ''"  # nothing to say until a run ends
    shim = Path(rw.__file__).resolve().parents[1] / "run_workflows.py"
    assert "raise SystemExit(main())" in shim.read_text(encoding="utf-8")


def _run_main_guard(run, sent: list) -> None:
    """Execute run_workflows' ``if __name__ == "__main__":`` block with ``main``
    and ``ntfy_notify`` stubbed; every ntfy it sends lands in ``sent``.

    ``run(namespace)`` is the stub main()'s body. Like the real main() it
    leaves the summary in the module's LAST_EYE_QC_SUMMARY; whatever it
    returns is main()'s return value, which the block must not read.
    """
    guard = next(
        n for n in _module_tree().body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'"
    )
    namespace = {
        "__name__": "__main__",
        "LAST_EYE_QC_SUMMARY": "",
        # The block also reads these module names (run lock, auto-freeze summary).
        "sys": sys,
        "RunLockTimeout": rw.RunLockTimeout,
        "_RUN_SUMMARY": "",
        "ntfy_notify": lambda title, message, **kwargs: sent.append((title, message, kwargs)),
    }
    namespace["main"] = lambda: run(namespace)
    exec(compile(ast.Module(body=[guard], type_ignores=[]), rw.__file__, "exec"), namespace)


def _finishing_with(summary):
    """A stub main() body that ends the way the real one does."""

    def _run(namespace):
        namespace["LAST_EYE_QC_SUMMARY"] = summary

    return _run


def test_the_pipeline_complete_ntfy_carries_the_summary():
    line = "EYE-QC: 1 of 4 assessed trial-flies flagged (fly came loose?) -> /qc/eye_drift_qc_flagged.csv"
    sent = []
    _run_main_guard(_finishing_with(line), sent)
    assert sent == [
        ("Pipeline Complete", f"Pipeline finished successfully.\n\n{line}", {"tags": "white_check_mark"})
    ]


@pytest.mark.parametrize("nothing", ["", None])
def test_without_a_summary_the_pipeline_complete_ntfy_is_unchanged(nothing):
    sent = []
    _run_main_guard(_finishing_with(nothing), sent)
    assert sent == [("Pipeline Complete", "Pipeline finished successfully.", {"tags": "white_check_mark"})]


def test_the_ntfy_reads_the_module_global_not_mains_return_value():
    sent = []
    _run_main_guard(lambda namespace: "EYE-QC: a stray return value", sent)
    assert sent == [("Pipeline Complete", "Pipeline finished successfully.", {"tags": "white_check_mark"})]


def test_a_failed_run_still_reports_failure_and_raises():
    def _crash(namespace):
        raise RuntimeError("yolo died")

    sent = []
    with pytest.raises(RuntimeError, match="yolo died"):
        _run_main_guard(_crash, sent)
    assert [title for title, _, _ in sent] == ["Pipeline FAILED"]
