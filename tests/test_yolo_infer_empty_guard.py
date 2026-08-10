"""yolo_infer must not leave a 'done-looking' trial folder for a zero-frame run.

Regression for the 2026-08-08 incident: corrupt source videos opened but
decoded zero frames, so ``_run_chunked_inference`` returned no rows, yet the
step still wrote a 261-byte header-only annotated mp4 and an empty parquet.
The ``out_dir.exists()`` skip-guard then treated those trials as processed
forever, and the stub mp4 crashed check_light_stimulus downstream.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

from fbpipe.steps import yolo_infer


def test_discard_empty_trial_outputs_removes_stub_and_dir(tmp_path: Path):
    out_dir = tmp_path / "august_07_batch_2_testing_9"
    out_dir.mkdir()
    out_mp4 = out_dir / "august_07_batch_2_testing_9_distance_annotated.mp4"
    out_mp4.write_bytes(b"\x00" * 261)

    yolo_infer._discard_empty_trial_outputs(out_dir, out_mp4, "video.mp4")

    assert not out_mp4.exists()
    assert not out_dir.exists()


def test_discard_empty_trial_outputs_keeps_dir_with_other_files(tmp_path: Path):
    out_dir = tmp_path / "august_07_batch_2_testing_9"
    out_dir.mkdir()
    out_mp4 = out_dir / "august_07_batch_2_testing_9_distance_annotated.mp4"
    out_mp4.write_bytes(b"\x00" * 261)
    other = out_dir / "_trial_meta.json"
    other.write_text("{}")

    yolo_infer._discard_empty_trial_outputs(out_dir, out_mp4, "video.mp4")

    assert not out_mp4.exists()
    assert out_dir.exists()
    assert other.exists()


def test_main_discards_outputs_when_inference_yields_no_rows():
    """The guard must run between inference and the merged-table write."""
    src = inspect.getsource(yolo_infer.main)
    m = re.search(
        r"_run_chunked_inference\(.*?if not rows:\s*\n(.*?)_discard_empty_trial_outputs\(.*?continue.*?write_table\(",
        src,
        re.DOTALL,
    )
    assert m, (
        "yolo_infer.main must call _discard_empty_trial_outputs and skip the "
        "trial (continue) when _run_chunked_inference returns no rows, before "
        "any output tables are written"
    )
