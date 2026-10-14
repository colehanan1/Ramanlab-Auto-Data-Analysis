"""eye_drift_qc is a registered pipeline step that run_workflows runs per dataset.

It reads the merged tables YOLO writes, so it must come after ``yolo``; it is
report-only QC like ``check_light_stimulus`` and sits right after it, before
``reject_bad_proboscis`` starts rewriting per-fly tables. run_workflows picks
it up through ``remaining_steps`` -- it must never ride along with the YOLO
workers, which are worker-sliced.
"""
from __future__ import annotations

import ast
import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbpipe import pipeline  # noqa: E402
from fbpipe.steps import eye_drift_qc  # noqa: E402
from scripts.pipeline import run_workflows as rw  # noqa: E402


def _names():
    return [step.name for step in pipeline.ORDERED_STEPS]


def test_eye_drift_qc_runs_after_yolo_and_before_reject_bad_proboscis():
    names = _names()
    assert names.index("yolo") < names.index("eye_drift_qc") < names.index("reject_bad_proboscis")
    assert names[names.index("check_light_stimulus") + 1] == "eye_drift_qc"
    assert pipeline.STEP_REGISTRY["eye_drift_qc"].runner is eye_drift_qc.main


def test_both_pipeline_import_branches_bring_in_the_step():
    """pipeline.py imports its steps twice, once for CUDA hosts and once for
    CPU-only ones; a step missing from either branch is a NameError there."""
    tree = ast.parse(Path(pipeline.__file__).read_text(encoding="utf-8"))
    step_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == "steps"
    ]
    assert len(step_imports) == 2
    for node in step_imports:
        assert "eye_drift_qc" in {alias.name for alias in node.names}


def test_run_workflows_runs_it_with_the_per_trial_steps_not_the_yolo_workers():
    src = inspect.getsource(getattr(rw, "_main_locked", rw.main))  # main() runs it under the run lock
    # main() derives the per-dataset step list from ORDERED_STEPS minus YOLO ...
    assert re.search(
        r'remaining_steps\s*=\s*\[step\.name for step in ORDERED_STEPS if step\.name != "yolo"\]', src
    )
    assert rw.ORDERED_STEPS is pipeline.ORDERED_STEPS
    remaining = [step.name for step in rw.ORDERED_STEPS if step.name != "yolo"]
    assert "eye_drift_qc" in remaining
    # ... and the worker-sliced YOLO launch carries the yolo step alone.
    assert 'steps=("yolo",)' in src
