"""The pipeline/YOLO stage cache must notice a change in freeze policy.

The pipeline cache decides "skip this dataset entirely" from two inputs: a
handful of config scalars, and a manifest of the dataset's raw per-trial table
files. Thawing a folder changes NEITHER -- the raw ``output_*.csv`` files were
already on disk (and already hashed into the manifest) while the folder sat
frozen, because ``_build_file_manifest`` walks the whole root and does not
consult the freeze rules.

So un-retiring a folder produced ``[CACHE HIT] No file changes`` and the whole
dataset was skipped, YOLO included -- the newly-live folder was never inferred.
The wide-CSV slice cache already guards this via ``freeze.build_fingerprint``
(which records ``freeze_folders`` + ``freeze_folders_before``); the pipeline
stage is the arm that was missing it.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fbpipe.utils.frozen_folders import FolderRule
from scripts.pipeline.run_workflows import (
    _pipeline_expectation,
    _should_skip_with_manifest,
    _write_state,
)


def _settings(tmp_path: Path, folders=(), cutoff=None):
    """Duck-typed stand-in carrying only what the cache path reads."""
    return SimpleNamespace(
        cache_dir=str(tmp_path / "cache"),
        non_reactive_span_px=100.0,
        class2_min=1,
        class2_max=5,
        freeze_folders_before=cutoff,
        dataset_overrides={
            "EB-Training-24-1": SimpleNamespace(
                freeze_data=False,
                freeze_folders=tuple(folders),
            )
        },
        force=SimpleNamespace(pipeline=False, yolo=False),
    )


def _dataset_root(tmp_path: Path) -> Path:
    """A root shaped like the real thing: <root>/<batch>/output_*.csv."""
    root = tmp_path / "EB-Training-24-1"
    for batch in ("july_20_batch_1_rig_3", "august_13_batch_2_rig_3"):
        d = root / batch
        d.mkdir(parents=True)
        (d / f"output_{batch}_testing_1_Hexanol_20260813_120000.csv").write_text(
            "frame,x\n0,1\n", encoding="utf-8"
        )
    return root


def _seed_cache(settings, root: Path) -> None:
    """Record state exactly as a completed run would."""
    _write_state(
        settings,
        "pipeline",
        str(root),
        dict(_pipeline_expectation(settings, root)),
    )


def test_unchanged_freeze_policy_still_hits_the_cache(tmp_path):
    """The guard must not defeat caching outright -- a no-op re-run still skips."""
    rule = FolderRule("*_rig_3", None)
    settings = _settings(tmp_path, folders=(rule,))
    root = _dataset_root(tmp_path)
    _seed_cache(settings, root)

    assert (
        _should_skip_with_manifest(
            settings,
            category="pipeline",
            key=str(root),
            expected=_pipeline_expectation(settings, root),
            force_flag=False,
            dataset_root=root,
        )
        is True
    )


def test_thawing_a_folder_invalidates_the_pipeline_cache(tmp_path):
    """Bounding the rig_3 rule un-retires august_13 -- the dataset must re-run.

    Not one byte of raw data changes between the two runs, so the file manifest
    is identical; only the freeze policy moved.
    """
    root = _dataset_root(tmp_path)

    frozen = _settings(tmp_path, folders=(FolderRule("*_rig_3", None),))
    _seed_cache(frozen, root)

    # Same cache dir, same files, bounded rule -- august_13 is now live.
    thawed = _settings(
        tmp_path, folders=(FolderRule("*_rig_3", dt.date(2026, 8, 11)),)
    )

    assert (
        _should_skip_with_manifest(
            thawed,
            category="pipeline",
            key=str(root),
            expected=_pipeline_expectation(thawed, root),
            force_flag=False,
            dataset_root=root,
        )
        is False
    )


def test_changing_the_global_cutoff_invalidates_the_pipeline_cache(tmp_path):
    """``freeze_folders_before`` gates every walked dataset, so it counts too."""
    root = _dataset_root(tmp_path)

    old = _settings(tmp_path, cutoff=dt.date(2026, 6, 26))
    _seed_cache(old, root)

    new = _settings(tmp_path, cutoff=dt.date(2026, 7, 1))

    assert (
        _should_skip_with_manifest(
            new,
            category="pipeline",
            key=str(root),
            expected=_pipeline_expectation(new, root),
            force_flag=False,
            dataset_root=root,
        )
        is False
    )


def test_thaw_flag_invalidates_the_pipeline_cache(tmp_path):
    """``--thaw <DATASET>`` is an escape hatch; a cache hit would neuter it."""
    root = _dataset_root(tmp_path)

    settings = _settings(tmp_path, folders=(FolderRule("*_rig_3", None),))
    _seed_cache(settings, root)

    settings._thawed = ("EB-Training-24-1",)

    assert (
        _should_skip_with_manifest(
            settings,
            category="pipeline",
            key=str(root),
            expected=_pipeline_expectation(settings, root),
            force_flag=False,
            dataset_root=root,
        )
        is False
    )


def test_rule_order_does_not_cause_spurious_invalidation(tmp_path):
    """Reordering the folders list changes nothing about what is frozen."""
    root = _dataset_root(tmp_path)
    a = FolderRule("*_rig_3", dt.date(2026, 8, 11))
    b = FolderRule("*batch_2*", dt.date(2026, 7, 27))

    first = _settings(tmp_path, folders=(a, b))
    _seed_cache(first, root)

    second = _settings(tmp_path, folders=(b, a))

    assert (
        _should_skip_with_manifest(
            second,
            category="pipeline",
            key=str(root),
            expected=_pipeline_expectation(second, root),
            force_flag=False,
            dataset_root=root,
        )
        is True
    )
