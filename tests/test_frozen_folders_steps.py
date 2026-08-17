"""The heavy per-trial steps skip frozen experiment folders.

A frozen folder costs zero GPU and zero I/O: YOLO inference, distance
normalisation, distance stats, acceleration, light checks and dropped-frame
detection all pass over it. Its per-trial CSVs already on disk stay exactly as
they are -- which is what lets ``build_wide_csv`` still emit its rows.

Every step enumerates batch dirs the same way, so they share one helper:
``iter_live_batch_dirs(cfg, root)``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from fbpipe.utils.frozen_folders import iter_live_batch_dirs


class _Override:
    def __init__(self, *, freeze_data=False, freeze_folders=()):
        self.freeze_data = freeze_data
        self.freeze_folders = tuple(freeze_folders)


class _Cfg:
    def __init__(self, *, freeze_folders_before=None, overrides=None):
        self.freeze_folders_before = freeze_folders_before
        self.dataset_overrides = dict(overrides or {})


def _batch(root: Path, name: str, stamp: str | None = None) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    if stamp:
        (d / f"output_{name}_testing_1_Hexanol_{stamp}_120000.csv").write_text(
            "frame\n0\n", encoding="utf-8"
        )
    return d


def test_iter_live_batch_dirs_skips_a_frozen_folder(tmp_path):
    root = tmp_path / "Hex-Training-24-0.1"
    root.mkdir()
    _batch(root, "may_22_batch_1_rig_2", "20260522")
    _batch(root, "august_10_batch_1_rig_2", "20260810")

    got = [d.name for d in iter_live_batch_dirs(_Cfg(freeze_folders_before=date(2026, 6, 26)), root)]
    assert got == ["august_10_batch_1_rig_2"]


def test_iter_live_batch_dirs_yields_everything_with_no_rules(tmp_path):
    """No config -> identical to today's ``for p in root.iterdir() if p.is_dir()``."""
    root = tmp_path / "EB-Training-24-1"
    root.mkdir()
    _batch(root, "july_14_batch_2_rig_2", "20260714")
    _batch(root, "july_15_batch_1", "20260715")

    got = [d.name for d in iter_live_batch_dirs(_Cfg(), root)]
    assert got == ["july_14_batch_2_rig_2", "july_15_batch_1"]


def test_iter_live_batch_dirs_is_sorted(tmp_path):
    """yolo_infer relies on a stable order to shard videos across workers."""
    root = tmp_path / "EB-Training-24-1"
    root.mkdir()
    for name in ("july_19_batch_1", "july_13_batch_1", "july_15_batch_1"):
        _batch(root, name, "20260715")
    got = [d.name for d in iter_live_batch_dirs(_Cfg(), root)]
    assert got == sorted(got)


def test_iter_live_batch_dirs_yields_only_directories(tmp_path):
    root = tmp_path / "EB-Training-24-1"
    root.mkdir()
    _batch(root, "july_14_batch_2_rig_2", "20260714")
    (root / "remux.log").write_text("noise", encoding="utf-8")
    got = [d.name for d in iter_live_batch_dirs(_Cfg(), root)]
    assert got == ["july_14_batch_2_rig_2"]


def test_iter_live_batch_dirs_honours_the_config_folder_list(tmp_path):
    root = tmp_path / "EB-Training-24-1"
    root.mkdir()
    _batch(root, "july_14_batch_2_rig_2", "20260714")
    _batch(root, "july_15_batch_1", "20260715")
    cfg = _Cfg(
        overrides={"EB-Training-24-1": _Override(freeze_folders=["july_14_batch_2_rig_2"])}
    )
    got = [d.name for d in iter_live_batch_dirs(cfg, root)]
    assert got == ["july_15_batch_1"]


def test_iter_live_batch_dirs_on_a_missing_root_is_empty(tmp_path):
    """Steps already tolerate a configured root that is not on disk."""
    assert list(iter_live_batch_dirs(_Cfg(), tmp_path / "nope")) == []


# ── the steps actually use it ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "module",
    [
        # Every compute step in pipeline.ORDERED_STEPS that walks batch dirs...
        "fbpipe.steps.yolo_infer",
        "fbpipe.steps.check_light_stimulus",
        "fbpipe.steps.reject_bad_proboscis",
        "fbpipe.steps.distance_stats",
        "fbpipe.steps.distance_normalize",
        "fbpipe.steps.detect_dropped_frames",
        "fbpipe.steps.rms_copy_filter",
        "fbpipe.steps.update_ofm_state",
        "fbpipe.steps.calculate_acceleration",
        # ...plus the alternate GPU/ultra implementations, which a config can
        # select instead and which would otherwise silently reprocess
        # everything the ordinary path skips.
        "fbpipe.steps.distance_normalize_gpu",
        "fbpipe.steps.distance_normalize_ultra",
        "fbpipe.steps.calculate_acceleration_gpu",
    ],
)
def test_step_enumerates_batch_dirs_through_the_helper(module):
    """Pins the wiring: a step that goes back to a bare ``root.iterdir()`` would
    silently reprocess every retired folder on the next run."""
    import importlib
    import inspect

    src = inspect.getsource(importlib.import_module(module))
    assert "iter_live_batch_dirs" in src, f"{module} does not skip frozen folders"
