"""run_workflows wiring for folder freeze.

Two things the pipeline must do:

* resolve the frozen folder set per wide root and hand it to ``build_wide_csv``;
* offer ``--include-frozen`` so a run can rebuild ``model_predictions.csv`` as a
  complete record without editing config.

``--thaw`` / ``--thaw-all`` already lift dataset freezes; they lift folder
freezes too, because ``folder_freeze_rules`` consults them.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

import scripts.pipeline.run_workflows as rw


# ── the CLI flag ──────────────────────────────────────────────────────────


def test_include_frozen_flag_exists_and_defaults_off():
    args = rw._build_arg_parser().parse_args([])
    assert args.include_frozen is False


def test_include_frozen_flag_parses():
    args = rw._build_arg_parser().parse_args(["--include-frozen"])
    assert args.include_frozen is True


# ── resolving the frozen set for a wide root ──────────────────────────────


class _Override:
    def __init__(self, *, freeze_data=False, freeze_folders=()):
        self.freeze_data = freeze_data
        self.freeze_figures = False
        self.freeze_folders = tuple(freeze_folders)


class _Tracking:
    apply_missing_frame_check = True
    max_missing_frames_per_trial = 1800
    max_missing_frames_pct_per_trial = 50.0


class _Cfg:
    def __init__(self, *, freeze_folders_before=None, overrides=None):
        self.freeze_folders_before = freeze_folders_before
        self.dataset_overrides = dict(overrides or {})
        self._thawed = ()
        self._thaw_all = False
        self.protocol = "v2"
        self.tracking = _Tracking()


def _batch(root: Path, name: str, stamp: str) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / f"output_{name}_testing_1_Hexanol_{stamp}_120000.csv").write_text(
        "frame\n0\n", encoding="utf-8"
    )
    return d


def test_resolve_frozen_folders_maps_dataset_to_its_frozen_folders(tmp_path):
    root = tmp_path / "Hex-Training-24-0.1"
    root.mkdir()
    _batch(root, "may_22_batch_1_rig_2", "20260522")
    _batch(root, "august_10_batch_1_rig_2", "20260810")

    got = rw._resolve_frozen_folders(
        _Cfg(freeze_folders_before=date(2026, 6, 26)), [root]
    )
    assert got == {"Hex-Training-24-0.1": {"may_22_batch_1_rig_2"}}


def test_resolve_frozen_folders_covers_every_root(tmp_path):
    a = tmp_path / "Hex-Training-24-0.1"
    b = tmp_path / "Hex-Control-24-0.1"
    a.mkdir()
    b.mkdir()
    _batch(a, "may_22_batch_1", "20260522")
    _batch(b, "may_23_batch_1", "20260523")

    got = rw._resolve_frozen_folders(
        _Cfg(freeze_folders_before=date(2026, 6, 26)), [a, b]
    )
    assert got == {
        "Hex-Training-24-0.1": {"may_22_batch_1"},
        "Hex-Control-24-0.1": {"may_23_batch_1"},
    }


def test_resolve_frozen_folders_omits_datasets_with_nothing_frozen(tmp_path):
    """An empty entry would print a misleading "0 frozen" line and make the
    caller's ``if frozen_folders:`` guard true for a no-op run."""
    root = tmp_path / "EB-Training-24-1"
    root.mkdir()
    _batch(root, "july_14_batch_2_rig_2", "20260714")
    got = rw._resolve_frozen_folders(_Cfg(freeze_folders_before=date(2026, 6, 26)), [root])
    assert got == {}


def test_resolve_frozen_folders_honours_thaw_all(tmp_path):
    root = tmp_path / "Hex-Training-24-0.1"
    root.mkdir()
    _batch(root, "may_22_batch_1", "20260522")
    cfg = _Cfg(freeze_folders_before=date(2026, 6, 26))
    cfg._thaw_all = True
    assert rw._resolve_frozen_folders(cfg, [root]) == {}


def test_resolve_frozen_folders_skips_data_frozen_datasets(tmp_path):
    """"Live datasets only": a data-frozen root is never walked, so it must not
    even be listed -- its cached rows splice in whole."""
    root = tmp_path / "EB-Control-24-0.1"
    root.mkdir()
    _batch(root, "april_11_batch_2", "20260411")
    cfg = _Cfg(
        freeze_folders_before=date(2026, 6, 26),
        overrides={"EB-Control-24-0.1": _Override(freeze_data=True)},
    )
    assert rw._resolve_frozen_folders(cfg, [root]) == {}


def test_resolve_frozen_folders_unions_the_config_list(tmp_path):
    root = tmp_path / "EB-Training-24-1"
    root.mkdir()
    _batch(root, "july_14_batch_2_rig_2", "20260714")
    _batch(root, "july_15_batch_1", "20260715")
    cfg = _Cfg(
        freeze_folders_before=date(2026, 6, 26),
        overrides={
            "EB-Training-24-1": _Override(freeze_folders=["july_14_batch_2_rig_2"])
        },
    )
    assert rw._resolve_frozen_folders(cfg, [root]) == {
        "EB-Training-24-1": {"july_14_batch_2_rig_2"}
    }


# ── the freeze fingerprint tracks the folder rules ────────────────────────


def test_freeze_fingerprint_changes_when_the_cutoff_changes():
    """Otherwise a dataset cached under one cutoff would keep serving those rows
    after the cutoff moved -- one CSV mixing two freeze policies."""
    kw = dict(
        measure_cols=["combined_base"],
        fps_fallback=40.0,
        distance_limits=None,
        non_reactive_threshold=None,
        low_max_threshold_px=20.0,
        use_per_trial_baseline=True,
        trial_type_filter=None,
    )
    a = rw._freeze_fingerprint(_Cfg(freeze_folders_before=date(2026, 6, 26)), "DS", **kw)
    b = rw._freeze_fingerprint(_Cfg(freeze_folders_before=date(2026, 7, 1)), "DS", **kw)
    assert a != b


def test_freeze_fingerprint_changes_when_the_folder_list_changes():
    kw = dict(
        measure_cols=["combined_base"],
        fps_fallback=40.0,
        distance_limits=None,
        non_reactive_threshold=None,
        low_max_threshold_px=20.0,
        use_per_trial_baseline=True,
        trial_type_filter=None,
    )
    a = rw._freeze_fingerprint(
        _Cfg(overrides={"DS": _Override(freeze_folders=[])}), "DS", **kw
    )
    b = rw._freeze_fingerprint(
        _Cfg(overrides={"DS": _Override(freeze_folders=["july_14_batch_2"])}), "DS", **kw
    )
    assert a != b
