"""Folder-level freeze: freeze individual experiment folders inside a live dataset.

The whole-dataset ``freeze:`` block (``data`` / ``figures``) is all-or-nothing.
Datasets like ``Hex-Training-24-0.1`` outlived their own rig plumbing: the
may/june batches ran a different odor panel than the august ones, so the old
batches must stop contributing to figures while the dataset itself stays live.

Two rules, unioned:

* ``freeze_folders_before: 2026-06-26`` (top level) — every batch recorded
  strictly BEFORE that date is frozen, resolved from the batch's own recording
  date via ``rig_gates.read_batch_date``;
* ``dataset_overrides.<ds>.freeze.folders: [july_14_batch_2_rig_2]`` — an
  explicit per-dataset list of experiment folder names.

Semantics pinned here:

* the cutoff is INCLUSIVE of the keep side — a batch recorded ON the cutoff
  date is KEPT (``freeze_folders_before`` reads "freeze what is before this");
* a dataset already frozen for DATA is never folder-frozen — its root is never
  walked, so there are no folders to judge (the "live datasets only" rule);
* a batch whose date cannot be derived stays LIVE and is reported, because
  silently dropping data is worse than silently keeping it;
* ``--thaw <dataset>`` / ``--thaw-all`` lift folder freezes too.
"""

from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path

import pytest

from fbpipe.utils.frozen_folders import (
    frozen_folders_for_root,
    is_frozen_folder,
)


SESSION_METADATA = textwrap.dedent(
    """\
    Session Metadata
    ===============

    Run Context
    -----------
    Intake Logged (UTC): 2026-05-22T20:02:09.137828Z
    Host: Flybehavior2 | OS: Linux 6.12.34+rpt-rpi-2712
    """
)


class _Override:
    """Stand-in for config.DatasetOverride (only the fields the resolver reads)."""

    def __init__(self, *, freeze_data=False, freeze_folders=()):
        self.freeze_data = freeze_data
        self.freeze_folders = tuple(freeze_folders)


class _Cfg:
    """Stand-in for config.Settings (only the fields the resolver reads)."""

    def __init__(self, *, freeze_folders_before=None, overrides=None):
        self.freeze_folders_before = freeze_folders_before
        self.dataset_overrides = dict(overrides or {})


def make_batch(root: Path, name: str, stamp: str | None) -> Path:
    """A batch dir whose recording date comes from its sidecar filename stamp."""
    batch = root / name
    batch.mkdir(parents=True, exist_ok=True)
    (batch / "session_metadata.txt").write_text(SESSION_METADATA, encoding="utf-8")
    if stamp is not None:
        (batch / f"output_{name}_training_2_Hexanol_{stamp}_160000.csv").write_text(
            "frame\n0\n", encoding="utf-8"
        )
    return batch


# ── the date cutoff ───────────────────────────────────────────────────────


def test_batch_recorded_before_cutoff_is_frozen(tmp_path):
    batch = make_batch(tmp_path, "may_22_batch_1_rig_2", "20260522")
    cfg = _Cfg(freeze_folders_before=date(2026, 6, 26))
    assert is_frozen_folder(cfg, "Hex-Training-24-0.1", batch) is True


def test_batch_recorded_after_cutoff_is_live(tmp_path):
    batch = make_batch(tmp_path, "august_10_batch_1_rig_2", "20260810")
    cfg = _Cfg(freeze_folders_before=date(2026, 6, 26))
    assert is_frozen_folder(cfg, "Hex-Training-24-0.1", batch) is False


def test_batch_recorded_on_the_cutoff_date_is_kept(tmp_path):
    """The boundary. ``freeze_folders_before`` freezes what is strictly before
    it, so 2026-06-26 itself is the first KEPT day -- matching "june 26 forward"."""
    batch = make_batch(tmp_path, "june_26_batch_1", "20260626")
    cfg = _Cfg(freeze_folders_before=date(2026, 6, 26))
    assert is_frozen_folder(cfg, "RandomPanel-Training-24-10", batch) is False


def test_day_before_the_cutoff_is_frozen(tmp_path):
    batch = make_batch(tmp_path, "june_25_batch_1", "20260625")
    cfg = _Cfg(freeze_folders_before=date(2026, 6, 26))
    assert is_frozen_folder(cfg, "RandomPanel-Training-24-10", batch) is True


def test_no_cutoff_configured_freezes_nothing(tmp_path):
    """An absent ``freeze_folders_before`` must reproduce today's behavior."""
    batch = make_batch(tmp_path, "april_16_batch_1", "20260416")
    assert is_frozen_folder(_Cfg(), "Hex-Control-24-0.01", batch) is False


# ── the explicit per-dataset folder list ──────────────────────────────────


def test_folder_named_in_config_is_frozen(tmp_path):
    batch = make_batch(tmp_path, "july_14_batch_2_rig_2", "20260714")
    cfg = _Cfg(
        overrides={
            "EB-Training-24-1": _Override(freeze_folders=["july_14_batch_2_rig_2"])
        }
    )
    assert is_frozen_folder(cfg, "EB-Training-24-1", batch) is True


def test_folder_list_is_scoped_to_its_own_dataset(tmp_path):
    """Naming a folder under one dataset must not freeze a same-named folder
    in another -- batch folder names repeat across datasets (both EB-Control-24-1
    and EB-Training-24-1 have a july_14_* batch)."""
    batch = make_batch(tmp_path, "july_14_batch_2_rig_2", "20260714")
    cfg = _Cfg(
        overrides={
            "EB-Training-24-1": _Override(freeze_folders=["july_14_batch_2_rig_2"])
        }
    )
    assert is_frozen_folder(cfg, "EB-Control-24-1", batch) is False


def test_config_list_freezes_a_folder_the_date_rule_would_keep(tmp_path):
    """The two rules union: a post-cutoff folder still freezes if listed."""
    batch = make_batch(tmp_path, "july_14_batch_2_rig_2", "20260714")
    cfg = _Cfg(
        freeze_folders_before=date(2026, 6, 26),
        overrides={
            "EB-Training-24-1": _Override(freeze_folders=["july_14_batch_2_rig_2"])
        },
    )
    assert is_frozen_folder(cfg, "EB-Training-24-1", batch) is True


def test_config_list_needs_no_recording_date(tmp_path):
    """An explicitly listed folder freezes even with no sidecar to date it --
    the list is a direct instruction, not a hint."""
    batch = make_batch(tmp_path, "mystery_batch", stamp=None)
    cfg = _Cfg(overrides={"EB-Training-24-1": _Override(freeze_folders=["mystery_batch"])})
    assert is_frozen_folder(cfg, "EB-Training-24-1", batch) is True


# ── interaction with the existing whole-dataset freeze ────────────────────


def test_data_frozen_dataset_is_never_folder_frozen(tmp_path):
    """"Live datasets only": a dataset with ``freeze.data: true`` never has its
    root walked, so its cached rows must splice in whole. Folder-freezing one of
    its batches would silently delete rows from the wide CSV."""
    batch = make_batch(tmp_path, "april_11_batch_2", "20260411")
    cfg = _Cfg(
        freeze_folders_before=date(2026, 6, 26),
        overrides={"EB-Control-24-0.1": _Override(freeze_data=True)},
    )
    assert is_frozen_folder(cfg, "EB-Control-24-0.1", batch) is False


# ── thaw ──────────────────────────────────────────────────────────────────


def test_thawed_dataset_ignores_folder_freeze(tmp_path):
    batch = make_batch(tmp_path, "may_22_batch_1_rig_2", "20260522")
    cfg = _Cfg(freeze_folders_before=date(2026, 6, 26))
    assert (
        is_frozen_folder(cfg, "Hex-Training-24-0.1", batch, thawed=["Hex-Training-24-0.1"])
        is False
    )


def test_thaw_all_ignores_folder_freeze(tmp_path):
    batch = make_batch(tmp_path, "may_22_batch_1_rig_2", "20260522")
    cfg = _Cfg(freeze_folders_before=date(2026, 6, 26))
    assert is_frozen_folder(cfg, "Hex-Training-24-0.1", batch, thaw_all=True) is False


# ── undated batches fail OPEN ─────────────────────────────────────────────


def test_undated_batch_stays_live(tmp_path, capsys):
    """No sidecar and no dated metadata line -> cannot judge -> keep it, and say
    so. Dropping data on a failed lookup would be silent loss."""
    batch = tmp_path / "undated_batch"
    batch.mkdir()
    cfg = _Cfg(freeze_folders_before=date(2026, 6, 26))
    assert is_frozen_folder(cfg, "Hex-Training-24-0.1", batch) is False
    assert "undated_batch" in capsys.readouterr().out


# ── whole-root resolution ─────────────────────────────────────────────────


def test_frozen_folders_for_root_names_only_the_frozen_ones(tmp_path):
    root = tmp_path / "Hex-Training-24-0.1"
    root.mkdir()
    make_batch(root, "may_22_batch_1_rig_2", "20260522")
    make_batch(root, "june_01_batch_1", "20260601")
    make_batch(root, "august_10_batch_1_rig_2", "20260810")
    cfg = _Cfg(freeze_folders_before=date(2026, 6, 26))
    assert frozen_folders_for_root(cfg, root) == {
        "may_22_batch_1_rig_2",
        "june_01_batch_1",
    }


def test_frozen_folders_for_root_infers_the_dataset_from_the_root_name(tmp_path):
    """Callers pass a root path, not a dataset name; the folder list must still
    be found. Roots are named for their dataset on both the live and secured mirrors."""
    root = tmp_path / "EB-Training-24-1"
    root.mkdir()
    make_batch(root, "july_14_batch_2_rig_2", "20260714")
    make_batch(root, "july_15_batch_1", "20260715")
    cfg = _Cfg(
        overrides={
            "EB-Training-24-1": _Override(freeze_folders=["july_14_batch_2_rig_2"])
        }
    )
    assert frozen_folders_for_root(cfg, root) == {"july_14_batch_2_rig_2"}


def test_frozen_folders_for_root_is_empty_when_nothing_is_configured(tmp_path):
    root = tmp_path / "EB-Training-24-1"
    root.mkdir()
    make_batch(root, "july_14_batch_2_rig_2", "20260714")
    assert frozen_folders_for_root(_Cfg(), root) == set()
