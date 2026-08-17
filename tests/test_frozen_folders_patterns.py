"""``freeze.folders`` entries may be globs, and may carry a date bound.

Retiring "every rig_3 batch" or "every batch 2 before july 27" by hand means
listing dozens of folder names that go stale the moment a batch is added or
renamed. Three entry forms, all in the same list:

    freeze:
      folders:
        - july_14_batch_2_rig_2            # one exact folder
        - "*_rig_3"                        # every folder matching the glob
        - match: "*batch_2*"               # glob, bounded by recording date
          before: 2026-07-27

A bare name and a glob are the same code path (fnmatch on a pattern with no
wildcard is an exact match); the mapping form adds the date bound.
"""

from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path

import pytest

from fbpipe.config import load_settings
from fbpipe.utils.frozen_folders import FolderRule, is_frozen_folder


class _Cfg:
    def __init__(self, *, freeze_folders_before=None, overrides=None):
        self.freeze_folders_before = freeze_folders_before
        self.dataset_overrides = dict(overrides or {})


class _Override:
    def __init__(self, *, freeze_data=False, freeze_folders=()):
        self.freeze_data = freeze_data
        self.freeze_folders = tuple(freeze_folders)


def _batch(root: Path, name: str, stamp: str | None = None) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    if stamp:
        (d / f"output_{name}_testing_1_Hexanol_{stamp}_120000.csv").write_text(
            "frame\n0\n", encoding="utf-8"
        )
    return d


def _cfg_with(rules):
    return _Cfg(overrides={"EB-Training-24-1": _Override(freeze_folders=rules)})


# ── globs ─────────────────────────────────────────────────────────────────


def test_glob_freezes_every_matching_folder(tmp_path):
    batch = _batch(tmp_path, "july_18_batch_1_rig_3", "20260718")
    assert is_frozen_folder(_cfg_with(["*_rig_3"]), "EB-Training-24-1", batch) is True


def test_glob_leaves_non_matching_folders_alone(tmp_path):
    batch = _batch(tmp_path, "july_18_batch_1_rig_2", "20260718")
    assert is_frozen_folder(_cfg_with(["*_rig_3"]), "EB-Training-24-1", batch) is False


def test_rig_3_glob_does_not_catch_a_rig_2_folder_ending_in_3(tmp_path):
    """``*_rig_3`` must anchor on the rig suffix, not merely contain a 3."""
    batch = _batch(tmp_path, "july_13_batch_3", "20260713")
    assert is_frozen_folder(_cfg_with(["*_rig_3"]), "EB-Training-24-1", batch) is False


def test_a_bare_name_still_matches_exactly(tmp_path):
    """The plain-name form must keep working -- fnmatch on a wildcard-free
    pattern is an exact match, but only if nothing else treats it as a prefix."""
    batch = _batch(tmp_path, "july_14_batch_2_rig_2_extra", "20260714")
    cfg = _cfg_with(["july_14_batch_2_rig_2"])
    assert is_frozen_folder(cfg, "EB-Training-24-1", batch) is False


def test_glob_needs_no_recording_date(tmp_path):
    batch = _batch(tmp_path, "mystery_rig_3", stamp=None)
    assert is_frozen_folder(_cfg_with(["*_rig_3"]), "EB-Training-24-1", batch) is True


# ── glob + date bound ─────────────────────────────────────────────────────


def test_bounded_rule_freezes_a_match_before_the_date(tmp_path):
    batch = _batch(tmp_path, "july_19_batch_2_rig_2", "20260719")
    cfg = _cfg_with([FolderRule("*batch_2*", date(2026, 7, 27))])
    assert is_frozen_folder(cfg, "EB-Training-24-1", batch) is True


def test_bounded_rule_keeps_a_match_on_or_after_the_date(tmp_path):
    batch = _batch(tmp_path, "july_28_batch_2_rig_2", "20260728")
    cfg = _cfg_with([FolderRule("*batch_2*", date(2026, 7, 27))])
    assert is_frozen_folder(cfg, "EB-Training-24-1", batch) is False


def test_bounded_rule_boundary_day_is_kept(tmp_path):
    batch = _batch(tmp_path, "july_27_batch_2_rig_2", "20260727")
    cfg = _cfg_with([FolderRule("*batch_2*", date(2026, 7, 27))])
    assert is_frozen_folder(cfg, "EB-Training-24-1", batch) is False


def test_bounded_rule_ignores_a_non_matching_folder_before_the_date(tmp_path):
    batch = _batch(tmp_path, "july_19_batch_1_rig_2", "20260719")
    cfg = _cfg_with([FolderRule("*batch_2*", date(2026, 7, 27))])
    assert is_frozen_folder(cfg, "EB-Training-24-1", batch) is False


def test_bounded_rule_keeps_an_undated_match(tmp_path):
    """Same fail-open stance as the global cutoff: a bounded rule cannot fire
    without a date, so an undated folder survives (and is reported)."""
    batch = _batch(tmp_path, "undated_batch_2", stamp=None)
    cfg = _cfg_with([FolderRule("*batch_2*", date(2026, 7, 27))])
    assert is_frozen_folder(cfg, "EB-Training-24-1", batch) is False


def test_a_raw_mapping_in_freeze_folders_is_rejected(tmp_path):
    """Config parsing turns a ``{match, before}`` mapping into a FolderRule. If
    something hands the resolver the raw mapping instead, that is a bug -- and
    stringifying it would produce a pattern matching nothing, silently freezing
    nothing. Fail loudly instead."""
    batch = _batch(tmp_path, "july_19_batch_2_rig_2", "20260719")
    cfg = _cfg_with([{"match": "*batch_2*", "before": date(2026, 7, 27)}])
    with pytest.raises(TypeError, match="FolderRule"):
        is_frozen_folder(cfg, "EB-Training-24-1", batch)


def test_rules_union(tmp_path):
    """rig_3 rule and the bounded batch_2 rule coexist on one dataset."""
    cfg = _cfg_with(["*_rig_3", FolderRule("*batch_2*", date(2026, 7, 27))])
    rig3_late = _batch(tmp_path, "august_09_batch_1_rig_3", "20260809")
    batch2_early = _batch(tmp_path, "july_13_batch_2_rig_2", "20260713")
    batch1_late = _batch(tmp_path, "august_09_batch_1_rig_2", "20260809")
    assert is_frozen_folder(cfg, "EB-Training-24-1", rig3_late) is True
    assert is_frozen_folder(cfg, "EB-Training-24-1", batch2_early) is True
    assert is_frozen_folder(cfg, "EB-Training-24-1", batch1_late) is False


# ── config parsing ────────────────────────────────────────────────────────

BASE = "model_path: /tmp/m.pt\nmain_directories:\n  - /tmp/d\ncache_dir: /tmp/c\n"


def _load(tmp_path, body):
    p = tmp_path / "config.yaml"
    p.write_text(BASE + textwrap.dedent(body), encoding="utf-8")
    return load_settings(p)


def test_config_parses_mixed_entry_forms(tmp_path):
    cfg = _load(
        tmp_path,
        """
        dataset_overrides:
          EB-Training-24-1:
            freeze:
              folders:
                - july_14_batch_2_rig_2
                - "*_rig_3"
                - match: "*batch_2*"
                  before: 2026-07-27
        """,
    )
    rules = cfg.dataset_overrides["EB-Training-24-1"].freeze_folders
    assert rules == (
        FolderRule("july_14_batch_2_rig_2", None),
        FolderRule("*_rig_3", None),
        FolderRule("*batch_2*", date(2026, 7, 27)),
    )


def test_config_rejects_a_mapping_without_match(tmp_path):
    with pytest.raises(ValueError, match="match"):
        _load(
            tmp_path,
            """
            dataset_overrides:
              EB-Training-24-1:
                freeze:
                  folders:
                    - before: 2026-07-27
            """,
        )


def test_config_rejects_an_unparseable_before(tmp_path):
    """A typo'd date must not degrade to an unbounded rule -- that would freeze
    every matching folder instead of only the old ones."""
    with pytest.raises(ValueError, match="before"):
        _load(
            tmp_path,
            """
            dataset_overrides:
              EB-Training-24-1:
                freeze:
                  folders:
                    - match: "*batch_2*"
                      before: end-of-july
            """,
        )
