"""Config surface for folder-level freeze.

Two new keys, both optional so an existing config parses unchanged:

    freeze_folders_before: 2026-06-26      # top level

    dataset_overrides:
      EB-Training-24-1:
        freeze:
          folders:
            - july_14_batch_2_rig_2

``folders:`` sits inside the SAME ``freeze:`` block as ``data:``/``figures:``,
so everything about retiring a dataset stays in one place.
"""

from __future__ import annotations

import datetime as dt
import textwrap

import pytest

from fbpipe.config import load_settings
from fbpipe.utils.frozen_folders import FolderRule


BASE = """\
model_path: /tmp/model.pt
main_directories:
  - /tmp/data
cache_dir: /tmp/cache
"""


def write_cfg(tmp_path, body: str):
    path = tmp_path / "config.yaml"
    path.write_text(BASE + textwrap.dedent(body), encoding="utf-8")
    return load_settings(path)


# ── freeze_folders_before ─────────────────────────────────────────────────


def test_freeze_folders_before_parses_to_a_date(tmp_path):
    cfg = write_cfg(tmp_path, "freeze_folders_before: 2026-06-26\n")
    assert cfg.freeze_folders_before == dt.date(2026, 6, 26)


def test_freeze_folders_before_accepts_a_quoted_string(tmp_path):
    """YAML parses a bare 2026-06-26 as a date but a quoted one as a string."""
    cfg = write_cfg(tmp_path, 'freeze_folders_before: "2026-06-26"\n')
    assert cfg.freeze_folders_before == dt.date(2026, 6, 26)


def test_freeze_folders_before_defaults_to_none(tmp_path):
    """Absent key == today's behavior: no folder is date-frozen."""
    assert write_cfg(tmp_path, "").freeze_folders_before is None


def test_unparseable_freeze_folders_before_is_rejected(tmp_path):
    """A typo'd date must not silently disable the whole rule -- that would
    quietly put every retired fly back into the figures."""
    with pytest.raises(ValueError, match="freeze_folders_before"):
        write_cfg(tmp_path, "freeze_folders_before: sometime-in-june\n")


# ── freeze.folders ────────────────────────────────────────────────────────


def test_freeze_folders_list_parses(tmp_path):
    cfg = write_cfg(
        tmp_path,
        """
        dataset_overrides:
          EB-Training-24-1:
            freeze:
              folders:
                - july_14_batch_2_rig_2
                - july_15_batch_1
        """,
    )
    assert cfg.dataset_overrides["EB-Training-24-1"].freeze_folders == (
        FolderRule("july_14_batch_2_rig_2", None),
        FolderRule("july_15_batch_1", None),
    )


def test_freeze_folders_defaults_to_empty(tmp_path):
    cfg = write_cfg(
        tmp_path,
        """
        dataset_overrides:
          EB-Training-24-1:
            freeze:
              data: true
        """,
    )
    assert cfg.dataset_overrides["EB-Training-24-1"].freeze_folders == ()


def test_freeze_folders_coexists_with_data_and_figures(tmp_path):
    """All three keys live in one block and must not clobber each other."""
    cfg = write_cfg(
        tmp_path,
        """
        dataset_overrides:
          Hex-Training-24-0.1:
            freeze:
              data: false
              figures: true
              folders: [may_22_batch_2]
        """,
    )
    ov = cfg.dataset_overrides["Hex-Training-24-0.1"]
    assert (ov.freeze_data, ov.freeze_figures, ov.freeze_folders) == (
        False,
        True,
        (FolderRule("may_22_batch_2", None),),
    )


def test_a_scalar_folders_value_is_rejected(tmp_path):
    """``folders: july_14_batch_2_rig_2`` (no list) would otherwise iterate as
    characters and freeze nothing, silently."""
    with pytest.raises(ValueError, match="folders"):
        write_cfg(
            tmp_path,
            """
            dataset_overrides:
              EB-Training-24-1:
                freeze:
                  folders: july_14_batch_2_rig_2
            """,
        )


# ── the real config still loads ───────────────────────────────────────────


def test_operative_config_still_loads():
    """config_new.yaml is the operative config; parsing must not regress."""
    cfg = load_settings("config/config_new.yaml")
    assert cfg.datasets


RIG3_GOOD_FROM = dt.date(2026, 8, 11)


def test_every_rig_3_rule_is_bounded_at_the_repair_date():
    """rig_3 was repaired 2026-08-11; only batches BEFORE that stay retired.

    An unbounded ``*_rig_3`` glob auto-freezes every rig_3 batch recorded from
    now on, which silently swallowed august_13_batch_2_rig_3. Every dataset
    carrying the rule must bound it, and all at the same date -- a per-dataset
    drift would retire rig_3 from one arm of a trained/control pair only.
    """
    cfg = load_settings("config/config_new.yaml")
    seen = []
    for dataset, override in (cfg.dataset_overrides or {}).items():
        for rule in getattr(override, "freeze_folders", ()) or ():
            if rule.pattern == "*_rig_3":
                seen.append(dataset)
                assert rule.before == RIG3_GOOD_FROM, (
                    f"{dataset}: *_rig_3 bound at {rule.before}, "
                    f"expected {RIG3_GOOD_FROM}"
                )
    # The rule spans both arms of all three live pairs; if it ever covers fewer
    # datasets than that, a cohort has been silently un-retired.
    assert set(seen) == {
        "EB-Control-24-1",
        "EB-Training-24-1",
        "Hex-Control-24-0.01",
        "Hex-Training-24-0.01",
        "3Oct-Control-24-0.1",
        "3Oct-Training-24-0.1",
    }


def test_rig_3_from_august_11_is_live(tmp_path):
    """The bounded rule keeps old rig_3 frozen and lets repaired rig_3 through."""
    from fbpipe.utils.frozen_folders import is_frozen_folder

    cfg = load_settings("config/config_new.yaml")

    def batch(name: str, stamp: str):
        d = tmp_path / name
        d.mkdir()
        (d / f"output_{name}_testing_1_Hexanol_{stamp}_120000.csv").write_text(
            "", encoding="utf-8"
        )
        return d

    before = batch("august_09_batch_1_rig_3", "20260809")
    on_the_day = batch("august_11_batch_1_rig_3", "20260811")
    after = batch("august_13_batch_2_rig_3", "20260813")

    assert is_frozen_folder(cfg, "EB-Training-24-1", before) is True
    assert is_frozen_folder(cfg, "EB-Training-24-1", on_the_day) is False
    assert is_frozen_folder(cfg, "EB-Training-24-1", after) is False
