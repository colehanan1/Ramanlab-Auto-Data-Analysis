"""Freeze folders by the SUBJECT's birth date, not the recording date.

A cohort question ("only flies born before 2026-08-11") cannot be answered with
``freeze_folders_before``: recording date and birth date are days apart and the
gap is not constant, so a recording-date cutoff either keeps flies that are too
young or drops batches that are fine.

    freeze_folders_born_on_or_after: 2026-08-11   # top level

freezes every experiment folder whose subjects were born ON OR AFTER that date,
leaving "born strictly before" live. The direction is the complement of
``freeze_folders_before`` (which freezes what is EARLIER), so the key spells its
boundary out rather than reusing the ``_before`` suffix.

The birth date comes from the batch's own ``session_metadata.txt``
(``Born (approx): 2026-08-10``) -- the same file ``rig_gates`` reads for host
and recording date. That file is full of unrelated dates (starvation, retinal
prep, odor vial), so only the labeled ``Born`` line counts.

Fail-open, matching every other freeze rule: a batch with no readable birth date
stays LIVE with a warning. The rule can only ever be reached by a positive
match, so a failed lookup keeps data rather than dropping it.
"""

from __future__ import annotations

import datetime as dt
import textwrap
from datetime import date
from pathlib import Path

import pytest

from fbpipe.utils.frozen_folders import (
    any_folder_rules,
    frozen_folders_for_root,
    is_frozen_folder,
    iter_live_batch_dirs,
)
from fbpipe.utils.rig_gates import read_batch_born_date


def metadata(born: str | None = "2026-08-10", *, label: str = "Born (approx)") -> str:
    """A session_metadata.txt body, optionally without its birth line.

    Carries the neighbouring dates from a real file so a test that passes here
    cannot be passing by grabbing "the first ISO date in the file".
    """
    born_line = "" if born is None else f"{label}: {born}\n"
    return textwrap.dedent(
        f"""\
        Session Metadata
        ===============

        Subject
        -------
        Fly ID: 1
        Fly Type: GR5a-OLD
        Sex: Female virgins
        {born_line}\
        Light cycle: 12L:12D

        Starvation Input
        ----------------
        Starved Date: 2026-08-13
        Starved Datetime (ISO, local naive): 2026-08-13T10:07:00

        Reagent Metadata
        ----------------
        Retinal vial activate / out-of-fridge date: 2026-08-03
        Odor vial creation date: 2026-08-10

        Run Context
        -----------
        Intake Logged (UTC): 2026-08-14T15:17:05.731097Z
        Host: BehaviorLocust | OS: Linux 6.12.34+rpt-rpi-2712
        """
    )


class _Override:
    def __init__(self, *, freeze_data=False, freeze_folders=()):
        self.freeze_data = freeze_data
        self.freeze_folders = tuple(freeze_folders)


class _Cfg:
    """Stand-in for config.Settings (only the fields the resolver reads)."""

    def __init__(
        self,
        *,
        freeze_folders_before=None,
        freeze_folders_born_on_or_after=None,
        overrides=None,
    ):
        self.freeze_folders_before = freeze_folders_before
        self.freeze_folders_born_on_or_after = freeze_folders_born_on_or_after
        self.dataset_overrides = dict(overrides or {})


def make_batch(
    root: Path, name: str, *, born: str | None = "2026-08-10", stamp: str = "20260814"
) -> Path:
    batch = root / name
    batch.mkdir(parents=True, exist_ok=True)
    (batch / "session_metadata.txt").write_text(metadata(born), encoding="utf-8")
    (batch / f"output_{name}_training_2_Hexanol_{stamp}_160000.csv").write_text(
        "frame\n0\n", encoding="utf-8"
    )
    return batch


# ── reading the birth date ────────────────────────────────────────────────


def test_born_date_is_read_from_the_labeled_line(tmp_path):
    batch = make_batch(tmp_path, "august_14_batch_1_rig_2", born="2026-08-10")
    assert read_batch_born_date(batch) == date(2026, 8, 10)


def test_born_date_is_not_confused_by_the_other_dates_in_the_file(tmp_path):
    """Starvation / retinal / odor-vial / intake dates all sit in this file and
    two of them precede the Born line. Only the labeled line may be read."""
    batch = make_batch(tmp_path, "august_14_batch_1_rig_2", born="2026-07-04")
    assert read_batch_born_date(batch) == date(2026, 7, 4)


def test_missing_born_line_reads_as_unknown(tmp_path):
    batch = make_batch(tmp_path, "may_21_batch_1_rig_2", born=None)
    assert read_batch_born_date(batch) is None


def test_missing_metadata_file_reads_as_unknown(tmp_path):
    batch = tmp_path / "august_14_batch_1_rig_2"
    batch.mkdir()
    assert read_batch_born_date(batch) is None


def test_unparseable_born_date_reads_as_unknown(tmp_path):
    batch = make_batch(tmp_path, "august_14_batch_1_rig_2", born="2026-02-31")
    assert read_batch_born_date(batch) is None


# ── the cutoff ────────────────────────────────────────────────────────────


def test_fly_born_before_the_cutoff_is_live(tmp_path):
    batch = make_batch(tmp_path, "august_14_batch_1_rig_2", born="2026-08-10")
    cfg = _Cfg(freeze_folders_born_on_or_after=date(2026, 8, 11))
    assert is_frozen_folder(cfg, "3Oct-Control-24-0.1", batch) is False


def test_fly_born_on_the_cutoff_is_frozen(tmp_path):
    """The boundary. "born before 8/11" excludes 8/11 itself."""
    batch = make_batch(tmp_path, "august_15_batch_1_rig_3", born="2026-08-11")
    cfg = _Cfg(freeze_folders_born_on_or_after=date(2026, 8, 11))
    assert is_frozen_folder(cfg, "3Oct-Training-24-0.1", batch) is True


def test_fly_born_after_the_cutoff_is_frozen(tmp_path):
    batch = make_batch(tmp_path, "august_25_batch_1_rig_2", born="2026-08-19")
    cfg = _Cfg(freeze_folders_born_on_or_after=date(2026, 8, 11))
    assert is_frozen_folder(cfg, "Hex-Control-24-0.1", batch) is True


def test_no_born_cutoff_configured_freezes_nothing(tmp_path):
    """Absent key must reproduce today's behavior exactly."""
    batch = make_batch(tmp_path, "august_25_batch_1_rig_2", born="2026-08-19")
    assert is_frozen_folder(_Cfg(), "Hex-Control-24-0.1", batch) is False


def test_undated_birth_stays_live_with_a_warning(tmp_path, capsys):
    """Fail-open, like the recording-date rule: no birth date, no freeze."""
    batch = make_batch(tmp_path, "may_21_batch_1_rig_2", born=None)
    cfg = _Cfg(freeze_folders_born_on_or_after=date(2026, 8, 11))
    assert is_frozen_folder(cfg, "Hex-Control-24-0.01", batch) is False
    assert "may_21_batch_1_rig_2" in capsys.readouterr().out


def test_born_cutoff_unions_with_the_recording_cutoff(tmp_path):
    """An old batch is frozen by the recording rule even though its flies were
    born long before the birth cutoff -- the rules union, never cancel."""
    batch = make_batch(
        tmp_path, "may_22_batch_1_rig_2", born="2026-05-18", stamp="20260522"
    )
    cfg = _Cfg(
        freeze_folders_before=date(2026, 6, 26),
        freeze_folders_born_on_or_after=date(2026, 8, 11),
    )
    assert is_frozen_folder(cfg, "Hex-Training-24-0.1", batch) is True


# ── interaction with the existing escape hatches ──────────────────────────


def test_data_frozen_dataset_is_never_born_frozen(tmp_path):
    """Its root is never walked and its rows splice in whole; folder-freezing
    one of its batches would silently delete rows from that cached slice."""
    batch = make_batch(tmp_path, "august_25_batch_1_rig_2", born="2026-08-19")
    cfg = _Cfg(
        freeze_folders_born_on_or_after=date(2026, 8, 11),
        overrides={"RandomPanel-24-1": _Override(freeze_data=True)},
    )
    assert is_frozen_folder(cfg, "RandomPanel-24-1", batch) is False


def test_thaw_all_reaches_the_born_cutoff(tmp_path):
    batch = make_batch(tmp_path, "august_25_batch_1_rig_2", born="2026-08-19")
    cfg = _Cfg(freeze_folders_born_on_or_after=date(2026, 8, 11))
    assert is_frozen_folder(cfg, "Hex-Control-24-0.1", batch, thaw_all=True) is False


def test_thawing_one_dataset_reaches_the_born_cutoff(tmp_path):
    batch = make_batch(tmp_path, "august_25_batch_1_rig_2", born="2026-08-19")
    cfg = _Cfg(freeze_folders_born_on_or_after=date(2026, 8, 11))
    assert (
        is_frozen_folder(
            cfg, "Hex-Control-24-0.1", batch, thawed=["Hex-Control-24-0.1"]
        )
        is False
    )


def test_any_folder_rules_sees_a_lone_born_cutoff(tmp_path):
    """``warn_if_unmarked`` guards --figures-only against a wide table built
    before the rule existed. A born-only config must trip that guard too."""
    assert any_folder_rules(_Cfg(freeze_folders_born_on_or_after=date(2026, 8, 11)))
    assert not any_folder_rules(_Cfg())


# ── the walkers ───────────────────────────────────────────────────────────


def test_iter_live_batch_dirs_skips_the_too_young(tmp_path):
    root = tmp_path / "Hex-Control-24-0.1"
    make_batch(root, "august_14_batch_1_rig_2", born="2026-08-10")
    make_batch(root, "august_25_batch_1_rig_2", born="2026-08-19")
    cfg = _Cfg(freeze_folders_born_on_or_after=date(2026, 8, 11))
    got = [d.name for d in iter_live_batch_dirs(cfg, root)]
    assert got == ["august_14_batch_1_rig_2"]


def test_frozen_folders_for_root_names_the_too_young(tmp_path):
    root = tmp_path / "Hex-Control-24-0.1"
    make_batch(root, "august_14_batch_1_rig_2", born="2026-08-10")
    make_batch(root, "august_25_batch_1_rig_2", born="2026-08-19")
    cfg = _Cfg(freeze_folders_born_on_or_after=date(2026, 8, 11))
    assert frozen_folders_for_root(cfg, root) == {"august_25_batch_1_rig_2"}


# ── config parsing ────────────────────────────────────────────────────────

BASE = """\
model_path: /tmp/model.pt
main_directories:
  - /tmp/data
cache_dir: /tmp/cache
"""


def write_cfg(tmp_path, body: str):
    from fbpipe.config import load_settings

    path = tmp_path / "config.yaml"
    path.write_text(BASE + textwrap.dedent(body), encoding="utf-8")
    return load_settings(path)


def test_born_cutoff_parses_to_a_date(tmp_path):
    cfg = write_cfg(tmp_path, "freeze_folders_born_on_or_after: 2026-08-11\n")
    assert cfg.freeze_folders_born_on_or_after == dt.date(2026, 8, 11)


def test_born_cutoff_accepts_a_quoted_string(tmp_path):
    cfg = write_cfg(tmp_path, 'freeze_folders_born_on_or_after: "2026-08-11"\n')
    assert cfg.freeze_folders_born_on_or_after == dt.date(2026, 8, 11)


def test_born_cutoff_defaults_to_none(tmp_path):
    assert write_cfg(tmp_path, "").freeze_folders_born_on_or_after is None


def test_unparseable_born_cutoff_is_rejected(tmp_path):
    """A typo must not degrade to "no rule" -- that quietly puts every excluded
    fly back into the figures, the exact outcome the key exists to prevent."""
    with pytest.raises(ValueError, match="freeze_folders_born_on_or_after"):
        write_cfg(tmp_path, "freeze_folders_born_on_or_after: mid-august\n")


# ── cache invalidation ────────────────────────────────────────────────────


def test_born_cutoff_changes_the_freeze_fingerprint():
    """A slice cached under one cohort policy must not be served under another."""
    from fbpipe.freeze import build_fingerprint

    kw = dict(
        protocol="v2",
        measure_cols=["distance"],
        fps_fallback=30.0,
        distance_limits=None,
        non_reactive_threshold=None,
        low_max_threshold_px=1.0,
        use_per_trial_baseline=False,
        trial_type_filter=None,
        threshold_rule=None,
        override=None,
        tracking=None,
    )
    a = build_fingerprint(**kw, freeze_folders_born_on_or_after=date(2026, 8, 11))
    b = build_fingerprint(**kw, freeze_folders_born_on_or_after=date(2026, 8, 20))
    c = build_fingerprint(**kw)
    assert a != b
    assert a != c


def test_born_cutoff_changes_the_pipeline_cache_key(tmp_path):
    """The manifest half of the cache only sees raw table bytes, and changing a
    cohort rule moves none -- so the expectation must carry it."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_rw_born", Path("scripts/pipeline/run_workflows.py").resolve()
    )
    rw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rw)

    root = tmp_path / "Hex-Control-24-0.1"
    root.mkdir()

    def expectation(cut):
        cfg = _Cfg(freeze_folders_born_on_or_after=cut)
        cfg.non_reactive_span_px = 1.0
        cfg.class2_min = 0.0
        cfg.class2_max = 1.0
        return rw._pipeline_expectation(cfg, root)

    assert expectation(date(2026, 8, 11)) != expectation(date(2026, 8, 20))
    assert expectation(date(2026, 8, 11)) != expectation(None)
