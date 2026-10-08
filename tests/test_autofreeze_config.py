"""Auto-freeze config semantics and the always-live folder rules.

Every dataset is managed automatically (fbpipe.autofreeze): an absent
``freeze:`` block, ``data: auto`` and the legacy ``data: true`` are AUTO; only an
explicit ``data: false`` is the user's standing unfreeze (LIVE). Folder
retirements and the global cutoffs apply whenever a dataset is walked --
thawed, live or rebuilt -- so a rebuild can never revive a retired folder.

``freeze.cutoff_exempt_before`` keeps finished datasets recorded entirely before
the global ``freeze_folders_before`` cutoff exactly as they are today: without
it a rules-live rebuild would retire every one of their trials.
"""

from __future__ import annotations

import datetime as dt
import textwrap
from pathlib import Path

import pytest
import yaml

from fbpipe import freeze
from fbpipe.analysis.threshold import ThresholdRule
from fbpipe.config import DatasetOverride, load_settings
from fbpipe.utils.frozen_folders import (
    FolderRule,
    born_cutoff,
    folder_freeze_rules,
    frozen_folders_for_root,
    is_frozen_folder,
    iter_live_batch_dirs,
)


def _write(tmp_path: Path, overrides: dict, **top) -> Path:
    cfg = {
        "model_path": str(tmp_path / "m.pt"),
        "main_directories": [],
        "cache_dir": str(tmp_path / "cache"),
        # Distinct values for the two gate keys (CLAUDE.md).
        "proboscis_filter": {"max_eye_prob_distance_px": 171.0},
        "distance_limits": {"three_fly_max_eye_prob_distance_px": 149.0},
        "dataset_overrides": overrides,
        **top,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    for var in ("AUTO_FREEZE", "CACHE_DIR", "MODEL_PATH"):
        monkeypatch.delenv(var, raising=False)


# --- data / figures modes ------------------------------------------------------


@pytest.mark.parametrize(
    "block,data_mode,legacy_data",
    [
        ({"data": True, "figures": True}, "auto", True),
        ({"data": "auto"}, "auto", False),
        ({}, "auto", False),
        ({"data": False, "figures": False}, "live", False),
        ({"data": "false"}, "live", False),
    ],
)
def test_data_mode_is_auto_unless_explicitly_false(tmp_path, block, data_mode, legacy_data):
    settings = load_settings(_write(tmp_path, {"DS": {"freeze": block}}))
    ov = settings.dataset_overrides["DS"]
    assert ov.data_mode == data_mode
    # The legacy flag (auto_freeze disabled) only ever reads an explicit true:
    # bool("auto") is True and must not freeze anything.
    assert ov.freeze_data is legacy_data


@pytest.mark.parametrize(
    "value,mode", [(True, "hold"), (False, "live"), (None, "auto"), ("auto", "auto")]
)
def test_figures_mode(tmp_path, value, mode):
    block = {} if value is None else {"figures": value}
    settings = load_settings(_write(tmp_path, {"DS": {"freeze": block}}))
    assert settings.dataset_overrides["DS"].figures_mode == mode


def test_a_dataset_without_an_override_block_defaults_to_auto():
    ov = DatasetOverride()
    assert ov.data_mode == "auto"
    assert ov.figures_mode == "auto"
    assert ov.cutoff_exempt_before is None


@pytest.mark.parametrize(
    "top,expected",
    [({}, True), ({"auto_freeze": {"enabled": False}}, False),
     ({"auto_freeze": False}, False), ({"auto_freeze": {"enabled": True}}, True)],
)
def test_auto_freeze_switch(tmp_path, top, expected):
    assert load_settings(_write(tmp_path, {}, **top)).auto_freeze is expected


# --- the cutoff exemption ------------------------------------------------------


def test_cutoff_exempt_before_parses(tmp_path):
    settings = load_settings(
        _write(tmp_path, {"EB-Control-24-0.1": {"freeze": {"data": True, "cutoff_exempt_before": "2026-06-26"}}})
    )
    assert settings.dataset_overrides["EB-Control-24-0.1"].cutoff_exempt_before == dt.date(2026, 6, 26)


def test_cutoff_exempt_before_typo_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="cutoff_exempt_before"):
        load_settings(_write(tmp_path, {"DS": {"freeze": {"cutoff_exempt_before": "June 26"}}}))


SESSION = textwrap.dedent(
    """\
    Session Metadata
    Born (approx): {born}
    Host: Flybehavior2 | OS: Linux
    """
)


def _batch(root: Path, name: str, stamp: str, born: str = "2026-04-01") -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / f"output_{name}_testing_1_Hexanol_{stamp}_120000.csv").write_text("frame\n0\n")
    (d / "session_metadata.txt").write_text(SESSION.format(born=born))
    return d


class _Cfg:
    def __init__(self, overrides=None, cutoff=dt.date(2026, 6, 26), born=None, keep=None):
        self.freeze_folders_before = cutoff
        self.freeze_folders_born_on_or_after = born
        self.keep_folders_born_on_or_after = keep
        self.dataset_overrides = dict(overrides or {})


def test_exemption_keeps_batches_recorded_before_its_date(tmp_path):
    root = tmp_path / "EB-Control-24-0.1"
    april = _batch(root, "april_11_batch_2", "20260411")
    cfg = _Cfg({"EB-Control-24-0.1": DatasetOverride(cutoff_exempt_before=dt.date(2026, 6, 26))})
    assert is_frozen_folder(cfg, "EB-Control-24-0.1", april) is False
    # Without the exemption the global cutoff retires it.
    assert is_frozen_folder(_Cfg({"EB-Control-24-0.1": DatasetOverride()}), "EB-Control-24-0.1", april) is True


def test_exemption_is_date_bounded(tmp_path):
    """A batch recorded on/after exempt_before follows the cutoff as usual: the
    exemption never silently swallows later batches (CLAUDE.md date bounds)."""
    root = tmp_path / "DS"
    june = _batch(root, "june_20_batch_1", "20260620")
    cfg = _Cfg(
        {"DS": DatasetOverride(cutoff_exempt_before=dt.date(2026, 6, 1))},
        cutoff=dt.date(2026, 6, 26),
    )
    assert is_frozen_folder(cfg, "DS", june) is True


def test_exemption_keeps_the_datasets_own_folder_rules(tmp_path):
    root = tmp_path / "DS"
    april = _batch(root, "april_11_batch_2_rig_3", "20260411")
    cfg = _Cfg({"DS": DatasetOverride(
        cutoff_exempt_before=dt.date(2026, 6, 26),
        freeze_folders=(FolderRule("*_rig_3", dt.date(2026, 8, 11)),),
    )})
    assert is_frozen_folder(cfg, "DS", april) is True


def test_exemption_keeps_the_born_window(tmp_path):
    root = tmp_path / "DS"
    batch = _batch(root, "april_11_batch_2", "20260411", born="2026-08-12")
    cfg = _Cfg(
        {"DS": DatasetOverride(cutoff_exempt_before=dt.date(2026, 6, 26))},
        born=dt.date(2026, 8, 11), keep=dt.date(2026, 8, 20),
    )
    assert is_frozen_folder(cfg, "DS", batch) is True


def _fp(override):
    return freeze.build_fingerprint(
        protocol="v2", measure_cols=["combined_pct"], fps_fallback=40.0,
        distance_limits=(70.0, 250.0), non_reactive_threshold=5.0,
        low_max_threshold_px=5.0, use_per_trial_baseline=True,
        trial_type_filter=None, threshold_rule=ThresholdRule(),
        override=override, tracking=None,
        freeze_folders_before=dt.date(2026, 6, 26),
    )


def test_fingerprint_changes_only_when_the_exemption_is_set():
    """Datasets without the key keep today's fingerprint (their slices stay
    valid); setting or moving it is a folder-policy change like any other."""
    plain = _fp(DatasetOverride())
    assert "cutoff_exempt_before" not in plain["override"]
    assert _fp(DatasetOverride(data_mode="live", figures_mode="hold")) == plain
    exempt = _fp(DatasetOverride(cutoff_exempt_before=dt.date(2026, 6, 26)))
    assert exempt["override"]["cutoff_exempt_before"] == "2026-06-26"
    assert exempt != plain
    assert _fp(DatasetOverride(cutoff_exempt_before=dt.date(2026, 6, 1))) != exempt


# --- folder rules are always live ------------------------------------------------


def test_a_data_frozen_dataset_keeps_its_folder_rules(tmp_path):
    """Its rows splice whole while it is frozen, but the day it is rebuilt the
    rules must apply -- they used to switch off, reviving retired folders."""
    root = tmp_path / "Hex-Training-24-0.1"
    may = _batch(root, "may_22_batch_1_rig_2", "20260522")
    cfg = _Cfg({"Hex-Training-24-0.1": DatasetOverride(freeze_data=True)})
    assert folder_freeze_rules(cfg, "Hex-Training-24-0.1") == (dt.date(2026, 6, 26), ())
    assert is_frozen_folder(cfg, "Hex-Training-24-0.1", may) is True


@pytest.mark.parametrize("kw", [{"thawed": ["Hex-Training-24-0.1"]}, {"thaw_all": True}])
def test_thaw_keeps_folder_retirements_and_cutoffs_live(tmp_path, kw):
    root = tmp_path / "Hex-Training-24-0.1"
    may = _batch(root, "may_22_batch_1_rig_2", "20260522", born="2026-05-15")
    aug = _batch(root, "august_14_batch_1", "20260814", born="2026-08-12")
    live = _batch(root, "august_26_batch_1", "20260826", born="2026-08-21")
    cfg = _Cfg(born=dt.date(2026, 8, 11), keep=dt.date(2026, 8, 20))
    assert born_cutoff(cfg, "Hex-Training-24-0.1", **kw) == dt.date(2026, 8, 11)
    assert is_frozen_folder(cfg, "Hex-Training-24-0.1", may, **kw) is True
    assert is_frozen_folder(cfg, "Hex-Training-24-0.1", aug, **kw) is True
    assert frozen_folders_for_root(cfg, root, **kw) == {may.name, aug.name}
    assert [p.name for p in iter_live_batch_dirs(cfg, root, **kw)] == [live.name]
