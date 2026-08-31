"""config_new.yaml: the four sensitivity cohorts are the only live datasets.

The no-light (``--control``) pre-test runs land in ``*-Sensitivity-*`` folders.
Everything else is finished for now and must be frozen for both data and
figures, so a pipeline run touches only the new cohorts.

config/ is gitignored, so this pins the operative file on disk — the half of
the change that can never be committed.
"""

from pathlib import Path

import pytest
import yaml

CONFIG = Path(__file__).resolve().parent.parent / "config" / "config_new.yaml"

# Deliberately thawed for a single run; empty this list when it is re-frozen,
# and the "everything else is frozen" test below starts guarding it again.
TEMPORARILY_THAWED = []  # re-frozen 2026-08-28

SENSITIVITY = [
    "Hex-Sensitivity-24-0.1",
    "IAA-Sensitivity-24-1",
    "EB-Sensitivity-24-1",
    "3Oct-Sensitivity-24-0.1",
]


class _DupCheckLoader(yaml.SafeLoader):
    """YAML keeps the LAST duplicate key; a silently-lost block is a real bug."""


def _no_duplicates(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise AssertionError(f"duplicate key {key!r} in config_new.yaml")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_DupCheckLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates)


@pytest.fixture(scope="module")
def cfg():
    if not CONFIG.exists():
        pytest.skip("config/config_new.yaml not present (gitignored)")
    return yaml.load(CONFIG.read_text(), Loader=_DupCheckLoader)


def _freeze(cfg, name):
    return ((cfg.get("dataset_overrides") or {}).get(name) or {}).get("freeze") or {}


@pytest.mark.parametrize("name", SENSITIVITY)
def test_sensitivity_dataset_listed(cfg, name):
    assert name in cfg["datasets"]


@pytest.mark.parametrize("name", SENSITIVITY)
def test_sensitivity_dataset_is_live(cfg, name):
    """Live = not data-frozen and not figure-frozen; it is still collecting."""
    f = _freeze(cfg, name)
    assert f.get("data") is not True
    assert f.get("figures") is not True


def test_every_other_dataset_is_frozen(cfg):
    stale = [d for d in cfg["datasets"]
             if d not in SENSITIVITY and d not in TEMPORARILY_THAWED
             and not (_freeze(cfg, d).get("data") is True
                      and _freeze(cfg, d).get("figures") is True)]
    assert stale == [], f"not frozen: {stale}"


def test_no_dataset_lost_from_the_list(cfg):
    """Freezing must not delete history — the old cohorts stay listed."""
    assert len(cfg["datasets"]) >= 20 + len(SENSITIVITY)


def test_frozen_datasets_keep_their_odor_remap(cfg):
    """Figures can be thawed later; the display labels must survive."""
    ov = cfg["dataset_overrides"]
    assert ov["Hex-Control-24-0.1"]["odor_remap"]["Hexanol"] == "Hexanol (0.1%)"
    assert "odor_remap" in ov["EB-Control-24-1"]


def test_folder_freeze_rules_survive(cfg):
    """A `freeze: {data,figures}` edit must not clobber a folder-freeze list."""
    folders = cfg["dataset_overrides"]["Hex-Control-24-0.1"]["freeze"]["folders"]
    assert "august_25_batch_1_rig_2" in folders


@pytest.mark.parametrize("name", TEMPORARILY_THAWED)
def test_temporary_thaw_keeps_its_folder_retirements(cfg, name):
    """A config thaw lifts the DATASET freeze only.

    The per-run ``--thaw`` flag would also drop this dataset's folder rules and
    the global recording/born cutoffs, silently readmitting retired batches --
    which is why the thaw is done here in the config instead.
    """
    fz = cfg["dataset_overrides"][name]["freeze"]
    assert fz.get("data") is False and fz.get("figures") is False
    assert fz.get("folders"), f"{name} lost its folder retirements in the thaw"
