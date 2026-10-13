"""config_new.yaml: the sensitivity cohorts are the only live datasets.

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

# Deliberately thawed for a single run; empty this mapping when it is
# re-frozen, and the "everything else is frozen" test below starts guarding it
# again. The value is the folder retirements the dataset carried BEFORE the
# thaw, which a config thaw must preserve -- ``None`` means it never had any
# and the global cutoffs at the top of config_new.yaml are its only folder
# rules, which is not the same thing as having lost them.
TEMPORARILY_THAWED = {
    # Thawed 2026-09-01, swapped with the 3Oct-24-0.1 pair, which went back to
    # data+figures frozen the same day.
    "Hex-Control-24-0.1": ["august_25_batch_1_rig_2", "august_25_batch_1_rig_3"],
    "Hex-Training-24-0.1": None,
}

SENSITIVITY = [
    "Hex-Sensitivity-24-0.1",
    "IAA-Sensitivity-24-1",
    # EB-Sensitivity-24-1 and IAA-Sensitivity-36-1 were registered here and in
    # config_new.yaml before their data existed. Neither has a folder under
    # flys_New or the secured mirror, and neither contributes a row to the wide
    # CSV, so every run crashed dataset_mean_traces_tvc.py ("No usable
    # post-training traces"). Deregistered 2026-10-12; re-add both here and in
    # the config's `datasets:` + figure lists once the recordings land.
    "3Oct-Sensitivity-24-0.1",
    # 1 h starved, 3-Octanol @ 0.1%.
    "3Oct-Sensitivity-1-0.1",
    # 40 h starved, IAA @ 1%. Added 2026-09-30.
    "IAA-Sensitivity-40-1",
    # The IAA starvation series (1-30 h), IAA @ 1%. Listed in `datasets:` from
    # early October but in none of the figure lists until 2026-10-10, so they
    # were analysed and never drawn.
    "IAA-Sensitivity-1-1",
    "IAA-Sensitivity-6-1",
    "IAA-Sensitivity-12-1",
    "IAA-Sensitivity-18-1",
    "IAA-Sensitivity-30-1",
]

# Live cohorts that are NOT sensitivity cohorts, so they carry no obligation
# to appear in the sensitivity figure lists below.
OTHER_LIVE = [
    # Hex test (PiCode/config_v2_hextest_L.yaml), 32 h starved, trained on
    # hexanol 1% with light. Added 2026-10-01.
    "Hex-HexTest-Training-32-1",
    # Same hex test at 24 h and 33 h starved.
    "Hex-HexTest-Training-24-1",
    "Hex-HexTest-Training-33-1",
]

# Every per-cohort figure list a sensitivity cohort must appear in. The
# pretest_vs_test step is absent on purpose: it renders every cohort with
# paired pre/post trials and takes no list.
COHORT_LISTS = [
    ("analysis", "dataset_mean_traces", "phase_cohorts"),
    ("analysis", "pretest_score_bars", "datasets"),
    ("analysis", "cohort_figures", "datasets"),
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


@pytest.mark.parametrize("path", COHORT_LISTS, ids=lambda p: ".".join(p[1:]))
@pytest.mark.parametrize("name", SENSITIVITY)
def test_sensitivity_dataset_gets_its_figures(cfg, name, path):
    """Listed in `datasets:` alone, a cohort is analysed but never drawn."""
    node = cfg
    for key in path:
        node = (node or {}).get(key)
    assert name in (node or []), f"{name} missing from {'.'.join(path)}"


def test_pretest_score_bars_datasets_are_registered(cfg):
    """The step passes each name as `--dataset`, whose argparse `choices` is
    the script's hardcoded PRETEST_COHORTS. One unregistered name exits the
    whole step with code 2, losing the figures of every OTHER cohort too."""
    import scripts.analysis.naive_vs_trial_score_bars as nvt

    listed = cfg["analysis"]["pretest_score_bars"]["datasets"]
    missing = [d for d in listed if d not in nvt.PRETEST_COHORTS]
    assert missing == [], f"not in PRETEST_COHORTS: {missing}"


def test_every_other_dataset_is_frozen(cfg):
    stale = [d for d in cfg["datasets"]
             if d not in SENSITIVITY and d not in TEMPORARILY_THAWED
             and d not in OTHER_LIVE
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
    expected = TEMPORARILY_THAWED[name]
    if expected is None:
        return
    folders = fz.get("folders") or []
    missing = [f for f in expected if f not in folders]
    assert not missing, f"{name} lost folder retirements in the thaw: {missing}"


@pytest.mark.parametrize("name", ["3Oct-Training-24-0.1", "3Oct-Control-24-0.1"])
def test_refrozen_3oct_keeps_its_date_bounded_rig3_retirement(cfg, name):
    """Re-freezing the 3Oct pair 2026-09-01 must not drop the rig_3 rule.

    It is inert while ``data: true`` (a data-frozen dataset is never
    folder-frozen) and applies again the moment the pair is thawed -- and it
    must stay date-bounded, or it swallows every batch after the 2026-08-11
    repair.
    """
    rules = cfg["dataset_overrides"][name]["freeze"].get("folders") or []
    rig3 = [r for r in rules if isinstance(r, dict) and r.get("match") == "*_rig_3"]
    assert rig3, f"{name} lost its rig_3 retirement"
    assert rig3[0].get("before"), f"{name} rig_3 retirement is no longer date-bounded"


@pytest.mark.parametrize("name", OTHER_LIVE)
def test_other_live_cohorts_are_listed_and_unfrozen(cfg, name):
    assert name in cfg["datasets"]
    assert not _freeze(cfg, name).get("data")
    assert not _freeze(cfg, name).get("figures")
