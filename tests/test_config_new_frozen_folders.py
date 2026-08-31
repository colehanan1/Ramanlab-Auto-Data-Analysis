"""What config_new.yaml actually freezes, pinned against the real data roots.

These assert the EFFECT of the rules, not their text. That matters: the rules
were first written as a second ``freeze:`` key inside datasets that already had
one, and YAML silently keeps only the last — so the rig_3 rule parsed fine,
loaded fine, and froze nothing. Only checking resolved folder names caught it.

Skipped when the data roots are absent (CI / another machine).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from fbpipe.config import load_settings
from fbpipe.utils.frozen_folders import frozen_folders_for_root
from fbpipe.utils.rig_gates import read_batch_born_date, read_batch_date

DATA_ROOT = Path("/home/ramanlab/Documents/cole/Data/flys_New")

# Retired 2026-08-12 at the user's request: rig_3 batches in these six --
# but only those recorded BEFORE the rig was repaired. rig_3 is good again
# from 2026-08-11 on (user, 2026-08-14), so that is the first KEPT day.
RIG3_CUTOFF = dt.date(2026, 8, 11)
RIG3_DATASETS = [
    "EB-Control-24-1",
    "EB-Training-24-1",
    "Hex-Control-24-0.01",
    "Hex-Training-24-0.01",
    "3Oct-Training-24-0.1",
    "3Oct-Control-24-0.1",
]
# ...plus every batch-2 fly before july 27 in the EB-24-1 pair.
EARLY_BATCH2_DATASETS = ["EB-Control-24-1", "EB-Training-24-1"]
BATCH2_CUTOFF = dt.date(2026, 7, 27)

pytestmark = pytest.mark.skipif(
    not DATA_ROOT.is_dir(), reason="live data roots not present on this machine"
)


@pytest.fixture(scope="module")
def cfg():
    """config_new.yaml with every dataset's DATA freeze lifted.

    As of 2026-08-27 every dataset except the four ``*-Sensitivity-*`` cohorts
    carries ``freeze: {data: true}``, and a data-frozen dataset is never
    folder-frozen — its root is not walked at all, so ``frozen_folders_for_root``
    correctly returns nothing. That would make every assertion below vacuously
    pass. These tests are about whether the retirement RULES are right, which
    still matters the moment a dataset is thawed, so the data freeze is lifted
    here and only here.
    """
    settings = load_settings("config/config_new.yaml")
    for override in (settings.dataset_overrides or {}).values():
        override.freeze_data = False
    return settings


def _live(cfg, dataset: str) -> list[Path]:
    root = DATA_ROOT / dataset
    frozen = frozen_folders_for_root(cfg, root)
    return [p for p in sorted(root.iterdir()) if p.is_dir() and p.name not in frozen]


@pytest.mark.parametrize("dataset", RIG3_DATASETS)
def test_no_pre_repair_rig_3_folder_survives(cfg, dataset):
    leaked = [
        p.name
        for p in _live(cfg, dataset)
        if p.name.endswith("_rig_3")
        and (read_batch_date(p) or dt.date(2099, 1, 1)) < RIG3_CUTOFF
    ]
    assert leaked == [], f"{dataset} still exposes pre-repair rig_3 folders: {leaked}"


@pytest.mark.parametrize("dataset", RIG3_DATASETS)
def test_post_repair_rig_3_folders_are_live(cfg, dataset):
    """The bound is a bound: rig_3 from 2026-08-11 on must reach the pipeline.

    An unbounded ``*_rig_3`` glob froze august_13_batch_2_rig_3 the moment it
    was recorded, so YOLO never touched it. Datasets with no post-repair rig_3
    batch yet simply have nothing to check.

    Folders retired by the COHORT rule (``freeze_folders_born_on_or_after``) are
    not evidence of a rig_3 leak — they are frozen on purpose, by an independent
    rule, and several post-repair rig_3 batches are also post-cutoff births. This
    test is about the rig_3 bound only, so it judges the rig_3 rule alone.
    """
    frozen = frozen_folders_for_root(cfg, DATA_ROOT / dataset)
    born_cut = getattr(cfg, "freeze_folders_born_on_or_after", None)

    def cohort_frozen(path: Path) -> bool:
        if born_cut is None:
            return False
        born = read_batch_born_date(path)
        return born is not None and born >= born_cut

    swallowed = [
        p.name
        for p in sorted((DATA_ROOT / dataset).iterdir())
        if p.is_dir()
        and p.name.endswith("_rig_3")
        and (read_batch_date(p) or dt.date(1970, 1, 1)) >= RIG3_CUTOFF
        and p.name in frozen
        and not cohort_frozen(p)
    ]
    assert swallowed == [], f"{dataset} froze repaired rig_3 folders: {swallowed}"


@pytest.mark.parametrize("dataset", RIG3_DATASETS)
def test_the_dataset_still_has_live_folders(cfg, dataset):
    """Guards the opposite failure: a too-greedy glob emptying a dataset."""
    assert _live(cfg, dataset), f"{dataset} has no live folders left"


@pytest.mark.parametrize("dataset", EARLY_BATCH2_DATASETS)
def test_no_batch_2_before_july_27_survives(cfg, dataset):
    leaked = [
        p.name
        for p in _live(cfg, dataset)
        if "batch_2" in p.name
        and (read_batch_date(p) or dt.date(2099, 1, 1)) < BATCH2_CUTOFF
    ]
    assert leaked == [], f"{dataset} still exposes early batch-2 folders: {leaked}"


@pytest.mark.parametrize("dataset", EARLY_BATCH2_DATASETS)
def test_a_late_batch_2_folder_is_frozen_only_if_it_is_rig_3(cfg, dataset):
    """The bound is a bound, not a blanket ban on batch 2.

    Asserting "some late batch 2 survives" would be wrong for
    EB-Training-24-1, whose only late batch 2 (july_28_batch_2_rig_3) is
    frozen by the SEPARATE rig_3 rule. The real invariant is that the date
    bound never reaches past the cutoff on its own.
    """
    root = DATA_ROOT / dataset
    frozen = frozen_folders_for_root(cfg, root)
    wrongly_frozen = [
        p.name
        for p in sorted(root.iterdir())
        if p.is_dir()
        and p.name in frozen
        and "batch_2" in p.name
        and not p.name.endswith("_rig_3")
        and (read_batch_date(p) or dt.date(2000, 1, 1)) >= BATCH2_CUTOFF
    ]
    assert wrongly_frozen == [], (
        f"{dataset} froze batch-2 folders at/after {BATCH2_CUTOFF} "
        f"that are not rig_3: {wrongly_frozen}"
    )


def test_the_eb_pair_still_has_a_late_batch_2_on_a_live_rig(cfg):
    """Sanity that the cutoff did not swallow the whole batch-2 series: at
    least one late, non-rig_3 batch 2 is still in the figures."""
    kept = [
        p.name
        for ds in EARLY_BATCH2_DATASETS
        for p in _live(cfg, ds)
        if "batch_2" in p.name
        and (read_batch_date(p) or dt.date(2000, 1, 1)) >= BATCH2_CUTOFF
    ]
    assert kept, "every batch-2 folder across the EB-24-1 pair was frozen"


def test_batch_1_is_untouched_in_the_eb_pair(cfg):
    """Only batch 2 was date-bounded; batch 1 must survive on rig_1/rig_2."""
    for dataset in EARLY_BATCH2_DATASETS:
        kept = [p.name for p in _live(cfg, dataset) if "batch_1" in p.name]
        assert kept, f"{dataset} lost all batch-1 folders"


def test_datasets_outside_the_six_keep_their_rig_3_folders(cfg):
    """The rig_3 rule is per-dataset, not global. 3Oct-Training-24-0.1-Manual
    was not in the request and must be unaffected."""
    root = DATA_ROOT / "3Oct-Training-24-0.1-Manual"
    if not root.is_dir():
        pytest.skip("dataset not present")
    frozen = frozen_folders_for_root(cfg, root)
    on_disk = {p.name for p in root.iterdir() if p.is_dir()}
    assert frozen == set(), f"unexpectedly frozen: {sorted(frozen)} of {sorted(on_disk)}"
