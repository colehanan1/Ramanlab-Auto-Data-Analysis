"""Regression tests for the 3Oct-24-0.1 odor label corrections.

The july_20 3-octanol cohorts reuse the EB-24-1 rig plumbing, so they inherit
its two corrections: the OFM_A ("ACV") channel actually delivered isoamyl
acetate, and every odor should carry its concentration in the figure label.
3-Octanol is the trained odor here and was delivered at 0.1% (dataset naming is
3Oct-{Training|Control}-{starvation_hours}-{3Oct_conc}), so it reads
"3-Octanol (0.1%)" rather than the EB cohort's "3-Octanol (1%)".

These datasets are also the only ones in the config whose *name*
("3Oct-Training-24-0.1") differs from the canonical dataset name the analysis
frames carry ("3OCT-Training-24-0.1", uppercased by ``canon_dataset``). Every
remap call site looks up ``dataset_canon``, so a config-keyed remap only lands
if registration canonicalises its keys — ``test_remap_applies_under_canon_name``
pins exactly that, since the failure mode is a silently un-remapped figure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fbpipe.config import load_settings
from fbpipe.odor_constants import canon_dataset
from scripts.analysis import envelope_visuals as ev

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config_new.yaml"
DATASETS = ("3Oct-Training-24-0.1", "3Oct-Control-24-0.1")
EXPECTED_REMAP = {
    "Apple Cider Vinegar": "Isoamyl Acetate (1%)",
    "3-Octanol": "3-Octanol (0.1%)",
    "Ethyl Butyrate": "Ethyl Butyrate (1%)",
    "Benzaldehyde": "Benzaldehyde (0.1%)",
    "Hexanol": "Hexanol (0.1%)",
    "Citral": "Citral (1%)",
    "Linalool": "Linalool (1%)",
}


@pytest.fixture()
def restore_protocol_and_remap():
    """Snapshot and restore the module-global protocol + remap state."""
    saved_protocol = ev.get_protocol()
    saved_remap = {ds: dict(m) for ds, m in ev._DATASET_ODOR_REMAP.items()}
    try:
        yield
    finally:
        ev.set_protocol(saved_protocol)
        ev.set_dataset_odor_remap(saved_remap)


def _register_config_remap() -> None:
    """Register the shipped config's remap exactly as run_workflows does."""
    settings = load_settings(CONFIG_PATH)
    ev.set_dataset_odor_remap(
        {
            str(ds): dict(ov.odor_remap)
            for ds, ov in settings.dataset_overrides.items()
            if ov.odor_remap
        }
    )


@pytest.mark.parametrize("dataset", DATASETS)
def test_config_declares_3oct_remap(dataset: str) -> None:
    """The shipped config must carry the full concentration-labelled remap."""
    settings = load_settings(CONFIG_PATH)
    override = settings.dataset_overrides.get(dataset)
    assert override is not None, (
        f"no dataset_overrides entry for {dataset}; the odor correction is missing"
    )
    assert dict(override.odor_remap) == EXPECTED_REMAP


@pytest.mark.parametrize("dataset", DATASETS)
def test_remap_applies_under_canon_name(dataset, restore_protocol_and_remap) -> None:
    """Figures look the remap up by canon name ("3OCT-..."), not the config key."""
    _register_config_remap()
    ev.set_protocol("v2")
    canon = canon_dataset(dataset)
    assert canon != dataset, "precondition: this dataset's canon name differs"

    # The rig spells it "3-Octonol"; it canonicalises to 3-Octanol, then remaps.
    assert ev._display_odor(canon, "testing_1_3-Octonol") == "3-Octanol (0.1%)"
    assert ev._display_odor(canon, "testing_2_ACV") == "Isoamyl Acetate (1%)"
    assert ev._display_odor(canon, "testing_4_EthylButyrate") == "Ethyl Butyrate (1%)"
    assert ev._display_odor(canon, "testing_3_Linalool") == "Linalool (1%)"
    assert ev._display_odor(canon, "testing_5_Hexanol") == "Hexanol (0.1%)"
    assert ev._display_odor(canon, "testing_6_Benzaldehyde") == "Benzaldehyde (0.1%)"
    assert ev._display_odor(canon, "testing_7_Citral") == "Citral (1%)"


def test_light_only_trials_are_untouched(restore_protocol_and_remap) -> None:
    """LightOnly is not an odor and must not pick up a concentration label."""
    _register_config_remap()
    ev.set_protocol("v2")
    canon = canon_dataset("3Oct-Training-24-0.1")
    assert ev._display_odor(canon, "testing_9_LightOnly") == "Optogenetic Light Control"


def test_remap_is_scoped_to_the_3oct_cohorts(restore_protocol_and_remap) -> None:
    """The 0.1% 3-Octanol label must not leak into the EB cohorts (1% there)."""
    _register_config_remap()
    ev.set_protocol("v2")
    assert ev._display_odor("EB-Training-24-1", "testing_1_3-Octonol") == "3-Octanol (1%)"
    # The Hex-24-0.01 cohorts run the same rig plumbing and so carry the *same*
    # 0.1% 3-octanol label (see test_hex_24_001_odor_remap.py) — they only differ
    # on the trained odor's own concentration.
    assert ev._display_odor("Hex-Training-24-0.01", "testing_1_3-Octonol") == (
        "3-Octanol (0.1%)"
    )
    # A dataset with no remap at all still renders the bare canonical name.
    assert ev._display_odor("Hex-Control-24-0.005", "testing_1_3-Octonol") == "3-Octanol"
