"""Regression tests for the Hex-24-0.01 odor label corrections.

The july_31/august_04/august_05 hexanol cohorts run the same rig plumbing as the
3Oct-24-0.1 cohorts, so they inherit the same two corrections: the OFM_A ("ACV")
channel actually delivered isoamyl acetate, and every odor carries its
concentration in the figure label. Hexanol is the trained odor here and was
delivered at 0.01% (dataset naming is
Hex-{Training|Control}-{starvation_hours}-{hexanol_conc}), so it reads
"Hexanol (0.01%)" rather than the 3Oct cohort's "Hexanol (0.1%)".

Two invariants are worth pinning beyond "the config says the right words":

``test_both_cohorts_share_one_remap`` — the training and control cohorts must
map every odor identically. Figures pair the two by *display* label, so a remap
present on one side and absent (or spelled differently) on the other splits one
odor into two columns and the train-vs-control bars quietly stop being paired.
That failure renders a perfectly plausible figure, which is why it needs a test.

``test_trained_odor_survives_the_remap`` — ``_trained_label`` reads
PRIMARY_ODOR_LABEL, which the remap never touches, so it returns a bare
"Hexanol" while the columns now read "Hexanol (0.01%) 1". Trained detection is a
prefix match and therefore still fires; an exact match would silently unbold the
trained ticks and recolour the trained bars.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fbpipe.config import load_settings
from fbpipe.odor_constants import canon_dataset
from scripts.analysis import envelope_visuals as ev
from scripts.analysis import odor_bar_palette as pal

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config_new.yaml"
DATASETS = ("Hex-Training-24-0.01", "Hex-Control-24-0.01")
EXPECTED_REMAP = {
    "Apple Cider Vinegar": "Isoamyl Acetate (1%)",
    "3-Octanol": "3-Octanol (0.1%)",
    "Ethyl Butyrate": "Ethyl Butyrate (1%)",
    "Benzaldehyde": "Benzaldehyde (0.1%)",
    "Hexanol": "Hexanol (0.01%)",
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
def test_config_declares_hex_remap(dataset: str) -> None:
    """The shipped config must carry the full concentration-labelled remap."""
    settings = load_settings(CONFIG_PATH)
    override = settings.dataset_overrides.get(dataset)
    assert override is not None, (
        f"no dataset_overrides entry for {dataset}; the odor correction is missing"
    )
    assert dict(override.odor_remap) == EXPECTED_REMAP


def test_both_cohorts_share_one_remap() -> None:
    """Train and control must agree, or the paired bars silently unpair."""
    settings = load_settings(CONFIG_PATH)
    train = dict(settings.dataset_overrides["Hex-Training-24-0.01"].odor_remap)
    ctrl = dict(settings.dataset_overrides["Hex-Control-24-0.01"].odor_remap)
    # Non-empty, or two missing remaps would satisfy `train == ctrl` vacuously.
    assert train, "training cohort has no odor_remap at all"
    assert train == ctrl


@pytest.mark.parametrize("dataset", DATASETS)
def test_remap_applies_to_every_trial_label(dataset, restore_protocol_and_remap) -> None:
    """Each rig trial label must resolve to its concentration-tagged display name."""
    _register_config_remap()
    ev.set_protocol("v2")
    canon = canon_dataset(dataset)

    # The rig spells it "3-octonol"; it canonicalises to 3-Octanol, then remaps.
    assert ev._display_odor(canon, "testing_2_3-octonol") == "3-Octanol (0.1%)"
    assert ev._display_odor(canon, "testing_2_acv") == "Isoamyl Acetate (1%)"
    assert ev._display_odor(canon, "testing_2_ethylbutyrate") == "Ethyl Butyrate (1%)"
    assert ev._display_odor(canon, "testing_3_linalool") == "Linalool (1%)"
    assert ev._display_odor(canon, "testing_2_benzaldehyde") == "Benzaldehyde (0.1%)"
    assert ev._display_odor(canon, "testing_2_citral") == "Citral (1%)"
    # Both hexanol presentations (testing_1 and testing_8) carry the label.
    assert ev._display_odor(canon, "testing_1_hexanol") == "Hexanol (0.01%)"
    assert ev._display_odor(canon, "testing_8_hexanol") == "Hexanol (0.01%)"


def test_light_only_trials_are_untouched(restore_protocol_and_remap) -> None:
    """LightOnly is not an odor and must not pick up a concentration label."""
    _register_config_remap()
    ev.set_protocol("v2")
    canon = canon_dataset("Hex-Training-24-0.01")
    assert ev._display_odor(canon, "testing_9_lightonly") == "Optogenetic Light Control"


def test_remap_is_scoped_to_the_24_001_cohorts(restore_protocol_and_remap) -> None:
    """The 0.01% hexanol label must not leak into the other hexanol cohorts.

    Hex-Training-24-0.1 delivers hexanol at 0.1%, a decimal place away, so a
    leak is silent and wrong rather than obviously wrong. (Its Citral channel
    delivered sour dough yeast in the may/june block; that block is frozen out
    and the dataset now carries the august panel -- see
    test_pubfig_pipeline_wiring.)
    """
    _register_config_remap()
    ev.set_protocol("v2")
    assert ev._display_odor("Hex-Training-24-0.1", "testing_2_citral") == "Citral (1%)"
    assert ev._display_odor("Hex-Training-24-0.1", "testing_1_hexanol") == (
        "Hexanol (0.1%)"
    )
    # A dataset with no remap at all still renders the bare canonical name.
    assert ev._display_odor("Hex-Control-24-0.005", "testing_2_citral") == "Citral"


def test_trained_odor_survives_the_remap(restore_protocol_and_remap) -> None:
    """Trained detection is a prefix match, so the tagged label still counts."""
    from scripts.analysis.envelope_visuals import _trained_label

    _register_config_remap()
    ev.set_protocol("v2")
    canon = canon_dataset("Hex-Training-24-0.01")
    trained = _trained_label(canon)
    assert trained == "Hexanol", "precondition: _trained_label ignores the remap"

    # This is exactly how pubfig_score_train_vs_control builds `is_trained`.
    for column in ("Hexanol (0.01%) 1", "Hexanol (0.01%) 2"):
        assert column.casefold().startswith(trained.casefold())
    assert not "Isoamyl Acetate (1%)".casefold().startswith(trained.casefold())


def test_remapped_labels_still_resolve_to_palette_colours() -> None:
    """Bars keep their odor colours once the concentration is appended.

    ACV becoming isoamyl acetate is a real colour change (orange -> purple);
    everything else must keep the colour it had before the concentration tag.
    """
    assert pal.odor_color("Isoamyl Acetate (1%)") == pal.ISOAMYL_PURPLE
    assert pal.odor_color("Hexanol (0.01%)") == pal.HEX_COLOR
    assert pal.odor_color("Hexanol (0.01%) 2") == pal.HEX_COLOR
    assert pal.odor_color("3-Octanol (0.1%)") == pal.DARK_GREEN
    assert pal.odor_color("Benzaldehyde (0.1%)") is pal.odor_color("Benzaldehyde")
    assert pal.odor_color("Citral (1%)") == pal.CITRAL_YELLOW
    assert pal.odor_color("Ethyl Butyrate (1%)") == pal.PINK
    assert pal.odor_color("Linalool (1%)") == pal.DARKER_GREEN
