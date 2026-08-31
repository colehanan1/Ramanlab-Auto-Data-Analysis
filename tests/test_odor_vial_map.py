"""OFM_A = isoamyl acetate, and the full per-odor vial map in session records.

Two coupled changes:

* the OFM_A valve now carries isoamyl acetate, so it is written as ``IAA`` in
  cohort folder names and ``IsoamylAcetate`` in trial filenames, and is
  addressable by the letter ``I`` as well as ``A``;
* ``--odor-vial-conc "E=1,H=0.1,..."`` previously collapsed to the one selected
  odor. The whole map is now kept so every odor's concentration reaches the
  session text file, the session CSV and each trial sidecar.
"""

import sys
from pathlib import Path

import pytest

PICODE = Path(__file__).resolve().parent.parent / "PiCode"
sys.path.insert(0, str(PICODE))

from experiment_scheduler import (  # noqa: E402
    ODOR_CONFIG_MAP, get_push_folder, resolve_odor_vial_conc,
    resolve_odor_pin, parse_odor_vial_conc_map, format_odor_vial_map,
)
from expand_config import _PIN_DISPLAY  # noqa: E402

USER_MAP = "E=1,H=0.1,O=0.1,I=1,C=1,B=0.1,L=1"


# ── OFM_A is isoamyl acetate ──────────────────────────────────────────

def test_ofm_a_folder_is_iaa():
    assert ODOR_CONFIG_MAP["OFM_A"]["folder"] == "IAA"


def test_ofm_a_display_label_is_isoamyl_acetate():
    assert _PIN_DISPLAY["OFM_A"] == "IsoamylAcetate"


def test_cohort_folder_uses_iaa():
    remote = get_push_folder("OFM_A", "Control", "r@h:/data",
                             starvation_hours=24, odor_vial_conc="1%")
    assert remote.rstrip("/").rsplit("/", 1)[-1] == "IAA-Control-24-1"


def test_letter_i_resolves_to_ofm_a():
    assert resolve_odor_pin("I") == "OFM_A"


def test_letter_a_still_resolves_to_ofm_a():
    """Old scripts and habits keep working."""
    assert resolve_odor_pin("A") == "OFM_A"


def test_full_pin_names_pass_through():
    assert resolve_odor_pin("OFM_H") == "OFM_H"
    assert resolve_odor_pin("h") == "OFM_H"


def test_unknown_letter_rejected():
    with pytest.raises(ValueError):
        resolve_odor_pin("Z")


def test_conc_mapping_accepts_the_i_key():
    """The user's real mapping string must resolve for the isoamyl valve."""
    assert resolve_odor_vial_conc(USER_MAP, "OFM_A") == "1"


# ── the full vial map ─────────────────────────────────────────────────

def test_parse_map_keys_every_named_odor():
    got = parse_odor_vial_conc_map(USER_MAP)
    assert got == {
        "OFM_E": "1", "OFM_H": "0.1", "OFM_O": "0.1", "OFM_A": "1",
        "OFM_C": "1", "OFM_B": "0.1", "OFM_L": "1",
    }


def test_parse_map_scalar_applies_to_every_odor():
    got = parse_odor_vial_conc_map("0.5%")
    assert set(got) == {p for p, i in ODOR_CONFIG_MAP.items() if i["config"]}
    assert set(got.values()) == {"0.5%"}


def test_parse_map_none_is_empty():
    assert parse_odor_vial_conc_map(None) == {}


def test_parse_map_rejects_unknown_odor():
    with pytest.raises(ValueError):
        parse_odor_vial_conc_map("Z=1")


def test_parse_map_rejects_malformed_entry():
    with pytest.raises(ValueError):
        parse_odor_vial_conc_map("H=")


def test_format_uses_display_names_in_odor_order():
    text = format_odor_vial_map(parse_odor_vial_conc_map(USER_MAP))
    assert text == ("IsoamylAcetate=1, Benzaldehyde=0.1, Citral=1, "
                    "EthylButyrate=1, Hexanol=0.1, Linalool=1, 3-Octonol=0.1")


def test_format_empty_map_is_na():
    assert format_odor_vial_map({}) == "N/A"


def test_format_omits_odors_the_mapping_never_named():
    text = format_odor_vial_map(parse_odor_vial_conc_map("H=0.1,E=1"))
    assert text == "EthylButyrate=1, Hexanol=0.1"
