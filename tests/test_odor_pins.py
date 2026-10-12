"""``~/odor_pins.txt``: which odor, at what concentration, sits in each valve.

The file replaces typing ``--odor-vial-conc`` on every launch and lets one
compound sit in two valves at two concentrations (hex 0.1 in H, hex 1 in L).
Applying it rewrites the rig's pin tables in place, so these tests snapshot
and restore them around every case.
"""

import copy
import sys
from pathlib import Path

import pytest

PICODE = Path(__file__).resolve().parent.parent / "PiCode"
sys.path.insert(0, str(PICODE))
import expand_config as ec  # noqa: E402
import experiment_scheduler as sched  # noqa: E402
import odor_pins  # noqa: E402
from odor_pins import (  # noqa: E402
    PinOdor, apply_pin_map, format_pin_table, load_and_apply_pin_file,
    parse_pin_text, vial_conc_arg,
)

# The vials on all three rigs as of 2026-10-01.
CURRENT = """\
# pin  odor  conc
H  hex   0.1
L  hex   1
B  benz  1
E  eb    1
C  eb    0.1
A  iaa   1
O  iaa   0.1
"""


@pytest.fixture(autouse=True)
def _restore_pin_tables():
    """apply_pin_map mutates module globals; put them back after each test."""
    display = dict(ec._PIN_DISPLAY)
    pins = list(ec._ALL_ODOR_PINS)
    cmap = copy.deepcopy(sched.ODOR_CONFIG_MAP)
    applied = ec._PINS_APPLIED
    yield
    ec._PIN_DISPLAY.clear()
    ec._PIN_DISPLAY.update(display)
    ec._ALL_ODOR_PINS[:] = pins
    sched.ODOR_CONFIG_MAP.clear()
    sched.ODOR_CONFIG_MAP.update(cmap)
    ec._PINS_APPLIED = applied


# ── parsing ───────────────────────────────────────────────────────────

def test_parses_the_current_vials():
    m = parse_pin_text(CURRENT)
    assert m["OFM_H"] == PinOdor("hex", "Hexanol", "Hex", "0.1")
    assert m["OFM_L"] == PinOdor("hex", "Hexanol", "Hex", "1")
    assert m["OFM_C"] == PinOdor("eb", "EthylButyrate", "EB", "0.1")
    assert m["OFM_O"] == PinOdor("iaa", "IsoamylAcetate", "IAA", "0.1")
    assert len(m) == 7


def test_label_carries_the_concentration():
    """Two vials of one compound must never share a trial label."""
    m = parse_pin_text(CURRENT)
    assert m["OFM_H"].label_with_conc == "Hexanol-0.1"
    assert m["OFM_L"].label_with_conc == "Hexanol-1"


@pytest.mark.parametrize("line", [
    "H hex 0.1",
    "h HEX 0.1",
    "OFM_H hex 0.1",
    "H = hex 0.1",
    "H: hex, 0.1",
    "H hexanol 0.1",
    "H hex 0.1%",
    "H hex 0.1   # the weak vial",
])
def test_forgiving_line_formats(line):
    assert parse_pin_text(line)["OFM_H"] == PinOdor("hex", "Hexanol", "Hex", "0.1")


def test_isoamyl_acetate_valve_letter_alias():
    """``I`` is the existing alias for the OFM_A valve (resolve_odor_pin)."""
    assert parse_pin_text("I iaa 1")["OFM_A"].folder == "IAA"


@pytest.mark.parametrize("name,label,folder", [
    ("benz", "Benzaldehyde", "Benz"),
    ("eb", "EthylButyrate", "EB"),
    ("iaa", "IsoamylAcetate", "IAA"),
    ("cit", "Citral", "Cit"),
    ("lin", "Linalool", "Lin"),
    ("3oct", "3-Octonol", "3Oct"),
    ("3-octanol", "3-Octonol", "3Oct"),
])
def test_known_odor_names(name, label, folder):
    """Labels and folders match the rig defaults the pipeline already parses."""
    p = parse_pin_text(f"B {name} 1")["OFM_B"]
    assert (p.label, p.folder) == (label, folder)


def test_labels_match_the_built_in_display_names():
    """A pins file naming the factory odors must reproduce today's labels."""
    for pin, (label, folder) in {
        "OFM_A": ("IsoamylAcetate", "IAA"), "OFM_B": ("Benzaldehyde", "Benz"),
        "OFM_C": ("Citral", "Cit"), "OFM_E": ("EthylButyrate", "EB"),
        "OFM_H": ("Hexanol", "Hex"), "OFM_L": ("Linalool", "Lin"),
        "OFM_O": ("3-Octonol", "3Oct"),
    }.items():
        assert ec._PIN_DISPLAY[pin] == label
        assert sched.ODOR_CONFIG_MAP[pin]["folder"] == folder


@pytest.mark.parametrize("text,needle", [
    ("H hexx 0.1", "Unknown odor 'hexx'"),
    ("Z hex 0.1", "Unknown valve 'Z'"),
    ("H hex", "line 1"),
    ("H hex 0.1 extra", "line 1"),
    ("H hex 1:10", "concentration"),
    ("H hex abc", "concentration"),
    ("H hex 0.1\nH benz 1", "listed twice"),
    ("H hex 0.1\nL hex 0.1", "same odor and concentration"),
    ("# nothing here\n\n", "no valves"),
])
def test_rejects_mistakes_before_anything_records(text, needle):
    with pytest.raises(ValueError, match=needle):
        parse_pin_text(text)


def test_error_names_the_line_number():
    with pytest.raises(ValueError, match="line 3"):
        parse_pin_text("H hex 0.1\n# comment\nL lemon 1\n")


# ── applying ──────────────────────────────────────────────────────────

def test_apply_rewrites_labels_folders_and_panel_pins():
    apply_pin_map(parse_pin_text(CURRENT))
    assert ec._PIN_DISPLAY["OFM_L"] == "Hexanol-1"
    assert ec._PIN_DISPLAY["OFM_C"] == "EthylButyrate-0.1"
    assert sched.ODOR_CONFIG_MAP["OFM_L"]["folder"] == "Hex"
    assert sched.ODOR_CONFIG_MAP["OFM_C"]["folder"] == "EB"
    assert ec._ALL_ODOR_PINS == ["OFM_A", "OFM_B", "OFM_C", "OFM_E",
                                 "OFM_H", "OFM_L", "OFM_O"]


def test_apply_keeps_config_files_and_pseudo_pins():
    apply_pin_map(parse_pin_text(CURRENT))
    assert sched.ODOR_CONFIG_MAP["OFM_L"]["config"] == "config_v2_L.yaml"
    assert sched.ODOR_CONFIG_MAP["OFM_PANEL"]["folder"] == "RandomPanel"


def test_unlisted_valves_drop_out_of_the_panels():
    """An unlisted valve may be empty — never deliver it in a panel."""
    apply_pin_map(parse_pin_text("H hex 0.1\nB benz 1\n"))
    assert ec._ALL_ODOR_PINS == ["OFM_B", "OFM_H"]


def test_panel_pin_order_stays_canonical():
    """File order must not leak into anything order-sensitive."""
    apply_pin_map(parse_pin_text("O iaa 0.1\nA iaa 1\nH hex 0.1\n"))
    assert ec._ALL_ODOR_PINS == ["OFM_A", "OFM_H", "OFM_O"]


def test_same_object_is_mutated_so_imported_aliases_follow():
    """The rigs do ``from expand_config import _PIN_DISPLAY as ODOR_DISPLAY``."""
    from expand_config import _PIN_DISPLAY as ODOR_DISPLAY
    from experiment_scheduler import ODOR_CONFIG_MAP
    apply_pin_map(parse_pin_text(CURRENT))
    assert ODOR_DISPLAY["OFM_L"] == "Hexanol-1"
    assert ODOR_CONFIG_MAP["OFM_L"]["folder"] == "Hex"


def test_vial_map_text_uses_the_new_labels():
    apply_pin_map(parse_pin_text(CURRENT))
    conc, _ = vial_conc_arg(parse_pin_text(CURRENT), None, "OFM_L")
    text = sched.format_odor_vial_map(sched.parse_odor_vial_conc_map(conc))
    assert "Hexanol-0.1=0.1" in text and "Hexanol-1=1" in text


# ── loading from disk ─────────────────────────────────────────────────

def test_missing_default_file_keeps_built_in_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(odor_pins, "DEFAULT_PIN_FILE", tmp_path / "odor_pins.txt")
    pin_map, path = load_and_apply_pin_file(None)
    assert pin_map == {} and path is None
    assert ec._PIN_DISPLAY["OFM_L"] == "Linalool"


def test_default_file_is_read_when_present(tmp_path, monkeypatch):
    f = tmp_path / "odor_pins.txt"
    f.write_text(CURRENT)
    monkeypatch.setattr(odor_pins, "DEFAULT_PIN_FILE", f)
    pin_map, path = load_and_apply_pin_file(None)
    assert path == f and len(pin_map) == 7
    assert ec._PIN_DISPLAY["OFM_L"] == "Hexanol-1"


def test_explicit_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_and_apply_pin_file(tmp_path / "nope.txt")


def test_bad_file_applies_nothing(tmp_path):
    f = tmp_path / "odor_pins.txt"
    f.write_text("H hex 0.1\nL lemon 1\n")
    with pytest.raises(ValueError):
        load_and_apply_pin_file(f)
    assert ec._PIN_DISPLAY["OFM_H"] == "Hexanol"


def test_default_location_is_outside_the_synced_code_dir():
    """A PiCode push must never overwrite the rig's own vial list."""
    assert odor_pins.DEFAULT_PIN_FILE == Path.home() / "odor_pins.txt"


def test_example_file_parses():
    """The shipped template must itself be a valid pins file."""
    m = parse_pin_text((PICODE / "odor_pins.txt.example").read_text())
    assert m == parse_pin_text(CURRENT)


def test_table_shows_every_valve(tmp_path):
    out = format_pin_table(parse_pin_text(CURRENT), tmp_path / "odor_pins.txt")
    for token in ("OFM_H", "Hexanol", "0.1", "OFM_L", "OFM_C", "EthylButyrate"):
        assert token in out


def test_table_says_when_no_file_is_used():
    assert "built-in" in format_pin_table({}, None)


# ── concentration hand-off to the existing --odor-vial-conc path ──────

def test_conc_arg_is_the_existing_mapping_form():
    conc, warn = vial_conc_arg(parse_pin_text(CURRENT), None, "OFM_L")
    assert sched.parse_odor_vial_conc_map(conc) == {
        "OFM_A": "1", "OFM_B": "1", "OFM_C": "0.1", "OFM_E": "1",
        "OFM_H": "0.1", "OFM_L": "1", "OFM_O": "0.1",
    }
    assert sched.resolve_odor_vial_conc(conc, "OFM_L") == "1"
    assert warn is None


def test_matching_cli_value_is_silent():
    _, warn = vial_conc_arg(parse_pin_text(CURRENT), "1%", "OFM_L")
    assert warn is None


@pytest.mark.parametrize("cli", ["0.1", "H=0.1,L=0.1", "H=0.1"])
def test_disagreeing_cli_value_warns_and_the_file_wins(cli):
    conc, warn = vial_conc_arg(parse_pin_text(CURRENT), cli, "OFM_L")
    assert sched.resolve_odor_vial_conc(conc, "OFM_L") == "1"
    assert warn and "odor_pins" in warn and cli in warn


def test_trained_valve_missing_from_file_is_an_error():
    with pytest.raises(ValueError, match="OFM_L"):
        vial_conc_arg(parse_pin_text("H hex 0.1\n"), None, "OFM_L")


@pytest.mark.parametrize("pseudo", ["OFM_PANEL", "OFM_LIGHT"])
def test_pseudo_pins_need_no_entry(pseudo):
    conc, warn = vial_conc_arg(parse_pin_text(CURRENT), None, pseudo)
    assert sched.resolve_odor_vial_conc(conc, pseudo) is None
    assert warn is None


# ── review fixes 2026-10-01 ───────────────────────────────────────────

@pytest.mark.parametrize("typed,canon", [
    ("1.0", "1"), ("0.10", "0.1"), ("01", "1"), ("10", "10"), ("100", "100"),
    ("0.1", "0.1"), ("0.005", "0.005"),
])
def test_concentration_has_one_spelling(typed, canon):
    """'1.0' on one rig and '1' on another must not split the cohort folder."""
    assert parse_pin_text(f"L hex {typed}")["OFM_L"].conc == canon


def test_equal_values_count_as_duplicates():
    with pytest.raises(ValueError, match="same odor and concentration"):
        parse_pin_text("H hex 0.1\nL hex 0.10\n")


def test_windows_bom_is_ignored(tmp_path):
    f = tmp_path / "odor_pins.txt"
    f.write_bytes(("\ufeff# header\n" + CURRENT).encode("utf-8"))
    pin_map, _ = load_and_apply_pin_file(f)
    assert len(pin_map) == 7


def test_apply_marks_the_pins_as_applied():
    ec._PINS_APPLIED = False
    apply_pin_map(parse_pin_text(CURRENT))
    assert ec._PINS_APPLIED is True


@pytest.mark.parametrize("pseudo", ["OFM_PANEL", "OFM_LIGHT"])
def test_pseudo_pins_keep_the_cli_value(pseudo):
    """RandomPanel / light sweeps name their folder from --odor-vial-conc."""
    conc, warn = vial_conc_arg(parse_pin_text(CURRENT), "10", pseudo)
    assert conc == "10" and warn is None
    assert sched.resolve_odor_vial_conc(conc, pseudo) == "10"


# ── odor + conc specs (hex test pre-test, 2026-10-03) ────────────────

@pytest.mark.parametrize("spec,label", [
    ("eb 10", "EthylButyrate-10"),
    ("EB 10", "EthylButyrate-10"),
    ("eb 10%", "EthylButyrate-10"),
    ("eb 10.0", "EthylButyrate-10"),
    ("ethylbutyrate 10", "EthylButyrate-10"),
    ("hex 0.1", "Hexanol-0.1"),
    ("3-octanol 1", "3-Octonol-1"),
])
def test_spec_label(spec, label):
    """Same parsing as a pins-file line, so the two can never disagree."""
    assert odor_pins.spec_label(spec) == label


@pytest.mark.parametrize("bad", ["eb", "lemon 10", "eb ten", "eb 1:10", "eb 10 x"])
def test_spec_label_rejects(bad):
    with pytest.raises(ValueError):
        odor_pins.spec_label(bad)
