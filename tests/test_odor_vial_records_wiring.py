"""Rig wiring for the per-odor vial record.

The scripts can't be imported off the Pi, so the wiring is AST-checked; the
CSV writer itself is lifted out and run for real against a temp dir.
"""

import ast
import csv
import sys
from pathlib import Path

import pytest

PICODE = Path(__file__).resolve().parent.parent / "PiCode"
sys.path.insert(0, str(PICODE))
from experiment_scheduler import ODOR_CONFIG_MAP, parse_odor_vial_conc_map  # noqa: E402
from expand_config import _PIN_DISPLAY  # noqa: E402

SCRIPTS = ["combinedv2_1.py", "combinedv2_1_pi1.py", "combinedv2_1_pi3.py"]
USER_MAP = "E=1,H=0.1,O=0.1,I=1,C=1,B=0.1,L=1"


def _src(script):
    return (PICODE / script).read_text(encoding="utf-8")


@pytest.mark.parametrize("script", SCRIPTS)
def test_full_map_parsed_before_it_is_collapsed(script):
    """resolve_odor_vial_conc() throws the other odors away — parse first."""
    src = _src(script)
    assert src.index("parse_odor_vial_conc_map(args.odor_vial_conc)") < \
           src.index("resolve_odor_vial_conc(args.odor_vial_conc")


@pytest.mark.parametrize("script", SCRIPTS)
def test_map_reaches_session_metadata(script):
    src = _src(script)
    assert 'md["odor_vial_concentration_map"]' in src
    assert 'md["odor_vial_concentrations"]' in src


@pytest.mark.parametrize("script", SCRIPTS)
def test_map_written_to_session_text_file(script):
    assert "Odor vial concentrations (all vials)" in _src(script)


@pytest.mark.parametrize("script", SCRIPTS)
def test_map_written_to_each_trial_sidecar(script):
    src = _src(script)
    assert "Trial odor vial concentration:" in src
    assert "All vial concentrations:" in src


@pytest.mark.parametrize("script", SCRIPTS)
def test_odor_vial_csv_written(script):
    src = _src(script)
    assert "def write_odor_vial_csv" in src
    # and actually called, not just defined
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "write_odor_vial_csv"]
    assert calls, f"{script} defines write_odor_vial_csv but never calls it"


@pytest.mark.parametrize("script", SCRIPTS)
def test_odors_flag_accepts_aliases(script):
    """--odors I must resolve; the old hardcoded OFM_<letter> could not."""
    assert "resolve_odor_pin(letter)" in _src(script)


def test_csv_writer_output(tmp_path):
    """Run the real writer: one row per vial, delivered odor flagged."""
    src = _src("combinedv2_1.py")
    tree = ast.parse(src)
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "write_odor_vial_csv")
    ns = {"csv": csv, "Path": Path, "ODOR_CONFIG_MAP": ODOR_CONFIG_MAP,
          "ODOR_DISPLAY": _PIN_DISPLAY, "print": lambda *a, **k: None}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "x", "exec"), ns)

    md = {"odor_vial_concentration_map": parse_odor_vial_conc_map(USER_MAP),
          "odor_pin": "OFM_H"}
    ns["write_odor_vial_csv"](tmp_path, md)

    rows = list(csv.DictReader((tmp_path / "odor_vials.csv").open()))
    assert len(rows) == 7
    by_odor = {r["odor"]: r for r in rows}
    assert by_odor["IsoamylAcetate"]["vial_concentration"] == "1"
    assert by_odor["IsoamylAcetate"]["folder"] == "IAA"
    assert by_odor["Hexanol"]["delivered_this_session"] == "yes"
    assert by_odor["Citral"]["delivered_this_session"] == "no"


def test_csv_writer_skips_when_no_map(tmp_path):
    src = _src("combinedv2_1.py")
    tree = ast.parse(src)
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "write_odor_vial_csv")
    ns = {"csv": csv, "Path": Path, "ODOR_CONFIG_MAP": ODOR_CONFIG_MAP,
          "ODOR_DISPLAY": _PIN_DISPLAY, "print": lambda *a, **k: None}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "x", "exec"), ns)
    ns["write_odor_vial_csv"](tmp_path, {"odor_vial_concentration_map": {}})
    assert not (tmp_path / "odor_vials.csv").exists()
