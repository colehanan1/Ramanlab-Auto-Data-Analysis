"""Per-odor ``--odor-vial-conc`` mapping (e.g. ``E=1,O=0.1,H=0.1``).

When the scheduler picks the odor at runtime, a single scalar concentration
cannot be right for every odor in the ``--odors`` pool. The mapping form keys
concentration by odor letter; ``resolve_odor_vial_conc`` returns the entry for
the selected odor so the push folder and session metadata carry the vial that
was actually loaded. Scalar values pass through unchanged.

Wiring into the rig scripts is asserted against their AST — the scripts open
the camera / claim GPIO at import time, so they cannot be imported here (same
approach as ``test_manual_session_flag.py``).
"""

import ast
import sys
from pathlib import Path

import pytest

PICODE = Path(__file__).resolve().parent.parent / "PiCode"
sys.path.insert(0, str(PICODE))

from experiment_scheduler import (  # noqa: E402
    get_push_folder,
    resolve_odor_vial_conc,
)

BASE = "ramanlab@host:/data/flys"


def _folder(remote: str) -> str:
    return remote.rstrip("/").rsplit("/", 1)[-1]


# ── scalar passthrough ───────────────────────────────────────────────────────

def test_none_passes_through():
    assert resolve_odor_vial_conc(None, "OFM_E") is None


def test_scalar_passes_through_unchanged():
    assert resolve_odor_vial_conc("0.1", "OFM_H") == "0.1"
    assert resolve_odor_vial_conc("10%", "OFM_E") == "10%"
    assert resolve_odor_vial_conc("1:10", "OFM_O") == "1:10"


# ── mapping form ─────────────────────────────────────────────────────────────

def test_mapping_resolves_per_selected_odor():
    arg = "E=1,O=0.1,H=0.1"
    assert resolve_odor_vial_conc(arg, "OFM_E") == "1"
    assert resolve_odor_vial_conc(arg, "OFM_O") == "0.1"
    assert resolve_odor_vial_conc(arg, "OFM_H") == "0.1"


def test_mapping_tolerates_case_and_whitespace():
    arg = " e = 1 , o = 0.1 , h = 0.1 "
    assert resolve_odor_vial_conc(arg, "OFM_E") == "1"
    assert resolve_odor_vial_conc(arg, "ofm_h") == "0.1"


def test_mapping_accepts_full_pin_names():
    assert resolve_odor_vial_conc("OFM_E=1,OFM_H=0.1", "OFM_E") == "1"


def test_mapping_missing_selected_odor_raises():
    with pytest.raises(ValueError, match="OFM_A"):
        resolve_odor_vial_conc("E=1,O=0.1,H=0.1", "OFM_A")


def test_mapping_unknown_odor_letter_raises():
    with pytest.raises(ValueError, match="[Uu]nknown"):
        resolve_odor_vial_conc("E=1,X=5", "OFM_E")


def test_mapping_skips_non_odor_pseudo_pins():
    """RandomPanel / LightSweep deliver all (or no) odors, so a per-odor
    mapping cannot apply — the conc segment is dropped rather than aborting."""
    arg = "E=1,O=0.1,H=0.1"
    assert resolve_odor_vial_conc(arg, "OFM_PANEL") is None
    assert resolve_odor_vial_conc(arg, "OFM_LIGHT") is None


def test_mapping_explicit_pseudo_pin_entry_wins():
    assert (
        resolve_odor_vial_conc("E=1,OFM_PANEL=mixed", "OFM_PANEL") == "mixed"
    )


def test_mapping_malformed_entry_raises():
    with pytest.raises(ValueError):
        resolve_odor_vial_conc("E=1,=0.1", "OFM_E")
    with pytest.raises(ValueError):
        resolve_odor_vial_conc("E=1,H=", "OFM_E")


# ── end-to-end folder names for the 3-odor / 2-mode run ──────────────────────

def test_resolved_conc_lands_in_push_folder():
    arg = "E=1,O=0.1,H=0.1"
    cases = {
        ("OFM_E", "Training"): "EB-Training-24-1",
        ("OFM_E", "Control"): "EB-Control-24-1",
        ("OFM_O", "Training"): "3Oct-Training-24-0.1",
        ("OFM_O", "Control"): "3Oct-Control-24-0.1",
        ("OFM_H", "Training"): "Hex-Training-24-0.1",
        ("OFM_H", "Control"): "Hex-Control-24-0.1",
    }
    for (pin, mode), expected in cases.items():
        conc = resolve_odor_vial_conc(arg, pin)
        remote = get_push_folder(
            pin, mode, BASE, starvation_hours=24, odor_vial_conc=conc
        )
        assert _folder(remote) == expected


# ── wiring into the rig scripts ──────────────────────────────────────────────

RIG_SCRIPTS = ["combinedv2_1.py", "combinedv2_1_pi1.py", "combinedv2_1_pi3.py"]


@pytest.mark.parametrize("script", RIG_SCRIPTS)
def test_rig_script_resolves_conc_before_building_push_folder(script):
    """Each rig script must resolve the mapping for the selected odor before
    ``get_push_folder`` bakes the conc into the push path (and before the
    metadata prompts read ``args.odor_vial_conc``)."""
    src = (PICODE / script).read_text(encoding="utf-8")
    tree = ast.parse(src)

    resolve_call_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None))
        == "resolve_odor_vial_conc"
    ]
    assert resolve_call_lines, f"{script} never calls resolve_odor_vial_conc"

    push_call_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None))
        == "get_push_folder"
    ]
    assert push_call_lines, f"{script} never calls get_push_folder"
    assert min(resolve_call_lines) < min(push_call_lines), (
        f"{script}: conc must be resolved before the push folder is built"
    )


# ── ATR reagent metadata constants ───────────────────────────────────────────
# All current batches are fed 300 µM all-trans-retinal (ATR) in both the
# agarose and food vials; session_metadata.txt must record that, not the old
# 150 µM retinol line.


@pytest.mark.parametrize("script", RIG_SCRIPTS)
def test_rig_script_records_300uM_atr(script):
    tree = ast.parse((PICODE / script).read_text(encoding="utf-8"))
    constants = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in (
                "RETINAL_CONCENTRATION", "RETINAL_PRODUCT"
            ):
                constants[target.id] = node.value.value
    assert constants["RETINAL_CONCENTRATION"] == "300 µM (agarose and food vials)"
    assert "all-trans-Retinal" in constants["RETINAL_PRODUCT"]
    assert "ATR" in constants["RETINAL_PRODUCT"]
