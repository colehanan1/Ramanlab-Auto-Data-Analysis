"""get_odor_from_config: the trained valve of a rig config.

A suffixed filename (config_v2_A_10s.yaml) used to resolve to OFM_LIGHT because
only the filename's last token was read; the rig then pushed to LightSweep and
died on "Missing --odor-vial-conc". The YAML's experiment.trained_odor wins.
"""
import sys
from pathlib import Path

import pytest

PICODE = Path(__file__).resolve().parent.parent / "PiCode"
sys.path.insert(0, str(PICODE))
from experiment_scheduler import get_odor_from_config  # noqa: E402


@pytest.mark.parametrize("name,pin", [
    ("config_v2_A_10s.yaml", "OFM_A"),
    ("config_v2_A_benzpre.yaml", "OFM_A"),
    ("config_v2_A.yaml", "OFM_A"),
    ("config_v2_hextest_H.yaml", "OFM_H"),
])
def test_real_configs(name, pin):
    if not (PICODE / name).exists():
        pytest.skip(f"{name} not on disk (PiCode is gitignored)")
    assert get_odor_from_config(PICODE / name) == pin


def test_trained_odor_beats_filename(tmp_path):
    p = tmp_path / "config_v2_E_whatever.yaml"
    p.write_text("format: v2\nexperiment:\n  trained_odor: OFM_O\n")
    assert get_odor_from_config(p) == "OFM_O"


def test_filename_fallback_without_trained_odor(tmp_path):
    p = tmp_path / "config_v2_H.yaml"
    p.write_text("cycles: []\n")
    assert get_odor_from_config(p) == "OFM_H"


def test_light_sweep_stays_light(tmp_path):
    p = tmp_path / "config_v2_lightsweep.yaml"
    p.write_text("cycles: []\n")
    assert get_odor_from_config(p) == "OFM_LIGHT"


def test_missing_file_uses_filename(tmp_path):
    assert get_odor_from_config(tmp_path / "config_v2_C.yaml") == "OFM_C"
