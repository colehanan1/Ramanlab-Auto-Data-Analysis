"""Light-on hold windows for the drift-following eye anchor comparison."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("des", ROOT / "scripts" / "analysis" / "drift_eye_shadow.py")
des = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(des)


def test_light_hold_covers_half_a_second_before_on_to_one_second_after_off():
    assert des.light_hold({"light_on_s": 30.0, "light_off_s": 55.0}, 40.0) == (1180, 2240)


def test_no_light_no_hold():
    assert des.light_hold({"light_on_s": None, "light_off_s": None}, 40.0) is None
    assert des.light_hold({}, 40.0) is None


def test_light_without_off_holds_to_the_end():
    assert des.light_hold({"light_on_s": 30.0, "n_frames": 3600}, 40.0) == (1180, 3640)
