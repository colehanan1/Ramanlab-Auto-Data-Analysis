"""Rig wiring for ``~/odor_pins.txt`` and the experiment ``cohort`` key.

The rig scripts can't be imported off the Pi, so the wiring is checked against
their source: the pins file must be applied before anything reads the pin
tables, its concentrations must feed the existing --odor-vial-conc path, and
the config's cohort must reach the push folder.
"""

import ast
import subprocess
from pathlib import Path

import pytest

PICODE = Path(__file__).resolve().parent.parent / "PiCode"
SCRIPTS = ["combinedv2_1.py", "combinedv2_1_pi1.py", "combinedv2_1_pi3.py"]


def _src(script):
    return (PICODE / script).read_text(encoding="utf-8")


def _module_line(src, needle):
    """Line of the first top-level statement whose source contains ``needle``."""
    tree = ast.parse(src)
    for node in tree.body:
        if needle in ast.get_source_segment(src, node):
            return node.lineno
    raise AssertionError(f"{needle!r} not found at module level")


@pytest.mark.parametrize("script", SCRIPTS)
def test_pins_file_flag(script):
    src = _src(script)
    assert '"--pins-file"' in src
    assert "~/odor_pins.txt" in src          # the help text names the default


@pytest.mark.parametrize("script", SCRIPTS)
def test_pins_applied_before_any_pin_table_is_read(script):
    """Labels, folders and the odor filter all read the tables at module level."""
    src = _src(script)
    apply_at = _module_line(src, "load_and_apply_pin_file(args.pins_file)")
    assert apply_at < _module_line(src, "_odor_filter = None")
    assert apply_at < _module_line(src, "if args.config is not None:")
    assert apply_at < _module_line(src, "expand_config(cfg")


@pytest.mark.parametrize("script", SCRIPTS)
def test_bad_pins_file_stops_the_script(script):
    src = _src(script)
    tree = ast.parse(src)
    tries = [n for n in tree.body if isinstance(n, ast.Try)
             and "load_and_apply_pin_file" in ast.get_source_segment(src, n)]
    assert tries, "load_and_apply_pin_file must be wrapped in try/except"
    handler_src = ast.get_source_segment(src, tries[0].handlers[0])
    assert "SystemExit" in handler_src


@pytest.mark.parametrize("script", SCRIPTS)
def test_pin_table_printed_at_startup(script):
    assert "format_pin_table(_pin_map, _pin_file)" in _src(script)


@pytest.mark.parametrize("script", SCRIPTS)
def test_file_concentrations_feed_the_existing_vial_path(script):
    """vial_conc_arg rewrites args.odor_vial_conc before it is parsed."""
    src = _src(script)
    tree = ast.parse(src)

    def calls(name):
        return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and getattr(n.func, "id", None) == name]

    (pins,) = calls("vial_conc_arg")
    assert [ast.unparse(a) for a in pins.args] == \
        ["_pin_map", "args.odor_vial_conc", "_odor_pin"]
    assert pins.lineno < calls("parse_odor_vial_conc_map")[0].lineno
    assert pins.lineno < calls("resolve_odor_vial_conc")[0].lineno
    # ...and its result is what the rest of the script reads.
    assign = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                  and n.value is pins)
    assert ast.unparse(assign.targets[0]) == "(args.odor_vial_conc, _pin_warning)"


@pytest.mark.parametrize("script", SCRIPTS)
def test_cohort_reaches_the_push_folder(script):
    src = _src(script)
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "get_push_folder"]
    assert calls
    for c in calls:
        assert "cohort" in {k.arg for k in c.keywords}, \
            f"{script}:{c.lineno} get_push_folder without cohort="


@pytest.mark.parametrize("script", SCRIPTS)
def test_pins_file_recorded_in_session_metadata(script):
    src = _src(script)
    assert 'md["odor_pins_file"]' in src
    assert "Odor pin file:" in src


@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_parse_on_the_pi_python(script):
    """Pi 1 runs Python 3.11; keep 3.12-only syntax out of the rig scripts."""
    ast.parse(_src(script), feature_version=(3, 11))


def test_run_sh_no_longer_requires_odor_vial_conc():
    src = (PICODE / "run.sh").read_text()
    assert "ODOR_VIAL_CONC:?" not in src
    assert "--odor-vial-conc" in src          # still passed through when set


def test_run_sh_is_valid_bash():
    subprocess.run(["bash", "-n", str(PICODE / "run.sh")], check=True)


@pytest.mark.parametrize("host,script", [
    ("BehaviorLocust", "combinedv2_1_pi1.py"),     # real hostnames (checked)
    ("behaviorlocust", "combinedv2_1_pi1.py"),
    ("Flybehavior2", "combinedv2_1.py"),
    ("Flybehavior3", "combinedv2_1_pi3.py"),
    ("flybehavior3", "combinedv2_1_pi3.py"),
])
def test_run_sh_picks_each_rigs_own_script(host, script):
    """Pi 3's valve GPIOs differ from Pi 2's, so the wrong script mis-fires."""
    src = (PICODE / "run.sh").read_text()
    case = src[src.index('case "$HOST" in'):src.index("esac") + 4]
    out = subprocess.run(["bash", "-c", f'HOST={host}\n{case}\necho "$PI_SCRIPT"'],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == script


@pytest.mark.parametrize("rig", ["pi1", "pi2", "pi3"])
def test_run_command_files_name_their_own_script(rig):
    script = {"pi1": "combinedv2_1_pi1.py", "pi2": "combinedv2_1.py",
              "pi3": "combinedv2_1_pi3.py"}[rig]
    text = (PICODE / f"HEXTEST_RUN_COMMAND_{rig}.txt").read_text()
    assert f"python3 {script} --config config_v2_hextest_L.yaml --training" in text
    assert "--odors" not in text.split("Variants")[0]
    assert "--odor-vial-conc" not in text.split("Variants")[0]
