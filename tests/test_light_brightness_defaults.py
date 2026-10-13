"""Per-rig LDD-L training-light brightness defaults (``--light-brightness``).

Set 2026-10-02 at the user's request: Pi 2 15 %, Pi 3 30 %. Pi 1 has no
brightness flag (Thorlabs TTL, digital only; see test_pi1_light_digital_only).

Pi 3's 30 % previously lived only as an on-Pi edit, which a plain PiCode push
silently reverted to 15; pinning it here keeps the repo the source of truth.
"""

import ast
from pathlib import Path

import pytest

PICODE = Path(__file__).resolve().parent.parent / "PiCode"


def _brightness_arg(script):
    tree = ast.parse((PICODE / script).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"
                and node.args and getattr(node.args[0], "value", None) == "--light-brightness"):
            kw = {k.arg: k.value for k in node.keywords}
            return ast.literal_eval(kw["default"]), ast.literal_eval(kw["help"])
    raise AssertionError(f"{script}: no --light-brightness argument")


@pytest.mark.parametrize("script,pct", [
    ("combinedv2_1.py", 15.0),        # Pi 2 (flybehavior2)
    ("combinedv2_1_pi3.py", 30.0),    # Pi 3 (flybehavior3)
])
def test_default_brightness(script, pct):
    default, help_text = _brightness_arg(script)
    assert default == pct
    assert f"Default {int(pct)}." in help_text      # help must not lie
