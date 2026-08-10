#!/usr/bin/env python3
"""Pi 1 light must be digital HIGH/LOW only — no PWM, no pulse train.

Pi 1 (`behaviorlocust`) drives a Thorlabs LED controller through its TTL
trigger input on BCM 25. Brightness is set on the controller itself, so any
modulation from the Pi — a PWM dimming carrier *or* a software pulse train —
is wrong: it just chops the LED. These tests pin that down so no code path in
``combinedv2_1_pi1.py`` can emit modulation again.

The module can't be imported (module-level argparse + lgpio + picamera2 +
config load), so this inspects the source with ``ast``.
"""

import ast
import os

PI1 = os.path.join(os.path.dirname(__file__), "..", "PiCode", "combinedv2_1_pi1.py")

with open(PI1, encoding="utf-8") as _fh:
    SOURCE = _fh.read()
TREE = ast.parse(SOURCE)


def _calls():
    return [n for n in ast.walk(TREE) if isinstance(n, ast.Call)]


def _call_name(node):
    """Dotted name of a call target, e.g. 'lgpio.tx_pwm'."""
    f = node.func
    parts = []
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
    return ".".join(reversed(parts))


def _functions():
    return {
        n.name: n
        for n in ast.walk(TREE)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _cli_flags():
    """Every flag string passed to parser.add_argument(...)."""
    flags = []
    for c in _calls():
        if _call_name(c).endswith("add_argument"):
            for a in c.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    flags.append(a.value)
    return flags


def test_no_pwm_calls_at_all():
    """lgpio.tx_pwm must not appear anywhere — the TTL input is on or off."""
    pwm = [c for c in _calls() if _call_name(c).endswith("tx_pwm")]
    assert pwm == [], (
        f"{len(pwm)} tx_pwm call(s) remain at lines "
        f"{sorted(c.lineno for c in pwm)}"
    )


def test_no_ramp_or_brightness_cli_flags():
    """The PWM brightness-ramp flags are gone from the CLI."""
    flags = _cli_flags()
    banned = [f for f in flags if f.startswith("--ramp") or f == "--light-brightness"]
    assert banned == [], f"PWM CLI flags still registered: {banned}"


def test_ramp_helpers_removed():
    """The gamma ramp and its standalone test mode are gone."""
    fns = _functions()
    for name in ("light_ramp_blocking", "run_ramp_test", "clamp_pwm_freq"):
        assert name not in fns, f"{name}() still defined at line {fns[name].lineno}"


def test_light_controller_takes_no_hz_or_duty():
    """LightController.on() is a bare digital ON — no pulse parameters."""
    cls = next(
        (n for n in ast.walk(TREE)
         if isinstance(n, ast.ClassDef) and n.name == "LightController"),
        None,
    )
    assert cls is not None, "LightController class missing"

    methods = {n.name: n for n in cls.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "_run" not in methods, "LightController._run() pulse loop still present"

    on = methods.get("on")
    assert on is not None, "LightController.on() missing"
    params = [a.arg for a in on.args.args if a.arg != "self"]
    params += [a.arg for a in on.args.kwonlyargs]
    assert params == [], f"LightController.on() still accepts {params}"


def test_no_call_site_passes_hz_or_duty_to_light_on():
    """No caller asks the light to pulse."""
    offenders = []
    for c in _calls():
        if _call_name(c) in ("light.on", "_light_solid_all"):
            kws = [k.arg for k in c.keywords if k.arg]
            if any(k in ("hz", "duty", "pwm_hz", "brightness") for k in kws):
                offenders.append((c.lineno, kws))
            if c.args:
                offenders.append((c.lineno, "positional args"))
    assert offenders == [], f"pulse/dim args still passed at {offenders}"


def test_light_solid_all_has_no_modulation_params():
    """_light_solid_all() drives HIGH; it has nothing to dim."""
    fn = _functions().get("_light_solid_all")
    assert fn is not None, "_light_solid_all() missing"
    params = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
    assert params == [], f"_light_solid_all() still accepts {params}"


def test_pwm_constants_removed():
    """No dimming-carrier constants left to tempt a future call site."""
    assigned = {
        t.id
        for n in ast.walk(TREE) if isinstance(n, ast.Assign)
        for t in n.targets if isinstance(t, ast.Name)
    }
    for const in ("PWM_LIGHT_HZ", "PWM_FREQ_MIN_HZ", "PWM_FREQ_MAX_HZ",
                  "LIGHT_BRIGHTNESS_PCT"):
        assert const not in assigned, f"{const} still defined"


def test_light_pin_is_still_the_thorlabs_ttl_pin():
    """Guard against the strip-out accidentally moving the pin."""
    assert "LIGHT_PINS = [25]" in SOURCE


if __name__ == "__main__":
    import sys
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS: {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL: {name}: {exc}")
    sys.exit(1 if failures else 0)
