"""Config for the batch-level eye anchor template (the ``eye_template:`` block).

The template fixes a batch's fly slot set from every trial's first frames, so
a fly whose eye sits at the frame edge cannot be numbered ``fly1`` in one
trial and dropped in the next. Pinned here:

* every key parses to its own field, with distinct values so a swapped key
  cannot pass by coincidence. ``match_radius_px`` here is the slot-to-anchor
  search for ONE trial's own detections -- it is not ``eye_qc.match_radius_px``
  (per-frame QC) and not the 80 px tracker / velocity gates;
* an absent, empty or malformed block, an empty value, or junk never raises.
  Every pipeline subprocess calls load_settings, so one raise aborts the whole
  nightly run;
* ``enabled: false`` is the escape hatch back to per-video sizing;
* it is a real Settings field, so dataclasses.replace (rig gate overrides) and
  pickling (parallel_map) keep it.

config/ is gitignored, so the live-config pins at the bottom skip on a clone
without config/config_new.yaml.
"""

from __future__ import annotations

import dataclasses
import pickle
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fbpipe.config import EyeTemplateSettings, Settings, load_settings  # noqa: E402
from fbpipe.utils.rig_gates import apply_rig_gate_overrides  # noqa: E402

CONFIG = REPO_ROOT / "config" / "config_new.yaml"


def _load(tmp_path: Path, body: str) -> Settings:
    path = tmp_path / "cfg.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return load_settings(str(path))


def test_defaults_are_on_with_two_trial_corroboration():
    cfg = EyeTemplateSettings()
    assert cfg.enabled is True
    assert cfg.min_trials == 2


def test_every_key_parses_to_its_own_field(tmp_path):
    cfg = _load(
        tmp_path,
        """
        eye_template:
          enabled: true
          min_trials: 3
          match_radius_px: 111.0
          cluster_tol_min_px: 22.0
          cluster_tol_frac: 0.33
        """,
    )
    assert cfg.eye_template.min_trials == 3
    assert cfg.eye_template.match_radius_px == pytest.approx(111.0)
    assert cfg.eye_template.cluster_tol_min_px == pytest.approx(22.0)
    assert cfg.eye_template.cluster_tol_frac == pytest.approx(0.33)


def test_the_template_match_radius_is_not_the_eye_qc_one(tmp_path):
    """Two different radii with two different jobs; reading the wrong key must
    not be correct by coincidence."""
    cfg = _load(
        tmp_path,
        """
        eye_qc:
          match_radius_px: 37.0
        eye_template:
          match_radius_px: 123.0
        """,
    )
    assert cfg.eye_qc.match_radius_px == pytest.approx(37.0)
    assert cfg.eye_template.match_radius_px == pytest.approx(123.0)


def test_an_absent_block_gives_defaults(tmp_path):
    cfg = _load(tmp_path, "max_flies: 4\n")
    assert cfg.eye_template == EyeTemplateSettings()


def test_a_bare_false_turns_it_off(tmp_path):
    cfg = _load(tmp_path, "eye_template: false\n")
    assert cfg.eye_template.enabled is False


def test_a_malformed_block_never_raises(tmp_path):
    cfg = _load(tmp_path, "eye_template: [1, 2, 3]\n")
    assert cfg.eye_template == EyeTemplateSettings()


def test_junk_values_fall_back_rather_than_raising(tmp_path):
    cfg = _load(
        tmp_path,
        """
        eye_template:
          min_trials: "not a number"
          match_radius_px: ""
          cluster_tol_frac: null
        """,
    )
    default = EyeTemplateSettings()
    assert cfg.eye_template.min_trials == default.min_trials
    assert cfg.eye_template.match_radius_px == pytest.approx(default.match_radius_px)
    assert cfg.eye_template.cluster_tol_frac == pytest.approx(default.cluster_tol_frac)


def test_it_survives_dataclasses_replace_and_pickling(tmp_path):
    cfg = _load(tmp_path, "eye_template:\n  min_trials: 4\n")
    assert dataclasses.replace(cfg, max_flies=9).eye_template.min_trials == 4
    assert pickle.loads(pickle.dumps(cfg)).eye_template.min_trials == 4


def test_rig_gate_overrides_keep_the_template_settings(tmp_path):
    cfg = _load(tmp_path, "eye_template:\n  min_trials: 4\n")
    widened = apply_rig_gate_overrides(cfg, Path("/tmp/flybehavior2_august_11_batch_4"))
    assert widened.eye_template.min_trials == 4


# ── The live config ──────────────────────────────────────────────────────────


@pytest.mark.skipif(not CONFIG.exists(), reason="config/ is gitignored")
def test_live_config_enables_the_template_with_min_trials_two():
    cfg = load_settings(str(CONFIG))
    assert cfg.eye_template.enabled is True
    assert cfg.eye_template.min_trials == 2


@pytest.mark.skipif(not CONFIG.exists(), reason="config/ is gitignored")
def test_live_config_declares_the_block_once():
    """YAML keeps the LAST duplicate key -- a second eye_template: block would
    silently win and parse fine."""
    text = CONFIG.read_text(encoding="utf-8")
    assert sum(1 for line in text.splitlines() if line.startswith("eye_template:")) == 1
    assert isinstance(yaml.safe_load(text).get("eye_template"), dict)


def test_require_proboscis_defaults_on():
    """An eye alone is not a measurable fly (june_03_batch_2's bottom-edge
    slot: detected in 2 trials, proboscis never in frame)."""
    assert EyeTemplateSettings().require_proboscis is True


def test_require_proboscis_can_be_turned_off(tmp_path):
    cfg = _load(tmp_path, "eye_template:\n  require_proboscis: false\n")
    assert cfg.eye_template.require_proboscis is False
    assert cfg.eye_template.enabled is True


@pytest.mark.skipif(not CONFIG.exists(), reason="config/ is gitignored")
def test_live_config_requires_a_pairable_proboscis():
    cfg = load_settings(str(CONFIG))
    assert cfg.eye_template.require_proboscis is True
