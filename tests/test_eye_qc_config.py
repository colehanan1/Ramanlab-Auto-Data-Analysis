"""Config for the per-frame eye-position QC (the top-level ``eye_qc:`` block).

The QC is report-only: distances, angles and gates keep the frozen eye anchor,
and the eye_drift_qc step only flags trials whose live eye drifts away from it
(a fly that came loose, a shifted mount). Pinned here:

* every key parses to its own field. The values are distinct, so a swapped or
  wrong key cannot pass by coincidence -- ``match_radius_px`` especially must
  not be confused with the two 80 px tracker / velocity gates;
* an absent, empty or malformed block, an empty value, or junk never raises.
  Every pipeline subprocess calls load_settings, so one raise aborts the whole
  nightly run; range checks belong to the QC step;
* ``eye_qc`` is a real Settings field, so dataclasses.replace (rig gate
  overrides, run_workflows' force flags) and pickling (parallel_map) keep it;
* it stays out of the pipeline cache key: tuning a QC threshold must not
  re-run YOLO on every dataset.

config/ is gitignored, so the live-config pins at the bottom skip on a clone
without config/config_new.yaml.
"""

from __future__ import annotations

import dataclasses
import pickle
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from fbpipe import config as fbconfig
from fbpipe.config import (
    DEFAULT_EYE_QC_REPORT_DIR,
    EyeQCSettings,
    Settings,
    load_settings,
    resolve_eye_qc_report_dir,
)
from fbpipe.utils.rig_gates import apply_rig_gate_overrides

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config" / "config_new.yaml"
EXAMPLE = REPO_ROOT / "config" / "example.yaml"

EYE_QC_FIELDS = [f.name for f in dataclasses.fields(EyeQCSettings)]
EYE_QC_ENV = (
    "EYE_QC_ENABLED",
    "EYE_QC_MAX_DRIFT_PX",
    "EYE_QC_SUSTAIN_S",
    "EYE_QC_MATCH_RADIUS_PX",
    "EYE_QC_MAX_MISSING_FRAC",
    "EYE_QC_MAX_MISSING_RUN_S",
    "EYE_QC_REPORT_DIR",
)


@pytest.fixture(autouse=True)
def _clean_eye_qc_env(monkeypatch):
    """A developer's shell must not leak an override into these tests."""
    for name in EYE_QC_ENV:
        monkeypatch.delenv(name, raising=False)


def _load(tmp_path: Path, body: str = "") -> Settings:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"model_path: /dev/null\nmain_directories:\n- {tmp_path}\n"
        + textwrap.dedent(body),
        encoding="utf-8",
    )
    return load_settings(path)


def _under_results_figures(path: Path) -> bool:
    parts = Path(path).parts
    return any(parts[i:i + 2] == ("Results", "Figures") for i in range(len(parts) - 1))


# Every eye_qc key off its default and distinct from every other key, next to
# the neighbouring gates at values of their own (none of them 80, the default
# both proboscis_match_max_dist_px and max_jump_px share).
FULL_BLOCK = """\
yolo:
  proboscis_match_max_dist_px: 81.0
proboscis_filter:
  max_eye_prob_distance_px: 161.0
  max_jump_px: 83.0
distance_limits:
  three_fly_max_eye_prob_distance_px: 162.0
eye_qc:
  enabled: false
  max_drift_px: 6.5
  sustain_s: 1.25
  match_radius_px: 37
  max_missing_frac: 0.35
  max_missing_run_s: 2.75
  draw_live_eye: false
  report_dir: {report_dir}
"""


def _full(tmp_path: Path, report_dir: Path | None = None) -> Settings:
    return _load(tmp_path, FULL_BLOCK.format(report_dir=report_dir or tmp_path / "yaml-report"))


# ── defaults ─────────────────────────────────────────────────────────


def test_dataclass_defaults():
    eq = EyeQCSettings()
    assert eq.enabled is True
    assert eq.max_drift_px == 5.0
    assert eq.sustain_s == 0.5
    assert eq.match_radius_px == 50.0
    assert eq.max_missing_frac == 0.2
    assert eq.max_missing_run_s == 2.0
    assert eq.draw_live_eye is True
    assert eq.report_dir == ""


def test_settings_carries_eye_qc_as_a_real_field():
    """A plain attribute set after construction would be dropped by
    dataclasses.replace -- and with it every per-batch rig-gate copy."""
    assert "eye_qc" in {f.name for f in dataclasses.fields(Settings)}
    a = Settings(model_path="m.pt", main_directories="/tmp")
    b = Settings(model_path="m.pt", main_directories="/tmp")
    assert a.eye_qc == EyeQCSettings()
    assert a.eye_qc is not b.eye_qc  # default_factory, not one shared instance


def test_absent_block_gives_defaults(tmp_path):
    assert _load(tmp_path).eye_qc == EyeQCSettings()


# ── YAML parse ───────────────────────────────────────────────────────


def test_full_block_parses_each_key_to_its_own_field(tmp_path):
    report = tmp_path / "eye-qc-report"
    s = _full(tmp_path, report)
    assert s.eye_qc == EyeQCSettings(
        enabled=False,
        max_drift_px=6.5,
        sustain_s=1.25,
        match_radius_px=37.0,
        max_missing_frac=0.35,
        max_missing_run_s=2.75,
        draw_live_eye=False,
        report_dir=str(report),
    )
    assert type(s.eye_qc.match_radius_px) is float  # YAML int, stored as float
    # The neighbouring gates keep their own values: match_radius_px is neither
    # read from nor written into the tracker / velocity / spatial keys.
    assert s.proboscis_match_max_dist_px == 81.0
    assert s.proboscis_filter.max_jump_px == 83.0
    assert s.proboscis_filter.max_eye_prob_distance_px == 161.0
    assert s.three_fly_max_eye_prob_distance_px == 162.0


@pytest.mark.parametrize(
    "body, enabled, draw",
    [
        ("eye_qc:\n  enabled: false\n", False, True),
        ("eye_qc:\n  draw_live_eye: false\n", True, False),
    ],
    ids=["only-enabled-off", "only-marker-off"],
)
def test_the_two_switches_are_independent(tmp_path, body, enabled, draw):
    eq = _load(tmp_path, body).eye_qc
    assert (eq.enabled, eq.draw_live_eye) == (enabled, draw)


@pytest.mark.parametrize(
    "body",
    [
        "eye_qc:\n",
        "eye_qc: true\n",
        "eye_qc: []\n",
        "eye_qc: [max_drift_px]\n",
        "eye_qc: loose\n",
        "eye_qc: 7\n",
    ],
    ids=["empty", "true", "empty-list", "list", "string", "number"],
)
def test_non_mapping_block_gives_defaults_without_raising(tmp_path, body):
    assert _load(tmp_path, body).eye_qc == EyeQCSettings()


def test_bare_false_turns_the_qc_off(tmp_path):
    """``eye_qc: false`` must not read as "all defaults", which is ON."""
    assert _load(tmp_path, "eye_qc: false\n").eye_qc == EyeQCSettings(enabled=False)


def test_empty_values_fall_back_to_the_defaults(tmp_path):
    """``max_drift_px:`` with nothing after it loads as None, and float(None)
    raises -- which would abort the run from load_settings."""
    body = "eye_qc:\n" + "".join(f"  {name}:\n" for name in EYE_QC_FIELDS)
    assert _load(tmp_path, body).eye_qc == EyeQCSettings()


def test_one_empty_value_leaves_the_others_alone(tmp_path):
    eq = _load(tmp_path, "eye_qc:\n  max_drift_px:\n  sustain_s: 1.25\n").eye_qc
    assert eq.max_drift_px == EyeQCSettings().max_drift_px
    assert eq.sustain_s == 1.25


@pytest.mark.parametrize(
    "key, junk",
    [
        ("max_drift_px", "five"),
        ("sustain_s", "[1, 2]"),
        ("match_radius_px", "{a: 1}"),
        ("max_missing_frac", "true"),  # a bool is not silently 1.0
        ("max_missing_run_s", "two seconds"),
    ],
)
def test_junk_number_warns_and_falls_back_without_raising(tmp_path, capsys, key, junk):
    eq = _load(tmp_path, f"eye_qc:\n  {key}: {junk}\n").eye_qc
    assert getattr(eq, key) == getattr(EyeQCSettings(), key)
    assert f"eye_qc.{key}" in capsys.readouterr().out


def test_out_of_range_numbers_are_left_for_the_qc_step(tmp_path):
    """load_settings only coerces types. The QC step range-checks, warns and
    falls back itself, so nothing is clamped silently here."""
    eq = _load(
        tmp_path, "eye_qc:\n  max_drift_px: -3\n  max_missing_frac: 1.5\n  max_missing_run_s: -4\n"
    ).eye_qc
    assert eq.max_drift_px == -3.0
    assert eq.max_missing_frac == 1.5
    assert eq.max_missing_run_s == -4.0


def test_parser_fallbacks_come_from_the_dataclass(tmp_path, monkeypatch):
    """No literal in load_settings can drift from EyeQCSettings: an empty
    value falls back to whatever the dataclass says today."""

    @dataclasses.dataclass
    class Retuned(EyeQCSettings):
        enabled: bool = False
        max_drift_px: float = 9.5
        sustain_s: float = 3.0
        match_radius_px: float = 44.0
        max_missing_frac: float = 0.65
        max_missing_run_s: float = 4.5
        draw_live_eye: bool = False
        report_dir: str = "/elsewhere"

    monkeypatch.setattr(fbconfig, "EyeQCSettings", Retuned)
    body = "eye_qc:\n" + "".join(f"  {name}:\n" for name in EYE_QC_FIELDS)
    assert _load(tmp_path, body).eye_qc == Retuned()


# ── env overrides ────────────────────────────────────────────────────


def test_env_overrides_beat_the_yaml(tmp_path, monkeypatch):
    env_report = tmp_path / "env-report"
    monkeypatch.setenv("EYE_QC_MAX_DRIFT_PX", "7.25")
    monkeypatch.setenv("EYE_QC_SUSTAIN_S", "2.5")
    monkeypatch.setenv("EYE_QC_MATCH_RADIUS_PX", "41")
    monkeypatch.setenv("EYE_QC_MAX_MISSING_FRAC", "0.45")
    monkeypatch.setenv("EYE_QC_MAX_MISSING_RUN_S", "3.25")
    monkeypatch.setenv("EYE_QC_REPORT_DIR", str(env_report))
    eq = _full(tmp_path).eye_qc
    assert (
        eq.max_drift_px, eq.sustain_s, eq.match_radius_px, eq.max_missing_frac, eq.max_missing_run_s
    ) == (7.25, 2.5, 41.0, 0.45, 3.25)
    assert eq.report_dir == str(env_report)
    assert eq.draw_live_eye is False  # no env override: the YAML value stands


@pytest.mark.parametrize(
    "env, yaml_value, expected",
    [
        ("false", "true", False),
        ("0", "true", False),
        ("off", "true", False),
        ("TRUE", "false", True),
        ("1", "false", True),
        ("yes", "false", True),
    ],
)
def test_enabled_env_follows_the_as_bool_convention(tmp_path, monkeypatch, env, yaml_value, expected):
    monkeypatch.setenv("EYE_QC_ENABLED", env)
    assert _load(tmp_path, f"eye_qc:\n  enabled: {yaml_value}\n").eye_qc.enabled is expected


def test_env_applies_without_a_yaml_block(tmp_path, monkeypatch):
    monkeypatch.setenv("EYE_QC_MAX_DRIFT_PX", "7.25")
    eq = _load(tmp_path).eye_qc
    assert eq.max_drift_px == 7.25
    assert eq.sustain_s == EyeQCSettings().sustain_s


def test_blank_env_var_is_ignored(tmp_path, monkeypatch):
    """An exported-but-empty variable is no override (float("") would raise)."""
    monkeypatch.setenv("EYE_QC_MAX_DRIFT_PX", "  ")
    monkeypatch.setenv("EYE_QC_ENABLED", "")
    eq = _load(tmp_path, "eye_qc:\n  max_drift_px: 6.5\n  enabled: true\n").eye_qc
    assert eq.max_drift_px == 6.5
    assert eq.enabled is True


def test_junk_env_number_falls_back_without_raising(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EYE_QC_SUSTAIN_S", "half a second")
    eq = _load(tmp_path, "eye_qc:\n  sustain_s: 1.25\n").eye_qc
    assert eq.sustain_s == EyeQCSettings().sustain_s
    assert "eye_qc.sustain_s" in capsys.readouterr().out


# ── report dir ───────────────────────────────────────────────────────


def test_default_report_dir_is_under_new_opto_fly_figures():
    """Pipeline output goes to New-Opto-Fly-Figures; Results/Figures holds
    hand-run one-offs that no pipeline step may write into."""
    s = Settings(model_path="m.pt", main_directories="/tmp")
    report = resolve_eye_qc_report_dir(s)
    assert report == DEFAULT_EYE_QC_REPORT_DIR
    assert report.is_absolute()
    assert report.parts[-3:] == ("Results", "New-Opto-Fly-Figures", "Eye-Tracking-QC")
    assert not _under_results_figures(report)


def test_report_dir_setting_wins(tmp_path):
    s = Settings(
        model_path="m.pt",
        main_directories="/tmp",
        eye_qc=EyeQCSettings(report_dir=str(tmp_path / "qc")),
    )
    assert resolve_eye_qc_report_dir(s) == tmp_path / "qc"


def test_report_dir_from_yaml_and_env(tmp_path, monkeypatch):
    assert resolve_eye_qc_report_dir(_full(tmp_path, tmp_path / "from-yaml")) == tmp_path / "from-yaml"
    monkeypatch.setenv("EYE_QC_REPORT_DIR", str(tmp_path / "from-env"))
    assert resolve_eye_qc_report_dir(_full(tmp_path, tmp_path / "from-yaml")) == tmp_path / "from-env"


def test_report_dir_expands_user(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    s = SimpleNamespace(eye_qc=EyeQCSettings(report_dir="~/qc"))
    assert resolve_eye_qc_report_dir(s) == tmp_path / "qc"


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_report_dir_is_the_default(blank):
    s = SimpleNamespace(eye_qc=EyeQCSettings(report_dir=blank))
    assert resolve_eye_qc_report_dir(s) == DEFAULT_EYE_QC_REPORT_DIR


def test_report_dir_inside_results_figures_is_refused(capsys):
    s = SimpleNamespace(
        eye_qc=EyeQCSettings(report_dir="/home/ramanlab/Documents/cole/Results/Figures/Eye-QC")
    )
    assert resolve_eye_qc_report_dir(s) == DEFAULT_EYE_QC_REPORT_DIR
    assert "Results/Figures" in capsys.readouterr().out


def test_resolver_tolerates_settings_without_eye_qc():
    """Duck-typed stand-ins (SimpleNamespace cfgs in step tests) carry no eye_qc."""
    assert resolve_eye_qc_report_dir(SimpleNamespace()) == DEFAULT_EYE_QC_REPORT_DIR


# ── survives copies ──────────────────────────────────────────────────


RIG_OVERRIDE = """\
rig_gate_overrides:
- host: Flybehavior2
  after: 2026-07-25
  max_eye_prob_distance_px: 280.0
  class2_max: 280.0
"""


def _flybehavior2_batch(tmp_path: Path) -> Path:
    batch = tmp_path / "august_11_batch_4"
    batch.mkdir()
    (batch / "session_metadata.txt").write_text(
        "Host: Flybehavior2 | OS: Linux\n", encoding="utf-8"
    )
    (batch / "output_august_11_batch_4_training_2_3-Octonol_20260811_163740.csv").write_text(
        "frame\n0\n", encoding="utf-8"
    )
    return batch


def test_rig_gate_override_copy_keeps_eye_qc(tmp_path):
    s = _load(tmp_path, FULL_BLOCK.format(report_dir=tmp_path / "qc") + RIG_OVERRIDE)
    eff = apply_rig_gate_overrides(s, _flybehavior2_batch(tmp_path))
    assert eff is not s and eff.class2_max == 280.0  # the override really applied
    assert eff.eye_qc == s.eye_qc
    assert eff.eye_qc != EyeQCSettings()  # not silently reset to the defaults


def test_loaded_settings_pickle_round_trip(tmp_path):
    """parallel_map ships cfg to worker processes by pickling it."""
    s = _full(tmp_path)
    back = pickle.loads(pickle.dumps(s))
    assert isinstance(back.eye_qc, EyeQCSettings)
    assert back.eye_qc == s.eye_qc


def test_eye_qc_is_not_part_of_the_pipeline_cache_key(tmp_path):
    """Retuning a QC threshold must not read as pipeline drift: that would
    re-run YOLO and every per-trial stage on every dataset."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.pipeline.run_workflows import _pipeline_expectation

    s = _load(tmp_path)
    root = tmp_path / "EB-Training-24-1"
    root.mkdir()
    retuned = dataclasses.replace(
        s,
        eye_qc=EyeQCSettings(
            enabled=False,
            max_drift_px=9.0,
            sustain_s=2.0,
            match_radius_px=30.0,
            max_missing_frac=0.5,
            max_missing_run_s=6.0,
            draw_live_eye=False,
            report_dir=str(tmp_path / "qc"),
        ),
    )
    assert _pipeline_expectation(retuned, root) == _pipeline_expectation(s, root)


# ── committed + live configs ─────────────────────────────────────────


class _DupCheckLoader(yaml.SafeLoader):
    """YAML keeps the LAST duplicate key; a silently-lost block is a real bug."""


def _no_duplicates(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise AssertionError(f"duplicate key {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_DupCheckLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates)


def test_example_config_documents_every_key_at_its_default(tmp_path):
    """config/example.yaml is the only committed config."""
    raw = yaml.load(EXAMPLE.read_text(encoding="utf-8"), Loader=_DupCheckLoader)
    block = raw.get("eye_qc")
    assert isinstance(block, dict), "example.yaml has no top-level eye_qc: mapping"
    assert sorted(block) == sorted(EYE_QC_FIELDS)
    assert _load(tmp_path, yaml.safe_dump({"eye_qc": block})).eye_qc == EyeQCSettings()


@pytest.fixture(scope="module")
def live_cfg():
    if not CONFIG.exists():
        pytest.skip("config/config_new.yaml not present (gitignored)")
    return yaml.load(CONFIG.read_text(encoding="utf-8"), Loader=_DupCheckLoader)


def test_live_config_declares_one_top_level_eye_qc_block(live_cfg):
    block = live_cfg.get("eye_qc")
    assert isinstance(block, dict), "config_new.yaml has no top-level eye_qc: mapping"
    unknown = sorted(set(block) - set(EYE_QC_FIELDS))
    assert not unknown, f"eye_qc keys load_settings would silently ignore: {unknown}"
    assert "eye_qc" not in (live_cfg.get("proboscis_filter") or {}), (
        "eye_qc is indented under proboscis_filter instead of top level"
    )


def test_live_config_eye_qc_values(live_cfg, tmp_path):
    """The operative thresholds, read through the production parser.

    Good trials (2026-10-07, Flybehavior2) drift a median 0.4-0.9 px, max
    2.7 px, so 5 px stands clear of detection jitter. The 50 px match radius
    must exceed the drift threshold, or a real shift reads as "eye missing".
    On six good trials (pretest, training with light, LightOnly) every eye
    matched in every frame -- longest unmatched run 0 s -- so 2 s is safe.
    """
    block = live_cfg.get("eye_qc")
    assert isinstance(block, dict), "config_new.yaml has no top-level eye_qc: mapping"
    eq = _load(tmp_path, yaml.safe_dump({"eye_qc": block})).eye_qc
    assert eq.enabled is True
    assert eq.max_drift_px == 5.0
    assert eq.sustain_s == 0.5
    assert eq.match_radius_px == 50.0
    assert eq.max_missing_frac == 0.2
    assert eq.max_missing_run_s == 2.0
    assert eq.draw_live_eye is True
    assert eq.match_radius_px > eq.max_drift_px
    report = resolve_eye_qc_report_dir(SimpleNamespace(eye_qc=eq))
    assert report.parts[-2:] == ("New-Opto-Fly-Figures", "Eye-Tracking-QC")
    assert not _under_results_figures(report)
