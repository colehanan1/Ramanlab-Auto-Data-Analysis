"""The training-AUC analysis, run on trained arms as well as controls.

It was written for control cohorts and says so on the figures: "Control cohort:
{odor} presented during conditioning with nothing paired to it." That sentence is
FALSE for a trained arm -- there the odor was paired with light, which is the
whole point of the cohort -- so pointing the script at a trained dataset without
touching the captions would ship figures making a false claim about the protocol.

The analysis itself transfers: "does conditioning vigor predict the test
response" is well posed for either arm. Only the wording has to follow the arm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from scripts.analysis import training_auc_vs_control_response as mod  # noqa: E402


CONTROLS = [
    "EB-Control-24-1", "3Oct-Control-24-0.1",
    "Hex-Control-24-0.01", "Hex-Control-24-0.1",
]
TRAINED = [
    "EB-Training-24-1", "3Oct-Training-24-0.1",
    "Hex-Training-24-0.01", "Hex-Training-24-0.1",
]


# ── which arm is this? ────────────────────────────────────────────────────


@pytest.mark.parametrize("ds", CONTROLS)
def test_control_datasets_read_as_control(ds):
    assert mod.is_trained_arm(ds) is False


@pytest.mark.parametrize("ds", TRAINED)
def test_training_datasets_read_as_trained(ds):
    assert mod.is_trained_arm(ds) is True


def test_an_unrecognised_name_is_not_claimed_as_trained():
    """Fail toward the existing wording rather than asserting a pairing that
    may not exist."""
    assert mod.is_trained_arm("RandomPanel-24-1") is False
    assert mod.is_trained_arm("") is False


# ── the noun used for the flies ───────────────────────────────────────────


def test_cohort_noun_follows_the_arm():
    assert "control" in mod.cohort_noun("EB-Control-24-1").lower()
    assert "control" not in mod.cohort_noun("EB-Training-24-1").lower()
    assert "trained" in mod.cohort_noun("EB-Training-24-1").lower()


# ── the protocol sentence ─────────────────────────────────────────────────


def test_control_protocol_line_says_nothing_was_paired():
    line = mod.protocol_line("EB-Control-24-1", "Ethyl Butyrate")
    assert "nothing" in line.lower()
    assert "light" not in line.lower()


def test_trained_protocol_line_says_the_odor_was_paired_with_light():
    """The correction that motivates this module: a trained arm must not claim
    its odor was presented with nothing paired to it."""
    line = mod.protocol_line("EB-Training-24-1", "Ethyl Butyrate")
    assert "nothing paired" not in line.lower()
    assert "light" in line.lower()


def test_protocol_line_names_the_odor_either_way():
    for ds in ("EB-Control-24-1", "EB-Training-24-1"):
        assert "Ethyl Butyrate" in mod.protocol_line(ds, "Ethyl Butyrate")


@pytest.mark.parametrize("ds", TRAINED)
def test_no_trained_caption_calls_its_flies_controls(ds):
    assert "control" not in mod.cohort_noun(ds).lower()
    assert "control" not in mod.protocol_line(ds, "Hexanol").lower()


# ── the wiring ────────────────────────────────────────────────────────────


def test_pipeline_sends_both_arms_to_the_auc_set():
    import yaml
    import scripts.pipeline.run_workflows as rw

    path = Path(rw.REPO_ROOT) / "config" / "config_new.yaml"
    if not path.is_file():
        pytest.skip("config_new.yaml not present")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    from fbpipe.config import load_settings

    cmds = rw._cohort_figure_commands(
        data.get("analysis"), load_settings(path),
        python_exec=sys.executable, config_path=path,
    )
    auc = [c for c in cmds
           if any(str(p).endswith("training_auc_vs_control_response.py") for p in c)]
    named = {c[i + 1] for c in auc for i, t in enumerate(c) if t == "--dataset"}
    assert set(CONTROLS) <= named
    assert set(TRAINED) <= named


def test_pipeline_makes_graded_rasters_for_the_trained_arms():
    import yaml
    import scripts.pipeline.run_workflows as rw
    from fbpipe.config import load_settings

    path = Path(rw.REPO_ROOT) / "config" / "config_new.yaml"
    if not path.is_file():
        pytest.skip("config_new.yaml not present")
    cmds = rw._cohort_figure_commands(
        (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("analysis"),
        load_settings(path), python_exec=sys.executable, config_path=path,
    )
    graded = [c for c in cmds
              if "--mode" in c and c[c.index("--mode") + 1] == "graded"]
    named = {c[i + 1] for c in graded for i, t in enumerate(c) if t == "--dataset"}
    assert set(TRAINED) <= named


def test_trained_rasters_now_sort_like_the_controls():
    import yaml
    import scripts.pipeline.run_workflows as rw
    from fbpipe.config import load_settings

    path = Path(rw.REPO_ROOT) / "config" / "config_new.yaml"
    if not path.is_file():
        pytest.skip("config_new.yaml not present")
    cmds = rw._cohort_figure_commands(
        (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("analysis"),
        load_settings(path), python_exec=sys.executable, config_path=path,
    )
    for cmd in cmds:
        if "--sort-by" in cmd:
            assert cmd[cmd.index("--sort-by") + 1] == "ratio"
