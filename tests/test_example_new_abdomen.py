"""The tracked template ``config/example_new_abdomen.yaml`` is a working abdomen/mounting overlay.

``config/*`` is gitignored, so the live ``config/config_new_abdomen.yaml`` exists only on disk: the user makes it with
``cp config/example_new_abdomen.yaml config/config_new_abdomen.yaml``. What git carries is the template, so the template
is what has to be right. Pinned here against the REAL parsers and the driver's own command builder (nothing runs, no
file outside ``tmp_path`` is written):

* it is ``extends: config_new.yaml`` plus the ``abdomen`` and ``mounting`` blocks, both switched on (the mounting
  analysis reads the abdomen tree, so it always needs the abdomen block), repeats no key (YAML keeps the last of a
  repeated key, so a repeated block would silently replace the first), and resolves as an overlay of a base config
  that sits beside it, leaving what PER readers see unchanged;
* both blocks parse -- the parsers are strict allow-lists, so a misspelt key or a bad value in the template fails here
  and not on the night -- the mounting block takes the abdomen block's ``batch_date_from`` as its cut-off, and its
  numbers are ``MountingSettings``' defaults (the dataclass documents them as the template's values);
* the nightly writes tables only (``annotated_video: false``, spelled out); the abdomen tree is its own and never lies
  inside the read-only production or secured trees; the six output directories are exactly the designed ones and every
  figure directory is below ``Results/New-Opto-Fly-Figures``;
* the driver builds all four stage commands from it. It does so silently: a mounting directory it refuses is only a
  warning, so a template it refused would cost the mounting analysis without a trace;
* ``.gitignore`` lets the template through while the live overlay, which holds real paths, stays out of git.
"""
from __future__ import annotations

import dataclasses
import logging
import shutil
import subprocess
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from fbpipe.analysis.mounting import MountingSettings, parse_mounting_block
from fbpipe.config import load_settings
from fbpipe.config_layers import OVERLAY_KEYS, resolve_layers
from fbpipe.geometry.config import SECURED_MOUNT, assert_out_tree_safe, parse_abdomen_block
from scripts.pipeline import run_workflows as rw

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "config" / "example_new_abdomen.yaml"
TEMPLATE_GIT_PATH = "config/example_new_abdomen.yaml"
LIVE_OVERLAY_GIT_PATH = "config/config_new_abdomen.yaml"

COLE = "/home/ramanlab/Documents/cole"
NEW_OPTO = f"{COLE}/Results/New-Opto-Fly-Figures"
# The six directories the abdomen and mounting stages write -- nothing else.
OUT_BASE = f"{COLE}/Data/flys_Abdomen"
GEOMETRY_TABLES = f"{COLE}/Data/CSVs-New-Opto-Flys/Abdomen-Geometry-New"
MOUNTING_TABLES = f"{COLE}/Data/CSVs-New-Opto-Flys/Mounting-Posture"
GEOMETRY_FIGURES = f"{NEW_OPTO}/Abdomen-Geometry-New"
MOUNTING_FIGURES = f"{NEW_OPTO}/Mounting-Posture"
ABDOMEN_FIGURES = f"{NEW_OPTO}/Abdomen-Movement"
SECURED_BASE = "/securedstorage/DATAsec/cole/Data-secured-New"
# What load_geometry_config refuses as an output tree: the production batches, the secured base and the secured mount.
READ_ONLY = [Path(f"{COLE}/Data/flys_New"), Path(SECURED_BASE), Path(SECURED_MOUNT)]
CUTOFF = date(2026, 10, 6)


def _repeated_keys(node, path=""):
    """Every key set twice within one mapping of a composed YAML tree, as ``block.key (line N)``."""
    found = []
    if isinstance(node, yaml.MappingNode):
        seen = set()
        for key_node, value_node in node.value:
            name = str(key_node.value)
            if name in seen:
                found.append(f"{path}{name} (line {key_node.start_mark.line + 1})")
            seen.add(name)
            found += _repeated_keys(value_node, f"{path}{name}.")
    elif isinstance(node, yaml.SequenceNode):
        for item in node.value:
            found += _repeated_keys(item, path)
    return found


@pytest.fixture
def template():
    return yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))


@pytest.fixture
def rig(tmp_path):
    """The template copied next to a base config, as ``cp`` leaves the two in ``config/``."""
    cfg = tmp_path / "config"
    cfg.mkdir()
    base = cfg / "config_new.yaml"
    base.write_text(yaml.safe_dump({
        "protocol": "v2",
        "model_path": "/x/per_model.engine",                     # the PER model: it must not become the abdomen model
        "datasets": [],
        "yolo": {"conf_thres": 0.4, "inference_batch_size": 16},
        "reaction_prediction": {"output_csv": str(tmp_path / "model_predictions.csv")},
    }, sort_keys=False), encoding="utf-8")
    overlay = cfg / "config_new_abdomen.yaml"
    shutil.copyfile(TEMPLATE, overlay)
    return SimpleNamespace(base=base, overlay=overlay, settings=load_settings(base))


# ---- the overlay ------------------------------------------------------------------------------------------

def test_the_template_extends_config_new_and_sets_nothing_else_but_the_overlay_blocks(template):
    assert template["extends"] == "config_new.yaml"
    assert set(template) <= OVERLAY_KEYS
    assert {"extends", "abdomen", "mounting"} <= set(template)


def test_no_key_is_set_twice_anywhere_in_the_template():
    """The parsers see the merged mapping, so a repeated key would pass every other test here."""
    tree = yaml.compose(TEMPLATE.read_text(encoding="utf-8"), Loader=yaml.SafeLoader)      # nodes only, nothing built
    assert _repeated_keys(tree) == []


def test_both_blocks_are_switched_on_because_the_mounting_analysis_needs_the_abdomen_tree(template):
    assert template["abdomen"]["enabled"] is True
    assert template["mounting"]["enabled"] is True


def test_copied_next_to_a_base_config_the_template_resolves_as_an_overlay_of_it(rig, template):
    layers = resolve_layers(rig.overlay)
    assert layers.is_layered and layers.base_path == rig.base.resolve()
    assert dict(layers.overlay) == {"abdomen": template["abdomen"], "mounting": template["mounting"]}
    assert load_settings(rig.overlay) == rig.settings            # what a PER reader sees is the base, untouched


# ---- the blocks parse with the real parsers ---------------------------------------------------------------

def test_the_abdomen_block_parses_into_the_designed_settings(rig, template):
    settings, gcfg = parse_abdomen_block(template["abdomen"], rig.settings)
    assert settings.model_path == f"{COLE}/Workflow&Code/runs/obb/train/weights/best.pt"
    assert (settings.conf_thres, settings.inference_batch_size, settings.engine_supports_batch,
            settings.max_flies, settings.yolo_num_workers) == (0.25, 8, True, 5, 3)
    assert gcfg.out_base == OUT_BASE
    assert gcfg.video_bases == (SECURED_BASE,)
    assert gcfg.batch_date_from == CUTOFF
    assert gcfg.datasets == () and gcfg.slot_seeds == "production"
    assert gcfg.table_format == "parquet"
    assert (gcfg.wide_out_dir, gcfg.figures_out_dir) == (GEOMETRY_TABLES, GEOMETRY_FIGURES)


def test_the_nightly_writes_tables_only_so_the_annotated_video_is_off_and_spelled_out(rig, template):
    assert template["abdomen"]["geometry"]["annotated_video"] is False      # not left to the parser's default
    _, gcfg = parse_abdomen_block(template["abdomen"], rig.settings)
    assert gcfg.annotated_video is False


def test_the_mounting_block_parses_and_takes_the_abdomen_cutoff(rig, template):
    _, gcfg = parse_abdomen_block(template["abdomen"], rig.settings)
    mcfg = parse_mounting_block(template["mounting"], default_batch_date_from=gcfg.batch_date_from)
    assert mcfg.enabled is True
    assert mcfg.batch_date_from == CUTOFF
    assert (mcfg.csv_out_dir, mcfg.out_dir, mcfg.abdomen_out_dir) == (MOUNTING_TABLES, MOUNTING_FIGURES, ABDOMEN_FIGURES)


def test_the_mounting_numbers_are_the_defaults_the_settings_class_documents(template):
    """``MountingSettings`` calls its defaults "the value the overlay template ships with": one set of numbers."""
    parsed = parse_mounting_block(template["mounting"], default_batch_date_from=CUTOFF)
    designed = dataclasses.replace(
        MountingSettings(), csv_out_dir=MOUNTING_TABLES, out_dir=MOUNTING_FIGURES, abdomen_out_dir=ABDOMEN_FIGURES,
        batch_date_from=CUTOFF)
    assert parsed == designed


# ---- where it writes --------------------------------------------------------------------------------------

def test_the_abdomen_output_tree_is_its_own_and_not_inside_a_read_only_root(rig, template):
    _, gcfg = parse_abdomen_block(template["abdomen"], rig.settings)
    assert_out_tree_safe(Path(gcfg.out_base), READ_ONLY)
    with pytest.raises(ValueError, match="only reads"):                       # the roots listed above are really enforced
        assert_out_tree_safe(READ_ONLY[0] / "flys_Abdomen", READ_ONLY)


def test_the_six_output_directories_are_exactly_the_designed_ones(template):
    abdomen, mounting = template["abdomen"], template["mounting"]
    written = [abdomen["out_base"], abdomen["geometry"]["wide_out_dir"], abdomen["geometry"]["figures_out_dir"],
               mounting["csv_out_dir"], mounting["out_dir"], mounting["abdomen_out_dir"]]
    assert sorted(written) == sorted([OUT_BASE, GEOMETRY_TABLES, GEOMETRY_FIGURES,
                                      MOUNTING_TABLES, MOUNTING_FIGURES, ABDOMEN_FIGURES])


def test_every_figure_directory_is_below_the_new_opto_figures_root(template):
    figures = [template["abdomen"]["geometry"]["figures_out_dir"], template["mounting"]["out_dir"],
               template["mounting"]["abdomen_out_dir"]]
    assert len(figures) == 3 and all(path.startswith(f"{NEW_OPTO}/") for path in figures)


# ---- the driver accepts it --------------------------------------------------------------------------------

def test_the_driver_builds_all_four_stage_commands_from_the_template_without_refusing_a_directory(rig, caplog):
    overlay = resolve_layers(rig.overlay).overlay
    kwargs = dict(python_exec="python3", overlay_path=rig.overlay, annotated_video=False, n_workers=3)
    with caplog.at_level(logging.WARNING):
        full = rw._abdomen_commands(overlay, rig.settings, figures_only=False, **kwargs)
        restyle = rw._abdomen_commands(overlay, rig.settings, figures_only=True, **kwargs)
    labels = [rw._abdomen_command_label(cmd) for cmd in full]
    assert labels == ["track", "normalize+wide+figures", "abdomen_geometry_stats", "mounting_posture"], caplog.messages
    assert [rw._abdomen_command_label(cmd) for cmd in restyle] == labels[1:]  # --figures-only: no tracking, still the rest
    mounting = full[-1]
    flags = {flag: mounting[mounting.index(flag) + 1] for flag in ("--out-dir", "--abdomen-out-dir", "--csv-out-dir")}
    assert flags == {"--out-dir": MOUNTING_FIGURES, "--abdomen-out-dir": ABDOMEN_FIGURES,
                     "--csv-out-dir": MOUNTING_TABLES}
    assert mounting[mounting.index("--predictions-csv") + 1] == rig.settings.reaction_prediction.output_csv
    assert caplog.messages == []


# ---- what git carries -------------------------------------------------------------------------------------

def _git_ignores(path: str) -> bool:
    """Whether the ignore rules match *path* (relative to the repo), whether or not the file is tracked or exists."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    probe = subprocess.run(["git", "check-ignore", "--no-index", "-q", "--", path], cwd=ROOT, capture_output=True)
    if probe.returncode not in (0, 1):                  # 128: not a git work tree (an exported source tree)
        pytest.skip("not inside a git work tree")
    return probe.returncode == 0


def test_gitignore_lets_the_template_through_the_config_wildcard():
    assert not _git_ignores(TEMPLATE_GIT_PATH)


def test_gitignore_keeps_the_live_overlay_with_its_real_paths_out_of_git():
    assert _git_ignores(LIVE_OVERLAY_GIT_PATH)
