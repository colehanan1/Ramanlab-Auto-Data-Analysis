"""The three cohort figure sets run on every pipeline run.

``Binarized-PER-Rasters``, ``Graded-PER-Rasters`` and
``Training-AUC-vs-Testing-Response`` were hand-run one-offs: they existed on disk
for whichever cohorts someone had last typed a command for, at whatever threshold
that command carried. Two of the three disagreed about which cohorts they
covered. Wiring them into ``run_workflows`` makes the set reproducible and ties
their threshold to the pipeline's.

All three read the wide envelope tables plus ``model_predictions.csv``, so they
run after ``combined`` and after the reactions step, alongside
``naive_vs_trained`` -- and like it, they are skipped when no predictions CSV
exists rather than rendering a figure from nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.pipeline.run_workflows as rw


DATASETS = [
    "EB-Control-24-1",
    "3Oct-Control-24-0.1",
    "Hex-Control-24-0.01",
    "Hex-Control-24-0.1",
]


class _ReactionPrediction:
    python = ""
    output_csv = "/tmp/preds/model_predictions.csv"


class _Settings:
    reaction_prediction = _ReactionPrediction()


def cfg_block(**over):
    block = {
        "enabled": True,
        "datasets": list(DATASETS),
        "training_wide_csv": "/tmp/csv/wide_training.parquet",
        "testing_wide_csv": "/tmp/csv/wide.parquet",
        "binarized_out_dir": "/tmp/fig/Binarized-PER-Rasters",
        "graded_out_dir": "/tmp/fig/Graded-PER-Rasters",
        "training_auc_out_dir": "/tmp/fig/Training-AUC-vs-Testing-Response",
    }
    block.update(over)
    return {"cohort_figures": block}


def commands(analysis_cfg, config_path=Path("/tmp/config_new.yaml")):
    return rw._cohort_figure_commands(
        analysis_cfg, _Settings(), python_exec="/usr/bin/python3",
        config_path=config_path,
    )


def by_script(cmds, name):
    hits = [c for c in cmds if any(str(p).endswith(name) for p in c)]
    assert len(hits) == 1, f"expected exactly one {name} command, got {len(hits)}"
    return hits[0]


def flag_values(cmd, flag):
    return [cmd[i + 1] for i, tok in enumerate(cmd) if tok == flag]


# ── what gets built ───────────────────────────────────────────────────────


def test_all_three_figure_sets_are_commanded():
    cmds = commands(cfg_block())
    assert len(cmds) == 3
    # The raster script appears twice (binary + graded); the AUC script once.
    scripts = [Path(c[1]).name for c in cmds]
    assert sorted(scripts) == [
        "binarized_per_rasters.py",
        "binarized_per_rasters.py",
        "training_auc_vs_control_response.py",
    ]


def test_binary_and_graded_are_separate_runs_of_the_raster_script():
    cmds = commands(cfg_block())
    raster = [c for c in cmds if any(str(p).endswith("binarized_per_rasters.py") for p in c)]
    assert len(raster) == 2
    modes = sorted(flag_values(c, "--mode")[0] for c in raster)
    assert modes == ["binary", "graded"]


def test_each_mode_writes_to_its_own_directory():
    cmds = commands(cfg_block())
    out = {
        flag_values(c, "--mode")[0]: flag_values(c, "--out-dir")[0]
        for c in cmds
        if any(str(p).endswith("binarized_per_rasters.py") for p in c)
    }
    assert out["binary"].endswith("Binarized-PER-Rasters")
    assert out["graded"].endswith("Graded-PER-Rasters")


def test_every_configured_cohort_is_passed_to_every_set():
    """The three sets disagreed about their cohort list before this existed."""
    for cmd in commands(cfg_block()):
        assert flag_values(cmd, "--dataset") == DATASETS


def test_training_auc_gets_both_wide_tables_and_the_predictions():
    cmd = by_script(commands(cfg_block()), "training_auc_vs_control_response.py")
    assert flag_values(cmd, "--training-wide-csv") == ["/tmp/csv/wide_training.parquet"]
    assert flag_values(cmd, "--testing-wide-csv") == ["/tmp/csv/wide.parquet"]
    assert flag_values(cmd, "--predictions-csv") == ["/tmp/preds/model_predictions.csv"]


def test_predictions_csv_defaults_to_the_reaction_step_output():
    cmd = by_script(commands(cfg_block()), "training_auc_vs_control_response.py")
    assert flag_values(cmd, "--predictions-csv") == [_ReactionPrediction.output_csv]


# ── the threshold tie-in ──────────────────────────────────────────────────


def test_rasters_are_handed_the_pipeline_config():
    """This is what makes them binarise at the pipeline's theta: the raster
    script resolves its ThresholdRule out of this file's wide block."""
    cmds = commands(cfg_block(), config_path=Path("/tmp/config_new.yaml"))
    for cmd in cmds:
        if any(str(p).endswith("binarized_per_rasters.py") for p in cmd):
            assert flag_values(cmd, "--config") == ["/tmp/config_new.yaml"]


def test_no_threshold_overrides_are_passed():
    """Passing -k here would re-introduce the drift: the rule must come from
    config alone, so retuning it in one place moves the rasters too."""
    for cmd in commands(cfg_block()):
        for flag in ("-k", "--threshold-k", "--threshold-min-delta",
                     "--threshold-anchor-s"):
            assert flag not in cmd


# ── when it must not run ──────────────────────────────────────────────────


def test_absent_block_produces_nothing():
    assert commands({}) == []
    assert commands(None) == []


def test_disabled_block_produces_nothing():
    assert commands(cfg_block(enabled=False)) == []


def test_empty_dataset_list_produces_nothing():
    assert commands(cfg_block(datasets=[])) == []


def test_missing_out_dir_drops_only_that_set():
    """A cohort figure set with nowhere to write is skipped; the others run."""
    cmds = commands(cfg_block(graded_out_dir=""))
    assert len(cmds) == 2
    assert not any(flag_values(c, "--mode") == ["graded"] for c in cmds)


# ── cache key ─────────────────────────────────────────────────────────────


def test_cache_key_tracks_the_dataset_list():
    a = rw._cohort_figures_expected(cfg_block(), _Settings(), config_path=None)
    b = rw._cohort_figures_expected(
        cfg_block(datasets=DATASETS[:2]), _Settings(), config_path=None
    )
    assert a != b


def test_cache_key_tracks_the_output_directories():
    a = rw._cohort_figures_expected(cfg_block(), _Settings(), config_path=None)
    b = rw._cohort_figures_expected(
        cfg_block(graded_out_dir="/tmp/other"), _Settings(), config_path=None
    )
    assert a != b


def test_cache_key_is_stable_for_an_unchanged_config():
    a = rw._cohort_figures_expected(cfg_block(), _Settings(), config_path=None)
    b = rw._cohort_figures_expected(cfg_block(), _Settings(), config_path=None)
    assert a == b


# ── force flag + figures-only ─────────────────────────────────────────────


def test_force_settings_carries_the_new_step():
    from fbpipe.config import ForceSettings

    assert hasattr(ForceSettings(), "cohort_figures")


def test_config_new_declares_the_block():
    """The operative config must actually turn this on, or 'every run' is a lie."""
    import yaml

    path = Path(rw.REPO_ROOT) / "config" / "config_new.yaml"
    if not path.is_file():
        pytest.skip("config_new.yaml not present")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    block = (data.get("analysis") or {}).get("cohort_figures")
    assert block, "analysis.cohort_figures missing from config_new.yaml"
    assert block.get("enabled", True) is True
    assert block.get("datasets"), "no cohorts configured"


# ── the trained arms, unranked ────────────────────────────────────────────


def trained_block(**over):
    block = dict(cfg_block()["cohort_figures"])
    block["trained_raster_datasets"] = [
        "EB-Training-24-1",
        "3Oct-Training-24-0.1",
        "Hex-Training-24-0.01",
        "Hex-Training-24-0.1",
    ]
    block.update(over)
    return {"cohort_figures": block}


def raster_cmds(analysis_cfg):
    return [
        c for c in commands(analysis_cfg)
        if any(str(p).endswith("binarized_per_rasters.py") for p in c)
    ]


def test_trained_arms_add_a_full_second_set():
    """Both arms get all three figure sets: 3 + 3."""
    assert len(commands(trained_block())) == 6


def test_trained_rasters_sort_like_the_controls():
    """Superseded both the earlier random order and the first-exposure ratio:
    the arms are meant to be read side by side, which a different sort on each
    would defeat."""
    trained = [c for c in raster_cmds(trained_block()) if "EB-Training-24-1" in c]
    assert len(trained) == 2                      # binary + graded
    for cmd in trained:
        assert flag_values(cmd, "--sort-by") == ["ratio"]


def test_control_rasters_use_the_mean_ratio():
    for cmd in raster_cmds(trained_block()):
        if "EB-Training-24-1" in cmd:
            continue
        assert flag_values(cmd, "--sort-by") == ["ratio"]


def test_trained_raster_covers_every_trained_cohort():
    trained = [c for c in raster_cmds(trained_block())
               if "EB-Training-24-1" in c]
    assert len(trained) == 2
    assert flag_values(trained[0], "--dataset") == [
        "EB-Training-24-1", "3Oct-Training-24-0.1",
        "Hex-Training-24-0.01", "Hex-Training-24-0.1",
    ]


def test_trained_raster_writes_into_the_binarized_dir():
    trained = [c for c in raster_cmds(trained_block())
               if "EB-Training-24-1" in c][0]
    assert flag_values(trained, "--out-dir")[0].endswith("Binarized-PER-Rasters")


def test_trained_raster_still_gets_the_pipeline_config():
    trained = [c for c in raster_cmds(trained_block())
               if "EB-Training-24-1" in c][0]
    assert flag_values(trained, "--config") == ["/tmp/config_new.yaml"]


def test_no_trained_datasets_changes_nothing():
    assert len(commands(trained_block(trained_raster_datasets=[]))) == 3
    assert len(commands(cfg_block())) == 3


def test_trained_arms_get_a_graded_raster_too():
    graded = [c for c in raster_cmds(trained_block())
              if flag_values(c, "--mode") == ["graded"]]
    named = {c[i + 1] for c in graded for i, t in enumerate(c) if t == "--dataset"}
    assert "EB-Training-24-1" in named


def test_both_arms_reach_the_training_auc_set():
    """It used to be control-only. The analysis transfers -- "does conditioning
    vigor predict the test response" is well posed for a trained arm too -- and
    that module's captions follow the arm, so no trained figure claims its odor
    was presented unpaired."""
    auc = [c for c in commands(trained_block())
           if any(str(p).endswith("training_auc_vs_control_response.py") for p in c)]
    assert len(auc) == 2
    named = {c[i + 1] for c in auc for i, t in enumerate(c) if t == "--dataset"}
    assert set(DATASETS) <= named
    assert "EB-Training-24-1" in named


def test_config_new_declares_the_trained_arms():
    import yaml

    path = Path(rw.REPO_ROOT) / "config" / "config_new.yaml"
    if not path.is_file():
        pytest.skip("config_new.yaml not present")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    block = (data.get("analysis") or {}).get("cohort_figures") or {}
    assert block.get("trained_raster_datasets"), "no trained arms configured"



# ── ratio sorting reaches the pipeline ────────────────────────────────────


def test_control_rasters_sort_by_ratio():
    """Fold-change over each fly's own baseline, averaged across conditioning
    trials -- not bare AUC-During, and not the odor-fraction default."""
    for cmd in raster_cmds(cfg_block()):
        assert flag_values(cmd, "--sort-by") == ["ratio"]


def test_every_raster_command_sorts_by_ratio():
    for cmd in raster_cmds(trained_block()):
        assert flag_values(cmd, "--sort-by") == ["ratio"]


def test_trained_rasters_are_no_longer_random():
    for cmd in raster_cmds(trained_block()):
        assert flag_values(cmd, "--sort-by") not in (["random"], ["ratio_first"])


def test_sort_keys_the_pipeline_asks_for_actually_exist():
    """Guards a typo that argparse would reject only at figure time, after the
    rest of the run had already succeeded."""
    import matplotlib
    matplotlib.use("Agg")
    from scripts.analysis import binarized_per_rasters as bpr

    for cmd in raster_cmds(trained_block()):
        for key in flag_values(cmd, "--sort-by"):
            assert key in bpr.SORT_KEYS, key
