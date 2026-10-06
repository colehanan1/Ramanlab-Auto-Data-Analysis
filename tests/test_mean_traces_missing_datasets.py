"""dataset_mean_traces must not build commands for datasets not on disk.

``EB-Sensitivity-24-1`` is listed in ``datasets`` and in
``analysis.dataset_mean_traces.phase_cohorts`` before its folder exists. The
config loader already drops it from every root list ("[config] Skipping
datasets not yet on disk"), but the mean-trace step still built its command,
which then exited 1 every night with "No usable post-training traces". A
dataset counts as on disk exactly as in ``fbpipe.config._expand_datasets``:
its folder exists under ``dataset_bases.data`` OR ``dataset_bases.secured``.

The failing command wrote nothing before it raised, so dropping it changes no
figure -- only the nightly warning goes away.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.pipeline import run_workflows as rw

ON_DATA = "Hex-Sensitivity-24-0.1"      # local data folder only
ON_SECURED = "IAA-Sensitivity-24-1"     # secured mirror only
MISSING = "EB-Sensitivity-24-1"         # neither


class _Settings:
    flagged_flies_csv = ""
    protocol = "v2"
    fps_default = 40.0
    odor_on_s = 30.0
    odor_off_s = 60.0
    dataset_overrides: dict = {}

    def __init__(self, datasets=()):
        self.datasets = tuple(datasets)


def _config(tmp_path: Path, datasets) -> Path:
    data, secured = tmp_path / "data", tmp_path / "secured"
    (data / ON_DATA).mkdir(parents=True)
    (secured / ON_SECURED).mkdir(parents=True)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "dataset_bases": {"data": str(data), "secured": str(secured)},
        "datasets": list(datasets),
    }))
    return path


def _analysis(**over):
    block = {
        "wide_csv": "/x/testing.parquet",
        "out_root": "/x/dataset_mean_comparisons",
        "pretest_wide_csv": "/x/pretest.parquet",
        "phase_cohorts": [ON_DATA, ON_SECURED, MISSING],
    }
    block.update(over)
    return {"dataset_mean_traces": block}


def _phase_datasets(cmds):
    return [c[c.index("--train-dataset") + 1] for c in cmds if "--pretest-wide-csv" in c]


def test_phase_cohort_not_on_disk_gets_no_command(tmp_path, capsys):
    cfg = _config(tmp_path, [ON_DATA, ON_SECURED, MISSING])
    cmds = rw._dataset_mean_traces_commands(
        _analysis(), _Settings(), python_exec="python3", config_path=cfg
    )
    assert _phase_datasets(cmds) == [ON_DATA, ON_SECURED]
    out = capsys.readouterr().out
    assert "not yet on disk" in out and MISSING in out


def test_secured_only_dataset_counts_as_on_disk(tmp_path):
    """Same rule as the config loader: data OR secured."""
    cfg = _config(tmp_path, [ON_SECURED])
    cmds = rw._dataset_mean_traces_commands(
        _analysis(phase_cohorts=[ON_SECURED]), _Settings(),
        python_exec="python3", config_path=cfg,
    )
    assert _phase_datasets(cmds) == [ON_SECURED]


def test_pair_with_an_arm_not_on_disk_gets_no_command(tmp_path):
    """Half a pair cannot draw the figure; the command could only exit 1."""
    data = tmp_path / "data"
    for name in ("EB-Training-24-1", "EB-Control-24-1", "Hex-Training-24-0.1"):
        (data / name).mkdir(parents=True)
    names = ["EB-Training-24-1", "EB-Control-24-1",
             "Hex-Training-24-0.1", "Hex-Control-24-0.1"]  # last one absent
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({
        "dataset_bases": {"data": str(data), "secured": str(tmp_path / "secured")},
        "datasets": names,
    }))
    cmds = rw._dataset_mean_traces_commands(
        _analysis(phase_cohorts=[]), _Settings(names),
        python_exec="python3", config_path=cfg,
    )
    trained = [c[c.index("--train-dataset") + 1] for c in cmds]
    assert trained == ["EB-Training-24-1"]


def test_explicit_cohort_with_an_arm_not_on_disk_gets_no_command(tmp_path):
    cfg = _config(tmp_path, [ON_DATA])
    block = _analysis(phase_cohorts=[], cohorts=[
        {"train_dataset": ON_DATA, "control_dataset": MISSING, "out_dir": "x"},
    ])
    assert rw._dataset_mean_traces_commands(
        block, _Settings(), python_exec="python3", config_path=cfg
    ) == []


def test_no_config_means_no_disk_filter():
    """Without a config there are no bases to check: keep every command, as
    before (fail open, like the config loader's own undated-batch stance)."""
    cmds = rw._dataset_mean_traces_commands(
        _analysis(), _Settings(), python_exec="python3", config_path=None
    )
    assert _phase_datasets(cmds) == [ON_DATA, ON_SECURED, MISSING]


def test_unreadable_config_means_no_disk_filter(tmp_path):
    cmds = rw._dataset_mean_traces_commands(
        _analysis(), _Settings(), python_exec="python3",
        config_path=tmp_path / "absent.yaml",
    )
    assert _phase_datasets(cmds) == [ON_DATA, ON_SECURED, MISSING]


def test_the_cache_key_follows_the_disk(tmp_path):
    """When the folder lands, the command list -- and so the cache key --
    changes, and the step re-runs with the new cohort."""
    cfg = _config(tmp_path, [ON_DATA, MISSING])
    before = rw._dataset_mean_traces_expected(
        _analysis(phase_cohorts=[ON_DATA, MISSING]), _Settings(), config_path=cfg
    )
    (tmp_path / "data" / MISSING).mkdir()
    after = rw._dataset_mean_traces_expected(
        _analysis(phase_cohorts=[ON_DATA, MISSING]), _Settings(), config_path=cfg
    )
    assert before["commands"] != after["commands"]
    assert len(after["commands"]) == len(before["commands"]) + 1
