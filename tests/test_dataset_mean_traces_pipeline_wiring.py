"""Per-odor trained-vs-control mean traces must be part of a pipeline run.

``dataset_mean_traces_tvc.py`` was hand-run only, so the figure sets under
``Results/Figures/<stem>_mean_traces_new/`` went stale whenever a pipeline run
rewrote the wide envelope table underneath them (two of the four cohort dirs
were still empty). ``analysis.dataset_mean_traces`` in the config names the
output tree and ``_dataset_mean_traces_commands`` turns each trained/control
dataset pair into one command.

Command construction is tested directly rather than through ``_run_pipeline``,
which needs a whole dataset tree before it will run -- the same reason
``_pubfig_commands`` is a pure function (see test_pubfig_pipeline_wiring.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fbpipe.config import load_settings  # noqa: E402
from scripts.pipeline import run_workflows as rw  # noqa: E402

CONFIG_PATH = ROOT / "config" / "config_new.yaml"


class _Settings:
    """Only the attributes ``_dataset_mean_traces_commands`` reads."""

    def __init__(self, **kw):
        self.flagged_flies_csv = kw.get("flagged_flies_csv", "")
        self.protocol = kw.get("protocol", "v2")
        self.fps_default = kw.get("fps_default", 40.0)
        self.odor_on_s = kw.get("odor_on_s", 30.0)
        self.odor_off_s = kw.get("odor_off_s", 60.0)
        self.datasets = kw.get(
            "datasets",
            [
                "3Oct-Training-24-0.1",
                "3Oct-Control-24-0.1",
                "EB-Training-24-1",
                "EB-Control-24-1",
            ],
        )


def _block(**kw):
    block = {
        "wide_csv": "/data/wide.parquet",
        "out_root": "/figs",
    }
    block.update(kw)
    return {"dataset_mean_traces": block}


def _cmds(analysis_cfg, settings=None):
    return rw._dataset_mean_traces_commands(
        analysis_cfg,
        settings or _Settings(),
        python_exec="/py",
        config_path=Path("/cfg.yaml"),
    )


def _value(cmd, flag):
    return cmd[cmd.index(flag) + 1]


# ---------------------------------------------------------------------------
# Which dataset pairs get a figure set
# ---------------------------------------------------------------------------


def test_training_and_control_arms_are_paired_by_name():
    pairs = rw._mean_trace_cohorts(
        ["3Oct-Training-24-0.1", "3Oct-Control-24-0.1"]
    )
    assert pairs == [("3Oct-Training-24-0.1", "3Oct-Control-24-0.1", "3Oct-24-0.1")]


def test_the_stem_drops_the_arm_token_matching_the_published_dirs():
    """The published sets are ``EB-24-1_mean_traces_new``, not ``EB-Training-``."""
    pairs = rw._mean_trace_cohorts(["EB-Training-24-1", "EB-Control-24-1"])
    assert pairs[0][2] == "EB-24-1"


def test_a_training_dataset_without_its_control_is_skipped():
    """Half a pair cannot draw a trained-vs-control figure."""
    assert rw._mean_trace_cohorts(["Hex-Training-48-0.1", "EB-Control-24-1"]) == []


def test_suffixed_variants_do_not_borrow_the_plain_control():
    """``-Manual`` is its own dataset; pairing it with the plain control would
    silently mix two experiments into one figure."""
    pairs = rw._mean_trace_cohorts(
        ["3Oct-Training-24-0.1-Manual", "3Oct-Control-24-0.1"]
    )
    assert pairs == []


def test_pairs_keep_config_order_and_do_not_repeat():
    pairs = rw._mean_trace_cohorts(
        [
            "EB-Training-24-1",
            "3Oct-Training-24-0.1",
            "EB-Control-24-1",
            "3Oct-Control-24-0.1",
            "EB-Training-24-1",
        ]
    )
    assert [p[2] for p in pairs] == ["EB-24-1", "3Oct-24-0.1"]


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


def test_no_block_means_no_commands():
    """An unconfigured pipeline must not start rendering figures."""
    assert _cmds({}) == []
    assert _cmds({"dataset_mean_traces": {}}) == []


def test_a_missing_out_root_or_wide_csv_yields_no_commands():
    assert _cmds({"dataset_mean_traces": {"wide_csv": "/w.parquet"}}) == []
    assert _cmds({"dataset_mean_traces": {"out_root": "/figs"}}) == []


def test_enabled_false_turns_the_step_off():
    assert _cmds(_block(enabled=False)) == []


def test_one_command_per_dataset_pair():
    cmds = _cmds(_block())
    assert len(cmds) == 2
    assert [_value(c, "--train-dataset") for c in cmds] == [
        "3Oct-Training-24-0.1",
        "EB-Training-24-1",
    ]


def test_each_pair_writes_a_directory_named_for_the_cohort():
    cmd = _cmds(_block())[0]
    assert _value(cmd, "--out-dir") == "/figs/3Oct-24-0.1"


def test_the_directory_suffix_is_configurable():
    """The published one-off sets carry a `_mean_traces_new` tail; the pipeline
    tree does not, so the suffix has to be an opt-in."""
    cmd = _cmds(_block(dir_suffix="_mean_traces_new"))[0]
    assert _value(cmd, "--out-dir") == "/figs/3Oct-24-0.1_mean_traces_new"


def test_command_carries_every_figure_parameter():
    cmd = _cmds(
        _block(),
        _Settings(flagged_flies_csv="/truth.csv", protocol="v2"),
    )[0]
    assert cmd[0] == "/py"
    assert cmd[1].endswith("dataset_mean_traces_tvc.py")
    for flag, value in (
        ("--wide-csv", "/data/wide.parquet"),
        ("--train-dataset", "3Oct-Training-24-0.1"),
        ("--control-dataset", "3Oct-Control-24-0.1"),
        ("--flagged-flies-csv", "/truth.csv"),
        ("--config", "/cfg.yaml"),
        ("--protocol", "v2"),
        ("--fps", "40.0"),
        ("--odor-on-s", "30.0"),
        ("--odor-off-s", "60.0"),
    ):
        assert flag in cmd, f"{flag} missing"
        assert _value(cmd, flag) == value, flag


def test_timing_falls_back_to_the_top_level_config_values():
    """One source of truth for fps / odor window, not a second copy that drifts."""
    cmd = _cmds(
        _block(),
        _Settings(fps_default=30.0, odor_on_s=20.0, odor_off_s=50.0),
    )[0]
    assert (_value(cmd, "--fps"), _value(cmd, "--odor-on-s"), _value(cmd, "--odor-off-s")) == (
        "30.0",
        "20.0",
        "50.0",
    )


def test_block_level_timing_overrides_the_top_level():
    cmd = _cmds(_block(fps=25.0, odor_on_s=10.0, odor_off_s=40.0))[0]
    assert _value(cmd, "--fps") == "25.0"
    assert _value(cmd, "--odor-on-s") == "10.0"
    assert _value(cmd, "--odor-off-s") == "40.0"


def test_flagged_csv_is_omitted_when_unset():
    assert "--flagged-flies-csv" not in _cmds(_block())[0]


def test_block_flagged_csv_wins_over_the_global_one():
    cmd = _cmds(_block(flagged_csv="/block.csv"), _Settings(flagged_flies_csv="/g.csv"))[0]
    assert _value(cmd, "--flagged-flies-csv") == "/block.csv"


def test_mean_trace_scoring_is_opt_in():
    assert "--score-mean-trace" not in _cmds(_block())[0]
    assert "--score-mean-trace" in _cmds(_block(score_mean_trace=True))[0]


def test_explicit_cohorts_replace_the_auto_pairing():
    cmds = _cmds(
        _block(
            cohorts=[
                {
                    "train_dataset": "Hex-Training-24-0.01",
                    "control_dataset": "Hex-Control-24-0.01",
                    "out_dir": "Hex-24-0.01_july_august_mean_traces",
                }
            ]
        )
    )
    assert len(cmds) == 1
    assert _value(cmds[0], "--out-dir") == "/figs/Hex-24-0.01_july_august_mean_traces"


def test_an_explicit_cohort_may_give_an_absolute_out_dir():
    cmds = _cmds(
        _block(
            cohorts=[
                {
                    "train_dataset": "A-Training-1",
                    "control_dataset": "A-Control-1",
                    "out_dir": "/elsewhere/A",
                }
            ]
        )
    )
    assert _value(cmds[0], "--out-dir") == "/elsewhere/A"


def test_an_explicit_cohort_without_out_dir_falls_back_to_the_stem():
    cmds = _cmds(
        _block(
            cohorts=[
                {
                    "train_dataset": "EB-Training-24-0.1",
                    "control_dataset": "EB-Control-24-0.1",
                }
            ]
        )
    )
    assert _value(cmds[0], "--out-dir") == "/figs/EB-24-0.1"


def test_a_cohort_missing_an_arm_is_dropped_not_rendered_half():
    assert _cmds(_block(cohorts=[{"train_dataset": "A-Training-1"}])) == []


def test_a_batch_restriction_is_forwarded():
    cmds = _cmds(
        _block(
            cohorts=[
                {
                    "train_dataset": "EB-Training-24-1",
                    "control_dataset": "EB-Control-24-1",
                    "batch": 1,
                    "out_dir": "EB-24-1_batch_1",
                }
            ]
        )
    )
    assert _value(cmds[0], "--batch") == "1"
    assert "--batch" not in _cmds(_block())[0]


def test_out_dirs_are_unique_so_cohorts_cannot_overwrite_each_other():
    cmds = _cmds(_block())
    dirs = [_value(c, "--out-dir") for c in cmds]
    assert len(set(dirs)) == len(dirs), dirs


# ---------------------------------------------------------------------------
# Concentration series (datasets with no control arm)
# ---------------------------------------------------------------------------


CONC_BLOCK = [
    {
        "out_dir": "RandomPanel",
        "datasets": {
            "RandomPanel-Training-24-10": 10,
            "RandomPanel-24-1": 1,
            "RandomPanel-24-0.1": 0.1,
        },
    }
]


def _conc_cmds(**kw):
    cmds = _cmds(_block(conc_series=CONC_BLOCK, **kw))
    return [c for c in cmds if c[1].endswith("randompanel_conc_traces.py")]


def test_a_conc_series_gets_its_own_command():
    cmd = _conc_cmds()[0]
    assert _value(cmd, "--out-dir") == "/figs/RandomPanel"
    specs = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--dataset"]
    assert specs == [
        "RandomPanel-Training-24-10=10",
        "RandomPanel-24-1=1",
        "RandomPanel-24-0.1=0.1",
    ]


def test_the_conc_command_carries_the_same_figure_parameters():
    cmd = _conc_cmds()[0]
    for flag, value in (
        ("--wide-csv", "/data/wide.parquet"),
        ("--config", "/cfg.yaml"),
        ("--protocol", "v2"),
        ("--fps", "40.0"),
        ("--odor-on-s", "30.0"),
        ("--odor-off-s", "60.0"),
    ):
        assert _value(cmd, flag) == value, flag


def test_a_dataset_in_a_conc_series_is_not_also_auto_paired():
    """RandomPanel-Control-24-10 exists in the config but holds no rows; the
    concentration series is what replaces that pair, so pairing it as well
    would fail every run."""
    settings = _Settings(
        datasets=[
            "RandomPanel-Training-24-10",
            "RandomPanel-Control-24-10",
            "EB-Training-24-1",
            "EB-Control-24-1",
        ]
    )
    cmds = _cmds(_block(conc_series=CONC_BLOCK), settings)
    trained = [
        _value(c, "--train-dataset") for c in cmds if "--train-dataset" in c
    ]
    assert trained == ["EB-Training-24-1"]


def test_a_conc_series_without_datasets_is_dropped():
    cmds = _cmds(_block(conc_series=[{"out_dir": "X"}]))
    assert not [c for c in cmds if c[1].endswith("randompanel_conc_traces.py")]


def test_conc_series_out_dir_may_be_absolute():
    cmds = _conc_cmds()
    assert cmds
    block = [dict(CONC_BLOCK[0], out_dir="/elsewhere/RP")]
    cmd = [
        c for c in _cmds(_block(conc_series=block))
        if c[1].endswith("randompanel_conc_traces.py")
    ][0]
    assert _value(cmd, "--out-dir") == "/elsewhere/RP"


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


def test_expected_state_changes_when_a_cohort_is_added(tmp_path):
    wide = tmp_path / "wide.parquet"
    wide.write_bytes(b"x")
    block = _block(wide_csv=str(wide))
    one = rw._dataset_mean_traces_expected(block, _Settings(), config_path=None)
    two = rw._dataset_mean_traces_expected(
        _block(wide_csv=str(wide), dir_suffix="_other"), _Settings(), config_path=None
    )
    assert one != two


def test_expected_state_tracks_the_wide_table(tmp_path):
    wide = tmp_path / "wide.parquet"
    wide.write_bytes(b"x")
    block = _block(wide_csv=str(wide))
    before = rw._dataset_mean_traces_expected(block, _Settings(), config_path=None)
    assert before["wide_csv_mtime"] is not None


# ---------------------------------------------------------------------------
# The shipped config must actually declare the step
# ---------------------------------------------------------------------------


def test_shipped_config_wires_the_mean_traces_step():
    import yaml

    raw = yaml.safe_load(CONFIG_PATH.read_text())
    block = (raw.get("analysis") or {}).get("dataset_mean_traces")
    assert block, "dataset_mean_traces is not wired into config_new.yaml"
    assert block.get("wide_csv", "").endswith(".parquet")
    assert block.get("out_root")


def test_shipped_config_renders_the_published_3oct_figure_set():
    import yaml

    raw = yaml.safe_load(CONFIG_PATH.read_text())
    # Command SHAPE test: thaw everything so the shipped config's freeze
    # state (all non-sensitivity datasets are frozen) does not empty the
    # command list. Freeze behaviour itself is covered by
    # tests/test_figure_freeze_adherence.py.
    settings = load_settings(CONFIG_PATH)
    settings._thaw_all = True
    cmds = rw._dataset_mean_traces_commands(
        raw.get("analysis") or {},
        settings,
        python_exec="/py",
        config_path=CONFIG_PATH,
    )
    out_dirs = [_value(c, "--out-dir") for c in cmds]
    # Folders are named for the cohort itself -- no "_mean_traces_new" tail.
    assert any(d.endswith("/3Oct-24-0.1") for d in out_dirs), out_dirs
    assert any(d.endswith("/EB-24-1") for d in out_dirs), out_dirs
    assert len(set(out_dirs)) == len(out_dirs), out_dirs


def test_shipped_config_writes_under_the_new_opto_figures_tree():
    """config_new outputs belong beside Matrix-PER-Reactions-Model, not in the
    hand-run Results/Figures tree."""
    import yaml

    raw = yaml.safe_load(CONFIG_PATH.read_text())
    block = raw["analysis"]["dataset_mean_traces"]
    root = block["out_root"].rstrip("/")
    assert root.endswith("New-Opto-Fly-Figures/dataset_mean_comparisons"), root


def test_shipped_config_covers_randompanel_with_a_concentration_series():
    """RandomPanel has no control arm, so it is served by the conc series."""
    import yaml

    raw = yaml.safe_load(CONFIG_PATH.read_text())
    # Command SHAPE test: thaw everything so the shipped config's freeze
    # state (all non-sensitivity datasets are frozen) does not empty the
    # command list. Freeze behaviour itself is covered by
    # tests/test_figure_freeze_adherence.py.
    settings = load_settings(CONFIG_PATH)
    settings._thaw_all = True
    cmds = rw._dataset_mean_traces_commands(
        raw.get("analysis") or {},
        settings,
        python_exec="/py",
        config_path=CONFIG_PATH,
    )
    conc = [c for c in cmds if c[1].endswith("randompanel_conc_traces.py")]
    assert conc, "no RandomPanel concentration series configured"
    specs = [conc[0][i + 1] for i, a in enumerate(conc[0]) if a == "--dataset"]
    assert specs == [
        "RandomPanel-Training-24-10=10",
        "RandomPanel-24-1=1",
        "RandomPanel-24-0.1=0.1",
    ]
    trained = [_value(c, "--train-dataset") for c in cmds if "--train-dataset" in c]
    assert not any(t.startswith("RandomPanel") for t in trained), trained
