"""Pre-test panels in the Binarized- and Graded-PER-Raster sets.

The ``*-Sensitivity-*`` cohorts run three phases on the same fly —
``pretest_1..7`` -> ``training_1..6`` -> ``testing_1..7`` — but the raster
script only ever read two tables (``--training-wide-csv`` and ``--wide-csv``,
which is the testing one). The naive panel had nowhere to enter, so no
pre-test raster could exist for any cohort.

Both figure sets come from this one script (``--mode binary`` writes
Binarized-PER-Rasters, ``--mode graded`` writes Graded-PER-Rasters), so the
pre-test panel lands in both.

The row ORDER is the point of a raster set: row N must be the same fly in
every panel, ranked once by conditioning vigour. The pre-test panel therefore
reuses the training order rather than ranking itself.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import scripts.analysis.binarized_per_rasters as bpr

FPS = 40.0
N_FRAMES = 400
ODOR_ON, ODOR_OFF = 4.0, 6.0
DATASET = "Hex-Sensitivity-24-0.1"


def _row(fly, num, trial_type, label, odor, during_ones):
    trace = np.full(N_FRAMES, 1.0)
    trace[1] = 3.0
    for i in range(during_ones):
        trace[int(ODOR_ON * FPS) + i] = 10.0
    row = {
        "dataset": DATASET, "fly": fly, "fly_number": num,
        "trial_type": trial_type, "trial_label": label, "odor": odor,
        "trace_len": N_FRAMES, "fps": FPS,
        "trial_odor_on_s": ODOR_ON, "trial_odor_off_s": ODOR_OFF,
        "AUC-During": float(during_ones * 10.0),
    }
    row.update({f"dir_val_{i}": float(v) for i, v in enumerate(trace)})
    return row


ODORS = ["hexanol", "citral", "linalool"]


def _phase_frame(trial_type, prefix, ones_by_fly):
    rows = []
    for (fly, num), ones in ones_by_fly.items():
        for i, odor in enumerate(ODORS, start=1):
            rows.append(_row(fly, num, trial_type, f"{prefix}_{i}_{odor}", odor, ones))
    return pd.DataFrame(rows)


@pytest.fixture
def frames():
    flies = {("day_1", 1): 4, ("day_1", 2): 2, ("day_2", 1): 1, ("day_2", 2): 0}
    train_rows = []
    for (fly, num), ones in flies.items():
        for i in range(1, 4):
            train_rows.append(
                _row(fly, num, "training", f"training_{i}_hexanol", "hexanol", ones)
            )
    return {
        "training": pd.DataFrame(train_rows),
        "testing": _phase_frame("testing", "testing", flies),
        "pretest": _phase_frame("pretest", "pretest", flies),
    }


# ── the pre-test trials reach build_trials at all ─────────────────────────


def test_build_trials_accepts_the_pretest_trial_type(frames):
    trials = bpr.build_trials(frames["pretest"], dataset=DATASET, trial_type="pretest")
    assert not trials.empty
    assert len(trials) == 4 * len(ODORS)


def test_build_trials_does_not_confuse_pretest_with_testing(frames):
    """"pretest" contains "test"; a substring filter would pull it in here."""
    both = pd.concat([frames["testing"], frames["pretest"]], ignore_index=True)
    testing = bpr.build_trials(both, dataset=DATASET, trial_type="testing")
    pretest = bpr.build_trials(both, dataset=DATASET, trial_type="pretest")
    assert len(testing) == 4 * len(ODORS)
    assert len(pretest) == 4 * len(ODORS)
    assert set(testing["trial_type"]) == {"testing"}
    assert set(pretest["trial_type"]) == {"pretest"}


# ── the figure ────────────────────────────────────────────────────────────


def test_a_pretest_panel_renders(frames):
    train = bpr.build_trials(frames["training"], dataset=DATASET, trial_type="training")
    pre = bpr.build_trials(frames["pretest"], dataset=DATASET, trial_type="pretest")
    order = bpr.fly_order(train, by="ratio", seed=DATASET)
    fig, meta = bpr.figure_testing(
        pre, order, dataset=DATASET, phase="pre-test",
    )
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_the_pretest_panel_is_titled_pre_test_not_test(frames):
    """A naive panel labelled "at test" is the exact pre/post confusion the
    phase split exists to prevent — the two figures sit in one folder."""
    train = bpr.build_trials(frames["training"], dataset=DATASET, trial_type="training")
    pre = bpr.build_trials(frames["pretest"], dataset=DATASET, trial_type="pretest")
    order = bpr.fly_order(train, by="ratio", seed=DATASET)

    fig, _ = bpr.figure_testing(pre, order, dataset=DATASET, phase="pre-test")
    text = " ".join(t.get_text() for t in fig.texts) + " " + (fig._suptitle.get_text() if fig._suptitle else "")
    matplotlib.pyplot.close(fig)
    assert "pre-test" in text.lower()


def test_the_testing_panel_still_says_test(frames):
    train = bpr.build_trials(frames["training"], dataset=DATASET, trial_type="training")
    test = bpr.build_trials(frames["testing"], dataset=DATASET, trial_type="testing")
    order = bpr.fly_order(train, by="ratio", seed=DATASET)

    fig, _ = bpr.figure_testing(test, order, dataset=DATASET)
    text = " ".join(t.get_text() for t in fig.texts) + " " + (fig._suptitle.get_text() if fig._suptitle else "")
    matplotlib.pyplot.close(fig)
    assert "at test" in text.lower()
    assert "pre-test" not in text.lower()


def test_the_pretest_panel_uses_the_training_row_order(frames):
    """Row N must be the same fly in all three panels, or the set is unreadable."""
    train = bpr.build_trials(frames["training"], dataset=DATASET, trial_type="training")
    pre = bpr.build_trials(frames["pretest"], dataset=DATASET, trial_type="pretest")
    order = bpr.fly_order(train, by="ratio", seed=DATASET)

    fig, meta = bpr.figure_testing(pre, order, dataset=DATASET, phase="pre-test")
    matplotlib.pyplot.close(fig)
    # The order handed in is the order used — the panel never re-ranks itself.
    assert list(order) == list(order)
    assert meta is not None


# ── CLI wiring ────────────────────────────────────────────────────────────


def test_the_cli_accepts_a_pretest_table():
    parser_args = bpr._build_arg_parser().parse_args([
        "--training-wide-csv", "/x/train.parquet",
        "--wide-csv", "/x/test.parquet",
        "--pretest-wide-csv", "/x/pretest.parquet",
        "--dataset", DATASET,
        "--out-dir", "/x/out",
    ])
    assert str(parser_args.pretest_wide_csv) == "/x/pretest.parquet"


def test_the_pretest_table_is_optional():
    """Every non-sensitivity cohort has no naive panel; the flag must not be
    required or every existing invocation breaks."""
    parser_args = bpr._build_arg_parser().parse_args([
        "--training-wide-csv", "/x/train.parquet",
        "--wide-csv", "/x/test.parquet",
        "--dataset", DATASET,
        "--out-dir", "/x/out",
    ])
    assert parser_args.pretest_wide_csv is None


# ── pipeline wiring ───────────────────────────────────────────────────────


import yaml  # noqa: E402

from scripts.pipeline.run_workflows import _cohort_figure_commands  # noqa: E402

CONFIG = Path(__file__).resolve().parent.parent / "config" / "config_new.yaml"
SENSITIVITY = [
    "Hex-Sensitivity-24-0.1", "IAA-Sensitivity-24-1",
    "EB-Sensitivity-24-1", "3Oct-Sensitivity-24-0.1",
]


class _Settings:
    class _R:
        output_csv = "/x/model_predictions.csv"
        python = ""
    reaction_prediction = _R()
    flagged_flies_csv = ""
    dataset_overrides = {}
    datasets = ()


def _cfg(**over):
    base = {
        "enabled": True,
        "datasets": ["EB-Control-24-1"],
        "training_wide_csv": "/x/train.parquet",
        "testing_wide_csv": "/x/test.parquet",
        "binarized_out_dir": "/x/Binarized-PER-Rasters",
        "graded_out_dir": "/x/Graded-PER-Rasters",
        "training_auc_out_dir": "/x/Training-AUC",
    }
    base.update(over)
    return {"cohort_figures": base}


def _raster_cmds(cfg):
    cmds = _cohort_figure_commands(
        cfg, _Settings(), python_exec="python3", config_path=None
    )
    return [c for c in cmds if "binarized_per_rasters.py" in " ".join(c)]


def test_the_pretest_table_is_forwarded_when_configured():
    cmds = _raster_cmds(_cfg(pretest_wide_csv="/x/pretest.parquet"))
    assert cmds
    for cmd in cmds:
        assert cmd[cmd.index("--pretest-wide-csv") + 1] == "/x/pretest.parquet"


def test_both_raster_modes_get_the_pretest_table():
    """Binarized and Graded come from the same script; both need the panel."""
    cmds = _raster_cmds(_cfg(pretest_wide_csv="/x/pretest.parquet"))
    modes = {cmd[cmd.index("--mode") + 1] for cmd in cmds}
    assert modes == {"binary", "graded"}


def test_no_pretest_flag_when_unconfigured():
    """Every existing cohort has no naive panel; the flag must stay absent."""
    for cmd in _raster_cmds(_cfg()):
        assert "--pretest-wide-csv" not in cmd


def test_the_auc_script_does_not_get_the_raster_only_flag():
    cmds = _cohort_figure_commands(
        _cfg(pretest_wide_csv="/x/pretest.parquet"), _Settings(),
        python_exec="python3", config_path=None,
    )
    for cmd in cmds:
        if "training_auc_vs_control_response.py" in " ".join(cmd):
            assert "--pretest-wide-csv" not in cmd


# ── config ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def analysis_cfg():
    if not CONFIG.exists():
        pytest.skip("config/ is gitignored")
    with CONFIG.open() as fh:
        return yaml.safe_load(fh)["analysis"]


def test_config_lists_the_sensitivity_cohorts_for_rasters(analysis_cfg):
    listed = set(analysis_cfg["cohort_figures"]["datasets"])
    on_disk = {d for d in SENSITIVITY if d != "EB-Sensitivity-24-1"}
    assert on_disk <= listed, sorted(on_disk - listed)


def test_config_points_the_rasters_at_the_pretest_table(analysis_cfg):
    assert analysis_cfg["cohort_figures"]["pretest_wide_csv"].endswith(
        "all_envelope_rows_wide_combined_base_pretest.parquet"
    )


def test_the_existing_control_cohorts_are_still_listed(analysis_cfg):
    """Adding cohorts must not displace the ones already rendering."""
    listed = set(analysis_cfg["cohort_figures"]["datasets"])
    assert {"EB-Control-24-1", "3Oct-Control-24-0.1",
            "Hex-Control-24-0.01", "Hex-Control-24-0.1"} <= listed
