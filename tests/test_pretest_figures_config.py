"""config_new.yaml: the pretest export and Raw-Pre-Testing-PER-Traces block.

``config/`` is gitignored, so the config half of this change can never be
committed — these tests are the only record of it. They pin the on-disk file.

The two pieces:
  * ``combined.wide.trial_type_exports`` gains a ``pretest`` entry, giving the
    naive panel its own wide CSV (and the ``.parquet`` sibling the figure step
    reads);
  * ``combined_base.envelopes`` gains a third block with ``trial_type: pretest``
    writing ``Raw-Pre-Testing-PER-Traces``.

Both mechanisms are generic in run_workflows.py, so this needs no Python.
"""

from pathlib import Path

import pytest
import yaml

CONFIG = Path(__file__).resolve().parent.parent / "config" / "config_new.yaml"

FIG_ROOT = "/home/ramanlab/Documents/cole/Results/New-Opto-Fly-Figures"
CSV_ROOT = "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys"
PRETEST_CSV = f"{CSV_ROOT}/all_envelope_rows_wide_combined_base_pretest.csv"
PRETEST_FIGS = f"{FIG_ROOT}/Raw-Pre-Testing-PER-Traces"


@pytest.fixture(scope="module")
def cfg():
    if not CONFIG.exists():
        pytest.skip(f"{CONFIG} not present (config/ is gitignored)")
    with CONFIG.open() as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def combined_base(cfg):
    return cfg["analysis"]["combined"]["combined_base"]


@pytest.fixture(scope="module")
def exports(combined_base):
    return combined_base["wide"].get("trial_type_exports") or []


def _by_type(exports):
    return {str(e.get("trial_type", "")).lower(): e for e in exports}


# ── the pretest wide export ───────────────────────────────────────────────


def test_a_pretest_export_exists(exports):
    assert "pretest" in _by_type(exports)


def test_the_pretest_export_writes_its_own_csv(exports):
    assert _by_type(exports)["pretest"]["output_csv"] == PRETEST_CSV


def test_the_training_export_is_untouched(exports):
    """Adding an arm must not disturb the one that already works."""
    training = _by_type(exports)["training"]
    assert training["output_csv"] == (
        f"{CSV_ROOT}/all_envelope_rows_wide_combined_base_training.csv"
    )


def test_the_three_wide_tables_are_distinct_files(cfg, combined_base, exports):
    paths = [combined_base["wide"]["output_csv"]] + [e["output_csv"] for e in exports]
    assert len(paths) == len(set(paths)) == 3


# ── the Raw-Pre-Testing-PER-Traces figure block ───────────────────────────


def _envelope_blocks(combined_base):
    return combined_base["envelopes"]


def test_there_are_three_envelope_blocks(combined_base):
    assert len(_envelope_blocks(combined_base)) == 3


def _block_for(combined_base, trial_type):
    for block in _envelope_blocks(combined_base):
        if str(block.get("trial_type", "testing")).lower() == trial_type:
            return block
    raise AssertionError(f"no {trial_type} envelope block")


def test_the_pretest_block_writes_raw_pre_testing_traces(combined_base):
    assert _block_for(combined_base, "pretest")["out_dir"] == PRETEST_FIGS


def test_the_pretest_block_reads_the_pretest_parquet(combined_base):
    """It must read its OWN table, not the pooled one."""
    block = _block_for(combined_base, "pretest")
    assert block["wide_input"] == PRETEST_CSV.replace(".csv", ".parquet")


def test_each_envelope_block_writes_a_different_directory(combined_base):
    out_dirs = [b["out_dir"] for b in _envelope_blocks(combined_base)]
    assert len(out_dirs) == len(set(out_dirs)) == 3


def test_the_testing_block_still_owns_raw_testing_traces(combined_base):
    assert _block_for(combined_base, "testing")["out_dir"] == f"{FIG_ROOT}/Raw-Testing-PER-Traces"


def test_the_pretest_block_shares_the_threshold_rule(combined_base):
    """One threshold rule, one place — the red line must match every other figure."""
    pretest = _block_for(combined_base, "pretest")
    testing = _block_for(combined_base, "testing")
    for key in ("threshold_std_mult", "threshold_min_delta", "threshold_anchor_s"):
        assert pretest[key] == testing[key], key


def test_the_pretest_block_shares_the_odor_window(combined_base):
    pretest = _block_for(combined_base, "pretest")
    testing = _block_for(combined_base, "testing")
    for key in ("odor_on_s", "odor_off_s", "odor_latency_s", "after_show_sec", "fps_default"):
        assert pretest[key] == testing[key], key


def test_the_pretest_figures_land_under_new_opto_fly_figures(combined_base):
    """A pipeline step must never write into Results/Figures (hand-run one-offs)."""
    out_dir = _block_for(combined_base, "pretest")["out_dir"]
    assert out_dir.startswith(FIG_ROOT)


# ── keys _envelope_plot_config REQUIRES ───────────────────────────────────

REQUIRED_ENVELOPE_KEYS = ("matrix_npy", "codes_json", "out_dir")


def test_every_envelope_block_carries_the_required_paths(combined_base):
    """_envelope_plot_config calls _ensure_path on these three and raises
    ValueError if any is missing — even when wide_input is set, where
    matrix_npy is only a label hint. Omitting them aborted a full pipeline
    run at the very end, after ~25 minutes of work, with
    "Missing required path for 'matrix_npy'".
    """
    for block in _envelope_blocks(combined_base):
        tt = str(block.get("trial_type", "testing"))
        for key in REQUIRED_ENVELOPE_KEYS:
            assert block.get(key), f"{tt} envelope block is missing {key}"


def test_the_pretest_block_builds_a_real_envelope_plot_config(combined_base):
    """The end-to-end check the missing-key test above only approximates."""
    from scripts.pipeline.run_workflows import _envelope_plot_config

    cfg, _smb = _envelope_plot_config(dict(_block_for(combined_base, "pretest")))
    assert cfg.trial_type == "pretest"
    assert str(cfg.out_dir).endswith("Raw-Pre-Testing-PER-Traces")


def test_all_three_blocks_build_without_error(combined_base):
    from scripts.pipeline.run_workflows import _envelope_plot_config

    types = set()
    for block in _envelope_blocks(combined_base):
        cfg, _smb = _envelope_plot_config(dict(block))
        types.add(cfg.trial_type)
    assert types == {"testing", "training", "pretest"}


def test_the_pretest_block_keeps_its_own_matrix_paths(combined_base):
    """It must not borrow the testing block's matrix dir, or the label hints
    would come from the wrong phase."""
    pretest = _block_for(combined_base, "pretest")
    testing = _block_for(combined_base, "testing")
    assert pretest["matrix_npy"] != testing["matrix_npy"]
    assert pretest["codes_json"] != testing["codes_json"]
