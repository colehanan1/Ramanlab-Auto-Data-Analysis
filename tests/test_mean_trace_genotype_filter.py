"""Mean-trace figures must be restricted to one genotype.

``fbpipe.utils.fly_type`` exists so that "different genotypes are never plotted
together", but the wide envelope table pools them: ``RandomPanel-Training-24-10``
carries 8 GR5a-GCaMP8 fly-ids alongside 21 GR5a-Old, so its 10% mean trace was
an average across two genotypes while the 1% and 0.1% lines it is compared
against are pure GR5a-Old.

``--genotype`` filters on the ``fly_type`` column for both mean-trace drivers,
and the pipeline block sets it once for every figure set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analysis import envelope_visuals as ev  # noqa: E402
from scripts.analysis.dataset_mean_traces_tvc import filter_by_genotype  # noqa: E402
from scripts.pipeline import run_workflows as rw  # noqa: E402

N_FRAMES = 120
FPS = 40.0
ODOR_ON_S = 0.5
BASELINE_FRAMES = int(round(ODOR_ON_S * FPS))

OLD = "GR5a-Old"
GCAMP = "GR5a-GCaMP8"


@pytest.fixture(autouse=True)
def _protocol():
    saved_protocol = ev.get_protocol()
    saved_remap = {ds: dict(m) for ds, m in ev._DATASET_ODOR_REMAP.items()}
    ev.set_protocol("v2")
    ev.set_dataset_odor_remap({})
    try:
        yield
    finally:
        ev.set_protocol(saved_protocol)
        ev.set_dataset_odor_remap(saved_remap)


def _frame(rows):
    """rows: (dataset, fly, fly_number, fly_type, trial_label, value)."""
    out = []
    for dataset, fly, fly_number, fly_type, label, value in rows:
        row = {
            "dataset": dataset,
            "fly": fly,
            "fly_number": fly_number,
            "fly_type": fly_type,
            "trial_type": "testing",
            "trial_label": label,
        }
        row.update(
            {
                f"dir_val_{i}": value * max(0, i - BASELINE_FRAMES)
                for i in range(N_FRAMES)
            }
        )
        out.append(row)
    return pd.DataFrame(out)


def _panel_frame():
    """Three concentrations; only the 10% dataset is genotype-mixed."""
    rows = []
    for dataset, types in (
        ("RandomPanel-Training-24-10", (OLD, OLD, GCAMP)),
        ("RandomPanel-24-1", (OLD, OLD)),
        ("RandomPanel-24-0.1", (OLD, OLD)),
    ):
        for i, fly_type in enumerate(types):
            for label, value in (("testing_1_hexanol", 1.0), ("testing_2_hexanol", 2.0)):
                rows.append((dataset, f"july_20_batch_{i}", 1, fly_type, label, value))
    return _frame(rows)


# ---------------------------------------------------------------------------
# The filter itself
# ---------------------------------------------------------------------------


def test_only_the_named_genotype_survives():
    df = _panel_frame()
    kept = filter_by_genotype(df, ["GR5a-Old"])
    assert set(kept["fly_type"]) == {OLD}
    assert len(kept) < len(df)


def test_matching_ignores_case():
    kept = filter_by_genotype(_panel_frame(), ["gr5a-old"])
    assert set(kept["fly_type"]) == {OLD}


def test_no_genotype_asked_for_leaves_the_frame_alone():
    df = _panel_frame()
    assert filter_by_genotype(df, []) is df


def test_several_genotypes_may_be_kept():
    kept = filter_by_genotype(_panel_frame(), [OLD, GCAMP])
    assert set(kept["fly_type"]) == {OLD, GCAMP}


def test_a_table_without_fly_type_is_an_error_not_a_silent_pass():
    """Silently keeping every fly would publish a mixed-genotype mean under a
    filename that claims one genotype."""
    df = _panel_frame().drop(columns=["fly_type"])
    with pytest.raises(SystemExit, match="fly_type"):
        filter_by_genotype(df, [OLD])


def test_an_unknown_genotype_names_what_was_available():
    with pytest.raises(SystemExit) as excinfo:
        filter_by_genotype(_panel_frame(), ["Canton-S"])
    message = str(excinfo.value)
    assert "Canton-S" in message and OLD in message


# ---------------------------------------------------------------------------
# Both drivers honour it end to end
# ---------------------------------------------------------------------------


def test_conc_series_drops_the_other_genotype(tmp_path):
    from scripts.analysis.randompanel_conc_traces import main

    wide = tmp_path / "wide.parquet"
    _panel_frame().to_parquet(wide)
    out_dir = tmp_path / "figs"
    main([
        "--wide-csv", str(wide),
        "--out-dir", str(out_dir),
        "--fps", str(FPS),
        "--odor-on-s", str(ODOR_ON_S),
        "--odor-off-s", "1.0",
        "--dataset", "RandomPanel-Training-24-10=10",
        "--dataset", "RandomPanel-24-1=1",
        "--dataset", "RandomPanel-24-0.1=0.1",
        "--genotype", OLD,
    ])
    meta = json.loads((out_dir / "conc_series.json").read_text())
    assert meta["genotypes"] == [OLD]
    # 3 fly-ids in the 10% dataset, one of them GCaMP8.
    assert meta["per_odor"]["Hexanol"]["10"] == 2


def test_conc_series_without_the_filter_pools_both_genotypes(tmp_path):
    from scripts.analysis.randompanel_conc_traces import main

    wide = tmp_path / "wide.parquet"
    _panel_frame().to_parquet(wide)
    out_dir = tmp_path / "figs"
    main([
        "--wide-csv", str(wide),
        "--out-dir", str(out_dir),
        "--fps", str(FPS),
        "--odor-on-s", str(ODOR_ON_S),
        "--odor-off-s", "1.0",
        "--dataset", "RandomPanel-Training-24-10=10",
        "--dataset", "RandomPanel-24-1=1",
        "--dataset", "RandomPanel-24-0.1=0.1",
    ])
    meta = json.loads((out_dir / "conc_series.json").read_text())
    assert meta["genotypes"] == []
    assert meta["per_odor"]["Hexanol"]["10"] == 3


def test_trained_vs_control_honours_the_filter(tmp_path):
    from scripts.analysis.dataset_mean_traces_tvc import main

    rows = []
    for dataset in ("EB-Training-24-1", "EB-Control-24-1"):
        for i, fly_type in enumerate((OLD, OLD, GCAMP)):
            rows.append((dataset, f"july_20_batch_{i}", 1, fly_type,
                         "testing_1_hexanol", 1.0 + i))
    wide = tmp_path / "wide.parquet"
    _frame(rows).to_parquet(wide)
    out_dir = tmp_path / "figs"
    main([
        "--wide-csv", str(wide),
        "--train-dataset", "EB-Training-24-1",
        "--control-dataset", "EB-Control-24-1",
        "--out-dir", str(out_dir),
        "--fps", str(FPS),
        "--odor-on-s", str(ODOR_ON_S),
        "--odor-off-s", "1.0",
        "--genotype", OLD,
    ])
    meta = json.loads((out_dir / "training_vs_control.json").read_text())
    assert meta["genotypes"] == [OLD]
    entry = next(iter(meta["per_odor"].values()))
    assert entry["n_training_flies"] == 2
    assert entry["n_control_flies"] == 2


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------


class _Settings:
    flagged_flies_csv = ""
    protocol = "v2"
    fps_default = 40.0
    odor_on_s = 30.0
    odor_off_s = 60.0
    datasets = ["EB-Training-24-1", "EB-Control-24-1"]


def _cmds(**block_kw):
    block = {"wide_csv": "/w.parquet", "out_root": "/figs"}
    block.update(block_kw)
    return rw._dataset_mean_traces_commands(
        {"dataset_mean_traces": block},
        _Settings(),
        python_exec="/py",
        config_path=None,
    )


CONC = [{"out_dir": "RandomPanel", "datasets": {"RandomPanel-24-1": 1}}]


def test_block_genotype_reaches_both_drivers():
    cmds = _cmds(genotype=OLD, conc_series=CONC)
    assert len(cmds) == 2
    for cmd in cmds:
        assert cmd[cmd.index("--genotype") + 1] == OLD


def test_no_genotype_configured_means_no_flag():
    for cmd in _cmds(conc_series=CONC):
        assert "--genotype" not in cmd


def test_a_series_may_override_the_block_genotype():
    series = [dict(CONC[0], genotype=GCAMP)]
    cmds = _cmds(genotype=OLD, conc_series=series)
    conc_cmd = [c for c in cmds if c[1].endswith("randompanel_conc_traces.py")][0]
    assert conc_cmd[conc_cmd.index("--genotype") + 1] == GCAMP


def test_shipped_config_restricts_the_figures_to_gr5a_old():
    import yaml

    raw = yaml.safe_load((ROOT / "config" / "config_new.yaml").read_text())
    block = raw["analysis"]["dataset_mean_traces"]
    assert block.get("genotype") == OLD
