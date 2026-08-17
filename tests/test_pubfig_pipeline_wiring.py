"""The publication figures must be part of a full pipeline run.

``pubfig_score_train_vs_control.py`` used to be hand-run only: a pipeline run
rewrote ``model_predictions.csv`` underneath it and the published PNGs silently
went stale until somebody remembered to re-run them. The blocker was that the
figure is not per-dataset -- it is per *cohort*, "the post-June block of
Hex-24-0.01", which the pipeline had no way to name. ``publication_figures``
in the config names them, and ``_pubfig_commands`` turns each into a command.

Command construction is tested directly rather than through ``_run_reactions``,
which needs a model, a predictions CSV and an output tree before it will run --
the same reason ``_thaw_cli_args`` exists (see test_freeze_figures.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fbpipe.config import load_settings  # noqa: E402
from scripts.pipeline import run_workflows as rw  # noqa: E402

CONFIG_PATH = ROOT / "config" / "config_new.yaml"


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, pub_block: dict) -> Path:
    cfg = {
        "model_path": "/tmp/model.pt",
        "main_directories": "/tmp/data",
        "reaction_prediction": {"publication_figures": pub_block},
    }
    out = tmp_path / "cfg.yaml"
    out.write_text(yaml.safe_dump(cfg))
    return out


def test_cohorts_parse_from_the_config(tmp_path):
    cfg = _write_config(
        tmp_path,
        {
            "figures_dir": "/figs",
            "cohorts": [
                {
                    "train_dataset": "Hex-Training-24-0.01",
                    "label": "Hex-24-0.01, August",
                    "out_stem": "Hex-24-0.01_august",
                    "fly_months": ["july", "august"],
                    "metrics": ["mean-score", "percent-responding"],
                }
            ],
        },
    )
    pub = load_settings(cfg).reaction_prediction.publication_figures
    assert pub.figures_dir == "/figs"
    assert len(pub.cohorts) == 1
    c = pub.cohorts[0]
    assert c.train_dataset == "Hex-Training-24-0.01"
    assert c.fly_months == ("july", "august")
    assert c.metrics == ("mean-score", "percent-responding")
    assert c.out_stem == "Hex-24-0.01_august"
    assert c.label == "Hex-24-0.01, August"


def test_a_cohort_defaults_to_both_metrics(tmp_path):
    """Omitting `metrics` must not silently produce zero figures."""
    cfg = _write_config(
        tmp_path, {"cohorts": [{"train_dataset": "Hex-Training-24-0.01"}]}
    )
    pub = load_settings(cfg).reaction_prediction.publication_figures
    assert pub.cohorts[0].metrics == ("mean-score", "percent-responding")


def test_no_publication_figures_block_means_no_cohorts(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"model_path": "/m.pt", "main_directories": "/d"}))
    pub = load_settings(cfg).reaction_prediction.publication_figures
    assert pub.cohorts == ()


def test_an_unknown_metric_is_rejected_at_load(tmp_path):
    """A typo must fail loudly, not drop that figure from the run."""
    cfg = _write_config(
        tmp_path,
        {"cohorts": [{"train_dataset": "D", "metrics": ["men-score"]}]},
    )
    with pytest.raises(ValueError, match="men-score"):
        load_settings(cfg)


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


class _Cohort:
    def __init__(self, **kw):
        self.train_dataset = kw.get("train_dataset", "Hex-Training-24-0.01")
        self.label = kw.get("label", "")
        self.out_stem = kw.get("out_stem", "")
        self.fly_months = kw.get("fly_months", ())
        self.metrics = kw.get("metrics", ("mean-score", "percent-responding"))
        self.genotype = kw.get("genotype", "")


class _Pub:
    def __init__(self, cohorts=(), figures_dir="/figs"):
        self.cohorts = tuple(cohorts)
        self.figures_dir = figures_dir


class _Reaction:
    def __init__(self, pub):
        self.publication_figures = pub


class _Settings:
    def __init__(self, pub, flagged_flies_csv=""):
        self.reaction_prediction = _Reaction(pub)
        self.flagged_flies_csv = flagged_flies_csv


def _cmds(cohorts, **kw):
    return rw._pubfig_commands(
        _Settings(_Pub(cohorts), **kw),
        python_exec="/py",
        csv_path=Path("/preds.csv"),
        config_path=Path("/cfg.yaml"),
    )


def test_no_cohorts_means_no_commands():
    """An unconfigured pipeline must not start rendering figures."""
    assert _cmds([]) == []


def test_one_command_per_cohort_per_metric():
    cmds = _cmds([_Cohort(), _Cohort(train_dataset="EB-Training-24-1")])
    assert len(cmds) == 4
    assert sum("percent-responding" in c for c in cmds) == 2


def test_command_carries_every_cohort_filter():
    cmd = _cmds(
        [_Cohort(fly_months=("july", "august"), out_stem="stem", label="Lbl")],
        flagged_flies_csv="/truth.csv",
    )[0]
    assert cmd[0] == "/py"
    assert cmd[1].endswith("pubfig_score_train_vs_control.py")
    assert "dataset" in cmd
    for flag, value in (
        ("--train-dataset", "Hex-Training-24-0.01"),
        ("--fly-months", "july,august"),
        ("--flagged-flies-csv", "/truth.csv"),
        ("--predictions-csv", "/preds.csv"),
        ("--config", "/cfg.yaml"),
        ("--figures-dir", "/figs"),
        ("--cohort-label", "Lbl"),
    ):
        assert flag in cmd, f"{flag} missing"
        assert cmd[cmd.index(flag) + 1] == value, flag


def test_months_are_omitted_when_the_cohort_has_none():
    """No months means the whole dataset -- not an empty --fly-months string,
    which the driver would treat as 'keep nothing'."""
    cmd = _cmds([_Cohort(fly_months=())])[0]
    assert "--fly-months" not in cmd


def test_flagged_csv_is_omitted_when_unset():
    cmd = _cmds([_Cohort()])[0]
    assert "--flagged-flies-csv" not in cmd


def test_each_metric_gets_its_own_out_stem():
    """Two metrics writing one stem would have the second overwrite the first."""
    cmds = _cmds([_Cohort(out_stem="Hex_aug")])
    stems = [c[c.index("--out-stem") + 1] for c in cmds]
    assert stems == [
        "pubfig_mean_score_train_vs_ctrl_Hex_aug",
        "pubfig_pct_responding_train_vs_ctrl_Hex_aug",
    ]
    assert len(set(stems)) == len(stems)


def test_out_stem_falls_back_to_the_dataset_name():
    cmds = _cmds([_Cohort(out_stem="")])
    assert cmds[0][cmds[0].index("--out-stem") + 1] == (
        "pubfig_mean_score_train_vs_ctrl_Hex-Training-24-0.01"
    )


def test_genotype_is_forwarded_only_when_set():
    assert "--genotype" not in _cmds([_Cohort()])[0]
    cmd = _cmds([_Cohort(genotype="GR5a-New")])[0]
    assert cmd[cmd.index("--genotype") + 1] == "GR5a-New"


# ---------------------------------------------------------------------------
# The shipped config must actually declare the cohort
# ---------------------------------------------------------------------------


def test_shipped_config_declares_the_hex_24_001_cohort():
    pub = load_settings(CONFIG_PATH).reaction_prediction.publication_figures
    hexes = [c for c in pub.cohorts if c.train_dataset == "Hex-Training-24-0.01"]
    assert hexes, "the Hex-24-0.01 cohort is not wired into the pipeline"
    c = hexes[0]
    assert c.fly_months == ("july", "august")
    assert set(c.metrics) == {"mean-score", "percent-responding"}
    assert pub.figures_dir, "no figures_dir, so the figures would land in cwd"


def test_shipped_config_cohorts_have_unique_out_stems():
    """Two cohorts sharing a stem would silently overwrite each other."""
    pub = load_settings(CONFIG_PATH).reaction_prediction.publication_figures
    stems = [c.out_stem or c.train_dataset for c in pub.cohorts]
    assert len(set(stems)) == len(stems), stems


# ---------------------------------------------------------------------------
# Cohort-scoped odor overrides
# ---------------------------------------------------------------------------


def test_cohort_odor_remap_is_forwarded_as_repeated_flags():
    class _C(_Cohort):
        pass

    c = _Cohort()
    c.odor_remap = (("Apple Cider Vinegar", "Isoamyl Acetate (1%)"), ("Citral", "Citral (1%)"))
    cmd = _cmds([c])[0]
    pairs = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--odor-remap"]
    assert pairs == ["Apple Cider Vinegar=Isoamyl Acetate (1%)", "Citral=Citral (1%)"]


def test_no_odor_remap_flag_when_the_cohort_has_none():
    assert "--odor-remap" not in _cmds([_Cohort()])[0]


def test_cohort_odor_remap_parses_from_config(tmp_path):
    cfg = _write_config(
        tmp_path,
        {
            "cohorts": [
                {
                    "train_dataset": "Hex-Training-24-0.1",
                    "odor_remap": {"Apple Cider Vinegar": "Isoamyl Acetate (1%)"},
                }
            ]
        },
    )
    pub = load_settings(cfg).reaction_prediction.publication_figures
    assert pub.cohorts[0].odor_remap == (
        ("Apple Cider Vinegar", "Isoamyl Acetate (1%)"),
    )


def test_shipped_config_uses_the_august_panel_for_hex_24_01():
    """The dataset-level remap tracks whichever block is LIVE.

    Hex-24-0.1 ran two rig plumbings: may/june sent sourdough on the Citral
    channel and isoamyl acetate on the Linalool channel, august ran the
    standard panel with isoamyl acetate on the ACV channel.
    ``freeze_folders_before: 2026-06-26`` retired the whole may/june block, so
    every live fly is august (verified 2026-08-13) and the dataset-level remap
    was switched to the august panel -- the sourdough labels would now
    mislabel real august flies in every figure. The may/june mapping is kept
    as a comment in the config for whoever thaws that block.
    """
    settings = load_settings(CONFIG_PATH)
    dataset_level = dict(settings.dataset_overrides["Hex-Training-24-0.1"].odor_remap)
    assert dataset_level["Citral"] == "Citral (1%)"
    assert dataset_level["Linalool"] == "Linalool (1%)"
    assert dataset_level["Apple Cider Vinegar"] == "Isoamyl Acetate (1%)"
    assert dataset_level["Hexanol"] == "Hexanol (0.1%)"
    # Both arms of the pair, or trained and control figures disagree on labels.
    assert dataset_level == dict(
        settings.dataset_overrides["Hex-Control-24-0.1"].odor_remap
    )


def test_every_hex_24_01_odor_carries_its_concentration():
    """An untagged label has no dose to match a naive panel against, so the
    naive-vs-trained sweep skips it."""
    settings = load_settings(CONFIG_PATH)
    remap = dict(settings.dataset_overrides["Hex-Training-24-0.1"].odor_remap)
    assert len(remap) == 7
    assert all("%" in label for label in remap.values()), remap


def test_shipped_config_scopes_the_24_01_august_remap_to_the_cohort():
    settings = load_settings(CONFIG_PATH)
    pub = settings.reaction_prediction.publication_figures
    aug = [c for c in pub.cohorts if c.train_dataset == "Hex-Training-24-0.1"]
    assert aug, "the Hex-24-0.1 August cohort is not wired in"
    override = dict(aug[0].odor_remap)
    assert override["Apple Cider Vinegar"] == "Isoamyl Acetate (1%)"
    assert override["Citral"] == "Citral (1%)"
    assert override["Linalool"] == "Linalool (1%)"
    assert override["Hexanol"] == "Hexanol (0.1%)", "0.1 cohort, not 0.01"
    assert aug[0].fly_months == ("august",)
    assert aug[0].genotype == "GR5a-Old"
