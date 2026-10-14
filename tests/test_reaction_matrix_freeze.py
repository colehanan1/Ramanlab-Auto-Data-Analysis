"""The reaction-matrix figures must skip datasets frozen for figures.

``reaction_matrix_from_spreadsheet`` renders one matrix per dataset present in
model_predictions.csv. Frozen datasets stay in that CSV (their rows are real
data), so without an explicit skip every run redraws figures that were declared
final -- restyling them with the current palette and threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
for _p in (str(_REPO), str(_REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scripts.analysis.reaction_matrix_from_spreadsheet as rm  # noqa: E402


def _predictions(path: Path) -> Path:
    rows = []
    for dataset in ("Live-Odor", "Frozen-Odor"):
        for fly in ("f1", "f2"):
            for trial in (1, 2, 3):
                rows.append({
                    "dataset": dataset,
                    "fly": f"{dataset}_{fly}",
                    "fly_number": "1",
                    "trial_label": f"testing_{trial}_hexanol",
                    "prediction": trial % 2,
                    "trial_type": "testing",
                })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _run(tmp_path, frozen=()):
    out = tmp_path / f"out_{'-'.join(frozen) or 'all'}"
    cfg = rm.SpreadsheetMatrixConfig(
        csv_path=_predictions(tmp_path / "predictions.csv"),
        out_dir=out,
        latency_sec=2.4,
        frozen_datasets=frozenset(frozen),
    )
    rm.generate_reaction_matrices_from_csv(cfg)
    return {p.name for p in out.rglob("*.png")}


def test_frozen_dataset_renders_no_matrix(tmp_path):
    names = _run(tmp_path, frozen=("Frozen-Odor",))
    assert names, "expected the live dataset to still render"
    assert not any("Frozen-Odor" in n for n in names), names
    assert any("Live-Odor" in n for n in names), names


def test_default_renders_every_dataset(tmp_path):
    names = _run(tmp_path)
    assert any("Frozen-Odor" in n for n in names), names
    assert any("Live-Odor" in n for n in names), names


def test_all_frozen_raises_nothing_and_writes_nothing(tmp_path):
    """Every dataset frozen is a legitimate no-op, not an error."""
    names = _run(tmp_path, frozen=("Live-Odor", "Frozen-Odor"))
    assert names == set()


def test_main_reads_the_freeze_from_config(tmp_path, monkeypatch):
    """--config carries the freeze; --thaw lifts it for one run."""
    config = tmp_path / "config.yaml"
    config.write_text(
        "dataset_overrides:\n"
        "  Frozen-Odor:\n"
        "    freeze:\n"
        "      data: true\n"
        "      figures: true\n"
    )
    seen = {}

    def _capture(cfg):
        seen["frozen"] = set(cfg.frozen_datasets)

    monkeypatch.setattr(rm, "generate_reaction_matrices_from_csv", _capture)
    base = [
        "--csv-path", str(_predictions(tmp_path / "predictions.csv")),
        "--out-dir", str(tmp_path / "out"),
        "--config", str(config),
    ]
    rm.main(base)
    assert seen["frozen"] == {"Frozen-Odor"}

    rm.main(base + ["--thaw", "Frozen-Odor"])
    assert seen["frozen"] == set()

    rm.main(base + ["--thaw-all"])
    assert seen["frozen"] == set()


# --- training-vs-control matrices -------------------------------------

import scripts.analysis.reaction_matrix_training_vs_control as tvc  # noqa: E402


def _tvc_predictions(path: Path) -> Path:
    rows = []
    for dataset in ("EB-Training-24-1", "EB-Control-24-1",
                    "Hex-Training-24-0.01", "Hex-Control-24-0.01"):
        for fly in ("f1", "f2"):
            for trial in (1, 2, 3):
                rows.append({
                    "dataset": dataset,
                    "fly": f"{dataset}_{fly}",
                    "fly_number": "1",
                    "trial_label": f"testing_{trial}_hexanol",
                    "prediction": trial % 2,
                    "trial_type": "testing",
                })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _tvc_run(tmp_path, frozen=()):
    out = tmp_path / f"tvc_{'-'.join(frozen) or 'all'}"
    cfg = rm.SpreadsheetMatrixConfig(
        csv_path=_tvc_predictions(tmp_path / "preds_tvc.csv"),
        out_dir=out,
        latency_sec=2.4,
        frozen_datasets=frozenset(frozen),
    )
    tvc.generate_training_vs_control_matrices(cfg)
    return {str(p.relative_to(out)) for p in out.rglob("*.png")}


def test_tvc_pair_frozen_on_both_arms_is_skipped(tmp_path):
    names = _tvc_run(tmp_path, frozen=("Hex-Training-24-0.01", "Hex-Control-24-0.01"))
    assert not any("Hex-Training" in n for n in names), names
    assert any("EB-Training" in n for n in names), names


def test_tvc_pair_with_one_live_arm_still_renders(tmp_path):
    """A frozen control against a live trained arm must keep redrawing, or new
    flies in the live arm never reach the comparison."""
    names = _tvc_run(tmp_path, frozen=("Hex-Control-24-0.01",))
    assert any("Hex-Training" in n for n in names), names


def test_tvc_default_renders_both_pairs(tmp_path):
    names = _tvc_run(tmp_path)
    assert any("Hex-Training" in n for n in names), names
    assert any("EB-Training" in n for n in names), names


def test_tvc_main_reads_the_freeze_from_config(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        "dataset_overrides:\n"
        "  Hex-Control-24-0.01:\n"
        "    freeze:\n"
        "      figures: true\n"
    )
    seen = {}
    monkeypatch.setattr(
        tvc, "generate_training_vs_control_matrices",
        lambda cfg: seen.__setitem__("frozen", set(cfg.frozen_datasets)),
    )
    base = [
        "--csv-path", str(_tvc_predictions(tmp_path / "preds_tvc.csv")),
        "--out-dir", str(tmp_path / "out"),
        "--config", str(config),
    ]
    tvc.main(base)
    assert seen["frozen"] == {"Hex-Control-24-0.01"}
    tvc.main(base + ["--thaw-all"])
    assert seen["frozen"] == set()


# --- the cross-dataset rate summary keeps held datasets' rows -----------------

import scripts.analysis.envelope_visuals as ev  # noqa: E402

_PANELS = {
    # Different odors per dataset, so the summary's columns depend on which
    # datasets are in it; the trained one repeats Hexanol (two presentations).
    "Hex-Training-24-0.1": ("hexanol", "ethylbutyrate", "hexanol"),
    "Alpha-Panel": ("3-octanol", "linalool", "ethylbutyrate"),
    "Beta-Panel": ("applecidervinegar", "linalool"),
}


def _panel_predictions(path: Path, night: int) -> Path:
    rows = []
    for dataset, odors in _PANELS.items():
        genotypes = ("GR5a-Old", "Orco") if dataset == "Beta-Panel" else ("GR5a-Old",)
        for g, genotype in enumerate(genotypes):
            for f in range(3):
                for t, odor in enumerate(odors, start=1):
                    hit = (f + t + g) % 2
                    if dataset == "Alpha-Panel" and night == 3:
                        hit = 1 - hit  # only Alpha-Panel changes on night 3
                    rows.append({
                        "dataset": dataset, "fly": f"{dataset}_{genotype}_f{f}", "fly_number": "1",
                        "fly_type": genotype, "trial_label": f"testing_{t}_{odor}",
                        "prediction": hit, "trial_type": "testing",
                    })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _summary_run(tmp_path, out, night, frozen=()):
    cfg = rm.SpreadsheetMatrixConfig(
        csv_path=_panel_predictions(tmp_path / f"preds_n{night}.csv", night),
        out_dir=out,
        latency_sec=2.4,
        trial_orders=("trained-first",),
        frozen_datasets=frozenset(frozen),
    )
    rm.generate_reaction_matrices_from_csv(cfg)
    return out / "reaction_rates_summary_unordered.csv"


@pytest.mark.parametrize("protocol", ["v2", "legacy"])
def test_held_datasets_keep_their_summary_rows_night_after_night(tmp_path, protocol):
    """Night 1 draws every dataset; on night 3 only Alpha-Panel changed and the
    other two are held. The summary must still list the held ones -- with the
    rows night 1 wrote -- and read exactly as a night that drew all three."""
    ev.set_protocol(protocol)
    out = tmp_path / "out"
    _summary_run(tmp_path, out, night=1)
    held = _summary_run(tmp_path, out, night=3, frozen=("Hex-Training-24-0.1", "Beta-Panel"))
    reference = _summary_run(tmp_path, tmp_path / "ref", night=3)
    assert held.read_bytes() == reference.read_bytes()
    text = held.read_text()
    assert "Beta-Panel [GR5a-Old]" in text and "Beta-Panel [Orco]" in text
    assert text.splitlines()[1].startswith("Hex-Training-24-0.1,")  # ODOR_ORDER first


def test_a_held_dataset_never_drawn_before_stays_out_of_the_summary(tmp_path):
    ev.set_protocol("v2")
    out = tmp_path / "out"
    _summary_run(tmp_path, out, night=1, frozen=("Beta-Panel",))
    summary = _summary_run(tmp_path, out, night=3, frozen=("Beta-Panel", "Hex-Training-24-0.1"))
    names = [line.split(",")[0] for line in summary.read_text().splitlines()[1:]]
    assert names == ["Hex-Training-24-0.1", "Alpha-Panel"]


def test_every_dataset_held_leaves_the_summary_untouched(tmp_path):
    ev.set_protocol("v2")
    out = tmp_path / "out"
    first = _summary_run(tmp_path, out, night=1).read_bytes()
    summary = _summary_run(tmp_path, out, night=3, frozen=tuple(_PANELS))
    assert summary.read_bytes() == first
