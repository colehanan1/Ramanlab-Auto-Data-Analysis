"""The geometry pipeline CLI and batch walking."""
import json
from pathlib import Path

import pytest
import yaml

from fbpipe import geometry_pipeline as GP
from fbpipe.config import Settings
from fbpipe.geometry import steps as S
from fbpipe.geometry.config import GeometrySettings


def test_step_names_and_list(capsys):
    assert GP.STEP_NAMES == ("track", "normalize", "wide", "figures")
    assert GP.main(["--config", "nope.yaml", "list"]) == 0
    out = capsys.readouterr().out
    for name in GP.STEP_NAMES:
        assert name in out


def test_unknown_step_is_rejected():
    with pytest.raises(SystemExit):
        GP.main(["--config", "nope.yaml", "fly"])


def test_run_steps_dispatches_only_supported_options(monkeypatch):
    calls = []
    monkeypatch.setattr(S, "run_track", lambda settings, gcfg, *, force=False, only_trial=None, max_frames=None: calls.append(("track", force, only_trial, max_frames)))
    monkeypatch.setattr(S, "run_normalize", lambda settings, gcfg, *, force=False: calls.append(("normalize", force)), raising=False)
    monkeypatch.setattr(S, "run_wide", lambda settings, gcfg: calls.append(("wide",)), raising=False)
    GP.run_steps(["track", "normalize", "wide"], object(), object(), force=True, only_trial="pretest_1", max_frames=5)
    assert calls == [("track", True, "pretest_1", 5), ("normalize", True), ("wide",)]


def test_iter_batches_and_videos(tmp_path):
    root = tmp_path / "flys_Abdomen" / "Hex-Sensitivity-24-0.1"
    b1 = root / "september_02_batch_1_rig_3_copy"
    b2 = root / "august_28_batch_1_rig_3_copy"
    for b in (b1, b2):
        b.mkdir(parents=True)
    (b1 / "output_september_02_batch_1_pretest_2_Citral_20260902_113000.mp4").write_bytes(b"x")
    (b1 / "output_september_02_batch_1_pretest_1_Hexanol_20260902_112000.mp4").write_bytes(b"x")
    (b1 / "output_september_02_batch_1_pretest_1_Hexanol_20260902_112000.csv").write_text("a\n")
    (b1 / "geometry").mkdir()
    settings = Settings(model_path="x", main_directories=[str(root)], datasets=("Hex-Sensitivity-24-0.1",))
    batches = list(S.iter_batches(settings))
    assert [(b.name, ds, fz) for b, ds, fz in batches] == [
        ("august_28_batch_1_rig_3_copy", "Hex-Sensitivity-24-0.1", False),
        ("september_02_batch_1_rig_3_copy", "Hex-Sensitivity-24-0.1", False)]
    vids = S.iter_videos(b1)
    assert [v.name[:40] for v in vids] == ["output_september_02_batch_1_pretest_1_He", "output_september_02_batch_1_pretest_2_Ci"]
    g = GeometrySettings()
    assert S.trial_out_dir(b1, g, vids[0]) == b1 / "geometry" / "september_02_batch_1_pretest_1"
    assert S.load_trial_entries(b1, g) == []


def test_worker_shard_reads_env(monkeypatch):
    monkeypatch.delenv("NUM_WORKERS", raising=False)
    monkeypatch.delenv("WORKER_INDEX", raising=False)
    assert S.worker_shard() == (1, 0)
    monkeypatch.setenv("NUM_WORKERS", "3")
    monkeypatch.setenv("WORKER_INDEX", "4")
    assert S.worker_shard() == (3, 1)


def test_wrapper_script_imports_main():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "pipeline" / "run_abdomen_workflows.py").read_text()
    assert "from fbpipe.geometry_pipeline import main" in src
