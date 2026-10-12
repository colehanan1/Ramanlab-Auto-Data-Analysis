"""Auto-freeze through run_workflows.main, night after night.

Every run is a full ``main()``: the YOLO/per-trial subprocesses and the figure
steps are recorders, but the combined step is the REAL one -- build_wide_csv
walks or splices each dataset on synthetic tables and the freeze cache is
really written -- so these pin what the nightly does:

* night 1 adopts the dataset frozen today and builds the live ones once;
* a quiet night runs no YOLO, no per-trial steps, no combine, splices every
  dataset and writes byte-identical wide tables;
* a new batch with videos thaws only its dataset, which is rebuilt (YOLO for
  the new videos) and refrozen with the new trials;
* ``data: false`` is rebuilt every night with folder retirements applied;
* a run that dies never records its rebuild, and the next night redoes it;
* a video landing mid-run before the dataset's YOLO slot is processed that
  night; figures follow the plan; the run holds the whole-run lock.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import scripts.analysis.envelope_visuals as ev
from fbpipe import autofreeze as af
from fbpipe import freeze as _freeze
from fbpipe.config import load_settings
from scripts.pipeline import run_workflows as rw

from _loop_harness import isolate_env

ADOPT = "EB-Control-24-1"       # data: true today, valid cache -> adopted
TRAIN = "EB-Training-24-1"      # its pair arm, no block -> built once
SENS = "IAA-Sensitivity-12-1"   # no block -> built once
BATCH = "august_09_batch_1_rig_2"


def _short(batch: str) -> str:
    return "_".join(batch.split("_")[:4])


def make_batch(root: Path, batch: str, trials=("testing_1", "testing_2"), *, processed=True,
               videos=False, born="2026-08-01", stamp="20260809", n=6):
    """A batch as the rig and the pipeline leave it."""
    d = root / batch
    env = d / "angle_distance_rms_envelope"
    d.mkdir(parents=True, exist_ok=True)
    (d / "session_metadata.txt").write_text(f"Born (approx): {born}\nHost: Flybehavior2\n")
    for i, trial in enumerate(trials):
        stem = f"output_{_short(batch)}_{trial}_Hexanol_{stamp}_12{i:02d}00"
        (d / f"{stem}.csv").write_text("frame,timestamp\n0,0\n")
        if videos:
            (d / f"{stem}.mp4").write_bytes(b"raw video")
        if processed:
            yolo_outputs(d, stem)
            combine_outputs(d, trial, n=n, seed=i)
    return d


def yolo_outputs(batch_dir: Path, stem: str) -> Path:
    from fbpipe.utils.yolo_discovery import trial_folder_name

    name = trial_folder_name(stem + ".mp4")
    trial = batch_dir / name
    trial.mkdir(exist_ok=True)
    pd.DataFrame({"frame": [0, 1], "eye_0_x": [1.0, 2.0]}).to_parquet(trial / f"{name}_distances_merged.parquet")
    pd.DataFrame({"frame": [0, 1]}).to_parquet(trial / f"{name}_fly1_distances.parquet")
    return trial


def combine_outputs(batch_dir: Path, trial: str, *, n=6, seed=0):
    env = batch_dir / "angle_distance_rms_envelope"
    env.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed + len(batch_dir.name))
    pd.DataFrame({"combined_pct": rng.uniform(0, 100, n), "combined_base": rng.uniform(0, 100, n)}).to_csv(
        env / f"{trial}_hexanol_fly1_distances_fly1_angle_distance_rms_envelope.csv", index=False
    )


def write_config(tmp_path: Path, datasets, overrides, **top) -> Path:
    model = tmp_path / "model.pt"
    model.write_bytes(b"weights")
    cfg = {
        "protocol": "v2",
        "dataset_bases": {"data": str(tmp_path / "data"), "secured": str(tmp_path / "secured")},
        "datasets": list(datasets),
        "dataset_overrides": overrides,
        "cache_dir": str(tmp_path / "cache"),
        "model_path": str(model),
        "force": {"pipeline": True, "combined": True, "yolo": False},
        "yolo": {"num_workers": 1},
        # DIFFERENT values for the two gate keys (CLAUDE.md).
        "proboscis_filter": {"max_eye_prob_distance_px": 170.0},
        "distance_limits": {"class2_min": 10.0, "class2_max": 160.0,
                            "three_fly_max_eye_prob_distance_px": 150.0},
        "freeze_folders_before": "2026-06-26",
        "analysis": {"combined": {
            "combine": {"roots": []},
            "combined_base": {"wide": {
                "roots": [],
                "output_csv": str(tmp_path / "out" / "wide.csv"),
                "measure_cols": ["combined_pct", "combined_base"],
                "use_per_trial_baseline": True,
                "trial_type_exports": [{"trial_type": "training",
                                        "output_csv": str(tmp_path / "out" / "wide_training.csv")}],
            }},
        }},
        **top,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


_FIGURE_STEPS = (
    "_run_envelope_visuals", "_run_training", "_run_light_only_traces",
    "_rerender_envelope_block_with_scores", "_run_dataset_means", "_run_dataset_means_training",
    "_run_dataset_mean_traces", "_run_naive_vs_trained", "_run_training_scores",
    "_run_pretest_score_bars", "_run_pretest_vs_test", "_run_cohort_figures",
)


class Night:
    """One full main() run; what it launched and what the figures saw."""

    def __init__(self, monkeypatch, cfg: Path, *argv, on_pipeline=None, fail_in=None, expect=None):
        self.pipeline: list[tuple[str, tuple]] = []
        self.combined: list[str] = []
        self.figures: dict[str, tuple[bool, bool]] = {}
        self.plan_env = None
        self.secure_cleanup = 0
        self.smb = 0

        def _run_pipeline(config_path, *, main_directory=None, steps=None, num_workers=1):
            self.pipeline.append((Path(main_directory).name, tuple(steps or ())))
            if on_pipeline is not None:
                on_pipeline(Path(main_directory), tuple(steps or ()))

        def _combine(cfg_):
            self.combined.append(Path(cfg_.root).name)

        def _reactions(settings, config_path=None):
            from fbpipe.freeze import freeze_flags

            self.plan_env = os.environ.get(af.PLAN_ENV)
            for ds in settings.datasets:
                self.figures[ds] = freeze_flags(settings, ds)
            if fail_in == "reactions":
                raise RuntimeError("a figure step crashed")

        def _cleanup(*a, **k):
            self.secure_cleanup += 1

        isolate_env(monkeypatch)
        monkeypatch.setenv("FBPIPE_DISABLE_NTFY", "1")
        monkeypatch.setenv("FBPIPE_RUN_LOCK_WAIT_MIN", "0")
        monkeypatch.setattr(rw, "_run_pipeline", _run_pipeline)
        monkeypatch.setattr(rw, "combine_distance_angle", _combine)
        monkeypatch.setattr(rw, "_run_reactions", _reactions)
        monkeypatch.setattr(rw, "_run_secure_cleanup", _cleanup)
        monkeypatch.setattr(rw, "_batch_copy_to_smb", lambda data: setattr(self, "smb", self.smb + 1))
        monkeypatch.setattr(rw, "_run_data_secured_sync", lambda data: None)
        monkeypatch.setattr(rw, "_load_light_checks_for_envelopes", lambda: None)
        monkeypatch.setattr(rw, "_load_model_scores_for_envelopes", lambda settings: None)
        for name in _FIGURE_STEPS:
            monkeypatch.setattr(rw, name, lambda *a, **k: None)
        monkeypatch.setattr(rw, "set_protocol", lambda protocol: None)
        monkeypatch.setattr(rw, "set_dataset_light_windows", lambda windows: None)
        monkeypatch.setattr(rw, "set_dataset_odor_remap", lambda remap: None)
        from scripts.analysis import envelope_combined, envelope_exports

        monkeypatch.setattr(envelope_combined, "set_runtime_settings", lambda cfg: None)
        monkeypatch.setattr(envelope_exports, "set_runtime_settings", lambda cfg: None)
        if expect is None:
            rw.main(["--config", str(cfg), *argv])
        else:
            with pytest.raises(expect):
                rw.main(["--config", str(cfg), *argv])

    def yolo(self):
        return [ds for ds, steps in self.pipeline if steps == ("yolo",)]

    def per_trial(self):
        return [ds for ds, steps in self.pipeline if steps != ("yolo",)]


@pytest.fixture
def world(tmp_path, monkeypatch):
    ev.set_protocol("v2")  # conftest restores it
    isolate_env(monkeypatch)
    data = tmp_path / "data"
    for ds in (ADOPT, TRAIN, SENS):
        make_batch(data / ds, BATCH)
    overrides = {ADOPT: {"freeze": {"data": True, "figures": True}}}
    cfg = write_config(tmp_path, [ADOPT, TRAIN, SENS], overrides)
    # ADOPT is frozen TODAY: its slice is what yesterday's wide build cached.
    _prime_slice(tmp_path, cfg, ADOPT)
    return {"tmp": tmp_path, "data": data, "cfg": cfg, "overrides": overrides}


def _prime_slice(tmp_path, cfg, dataset):
    settings = load_settings(cfg)
    with cfg.open() as fh:
        raw = rw._expand_datasets(yaml.safe_load(fh))
    roots = [r for r in raw["main_directories"]]
    out = tmp_path / "prime.csv"
    from scripts.analysis.envelope_combined import build_wide_csv

    build_wide_csv(roots, str(out), measure_cols=["combined_pct", "combined_base"],
                   use_per_trial_baseline=True, config_path=cfg,
                   extra_trial_exports={"training": str(tmp_path / "prime_training.csv")})
    rows = pd.read_csv(out)
    rows = rows[rows["dataset"] == dataset].reset_index(drop=True)
    blocks = rw._freeze_cache_blocks(raw["analysis"]["combined"], settings)
    for block, names, kw in blocks:
        if dataset in names:
            _freeze.save_slice(settings.cache_dir, block, dataset, rows,
                               rw._freeze_fingerprint(settings, dataset, **kw))


def _record(world, ds):
    settings = load_settings(world["cfg"])
    return af.load_record(af.scope_dir(settings.cache_dir, world["cfg"]), ds)


def _outputs(world):
    out = world["tmp"] / "out"
    return {p.name: p.read_bytes() for p in sorted(out.glob("wide*.csv"))}


def test_night_1_adopts_frozen_and_builds_the_rest_once(world, monkeypatch):
    n1 = Night(monkeypatch, world["cfg"])
    assert n1.per_trial() == [TRAIN, SENS]  # ADOPT sits out
    assert n1.combined == [TRAIN, SENS]
    assert _record(world, ADOPT)["mode"] == "adopted"
    for ds in (TRAIN, SENS):
        rec = _record(world, ds)
        assert rec["state"] == "frozen" and rec["inputs"]["dirs"] == [BATCH]
    assert n1.figures[ADOPT] == (True, True)
    assert n1.figures[SENS] == (False, False)
    assert n1.plan_env and Path(n1.plan_env).is_file()
    assert os.environ.get(af.PLAN_ENV) is None  # cleared when main() ends
    assert af.active_plan() is None


def test_a_quiet_night_does_no_work_and_writes_identical_tables(world, monkeypatch):
    Night(monkeypatch, world["cfg"])
    first = _outputs(world)
    n2 = Night(monkeypatch, world["cfg"])
    assert n2.yolo() == [] and n2.per_trial() == [] and n2.combined == []
    assert n2.figures == {ADOPT: (True, True), TRAIN: (True, True), SENS: (True, True)}
    # Byte for byte, last float digit included: the walked rows were cached
    # round-trip exact, so splicing them back writes the same text.
    assert _outputs(world) == first


def test_a_new_batch_with_videos_thaws_only_its_dataset(world, monkeypatch):
    Night(monkeypatch, world["cfg"])
    before = pd.read_csv(world["tmp"] / "out" / "wide.csv")
    new = make_batch(world["data"] / SENS, "august_20_batch_1", processed=False, videos=True,
                     born="2026-08-21", stamp="20260820")

    def _yolo(root, steps):
        # What YOLO + the per-trial steps + combine leave behind.
        if steps == ("yolo",) and root.name == SENS:
            for video in sorted(new.glob("*.mp4")):
                yolo_outputs(new, video.stem)
        if steps != ("yolo",) and root.name == SENS:
            for i, trial in enumerate(("testing_1", "testing_2")):
                combine_outputs(new, trial, seed=10 + i)

    n3 = Night(monkeypatch, world["cfg"], on_pipeline=_yolo)
    assert n3.yolo() == [SENS] and n3.per_trial() == [SENS] and n3.combined == [SENS]
    after = pd.read_csv(world["tmp"] / "out" / "wide.csv")
    assert set(after.loc[after["dataset"] == SENS, "fly"]) == {BATCH, "august_20_batch_1"}
    # The other datasets' rows are byte-for-byte what they were.
    keep = after[after["dataset"] != SENS].reset_index(drop=True)
    pd.testing.assert_frame_equal(keep, before[before["dataset"] != SENS].reset_index(drop=True))
    assert n3.figures[SENS] == (False, False) and n3.figures[ADOPT] == (True, True)
    rec = _record(world, SENS)
    assert rec["state"] == "frozen" and "august_20_batch_1" in rec["inputs"]["dirs"]
    n4 = Night(monkeypatch, world["cfg"])
    assert n4.per_trial() == [] and n4.combined == []


def test_data_false_is_rebuilt_every_night_with_folder_retirements(world, monkeypatch):
    make_batch(world["data"] / TRAIN, "may_22_batch_1_rig_2", stamp="20260522", born="2026-05-15")
    world["overrides"][TRAIN] = {"freeze": {"data": False, "figures": False}}
    cfg = write_config(world["tmp"], [ADOPT, TRAIN, SENS], world["overrides"])
    for _ in range(2):
        night = Night(monkeypatch, cfg)
        assert TRAIN in night.per_trial() and TRAIN in night.combined
        wide = pd.read_csv(world["tmp"] / "out" / "wide.csv")
        marks = dict(zip(wide.loc[wide["dataset"] == TRAIN, "fly"], wide.loc[wide["dataset"] == TRAIN, "frozen"]))
        assert marks == {BATCH: False, "may_22_batch_1_rig_2": True}  # retired by the cutoff
        assert night.figures[TRAIN] == (False, False)


def test_thaw_keeps_folder_retirements_live(world, monkeypatch):
    make_batch(world["data"] / ADOPT, "may_22_batch_1_rig_2", stamp="20260522", born="2026-05-15")
    night = Night(monkeypatch, world["cfg"], "--thaw", ADOPT)
    assert ADOPT in night.per_trial()
    wide = pd.read_csv(world["tmp"] / "out" / "wide.csv")
    sub = wide[wide["dataset"] == ADOPT]
    assert dict(zip(sub["fly"], sub["frozen"])) == {BATCH: False, "may_22_batch_1_rig_2": True}


def test_a_run_that_dies_never_records_its_rebuild(world, monkeypatch):
    Night(monkeypatch, world["cfg"], fail_in="reactions", expect=RuntimeError)
    rec = _record(world, SENS)
    assert rec["state"] == "thawing"
    n2 = Night(monkeypatch, world["cfg"])
    assert SENS in n2.per_trial()
    assert _record(world, SENS)["state"] == "frozen"


def test_a_video_landing_before_its_yolo_slot_is_processed_that_night(world, monkeypatch):
    Night(monkeypatch, world["cfg"])
    late_batch = world["data"] / SENS / BATCH
    stem = f"output_{_short(BATCH)}_testing_3_Hexanol_20260809_123000"

    def _drop(root, steps):
        if steps == ("yolo",):
            yolo_outputs(late_batch, stem)
        if steps != ("yolo",) and root.name == SENS:
            combine_outputs(late_batch, "testing_3", seed=3)

    # Drop the video while the plan is being made: patch scan_inputs once.
    real_plan = af.plan_run

    def _plan_then_drop(*a, **k):
        plan = real_plan(*a, **k)
        (late_batch / f"{stem}.csv").write_text("frame,timestamp\n0,0\n")
        (late_batch / f"{stem}.mp4").write_bytes(b"late video")
        return plan

    monkeypatch.setattr(af, "plan_run", _plan_then_drop)
    night = Night(monkeypatch, world["cfg"], on_pipeline=_drop)
    assert night.yolo() == [SENS] and SENS in night.per_trial()
    wide = pd.read_csv(world["tmp"] / "out" / "wide.csv")
    assert "testing_3_hexanol" in set(wide.loc[wide["dataset"] == SENS, "trial_label"])
    assert night.figures[SENS] == (False, False)
    # Its inputs were re-snapshotted when it thawed, before the rebuild that
    # included the late trial, so the next night has nothing new.
    monkeypatch.setattr(af, "plan_run", real_plan)
    assert _record(world, SENS)["state"] == "frozen"
    assert Night(monkeypatch, world["cfg"]).per_trial() == []


def test_a_corrupt_video_rebuilds_its_dataset_twice_then_freezes(world, monkeypatch):
    """YOLO leaves a corrupt video unprocessed (0 frames decoded) and the
    cleanup now keeps it: two nights try it, then nothing new was added and the
    dataset freezes -- no YOLO launch at its slot either -- with a notice."""
    Night(monkeypatch, world["cfg"])
    stem = f"output_{_short(BATCH)}_testing_3_Hexanol_20260809_123000"
    (world["data"] / SENS / BATCH / f"{stem}.mp4").write_bytes(b"corrupt")
    for _ in range(2):
        night = Night(monkeypatch, world["cfg"])  # YOLO "runs" and writes nothing
        assert night.yolo() == [SENS] and SENS in night.per_trial()
    night = Night(monkeypatch, world["cfg"])
    assert night.yolo() == [] and night.per_trial() == []
    status = (af.scope_dir(load_settings(world["cfg"]).cache_dir, world["cfg"]) / "status.txt").read_text()
    assert f"{SENS}: 1 video(s) YOLO could not process in 2 attempts" in status


def test_thaw_all_no_longer_scores_retired_folders(world, monkeypatch, capsys):
    Night(monkeypatch, world["cfg"], "--thaw-all")
    assert "Frozen experiment folders WILL be scored" not in capsys.readouterr().out


def test_include_frozen_still_scores_retired_folders(world, monkeypatch, capsys):
    Night(monkeypatch, world["cfg"], "--include-frozen")
    assert "Frozen experiment folders WILL be scored" in capsys.readouterr().out


def test_the_run_takes_the_whole_run_lock(world, monkeypatch):
    from fbpipe.utils.run_lock import RunLockTimeout

    settings = load_settings(world["cfg"])
    import fcntl

    lock = Path(settings.cache_dir) / "run_workflows.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT)
    # A different open file description: blocks this process's run too.
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        night = Night(monkeypatch, world["cfg"], expect=RunLockTimeout)
        assert night.pipeline == []
    finally:
        os.close(fd)


def test_figures_only_holds_the_lock_and_plans_nothing(world, monkeypatch):
    night = Night(monkeypatch, world["cfg"], "--figures-only")
    assert night.pipeline == [] and night.plan_env is None
    assert night.figures[ADOPT] == (True, True)  # the explicit flags, as before
    assert night.figures[SENS] == (False, False)


def test_auto_freeze_disabled_keeps_the_explicit_flags(world, monkeypatch):
    cfg = write_config(world["tmp"], [ADOPT, TRAIN, SENS], world["overrides"],
                       auto_freeze={"enabled": False})
    for _ in range(2):
        night = Night(monkeypatch, cfg)
        assert night.per_trial() == [TRAIN, SENS]
        assert night.figures[ADOPT] == (True, True) and night.figures[SENS] == (False, False)
    settings = load_settings(cfg)
    assert af.load_record(af.scope_dir(settings.cache_dir, cfg), SENS) is None


def test_the_plan_reaches_figure_subprocesses(world, monkeypatch):
    seen = {}

    def _reactions(settings, config_path=None):
        code = (
            "import json, sys; from fbpipe.freeze import figure_frozen_from_raw; "
            "print(json.dumps(sorted(figure_frozen_from_raw({}))))"
        )
        env = dict(os.environ, PYTHONPATH=str(Path(af.__file__).resolve().parents[1]))
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        seen["frozen"] = json.loads(out.stdout.strip().splitlines()[-1])

    Night(monkeypatch, world["cfg"])
    monkeypatch.setattr(rw, "_run_reactions", _reactions)
    # Night 2: everything frozen, so the subprocess must see all three held,
    # though only ADOPT has a freeze block in config.
    isolate_env(monkeypatch)
    monkeypatch.setenv("FBPIPE_DISABLE_NTFY", "1")
    monkeypatch.setenv("FBPIPE_RUN_LOCK_WAIT_MIN", "0")
    monkeypatch.setattr(rw, "_run_pipeline", lambda *a, **k: None)
    rw.main(["--config", str(world["cfg"])])
    assert {ADOPT, TRAIN, SENS} <= set(seen["frozen"])


def test_freeze_status_previews_tonight_and_writes_nothing(world, monkeypatch, capsys):
    isolate_env(monkeypatch)
    cache = world["tmp"] / "cache"
    before = {p: p.stat().st_mtime_ns for p in cache.rglob("*") if p.is_file()}
    rw.main(["--config", str(world["cfg"]), "--freeze-status"])
    out = capsys.readouterr().out
    assert f"[FREEZE] {ADOPT}: FROZEN - adopted" in out
    assert f"[FREEZE] {SENS}: THAWED - no freeze record yet" in out
    after = {p: p.stat().st_mtime_ns for p in cache.rglob("*") if p.is_file()}
    assert after == before
    assert not (cache / "run_workflows.lock").exists()


def test_freeze_status_never_sends_the_end_of_run_notification():
    """It is a read-only report: the script's __main__ must not ntfy
    "Pipeline Complete" for it (checked on the source -- running __main__ would
    post to the user's phone)."""
    src = Path(rw.__file__).read_text(encoding="utf-8")
    tail = src[src.index('if __name__ == "__main__"'):]
    status_branch, rest = tail.split("elif __name__", 1)
    assert '"--freeze-status" in sys.argv[1:]' in status_branch
    assert "ntfy_notify" not in status_branch
    assert "ntfy_notify(\"Pipeline Complete\"" in rest
