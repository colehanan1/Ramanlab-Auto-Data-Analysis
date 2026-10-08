"""Auto-freeze decisions, one per dataset per run (fbpipe.autofreeze.plan_run).

The user's rule: unfreeze (rebuild) a dataset only when something new was
added to it; otherwise freeze it -- unless the user explicitly unfreezes it
(``data: false``, ``--thaw``, ``--thaw-all``). A missing record, an unfinished
rebuild or a cache that fails the current fingerprint is rebuilt once. A
dataset frozen today with a valid cache is adopted without a rebuild, its
byte-identical duplicate rows removed. Cascades are only reported.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fbpipe import autofreeze as af
from fbpipe import freeze as _freeze
from fbpipe.config import load_settings
from scripts.pipeline import run_workflows as rw

from _loop_harness import isolate_env

FROZEN = "EB-Control-24-1"
PARTNER = "EB-Training-24-1"
LIVE = "IAA-Sensitivity-12-1"
BATCH = "august_09_batch_1_rig_2"


def _stem(batch=BATCH, trial="testing_1", stamp="20260809_120000"):
    # Rigs name their files after the batch WITHOUT its rig suffix.
    return f"output_{'_'.join(batch.split('_')[:4])}_{trial}_Hexanol_{stamp}"


def _batch(root: Path, name=BATCH, trials=("testing_1",), processed=True, stamp="20260809"):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "session_metadata.txt").write_text("Born (approx): 2026-08-01\nHost: Flybehavior2\n")
    for i, trial in enumerate(trials):
        stem = _stem(name, trial, f"{stamp}_12{i:02d}00")
        (d / f"{stem}.csv").write_text("frame,timestamp\n0,0\n")
        if processed:
            tdir = d / f"{'_'.join(name.split('_')[:4])}_{trial}"
            tdir.mkdir(exist_ok=True)
            (tdir / f"{tdir.name}_distances_merged.parquet").write_bytes(b"m")
            (tdir / f"{tdir.name}_fly1_distances.parquet").write_bytes(b"f")
    return d


def _config(tmp_path, datasets, overrides, **top):
    cfg = {
        "dataset_bases": {"data": str(tmp_path / "data"), "secured": str(tmp_path / "secured")},
        "datasets": list(datasets),
        "dataset_overrides": overrides,
        "cache_dir": str(tmp_path / "cache"),
        "model_path": str(tmp_path / "model.pt"),
        "proboscis_filter": {"max_eye_prob_distance_px": 170.0},
        "distance_limits": {"class2_min": 10.0, "class2_max": 160.0,
                            "three_fly_max_eye_prob_distance_px": 150.0},
        "analysis": {"combined": {"combine": {"roots": []}, "combined_base": {"wide": {
            "roots": [], "output_csv": str(tmp_path / "out" / "wide.csv"),
            "measure_cols": ["combined_pct", "combined_base"]}}}},
        **top,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


class Scene:
    def __init__(self, tmp_path, monkeypatch, *, overrides=None, datasets=(FROZEN, PARTNER, LIVE), **top):
        isolate_env(monkeypatch)
        monkeypatch.setenv("FBPIPE_DISABLE_NTFY", "1")
        self.tmp = tmp_path
        self.datasets = list(datasets)
        for ds in self.datasets:
            _batch(tmp_path / "data" / ds)
        if overrides is None:
            overrides = {FROZEN: {"freeze": {"data": True, "figures": True}},
                         PARTNER: {"freeze": {"data": True, "figures": True}}}
        self.overrides = overrides
        self.top = top
        self.reload()

    def reload(self):
        self.cfg = _config(self.tmp, self.datasets, self.overrides, **self.top)
        self.settings = load_settings(self.cfg)
        with self.cfg.open() as fh:
            data = rw._expand_datasets(yaml.safe_load(fh))
        self.roots = [str(Path(r)) for r in data["main_directories"]]
        self.blocks = rw._freeze_cache_blocks(data["analysis"]["combined"], self.settings)

    def root(self, ds):
        return self.tmp / "data" / ds

    def fingerprint(self, ds, kw):
        return rw._freeze_fingerprint(self.settings, ds, **kw)

    def seed_slice(self, ds, rows=None, fingerprint=None):
        if rows is None:
            rows = pd.DataFrame({"dataset": [ds], "fly": [BATCH], "fly_number": [1],
                                 "trial_type": ["testing"], "trial_label": ["testing_1_hexanol"],
                                 "frozen": [False], "trace_len": [2],
                                 "dir_val_0": [0.1], "dir_val_1": [0.2]})
        for block, names, kw in self.blocks:
            if ds in names:
                _freeze.save_slice(self.settings.cache_dir, block, ds, rows,
                                   fingerprint or self.fingerprint(ds, kw))

    def plan(self, **kw):
        kw.setdefault("pairs", [(PARTNER, FROZEN)])
        return af.plan_run(self.settings, config_path=self.cfg, roots=self.roots,
                           blocks=self.blocks, fingerprint=self.fingerprint, **kw)

    def scope(self):
        return af.scope_dir(self.settings.cache_dir, self.cfg)

    def record(self, ds):
        return af.load_record(self.scope(), ds)

    def build_and_commit(self, plan, datasets=None):
        """What a successful run does: rebuild (here: cache) every walked
        dataset's rows, then commit at the end."""
        for ds, d in plan.decisions.items():
            if d.decision != "frozen" and (datasets is None or ds in datasets):
                self.seed_slice(ds)
        return plan.commit(combined_ran=True)


@pytest.fixture
def scene(tmp_path, monkeypatch):
    return Scene(tmp_path, monkeypatch)


def _settle(scene):
    """Night 1: adopt the frozen pair, build the live one, commit."""
    for ds in (FROZEN, PARTNER):
        scene.seed_slice(ds)
    plan = scene.plan()
    scene.build_and_commit(plan)
    return plan


# --- first run: adopt or build ---------------------------------------------------


def test_a_dataset_frozen_today_with_a_valid_cache_is_adopted(scene):
    scene.seed_slice(FROZEN)
    scene.seed_slice(PARTNER)
    plan = scene.plan()
    d = plan.decisions[FROZEN]
    assert d.decision == "frozen" and d.adopted and not d.commit
    assert d.reasons[0].startswith("adopted")
    assert d.figures_frozen is True
    rec = scene.record(FROZEN)
    assert rec["state"] == "frozen" and rec["mode"] == "adopted"
    assert rec["inputs"]["dirs"] == [BATCH]


def test_a_dataset_without_a_freeze_block_is_built_once(scene):
    plan = scene.plan()
    d = plan.decisions[LIVE]
    assert d.decision == "thawed" and d.commit
    assert "no freeze record yet" in d.reasons[0]
    assert d.figures_frozen is False
    assert scene.record(LIVE)["state"] == "thawing"


def test_adoption_needs_a_valid_cache(scene):
    plan = scene.plan()  # no slices at all
    d = plan.decisions[FROZEN]
    assert d.decision == "thawed" and "cache missing" in d.reasons[0]


def test_adoption_needs_every_processed_batch_in_the_cached_rows(scene):
    """A batch YOLO processed after the dataset was frozen never reached its
    rows: something new was added, so it is rebuilt, not adopted."""
    _batch(scene.root(FROZEN), "august_14_batch_1_rig_3")
    scene.seed_slice(FROZEN)
    scene.seed_slice(PARTNER)
    d = scene.plan().decisions[FROZEN]
    assert d.decision == "thawed"
    assert "processed batch folder(s) missing from its cached rows (august_14_batch_1_rig_3)" in d.reasons[0]


def test_adoption_removes_exact_duplicate_rows_losslessly(scene):
    base = pd.DataFrame({
        "dataset": [FROZEN] * 3, "fly": [BATCH] * 3, "fly_number": [1, 2, 3],
        "trial_type": ["testing"] * 3, "trial_label": ["testing_1_hexanol"] * 3,
        "frozen": [False] * 3, "trace_len": [2] * 3,
        "dir_val_0": [0.1, np.nan, 0.3], "dir_val_1": [0.2, 0.4, np.nan],
    })
    # The doubled block the real slices carry, plus a same-key row with a
    # different value: that one is NOT a duplicate and must stay.
    collision = base.iloc[[0]].assign(dir_val_0=9.9)
    rows = pd.concat([base, base, collision], ignore_index=True)
    scene.seed_slice(FROZEN, rows)
    scene.seed_slice(PARTNER)
    plan = scene.plan()
    d = plan.decisions[FROZEN]
    assert d.deduped == {"combined_base": (7, 4)}
    assert d.figures_frozen is False  # its rows changed: figures redraw once
    kept = pd.read_parquet(_freeze.slice_dir(scene.settings.cache_dir, "combined_base", FROZEN) / "rows.parquet")
    expect = pd.concat([base, collision], ignore_index=True)
    pd.testing.assert_frame_equal(kept, expect)
    assert scene.record(FROZEN)["figures_pending"] is True
    # The slice stays valid under its fingerprint.
    kw = next(kw for b, n, kw in scene.blocks if FROZEN in n)
    assert _freeze.slice_status(scene.settings.cache_dir, "combined_base", FROZEN, scene.fingerprint(FROZEN, kw)) == (True, "")
    # After a successful run the figures are held again.
    plan.commit(combined_ran=True)
    assert "figures_pending" not in scene.record(FROZEN)
    assert scene.plan().decisions[FROZEN].figures_frozen is True


# --- steady state ------------------------------------------------------------------


def test_nothing_new_stays_frozen(scene):
    _settle(scene)
    plan = scene.plan()
    assert {ds: d.decision for ds, d in plan.decisions.items()} == {
        FROZEN: "frozen", PARTNER: "frozen", LIVE: "frozen"}
    assert plan.decisions[LIVE].reasons == ["nothing new"]
    assert all(d.figures_frozen for d in plan.decisions.values())


@pytest.mark.parametrize(
    "change,expect",
    [
        (lambda r: _batch(r, "august_15_batch_1_rig_2"), "+1 batch folder(s) (august_15_batch_1_rig_2)"),
        (lambda r: (r / BATCH / "odor_vials.csv").write_text("vials"), "+1 raw file(s)"),
        (lambda r: (r / BATCH / "session_metadata.txt").write_text("Born (approx): 2026-08-02\n"), "~1 raw file(s)"),
        (lambda r: (r / BATCH / f"{_stem(BATCH, 'testing_1', '20260809_120000')}.csv").unlink(), "-1 raw file(s)"),
    ],
    ids=["new batch", "new raw file", "changed raw file", "removed raw file"],
)
def test_something_new_thaws_only_that_dataset(scene, change, expect):
    _settle(scene)
    change(scene.root(LIVE))
    plan = scene.plan()
    assert plan.decisions[LIVE].decision == "thawed"
    assert plan.decisions[LIVE].reasons[0].startswith("new data:")
    assert expect in plan.decisions[LIVE].reasons[0]
    assert plan.decisions[FROZEN].decision == "frozen"
    assert scene.record(LIVE)["state"] == "thawing"


def test_a_removed_batch_folder_thaws(scene):
    _batch(scene.root(LIVE), "august_15_batch_1_rig_2")
    _settle(scene)
    import shutil

    shutil.rmtree(scene.root(LIVE) / "august_15_batch_1_rig_2")
    d = scene.plan().decisions[LIVE]
    assert d.decision == "thawed" and "-1 batch folder(s)" in d.reasons[0]


def test_a_touched_file_stays_frozen_and_its_stat_is_refreshed(scene):
    _settle(scene)
    target = scene.root(LIVE) / BATCH / "session_metadata.txt"
    st = target.stat()
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 3_000_000_000))
    d = scene.plan().decisions[LIVE]
    assert d.decision == "frozen" and "touched, same content" in d.reasons[0]
    rec = scene.record(LIVE)
    assert rec["inputs"]["files"][f"{BATCH}/session_metadata.txt"][1] == st.st_mtime_ns + 3_000_000_000
    assert scene.plan().decisions[LIVE].reasons == ["nothing new"]


def test_pipeline_writes_never_thaw(scene):
    _settle(scene)
    batch = scene.root(LIVE) / BATCH
    (batch / "fly1_global_distance_stats_class_0.json").write_text("{}")
    (batch / "angle_distance_rms_envelope").mkdir()
    (batch / "angle_distance_rms_envelope" / "x.csv").write_text("x")
    assert scene.plan().decisions[LIVE].decision == "frozen"


def test_a_video_yolo_would_process_thaws(scene):
    _settle(scene)
    (scene.root(LIVE) / BATCH / f"{_stem(BATCH, 'testing_2', '20260809_121000')}.mp4").write_bytes(b"v")
    d = scene.plan().decisions[LIVE]
    assert d.decision == "thawed" and "1 video(s) YOLO has not processed" in d.reasons[0]


def test_a_processed_video_does_not_thaw(scene):
    """Its trial folder exists: YOLO skips it, the cleanup may delete it."""
    _settle(scene)
    video = scene.root(LIVE) / BATCH / f"{_stem(BATCH, 'testing_1', '20260809_120000')}.mp4"
    video.write_bytes(b"v")
    assert scene.plan().decisions[LIVE].decision == "frozen"
    video.unlink()  # the secure cleanup
    assert scene.plan().decisions[LIVE].decision == "frozen"


def test_a_video_in_a_retired_folder_does_not_thaw(scene):
    """yolo_infer never infers a retired folder, so nothing would change."""
    scene.overrides[LIVE] = {"freeze": {"folders": ["*_rig_3"]}}
    scene.reload()
    _batch(scene.root(LIVE), "august_10_batch_1_rig_3", processed=False)
    _settle(scene)
    (scene.root(LIVE) / "august_10_batch_1_rig_3" / f"{_stem('august_10_batch_1_rig_3', 'testing_2')}.mp4").write_bytes(b"v")
    assert scene.plan().decisions[LIVE].decision == "frozen"


# --- explicit unfreeze ---------------------------------------------------------------


def test_data_false_is_live_every_run(scene):
    scene.overrides[LIVE] = {"freeze": {"data": False, "figures": False}}
    scene.reload()
    _settle(scene)
    for _ in range(2):
        plan = scene.plan()
        d = plan.decisions[LIVE]
        assert d.decision == "live" and d.commit and d.figures_frozen is False
        scene.build_and_commit(plan)


def test_a_live_dataset_is_logged_as_live_not_as_frozen(scene):
    scene.overrides[LIVE] = {"freeze": {"data": False, "figures": False}}
    scene.reload()
    _settle(scene)
    lines = scene.build_and_commit(scene.plan())
    mine = [line for line in lines if f"{LIVE}:" in line]
    assert len(mine) == 1
    assert "stays live (freeze.data: false)" in mine[0]
    assert "recorded frozen" not in mine[0]


def test_removing_data_false_freezes_with_no_rebuild(scene):
    """The live runs keep its record current: once the user drops the explicit
    unfreeze, a dataset that got nothing new freezes on the very next run."""
    scene.overrides[LIVE] = {"freeze": {"data": False, "figures": False}}
    scene.reload()
    _settle(scene)
    scene.build_and_commit(scene.plan())
    scene.overrides[LIVE] = {}
    scene.reload()
    d = scene.plan().decisions[LIVE]
    assert d.decision == "frozen" and d.reasons == ["nothing new"]


def test_removing_data_false_after_new_data_rebuilds_once(scene):
    scene.overrides[LIVE] = {"freeze": {"data": False, "figures": False}}
    scene.reload()
    _settle(scene)
    plan = scene.plan()
    _batch(scene.root(LIVE), "august_15_batch_1_rig_2")  # arrives after this run's scan
    scene.build_and_commit(plan)
    scene.overrides[LIVE] = {}
    scene.reload()
    d = scene.plan().decisions[LIVE]
    assert d.decision == "thawed" and "new data: +1 batch folder(s) (august_15_batch_1_rig_2)" in d.reasons[0]


def test_thaw_rebuilds_one_dataset_and_refreezes_it(scene):
    _settle(scene)
    plan = scene.plan(thawed=[FROZEN])
    assert plan.decisions[FROZEN].decision == "thawed"
    assert plan.decisions[FROZEN].reasons == ["--thaw: rebuilt this run"]
    assert plan.decisions[PARTNER].decision == "frozen"
    scene.build_and_commit(plan)
    assert scene.record(FROZEN)["state"] == "frozen"
    assert scene.plan().decisions[FROZEN].decision == "frozen"


def test_thaw_all_rebuilds_everything(scene):
    _settle(scene)
    plan = scene.plan(thaw_all=True)
    assert {d.decision for d in plan.decisions.values()} == {"thawed"}


# --- repairs ----------------------------------------------------------------------------


def test_a_run_that_died_leaves_its_dataset_thawing(scene):
    _settle(scene)
    _batch(scene.root(LIVE), "august_15_batch_1_rig_2")
    scene.plan()  # marks LIVE thawing ... and the run dies: no commit
    d = scene.plan().decisions[LIVE]
    assert d.decision == "thawed"
    assert "the previous rebuild did not finish" in d.reasons[0]


def test_a_config_edit_that_changes_rows_rebuilds_once(scene):
    _settle(scene)
    scene.top["distance_limits"] = {"class2_min": 10.0, "class2_max": 175.0,
                                    "three_fly_max_eye_prob_distance_px": 150.0}
    scene.reload()
    plan = scene.plan()
    d = plan.decisions[FROZEN]
    assert d.decision == "thawed" and "fingerprint drift" in d.reasons[0]
    assert "folder rules live" in d.reasons[0]
    scene.build_and_commit(plan)
    assert scene.plan().decisions[FROZEN].decision == "frozen"


def test_a_folder_policy_edit_that_flips_no_mark_is_restamped(scene):
    """Adding cutoff_exempt_before to a finished dataset whose rows are all
    live changes its fingerprint but no row: no rebuild."""
    _settle(scene)
    scene.overrides[FROZEN]["freeze"]["cutoff_exempt_before"] = "2026-06-26"
    scene.reload()
    plan = scene.plan()
    assert plan.decisions[FROZEN].decision == "frozen"
    assert any("re-stamped" in n for n in plan.notices)
    kw = next(kw for b, n, kw in scene.blocks if FROZEN in n)
    assert _freeze.slice_status(scene.settings.cache_dir, "combined_base", FROZEN,
                                scene.fingerprint(FROZEN, kw)) == (True, "")


def test_a_folder_policy_edit_that_flips_a_mark_rebuilds(scene):
    _settle(scene)
    scene.overrides[FROZEN]["freeze"]["folders"] = [BATCH]
    scene.reload()
    d = scene.plan().decisions[FROZEN]
    assert d.decision == "thawed"
    assert "the rules now mark 1 folder(s) differently" in d.reasons[0]


# --- --folder ------------------------------------------------------------------------------


def test_folder_mode_rebuilds_its_target_and_leaves_the_rest(scene):
    _settle(scene)
    _batch(scene.root(PARTNER), "august_15_batch_1_rig_2")
    plan = scene.plan(folder=LIVE)
    assert plan.decisions[LIVE].decision == "thawed" and plan.decisions[LIVE].commit
    assert plan.decisions[PARTNER].decision == "untouched"
    assert plan.decisions[PARTNER].commit is False
    assert plan.decisions[FROZEN].decision == "frozen"


# --- commit -------------------------------------------------------------------------------------


def test_commit_records_the_inputs_seen_at_the_start_of_the_run(scene):
    _settle(scene)
    _batch(scene.root(LIVE), "august_15_batch_1_rig_2")
    plan = scene.plan()
    # A batch landing mid-run is new to the NEXT run.
    _batch(scene.root(LIVE), "august_16_batch_1_rig_2")
    scene.build_and_commit(plan)
    rec = scene.record(LIVE)
    assert rec["state"] == "frozen"
    assert "august_16_batch_1_rig_2" not in rec["inputs"]["dirs"]
    d = scene.plan().decisions[LIVE]
    assert d.decision == "thawed" and "august_16_batch_1_rig_2" in d.reasons[0]


def test_commit_refuses_a_dataset_whose_rows_were_not_cached(scene):
    plan = scene.plan()
    lines = plan.commit(combined_ran=True)  # nothing cached LIVE's rows
    assert any(f"{LIVE}: NOT refrozen" in line for line in lines)
    assert scene.record(LIVE)["state"] == "thawing"


def test_commit_refuses_when_the_combined_stage_did_not_run(scene):
    plan = scene.plan()
    for ds in scene.datasets:
        scene.seed_slice(ds)
    plan.commit(combined_ran=False)
    assert scene.record(LIVE)["state"] == "thawing"


def test_an_empty_dataset_freezes_with_an_empty_cache(tmp_path, monkeypatch):
    """RandomPanel-Control-24-10 has no batch at all: nothing to cache, so it
    must not stay 'missing or stale' -- and rebuilt -- every night."""
    s = Scene(tmp_path, monkeypatch, datasets=(FROZEN, "RandomPanel-Control-24-10"),
              overrides={"RandomPanel-Control-24-10": {"freeze": {"data": True}}})
    import shutil

    shutil.rmtree(s.root("RandomPanel-Control-24-10") / BATCH)
    plan = s.plan(pairs=[])
    assert plan.decisions["RandomPanel-Control-24-10"].decision == "thawed"
    s.seed_slice(FROZEN)
    plan.commit(combined_ran=True)
    assert s.record("RandomPanel-Control-24-10")["state"] == "frozen"
    assert s.plan(pairs=[]).decisions["RandomPanel-Control-24-10"].decision == "frozen"


# --- cascades are reported, never acted on ----------------------------------------------------


def test_a_frozen_pair_arm_is_reported_when_its_partner_gets_new_data(scene):
    _settle(scene)
    _batch(scene.root(PARTNER), "august_15_batch_1_rig_2")
    plan = scene.plan()
    assert plan.decisions[PARTNER].decision == "thawed"
    assert plan.decisions[FROZEN].decision == "frozen"
    notice = next(n for n in plan.notices if "cascade" in n and "pair arm" in n)
    assert f"{PARTNER} got new data" in notice
    assert f"--thaw {FROZEN}" in notice and "run_workflows.py --config" in notice


def test_frozen_datasets_built_under_other_code_are_reported(scene, monkeypatch):
    _settle(scene)
    real = af.code_provenance
    monkeypatch.setattr(af, "code_provenance", lambda: {**real(), "src/fbpipe/steps/reject_bad_proboscis.py": "changed"})
    plan = scene.plan()
    assert {d.decision for d in plan.decisions.values()} == {"frozen"}
    notice = next(n for n in plan.notices if "per-trial code changed" in n)
    assert "src/fbpipe/steps/reject_bad_proboscis.py" in notice
    for ds in (FROZEN, PARTNER, LIVE):
        assert f"--thaw {ds}" in notice


def test_frozen_datasets_built_under_other_settings_are_reported(scene):
    _settle(scene)
    scene.top["proboscis_filter"] = {"max_eye_prob_distance_px": 190.0}
    scene.reload()
    plan = scene.plan()
    notice = next(n for n in plan.notices if "per-trial settings changed" in n)
    assert "proboscis_filter" in notice


# --- isolation --------------------------------------------------------------------------------------


def test_records_are_scoped_by_config(scene, tmp_path):
    _settle(scene)
    other = tmp_path / "other.yaml"
    other.write_text(scene.cfg.read_text())
    plan = af.plan_run(scene.settings, config_path=other, roots=scene.roots, blocks=scene.blocks,
                       fingerprint=scene.fingerprint, read_only=True)
    assert plan.decisions[LIVE].decision == "thawed"
    assert "no freeze record yet" in plan.decisions[LIVE].reasons[0]


def test_a_read_only_plan_writes_nothing(scene):
    scene.seed_slice(FROZEN, pd.concat([pd.DataFrame({
        "dataset": [FROZEN], "fly": [BATCH], "fly_number": [1], "trial_type": ["testing"],
        "trial_label": ["testing_1_hexanol"], "frozen": [False], "trace_len": [2],
        "dir_val_0": [0.1], "dir_val_1": [0.2]})] * 2, ignore_index=True))
    scene.seed_slice(PARTNER)
    before = {p: p.stat().st_mtime_ns for p in Path(scene.settings.cache_dir).rglob("*") if p.is_file()}
    plan = scene.plan(read_only=True)
    assert plan.decisions[FROZEN].decision == "frozen"
    assert plan.decisions[FROZEN].deduped == {"combined_base": (2, 1)}
    after = {p: p.stat().st_mtime_ns for p in Path(scene.settings.cache_dir).rglob("*") if p.is_file()}
    assert after == before


def test_a_dataset_no_wide_table_reads_freezes_after_one_build(scene):
    """Nothing to splice, so frozen means only that its per-trial work, whose
    inputs did not change, is skipped."""
    scene.blocks = [(b, frozenset(n - {LIVE}), kw) for b, n, kw in scene.blocks]
    plan = scene.plan()
    assert plan.decisions[LIVE].decision == "thawed"
    for ds in (FROZEN, PARTNER):
        scene.seed_slice(ds)
    plan.commit(combined_ran=True)
    assert scene.record(LIVE)["state"] == "frozen"
    assert scene.plan().decisions[LIVE].decision == "frozen"
