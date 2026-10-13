"""Auto-freeze decisions, one per dataset per run (fbpipe.autofreeze.plan_run).

The user's rule: unfreeze (rebuild) a dataset only when something new was
added to it; otherwise freeze it -- unless the user explicitly unfreezes it
(``data: false``, ``--thaw``, ``--thaw-all``). A missing record, an unfinished
rebuild or a cache that fails the current fingerprint is rebuilt once. A
dataset frozen today with a valid cache is adopted without a rebuild, its
byte-identical duplicate rows removed. Cascades are only reported.
"""

from __future__ import annotations

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


def _rows(ds, batches):
    return pd.DataFrame({"dataset": [ds] * len(batches), "fly": list(batches),
                         "fly_number": [1] * len(batches), "trial_type": ["testing"] * len(batches),
                         "trial_label": ["testing_1_hexanol"] * len(batches), "frozen": [False] * len(batches),
                         "trace_len": [2] * len(batches), "dir_val_0": [0.1] * len(batches),
                         "dir_val_1": [0.2] * len(batches)})


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

    def seed_slice(self, ds, rows=None, fingerprint=None, run=None):
        """Cache *ds*'s rows; *run* stamps them as written by that run's walk.

        By default one row per processed batch folder, as a real build emits.
        """
        if rows is None:
            batches = sorted(af.processed_batches(self.root(ds)) | {BATCH}, key=lambda b: (b != BATCH, b))
            rows = _rows(ds, batches)
        for block, names, kw in self.blocks:
            if ds in names:
                _freeze.save_slice(self.settings.cache_dir, block, ds, rows,
                                   fingerprint or self.fingerprint(ds, kw), built_by=run)

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
                self.seed_slice(ds, run=plan.run_id)
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
    scene.seed_slice(FROZEN, rows=_rows(FROZEN, [BATCH]))  # cached before that batch
    scene.seed_slice(PARTNER)
    d = scene.plan().decisions[FROZEN]
    assert d.decision == "thawed"
    assert "processed batch folder(s) missing from its cached rows (august_14_batch_1_rig_3)" in d.reasons[0]


def _doubled_rows():
    base = pd.DataFrame({
        "dataset": [FROZEN] * 3, "fly": [BATCH] * 3, "fly_number": [1, 2, 3],
        "trial_type": ["testing"] * 3, "trial_label": ["testing_1_hexanol"] * 3,
        "frozen": [False] * 3, "trace_len": [2] * 3,
        "dir_val_0": [0.1, np.nan, 0.3], "dir_val_1": [0.2, 0.4, np.nan],
    })
    # The doubled block the real slices carry, plus a same-key row with a
    # different value: that one is NOT a duplicate and must stay.
    collision = base.iloc[[0]].assign(dir_val_0=9.9)
    return pd.concat([base, base, collision], ignore_index=True), pd.concat([base, collision], ignore_index=True)


def test_adoption_waits_for_a_video_yolo_has_not_processed(scene):
    """A frozen-today dataset with a new video is not adopted as it is: the
    video is something new, so the dataset is built once with it."""
    scene.seed_slice(FROZEN)
    scene.seed_slice(PARTNER)
    stem = _stem(BATCH, "testing_2", "20260809_121500")
    (scene.root(FROZEN) / BATCH / f"{stem}.csv").write_text("frame,timestamp\n0,0\n")
    (scene.root(FROZEN) / BATCH / f"{stem}.mp4").write_bytes(b"video")
    d = scene.plan().decisions[FROZEN]
    assert d.decision == "thawed" and not d.adopted
    assert d.reasons == ["no freeze record yet; 1 video(s) YOLO has not processed"]


def test_a_record_from_another_root_is_rebuilt_once(scene):
    """The dataset's folder moved (same name, other base): its record's inputs
    describe the old folder, so they prove nothing -- rebuilt once."""
    _settle(scene)
    record = scene.record(LIVE)
    record["root"] = str(scene.tmp / "elsewhere" / LIVE)
    af.save_record(scene.scope(), LIVE, record)
    d = scene.plan().decisions[LIVE]
    assert d.decision == "thawed"
    assert d.reasons == ["freeze record belongs to another root: rebuilt once"]


def test_adoption_removes_exact_duplicate_rows_losslessly(scene):
    rows, expect = _doubled_rows()
    scene.seed_slice(FROZEN, rows)
    scene.seed_slice(PARTNER)
    plan = scene.plan()
    d = plan.decisions[FROZEN]
    assert d.deduped == {"combined_base": (7, 4)}
    kept = pd.read_parquet(_freeze.slice_dir(scene.settings.cache_dir, "combined_base", FROZEN) / "rows.parquet")
    pd.testing.assert_frame_equal(kept, expect)
    # The slice stays valid under its fingerprint.
    kw = next(kw for b, n, kw in scene.blocks if FROZEN in n)
    assert _freeze.slice_status(scene.settings.cache_dir, "combined_base", FROZEN, scene.fingerprint(FROZEN, kw)) == (True, "")


def test_a_deduped_dataset_with_frozen_figures_keeps_its_published_figures(scene):
    """``figures: true`` means published: those figures were drawn before the
    duplicates existed, and a redraw would re-score them with today's model.
    The dedupe reaches the data only; the notice names the command that
    redraws them from the deduplicated rows."""
    rows, _ = _doubled_rows()
    scene.seed_slice(FROZEN, rows)
    scene.seed_slice(PARTNER)
    plan = scene.plan()
    d = plan.decisions[FROZEN]
    assert d.deduped and d.figures_mode == "hold"
    assert d.figures_frozen is True
    assert "figures_pending" not in scene.record(FROZEN)
    notice = next(n for n in plan.notices if n.startswith(f"[FREEZE] {FROZEN}:") and "duplicate" in n)
    assert "3 exact duplicate row(s) removed" in notice
    assert f"--figures-only --thaw {FROZEN}" in notice
    plan.commit(combined_ran=True)
    assert scene.plan().decisions[FROZEN].figures_frozen is True


def test_a_deduped_dataset_whose_figures_were_live_redraws_them_once(scene):
    """Without ``figures: true`` its figures were redrawn every night from the
    duplicated rows, so they redraw once from the deduplicated ones."""
    scene.overrides[FROZEN] = {"freeze": {"data": True}}
    scene.reload()
    rows, _ = _doubled_rows()
    scene.seed_slice(FROZEN, rows)
    scene.seed_slice(PARTNER)
    plan = scene.plan()
    d = plan.decisions[FROZEN]
    assert d.deduped and d.figures_mode == "auto"
    assert d.figures_frozen is False  # its rows changed: figures redraw once
    assert scene.record(FROZEN)["figures_pending"] is True
    assert not any(n.startswith(f"[FREEZE] {FROZEN}:") and "duplicate" in n for n in plan.notices)
    # After a successful run the figures are held again.
    plan.commit(combined_ran=True)
    assert "figures_pending" not in scene.record(FROZEN)
    assert scene.plan().decisions[FROZEN].figures_frozen is True


def test_adoption_keeps_rows_whose_marks_the_folder_rules_disagree_with(scene):
    """A data-frozen dataset's rows were built with the folder rules OFF, so its
    slice can mark a folder live that today's always-live rules retire -- under
    an unchanged fingerprint. Adopting keeps the rows exactly as they are (that
    is what freezing means) and says so: its first rebuild would retire it."""
    scene.overrides[FROZEN] = {"freeze": {"data": True, "figures": True, "folders": [BATCH]}}
    scene.reload()
    scene.seed_slice(FROZEN)  # BATCH marked frozen=False, current fingerprint
    scene.seed_slice(PARTNER)
    plan = scene.plan()
    d = plan.decisions[FROZEN]
    assert d.decision == "frozen" and d.adopted
    notice = next(n for n in plan.notices if n.startswith(f"[FREEZE] {FROZEN}:") and "folder rules" in n)
    assert f"would retire 1 folder(s) its rows show today ({BATCH})" in notice
    rows = pd.read_parquet(_freeze.slice_dir(scene.settings.cache_dir, "combined_base", FROZEN) / "rows.parquet")
    assert rows["frozen"].tolist() == [False]  # spliced as it is


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


def test_a_video_yolo_cannot_process_stops_thawing_after_two_attempts(scene):
    """A corrupt video (YOLO decodes 0 frames and leaves the trial for the next
    run -- IAA-Sensitivity-12-1 october_06_batch_1's mirror copies) is never
    processed and, now, never deleted. Two attempts on the same bytes; after
    that nothing new was added, so the dataset freezes and the video is
    reported instead. A re-pushed (changed) file is new data again."""
    _settle(scene)
    video = scene.root(LIVE) / BATCH / f"{_stem(BATCH, 'testing_2', '20260809_121000')}.mp4"
    video.write_bytes(b"corrupt")
    for attempt in (1, 2):
        plan = scene.plan()
        d = plan.decisions[LIVE]
        assert d.decision == "thawed" and "1 video(s) YOLO has not processed" in d.reasons[0]
        scene.build_and_commit(plan)  # YOLO ran; the video is still unprocessed
        rel = f"{BATCH}/{video.name}"
        assert scene.record(LIVE)["yolo_attempts"][rel][2] == attempt
    plan = scene.plan()
    d = plan.decisions[LIVE]
    assert d.decision == "frozen" and d.reasons == ["nothing new"]
    notice = next(n for n in plan.notices if "could not process" in n)
    assert notice.startswith(f"[FREEZE] {LIVE}: 1 video(s) YOLO could not process in 2 attempts")
    assert video.name in notice
    assert plan.pending_now(scene.root(LIVE)) == []  # its YOLO slot does not thaw it either
    video.write_bytes(b"re-pushed, complete")
    assert scene.plan().decisions[LIVE].decision == "thawed"


def test_a_video_that_arrives_after_the_yolo_slot_is_not_an_attempt(scene):
    """Videos YOLO never saw this run (they landed after its slot) keep a clean
    count: the next run processes them as usual."""
    _settle(scene)
    plan = scene.plan()  # nothing pending: frozen
    assert plan.decisions[LIVE].decision == "frozen"
    _batch(scene.root(LIVE), "august_15_batch_1_rig_2")  # rebuild LIVE for a new batch ...
    plan = scene.plan()
    late = scene.root(LIVE) / BATCH / f"{_stem(BATCH, 'testing_3', '20260809_123000')}.mp4"
    late.write_bytes(b"landed after the slot")  # ... and a video lands mid-run
    scene.build_and_commit(plan)
    assert not scene.record(LIVE).get("yolo_attempts")


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


def test_a_read_only_plan_says_what_it_would_restamp(scene):
    """--freeze-status previews the re-stamp without writing it."""
    _settle(scene)
    scene.overrides[FROZEN]["freeze"]["cutoff_exempt_before"] = "2026-06-26"
    scene.reload()
    meta = _freeze.slice_dir(scene.settings.cache_dir, "combined_base", FROZEN) / "meta.json"
    before = meta.read_bytes()
    plan = scene.plan(read_only=True)
    assert plan.decisions[FROZEN].decision == "frozen"
    assert any("would be re-stamped" in n for n in plan.notices)
    assert not any(n.endswith("marks no folder differently") and "would" not in n for n in plan.notices)
    assert meta.read_bytes() == before


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


def test_commit_refuses_a_valid_slice_this_run_did_not_rewrite(scene):
    """New data arrived but the rebuilt rows never reached the cache (a cache
    write that failed). Yesterday's slice still validates under the same
    fingerprint: recording it frozen would drop the new batch for good."""
    _settle(scene)
    _batch(scene.root(LIVE), "august_15_batch_1_rig_2")
    plan = scene.plan()
    assert plan.decisions[LIVE].decision == "thawed"
    kw = next(kw for b, n, kw in scene.blocks if LIVE in n)
    assert _freeze.slice_status(scene.settings.cache_dir, "combined_base", LIVE,
                                scene.fingerprint(LIVE, kw)) == (True, "")
    lines = plan.commit(combined_ran=True)  # nothing re-cached LIVE this run
    line = next(line for line in lines if f"{LIVE}:" in line)
    assert "NOT refrozen" in line and "not rewritten by this run" in line
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


# --- a rebuild whose rows went missing is not recorded frozen --------------------------------


def _rebuild_with(scene, ds, rows):
    plan = scene.plan()
    assert plan.decisions[ds].decision == "thawed"
    scene.seed_slice(ds, rows=rows, run=plan.run_id)
    return plan, plan.commit(combined_ran=True)


def _line(lines, ds):
    return next(line for line in lines if line.startswith(f"[FREEZE] {ds}:"))


NEW_BATCH = "august_14_batch_1_rig_3"


def test_a_rebuild_whose_rows_miss_a_processed_batch_stays_thawing(scene):
    """combine logs and skips a trial it cannot read, and build_wide_csv drops
    unreadable trials, so one bad night can lose a batch's rows. Frozen like
    that, they would stay lost until new data arrived -- base re-walked a live
    dataset every night. The rebuild is not recorded frozen and runs again;
    after MAX_LOST_REBUILDS such rebuilds the dataset freezes, with a notice."""
    assert af.MAX_LOST_REBUILDS == 2
    _settle(scene)
    _batch(scene.root(LIVE), NEW_BATCH)
    _, lines = _rebuild_with(scene, LIVE, _rows(LIVE, [BATCH]))
    assert "NOT refrozen" in _line(lines, LIVE) and NEW_BATCH in _line(lines, LIVE)
    rec = scene.record(LIVE)
    assert rec["state"] == "thawing" and rec["lost_rebuilds"] == 1
    plan, lines = _rebuild_with(scene, LIVE, _rows(LIVE, [BATCH]))
    assert "missed 1 processed batch folder(s)" in plan.decisions[LIVE].reasons[0]
    rec = scene.record(LIVE)
    assert rec["state"] == "frozen" and "lost_rebuilds" not in rec
    assert _line(lines, LIVE).startswith(f"[FREEZE] {LIVE}: recorded frozen after its rebuild")
    notice = next(line for line in lines if "recorded although" in line)
    assert NEW_BATCH in notice and "after 2 rebuilds" in notice and f"--thaw {LIVE}" in notice
    assert scene.plan().decisions[LIVE].decision == "frozen"


def test_an_empty_rebuild_of_a_dataset_with_processed_trials_stays_thawing(scene):
    """A wide build that emitted none of its rows caches an empty slice."""
    _settle(scene)
    _batch(scene.root(LIVE), NEW_BATCH)
    _, lines = _rebuild_with(scene, LIVE, pd.DataFrame())
    assert "NOT refrozen" in _line(lines, LIVE) and "2 processed batch folder(s)" in _line(lines, LIVE)
    assert scene.record(LIVE)["state"] == "thawing"


def test_a_complete_rebuild_after_a_lossy_one_is_recorded_frozen(scene):
    _settle(scene)
    _batch(scene.root(LIVE), NEW_BATCH)
    _rebuild_with(scene, LIVE, _rows(LIVE, [BATCH]))
    assert scene.record(LIVE)["lost_rebuilds"] == 1
    _, lines = _rebuild_with(scene, LIVE, _rows(LIVE, [BATCH, NEW_BATCH]))
    rec = scene.record(LIVE)
    assert rec["state"] == "frozen" and "lost_rebuilds" not in rec
    assert _line(lines, LIVE).startswith(f"[FREEZE] {LIVE}: recorded frozen after its rebuild")


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


# --- trials the plan reports (never a reason to rebuild by themselves) -------------------------------


def test_a_trial_recorded_more_than_once_is_reported(scene):
    """EB-Control-24-1 july_17_batch_1_rig_3 training_1 was recorded three
    times. YOLO processes only the latest session (the one the trial's metadata
    is read from); the run says so every time."""
    _settle(scene)
    (scene.root(LIVE) / BATCH / f"{_stem(BATCH, 'testing_1', '20260809_093939')}.csv").write_text("frame\n0\n")
    plan = scene.plan()
    notice = next(n for n in plan.notices if "recorded more than once" in n)
    assert notice.startswith(f"[FREEZE] {LIVE}: 1 trial(s) recorded more than once")
    assert f"{BATCH}/august_09_batch_1_testing_1 (2 sessions)" in notice
    assert "only the latest session" in notice
    assert "the one its metadata is read from; none while that one has no video" in notice


def test_a_trial_folder_without_a_merged_table_beside_its_video_is_reported(scene):
    """A video OpenCV could not open: yolo_infer left its trial folder with no
    merged table (here only the per-trial steps' _trial_meta.json), so it is
    never retried and never pending -- the run says so every time."""
    _settle(scene)
    stem = _stem(BATCH, "testing_2", "20260809_121500")
    (scene.root(LIVE) / BATCH / f"{stem}.csv").write_text("frame,timestamp\n0,0\n")
    (scene.root(LIVE) / BATCH / f"{stem}.mp4").write_bytes(b"not a video")
    trial = scene.root(LIVE) / BATCH / "august_09_batch_1_testing_2"
    trial.mkdir()
    (trial / "_trial_meta.json").write_text("{}")
    plan = scene.plan()
    notice = next(n for n in plan.notices if "no merged table" in n)
    assert notice.startswith(f"[FREEZE] {LIVE}: 1 trial folder(s) hold no merged table beside their video")
    assert f"{BATCH}/august_09_batch_1_testing_2" in notice
    assert "move the trial folder aside" in notice


def test_untracked_trials_are_reported_and_never_thaw(scene):
    """Rig files, but no trial folder and no local video: the video was deleted
    before YOLO saw it (Hex-Control-24-0.1 august_27_batch_2_rig_2 x8). There is
    nothing YOLO could process, so it is no reason to rebuild -- only reported."""
    (scene.root(LIVE) / BATCH / f"{_stem(BATCH, 'training_1', '20260809_110000')}.csv").write_text("frame\n0\n")
    _settle(scene)
    plan = scene.plan()
    assert plan.decisions[LIVE].decision == "frozen"
    notice = next(n for n in plan.notices if "never tracked" in n)
    assert notice.startswith(f"[FREEZE] {LIVE}: 1 trial(s)")
    assert f"{BATCH}/august_09_batch_1_training_1" in notice


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
