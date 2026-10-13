"""Abdomen fly N is PRODUCTION fly N.

In integrated mode (``abdomen.slot_seeds: production``) a trial's slots are not numbered by the abdomen model's own
top-to-bottom order but by the production eye anchors (``x_class0`` / ``y_class0`` of ``<trial>_fly{N}_distances``), so
the abdomen tables join the PER scores. Pinned here: how the seeds are read and digested; that ``freeze_slots`` gives
each seed's number to the best warm-up eye within ``seed_radius_frac`` x eye scale (any eye, not only ones with a body
attached) and leaves a seed that finds none unmatched; what the meta records; which trials a run tracks (a trial PER has
not processed yet is awaited, with no meta; one whose meta records other seeds loses its old outputs and is tracked
again; one with the same seeds is skipped); and that non-contiguous slots ({1, 3}) run from track to figures.
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fbpipe.config import Settings
from fbpipe.geometry import columns as C
from fbpipe.geometry import stats as ST
from fbpipe.geometry import steps as S
from fbpipe.geometry import track as T
from fbpipe.geometry.config import GeometrySettings
from fbpipe.geometry.detect import id_to_label_map
from fbpipe.utils.fly_geometry import EYE, Part
from test_fly_geometry import EYE_SCALE, box
# ``tree``, ``fake_model`` and ``video`` are fixtures: imported so pytest finds them, then used as test arguments
from test_geometry_jobs import (BATCH, DATASET, STAMP, TRIALS, Tree, add_trial, fake_model,  # noqa: F401
                                production_batch, trial_stem, tree)
from test_geometry_track import H, NAMES, W, _gcfg, _info, _settings, fake_predict, scene, video  # noqa: F401

# The three stacked flies of test_geometry_track.scene(), in the 1080x1080 model frame -- the space production's
# x_class0 / y_class0 are in.
TOP, MID, BOTTOM = (300.0, 200.0), (300.0, 500.0), (300.0, 800.0)


# ---- helpers ---------------------------------------------------------------------------------------

def write_fly_table(folder: Path, trial: str, n: int, x: float, y: float, *, fmt: str = "parquet",
                    lead_nan: int = 0, rows: int = 6) -> Path:
    """One production ``<trial>_fly{n}_distances.<fmt>``: the anchor columns are constant, the first ``lead_nan`` rows NaN."""
    folder.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"frame": np.arange(rows),
                       "x_class0": [np.nan] * lead_nan + [x] * (rows - lead_nan),
                       "y_class0": [np.nan] * lead_nan + [y] * (rows - lead_nan),
                       "distance_0_1": np.linspace(10.0, 60.0, rows)})
    path = folder / f"{trial}_fly{n}_distances.{fmt}"
    df.to_parquet(path, index=False) if fmt == "parquet" else df.to_csv(path, index=False)
    return path


def write_seeds(folder: Path, trial: str, seeds: dict, **kw) -> None:
    for n, (x, y) in seeds.items():
        write_fly_table(folder, trial, n, x, y, **kw)


def put_seeds(tree: Tree, seeds: dict, *, trials=(1, 2, 3), where: str = "prod") -> None:
    """Production's seed tables for the given trials of BATCH, replacing what was there (a PER re-run rewrites them).
    ``where``: the production trial folder ("prod") or the secured mirror's ("mirror")."""
    for n in trials:
        folder = (tree.batch_dir if where == "prod" else tree.secured_batch) / trial_stem(n)
        shutil.rmtree(folder, ignore_errors=True)
        write_seeds(folder, trial_stem(n), seeds)


def production(tree: Tree):
    return tree.settings(), tree.gcfg(slot_seeds="production")


def read_meta(tree: Tree, n: int) -> dict:
    return json.loads(tree.meta(n).read_text(encoding="utf-8"))


def slots(tree: Tree, n: int) -> list:
    return [slot for slot, _ in C.iter_geometry_tables(tree.trial_dir(n))]


def track_with(video, seeds, **gcfg_kw):
    return T.track_video(video, _info(video), predict_fn=fake_predict, id_to_label=id_to_label_map(NAMES),
                         settings=_settings(), gcfg=_gcfg(**gcfg_kw), seed_centers=seeds)


def _vg():
    return T.VideoGeometry(W, H, 1080, 1080, measure_native=True)


def _freeze(seeds, warm=None, **kw):
    warm = [scene(i) for i in range(5)] if warm is None else warm
    return T.freeze_slots(warm, _vg(), _gcfg().limits, max_flies=5, presence_min=0.5, seed_centers=seeds, **kw)


# ---- reading the seeds -----------------------------------------------------------------------------

def test_the_seeds_are_each_production_flys_frozen_eye_anchor_keyed_by_its_fly_number(tmp_path):
    trial = "october_07_batch_2_pretest_1"
    write_seeds(tmp_path, trial, {1: (754.3, 116.8), 3: (624.8, 872.4)})
    # the rest of a production trial folder: none of it is a per-fly table
    pd.DataFrame({"x_class0": [1.0], "y_class0": [2.0]}).to_parquet(tmp_path / f"{trial}_distances_merged.parquet")
    (tmp_path / f"{trial}_fly1_distances_dropped_frames.txt").write_text("0\n", encoding="utf-8")
    (tmp_path / "_trial_meta.json").write_text("{}", encoding="utf-8")
    seeds = T.production_eye_seeds(tmp_path)
    assert seeds == {1: (754.3, 116.8), 3: (624.8, 872.4)}
    assert all(type(n) is int and all(type(v) is float for v in xy) for n, xy in seeds.items()), "plain, JSON-able numbers"


def test_leading_rows_without_an_anchor_are_skipped(tmp_path):
    write_fly_table(tmp_path, "t", 1, 754.3, 116.8, lead_nan=4)
    # half a row (x known, y not) is no anchor either
    pd.DataFrame({"x_class0": [500.0, np.nan, 600.5], "y_class0": [np.nan, 10.0, 20.5]}).to_parquet(
        tmp_path / "t_fly3_distances.parquet")
    assert T.production_eye_seeds(tmp_path) == {1: (754.3, 116.8), 3: (600.5, 20.5)}


def test_a_parquet_table_wins_over_its_csv_sibling_and_a_csv_alone_is_read(tmp_path):
    write_fly_table(tmp_path, "t", 1, 754.3, 116.8, fmt="parquet")
    write_fly_table(tmp_path, "t", 1, 1.0, 2.0, fmt="csv")
    write_fly_table(tmp_path, "t", 2, 634.9, 281.2, fmt="csv")
    assert T.production_eye_seeds(tmp_path) == {1: (754.3, 116.8), 2: (634.9, 281.2)}


def test_only_the_two_anchor_columns_are_read(tmp_path, monkeypatch):
    write_fly_table(tmp_path, "t", 1, 754.3, 116.8)
    asked = []
    real = pd.read_parquet
    monkeypatch.setattr(pd, "read_parquet", lambda path, *a, **kw: asked.append(kw.get("columns")) or real(path, *a, **kw))
    T.production_eye_seeds(tmp_path)
    assert len(asked) == 1 and set(asked[0]) == {"x_class0", "y_class0"}


def test_no_tables_means_no_seeds(tmp_path):
    assert T.production_eye_seeds(tmp_path) == {}
    assert T.production_eye_seeds(tmp_path / "no_such_trial") == {}
    (tmp_path / "t_distances_merged.parquet").write_bytes(b"")                      # not a per-fly table
    assert T.production_eye_seeds(tmp_path) == {}


def test_a_table_with_no_usable_anchor_gives_no_seed_for_that_fly_and_the_others_still_count(tmp_path, capsys):
    write_fly_table(tmp_path, "t", 1, 754.3, 116.8)
    write_fly_table(tmp_path, "t", 2, np.nan, np.nan)                               # production never found its eye
    (tmp_path / "t_fly3_distances.parquet").write_bytes(b"not a parquet file")      # a torn write
    pd.DataFrame({"frame": [0, 1]}).to_parquet(tmp_path / "t_fly4_distances.parquet")   # some other schema
    assert T.production_eye_seeds(tmp_path) == {1: (754.3, 116.8)}
    out = capsys.readouterr().out
    assert "t_fly3_distances.parquet" in out, "the unreadable table is named"
    assert "FLYFILES" not in out, "the file finder's per-file chatter stays out of the track log"


def test_a_folder_that_cannot_be_listed_has_no_seeds_instead_of_stopping_the_run(tmp_path, monkeypatch, capsys):
    write_fly_table(tmp_path, "t", 1, 754.3, 116.8)

    def stale_handle(base, **kw):
        raise OSError(116, "Stale file handle")

    monkeypatch.setattr(T, "iter_fly_distance_csvs", stale_handle)
    assert T.production_eye_seeds(tmp_path) == {}
    assert str(tmp_path) in capsys.readouterr().out, "the folder that failed is named"


# ---- the digest ------------------------------------------------------------------------------------

def test_the_digest_recipe_is_pinned_because_changing_it_re_tracks_every_trial():
    seeds = {3: (624.8, 872.4), 1: (754.3, 116.8)}
    expected = hashlib.sha1(repr([(1, 754.3, 116.8), (3, 624.8, 872.4)]).encode("utf-8")).hexdigest()
    assert T.seed_digest(seeds) == expected


def test_the_digest_ignores_order_and_sub_tenth_pixel_noise_but_not_the_numbering():
    base = {1: (754.3, 116.8), 3: (624.8, 872.4)}
    digest = T.seed_digest(base)
    assert T.seed_digest(dict(reversed(list(base.items())))) == digest
    assert T.seed_digest({1: (754.31, 116.79), 3: (624.8, 872.4)}) == digest        # rounds to the same 0.1 px
    assert T.seed_digest({1: (754.4, 116.8), 3: (624.8, 872.4)}) != digest          # one 0.1 px step
    assert T.seed_digest({2: (754.3, 116.8), 3: (624.8, 872.4)}) != digest          # same places, other fly numbers
    assert T.seed_digest({1: (624.8, 872.4), 3: (754.3, 116.8)}) != digest          # numbers swapped
    assert T.seed_digest({1: (754.3, 116.8)}) != digest                             # a fly fewer


def test_the_digest_rounds_python_floats_so_numpy_input_cannot_change_it():
    """Seeds read from a table are numpy numbers, and numpy rounds 754.15 to 754.2 where Python gives 754.1."""
    if round(np.float64(754.15), 1) == round(754.15, 1):
        pytest.skip("this numpy rounds like Python, so the conversion is not observable here")
    assert T.seed_digest({np.int64(1): (np.float64(754.15), np.float64(116.8))}) == T.seed_digest({1: (754.15, 116.8)})


# ---- freezing the slots from seeds -----------------------------------------------------------------

def test_slots_carry_the_production_fly_numbers_of_the_seeds():
    frozen, axis, chirality = _freeze({1: TOP, 3: BOTTOM})
    assert [f.slot for f in frozen] == [1, 3]
    assert [tuple(np.round(f.eye_model.centroid)) for f in frozen] == [TOP, BOTTOM]
    assert all(f.presence == 1.0 and f.conf == pytest.approx(0.9) for f in frozen)
    # the posterior axis and handedness priors do not depend on how the slots are numbered
    unseeded = _freeze(None)
    assert np.allclose(axis, unseeded[1]) and chirality == unseeded[2] == -1.0


def test_slots_follow_the_seeds_not_the_vertical_order_and_come_back_in_slot_order():
    frozen = _freeze({4: TOP, 2: MID})[0]
    assert [f.slot for f in frozen] == [2, 4]
    assert [tuple(np.round(f.eye_model.centroid)) for f in frozen] == [MID, TOP]


def test_a_seed_with_no_eye_within_half_an_eye_scale_is_left_unmatched():
    near = (TOP[0] + 23.0, TOP[1])                  # eye scale 48 px: reach 24 px
    far = (BOTTOM[0] + 25.0, BOTTOM[1])
    frozen = _freeze({1: near, 3: far})[0]
    assert [f.slot for f in frozen] == [1], "slot 3 finds no eye, so it has no slot"
    # the reach is seed_radius_frac x eye scale
    assert [f.slot for f in _freeze({1: near, 3: far}, seed_radius_frac=0.6)[0]] == [1, 3]
    assert [f.slot for f in _freeze({1: near}, seed_radius_frac=0.4)[0]] == []


def test_a_seed_claims_a_bare_eye_with_no_proboscis_or_abdomen_where_the_warmup_pass_would_not():
    bare = (900.0, 350.0)
    warm = [scene(i) + [Part(EYE, box(bare, EYE_SCALE), 0.8)] for i in range(5)]    # a bare eye in every warm-up frame
    assert [f.slot for f in _freeze(None, warm)[0]] == [1, 2, 3], "no body attached, so no fly: left out unseeded"
    frozen = _freeze({1: TOP, 5: bare}, warm)[0]
    assert [f.slot for f in frozen] == [1, 5], "production already established the fly"
    assert frozen[1].presence == 1.0 and frozen[1].conf == pytest.approx(0.8)


def test_presence_min_does_not_apply_to_a_seeded_slot_but_the_presence_is_recorded():
    seen = [Part(p.label, p.pts, 0.9) for p in scene(0) if p.centroid[1] < 350.0]          # the top fly, in 2 of 5 frames
    warm = [seen, seen, [], [], []]
    assert _freeze(None, warm)[0] == [], "unseeded, 0.4 < presence_min drops the anchor"
    frozen = _freeze({1: TOP}, warm)[0]
    assert [f.slot for f in frozen] == [1] and frozen[0].presence == pytest.approx(0.4)


def test_the_best_confidence_eye_in_reach_is_the_frozen_one():
    def eye(center, score):
        return [Part(EYE, box(center, EYE_SCALE), score)]
    warm = [eye((300.0, 200.0), 0.6), eye((310.0, 205.0), 0.95), eye((296.0, 198.0), 0.7)]
    frozen = _freeze({2: TOP}, warm)[0]
    assert frozen[0].conf == pytest.approx(0.95)
    assert tuple(np.round(frozen[0].eye_model.centroid)) == (310.0, 205.0)
    assert frozen[0].presence == 1.0 and frozen[0].scale == pytest.approx(EYE_SCALE)


def test_without_seeds_the_slots_are_what_they_were_whatever_the_radius():
    frozen, axis, chirality = _freeze(None, seed_radius_frac=0.9)
    assert [f.slot for f in frozen] == [1, 2, 3]
    assert [tuple(np.round(f.eye_model.centroid)) for f in frozen] == [TOP, MID, BOTTOM]
    assert all(f.presence == 1.0 for f in frozen) and chirality == -1.0
    plain = T.freeze_slots([scene(i) for i in range(5)], _vg(), _gcfg().limits, max_flies=5, presence_min=0.5)
    assert [(f.slot, f.presence, f.conf, f.scale) for f in plain[0]] == [(f.slot, f.presence, f.conf, f.scale) for f in frozen]
    assert all(np.array_equal(a.eye_model.pts, b.eye_model.pts) for a, b in zip(plain[0], frozen))


# ---- tracking a video with seeds -------------------------------------------------------------------

def test_track_video_with_seeds_writes_tables_keyed_by_production_number_and_records_the_meta(video):
    seeds = {1: TOP, 3: BOTTOM}
    res = track_with(video, seeds)
    assert sorted(res.tables) == [1, 3], "the middle fly production never tracked is not tracked here"
    assert [s["slot"] for s in res.meta["slots"]] == [1, 3] and [fz.slot for fz in res.frozen] == [1, 3]
    df1, df3 = res.tables[1], res.tables[3]
    assert df1[C.EYE_Y].iloc[0] == pytest.approx(200.0, abs=1.0) and df3[C.EYE_Y].iloc[0] == pytest.approx(800.0, abs=1.0)
    # scene(): the bottom fly loses its tip from frame 10, the top fly does not
    assert df3[C.STATUS].iloc[15] == C.STATUS_ABDOMEN_ONLY and df1[C.STATUS].iloc[15] == C.STATUS_OK
    meta = res.meta
    assert meta["slot_source"] == "production"
    assert meta["seed_centers"] == {"1": [300.0, 200.0], "3": [300.0, 800.0]}
    assert meta["seed_digest"] == T.seed_digest(seeds)
    assert meta["unmatched_seeds"] == []
    json.dumps(meta)                                                                # must be JSON-serialisable


def test_a_slots_table_follows_the_fly_its_seed_names_even_when_the_numbers_do_not_run_top_to_bottom(video):
    res = track_with(video, {4: TOP, 2: MID})
    assert sorted(res.tables) == [2, 4]
    assert res.tables[2][C.EYE_Y].iloc[0] == pytest.approx(500.0, abs=1.0)
    assert res.tables[4][C.EYE_Y].iloc[0] == pytest.approx(200.0, abs=1.0)
    # scene(): only the middle fly's proboscis jumps (frame 20), and the velocity gate must blank it in ITS table
    assert res.tables[2][C.PRO_STATUS].iloc[20] == C.PRO_VELOCITY
    assert res.tables[4][C.PRO_STATUS].iloc[20] == C.PRO_OK


def test_track_video_lists_the_seeds_that_found_no_eye(video):
    res = track_with(video, {1: TOP, 2: (900.0, 500.0)})                           # nothing anywhere near the second
    assert sorted(res.tables) == [1]
    assert res.meta["unmatched_seeds"] == [2]
    assert res.meta["seed_centers"] == {"1": [300.0, 200.0], "2": [900.0, 500.0]}, "the seeds asked for, not only the matched"
    nothing = track_with(video, {7: (900.0, 500.0)})
    assert nothing.tables == {} and nothing.meta["unmatched_seeds"] == [7] and nothing.meta["slot_source"] == "production"
    assert nothing.meta["seed_digest"] == T.seed_digest({7: (900.0, 500.0)})


def test_track_video_without_seeds_numbers_the_slots_top_to_bottom_and_says_so_in_the_meta(video):
    res = track_with(video, None)
    assert sorted(res.tables) == [1, 2, 3]
    assert res.meta["slot_source"] == "warmup"
    assert res.meta["seed_centers"] == {} and res.meta["seed_digest"] is None and res.meta["unmatched_seeds"] == []


# ---- which trials a run tracks ---------------------------------------------------------------------

def test_a_trial_whose_production_seeds_are_not_there_yet_is_awaited_not_tracked(tree, fake_model, capsys):
    settings, gcfg = production(tree)
    assert S.pending_trials(settings, gcfg) == 0, "waiting for PER is no work for a worker: no fan-out, no model load"
    result = S.run_track(settings, gcfg)
    assert (result["tracked"], result["skipped"], result["failed"], result["awaiting_seeds"]) == (0, 0, 0, 3)
    assert not tree.out_base.exists(), "a trial without a production identity leaves no folder, no meta"
    assert "awaiting" in capsys.readouterr().out
    put_seeds(tree, {1: TOP, 3: BOTTOM}, trials=(1,))                               # PER gets to trial 1
    assert S.pending_trials(settings, gcfg) == 1
    result = S.run_track(settings, gcfg)
    assert (result["tracked"], result["awaiting_seeds"]) == (1, 2)
    assert tree.meta(1).is_file() and not tree.meta(2).exists() and not tree.meta(3).exists()
    assert S.pending_trials(settings, gcfg) == 0


def test_seeds_come_from_the_production_trial_folder_then_from_the_secured_mirror(tree, fake_model):
    settings, gcfg = production(tree)
    put_seeds(tree, {1: TOP, 3: BOTTOM}, trials=(1,), where="mirror")
    (tree.batch_dir / trial_stem(1)).mkdir()                                        # present but empty: counts as absent
    assert S.pending_trials(settings, gcfg) == 1
    S.run_track(settings, gcfg)
    assert read_meta(tree, 1)["seed_centers"] == {"1": [300.0, 200.0], "3": [300.0, 800.0]}
    assert slots(tree, 1) == [1, 3]
    # both hold seeds: the production folder wins, so the mirror's numbering is replaced
    put_seeds(tree, {2: MID}, trials=(1,))
    assert S.pending_trials(settings, gcfg) == 1
    S.run_track(settings, gcfg)
    assert read_meta(tree, 1)["seed_centers"] == {"2": [300.0, 500.0]} and slots(tree, 1) == [2]


def test_a_trial_tracked_before_its_seeds_existed_is_tracked_again_numbered_by_production(tree, fake_model):
    settings = tree.settings()
    assert S.run_track(settings, tree.gcfg())["tracked"] == 3                      # the warm-up numbering
    assert read_meta(tree, 1)["slot_source"] == "warmup" and slots(tree, 1) == [1, 2, 3]
    gcfg = tree.gcfg(slot_seeds="production")
    assert S.pending_trials(settings, gcfg) == 0, "no seeds anywhere yet: nothing to number them by"
    put_seeds(tree, {1: TOP, 3: BOTTOM}, trials=(1,))
    assert S.pending_trials(settings, gcfg) == 1
    result = S.run_track(settings, gcfg)
    assert (result["tracked"], result["skipped"], result["failed"], result["awaiting_seeds"]) == (1, 0, 0, 2)
    meta = read_meta(tree, 1)
    assert meta["slot_source"] == "production" and meta["seed_digest"] == T.seed_digest({1: TOP, 3: BOTTOM})
    assert slots(tree, 1) == [1, 3]
    assert not list(tree.trial_dir(1).glob("*_fly2_geometry.*")), "the warm-up numbering's fly 2 is gone"
    for n in (2, 3):                                                                # still waiting: left exactly as they were
        assert read_meta(tree, n)["slot_source"] == "warmup" and slots(tree, n) == [1, 2, 3]


def test_a_trial_whose_seed_digest_is_unchanged_is_skipped(tree, fake_model):
    settings, gcfg = production(tree)
    put_seeds(tree, {1: TOP, 3: BOTTOM})
    assert S.run_track(settings, gcfg)["tracked"] == 3
    stamps = {n: tree.meta(n).stat().st_mtime_ns for n in (1, 2, 3)}
    put_seeds(tree, {1: (TOP[0] + 0.01, TOP[1]), 3: BOTTOM}, trials=(1,))           # a PER re-run: same anchors to 0.1 px
    assert S.pending_trials(settings, gcfg) == 0
    result = S.run_track(settings, gcfg)
    assert (result["tracked"], result["skipped"], result["failed"], result["awaiting_seeds"]) == (0, 3, 0, 0)
    assert {n: tree.meta(n).stat().st_mtime_ns for n in (1, 2, 3)} == stamps
    assert S.run_track(settings, gcfg, force=True)["tracked"] == 3, "--force still redoes it"


def test_a_changed_seed_digest_removes_the_old_slots_and_tracks_the_trial_again(tree, fake_model):
    settings, gcfg = production(tree)
    put_seeds(tree, {1: TOP, 2: MID, 3: BOTTOM})
    S.run_track(settings, gcfg)
    assert all(slots(tree, n) == [1, 2, 3] for n in (1, 2, 3))
    # PER ran again on trial 1 and numbered its flies differently: the middle fly is gone, the bottom one is now 2
    put_seeds(tree, {1: TOP, 2: BOTTOM}, trials=(1,))
    assert S.pending_trials(settings, gcfg) == 1
    result = S.run_track(settings, gcfg)
    assert (result["tracked"], result["skipped"]) == (1, 2)
    assert slots(tree, 1) == [1, 2]
    assert not list(tree.trial_dir(1).glob("*_fly3_geometry.*")), "no table of the old fly 3 is left behind"
    df2 = C.read_table_any(C.geometry_table_path(tree.trial_dir(1), trial_stem(1), 2, "csv"))
    assert df2[C.EYE_Y].iloc[0] == pytest.approx(800.0, abs=1.0), "fly 2 now follows the bottom fly, not the middle one"
    assert read_meta(tree, 1)["seed_digest"] == T.seed_digest({1: TOP, 2: BOTTOM})
    assert all(slots(tree, n) == [1, 2, 3] for n in (2, 3))
    assert S.pending_trials(settings, gcfg) == 0


def test_an_unreadable_older_or_warmup_sourced_meta_is_re_tracked_rather_than_trusted(tree, fake_model):
    settings, gcfg = production(tree)
    put_seeds(tree, {1: TOP, 3: BOTTOM})
    S.run_track(settings, gcfg)
    older = {"trial": read_meta(tree, 3)["trial"]}                                  # from before the seed fields existed
    warmup = dict(read_meta(tree, 3), slot_source="warmup")                          # the digest is current; the source is not
    tree.meta(1).write_text("{ torn", encoding="utf-8")                              # a half-written meta
    tree.meta(2).write_text(json.dumps(older), encoding="utf-8")
    tree.meta(3).write_text(json.dumps(warmup), encoding="utf-8")
    assert S.pending_trials(settings, gcfg) == 3
    result = S.run_track(settings, gcfg)
    assert (result["tracked"], result["skipped"], result["failed"]) == (3, 0, 0)
    assert all(read_meta(tree, n)["slot_source"] == "production" and slots(tree, n) == [1, 3] for n in (1, 2, 3))
    assert S.pending_trials(settings, gcfg) == 0


def test_a_re_track_that_fails_leaves_no_outputs_of_the_old_numbering_and_is_retried(tree, fake_model, monkeypatch):
    settings, gcfg = production(tree)
    put_seeds(tree, {1: TOP, 2: MID, 3: BOTTOM})
    S.run_track(settings, gcfg)
    put_seeds(tree, {1: TOP}, trials=(2,))
    real = T.track_video

    def flaky(video_path, *a, **kw):
        if "_pretest_2_" in Path(video_path).name:
            raise ValueError("boom")
        return real(video_path, *a, **kw)

    monkeypatch.setattr(T, "track_video", flaky)
    result = S.run_track(settings, gcfg)
    assert (result["tracked"], result["failed"], result["skipped"]) == (0, 1, 2)
    assert not tree.meta(2).exists() and slots(tree, 2) == [], "never a table of a numbering production has dropped"
    assert S.pending_trials(settings, gcfg) == 1
    monkeypatch.setattr(T, "track_video", real)
    assert S.run_track(settings, gcfg)["tracked"] == 1
    assert slots(tree, 2) == [1] and read_meta(tree, 2)["slot_source"] == "production"


def test_a_forced_run_still_waits_for_the_seeds_of_a_trial_that_has_none(tree, fake_model):
    settings, gcfg = production(tree)
    put_seeds(tree, {1: TOP, 3: BOTTOM}, trials=(1,))
    result = S.run_track(settings, gcfg, force=True)
    assert (result["tracked"], result["awaiting_seeds"]) == (1, 2), "without a production number there is no slot to name"
    assert not tree.meta(2).exists() and not tree.meta(3).exists()


def test_warmup_mode_ignores_production_seeds_entirely(tree, fake_model):
    settings, gcfg = tree.settings(), tree.gcfg()                                   # slot_seeds="warmup", the default
    put_seeds(tree, {1: TOP, 3: BOTTOM})
    assert S.pending_trials(settings, gcfg) == 3
    result = S.run_track(settings, gcfg)
    assert (result["tracked"], result["awaiting_seeds"]) == (3, 0)
    meta = read_meta(tree, 1)
    assert meta["slot_source"] == "warmup" and meta["seed_centers"] == {} and meta["seed_digest"] is None
    assert slots(tree, 1) == [1, 2, 3]
    put_seeds(tree, {1: BOTTOM})                                                    # other seeds later change nothing here
    assert S.pending_trials(settings, gcfg) == 0
    again = S.run_track(settings, gcfg)
    assert (again["tracked"], again["skipped"]) == (0, 3)


def test_legacy_mode_never_waits_for_seeds_whatever_slot_seeds_says(tmp_path, video, fake_model):
    batch = production_batch(tmp_path / "flys_Abdomen" / DATASET, BATCH, video, day=None)
    for kind, n, odor in TRIALS[:2]:
        add_trial(batch, video, BATCH, kind, n, odor, day=STAMP)
    settings = Settings(model_path=str(tmp_path / "m.pt"), main_directories=[str(batch.parent)], datasets=(DATASET,),
                        conf_thres=0.25, max_flies=5, inference_batch_size=7, allow_cpu=True, fps_default=40.0)
    gcfg = GeometrySettings(warmup_frames=5, annotated_video=False, slot_seeds="production")      # no out_base: legacy
    assert S.pending_trials(settings, gcfg) == 2
    result = S.run_track(settings, gcfg)
    assert (result["tracked"], result["awaiting_seeds"]) == (2, 0)
    tdir = batch / "geometry" / trial_stem(1)
    assert json.loads(C.meta_path(tdir, trial_stem(1)).read_text(encoding="utf-8"))["slot_source"] == "warmup"
    assert [slot for slot, _ in C.iter_geometry_tables(tdir)] == [1, 2, 3]
    assert S.pending_trials(settings, gcfg) == 0


def test_tracking_with_seeds_only_reads_production_and_the_mirror(tree, fake_model):
    put_seeds(tree, {1: TOP, 3: BOTTOM})
    put_seeds(tree, {1: TOP, 3: BOTTOM}, trials=(1, 2), where="mirror")
    before = tree.read_only()
    S.run_track(*production(tree))
    assert all(slots(tree, n) == [1, 3] for n in (1, 2, 3)), "precondition: the seeds were read and used"
    assert tree.read_only() == before


# ---- clearing a trial's outputs --------------------------------------------------------------------

def test_clearing_deletes_one_trials_geometry_outputs_and_nothing_else(tmp_path):
    out_base = tmp_path / "flys_Abdomen"
    geometry = out_base / "IAA-X" / "b" / "geometry"
    trial, other = geometry / "b_pretest_1", geometry / "b_pretest_2"
    ours = ["b_pretest_1_fly1_geometry.csv", "b_pretest_1_fly1_geometry.parquet", "b_pretest_1_fly12_geometry.csv",
            "b_pretest_1_geometry_annotated.mp4", "b_pretest_1_geometry_meta.json"]
    kept = ["notes.txt", "b_pretest_1_fly1_distances.parquet", "b_pretest_1_fly1_geometry.csv.bak"]
    for folder, names in ((trial, ours + kept), (other, [n.replace("pretest_1", "pretest_2") for n in ours])):
        folder.mkdir(parents=True)
        for name in names:
            (folder / name).write_text("x", encoding="utf-8")
    (geometry / "fly1_geometry_stats.json").write_text("{}", encoding="utf-8")
    removed = S._clear_trial_outputs(trial, "b_pretest_1", out_base=out_base)
    assert sorted(p.name for p in removed) == sorted(ours)
    assert sorted(p.name for p in trial.iterdir()) == sorted(kept)
    assert len(list(other.iterdir())) == len(ours), "another trial is not touched"
    assert (geometry / "fly1_geometry_stats.json").exists()
    assert S._clear_trial_outputs(trial, "b_pretest_1", out_base=out_base) == [], "nothing left is not an error"
    assert S._clear_trial_outputs(geometry / "b_pretest_9", "b_pretest_9", out_base=out_base) == []


def test_clearing_refuses_a_folder_outside_the_out_tree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                                                     # an empty out_base would resolve to here
    out_base = tmp_path / "flys_Abdomen"
    out_base.mkdir()
    victim = tmp_path / "flys_New" / "IAA-X" / "b" / "b_pretest_1"                  # a PRODUCTION trial folder
    victim.mkdir(parents=True)
    files = [victim / "b_pretest_1_fly1_geometry.csv", victim / "b_pretest_1_geometry_meta.json"]
    for f in files:
        f.write_text("x", encoding="utf-8")
    (out_base / "link").symlink_to(victim, target_is_directory=True)
    bad = [victim,                                                                  # elsewhere
           out_base,                                                                # the tree itself is no trial
           out_base / ".." / "flys_New" / "IAA-X" / "b" / "b_pretest_1",            # climbs out
           tmp_path / "flys_Abdomen2" / "t",                                        # shares a name prefix, not a parent
           out_base / "link"]                                                       # inside by name, outside once resolved
    for folder in bad:
        with pytest.raises(ValueError):
            S._clear_trial_outputs(folder, "b_pretest_1", out_base=out_base)
    with pytest.raises(ValueError):
        S._clear_trial_outputs(victim, "b_pretest_1", out_base="")                  # no out tree configured (legacy)
    assert all(f.exists() for f in files)


# ---- non-contiguous slots, end to end --------------------------------------------------------------

def test_production_numbers_with_a_gap_run_from_track_to_the_figures(tree, fake_model):
    settings, gcfg = production(tree)
    put_seeds(tree, {1: TOP, 3: BOTTOM})
    assert S.run_track(settings, gcfg)["tracked"] == 3
    for n in (1, 2, 3):
        assert slots(tree, n) == [1, 3]
        assert sorted(p.name for p in tree.trial_dir(n).glob("*_geometry.csv")) == [
            f"{trial_stem(n)}_fly1_geometry.csv", f"{trial_stem(n)}_fly3_geometry.csv"]
    assert S.run_normalize(settings, gcfg) == {str(tree.out_batch()): [1, 3]}
    assert sorted(p.name for p in (tree.out_batch() / "geometry").glob("fly*_geometry_stats.json")) == [
        "fly1_geometry_stats.json", "fly3_geometry_stats.json"]
    wide = S.run_wide(settings, gcfg)
    assert wide["rows"] == {m: 6 for m in wide["rows"]}, "3 trials x 2 flies per measure"
    assert sorted(set(pd.read_csv(wide["files"]["per"])["fly_number"].astype(str))) == ["1", "3"]
    assert S.run_figures(settings, gcfg)["n_figures"] == 5                          # (traces + size) x 2 flies + the QC grid
    figures = sorted(p.name for p in (Path(gcfg.figures_out_dir) / DATASET / BATCH).glob("*.png"))
    assert [n for n in figures if "_fly" in n] == [
        f"{BATCH}_fly1_abdomen_size_traces.png", f"{BATCH}_fly1_per_abdomen_traces.png",
        f"{BATCH}_fly3_abdomen_size_traces.png", f"{BATCH}_fly3_per_abdomen_traces.png"]
    metrics = ST.collect_fly_trial_metrics(settings, gcfg)
    assert len(metrics) == 6 and set(metrics["fly_number"]) == {1, 3}
