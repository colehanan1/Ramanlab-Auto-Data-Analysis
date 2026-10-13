"""The mounting-posture tables: ``fly_trial_posture.csv`` (one row per fly-trial) and ``fly_posture.csv`` (one per fly).

Rules pinned here, on a synthetic production + geometry + secured-mirror tree under ``tmp_path``:

* scope -- only live batches on or after the cut-off date, and only trials that have a geometry meta;
* the join -- production fly N is geometry fly N, and a geometry eye further than ``eye_match_px`` from the
  production anchor drops the row (counted); a fly whose anchor wanders keeps its trial rows but leaves the fly
  table and the rig reference (counted);
* units and windows -- eye units are pixels over the geometry ``eye_scale_px``; the stimulus window comes from
  the geometry meta, so nothing is ever written next to production;
* gates -- an unknown rest, rig reference, pre-window or poorly tracked tip gives NaN, never a guess;
* pooling -- fly rates divide pooled events by pooled window-minutes, proportions pool trials, and the
  continuous abdomen measures are medians over the fly's pooled frames.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from scripts.analysis import mounting_posture as MP  # noqa: E402
from fbpipe.analysis import mounting as M  # noqa: E402
from fbpipe.config import DatasetOverride, Settings  # noqa: E402
from fbpipe.geometry import columns as C  # noqa: E402
from fbpipe.geometry.config import GeometrySettings  # noqa: E402
from fbpipe.geometry.trial_meta import TrialInfo  # noqa: E402
from fbpipe.utils.frozen_folders import FolderRule  # noqa: E402

DATASET = "IAA-X"
NEW, OLD = "october_07_batch_2", "september_30_batch_1"
HOST = "Flybehavior2"
CUTOFF = date(2026, 10, 6)
FPS, N, SCALE = 10.0, 400, 48.0
ONSET, OFFSET = 300, 340            # odor on at 30 s for 4 s: every pre-stimulus window is 30 s = 0.5 min
EYES = {1: (300.0, 200.0), 2: (300.0, 700.0)}
TRIALS = (("pretest", 1), ("testing", 1), ("testing", 2))
FLIES = (1, 2)
PITCH_5 = math.degrees(math.atan2(20.0, 200.0))       # abdomen 20 px below the eye, 200 px behind it
PITCH_11 = math.degrees(math.atan2(40.0, 200.0))
TIP_PITCH = math.degrees(math.atan2(10.0, 60.0))
MCFG = M.parse_mounting_block({}, default_batch_date_from=CUTOFF)

# ---------------------------------------------------------------------------
# The synthetic tree
# ---------------------------------------------------------------------------


def recipe(fly, kind, idx):
    """One fly-trial of the base scenario as N-frame arrays, before it is written as tables.

    Fly 1 is always retracted (rest 24 px = 0.5 eu); fly 2 holds its proboscis out in testing_1 only.
    Fly 1's stimulus window is untouched by spontaneous events: the evoked ones sit inside it, the other ones outside.
    """
    r = dict(eye=EYES[fly], dist=np.full(N, 24.0), theta=0.0, bend=np.full(N, 5.0), abd=(200.0, 40.0),
             tip_dx=np.zeros(N), tip_dy=np.full(N, 10.0), tip_ok=np.ones(N, bool))
    f = np.arange(N)
    key = (fly, kind, idx)
    if key == (1, "pretest", 1):         # +20 px for 0.5 s (a spontaneous PER) and a +15 degree curl before onset
        r["abd"] = (200.0, 20.0)
        r["dist"][100:105] += 20.0
        r["bend"] = 4.0 + 0.01 * f
        r["bend"][150:155] += 15.0
    elif key == (1, "testing", 1):       # a too-small and a too-short blip, then an evoked curl and a moving tip
        r["dist"][100:105] += 10.0
        r["bend"][100:102] += 15.0
        r["bend"][305:335] += 15.0
        r["bend"][360:365] += 15.0                       # after the stimulus ended: not an evoked curl
        r["tip_dx"] = 4.8 * np.clip(f - 300, 0, 45)      # 4.8 px/frame = 1.0 eu/s from frame 300 on
    elif key == (2, "pretest", 1):       # an evoked PER inside the stimulus window: not a spontaneous one
        r["dist"][310:315] += 20.0
    elif key == (1, "testing", 2):       # a second curl and a pull-in before onset; a +20 px blip that is only 0.2 s long
        r["abd"] = (200.0, 0.0)
        r["dist"][100:102] += 20.0
        r["dist"][250:255] -= 15.0
        r["bend"][200:205] += 15.0
    elif key == (2, "testing", 1):       # held back at -15 degrees, a dip in the stimulus, tip tracked in 30% of frames
        r["dist"] = np.full(N, 60.0)
        r["dist"][310:315] -= 20.0
        r["theta"] = -15.0
        r["tip_ok"] = (f % 10) < 3
    return r


def frames(r):
    """``(production, geometry)`` tables for a recipe. The anchor is on the left edge, so forward is +x."""
    f = np.arange(N)
    (ex, ey), (gx, gy) = r["eye"], r.get("geo_eye", r["eye"])
    th = np.radians(r["theta"])
    prod = pd.DataFrame({"frame": f, "x_class0": ex, "y_class0": ey, "eye_live_x": ex, "eye_live_y": ey,
                         "x_class1": ex + r["dist"] * np.sin(th), "y_class1": ey + r["dist"] * np.cos(th),
                         "x_anchor": 0.0})
    ax, ay = gx + r["abd"][0], gy + r["abd"][1]
    ok = r["tip_ok"]
    geo = pd.DataFrame({"frame": f, "eye_x": gx, "eye_y": gy, "eye_scale_px": SCALE,
                        "proboscis_x": gx + r["dist"] * np.sin(th), "proboscis_y": gy + r["dist"] * np.cos(th),
                        "proboscis_distance_px": r["dist"], "abdomen_x": ax, "abdomen_y": ay,
                        "tip_x": np.where(ok, ax + 60.0 + r["tip_dx"], np.nan), "tip_y": np.where(ok, ay + r["tip_dy"], np.nan),
                        "abdomen_angle_deg": np.where(ok, 175.0 - r["bend"], np.nan),
                        "abdomen_bend_deg": np.where(ok, r["bend"], np.nan)})
    return (r["prod_edit"](prod) if "prod_edit" in r else prod), geo


def trial_meta(batch, kind, idx):
    info = TrialInfo(video=None, batch_dir=batch, dataset=DATASET, trial_name=f"{batch}_{kind}_{idx}", out_dir="",
                     trial_type=kind, trial_index=idx, odor="Citral-1", label=f"{kind}_{idx}_Citral-1", n_frames=N,
                     fps=FPS, has_odor=True, odor_on_frame=ONSET, odor_off_frame=OFFSET, light_on_s=None,
                     light_off_s=None, fly_type="GR5a-Old", sidecar_txt=None, sidecar_csv=None,
                     recorded_at="20261007_100000")
    return {"pipeline": "fbpipe.geometry", "schema_version": 1, "trial": info.to_dict(),
            "frame": {"width": 1080, "height": 1080, "model_w": 1080, "model_h": 1080, "sx": 1.0, "sy": 1.0,
                      "measure_in": "native"},
            "fps": FPS, "n_frames_processed": N, "slot_source": "production", "seed_digest": "digest"}


class Tree:
    """Production (``flys_New``), the geometry out tree (``flys_Abdomen``) and the secured mirror, under ``root``."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.prod, self.out, self.mirror = self.root / "flys_New", self.root / "flys_Abdomen", self.root / "secured"
        self.ds = self.prod / DATASET
        self.ds.mkdir(parents=True)
        self.mirror.mkdir()

    def write_batch(self, batch, day, *, host=HOST, edits=None, meta_edits=None, all_edit=None, mirror_only=(),
                    no_prod=()):
        """A production batch (metadata + a dated sidecar stub) and its geometry. ``edits[(fly, kind, idx)]`` and
        ``all_edit`` change a recipe, ``meta_edits[(kind, idx)]`` a trial's meta; ``mirror_only`` puts a production
        table in the secured mirror instead of the batch, ``no_prod`` writes none at all."""
        bdir = self.ds / batch
        bdir.mkdir()
        (bdir / "session_metadata.txt").write_text(
            "Session Metadata\n" + (f"Host: {host} | OS: Linux\n" if host else ""), encoding="utf-8")
        (bdir / f"output_{batch}_pretest_1_Citral-1_{day}_100000.csv").write_text("", encoding="utf-8")
        for kind, idx in TRIALS:
            trial = f"{batch}_{kind}_{idx}"
            tdir = self.out / DATASET / batch / "geometry" / trial
            tdir.mkdir(parents=True)
            for fly in FLIES:
                r = recipe(fly, kind, idx)
                if all_edit is not None:
                    all_edit(r)
                if (fly, kind, idx) in (edits or {}):
                    edits[(fly, kind, idx)](r)
                prod, geo = frames(r)
                C.write_table_fmt(geo, C.geometry_table_path(tdir, trial, fly, "parquet"), "parquet")
                if (fly, kind, idx) in no_prod:
                    continue
                home = (self.mirror / DATASET / batch) if (fly, kind, idx) in mirror_only else bdir
                (home / trial).mkdir(parents=True, exist_ok=True)
                prod.to_parquet(home / trial / f"{trial}_fly{fly}_distances.parquet", index=False)
            meta = trial_meta(batch, kind, idx)
            if (kind, idx) in (meta_edits or {}):
                meta_edits[(kind, idx)](meta)
            C.meta_path(tdir, trial).write_text(json.dumps(meta), encoding="utf-8")

    def settings(self, flagged=None, freeze=()):
        over = {DATASET: DatasetOverride(freeze_folders=tuple(FolderRule(f) for f in freeze))} if freeze else {}
        return Settings(model_path="x", main_directories=[str(self.ds)], datasets=(DATASET,),
                        flagged_flies_csv=str(flagged or ""), dataset_overrides=over)

    def gcfg(self, **kw):
        base = dict(out_base=str(self.out), video_bases=(str(self.mirror),), batch_date_from=CUTOFF,
                    slot_seeds="production", table_format="parquet")
        return GeometrySettings(**{**base, **kw})


def default_predictions():
    rows = [(DATASET, NEW, 1, "pretest_1_citral-1", "pretest", 0),
            (DATASET, NEW, 1, "testing_1_citral-1", "testing", 3),
            (DATASET, NEW, 1, "testing_1_citral-1", "testing", 3),            # an exact duplicate row
            (DATASET, NEW, 1, "training_1_citral-1", "testing", 4),           # the real file calls training rows "testing"
            (DATASET, NEW, 2, "pretest_1_citral-1", "pretest", 2),            # exactly the threshold: a response
            (DATASET, NEW, 2, "testing_1_citral-1", "testing", 1),
            (DATASET, NEW, 2, "testing_2_citral-1", "testing", 4),
            (DATASET, OLD, 1, "testing_1_citral-1", "testing", 4)]
    df = pd.DataFrame(rows, columns=["dataset", "fly", "fly_number", "trial_label", "trial_type", "score"])
    df["prediction"] = (df["score"] >= 2).astype(int)
    return df


def write_predictions(path: Path, df=None) -> Path:
    (default_predictions() if df is None else df).to_csv(path, index=False)
    return path


def tables(tree, *, mcfg=MCFG, predictions=None, flagged=None, freeze=(), **gcfg_kw):
    ft, exc = MP.build_fly_trial_table(tree.settings(flagged, freeze), tree.gcfg(**gcfg_kw), mcfg, predictions)
    return ft, exc, MP.build_fly_table(ft, mcfg)


def row(df, fly, kind, idx):
    sel = df[(df["fly_number"] == fly) & (df["trial_type"] == kind) & (df["trial_index"] == idx)]
    assert len(sel) == 1, f"expected exactly one row for fly {fly} {kind}_{idx}, got {len(sel)}"
    return sel.iloc[0]


def variant(tmp_path, **batch_kw):
    """A tree with just the new batch (no September one), built with the given edits."""
    tree = Tree(tmp_path)
    tree.write_batch(NEW, "20261007", **batch_kw)
    return tree


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    """The base scenario: 2 flies x 3 trials in a new batch, a September batch, an untracked trial and predictions."""
    tree = Tree(tmp_path_factory.mktemp("mounting_base"))
    tree.write_batch(NEW, "20261007")
    tree.write_batch(OLD, "20260930")
    untracked = tree.out / DATASET / NEW / "geometry" / f"{NEW}_testing_9"       # tables but no meta: not tracked yet
    untracked.mkdir()
    prod, geo = frames(recipe(1, "testing", 9))
    C.write_table_fmt(geo, C.geometry_table_path(untracked, untracked.name, 1, "parquet"), "parquet")
    preds = write_predictions(tree.root / "model_predictions.csv")
    ft, exc, fly = tables(tree, predictions=preds)
    return SimpleNamespace(tree=tree, ft=ft, exc=exc, fly=fly, preds=preds)


# ---------------------------------------------------------------------------
# The mounting settings block
# ---------------------------------------------------------------------------

DESIGN_YAML = """
mounting:
  enabled: true
  csv_out_dir: /home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/Mounting-Posture
  out_dir: /home/ramanlab/Documents/cole/Results/New-Opto-Fly-Figures/Mounting-Posture
  rest_quantile: 0.10
  retracted_ref_eu: null
  held_margin_eu: 0.25
  held_frac_min: 0.8
  held_back_deg: -10.0
  held_forward_deg: 10.0
  event_rise_eu: 0.30
  event_drop_eu: 0.30
  event_min_s: 0.3
  identity_tol_px: 25
  eye_match_px: 25
  binary_threshold: 2
  n_permutations: 10000
  abdomen_out_dir: /home/ramanlab/Documents/cole/Results/New-Opto-Fly-Figures/Abdomen-Movement
  curl_rise_deg: 10.0
  curl_min_s: 0.3
  tip_move_eu_s: 0.5
  speed_smooth_frames: 5
  abd_min_tracked_frac: 0.5
"""
DESIGN_NUMBERS = dict(rest_quantile=0.10, retracted_ref_eu=None, held_margin_eu=0.25, held_frac_min=0.8,
                      held_back_deg=-10.0, held_forward_deg=10.0, event_rise_eu=0.30, event_drop_eu=0.30,
                      event_min_s=0.3, identity_tol_px=25, eye_match_px=25, binary_threshold=2, n_permutations=10000,
                      curl_rise_deg=10.0, curl_min_s=0.3, tip_move_eu_s=0.5, speed_smooth_frames=5,
                      abd_min_tracked_frac=0.5)


def test_an_empty_block_gives_the_design_values():
    mc = M.parse_mounting_block({})
    for key, want in DESIGN_NUMBERS.items():
        assert getattr(mc, key) == want, key
    assert (mc.enabled, mc.csv_out_dir, mc.out_dir, mc.abdomen_out_dir, mc.batch_date_from) == (True, "", "", "", None)


def test_the_design_yaml_block_parses_to_the_same_numbers():
    block = yaml.safe_load(DESIGN_YAML)["mounting"]
    mc = M.parse_mounting_block(block)
    for key, want in DESIGN_NUMBERS.items():
        assert getattr(mc, key) == want, key
    assert mc.csv_out_dir.endswith("CSVs-New-Opto-Flys/Mounting-Posture")
    assert mc.out_dir.endswith("New-Opto-Fly-Figures/Mounting-Posture")
    assert mc.abdomen_out_dir.endswith("New-Opto-Fly-Figures/Abdomen-Movement")


def test_an_unknown_mounting_key_is_refused_by_name():
    with pytest.raises(ValueError, match="held_marign_eu"):
        M.parse_mounting_block({"held_marign_eu": 0.3})


def test_a_block_that_is_not_a_mapping_is_refused():
    with pytest.raises(ValueError):
        M.parse_mounting_block(None)
    with pytest.raises(ValueError):
        M.parse_mounting_block(["rest_quantile"])


def test_the_batch_cutoff_defaults_to_the_abdomen_blocks_and_an_explicit_one_wins():
    assert M.parse_mounting_block({}, default_batch_date_from=CUTOFF).batch_date_from == CUTOFF
    assert M.parse_mounting_block({"batch_date_from": date(2026, 10, 8)},
                                  default_batch_date_from=CUTOFF).batch_date_from == date(2026, 10, 8)
    assert M.parse_mounting_block({"batch_date_from": "2026-10-09"}).batch_date_from == date(2026, 10, 9)


@pytest.mark.parametrize("bad", [{"batch_date_from": "yesterday"}, {"batch_date_from": "2026-13-40"},
                                 {"rest_quantile": 0.0}, {"rest_quantile": 1.0}, {"held_frac_min": 0.0},
                                 {"held_frac_min": 1.5}, {"held_margin_eu": -0.1}, {"event_min_s": 0.0},
                                 {"curl_min_s": -1}, {"curl_rise_deg": 0}, {"n_permutations": 0},
                                 {"n_permutations": 10.5}, {"speed_smooth_frames": 0}, {"speed_smooth_frames": True},
                                 {"abd_min_tracked_frac": 1.2}, {"eye_match_px": -1}, {"identity_tol_px": "wide"},
                                 {"held_back_deg": 20.0}, {"event_rise_eu": float("nan")}, {"retracted_ref_eu": "x"},
                                 {"enabled": "yes"}, {"csv_out_dir": 5}])
def test_out_of_range_or_mistyped_values_are_refused(bad):
    with pytest.raises(ValueError):
        M.parse_mounting_block(bad)


def test_a_configured_retracted_reference_and_paths_are_kept():
    mc = M.parse_mounting_block({"retracted_ref_eu": 0.6, "csv_out_dir": "~/x/csv", "enabled": False})
    assert mc.retracted_ref_eu == 0.6 and mc.enabled is False
    assert mc.csv_out_dir == str(Path("~/x/csv").expanduser())


# ---------------------------------------------------------------------------
# Rows and scope
# ---------------------------------------------------------------------------

def test_rows_are_the_in_scope_fly_trials_and_the_september_batch_is_absent(base):
    ft = base.ft
    assert len(ft) == 6
    assert set(ft["batch"]) == {NEW}
    assert sorted(zip(ft["fly_number"], ft["trial_type"], ft["trial_index"])) == \
        sorted((f, k, i) for f in FLIES for k, i in TRIALS)
    assert (ft["dataset"] == DATASET).all() and (ft["rig"] == HOST).all()
    assert list(ft.columns) == MP.FLY_TRIAL_COLUMNS
    assert (base.exc["n_batches"], base.exc["n_trials"], base.exc["n_fly_trials"]) == (1, 3, 6)


def test_a_trial_folder_without_a_geometry_meta_is_not_a_row(base):
    assert f"{NEW}_testing_9" not in set(base.ft["trial"])


def test_a_frozen_batch_is_not_read(tmp_path):
    tree = variant(tmp_path)
    ft, exc, fly = tables(tree, freeze=(NEW,))
    assert len(ft) == 0 and len(fly) == 0 and exc["n_batches"] == 0


def test_the_mounting_cutoff_can_be_later_than_the_abdomen_one(tmp_path):
    tree = variant(tmp_path)
    late = M.parse_mounting_block({"batch_date_from": date(2026, 10, 8)})
    ft, _, _ = tables(tree, mcfg=late)
    assert len(ft) == 0
    early = M.parse_mounting_block({"batch_date_from": date(2026, 10, 7)})
    assert len(tables(tree, mcfg=early)[0]) == 6


def test_the_column_lists_have_no_repeats():
    assert len(set(MP.FLY_TRIAL_COLUMNS)) == len(MP.FLY_TRIAL_COLUMNS)
    assert len(set(MP.FLY_COLUMNS)) == len(MP.FLY_COLUMNS)


def test_every_column_is_filled_somewhere_in_the_base_scenario(base):
    empty_ft = [c for c in MP.FLY_TRIAL_COLUMNS if base.ft[c].isna().all()]
    empty_fly = [c for c in MP.FLY_COLUMNS if base.fly[c].isna().all()]
    assert empty_ft == [] and empty_fly == []


def test_the_exclusion_counts_of_a_clean_tree_are_zero(base):
    for key in ("eye_mismatch", "non_square_frame", "warmup_slots", "no_production_table", "unreadable_geometry",
                "flagged_flies", "flagged_fly_trials", "prediction_conflicts", "prediction_same_score_duplicates"):
        assert base.exc[key] == 0, key
    assert base.exc["identity_drift"] == [] and base.exc["predictions_missing"] is False


# ---------------------------------------------------------------------------
# Mounting angle
# ---------------------------------------------------------------------------

def test_the_trial_pitch_is_the_body_pitch_and_the_fly_value_the_median_of_its_trials(base):
    ft, fly = base.ft, base.fly
    assert row(ft, 1, "pretest", 1)["trial_body_pitch_deg"] == pytest.approx(PITCH_5)
    assert row(ft, 1, "testing", 1)["trial_body_pitch_deg"] == pytest.approx(PITCH_11)
    assert row(ft, 1, "testing", 2)["trial_body_pitch_deg"] == pytest.approx(0.0)
    f1, f2 = (fly[fly["fly_number"] == n].iloc[0] for n in FLIES)
    assert np.median([PITCH_5, PITCH_11, 0.0]) == pytest.approx(PITCH_5)
    assert f1["mount_pitch_deg"] == pytest.approx(PITCH_5)
    assert f2["mount_pitch_deg"] == pytest.approx(PITCH_11)


def test_the_centred_pitch_is_the_fly_value_minus_the_rig_median(base):
    fly = base.fly
    median = np.median([PITCH_5, PITCH_11])
    centred = dict(zip(fly["fly_number"], fly["mount_pitch_centred_deg"]))
    assert centred[1] == pytest.approx(PITCH_5 - median) and centred[2] == pytest.approx(PITCH_11 - median)
    assert sum(centred.values()) == pytest.approx(0.0)


def test_the_centring_is_within_each_rig(tmp_path):
    def rolled(r):                       # another camera roll: every fly of this rig sits at 26.6 degrees
        r["abd"] = (200.0, 100.0)
    tree = Tree(tmp_path)
    tree.write_batch(NEW, "20261007", host="Flybehavior2")
    tree.write_batch("october_08_batch_1_rig_3", "20261008", host="Flybehavior3", all_edit=rolled)
    _, _, fly = tables(tree)
    assert set(fly["rig"]) == {"Flybehavior2", "Flybehavior3"} and len(fly) == 4
    for _, per_rig in fly.groupby("rig"):
        assert per_rig["mount_pitch_centred_deg"].median() == pytest.approx(0.0)
    assert fly.groupby("rig")["mount_pitch_deg"].median().nunique() == 2


def test_the_trial_pitch_is_the_median_over_the_pre_stimulus_frames_only(tmp_path):
    tree = Tree(tmp_path)
    tree.write_batch(NEW, "20261007")
    # rewrite one geometry table: the abdomen sits at 5.7 degrees for 200 frames, then at 11.3 degrees
    tdir = tree.out / DATASET / NEW / "geometry" / f"{NEW}_pretest_1"
    path = C.geometry_table_path(tdir, tdir.name, 1, "parquet")
    geo = pd.read_parquet(path)
    geo["abdomen_y"] = geo["eye_y"] + np.where(np.arange(N) < 200, 20.0, 40.0)
    C.write_table_fmt(geo, path, "parquet")
    pitch = row(tables(tree)[0], 1, "pretest", 1)["trial_body_pitch_deg"]
    assert pitch == pytest.approx(PITCH_5)                      # 200 + 100 pre frames
    assert np.median(np.where(np.arange(N) < 200, PITCH_5, PITCH_11)) != pytest.approx(PITCH_5)     # the whole trial: 8.5
    assert np.mean(np.r_[np.full(200, PITCH_5), np.full(100, PITCH_11)]) != pytest.approx(PITCH_5)  # and a mean: 7.6


# ---------------------------------------------------------------------------
# Eye units, spontaneous PER and pull-ins
# ---------------------------------------------------------------------------

def test_rest_is_in_eye_units_and_a_pre_onset_excursion_is_a_spontaneous_per_event(base):
    t1 = row(base.ft, 1, "pretest", 1)
    assert t1["rest_eu"] == pytest.approx(24.0 / 48.0)
    assert t1["rest_theta_deg"] == pytest.approx(0.0, abs=1e-9)
    assert t1["spont_per_events"] == 1 and t1["spont_per_any"] == 1
    assert t1["pullin_events_pre"] == 0 and t1["pullin_events_stim"] == 0


def test_a_too_small_or_too_short_excursion_is_not_an_event(base):
    small = row(base.ft, 1, "testing", 1)          # +10 px = 0.21 eu for 0.5 s
    short = row(base.ft, 1, "testing", 2)          # +20 px for 0.2 s
    for t in (small, short):
        assert t["spont_per_events"] == 0 and t["spont_per_any"] == 0


@pytest.mark.parametrize("setting, who, column, want", [
    ({"event_rise_eu": 0.5}, (1, "pretest", 1), "spont_per_events", 0),          # +20 px is 0.417 eu
    ({"event_min_s": 0.6}, (1, "pretest", 1), "spont_per_events", 0),            # the excursion lasts 0.5 s
    ({"event_rise_eu": 0.1, "event_min_s": 0.2}, (1, "pretest", 1), "spont_per_events", 1),
    ({"event_drop_eu": 0.5}, (2, "testing", 1), "pullin_events_stim", 0),        # the dip is 20 px = 0.417 eu
    ({"curl_rise_deg": 20.0}, (1, "pretest", 1), "abd_curl_events_pre", 0),      # the curl is +15 degrees
    ({"curl_min_s": 0.6}, (1, "pretest", 1), "abd_curl_events_pre", 0),
    ({"tip_move_eu_s": 1.5}, (1, "testing", 1), "tip_moving_frac_stim", 0.0),    # the tip moves at 1.0 eu/s
])
def test_the_event_thresholds_come_from_the_mounting_settings(tmp_path, setting, who, column, want):
    mc = M.parse_mounting_block(setting, default_batch_date_from=CUTOFF)
    ft, _, _ = tables(variant(tmp_path), mcfg=mc)
    assert row(ft, *who)[column] == want


def test_the_rest_is_the_median_of_the_frames_at_or_below_the_rest_quantile(tmp_path):
    ramp = np.linspace(24.0, 64.0, N)                           # a proboscis that never settles: the quantile matters

    def drifting_proboscis(r):
        r["dist"] = ramp.copy()
    tree = variant(tmp_path, edits={(2, "pretest", 1): drifting_proboscis})
    eu = ramp / SCALE
    for q in (0.10, 0.50):
        rest = np.median(eu[eu <= np.quantile(eu, q)])
        mc = M.parse_mounting_block({"rest_quantile": q}, default_batch_date_from=CUTOFF)
        assert row(tables(tree, mcfg=mc)[0], 2, "pretest", 1)["rest_eu"] == pytest.approx(rest), q
    assert np.median(eu[eu <= np.quantile(eu, 0.10)]) != pytest.approx(np.median(eu[eu <= np.quantile(eu, 0.50)]))


def test_a_dip_before_the_stimulus_is_a_pre_pull_in_and_not_a_stimulus_one(base):
    t = row(base.ft, 1, "testing", 2)
    assert (t["pullin_events_pre"], t["pullin_events_stim"]) == (1, 0)


def test_a_dip_during_the_stimulus_on_a_held_trial_is_one_pull_in(base):
    held = row(base.ft, 2, "testing", 1)
    assert held["pullin_events_stim"] == 1 and held["pullin_events_pre"] == 0
    assert held["rest_eu"] == pytest.approx(60.0 / 48.0)
    assert held["spont_per_events"] == 0


def test_the_eye_is_the_live_eye_when_there_is_one_and_the_frozen_anchor_otherwise(tmp_path):
    def live(r):
        r["prod_edit"] = lambda p: p.assign(eye_live_x=p["x_class0"] + 6.0)      # the live eye sits 6 px to the right

    def frozen(r):
        r["prod_edit"] = lambda p: p.drop(columns=["eye_live_x", "eye_live_y"])
    ft_live, _, _ = tables(variant(tmp_path / "a", edits={(1, "testing", 2): live}))
    ft_froz, _, _ = tables(variant(tmp_path / "b", edits={(1, "testing", 2): frozen}))
    assert row(ft_live, 1, "testing", 2)["rest_eu"] == pytest.approx(math.hypot(6.0, 24.0) / 48.0)
    assert row(ft_live, 1, "testing", 2)["rest_theta_deg"] == pytest.approx(math.degrees(math.atan2(-6.0, 24.0)))
    assert row(ft_live, 1, "testing", 2)["anchor_x"] == EYES[1][0]            # the identity is the frozen anchor, not the live eye
    assert row(ft_froz, 1, "testing", 2)["rest_eu"] == pytest.approx(0.5)


def test_the_anchor_side_decides_which_way_is_forward(tmp_path):
    def mirrored(r):              # anchor on the right edge and the proboscis mirrored: same theta
        r["theta"] = -15.0
        r["dist"] = np.full(N, 60.0)

        def flip(p):
            p = p.assign(x_anchor=1080.0)
            p["x_class1"] = 2 * p["x_class0"] - p["x_class1"]
            return p
        r["prod_edit"] = flip
    ft, _, _ = tables(variant(tmp_path, edits={(1, "testing", 2): mirrored}))
    assert row(ft, 1, "testing", 2)["rest_theta_deg"] == pytest.approx(-15.0)


def test_production_frames_are_aligned_by_their_frame_index(tmp_path):
    def dropped(r):               # production lost frames 0-9, and a dip straddles the stimulus onset
        r["dist"][296:304] -= 20.0
        r["prod_edit"] = lambda p: p[p["frame"] >= 10].reset_index(drop=True)
    ft, _, _ = tables(variant(tmp_path, edits={(2, "testing", 2): dropped}))
    t = row(ft, 2, "testing", 2)
    assert (t["pullin_events_pre"], t["pullin_events_stim"]) == (1, 1)


def test_a_production_table_without_a_frame_column_is_read_by_position(tmp_path):
    def bare(r):
        r["dist"][100:105] += 20.0
        r["prod_edit"] = lambda p: p.drop(columns=["frame"])
    ft, _, _ = tables(variant(tmp_path, edits={(2, "testing", 2): bare}))
    assert row(ft, 2, "testing", 2)["spont_per_events"] == 1


def test_the_proboscis_to_body_angle_is_measured_at_the_rest_frames(base):
    # eye -> proboscis points straight down; the body axis (abdomen -> eye) points left and 5.7 degrees up
    t1 = row(base.ft, 1, "pretest", 1)
    assert t1["proboscis_rest_vs_body_deg"] == pytest.approx(90.0 + PITCH_5, abs=1e-6)


# ---------------------------------------------------------------------------
# Held out and the retracted reference
# ---------------------------------------------------------------------------

def test_a_trial_parked_behind_the_eye_is_held_out_with_subtype_held_back(base):
    held = row(base.ft, 2, "testing", 1)
    assert held["held_out"] == 1 and held["held_subtype"] == "held_back"
    assert held["rest_theta_deg"] == pytest.approx(-15.0, abs=1e-6)


def test_retracted_trials_are_labelled_retracted_and_the_reference_is_the_rig_median_minimum(base):
    ft = base.ft
    retracted = ft[~((ft["fly_number"] == 2) & (ft["trial_index"] == 1) & (ft["trial_type"] == "testing"))]
    assert (retracted["held_out"] == 0).all() and (retracted["held_subtype"] == "retracted").all()
    assert ft["retracted_ref_eu"].to_numpy() == pytest.approx(0.5)


@pytest.mark.parametrize("ref, held", [(0.9, True), (1.2, False)])
def test_a_configured_reference_overrides_the_rig_derived_one(tmp_path, ref, held):
    mc = M.parse_mounting_block({"retracted_ref_eu": ref}, default_batch_date_from=CUTOFF)
    ft, _, _ = tables(variant(tmp_path), mcfg=mc)
    assert (ft["retracted_ref_eu"] == ref).all()
    t = row(ft, 2, "testing", 1)                    # rest 1.25 eu against ref + 0.25
    assert t["held_out"] == (1 if held else 0)
    assert t["held_subtype"] == ("held_back" if held else "retracted")


def test_the_held_margin_and_fraction_come_from_the_settings(tmp_path):
    tree = variant(tmp_path)
    wide = M.parse_mounting_block({"held_margin_eu": 1.0}, default_batch_date_from=CUTOFF)     # 0.5 + 1.0 > 1.25
    assert row(tables(tree, mcfg=wide)[0], 2, "testing", 1)["held_out"] == 0


def test_the_reference_is_built_from_each_flys_most_retracted_trial(tmp_path):
    def held(r):
        r["dist"] = np.full(N, 60.0)
    tree = variant(tmp_path, edits={(2, "testing", 1): held, (2, "testing", 2): held})         # fly 2: 0.5, 1.25, 1.25
    ft, _, _ = tables(tree)
    assert ft["retracted_ref_eu"].to_numpy() == pytest.approx(0.5)          # a median over its trials would give 0.875


def test_each_rig_has_its_own_retracted_reference(tmp_path):
    def held(r):
        r["dist"] = np.full(N, 96.0)                            # rest 2.0 eu in every trial of this rig
    tree = Tree(tmp_path)
    tree.write_batch(NEW, "20261007", host="Flybehavior2")
    tree.write_batch("october_08_batch_1_rig_3", "20261008", host="Flybehavior3", all_edit=held)
    ft, _, _ = tables(tree)
    assert dict(ft.groupby("rig")["retracted_ref_eu"].first()) == pytest.approx({"Flybehavior2": 0.5, "Flybehavior3": 2.0})
    assert row(ft[ft["rig"] == "Flybehavior2"], 2, "testing", 1)["held_out"] == 1       # 1.25 eu against 0.5 + 0.25
    assert ft.loc[ft["rig"] == "Flybehavior3", "held_out"].eq(0).all()


def test_held_forward_and_held_down_follow_the_rest_angle(tmp_path):
    def at(theta):
        def edit(r):
            r["dist"] = np.full(N, 60.0)
            r["theta"] = theta
        return edit
    tree = variant(tmp_path, edits={(2, "testing", 2): at(15.0), (2, "pretest", 1): at(3.0)})
    ft, _, _ = tables(tree)
    assert row(ft, 2, "testing", 2)["held_subtype"] == "held_forward"
    assert row(ft, 2, "pretest", 1)["held_subtype"] == "held_down"


def test_held_out_is_unknown_for_every_trial_when_no_rest_is_known(tmp_path):
    def blind(r):
        r["dist"] = np.full(N, np.nan)
    tree = variant(tmp_path, all_edit=blind)
    ft, _, fly = tables(tree)
    assert ft["rest_eu"].isna().all() and ft["retracted_ref_eu"].isna().all()
    for col in ("held_out", "held_subtype", "spont_per_events", "spont_per_any", "pullin_events_pre",
                "pullin_events_stim", "proboscis_rest_vs_body_deg", "per_abdomen_coupling_rho"):
        assert ft[col].isna().all(), col
    assert ft["abd_quality_ok"].sum() == 5 and ft.loc[ft["abd_quality_ok"], "abd_rest_bend_deg"].notna().all()
    assert fly["held_out_frac"].isna().all() and fly["n_held_known"].eq(0).all()


def test_held_out_is_unknown_for_a_trial_whose_own_rest_is_unknown(tmp_path):
    def sparse(r):                  # only 30 finite proboscis frames: below the 40 a rest level needs
        r["dist"] = np.full(N, np.nan)
        r["dist"][:30] = 60.0
    ft, _, _ = tables(variant(tmp_path, edits={(2, "testing", 2): sparse}))
    t = row(ft, 2, "testing", 2)
    assert math.isnan(t["rest_eu"]) and math.isnan(t["held_out"]) and pd.isna(t["held_subtype"])
    assert math.isnan(t["spont_per_events"])
    assert row(ft, 2, "pretest", 1)["held_out"] == 0 and row(ft, 2, "testing", 2)["retracted_ref_eu"] == 0.5


def test_a_held_subtype_needs_a_rest_angle(tmp_path):
    def no_anchor(r):
        r["dist"] = np.full(N, 60.0)
        r["prod_edit"] = lambda p: p.assign(x_anchor=np.nan)
    ft, _, _ = tables(variant(tmp_path, edits={(2, "testing", 2): no_anchor}))
    t = row(ft, 2, "testing", 2)
    assert t["held_out"] == 1 and pd.isna(t["held_subtype"]) and math.isnan(t["rest_theta_deg"])


# ---------------------------------------------------------------------------
# The join with production and the geometry
# ---------------------------------------------------------------------------

def test_a_geometry_eye_far_from_the_production_anchor_drops_the_row(tmp_path):
    def far(r):
        r["geo_eye"] = (r["eye"][0] + 40.0, r["eye"][1])
    ft, exc, _ = tables(variant(tmp_path, edits={(2, "testing", 2): far}))
    assert len(ft) == 5 and exc["eye_mismatch"] == 1
    assert ((ft["fly_number"] == 2) & (ft["trial_index"] == 2) & (ft["trial_type"] == "testing")).sum() == 0


def test_a_geometry_eye_within_the_tolerance_is_joined(tmp_path):
    def near(r):
        r["geo_eye"] = (r["eye"][0] + 20.0, r["eye"][1])
    ft, exc, _ = tables(variant(tmp_path, edits={(2, "testing", 2): near}))
    assert len(ft) == 6 and exc["eye_mismatch"] == 0


def test_a_geometry_eye_exactly_at_the_tolerance_is_joined(tmp_path):
    def edge(r):
        r["geo_eye"] = (r["eye"][0] + 25.0, r["eye"][1])
    ft, exc, _ = tables(variant(tmp_path, edits={(2, "testing", 2): edge}))
    assert len(ft) == 6 and exc["eye_mismatch"] == 0


def test_the_eye_tolerance_is_a_setting(tmp_path):
    def near(r):
        r["geo_eye"] = (r["eye"][0] + 20.0, r["eye"][1])
    tight = M.parse_mounting_block({"eye_match_px": 10}, default_batch_date_from=CUTOFF)
    ft, exc, _ = tables(variant(tmp_path, edits={(2, "testing", 2): near}), mcfg=tight)
    assert len(ft) == 5 and exc["eye_mismatch"] == 1


def test_a_trial_without_a_production_table_anywhere_is_dropped_and_counted(tmp_path):
    ft, exc, _ = tables(variant(tmp_path, no_prod={(1, "testing", 2)}))
    assert len(ft) == 5 and exc["no_production_table"] == 1


def test_a_missing_local_production_trial_is_read_from_the_secured_mirror(tmp_path):
    ft, exc, _ = tables(variant(tmp_path, mirror_only={(1, "pretest", 1)}))
    t = row(ft, 1, "pretest", 1)
    assert len(ft) == 6 and exc["no_production_table"] == 0 and t["rest_eu"] == pytest.approx(0.5)


def prod_path(tree, fly, kind, idx, batch=NEW):
    trial = f"{batch}_{kind}_{idx}"
    return tree.ds / batch / trial / f"{trial}_fly{fly}_distances.parquet"


def test_a_production_table_in_csv_form_is_read(tmp_path):
    tree = variant(tmp_path)
    path = prod_path(tree, 1, "pretest", 1)
    pd.read_parquet(path).to_csv(path.with_suffix(".csv"), index=False)
    path.unlink()
    ft, exc, _ = tables(tree)
    assert exc["no_production_table"] == 0 and row(ft, 1, "pretest", 1)["rest_eu"] == pytest.approx(0.5)


def test_a_corrupt_production_table_is_a_missing_one(tmp_path):
    tree = variant(tmp_path)
    prod_path(tree, 1, "pretest", 1).write_bytes(b"not a parquet file")
    ft, exc, _ = tables(tree)
    assert len(ft) == 5 and exc["no_production_table"] == 1


def test_a_production_table_without_the_anchor_column_is_not_used(tmp_path):
    tree = variant(tmp_path)
    path = prod_path(tree, 1, "pretest", 1)
    pd.read_parquet(path).drop(columns=["x_anchor"]).to_parquet(path, index=False)
    ft, exc, _ = tables(tree)
    assert len(ft) == 5 and exc["no_production_table"] == 1


def test_the_rms_calculations_copies_are_never_read(tmp_path):
    tree = variant(tmp_path)
    primary = prod_path(tree, 1, "pretest", 1)
    copy = tree.ds / NEW / "RMS_calculations" / f"updated_{primary.name}"
    copy.parent.mkdir()
    primary.rename(copy)
    ft, exc, _ = tables(tree)
    assert len(ft) == 5 and exc["no_production_table"] == 1


def test_a_corrupt_geometry_table_is_counted_and_dropped(tmp_path):
    tree = variant(tmp_path)
    tdir = tree.out / DATASET / NEW / "geometry" / f"{NEW}_pretest_1"
    C.geometry_table_path(tdir, tdir.name, 1, "parquet").write_bytes(b"junk")
    ft, exc, _ = tables(tree)
    assert len(ft) == 5 and exc["unreadable_geometry"] == 1


@pytest.mark.parametrize("damage", ["garbage", "no_fps", "no_trial_block"])
def test_an_unusable_geometry_meta_is_counted_and_the_other_trials_still_load(tmp_path, damage):
    tree = variant(tmp_path)
    tdir = tree.out / DATASET / NEW / "geometry" / f"{NEW}_pretest_1"
    meta_file = C.meta_path(tdir, tdir.name)
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    if damage == "garbage":
        meta_file.write_text("{not json", encoding="utf-8")
    elif damage == "no_fps":
        del meta["fps"], meta["trial"]["fps"]
        meta_file.write_text(json.dumps(meta), encoding="utf-8")
    else:
        del meta["trial"]
        meta_file.write_text(json.dumps(meta), encoding="utf-8")
    ft, exc, _ = tables(tree)
    assert len(ft) == 4 and exc["unreadable_geometry"] == 1 and exc["n_trials"] == 2


def test_warm_up_numbered_slots_are_dropped_when_slots_follow_production(tmp_path):
    def warm(meta):
        meta["slot_source"] = "warmup"
    tree = variant(tmp_path, meta_edits={("testing", 2): warm})
    ft, exc, _ = tables(tree)
    assert len(ft) == 4 and exc["warmup_slots"] == 2
    ft_all, exc_all, _ = tables(tree, slot_seeds="warmup")
    assert len(ft_all) == 6 and exc_all["warmup_slots"] == 0


@pytest.mark.parametrize("sx, sy", [(1.5, 1.5), (1.0, 1.5), (1.5, 1.0)])
def test_a_non_square_frame_gets_no_eye_unit_metrics_but_keeps_the_angles(tmp_path, sx, sy):
    def wide(meta):
        meta["frame"].update(width=int(1080 * sx), height=int(1080 * sy), sx=sx, sy=sy)
    ft, exc, _ = tables(variant(tmp_path, meta_edits={("testing", 2): wide}))
    t = row(ft, 1, "testing", 2)
    assert exc["non_square_frame"] == 2 and len(ft) == 6
    for col in ("rest_eu", "rest_theta_deg", "held_out", "spont_per_events", "pullin_events_stim",
                "tip_speed_med_pre_eu_s", "per_abdomen_coupling_rho", "anchor_x"):
        assert pd.isna(t[col]), col
    assert t["trial_body_pitch_deg"] == pytest.approx(0.0) and t["abd_curl_events_pre"] == 1
    assert row(ft, 1, "pretest", 1)["rest_eu"] == pytest.approx(0.5)


def test_the_rig_is_the_recording_host_else_the_rig_in_the_batch_name_else_unknown(tmp_path):
    tree = Tree(tmp_path)
    tree.write_batch(NEW, "20261007", host=None)
    tree.write_batch("october_08_batch_1_rig_3", "20261008", host=None)
    tree.write_batch("october_09_batch_1", "20261009", host="BehaviorLocust")
    ft, _, _ = tables(tree)
    assert dict(ft.groupby("batch")["rig"].first()) == {NEW: "unknown", "october_08_batch_1_rig_3": "rig_3",
                                                        "october_09_batch_1": "BehaviorLocust"}


def test_the_anchors_are_the_production_eye_of_the_trial(base):
    t = row(base.ft, 2, "testing", 1)
    assert (t["anchor_x"], t["anchor_y"]) == EYES[2]


# ---------------------------------------------------------------------------
# Fly identity
# ---------------------------------------------------------------------------

def drifting(r):
    r["eye"] = r["geo_eye"] = (r["eye"][0], r["eye"][1] + 60.0)           # production and geometry both move 60 px


def test_a_fly_whose_anchor_moves_keeps_its_rows_but_leaves_the_fly_table(tmp_path):
    ft, exc, fly = tables(variant(tmp_path, edits={(2, "testing", 2): drifting}))
    assert len(ft) == 6
    assert ft.loc[ft["fly_number"] == 2, "identity_consistent"].eq(False).all()
    assert ft.loc[ft["fly_number"] == 1, "identity_consistent"].eq(True).all()
    assert list(fly["fly_number"]) == [1]
    assert len(exc["identity_drift"]) == 1
    drift = exc["identity_drift"][0]
    assert drift["fly_id"] == f"{DATASET}/{NEW}/fly2" and drift["spread_px"] == pytest.approx(60.0)
    assert exc["eye_mismatch"] == 0


def test_the_identity_tolerance_is_a_setting(tmp_path):
    loose = M.parse_mounting_block({"identity_tol_px": 80}, default_batch_date_from=CUTOFF)
    ft, exc, fly = tables(variant(tmp_path, edits={(2, "testing", 2): drifting}), mcfg=loose)
    assert exc["identity_drift"] == [] and len(fly) == 2 and ft["identity_consistent"].all()


def hold_fly_2(r):
    if r["eye"] == EYES[2]:
        r["dist"] = np.full(N, 96.0)                            # rest 2.0 eu in every trial of fly 2


def test_a_drifting_fly_does_not_set_the_rig_reference(tmp_path):
    tree = variant(tmp_path, all_edit=hold_fly_2, edits={(2, "testing", 2): drifting})
    ft, _, _ = tables(tree)
    assert ft["retracted_ref_eu"].to_numpy() == pytest.approx(0.5)        # fly 1 alone; with fly 2 it would be 1.25
    assert ft.loc[ft["fly_number"] == 2, "held_out"].eq(1).all()


def test_a_rig_whose_flies_all_drift_has_no_reference_and_so_no_held_out_state(tmp_path):
    tree = variant(tmp_path, edits={(1, "testing", 2): drifting, (2, "testing", 2): drifting})
    ft, exc, fly = tables(tree)
    assert len(ft) == 6 and ft["rest_eu"].notna().all() and len(exc["identity_drift"]) == 2
    assert ft["retracted_ref_eu"].isna().all() and ft["held_out"].isna().all() and ft["held_subtype"].isna().all()
    assert len(fly) == 0 and list(fly.columns) == MP.FLY_COLUMNS


def test_flagged_flies_are_removed_and_counted(tmp_path):
    tree = variant(tmp_path)
    flagged = tree.root / "flagged-flys-truth.csv"
    flagged.write_text('dataset,fly,fly_number,"FLY-State(1, 0, -1)",comment\n'
                       f"{DATASET},{NEW},2,0,Bad\n{DATASET},{NEW},1,1,Fine\n", encoding="utf-8")
    ft, exc, fly = tables(tree, flagged=flagged)
    assert set(ft["fly_number"]) == {1} and list(fly["fly_number"]) == [1]
    assert (exc["flagged_flies"], exc["flagged_fly_trials"]) == (1, 3)



# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def test_scores_join_on_the_trial_parsed_from_the_label_and_respond_at_the_threshold(base):
    ft = base.ft
    want = {(1, "pretest", 1): (0, 0), (1, "testing", 1): (3, 1), (2, "pretest", 1): (2, 1), (2, "testing", 1): (1, 0),
            (2, "testing", 2): (4, 1)}
    for (fly, kind, idx), (score, responded) in want.items():
        t = row(ft, fly, kind, idx)
        assert (t["score"], t["responded"]) == (score, responded), (fly, kind, idx)
    t = row(ft, 1, "testing", 2)                                            # no prediction row: NaN, not 0
    assert math.isnan(t["score"]) and math.isnan(t["responded"])


def test_a_training_label_scored_as_testing_does_not_overwrite_a_testing_trial(base):
    assert row(base.ft, 1, "testing", 1)["score"] == 3          # the file also has training_1 = 4 for this fly


def test_a_training_trial_takes_no_score_even_when_the_file_calls_its_row_testing(tmp_path):
    df = pd.DataFrame({"dataset": DATASET, "batch": NEW, "fly_number": 1, "trial_type": ["training", "testing"],
                       "trial_index": 1})
    MP._attach_scores(df, write_predictions(tmp_path / "p.csv"), MCFG, {})
    assert math.isnan(df["score"][0]) and df["score"][1] == 3


def test_float_fly_numbers_in_the_predictions_still_join(tmp_path):
    tree = variant(tmp_path)
    preds = default_predictions()
    odd = preds.iloc[[0]].assign(dataset="Other-Dataset", fly_number=np.nan)       # one blank makes the column float
    path = write_predictions(tree.root / "p.csv", pd.concat([preds, odd]))
    assert pd.read_csv(path)["fly_number"].dtype == float
    ft, _, _ = tables(tree, predictions=path)
    assert row(ft, 1, "testing", 1)["score"] == 3 and row(ft, 2, "testing", 2)["score"] == 4


def test_a_duplicate_prediction_row_is_kept_once_not_dropped(base):
    assert base.exc["prediction_exact_duplicates"] == 1 and base.exc["prediction_conflicts"] == 0
    assert row(base.ft, 1, "testing", 1)["score"] == 3


def test_conflicting_scores_drop_the_trial_score_and_are_counted(tmp_path):
    tree = variant(tmp_path)
    preds = default_predictions()
    extra = preds[(preds["fly_number"] == 2) & (preds["trial_label"] == "testing_2_citral-1")].assign(score=0, prediction=0)
    ft, exc, _ = tables(tree, predictions=write_predictions(tree.root / "p.csv", pd.concat([preds, extra])))
    assert exc["prediction_conflicts"] == 1
    assert math.isnan(row(ft, 2, "testing", 2)["score"]) and row(ft, 2, "testing", 1)["score"] == 1


def test_labels_that_differ_only_in_the_odor_and_agree_on_the_score_join_once(tmp_path):
    tree = variant(tmp_path)
    preds = default_predictions()
    twin = preds[(preds["fly_number"] == 2) & (preds["trial_label"] == "testing_2_citral-1")].assign(
        trial_label="testing_2_citral-0.1")
    ft, exc, _ = tables(tree, predictions=write_predictions(tree.root / "p.csv", pd.concat([preds, twin])))
    assert len(ft) == 6 and row(ft, 2, "testing", 2)["score"] == 4 and exc["prediction_key_conflicts"] == 0


def test_predictions_without_a_pretest_or_testing_row_leave_the_scores_empty(tmp_path):
    tree = variant(tmp_path)
    only_training = pd.DataFrame([(DATASET, NEW, 1, "training_1_citral-1", "testing", 4, 1)],
                                 columns=["dataset", "fly", "fly_number", "trial_label", "trial_type", "score", "prediction"])
    ft, exc, _ = tables(tree, predictions=write_predictions(tree.root / "p.csv", only_training))
    assert len(ft) == 6 and ft["score"].isna().all() and ft["responded"].isna().all() and not exc["predictions_missing"]


def test_the_missing_predictions_flag_is_set_even_when_nothing_is_in_scope(tmp_path):
    late = M.parse_mounting_block({"batch_date_from": date(2030, 1, 1)})
    ft, exc, _ = tables(variant(tmp_path), mcfg=late, predictions=tmp_path / "nope.csv")
    assert len(ft) == 0 and exc["predictions_missing"] is True and exc["predictions_csv"].endswith("nope.csv")


def test_the_response_threshold_is_a_setting(tmp_path):
    tree = variant(tmp_path)
    strict = M.parse_mounting_block({"binary_threshold": 4}, default_batch_date_from=CUTOFF)
    ft, _, _ = tables(tree, mcfg=strict, predictions=write_predictions(tree.root / "p.csv"))
    assert row(ft, 1, "testing", 1)["responded"] == 0 and row(ft, 2, "testing", 2)["responded"] == 1
    assert row(ft, 2, "pretest", 1)["responded"] == 0


def test_a_missing_predictions_file_leaves_the_scores_empty_and_says_so(tmp_path):
    tree = variant(tmp_path)
    ft, exc, fly = tables(tree, predictions=tree.root / "nope.csv")
    assert ft["score"].isna().all() and exc["predictions_missing"] is True
    assert fly["n_scored"].eq(0).all() and fly["response_rate"].isna().all()


def test_the_fly_response_rate_pools_the_scored_trials(base):
    f1, f2 = (base.fly[base.fly["fly_number"] == n].iloc[0] for n in FLIES)
    assert (f1["n_scored"], f1["n_responded"], f1["response_rate"]) == (2, 1, 0.5)
    assert (f2["n_scored"], f2["n_responded"]) == (3, 2) and f2["response_rate"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def test_the_windows_come_from_the_geometry_meta(base):
    t = row(base.ft, 1, "pretest", 1)
    assert (t["onset_frame"], t["offset_frame"], t["fps"], t["n_frames"]) == (ONSET, OFFSET, FPS, N)
    assert t["pre_window_known"] and t["pre_min"] == pytest.approx(0.5) and t["stim_min"] == pytest.approx(40 / 10 / 60)
    assert t["stim_kind"] == "odor" and t["eye_scale_px"] == SCALE


def test_a_light_only_trial_never_uses_its_filler_odor_frame(tmp_path):
    def light_only(meta):
        meta["trial"].update(has_odor=False, odor_on_frame=1201, odor_off_frame=None, light_on_s=15.0, light_off_s=19.0)
    ft, _, _ = tables(variant(tmp_path, meta_edits={("testing", 2): light_only}))
    t = row(ft, 1, "testing", 2)
    assert (t["onset_frame"], t["offset_frame"], t["stim_kind"]) == (150, 190, "light")
    assert t["pre_min"] == pytest.approx(150 / 10 / 60)


def test_a_meta_without_the_top_level_fps_uses_the_trials(tmp_path):
    def trial_only(meta):
        del meta["fps"]
    ft, exc, _ = tables(variant(tmp_path, meta_edits={("testing", 2): trial_only}))
    assert len(ft) == 6 and exc["unreadable_geometry"] == 0 and row(ft, 1, "testing", 2)["fps"] == FPS


def test_a_stimulus_with_no_known_end_runs_to_the_end_of_the_trial(tmp_path):
    def open_ended(meta):
        meta["trial"].update(has_odor=False, odor_on_frame=1201, odor_off_frame=None, light_on_s=15.0, light_off_s=None)
    ft, _, _ = tables(variant(tmp_path, meta_edits={("testing", 2): open_ended}))
    t = row(ft, 1, "testing", 2)
    assert t["onset_frame"] == 150 and math.isnan(t["offset_frame"]) and t["stim_min"] == pytest.approx(250 / 10 / 60)
    assert (t["abd_curl_events_pre"], t["abd_curl_events_stim"]) == (0, 1)          # the curl at frame 200 is now evoked


def test_a_trial_with_odor_and_light_has_the_union_of_the_two_windows(tmp_path):
    def both(meta):
        meta["trial"].update(light_on_s=29.0, light_off_s=33.0)
    ft, _, _ = tables(variant(tmp_path, meta_edits={("testing", 2): both}))
    t = row(ft, 1, "testing", 2)
    assert (t["onset_frame"], t["offset_frame"], t["stim_kind"]) == (290, 340, "odor+light")


def test_a_trial_with_no_known_stimulus_has_no_event_statistics(tmp_path):
    def none(meta):
        meta["trial"].update(has_odor=False, odor_on_frame=1201, odor_off_frame=None, light_on_s=None, light_off_s=None)
    ft, _, fly = tables(variant(tmp_path, meta_edits={("testing", 2): none}))
    t = row(ft, 1, "testing", 2)
    assert not t["pre_window_known"] and t["stim_kind"] == "none"
    for col in ("onset_frame", "pre_min", "stim_min", "trial_body_pitch_deg", "spont_per_events", "spont_per_any",
                "pullin_events_pre", "pullin_events_stim", "held_out", "abd_curl_events_pre", "abd_curl_events_stim",
                "abd_rest_bend_deg", "abd_tracked_frac_pre"):
        assert pd.isna(t[col]), col
    assert not t["abd_quality_ok"] and t["rest_eu"] == pytest.approx(0.5)          # the rest level needs no window
    f1 = fly[fly["fly_number"] == 1].iloc[0]
    assert f1["n_spont_known"] == 2                                              # that trial leaves the denominator


# ---------------------------------------------------------------------------
# Abdomen movement
# ---------------------------------------------------------------------------

def test_a_pre_onset_ventral_curl_is_one_spontaneous_curl_event(base):
    t1 = row(base.ft, 1, "pretest", 1)
    assert t1["abd_curl_events_pre"] == 1 and t1["abd_straighten_events_pre"] == 0
    assert t1["abd_curl_events_stim"] == 0 and t1["abd_quality_ok"] and t1["abd_quality_ok_stim"]
    t2 = row(base.ft, 1, "testing", 1)                      # a 0.2 s +15 degree blip is below curl_min_s
    assert t2["abd_curl_events_pre"] == 0


def test_a_stimulus_window_curl_is_an_evoked_event_with_a_positive_delta(base):
    t2 = row(base.ft, 1, "testing", 1)
    assert t2["abd_curl_events_stim"] == 1 and t2["abd_curl_events_pre"] == 0
    assert t2["abd_curl_delta_deg"] == pytest.approx(15.0) and t2["abd_curl_delta_deg"] > 0
    assert t2["abd_rest_bend_deg"] == pytest.approx(5.0)


def test_the_resting_bend_is_the_pre_window_median(base):
    bend = 4.0 + 0.01 * np.arange(N)
    bend[150:155] += 15.0
    assert row(base.ft, 1, "pretest", 1)["abd_rest_bend_deg"] == pytest.approx(np.median(bend[:ONSET]))


def test_the_bend_sd_is_per_window(base):
    t1 = row(base.ft, 1, "pretest", 1)
    bend = 4.0 + 0.01 * np.arange(N)
    bend[150:155] += 15.0
    assert t1["abd_bend_sd_pre_deg"] == pytest.approx(np.std(bend[:ONSET]))
    assert t1["abd_bend_sd_stim_deg"] == pytest.approx(np.std(bend[ONSET:OFFSET]))
    assert row(base.ft, 2, "pretest", 1)["abd_bend_sd_pre_deg"] == pytest.approx(0.0)


def test_a_straightening_is_counted_apart_from_a_curl(tmp_path):
    def straighten(r):
        r["bend"][100:110] -= 15.0
    ft, _, _ = tables(variant(tmp_path, edits={(2, "pretest", 1): straighten}))
    t = row(ft, 2, "pretest", 1)
    assert (t["abd_curl_events_pre"], t["abd_straighten_events_pre"]) == (0, 1)


def test_the_tip_pitch_at_rest_is_the_pre_window_median(base):
    assert row(base.ft, 2, "pretest", 1)["tip_pitch_rest_deg"] == pytest.approx(TIP_PITCH)


def test_the_tip_pitch_at_rest_uses_the_pre_stimulus_frames_only(tmp_path):
    def stepped(r):                      # 200 pre frames at 9.5 degrees, 100 at 33.7; the whole trial would be split 50/50
        r["tip_dy"] = np.where(np.arange(N) < 200, 10.0, 40.0)
    ft, _, _ = tables(variant(tmp_path, edits={(2, "pretest", 1): stepped}))
    assert row(ft, 2, "pretest", 1)["tip_pitch_rest_deg"] == pytest.approx(TIP_PITCH)


def test_the_tip_speed_is_smoothed_before_it_is_differenced(tmp_path):
    def jitter(r):
        r["tip_dx"][100] += 30.0         # a single frame, 30 px off: detection jitter, not movement
    tree = variant(tmp_path, edits={(2, "pretest", 1): jitter})
    smooth = row(tables(tree)[0], 2, "pretest", 1)
    raw = row(tables(tree, mcfg=M.parse_mounting_block({"speed_smooth_frames": 1}, default_batch_date_from=CUTOFF))[0],
              2, "pretest", 1)
    assert smooth["tip_moving_frac_pre"] == 0 and smooth["tip_speed_p95_pre_eu_s"] == 0
    assert raw["tip_moving_frac_pre"] == pytest.approx(2 / 299)          # the jump and the jump back, in 299 speeds


def test_tip_speed_is_in_eye_units_per_second_by_window(base):
    t2 = row(base.ft, 1, "testing", 1)
    assert t2["tip_speed_med_stim_eu_s"] == pytest.approx(1.0)               # 4.8 px/frame / 48 px * 10 fps
    assert t2["tip_speed_p95_stim_eu_s"] == pytest.approx(1.0)
    assert t2["tip_moving_frac_stim"] > 0.9
    assert (t2["tip_speed_med_pre_eu_s"], t2["tip_moving_frac_pre"]) == (0.0, 0.0)
    still = row(base.ft, 2, "pretest", 1)
    assert (still["tip_speed_med_pre_eu_s"], still["tip_speed_p95_pre_eu_s"], still["tip_moving_frac_stim"]) == (0, 0, 0)


def test_a_trial_with_the_tip_tracked_in_30_percent_of_frames_is_nan_but_keeps_its_per_columns(base):
    t = row(base.ft, 2, "testing", 1)
    assert t["abd_tracked_frac_pre"] == pytest.approx(0.3) and t["abd_tracked_frac_stim"] == pytest.approx(0.3)
    assert not t["abd_quality_ok"] and not t["abd_quality_ok_stim"]
    for col in ("abd_rest_bend_deg", "abd_bend_sd_pre_deg", "abd_bend_sd_stim_deg", "abd_curl_events_pre",
                "abd_curl_events_stim", "abd_straighten_events_pre", "abd_straighten_events_stim",
                "abd_curl_delta_deg", "tip_pitch_rest_deg", "tip_speed_med_pre_eu_s", "tip_speed_p95_pre_eu_s",
                "tip_moving_frac_pre", "tip_speed_med_stim_eu_s", "tip_moving_frac_stim", "per_abdomen_coupling_rho"):
        assert pd.isna(t[col]), col
    assert t["held_out"] == 1 and t["pullin_events_stim"] == 1 and t["rest_eu"] == pytest.approx(1.25)
    assert t["trial_body_pitch_deg"] == pytest.approx(PITCH_11)         # the abdomen centroid is not the tip


def test_the_stimulus_window_is_judged_on_its_own_tracking(tmp_path):
    def poor_stim(r):
        r["tip_ok"] = ~((np.arange(N) >= ONSET) & (np.arange(N) % 10 >= 3))       # 30% of the stimulus window
    ft, _, _ = tables(variant(tmp_path, edits={(2, "pretest", 1): poor_stim}))
    t = row(ft, 2, "pretest", 1)
    assert t["abd_quality_ok"] and not t["abd_quality_ok_stim"]
    assert t["abd_rest_bend_deg"] == pytest.approx(5.0) and t["abd_curl_events_pre"] == 0
    for col in ("abd_curl_events_stim", "abd_curl_delta_deg", "abd_bend_sd_stim_deg", "tip_speed_med_stim_eu_s"):
        assert pd.isna(t[col]), col


def test_a_poorly_tracked_pre_window_blanks_the_stimulus_window_too(tmp_path):
    def poor_pre(r):                     # 30% of the frames before the onset, every frame after it
        r["tip_ok"] = (np.arange(N) >= ONSET) | ((np.arange(N) % 10) < 3)
    ft, _, _ = tables(variant(tmp_path, edits={(2, "pretest", 1): poor_pre}))
    t = row(ft, 2, "pretest", 1)
    assert t["abd_tracked_frac_stim"] == 1.0 and not t["abd_quality_ok"] and not t["abd_quality_ok_stim"]
    for col in ("abd_curl_events_stim", "abd_curl_delta_deg", "abd_bend_sd_stim_deg", "tip_speed_med_stim_eu_s"):
        assert pd.isna(t[col]), col        # no resting bend to measure them against


def test_csv_geometry_tables_read_the_same_as_parquet_ones(tmp_path):
    tree = variant(tmp_path)
    first, _, _ = tables(tree)
    for path in sorted(tree.out.rglob("*_geometry.parquet")):
        pd.read_parquet(path).to_csv(path.with_suffix(".csv"), index=False)
        path.unlink()
    second, _, _ = tables(tree)
    pd.testing.assert_frame_equal(first, second)


def test_a_window_with_no_tracked_bend_has_no_resting_bend_even_when_the_gate_is_open(tmp_path):
    def blind(r):
        r["tip_ok"] = np.zeros(N, bool)
    mc = M.parse_mounting_block({"abd_min_tracked_frac": 0.0}, default_batch_date_from=CUTOFF)
    t = row(tables(variant(tmp_path, edits={(2, "pretest", 1): blind}), mcfg=mc)[0], 2, "pretest", 1)
    assert t["abd_tracked_frac_pre"] == 0 and not t["abd_quality_ok"] and math.isnan(t["abd_rest_bend_deg"])
    assert math.isnan(t["abd_curl_events_pre"])               # not 0: with no resting bend nothing can be a curl


def test_the_tracked_fraction_gate_is_a_setting(tmp_path):
    def sparse(r):
        r["tip_ok"] = (np.arange(N) % 10) < 3
    edits = {(2, "pretest", 1): sparse}
    ft, _, _ = tables(variant(tmp_path / "a", edits=edits), mcfg=M.parse_mounting_block(
        {"abd_min_tracked_frac": 0.3}, default_batch_date_from=CUTOFF))
    assert row(ft, 2, "pretest", 1)["abd_quality_ok"]                   # 0.3 >= 0.3
    assert not row(tables(variant(tmp_path / "b", edits=edits))[0], 2, "pretest", 1)["abd_quality_ok"]


def test_the_coupling_is_the_within_trial_spearman_of_proboscis_distance_and_bend(tmp_path):
    def together(r):
        r["dist"] = 24.0 + 0.1 * np.arange(N)
        r["bend"] = 5.0 + 0.05 * np.arange(N)

    def opposed(r):
        together(r)
        r["bend"] = r["bend"][::-1].copy()
    ft, _, _ = tables(variant(tmp_path, edits={(2, "pretest", 1): together, (2, "testing", 2): opposed}))
    assert row(ft, 2, "pretest", 1)["per_abdomen_coupling_rho"] == pytest.approx(1.0)
    assert row(ft, 2, "testing", 2)["per_abdomen_coupling_rho"] == pytest.approx(-1.0)


def test_the_coupling_needs_200_frames_where_both_are_finite(tmp_path):
    def short(r):
        r["dist"] = 24.0 + 0.1 * np.arange(N)
        r["bend"] = 5.0 + 0.05 * np.arange(N)
        r["dist"][150:] = np.nan                       # 150 frames left
    ft, _, _ = tables(variant(tmp_path, edits={(2, "pretest", 1): short}))
    t = row(ft, 2, "pretest", 1)
    assert math.isnan(t["per_abdomen_coupling_rho"]) and t["abd_quality_ok"]


# ---------------------------------------------------------------------------
# The fly table: pooling
# ---------------------------------------------------------------------------

def test_the_fly_table_has_one_row_per_consistent_fly_in_the_documented_columns(base):
    assert list(base.fly.columns) == MP.FLY_COLUMNS
    assert sorted(base.fly["fly_number"]) == [1, 2] and base.fly["n_trials"].eq(3).all()
    assert list(base.fly["fly_id"]) == [f"{DATASET}/{NEW}/fly1", f"{DATASET}/{NEW}/fly2"]


def test_the_spontaneous_per_proportion_pools_trials(base):
    f1, f2 = (base.fly[base.fly["fly_number"] == n].iloc[0] for n in FLIES)
    assert (f1["n_spont_known"], f1["n_spont_per"]) == (3, 1)
    assert f1["spont_per_frac"] == pytest.approx(1 / 3)                # 1 spontaneous of 3 trials
    assert f2["spont_per_frac"] == 0
    assert f1["spont_per_events"] == 1


def test_the_spontaneous_per_rate_is_events_over_pooled_window_minutes(base):
    f1 = base.fly[base.fly["fly_number"] == 1].iloc[0]
    assert f1["spont_per_rate_per_min"] == pytest.approx(1 / (3 * 0.5))


def test_the_curl_rate_pools_events_over_pooled_minutes(base):
    f1 = base.fly[base.fly["fly_number"] == 1].iloc[0]
    assert f1["abd_curl_rate_pre_per_min"] == pytest.approx(2 / 1.5)          # 2 events over 3 x 30 s = 1.333 / min
    assert f1["abd_curl_rate_stim_per_min"] == pytest.approx(1 / (3 * 40 / 10 / 60))
    assert f1["abd_straighten_rate_pre_per_min"] == 0


def test_held_out_pull_ins_and_states_pool_trials(base):
    f1, f2 = (base.fly[base.fly["fly_number"] == n].iloc[0] for n in FLIES)
    assert (f2["n_held_known"], f2["n_held_out"]) == (3, 1) and f2["held_out_frac"] == pytest.approx(1 / 3)
    assert f1["held_out_frac"] == 0 and pd.isna(f1["held_subtype_dominant"])
    assert f2["held_subtype_dominant"] == "held_back"
    assert (f1["proboscis_state_dominant"], f2["proboscis_state_dominant"]) == ("retracted", "retracted")
    assert f2["pullin_rate_stim_per_min"] == pytest.approx(1 / (3 * 40 / 10 / 60)) and f2["pullin_rate_pre_per_min"] == 0
    assert f1["pullin_rate_stim_per_min"] == 0 and f1["pullin_rate_pre_per_min"] == pytest.approx(1 / 1.5)


def test_the_reference_audit_columns_are_the_fly_minimum_and_median_rest(base):
    f2 = base.fly[base.fly["fly_number"] == 2].iloc[0]
    assert f2["rest_eu_min"] == pytest.approx(0.5) and f2["rest_eu_median"] == pytest.approx(0.5)


def test_the_fly_table_leaves_out_the_poorly_tracked_trial_from_abdomen_pools(base):
    f2 = base.fly[base.fly["fly_number"] == 2].iloc[0]
    assert f2["abd_rest_bend_deg"] == pytest.approx(5.0)
    assert f2["abd_curl_rate_pre_per_min"] == 0 and f2["abd_bend_sd_pre_deg"] == pytest.approx(0.0)
    assert f2["tip_pitch_rest_deg"] == pytest.approx(TIP_PITCH)


def test_the_median_coupling_and_its_trial_count_are_reported(tmp_path):
    def together(r):
        r["dist"] = 24.0 + 0.1 * np.arange(N)
        r["bend"] = 5.0 + 0.05 * np.arange(N)

    def opposed(r):
        together(r)
        r["bend"] = r["bend"][::-1].copy()
    _, _, fly = tables(variant(tmp_path, edits={(1, "pretest", 1): together, (1, "testing", 1): together,
                                                (1, "testing", 2): opposed}))
    f1, f2 = (fly[fly["fly_number"] == n].iloc[0] for n in FLIES)
    assert f1["n_coupling_trials"] == 3 and f1["per_abdomen_coupling_rho"] == pytest.approx(1.0)   # a mean would say 0.33
    assert f2["n_coupling_trials"] == 0 and math.isnan(f2["per_abdomen_coupling_rho"])


def test_the_coupling_uses_every_frame_of_the_trial(tmp_path):
    from scipy import stats
    f = np.arange(N)
    dist = 24.0 + 0.1 * f
    bend = np.where(f < ONSET, 5.0 + 0.05 * f, 20.0 - 0.2 * (f - ONSET))      # in step before the onset, opposed after it

    def turning(r):
        r["dist"], r["bend"] = dist, bend
    ft, _, _ = tables(variant(tmp_path, edits={(2, "pretest", 1): turning}))
    want = stats.spearmanr(dist, bend)[0]
    assert row(ft, 2, "pretest", 1)["per_abdomen_coupling_rho"] == pytest.approx(want)
    assert want < 0.5                                       # the frames before the onset alone would give 1.0


def test_fly_level_resting_bend_is_a_median_over_pooled_frames_not_over_trial_medians(tmp_path):
    def trial(level, tracked):
        def edit(r):
            r["bend"] = np.where(np.arange(N) < ONSET, level, 5.0)
            r["tip_ok"] = tracked
        return edit
    f = np.arange(N)
    half = (f % 2 == 0) | (f >= ONSET)                  # exactly 50% of the pre window: the gate is >=
    tree = variant(tmp_path, edits={(1, "pretest", 1): trial(4.0, np.ones(N, bool)), (1, "testing", 1): trial(6.0, half),
                                    (1, "testing", 2): trial(10.0, half)})
    ft, _, fly = tables(tree)
    assert [row(ft, 1, k, i)["abd_rest_bend_deg"] for k, i in TRIALS] == pytest.approx([4.0, 6.0, 10.0])
    assert row(ft, 1, "testing", 1)["abd_tracked_frac_pre"] == 0.5 and row(ft, 1, "testing", 1)["abd_quality_ok"]
    f1 = fly[fly["fly_number"] == 1].iloc[0]
    assert np.median([4.0, 6.0, 10.0]) == 6.0
    assert f1["abd_rest_bend_deg"] == pytest.approx(5.0)         # 300 frames at 4, 150 at 6, 150 at 10 -> (4 + 6) / 2


def test_fly_level_tip_speed_and_moving_fraction_pool_frames(base):
    f1 = base.fly[base.fly["fly_number"] == 1].iloc[0]
    assert f1["tip_speed_med_stim_eu_s"] == pytest.approx(0.0)                  # 80 still frames vs 39 moving ones
    assert f1["tip_speed_p95_stim_eu_s"] == pytest.approx(1.0)
    assert 0.2 < f1["tip_moving_frac_stim"] < 0.4                               # 39 of 120 pooled stimulus frames
    assert f1["tip_moving_frac_pre"] == 0 and f1["tip_speed_med_pre_eu_s"] == 0


def test_speed_statistics_are_the_median_the_95th_percentile_and_the_moving_fraction():
    med, p95, moving = MP._speed_stats(np.r_[np.arange(101.0), [np.nan]], 50.0)
    assert (med, p95) == (50.0, 95.0) and moving == pytest.approx(50 / 101)
    assert all(math.isnan(v) for v in MP._speed_stats([np.nan], 0.5))


# -- hand-built trial tables: the pooling arithmetic on its own ---------------------------------------------

def blank(*rows):
    base_row = {c: np.nan for c in MP.FLY_TRIAL_COLUMNS}
    base_row.update(dataset=DATASET, batch=NEW, fly_number=1, fly_id=f"{DATASET}/{NEW}/fly1", rig=HOST,
                    identity_consistent=True)
    return pd.DataFrame([{**base_row, **r} for r in rows], columns=MP.FLY_TRIAL_COLUMNS)


def test_a_rate_is_pooled_events_over_pooled_minutes_not_a_mean_of_trial_rates():
    df = blank({"abd_curl_events_pre": 1, "pre_min": 0.5}, {"abd_curl_events_pre": 1, "pre_min": 0.25})
    fly = MP.build_fly_table(df, MCFG).iloc[0]
    assert fly["abd_curl_rate_pre_per_min"] == pytest.approx(2 / 0.75)          # not (2 + 4) / 2


def test_the_minutes_of_an_unusable_window_do_not_enter_the_rate():
    df = blank({"abd_curl_events_pre": 1, "pre_min": 0.5}, {"abd_curl_events_pre": 1, "pre_min": 0.25},
               {"abd_curl_events_pre": np.nan, "pre_min": 0.5})
    assert MP.build_fly_table(df, MCFG).iloc[0]["abd_curl_rate_pre_per_min"] == pytest.approx(2 / 0.75)


def test_a_rate_with_no_usable_window_is_nan_not_zero():
    df = blank({"abd_curl_events_pre": np.nan, "pre_min": 0.5})
    fly = MP.build_fly_table(df, MCFG).iloc[0]
    assert math.isnan(fly["abd_curl_rate_pre_per_min"]) and math.isnan(fly["spont_per_rate_per_min"])


def test_proportions_divide_by_the_trials_with_a_known_state():
    df = blank({"spont_per_any": 1.0}, {"spont_per_any": 0.0}, {"spont_per_any": np.nan}, {"spont_per_any": 0.0})
    fly = MP.build_fly_table(df, MCFG).iloc[0]
    assert (fly["n_spont_known"], fly["n_spont_per"]) == (3, 1) and fly["spont_per_frac"] == pytest.approx(1 / 3)


def test_the_dominant_state_is_the_most_common_one_with_a_fixed_tie_order():
    df = blank({"held_subtype": "held_back"}, {"held_subtype": "held_down"}, {"held_subtype": "retracted"},
               {"held_subtype": "held_down"}, {"held_subtype": "retracted"}, {"held_subtype": np.nan})
    fly = MP.build_fly_table(df, MCFG).iloc[0]
    assert fly["proboscis_state_dominant"] == "retracted"           # 2 retracted, 2 held_down: retracted comes first
    assert fly["held_subtype_dominant"] == "held_down"              # among the held trials: held_down 2, held_back 1


def test_flies_flagged_inconsistent_are_not_in_the_fly_table():
    ok = blank({"trial_body_pitch_deg": 5.0})
    bad = blank({"trial_body_pitch_deg": 9.0, "identity_consistent": False, "fly_number": 2,
                 "fly_id": f"{DATASET}/{NEW}/fly2"})
    fly = MP.build_fly_table(pd.concat([ok, bad], ignore_index=True), MCFG)
    assert list(fly["fly_number"]) == [1]


def test_an_empty_trial_table_gives_a_fly_table_with_headers_only():
    fly = MP.build_fly_table(pd.DataFrame(columns=MP.FLY_TRIAL_COLUMNS), MCFG)
    assert len(fly) == 0 and list(fly.columns) == MP.FLY_COLUMNS


# ---------------------------------------------------------------------------
# The command line, and leaving production alone
# ---------------------------------------------------------------------------

def write_configs(tree, preds, *, flagged="", cutoff=CUTOFF, **mounting):
    cfg = tree.root / "cfg"
    cfg.mkdir(exist_ok=True)
    base_cfg = {"protocol": "v2", "model_path": "/x/per.engine",
                "dataset_bases": {"data": str(tree.prod), "secured": str(tree.mirror)},
                "datasets": [DATASET], "cache_dir": str(tree.root / "cache"),
                "reaction_prediction": {"output_csv": str(preds)}}
    if flagged:
        base_cfg["flagged_flies_csv"] = str(flagged)
    (cfg / "config_new.yaml").write_text(yaml.safe_dump(base_cfg), encoding="utf-8")
    (tree.root / "abdomen_best.pt").write_bytes(b"x")
    overlay = {"extends": "config_new.yaml",
               "abdomen": {"model_path": str(tree.root / "abdomen_best.pt"), "out_base": str(tree.out),
                           "video_bases": [str(tree.mirror)], "batch_date_from": cutoff, "slot_seeds": "production",
                           "geometry": {"table_format": "parquet", "wide_out_dir": str(tree.root / "wide"),
                                        "figures_out_dir": str(tree.root / "figs")}},
               "mounting": {"csv_out_dir": str(tree.root / "csvs"), "out_dir": str(tree.root / "mfigs"), **mounting}}
    path = cfg / "config_new_abdomen.yaml"
    path.write_text(yaml.safe_dump(overlay), encoding="utf-8")
    return path


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.delenv("FLAGGED_FLIES_CSV", raising=False)
    monkeypatch.delenv("REACTION_OUTPUT_CSV", raising=False)
    tree = Tree(tmp_path)
    tree.write_batch(NEW, "20261007")
    tree.write_batch(OLD, "20260930")
    preds = write_predictions(tree.root / "model_predictions.csv")
    return SimpleNamespace(tree=tree, preds=preds, overlay=write_configs(tree, preds))


def run_main(cli, *extra):
    t = cli.tree.root
    return MP.main(["--config", str(cli.overlay), "--out-dir", str(t / "figs_out"), "--csv-out-dir", str(t / "csv_out"),
                    *extra])


def test_main_writes_both_tables_and_the_exclusions_and_creates_the_figure_dir(cli):
    assert run_main(cli, "--predictions-csv", str(cli.preds)) == 0
    out = cli.tree.root / "csv_out"
    ft = pd.read_csv(out / "fly_trial_posture.csv")
    fly = pd.read_csv(out / "fly_posture.csv")
    exc = json.loads((out / "exclusions.json").read_text(encoding="utf-8"))
    assert len(ft) == 6 and list(ft.columns) == MP.FLY_TRIAL_COLUMNS
    assert len(fly) == 2 and list(fly.columns) == MP.FLY_COLUMNS
    assert exc["n_fly_trials"] == 6 and exc["prediction_exact_duplicates"] == 1 and exc["identity_drift"] == []
    assert (cli.tree.root / "figs_out").is_dir()
    held = ft[(ft["fly_number"] == 2) & (ft["trial_type"] == "testing") & (ft["trial_index"] == 1)].iloc[0]
    assert held["held_out"] == 1 and held["held_subtype"] == "held_back" and held["score"] == 1


def test_main_defaults_to_the_directories_and_predictions_of_the_config(cli):
    assert MP.main(["--config", str(cli.overlay)]) == 0
    ft = pd.read_csv(cli.tree.root / "csvs" / "fly_trial_posture.csv")
    assert ft["score"].notna().sum() == 5                      # the base config's reaction_prediction.output_csv
    assert (cli.tree.root / "mfigs").is_dir()


def test_main_needs_somewhere_to_write(tmp_path, monkeypatch):
    monkeypatch.delenv("FLAGGED_FLIES_CSV", raising=False)
    tree = variant(tmp_path)
    overlay = write_configs(tree, write_predictions(tree.root / "p.csv"))
    cfg = yaml.safe_load(overlay.read_text(encoding="utf-8"))
    del cfg["mounting"]["csv_out_dir"]
    overlay.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(SystemExit) as stop:
        MP.main(["--config", str(overlay)])
    assert stop.value.code == 2


def test_main_takes_flagged_flies_from_the_base_config(tmp_path, monkeypatch):
    monkeypatch.delenv("FLAGGED_FLIES_CSV", raising=False)
    tree = variant(tmp_path)
    flagged = tree.root / "flagged.csv"
    flagged.write_text(f'dataset,fly,fly_number,"FLY-State(1, 0, -1)"\n{DATASET},{NEW},2,0\n', encoding="utf-8")
    overlay = write_configs(tree, write_predictions(tree.root / "p.csv"), flagged=flagged)
    assert MP.main(["--config", str(overlay)]) == 0
    assert set(pd.read_csv(tree.root / "csvs" / "fly_trial_posture.csv")["fly_number"]) == {1}


def test_main_writes_headers_only_and_exits_zero_when_nothing_is_in_scope(tmp_path, monkeypatch):
    monkeypatch.delenv("FLAGGED_FLIES_CSV", raising=False)
    tree = variant(tmp_path)
    overlay = write_configs(tree, write_predictions(tree.root / "p.csv"), cutoff=date(2030, 1, 1))
    assert MP.main(["--config", str(overlay)]) == 0
    out = tree.root / "csvs"
    ft, fly = pd.read_csv(out / "fly_trial_posture.csv"), pd.read_csv(out / "fly_posture.csv")
    assert len(ft) == 0 and list(ft.columns) == MP.FLY_TRIAL_COLUMNS
    assert len(fly) == 0 and list(fly.columns) == MP.FLY_COLUMNS
    assert json.loads((out / "exclusions.json").read_text(encoding="utf-8"))["n_fly_trials"] == 0


def test_main_refuses_a_legacy_config_without_an_abdomen_overlay(tmp_path, monkeypatch):
    monkeypatch.delenv("FLAGGED_FLIES_CSV", raising=False)
    tree = variant(tmp_path)
    (tree.root / "model.pt").write_bytes(b"x")
    legacy = tree.root / "config_abdomen.yaml"
    legacy.write_text(yaml.safe_dump({
        "protocol": "v2", "model_path": str(tree.root / "model.pt"),
        "dataset_bases": {"data": str(tree.prod)}, "datasets": [DATASET], "cache_dir": str(tree.root / "cache"),
        "geometry": {"wide_out_dir": str(tree.root / "wide"), "figures_out_dir": str(tree.root / "figs")}}),
        encoding="utf-8")
    with pytest.raises(SystemExit) as stop:
        MP.main(["--config", str(legacy), "--csv-out-dir", str(tree.root / "o")])
    assert stop.value.code == 2
    assert not (tree.root / "o").exists()


def snapshot(*roots):
    """(is_dir, size, mtime_ns) of every path under the roots, the roots themselves included."""
    out = {}
    for root in roots:
        for p in [root, *sorted(root.rglob("*"))]:
            st = p.stat()
            out[str(p)] = (p.is_dir(), st.st_size, st.st_mtime_ns)
    return out


def test_main_changes_nothing_under_production_the_mirror_or_the_geometry_tree(cli):
    t = cli.tree
    watched = (t.prod, t.mirror, t.out)
    before = snapshot(*watched)
    assert run_main(cli, "--predictions-csv", str(cli.preds)) == 0
    assert snapshot(*watched) == before
    assert list(t.root.rglob("_trial_meta.json")) == []
    assert (t.root / "csv_out" / "fly_trial_posture.csv").is_file()          # the outputs went elsewhere
