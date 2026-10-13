"""The integrated abdomen config: an ``abdomen:`` block in an extends-overlay.

Pins the rules that keep it safe beside the PER pipeline: the abdomen detector settings are
self-contained (no PER ``yolo:`` block, no PER ``model_path``, no env override leaks in), the
abdomen output tree can never be -- or sit inside -- the read-only production tree, every malformed
key is refused by name, and ``--annotated-video`` reaches the step runners and every fan-out track
worker. Legacy configs (no ``abdomen:`` block) stay covered by test_geometry_config.py.
"""
import dataclasses
import datetime as dt
import math
import shutil
from pathlib import Path

import pytest
import yaml

from fbpipe import geometry_pipeline as GP
from fbpipe.config import Settings, load_settings
from fbpipe.geometry import config as GC
from fbpipe.geometry import steps as S
from fbpipe.geometry.config import GeometrySettings, load_geometry_config

DATASETS = ("IAA-X", "Hex-Y")
DROP = object()                      # an override value that removes the key
YOLO_ENV = ("MODEL_PATH", "INFERENCE_BATCH_SIZE", "MAX_FLIES", "CONF_THRES", "ENGINE_SUPPORTS_BATCH",
            "YOLO_NUM_WORKERS", "ZERO_IOU_EPSILON")


def _write(path: Path, data) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _override(block: dict, overrides: dict) -> dict:
    for key, val in overrides.items():
        if val is DROP:
            block.pop(key, None)
        else:
            block[key] = val
    return block


class Rig:
    """A tmp production tree plus an extends-overlay whose ``abdomen:`` block is valid."""

    def __init__(self, tmp_path: Path):
        self.root = tmp_path
        self.flys_new = tmp_path / "flys_New"
        for ds in DATASETS:
            (self.flys_new / ds / "october_07_batch_2").mkdir(parents=True)
        self.secured = tmp_path / "secured"
        self.secured.mkdir()
        self.model = tmp_path / "abdomen_best.pt"
        self.model.write_bytes(b"x")
        self.out_base = tmp_path / "flys_Abdomen"
        self.wide = tmp_path / "wide"
        self.figs = tmp_path / "figs"
        cfg = tmp_path / "cfg"
        cfg.mkdir()
        self.base = _write(cfg / "config_new.yaml", {
            "protocol": "v2",
            "model_path": "/x/per.engine",            # the PER model: it must never reach the abdomen
            "dataset_bases": {"data": str(self.flys_new), "secured": str(self.secured)},
            "datasets": list(DATASETS),
            "cache_dir": str(tmp_path / "cache"),
            "yolo": {"conf_thres": 0.4, "inference_batch_size": 16, "max_flies": 4, "num_workers": 1},
        })
        self.overlay = cfg / "config_new_abdomen.yaml"
        self.write()

    def geometry(self, **overrides) -> dict:
        """``abdomen.geometry`` as the template writes it (tables only), minus/plus the given keys."""
        return _override({"table_format": "parquet", "annotated_video": False,
                          "wide_out_dir": str(self.wide), "figures_out_dir": str(self.figs)}, overrides)

    def block(self, **overrides) -> dict:
        return _override({
            "enabled": True,
            "model_path": str(self.model),
            "out_base": str(self.out_base),
            "video_bases": [str(self.secured)],
            "batch_date_from": dt.date(2026, 10, 6),
            "datasets": [],
            "slot_seeds": "production",
            "max_hours": 10,
            "yolo": {"conf_thres": 0.25, "inference_batch_size": 8, "engine_supports_batch": True,
                     "max_flies": 5, "num_workers": 3},
            "geometry": self.geometry(),
        }, overrides)

    def legacy(self, **geometry) -> Path:
        """A pre-overlay ``config_abdomen.yaml`` over the same tmp tree; ``geometry`` adds keys to its block."""
        return _write(self.root / "config_abdomen.yaml", {
            "protocol": "v2", "model_path": str(self.model),
            "dataset_bases": {"data": str(self.flys_new), "secured": str(self.secured)},
            "datasets": list(DATASETS), "cache_dir": str(self.root / "cache"),
            "yolo": {"conf_thres": 0.3, "max_flies": 3},
            "geometry": {"wide_out_dir": str(self.wide), "figures_out_dir": str(self.figs), **geometry}})

    def write(self, **overrides) -> Path:
        return _write(self.overlay, {"extends": "config_new.yaml", "abdomen": self.block(**overrides)})


# ---- the block parser, without the filesystem -----------------------------------------------------

PER_BASE = Settings(model_path="/x/per.engine", main_directories=["/p/flys_New/IAA-X/"], datasets=DATASETS,
                    conf_thres=0.4, inference_batch_size=16, max_flies=4, zero_iou_epsilon=1e-6)


def _parse(**overrides):
    block = _override({"model_path": "/m/abdomen.pt", "out_base": "/o/flys_Abdomen"}, overrides)
    return GC.parse_abdomen_block(block, PER_BASE)


def test_the_abdomen_yolo_defaults_are_the_pinned_values():
    assert GC.ABDOMEN_YOLO_DEFAULTS == {"conf_thres": 0.25, "inference_batch_size": 8, "engine_supports_batch": True,
                                        "max_flies": 5, "num_workers": 3, "zero_iou_epsilon": 1e-8}


def test_the_integrated_fields_default_to_legacy_behaviour():
    g = GeometrySettings()
    assert (g.out_base, g.video_bases, g.batch_date_from, g.datasets) == ("", (), None, ())
    assert (g.slot_seeds, g.seed_radius_frac) == ("warmup", 0.5)
    assert g.annotated_video is True and GC.parse_geometry_block(None).annotated_video is True   # legacy: videos on


def test_a_missing_yolo_block_takes_the_abdomen_defaults_not_the_per_values():
    s, _ = _parse()
    assert s.model_path == "/m/abdomen.pt"
    assert (s.conf_thres, s.inference_batch_size, s.engine_supports_batch) == (0.25, 8, True)
    assert (s.max_flies, s.yolo_num_workers, s.zero_iou_epsilon) == (5, 3, 1e-8)


def test_a_partial_yolo_block_fills_the_rest_from_the_abdomen_defaults():
    s, _ = _parse(yolo={"conf_thres": 0.3, "num_workers": 2})
    assert (s.conf_thres, s.yolo_num_workers) == (0.3, 2)
    assert (s.inference_batch_size, s.max_flies) == (8, 5)           # not the PER block's 16 / 4


def test_only_the_detector_fields_differ_from_the_base_settings():
    s, _ = _parse()
    changed = {f.name for f in dataclasses.fields(Settings) if getattr(s, f.name) != getattr(PER_BASE, f.name)}
    assert changed == {"model_path", "conf_thres", "inference_batch_size", "engine_supports_batch", "max_flies",
                       "yolo_num_workers", "zero_iou_epsilon"}
    assert s.main_directories == PER_BASE.main_directories and s.datasets == PER_BASE.datasets


def test_parsing_leaves_the_base_settings_and_the_defaults_untouched():
    defaults = dict(GC.ABDOMEN_YOLO_DEFAULTS)
    s, _ = _parse(yolo={"conf_thres": 0.3, "max_flies": 2})
    assert s is not PER_BASE
    assert (PER_BASE.model_path, PER_BASE.conf_thres, PER_BASE.max_flies) == ("/x/per.engine", 0.4, 4)
    assert GC.ABDOMEN_YOLO_DEFAULTS == defaults


def test_the_abdomen_tree_fields_land_on_the_geometry_settings():
    _, g = _parse(video_bases=["/v1", "/v2"], datasets=["IAA-X"], slot_seeds="production", seed_radius_frac=0.4,
                  batch_date_from=dt.date(2026, 10, 6), geometry={"table_format": "parquet", "annotated_video": False})
    assert g.out_base == "/o/flys_Abdomen" and g.video_bases == ("/v1", "/v2")
    assert g.datasets == ("IAA-X",) and g.slot_seeds == "production" and g.seed_radius_frac == 0.4
    assert g.batch_date_from == dt.date(2026, 10, 6)
    assert g.table_format == "parquet" and g.annotated_video is False


def test_annotated_videos_are_off_in_integrated_mode_unless_the_geometry_block_sets_them():
    assert _parse()[1].annotated_video is False                                          # no geometry block at all
    assert _parse(geometry={"table_format": "parquet"})[1].annotated_video is False      # a block that omits the key
    assert _parse(geometry={"annotated_video": False})[1].annotated_video is False
    assert _parse(geometry={"annotated_video": True})[1].annotated_video is True         # an explicit true wins


def test_every_documented_abdomen_key_is_accepted():
    _parse(enabled=False, python="/envs/yolo-env/bin/python", max_hours=10, datasets=["IAA-X"], video_bases=["/v"],
           batch_date_from="2026-10-06", slot_seeds="warmup", seed_radius_frac=0.5,
           yolo=dict(GC.ABDOMEN_YOLO_DEFAULTS), geometry={"wide_out_dir": "/w", "figures_out_dir": "/f"})


@pytest.mark.parametrize("block", [None, [], "abdomen", 3], ids=repr)
def test_a_non_mapping_abdomen_block_is_refused(block):
    with pytest.raises(ValueError, match="abdomen block must be a mapping"):
        GC.parse_abdomen_block(block, PER_BASE)


def test_an_unknown_abdomen_key_is_refused_by_name():
    with pytest.raises(ValueError, match="seed_radiusfrac"):
        _parse(seed_radiusfrac=0.5)


def test_an_unknown_abdomen_yolo_key_is_refused_by_name():
    with pytest.raises(ValueError, match="iou_match_thres"):
        _parse(yolo={"conf_thres": 0.25, "iou_match_thres": 0.3})      # a PER yolo key, not an abdomen one


@pytest.mark.parametrize("bad", [
    {"conf_thres": 1.5}, {"conf_thres": -0.1}, {"conf_thres": "high"}, {"conf_thres": math.nan},
    {"inference_batch_size": 0}, {"max_flies": 0}, {"num_workers": 0},
    {"engine_supports_batch": "maybe"}, {"engine_supports_batch": 1}, {"zero_iou_epsilon": -1e-8},
], ids=repr)
def test_invalid_abdomen_yolo_values_raise(bad):
    with pytest.raises(ValueError, match="abdomen.yolo"):
        _parse(yolo=bad)


def test_a_non_mapping_yolo_or_geometry_block_is_refused_but_a_null_yolo_means_defaults():
    with pytest.raises(ValueError, match="abdomen.yolo"):
        _parse(yolo=[1, 2])
    with pytest.raises(ValueError, match="abdomen.geometry"):
        _parse(geometry="parquet")
    assert _parse(yolo=None)[0].max_flies == 5


@pytest.mark.parametrize("val", [DROP, "", "  ", None], ids=["absent", "empty", "blank", "null"])
def test_an_absent_model_path_or_out_base_stays_empty_and_never_falls_back_to_the_base(val):
    s, g = _parse(model_path=val, out_base=val)
    assert s.model_path == "" and s.model_path != PER_BASE.model_path       # the PER model must not leak in
    assert g.out_base == ""


@pytest.mark.parametrize("key", ["model_path", "out_base"])
def test_a_path_that_is_not_a_string_is_refused(key):
    with pytest.raises(ValueError, match=key):
        _parse(**{key: 5})


@pytest.mark.parametrize("key,val", [
    ("video_bases", "/v"), ("video_bases", [""]), ("video_bases", [3]), ("video_bases", "  "),
    ("datasets", "IAA-X"), ("datasets", [None]), ("datasets", [""]),
], ids=repr)
def test_list_keys_must_hold_non_empty_strings(key, val):
    with pytest.raises(ValueError, match=key):
        _parse(**{key: val})


@pytest.mark.parametrize("raw,want", [
    (dt.date(2026, 10, 6), dt.date(2026, 10, 6)),
    ("2026-10-06", dt.date(2026, 10, 6)),
    (dt.datetime(2026, 10, 6, 3, 30), dt.date(2026, 10, 6)),          # a datetime is cut to its day
    (None, None),                                                      # no cut-off
], ids=["yaml-date", "iso-string", "datetime", "null"])
def test_batch_date_from_accepts_a_yaml_date_or_an_iso_string(raw, want):
    assert _parse(batch_date_from=raw)[1].batch_date_from == want
    assert _parse(batch_date_from=DROP)[1].batch_date_from is None


@pytest.mark.parametrize("raw", [
    "2026-13-40", "2026-02-30", "2026-10-6", "20261006", "10/06/2026", "2026-10-06T00:00:00", "yesterday", "",
    20261006, 2026.1006, True, ["2026-10-06"],
], ids=repr)
def test_batch_date_from_rejects_anything_else(raw):
    with pytest.raises(ValueError, match="batch_date_from"):
        _parse(batch_date_from=raw)


def test_abdomen_datasets_must_be_a_subset_of_the_base_datasets():
    assert _parse(datasets=["Hex-Y"])[1].datasets == ("Hex-Y",)
    assert _parse(datasets=[])[1].datasets == ()                       # empty = every base dataset
    with pytest.raises(ValueError, match="Not-A-Dataset"):
        _parse(datasets=["IAA-X", "Not-A-Dataset"])


def test_datasets_are_refused_when_the_base_config_lists_none():
    bare = dataclasses.replace(PER_BASE, datasets=())
    with pytest.raises(ValueError, match="IAA-X"):
        GC.parse_abdomen_block({"model_path": "/m", "out_base": "/o", "datasets": ["IAA-X"]}, bare)


@pytest.mark.parametrize("mode", ["warmup", "production"])
def test_slot_seeds_accepts_the_two_modes(mode):
    assert _parse(slot_seeds=mode)[1].slot_seeds == mode
    assert _parse(slot_seeds=DROP)[1].slot_seeds == "warmup"


@pytest.mark.parametrize("bad", ["fly", "", None, 1, True], ids=repr)
def test_slot_seeds_rejects_anything_else(bad):
    with pytest.raises(ValueError, match="slot_seeds"):
        _parse(slot_seeds=bad)


def test_seed_radius_frac_takes_a_positive_number():
    assert _parse(seed_radius_frac=0.35)[1].seed_radius_frac == 0.35
    assert _parse(seed_radius_frac=1)[1].seed_radius_frac == 1.0
    assert _parse(seed_radius_frac=DROP)[1].seed_radius_frac == 0.5


@pytest.mark.parametrize("bad", [0, 0.0, -0.5, math.nan, "wide", None], ids=repr)
def test_seed_radius_frac_must_be_positive(bad):
    with pytest.raises(ValueError, match="seed_radius_frac"):
        _parse(seed_radius_frac=bad)


# ---- the output tree can never be production ------------------------------------------------------

def test_assert_out_tree_safe_refuses_an_equal_or_nested_path(tmp_path):
    prod = tmp_path / "flys_New"
    prod.mkdir()
    for bad in (prod, prod / "IAA-X", prod / "IAA-X" / "b" / "geometry", tmp_path / "x" / ".." / "flys_New" / "z"):
        with pytest.raises(ValueError, match="flys_New"):
            GC.assert_out_tree_safe(bad, [prod])


def test_assert_out_tree_safe_allows_siblings_and_prefix_lookalikes(tmp_path):
    prod = tmp_path / "flys_New"
    prod.mkdir()
    for ok in (tmp_path / "flys_Abdomen", tmp_path / "flys_New2" / "x", tmp_path / "other", tmp_path):
        GC.assert_out_tree_safe(ok, [prod])


def test_assert_out_tree_safe_checks_every_forbidden_root(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    GC.assert_out_tree_safe(tmp_path / "c", [a, b])
    with pytest.raises(ValueError):
        GC.assert_out_tree_safe(b / "deep" / "tree", [a, b])


def test_assert_out_tree_safe_sees_through_a_symlink_into_a_forbidden_root(tmp_path):
    secured = tmp_path / "securedstorage"
    secured.mkdir()
    link = tmp_path / "looks_innocent"
    link.symlink_to(secured, target_is_directory=True)
    with pytest.raises(ValueError):
        GC.assert_out_tree_safe(link / "abdomen", [secured])


UNSAFE = {
    "the production base itself": lambda r: r.flys_new,
    "a dataset folder of the production tree": lambda r: r.flys_new / "IAA-X",
    "a new folder inside a production batch": lambda r: r.flys_new / "IAA-X" / "october_07_batch_2" / "abd",
    "the secured base of the config": lambda r: r.secured / "abd",
    "the literal /securedstorage mount": lambda r: Path("/securedstorage/DATAsec/cole/Data-secured-New/abd"),
}


@pytest.mark.parametrize("pick", UNSAFE.values(), ids=UNSAFE.keys())
def test_an_out_base_inside_the_production_tree_is_refused(tmp_path, pick):
    rig = Rig(tmp_path)
    before = set(rig.flys_new.rglob("*"))
    rig.write(out_base=str(pick(rig)))
    with pytest.raises(ValueError, match="out_base"):
        load_geometry_config(rig.overlay)
    assert set(rig.flys_new.rglob("*")) == before, "refusing must not have touched the production tree"


# ---- load_geometry_config, integrated -------------------------------------------------------------

def test_the_overlay_yields_abdomen_settings_over_the_production_roots(tmp_path):
    rig = Rig(tmp_path)
    settings, g = load_geometry_config(rig.overlay)
    base = load_settings(rig.base)
    assert settings.main_directories == base.main_directories           # the PRODUCTION roots, untouched
    assert [Path(d).name for d in settings.main_directories] == list(DATASETS)
    assert settings.cache_dir == base.cache_dir and settings.datasets == base.datasets
    assert settings.model_path == str(rig.model)
    assert isinstance(g, GeometrySettings)
    assert g.out_base == str(rig.out_base) and g.video_bases == (str(rig.secured),)
    assert g.batch_date_from == dt.date(2026, 10, 6) and g.datasets == ()
    assert g.slot_seeds == "production" and g.seed_radius_frac == 0.5
    assert g.table_format == "parquet" and g.annotated_video is False
    assert g.wide_out_dir == str(rig.wide) and g.figures_out_dir == str(rig.figs)
    assert not rig.out_base.exists(), "loading a config creates nothing"


def test_the_abdomen_detector_ignores_the_per_yolo_block(tmp_path, monkeypatch):
    for var in YOLO_ENV:
        monkeypatch.delenv(var, raising=False)
    rig = Rig(tmp_path)
    per = load_settings(rig.overlay)                                    # what a PER stage reads from the base
    assert (per.model_path, per.conf_thres, per.inference_batch_size, per.max_flies) == ("/x/per.engine", 0.4, 16, 4)
    settings, _ = load_geometry_config(rig.overlay)
    assert settings.model_path == str(rig.model)
    assert (settings.conf_thres, settings.inference_batch_size, settings.max_flies) == (0.25, 8, 5)
    assert (settings.engine_supports_batch, settings.yolo_num_workers, settings.zero_iou_epsilon) == (True, 3, 1e-8)


def test_the_abdomen_detector_ignores_the_env_overrides(tmp_path, monkeypatch):
    rig = Rig(tmp_path)
    for var, val in zip(YOLO_ENV, ("/env/per.engine", "32", "2", "0.9", "false", "9", "0.5")):
        monkeypatch.setenv(var, val)
    per = load_settings(rig.overlay)
    assert (per.model_path, per.inference_batch_size, per.max_flies, per.conf_thres) == ("/env/per.engine", 32, 2, 0.9)
    assert (per.engine_supports_batch, per.yolo_num_workers, per.zero_iou_epsilon) == (False, 9, 0.5)   # the leak is real
    settings, _ = load_geometry_config(rig.overlay)
    assert settings.model_path == str(rig.model)
    assert (settings.conf_thres, settings.inference_batch_size, settings.max_flies) == (0.25, 8, 5)
    assert (settings.engine_supports_batch, settings.yolo_num_workers, settings.zero_iou_epsilon) == (True, 3, 1e-8)


def test_a_plain_config_carrying_an_inline_abdomen_block_is_integrated_too(tmp_path):
    rig = Rig(tmp_path)
    inline = yaml.safe_load(rig.base.read_text())
    inline["abdomen"] = rig.block()
    settings, g = load_geometry_config(_write(tmp_path / "inline.yaml", inline))
    assert settings.model_path == str(rig.model) and g.out_base == str(rig.out_base)


def test_a_legacy_config_leaves_the_integrated_fields_at_their_defaults(tmp_path):
    rig = Rig(tmp_path)
    settings, g = load_geometry_config(rig.legacy())
    assert settings.model_path == str(rig.model) and (settings.conf_thres, settings.max_flies) == (0.3, 3)
    assert (g.out_base, g.video_bases, g.batch_date_from, g.datasets, g.slot_seeds, g.seed_radius_frac) == \
        ("", (), None, (), "warmup", 0.5)


def test_a_legacy_config_keeps_annotated_videos_on_by_default(tmp_path):
    rig = Rig(tmp_path)
    assert load_geometry_config(rig.legacy())[1].annotated_video is True
    assert load_geometry_config(rig.legacy(annotated_video=False))[1].annotated_video is False


def test_an_integrated_config_that_omits_annotated_video_writes_no_annotated_videos(tmp_path):
    rig = Rig(tmp_path)
    rig.write(geometry=rig.geometry(annotated_video=DROP))
    assert "annotated_video" not in yaml.safe_load(rig.overlay.read_text())["abdomen"]["geometry"]
    assert load_geometry_config(rig.overlay)[1].annotated_video is False


def test_an_explicit_annotated_video_true_in_the_abdomen_geometry_block_still_wins(tmp_path):
    rig = Rig(tmp_path)
    rig.write(geometry=rig.geometry(annotated_video=True))
    assert load_geometry_config(rig.overlay)[1].annotated_video is True


def test_a_null_abdomen_block_in_the_overlay_is_refused(tmp_path):
    rig = Rig(tmp_path)
    _write(rig.overlay, {"extends": "config_new.yaml", "abdomen": None})
    with pytest.raises(ValueError, match="abdomen block must be a mapping"):
        load_geometry_config(rig.overlay)


def test_an_overlay_that_adds_no_abdomen_block_is_not_a_geometry_config(tmp_path):
    rig = Rig(tmp_path)
    mounting_only = _write(rig.overlay.with_name("mounting_only.yaml"),
                           {"extends": "config_new.yaml", "mounting": {"enabled": True}})
    with pytest.raises(ValueError, match="abdomen"):
        load_geometry_config(mounting_only)


def test_the_batch_cut_off_survives_the_yaml_round_trip(tmp_path):
    rig = Rig(tmp_path)
    for written in (dt.date(2026, 10, 6), "2026-10-06"):                # unquoted YAML date, quoted ISO string
        rig.write(batch_date_from=written)
        assert load_geometry_config(rig.overlay)[1].batch_date_from == dt.date(2026, 10, 6)
    rig.write(batch_date_from="2026-13-40")
    with pytest.raises(ValueError, match="batch_date_from"):
        load_geometry_config(rig.overlay)


def test_abdomen_datasets_are_checked_against_the_base_config_file(tmp_path):
    rig = Rig(tmp_path)
    rig.write(datasets=["IAA-X"])
    assert load_geometry_config(rig.overlay)[1].datasets == ("IAA-X",)
    rig.write(datasets=["IAA-X", "Ghost-Dataset"])
    with pytest.raises(ValueError, match="Ghost-Dataset"):
        load_geometry_config(rig.overlay)


@pytest.mark.parametrize("key", ["model_path", "out_base"])
@pytest.mark.parametrize("val", [DROP, "", "  ", None, 5], ids=["absent", "empty", "blank", "null", "number"])
def test_the_model_path_and_out_base_are_required_in_the_abdomen_block(tmp_path, key, val):
    rig = Rig(tmp_path)
    rig.write(**{key: val})                  # the base config's PER model_path is no substitute
    with pytest.raises(ValueError, match=key):
        load_geometry_config(rig.overlay)


def test_a_missing_abdomen_model_file_is_refused(tmp_path):
    rig = Rig(tmp_path)
    rig.write(model_path=str(tmp_path / "nope.pt"))
    with pytest.raises(FileNotFoundError, match=r"model_path is not a file: .*nope\.pt"):   # not the PER /x/per.engine
        load_geometry_config(rig.overlay)


def test_at_least_one_video_base_must_exist(tmp_path):
    rig = Rig(tmp_path)
    gone = [str(tmp_path / "no_such_a"), str(tmp_path / "no_such_b")]
    for bases in (gone, [], DROP):
        rig.write(video_bases=bases)
        with pytest.raises(FileNotFoundError, match="video_bases"):
            load_geometry_config(rig.overlay)
    rig.write(video_bases=[gone[0], str(rig.secured)])
    assert load_geometry_config(rig.overlay)[1].video_bases == (gone[0], str(rig.secured))   # one is enough


def test_both_output_dirs_are_still_required(tmp_path):
    rig = Rig(tmp_path)
    for geometry in ({"table_format": "parquet"}, {"wide_out_dir": str(rig.wide)}, {"figures_out_dir": str(rig.figs)}):
        rig.write(geometry=geometry)
        with pytest.raises(ValueError, match="wide_out_dir"):
            load_geometry_config(rig.overlay)


def test_a_production_tree_with_nothing_to_process_is_refused(tmp_path):
    rig = Rig(tmp_path)
    for ds in DATASETS:
        shutil.rmtree(rig.flys_new / ds)                                # flys_New exists but holds no dataset
    with pytest.raises(RuntimeError, match="main_directories"):
        load_geometry_config(rig.overlay)
    shutil.rmtree(rig.flys_new)
    with pytest.raises(FileNotFoundError, match="dataset_bases.data"):
        load_geometry_config(rig.overlay)


# ---- --annotated-video ----------------------------------------------------------------------------

def test_the_annotated_video_flag_is_off_unless_given():
    parser = GP.build_parser()
    assert parser.parse_args(["--config", "c.yaml"]).annotated_video is False
    assert parser.parse_args(["--config", "c.yaml", "--annotated-video", "track"]).annotated_video is True


def test_the_annotated_video_flag_overrides_the_overlay_for_the_steps(tmp_path, monkeypatch):
    rig = Rig(tmp_path)                                                 # the overlay says annotated_video: false
    seen = []
    monkeypatch.setattr(S, "run_normalize", lambda settings, gcfg, *, force=False: seen.append(gcfg))
    assert GP.main(["--config", str(rig.overlay), "normalize"]) == 0
    assert GP.main(["--config", str(rig.overlay), "--annotated-video", "normalize"]) == 0
    assert [g.annotated_video for g in seen] == [False, True]
    assert seen[1] == dataclasses.replace(seen[0], annotated_video=True), "nothing but annotated_video changes"


def test_the_annotated_video_flag_still_forces_videos_on_when_the_block_omits_the_key(tmp_path, monkeypatch):
    rig = Rig(tmp_path)
    rig.write(geometry=rig.geometry(annotated_video=DROP))
    seen = []
    monkeypatch.setattr(S, "run_normalize", lambda settings, gcfg, *, force=False: seen.append(gcfg))
    assert GP.main(["--config", str(rig.overlay), "normalize"]) == 0                         # the nightly
    assert GP.main(["--config", str(rig.overlay), "--annotated-video", "normalize"]) == 0    # the backfill
    assert [g.annotated_video for g in seen] == [False, True]


def test_worker_commands_forward_the_annotated_video_flag_to_every_worker(tmp_path):
    cfg = tmp_path / "o.yaml"
    on = GP.worker_commands(cfg, 3, options={"annotated_video": True})
    assert len(on) == 3
    for cmd, _env in on:
        assert "--annotated-video" in cmd and cmd.index("--annotated-video") < cmd.index("track")
    for options in ({"annotated_video": False}, {}, None):
        for cmd, _env in GP.worker_commands(cfg, 3, options=options):
            assert "--annotated-video" not in cmd


def test_a_worker_command_line_parses_back_to_the_options_it_was_built_from(tmp_path):
    ((cmd, _env),) = GP.worker_commands(tmp_path / "o.yaml", 1, options={"annotated_video": True, "force": True})
    args = GP.build_parser().parse_args(cmd[3:])                        # drop `python -m fbpipe.geometry_pipeline`
    assert args.annotated_video is True and args.force is True and args.no_fan_out is True
    assert args.steps == ["track"]


@pytest.fixture
def launched(monkeypatch):
    """Capture the worker command lines instead of starting processes."""
    cmds = []

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            cmds.append(list(cmd))
            self.pid = 4242

        def wait(self):
            return 0

    monkeypatch.setattr(GP.subprocess, "Popen", FakePopen)
    return cmds


@pytest.mark.parametrize("flags,want", [([], False), (["--annotated-video"], True)], ids=["nightly", "backfill"])
def test_main_gives_every_track_worker_the_annotated_video_flag_only_when_asked(tmp_path, launched, flags, want):
    rig = Rig(tmp_path)
    # one untracked trial video: with nothing pending `track` starts no workers (test_geometry_jobs.py)
    (rig.flys_new / DATASETS[0] / "october_07_batch_2" / "output_october_07_batch_2_pretest_1_Citral_20261007_100000.mp4"
     ).write_bytes(b"x")
    assert GP.main(["--config", str(rig.overlay), *flags, "track"]) == 0
    assert len(launched) == 3, "abdomen.yolo.num_workers, not the PER block's 1"
    assert all(("--annotated-video" in cmd) is want for cmd in launched)
    assert all(str(rig.overlay) in cmd for cmd in launched), "workers load the overlay, not the base"
