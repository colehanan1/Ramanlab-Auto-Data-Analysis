"""Data-frozen datasets skip YOLO + the per-trial steps when nothing can change.

A ``freeze.data`` dataset's wide rows are spliced from the freeze cache, so its
per-trial files are never read -- yet every night the forced per-trial steps
re-derived all of them (~6.5 min for 16 datasets). The dataset loop now skips
BOTH its YOLO launch and its per-trial subprocess, but only when ALL hold:

  (a) it is data-frozen this run (``--thaw`` / ``--thaw-all`` honoured);
  (b) its YOLO launch would certainly succeed without finding a video: no
      video file at all (the YOLO pre-check's superset rule), and CUDA and the
      model file are there -- without them today's launch fails the night;
  (c) every freeze-cache slice the combined step will consult for it exists
      and matches the current fingerprint -- i.e. build_wide_csv will splice
      it. A missing/stale slice means build_wide_csv rebuilds the dataset's
      rows tonight, so the dataset runs exactly as today and that rebuild (and
      the slice it re-caches) sees exactly today's inputs;
  (d) not ``--folder`` mode, which keeps today's force-everything behaviour.

A skipped dataset's pipeline state is left untouched. Its YOLO slot still
re-checks for videos just before the launch would have happened, so a video
landing after the start of the run is inferred exactly as today.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import subprocess  # noqa: E402

import torch  # noqa: E402

from fbpipe import freeze as _freeze  # noqa: E402
from fbpipe.config import load_settings  # noqa: E402
from scripts.pipeline import run_workflows as rw  # noqa: E402

from _loop_harness import (  # noqa: E402
    isolate_env,
    make_dataset,
    per_trial_runs,
    run_main,
    write_config,
    yolo_launches,
)

LIVE = "Hex-HexTest-Training-24-1"
FROZEN = "EB-Control-24-0.1"
BATCH = "august_01_batch_1"
STEM = "output_august_01_batch_1_testing_1_Hexanol_20260801_120000"


def _tables(batch=BATCH):
    return [f"{batch}/{STEM}.csv", f"{batch}/august_01_batch_1_testing_1/fly1_distances.parquet"]


def _load(cfg_path: Path):
    settings = load_settings(cfg_path)
    with cfg_path.open(encoding="utf-8") as fh:
        data = rw._expand_datasets(yaml.safe_load(fh))
    return settings, data


def _real_resolve_calls(combined_cfg, settings, cfg_path) -> list[tuple]:
    """``(wide_block, root names, fingerprint kwargs)`` of every
    ``_resolve_frozen_slices`` call a real ``_run_combined`` makes."""
    calls: list[tuple] = []

    def _record(settings_, roots, wide_block, *, thawed=(), thaw_all=False, **kw):
        calls.append((wide_block, sorted(Path(str(r)).name for r in roots), kw))
        return {}

    noop = lambda *a, **k: None  # noqa: E731
    # A private patch context: the caller's monkeypatch (env isolation) stays.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rw, "_resolve_frozen_slices", _record)
        for name in (
            "combine_distance_angle",
            "build_wide_csv",
            "_write_freeze_cache",
            "wide_to_matrix",
            "generate_envelope_plots",
            "generate_reaction_matrices",
            "_render_pair_visuals",
            "_run_secure_cleanup",
        ):
            mp.setattr(rw, name, noop)
        # The parallel combine runner, where one exists (parallel.enabled).
        mp.setattr(rw, "run_combine_plan", noop, raising=False)
        mp.setattr(rw, "mirror_directory", lambda *a, **k: (0, 0))
        rw._run_combined(
            combined_cfg,
            settings,
            defer_envelopes=True,
            defer_secure_cleanup=True,
            config_path=cfg_path,
        )
    return calls


def _seed_slices(settings, calls, dataset, *, skip_blocks=()):
    """Cache *dataset*'s slice exactly as _write_freeze_cache would."""
    rows = pd.DataFrame(
        {"dataset": [dataset], "fly": ["f1"], "trace_len": [2], "dir_val_0": [0.1], "dir_val_1": [0.2]}
    )
    for block, names, kw in calls:
        if dataset in names and block not in skip_blocks:
            _freeze.save_slice(
                settings.cache_dir, block, dataset, rows,
                rw._freeze_fingerprint(settings, dataset, **kw),
            )


@pytest.fixture
def scene(tmp_path, monkeypatch):
    """A live dataset plus a data-frozen one whose cache a real combine wrote."""
    isolate_env(monkeypatch)
    make_dataset(tmp_path, LIVE, _tables())
    make_dataset(tmp_path, FROZEN, _tables())
    cfg_path = write_config(tmp_path, [LIVE, FROZEN], frozen=[FROZEN])
    settings, data = _load(cfg_path)
    combined_cfg = data["analysis"]["combined"]
    calls = _real_resolve_calls(combined_cfg, settings, cfg_path)
    _seed_slices(settings, calls, FROZEN)
    return {
        "tmp": tmp_path,
        "cfg": cfg_path,
        "settings": settings,
        "combined": combined_cfg,
        "root": str(tmp_path / "data" / FROZEN),
        "live_root": str(tmp_path / "data" / LIVE),
    }


def _decide(scene, root=None, *, thawed=(), thaw_all=False, settings=None):
    settings = settings or scene["settings"]
    blocks = rw._freeze_cache_blocks(scene["combined"], settings)
    return rw._skip_frozen_dataset(
        settings, root or scene["root"], blocks, thawed=thawed, thaw_all=thaw_all
    )


# --- (c) the prediction matches _run_combined ---------------------------------

def _full_combined_cfg(tmp_path: Path, roots, *, pair_b_filter="testing") -> dict:
    """Every wide block type _run_combined knows, with differing parameters."""
    out = tmp_path / "out"
    return {
        "combine": {"roots": list(roots)},
        "wide": {
            "roots": list(roots),
            "output_csv": str(out / "wide.csv"),
            "measure_cols": ["envelope_of_rms"],
            "fps_fallback": 41.0,
            "use_per_trial_baseline": False,
            "trial_type_filter": ["testing", "training"],
            "threshold_std_mult": 4.0,
            "trial_type_exports": [
                {"trial_type": "training", "output_csv": str(out / "wide_training.csv")}
            ],
        },
        "combined_base": {
            "wide": {
                "roots": list(roots)[:1],
                "output_csv": str(out / "cb.csv"),
                "use_per_trial_baseline": True,
                "threshold_min_delta": 5.0,
            }
        },
        # No roots: inherits combined.wide's, and its default measure column.
        "distance_base": {"wide": {"output_csv": str(out / "db.csv"), "fps_fallback": 39.0}},
        "pair_groups": [
            {"name": "pair-a", "datasets": [Path(r).name for r in roots], "out_dir": str(out / "pa")},
            {
                "name": "pair-b",
                "datasets": [Path(roots[-1]).name],
                "out_dir": str(out / "pb"),
                "trial_type_filter": pair_b_filter,
            },
        ],
    }


def test_blocks_match_every_resolve_call_run_combined_makes(tmp_path, monkeypatch):
    isolate_env(monkeypatch)
    roots = [str(make_dataset(tmp_path, n, _tables())) for n in (LIVE, FROZEN)]
    cfg_path = write_config(tmp_path, [LIVE, FROZEN], frozen=[FROZEN])
    settings, _ = _load(cfg_path)
    combined_cfg = _full_combined_cfg(tmp_path, roots)

    real = _real_resolve_calls(combined_cfg, settings, cfg_path)
    predicted = [
        (block, sorted(names), kw)
        for block, names, kw in rw._freeze_cache_blocks(combined_cfg, settings)
    ]
    assert [b for b, _, _ in real] == [
        "wide", "combined_base", "distance_base", "pair_groups", "pair_groups"
    ]
    assert predicted == real


def test_blocks_match_with_flagged_dirs_riding_along(tmp_path, monkeypatch):
    isolate_env(monkeypatch)
    roots = [str(make_dataset(tmp_path, n, _tables())) for n in (LIVE, FROZEN)]
    flagged = tmp_path / "flagged_root"
    (flagged / "Odd-Flagged-Experiment").mkdir(parents=True)
    cfg_path = write_config(tmp_path, [LIVE, FROZEN], frozen=[FROZEN])
    settings, _ = _load(cfg_path)
    settings.flagged_root = str(flagged)
    combined_cfg = _full_combined_cfg(tmp_path, roots)

    real = _real_resolve_calls(combined_cfg, settings, cfg_path)
    predicted = [
        (block, sorted(names), kw)
        for block, names, kw in rw._freeze_cache_blocks(combined_cfg, settings)
    ]
    assert "Odd-Flagged-Experiment" in real[0][1]
    assert predicted == real


def test_a_base_blocks_own_trial_type_filter_beats_the_wide_one(tmp_path, monkeypatch):
    # _process_base_block reads the block's own trial_type_filter and falls back
    # to combined.wide's only when it sets none. Predicting the wide filter for
    # every base block would check the slice under the wrong fingerprint.
    isolate_env(monkeypatch)
    roots = [str(make_dataset(tmp_path, n, _tables())) for n in (LIVE, FROZEN)]
    cfg_path = write_config(tmp_path, [LIVE, FROZEN], frozen=[FROZEN])
    settings, _ = _load(cfg_path)
    combined_cfg = _full_combined_cfg(tmp_path, roots)
    combined_cfg["combined_base"]["wide"]["trial_type_filter"] = "testing"

    real = _real_resolve_calls(combined_cfg, settings, cfg_path)
    predicted = [
        (block, sorted(names), kw)
        for block, names, kw in rw._freeze_cache_blocks(combined_cfg, settings)
    ]
    assert {b: kw["trial_type_filter"] for b, _, kw in real if b != "pair_groups"} == {
        "wide": ["testing", "training"],
        "combined_base": "testing",
        "distance_base": ["testing", "training"],
    }
    assert predicted == real


def test_live_config_shape_consults_combined_base_only(scene):
    blocks = rw._freeze_cache_blocks(scene["combined"], scene["settings"])
    assert [(b, sorted(n)) for b, n, _ in blocks] == [
        ("combined_base", sorted([LIVE, FROZEN]))
    ]


def test_no_combined_block_consults_nothing(scene):
    assert rw._freeze_cache_blocks(None, scene["settings"]) == []
    assert rw._freeze_cache_blocks({}, scene["settings"]) == []


# --- the decision -------------------------------------------------------------

def test_frozen_no_video_valid_cache_is_skipped(scene, capsys):
    assert _decide(scene) is True
    assert (
        f"[FROZEN] {FROZEN}: data-frozen, no videos, cache valid - skipping YOLO + per-trial steps"
        in capsys.readouterr().out
    )


def test_live_dataset_is_never_skipped(scene, capsys):
    assert _decide(scene, scene["live_root"]) is False
    assert "[FROZEN]" not in capsys.readouterr().out


def test_figures_only_freeze_is_not_a_data_freeze(scene):
    settings = scene["settings"]
    settings.dataset_overrides[FROZEN].freeze_data = False
    settings.dataset_overrides[FROZEN].freeze_figures = True
    assert _decide(scene, settings=settings) is False


@pytest.mark.parametrize("thawed,thaw_all", [((FROZEN,), False), ((), True)])
def test_thaw_runs_the_steps(scene, thawed, thaw_all):
    assert _decide(scene, thawed=thawed, thaw_all=thaw_all) is False


@pytest.mark.parametrize(
    "rel",
    [
        f"{BATCH}/{STEM}.mp4",
        f"{BATCH}/{STEM}.h264",
        f"flagged/{BATCH}/{STEM}.mp4",
        # Not a YOLO input, but the per-trial move_videos step relocates it.
        f"{BATCH}/august_01_batch_1_testing_1/clip.mkv",
    ],
    ids=["batch-level mp4", "raw h264", "nested", "move_videos container"],
)
def test_any_video_runs_the_steps(scene, rel, capsys):
    target = Path(scene["root"]) / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x\n", encoding="utf-8")
    assert _decide(scene) is False
    assert "video" in capsys.readouterr().out


def test_a_root_the_worker_would_not_read_as_given_runs_the_steps(scene, tmp_path):
    # MAIN_DIRECTORY ".../EB,Control" reaches the worker split on the comma
    # (config._parse_env_paths), so skipping it says nothing about what that
    # run would have done. Everything else here would allow the skip.
    settings = scene["settings"]
    name = f"{FROZEN},odd"
    root = make_dataset(tmp_path, name, _tables())
    settings.dataset_overrides[name] = settings.dataset_overrides[FROZEN]
    calls = _real_resolve_calls(scene["combined"], settings, scene["cfg"])
    kw = next(kw for block, _, kw in calls if block == "combined_base")
    blocks = [("combined_base", frozenset({name}), kw)]
    _seed_slices(settings, [("combined_base", [name], kw)], name)
    assert rw._skip_frozen_dataset(settings, str(root), blocks, thawed=(), thaw_all=False) is False
    # ...and the same layout under a plain name does skip.
    plain = make_dataset(tmp_path, f"{FROZEN}-plain", _tables())
    settings.dataset_overrides[plain.name] = settings.dataset_overrides[FROZEN]
    _seed_slices(settings, [("combined_base", [plain.name], kw)], plain.name)
    blocks = [("combined_base", frozenset({plain.name}), kw)]
    assert rw._skip_frozen_dataset(settings, str(plain), blocks, thawed=(), thaw_all=False) is True


def _drop_slice(scene):
    for f in _freeze.slice_dir(scene["settings"].cache_dir, "combined_base", FROZEN).iterdir():
        f.unlink()


def test_missing_cache_runs_the_steps(scene, capsys):
    _drop_slice(scene)
    assert _decide(scene) is False
    assert "missing or stale" in capsys.readouterr().out


def test_fingerprint_drift_runs_the_steps(scene):
    # distance_limits feeds both distance_stats and the wide rows: tonight's
    # rebuild must read stats derived under the NEW limits.
    scene["settings"].class2_max = 170.0
    assert _decide(scene) is False


def test_unreadable_rows_run_the_steps(scene):
    # load_slice itself reads the rows; a corrupt parquet is a rebuild too.
    rows = _freeze.slice_dir(scene["settings"].cache_dir, "combined_base", FROZEN) / "rows.parquet"
    rows.write_bytes(b"not a parquet file")
    assert _decide(scene) is False


def test_schema_bump_runs_the_steps(scene):
    meta = _freeze.slice_dir(scene["settings"].cache_dir, "combined_base", FROZEN) / "meta.json"
    data = json.loads(meta.read_text())
    data["schema_version"] = _freeze.FREEZE_SCHEMA_VERSION + 1
    meta.write_text(json.dumps(data))
    assert _decide(scene) is False


def _multi_block_scene(tmp_path, monkeypatch, **cfg_kw):
    isolate_env(monkeypatch)
    roots = [str(make_dataset(tmp_path, n, _tables())) for n in (LIVE, FROZEN)]
    cfg_path = write_config(tmp_path, [LIVE, FROZEN], frozen=[FROZEN])
    settings, _ = _load(cfg_path)
    combined_cfg = _full_combined_cfg(tmp_path, roots, **cfg_kw)
    calls = _real_resolve_calls(combined_cfg, settings, cfg_path)
    blocks = rw._freeze_cache_blocks(combined_cfg, settings)
    return settings, roots[1], calls, blocks


def test_every_consulted_block_must_be_valid(tmp_path, monkeypatch):
    settings, root, calls, blocks = _multi_block_scene(
        tmp_path, monkeypatch, pair_b_filter=None
    )
    _seed_slices(settings, calls, FROZEN, skip_blocks=("distance_base",))
    assert rw._skip_frozen_dataset(settings, root, blocks, thawed=(), thaw_all=False) is False
    _seed_slices(settings, calls, FROZEN)
    assert rw._skip_frozen_dataset(settings, root, blocks, thawed=(), thaw_all=False) is True


def test_pair_groups_sharing_a_slice_dir_with_different_filters_never_skip(tmp_path, monkeypatch):
    # Both pair groups cache into frozen/pair_groups/<ds>; the second write
    # always evicts the first's fingerprint, so the first always rebuilds --
    # and its per-trial inputs must be fresh for that.
    settings, root, calls, blocks = _multi_block_scene(tmp_path, monkeypatch)
    _seed_slices(settings, calls, FROZEN)
    assert rw._skip_frozen_dataset(settings, root, blocks, thawed=(), thaw_all=False) is False


def test_a_dataset_no_wide_table_reads_is_not_skipped(scene):
    # "Its slice exists" cannot hold when nothing caches it.
    settings = scene["settings"]
    assert rw._skip_frozen_dataset(settings, scene["root"], [], thawed=(), thaw_all=False) is False


def test_flagged_dirs_riding_along_run_the_steps(scene, tmp_path):
    flagged = tmp_path / "flagged_root"
    (flagged / "Odd-Flagged-Experiment").mkdir(parents=True)
    scene["settings"].flagged_root = str(flagged)
    assert _decide(scene) is False


def test_main_directories_env_runs_the_steps(scene, monkeypatch):
    monkeypatch.setenv("MAIN_DIRECTORIES", scene["root"])
    assert _decide(scene) is False


def test_no_cuda_runs_the_steps(scene, monkeypatch, capsys):
    # Today's YOLO launch for it fails the night without CUDA (allow_cpu off);
    # that launch has to happen, not be skipped along with the steps.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _decide(scene) is False
    assert f"[FROZEN] {FROZEN}: data-frozen, but CUDA is unavailable" in capsys.readouterr().out
    scene["settings"].allow_cpu = True
    assert _decide(scene) is True


def test_a_missing_model_runs_the_steps(scene, capsys):
    Path(scene["settings"].model_path).unlink()
    assert _decide(scene) is False
    assert "is missing" in capsys.readouterr().out


# --- the dataset loop ---------------------------------------------------------

def _state_file(tmp_path: Path, root: Path) -> Path:
    return tmp_path / "cache" / "pipeline" / rw._safe_key(str(root)) / "state.json"


def test_loop_skips_the_frozen_dataset_entirely(scene, monkeypatch, capsys):
    # A video in the live dataset, so its YOLO launch is real.
    live_video = Path(scene["live_root"]) / BATCH / f"{STEM}.mp4"
    live_video.write_text("x\n", encoding="utf-8")
    # Seed a stale-looking pipeline state: a skip must leave it byte-identical.
    state = _state_file(scene["tmp"], Path(scene["root"]))
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text('{"version": 2, "sentinel": true}', encoding="utf-8")

    calls = run_main(monkeypatch, scene["cfg"])

    assert yolo_launches(calls) == [LIVE]
    assert per_trial_runs(calls) == [LIVE]
    assert state.read_text(encoding="utf-8") == '{"version": 2, "sentinel": true}'
    assert "skipping YOLO + per-trial steps" in capsys.readouterr().out


def test_loop_runs_a_frozen_dataset_whose_cache_is_stale(scene, monkeypatch):
    _drop_slice(scene)
    calls = run_main(monkeypatch, scene["cfg"])
    # No video anywhere, so no YOLO launch -- but on a rebuild night the
    # dataset runs its per-trial steps exactly as today.
    assert yolo_launches(calls) == []
    assert per_trial_runs(calls) == [LIVE, FROZEN]


def test_loop_without_cuda_runs_the_frozen_dataset_as_today(scene, monkeypatch, capsys):
    calls = run_main(monkeypatch, scene["cfg"], cuda=False)
    assert yolo_launches(calls) == [LIVE, FROZEN]
    assert per_trial_runs(calls) == [LIVE, FROZEN]
    assert "skipping YOLO + per-trial steps" not in capsys.readouterr().out


def test_loop_without_cuda_fails_at_the_first_launch_as_today(scene, monkeypatch):
    def _worker_fails(main_directory, steps):
        if steps == ("yolo",):
            raise subprocess.CalledProcessError(1, ["fbpipe.pipeline", "yolo"])

    calls = run_main(
        monkeypatch, scene["cfg"], cuda=False, on_pipeline=_worker_fails,
        expect=subprocess.CalledProcessError,
    )
    assert yolo_launches(calls) == [LIVE]
    assert per_trial_runs(calls) == []


def test_loop_runs_a_frozen_dataset_when_the_decision_itself_fails(scene, monkeypatch, capsys):
    # The skip is an optimisation: an error while deciding must neither kill
    # the night nor skip the dataset -- it runs exactly as today.
    def _boom(*a, **k):
        raise OSError("cannot list the flagged root")

    monkeypatch.setattr(rw, "_skip_frozen_dataset", _boom)
    calls = run_main(monkeypatch, scene["cfg"])
    assert per_trial_runs(calls) == [LIVE, FROZEN]
    assert "cannot list the flagged root" in capsys.readouterr().out


def test_loop_thaw_runs_the_frozen_dataset(scene, monkeypatch):
    calls = run_main(monkeypatch, scene["cfg"], "--thaw", FROZEN)
    assert per_trial_runs(calls) == [LIVE, FROZEN]


def test_loop_folder_mode_keeps_todays_behaviour(scene, monkeypatch):
    calls = run_main(monkeypatch, scene["cfg"], "--folder", FROZEN)
    assert per_trial_runs(calls) == [FROZEN]


def test_video_landing_after_the_start_of_the_run_is_still_inferred(scene, monkeypatch, capsys):
    """The skip is provisional until the dataset's own YOLO slot: a video that
    appears before then is launched for, exactly as today's launch would."""
    (Path(scene["live_root"]) / BATCH / f"{STEM}.mp4").write_text("x\n", encoding="utf-8")
    late = Path(scene["root"]) / BATCH / f"{STEM}.mp4"

    def _drop_late_video(main_directory, steps):
        if steps == ("yolo",) and Path(main_directory).name == LIVE:
            late.write_text("x\n", encoding="utf-8")

    calls = run_main(monkeypatch, scene["cfg"], on_pipeline=_drop_late_video)

    assert yolo_launches(calls) == [LIVE, FROZEN]
    assert per_trial_runs(calls) == [LIVE, FROZEN]
    assert (
        f"[FROZEN] {FROZEN}: no longer skippable at its YOLO slot (it holds video files)"
        in capsys.readouterr().out
    )


def test_a_held_back_dataset_whose_slot_check_fails_says_so(scene, monkeypatch, capsys):
    # It runs as today -- and the log names the failed check, not a video.
    real = rw._yolo_work_reason
    frozen_checks = []

    def _slot_check_fails(settings, root):
        if Path(root).name == FROZEN:
            frozen_checks.append(root)
            if len(frozen_checks) > 1:  # the decision's check passed; the slot's fails
                raise OSError("flagged root unreadable")
        return real(settings, root)

    monkeypatch.setattr(rw, "_yolo_work_reason", _slot_check_fails)
    calls = run_main(monkeypatch, scene["cfg"])

    assert yolo_launches(calls) == [FROZEN]
    assert per_trial_runs(calls) == [LIVE, FROZEN]
    out = capsys.readouterr().out
    assert f"[FROZEN] {FROZEN}: no longer skippable at its YOLO slot" in out
    assert "flagged root unreadable" in out
    assert "video appeared" not in out
