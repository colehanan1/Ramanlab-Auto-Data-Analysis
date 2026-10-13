"""A config_new.yaml run and the same run through an extends-overlay do identical PER work.

``config_new_abdomen.yaml`` is ``extends: config_new.yaml`` plus an ``abdomen:`` block. ``main()`` must hand every PER
reader and child process the BASE file -- the same absolute path string a plain run passes (binarized_per_rasters
even writes its --config argument into a JSON, and three cache keys embed it) -- and add exactly one thing: the
abdomen stage, once PER is done. This test runs ``main()`` twice, once per config, with every call that writes a
file, launches a process or mutates module state replaced by a recorder, and compares the two call traces: the PER
calls (names, arguments, order) must be identical, every config path in them must be ``config_new.yaml`` itself, and
the overlay run may add only ``_run_abdomen_stages``. It also pins the one thing a PER run gains, the shared run
lock: held across the PER stages, released before the abdomen stage, and the only file created.

The overlay is built in ``tmp_path`` and extends the LIVE ``config/config_new.yaml`` by absolute path (skipped when
that gitignored file is absent). The cache dir and the predictions CSV are redirected into ``tmp_path`` so nothing
outside it is touched.
"""
from __future__ import annotations

import dataclasses
import inspect
import os
import subprocess
import types
from pathlib import Path

import pytest
import yaml

from fbpipe.config import resolve_config_path
from fbpipe.utils.run_locks import per_run_active
from scripts.analysis import envelope_combined, envelope_exports
from scripts.pipeline import run_workflows as rw

LIVE = Path(__file__).resolve().parents[1] / "config" / "config_new.yaml"
RELATIVE = "config/config_new.yaml"

pytestmark = pytest.mark.skipif(not LIVE.is_file(), reason="config/config_new.yaml is gitignored and absent")


def _plain(value):
    """Plain data a trace can be compared on: Paths as strings, mappings and sequences copied, objects (Settings,
    ConfigLayers) by their repr -- which for these dataclasses spells out every field."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "items"):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)) or hasattr(value, "isoformat"):
        return value
    return repr(value)


class Trace:
    """What one ``main()`` did: ``calls`` as (name, args, kwargs) plain data, ``raw`` the same as received, and
    ``lock_held`` whether a PER run held the shared run lock when each call was made."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.calls, self.raw, self.lock_held = [], [], []

    def recorder(self, name, result=None):
        def record(*args, **kwargs):
            self.calls.append((name, _plain(args), _plain(kwargs)))
            self.raw.append((name, args, kwargs))
            self.lock_held.append(per_run_active(self.cache_dir))
            return result(*args, **kwargs) if callable(result) else result

        record.__name__ = name
        return record

    def names(self):
        return [name for name, _, _ in self.calls]

    def without(self, name):
        keep = [i for i, call in enumerate(self.calls) if call[0] != name]
        return [self.calls[i] for i in keep], [self.lock_held[i] for i in keep]


@pytest.fixture
def harness(tmp_path):
    """``run(argv) -> Trace`` for rw.main with every side effect stubbed by a recorder (undone after each run)."""
    cache = tmp_path / "cache"
    predictions = tmp_path / "model_predictions.csv"
    predictions.write_text("dataset,fly,fly_number,trial_label,score\n", encoding="utf-8")
    environment = dict(os.environ)

    def run(argv) -> Trace:
        trace = Trace(cache)
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("FBPIPE_DISABLE_NTFY", "1")
            for name, empty in (("LAST_EYE_QC_SUMMARY", ""), ("LAST_ABDOMEN_SUMMARY", ""),
                                ("LAST_RUN_ABDOMEN_ONLY", False)):
                mp.setattr(rw, name, empty)             # main() sets these; put them back afterwards
            # Every _run_* stage (the pipeline, combine, reactions, every figure set, the SMB copy and the
            # data-secured sync, ...), whatever it is called today; _run_abdomen_stages is one of them.
            for name, fn in list(vars(rw).items()):
                if name.startswith("_run_") and inspect.isfunction(fn):
                    mp.setattr(rw, name, trace.recorder(name))
            # The combined stage returns its deferred secure-copy, which main() calls after the figures.
            mp.setattr(rw, "_run_combined", trace.recorder(
                "_run_combined", lambda *a, **k: trace.recorder("deferred_secure_cleanup")))
            for name in ("_batch_copy_to_smb", "_write_state", "_rerender_envelope_block_with_scores",
                         "_load_model_scores_for_envelopes", "_load_light_checks_for_envelopes", "set_protocol",
                         "set_dataset_light_windows", "set_dataset_odor_remap", "set_model_scores",
                         "set_light_check_fractions"):
                mp.setattr(rw, name, trace.recorder(name))
            mp.setattr(rw, "_should_skip", trace.recorder("_should_skip", False))
            mp.setattr(rw, "_should_skip_with_manifest", trace.recorder("_should_skip_with_manifest", False))
            mp.setattr(rw, "_eye_qc_summary", trace.recorder("_eye_qc_summary", ""))
            mp.setattr(rw, "ntfy_notify", trace.recorder("ntfy_notify"))
            for module in (envelope_combined, envelope_exports):
                mp.setattr(module, "set_runtime_settings",
                           trace.recorder(f"{module.__name__}.set_runtime_settings"))
            mp.setattr(subprocess, "run", trace.recorder("subprocess.run",
                                                         lambda *a, **k: types.SimpleNamespace(returncode=0)))
            mp.setattr(subprocess, "Popen", trace.recorder(
                "subprocess.Popen", lambda *a, **k: types.SimpleNamespace(returncode=0, pid=1, poll=lambda: 0,
                                                                          wait=lambda *a, **k: 0)))
            # Real files' mtimes are cache keys; they must not make two back-to-back runs differ.
            mp.setattr(rw, "_file_mtime_key", lambda path: None)

            load_settings = rw.load_settings

            def recorded_load_settings(path):
                trace.recorder("load_settings")(path)
                settings = load_settings(path)
                return dataclasses.replace(
                    settings, cache_dir=str(cache),
                    reaction_prediction=dataclasses.replace(settings.reaction_prediction,
                                                            output_csv=str(predictions)))

            mp.setattr(rw, "load_settings", recorded_load_settings)
            assert rw.main(argv) is None
        return trace

    yield run
    os.environ.clear()                         # load_settings may read a .env into the process environment
    os.environ.update(environment)


@pytest.fixture
def overlay(tmp_path) -> Path:
    path = tmp_path / "config_new_abdomen.yaml"
    path.write_text(yaml.safe_dump({
        "extends": str(LIVE),                                # the ABSOLUTE path of the live config_new.yaml
        "abdomen": {"enabled": True, "model_path": "/not/used/best.pt", "out_base": str(tmp_path / "flys_Abdomen")},
    }), encoding="utf-8")
    return path


def _dataset_folder() -> str:
    roots = rw.load_settings(LIVE).main_directories
    if not roots:
        pytest.skip("no dataset root of config_new.yaml exists on this machine")
    return Path(roots[0]).name


def _config_arguments(trace):
    """(call, path) for every config file a recorded call received: a ``*.yaml`` argument or the value after
    ``--config`` in a command list."""
    found = []
    for name, args, kwargs in trace.calls:
        for value in [*args, *kwargs.values()]:
            if isinstance(value, str) and value.endswith((".yaml", ".yml")):
                found.append((name, value))
            elif isinstance(value, list) and "--config" in value:
                found.append((name, value[value.index("--config") + 1]))
    return found


ARGVS = [
    pytest.param([], id="full-run"),
    pytest.param(["--figures-only"], id="figures-only"),
    pytest.param(["--folder", "<first dataset>"], id="single-folder"),
    pytest.param(["--abdomen-video"], id="abdomen-video-flag"),
]


@pytest.mark.parametrize("argv", ARGVS)
def test_an_overlay_run_does_the_same_per_work_as_a_plain_run_and_adds_only_the_abdomen_stage(harness, overlay, argv):
    argv = [_dataset_folder() if a == "<first dataset>" else a for a in argv]
    plain = harness(["--config", RELATIVE, *argv])
    layered = harness(["--config", str(overlay), *argv])

    stages = {"_run_reactions", "_run_dataset_means", "_batch_copy_to_smb", "_run_data_secured_sync"}
    assert stages <= set(plain.names()), "a real run's worth of stages must have been traced"
    assert "_run_abdomen_stages" not in plain.names(), "a plain config_new.yaml run must not even call the stage"
    assert layered.names().count("_run_abdomen_stages") == 1

    per_calls, per_locks = layered.without("_run_abdomen_stages")
    assert per_calls == plain.calls, "the overlay must not change a single PER call, its arguments or its order"
    assert per_locks == plain.lock_held

    expected = str(resolve_config_path(RELATIVE))
    for trace in (plain, layered):
        seen = _config_arguments(trace)
        assert {path for _, path in seen} == {expected}, "every PER reader and child gets config_new.yaml itself"
        required = {"load_settings", "_run_reactions", "_run_dataset_means"}
        if "--figures-only" not in argv:
            required |= {"_run_pipeline", "_run_combined"}
        assert required <= {name for name, _ in seen}


@pytest.mark.parametrize("argv", ARGVS)
def test_the_abdomen_stage_gets_the_overlay_and_the_flags_of_the_run(harness, overlay, argv):
    argv = [_dataset_folder() if a == "<first dataset>" else a for a in argv]
    layered = harness(["--config", str(overlay), *argv])
    ((_, (layers, settings), kwargs),) = [c for c in layered.raw if c[0] == "_run_abdomen_stages"]
    assert layers.path == overlay and layers.base_path == resolve_config_path(RELATIVE)
    assert dict(layers.overlay)["abdomen"]["enabled"] is True
    assert kwargs == {"figures_only": "--figures-only" in argv,
                      "folder": argv[argv.index("--folder") + 1] if "--folder" in argv else None,
                      "annotated_video": "--abdomen-video" in argv}
    assert layered.names()[-2:] == ["_run_abdomen_stages", "_eye_qc_summary"], "after PER, before the QC line"


def test_the_shared_lock_covers_the_per_stages_and_is_released_before_the_abdomen_stage(harness, overlay):
    layered = harness(["--config", str(overlay)])
    stages = [(name, held) for name, held in zip(layered.names(), layered.lock_held)
              if name.startswith("_run_") or name in ("_batch_copy_to_smb", "_write_state", "deferred_secure_cleanup")]
    per = [(name, held) for name, held in stages if name != "_run_abdomen_stages"]
    assert {"_run_reactions", "_run_dataset_means", "_batch_copy_to_smb"} <= {name for name, _ in per}
    assert all(held for _, held in per), f"PER work ran outside the lock: {[n for n, held in per if not held]}"
    assert stages[-1] == ("_run_abdomen_stages", False), "the abdomen tracker must not see this very run as PER"
    assert not per_run_active(layered.cache_dir), "nothing holds the lock after main() returns"


def test_the_shared_lock_file_is_the_only_thing_a_run_adds_under_the_cache_dir(harness, overlay, tmp_path):
    harness(["--config", RELATIVE])
    cache = tmp_path / "cache"
    assert sorted(str(p.relative_to(cache)) for p in cache.rglob("*")) == ["locks", "locks/per.lock"]
    assert not per_run_active(cache)
    harness(["--config", str(overlay)])
    assert sorted(str(p.relative_to(cache)) for p in cache.rglob("*")) == ["locks", "locks/per.lock"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "cache", "config_new_abdomen.yaml", "model_predictions.csv"], "nothing else was written"
