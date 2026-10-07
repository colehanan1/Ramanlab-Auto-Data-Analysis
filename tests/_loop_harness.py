"""Scaffolding for the run_workflows dataset-loop tests.

``write_config`` builds a scratch config whose datasets live under *tmp_path*;
``run_main`` runs ``run_workflows.main`` on it with every subprocess launch and
the combined step replaced by recorders, and stops the run right after the
combined-state write (``_load_model_scores_for_envelopes`` is the next call),
so no figure, cleanup, sync or notification code is ever reached.

The YOLO pre-check keeps every launch when CUDA is missing (allow_cpu off) or
the model file is not there, as today's launch would fail the night. So the
harness gives each run a GPU (``torch.cuda.is_available`` patched, never the
real device) and a model file that exists; tests about those two pass
``cuda=False`` or delete the file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.pipeline import run_workflows as rw

# Env overrides load_settings honours; any of them leaking in from the shell
# would change which roots or force flags a test sees.
_ENV_OVERRIDES = (
    "MAIN_DIRECTORY",
    "MAIN_DIRECTORIES",
    "CACHE_DIR",
    "FORCE_PIPELINE",
    "FORCE_YOLO",
    "FORCE_COMBINED",
    "YOLO_NUM_WORKERS",
    "FLAGGED_ROOT",
    "FLAGGED_SECURED_ROOT",
    "AUTO_DISCOVER_FLAGGED",
    "MODEL_PATH",
    "PARALLEL_ENABLED",
    "PARALLEL_N_JOBS",
)


class StopAfterCombined(Exception):
    """Raised in place of the first call after the combined-state write."""


def write_config(
    tmp_path: Path,
    datasets,
    *,
    frozen=(),
    force_pipeline: bool = True,
    force_combined: bool = True,
    num_workers: int = 3,
    **extra,
) -> Path:
    """*extra* sets further top-level keys (e.g. ``allow_cpu=True``)."""
    model = tmp_path / "model.pt"
    model.write_bytes(b"stand-in weights")
    cfg = {
        "dataset_bases": {
            "data": str(tmp_path / "data"),
            "secured": str(tmp_path / "secured"),
        },
        "datasets": list(datasets),
        "dataset_overrides": {
            ds: {"freeze": {"data": True, "figures": True}} for ds in frozen
        },
        "cache_dir": str(tmp_path / "cache"),
        "model_path": str(model),
        "force": {
            "pipeline": force_pipeline,
            "combined": force_combined,
            "yolo": False,
        },
        "yolo": {"num_workers": num_workers},
        # The two gate keys get DIFFERENT values (CLAUDE.md), so code reading
        # the wrong one could never pass by coincidence.
        "proboscis_filter": {"max_eye_prob_distance_px": 170.0},
        "distance_limits": {
            "class2_min": 10.0,
            "class2_max": 160.0,
            "three_fly_max_eye_prob_distance_px": 150.0,
        },
        "analysis": {
            "combined": {
                "combine": {"roots": []},
                "combined_base": {
                    "wide": {
                        "roots": [],
                        "output_csv": str(tmp_path / "out" / "wide.csv"),
                        "measure_cols": ["combined_pct", "combined_base"],
                        "use_per_trial_baseline": True,
                    }
                },
            }
        },
        **extra,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def make_dataset(tmp_path: Path, name: str, files=()) -> Path:
    """``<tmp>/data/<name>/`` holding *files* (paths relative to the root)."""
    root = tmp_path / "data" / name
    root.mkdir(parents=True, exist_ok=True)
    for rel in files:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x\n", encoding="utf-8")
    return root


def isolate_env(monkeypatch, *, cuda: bool = True) -> None:
    for var in _ENV_OVERRIDES:
        monkeypatch.delenv(var, raising=False)
    import torch

    # What yolo_infer.main and fbpipe.pipeline ask too; never the real device,
    # so these tests decide the same on a CPU-only machine.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)


def run_main(
    monkeypatch,
    config_path: Path,
    *argv: str,
    on_pipeline=None,
    cuda: bool = True,
    expect=None,
) -> dict:
    """Run main() up to the combined-state write; return what it launched.

    *on_pipeline*, if given, is called with ``(main_directory, steps)`` for
    every recorded launch -- e.g. to drop a file mid-run, or to raise as a
    failing worker would. *expect* is the exception main() must end with
    (default: the stop after the combined-state write).
    """
    calls: dict = {"pipeline": [], "combined": []}

    def _run_pipeline(config_path, *, main_directory=None, steps=None, num_workers=1):
        calls["pipeline"].append(
            (main_directory, tuple(steps) if steps else None, num_workers)
        )
        if on_pipeline is not None:
            on_pipeline(main_directory, tuple(steps) if steps else None)

    def _run_combined(cfg, settings, **kwargs):
        calls["combined"].append(kwargs)
        return None

    def _stop(settings):
        raise StopAfterCombined

    isolate_env(monkeypatch, cuda=cuda)
    monkeypatch.setattr(rw, "_run_pipeline", _run_pipeline)
    monkeypatch.setattr(rw, "_run_combined", _run_combined)
    monkeypatch.setattr(rw, "_load_light_checks_for_envelopes", lambda: None)
    monkeypatch.setattr(rw, "_load_model_scores_for_envelopes", _stop)
    # main() installs these as module globals; keep them out of other tests.
    monkeypatch.setattr(rw, "set_protocol", lambda protocol: None)
    monkeypatch.setattr(rw, "set_dataset_light_windows", lambda windows: None)
    monkeypatch.setattr(rw, "set_dataset_odor_remap", lambda remap: None)
    from scripts.analysis import envelope_combined, envelope_exports

    monkeypatch.setattr(envelope_combined, "set_runtime_settings", lambda cfg: None)
    monkeypatch.setattr(envelope_exports, "set_runtime_settings", lambda cfg: None)

    with pytest.raises(expect or StopAfterCombined):
        rw.main(["--config", str(config_path), *argv])
    return calls


def yolo_launches(calls: dict) -> list[str]:
    return [Path(d).name for d, steps, _ in calls["pipeline"] if steps == ("yolo",)]


def per_trial_runs(calls: dict) -> list[str]:
    return [Path(d).name for d, steps, _ in calls["pipeline"] if steps != ("yolo",)]
