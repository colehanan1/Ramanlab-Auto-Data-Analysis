"""Per-fly envelope trace figures rendered by forked worker processes.

``EnvelopePlotConfig.render_workers`` > 1 splits the (fly, fly_number) groups
of one ``generate_envelope_plots`` call across that many FORKED workers. These
tests pin that:

  * the default (1) is today's serial loop -- no process pool is ever built;
  * the forked render is byte-identical to the serial one, including every
    annotation drawn from the module-level registries the pipeline driver
    fills at runtime (model scores, light checks, protocol, light-only
    windows, odor remap). A spawned/loky worker re-imports the module and sees
    those registries empty, which is why the start method must be ``fork``;
  * frozen datasets stay skipped and thawed ones redraw, exactly as serially;
  * the parent ends with the rcParams the serial loop leaves behind;
  * a failing or dying worker raises in the parent instead of hanging;
  * ``max_flies``, GUI matplotlib backends and a CUDA context in the parent
    fall back to the serial loop;
  * the knob reaches the renderer from a YAML envelope entry.
"""

from __future__ import annotations

import concurrent.futures
import faulthandler
import hashlib
import os
import re
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from fbpipe.config import DatasetOverride  # noqa: E402
from scripts.analysis import envelope_visuals as ev  # noqa: E402

# The test process carries native BLAS threads, so Python 3.12 warns on every
# fork. The production fork is deliberate (see envelope_visuals); keep the
# warnings summary for real problems.
pytestmark = pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded:DeprecationWarning"
)

LIVE = "Hex-Training-24-0.1"
FROZEN = "Hex-Control-24-0.1"
LIGHT_ONLY = "LightSweep-Control-24-0.01"
N_SAMPLES = 2400  # 60 s at 40 fps

# (dataset, fly, fly_number, trial labels)
FLIES = [
    (LIVE, "august_10_batch_1_rig_2", 1, ("testing_1_hexanol", "testing_2_linalool", "testing_3_citral")),
    (LIVE, "august_10_batch_1_rig_2", 2, ("testing_1_hexanol", "testing_2_linalool")),
    (LIVE, "august_11_batch_1_rig_2", 1, ("testing_1_hexanol", "testing_2_linalool", "testing_3_citral")),
    (LIVE, "august_11_batch_1_rig_2", 2, ("testing_1_hexanol", "testing_2_linalool")),
    (FROZEN, "august_12_batch_1_rig_3", 1, ("testing_1_hexanol", "testing_2_linalool")),
    (FROZEN, "august_12_batch_1_rig_3", 2, ("testing_1_hexanol", "testing_2_linalool")),
    (LIGHT_ONLY, "september_01_batch_1_rig_1", 1, ("testing_1", "testing_2")),
]


def _write_wide_table(path: Path) -> None:
    rows = []
    t = np.arange(N_SAMPLES) / 40.0
    for i, (dataset, fly, fly_number, labels) in enumerate(FLIES):
        for j, label in enumerate(labels):
            trace = 10.0 + 3.0 * np.sin(t * (0.3 + 0.07 * i + 0.11 * j))
            trace[t >= 31.0] += 25.0 + 10.0 * j  # a response after odor onset
            row = {
                "dataset": dataset,
                "fly": fly,
                "fly_number": fly_number,
                "fly_type": "GR5a-Old",
                "trial_type": "testing",
                "trial_label": label,
                "fps": 40.0,
                "global_min": 1.0,
                "global_max": 80.0,
                "trimmed_global_min": 1.0,
                "trimmed_global_max": 80.0,
                "trace_len": N_SAMPLES,
                "trial_odor_on_s": 30.0,
                "trial_odor_off_s": 60.0,
                "trial_duration_s": 90.0,
                "trial_light_on_s": np.nan,
            }
            row.update({f"dir_val_{k}": float(v) for k, v in enumerate(trace)})
            rows.append(row)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _registries() -> dict:
    """What the pipeline driver registers before the score re-render."""
    scores = {}
    checks = {}
    for dataset, fly, fly_number, labels in FLIES:
        for j, label in enumerate(labels):
            scores[(dataset, fly, str(fly_number), label)] = (j + fly_number) % 6
            checks[(dataset, fly, "testing", j + 1)] = 1.0 if j % 2 == 0 else 0.42
    return {
        "protocol": "v2",
        "scores": scores,
        "checks": checks,
        "light_windows": {LIGHT_ONLY: (15.0, 30.0)},
        "odor_remap": {LIVE: {"Linalool": "Linalool (1%)"}},
    }


def _install(reg: dict | None) -> None:
    reg = reg or {"protocol": "legacy", "scores": {}, "checks": {}, "light_windows": {}, "odor_remap": {}}
    ev.set_protocol(reg["protocol"])
    ev.set_model_scores(reg["scores"])
    ev.set_light_check_fractions(reg["checks"])
    ev.set_dataset_light_windows(reg["light_windows"])
    ev.set_dataset_odor_remap(reg["odor_remap"])


@pytest.fixture(autouse=True)
def _isolate_registries():
    """Snapshot/restore every module-level registry these tests populate."""
    saved = (
        dict(ev._MODEL_SCORES),
        dict(ev._LIGHT_CHECKS),
        dict(ev._DATASET_LIGHT_WINDOWS),
    )
    try:
        yield
    finally:
        ev.set_model_scores(saved[0])
        ev.set_light_check_fractions(saved[1])
        ev.set_dataset_light_windows(saved[2])


@pytest.fixture
def no_hang():
    """Fail loudly (dump tracebacks, exit) instead of hanging the suite."""
    faulthandler.dump_traceback_later(180, exit=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def wide_table(tmp_path) -> Path:
    path = tmp_path / "wide.parquet"
    _write_wide_table(path)
    return path


def _cfg(wide: Path, out_dir: Path, **overrides) -> ev.EnvelopePlotConfig:
    opts = dict(
        matrix_npy=Path("/does/not/exist/combined_base/envelope_matrix_float16.npy"),
        codes_json=Path("/does/not/exist/code_maps.json"),
        out_dir=out_dir,
        latency_sec=0.0,
        odor_latency_s=2.15,
        trial_type="testing",
        wide_input=wide,
        overwrite=True,
        threshold_min_delta=5.0,
        threshold_anchor_s=5.0,
        dataset_overrides={FROZEN: DatasetOverride(freeze_figures=True)},
    )
    opts.update(overrides)
    return ev.EnvelopePlotConfig(**opts)


def _digests(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.md5(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


class _PoolSpy:
    """Stands in for ProcessPoolExecutor: records the start method, delegates."""

    def __init__(self, real):
        self.real = real
        self.start_methods: list[str] = []

    def __call__(self, *args, **kwargs):
        self.start_methods.append(kwargs["mp_context"].get_start_method())
        return self.real(*args, **kwargs)


def _forbid_pool(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise AssertionError("a process pool was built on the serial path")

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _boom)


# ---------------------------------------------------------------------------
# Current behaviour, pinned (passes before and after the change)
# ---------------------------------------------------------------------------


def test_serial_render_draws_live_and_light_only_skips_frozen(tmp_path, wide_table):
    _install(_registries())
    out = tmp_path / "serial"
    ev.generate_envelope_plots(_cfg(wide_table, out))

    names = sorted(_digests(out))
    assert names == [
        f"{LIVE}/august_10_batch_1_rig_2_fly1_testing_envelope_trials_by_odor_30_shifted.png",
        f"{LIVE}/august_10_batch_1_rig_2_fly2_testing_envelope_trials_by_odor_30_shifted.png",
        f"{LIVE}/august_11_batch_1_rig_2_fly1_testing_envelope_trials_by_odor_30_shifted.png",
        f"{LIVE}/august_11_batch_1_rig_2_fly2_testing_envelope_trials_by_odor_30_shifted.png",
        f"{LIGHT_ONLY}/september_01_batch_1_rig_1_fly1_testing_envelope_trials_by_odor_30_shifted.png",
    ]


# ---------------------------------------------------------------------------
# The knob
# ---------------------------------------------------------------------------


def test_render_workers_defaults_to_serial_and_builds_no_pool(tmp_path, wide_table, monkeypatch):
    cfg = _cfg(wide_table, tmp_path / "out")
    assert cfg.render_workers == 1
    _forbid_pool(monkeypatch)
    _install(_registries())
    ev.generate_envelope_plots(cfg)
    assert len(_digests(tmp_path / "out")) == 5


def test_forked_render_is_byte_identical_to_serial(
    tmp_path, wide_table, monkeypatch, no_hang
):
    reg = _registries()

    _install(reg)
    ev.generate_envelope_plots(_cfg(wide_table, tmp_path / "serial"))
    serial = _digests(tmp_path / "serial")

    spy = _PoolSpy(concurrent.futures.ProcessPoolExecutor)
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", spy)
    for workers in (2, 3):
        _install(reg)
        out = tmp_path / f"forked_{workers}"
        ev.generate_envelope_plots(_cfg(wide_table, out, render_workers=workers))
        assert _digests(out) == serial, f"render_workers={workers} differs from serial"
    # Requested explicitly: Python 3.14 stops defaulting to fork on Linux.
    assert spy.start_methods == ["fork", "fork"]

    # The registries really reach the figures: rendered without them (what a
    # spawned worker would see), the same flies come out different.
    _install(None)
    ev.generate_envelope_plots(_cfg(wide_table, tmp_path / "bare"))
    bare = _digests(tmp_path / "bare")
    differing = {name for name in serial if bare.get(name) != serial[name]}
    assert differing == set(serial)


def test_forked_render_honours_thaw_like_serial(tmp_path, wide_table, no_hang):
    _install(_registries())
    ev.generate_envelope_plots(_cfg(wide_table, tmp_path / "serial", thawed=(FROZEN,)))
    serial = _digests(tmp_path / "serial")
    assert any(name.startswith(f"{FROZEN}/") for name in serial)

    ev.generate_envelope_plots(
        _cfg(wide_table, tmp_path / "forked", thawed=(FROZEN,), render_workers=3)
    )
    assert _digests(tmp_path / "forked") == serial


def test_forked_render_prints_correct_summary(tmp_path, wide_table, capfd, no_hang):
    _install(_registries())
    capfd.readouterr()
    ev.generate_envelope_plots(_cfg(wide_table, tmp_path / "out", render_workers=3))
    out = capfd.readouterr().out

    # Every group's per-fly line still appears (order may interleave).
    assert out.count("[DEBUG] envelope_plots: generating") == len(FLIES)
    assert out.count("[FROZEN] Skipping figure") == 2
    summary = [line for line in out.splitlines() if line.startswith("[envelope_plots] ")]
    assert summary == [
        "[envelope_plots] 3 fork workers: 7 fly groups -> 5 rendered, 2 frozen, "
        "0 already on disk, 0 without plottable trials."
    ]
    assert len(_digests(tmp_path / "out")) == 5


def test_forked_render_counts_existing_figures(tmp_path, wide_table, capfd, no_hang):
    _install(_registries())
    out_dir = tmp_path / "out"
    ev.generate_envelope_plots(_cfg(wide_table, out_dir))
    before = {p: p.stat().st_mtime_ns for p in out_dir.rglob("*.png")}
    capfd.readouterr()

    ev.generate_envelope_plots(_cfg(wide_table, out_dir, overwrite=False, render_workers=2))
    out = capfd.readouterr().out
    assert "2 fork workers: 7 fly groups -> 0 rendered, 2 frozen, 5 already on disk" in out
    assert {p: p.stat().st_mtime_ns for p in out_dir.rglob("*.png")} == before


@pytest.mark.parametrize("also_frozen", [(), (LIVE, LIGHT_ONLY)], ids=["drawn", "all-frozen"])
def test_forked_render_leaves_parent_rcparams_as_serial_does(
    tmp_path, wide_table, also_frozen, no_hang
):
    """The per-fly body styles pyplot's global rcParams before it draws, so a
    serial render leaves them styled for whatever this process draws next.
    Forked workers style only their own copies: the parent must end in the
    same state -- styled iff some figure was drawn."""
    overrides = {ds: DatasetOverride(freeze_figures=True) for ds in (FROZEN, *also_frozen)}
    after = {}
    for workers in (1, 2):
        _install(_registries())
        with plt.rc_context():
            before = plt.rcParams["font.size"]
            ev.generate_envelope_plots(
                _cfg(
                    wide_table,
                    tmp_path / f"workers_{workers}",
                    style_scale=1.37,
                    dataset_overrides=overrides,
                    render_workers=workers,
                )
            )
            after[workers] = dict(plt.rcParams)
    assert after[2] == after[1]
    # Not vacuous: drawing really restyles (font.size = 10 * style_scale).
    expected = before if also_frozen else pytest.approx(13.7)
    assert after[1]["font.size"] == expected


def test_workers_freeze_inherited_objects_parent_gc_untouched(tmp_path, wide_table, monkeypatch, no_hang):
    """Each worker freezes what it inherited, so its GC never walks (and so
    copy-on-write copies) the parent's objects; the parent's GC is untouched."""
    import gc

    real = ev._select_trial_rows
    seen = tmp_path / "freeze_counts"
    seen.mkdir()

    def _recording(fly_df):
        (seen / f"{os.getpid()}.txt").write_text(str(gc.get_freeze_count()))
        return real(fly_df)

    monkeypatch.setattr(ev, "_select_trial_rows", _recording)
    _install(_registries())
    before = gc.get_freeze_count()  # CPython itself parks a few hundred tuples here
    ev.generate_envelope_plots(_cfg(wide_table, tmp_path / "out", render_workers=2))

    counts = [int(p.read_text()) for p in seen.glob("*.txt")]
    assert len(counts) == 2 and all(n > before + 10_000 for n in counts), (before, counts)
    assert gc.get_freeze_count() == before


# ---------------------------------------------------------------------------
# Failures surface in the parent
# ---------------------------------------------------------------------------


def test_worker_exception_is_raised_in_parent(tmp_path, wide_table, monkeypatch, no_hang):
    real = ev._select_trial_rows

    def _failing(fly_df):
        if str(fly_df["fly"].iloc[0]) == "august_11_batch_1_rig_2":
            raise ValueError("boom: bad fly")
        return real(fly_df)

    monkeypatch.setattr(ev, "_select_trial_rows", _failing)
    _install(_registries())
    with pytest.raises(ValueError, match="boom: bad fly"):
        ev.generate_envelope_plots(_cfg(wide_table, tmp_path / "out", render_workers=3))


def test_dead_worker_raises_instead_of_hanging(tmp_path, wide_table, monkeypatch, no_hang):
    real = ev._select_trial_rows

    def _dying(fly_df):
        if str(fly_df["fly"].iloc[0]) == "august_11_batch_1_rig_2":
            os._exit(1)  # what an OOM kill looks like to the pool
        return real(fly_df)

    monkeypatch.setattr(ev, "_select_trial_rows", _dying)
    _install(_registries())
    with pytest.raises(concurrent.futures.process.BrokenProcessPool):
        ev.generate_envelope_plots(_cfg(wide_table, tmp_path / "out", render_workers=3))


# ---------------------------------------------------------------------------
# Serial fallbacks
# ---------------------------------------------------------------------------


def test_max_flies_renders_serially(tmp_path, wide_table, monkeypatch):
    _forbid_pool(monkeypatch)
    _install(_registries())
    ev.generate_envelope_plots(_cfg(wide_table, tmp_path / "out", render_workers=4, max_flies=2))
    assert len(_digests(tmp_path / "out")) == 2


def test_gui_backend_renders_serially(tmp_path, wide_table, monkeypatch, capfd):
    """Forking a process that holds a Qt/Tk display connection is unsafe."""
    _forbid_pool(monkeypatch)
    monkeypatch.setattr(ev.matplotlib, "get_backend", lambda: "QtAgg")
    _install(_registries())
    capfd.readouterr()
    ev.generate_envelope_plots(_cfg(wide_table, tmp_path / "out", render_workers=4))
    assert len(_digests(tmp_path / "out")) == 5
    assert re.search(r"render_workers=4 ignored: matplotlib backend 'qtagg'", capfd.readouterr().out)


def _fake_torch(is_initialized) -> types.SimpleNamespace:
    return types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=is_initialized))


def _unreadable() -> bool:
    raise RuntimeError("torch is half-imported")


@pytest.mark.parametrize("is_initialized", [lambda: True, _unreadable], ids=["context", "unreadable"])
def test_cuda_context_in_parent_renders_serially(
    tmp_path, wide_table, monkeypatch, capfd, is_initialized
):
    """A fork must never copy a live CUDA context into the workers; when torch
    cannot say whether this process holds one, assume it does."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(is_initialized))
    _forbid_pool(monkeypatch)
    _install(_registries())
    capfd.readouterr()
    ev.generate_envelope_plots(_cfg(wide_table, tmp_path / "out", render_workers=4))
    assert len(_digests(tmp_path / "out")) == 5
    assert "render_workers=4 ignored: this process may hold a CUDA context" in capfd.readouterr().out


def test_cuda_driver_without_context_still_forks(tmp_path, wide_table, monkeypatch, no_hang):
    """The driver's real state: importing fbpipe.pipeline runs
    torch.cuda.is_available(), which loads the CUDA driver but creates no
    context (torch.cuda.is_initialized() stays False). That alone must not cost
    the nightly its parallel render."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(lambda: False))
    spy = _PoolSpy(concurrent.futures.ProcessPoolExecutor)
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", spy)
    _install(_registries())
    ev.generate_envelope_plots(_cfg(wide_table, tmp_path / "out", render_workers=2))
    assert spy.start_methods == ["fork"]
    assert len(_digests(tmp_path / "out")) == 5


def test_more_workers_than_groups_is_fine(tmp_path, wide_table, no_hang):
    _install(_registries())
    ev.generate_envelope_plots(_cfg(wide_table, tmp_path / "serial"))
    ev.generate_envelope_plots(_cfg(wide_table, tmp_path / "forked", render_workers=32))
    assert _digests(tmp_path / "forked") == _digests(tmp_path / "serial")


def test_worker_count_is_capped_at_cpu_count(tmp_path, wide_table, monkeypatch, capfd, no_hang):
    """A typo such as render_workers: 60 must not fork 60 ~1 GB workers."""
    monkeypatch.setattr(ev.os, "cpu_count", lambda: 2)
    _install(_registries())
    capfd.readouterr()
    ev.generate_envelope_plots(_cfg(wide_table, tmp_path / "out", render_workers=60))
    assert "[envelope_plots] 2 fork workers: 7 fly groups" in capfd.readouterr().out


# ---------------------------------------------------------------------------
# Plumbing: YAML envelope entry -> EnvelopePlotConfig -> renderer
# ---------------------------------------------------------------------------


def test_render_workers_flows_from_yaml_entry_through_rerender(tmp_path, monkeypatch):
    import scripts.pipeline.run_workflows as rw

    seen: list[ev.EnvelopePlotConfig] = []
    monkeypatch.setattr(rw, "generate_envelope_plots", seen.append)
    entry = {
        "matrix_npy": str(tmp_path / "m.npy"),
        "codes_json": str(tmp_path / "c.json"),
        "out_dir": str(tmp_path / "out"),
        "latency_sec": 0.0,
        "overwrite": False,
        "render_workers": 6,
    }
    rw._rerender_envelope_block_with_scores([entry, {**entry, "render_workers": 1}], "test")
    assert [(c.render_workers, c.overwrite) for c in seen] == [(6, True), (1, True)]

    # Absent key -> serial default.
    entry.pop("render_workers")
    config, _ = rw._envelope_plot_config(entry)
    assert config.render_workers == 1
