"""combined.combine over worker processes must match the serial run exactly.

The batches of every live dataset are independent -- each one reads and writes
only inside its own directory -- so run_workflows can fan them out when
``parallel.enabled`` is set. The trap is module state. The driver configures
the analysis modules once, in its own process (``set_protocol``,
``set_runtime_settings`` and friends), and a fresh worker starts from the
import-time defaults: protocol "legacy", no settings. The 2026-10-05 audit ran
combine through plain ``parallel_map`` and got different files -- workers took
the legacy path, skipped the odor map, and wrote
``pretest_1_fly1_distances_...`` instead of ``pretest_1_ethylbutyrate_fly1_...``.

These tests hold the parallel path to the serial one: same files byte for byte,
same log in the same order, the same exception, and every driver-set global
present in the workers.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import io
import shutil
import types
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import scripts.analysis.envelope_combined as ec
import scripts.analysis.envelope_exports as ee
import scripts.analysis.envelope_visuals as ev
from fbpipe import figure_export
from fbpipe.config import load_settings
from _combine_fixtures import ODORS, tree_digest, write_batch

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _restore_driver_globals():
    """Every test here sets driver globals; put them all back afterwards."""

    saved = {key: _copy(value) for key, value in ec._capture_driver_state()["globals"].items()}
    try:
        yield
    finally:
        ec._combine_worker_init({"globals": saved, "svg_sidecar": False})


def _copy(value):
    return dict(value) if isinstance(value, dict) else value


def _settings(tmp_path: Path, *, protocol: str = "v2", n_jobs: int = 3, enabled: bool = True):
    """A real Settings from a minimal config. RandomPanel-Probe forces every
    trial to testing -- that only happens if a worker can see the settings."""

    cfg = tmp_path / f"config_{protocol}_{n_jobs}_{enabled}.yaml"
    cfg.write_text(
        "\n".join(
            [
                f"protocol: {protocol}",
                f"cache_dir: {tmp_path / 'cache'}",
                "datasets:",
                "  - Hex-Sensitivity-24-0.1",
                "  - RandomPanel-Probe",
                "dataset_overrides:",
                "  RandomPanel-Probe:",
                "    trial_type_override: testing",
                "  Hex-Sensitivity-24-0.1:",
                "    freeze:",
                "      folders:",
                "        - august_29_batch_1_rig_3",
                "parallel:",
                f"  enabled: {str(enabled).lower()}",
                f"  n_jobs: {n_jobs}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return load_settings(cfg)


def _install(settings) -> None:
    """What run_workflows.main does before combine runs."""

    ev.set_protocol(settings.protocol)
    ec.set_runtime_settings(settings)
    ee.set_runtime_settings(settings)


def _datasets(base: Path) -> list[Path]:
    hex_root = base / "Hex-Sensitivity-24-0.1"
    write_batch(hex_root, "august_28_batch_1_rig_3", flies=(1, 2, 3, 4))
    write_batch(hex_root, "august_28_batch_3_rig_3", flies=(1, 2, 3))
    write_batch(hex_root, "august_29_batch_1_rig_3", flies=(1,))  # folder-frozen
    write_batch(
        hex_root, "september_29_batch_1", flies=(1, 2), host="Flybehavior2", rec_date="20260929"
    )
    (hex_root / "october_02_batch_1_rig_3").mkdir()  # an empty batch: nothing to combine
    panel = base / "RandomPanel-Probe"
    write_batch(panel, "june_30_batch_2", flies=(1, 2), phases=(("training", 3),))
    write_batch(panel, "july_01_batch_1_rig_2", flies=(1,), phases=(("training", 2),))
    return [hex_root, panel]


def _plan(roots: list[Path], settings) -> list:
    from fbpipe.utils.frozen_folders import frozen_folders_for_root

    plan: list = []
    for root in roots:
        skip = frozen_folders_for_root(settings, root)
        plan.append(f"[analysis] combined.combine → {root}")
        plan.append(ec.CombineConfig(root=root, skip_folders=frozenset(skip)))
    return plan


def _run_serial(plan, buf: io.StringIO | None = None) -> str:
    """What run_workflows' loop does with parallel off: print, combine, in order."""

    buf = buf if buf is not None else io.StringIO()
    with contextlib.redirect_stdout(buf):
        for step in plan:
            if isinstance(step, str):
                print(step)
            else:
                ec.combine_distance_angle(step)
    return buf.getvalue()


def _run_parallel(plan, n_jobs: int, buf: io.StringIO | None = None) -> str:
    buf = buf if buf is not None else io.StringIO()
    with contextlib.redirect_stdout(buf):
        ec.run_combine_plan(plan, n_jobs=n_jobs)
    return buf.getvalue()


def _twins(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    serial = _datasets(tmp_path / "serial")
    parallel = []
    for root in serial:
        twin = tmp_path / "parallel" / root.name
        shutil.copytree(root, twin)
        parallel.append(twin)
    return serial, parallel


def _same_trees(a: list[Path], b: list[Path]) -> None:
    for x, y in zip(a, b):
        assert tree_digest(x) == tree_digest(y), x.name


def _norm(log: str, base: Path) -> str:
    return log.replace(str(base), "<BASE>")


@pytest.mark.parametrize("protocol", ["v2", "legacy"])
def test_parallel_combine_matches_serial(tmp_path, protocol):
    settings = _settings(tmp_path, protocol=protocol)
    _install(settings)
    serial, parallel = _twins(tmp_path)

    log_serial = _run_serial(_plan(serial, settings))
    log_parallel = _run_parallel(_plan(parallel, settings), n_jobs=3)

    _same_trees(serial, parallel)
    assert _norm(log_parallel, tmp_path / "parallel") == _norm(log_serial, tmp_path / "serial")
    # the frozen folder stayed frozen, the empty batch produced nothing
    hex_root = parallel[0]
    assert "[FROZEN] combine → skipping recompute for august_29_batch_1_rig_3" in log_parallel
    assert not (hex_root / "august_29_batch_1_rig_3" / "angle_distance_rms_envelope").exists()
    envelopes = {p.name for p in hex_root.rglob("angle_distance_rms_envelope/*.csv")}
    if protocol == "v2":
        # the audit's failure: workers without the driver state lost the odor token
        assert any(n.startswith("pretest_1_") and "_fly1_distances" in n
                   and not n.startswith("pretest_1_fly1") for n in envelopes), envelopes
    else:
        # v1 has no odor map: labels keep the filename's own suffix
        assert not any(odor.lower() in n for odor in ODORS for n in envelopes), envelopes


def test_parallel_combine_saves_identical_trial_plots(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    _install(settings)
    # The parent's styling as an import left it (fbpipe.steps.compose_videos_rms
    # restyles at import time); workers must render with the same.
    monkeypatch.setitem(plt.rcParams, "font.size", 14.0)
    monkeypatch.setitem(plt.rcParams, "axes.linewidth", 2.0)
    serial = tmp_path / "serial" / "Hex-Sensitivity-24-0.1"
    write_batch(serial, "august_28_batch_1_rig_3", flies=(1,), phases=(("testing", 1),))
    write_batch(serial, "august_28_batch_3_rig_3", flies=(1,), phases=(("testing", 1),))
    parallel = tmp_path / "parallel" / serial.name
    shutil.copytree(serial, parallel)

    def plan(root):
        return [ec.CombineConfig(root=root, save_trial_plots=True)]

    _run_serial(plan(serial))
    _run_parallel(plan(parallel), n_jobs=2)
    assert list(parallel.rglob("*.png"))
    _same_trees([serial], [parallel])


def test_workers_receive_every_driver_global(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    _install(settings)
    monkeypatch.setitem(plt.rcParams, "axes.linewidth", 1.25)
    ev.set_dataset_light_windows({"LightSweep-Probe": (30.0, 32.0)})
    ev.set_dataset_odor_remap({"Hex-Probe": {"Citral": "Sour Dough Yeast (25%)"}})
    ev.set_light_check_fractions({("Hex-Probe", "june_30_batch_2", "testing", 3): 0.5})
    ev.set_model_scores({("Hex-Probe", "june_30_batch_2", "1", "testing_3"): 4})

    figure_export.install_svg_sidecar()  # run_workflows --svg
    try:
        state = ec._capture_driver_state()
        with ec._combine_executor(2, state) as pool:
            seen = pool.submit(_probe_worker, None).result()
    finally:
        figure_export.uninstall_svg_sidecar()

    assert seen["module"] == ec.__file__
    assert seen["protocol"] == "v2"
    assert seen["runtime_settings"] == settings
    assert seen["exports_settings"] == settings
    assert seen["globals"] == {key: _copy(v) for key, v in state["globals"].items()}
    assert state["svg_sidecar"] and seen["svg_sidecar"]
    assert seen["axes.linewidth"] == 1.25


def _probe_worker(_):
    """Runs in a worker: report what that process sees."""

    import matplotlib

    import scripts.analysis.envelope_combined as wec
    import scripts.analysis.envelope_exports as wee
    import scripts.analysis.envelope_visuals as wev
    from fbpipe import figure_export as wfe

    return {
        "axes.linewidth": matplotlib.rcParams["axes.linewidth"],
        "module": wec.__file__,
        "protocol": wev.get_protocol(),
        "runtime_settings": wec._RUNTIME_SETTINGS,
        "exports_settings": wee._RUNTIME_SETTINGS,
        "globals": {k: _copy(v) for k, v in wec._capture_driver_state()["globals"].items()},
        "svg_sidecar": wfe._INSTALLED,
    }


def test_every_setter_run_workflows_calls_is_carried_to_workers():
    """Static guard: a new ``set_*`` call in the driver must be added to
    ``_DRIVER_STATE`` or this fails -- a worker would silently miss it."""

    driver = (REPO / "scripts" / "pipeline" / "run_workflows.py").read_text(encoding="utf-8")
    called = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(ast.parse(driver))
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
        and (node.func.id if isinstance(node.func, ast.Name) else node.func.attr).startswith("set_")
    }
    covered = set(ec._DRIVER_STATE)
    checked = 0
    for module in (ev, ec, ee):
        for name, fn in vars(module).items():
            if name not in called or not isinstance(fn, types.FunctionType):
                continue
            if fn.__module__ != module.__name__:
                continue
            state = {
                n for n in fn.__code__.co_names
                if n in vars(module)
                and not callable(vars(module)[n])
                and not inspect.ismodule(vars(module)[n])
            }
            assert state, f"{module.__name__}.{name} sets no module state?"
            for n in state:
                assert (module.__name__, n) in covered, f"{module.__name__}.{n} ({name})"
            checked += 1
    assert checked >= 6  # protocol, light windows, odor remap, light checks, scores, settings


def test_a_failing_batch_raises_after_the_output_before_it(tmp_path):
    settings = _settings(tmp_path)
    _install(settings)
    serial, parallel = _twins(tmp_path)
    for root in (serial[0], parallel[0]):
        # mkdir(exist_ok=True) of the output dir raises when it is a file
        (root / "august_28_batch_3_rig_3" / "angle_distance_rms_envelope").write_text("x")

    buf_serial, buf_parallel = io.StringIO(), io.StringIO()
    with pytest.raises(FileExistsError):
        _run_serial(_plan(serial, settings), buf_serial)
    with pytest.raises(FileExistsError) as raised:
        _run_parallel(_plan(parallel, settings), 3, buf_parallel)

    assert _norm(buf_parallel.getvalue(), tmp_path / "parallel") == _norm(
        buf_serial.getvalue(), tmp_path / "serial"
    )
    # the batch before the failure was completed; the worker's traceback rides along
    assert (parallel[0] / "august_28_batch_1_rig_3" / "angle_distance_rms_envelope").is_dir()
    assert "_combine_fly_dir" in str(raised.value.__cause__)


def test_a_root_listed_twice_matches_serial(tmp_path):
    """The same batch dir twice in one plan must not run concurrently with
    itself; the second pass waits for the first, as it does serially."""

    settings = _settings(tmp_path)
    _install(settings)
    serial, parallel = _twins(tmp_path)

    log_serial = _run_serial(_plan(serial[:1] * 2, settings))
    log_parallel = _run_parallel(_plan(parallel[:1] * 2, settings), n_jobs=3)

    _same_trees(serial[:1], parallel[:1])
    assert _norm(log_parallel, tmp_path / "parallel") == _norm(log_serial, tmp_path / "serial")


class _RecordingPool:
    """Stands in for the worker pool: runs nothing, records the order in which
    batches are handed out and their results collected."""

    def __init__(self, events: list):
        self.events = events

    def submit(self, fn, cfg, fly_dir):
        self.events.append(("submit", fly_dir.name))
        events = self.events

        class _Future:
            def result(self):
                events.append(("result", fly_dir.name))
                return "", "", None

            def cancel(self):
                return True

        return _Future()


def test_nested_batch_dirs_never_run_at_the_same_time(tmp_path, monkeypatch):
    """A root that is itself listed as another root's batch (a parent dir in
    ``roots`` next to one of its datasets) gives two plan entries over the same
    files. The inner batches may start only once the outer one has finished,
    as serially; batches that share nothing are still handed out up front."""

    data = tmp_path / "data"
    hex_root = data / "Hex-Sensitivity-24-0.1"
    for batch in ("august_28_batch_1_rig_3", "august_28_batch_3_rig_3"):
        (hex_root / batch).mkdir(parents=True)
    (data / "IAA-Sensitivity-24-1" / "september_29_batch_1").mkdir(parents=True)
    plan = [ec.CombineConfig(root=data), ec.CombineConfig(root=hex_root)]

    events: list = []

    @contextlib.contextmanager
    def recording_executor(n_workers, state):
        yield _RecordingPool(events)

    monkeypatch.setattr(ec, "_combine_executor", recording_executor)
    with contextlib.redirect_stdout(io.StringIO()):
        ec.run_combine_plan(plan, n_jobs=4)

    first_result = events.index(("result", hex_root.name))
    # the two datasets share nothing: both are handed out before any result
    assert events[:2] == [("submit", hex_root.name), ("submit", "IAA-Sensitivity-24-1")]
    for batch in ("august_28_batch_1_rig_3", "august_28_batch_3_rig_3"):
        assert events.index(("submit", batch)) > first_result, events
    assert [e for e in events if e[0] == "result"] == [
        ("result", hex_root.name),
        ("result", "IAA-Sensitivity-24-1"),
        ("result", "august_28_batch_1_rig_3"),
        ("result", "august_28_batch_3_rig_3"),
    ]


def test_a_missing_root_raises_where_the_serial_run_does(tmp_path):
    settings = _settings(tmp_path)
    _install(settings)
    serial, parallel = _twins(tmp_path)

    def plan(roots, base):
        steps = _plan(roots, settings)
        steps.insert(2, ec.CombineConfig(root=base / "Not-On-Disk"))
        return steps

    buf_serial, buf_parallel = io.StringIO(), io.StringIO()
    with pytest.raises(FileNotFoundError):
        _run_serial(plan(serial, tmp_path / "serial"), buf_serial)
    with pytest.raises(FileNotFoundError):
        _run_parallel(plan(parallel, tmp_path / "parallel"), 3, buf_parallel)

    assert _norm(buf_parallel.getvalue(), tmp_path / "parallel") == _norm(
        buf_serial.getvalue(), tmp_path / "serial"
    )
    # nothing after the missing root ran
    assert not list(parallel[1].rglob("angle_distance_rms_envelope"))


# ── the gate in run_workflows._run_combined ───────────────────────────────


def _combine_block(roots: list[Path]) -> dict:
    return {"combine": {"roots": [str(r) for r in roots], "fps_default": 40.0, "window_sec": 0.25}}


@pytest.mark.parametrize(
    "enabled,n_jobs,expect_parallel",
    [(False, 4, False), (True, 1, False), (True, 3, True)],
)
def test_run_combined_gate(tmp_path, monkeypatch, enabled, n_jobs, expect_parallel):
    import scripts.pipeline.run_workflows as rw

    settings = _settings(tmp_path, n_jobs=n_jobs, enabled=enabled)
    _install(settings)
    roots = _datasets(tmp_path / "data")
    serial_calls: list = []
    plans: list = []
    monkeypatch.setattr(rw, "combine_distance_angle", lambda cfg: serial_calls.append(cfg.root))
    monkeypatch.setattr(
        rw, "run_combine_plan", lambda plan, n_jobs: plans.append((list(plan), n_jobs))
    )

    with contextlib.redirect_stdout(io.StringIO()):
        rw._run_combined(_combine_block(roots), settings, defer_secure_cleanup=True)

    if expect_parallel:
        assert serial_calls == []
        assert len(plans) == 1
        plan, jobs = plans[0]
        assert jobs == n_jobs
        assert [s for s in plan if isinstance(s, str)] == [
            f"[analysis] combined.combine → {r}" for r in roots
        ]
        configs = [s for s in plan if not isinstance(s, str)]
        assert [c.root for c in configs] == roots
        assert configs[0].skip_folders == frozenset({"august_29_batch_1_rig_3"})
    else:
        assert plans == []
        assert serial_calls == roots


def test_run_combined_parallel_matches_serial_end_to_end(tmp_path):
    import scripts.pipeline.run_workflows as rw

    serial, parallel = _twins(tmp_path)
    logs = {}
    for name, roots, enabled in (("serial", serial, False), ("parallel", parallel, True)):
        settings = _settings(tmp_path, enabled=enabled)
        _install(settings)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rw._run_combined(_combine_block(roots), settings, defer_secure_cleanup=True)
        logs[name] = _norm(buf.getvalue(), tmp_path / name)

    _same_trees(serial, parallel)
    assert logs["parallel"] == logs["serial"]
