"""A forced run does not hash a manifest nobody reads.

``_write_state`` blake2b-hashes every tracked table for the pipeline state
after each dataset (~43 s a night) and for the combined state after combine
(~52 s). Under ``force.pipeline`` / ``force.combined`` the matching check
returns before it ever loads that state, so the hashing was pure waste.

The forced write therefore builds no manifest -- and keeps no state at all,
rather than one with an empty or stale manifest. A later NON-force run then
finds no state and recomputes (a cache MISS), never a hit and never a crash.
An empty manifest would not be safe on its own: a dataset that has lost all of
its tracked tables compares {} against {} and would read as "unchanged".
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline import run_workflows as rw  # noqa: E402

from _loop_harness import make_dataset, run_main, write_config  # noqa: E402

BATCH = "august_13_batch_2_rig_3"
STEM = "output_august_13_batch_2_rig_3_testing_1_Hexanol_20260813_120000"


def _settings(tmp_path: Path):
    return SimpleNamespace(cache_dir=str(tmp_path / "cache"))


def _root(tmp_path: Path, name="EB-Training-24-1") -> Path:
    root = tmp_path / name
    trial = root / BATCH / "august_13_batch_2_rig_3_testing_1"
    trial.mkdir(parents=True)
    (root / BATCH / f"{STEM}.csv").write_text("frame,x\n0,1\n", encoding="utf-8")
    (trial / "fly1_distances.csv").write_text("frame,d\n0,1.5\n", encoding="utf-8")
    return root


EXPECTED = {"non_reactive_span_px": 20, "class2_min": 10.0, "class2_max": 160.0}


def _pipeline_hit(settings, root: Path) -> bool:
    return rw._should_skip_with_manifest(
        settings, category="pipeline", key=str(root), expected=dict(EXPECTED),
        force_flag=False, dataset_root=root,
    )


def _combined_expected(roots) -> dict:
    return {**EXPECTED, "dataset_roots": sorted(str(r) for r in roots)}


def _combined_hit(settings, roots) -> bool:
    return rw._should_skip_with_manifest(
        settings, category="combined", key="analysis",
        expected=_combined_expected(roots), force_flag=False,
    )


class _ManifestSpy:
    def __init__(self, monkeypatch):
        self.calls = []
        real = rw._build_file_manifest

        def _spy(root):
            self.calls.append(Path(root))
            return real(root)

        monkeypatch.setattr(rw, "_build_file_manifest", _spy)


# --- today's behaviour, pinned ------------------------------------------------

def test_unforced_pipeline_write_records_a_manifest_that_hits(tmp_path):
    settings, root = _settings(tmp_path), _root(tmp_path)
    rw._write_state(settings, "pipeline", str(root), dict(EXPECTED, version=rw.STATE_VERSION))
    state = rw._load_state(settings, "pipeline", str(root))
    assert len(state["file_manifest"]) == 2
    assert _pipeline_hit(settings, root) is True


def test_unforced_combined_write_records_a_manifest_that_hits(tmp_path):
    settings = _settings(tmp_path)
    roots = [_root(tmp_path, "A"), _root(tmp_path, "B")]
    rw._write_state(settings, "combined", "analysis", _combined_expected(roots))
    assert len(rw._load_state(settings, "combined", "analysis")["file_manifest"]) == 4
    assert _combined_hit(settings, roots) is True


# --- forced writes ------------------------------------------------------------

def test_forced_pipeline_write_builds_no_manifest_and_later_misses(tmp_path, monkeypatch):
    settings, root = _settings(tmp_path), _root(tmp_path)
    spy = _ManifestSpy(monkeypatch)
    rw._write_state(
        settings, "pipeline", str(root), dict(EXPECTED, version=rw.STATE_VERSION),
        build_manifest=False,
    )
    assert spy.calls == []
    assert not rw._state_path(settings, "pipeline", str(root)).exists()
    assert _pipeline_hit(settings, root) is False


def test_forced_write_still_logs_a_timestamped_line_per_dataset(tmp_path, caplog):
    # The nightly log's per-dataset marker used to be "Built file manifest for
    # <dataset>"; a forced write must leave an INFO line in its place.
    settings, root = _settings(tmp_path), _root(tmp_path)
    with caplog.at_level("INFO", logger=rw.LOGGER.name):
        rw._write_state(
            settings, "pipeline", str(root), dict(EXPECTED, version=rw.STATE_VERSION),
            build_manifest=False,
        )
    assert any(
        r.levelname == "INFO" and root.name in r.getMessage() and "no file manifest" in r.getMessage()
        for r in caplog.records
    )


def test_forced_pipeline_write_drops_an_older_state(tmp_path):
    """An old snapshot must not outlive a run that never verified it."""
    settings, root = _settings(tmp_path), _root(tmp_path)
    payload = dict(EXPECTED, version=rw.STATE_VERSION)
    rw._write_state(settings, "pipeline", str(root), payload)
    assert _pipeline_hit(settings, root) is True
    rw._write_state(settings, "pipeline", str(root), payload, build_manifest=False)
    assert rw._load_state(settings, "pipeline", str(root)) is None
    assert _pipeline_hit(settings, root) is False


def test_forced_write_never_hits_even_for_a_dataset_that_lost_every_table(tmp_path):
    # With an empty manifest written instead, {} vs {} would read "unchanged"
    # here and skip the dataset -- YOLO included -- on the next non-force run.
    settings, root = _settings(tmp_path), _root(tmp_path)
    payload = dict(EXPECTED, version=rw.STATE_VERSION)
    rw._write_state(settings, "pipeline", str(root), payload, build_manifest=False)
    for table in [*root.rglob("*.csv")]:
        table.unlink()
    assert _pipeline_hit(settings, root) is False


def test_forced_combined_write_builds_no_manifest_and_later_misses(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    roots = [_root(tmp_path, "A"), _root(tmp_path, "B")]
    rw._write_state(settings, "combined", "analysis", _combined_expected(roots))
    spy = _ManifestSpy(monkeypatch)
    rw._write_state(
        settings, "combined", "analysis", _combined_expected(roots), build_manifest=False
    )
    assert spy.calls == []
    assert rw._load_state(settings, "combined", "analysis") is None
    assert _combined_hit(settings, roots) is False


def test_categories_without_a_manifest_are_unaffected(tmp_path):
    settings = _settings(tmp_path)
    payload = {"wide_csv_mtime": 1.0, "version": rw.STATE_VERSION}
    rw._write_state(settings, "dataset_means", "analysis", payload, build_manifest=False)
    assert rw._load_state(settings, "dataset_means", "analysis") == payload


@pytest.mark.parametrize("bad", ["", "{not json", "null"])
def test_a_damaged_state_is_a_miss_not_a_crash(tmp_path, bad):
    settings, root = _settings(tmp_path), _root(tmp_path)
    path = rw._state_path(settings, "pipeline", str(root))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(bad, encoding="utf-8")
    assert _pipeline_hit(settings, root) is False


# --- wired into main() --------------------------------------------------------

@pytest.mark.parametrize(
    "force_pipeline,force_combined",
    [(True, True), (False, False), (True, False), (False, True)],
    ids=["both forced", "neither forced", "pipeline forced", "combined forced"],
)
def test_main_builds_manifests_only_when_not_forced(tmp_path, monkeypatch, force_pipeline, force_combined):
    # The mixed cases pin each write to ITS OWN force flag.
    roots = [
        make_dataset(tmp_path, name, [f"{BATCH}/{STEM}.csv", f"{BATCH}/t/fly1_distances.csv"])
        for name in ("Live-A", "Live-B")
    ]
    cfg = write_config(
        tmp_path, ["Live-A", "Live-B"], force_pipeline=force_pipeline, force_combined=force_combined
    )
    settings = SimpleNamespace(cache_dir=str(tmp_path / "cache"))

    calls = run_main(monkeypatch, cfg)

    assert len(calls["combined"]) == 1
    for root in roots:
        state = rw._load_state(settings, "pipeline", str(root))
        if force_pipeline:
            assert state is None
        else:
            assert len(state["file_manifest"]) == 2
    combined = rw._load_state(settings, "combined", "analysis")
    if force_combined:
        assert combined is None
    else:
        assert len(combined["file_manifest"]) == 4


def test_main_forced_never_hashes_in_the_dataset_loop(tmp_path, monkeypatch):
    for name in ("Live-A", "Live-B"):
        make_dataset(tmp_path, name, [f"{BATCH}/{STEM}.csv", f"{BATCH}/t/fly1_distances.csv"])
    cfg = write_config(tmp_path, ["Live-A", "Live-B"])
    spy = _ManifestSpy(monkeypatch)
    run_main(monkeypatch, cfg)
    assert spy.calls == []
