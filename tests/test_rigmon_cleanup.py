import os
import time
from datetime import datetime, timedelta
from pathlib import Path

from rigmon.cleanup import run_cleanup
from rigmon.config import HubConfig, RigConfig
from rigmon.store import Store

R = RigConfig("pi2", "Pi 2", "1.2.3.4")
NOW = datetime(2026, 10, 15, 0, 10).timestamp()
YESTERDAY = datetime(2026, 10, 14, 18, 0).timestamp()
TODAY = datetime(2026, 10, 15, 0, 5).timestamp()


def setup(tmp_path):
    cfg = HubConfig(rigs=(R,), staging_root=tmp_path / "staging", data_root=tmp_path / "Data",
                    db_path=tmp_path / "db.sqlite", qc_root=tmp_path / "qc")
    store = Store(cfg.db_path)
    for batch, ended in (("old", YESTERDAY), ("new", TODAY), ("cur", YESTERDAY), ("landing", YESTERDAY)):
        store.upsert_trial("pi2", batch, {"index": 1, "video": "t.h264", "ended_at": ended})
        store.set_trial_trace("pi2", batch, 1, trace="t")
        (cfg.qc_root / "pi2" / batch / "traces").mkdir(parents=True)
        (cfg.qc_root / "pi2" / batch / "traces" / "t.png").write_bytes(b"x")
        (cfg.qc_root / "pi2" / batch / "mount_check_1.json").write_text("{}")
        (cfg.staging_root / "pi2" / batch).mkdir(parents=True)
        (cfg.staging_root / "pi2" / batch / "t.mp4").write_bytes(b"x")
    store.set_session("pi2", "landing", "pulling")
    return cfg, store


def test_cleanup_deletes_old_and_keeps_today(tmp_path):
    cfg, store = setup(tmp_path)
    run_cleanup(cfg, store, {"pi2": "cur"}, NOW)
    assert not (cfg.qc_root / "pi2" / "old" / "traces").exists()
    assert (cfg.qc_root / "pi2" / "old" / "mount_check_1.json").exists()
    assert not (cfg.staging_root / "pi2" / "old").exists()
    assert (cfg.qc_root / "pi2" / "new" / "traces" / "t.png").exists()
    assert (cfg.staging_root / "pi2" / "new").exists()
    assert store.get_trial("pi2", "old", 1)["trace"] == "" and store.get_trial("pi2", "new", 1)["trace"] == "t"


def test_cleanup_keeps_current_and_landing(tmp_path):
    cfg, store = setup(tmp_path)
    run_cleanup(cfg, store, {"pi2": "cur"}, NOW)
    assert (cfg.qc_root / "pi2" / "cur" / "traces").exists() and (cfg.staging_root / "pi2" / "cur").exists()
    assert (cfg.staging_root / "pi2" / "landing").exists()


def test_cleanup_never_follows_symlinks(tmp_path):
    cfg, store = setup(tmp_path)
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "keep.txt").write_text("x")
    os.symlink(outside, cfg.staging_root / "pi2" / "evil")
    os.utime(outside, (YESTERDAY, YESTERDAY))
    (cfg.qc_root / "pi2" / "evil2").mkdir()
    os.symlink(outside, cfg.qc_root / "pi2" / "evil2" / "traces")
    run_cleanup(cfg, store, {}, NOW)
    assert (outside / "keep.txt").exists()


def test_cleanup_unknown_batch_uses_mtime(tmp_path):
    cfg, store = setup(tmp_path)
    d = cfg.staging_root / "pi2" / "orphan"
    d.mkdir()
    os.utime(d, (YESTERDAY, YESTERDAY))
    run_cleanup(cfg, store, {}, NOW)
    assert not d.exists()


def test_staging_kept_when_any_session_row_exists(tmp_path):
    cfg, store = setup(tmp_path)
    for b, st in (("old", "failed"), ("cur", "aborted"), ("new", "landed")):
        store.set_session("pi2", b, st)
    store.set_session("pi2", "pend", "pending")
    (cfg.staging_root / "pi2" / "pend").mkdir()
    os.utime(cfg.staging_root / "pi2" / "pend", (YESTERDAY, YESTERDAY))
    (cfg.staging_root / "pi2" / "nosess").mkdir()
    os.utime(cfg.staging_root / "pi2" / "nosess", (YESTERDAY, YESTERDAY))
    run_cleanup(cfg, store, {}, NOW)
    for b in ("old", "cur", "new", "pend", "landing"):
        assert (cfg.staging_root / "pi2" / b).exists(), b
    assert not (cfg.staging_root / "pi2" / "nosess").exists()


def test_skip_rigs_keeps_everything(tmp_path):
    cfg, store = setup(tmp_path)
    run_cleanup(cfg, store, {}, NOW, skip_rigs=frozenset({"pi2"}))
    assert (cfg.qc_root / "pi2" / "old" / "traces").exists()
    assert (cfg.staging_root / "pi2" / "old").exists()


def test_mtime_fallback_uses_newest_inside(tmp_path):
    cfg, store = setup(tmp_path)
    d = cfg.staging_root / "pi2" / "orphan"
    d.mkdir()
    f = d / "recent.mp4"
    f.write_bytes(b"x")
    os.utime(f, (TODAY, TODAY))
    os.utime(d, (YESTERDAY, YESTERDAY))
    run_cleanup(cfg, store, {}, NOW)
    assert d.exists()


def test_zero_ended_at_treated_as_unknown(tmp_path):
    cfg, store = setup(tmp_path)
    store.upsert_trial("pi2", "zero", {"index": 1, "video": "t.h264"})
    d = cfg.staging_root / "pi2" / "zero"
    d.mkdir()
    (d / "f").write_bytes(b"x")
    os.utime(d / "f", (TODAY, TODAY))
    os.utime(d, (TODAY, TODAY))
    run_cleanup(cfg, store, {}, NOW)
    assert d.exists()
