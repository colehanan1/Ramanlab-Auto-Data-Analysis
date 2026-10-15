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
