import threading
from pathlib import Path

import pytest

from rigmon.config import HubConfig, RigConfig
from rigmon.finalize import FinalizeError, Finalizer, dest_from_push_to, outstanding
from rigmon.store import Store
from rigmon.transfer import TrialTransfer
from test_rigmon_transfer import FakeRunner  # tests/ is on sys.path (conftest)

LOCAL = RigConfig("fake", "Fake", "local", 1)


def _setup(tmp_path, free_gb=500.0):
    cfg = HubConfig(rigs=(LOCAL,), staging_root=tmp_path / "staging",
                    data_root=tmp_path / "Data", db_path=tmp_path / "db.sqlite")
    pi = tmp_path / "pi" / "b"
    pi.mkdir(parents=True)
    for stem in ("t1", "t2"):
        (pi / f"{stem}.h264").write_bytes(b"h" * 50)
        (pi / f"{stem}.csv").write_bytes(b"SensorNs\n1\n")
    (pi / "session.txt").write_text("fly 3")
    manifest = [{"path": p.name, "bytes": p.stat().st_size} for p in sorted(pi.iterdir())]
    store = Store(cfg.db_path)
    tt = TrialTransfer(cfg, store, runner=FakeRunner(pi), sleep=lambda s: None)
    alerts = []
    fin = Finalizer(cfg, store, tt, disk_free=lambda p: free_gb * 1e9,
                    alert=lambda t, b: alerts.append(t))
    push_to = f"ramanlab@100.94.160.60:{tmp_path}/Data/flys_New/Cohort/"
    return cfg, pi, manifest, store, tt, fin, push_to, alerts


def test_dest_from_push_to(tmp_path):
    assert dest_from_push_to(f"u@h:{tmp_path}/Data/flys_New/C/", "b", tmp_path / "Data") == \
        (tmp_path / "Data" / "flys_New" / "C" / "b")


def test_dest_outside_data_root_is_refused(tmp_path):
    with pytest.raises(FinalizeError):
        dest_from_push_to("u@h:/etc/", "b", tmp_path / "Data")
    with pytest.raises(FinalizeError):
        dest_from_push_to(f"u@h:{tmp_path}/Data/x/", "../../etc", tmp_path / "Data")
    with pytest.raises(FinalizeError):
        dest_from_push_to("", "b", tmp_path / "Data")


def test_outstanding_counts_h264_as_present_once_mp4_exists(tmp_path):
    (tmp_path / "t1.mp4").write_bytes(b"m")
    (tmp_path / "t1.csv").write_bytes(b"12")
    manifest = [{"path": "t1.h264", "bytes": 50}, {"path": "t1.csv", "bytes": 2},
                {"path": "t2.csv", "bytes": 2}]
    assert outstanding(manifest, tmp_path) == ["t2.csv"]


def test_finalize_lands_the_whole_session(tmp_path):
    cfg, pi, manifest, store, tt, fin, push_to, _ = _setup(tmp_path)
    tt.process(LOCAL, "b", str(pi), {"index": 1, "video": "t1.h264", "frames_csv": "t1.csv",
                                     "video_bytes": 50, "csv_bytes": 11, "closed": True})
    assert fin.run(LOCAL, "b", str(pi), push_to, manifest) == "landed"
    dest = tmp_path / "Data" / "flys_New" / "Cohort" / "b"
    assert sorted(p.name for p in dest.iterdir()) == \
        ["session.txt", "t1.csv", "t1.mp4", "t2.csv", "t2.mp4"]
    assert not tt.staging_dir("fake", "b").exists()
    assert store.get_session("fake", "b") ["state"] == "landed"


def test_existing_destination_is_never_merged(tmp_path):
    cfg, pi, manifest, store, tt, fin, push_to, alerts = _setup(tmp_path)
    dest = tmp_path / "Data" / "flys_New" / "Cohort" / "b"
    dest.mkdir(parents=True)
    (dest / "legacy.mp4").write_bytes(b"x")
    assert fin.run(LOCAL, "b", str(pi), push_to, manifest) == "failed"
    assert sorted(p.name for p in dest.iterdir()) == ["legacy.mp4"]
    assert tt.staging_dir("fake", "b").exists()
    assert "already exists" in store.get_session("fake", "b")["error"]
    assert alerts == ["Fake finalize failed"]


def test_abort_wins(tmp_path):
    cfg, pi, manifest, store, tt, fin, push_to, _ = _setup(tmp_path)
    assert fin.abort("fake", "b") == "aborted"
    assert fin.run(LOCAL, "b", str(pi), push_to, manifest) == "aborted"
    assert not (tmp_path / "Data" / "flys_New").exists()


def test_abort_after_landing_reports_landed(tmp_path):
    cfg, pi, manifest, store, tt, fin, push_to, _ = _setup(tmp_path)
    fin.run(LOCAL, "b", str(pi), push_to, manifest)
    assert fin.abort("fake", "b") == "landed"


def test_low_disk_is_refused(tmp_path):
    cfg, pi, manifest, store, tt, fin, push_to, _ = _setup(tmp_path, free_gb=5.0)
    assert fin.run(LOCAL, "b", str(pi), push_to, manifest) == "failed"
    assert "free" in store.get_session("fake", "b")["error"]


def test_manifest_path_escape_is_refused(tmp_path):
    cfg, pi, manifest, store, tt, fin, push_to, _ = _setup(tmp_path)
    bad = manifest + [{"path": "../../outside.txt", "bytes": 1}]
    assert fin.run(LOCAL, "b", str(pi), push_to, bad) == "failed"


def test_landed_session_is_not_rerun(tmp_path):
    cfg, pi, manifest, store, tt, fin, push_to, alerts = _setup(tmp_path)
    assert fin.run(LOCAL, "b", str(pi), push_to, manifest) == "landed"
    dest = tmp_path / "Data" / "flys_New" / "Cohort" / "b"
    before = sorted(p.name for p in dest.iterdir())
    assert fin.run(LOCAL, "b", str(pi), push_to, manifest) == "landed"
    assert sorted(p.name for p in dest.iterdir()) == before
    assert not tt.staging_dir("fake", "b").exists()
    assert store.get_session("fake", "b")["state"] == "landed"
    assert alerts == []


def test_abort_shares_the_move_lock(tmp_path):
    cfg, pi, manifest, store, tt, fin, push_to, _ = _setup(tmp_path)
    result = []
    with fin._move_lock:
        th = threading.Thread(target=lambda: result.append(fin.abort("fake", "b")))
        th.start()
        th.join(0.3)
        assert th.is_alive()
    th.join(5)
    assert result == ["aborted"]


def test_relative_push_to_is_refused(tmp_path):
    for bad in ("u@h:relative/dir/", "u@h:~/x/"):
        with pytest.raises(FinalizeError):
            dest_from_push_to(bad, "b", tmp_path / "Data")


def test_destination_inside_staging_is_refused(tmp_path):
    cfg, pi, manifest, store, tt, fin, push_to, alerts = _setup(tmp_path)
    cfg2 = HubConfig(rigs=(LOCAL,), staging_root=tmp_path / "Data" / "stage",
                     data_root=tmp_path / "Data", db_path=tmp_path / "db2.sqlite")
    tt2 = TrialTransfer(cfg2, store, runner=FakeRunner(pi), sleep=lambda s: None)
    fin2 = Finalizer(cfg2, store, tt2, disk_free=lambda p: 500e9,
                     alert=lambda t, b: alerts.append(t))
    bad = f"u@h:{tmp_path}/Data/stage/fake/"
    assert fin2.run(LOCAL, "b", str(pi), bad, manifest) == "failed"
    assert "staging" in store.get_session("fake", "b")["error"]


def test_malformed_manifest_item_fails_not_sticks(tmp_path):
    cfg, pi, manifest, store, tt, fin, push_to, alerts = _setup(tmp_path)
    bad = manifest + [{"path": "x.txt"}]
    assert fin.run(LOCAL, "b", str(pi), push_to, bad) == "failed"
    assert store.get_session("fake", "b")["state"] == "failed"
    assert alerts == ["Fake finalize failed"]
