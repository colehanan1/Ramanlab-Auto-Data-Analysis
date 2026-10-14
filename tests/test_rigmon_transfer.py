from pathlib import Path

from rigmon.config import HubConfig, RigConfig
from rigmon.store import Store
from rigmon.transfer import (TransferWorkers, TrialTransfer, remux_argv,
                             rsync_pull_argv)

PI = RigConfig("flybehavior2", "Pi 2", "100.93.23.70")
LOCAL = RigConfig("fake", "Fake", "local", 1)


def _cfg(tmp_path):
    return HubConfig(rigs=(PI, LOCAL), staging_root=tmp_path / "staging",
                     data_root=tmp_path / "Data", db_path=tmp_path / "db.sqlite")


class FakeRunner:
    """Pretends to be rsync (copies from a local 'Pi' dir) and the remux script."""

    def __init__(self, pi_dir: Path, remux_ok=True, rsync_fail=0):
        self.pi_dir, self.remux_ok, self.rsync_fail = pi_dir, remux_ok, rsync_fail
        self.calls = []

    def __call__(self, argv, timeout, env=None):
        self.calls.append(argv)
        if argv[0] == "rsync":
            if self.rsync_fail:
                self.rsync_fail -= 1
                return 23, "rsync: connection reset"
            src, dest = Path(argv[-2].split(":", 1)[-1]), Path(argv[-1])
            (dest / src.name).write_bytes(src.read_bytes())
            return 0, ""
        h264 = Path(argv[argv.index("--input-dir") + 1]) / argv[argv.index("--only") + 1]
        csv = h264.with_suffix(".csv")
        if not self.remux_ok or not csv.exists() or csv.stat().st_size == 0:
            return 1, "[SKIP] no csv"
        h264.with_suffix(".mp4").write_bytes(b"mp4" * 400)
        h264.unlink()
        return 0, "ok"


def _pi_trial(pi_dir, stem="t1", csv=b"SensorNs\n1\n"):
    pi_dir.mkdir(parents=True, exist_ok=True)
    (pi_dir / f"{stem}.h264").write_bytes(b"h" * 50)
    (pi_dir / f"{stem}.csv").write_bytes(csv)
    return {"index": 1, "label": "Training 1/6", "video": f"{stem}.h264",
            "frames_csv": f"{stem}.csv", "video_bytes": 50, "csv_bytes": len(csv),
            "ended_at": 1.0, "closed": True}


def test_rsync_argv_is_additive_and_uses_the_key(tmp_path):
    argv = rsync_pull_argv(PI, Path("/k/pi_backup"), "/pi/b/t1.h264", tmp_path)
    assert argv[:3] == ["rsync", "-a", "--partial"]
    assert not any("delete" in a for a in argv)
    assert "ssh -i /k/pi_backup -o BatchMode=yes -o ConnectTimeout=10" in argv
    assert argv[-2:] == ["ramanlab@100.93.23.70:/pi/b/t1.h264", f"{tmp_path}/"]


def test_local_rig_rsyncs_a_plain_path(tmp_path):
    argv = rsync_pull_argv(LOCAL, Path("/k"), "/pi/b/t1.h264", tmp_path)
    assert "-e" not in argv and argv[-2] == "/pi/b/t1.h264"


def test_remux_argv_matches_the_pi_recipe(tmp_path):
    argv = remux_argv(_cfg(tmp_path), tmp_path / "x" / "t1.h264")
    assert argv[:4] == ["nice", "-n", "10", "/usr/bin/python3"]
    assert argv[5:9] == ["--input-dir", str(tmp_path / "x"), "--only", "t1.h264"]
    assert argv[9:] == ["--crf", "22", "--preset", "slow", "--faststart",
                        "--scale", "1.0", "--fps-cap", "50.0"]


def test_process_pulls_verifies_and_remuxes(tmp_path):
    cfg, pi_dir = _cfg(tmp_path), tmp_path / "pi" / "b"
    store, trial = Store(cfg.db_path), _pi_trial(pi_dir)
    tt = TrialTransfer(cfg, store, runner=FakeRunner(pi_dir), sleep=lambda s: None)
    assert tt.process(LOCAL, "b", str(pi_dir), trial) == "done"
    staged = tt.staging_dir("fake", "b")
    assert (staged / "t1.mp4").exists() and not (staged / "t1.h264").exists()
    assert store.get_trial("fake", "b", 1)["mp4"] == "t1.mp4"


def test_done_trial_is_not_pulled_again(tmp_path):
    cfg, pi_dir = _cfg(tmp_path), tmp_path / "pi" / "b"
    store, trial = Store(cfg.db_path), _pi_trial(pi_dir)
    runner = FakeRunner(pi_dir)
    tt = TrialTransfer(cfg, store, runner=runner, sleep=lambda s: None)
    tt.process(LOCAL, "b", str(pi_dir), trial)
    n = len(runner.calls)
    assert tt.process(LOCAL, "b", str(pi_dir), trial) == "done"
    assert len(runner.calls) == n


def test_transient_rsync_failure_is_retried(tmp_path):
    cfg, pi_dir = _cfg(tmp_path), tmp_path / "pi" / "b"
    store, trial = Store(cfg.db_path), _pi_trial(pi_dir)
    sleeps = []
    tt = TrialTransfer(cfg, store, runner=FakeRunner(pi_dir, rsync_fail=2), sleep=sleeps.append)
    assert tt.process(LOCAL, "b", str(pi_dir), trial) == "done"
    assert sleeps == [5.0, 15.0]


def test_missing_csv_fails_the_trial_with_a_reason(tmp_path):
    cfg, pi_dir = _cfg(tmp_path), tmp_path / "pi" / "b"
    store, trial = Store(cfg.db_path), _pi_trial(pi_dir, csv=b"")
    tt = TrialTransfer(cfg, store, runner=FakeRunner(pi_dir), sleep=lambda s: None)
    assert tt.process(LOCAL, "b", str(pi_dir), trial) == "failed"
    row = store.get_trial("fake", "b", 1)
    assert row["state"] == "failed" and "remux t1.h264 failed" in row["error"]


def test_size_mismatch_fails(tmp_path):
    cfg, pi_dir = _cfg(tmp_path), tmp_path / "pi" / "b"
    store, trial = Store(cfg.db_path), {**_pi_trial(pi_dir), "video_bytes": 999}
    tt = TrialTransfer(cfg, store, runner=FakeRunner(pi_dir), sleep=lambda s: None)
    assert tt.process(LOCAL, "b", str(pi_dir), trial) == "failed"
    assert "Pi reported 999" in store.get_trial("fake", "b", 1)["error"]


def test_unclosed_trial_is_left_for_finalize(tmp_path):
    cfg, pi_dir = _cfg(tmp_path), tmp_path / "pi" / "b"
    store, trial = Store(cfg.db_path), {**_pi_trial(pi_dir), "closed": False}
    runner = FakeRunner(pi_dir)
    tt = TrialTransfer(cfg, store, runner=runner, sleep=lambda s: None)
    assert tt.process(LOCAL, "b", str(pi_dir), trial) == "waiting"
    assert runner.calls == []


def test_workers_dedupe_and_run_in_order():
    w, seen = TransferWorkers(threaded=False), []
    assert w.submit("pi2", ("t", 1), lambda: seen.append(1))
    assert not w.submit("pi2", ("t", 1), lambda: seen.append("dup"))
    assert w.submit("pi2", ("t", 2), lambda: seen.append(2))
    w.drain()
    assert seen == [1, 2]
    assert w.submit("pi2", ("t", 1), lambda: seen.append(3))  # key free again


def test_worker_exception_does_not_kill_the_queue():
    w, seen = TransferWorkers(threaded=False), []
    w.submit("pi2", "a", lambda: 1 / 0)
    w.submit("pi2", "b", lambda: seen.append("b"))
    w.drain()
    assert seen == ["b"]
