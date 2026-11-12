import importlib.util
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location(
    "pi_session_backup", Path(__file__).resolve().parents[1] / "scripts/dev/pi_session_backup.py")
psb = importlib.util.module_from_spec(_spec)
import sys
sys.modules["pi_session_backup"] = psb
_spec.loader.exec_module(psb)

NOW = 1_800_000_000.0
H = 3600.0


def test_week_tag():
    assert psb.week_tag(date(2026, 11, 9)) == "2026-W46-2026-11-09"   # Monday
    assert psb.week_tag(date(2026, 11, 12)) == "2026-W46-2026-11-09"  # Thursday
    assert psb.week_tag(date(2026, 11, 15)) == "2026-W46-2026-11-09"  # Sunday


def test_week_tag_year_boundary():
    assert psb.week_tag(date(2025, 12, 29)) == "2026-W01-2025-12-29"
    assert psb.week_tag(date(2026, 1, 1)) == "2026-W01-2025-12-29"
    assert psb.week_tag(date(2027, 1, 1)) == "2026-W53-2026-12-28"


def elig(batches, hub, cur=None):
    return psb.eligible_batches(batches, hub, cur, NOW)


def test_landed_ok():
    assert elig({"a": NOW - 2 * H}, {"a": "landed"}) == ["a"]


def test_current_batch_excluded_even_if_landed():
    assert elig({"a": NOW - 2 * H}, {"a": "landed"}, cur="a") == []


def test_recent_file_excluded():
    assert elig({"a": NOW - 10 * 60}, {"a": "landed"}) == []


def test_no_record_old_ok_and_young_excluded():
    assert elig({"old": NOW - 13 * H, "young": NOW - 11 * H}, {}) == ["old"]


def test_other_states_not_eligible():
    assert elig({"a": NOW - 20 * H, "b": NOW - 20 * H, "c": NOW - 20 * H},
                {"a": "aborted", "b": "pending", "c": "failed"}) == ["a"]
    # a recently aborted session waits out the idle window like an unknown one
    assert elig({"a": NOW - 2 * H}, {"a": "aborted"}) == []


# ── backup_batch with a fake runner ───────────────────────────────────────
PI = psb.Pi("flybehavior3", "Flybehavior3", "/pi/root")


class FakeRunner:
    def __init__(self, pi_md5, local_md5, rsync_rc=0):
        self.calls, self.pi_md5, self.local_md5, self.rsync_rc = [], pi_md5, local_md5, rsync_rc

    def __call__(self, argv, input=None):
        self.calls.append((argv, input))
        out = ""
        if argv[0] == "rsync":
            return SimpleNamespace(returncode=self.rsync_rc, stdout="", stderr="")
        if argv[0] == "ssh" and "md5sum" in argv[-1]:
            out = "".join(f"{h}  /pi/root/b1/{p}\n" for p, h in self.pi_md5.items())
        if argv[0] == "find":
            out = "".join(f"{h}  {argv[1]}/{p}\n" for p, h in self.local_md5.items())
        return SimpleNamespace(returncode=0, stdout=out, stderr="")

    def cmds(self, word):
        return [a for a, _ in self.calls if word in " ".join(a)]


def test_md5_mismatch_no_delete(tmp_path):
    r = FakeRunner({"x.csv": "a" * 32, "y.mp4": "b" * 32}, {"x.csv": "a" * 32, "y.mp4": "c" * 32})
    assert psb.backup_batch(PI, "b1", tmp_path, r) is False
    assert not r.cmds("rm -f") and not r.cmds("rmdir")


def test_missing_local_file_no_delete(tmp_path):
    r = FakeRunner({"x.csv": "a" * 32}, {})
    assert psb.backup_batch(PI, "b1", tmp_path, r) is False
    assert not r.cmds("rm -f")


def test_rsync_failure_no_delete(tmp_path):
    r = FakeRunner({"x.csv": "a" * 32}, {"x.csv": "a" * 32}, rsync_rc=23)
    assert psb.backup_batch(PI, "b1", tmp_path, r) is False
    assert not r.cmds("rm -f") and not r.cmds("md5sum")


def test_match_deletes_exactly_listed_files(tmp_path):
    r = FakeRunner({"x.csv": "a" * 32, "s/y.mp4": "b" * 32}, {"x.csv": "a" * 32, "s/y.mp4": "b" * 32})
    assert psb.backup_batch(PI, "b1", tmp_path, r) is True
    rm = [(a, i) for a, i in r.calls if "rm -f" in " ".join(a)]
    assert len(rm) == 1
    assert rm[0][1].splitlines() == ["/pi/root/b1/s/y.mp4", "/pi/root/b1/x.csv"]
    assert any("rmdir" in " ".join(a) for a, _ in r.calls)


def test_no_delete_flag_in_rsync_argv(tmp_path):
    r = FakeRunner({"x": "a" * 32}, {"x": "a" * 32})
    psb.backup_batch(PI, "b1", tmp_path, r)
    rs = [a for a, _ in r.calls if a[0] == "rsync"]
    assert rs and not any("delete" in arg for a in rs for arg in a[:5] + [a[5]])
    assert "--partial" in rs[0] and "-a" in rs[0]
    assert "delete" not in " ".join(psb.rsync_argv(PI, "b1", Path("/x/dest")))


def test_dry_run_issues_nothing(tmp_path):
    r = FakeRunner({}, {})
    assert psb.backup_batch(PI, "b1", tmp_path, r, dry_run=True) is True
    assert r.calls == []
    assert not (tmp_path / PI.name).exists()


# ── driver ────────────────────────────────────────────────────────────────
def hub_json(running):
    rigs = {}
    for rig, batch in running.items():
        rigs[rig] = {"status": {"run": {"state": "running" if batch else "done", "batch_dir": batch}}}
    return json.dumps({"rigs": rigs})


def listing_runner(lines):
    calls = []

    def runner(argv, input=None):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="\n".join(lines) + "\n", stderr="")
    runner.calls = calls
    return runner


def test_run_dry_skips_running_batch(tmp_path):
    lines = [f"{NOW - 5 * H}|1000|old_landed", f"{NOW - 60}|2000|today"]
    r = listing_runner(lines)
    sess = lambda rig: {"old_landed": "landed", "today": "landed"}  # noqa: E731
    rc = psb.run([PI], r, lambda url: hub_json({"Flybehavior3": "today"}), True, now=NOW,
                 dest_base=tmp_path, sessions_loader=sess)
    assert rc == 0
    assert all(a[0] == "ssh" for a in r.calls)   # listing only, no rsync/rm


def test_recording_pi_is_not_touched(tmp_path, monkeypatch):
    # rsync + md5 load on a Pi competes with its software trial encoder:
    # a Pi with a running session gets no backup at all, even of old batches.
    done = []
    monkeypatch.setattr(psb, "backup_batch", lambda pi, b, *a, **k: done.append(b) or True)
    r = listing_runner([f"{NOW - 5 * H}|1000|old_landed", f"{NOW - 60}|2000|today"])
    sess = lambda rig: {"old_landed": "landed"}  # noqa: E731
    rc = psb.run([PI], r, lambda url: hub_json({"Flybehavior3": "today"}), False, now=NOW,
                 dest_base=tmp_path, sessions_loader=sess)
    assert rc == 0
    assert done == []


def test_session_starting_mid_backup_stops_that_pi(tmp_path, monkeypatch):
    done = []
    monkeypatch.setattr(psb, "backup_batch", lambda pi, b, *a, **k: done.append(b) or True)
    r = listing_runner([f"{NOW - 5 * H}|1000|b1", f"{NOW - 5 * H}|1000|b2"])
    sess = lambda rig: {"b1": "landed", "b2": "landed"}  # noqa: E731
    calls = []

    def hub(url):   # idle at the start and before b1; a session starts after b1
        calls.append(url)
        return hub_json({"Flybehavior3": None if len(calls) <= 2 else "new_batch"})
    rc = psb.run([PI], r, hub, False, now=NOW, dest_base=tmp_path, sessions_loader=sess)
    assert rc == 0
    assert done == ["b1"]


def test_hub_unreachable_backs_up_nothing(tmp_path):
    def boom(url):
        raise OSError("down")
    r = listing_runner([])
    assert psb.run([PI], r, boom, False, now=NOW, dest_base=tmp_path) == 1
    assert r.calls == []


def test_failed_remux_leaves_pi_untouched(tmp_path, monkeypatch):
    """Remux runs before the Pi delete: a failure keeps the batch on the Pi so a rerun retries."""
    import pi_session_backup as m
    calls = []

    class R:
        def __init__(self, rc=0, out=""):
            self.returncode, self.stdout, self.stderr = rc, out, ""

    def runner(argv, input=None):
        calls.append(argv)
        if argv[0] == "ssh" and "md5sum" in argv[-1]:
            return R(0, "")
        return R(0, "")

    monkeypatch.setattr(m, "remux_continuous", lambda dest, runner: False)
    pi = m.PIS[0]
    assert m.backup_batch(pi, "b1", tmp_path, runner) is False
    assert not any(a[0] == "ssh" and ("rm -f" in a[-1] or "rmdir" in a[-1]) for a in calls)
