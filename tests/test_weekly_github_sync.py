"""Tests for scripts/dev/weekly_github_sync.py — the Saturday commit/merge/push job.

Every git test builds a throwaway bare "origin" plus a clone in tmp_path, so
nothing here touches the real repo or GitHub.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "weekly_github_sync",
    Path(__file__).resolve().parents[1] / "scripts/dev/weekly_github_sync.py",
)
wgs = importlib.util.module_from_spec(_spec)
sys.modules["weekly_github_sync"] = wgs
_spec.loader.exec_module(wgs)


# --------------------------------------------------------------------- helpers

def _git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A clone of a bare origin, on main, with one pushed commit."""
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(origin))
    work = tmp_path / "work"
    _git(tmp_path, "clone", str(origin), str(work))
    _git(work, "config", "user.name", "Test")
    _git(work, "config", "user.email", "test@example.com")
    _git(work, "checkout", "-b", "main")
    (work / ".gitignore").write_text(".env\nPiCode/\nconfig/*\nlogs/\n")
    (work / "README.md").write_text("hello\n")
    (work / "tests").mkdir()
    (work / "tests" / "test_a.py").write_text("def test_a(): pass\n")
    _git(work, "add", ".")
    _git(work, "commit", "-m", "init")
    _git(work, "push", "-u", "origin", "main")
    return work


def _commit(repo, name, text, msg=None, force=False):
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    _git(repo, "add", *(["-f"] if force else []), name)
    _git(repo, "commit", "-m", msg or f"add {name}")
    return _git(repo, "rev-parse", "HEAD")


def _second_clone_push(repo, tmp_path, name, text):
    """Push a commit to origin from another clone, so origin/main moves on."""
    other = tmp_path / "other"
    origin = _git(repo, "remote", "get-url", "origin")
    _git(tmp_path, "clone", "-b", "main", origin, str(other))
    _git(other, "config", "user.name", "Other")
    _git(other, "config", "user.email", "other@example.com")
    _commit(other, name, text)
    _git(other, "push", "origin", "main")


# ------------------------------------------------------------- secret values

def test_secret_values_come_from_secret_env_keys_and_picode_topics(repo):
    (repo / ".env").write_text(
        "INFLUXDB_HOST=10.0.0.5\n"
        "INFLUXDB_DB=homeassistant\n"
        "INFLUXDB_USER=homeassistant\n"
        "INFLUXDB_PASS=hunter2secret\n"
        "SOME_API_TOKEN='tok-abcdef123'\n"
    )
    (repo / "PiCode").mkdir()
    (repo / "PiCode" / "rig.py").write_text('NTFY_TOPIC = "private-rig-topic"\n')
    (repo / "PiCode" / "other.py").write_text(
        'URL = "https://ntfy.sh/another-private-topic"\n'
        'F = f"https://ntfy.sh/{NTFY_TOPIC}"\n'
    )

    values = wgs.load_secret_values(repo)

    assert "hunter2secret" in values
    assert "tok-abcdef123" in values  # quotes stripped
    assert "private-rig-topic" in values
    assert "another-private-topic" in values
    # Non-secret keys are not treated as secrets: "homeassistant" is in the public CLAUDE.md.
    assert "homeassistant" not in values
    assert "10.0.0.5" not in values
    # An f-string placeholder is not a topic.
    assert not any("{" in v for v in values)


def test_secret_values_missing_files_is_empty(repo):
    assert wgs.load_secret_values(repo) == set()


def test_values_already_public_on_origin_are_dropped(repo):
    # Already on GitHub: blocking every later push over it would protect nothing.
    _commit(repo, "notify.py", 'TOPIC = "ramanlab-pipeline"\n')
    _git(repo, "push", "origin", "main")
    kept = wgs.drop_public_values(repo, {"ramanlab-pipeline", "private-rig-topic"}, "origin/main")
    assert kept == {"private-rig-topic"}


# ------------------------------------------------------------------- scanning

def test_scan_flags_known_values_and_token_shapes_without_echoing_them():
    gh_token = "ghp_" + "A1b2C3d4" * 5
    text = "pw = hunter2secret\nsafe line\ntoken = " + gh_token + "\n"

    findings = wgs.scan_for_secrets(text, {"hunter2secret"})

    assert len(findings) == 2
    joined = "\n".join(findings)
    assert "hunter2secret" not in joined
    assert gh_token not in joined


@pytest.mark.parametrize(
    "line",
    [
        "-----BEGIN " + "OPENSSH PRIVATE KEY-----",  # split so this file is pushable
        "key = sk-ant-api03-" + "x" * 40,
        "aws = AKIA" + "ABCDEFGHIJKLMNOP",
        "pat = github_pat_" + "a" * 40,
    ],
)
def test_scan_flags_generic_token_patterns(line):
    assert wgs.scan_for_secrets(line, set())


def test_scan_ignores_clean_text():
    assert wgs.scan_for_secrets("topic = ramanlab-pipeline\nx = 1\n", {"hunter2secret"}) == []


# ----------------------------------------------------------------- push guard

def test_push_guard_passes_a_clean_fast_forward(repo):
    _commit(repo, "a.py", "x = 1\n")
    assert wgs.push_guard(repo, "main", "origin/main", {"hunter2secret"}) == []


def test_push_guard_refuses_when_origin_has_commits_we_lack(repo, tmp_path):
    _second_clone_push(repo, tmp_path, "theirs.py", "y = 2\n")
    _git(repo, "fetch", "origin")
    _commit(repo, "ours.py", "x = 1\n")

    problems = wgs.push_guard(repo, "main", "origin/main", set())

    assert any("fast-forward" in p for p in problems)


def test_push_guard_refuses_a_secret_added_in_a_new_commit(repo):
    _commit(repo, "a.py", "PASSWORD = 'hunter2secret'\n")
    problems = wgs.push_guard(repo, "main", "origin/main", {"hunter2secret"})
    assert any("secret" in p.lower() for p in problems)
    assert not any("hunter2secret" in p for p in problems)


def test_push_guard_ignores_a_secret_being_removed(repo):
    # Already public in pushed history: removing it must not block the push.
    _commit(repo, "a.py", "PASSWORD = 'hunter2secret'\n")
    _git(repo, "push", "origin", "main")
    _commit(repo, "a.py", "PASSWORD = None\n", msg="scrub")
    assert wgs.push_guard(repo, "main", "origin/main", {"hunter2secret"}) == []


def test_push_guard_refuses_an_oversized_file(repo):
    _commit(repo, "big.bin", "x" * 500)
    problems = wgs.push_guard(repo, "main", "origin/main", set(), max_bytes=100)
    assert any("big.bin" in p for p in problems)


@pytest.mark.parametrize("path", [
    "config/config_new.yaml", "PiCode/combinedv2_1.py", ".env", "sub/.env.local",
    "Results/fig.png", "logs/x.txt", ".remember/now.md", ".claude/settings.json",
])
def test_push_guard_refuses_local_only_paths(repo, path):
    _commit(repo, path, "x\n", force=True)
    problems = wgs.push_guard(repo, "main", "origin/main", set())
    assert any(path in p for p in problems)


@pytest.mark.parametrize("path", [
    "config/example.rigmon.yaml", "config/example.env", "config/environment.yml",
    "docs/superpowers/plans/x.md",  # ignored by docs/ but tracked on purpose
])
def test_push_guard_allows_deliberately_tracked_ignored_paths(repo, path):
    (repo / ".gitignore").write_text((repo / ".gitignore").read_text() + "docs/\n")
    _commit(repo, path, "x\n", force=True)
    assert wgs.push_guard(repo, "main", "origin/main", set()) == []


def test_new_branch_only_scans_its_own_unpushed_commits(repo):
    # A secret already on origin (in main) must not block a branch that merely contains it.
    _commit(repo, "old.py", "PASSWORD = 'hunter2secret'\n")
    _git(repo, "push", "origin", "main")
    _git(repo, "checkout", "-b", "feature/x")
    _commit(repo, "feat.py", "y = 2\n")
    _git(repo, "checkout", "main")

    commits = wgs.new_commits(repo, "feature/x", None)

    assert len(commits) == 1
    assert wgs.push_guard(repo, "feature/x", None, {"hunter2secret"}) == []


# ------------------------------------------------------------------ preflight

def test_preflight_passes_on_main(repo):
    assert wgs.preflight(repo) == []


def test_preflight_refuses_another_branch(repo):
    _git(repo, "checkout", "-b", "feature/y")
    assert any("main" in p for p in wgs.preflight(repo))


def test_preflight_refuses_a_merge_in_progress(repo):
    (repo / ".git" / "MERGE_HEAD").write_text("0" * 40 + "\n")
    assert any("merge" in p.lower() for p in wgs.preflight(repo))


# --------------------------------------------------------- pipeline detection

def test_pipeline_running_reads_proc_cmdlines(tmp_path):
    proc = tmp_path / "proc"
    (proc / "100").mkdir(parents=True)
    (proc / "100" / "cmdline").write_bytes(b"python3\0scripts/pipeline/run_workflows.py\0--config\0x\0")
    (proc / "200").mkdir()
    (proc / "200" / "cmdline").write_bytes(b"bash\0")
    (proc / "self").mkdir()

    assert wgs.pipeline_running(proc_root=proc, self_pid=999) is True
    assert wgs.pipeline_running(proc_root=proc, self_pid=100) is False


def test_pipeline_not_running(tmp_path):
    proc = tmp_path / "proc"
    (proc / "300").mkdir(parents=True)
    (proc / "300" / "cmdline").write_bytes(b"sleep\0" + b"10\0")
    assert wgs.pipeline_running(proc_root=proc, self_pid=1) is False


# ---------------------------------------------------------------------- junit

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest">
  <testcase classname="tests.test_a" name="test_ok" file="tests/test_a.py" line="1"/>
  <testcase classname="tests.test_a" name="test_bad" file="tests/test_a.py" line="5"><failure message="x"/></testcase>
  <testcase classname="tests.test_b.TestK" name="test_err[p1]" file="tests/test_b.py" line="9"><error message="e"/></testcase>
  <testcase classname="tests.test_b.TestK" name="test_skip" file="tests/test_b.py" line="12"><skipped message="s"/></testcase>
  <testcase classname="" name="tests.test_c"><error message="collection failure"/></testcase>
</testsuite></testsuites>
"""


def test_parse_junit_returns_all_and_failed_node_ids(tmp_path):
    xml = tmp_path / "j.xml"
    xml.write_text(JUNIT)

    seen, failed = wgs.parse_junit(xml)

    assert failed == {
        "tests/test_a.py::test_bad",
        "tests/test_b.py::TestK::test_err[p1]",
        "tests/test_c.py",
    }
    assert "tests/test_a.py::test_ok" in seen
    assert "tests/test_b.py::TestK::test_skip" in seen
    assert failed <= seen


def test_classify_failures():
    head = {"t::regressed", "t::always", "t::brand_new"}
    baseline_seen = {"t::regressed", "t::always", "t::other"}
    baseline_failed = {"t::always"}

    c = wgs.classify_failures(head, baseline_seen, baseline_failed)

    assert c["regressions"] == ["t::regressed"]
    assert c["preexisting"] == ["t::always"]
    assert c["new"] == ["t::brand_new"]


# ---------------------------------------------------------------- claude cmd

def test_claude_cmd_never_lets_claude_push_or_rewrite_history():
    cmd = wgs.build_claude_cmd("do it", allow_edits=True, claude_bin="claude")
    disallowed = cmd[cmd.index("--disallowedTools") + 1:]
    allowed = cmd[cmd.index("--allowedTools") + 1: cmd.index("--disallowedTools")]

    for banned in ("Bash(git push*)", "Bash(git reset*)", "Bash(git rebase*)",
                   "Bash(git add -f*)", "Bash(git add --force*)", "Bash(git commit --amend*)",
                   "Bash(git -C*)", "Bash(rm *)"):
        assert banned in disallowed
    assert not any("push" in a for a in allowed)
    assert cmd[:2] == ["claude", "-p"]
    # The repo's .claude/settings.local.json allows Bash(python3:*), Edit(*), Write(*)
    # and /tmp: loading it would let Claude script its way around every rule above.
    # "project" stays: it is what loads the repo's CLAUDE.md rules.
    assert cmd[cmd.index("--setting-sources") + 1] == "user,project"


def test_claude_edits_are_scoped_to_the_repo_minus_local_only_folders(tmp_path):
    cmd = wgs.build_claude_cmd("do it", allow_edits=True, claude_bin="claude", repo=tmp_path)
    allowed = cmd[cmd.index("--allowedTools") + 1: cmd.index("--disallowedTools")]
    disallowed = cmd[cmd.index("--disallowedTools") + 1:]
    root = "/" + str(tmp_path)  # "//abs/path" is Claude Code's absolute-path rule form

    # A bare "Edit"/"Write" allow rule would cover every path on the machine.
    assert "Edit" not in allowed and "Write" not in allowed
    assert f"Edit({root}/**)" in allowed and f"Write({root}/**)" in allowed
    for local in ("config", "PiCode", "Results"):
        assert f"Edit({root}/{local}/**)" in disallowed
        assert f"Write({root}/{local}/**)" in disallowed
    assert f"Edit({root}/.env)" in disallowed


def test_claude_cmd_without_edits_cannot_edit_files():
    cmd = wgs.build_claude_cmd("do it", allow_edits=False, claude_bin="claude")
    allowed = cmd[cmd.index("--allowedTools") + 1: cmd.index("--disallowedTools")]
    disallowed = cmd[cmd.index("--disallowedTools") + 1:]
    assert not any(a.startswith(("Edit", "Write")) for a in allowed)
    assert "Edit" in disallowed and "Write" in disallowed


# ------------------------------------------------------------------ end to end

class FakeTests:
    """Stands in for pytest: returns queued (seen, failed) results per call."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, cwd, files):
        self.calls.append((Path(cwd), files))
        if self.results:
            return self.results.pop(0)
        return set(), set()


def _committing_claude(repo):
    """A fake Claude that commits whatever is uncommitted."""
    prompts = []

    def run(prompt, allow_edits):
        prompts.append((prompt, allow_edits))
        if _git(repo, "status", "--porcelain"):
            _git(repo, "add", "-A")
            _git(repo, "commit", "-m", "feat: the week's work")
        return "committed the week's work"

    run.prompts = prompts
    return run


def _run(repo, **kw):
    kw.setdefault("test_runner", FakeTests())
    kw.setdefault("notifier", lambda title, msg, priority: None)
    kw.setdefault("busy", False)
    return wgs.run(repo, **kw)


def test_end_to_end_commits_and_pushes_main(repo):
    (repo / "new.py").write_text("x = 1\n")
    claude = _committing_claude(repo)

    report = _run(repo, claude_runner=claude)

    assert _git(repo, "rev-parse", "main") == _git(repo, "rev-parse", "origin/main")
    assert _git(repo, "log", "-1", "--format=%s", "origin/main") == "feat: the week's work"
    assert report.pushed_main is True
    assert report.ok


def test_dry_run_never_pushes(repo):
    (repo / "new.py").write_text("x = 1\n")
    before = _git(repo, "rev-parse", "origin/main")

    report = _run(repo, claude_runner=_committing_claude(repo), dry_run=True)

    _git(repo, "fetch", "origin")
    assert _git(repo, "rev-parse", "origin/main") == before
    assert report.pushed_main is False


def test_secret_in_the_weeks_work_blocks_the_push(repo):
    (repo / ".env").write_text("INFLUXDB_PASS=hunter2secret\n")
    (repo / "leak.py").write_text("PW = 'hunter2secret'\n")
    before = _git(repo, "rev-parse", "origin/main")

    report = _run(repo, claude_runner=_committing_claude(repo))

    _git(repo, "fetch", "origin")
    assert _git(repo, "rev-parse", "origin/main") == before
    assert report.pushed_main is False
    assert not report.ok
    assert "hunter2secret" not in report.to_markdown()


def test_nothing_to_do_skips_claude(repo):
    calls = []
    report = _run(repo, claude_runner=lambda p, allow_edits: calls.append(p) or "")
    assert calls == []
    assert report.pushed_main is False
    assert report.ok


def test_pipeline_running_means_no_file_edits(repo):
    (repo / "new.py").write_text("x = 1\n")
    claude = _committing_claude(repo)

    _run(repo, claude_runner=claude, busy=True)

    assert claude.prompts and all(allow is False for _, allow in claude.prompts)


def test_regressions_go_to_a_fix_phase(repo):
    (repo / "new.py").write_text("x = 1\n")
    fix_prompts = []

    def claude(prompt, allow_edits):
        if "REGRESSIONS" in prompt:
            fix_prompts.append(prompt)
            _commit(repo, "fix.py", "fixed = True\n", msg="fix: restore behaviour")
            return "fixed"
        _git(repo, "add", "-A")
        _git(repo, "commit", "-m", "feat: work")
        return "committed"

    tests = FakeTests(
        ({"tests/test_a.py::a", "tests/test_a.py::b"}, {"tests/test_a.py::a"}),   # HEAD: t::a fails
        ({"tests/test_a.py::a", "tests/test_a.py::b"}, set()),      # baseline: t::a passed -> regression
        ({"tests/test_a.py::a", "tests/test_a.py::b"}, set()),      # after fix: all pass
    )

    report = _run(repo, claude_runner=claude, test_runner=tests)

    assert len(fix_prompts) == 1 and "tests/test_a.py::a" in fix_prompts[0]
    assert report.fixed == ["tests/test_a.py::a"]
    assert report.still_failing == []
    assert _git(repo, "log", "-1", "--format=%s", "origin/main") == "fix: restore behaviour"


def test_a_fix_that_breaks_other_tests_is_reverted(repo):
    (repo / "new.py").write_text("x = 1\n")

    def claude(prompt, allow_edits):
        if "REGRESSIONS" in prompt:
            _commit(repo, "fix.py", "bad = True\n", msg="fix: attempt")
            return "tried"
        _git(repo, "add", "-A")
        _git(repo, "commit", "-m", "feat: work")
        return "committed"

    tests = FakeTests(
        ({"tests/test_a.py::a", "tests/test_a.py::b"}, {"tests/test_a.py::a"}),   # HEAD
        ({"tests/test_a.py::a", "tests/test_a.py::b"}, set()),      # baseline
        ({"tests/test_a.py::a", "tests/test_a.py::b"}, {"tests/test_a.py::b"}),   # after fix: t::b now fails
    )

    report = _run(repo, claude_runner=claude, test_runner=tests)

    assert not (repo / "fix.py").exists()
    assert report.fix_reverted is True
    assert report.still_failing == ["tests/test_a.py::a"]
    # Still pushed: the job backs the week up even when a fix does not work out.
    assert report.pushed_main is True


def test_preexisting_failures_do_not_trigger_a_fix(repo):
    (repo / "new.py").write_text("x = 1\n")
    claude = _committing_claude(repo)
    tests = FakeTests(({"tests/test_a.py::a"}, {"tests/test_a.py::a"}), ({"tests/test_a.py::a"}, {"tests/test_a.py::a"}))

    report = _run(repo, claude_runner=claude, test_runner=tests)

    assert all("REGRESSIONS" not in p for p, _ in claude.prompts)
    assert report.preexisting == ["tests/test_a.py::a"]


def test_baseline_tests_run_in_a_worktree_of_the_last_pushed_commit(repo):
    pushed = _git(repo, "rev-parse", "origin/main")
    (repo / "new.py").write_text("x = 1\n")
    tests = FakeTests(({"tests/test_a.py::t", "tests/test_new.py::n"},
                       {"tests/test_a.py::t", "tests/test_new.py::n"}), (set(), set()))

    _run(repo, claude_runner=_committing_claude(repo), test_runner=tests)

    baseline_cwd, files = tests.calls[1]
    assert baseline_cwd != repo
    # Only files that existed at the last push; tests/test_new.py is new this week.
    assert files == ["tests/test_a.py"]
    # Worktree was removed afterwards.
    assert str(baseline_cwd) not in _git(repo, "worktree", "list")
    assert pushed  # sanity


def test_branch_backups_push_new_and_ff_branches_but_skip_diverged(repo, tmp_path):
    _git(repo, "checkout", "-b", "feature/new")
    _commit(repo, "n.py", "n = 1\n")
    _git(repo, "checkout", "main")

    _git(repo, "checkout", "-b", "feature/diverged")
    _commit(repo, "d.py", "d = 1\n")
    _git(repo, "push", "origin", "feature/diverged")
    _git(repo, "reset", "--hard", "HEAD~1")
    _commit(repo, "d2.py", "d = 2\n")
    _git(repo, "checkout", "main")

    report = _run(repo, claude_runner=_committing_claude(repo))

    remote = _git(repo, "ls-remote", "--heads", "origin")
    assert "refs/heads/feature/new" in remote
    assert _git(repo, "rev-parse", "feature/new") == _git(repo, "rev-parse", "origin/feature/new")
    assert "feature/new" in report.branches_pushed
    assert any(b.startswith("feature/diverged") for b in report.branches_skipped)


def test_report_counts_every_commit_going_to_github(repo):
    # Two commits made by hand earlier in the week plus the one Claude makes now.
    _commit(repo, "a.py", "a = 1\n")
    _commit(repo, "b.py", "b = 1\n")
    (repo / "c.py").write_text("c = 1\n")

    report = _run(repo, claude_runner=_committing_claude(repo))

    assert len(report.commits) == 3
    assert report.headline().startswith("3 commit(s)")


def test_leftovers_keep_porcelain_columns(repo):
    (repo / "README.md").write_text("changed\n")  # " M README.md": leading space matters
    report = _run(repo, claude_runner=lambda p, allow_edits: "left it")
    assert report.leftovers == [" M README.md"]


def test_a_failed_claude_run_is_an_error_but_committed_work_still_goes_up(repo):
    _commit(repo, "done.py", "d = 1\n")          # committed by hand earlier in the week
    (repo / "wip.py").write_text("w = 1\n")      # left for Claude, which then fails

    def broken(prompt, allow_edits):
        raise wgs.ClaudeError("claude exited 1: sign-in expired")

    report = _run(repo, claude_runner=broken)

    assert any("sign-in expired" in e for e in report.errors)
    assert not report.ok
    assert report.pushed_main is True
    assert (repo / "wip.py").exists() and "?? wip.py" in report.leftovers


def test_report_is_written_to_logs(repo):
    report = _run(repo, claude_runner=_committing_claude(repo))
    path = report.write(repo / "logs" / "weekly_github_sync")
    assert path.exists()
    assert "# Weekly GitHub sync" in path.read_text()


def test_a_secret_typed_into_a_merge_resolution_is_caught(repo, tmp_path):
    _second_clone_push(repo, tmp_path, "shared.py", "x = 'theirs'\n")
    _git(repo, "fetch", "origin")
    _commit(repo, "shared.py", "x = 'ours'\n")
    subprocess.run(["git", "merge", "origin/main"], cwd=repo, capture_output=True)
    (repo / "shared.py").write_text("x = 'hunter2secret'\n")
    _git(repo, "add", "shared.py")
    _git(repo, "commit", "--no-edit")

    problems = wgs.push_guard(repo, "main", "origin/main", {"hunter2secret"})

    assert any("shared.py" in p and "secret" in p for p in problems)


def test_busy_commit_prompt_forbids_the_merge_and_fix_prompt_lists_regressions():
    busy = wgs.commit_prompt(busy=True, origin_sha="a" * 40, today="2026-11-21")
    idle = wgs.commit_prompt(busy=False, origin_sha="a" * 40, today="2026-11-21")
    fix = wgs.fix_prompt(regressions=["tests/test_a.py::a"], baseline_sha="b" * 40, today="2026-11-21")

    assert "must NOT merge" in busy and "git merge --no-edit origin/main" not in busy
    assert "git merge --no-edit origin/main" in idle
    assert "REGRESSIONS" in fix and "tests/test_a.py::a" in fix and "b" * 40 in fix
    for prompt in (busy, idle, fix):
        assert "$" not in prompt.replace("$(", "")  # every placeholder filled
        assert wgs.CO_AUTHOR in prompt


def test_claude_gets_no_exec_capable_helpers_and_cannot_touch_dot_git(tmp_path):
    cmd = wgs.build_claude_cmd("do it", allow_edits=True, claude_bin="claude", repo=tmp_path)
    allowed = cmd[cmd.index("--allowedTools") + 1: cmd.index("--disallowedTools")]
    disallowed = cmd[cmd.index("--disallowedTools") + 1:]
    root = "/" + str(tmp_path)
    # find -exec / sort -o would run or write anything, around every other rule.
    assert not any(a.startswith(("Bash(find", "Bash(sort")) for a in allowed)
    # A planted hook would run inside the driver's own git commands.
    assert f"Edit({root}/.git/**)" in disallowed and f"Write({root}/.git/**)" in disallowed


def test_claude_env_has_no_way_to_reach_github(monkeypatch):
    monkeypatch.setenv("SSH_AUTH_SOCK", "/run/agent.sock")
    monkeypatch.setenv("GH_TOKEN", "x")
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    env = wgs.claude_env()
    assert env["GIT_SSH_COMMAND"] == "/bin/false"
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    for var in ("SSH_AUTH_SOCK", "GH_TOKEN", "GITHUB_TOKEN"):
        assert var not in env
    assert env["FBPIPE_DISABLE_NTFY"] == "1"


def test_driver_push_ignores_a_planted_hook(repo):
    hook = repo / ".git" / "hooks" / "pre-push"
    hook.write_text("#!/bin/sh\necho planted >&2\nexit 1\n")
    hook.chmod(0o755)
    (repo / "new.py").write_text("x = 1\n")

    report = _run(repo, claude_runner=_committing_claude(repo))

    assert report.pushed_main is True
