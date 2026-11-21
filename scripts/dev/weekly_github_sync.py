#!/usr/bin/env python3
"""Weekly commit / merge / test / push of this repo's ``main`` to GitHub.

Run by the ``weekly-github-sync.timer`` systemd user unit every Saturday at
12:00 (``scripts/dev/systemd/``). Scope is ``main`` only: other local branches
are pushed to GitHub as backups but never merged.

Phases
------
1. Preflight: the repo must be on ``main`` with no merge/rebase in progress.
2. Commit (Claude Code, headless): commit the week's uncommitted work in
   logical commits and merge ``origin/main``. Claude cannot push, reset,
   rebase, switch branches or force-add ignored files.
3. Test (here): run the full suite. Any failure is re-run at the last pushed
   commit in a throwaway worktree, splitting failures into regressions (passed
   at the last push, fail now), pre-existing and new tests.
4. Fix (Claude Code): only for regressions, only when no pipeline run is
   active. If the fix makes any other test fail, its commits are reverted.
5. Push (here, never Claude): ``main`` goes up only if it is a fast-forward
   of ``origin/main``, adds no known secret value or token-shaped string, no
   file over 10 MB and no gitignored path. A week with an unfixed regression
   is still pushed: the report says so.
6. Back up every other local branch the same way (diverged ones are skipped).
7. Write ``logs/weekly_github_sync/<date>.md`` and send one ntfy summary.

While ``run_workflows.py`` is running, phase 2 runs without file edits (no
merge, no conflict resolution) and phase 4 is skipped, so the running
pipeline never sees its code change underneath it.

Usage::

    python3 scripts/dev/weekly_github_sync.py              # the real thing
    python3 scripts/dev/weekly_github_sync.py --dry-run    # no Claude, no push
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fcntl
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from string import Template
from typing import Callable, Iterable, Iterator

REPO = Path(__file__).resolve().parents[2]
PROMPT_DIR = Path(__file__).resolve().parent / "weekly_github_sync_prompts"
REPORT_DIR = REPO / "logs" / "weekly_github_sync"

MAIN = "main"
REMOTE = "origin"
MAX_FILE_BYTES = 10 * 1024 * 1024
MIN_SECRET_LEN = 6
CLAUDE_TIMEOUT_S = 2 * 60 * 60
TEST_TIMEOUT_S = 3 * 60 * 60
NTFY_DEFAULT_TOPIC = "ramanlab-pipeline"  # already public in src/fbpipe/utils/notify.py
# Gitignored paths the tests may read; symlinked into the baseline worktree.
BASELINE_LINKS = ("config", "PiCode", ".env")
# Top-level folders that never go to GitHub, even force-added.
LOCAL_ONLY_DIRS = {"PiCode", "Results", "logs", ".remember", ".claude"}

SECRET_ENV_KEY = re.compile(r"PASS|PWD|TOKEN|SECRET|KEY|CREDENTIAL", re.I)
TOPIC_ASSIGN = re.compile(r"""NTFY_TOPIC\s*[:=]\s*['"]([^'"{}\s]+)['"]""")
NTFY_URL = re.compile(r"ntfy\.sh/([A-Za-z0-9_-]+)")
TOKEN_PATTERNS = [
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}")),
    ("GitHub fine-grained token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}")),
    ("Anthropic API key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
]
PICODE_SUFFIXES = {".py", ".yaml", ".yml", ".sh", ".json", ".txt", ".md", ".cfg", ".ini"}
CO_AUTHOR = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
# Every git command the driver runs ignores repo hooks.
NO_HOOKS = ("-c", "core.hooksPath=/dev/null")


# ----------------------------------------------------------------------- git

def git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *NO_HOOKS, "-c", "core.quotePath=false", *args],
        cwd=repo, capture_output=True, text=True,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _ref_exists(repo: Path, ref: str) -> bool:
    return subprocess.run(
        ["git", "rev-parse", "--verify", "-q", ref], cwd=repo, capture_output=True
    ).returncode == 0


def _is_ancestor(repo: Path, older: str, newer: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", older, newer], cwd=repo, capture_output=True
    ).returncode == 0


def _head(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD")


def _dirty(repo: Path) -> list[str]:
    # Not git(): its strip() would eat the first line's leading status column.
    out = subprocess.run(["git", "-c", "core.quotePath=false", "status", "--porcelain"],
                         cwd=repo, capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def _git_dir(repo: Path) -> Path:
    return Path(git(repo, "rev-parse", "--absolute-git-dir"))


def _merge_in_progress(repo: Path) -> bool:
    return (_git_dir(repo) / "MERGE_HEAD").exists()


# ------------------------------------------------------------------- secrets

def load_secret_values(repo: Path) -> set[str]:
    """Literal values that must never reach GitHub.

    Values of secret-looking keys in ``.env`` (passwords, tokens, keys; not hosts
    or database names) plus every ntfy topic name in ``PiCode/``: the Pi topics
    act as passwords and the user chose not to publish them.
    """
    values: set[str] = set()
    env = repo / ".env"
    if env.is_file():
        for raw in env.read_text(errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.removeprefix("export ").partition("=")
            value = value.strip().strip("'\"")
            if SECRET_ENV_KEY.search(key) and len(value) >= MIN_SECRET_LEN:
                values.add(value)
    picode = repo / "PiCode"
    if picode.is_dir():
        for path in picode.rglob("*"):
            if ".git" in path.parts or not path.is_file() or path.suffix not in PICODE_SUFFIXES:
                continue
            try:
                if path.stat().st_size > 2 * 1024 * 1024:
                    continue
                text = path.read_text(errors="replace")
            except OSError:
                continue
            for rx in (TOPIC_ASSIGN, NTFY_URL):
                for m in rx.finditer(text):
                    if len(m.group(1)) >= MIN_SECRET_LEN:
                        values.add(m.group(1))
    return values


def drop_public_values(repo: Path, values: Iterable[str], ref: str) -> set[str]:
    """Drop values already present in ``ref``'s tree: they are public already."""
    kept = set()
    for value in values:
        hit = subprocess.run(
            ["git", "grep", "-q", "-F", "-e", value, ref, "--"], cwd=repo, capture_output=True
        )
        if hit.returncode != 0:
            kept.add(value)
    return kept


def scan_for_secrets(text: str, secret_values: Iterable[str]) -> list[str]:
    """Describe each secret-looking line without ever echoing the secret."""
    values = [v for v in secret_values if v]
    findings = []
    for n, line in enumerate(text.splitlines(), 1):
        if any(v in line for v in values):
            findings.append(f"line {n}: a known secret value from .env/PiCode")
        for kind, rx in TOKEN_PATTERNS:
            if rx.search(line):
                findings.append(f"line {n}: {kind}")
    return findings


# ---------------------------------------------------------------- push guard

def new_commits(repo: Path, local_ref: str, remote_ref: str | None) -> list[str]:
    """Commits a push of ``local_ref`` would send to GitHub, oldest first."""
    if remote_ref:
        out = git(repo, "rev-list", "--reverse", f"{remote_ref}..{local_ref}")
    else:
        out = git(repo, "rev-list", "--reverse", local_ref, "--not", f"--remotes={REMOTE}")
    return out.split()


def _parent_count(repo: Path, commit: str) -> int:
    return len(git(repo, "rev-list", "--parents", "-n", "1", commit).split()) - 1


def _added_lines(repo: Path, commit: str) -> Iterator[tuple[str, str]]:
    """(path, text) for every line the commit adds.

    For a merge commit ``git show`` prints a combined diff; only lines new
    relative to every parent (the conflict resolution) count, because the
    merged-in side's own commits are scanned on their own.
    """
    nparents = max(1, _parent_count(repo, commit))
    out = git(repo, "show", "--format=", "--unified=0", "--no-color", "--no-ext-diff", commit)
    path, in_hunk = "?", False
    for line in out.splitlines():
        if line.startswith(("diff --git ", "diff --cc ", "diff --combined ")):
            in_hunk = False
            if line.startswith("diff --cc ") or line.startswith("diff --combined "):
                path = line.split(" ", 2)[2]
            continue
        if not in_hunk:
            if line.startswith("+++ "):
                target = line[4:]
                if target != "/dev/null":
                    path = target[2:] if target.startswith("b/") else target
            elif line.startswith("@@"):
                in_hunk = True
            continue
        if line.startswith("@@"):
            continue
        if line[:nparents] == "+" * nparents:
            yield path, line[nparents:]


def _changed_blobs(repo: Path, commit: str) -> list[tuple[str, str, str]]:
    """(status, path, blob sha) for files a non-merge commit adds or modifies."""
    out = subprocess.run(
        ["git", "diff-tree", "-r", "--root", "--no-commit-id", "-z", "--diff-filter=AM", commit],
        cwd=repo, capture_output=True, check=True,
    ).stdout.decode(errors="replace")
    parts = [p for p in out.split("\0") if p]
    rows = []
    for meta, path in zip(parts[0::2], parts[1::2]):
        fields = meta.split()
        rows.append((fields[4], path, fields[3]))
    return rows


def push_guard(repo: Path, local_ref: str, remote_ref: str | None,
               secret_values: Iterable[str], max_bytes: int = MAX_FILE_BYTES) -> list[str]:
    """Reasons not to push ``local_ref``; empty when the push is safe."""
    values = set(secret_values)
    problems: list[str] = []
    if remote_ref and not _is_ancestor(repo, remote_ref, local_ref):
        problems.append(
            f"{local_ref} is not a fast-forward of {remote_ref}: pushing would rewrite GitHub history"
        )
    for commit in new_commits(repo, local_ref, remote_ref):
        short = commit[:8]
        for path, text in _added_lines(repo, commit):
            for finding in scan_for_secrets(text, values):
                kind = finding.split(": ", 1)[1]
                problems.append(f"{short} {path}: possible secret ({kind})")
        if _parent_count(repo, commit) > 1:
            continue
        blobs = _changed_blobs(repo, commit)
        for _status, path, sha in blobs:
            size = int(git(repo, "cat-file", "-s", sha))
            if size > max_bytes:
                problems.append(f"{short} {path}: {size / 1e6:.1f} MB is over the {max_bytes / 1e6:.0f} MB limit")
        for status, path, _ in blobs:
            if status == "A" and is_local_only(path):
                problems.append(f"{short} {path}: local-only path (must never reach GitHub)")
    return list(dict.fromkeys(problems))


def is_local_only(path: str) -> bool:
    """Paths that stay on this machine even when force-added.

    Not "anything matching .gitignore": docs/ and config/example* are ignored
    yet tracked on purpose. These are the ones that hold secrets, the operative
    config, data or agent state.
    """
    parts = path.split("/")
    name = parts[-1]
    if name == ".env" or name.startswith(".env."):
        return not (parts[0] == "config" and name.startswith("example"))
    if parts[0] in LOCAL_ONLY_DIRS:
        return True
    if parts[0] == "config" and len(parts) > 1:
        return not (name.startswith("example") or name == "environment.yml")
    return False


# ----------------------------------------------------------------- preflight

def preflight(repo: Path) -> list[str]:
    problems = []
    branch = git(repo, "symbolic-ref", "--short", "-q", "HEAD", check=False)
    if branch != MAIN:
        problems.append(f"the repo is on '{branch or 'a detached HEAD'}', not {MAIN}")
    gd = _git_dir(repo)
    for marker, what in (("MERGE_HEAD", "a merge"), ("CHERRY_PICK_HEAD", "a cherry-pick"),
                         ("REVERT_HEAD", "a revert"), ("rebase-merge", "a rebase"),
                         ("rebase-apply", "a rebase or am")):
        if (gd / marker).exists():
            problems.append(f"{what} is in progress ({marker})")
    return problems


def pipeline_running(proc_root: Path = Path("/proc"), self_pid: int | None = None,
                     needle: str = "run_workflows.py") -> bool:
    self_pid = os.getpid() if self_pid is None else self_pid
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == self_pid:
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if any(needle in a.decode(errors="replace") for a in args):
            return True
    return False


# --------------------------------------------------------------------- tests

def parse_junit(xml_path: Path) -> tuple[set[str], set[str]]:
    """(every test node id seen, the failed/errored ones) from an xunit1 report."""
    seen: set[str] = set()
    failed: set[str] = set()
    for tc in ET.parse(xml_path).getroot().iter("testcase"):
        name = tc.get("name", "")
        cls = tc.get("classname", "") or ""
        file = tc.get("file")
        if file:
            module = file[:-3].replace("/", ".") if file.endswith(".py") else file
            rest = cls[len(module) + 1:] if cls.startswith(module + ".") else ""
            node = f"{file}::{rest}::{name}" if rest else f"{file}::{name}"
        elif not cls:  # collection error: name is the dotted module
            node = name.replace(".", "/") + ".py"
        else:
            node = f"{cls.replace('.', '/')}.py::{name}"
        seen.add(node)
        if tc.find("failure") is not None or tc.find("error") is not None:
            failed.add(node)
    return seen, failed


def classify_failures(head_failed: Iterable[str], baseline_seen: set[str],
                      baseline_failed: set[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"regressions": [], "preexisting": [], "new": []}
    for node in sorted(head_failed):
        if node not in baseline_seen:
            out["new"].append(node)
        elif node in baseline_failed:
            out["preexisting"].append(node)
        else:
            out["regressions"].append(node)
    return out


def run_pytest(cwd: Path, files: list[str] | None) -> tuple[set[str], set[str]]:
    with tempfile.TemporaryDirectory(prefix="weekly-sync-junit-") as td:
        xml = Path(td) / "junit.xml"
        env = os.environ.copy()
        env["FBPIPE_DISABLE_NTFY"] = "1"
        # The editable fbpipe install points at the main checkout; a baseline
        # worktree must import its own src/ instead.
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(Path(cwd) / "src"), env.get("PYTHONPATH")]))
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "-o", "junit_family=xunit1", f"--junitxml={xml}", *(files or ["tests/"])]
        proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                              timeout=TEST_TIMEOUT_S)
        tail = "\n".join(proc.stdout.strip().splitlines()[-3:])
        print(f"[tests] {cwd}: {tail}", flush=True)
        if not xml.exists():
            raise RuntimeError(f"pytest wrote no report (exit {proc.returncode}): {proc.stdout[-1500:]}")
        return parse_junit(xml)


def _link_missing(src: Path, dst: Path) -> None:
    """Symlink ``src`` at ``dst``; if ``dst`` is already a folder (``config/``
    holds tracked ``example.*`` files), link each entry it lacks instead, so a
    gitignored file beside tracked ones still reaches the baseline. Entries the
    worktree already has stay the pushed commit's version."""
    if not src.exists():
        return
    if not dst.exists() and not dst.is_symlink():
        dst.symlink_to(src)
    elif src.is_dir() and dst.is_dir() and not dst.is_symlink():
        for child in src.iterdir():
            _link_missing(child, dst / child.name)


def baseline_results(repo: Path, sha: str, files: list[str],
                     test_runner: Callable) -> tuple[set[str], set[str]]:
    """Run ``files`` at ``sha`` in a throwaway worktree (only files that existed then)."""
    td = Path(tempfile.mkdtemp(prefix="weekly-sync-baseline-"))
    wt = td / "wt"
    git(repo, "worktree", "add", "--detach", str(wt), sha)
    try:
        for name in BASELINE_LINKS:
            _link_missing(repo / name, wt / name)
        existing = [f for f in files if (wt / f).is_file()]
        if not existing:
            return set(), set()
        return test_runner(wt, existing)
    finally:
        git(repo, "worktree", "remove", "--force", str(wt), check=False)
        git(repo, "worktree", "prune", check=False)
        shutil.rmtree(td, ignore_errors=True)


# -------------------------------------------------------------------- claude

_READ_ONLY = [
    "Read", "Glob", "Grep",
    "Bash(git status*)", "Bash(git diff*)", "Bash(git log*)", "Bash(git show*)",
    "Bash(git rev-parse*)", "Bash(git ls-files*)", "Bash(git check-ignore*)",
    "Bash(git branch --show-current*)", "Bash(git merge-base*)",
    "Bash(ls*)", "Bash(cat *)", "Bash(head *)", "Bash(tail *)", "Bash(wc *)",
    "Bash(du *)", "Bash(file *)", "Bash(grep *)", "Bash(cd *)",
    # No find (-exec/-delete) or sort (-o): the Glob and Grep tools cover them.
]
_COMMIT = ["Bash(git add *)", "Bash(git commit *)", "Bash(git restore --staged *)",
           "Bash(git rm --cached *)"]
_EDIT = ["Bash(git merge --no-edit origin/main*)", "Bash(git merge origin/main*)",
         "Bash(git merge --abort*)", "Bash(git merge --continue*)",
         "Bash(python3 -m pytest *)"]
_NEVER = [
    "Bash(git push*)", "Bash(git reset*)", "Bash(git rebase*)", "Bash(git checkout*)",
    "Bash(git switch*)", "Bash(git stash*)", "Bash(git clean*)", "Bash(git restore --worktree*)",
    "Bash(git add -f*)", "Bash(git add --force*)", "Bash(git add * -f*)", "Bash(git add * --force*)",
    "Bash(git commit --amend*)", "Bash(git commit * --amend*)",
    "Bash(git branch -D*)", "Bash(git branch -d*)", "Bash(git branch --delete*)",
    "Bash(git tag*)", "Bash(git remote*)", "Bash(git config*)", "Bash(git worktree*)",
    "Bash(git update-ref*)", "Bash(git filter-branch*)", "Bash(git -C*)",
    "Bash(rm *)", "Bash(sudo *)", "Bash(ssh *)", "Bash(scp *)", "Bash(rsync *)",
    "Bash(curl *)", "Bash(wget *)", "WebFetch", "WebSearch",
]


class ClaudeError(RuntimeError):
    """A headless Claude run failed (non-zero exit, timeout, expired sign-in)."""


def build_claude_cmd(prompt: str, *, allow_edits: bool, claude_bin: str = "claude",
                     repo: Path = REPO) -> list[str]:
    # "//abs/path" is Claude Code's absolute-path rule form. A bare "Edit" or
    # "Write" allow rule would cover every path on the machine.
    root = "/" + str(Path(repo).resolve())
    allowed = _READ_ONLY + _COMMIT
    disallowed = list(_NEVER)
    if allow_edits:
        allowed += [f"Edit({root}/**)", f"Write({root}/**)"] + _EDIT
        for tool in ("Edit", "Write"):
            disallowed += [f"{tool}({root}/{d}/**)"
                           for d in (".git", "config", "PiCode", "Results", ".claude", ".remember")]
            disallowed += [f"{tool}({root}/.env)", f"{tool}({root}/**/.env)"]
    else:
        disallowed += ["Edit", "Write", "NotebookEdit"]
    return [claude_bin, "-p", prompt,
            "--permission-mode", "dontAsk",
            "--output-format", "text",
            "--disable-slash-commands",
            "--strict-mcp-config",
            # Not "local": the repo's .claude/settings.local.json allows Bash(python3:*),
            # Edit(*), Write(*) and /tmp, which would undo the rules here. "project"
            # is what loads CLAUDE.md (there is no .claude/settings.json).
            "--setting-sources", "user,project",
            "--allowedTools", *allowed,
            "--disallowedTools", *disallowed]


def claude_env() -> dict[str, str]:
    """Environment for Claude and everything it starts (pytest included).

    Claude's allowed git commands are all local, so it loses nothing by having
    no way to reach GitHub: SSH is /bin/false, no agent socket, no tokens.
    """
    env = os.environ.copy()
    for var in ("SSH_AUTH_SOCK", "GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN"):
        env.pop(var, None)
    env["GIT_SSH_COMMAND"] = "/bin/false"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["FBPIPE_DISABLE_NTFY"] = "1"
    env["WEEKLY_GITHUB_SYNC"] = "1"
    return env


def run_claude(prompt: str, allow_edits: bool, repo: Path = REPO) -> str:
    claude_bin = shutil.which("claude") or str(Path.home() / ".local/bin/claude")
    env = claude_env()
    cmd = build_claude_cmd(prompt, allow_edits=allow_edits, claude_bin=claude_bin, repo=repo)
    try:
        proc = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True,
                              timeout=CLAUDE_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise ClaudeError(f"Claude timed out after {CLAUDE_TIMEOUT_S // 60} min") from exc
    if proc.returncode != 0:
        raise ClaudeError(f"claude exited {proc.returncode}: "
                          f"{(proc.stderr.strip() or proc.stdout.strip())[-1500:]}")
    # Surface the sign-in expiry warning: once it lapses every run fails.
    warn = [ln for ln in proc.stderr.splitlines() if "sign-in ends" in ln]
    return proc.stdout.strip() + ("\n\n(" + warn[0].strip() + ")" if warn else "")


def _prompt(name: str, **values: str) -> str:
    return Template((PROMPT_DIR / f"{name}.md").read_text()).safe_substitute(co_author=CO_AUTHOR, **values)


def commit_prompt(*, busy: bool, origin_sha: str, today: str) -> str:
    if busy:
        mode = ("A pipeline run (run_workflows.py) is active right now, so you cannot edit files "
                "and must NOT merge origin/main this week. Only commit.")
    else:
        mode = (f"After committing, merge GitHub's main into ours: `git merge --no-edit origin/main` "
                f"(origin/main is {origin_sha[:10]}). If it conflicts, resolve every conflict keeping "
                f"both sides' intent, run the tests of the files you touched, and conclude the merge "
                f"with `git commit --no-edit`. If you cannot resolve it with confidence, run "
                f"`git merge --abort` and say why.")
    return _prompt("commit", mode=mode, today=today)


def fix_prompt(*, regressions: list[str], baseline_sha: str, today: str) -> str:
    listing = "\n".join(f"- `{node}`" for node in regressions)
    return _prompt("fix", regressions=listing, baseline_sha=baseline_sha, today=today)


# -------------------------------------------------------------------- report

@dataclasses.dataclass
class Report:
    date: str
    dry_run: bool = False
    busy: bool = False
    aborted: list[str] = dataclasses.field(default_factory=list)
    commits: list[str] = dataclasses.field(default_factory=list)
    leftovers: list[str] = dataclasses.field(default_factory=list)
    claude: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    tests_ran: bool = False
    tests_skipped: str = "nothing changed since the last push"
    regressions: list[str] = dataclasses.field(default_factory=list)
    preexisting: list[str] = dataclasses.field(default_factory=list)
    new_failing: list[str] = dataclasses.field(default_factory=list)
    fixed: list[str] = dataclasses.field(default_factory=list)
    still_failing: list[str] = dataclasses.field(default_factory=list)
    fix_reverted: bool = False
    pushed_main: bool = False
    push_problems: list[str] = dataclasses.field(default_factory=list)
    branches_pushed: list[str] = dataclasses.field(default_factory=list)
    branches_skipped: list[str] = dataclasses.field(default_factory=list)
    notes: list[str] = dataclasses.field(default_factory=list)
    errors: list[str] = dataclasses.field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.aborted or self.push_problems or self.errors)

    def headline(self) -> str:
        if self.aborted:
            return "Weekly GitHub sync did nothing: " + "; ".join(self.aborted)
        bits = []
        if self.dry_run:
            bits.append("DRY RUN")
        bits.append(f"{len(self.commits)} commit(s) ahead of GitHub")
        bits.append("pushed" if self.pushed_main else "main NOT pushed")
        if self.fixed:
            bits.append(f"{len(self.fixed)} regression(s) fixed")
        if self.still_failing:
            bits.append(f"{len(self.still_failing)} regression(s) still failing")
        if self.branches_pushed:
            bits.append(f"{len(self.branches_pushed)} branch backup(s)")
        return ", ".join(bits)

    def to_markdown(self) -> str:
        def section(title: str, items: list[str]) -> list[str]:
            return [f"## {title}", "", *([f"- {i}" for i in items] or ["- none"]), ""]

        lines = [f"# Weekly GitHub sync — {self.date}", "", f"**{self.headline()}**", ""]
        if self.busy:
            lines += ["A pipeline run was active: committed only, no merge and no fixes.", ""]
        if self.aborted:
            lines += section("Stopped before doing anything", self.aborted)
        lines += section("Commits on main that were not on GitHub", self.commits)
        lines += section("Left uncommitted", self.leftovers)
        if self.tests_ran:
            lines += section("Regressions (passed at the last push, failed this week)", self.regressions)
            lines += section("Fixed by the fix step", self.fixed)
            lines += section("Regressions still failing", self.still_failing)
            if self.fix_reverted:
                lines += ["The fix step made other tests fail, so its commits were reverted.", ""]
            lines += section("Failing before this week too (left alone)", self.preexisting)
            lines += section("New tests that fail (left alone)", self.new_failing)
        else:
            lines += [f"Tests were not run ({self.tests_skipped}).", ""]
        lines += section("Push refused because", self.push_problems)
        lines += section("Branches backed up to GitHub", self.branches_pushed)
        lines += section("Branches not backed up", self.branches_skipped)
        if self.notes:
            lines += section("Notes", self.notes)
        if self.errors:
            lines += section("Errors", self.errors)
        for phase, text in self.claude:
            lines += [f"## Claude — {phase} step", "", text.strip() or "(no output)", ""]
        return "\n".join(lines)

    def write(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.date}.md"
        n = 2
        while path.exists():
            path = out_dir / f"{self.date}-{n}.md"
            n += 1
        path.write_text(self.to_markdown())
        return path


def ntfy(title: str, message: str, priority: str = "default") -> None:
    topic = os.getenv("NTFY_TOPIC", NTFY_DEFAULT_TOPIC)
    try:
        req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=message.encode(), method="POST")
        req.add_header("Title", title)
        req.add_header("Priority", priority)
        req.add_header("Tags", "octocat")
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:  # network trouble must not fail the job
        print(f"[ntfy] failed: {exc}", flush=True)


# ----------------------------------------------------------------------- run

def _log(msg: str) -> None:
    print(f"[weekly-github-sync] {msg}", flush=True)


def _abort_merge_if_left(repo: Path, report: Report) -> None:
    if _merge_in_progress(repo):
        git(repo, "merge", "--abort", check=False)
        report.notes.append("A merge was left half-done; it was aborted (origin/main not merged).")


def _test_phase(repo: Path, report: Report, pushed_sha: str, test_runner: Callable,
                claude_runner: Callable, busy: bool, today: str) -> None:
    _, failed = test_runner(repo, None)
    report.tests_ran = True
    if not failed:
        return
    files = sorted({node.split("::")[0] for node in failed})
    try:
        base_seen, base_failed = baseline_results(repo, pushed_sha, files, test_runner)
    except Exception as exc:
        report.errors.append(f"baseline test run failed: {exc}")
        base_seen, base_failed = set(failed), set(failed)
    c = classify_failures(failed, base_seen, base_failed)
    report.regressions, report.preexisting, report.new_failing = c["regressions"], c["preexisting"], c["new"]
    report.still_failing = list(c["regressions"])
    if not c["regressions"]:
        return
    if busy or claude_runner is None:
        report.notes.append("Regressions were not sent to the fix step "
                            + ("(a pipeline run was active)." if busy else "(Claude disabled)."))
        return

    _log(f"fix step for {len(c['regressions'])} regression(s)")
    before_fix = _head(repo)
    try:
        out = claude_runner(fix_prompt(regressions=c["regressions"], baseline_sha=pushed_sha, today=today), True)
    except ClaudeError as exc:
        out = ""
        report.errors.append(f"fix step: {exc}")
    report.claude.append(("fix", out))
    _abort_merge_if_left(repo, report)
    if preflight(repo):
        report.errors.append("after the fix step: " + "; ".join(preflight(repo)))
        return
    if _head(repo) == before_fix:
        report.notes.append("The fix step made no commit.")
        return
    _, failed_after = test_runner(repo, None)
    broke = sorted(set(failed_after) - set(failed))
    if broke:
        report.fix_reverted = True
        report.notes.append("Reverted the fix step: it made these pass→fail: " + ", ".join(broke))
        rev = subprocess.run(["git", *NO_HOOKS, "revert", "--no-edit", f"{before_fix}..HEAD"],
                             cwd=repo, capture_output=True, text=True)
        if rev.returncode != 0:
            git(repo, "revert", "--abort", check=False)
            report.errors.append(f"could not revert the fix commits: {rev.stderr.strip()[-500:]}")
        return
    report.still_failing = [n for n in c["regressions"] if n in failed_after]
    report.fixed = [n for n in c["regressions"] if n not in failed_after]


def _push(repo: Path, local: str, remote_branch: str) -> str | None:
    proc = subprocess.run(["git", *NO_HOOKS, "push", REMOTE, f"refs/heads/{local}:refs/heads/{remote_branch}"],
                          cwd=repo, capture_output=True, text=True)
    return None if proc.returncode == 0 else proc.stderr.strip()[-500:]


def _backup_branches(repo: Path, report: Report, secrets: set[str], dry_run: bool) -> None:
    branches = git(repo, "for-each-ref", "--format=%(refname:short)", "refs/heads/").split()
    for branch in branches:
        if branch == MAIN:
            continue
        remote_ref = f"{REMOTE}/{branch}" if _ref_exists(repo, f"refs/remotes/{REMOTE}/{branch}") else None
        if not new_commits(repo, branch, remote_ref):
            continue
        problems = push_guard(repo, branch, remote_ref, secrets)
        if problems:
            report.branches_skipped.append(f"{branch}: " + "; ".join(problems[:3])
                                           + (f" (+{len(problems) - 3} more)" if len(problems) > 3 else ""))
            continue
        if dry_run:
            report.branches_pushed.append(f"{branch} (dry run: would push)")
            continue
        err = _push(repo, branch, branch)
        if err:
            report.branches_skipped.append(f"{branch}: push failed: {err}")
        else:
            report.branches_pushed.append(branch)


def run(repo: Path = REPO, *, claude_runner: Callable | None = None,
        test_runner: Callable = run_pytest, notifier: Callable | None = ntfy,
        busy: bool | None = None, dry_run: bool = False, run_tests: bool = True,
        today: str | None = None) -> Report:
    today = today or dt.date.today().isoformat()
    report = Report(date=today, dry_run=dry_run)
    try:
        _run(repo, report, claude_runner, test_runner, busy, dry_run, run_tests, today)
    except Exception as exc:
        report.errors.append(f"{type(exc).__name__}: {exc}")
    if notifier:
        msg = report.headline()
        if report.push_problems:
            msg += "\nPush refused: " + "; ".join(report.push_problems[:3])
        if report.still_failing:
            msg += f"\nStill failing: {', '.join(report.still_failing[:5])}"
        if report.errors:
            msg += "\nErrors: " + "; ".join(report.errors[:2])
        priority = "high" if (not report.ok or report.still_failing) else "default"
        notifier("Weekly GitHub sync" + ("" if report.ok else ": needs attention"), msg, priority)
    return report


def _run(repo, report, claude_runner, test_runner, busy, dry_run, run_tests, today):
    problems = preflight(repo)
    if problems:
        report.aborted = problems
        return
    git(repo, "fetch", REMOTE)
    remote_main = f"{REMOTE}/{MAIN}"
    pushed_sha = git(repo, "rev-parse", remote_main)
    busy = pipeline_running() if busy is None else busy
    report.busy = busy

    # 2. commit (+ merge origin/main)
    needs_merge = not _is_ancestor(repo, remote_main, "HEAD")
    if _dirty(repo) or needs_merge:
        if claude_runner is None:
            report.notes.append("Claude disabled: uncommitted work and origin/main left as they were.")
        else:
            _log("commit step")
            try:
                out = claude_runner(commit_prompt(busy=busy, origin_sha=pushed_sha, today=today), not busy)
            except ClaudeError as exc:
                out = ""
                report.errors.append(f"commit step: {exc}")
            report.claude.append(("commit", out))
            _abort_merge_if_left(repo, report)
            after = preflight(repo)
            if after:
                report.errors.append("after the commit step: " + "; ".join(after))
                return
    report.leftovers = _dirty(repo)

    # 3/4. test, classify, fix
    if not run_tests:
        report.tests_skipped = "--no-tests"
    elif _head(repo) != pushed_sha:
        _log("test step")
        _test_phase(repo, report, pushed_sha, test_runner, claude_runner, busy, today)

    report.commits = [line for line in git(
        repo, "log", "--reverse", "--format=%h %s", f"{pushed_sha}..HEAD").splitlines() if line]

    # 5. push main
    git(repo, "fetch", REMOTE)
    secrets = drop_public_values(repo, load_secret_values(repo), remote_main)
    if new_commits(repo, MAIN, remote_main):
        report.push_problems = push_guard(repo, MAIN, remote_main, secrets)
        if not report.push_problems:
            if dry_run:
                report.notes.append("Dry run: main would have been pushed.")
            else:
                err = _push(repo, MAIN, MAIN)
                if err:
                    report.push_problems.append(f"git push failed: {err}")
                else:
                    report.pushed_main = True
    # 6. back up the other branches
    _backup_branches(repo, report, secrets, dry_run)


# ----------------------------------------------------------------------- cli

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="no Claude, no push: run the checks and tests and report what would happen")
    ap.add_argument("--no-claude", action="store_true", help="skip the commit and fix steps")
    ap.add_argument("--no-tests", action="store_true", help="skip the test step")
    ap.add_argument("--no-notify", action="store_true", help="do not send the ntfy summary")
    args = ap.parse_args(argv)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lock = open(REPORT_DIR / ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        _log("another sync is running; exiting")
        return 0

    use_claude = not (args.dry_run or args.no_claude)
    report = run(REPO,
                 claude_runner=(lambda p, e: run_claude(p, e, REPO)) if use_claude else None,
                 notifier=None if args.no_notify else ntfy,
                 dry_run=args.dry_run, run_tests=not args.no_tests)
    path = report.write(REPORT_DIR)
    _log(report.headline())
    _log(f"report: {path}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
