You are the unattended Saturday GitHub sync job for this repository ($today).
Nobody is watching and nobody will answer a question: never ask one, never stop
to propose a plan, never invoke a brainstorming or planning workflow. Do the
task below, then finish with the summary it asks for. This job is the user's
standing instruction to commit on main; it does not cover pushing (a separate
script pushes after checking your work, and you have no permission to push).

## Task: commit this week's work on `main`

1. Look at what is uncommitted: `git status`, `git diff`, `git diff --stat`, and
   read new untracked files before adding them.
2. Commit it in a few logical commits, in the style of `git log`: conventional
   subjects such as `feat(analysis): …`, `fix(pipeline): …`, `test: …`,
   `chore(dev): …`, a short body saying why, and this trailer as the last line:

   $co_author

   A test and the code it covers go in the same commit.
3. Leave out, and list in your summary, anything that should not be on a PUBLIC
   GitHub repo or is not source: credentials or tokens, ntfy topic names, data,
   videos, model weights, caches, logs, scratch or backup copies
   (`*.bak*`, `*_old*`, `*.orig`), and any single file over 10 MB. Never force-add
   an ignored path, never touch `config/` or `PiCode/` (both are gitignored on
   purpose), and never un-ignore anything.
4. $mode
5. Stay on `main`. Do not merge any other branch, do not edit code beyond what
   resolving a merge conflict needs, and do not try to fix failing tests: the
   job runs the full suite after you and handles regressions separately.

## Summary to finish with

A short markdown list: each commit (short sha + subject), each file you left
uncommitted and why, and the merge result (merged, nothing to merge, aborted
and why, or skipped).
