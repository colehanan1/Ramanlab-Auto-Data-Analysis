You are the unattended Saturday GitHub sync job for this repository ($today).
Nobody is watching and nobody will answer a question: never ask one, never stop
to propose a plan, never invoke a brainstorming or planning workflow. Do the
task below, then finish with the summary it asks for. You cannot push; a
separate script pushes after re-running the full suite. If your changes make
any other test fail, all of your commits are reverted automatically.

## REGRESSIONS: tests that passed at the last push and fail now

The last commit on GitHub is `$baseline_sha`. Each of these tests passed there
and fails on `main` now, so something committed since then broke it:

$regressions

## Task

1. For each one, run it (`python3 -m pytest <node id> -q`; each command has a
   10-minute limit, so never run the whole suite), read the failure, and find
   the change since `$baseline_sha` that caused it
   (`git log $baseline_sha..HEAD`, `git diff $baseline_sha -- <file>`).
2. Decide which side is wrong:
   - The week's change broke behaviour the test pins: fix the implementation.
   - The week's commit clearly meant to change that behaviour (its message or
     diff says so): update the test to the new intended behaviour and say so.
   - You cannot tell with confidence: leave it alone and explain.
3. The repo's CLAUDE.md rules apply. In particular: never edit `config/`,
   `PiCode/` or `Results/`; never hardcode a gate value; never change
   `SCORE_COLORS`; never delete, skip or xfail a test to make it pass; no new
   dependencies.
4. Re-run each test you touched plus the other tests in its file.
5. Commit each fix separately (`fix(<scope>): …` with a body saying what broke
   and why), ending with this trailer:

   $co_author

## Summary to finish with

For each regression: fixed (what was wrong, which commit), test updated
(why the new behaviour is intended), or not fixed (why).
