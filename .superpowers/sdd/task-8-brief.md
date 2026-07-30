### Task 8: End-to-end verification and legacy regression

**Files:**
- Test: `tests/test_freeze_e2e.py` (create)

**Interfaces:**
- Consumes: everything from Tasks 1-7

- [ ] **Step 1: Write the end-to-end test**

Create `tests/test_freeze_e2e.py`:

```python
"""Freeze end-to-end: derive live, freeze, re-run, output is unchanged."""

import numpy as np
import pandas as pd
import pytest

import scripts.analysis.envelope_combined as ec
import scripts.analysis.envelope_visuals as ev
from fbpipe import freeze
from fbpipe.config import DatasetOverride


class _Tracking:
    """Stub of fbpipe.config.TrackingConfig -- build_fingerprint duck-types it.

    build_wide_csv reads settings.tracking internally (envelope_combined.py:2622)
    to derive the tracking_* columns, so it is part of the fingerprint.
    """

    apply_missing_frame_check = True
    max_missing_frames_per_trial = 5000
    max_missing_frames_pct_per_trial = 50.0


def _make(root, n, fly="october_01_fly1"):
    out = root / fly / "angle_distance_rms_envelope"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"envelope_of_rms": np.linspace(0, 100, n)}).to_csv(
        out / f"{fly}_testing_1_angle_distance_rms_envelope.csv", index=False
    )
    return root


@pytest.fixture(autouse=True)
def _v2():
    ev.set_protocol("v2")


def test_freeze_then_rerun_is_byte_identical(tmp_path):
    """The user-facing promise, exercised through the real cache.

    FROZEN is deliberately longer than LIVE so the own_max_len fold is genuinely
    exercised rather than correct by accident.
    """
    live = _make(tmp_path / "LIVE", 9)
    frozen = _make(tmp_path / "FROZEN", 21)
    cache = tmp_path / "cache"

    fp_kw = dict(
        protocol="v2",
        measure_cols=["envelope_of_rms"],
        fps_fallback=40.0,
        distance_limits=None,
        non_reactive_threshold=None,
        low_max_threshold_px=ec.LOW_MAX_FLAG_THRESHOLD_PX,
        use_per_trial_baseline=False,
        override=DatasetOverride(),
        tracking=_Tracking(),
    )
    fp = freeze.build_fingerprint(**fp_kw)

    # Pass 1: fully live.
    first = tmp_path / "first.csv"
    ec.build_wide_csv(
        [str(live), str(frozen)], str(first), measure_cols=["envelope_of_rms"]
    )
    baseline = pd.read_csv(first)

    # Cache FROZEN's slice, as the pipeline does after a live derivation.
    freeze.save_slice(
        cache, "wide", "FROZEN",
        baseline[baseline["dataset"] == "FROZEN"].reset_index(drop=True), fp,
    )

    # Pass 2: FROZEN spliced from cache.
    got = freeze.load_slice(cache, "wide", "FROZEN", fp)
    assert got is not None and got.own_max_len == 21

    second = tmp_path / "second.csv"
    ec.build_wide_csv(
        [str(live), str(frozen)], str(second),
        measure_cols=["envelope_of_rms"],
        frozen_slices={"FROZEN": (got.rows, got.own_max_len)},
    )

    pd.testing.assert_frame_equal(
        baseline.sort_values(["dataset", "fly"]).reset_index(drop=True),
        pd.read_csv(second).sort_values(["dataset", "fly"]).reset_index(drop=True),
    )


def test_deleting_the_cache_is_safe(tmp_path):
    """The cache is a derived artifact: deleting it must only cost time."""
    import shutil

    live = _make(tmp_path / "LIVE", 9)
    frozen = _make(tmp_path / "FROZEN", 21)
    cache = tmp_path / "cache"
    fp = freeze.build_fingerprint(
        protocol="v2", measure_cols=["envelope_of_rms"], fps_fallback=40.0,
        distance_limits=None, non_reactive_threshold=None,
        low_max_threshold_px=ec.LOW_MAX_FLAG_THRESHOLD_PX,
        use_per_trial_baseline=False, override=DatasetOverride(),
        tracking=_Tracking(),
    )
    out1 = tmp_path / "a.csv"
    ec.build_wide_csv([str(live), str(frozen)], str(out1), measure_cols=["envelope_of_rms"])
    base = pd.read_csv(out1)
    freeze.save_slice(
        cache, "wide", "FROZEN", base[base["dataset"] == "FROZEN"].reset_index(drop=True), fp
    )
    shutil.rmtree(cache)
    assert freeze.load_slice(cache, "wide", "FROZEN", fp) is None

    out2 = tmp_path / "b.csv"
    ec.build_wide_csv([str(live), str(frozen)], str(out2), measure_cols=["envelope_of_rms"])
    pd.testing.assert_frame_equal(base, pd.read_csv(out2))
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/test_freeze_e2e.py -v`
Expected: PASS (2 passed)

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: PASS, ≥538 passed plus the new freeze tests.

- [ ] **Step 4: Verify legacy is byte-for-byte identical**

Legacy output must not move. Create a worktree at the branch point and compare
against the SAME data:

```bash
git worktree add /tmp/freeze-baseline 47d6fe3
```

Run the legacy protocol suite in both trees and diff the emitted wide CSVs.

Do NOT verify this with `git stash` — a no-op diff produces a vacuous pass. Do
NOT compare against a stale baseline — the pipeline regenerates
`model_predictions.csv`.

Run: `python -m pytest tests/test_protocol_legacy_regression.py -v`
Expected: PASS

```bash
git worktree remove /tmp/freeze-baseline
```

- [ ] **Step 5: Verify the real pipeline runs**

Run: `python scripts/pipeline/run_workflows.py --config config/config_new.yaml --figures-only`
Expected: exit 0, no freeze warnings (no dataset is frozen yet, so behavior is unchanged).

Then, on a dataset that has been derived at least once, add to
`config/config_new.yaml`:

```yaml
dataset_overrides:
  EB-Control-24-1:
    freeze:
      data: true
      figures: true
```

Re-run and confirm the log shows `[FROZEN] Not walking root ...` for that dataset
and that its rows are still present in the wide CSV.

- [ ] **Step 6: Commit**

```bash
git add tests/test_freeze_e2e.py
git commit -m "test(freeze): end-to-end round-trip and cache-deletion safety"
```

---

## Notes for the implementer

**The one thing that will silently break this:** `build_wide_csv` truncates rows
longer than the global `max_len` (`:3283-3284`). If a frozen dataset's
`own_max_len` is not folded into that global max (Task 3, Step 6), its cached
rows are chopped with no error. Every round-trip test therefore uses a frozen
dataset LONGER than the live one — with equal lengths the tests pass even with
the fold deleted. If you find yourself simplifying a fixture to equal lengths,
stop: you are deleting the only thing the test proves.

**This branch's recorded failure modes** — check your own tests against them:
- An assertion that reads its expectation from the constant it validates.
- An assertion over an empty collection (empty for the wrong reason).
- An assertion on a quantity that is equal by construction.
- Place mutants INSIDE the unit under test and re-run. Do not trust a passing
  report you have not tried to break.

**Out of scope; do not fix here:**
- ~~`wide_measure_cols` is bound only inside `if wide_cfg:`, so the `pair_groups` call site would `NameError` if configured.~~ **RETRACTED — this was false.** `wide_measure_cols` (`:1093`), `wide_fps_fallback` (`:1094`), `wide_exclude_cfg` (`:1095`) and `use_per_trial_baseline` (`:1099`) all have function-level defaults and are always bound; `if wide_cfg:` (`:1156`) only overrides them. The claim originated in an exploration report and was propagated into this plan without verification. There is no latent bug.
- `_style_trained_xticks` (`envelope_visuals.py:1309`) compares odor names against a dataset display label, so sibling figures disagree about the trained odor. Pre-existing.
- `docs/` and `config/` are gitignored (`.gitignore:116`), so `config/config_new.yaml` — and any `freeze:` block in it — is not tracked. Flagged in the spec.
