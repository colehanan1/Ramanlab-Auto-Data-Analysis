# Per-rig Anchor (Mirrored rig_3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the proboscis angle multiplier correct for the physically mirrored rig_3 by resolving the geometric anchor per rig, using only recalculation from already-stored coordinates.

**Architecture:** A single shared helper maps a trial path to its anchor (`rig_3` → left edge, everything else → right edge). That anchor is threaded as an optional parameter through the angle functions in the two modules that compute angles, defaulting to today's constants so every existing caller is unchanged. Existing rig_3 data is fixed by invalidating cached angle columns and recomputing from stored eye/proboscis coordinates.

**Tech Stack:** Python 3.12, numpy, pandas, pytest. Conda env `yolo-env`.

## Global Constraints

- **No video re-encode and no YOLO re-run.** Every correction is a recalculation from stored `x_class0/y_class0/x_class1/y_class1` columns.
- **rig_2 output must remain byte-identical.** Only rig_3 values may change. This is the primary regression gate.
- Default anchor is exactly `(1080.0, 540.0)`. Mirrored (rig_3) anchor is exactly `(0.0, 540.0)`.
- The angle is **unsigned** (`arctan2(abs(cross), dot)`, range 0–180). Do not introduce a signed angle.
- A reference/baseline angle and the measurement it centers **must** use the same anchor. Mixing them is silently wrong, not an error.
- Run tests from the repo root with `conda run -n yolo-env python -m pytest`.
- The repo default branch is `main`; work happens on `feature/rig3-mirrored-anchor`.

## Deviation from the spec

The spec proposed a YAML `rig_anchors:` config block. During planning it emerged
that `envelope_combined._compute_angle_deg`, `_find_reference_angle`, and
`_fly_max_centered` are private module functions with no `Settings` access, so a
config-driven anchor would require threading `Settings` through six functions in
two modules. This plan uses module-level constants in the shared helper instead.
Adding a future rig is still a one-line edit (`MIRRORED_RIGS`), with far less
plumbing. Everything else in the spec is unchanged.

---

### Task 1: Shared per-rig anchor helper

**Files:**

- Create: `src/fbpipe/utils/rig_anchor.py`
- Test: `tests/test_rig_anchor.py`

**Interfaces:**

- Consumes: nothing (leaf module).
- Produces:
  - `DEFAULT_ANCHOR: tuple[float, float]` = `(1080.0, 540.0)`
  - `MIRRORED_ANCHOR: tuple[float, float]` = `(0.0, 540.0)`
  - `MIRRORED_RIGS: frozenset[str]` = `{"rig_3"}`
  - `rig_token(path: str | Path) -> str | None` — normalized token e.g. `"rig_3"`, or `None`
  - `resolve_anchor(path: str | Path) -> tuple[float, float]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_rig_anchor.py`:

```python
"""Per-rig anchor resolution.

rig_3 is a physically mirrored rig (flies and odor tube on the opposite side),
so its geometric anchor sits on the left edge instead of the right.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from fbpipe.utils.rig_anchor import (
    DEFAULT_ANCHOR,
    MIRRORED_ANCHOR,
    resolve_anchor,
    rig_token,
)


def test_default_and_mirrored_anchor_values():
    assert DEFAULT_ANCHOR == (1080.0, 540.0)
    assert MIRRORED_ANCHOR == (0.0, 540.0)


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/data/EB-Training-24-1/july_17_batch_2_rig_3/trial_1", "rig_3"),
        ("/data/EB-Training-24-1/july_17_batch_2_rig_2/trial_1", "rig_2"),
        ("/data/3Oct-Training-24-0.1/july_20_batch_1_rig_3", "rig_3"),
        ("relative/july_18_batch_1_rig_3/x.parquet", "rig_3"),
        ("/data/no_rig_here/trial_1", None),
    ],
)
def test_rig_token(path, expected):
    assert rig_token(path) == expected


def test_rig_3_resolves_to_mirrored_anchor():
    p = "/data/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_testing_1"
    assert resolve_anchor(p) == MIRRORED_ANCHOR


def test_rig_2_resolves_to_default_anchor():
    p = "/data/EB-Training-24-1/july_17_batch_2_rig_2/july_17_batch_2_testing_1"
    assert resolve_anchor(p) == DEFAULT_ANCHOR


def test_unknown_path_falls_back_to_default():
    assert resolve_anchor("/tmp/somewhere/else") == DEFAULT_ANCHOR


def test_accepts_path_objects():
    p = Path("/data/x/july_17_batch_2_rig_3/trial")
    assert resolve_anchor(p) == MIRRORED_ANCHOR


def test_deepest_rig_token_wins():
    """A nested path must resolve to the rig closest to the trial."""
    p = "/data/rig_2_archive/july_17_batch_2_rig_3/trial"
    assert resolve_anchor(p) == MIRRORED_ANCHOR
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n yolo-env python -m pytest tests/test_rig_anchor.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'fbpipe.utils.rig_anchor'`

- [ ] **Step 3: Write minimal implementation**

Create `src/fbpipe/utils/rig_anchor.py`:

```python
"""Resolve the geometric anchor for a trial, per recording rig.

The proboscis angle is measured as the unsigned angle between the eye->anchor
and eye->proboscis vectors, where the anchor is a fixed point in the rig (the
odor tube). rig_3 is physically mirrored -- both the flies and the odor tube sit
on the opposite side -- so measuring it against the right-edge anchor used by
the other rigs inverts the extension signal and flips its angle multiplier into
0.5-1.0 where it should be 1.0-2.0.

Adding another mirrored rig later is a one-line edit to ``MIRRORED_RIGS``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

# Right edge, mid-height: the non-mirrored rigs.
DEFAULT_ANCHOR: Tuple[float, float] = (1080.0, 540.0)
# Left edge, mid-height: the mirrored rigs.
MIRRORED_ANCHOR: Tuple[float, float] = (0.0, 540.0)

MIRRORED_RIGS = frozenset({"rig_3"})

# Matches "rig_3" and "rig3" in a path component such as
# "july_17_batch_2_rig_3".
_RIG_RE = re.compile(r"rig_?(\d+)", re.IGNORECASE)


def rig_token(path: str | Path) -> Optional[str]:
    """Return the normalized rig token (e.g. ``"rig_3"``) for *path*.

    Scans path components from the deepest upward so the rig nearest the trial
    wins over any similarly named ancestor. Returns ``None`` when no component
    names a rig.
    """
    for part in reversed(Path(path).parts):
        m = _RIG_RE.search(part)
        if m:
            return f"rig_{m.group(1)}"
    return None


def resolve_anchor(path: str | Path) -> Tuple[float, float]:
    """Return the ``(x, y)`` anchor to measure angles against for *path*.

    Falls back to :data:`DEFAULT_ANCHOR` for unknown or rig-less paths, so
    anything not explicitly mirrored keeps its current behaviour.
    """
    return MIRRORED_ANCHOR if rig_token(path) in MIRRORED_RIGS else DEFAULT_ANCHOR
```

- [ ] **Step 4: Run test to verify it passes**

```bash
conda run -n yolo-env python -m pytest tests/test_rig_anchor.py -v
```

Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fbpipe/utils/rig_anchor.py tests/test_rig_anchor.py
git commit -m "feat(rig_anchor): resolve the geometric anchor per rig

rig_3 is physically mirrored, so its anchor sits on the left edge. Unknown
and rig-less paths fall back to the existing right-edge anchor."
```

---

### Task 2: Thread the anchor through `envelope_combined` (authoritative path)

This module is done first because it **overwrites** `angle_ARB_deg`,
`angle_centered_deg`, and `angle_centered_pct`
(`scripts/analysis/envelope_combined.py:1717-1722`). Fixing only
`compose_videos_rms` would be silently undone here.

**Files:**

- Modify: `scripts/analysis/envelope_combined.py:1493` (`_compute_angle_deg`), `:1588` (`_find_reference_angle`), `:1629` (`_fly_max_centered`), `:1670` (`_ensure_angle_percentages`)
- Test: `tests/test_rig3_anchor_angles.py`

**Interfaces:**

- Consumes: `resolve_anchor`, `DEFAULT_ANCHOR`, `MIRRORED_ANCHOR` from Task 1.
- Produces:
  - `_compute_angle_deg(df: pd.DataFrame, anchor: tuple[float, float] | None = None) -> pd.Series`
  - `_find_reference_angle(csv_paths: Sequence[Path], anchor: tuple[float, float] | None = None) -> float`
  - `_fly_max_centered(csv_paths: Sequence[Path], reference_angle: float, anchor: tuple[float, float] | None = None) -> float`
  - `_ensure_angle_percentages(fly_dir, suffix_globs)` — signature unchanged; resolves the anchor internally from `fly_dir`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_rig3_anchor_angles.py`:

```python
"""The mirrored rig_3 anchor must invert the measured angle.

Flipping the anchor from the right edge to the left edge is equivalent to
``angle -> 180 - angle`` for a fly on the mid-line, which is what restores the
correct sign of the extension->angle relationship for rig_3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.envelope_combined import _compute_angle_deg
from fbpipe.utils.rig_anchor import DEFAULT_ANCHOR, MIRRORED_ANCHOR


def _frame():
    """One fly on the mid-line (y=540) with the proboscis extended right."""
    return pd.DataFrame(
        {
            "x_class0": [500.0],
            "y_class0": [540.0],
            "x_class1": [550.0],
            "y_class1": [540.0],
        }
    )


def test_default_anchor_matches_module_constant_behaviour():
    df = _frame()
    assert np.allclose(
        _compute_angle_deg(df).to_numpy(dtype=float),
        _compute_angle_deg(df, DEFAULT_ANCHOR).to_numpy(dtype=float),
        equal_nan=True,
    )


def test_mirrored_anchor_is_supplement_of_default():
    df = _frame()
    right = _compute_angle_deg(df, DEFAULT_ANCHOR).to_numpy(dtype=float)
    left = _compute_angle_deg(df, MIRRORED_ANCHOR).to_numpy(dtype=float)
    assert np.allclose(left, 180.0 - right, atol=1e-9)


def test_extension_toward_anchor_reads_as_zero_degrees():
    """Proboscis pointing at the anchor is 0 deg; away from it is 180 deg."""
    df = _frame()
    assert _compute_angle_deg(df, DEFAULT_ANCHOR).iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert _compute_angle_deg(df, MIRRORED_ANCHOR).iloc[0] == pytest.approx(180.0, abs=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n yolo-env python -m pytest tests/test_rig3_anchor_angles.py -v
```

Expected: FAIL — `TypeError: _compute_angle_deg() takes 1 positional argument but 2 were given`

- [ ] **Step 3: Add the anchor parameter to `_compute_angle_deg`**

In `scripts/analysis/envelope_combined.py`, change the signature at line 1493 and the vector setup (currently lines 1546-1547):

```python
def _compute_angle_deg(
    df: pd.DataFrame, anchor: tuple[float, float] | None = None
) -> pd.Series:
```

Replace:

```python
    ux = ANCHOR_X - p2x
    uy = ANCHOR_Y - p2y
```

with:

```python
    ax, ay = (ANCHOR_X, ANCHOR_Y) if anchor is None else anchor
    ux = ax - p2x
    uy = ay - p2y
```

- [ ] **Step 4: Run test to verify it passes**

```bash
conda run -n yolo-env python -m pytest tests/test_rig3_anchor_angles.py -v
```

Expected: PASS (3 tests)

- [ ] **Step 5: Thread the anchor through the three callers**

In `_find_reference_angle` (line 1588):

```python
def _find_reference_angle(
    csv_paths: Sequence[Path], anchor: tuple[float, float] | None = None
) -> float:
```

and inside its loop change `_compute_angle_deg(df)` to `_compute_angle_deg(df, anchor)`.

In `_fly_max_centered` (line 1629):

```python
def _fly_max_centered(
    csv_paths: Sequence[Path],
    reference_angle: float,
    anchor: tuple[float, float] | None = None,
) -> float:
```

and inside its loop change `_compute_angle_deg(df)` to `_compute_angle_deg(df, anchor)`.

In `_ensure_angle_percentages` (line 1670), resolve once and pass it down:

```python
def _ensure_angle_percentages(fly_dir: Path, suffix_globs: Iterable[str] | str) -> None:
    csv_paths = _trial_csv_candidates(fly_dir, suffix_globs)
    if not csv_paths:
        return

    anchor = resolve_anchor(fly_dir)
    reference = _find_reference_angle(csv_paths, anchor)
    fly_max = _fly_max_centered(csv_paths, reference, anchor)
```

and in its per-file loop change `angles = _compute_angle_deg(df)` to
`angles = _compute_angle_deg(df, anchor)`.

Add the import near the other `fbpipe` imports at the top of the file:

```python
from fbpipe.utils.rig_anchor import resolve_anchor
```

- [ ] **Step 6: Write the baseline-consistency test**

Append to `tests/test_rig3_anchor_angles.py`:

```python
def test_reference_and_measurement_share_one_anchor(monkeypatch, tmp_path):
    """The baseline and the measurement must be computed with the same anchor.

    A mismatch is silently wrong rather than an error, so pin it with a test.
    """
    from scripts.analysis import envelope_combined as ec

    seen: list[tuple[float, float] | None] = []
    real = ec._compute_angle_deg

    def spy(df, anchor=None):
        seen.append(anchor)
        return real(df, anchor)

    monkeypatch.setattr(ec, "_compute_angle_deg", spy)

    fly_dir = tmp_path / "july_17_batch_2_rig_3"
    fly_dir.mkdir()
    df = pd.DataFrame(
        {
            "frame": [0, 1],
            "x_class0": [500.0, 500.0],
            "y_class0": [540.0, 540.0],
            "x_class1": [550.0, 560.0],
            "y_class1": [540.0, 540.0],
            "distance_percentage_0_1": [10.0, 20.0],
        }
    )
    df.to_parquet(fly_dir / "updated_t1_fly1_distances.parquet")

    monkeypatch.setattr(ec, "_trial_csv_candidates",
                        lambda d, s: [fly_dir / "updated_t1_fly1_distances.parquet"])

    ec._ensure_angle_percentages(fly_dir, "*.parquet")

    assert seen, "_compute_angle_deg was never called"
    assert set(seen) == {MIRRORED_ANCHOR}, f"anchors disagreed: {set(seen)}"
```

- [ ] **Step 7: Run the full test file**

```bash
conda run -n yolo-env python -m pytest tests/test_rig3_anchor_angles.py -v
```

Expected: PASS (4 tests)

- [ ] **Step 8: Commit**

```bash
git add scripts/analysis/envelope_combined.py tests/test_rig3_anchor_angles.py
git commit -m "feat(envelope_combined): resolve the angle anchor per rig

_ensure_angle_percentages resolves the anchor from fly_dir and threads it
through the reference angle, the fly-max scale, and the measurement so all
three agree. Default argument preserves existing behaviour."
```

---

### Task 3: Thread the anchor through `compose_videos_rms`

**Files:**

- Modify: `src/fbpipe/steps/compose_videos_rms.py:214` (`compute_angle_deg_at_point2`), `:327` (`find_fly_reference_angle`), `:400` (`compute_fly_max_abs_centered`), `:792` (`_process_fly_angles`)
- Test: `tests/test_rig3_anchor_compose.py`

**Interfaces:**

- Consumes: `resolve_anchor`, `DEFAULT_ANCHOR`, `MIRRORED_ANCHOR` from Task 1.
- Produces:
  - `compute_angle_deg_at_point2(df: pd.DataFrame, anchor: tuple[float, float] | None = None) -> pd.Series`
  - `find_fly_reference_angle(csvs_raw: List[Path], trimmed_min: Optional[float] = None, anchor: tuple[float, float] | None = None) -> float`
  - `compute_fly_max_abs_centered(csvs_raw: List[Path], ref_angle: float, anchor: tuple[float, float] | None = None) -> float`

- [ ] **Step 1: Write the failing test**

Create `tests/test_rig3_anchor_compose.py`:

```python
"""compose_videos_rms must honour the same per-rig anchor as envelope_combined."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fbpipe.steps.compose_videos_rms import compute_angle_deg_at_point2
from fbpipe.utils.rig_anchor import DEFAULT_ANCHOR, MIRRORED_ANCHOR


def _frame():
    return pd.DataFrame(
        {
            "x_class0": [500.0],
            "y_class0": [540.0],
            "x_class1": [550.0],
            "y_class1": [540.0],
        }
    )


def test_default_argument_preserves_current_values():
    df = _frame()
    assert np.allclose(
        compute_angle_deg_at_point2(df).to_numpy(dtype=float),
        compute_angle_deg_at_point2(df, DEFAULT_ANCHOR).to_numpy(dtype=float),
        equal_nan=True,
    )


def test_mirrored_anchor_is_supplement_of_default():
    df = _frame()
    right = compute_angle_deg_at_point2(df, DEFAULT_ANCHOR).to_numpy(dtype=float)
    left = compute_angle_deg_at_point2(df, MIRRORED_ANCHOR).to_numpy(dtype=float)
    assert np.allclose(left, 180.0 - right, atol=1e-9)


def test_agrees_with_envelope_combined_implementation():
    """The two duplicate implementations must not drift apart."""
    from scripts.analysis.envelope_combined import _compute_angle_deg

    df = _frame()
    for anchor in (DEFAULT_ANCHOR, MIRRORED_ANCHOR):
        a = compute_angle_deg_at_point2(df, anchor).to_numpy(dtype=float)
        b = _compute_angle_deg(df, anchor).to_numpy(dtype=float)
        assert np.allclose(a, b, atol=1e-9, equal_nan=True)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n yolo-env python -m pytest tests/test_rig3_anchor_compose.py -v
```

Expected: FAIL — `TypeError: compute_angle_deg_at_point2() takes 1 positional argument but 2 were given`

- [ ] **Step 3: Add the anchor parameter**

In `src/fbpipe/steps/compose_videos_rms.py`, change the signature at line 214:

```python
def compute_angle_deg_at_point2(
    df: pd.DataFrame, anchor: tuple[float, float] | None = None
) -> pd.Series:
```

Replace:

```python
    ux, uy = (ANCHOR_X - p2x), (ANCHOR_Y - p2y)
```

with:

```python
    ax, ay = (ANCHOR_X, ANCHOR_Y) if anchor is None else anchor
    ux, uy = (ax - p2x), (ay - p2y)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
conda run -n yolo-env python -m pytest tests/test_rig3_anchor_compose.py -v
```

Expected: PASS (3 tests)

- [ ] **Step 5: Thread the anchor through the callers**

In `find_fly_reference_angle` (line 327):

```python
def find_fly_reference_angle(
    csvs_raw: List[Path],
    trimmed_min: Optional[float] = None,
    anchor: tuple[float, float] | None = None,
) -> float:
```

and change `angle = compute_angle_deg_at_point2(df)` to
`angle = compute_angle_deg_at_point2(df, anchor)`.

In `compute_fly_max_abs_centered` (line 405) — currently uncalled anywhere in
the repo, updated only to keep the module internally consistent:

```python
def compute_fly_max_abs_centered(
    csvs_raw: List[Path],
    ref_angle: float,
    anchor: tuple[float, float] | None = None,
) -> float:
```

and change `angle = compute_angle_deg_at_point2(df)` to
`angle = compute_angle_deg_at_point2(df, anchor)`.

In `_process_fly_angles` (line 792), resolve once from `fly_dir`:

```python
    anchor = resolve_anchor(fly_dir)
    reference_angle = find_fly_reference_angle(
        csv_paths, trimmed_min=trimmed_min, anchor=anchor
    )
```

and in the per-file loop change `angles = compute_angle_deg_at_point2(df)` to
`angles = compute_angle_deg_at_point2(df, anchor)`.

Add the import alongside the other `fbpipe` imports:

```python
from ..utils.rig_anchor import resolve_anchor
```

- [ ] **Step 6: Run the existing compose tests for regressions**

```bash
conda run -n yolo-env python -m pytest tests/test_compose_videos_rms.py tests/test_compose_videos_rms_parquet.py tests/test_rig3_anchor_compose.py -v
```

Expected: PASS, no pre-existing test newly failing.

- [ ] **Step 7: Commit**

```bash
git add src/fbpipe/steps/compose_videos_rms.py tests/test_rig3_anchor_compose.py
git commit -m "feat(compose_videos_rms): resolve the angle anchor per rig

_process_fly_angles resolves the anchor from fly_dir and passes it to both
the reference angle and the measurement. Adds a cross-check test so the two
duplicate angle implementations cannot drift apart."
```

---

### Task 4: Correct-on-write for new rig_3 recordings

Without this, every newly recorded rig_3 trial writes an inverted
`angle_deg_c0_c1_vs_anchor` at inference time and needs the Task 5 pass again.

**Files:**

- Modify: `src/fbpipe/steps/yolo_infer.py:547`
- Test: `tests/test_rig3_anchor_yolo.py`

**Interfaces:**

- Consumes: `resolve_anchor` from Task 1.
- Produces: no new public API. `yolo_infer.main` resolves `AX, AY` per video instead of once per run.

- [ ] **Step 1: Write the failing test**

Create `tests/test_rig3_anchor_yolo.py`:

```python
"""yolo_infer must write angles against the per-rig anchor.

The anchor is currently hoisted out of the per-video loop, so a single run
covering both rigs would apply one rig's anchor to the other's videos.
"""
from __future__ import annotations

import inspect

from fbpipe.steps import yolo_infer
from fbpipe.utils.rig_anchor import resolve_anchor


def test_yolo_infer_imports_resolve_anchor():
    src = inspect.getsource(yolo_infer)
    assert "resolve_anchor" in src, "yolo_infer must resolve the anchor per rig"


def test_anchor_is_resolved_inside_the_video_loop():
    """AX, AY must be assigned after the per-video loop starts."""
    src = inspect.getsource(yolo_infer.main)
    assert "for video_path in video_files:" in src
    loop_at = src.index("for video_path in video_files:")
    anchor_at = src.index("resolve_anchor(")
    assert anchor_at > loop_at, "anchor must be resolved per video, not once per run"


def test_rig_3_video_path_resolves_mirrored_anchor():
    p = "/data/EB-Training-24-1/july_17_batch_2_rig_3/output_x.mp4"
    assert resolve_anchor(p) == (0.0, 540.0)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n yolo-env python -m pytest tests/test_rig3_anchor_yolo.py -v
```

Expected: FAIL on `test_yolo_infer_imports_resolve_anchor` — `resolve_anchor` not present.

- [ ] **Step 3: Resolve the anchor per video**

In `src/fbpipe/steps/yolo_infer.py`, add the import beside the other `..utils` imports:

```python
from ..utils.rig_anchor import resolve_anchor
```

Delete the hoisted assignment at line 547:

```python
    AX, AY = cfg.anchor_x, cfg.anchor_y
```

and re-establish it per video, immediately after `base = video_path.stem`
inside the `for video_path in video_files:` loop:

```python
                # rig_3 is physically mirrored, so its anchor is on the other
                # side. Resolved per video because one run can span both rigs.
                AX, AY = resolve_anchor(video_path)
```

**Deliberate side effect, confirm before committing:** this stops `yolo_infer`
reading `cfg.anchor_x`/`cfg.anchor_y` (`1079.0`, `540.0`) and uses
`DEFAULT_ANCHOR` (`1080.0`, `540.0`) instead, collapsing the duplicate-anchor
inconsistency called out in the spec. For non-mirrored rigs this shifts the
anchor by 1 px on a 1080 px frame — an angle change on the order of 0.05°,
numerically negligible. It affects only `angle_deg_c0_c1_vs_anchor` written by
*future* YOLO runs; it cannot alter existing data, and the analysis path
already used `1080.0`. If that 1 px shift is unacceptable, set
`DEFAULT_ANCHOR = (1079.0, 540.0)` in Task 1 instead and re-run Task 1's tests
plus this task's — but then the two anchors still disagree and the spec's
item 5 is not satisfied.

- [ ] **Step 4: Run test to verify it passes**

```bash
conda run -n yolo-env python -m pytest tests/test_rig3_anchor_yolo.py -v
```

Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fbpipe/steps/yolo_infer.py tests/test_rig3_anchor_yolo.py
git commit -m "fix(yolo_infer): resolve the angle anchor per video, per rig

A single run can span rig_2 and rig_3, so the anchor cannot be hoisted out
of the video loop. New rig_3 trials are now correct on first write."
```

---

### Task 5: Recompute existing rig_3 data and validate

> **Post-merge review note:** the invalidation pass below (Steps 6-7's
> write) is **not required** for today's data. `envelope_combined._ensure_angle_percentages`
> has no short-circuit guard on the cached angle columns -- it recomputes
> them unconditionally and rewrites whenever `_series_matches` finds the
> recomputed values differ. Rerunning the analysis for `EB-Training-24-1`
> and `EB-Control-24-1` alone is sufficient to correct rig_3 (verified: corr
> went from -0.269 to +0.269 on a scratch copy with no invalidation pass at
> all). `recompute_rig3_angles.py`'s main value is its `--dry-run` audit,
> which confirms whether any table carries a stale `angle_multiplier` (today
> none of the 474 real rig_3 parquets do) -- see the module docstring.

**Files:**

- Create: `scripts/pipeline/recompute_rig3_angles.py`
- Test: `tests/test_recompute_rig3_angles.py`

**Interfaces:**

- Consumes: `resolve_anchor`, `MIRRORED_RIGS` from Task 1.
- Produces:
  - `find_rig3_tables(roots: Sequence[Path]) -> list[Path]`
  - `invalidate_angle_columns(df: pd.DataFrame) -> pd.DataFrame`
  - `ANGLE_COLUMNS: tuple[str, ...]` = `("angle_ARB_deg", "angle_centered_deg", "angle_centered_pct", "angle_multiplier")`

- [ ] **Step 1: Write the failing test**

Create `tests/test_recompute_rig3_angles.py`:

```python
"""Invalidation pass for cached rig_3 angle columns.

_process_fly_angles and _ensure_angle_percentages both short-circuit when the
cached columns are already present, so they must be dropped before a recompute.
"""
from __future__ import annotations

import pandas as pd

from scripts.pipeline.recompute_rig3_angles import (
    ANGLE_COLUMNS,
    find_rig3_tables,
    invalidate_angle_columns,
)


def test_angle_columns_cover_both_producers():
    assert set(ANGLE_COLUMNS) == {
        "angle_ARB_deg",
        "angle_centered_deg",
        "angle_centered_pct",
        "angle_multiplier",
    }


def test_invalidate_drops_only_angle_columns():
    df = pd.DataFrame(
        {
            "frame": [0, 1],
            "x_class0": [1.0, 2.0],
            "angle_ARB_deg": [10.0, 20.0],
            "angle_centered_deg": [1.0, 2.0],
            "angle_centered_pct": [5.0, 6.0],
            "angle_multiplier": [1.1, 1.2],
        }
    )
    out = invalidate_angle_columns(df)
    assert list(out.columns) == ["frame", "x_class0"]
    # input must not be mutated
    assert "angle_ARB_deg" in df.columns


def test_invalidate_is_safe_when_columns_absent():
    df = pd.DataFrame({"frame": [0], "x_class0": [1.0]})
    assert list(invalidate_angle_columns(df).columns) == ["frame", "x_class0"]


def test_find_rig3_tables_selects_only_rig3(tmp_path):
    r3 = tmp_path / "july_17_batch_2_rig_3" / "trial_1"
    r2 = tmp_path / "july_17_batch_2_rig_2" / "trial_1"
    r3.mkdir(parents=True)
    r2.mkdir(parents=True)
    (r3 / "updated_t_fly1_distances.parquet").write_bytes(b"")
    (r2 / "updated_t_fly1_distances.parquet").write_bytes(b"")

    found = find_rig3_tables([tmp_path])
    assert len(found) == 1
    assert "rig_3" in str(found[0])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n yolo-env python -m pytest tests/test_recompute_rig3_angles.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.pipeline.recompute_rig3_angles'`

- [ ] **Step 3: Write the implementation**

Create `scripts/pipeline/recompute_rig3_angles.py`:

```python
"""Drop cached angle columns from rig_3 tables so they recompute correctly.

rig_3 is a physically mirrored rig and its angles were computed against the
wrong (right-edge) anchor. Both producers of the angle columns short-circuit
when the columns already exist:

  compose_videos_rms._process_fly_angles      -- "if 'angle_multiplier' in df"
  envelope_combined._ensure_angle_percentages -- via _series_matches

so the stale values must be removed before rerunning the analysis. This is a
pure recalculation from the stored eye/proboscis coordinates: no video is
re-encoded and YOLO is not re-run.

Usage:
    python -m scripts.pipeline.recompute_rig3_angles --dry-run ROOT [ROOT ...]
    python -m scripts.pipeline.recompute_rig3_angles ROOT [ROOT ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from fbpipe.utils.rig_anchor import MIRRORED_RIGS, rig_token  # noqa: E402

ANGLE_COLUMNS: tuple[str, ...] = (
    "angle_ARB_deg",
    "angle_centered_deg",
    "angle_centered_pct",
    "angle_multiplier",
)


def find_rig3_tables(roots: Sequence[Path]) -> list[Path]:
    """Return every per-fly distance table living under a mirrored rig."""
    out: list[Path] = []
    for root in roots:
        for path in Path(root).rglob("*_distances.parquet"):
            if rig_token(path) in MIRRORED_RIGS:
                out.append(path)
    return sorted(out)


def invalidate_angle_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with every cached angle column removed."""
    present = [c for c in ANGLE_COLUMNS if c in df.columns]
    return df.drop(columns=present)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would change without writing.")
    args = ap.parse_args(argv)

    tables = find_rig3_tables(args.roots)
    print(f"found {len(tables)} rig_3 tables")

    changed = 0
    for path in tables:
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # a partial/corrupt table must not abort the run
            print(f"  SKIP (unreadable) {path}: {exc}")
            continue
        present = [c for c in ANGLE_COLUMNS if c in df.columns]
        if not present:
            continue
        changed += 1
        if args.dry_run:
            print(f"  would drop {present} from {path}")
            continue
        invalidate_angle_columns(df).to_parquet(path, index=False)
        print(f"  dropped {present} from {path}")

    verb = "would update" if args.dry_run else "updated"
    print(f"{verb} {changed} of {len(tables)} tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
conda run -n yolo-env python -m pytest tests/test_recompute_rig3_angles.py -v
```

Expected: PASS (4 tests)

- [ ] **Step 5: Dry-run against the real trees**

```bash
conda run -n yolo-env python -m scripts.pipeline.recompute_rig3_angles --dry-run \
  /home/ramanlab/Documents/cole/Data/flys_New \
  /securedstorage/DATAsec/cole/Data-secured-New
```

Expected: every reported path contains `rig_3` and sits under `EB-Training-24-1`
or `EB-Control-24-1` (the `-0.11` datasets hold no `*_distances.parquet`).
**No `rig_2` path may appear.** If one does, stop and fix `find_rig3_tables`
before writing anything.

- [ ] **Step 6: Capture the rig_2 regression baseline BEFORE writing anything**

```bash
conda run -n yolo-env python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path("/home/ramanlab/Documents/cole/Data/flys_New")
sums = {}
for p in sorted(root.rglob("*_distances.parquet")):
    if "rig_2" in str(p):
        sums[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
Path("/tmp/rig2_baseline.json").write_text(json.dumps(sums, indent=0))
print(f"hashed {len(sums)} rig_2 tables -> /tmp/rig2_baseline.json")
PY
```

- [ ] **Step 7: Apply the invalidation, then rerun the analysis**

```bash
conda run -n yolo-env python -m scripts.pipeline.recompute_rig3_angles \
  /home/ramanlab/Documents/cole/Data/flys_New \
  /securedstorage/DATAsec/cole/Data-secured-New
```

Then rerun the analysis for the datasets holding rig_3 data. Only two qualify:

```bash
export MPLBACKEND=Agg ORT_LOGGING_LEVEL=3 PYTHONNOUSERSITE=1
for DS in EB-Training-24-1 EB-Control-24-1; do
  conda run -n yolo-env python scripts/pipeline/run_workflows.py \
    --config config/config_new.yaml --folder "$DS"
done
```

**Why only these two.** Eight rig_3 directories exist, but they split as:

| dataset | rig_3 dirs | distance parquets | action |
| --- | --- | --- | --- |
| EB-Training-24-1 | 5 | present | recompute now |
| EB-Control-24-1 | 1 | present | recompute now |
| 3Oct-Training-24-0.11 | 1 | none | nothing to fix |
| 3Oct-Control-24-0.11 | 1 | none | nothing to fix |

The two `-0.11` datasets have been recorded but never processed — zero
`*_distances.parquet` files — so there is no stale angle data to correct. They
are also absent from `config_new.yaml`'s `datasets:` list (which contains
`3Oct-Training-24-0.1` / `3Oct-Control-24-0.1`, different datasets), so
`--folder` on them would be a no-op. When they are eventually processed, Task 4
writes their angles correctly on the first pass — **provided they are added to
`datasets:` first.** Flag this to the user rather than editing the list here;
it is outside this change.

Both `EB-Training-24-1` and `EB-Control-24-1` are deliberately left LIVE (not
frozen) per the comment at `config_new.yaml:194-196`, so the frozen-dataset
skip added to `_run_combined` does not suppress this recompute. Confirm the run
log does **not** print `[FROZEN] combined.combine → skipping recompute` for
either dataset; if it does, the recompute silently did nothing.

Each dataset also contains rig_2 directories, which is expected and safe:
`resolve_anchor` returns the identical `DEFAULT_ANCHOR` for rig_2 as before
this branch, so the recomputed angle arrays are bitwise-identical to the
cached ones and `_ensure_angle_percentages` writes nothing because
`_series_matches` finds them equal. (`_process_fly_angles` in
`compose_videos_rms` is not part of this path -- it is not wired into
`ORDERED_STEPS` and is not called by `pipeline.py` or `run_workflows.py`. Its
"short-circuits on `angle_multiplier` in df.columns" behavior does not apply
here; when run manually against a real rig_2 copy it in fact rewrote 60
files and changed `angle_centered_deg`, because it re-centers on compose's
own reference angle, which differs from envelope's. Do not rely on it for
rig_2 safety.) That is exactly what Step 8 verifies.

YOLO does not re-run: `config_new.yaml` sets `force.yolo: false`, and
`yolo_infer` skips any video whose output directory already exists
(`yolo_infer.py:588`). This step recomputes from stored coordinates only.

- [ ] **Step 8: Verify rig_2 is byte-identical**

```bash
conda run -n yolo-env python - <<'PY'
import hashlib, json
from pathlib import Path
sums = json.loads(Path("/tmp/rig2_baseline.json").read_text())
bad = [p for p, h in sums.items()
       if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest() != h]
print("rig_2 tables changed:", len(bad))
for p in bad[:10]:
    print("  ", p)
assert not bad, "rig_2 output changed -- the fix leaked outside rig_3"
print("OK: rig_2 byte-identical")
PY
```

Expected: `rig_2 tables changed: 0` and `OK: rig_2 byte-identical`.

- [ ] **Step 9: Verify rig_3 is corrected**

```bash
conda run -n yolo-env python - <<'PY'
import numpy as np, pandas as pd
from pathlib import Path
base = Path("/home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1")
for rig in ("rig_2", "rig_3"):
    cs = []
    for f in sorted((base / f"july_17_batch_2_{rig}").glob("*/*_fly*_distances.parquet"))[:20]:
        df = pd.read_parquet(f)
        if not {"angle_ARB_deg", "distance_percentage_0_1"} <= set(df.columns):
            continue
        a = pd.to_numeric(df["angle_ARB_deg"], errors="coerce").to_numpy(float)
        d = pd.to_numeric(df["distance_percentage_0_1"], errors="coerce").to_numpy(float)
        m = np.isfinite(a) & np.isfinite(d)
        if m.sum() > 50:
            cs.append(np.corrcoef(d[m], a[m])[0, 1])
    print(f"{rig}: corr(extension, angle) = {np.mean(cs):+.3f}  (n={len(cs)})")
PY
```

Expected: **both rigs report the same sign** (rig_2 near `+0.054`, rig_3 flipped from `-0.220` to roughly `+0.220`). A negative rig_3 value means the recompute did not take effect.

- [ ] **Step 10: Commit**

```bash
git add scripts/pipeline/recompute_rig3_angles.py tests/test_recompute_rig3_angles.py
git commit -m "feat(recompute_rig3_angles): invalidate cached rig_3 angle columns

Both angle producers short-circuit on the cached columns, so stale rig_3
values must be dropped before the corrected anchor can take effect. Pure
recalculation from stored coordinates: no re-encode, no YOLO re-run."
```

---

### Task 6: Full suite and branch wrap-up

**Files:** none modified.

- [ ] **Step 1: Run the full test suite**

```bash
conda run -n yolo-env python -m pytest tests/ -q
```

Expected: no failures introduced by this branch. Record any pre-existing failure explicitly rather than assuming it is unrelated.

- [ ] **Step 2: Review the complete diff**

```bash
git diff main...feature/rig3-mirrored-anchor --stat
```

Expected files: `src/fbpipe/utils/rig_anchor.py`, `src/fbpipe/steps/compose_videos_rms.py`, `src/fbpipe/steps/yolo_infer.py`, `scripts/analysis/envelope_combined.py`, `scripts/pipeline/recompute_rig3_angles.py`, plus four test files and the spec. Nothing else.

- [ ] **Step 3: Report results to the user**

State plainly: which tests pass, the rig_2 byte-identity result, and the before/after correlation for rig_3. Do not claim success without the Step 8 and Step 9 output from Task 5.
