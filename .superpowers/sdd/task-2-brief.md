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

