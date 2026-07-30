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

