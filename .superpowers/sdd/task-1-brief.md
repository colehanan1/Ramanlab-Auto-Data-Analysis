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

