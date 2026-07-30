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

