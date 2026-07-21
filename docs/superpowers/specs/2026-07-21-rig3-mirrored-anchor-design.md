# Per-rig anchor for the mirrored rig_3 (angle multiplier fix)

**Date:** 2026-07-21
**Status:** approved, pending implementation

## Problem

Rig 3 is physically mirrored relative to the other rigs: both the flies and the
odor tube (the anchor the angle is measured against) sit on the opposite side.
The angle calculation uses a single hardcoded right-edge anchor for every rig,
so rig_3's proboscis-extension signal comes out inverted and its
`angle_multiplier` lands in 0.5–1.0 where it should be 1.0–2.0.

Scope of affected data (survey of both trees):

| rig | working tree | secured |
| --- | --- | --- |
| rig_2 (non-mirrored) | 92 dirs | 92 dirs |
| rig_3 (mirrored) | 8 dirs | 6 dirs |

No `rig_1` exists in either tree, so `rig_2` is the non-mirrored reference
throughout this document.

Of the 8 rig_3 directories only 6 hold `*_distances.parquet` data, all under
`EB-Training-24-1` (5) and `EB-Control-24-1` (1). The remaining two live in
`3Oct-Training-24-0.11` / `3Oct-Control-24-0.11`, which have been recorded but
never processed and are absent from the config's `datasets:` list — they have
no stale angle data to correct, and will be written correctly on first
processing once the correct-on-write change is in place.

### Mechanism

`compute_angle_deg_at_point2` (`src/fbpipe/steps/compose_videos_rms.py:214`)
computes an **unsigned** angle between the eye→anchor and eye→proboscis vectors:

```python
ux, uy = (ANCHOR_X - p2x), (ANCHOR_Y - p2y)   # eye -> anchor
vx, vy = (p3x - p2x), (p3y - p2y)             # eye -> proboscis
angles = degrees(arctan2(abs(cross), dot))     # 0..180, unsigned
```

with `ANCHOR_X, ANCHOR_Y = 1080.0, 540.0` hardcoded at module scope
(`compose_videos_rms.py:57`).

The multiplier is then derived from the angle *centered on the fly's own
resting baseline*:

```python
centered_angles = angles - reference_angle
multipliers     = compute_angle_multiplier_series(centered_angles)  # -> [0.5, 2.0]
```

In the non-mirrored rigs the flies face away from the anchor, so extending the
proboscis raises the angle above the resting baseline → positive deviation →
multiplier > 1. For the mirrored rig_3 the same physical extension *lowers* the
angle → negative deviation → multiplier < 1. The signal is inverted, not merely
offset.

### Evidence

Measured over 20 fly-trials per rig from `EB-Training-24-1/july_17_batch_2_*`:

| Measurement | rig_2 | rig_3 |
| --- | --- | --- |
| mean proboscis dx (+right / −left) | +2.9 px | −3.6 px |
| corr(extension, angle), **right** anchor (current) | +0.054 | **−0.220** |
| corr(extension, angle), **left** anchor (proposed) | −0.054 | **+0.220** |
| recomputed vs stored `angle_ARB_deg` | max diff 0.0000° | max diff 0.0000° |

Two conclusions follow. The sign of the extension→angle relationship is
inverted for rig_3 and flipping the anchor restores it. And because a
recomputation from the stored `x_class0/y_class0/x_class1/y_class1` columns
reproduces the stored `angle_ARB_deg` exactly, **the fix requires no
reprocessing** — no video re-encode, no YOLO re-run.

Flipping the anchor is equivalent to `angle → 180 − angle` for a fly near the
vertical centre, which is why the correlation negates cleanly.

## Design

### 1. Anchor resolution

A single helper resolves a trial/fly path to the anchor for its rig. Rig
membership is read from a config mapping rather than a hardcoded path
substring, so a future rig_4 is a config edit and not a code change.

```yaml
# config_new.yaml
rig_anchors:
  default: [1080.0, 540.0]
  rig_3:   [0.0, 540.0]
```

The helper detects the rig token (`_rig_<n>`) in the path, looks it up, and
falls back to `default`. An unrecognised path yields `default`, so behaviour is
unchanged for anything not explicitly mirrored.

### 2. Thread the anchor through the angle calculation

`compute_angle_deg_at_point2(df, anchor=None)` gains an optional anchor
parameter defaulting to the existing module constants, keeping every current
caller working unchanged.

All three call sites must pass the *same* resolved anchor:

- `find_fly_reference_angle` (`compose_videos_rms.py:348`)
- the RMS angle pass (`compose_videos_rms.py:408`)
- `_process_fly_angles` (`compose_videos_rms.py:841`)

`find_fly_reference_angle` is the critical one: it establishes the per-fly
resting baseline that the measurement is centered against. If the baseline and
the measurement use different anchors the centering is meaningless.

### 3. Recompute existing rig_3 data

`_process_fly_angles` short-circuits when the column already exists
(`compose_videos_rms.py:836`):

```python
if "angle_multiplier" in df.columns:
    continue
```

Recomputation therefore needs the cached columns invalidated for rig_3 tables:
`angle_ARB_deg`, `angle_centered_deg`, `angle_centered_pct`, `angle_multiplier`.
This is a pure recalculation from stored coordinates — 8 rig_3 directories, no
GPU, no video decode.

### 4. Correct-on-write for future runs

`yolo_infer` writes `angle_deg_c0_c1_vs_anchor` at inference time from
`cfg.anchor_x/anchor_y` (`yolo_infer.py:547`). It resolves the anchor per rig
from the same helper so newly recorded rig_3 trials are correct on first write
and need no post-hoc pass.

### 5. Collapse the duplicate anchor definition

The anchor is currently defined twice and the two disagree: `cfg.anchor_x =
1079.0` (config, used during inference) versus `ANCHOR_X = 1080.0` (hardcoded,
used downstream). The 1 px difference is numerically negligible but the
duplication is a live trap. Both paths resolve through the single helper, with
the config as the source of truth.

## Data flow

```text
trial path ──► resolve_anchor(path) ──► (ax, ay)
                                         │
                    ┌────────────────────┼─────────────────────┐
                    ▼                    ▼                     ▼
        find_fly_reference_angle   compute_angle_deg_    yolo_infer
        (baseline)                 at_point2 (measure)   (write-time)
                    └────────────► centered_angles ◄─────┘
                                         │
                                         ▼
                              compute_angle_multiplier_series
                                    angle_multiplier
```

## Error handling

- Unknown or missing rig token → `default` anchor; existing behaviour preserved.
- Malformed `rig_anchors` block (non-list, wrong arity, non-numeric) → log a
  warning and fall back to `default` rather than crashing a long pipeline run.
- Recompute pass skips tables lacking the coordinate columns, logging the file,
  matching the existing `[ANGLES]` failure style.

## Testing

1. **Unit** — `compute_angle_deg_at_point2` with an explicit mirrored anchor
   returns `180 − angle` on a symmetric fixture; the default argument path
   reproduces today's values exactly.
2. **Anchor resolution** — `_rig_3` paths resolve to the mirrored anchor,
   `_rig_2`/unknown/malformed paths to the default.
3. **Regression (the important one)** — rig_2 outputs are byte-identical before
   and after the change. Nothing but rig_3 may move.
4. **Baseline consistency** — a test that fails if `find_fly_reference_angle`
   and the measurement are given different anchors.
5. **End-to-end validation** — after the recompute, `corr(extension, angle)`
   for rig_3 is positive and comparable in sign to rig_2, reproducing the
   evidence table above.

## Out of scope

- Mirroring or re-encoding any video.
- Re-running YOLO inference on existing data.
- The separate `_angle_multiplier` implementation in
  `scripts/analysis/per_polygon_distance.py`, which has its own semantics and
  is not part of the main pipeline path.
- The YOLO tracking-corruption investigation (transcode/YOLO race), tracked
  separately.
