# PER gate rejection figure — design

**Date:** 2026-07-30
**Purpose:** A single methods figure for the master's thesis defense showing how the
tracking pipeline rejects physically impossible proboscis measurements.

## Goal

One figure, five rules. A real fly whose PER runs close to the acceptance boundary
carries the argument; a constructed bad detection shows what rejection looks like.
The figure must be readable from the back of a room, so text is kept to panel
titles, two anatomical direction words, and the gate values themselves. No sentence
appears anywhere inside the figure; everything else lives in the caption.

## The five rules and where they live in the code

All five are already implemented. The figure reports them; it does not restate them.

| Rule | Implementation | Value |
|---|---|---|
| Proboscis detections capped at the number of resolved flies, highest confidence kept | `_limit_proboscis_detections`, `src/fbpipe/steps/yolo_infer.py:118-127` (`argsort(-scores)[:limit]`) | limit = confirmed eye anchors |
| Anisotropic spatial gate around each eye | `anisotropic_semi_axes`, `src/fbpipe/utils/distance_sanity.py:64-76` | 160 px lateral and ventral, `160 / up_divisor` = **40 px** dorsal |
| With ≥3 flies, eye–proboscis pairings beyond the limit are rejected and the binding released | `_max_valid_eye_prob_distance_px`, `src/fbpipe/steps/yolo_infer.py:145-160`; release at `pairer.eye_to_cls8[eye_id] = None`, `:319` | `three_fly_max_eye_prob_distance_px: 160.0` |
| Displacement gate against the last *accepted* position | `sanitize_proboscis_velocity_dataframe`, `src/fbpipe/utils/distance_sanity.py:136-174` | `max_jump_px: 80.0` |
| Only eye→proboscis distances in range contribute to each fly's normalization | `distance_limits`, `config/config_new.yaml:415-418` | `class2_min: 10.0`, `class2_max: 160.0` |

### Config provenance

These values come from `config/config_new.yaml`, which is the operative config. The
repo defaults (`config.yaml`, `example.yaml`, and the dataclass defaults in
`src/fbpipe/config.py`) carry different numbers — 150 / 180 / 250. The figure script
must **load the gate constants from `config/config_new.yaml` at runtime and never
hardcode them**, so the figure cannot silently disagree with the pipeline. The
caption cites `config_new.yaml` by name.

## Subject

`3Oct-Control-24-0.1 / july_26_batch_1 / july_26_batch_1_testing_1 / fly1`

- Odor 3-Octonol, testing trial, 2 flies in the arena
- **3605 / 3605 frames detected — no gaps**
- Frozen eye anchor at (845, 183) in 1080×1080 space
- Peak PER at **frame 1102**: `dx +28.9, dy +142.8`, r = **145.8 px**, gate norm 0.830
- 220 frames past half-gate, 7 past 0.8 of the gate
- Purely ventral: `dy` spans +90.8 → +142.8, `|dx| ≤ 32.1`, never dorsal
- Max frame-to-frame displacement 5.7 px

Selected from a scan of 2029 fly-trials that still have their source video. It is the
closest-to-boundary **odor testing** trial in the set and the only near-edge candidate
with a perfect detection record.

Source video: `output_july_26_batch_1_testing_1_3-Octonol_20260726_161155.mp4` under
`/securedstorage/DATAsec/cole/Data-secured-New/3Oct-Control-24-0.1/july_26_batch_1/`.
It is the raw camera recording, 1080×1080, 3605 frames, 40 fps. The sibling
`*_distance_annotated.mp4` is the pipeline's own overlay and is **not** used.
Detection coordinates are already in 1080×1080 space (`yolo_infer.py:603`), so no
rescaling is needed.

### What the data actually shows

Across every candidate examined, PER extension is a nearly vertical ventral
excursion. The dorsal allowance is almost never approached and the lateral allowance
never is. This is the figure's real argument: the gate is generous exactly where PER
goes and tight where anatomy forbids it. The asymmetric shape is self-justifying, and
the figure should let the shape make that point rather than asserting it in text.

## Composition

Hero panel at left (~62% of width), four small gate panels stacked at right.

```text
┌──────────────────────────────────────┬──────────────┐
│              dorsal                  │ B  cap = 2   │
│      ┌──────── 40 ────────┐          │  ●.93 ●.88 ✓ │
│    ┌─┘                    └─┐        │  ○.71 ○.40 ✗ │
│    │        ⊕ eye           │        ├──────────────┤
│  160        ╎               160  ✗   │ C  ≥3 flies  │
│    │        ╎ ▒▒▒           │   220  │  ⊕╌╌╌✗ prob  │
│     ╲       ╎▒▒▒▒▒         ╱         │      160     │
│      ╲      ●146 px      ╱           ├──────────────┤
│       └───── 160 ───────┘            │ D  jump      │
│              ventral                 │ ●─●─● ✗ 80   │
│      ACCEPTANCE BOUNDARY             ├──────────────┤
│                                      │ E  normalize │
│                                      │ 10▐████▌160  │
└──────────────────────────────────────┴──────────────┘
```

## Panel A — hero

Crop `x ∈ [645, 1045], y ∈ [63, 383]` of frame 1102, centred so the full gate plus
margin is visible. Verified to fit inside the 1080×1080 frame.

**Frame treatment:** grayscale, contrast-stretch (1st–99.5th percentile), invert,
then blend 40% toward white. This is a contrast requirement, not a style choice, and
the recipe is dictated by the footage.

The raw crop is near-black IR video — mean 17, min 5, max 79. The originally
specified treatment (grayscale then blend 60% toward white) was implemented and
rendered, and it compressed the entire image into 14 grey levels, 5.5% of full
scale: a blank pale rectangle with no visible fly. It passed every threshold it was
given, because those thresholds measured paleness and never measured whether any
structure survived.

The stretch restores dynamic range; the inversion puts the fly dark-on-pale, which is
the correct polarity for coloured overlay marks on a light surface. Final result:
mean 199.3, range 153 levels, with head, eye and extended proboscis clearly visible.
The palette validator WARNs at every mid-gray surface tested — orange falls to 1.61:1
against `#b8b8b6` — so the pale background is what preserves the overlay palette's
contrast guarantees.

Because the displayed image is inverted relative to the raw recording, the caption
must say so.

**Boundary:** traced by calling `anisotropic_boundary_offsets(max_px, up_divisor, n=360)`
from `src/fbpipe/utils/distance_sanity.py:78`. This is the production drawing
function that already exists for exactly this purpose. The figure cannot drift from
the implementation because it calls the implementation.

| Mark | Encoding |
|---|---|
| Frozen eye anchor | violet `#4a3aa7`, ⊕ crosshair, 10 px |
| Accepted cloud (all 3605 detections) | blue `#2a78d6`, 3 px, α 0.10 — reads as the narrow ventral column |
| Peak PER, frame 1102 | blue filled circle, 11 px, 2 px white ring, thin eye→proboscis line, direct label `146 px` |
| Constructed rejection | orange `#eb6834`, hollow ✗, 13 px, 2.5 px stroke, white ring, dashed lead line from the eye, direct label `220 px` |
| Boundary | 2.5 px primary ink with a 4 px white halo beneath |

**Text in this panel:** the words `dorsal` and `ventral`, the title
`ACCEPTANCE BOUNDARY`, the three gate numbers `160` / `160` / `40`, and the two mark
labels. Nothing else.

The constructed rejection is placed outside the boundary in the lateral-ventral
quadrant at exactly **eye + (185, 120)**, i.e. (1030, 303) in frame coordinates.
That gives r = 220.5 px, labelled `220 px`, and a gate norm of
`(185/160)² + (120/160)² = 1.90`, comfortably outside the boundary.

## Panels B–E

Each panel gets a two-or-three-word title. Numbers appear only where they are the
point of the panel: the gate value being illustrated, and — in panel B only — the
four detection confidences, since ranking by confidence *is* that panel's rule.

- **B — detection cap.** Four proboscis detections ordered by confidence. The top two
  filled blue with ✓, the bottom two hollow gray with ✗. Title carries the real
  count for this trial: `cap = 2 flies`.
- **C — ≥3 flies, binding released.** Two eyes and one proboscis. The over-distance
  pairing is drawn as a dashed orange line terminating in ✗, labelled `160`. Shows
  the binding being dropped rather than stretched.
- **D — displacement gate.** Four short accepted steps followed by one long orange ✗
  step. An 80 px ring is drawn around the **last accepted** point, not around the
  rejected one — this is what makes visible the rule that a rejected point never
  becomes the new reference.
- **E — normalization window.** A 0→200 px axis with the 10–160 band shaded and this
  fly's real r spread (p01 94.4, p99 126.7, max 145.8) drawn inside it.

## Color

**Blue `#2a78d6` = accepted. Orange `#eb6834` = rejected. Violet `#4a3aa7` = eye anchor.**

Validated with the dataviz skill's validator under `--pairs all --mode light`:

```text
[PASS] Lightness band      all 3 inside L 0.43–0.77
[PASS] Chroma floor        all 3 >= 0.1
[PASS] CVD separation      worst #4a3aa7↔#2a78d6 ΔE 13.0 (deutan) · tritan 17.4
[PASS] Normal-vision floor worst #4a3aa7↔#2a78d6 ΔE 16.3
[PASS] Contrast vs surface all 3 >= 3:1
```

A green-accepted / red-rejected palette was tested first and **FAILED**: worst pair
`#d03b3b ↔ #0ca30c` at deutan ΔE 4.1. This is the identical failure that ruled out
the red→green score palette, so it is not revisited.

Rejection is encoded by **shape first** — a hollow ✗ against filled circles — with
color as the secondary channel, so no meaning depends on hue alone. Every mark
carrying a status also carries a glyph and a direct label.

## Honesty

Panels B, C and D are schematic. This fly has 2 flies rather than 3, and its real
maximum frame-to-frame displacement is only 5.7 px, so neither the pairing-release
rule nor the displacement gate is exercised by its data. Those three panels get a
hairline dashed border, and the caption states once:

> Blue marks are measured from this fly; orange ✗ marks are constructed rejections.
> The video frame is contrast-stretched and shown inverted.

The hero panel's accepted cloud and panel E are real measured data. Only the single
✗ in the hero panel is constructed.

## Deliverables

`scripts/analysis/per_gate_rejection_figure.py`, following the conventions already
used by `scripts/analysis/yolo_geometry_figure.py`.

Outputs to `figures/`:

- `per_gate_rejection.png` — 300 dpi
- `per_gate_rejection.pdf` — vector
- `per_gate_rejection.svg` — editable text, matching the treatment used for the PID
  figures

Default canvas 13.33 × 7.5 in for a 16:9 slide, overridable by flag.

## Testing

Tests are written and run before the figure code, per the project workflow rule.
`tests/test_per_gate_rejection_figure.py`:

1. **The constructed rejection is genuinely rejected.** Feed the invented ✗ point
   through the real `sanitize_eye_prob_geometry_dataframe` and assert it is blanked.
   This guarantees the figure shows a rejection the model would actually make.
2. **Every plotted accepted point survives.** Feed the plotted cloud through the same
   function and assert nothing is blanked.
3. **Gate constants come from config.** Assert the script reads 160 / 4.0 / 80 / 10 /
   160 from `config/config_new.yaml` rather than from literals in the script.
4. **The boundary is the production boundary.** Assert the traced boundary comes from
   `anisotropic_boundary_offsets` and that its extreme offsets match 160 lateral,
   160 ventral, 40 dorsal.
5. **Subject data is as described.** Assert the chosen fly's detection count, peak
   frame and peak radius match the values in this spec, so the figure fails loudly if
   the underlying parquet is ever reprocessed.
6. **Outputs are produced.** All three files written and non-empty.

Tests that need the source video or secured storage skip cleanly when those are
unavailable, so the suite still runs on a machine without the data mounted.
