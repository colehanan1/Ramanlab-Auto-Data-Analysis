# PER Gate Rejection Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single methods figure for the master's thesis defense showing the five gates that reject physically impossible proboscis measurements, anchored on a real video frame of a fly whose PER runs close to the acceptance boundary.

**Architecture:** One standalone script, `scripts/analysis/per_gate_rejection_figure.py`, following the conventions already used by `scripts/analysis/yolo_geometry_figure.py` and `scripts/analysis/pid_odor_figures.py`. It loads gate constants from `config/config_new.yaml` at runtime, loads one fly's real detections from parquet, pulls one raw video frame, and traces the acceptance boundary by calling the production function `anisotropic_boundary_offsets` rather than re-deriving it. A hero panel carries the spatial gate; four small panels carry the remaining four rules.

**Tech Stack:** Python 3.12, matplotlib, numpy, pandas, OpenCV (`cv2`), pytest. All already in use in this repo.

**Spec:** `docs/superpowers/specs/2026-07-30-per-gate-rejection-figure-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **Gate constants are never hardcoded.** They are read at runtime from `config/config_new.yaml`: `proboscis_filter.max_eye_prob_distance_px: 160.0`, `proboscis_filter.up_divisor: 4.0`, `proboscis_filter.max_jump_px: 80.0`, `distance_limits.class2_min: 10.0`, `distance_limits.class2_max: 160.0`. The repo defaults in `config.yaml` and `src/fbpipe/config.py` carry *different* numbers (150 / 180 / 250) and must not be used.
- **The boundary is traced by the production function.** Call `fbpipe.utils.distance_sanity.anisotropic_boundary_offsets(max_px, up_divisor, n)`. Do not re-implement the ellipse maths in the figure script.
- **Palette is fixed and validated.** Accepted `#2a78d6` (blue), rejected `#eb6834` (orange), eye anchor `#4a3aa7` (violet), primary ink `#0b0b0b`, muted ink `#898781`, surface `#fcfcfb`. Validated all-pairs light mode: worst CVD ΔE 13.0, normal-vision 16.3, all ≥3:1. **Never substitute a green/red accept/reject pair** — it fails at deutan ΔE 4.1.
- **Rejection is encoded by shape first.** Hollow ✗ markers against filled circles; colour is the secondary channel only. Every status mark also carries a direct label.
- **No sentence appears inside the figure.** Only panel titles, the words `dorsal` and `ventral`, gate values, and short mark labels. Prose belongs in the caption.
- **Subject is fixed:** `3Oct-Control-24-0.1 / july_26_batch_1 / july_26_batch_1_testing_1 / fly1`, eye anchor at (845, 183), peak PER at frame 1102, 3605/3605 frames detected.
- **Outputs:** `figures/per_gate_rejection.png` (300 dpi), `.pdf` (vector), `.svg` (editable text via `svg.fonttype: "none"`). Default canvas 13.33 × 7.5 in.
- **Frame treatment (AMENDED during Task 4).** The raw footage is near-black (mean 17, max 79). The originally planned "grayscale then blend 60% toward white" flattens it to 14 grey levels — a blank rectangle that passed every threshold it was given. The approved treatment is: grayscale → contrast-stretch (p1–p99.5) → INVERT → blend 40% toward white, giving mean 199.3 and 153 levels of range with the fly dark on a pale surface. The thresholds are `percentile(img, 90) > 230` (surface is light, so the palette's contrast holds) and `percentile(img, 99) - percentile(img, 1) > 100` (the fly is actually visible). The old naive treatment scores 14 on the second, so it genuinely discriminates.
- **Tests needing the parquet or the video must skip cleanly** when those paths are absent, so the suite still runs on a machine without the data mounted.
- Run tests with `python -m pytest` from the repo root (`tests/conftest.py` puts the repo root on `sys.path`, which is how `from scripts.analysis... import ...` resolves).

---

## File Structure

**Create: `scripts/analysis/per_gate_rejection_figure.py`**

One file, sectioned in this order. It stays a single module to match the repo's one-script-per-figure convention (`yolo_geometry_figure.py`, `pid_odor_figures.py`).

| Section | Responsibility |
|---|---|
| constants | palette, paths, `SUBJECT`, `CROP`, `REJECTED_OFFSET`, rcParams |
| `GateSettings` + `load_gate_settings` | read the five gate values from `config_new.yaml` |
| `load_subject_offsets` | read the fly's real (dx, dy) detections from parquet |
| `gate_boundary_offsets`, `gate_norm` | geometry, delegating to the production function |
| `load_frame_crop` | pull frame 1102, crop, grayscale, ghost |
| `draw_hero` | panel A |
| `draw_cap_panel`, `draw_release_panel`, `draw_jump_panel`, `draw_norm_panel` | panels B–E |
| `make_figure`, `_save`, `_parse_args`, `main` | assembly and CLI |

**Create: `tests/test_per_gate_rejection_figure.py`** — one test module covering all seven tasks.

---

### Task 1: Gate settings loaded from config, never hardcoded

**Files:**
- Create: `scripts/analysis/per_gate_rejection_figure.py`
- Test: `tests/test_per_gate_rejection_figure.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `GateSettings` dataclass with float fields `max_px`, `up_divisor`, `max_jump_px`, `norm_min_px`, `norm_max_px`, and property `dorsal_px` (= `max_px / up_divisor`). Function `load_gate_settings(config_path: Path | str = CONFIG_PATH) -> GateSettings`. Module constants `REPO_ROOT: Path`, `CONFIG_PATH: Path`, `OUT_DIR: Path`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_per_gate_rejection_figure.py`:

```python
"""Tests for the PER gate rejection methods figure.

The gate values asserted here are the ones the thesis text cites, and they live
in ``config/config_new.yaml`` -- NOT in the repo defaults (``config.yaml`` and
``src/fbpipe/config.py`` carry 150 / 180 / 250). The figure must read them at
runtime so it can never silently disagree with the pipeline.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.analysis.per_gate_rejection_figure import (
    CONFIG_PATH,
    GateSettings,
    load_gate_settings,
)


def test_gate_settings_match_config_new() -> None:
    """The five gate values the figure reports come from config_new.yaml."""
    s = load_gate_settings()
    assert s.max_px == 160.0
    assert s.up_divisor == 4.0
    assert s.dorsal_px == 40.0
    assert s.max_jump_px == 80.0
    assert s.norm_min_px == 10.0
    assert s.norm_max_px == 160.0


def test_config_path_points_at_config_new() -> None:
    assert CONFIG_PATH.name == "config_new.yaml"
    assert CONFIG_PATH.exists()


def test_gate_settings_are_not_hardcoded(tmp_path: Path) -> None:
    """Feeding a different config must change the values -- proving they are read,
    not baked into the script."""
    alt = tmp_path / "alt.yaml"
    alt.write_text(
        "proboscis_filter:\n"
        "  max_eye_prob_distance_px: 99.0\n"
        "  up_divisor: 3.0\n"
        "  max_jump_px: 55.0\n"
        "distance_limits:\n"
        "  class2_min: 5.0\n"
        "  class2_max: 99.0\n",
        encoding="utf-8",
    )
    s = load_gate_settings(alt)
    assert s.max_px == 99.0
    assert s.up_divisor == 3.0
    assert s.dorsal_px == 33.0
    assert s.max_jump_px == 55.0
    assert s.norm_min_px == 5.0
    assert s.norm_max_px == 99.0


def test_gate_settings_rejects_missing_keys(tmp_path: Path) -> None:
    """A config without the gate blocks must fail loudly, not silently default."""
    empty = tmp_path / "empty.yaml"
    empty.write_text("{}\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_gate_settings(empty)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_per_gate_rejection_figure.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.analysis.per_gate_rejection_figure'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/analysis/per_gate_rejection_figure.py`:

```python
"""Methods figure: rejecting physically impossible proboscis measurements.

One hero panel (a real video frame with the acceptance boundary traced on it)
plus four small panels, covering the five gates the pipeline applies:

  cap        proboscis detections capped at the number of resolved flies,
             highest confidence kept        (yolo_infer._limit_proboscis_detections)
  spatial    anisotropic gate around each eye, 160 px lateral/ventral,
             40 px dorsal                   (distance_sanity.anisotropic_semi_axes)
  release    with >=3 flies, pairings beyond 160 px are dropped and the
             binding released               (yolo_infer._max_valid_eye_prob_distance_px)
  jump       accepted positions must be within 80 px of the previous
             ACCEPTED position              (distance_sanity.sanitize_proboscis_velocity_dataframe)
  normalize  only 10-160 px contributes to each fly's normalization range
                                            (config_new.yaml distance_limits)

Gate values are read from config/config_new.yaml at runtime; the boundary is
traced by the production function, not re-derived here.

Run:
    python scripts/analysis/per_gate_rejection_figure.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(REPO_ROOT), str(SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fbpipe.config import load_raw_config  # noqa: E402

CONFIG_PATH = REPO_ROOT / "config" / "config_new.yaml"
OUT_DIR = REPO_ROOT / "figures"


@dataclass(frozen=True)
class GateSettings:
    """The five gate values the figure reports, as read from the config."""

    max_px: float
    up_divisor: float
    max_jump_px: float
    norm_min_px: float
    norm_max_px: float

    @property
    def dorsal_px(self) -> float:
        """Upward (dorsal) allowance -- the tightened semi-axis."""
        return self.max_px / self.up_divisor


def load_gate_settings(config_path: Path | str = CONFIG_PATH) -> GateSettings:
    """Read the gate constants from *config_path*.

    Raises KeyError if a gate block is missing: a figure that silently fell back
    to defaults would print numbers the pipeline does not use.
    """
    raw = load_raw_config(config_path)
    try:
        pf = raw["proboscis_filter"]
        dl = raw["distance_limits"]
        return GateSettings(
            max_px=float(pf["max_eye_prob_distance_px"]),
            up_divisor=float(pf["up_divisor"]),
            max_jump_px=float(pf["max_jump_px"]),
            norm_min_px=float(dl["class2_min"]),
            norm_max_px=float(dl["class2_max"]),
        )
    except KeyError as exc:
        raise KeyError(
            f"{config_path} is missing gate settings: {exc}. "
            "The figure must not fall back to repo defaults (150/180/250)."
        ) from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_per_gate_rejection_figure.py -v`
Expected: PASS — 4 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/analysis/per_gate_rejection_figure.py tests/test_per_gate_rejection_figure.py
git commit -m "feat(figure): load PER gate constants from config_new.yaml"
```

---

### Task 2: Load the subject fly's real detections

**Files:**
- Modify: `scripts/analysis/per_gate_rejection_figure.py`
- Test: `tests/test_per_gate_rejection_figure.py`

**Interfaces:**
- Consumes: `GateSettings` from Task 1.
- Produces: `Subject` dataclass (fields `dataset: str`, `trial_rel: str`, `slot: str`, `odor: str`, `eye_xy: tuple[float, float]`, `peak_frame: int`, `video_name: str`); module constants `SUBJECT: Subject`, `DATA_ROOT: Path`, `VIDEO_ROOT: Path`. Function `subject_parquet_path(subject: Subject = SUBJECT) -> Path`, `subject_video_path(subject: Subject = SUBJECT) -> Path`, and `load_subject_offsets(subject: Subject = SUBJECT) -> Offsets` where `Offsets` is a dataclass with `dx: np.ndarray`, `dy: np.ndarray`, `frames: np.ndarray`, `eye_xy: tuple[float, float]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_per_gate_rejection_figure.py`:

```python
import numpy as np

from scripts.analysis.per_gate_rejection_figure import (
    SUBJECT,
    Offsets,
    load_subject_offsets,
    subject_parquet_path,
    subject_video_path,
)

requires_data = pytest.mark.skipif(
    not subject_parquet_path().exists(),
    reason="subject parquet not mounted on this machine",
)


def test_subject_identity() -> None:
    """The subject is pinned so the figure caption cannot drift from the data."""
    assert SUBJECT.dataset == "3Oct-Control-24-0.1"
    assert SUBJECT.trial_rel == "july_26_batch_1/july_26_batch_1_testing_1"
    assert SUBJECT.slot == "fly1"
    assert SUBJECT.odor == "3-Octonol"
    assert SUBJECT.eye_xy == (845.0, 183.0)
    assert SUBJECT.peak_frame == 1102


@requires_data
def test_subject_offsets_match_spec() -> None:
    """Golden numbers from the candidate scan. If the parquet is ever
    reprocessed, this fails loudly rather than the figure quietly changing."""
    off = load_subject_offsets()
    assert len(off.dx) == 3605, "subject was chosen for its perfect detection record"
    assert len(off.dx) == len(off.dy) == len(off.frames)

    ex, ey = off.eye_xy
    assert round(ex) == 845 and round(ey) == 183

    r = np.hypot(off.dx, off.dy)
    assert r.max() == pytest.approx(145.8, abs=0.1)

    peak_i = int(np.argmax(r))
    assert off.frames[peak_i] == SUBJECT.peak_frame
    assert off.dx[peak_i] == pytest.approx(28.9, abs=0.1)
    assert off.dy[peak_i] == pytest.approx(142.8, abs=0.1)


@requires_data
def test_subject_per_is_ventral() -> None:
    """The figure's argument: PER is a near-vertical ventral excursion, so the
    gate is generous ventrally and tight dorsally."""
    off = load_subject_offsets()
    assert off.dy.min() > 0, "this fly never goes dorsal"
    assert np.abs(off.dx).max() < 40.0
    assert off.dy.max() > 140.0


def test_subject_video_path_is_the_raw_recording() -> None:
    """The '*_distance_annotated.mp4' sibling is the pipeline's own overlay and
    must not be used -- we draw our own."""
    path = subject_video_path()
    assert path.name.startswith("output_")
    assert "distance_annotated" not in path.name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_per_gate_rejection_figure.py -v`
Expected: FAIL — `ImportError: cannot import name 'SUBJECT'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/analysis/per_gate_rejection_figure.py`, after `load_gate_settings`:

```python
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fbpipe.utils.columns import (  # noqa: E402
    find_eye_xy_columns,
    find_proboscis_xy_columns,
)
from fbpipe.utils.tables import read_table  # noqa: E402

DATA_ROOT = Path("/home/ramanlab/Documents/cole/Data/flys_New")
VIDEO_ROOT = Path("/securedstorage/DATAsec/cole/Data-secured-New")


@dataclass(frozen=True)
class Subject:
    """The one fly the hero panel zooms in on.

    Chosen from a scan of 2029 fly-trials that still have their source video: it
    is the closest-to-boundary *odor testing* trial in the set, and the only
    near-edge candidate with a perfect detection record (3605/3605).
    """

    dataset: str
    trial_rel: str
    slot: str
    odor: str
    eye_xy: tuple[float, float]
    peak_frame: int
    video_name: str


SUBJECT = Subject(
    dataset="3Oct-Control-24-0.1",
    trial_rel="july_26_batch_1/july_26_batch_1_testing_1",
    slot="fly1",
    odor="3-Octonol",
    eye_xy=(845.0, 183.0),
    peak_frame=1102,
    video_name="output_july_26_batch_1_testing_1_3-Octonol_20260726_161155.mp4",
)


@dataclass(frozen=True)
class Offsets:
    """Accepted eye->proboscis offsets, in pixels, for one fly-trial."""

    dx: np.ndarray
    dy: np.ndarray
    frames: np.ndarray
    eye_xy: tuple[float, float]


def subject_parquet_path(subject: Subject = SUBJECT) -> Path:
    trial_name = subject.trial_rel.split("/")[-1]
    return (
        DATA_ROOT
        / subject.dataset
        / subject.trial_rel
        / f"{trial_name}_{subject.slot}_distances.parquet"
    )


def subject_video_path(subject: Subject = SUBJECT) -> Path:
    batch_dir = subject.trial_rel.split("/")[0]
    return VIDEO_ROOT / subject.dataset / batch_dir / subject.video_name


def load_subject_offsets(subject: Subject = SUBJECT) -> Offsets:
    """Read the fly's accepted proboscis positions as offsets from its frozen eye."""
    df = read_table(subject_parquet_path(subject))
    ex_col, ey_col = find_eye_xy_columns(df)
    px_col, py_col = find_proboscis_xy_columns(df)
    if not (ex_col and ey_col and px_col and py_col):
        raise ValueError(f"missing eye/proboscis columns in {subject_parquet_path(subject)}")

    ex = pd.to_numeric(df[ex_col], errors="coerce").to_numpy(float)
    ey = pd.to_numeric(df[ey_col], errors="coerce").to_numpy(float)
    px = pd.to_numeric(df[px_col], errors="coerce").to_numpy(float)
    py = pd.to_numeric(df[py_col], errors="coerce").to_numpy(float)

    dx, dy = px - ex, py - ey
    ok = np.isfinite(dx) & np.isfinite(dy)
    return Offsets(
        dx=dx[ok],
        dy=dy[ok],
        frames=np.flatnonzero(ok),
        eye_xy=(float(np.nanmedian(ex)), float(np.nanmedian(ey))),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_per_gate_rejection_figure.py -v`
Expected: PASS — 8 passed (or 5 passed, 3 skipped if the data is not mounted)

- [ ] **Step 5: Commit**

```bash
git add scripts/analysis/per_gate_rejection_figure.py tests/test_per_gate_rejection_figure.py
git commit -m "feat(figure): load subject fly detections from parquet"
```

---

### Task 3: Gate geometry, and prove the constructed rejection is real

This is the honesty-critical task. The invented ✗ mark must be a rejection the actual production gate would make.

**Files:**
- Modify: `scripts/analysis/per_gate_rejection_figure.py`
- Test: `tests/test_per_gate_rejection_figure.py`

**Interfaces:**
- Consumes: `GateSettings`, `Offsets`, `load_subject_offsets` from Tasks 1–2.
- Produces: module constant `REJECTED_OFFSET: tuple[float, float]`; functions `gate_boundary_offsets(settings: GateSettings, n: int = 360) -> np.ndarray` (returns shape `(n, 2)` of dx/dy), `gate_norm(dx, dy, settings: GateSettings) -> np.ndarray`, and `offsets_survive_geometry_gate(dx, dy, settings) -> np.ndarray[bool]` which runs the real `sanitize_eye_prob_geometry_dataframe`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_per_gate_rejection_figure.py`:

```python
from scripts.analysis.per_gate_rejection_figure import (
    REJECTED_OFFSET,
    gate_boundary_offsets,
    gate_norm,
    offsets_survive_geometry_gate,
)


def test_boundary_extremes_match_the_gate() -> None:
    """160 px lateral, 160 px ventral, 40 px dorsal -- traced by the production
    function, not re-derived here."""
    s = load_gate_settings()
    pts = gate_boundary_offsets(s, n=720)
    dx, dy = pts[:, 0], pts[:, 1]

    assert dx.max() == pytest.approx(160.0, abs=0.5)
    assert dx.min() == pytest.approx(-160.0, abs=0.5)
    assert dy.max() == pytest.approx(160.0, abs=0.5), "ventral (dy > 0) is generous"
    assert dy.min() == pytest.approx(-40.0, abs=0.5), "dorsal (dy < 0) is tightened"


def test_boundary_comes_from_the_production_function(monkeypatch) -> None:
    """If someone re-implements the ellipse maths locally, this fails."""
    import scripts.analysis.per_gate_rejection_figure as mod

    called = {}

    def spy(max_px, up_divisor, n=72):
        called["args"] = (max_px, up_divisor, n)
        return [(0.0, 0.0)]

    monkeypatch.setattr(mod, "anisotropic_boundary_offsets", spy)
    gate_boundary_offsets(load_gate_settings(), n=123)
    assert called["args"] == (160.0, 4.0, 123)


def test_constructed_rejection_is_genuinely_rejected() -> None:
    """The invented X mark must be a rejection the real model would make.

    This is what keeps the figure honest: the point is fed through the actual
    production gate, not merely drawn outside a line we chose.
    """
    s = load_gate_settings()
    dx, dy = REJECTED_OFFSET

    assert gate_norm(np.array([dx]), np.array([dy]), s)[0] > 1.0

    survives = offsets_survive_geometry_gate(np.array([dx]), np.array([dy]), s)
    assert not survives[0], "the constructed bad detection must be blanked by the gate"

    r = float(np.hypot(dx, dy))
    assert r == pytest.approx(220.5, abs=0.2), "label reads '220 px'"


@requires_data
def test_every_plotted_accepted_point_survives_the_gate() -> None:
    """The blue cloud must contain nothing the gate would have removed."""
    s = load_gate_settings()
    off = load_subject_offsets()
    survives = offsets_survive_geometry_gate(off.dx, off.dy, s)
    assert survives.all()
    assert gate_norm(off.dx, off.dy, s).max() < 1.0


@requires_data
def test_peak_per_is_near_the_boundary_but_inside() -> None:
    """The whole point of this fly: it rides the edge without crossing it."""
    s = load_gate_settings()
    off = load_subject_offsets()
    norms = gate_norm(off.dx, off.dy, s)
    assert 0.80 < norms.max() < 1.0
    assert norms.max() == pytest.approx(0.830, abs=0.005)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_per_gate_rejection_figure.py -v`
Expected: FAIL — `ImportError: cannot import name 'REJECTED_OFFSET'`

- [ ] **Step 3: Write minimal implementation**

Add the import near the other `fbpipe` imports:

```python
from fbpipe.utils.distance_sanity import (  # noqa: E402
    anisotropic_boundary_offsets,
    anisotropic_semi_axes,
    sanitize_eye_prob_geometry_dataframe,
)
```

Then add after `load_subject_offsets`:

```python
# The constructed bad detection, as an offset from the eye. Lateral-ventral
# quadrant, well outside the gate: r = 220.5 px, gate norm 1.90. Task 3's tests
# push this through the real production gate to prove it is genuinely rejected.
REJECTED_OFFSET: tuple[float, float] = (185.0, 120.0)


def gate_boundary_offsets(settings: GateSettings, n: int = 360) -> np.ndarray:
    """Trace the acceptance boundary as (n, 2) dx/dy offsets from the eye.

    Delegates to the production drawing function so the figure cannot drift from
    the implementation.
    """
    pts = anisotropic_boundary_offsets(settings.max_px, settings.up_divisor, n)
    return np.asarray(pts, dtype=float)


def gate_norm(dx, dy, settings: GateSettings) -> np.ndarray:
    """Gate-normalised radius. 1.0 is exactly on the boundary; > 1.0 is rejected."""
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    a, b = anisotropic_semi_axes(dx, dy, settings.max_px, settings.up_divisor)
    return (dx / a) ** 2 + (dy / b) ** 2


def offsets_survive_geometry_gate(dx, dy, settings: GateSettings) -> np.ndarray:
    """Run offsets through the REAL production geometry gate.

    Returns a boolean mask: True where the point survives, False where the
    pipeline would blank it. Used to guarantee the figure's rejected example is
    a rejection the model actually makes.
    """
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    frame = pd.DataFrame(
        {
            "x_class0": np.zeros_like(dx),
            "y_class0": np.zeros_like(dy),
            "x_class1": dx,
            "y_class1": dy,
        }
    )
    cleaned, _ = sanitize_eye_prob_geometry_dataframe(
        frame, settings.max_px, settings.up_divisor
    )
    return pd.to_numeric(cleaned["x_class1"], errors="coerce").notna().to_numpy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_per_gate_rejection_figure.py -v`
Expected: PASS — 13 passed (or 10 passed, 3 skipped without data)

- [ ] **Step 5: Commit**

```bash
git add scripts/analysis/per_gate_rejection_figure.py tests/test_per_gate_rejection_figure.py
git commit -m "feat(figure): trace gate from production fn, verify rejected example"
```

---

### Task 4: Frame crop and ghosting

**Files:**
- Modify: `scripts/analysis/per_gate_rejection_figure.py`
- Test: `tests/test_per_gate_rejection_figure.py`

**Interfaces:**
- Consumes: `SUBJECT`, `subject_video_path` from Task 2.
- Produces: module constants `CROP: tuple[int, int, int, int]` (x0, y0, x1, y1) and `GHOST_BLEND: float`; function `load_frame_crop(video_path: Path, frame_index: int, crop: tuple[int, int, int, int] = CROP, ghost: float = GHOST_BLEND) -> np.ndarray` returning an RGB `uint8` array of shape `(y1 - y0, x1 - x0, 3)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_per_gate_rejection_figure.py`:

```python
from scripts.analysis.per_gate_rejection_figure import (
    CROP,
    GHOST_BLEND,
    load_frame_crop,
)

requires_video = pytest.mark.skipif(
    not subject_video_path().exists(),
    reason="subject video not mounted on this machine",
)


def test_crop_contains_the_whole_gate() -> None:
    """The crop must show the full boundary plus margin, and stay inside the
    1080x1080 frame."""
    s = load_gate_settings()
    ex, ey = SUBJECT.eye_xy
    x0, y0, x1, y1 = CROP

    assert 0 <= x0 < x1 <= 1080
    assert 0 <= y0 < y1 <= 1080
    assert x0 <= ex - s.max_px and x1 >= ex + s.max_px
    assert y0 <= ey - s.dorsal_px and y1 >= ey + s.max_px
    # and room for the constructed rejection
    rdx, rdy = REJECTED_OFFSET
    assert x1 >= ex + rdx and y1 >= ey + rdy


@requires_video
def test_frame_crop_shape_and_type() -> None:
    x0, y0, x1, y1 = CROP
    img = load_frame_crop(subject_video_path(), SUBJECT.peak_frame)
    assert img.shape == (y1 - y0, x1 - x0, 3)
    assert img.dtype == np.uint8


@requires_video
def test_frame_crop_is_grayscale() -> None:
    """Ghosting is grayscale: the three channels must be identical."""
    img = load_frame_crop(subject_video_path(), SUBJECT.peak_frame)
    assert np.array_equal(img[:, :, 0], img[:, :, 1])
    assert np.array_equal(img[:, :, 1], img[:, :, 2])


@requires_video
def test_frame_crop_is_ghosted_pale_enough_for_the_palette() -> None:
    """A raw photographic background voids the palette's contrast guarantees --
    orange falls to 1.61:1 on mid-gray. Ghosting restores a light surface."""
    img = load_frame_crop(subject_video_path(), SUBJECT.peak_frame)
    assert img.mean() > 200, "crop must read as a pale surface, not a photo"
    assert img.min() > 120, "even the darkest pixel stays well clear of mid-gray"


@requires_video
def test_ghost_blend_zero_returns_the_unghosted_frame() -> None:
    """Sanity check that the ghosting parameter actually does the work."""
    raw = load_frame_crop(subject_video_path(), SUBJECT.peak_frame, ghost=0.0)
    ghosted = load_frame_crop(subject_video_path(), SUBJECT.peak_frame)
    assert GHOST_BLEND > 0
    assert ghosted.mean() > raw.mean()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_per_gate_rejection_figure.py -v`
Expected: FAIL — `ImportError: cannot import name 'CROP'`

- [ ] **Step 3: Write minimal implementation**

Add `import cv2  # noqa: E402` with the other third-party imports, then add after `offsets_survive_geometry_gate`:

```python
# Crop window around the frozen eye at (845, 183): the full gate (160 px lateral
# and ventral, 40 px dorsal) plus margin, and room for the constructed rejection
# at eye + (185, 120). Verified to sit inside the 1080x1080 frame.
CROP: tuple[int, int, int, int] = (645, 63, 1045, 383)

# Blend fraction toward white. Not stylistic: the palette validator WARNs at
# every mid-gray surface tested (orange falls to 1.61:1 on #b8b8b6), so a raw
# photographic background would void the contrast guarantees.
GHOST_BLEND = 0.60


def load_frame_crop(
    video_path: Path,
    frame_index: int,
    crop: tuple[int, int, int, int] = CROP,
    ghost: float = GHOST_BLEND,
) -> np.ndarray:
    """Pull one frame, crop it, grayscale it, and blend it toward white.

    Returns an RGB uint8 array so matplotlib can draw it directly.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise ValueError(f"cannot read frame {frame_index} of {video_path}")

    x0, y0, x1, y1 = crop
    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(float)
    pale = gray + (255.0 - gray) * float(ghost)
    return np.repeat(np.clip(pale, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_per_gate_rejection_figure.py -v`
Expected: PASS — 18 passed (or fewer with skips)

If `test_frame_crop_is_ghosted_pale_enough_for_the_palette` fails because the arena is darker than expected, raise `GHOST_BLEND` until it passes — do not weaken the assertion, because the assertion is the contrast guarantee.

- [ ] **Step 5: Commit**

```bash
git add scripts/analysis/per_gate_rejection_figure.py tests/test_per_gate_rejection_figure.py
git commit -m "feat(figure): extract and ghost the hero video frame"
```

---

### Task 5: Hero panel

**Files:**
- Modify: `scripts/analysis/per_gate_rejection_figure.py`
- Test: `tests/test_per_gate_rejection_figure.py`

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: palette constants `ACCEPTED`, `REJECTED`, `EYE`, `INK`, `MUTED`, `SURFACE` (all `str` hex); `HERO_TEXTS: frozenset[str]`; function `draw_hero(ax, offsets: Offsets, settings: GateSettings, image: np.ndarray | None) -> None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_per_gate_rejection_figure.py`:

```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analysis.per_gate_rejection_figure import (  # noqa: E402
    ACCEPTED,
    EYE,
    HERO_TEXTS,
    REJECTED,
    draw_hero,
)


def test_palette_is_the_validated_one() -> None:
    """Validated all-pairs, light mode: worst CVD dE 13.0, normal-vision 16.3.
    Green/red was tested first and FAILED at deutan dE 4.1 -- never reinstate it.
    """
    assert ACCEPTED == "#2a78d6"
    assert REJECTED == "#eb6834"
    assert EYE == "#4a3aa7"
    for hexcode in (ACCEPTED, REJECTED, EYE):
        assert hexcode.lower() not in {"#0ca30c", "#d03b3b", "#008300", "#e34948"}


def test_hero_contains_no_sentences() -> None:
    """The figure carries panel titles, two direction words, gate values and
    short mark labels -- never prose. Longest permitted label is 3 words."""
    for text in HERO_TEXTS:
        assert len(text.split()) <= 3, f"too wordy for a slide: {text!r}"
    assert "dorsal" in HERO_TEXTS
    assert "ventral" in HERO_TEXTS


def _hero_axes(image=None):
    s = load_gate_settings()
    off = Offsets(
        dx=np.array([5.0, 10.0, 28.9]),
        dy=np.array([95.0, 110.0, 142.8]),
        frames=np.array([0, 1, 1102]),
        eye_xy=SUBJECT.eye_xy,
    )
    fig, ax = plt.subplots()
    draw_hero(ax, off, s, image)
    return fig, ax


def test_hero_draws_exactly_the_expected_text() -> None:
    fig, ax = _hero_axes()
    drawn = {t.get_text() for t in ax.texts if t.get_text()}
    assert drawn == set(HERO_TEXTS)
    plt.close(fig)


def test_hero_labels_the_acceptance_boundary_and_both_marks() -> None:
    fig, ax = _hero_axes()
    drawn = {t.get_text() for t in ax.texts}
    assert any("ACCEPTANCE" in t.upper() for t in drawn)
    assert "146 px" in drawn, "the accepted peak PER is direct-labelled"
    assert "220 px" in drawn, "the rejected example is direct-labelled"
    plt.close(fig)


def test_hero_gate_numbers_come_from_settings() -> None:
    """Change the config, and the numbers on the boundary change with it."""
    fig, ax = _hero_axes()
    drawn = {t.get_text() for t in ax.texts}
    s = load_gate_settings()
    assert str(int(s.max_px)) in drawn
    assert str(int(s.dorsal_px)) in drawn
    plt.close(fig)


def test_hero_rejected_marker_is_hollow() -> None:
    """Rejection is encoded by SHAPE first; colour is secondary, so nothing
    depends on hue alone."""
    fig, ax = _hero_axes()
    marks = [ln for ln in ax.lines if ln.get_marker() in {"x", "X"}]
    assert marks, "the rejected example must be an X marker"
    assert any(m.get_markerfacecolor() in ("none", "None") for m in marks)
    plt.close(fig)


def test_hero_axis_is_equal_aspect() -> None:
    """Pixels are square; an unequal aspect would misrepresent the gate shape."""
    fig, ax = _hero_axes()
    assert ax.get_aspect() == 1.0
    plt.close(fig)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_per_gate_rejection_figure.py -v`
Expected: FAIL — `ImportError: cannot import name 'ACCEPTED'`

- [ ] **Step 3: Write minimal implementation**

Add near the top of the module, after the path bootstrap:

```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402

# Validated all-pairs (light mode): worst CVD dE 13.0, normal-vision 16.3, all
# >= 3:1. A green/red accept/reject pair was tested first and FAILED at deutan
# dE 4.1 -- the same failure that ruled out the red/green score palette.
ACCEPTED = "#2a78d6"   # blue
REJECTED = "#eb6834"   # orange
EYE = "#4a3aa7"        # violet
INK = "#0b0b0b"
MUTED = "#898781"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "text.color": INK,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,       # editable text in Illustrator / Inkscape
        "ps.fonttype": 42,
        "svg.fonttype": "none",   # keep <text> as text, not paths, in the SVG
    }
)

# Every string the hero panel is allowed to draw. The test asserts this set
# exactly, which is what keeps the slide from accreting prose.
HERO_TEXTS = frozenset(
    {"dorsal", "ventral", "ACCEPTANCE BOUNDARY", "146 px", "220 px", "160", "40"}
)
```

Then add the drawing function after `load_frame_crop`:

```python
def _halo(width: float = 3.0):
    """White relief so a mark stays legible over the ghosted frame."""
    return [withStroke(linewidth=width, foreground="white")]


def draw_hero(ax, offsets: Offsets, settings: GateSettings, image: np.ndarray | None) -> None:
    """Panel A: the acceptance boundary drawn over a real frame, in eye-centred px."""
    ex, ey = offsets.eye_xy
    x0, y0, x1, y1 = CROP

    if image is not None:
        ax.imshow(image, extent=(x0 - ex, x1 - ex, y1 - ey, y0 - ey),
                  interpolation="bilinear", zorder=0)

    # accepted cloud -- all real detections
    ax.scatter(offsets.dx, offsets.dy, s=9, c=ACCEPTED, alpha=0.10,
               linewidths=0, zorder=2)

    # the acceptance boundary, traced by the production function
    pts = gate_boundary_offsets(settings, n=360)
    ring = np.vstack([pts, pts[:1]])
    ax.plot(ring[:, 0], ring[:, 1], color=INK, lw=2.5, zorder=4,
            path_effects=_halo(5.0))

    # frozen eye anchor
    ax.plot([0], [0], marker="P", ms=11, color=EYE, mew=0, zorder=6,
            path_effects=_halo())

    # peak PER -- the accepted detection nearest the boundary
    r = np.hypot(offsets.dx, offsets.dy)
    pk = int(np.argmax(r))
    pdx, pdy = float(offsets.dx[pk]), float(offsets.dy[pk])
    ax.plot([0, pdx], [0, pdy], color=ACCEPTED, lw=1.6, zorder=5)
    ax.plot([pdx], [pdy], marker="o", ms=11, color=ACCEPTED, mew=2.0,
            mec="white", zorder=7)
    ax.text(pdx + 12, pdy, "146 px", color=ACCEPTED, fontsize=12,
            fontweight="bold", va="center", ha="left", path_effects=_halo())

    # constructed rejection -- verified against the real gate in the tests
    rdx, rdy = REJECTED_OFFSET
    ax.plot([0, rdx], [0, rdy], color=REJECTED, lw=1.6, ls=(0, (4, 3)), zorder=5)
    ax.plot([rdx], [rdy], marker="X", ms=15, mfc="none", mec=REJECTED, mew=3.0,
            zorder=7, path_effects=_halo())
    ax.text(rdx, rdy + 16, "220 px", color=REJECTED, fontsize=12,
            fontweight="bold", va="top", ha="center", path_effects=_halo())

    # gate values, on the boundary itself
    lat, dorsal = settings.max_px, settings.dorsal_px
    ax.text(lat + 6, 0, str(int(lat)), color=INK, fontsize=12, va="center",
            ha="left", path_effects=_halo())
    ax.text(0, lat + 6, str(int(lat)), color=INK, fontsize=12, va="top",
            ha="center", path_effects=_halo())
    ax.text(0, -dorsal - 6, str(int(dorsal)), color=INK, fontsize=12,
            va="bottom", ha="center", path_effects=_halo())

    ax.text(0, -dorsal - 34, "dorsal", color=MUTED, fontsize=11, va="bottom",
            ha="center", style="italic")
    ax.text(0, lat + 34, "ventral", color=MUTED, fontsize=11, va="top",
            ha="center", style="italic")
    ax.text(0.02, 0.02, "ACCEPTANCE BOUNDARY", transform=ax.transAxes,
            color=INK, fontsize=13, fontweight="bold", va="bottom", ha="left")

    ax.set_xlim(x0 - ex, x1 - ex)
    ax.set_ylim(y1 - ey, y0 - ey)   # image convention: +dy is ventral, downward
    ax.set_aspect(1.0)
    ax.axis("off")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_per_gate_rejection_figure.py -v`
Expected: PASS — 25 passed (or fewer with skips)

- [ ] **Step 5: Commit**

```bash
git add scripts/analysis/per_gate_rejection_figure.py tests/test_per_gate_rejection_figure.py
git commit -m "feat(figure): draw hero acceptance-boundary panel"
```

---

### Task 6: Side panels B–E

**Files:**
- Modify: `scripts/analysis/per_gate_rejection_figure.py`
- Test: `tests/test_per_gate_rejection_figure.py`

**Interfaces:**
- Consumes: `GateSettings`, `Offsets`, palette constants.
- Produces: `SCHEMATIC_PANELS: frozenset[str]`; functions `draw_cap_panel(ax, n_flies: int) -> None`, `draw_release_panel(ax, settings) -> None`, `draw_jump_panel(ax, settings) -> dict` (returns `{"ring_centre": tuple[float, float], "ring_radius": float}`), `draw_norm_panel(ax, offsets: Offsets, settings) -> dict` (returns `{"band": tuple[float, float]}`), and `_mark_schematic(ax) -> None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_per_gate_rejection_figure.py`:

```python
from scripts.analysis.per_gate_rejection_figure import (
    SCHEMATIC_PANELS,
    draw_cap_panel,
    draw_jump_panel,
    draw_norm_panel,
    draw_release_panel,
)


def test_schematic_panels_are_declared() -> None:
    """This fly has 2 flies and a max jump of 5.7 px, so the cap, release and
    jump panels are illustrations -- they must be flagged, not passed off as data."""
    assert SCHEMATIC_PANELS == frozenset({"cap", "release", "jump"})
    assert "norm" not in SCHEMATIC_PANELS, "panel E plots this fly's real spread"


def test_jump_ring_is_centred_on_the_last_ACCEPTED_point() -> None:
    """The rule the panel exists to show: a rejected point never becomes the new
    reference, so the 80 px ring stays on the last accepted position."""
    s = load_gate_settings()
    fig, ax = plt.subplots()
    info = draw_jump_panel(ax, s)
    assert info["ring_radius"] == s.max_jump_px

    # The DRAWN circle must use the same value, or the test proves nothing about
    # the picture. Panel D works in px data coordinates for exactly this reason.
    circles = [p for p in ax.patches if isinstance(p, plt.Circle)]
    assert len(circles) == 1
    assert circles[0].get_radius() == pytest.approx(s.max_jump_px)
    assert circles[0].center == pytest.approx(info["ring_centre"])

    # The ring must sit on the LAST ACCEPTED point, never on the rejected one.
    # Asserting the exact coordinates is what makes this test able to fail.
    assert info["ring_centre"] == pytest.approx((130.0, 114.0))
    assert info["ring_centre"] != pytest.approx((250.0, 112.0))
    plt.close(fig)


def test_norm_panel_band_edges_come_from_settings() -> None:
    s = load_gate_settings()
    off = Offsets(
        dx=np.array([0.0, 0.0]), dy=np.array([94.4, 145.8]),
        frames=np.array([0, 1]), eye_xy=SUBJECT.eye_xy,
    )
    fig, ax = plt.subplots()
    info = draw_norm_panel(ax, off, s)
    assert info["band"] == (s.norm_min_px, s.norm_max_px) == (10.0, 160.0)
    drawn = {t.get_text() for t in ax.texts}
    assert "10" in drawn and "160" in drawn
    plt.close(fig)


def test_release_panel_shows_the_three_fly_limit() -> None:
    s = load_gate_settings()
    fig, ax = plt.subplots()
    draw_release_panel(ax, s)
    drawn = {t.get_text() for t in ax.texts}
    assert str(int(s.max_px)) in drawn
    marks = [ln for ln in ax.lines if ln.get_marker() in {"x", "X"}]
    assert marks, "the released binding terminates in an X"
    plt.close(fig)


def test_cap_panel_keeps_the_highest_confidence_detections() -> None:
    """Two flies resolved -> the two highest-confidence proboscis detections are
    kept and the rest dropped."""
    fig, ax = plt.subplots()
    draw_cap_panel(ax, n_flies=2)
    drawn = {t.get_text() for t in ax.texts}
    assert "2-fly cap" in drawn
    kept = [ln for ln in ax.lines if ln.get_marker() == "o"
            and ln.get_markerfacecolor() not in ("none", "None")]
    dropped = [ln for ln in ax.lines if ln.get_marker() in {"x", "X"}]
    assert len(kept) == 2
    assert len(dropped) == 2
    plt.close(fig)


def test_every_side_panel_title_is_short() -> None:
    """Minimal-text rule applies to the side strip too."""
    s = load_gate_settings()
    off = Offsets(dx=np.array([0.0]), dy=np.array([100.0]),
                  frames=np.array([0]), eye_xy=SUBJECT.eye_xy)
    for draw in (
        lambda ax: draw_cap_panel(ax, 2),
        lambda ax: draw_release_panel(ax, s),
        lambda ax: draw_jump_panel(ax, s),
        lambda ax: draw_norm_panel(ax, off, s),
    ):
        fig, ax = plt.subplots()
        draw(ax)
        for t in ax.texts:
            assert len(t.get_text().split()) <= 3, f"too wordy: {t.get_text()!r}"
        plt.close(fig)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_per_gate_rejection_figure.py -v`
Expected: FAIL — `ImportError: cannot import name 'SCHEMATIC_PANELS'`

- [ ] **Step 3: Write minimal implementation**

Add after `draw_hero`:

```python
# Panels whose marks are illustrations rather than this fly's measurements. The
# subject has 2 flies (so the >=3-fly release never fires) and a max real
# displacement of 5.7 px (so the 80 px gate is never approached). These get a
# dashed border, and the caption says so once.
SCHEMATIC_PANELS = frozenset({"cap", "release", "jump"})


def _mark_schematic(ax) -> None:
    """Hairline dashed border: this panel is an illustration, not measured data."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linestyle((0, (3, 3)))
        spine.set_linewidth(0.8)
        spine.set_edgecolor(MUTED)


def _panel_title(ax, text: str) -> None:
    ax.text(0.0, 1.0, text, transform=ax.transAxes, color=INK, fontsize=11,
            fontweight="bold", va="bottom", ha="left")


def draw_cap_panel(ax, n_flies: int) -> None:
    """Panel B: detections ranked by confidence, capped at the resolved fly count."""
    confidences = [0.93, 0.88, 0.71, 0.40]
    for i, conf in enumerate(confidences):
        x = 0.12 + 0.25 * i
        keep = i < n_flies
        if keep:
            ax.plot([x], [0.55], marker="o", ms=13, color=ACCEPTED, mew=2.0,
                    mec="white")
        else:
            ax.plot([x], [0.55], marker="X", ms=13, mfc="none", mec=REJECTED,
                    mew=2.5)
        ax.text(x, 0.28, f"{conf:.2f}",
                color=ACCEPTED if keep else REJECTED,
                fontsize=10, ha="center", va="top")

    _panel_title(ax, f"{n_flies}-fly cap")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    _mark_schematic(ax)


def draw_release_panel(ax, settings: GateSettings) -> None:
    """Panel C: with >=3 flies, an over-distance pairing is dropped, not stretched."""
    ax.plot([0.15], [0.55], marker="P", ms=12, color=EYE, mew=0)
    ax.plot([0.85], [0.55], marker="P", ms=12, color=EYE, mew=0)
    ax.plot([0.42], [0.55], marker="o", ms=11, color=ACCEPTED, mew=2.0, mec="white")

    ax.plot([0.15, 0.42], [0.55, 0.55], color=ACCEPTED, lw=1.8)
    ax.plot([0.85, 0.56], [0.55, 0.55], color=REJECTED, lw=1.8, ls=(0, (4, 3)))
    ax.plot([0.55], [0.55], marker="X", ms=13, mfc="none", mec=REJECTED, mew=2.5)

    ax.text(0.70, 0.30, str(int(settings.max_px)), color=REJECTED, fontsize=11,
            ha="center", va="top")
    _panel_title(ax, "≥3 flies")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    _mark_schematic(ax)


def draw_jump_panel(ax, settings: GateSettings) -> dict:
    """Panel D: the displacement gate, measured from the last ACCEPTED position.

    Drawn in PIXEL data coordinates so the ring radius on screen is literally
    ``settings.max_jump_px``. Returns the ring geometry so the tests can assert
    both that the drawn circle uses the gate value and that it is centred on the
    last accepted point rather than on the rejected one.
    """
    accepted_x = [40.0, 70.0, 100.0, 130.0]
    accepted_y = [110.0, 118.0, 106.0, 114.0]
    ax.plot(accepted_x, accepted_y, color=ACCEPTED, lw=1.8, marker="o", ms=9,
            mew=1.6, mec="white")

    last_x, last_y = accepted_x[-1], accepted_y[-1]
    rej_x, rej_y = 250.0, 112.0  # 120 px away -- outside the 80 px gate
    ax.plot([last_x, rej_x], [last_y, rej_y], color=REJECTED, lw=1.8,
            ls=(0, (4, 3)))
    ax.plot([rej_x], [rej_y], marker="X", ms=13, mfc="none", mec=REJECTED,
            mew=2.5)

    ax.add_patch(plt.Circle((last_x, last_y), settings.max_jump_px, fill=False,
                            color=INK, lw=1.2, ls=(0, (2, 2))))
    ax.text(last_x, last_y - settings.max_jump_px - 6, str(int(settings.max_jump_px)),
            color=INK, fontsize=11, ha="center", va="top")

    _panel_title(ax, "jump gate")
    ax.set_xlim(0, 320)
    ax.set_ylim(0, 210)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1.0)
    _mark_schematic(ax)
    return {"ring_centre": (last_x, last_y), "ring_radius": settings.max_jump_px}


def draw_norm_panel(ax, offsets: Offsets, settings: GateSettings) -> dict:
    """Panel E: the normalization window, with this fly's real spread inside it."""
    lo, hi = settings.norm_min_px, settings.norm_max_px
    axis_max = 200.0

    ax.axhspan(0.42, 0.68, xmin=lo / axis_max, xmax=hi / axis_max,
               color=ACCEPTED, alpha=0.16, lw=0)

    r = np.hypot(offsets.dx, offsets.dy)
    ax.plot([np.percentile(r, 1), np.percentile(r, 99)], [0.55, 0.55],
            color=ACCEPTED, lw=4.0, solid_capstyle="round")
    ax.plot([r.max()], [0.55], marker="o", ms=9, color=ACCEPTED, mew=1.6,
            mec="white")

    for value in (lo, hi):
        ax.plot([value, value], [0.42, 0.68], color=INK, lw=1.2)
        ax.text(value, 0.34, str(int(value)), color=INK, fontsize=11,
                ha="center", va="top")

    _panel_title(ax, "normalize")
    ax.set_xlim(0, axis_max)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return {"band": (lo, hi)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_per_gate_rejection_figure.py -v`
Expected: PASS — 31 passed (or fewer with skips)

- [ ] **Step 5: Commit**

```bash
git add scripts/analysis/per_gate_rejection_figure.py tests/test_per_gate_rejection_figure.py
git commit -m "feat(figure): draw cap, release, jump and normalize panels"
```

---

### Task 7: Assemble, save, CLI — and look at the result

**Files:**
- Modify: `scripts/analysis/per_gate_rejection_figure.py`
- Test: `tests/test_per_gate_rejection_figure.py`

**Interfaces:**
- Consumes: every draw function from Tasks 5–6.
- Produces: `FIGSIZE: tuple[float, float]`, `CAPTION: str`, `make_figure(offsets, settings, image) -> plt.Figure`, `save_figure(fig, outdir: Path = OUT_DIR, stem: str = "per_gate_rejection") -> tuple[Path, ...]`, `main(argv: Sequence[str] | None = None) -> int`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_per_gate_rejection_figure.py`:

```python
from scripts.analysis.per_gate_rejection_figure import (
    CAPTION,
    FIGSIZE,
    make_figure,
    save_figure,
)


def test_default_canvas_is_a_16_by_9_slide() -> None:
    assert FIGSIZE == (13.33, 7.5)
    assert abs(FIGSIZE[0] / FIGSIZE[1] - 16 / 9) < 0.01


def test_caption_declares_what_is_measured_and_what_is_constructed() -> None:
    """The one honesty sentence. It belongs in the caption, not in the figure."""
    assert "constructed" in CAPTION.lower()
    assert "measured" in CAPTION.lower()
    assert "config_new.yaml" in CAPTION


def test_figure_has_five_panels() -> None:
    s = load_gate_settings()
    off = Offsets(
        dx=np.array([5.0, 28.9]), dy=np.array([95.0, 142.8]),
        frames=np.array([0, 1102]), eye_xy=SUBJECT.eye_xy,
    )
    fig = make_figure(off, s, image=None)
    assert len(fig.axes) == 5, "one hero + four gate panels"
    plt.close(fig)


def test_save_writes_png_pdf_and_svg(tmp_path: Path) -> None:
    s = load_gate_settings()
    off = Offsets(
        dx=np.array([5.0, 28.9]), dy=np.array([95.0, 142.8]),
        frames=np.array([0, 1102]), eye_xy=SUBJECT.eye_xy,
    )
    fig = make_figure(off, s, image=None)
    paths = save_figure(fig, outdir=tmp_path)

    assert {p.suffix for p in paths} == {".png", ".pdf", ".svg"}
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


def test_svg_keeps_text_editable(tmp_path: Path) -> None:
    """svg.fonttype='none' -- so the labels stay editable in Illustrator."""
    s = load_gate_settings()
    off = Offsets(
        dx=np.array([5.0, 28.9]), dy=np.array([95.0, 142.8]),
        frames=np.array([0, 1102]), eye_xy=SUBJECT.eye_xy,
    )
    fig = make_figure(off, s, image=None)
    paths = save_figure(fig, outdir=tmp_path)
    svg = next(p for p in paths if p.suffix == ".svg").read_text(encoding="utf-8")
    assert "<text" in svg, "text was converted to paths -- not editable"
    assert "ACCEPTANCE BOUNDARY" in svg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_per_gate_rejection_figure.py -v`
Expected: FAIL — `ImportError: cannot import name 'CAPTION'`

- [ ] **Step 3: Write minimal implementation**

Add at the end of the module:

```python
import argparse  # noqa: E402
from typing import Sequence  # noqa: E402

FIGSIZE = (13.33, 7.5)  # 16:9 defense slide

CAPTION = (
    "Blue marks are measured from this fly; orange ✗ marks are constructed "
    "rejections. Gate values are read from config/config_new.yaml. Panels with a "
    "dashed border are schematic."
)


def make_figure(offsets: Offsets, settings: GateSettings,
                image: np.ndarray | None) -> plt.Figure:
    """Hero panel at left, four gate panels stacked at right."""
    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(
        4, 2, width_ratios=[0.62, 0.38], hspace=0.45, wspace=0.06,
        left=0.02, right=0.98, top=0.94, bottom=0.08,
    )

    draw_hero(fig.add_subplot(gs[:, 0]), offsets, settings, image)
    draw_cap_panel(fig.add_subplot(gs[0, 1]), n_flies=2)
    draw_release_panel(fig.add_subplot(gs[1, 1]), settings)
    draw_jump_panel(fig.add_subplot(gs[2, 1]), settings)
    draw_norm_panel(fig.add_subplot(gs[3, 1]), offsets, settings)

    fig.text(0.02, 0.015, CAPTION, fontsize=8, color=MUTED, va="bottom", ha="left")
    return fig


def save_figure(fig: plt.Figure, outdir: Path = OUT_DIR,
                stem: str = "per_gate_rejection") -> tuple[Path, ...]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in (".png", ".pdf", ".svg"):
        path = outdir / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
        paths.append(path)
        print(f"wrote {path}")
    plt.close(fig)
    return tuple(paths)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--outdir", type=Path, default=OUT_DIR)
    parser.add_argument("--stem", default="per_gate_rejection")
    parser.add_argument("--no-frame", action="store_true",
                        help="skip the video frame (useful without secured storage)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = load_gate_settings(args.config)
    offsets = load_subject_offsets()

    image = None
    if not args.no_frame:
        video = subject_video_path()
        if video.exists():
            image = load_frame_crop(video, SUBJECT.peak_frame)
        else:
            print(f"[warn] video not found, drawing without it: {video}")

    save_figure(make_figure(offsets, settings, image), args.outdir, args.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the full test suite**

Run: `python -m pytest tests/test_per_gate_rejection_figure.py -v`
Expected: PASS — 36 passed (or fewer with skips)

Then confirm nothing else broke:

Run: `python -m pytest tests/ -q`
Expected: no new failures relative to the pre-existing baseline.

- [ ] **Step 5: Generate the figure and look at it**

Run: `python scripts/analysis/per_gate_rejection_figure.py`
Expected: three files written to `figures/`.

Now **open `figures/per_gate_rejection.png` and actually look at it.** The tests check colour, geometry and text content; they cannot check layout. Verify by eye:

- the fly is visible through the ghosting and the boundary reads clearly over it
- no label collides with another label or with the boundary line
- the accepted cloud reads as a ventral column, not a blob
- the ✗ sits clearly outside the boundary and its label does not run off the panel
- the four side panels are legible at slide size and are not squashed
- nothing overflows the canvas

Fix any layout problems by adjusting the gridspec ratios and label offsets, then re-run the tests.

- [ ] **Step 6: Commit**

```bash
git add scripts/analysis/per_gate_rejection_figure.py tests/test_per_gate_rejection_figure.py figures/per_gate_rejection.png figures/per_gate_rejection.pdf figures/per_gate_rejection.svg
git commit -m "feat(figure): assemble PER gate rejection figure with CLI"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| Five rules and their code locations | module docstring, Task 1 |
| Config provenance, never hardcoded | Task 1 (incl. a test that a different config changes the values) |
| Subject identity and golden numbers | Task 2 |
| Raw video, not the annotated sibling | Task 2 |
| "What the data actually shows" (ventral argument) | Task 2 `test_subject_per_is_ventral` |
| Composition / panel inventory | Task 7 `make_figure` |
| Crop and ghosting | Task 4 |
| Boundary from the production function | Task 3 |
| Hero mark table | Task 5 |
| Constructed rejection at eye + (185, 120), r = 220.5 | Task 3 (verified against the real gate) |
| Panels B–E | Task 6 |
| Palette and CVD constraint | Task 5 `test_palette_is_the_validated_one` |
| Shape-first rejection encoding | Task 5, Task 6 |
| Honesty: schematic panels + caption | Task 6 `SCHEMATIC_PANELS`, Task 7 `CAPTION` |
| Deliverables: png / pdf / svg, 13.33 × 7.5 | Task 7 |
| Six spec tests | Tasks 1–7 cover all six, plus extras |
| Clean skips without data | `requires_data` / `requires_video` markers |

No gaps.

**Placeholder scan:** none. Every step contains runnable code.

**Type consistency:** `GateSettings` fields (`max_px`, `up_divisor`, `max_jump_px`, `norm_min_px`, `norm_max_px`, `dorsal_px`) are used identically in Tasks 3–7. `Offsets` fields (`dx`, `dy`, `frames`, `eye_xy`) are used identically in Tasks 3, 5, 6, 7. `draw_jump_panel` returns `{"ring_centre", "ring_radius"}` and `draw_norm_panel` returns `{"band"}` — both consumed with those exact keys in Task 6's tests. `anisotropic_boundary_offsets` is imported into module scope in Task 3 so Task 3's monkeypatch test can spy on it.

**Known risk:** `test_frame_crop_is_ghosted_pale_enough_for_the_palette` asserts `mean > 200` and `min > 120` on the real arena image. If the arena is darker than expected, raise `GHOST_BLEND` rather than lowering the thresholds — the thresholds *are* the contrast guarantee that justifies the ghosting.
