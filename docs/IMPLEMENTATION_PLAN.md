# Implementation Plan: 6-Point Proboscis Distance & Angle Analysis Modifications

## Executive Summary

This plan details the step-by-step implementation of 6 interconnected modifications to the Drosophila behavioral analysis pipeline. Each modification builds on previous ones, requiring careful sequencing and integration testing.

**Estimated complexity:** High (touching 6+ files, adding new pipeline step, restructuring result organization)

---

## Phase 1: Core Distance & Angle Calculations (Modifications #1, #2, #3)

### Modification #1: Dynamic RMS Distance Percentage with 95-Pixel Threshold

**Current Behavior:**
- File: `src/fbpipe/steps/distance_normalize.py`
- Formula: `perc = 100 * (d - gmin) / (gmax - gmin)`
- Uses global min/max per fly calculated from all trials

**Changes Required:**

#### 1.1 Modify `distance_stats.py` (Lines ~80-130)

**Add to JSON output:**
```python
# Current output:
{"global_min": float, "global_max": float}

# New output:
{
    "global_min": float,
    "global_max": float,
    "fly_max_distance": float,  # NEW: actual max before 95px floor
    "effective_max_threshold": 95.0  # NEW: threshold constant
}
```

**Implementation location:** After line where `slot_ranges[token] = (...)` is calculated
- Track actual max distance: `fly_max = max([local_max for all CSVs])`
- Add to JSON: `stats["fly_max_distance"] = float(fly_max)`
- Add constant: `stats["effective_max_threshold"] = 95.0`

#### 1.2 Modify `distance_normalize.py` (Lines ~60-90)

**Change normalization formula:**
```python
# Current (lines ~85-90):
if gmax != gmin:
    perc[inr] = 100.0 * (d[inr] - gmin) / (gmax - gmin)

# New logic:
fly_max = float(stats.get("fly_max_distance", gmax))  # Read from JSON
threshold = float(stats.get("effective_max_threshold", 95.0))
effective_max = max(fly_max, threshold)  # Use max(actual, 95px)

if effective_max != gmin:
    perc[inr] = 100.0 * (d[inr] - gmin) / (effective_max - gmin)
```

**Add new column to CSV:**
```python
df["effective_max_distance_2_8"] = effective_max  # Store for reference
```

**Test cases:**
- Fly with max=30px → effective_max=95px → max percentage = 31.6%
- Fly with max=120px → effective_max=120px → max percentage = 100%
- Fly with max=200px → effective_max=200px → max percentage = 100%

---

### Modification #2: Increase Pixel Trimming (Bottom 5% → 10%)

**Current Behavior:**
- File: `src/fbpipe/steps/compose_videos_rms.py`, Line 48
- Constant: `TRIM_FRAC = 0.05` (5th percentile cutoff)
- Used in: `_ead_compute_trim_min_max()` function (lines 440-464)

**Changes Required:**

#### 2.1 Update TRIM_FRAC constant (Line 48)

```python
# Current:
TRIM_FRAC = 0.05  # Lower 5th percentile

# New:
TRIM_FRAC = 0.10  # Lower 10th percentile (doubling trim percentage)
```

**Impact verification:**
- Function `_ead_compute_trim_min_max()` already uses this constant correctly
- Line 459: `percentile = float(np.percentile(all_values, 100 * TRIM_FRAC, method="linear"))`
- This will automatically trim bottom 10% instead of 5%
- NO OTHER CHANGES NEEDED in this function

**Test cases:**
- Verify more low-value outliers are excluded
- Log: Number of values below percentile before/after change
- Ensure `trimmed_min` increases (more aggressive filtering)

---

### Modification #3: Convert Bin-Based Angle Scaling to Continuous Exponential

**Current Behavior:**
- Angles are calculated in `compose_videos_rms.py` (`compute_angle_deg_at_point2()`, lines 201-220)
- Reference angle found in `find_fly_reference_angle()` (lines 223-261)
- **NO EXISTING MULTIPLIER LOGIC FOUND** - this is a NEW feature

**Changes Required:**

#### 3.1 Create new function in `compose_videos_rms.py` (Insert after line 220)

```python
def compute_angle_multiplier_continuous(angle_deg: float) -> float:
    """
    Convert angle in degrees to continuous multiplier for distance percentage.

    Mapping:
        -100° → 0.5× (proboscis retracted toward body)
        0° → 1.0× (neutral, baseline)
        +100° → 2.0× (proboscis extended)

    Uses piecewise linear interpolation for smooth transitions.

    Args:
        angle_deg: Angle in degrees (will be clamped to [-100, 100])

    Returns:
        Multiplier value in range [0.5, 2.0]
    """
    # Clamp angle to valid range
    clamped = np.clip(angle_deg, -100.0, 100.0)

    if clamped < 0:
        # Linear interpolation: -100° → 0.5, 0° → 1.0
        # Formula: 0.5 + 0.5 * (1.0 + normalized)
        normalized = clamped / 100.0  # Range: -1.0 to 0.0
        multiplier = 0.5 + 0.5 * (1.0 + normalized)
    else:
        # Linear interpolation: 0° → 1.0, +100° → 2.0
        # Formula: 1.0 + normalized
        normalized = clamped / 100.0  # Range: 0.0 to 1.0
        multiplier = 1.0 + normalized

    return float(multiplier)


def compute_angle_multiplier_series(angle_series: pd.Series) -> pd.Series:
    """
    Vectorized version for applying to pandas Series.

    Args:
        angle_series: Series of angle values in degrees

    Returns:
        Series of multiplier values
    """
    return angle_series.apply(compute_angle_multiplier_continuous)
```

#### 3.2 Apply multipliers in RMS calculations (Modify existing functions)

**Location:** In `_series_rms_from_rmscalc()` function (lines 566-635)

**After line where angles are computed, add:**
```python
# After: angle_centered = compute_angle_deg_at_point2(df) - reference_angle
angle_multipliers = compute_angle_multiplier_series(angle_centered)

# Store in dataframe for later use
df["angle_centered_deg"] = angle_centered
df["angle_multiplier"] = angle_multipliers
```

**Test cases:**
- Test angles: -100, -75, -50, -25, 0, 25, 50, 75, 100
- Expected multipliers: 0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0
- Verify 40° ≠ 50° (should be 1.4 vs 1.5)
- Verify clamping works for angles > ±100°

---

## Phase 2: Acceleration Analysis (Modification #4)

### Modification #4: Calculate Acceleration from Distance% × Angle Multiplier

**Current Behavior:**
- NO EXISTING ACCELERATION ANALYSIS
- This is a completely NEW pipeline step

**Changes Required:**

#### 4.1 Create new file: `src/fbpipe/steps/calculate_acceleration.py`

```python
"""
Calculate frame-to-frame acceleration from combined distance% × angle multiplier.

This step detects potential model errors or dropped frames by identifying
physically implausible rapid changes in proboscis position.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from ..config import Settings
from ..helpers import iter_fly_distance_csvs, find_col

# Acceleration threshold for flagging suspicious frames (% per frame)
ACCELERATION_THRESHOLD = 100.0

# Columns required
DISTANCE_PCT_COL = "distance_percentage_2_8"
ANGLE_MULT_COL = "angle_multiplier"
FRAME_COL = "frame"


def calculate_acceleration_for_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """
    Calculate acceleration for a single CSV file.

    Returns:
        DataFrame with new columns added, or None if missing required columns
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ACCEL] Error reading {csv_path.name}: {e}")
        return None

    # Check for required columns
    if DISTANCE_PCT_COL not in df.columns:
        print(f"[ACCEL] Missing {DISTANCE_PCT_COL} in {csv_path.name}")
        return None

    if ANGLE_MULT_COL not in df.columns:
        print(f"[ACCEL] Missing {ANGLE_MULT_COL} in {csv_path.name}")
        return None

    # Extract data
    dist_pct = pd.to_numeric(df[DISTANCE_PCT_COL], errors="coerce").to_numpy()
    angle_mult = pd.to_numeric(df[ANGLE_MULT_COL], errors="coerce").to_numpy()

    # Calculate combined metric
    combined = dist_pct * angle_mult
    df["combined_distance_x_angle"] = combined

    # Calculate frame-to-frame acceleration (delta)
    acceleration = np.full(len(combined), np.nan)
    acceleration[1:] = np.diff(combined)  # Frame i - Frame i-1
    df["acceleration_pct_per_frame"] = acceleration

    # Flag high-acceleration frames
    df["acceleration_flag"] = np.abs(acceleration) > ACCELERATION_THRESHOLD

    return df


def main(cfg: Settings) -> None:
    """
    Calculate acceleration for all RMS calculation CSVs.

    Process:
    1. Find all CSVs in RMS_calculations directories
    2. Calculate combined distance × angle multiplier
    3. Calculate frame-to-frame acceleration
    4. Flag suspicious frames (acceleration > threshold)
    5. Save updated CSVs with new columns
    """
    root = Path(cfg.main_directory).expanduser().resolve()

    total_processed = 0
    total_flagged = 0

    for fly_dir in root.iterdir():
        if not fly_dir.is_dir():
            continue

        rms_dir = fly_dir / "RMS_calculations"
        if not rms_dir.is_dir():
            continue

        for csv_path, token, _ in iter_fly_distance_csvs(rms_dir, recursive=False):
            # Skip if already processed (has acceleration column)
            try:
                temp_df = pd.read_csv(csv_path, nrows=1)
                if "acceleration_pct_per_frame" in temp_df.columns:
                    continue
            except Exception:
                pass

            # Calculate acceleration
            df = calculate_acceleration_for_csv(csv_path)
            if df is None:
                continue

            # Count flagged frames
            n_flagged = int(df["acceleration_flag"].sum())
            total_flagged += n_flagged

            # Save updated CSV
            df.to_csv(csv_path, index=False)
            total_processed += 1

            if n_flagged > 0:
                print(f"[ACCEL] {csv_path.name}: {n_flagged} high-acceleration frames flagged")

        # Generate summary report per fly
        if total_processed > 0:
            print(f"[ACCEL] {fly_dir.name}: {total_processed} files processed, {total_flagged} suspicious frames")

    print(f"[ACCEL] Complete: {total_processed} CSVs processed, {total_flagged} total flagged frames")


if __name__ == "__main__":
    from ..config import load_settings
    main(load_settings())
```

#### 4.2 Register step in pipeline (Modify `src/fbpipe/pipeline.py`)

**Location:** Find `ORDERED_STEPS` tuple (currently lines ~20-30)

**Add new step AFTER `compose_videos_rms`:**
```python
ORDERED_STEPS = (
    "yolo",
    "distance_stats",
    "distance_normalize",
    "detect_dropped_frames",
    "rms_copy_filter",
    "update_ofm_state",
    "move_videos",
    "compose_videos_rms",
    "calculate_acceleration",  # NEW STEP
)
```

**Register the step function:**
```python
# Add import at top of file
from .steps.calculate_acceleration import main as calculate_acceleration_main

# In the step dispatch/registry (find where steps are mapped to functions)
STEP_FUNCTIONS = {
    "yolo": yolo_main,
    "distance_stats": distance_stats_main,
    # ... existing steps ...
    "compose_videos_rms": compose_videos_rms_main,
    "calculate_acceleration": calculate_acceleration_main,  # NEW
}
```

**Test cases:**
- Frame 0: dist_pct=20%, angle_mult=1.0 → combined=20%
- Frame 1: dist_pct=45%, angle_mult=1.1 → combined=49.5%, accel=+29.5%
- Frame 2: dist_pct=98%, angle_mult=1.2 → combined=117.6%, accel=+68.1% (flagged)
- Verify flagging works for threshold exceedance

---

## Phase 3: Reference Angle Stability (Modification #5)

### Modification #5: Recalculate 0° Reference Angle Using Trimmed Data

**Current Behavior:**
- File: `compose_videos_rms.py`, Function: `find_fly_reference_angle()` (lines 223-261)
- Current logic: Find angle at frame where distance == 0, or argmin(abs(distance))
- Uses raw distances (no trimming consideration)

**Changes Required:**

#### 5.1 Modify `find_fly_reference_angle()` function signature

**Current signature (line 223):**
```python
def find_fly_reference_angle(csvs_raw: List[Path]) -> float:
```

**New signature:**
```python
def find_fly_reference_angle(csvs_raw: List[Path], trimmed_min: Optional[float] = None) -> float:
```

#### 5.2 Update algorithm logic (lines 230-260)

**Replace the distance-based selection:**
```python
# OLD LOGIC (lines ~240-255):
exact = np.flatnonzero(dist == 0)
if exact.size > 0:
    idx = int(exact[0])
    angle_here = float(angle.iloc[idx])
    candidate = (0, 0.0, angle_here)
else:
    absd = np.abs(dist)
    idx = int(np.nanargmin(absd))
    angle_here = float(angle.iloc[idx])
    candidate = (1, float(absd[idx]), angle_here)

# NEW LOGIC:
if trimmed_min is not None:
    # Find frame nearest to trimmed_min (more stable reference)
    distance_from_trim = np.abs(dist - trimmed_min)
    idx = int(np.nanargmin(distance_from_trim))
    angle_here = float(angle.iloc[idx])
    candidate = (0, float(distance_from_trim[idx]), angle_here)
else:
    # Fallback to old logic if trimmed_min not provided
    exact = np.flatnonzero(dist == 0)
    if exact.size > 0:
        idx = int(exact[0])
        angle_here = float(angle.iloc[idx])
        candidate = (0, 0.0, angle_here)
    else:
        absd = np.abs(dist)
        idx = int(np.nanargmin(absd))
        angle_here = float(angle.iloc[idx])
        candidate = (1, float(absd[idx]), angle_here)
```

#### 5.3 Update calling code to pass trimmed_min

**Location:** Find where `find_fly_reference_angle()` is called in `compose_videos_rms.py`

**Likely in `main()` function or render loop:**
```python
# Get trimmed min from _ead_compute_trim_min_max()
trimmed_stats = _ead_compute_trim_min_max(fly_dir)
if trimmed_stats:
    trimmed_min, trimmed_max = trimmed_stats
else:
    trimmed_min = None

# Pass to reference angle function
reference_angle = find_fly_reference_angle(csv_list, trimmed_min=trimmed_min)
```

**Test cases:**
- Verify reference angle uses frame near trimmed_min, not global min
- Compare stability: old vs new reference angles across flies
- Log which frame was selected and its distance value

---

## Phase 4: Result Organization by Condition (Modification #6)

### Modification #6: Separate Results by Experimental Condition

**Current Behavior:**
- Results organized by month-based directories (e.g., `october/`, `november/`)
- Function: `_discover_month_folders()` in `compose_videos_rms.py` (lines 97-107)
- No condition separation

**Changes Required:**

#### 6.1 Create condition inference function (New file or add to helpers)

**Location:** Add to `src/fbpipe/helpers.py` or create `src/fbpipe/conditions.py`

```python
"""
Experimental condition detection and classification.
"""

from pathlib import Path
from typing import Optional
import re

# Condition patterns (order matters - check specific before general)
CONDITION_PATTERNS = [
    (r"opto.*hex", "opto_hex"),
    (r"hex.*control", "hex_control"),
    (r"opto.*benz", "opto_benz_1"),
    (r"benz.*control", "benz_control"),
    (r"opto.*eb", "opto_Eb"),
    (r"opto.*Eb", "opto_Eb"),  # Case variation
    (r"eb.*control", "Eb_control"),
    (r"Eb.*control", "Eb_control"),  # Case variation
]


def infer_condition_from_path(fly_dir: Path) -> Optional[str]:
    """
    Infer experimental condition from fly directory name or parent directory.

    Searches for condition markers in directory path components.

    Args:
        fly_dir: Path to fly directory (e.g., .../opto_EB/october_fly_1/)

    Returns:
        Condition string or None if no match found

    Examples:
        /data/opto_EB/october_fly_1/ → "opto_Eb"
        /data/controls/hex_control_batch/ → "hex_control"
        /data/september_opto_hex/ → "opto_hex"
    """
    # Check fly directory name and all parent directories
    path_str = str(fly_dir).lower()

    for pattern, condition in CONDITION_PATTERNS:
        if re.search(pattern, path_str, re.IGNORECASE):
            return condition

    # Check for metadata file (optional, if exists)
    metadata_file = fly_dir / "experiment_metadata.txt"
    if metadata_file.exists():
        try:
            content = metadata_file.read_text().lower()
            for pattern, condition in CONDITION_PATTERNS:
                if re.search(pattern, content, re.IGNORECASE):
                    return condition
        except Exception:
            pass

    return None  # No condition detected


def extract_month_from_path(fly_dir: Path) -> Optional[str]:
    """
    Extract month identifier from fly directory name.

    Args:
        fly_dir: Path to fly directory (e.g., october_fly_1)

    Returns:
        Month string (e.g., "october") or None

    Examples:
        october_fly_1 → "october"
        jan_batch_1 → "january"
        september_09_fly_1 → "september"
    """
    MONTHS_FULL = (
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    )
    MONTHS_SHORT = ("jan", "feb", "mar", "apr", "may", "jun",
                    "jul", "aug", "sep", "oct", "nov", "dec")
    MONTH_MAP = dict(zip(MONTHS_SHORT, MONTHS_FULL))

    name_lower = fly_dir.name.lower()

    # Check full month names
    for month in MONTHS_FULL:
        if name_lower.startswith(month):
            return month

    # Check short month names
    for short, full in MONTH_MAP.items():
        if name_lower.startswith(short):
            return full

    return None
```

#### 6.2 Modify result organization functions

**Files to modify:**
1. `compose_videos_rms.py` - Update `_discover_month_folders()` and result paths
2. `move_videos.py` - Update video staging paths
3. `rms_copy_filter.py` - Update RMS_calculations output paths

**Key changes in `compose_videos_rms.py`:**

**Replace `_discover_month_folders()` (lines 97-107):**
```python
def _discover_condition_folders(root: Path) -> Dict[str, List[Path]]:
    """
    Discover fly directories organized by month and condition.

    Returns:
        Dictionary mapping "month_condition" to list of fly directories

    Example:
        {
            "october_hex_control": [Path("october_fly_1"), Path("october_fly_2")],
            "october_opto_hex": [Path("october_fly_3")],
            "november_benz_control": [Path("november_batch_1")],
        }
    """
    from .conditions import infer_condition_from_path, extract_month_from_path

    organized: Dict[str, List[Path]] = {}

    for fly_dir in root.iterdir():
        if not fly_dir.is_dir():
            continue

        # Extract month and condition
        month = extract_month_from_path(fly_dir)
        condition = infer_condition_from_path(fly_dir)

        if month is None:
            print(f"[WARN] Could not determine month for {fly_dir.name}, skipping")
            continue

        # Create combined key
        if condition:
            key = f"{month}_{condition}"
        else:
            key = f"{month}_unknown"
            print(f"[WARN] Could not determine condition for {fly_dir.name}, using 'unknown'")

        if key not in organized:
            organized[key] = []
        organized[key].append(fly_dir)

    return organized
```

**Update main() function to use condition-based organization:**
```python
def main(cfg: Settings) -> None:
    root = Path(cfg.main_directory).expanduser().resolve()

    # Organize by condition
    condition_groups = _discover_condition_folders(root)

    for condition_key, fly_dirs in condition_groups.items():
        print(f"\n[RMS] Processing condition: {condition_key}")
        print(f"[RMS] Flies: {[d.name for d in fly_dirs]}")

        # Create output directory for this condition
        output_dir = root / condition_key
        output_dir.mkdir(exist_ok=True)

        for fly_dir in fly_dirs:
            # Move/organize results to condition-specific folder
            # ... existing processing logic ...

            # Update output paths to include condition
            rms_output = output_dir / fly_dir.name / "videos_with_rms"
            # ... rest of processing ...
```

#### 6.3 Update all result-writing steps

**Files requiring path updates:**
1. `move_videos.py` - Line where videos are moved
2. `rms_copy_filter.py` - Line where RMS_calculations are written
3. Any summary/report generation code

**General pattern:**
```python
# OLD:
output_path = fly_dir / "RMS_calculations"

# NEW:
from .conditions import infer_condition_from_path, extract_month_from_path
month = extract_month_from_path(fly_dir)
condition = infer_condition_from_path(fly_dir)
condition_key = f"{month}_{condition}" if condition else month
output_base = root / condition_key / fly_dir.name
output_path = output_base / "RMS_calculations"
output_path.mkdir(parents=True, exist_ok=True)
```

**Test cases:**
- Verify condition detection works for all 6 condition types
- Verify directory structure created correctly
- Verify no data loss during reorganization
- Generate summary: fly count per condition

---

## Phase 5: Integration Dependencies

### Dependency Chain

```
Modification #1 (95px threshold)
    ↓ provides: effective_max, fly_max_distance
Modification #2 (trimming 10%)
    ↓ provides: trimmed_min
Modification #5 (reference angle)
    ↑ requires: trimmed_min from #2
    ↓ provides: stable reference_angle
Modification #3 (continuous angle multiplier)
    ↑ requires: reference_angle from #5
    ↓ provides: angle_multiplier column
Modification #4 (acceleration)
    ↑ requires: distance_percentage from #1, angle_multiplier from #3
    ↓ provides: acceleration metrics
Modification #6 (condition organization)
    ↑ requires: all above calculations complete
    ↓ provides: organized result structure
```

### Critical Integration Points

1. **JSON Schema Update (Mod #1):**
   - `distance_stats.py` must write new fields before `distance_normalize.py` reads them
   - Backward compatibility: Handle missing fields gracefully

2. **Angle Multiplier Availability (Mod #3 → #4):**
   - `compose_videos_rms.py` must add `angle_multiplier` column before `calculate_acceleration.py` runs
   - Column must be present in RMS_calculations CSVs

3. **Trimmed Min Propagation (Mod #2 → #5):**
   - `_ead_compute_trim_min_max()` output must be passed to `find_fly_reference_angle()`
   - Need to ensure both functions are in same scope or pass via parameters

4. **Pipeline Step Order (Mod #4):**
   - `calculate_acceleration` must run AFTER `compose_videos_rms`
   - Must run BEFORE result organization (Mod #6)

---

## Phase 6: Testing Strategy

### Unit Tests (Per Modification)

**Test #1: 95px Threshold**
```python
def test_effective_max_threshold():
    # Fly with max < 95
    assert effective_max(fly_max=30) == 95
    assert distance_percentage(d=30, min=0, effective_max=95) == 31.58

    # Fly with max > 95
    assert effective_max(fly_max=120) == 120
    assert distance_percentage(d=120, min=0, effective_max=120) == 100.0
```

**Test #2: Trimming Fraction**
```python
def test_trimming_increase():
    values = np.random.randn(1000)

    old_trim = np.percentile(values, 5)
    new_trim = np.percentile(values, 10)

    assert new_trim > old_trim  # More aggressive trimming
    assert len(values[values >= new_trim]) < len(values[values >= old_trim])
```

**Test #3: Continuous Angle Multiplier**
```python
def test_angle_multiplier():
    assert compute_angle_multiplier_continuous(-100) == 0.5
    assert compute_angle_multiplier_continuous(-50) == 0.75
    assert compute_angle_multiplier_continuous(0) == 1.0
    assert compute_angle_multiplier_continuous(50) == 1.5
    assert compute_angle_multiplier_continuous(100) == 2.0

    # Test clamping
    assert compute_angle_multiplier_continuous(-200) == 0.5
    assert compute_angle_multiplier_continuous(200) == 2.0

    # Test continuity (no jumps)
    assert compute_angle_multiplier_continuous(40) != compute_angle_multiplier_continuous(50)
```

**Test #4: Acceleration Calculation**
```python
def test_acceleration():
    df = pd.DataFrame({
        "frame": [0, 1, 2],
        "distance_percentage_2_8": [20, 45, 98],
        "angle_multiplier": [1.0, 1.1, 1.2]
    })

    result = calculate_acceleration_for_csv_df(df)

    assert result["combined_distance_x_angle"].iloc[0] == 20.0
    assert result["combined_distance_x_angle"].iloc[1] == 49.5
    assert result["combined_distance_x_angle"].iloc[2] == 117.6

    assert np.isnan(result["acceleration_pct_per_frame"].iloc[0])
    assert abs(result["acceleration_pct_per_frame"].iloc[1] - 29.5) < 0.1
    assert abs(result["acceleration_pct_per_frame"].iloc[2] - 68.1) < 0.1
```

**Test #5: Reference Angle with Trimmed Min**
```python
def test_reference_angle_stability():
    distances = np.array([5, 8, 12, 15, 18, 150, 20, 19, 14, 9, 6])
    angles = np.array([30, 35, 40, 45, 50, 90, 55, 53, 44, 38, 32])

    trimmed_min = np.percentile(distances, 10)  # ~8.5px

    # Old method: uses argmin(distances) → frame 0 (5px, angle=30°)
    old_ref = angles[np.argmin(distances)]

    # New method: uses frame nearest trimmed_min → frame 1 (8px, angle=35°)
    distance_from_trim = np.abs(distances - trimmed_min)
    new_ref = angles[np.argmin(distance_from_trim)]

    assert new_ref != old_ref
    assert new_ref == 35  # More stable (not outlier)
```

**Test #6: Condition Detection**
```python
def test_condition_inference():
    assert infer_condition_from_path(Path("/data/opto_EB/october_fly_1")) == "opto_Eb"
    assert infer_condition_from_path(Path("/data/hex_control/november_batch")) == "hex_control"
    assert infer_condition_from_path(Path("/data/opto_hex_batch")) == "opto_hex"

    assert extract_month_from_path(Path("october_fly_1")) == "october"
    assert extract_month_from_path(Path("jan_batch_1")) == "january"
```

### Integration Tests (Full Pipeline)

**Test End-to-End:**
1. Create test dataset with 3 flies (different conditions)
2. Run full pipeline with all modifications
3. Verify:
   - JSON stats include `fly_max_distance` and `effective_max_threshold`
   - Distance percentages respect 95px threshold
   - Trimming uses 10th percentile
   - Angle multipliers in range [0.5, 2.0]
   - Acceleration columns present in RMS_calculations CSVs
   - Results organized by condition folders
   - All CSVs have required columns

**Expected Output Structure:**
```
main_directory/
├── october_hex_control/
│   ├── october_fly_1/
│   │   ├── RMS_calculations/
│   │   │   └── updated_*_distances.csv
│   │   │       [columns: frame, timestamp, distance_percentage_2_8,
│   │   │        angle_multiplier, combined_distance_x_angle,
│   │   │        acceleration_pct_per_frame, acceleration_flag, ...]
│   │   └── videos_with_rms/
├── october_opto_hex/
├── october_benz_control/
├── october_opto_benz_1/
├── october_Eb_control/
├── october_opto_Eb/
└── ...
```

---

## Phase 7: Implementation Sequence

### Step-by-Step Execution Order

1. **Implement Mod #1 (95px threshold):**
   - Edit `distance_stats.py`
   - Edit `distance_normalize.py`
   - Test: Run distance_stats + distance_normalize on sample fly

2. **Implement Mod #2 (trimming):**
   - Edit `compose_videos_rms.py` (change TRIM_FRAC)
   - Test: Check percentile calculation output

3. **Implement Mod #5 (reference angle) - BEFORE #3:**
   - Edit `find_fly_reference_angle()` in `compose_videos_rms.py`
   - Update calling code to pass trimmed_min
   - Test: Verify reference angle uses trimmed data

4. **Implement Mod #3 (continuous angle multiplier):**
   - Add `compute_angle_multiplier_continuous()` to `compose_videos_rms.py`
   - Add column to RMS CSVs
   - Test: Verify multipliers calculated correctly

5. **Implement Mod #4 (acceleration):**
   - Create `calculate_acceleration.py`
   - Register in `pipeline.py`
   - Test: Run on sample CSV with angle_multiplier column

6. **Implement Mod #6 (condition organization):**
   - Create condition inference functions
   - Update all result-writing paths
   - Test: Verify directory structure

7. **Full Integration Test:**
   - Run entire pipeline on test dataset
   - Verify all modifications work together
   - Check output structure and data integrity

---

## Phase 8: Rollback Plan

### Rollback Strategy

If any modification causes issues:

1. **Git-based rollback:**
   - Each modification should be a separate commit
   - Can revert individual commits without affecting others

2. **Feature flags (optional):**
   - Add config.yaml flags to enable/disable each modification
   - Example: `use_95px_threshold: true`

3. **Backward compatibility:**
   - All CSV readers should handle missing columns gracefully
   - JSON readers should provide defaults for missing fields

---

## Phase 9: Documentation & Validation

### Required Documentation

1. **Update README.md:**
   - Document new CSV columns
   - Document condition organization structure
   - Document acceleration threshold configuration

2. **Update config.yaml schema:**
   - Add `effective_max_threshold: 95.0`
   - Add `trim_fraction: 0.10`
   - Add `acceleration_threshold: 100.0`

3. **Create changelog:**
   - Document all modifications with justifications
   - Include migration notes for existing datasets

### Validation Checklist

- [ ] All unit tests pass
- [ ] Integration test on sample dataset passes
- [ ] Output CSVs have correct columns
- [ ] Result directories organized by condition
- [ ] No data loss during reorganization
- [ ] Acceleration flagging works correctly
- [ ] Reference angles are stable
- [ ] Distance percentages respect 95px threshold
- [ ] Angle multipliers are continuous (no binning)
- [ ] Pipeline completes without errors

---

## Summary

This implementation plan provides:
- Clear sequence of modifications with dependencies mapped
- Specific file locations and line numbers for each change
- Complete code implementations for new functions
- Comprehensive test cases for validation
- Rollback strategy for risk mitigation

**Total estimated changes:**
- 6 files modified
- 1 new file created (calculate_acceleration.py)
- ~500 lines of code added/modified
- ~200 lines of test code added

**Risk areas:**
- Modification #6 (condition organization) has highest risk (affects all result paths)
- Modification #4 (acceleration) requires careful integration into pipeline
- Modification #5 (reference angle) must be done before #3 (angle multiplier)

**Next step:** Begin implementation with Modification #1 (95px threshold)
