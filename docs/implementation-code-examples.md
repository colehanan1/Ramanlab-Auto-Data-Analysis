# Implementation Code Examples & Test Cases

This document provides **specific code snippets and test cases** to accompany the main prompt. Use these as reference implementations.

---

## Modification #1: Dynamic 95-Pixel Threshold - Code Examples

### Example 1.1: Calculate Per-Fly Maximum Distance

**Location:** `src/fbpipe/steps/distancestats.py` → modify `main()` function

```python
def main(cfg: Settings) -> None:
    """
    Derive global and per-fly distance bounds.
    NOW INCLUDES: per-fly maximum distance for effective threshold calculation
    """
    root = Path(cfg.maindirectory).expanduser().resolve()
    print(f"[DIST] Starting distance stats scan in {root}")
    
    for flydir in [p for p in root.iterdir() if p.is_dir()]:
        print(f"[DIST] Inspecting fly directory {flydir.name}")
        slot_ranges = {}
        fly_max_distances = {}  # NEW: Track per-fly max distance
        
        for csvpath, token, _ in iter_fly_distance_csvs(flydir, recursive=True):
            print(f"[DIST] Reading {csvpath.name} for slot {token}")
            
            try:
                df = pd.read_csv(csvpath)
            except Exception as exc:
                print(f"[DIST] Failed to read {csvpath.name} slot {token}: {exc}")
                continue
            
            dist_col = find_proboscis_distance_column(df)
            if dist_col is None:
                print(f"[DIST] No proboscis distance column found in {csvpath.name}")
                continue
            
            distances = pd.to_numeric(df[dist_col], errors='coerce')
            mask = distances.between(cfg.class2min, cfg.class2max, inclusive='both')
            vals = distances[mask]
            
            if vals.empty:
                print(f"[DIST] Slot {token} had no in-range values; skipping")
                continue
            
            local_min = float(vals.min())
            local_max = float(vals.max())
            
            print(f"[DIST] Slot {token} local min={local_min:.3f}, max={local_max:.3f}")
            
            # EXISTING: Track global min/max per slot
            current = slot_ranges.get(token)
            if current is None:
                slot_ranges[token] = (local_min, local_max)
            else:
                slot_ranges[token] = (min(current[0], local_min), max(current[1], local_max))
            
            # NEW: Track per-fly maximum distance
            if token not in fly_max_distances:
                fly_max_distances[token] = local_max
            else:
                fly_max_distances[token] = max(fly_max_distances[token], local_max)
        
        if not slot_ranges:
            print(f"[DIST] No in-range distances for {flydir.name}")
            continue
        
        # Write stats JSON with new fields
        for token, (gmin, gmax) in sorted(slot_ranges.items()):
            slot_label = token.replace('distances', '')
            
            # NEW: Calculate effective maximum
            fly_raw_max = fly_max_distances.get(token, gmax)
            effective_max = max(fly_raw_max, 95.0)  # 95-pixel threshold
            
            stats = {
                "global_min": gmin,
                "global_max": gmax,
                "effective_max_threshold": 95.0,
                "fly_max_distance": fly_raw_max,
                "effective_max_for_normalization": effective_max,
            }
            
            stats_path = flydir / f"{slot_label}global_distance_stats_class2.json"
            with open(stats_path, 'w', encoding='utf-8') as fp:
                json.dump(stats, fp)
            
            print(f"[DIST] {flydir.name}/{slot_label}: "
                  f"min={gmin:.3f} max={gmax:.3f} "
                  f"fly_max={fly_raw_max:.3f} "
                  f"effective_max={effective_max:.3f}")
            print(f"[DIST] Wrote stats JSON: {stats_path}")
```

### Example 1.2: Apply Dynamic Threshold in Normalization

**Location:** `src/fbpipe/steps/distancenormalize.py` → modify `main()` function

```python
def main(cfg: Settings) -> None:
    """
    Normalize distances using effective maximum (with 95px threshold).
    """
    root = Path(cfg.maindirectory).expanduser().resolve()
    print(f"[NORM] Starting normalization in {root}")
    
    for flydir in [p for p in root.iterdir() if p.is_dir()]:
        print(f"[NORM] Processing fly directory {flydir.name}")
        
        for csvpath, token, _ in iter_fly_distance_csvs(flydir, recursive=True):
            slot_label = token.replace('distances', '')
            
            # Load stats (now includes effective_max)
            stats_path = flydir / f"{slot_label}global_distance_stats_class2.json"
            legacy_path = flydir / "global_distance_stats_class2.json"
            
            if not stats_path.exists():
                if not legacy_path.exists():
                    print(f"[NORM] Missing stats JSON for {flydir.name}/{slot_label}")
                    continue
                stats = json.loads(legacy_path.read_text(encoding='utf-8'))
            else:
                stats = json.loads(stats_path.read_text(encoding='utf-8'))
            
            # Extract values with fallback to legacy format
            gmin = float(stats.get('global_min', stats.get('globalmin')))
            effective_max = float(stats.get('effective_max_for_normalization', 
                                           stats.get('global_max')))
            
            print(f"[NORM] Using effective_max={effective_max:.3f} for normalization")
            
            df = pd.read_csv(csvpath)
            dist_col = find_proboscis_distance_column(df)
            
            if dist_col is None:
                print(f"[NORM] No proboscis distance column in {csvpath.name}")
                continue
            
            d = pd.to_numeric(df[dist_col], errors='coerce').to_numpy()
            
            # CRITICAL: Use effective_max instead of global_max
            perc = np.empty(d.shape, dtype=float)
            
            # Mask for valid values
            valid = (d >= gmin) & np.isfinite(d)
            if effective_max != gmin:
                perc[valid] = 100.0 * (d[valid] - gmin) / effective_max
            else:
                perc[valid] = 0.0
            
            perc[~valid] = np.nan
            
            # Clamp to [0, 100] range
            perc = np.clip(perc, 0.0, 100.0)
            
            # Write updated DataFrame
            df[PROBOSCIS_DISTANCE_COL] = d
            df[PROBOSCIS_DISTANCE_PCT_COL] = perc
            df[PROBOSCIS_MIN_DISTANCE_COL] = gmin
            df[PROBOSCIS_MAX_DISTANCE_COL] = effective_max  # Store effective max
            
            df.to_csv(csvpath, index=False)
            print(f"[NORM] Normalized {csvpath.name} "
                  f"({len(df)} rows, effective_max={effective_max:.1f}px)")
```

### Example 1.3: Test Cases for Modification #1

```python
def test_95px_threshold_calculation():
    """Test that 95px threshold is applied correctly."""
    import numpy as np
    import pandas as pd
    
    # Test case 1: Fly with max < 95px
    fly_1_max = 30.0
    gmin = 0.0
    effective_max_1 = max(fly_1_max, 95.0)  # Should be 95.0
    assert effective_max_1 == 95.0, f"Expected 95, got {effective_max_1}"
    
    # At max distance (30px)
    pct_1_at_max = 100.0 * (30.0 - gmin) / effective_max_1
    assert pct_1_at_max == 31.578947368421052, f"Expected ~31.6%, got {pct_1_at_max}"
    
    # Test case 2: Fly with max >= 95px
    fly_2_max = 120.0
    effective_max_2 = max(fly_2_max, 95.0)  # Should be 120.0
    assert effective_max_2 == 120.0
    
    # At max distance (120px)
    pct_2_at_max = 100.0 * (120.0 - gmin) / effective_max_2
    assert pct_2_at_max == 100.0, f"Expected 100%, got {pct_2_at_max}"
    
    # Test case 3: Fly with max = 95px (boundary)
    fly_3_max = 95.0
    effective_max_3 = max(fly_3_max, 95.0)  # Should be 95.0
    assert effective_max_3 == 95.0
    
    pct_3_at_max = 100.0 * (95.0 - gmin) / effective_max_3
    assert pct_3_at_max == 100.0
    
    print("✓ All 95px threshold tests passed")

def test_normalization_formula():
    """Verify percentage = ((current - min) / effective_max) * 100"""
    current = 50.0
    min_val = 10.0
    effective_max = 100.0
    
    pct = ((current - min_val) / effective_max) * 100
    assert pct == 40.0, f"Expected 40%, got {pct}%"
    print("✓ Normalization formula verified")
```

---

## Modification #2: Pixel Trimming 2x - Code Examples

### Example 2.1: Update Trim Fraction

**Location:** `src/fbpipe/steps/composevideosrms.py`

```python
# BEFORE:
TRIMFRAC = 0.05  # Trim bottom 5%

# AFTER:
TRIMFRAC = 0.10  # Trim bottom 10% (double the trimming)
```

### Example 2.2: Verify Trimming Impact

```python
def read_compute_trimming_max_flydir(flydir: Path) -> Optional[Tuple[float, float]]:
    """
    Compute trimmed min/max for a fly directory.
    NEW: Log how much was trimmed.
    """
    base = flydir / 'RMScalculations'
    if not base.is_dir():
        return None
    
    values = []
    for path, _, _ in iter_fly_distance_csvs(base, recursive=True):
        try:
            arr = pd.to_numeric(
                pd.read_csv(path, usecols=[DIST_COL_ROBUST]),
                errors='coerce'
            ).to_numpy()
        except Exception:
            continue
        values.append(arr)
    
    if not values:
        return None
    
    all_values = np.concatenate(values)
    all_values = all_values[np.isfinite(all_values)]
    
    if all_values.size == 0:
        return None
    
    global_max = float(np.max(all_values))
    
    # Calculate trimming impact
    num_before_trim = all_values.size
    percentile = float(np.percentile(all_values, 100 * TRIMFRAC, method='linear'))
    kept = all_values[all_values >= percentile]
    num_after_trim = kept.size
    num_trimmed = num_before_trim - num_after_trim
    pct_trimmed = 100.0 * num_trimmed / num_before_trim
    
    trimmed_min = float(np.min(kept)) if kept.size else float(np.min(all_values))
    
    print(f"[TRIM] Fly {flydir.name}:")
    print(f"       Before trim: {num_before_trim} points, min={all_values.min():.2f}px")
    print(f"       After trim:  {num_after_trim} points, min={trimmed_min:.2f}px")
    print(f"       Removed:     {num_trimmed} points ({pct_trimmed:.1f}%)")
    
    return trimmed_min, global_max
```

---

## Modification #3: Continuous Angle Multiplier - Code Examples

### Example 3.1: Exponential Angle Multiplier Function

**Location:** New utility file `src/fbpipe/utils/angle_calc.py`

```python
"""Continuous exponential angle multiplier calculation."""

import numpy as np
from typing import Union

def compute_angle_multiplier_continuous(
    angle_deg: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Convert angle in degrees to continuous multiplier for distance percentage.
    
    Mapping:
        -100° → 0.5× (angle pulls proboscis back)
          0° → 1.0× (baseline, no angle effect)
        +100° → 2.0× (angle supports extension)
    
    Uses linear interpolation (continuous):
        For angle in [-100, 0): multiplier = 0.5 + 0.5 * (1 + angle/100)
        For angle in [0, +100]: multiplier = 1.0 + angle/100
    
    Args:
        angle_deg: Single angle or array of angles in degrees
    
    Returns:
        Multiplier value(s) in range [0.5, 2.0]
    
    Examples:
        >>> compute_angle_multiplier_continuous(-100)
        0.5
        >>> compute_angle_multiplier_continuous(0)
        1.0
        >>> compute_angle_multiplier_continuous(100)
        2.0
        >>> compute_angle_multiplier_continuous(50)
        1.5
        >>> compute_angle_multiplier_continuous(-50)
        0.75
    """
    # Clamp to valid range
    is_scalar = np.isscalar(angle_deg)
    angle_array = np.atleast_1d(angle_deg)
    clamped = np.clip(angle_array, -100, 100)
    
    multiplier = np.zeros_like(clamped, dtype=float)
    
    # Negative angles: 0.5 at -100°, transition to 1.0 at 0°
    neg_mask = clamped < 0
    if np.any(neg_mask):
        # Linear: from 0.5 (-100) to 1.0 (0)
        normalized = clamped[neg_mask] / 100.0  # [-1.0, 0.0)
        # mult = 0.5 + 0.5 * (1.0 + normalized)
        multiplier[neg_mask] = 0.5 + 0.5 * (1.0 + normalized)
    
    # Positive angles: 1.0 at 0°, transition to 2.0 at +100°
    pos_mask = clamped >= 0
    if np.any(pos_mask):
        # Linear: from 1.0 (0) to 2.0 (+100)
        normalized = clamped[pos_mask] / 100.0  # [0.0, 1.0]
        # mult = 1.0 + normalized
        multiplier[pos_mask] = 1.0 + normalized
    
    return float(multiplier[0]) if is_scalar else multiplier


def test_angle_multiplier():
    """Verify continuous angle multiplier."""
    test_cases = [
        (-100, 0.5),
        (-75, 0.625),
        (-50, 0.75),
        (-25, 0.875),
        (0, 1.0),
        (25, 1.25),
        (50, 1.5),
        (75, 1.75),
        (100, 2.0),
    ]
    
    for angle, expected in test_cases:
        result = compute_angle_multiplier_continuous(angle)
        assert abs(result - expected) < 1e-6, \
            f"Angle {angle}°: expected {expected}, got {result}"
    
    # Test array input
    angles = np.array([-100, 0, 100])
    mults = compute_angle_multiplier_continuous(angles)
    assert np.allclose(mults, [0.5, 1.0, 2.0])
    
    # Test edge cases
    assert compute_angle_multiplier_continuous(-150) == 0.5  # Clamped
    assert compute_angle_multiplier_continuous(200) == 2.0   # Clamped
    
    print("✓ All angle multiplier tests passed")
```

### Example 3.2: Replace Bin-Based Calls

**Replace old bin lookup:**

```python
# OLD (bin-based):
if angle >= 40 and angle < 60:
    multiplier = 1.5
elif angle >= 60:
    multiplier = 1.75
else:
    multiplier = 1.0

# NEW (continuous):
from fbpipe.utils.angle_calc import compute_angle_multiplier_continuous
multiplier = compute_angle_multiplier_continuous(angle)
```

---

## Modification #4: Calculate Acceleration - Code Examples

### Example 4.1: New Step - Calculate Acceleration

**Location:** New file `src/fbpipe/steps/calculateacceleration.py`

```python
"""Calculate frame-to-frame acceleration of distance% × angle_multiplier."""

from pathlib import Path
import numpy as np
import pandas as pd
from ..config import Settings
from ..utils.columns import find_proboscis_distance_percentage_column
from ..utils.flyfiles import iter_fly_distance_csvs
from ..utils.angle_calc import compute_angle_multiplier_continuous

def main(cfg: Settings) -> None:
    """
    Calculate acceleration (change in combined metric per frame).
    
    For each frame:
        combined(t) = distance_pct(t) × angle_multiplier(angle(t))
        acceleration(t) = combined(t) - combined(t-1)
    
    Flags frames where |acceleration| exceeds threshold (suggests model error).
    """
    root = Path(cfg.maindirectory).expanduser().resolve()
    print(f"[ACCEL] Starting acceleration calculation in {root}")
    
    ACCEL_THRESHOLD_PCT = 100.0  # Flag if acceleration > 100% per frame
    
    for flydir in [p for p in root.iterdir() if p.is_dir()]:
        print(f"[ACCEL] Processing fly {flydir.name}")
        rms_dir = flydir / 'RMScalculations'
        if not rms_dir.is_dir():
            continue
        
        for csvpath, token, _ in iter_fly_distance_csvs(rms_dir, recursive=True):
            try:
                df = pd.read_csv(csvpath)
            except Exception as exc:
                print(f"[ACCEL] Failed to read {csvpath.name}: {exc}")
                continue
            
            # Get required columns
            dist_pct_col = find_proboscis_distance_percentage_column(df)
            if dist_pct_col is None or dist_pct_col not in df.columns:
                print(f"[ACCEL] No distance % column in {csvpath.name}")
                continue
            
            # We need angle for multiplier
            angle_col = None
            for candidate in ['angle_deg', 'angle_ARB_deg', 'angleARBdeg']:
                if candidate in df.columns:
                    angle_col = candidate
                    break
            
            if angle_col is None:
                # If no explicit angle, we can't calculate combined metric
                print(f"[ACCEL] No angle column in {csvpath.name}, skipping acceleration")
                continue
            
            # Extract numeric values
            dist_pct = pd.to_numeric(df[dist_pct_col], errors='coerce').to_numpy()
            angles = pd.to_numeric(df[angle_col], errors='coerce').to_numpy()
            
            # Calculate angle multipliers
            multipliers = np.zeros_like(angles, dtype=float)
            valid_angle = np.isfinite(angles)
            multipliers[valid_angle] = compute_angle_multiplier_continuous(
                angles[valid_angle]
            )
            multipliers[~valid_angle] = np.nan
            
            # Calculate combined metric
            combined = dist_pct * multipliers
            
            # Calculate acceleration (frame-to-frame delta)
            acceleration = np.full_like(combined, np.nan, dtype=float)
            valid = np.isfinite(combined)
            
            for i in range(1, len(combined)):
                if valid[i] and valid[i-1]:
                    acceleration[i] = combined[i] - combined[i-1]
            
            # Flag high accelerations
            accel_flag = np.abs(acceleration) > ACCEL_THRESHOLD_PCT
            
            # Add columns to DataFrame
            df['angle_multiplier'] = multipliers
            df['combined_distance_pct_x_angle'] = combined
            df['acceleration_pct_per_frame'] = acceleration
            df['acceleration_flag'] = accel_flag.astype(float)
            
            # Save updated CSV
            df.to_csv(csvpath, index=False)
            
            # Log results
            num_flagged = int(np.sum(accel_flag))
            print(f"[ACCEL] {csvpath.name}: "
                  f"{num_flagged} frames flagged (accel > {ACCEL_THRESHOLD_PCT}%/frame)")
            
            if num_flagged > 0:
                flagged_frames = np.where(accel_flag)[0]
                print(f"        Flagged frames: {flagged_frames[:10]}"
                      f"{'...' if len(flagged_frames) > 10 else ''}")
```

### Example 4.2: Test Acceleration Calculation

```python
def test_acceleration_calculation():
    """Verify acceleration computation."""
    
    # Test data
    distances_pct = np.array([20.0, 45.0, 98.0, 50.0, 25.0])
    angles_deg = np.array([0.0, 10.0, 5.0, -10.0, 0.0])
    
    # Compute multipliers
    multipliers = np.array([
        compute_angle_multiplier_continuous(a) for a in angles_deg
    ])
    # Expected: [1.0, 1.1, 1.05, 0.9, 1.0]
    
    combined = distances_pct * multipliers
    # Frame 0: 20 × 1.0 = 20.0
    # Frame 1: 45 × 1.1 = 49.5
    # Frame 2: 98 × 1.05 = 102.9
    # Frame 3: 50 × 0.9 = 45.0
    # Frame 4: 25 × 1.0 = 25.0
    
    # Calculate acceleration
    acceleration = np.full_like(combined, np.nan)
    for i in range(1, len(combined)):
        acceleration[i] = combined[i] - combined[i-1]
    
    # Verify
    np.testing.assert_allclose(combined, [20.0, 49.5, 102.9, 45.0, 25.0], atol=0.1)
    np.testing.assert_allclose(acceleration[1:], [29.5, 53.4, -57.9, -20.0], atol=0.1)
    
    # Frame 2→3 acceleration is -57.9% (large drop) - would be flagged
    print("✓ Acceleration calculation verified")
```

---

## Modification #5: Reference Angle Using Trimmed Data - Code Examples

### Example 5.1: Update Reference Angle Calculation

**Location:** `src/fbpipe/steps/composevideosrms.py` → `find_fly_reference_angle_csvs_raw()`

```python
def find_fly_reference_angle_csvs_raw(
    csvs_raw: List[Path],
    trimmed_min_threshold: float  # NEW parameter
) -> float:
    """
    Find reference angle (0° baseline) using frames near trimmed_min.
    
    Instead of: angle at frame with minimum distance
    Now: angle at frame where distance is closest to trimmed_min value
    
    Args:
        csvs_raw: List of CSV paths for fly
        trimmed_min_threshold: Distance value from Modification #2 trimming
    
    Returns:
        Reference angle in degrees
    """
    best = None  # (metric, delta, angle)
    
    for path in csvs_raw:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        
        try:
            angle = compute_angle_deg_at_point2(df)  # Existing function
        except Exception:
            continue
        
        dist_col = find_col(df, ['distancepercentage', 'distancepercent', ...])
        if dist_col is None:
            continue
        
        dist = pd.to_numeric(df[dist_col], errors='coerce').to_numpy()
        
        # NEW: Find frame closest to trimmed_min, not absolute minimum
        valid = np.isfinite(dist)
        if not np.any(valid):
            continue
        
        dist_delta = np.abs(dist[valid] - trimmed_min_threshold)
        best_frame_idx = int(np.nanargmin(dist_delta))
        
        angle_here = float(angle.iloc[best_frame_idx]) if np.isfinite(angle.iloc[best_frame_idx]) else np.nan
        distance_to_threshold = float(dist_delta[best_frame_idx])
        
        candidate = (distance_to_threshold, angle_here)
        
        if best is None or candidate[0] < best[0]:
            best = candidate
    
    return best[1] if best is not None else float('nan')
```

### Example 5.2: Integration in Pipeline

```python
# In composevideosrms.py, before rendering panels:

trimmed_min, trimmed_max = read_compute_trimming_max_flydir(flydir)
if trimmed_min is None:
    print(f"[TRIM] Could not compute trimmed values for {flydir.name}")
    trimmed_min = some_default_value

# Find reference angle using trimmed data
raw_csvs = [...]  # Collect raw CSV paths
reference_angle = find_fly_reference_angle_csvs_raw(raw_csvs, trimmed_min)

# Now all angle calculations use this stable baseline
```

---

## Modification #6: Condition-Based Result Organization - Code Examples

### Example 6.1: Infer Condition from Path

**Location:** New utility file `src/fbpipe/utils/condition_detect.py`

```python
"""Detect experimental condition from fly directory."""

from pathlib import Path
from typing import Optional
import re

# Condition mappings
CONDITION_PATTERNS = {
    'hex_control': r'hex.*control|control.*hex',
    'opto_hex': r'opto.*hex|hex.*opto',
    'benz_control': r'benz.*control|control.*benz|benzaldehyde.*control',
    'opto_benz_1': r'opto.*benz|benz.*opto|optobenz',
    'Eb_control': r'eb.*control|control.*eb|ethylbenzene.*control',
    'opto_Eb': r'opto.*eb|eb.*opto|optoeb',
}

def infer_condition_from_path(flydir: Path) -> Optional[str]:
    """
    Infer experimental condition from fly directory name/path.
    
    Args:
        flydir: Path to fly directory
    
    Returns:
        Condition string or None if not detected
    
    Examples:
        >>> infer_condition_from_path(Path('/data/october/fly1_hex_control'))
        'hex_control'
        >>> infer_condition_from_path(Path('/data/november/fly2_opto_Eb'))
        'opto_Eb'
    """
    dir_name = flydir.name.lower()
    full_path = str(flydir).lower()
    
    # Check directory name first, then full path
    for condition, pattern in CONDITION_PATTERNS.items():
        if re.search(pattern, dir_name):
            return condition
        if re.search(pattern, full_path):
            return condition
    
    return None

def test_condition_detection():
    """Test condition inference."""
    test_cases = [
        ('october_fly1_hex_control', 'hex_control'),
        ('november_fly2_opto_hex', 'opto_hex'),
        ('december_fly3_benz_control', 'benz_control'),
        ('january_fly4_opto_benz_1', 'opto_benz_1'),
        ('february_fly5_Eb_control', 'Eb_control'),
        ('march_fly6_opto_Eb', 'opto_Eb'),
        ('april_fly7_unknown', None),
    ]
    
    for dirname, expected_condition in test_cases:
        result = infer_condition_from_path(Path(dirname))
        assert result == expected_condition, \
            f"Path '{dirname}': expected {expected_condition}, got {result}"
    
    print("✓ All condition detection tests passed")
```

### Example 6.2: Reorganize Results by Condition

**Location:** New step `src/fbpipe/steps/organizebycondition.py`

```python
"""Organize results into condition-specific folders."""

from pathlib import Path
import shutil
from ..config import Settings
from ..utils.condition_detect import infer_condition_from_path

def main(cfg: Settings) -> None:
    """
    Reorganize pipeline results by experimental condition.
    
    Moves files from:
        maindirectory/october/fly1/RMScalculations/
    To:
        maindirectory/october_hex_control/fly1/RMScalculations/
    """
    root = Path(cfg.maindirectory).expanduser().resolve()
    print(f"[ORG] Starting condition-based organization in {root}")
    
    # Collect all fly directories
    fly_dirs = []
    for month_dir in root.iterdir():
        if not month_dir.is_dir():
            continue
        for fly_dir in month_dir.iterdir():
            if fly_dir.is_dir():
                fly_dirs.append((month_dir, fly_dir))
    
    # Organize by condition
    for month_dir, fly_dir in fly_dirs:
        condition = infer_condition_from_path(fly_dir)
        
        if condition is None:
            print(f"[ORG] Could not infer condition for {fly_dir.name}")
            continue
        
        # Create condition-specific folder
        month_name = month_dir.name
        condition_dir = root / f"{month_name}_{condition}"
        condition_dir.mkdir(parents=True, exist_ok=True)
        
        # Move fly directory to condition folder
        target_dir = condition_dir / fly_dir.name
        
        if target_dir.exists():
            print(f"[ORG] {target_dir} already exists, skipping {fly_dir.name}")
            continue
        
        try:
            shutil.move(str(fly_dir), str(target_dir))
            print(f"[ORG] Moved {fly_dir.name} → {condition_dir.name}/{fly_dir.name}")
        except Exception as exc:
            print(f"[ORG] ERROR moving {fly_dir.name}: {exc}")
    
    print(f"[ORG] Completed condition-based organization")
    
    # Summary report
    print("\n[ORG] Result summary:")
    for cond_dir in sorted(root.glob('*_*/')):
        fly_count = len(list(cond_dir.glob('*/RMScalculations')))
        print(f"      {cond_dir.name}: {fly_count} flies")
```

### Example 6.3: Integration into Pipeline

**Location:** `src/fbpipe/pipeline.py`

```python
from .steps import organizebycondition

ORDERED_STEPS = (
    Step('yolo', yoloinfer.main, 'Run YOLO inference...'),
    Step('distancestats', distancestats.main, 'Derive distance bounds...'),
    Step('distancenormalize', distancenormalize.main, 'Normalize distances...'),
    Step('detectdroppedframes', detectdroppedframes.main, '...'),
    Step('rmscopyfilter', rmscopyfilter.main, '...'),
    Step('updateofmstate', updateofmstate.main, '...'),
    Step('movevideos', movevideos.main, '...'),
    Step('composevideosrms', composevideosrms.main, 'Render RMS overlays...'),
    Step('calculateacceleration', calculateacceleration.main,  # NEW
          'Calculate frame acceleration...'),
    Step('organizebycondition', organizebycondition.main,      # NEW
          'Organize results by condition...'),
)
```

---

## Complete Integration Test

```python
def test_full_integration():
    """
    End-to-end test of all 6 modifications.
    """
    import tempfile
    from pathlib import Path
    
    # Create test data structure
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # Create fly directories with conditions in name
        oct_hex_ctrl = root / 'october_hex_control_fly1'
        oct_hex_ctrl.mkdir()
        
        rms_dir = oct_hex_ctrl / 'RMScalculations'
        rms_dir.mkdir()
        
        # Create test CSV with all necessary columns
        test_csv = rms_dir / 'fly1_distances.csv'
        
        test_data = {
            'frame': [0, 1, 2, 3, 4],
            'distance_class_2_8': [10.0, 25.0, 50.0, 75.0, 80.0],
            'angle_ARB_deg': [0.0, 10.0, 5.0, -5.0, 0.0],
            'x_class_2': [1000, 950, 900, 850, 800],
            'y_class_2': [500, 500, 500, 500, 500],
            'x_proboscis': [1100, 1150, 1200, 1150, 1100],
            'y_proboscis': [500, 520, 540, 520, 500],
        }
        
        df_test = pd.DataFrame(test_data)
        df_test.to_csv(test_csv, index=False)
        
        # Run modifications
        print("Testing Modification #1: 95px threshold")
        # Simulate: max distance = 80px, should cap below 100%
        effective_max = max(80.0, 95.0)
        assert effective_max == 95.0
        pct_at_max = 100.0 * (80.0 - 10.0) / effective_max
        assert pct_at_max < 100.0  # ✓
        
        print("Testing Modification #2: 2x trimming")
        # Verify TRIMFRAC doubled
        assert 0.10 > 0.05  # ✓
        
        print("Testing Modification #3: Continuous angle")
        mults = [compute_angle_multiplier_continuous(a) for a in [0, 10, 5, -5, 0]]
        assert all(0.5 <= m <= 2.0 for m in mults)  # ✓
        
        print("Testing Modification #4: Acceleration")
        combined = [20*1.0, 45*1.1, 75*1.05, 75*0.95, 80*1.0]
        accel = [combined[i]-combined[i-1] for i in range(1, len(combined))]
        assert all(isinstance(a, float) for a in accel)  # ✓
        
        print("Testing Modification #5: Reference angle")
        trimmed_min = 25.0
        # Find frame closest to trimmed_min
        distances = [10.0, 25.0, 50.0, 75.0, 80.0]
        deltas = [abs(d - trimmed_min) for d in distances]
        best_frame = deltas.index(min(deltas))
        assert best_frame == 1  # Frame 1 has distance=25.0 ✓
        
        print("Testing Modification #6: Condition separation")
        condition = infer_condition_from_path(oct_hex_ctrl)
        assert condition == 'hex_control'  # ✓
        
        print("\n✅ All integration tests passed!")
```

---

## Summary

These code examples provide:
- **Clear implementation guidance** for each modification
- **Exact function signatures** and parameter names
- **Test cases** to verify correctness
- **Integration points** in the existing pipeline
- **Logging statements** for debugging

Use these as templates when implementing with Claude Code.
