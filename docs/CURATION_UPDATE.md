# YOLO Curation Module - Secure Storage Update

## Summary

Updated the YOLO dataset curation module to support searching for videos in secure storage locations separate from YOLO output CSVs.

## Changes Made

### 1. Configuration ([config/config.yaml](../config/config.yaml#L96-L106))

Added `video_source_dirs` to specify additional directories to search for videos:

```yaml
yolo_curation:
  enabled: false
  # ... other settings ...
  video_source_dirs:
    - "/path/to/secure/storage/opto_EB/"
    - "/path/to/secure/storage/opto_EB(6-training)/"
    - "/path/to/secure/storage/opto_benz_1/"
    - "/path/to/secure/storage/opto_hex/"
    - "/path/to/secure/storage/opto_ACV/"
    - "/path/to/secure/storage/hex_control/"
    - "/path/to/secure/storage/EB_control/"
    - "/path/to/secure/storage/Benz_control/"
    - "/path/to/secure/storage/opto_AIR/"
    - "/path/to/secure/storage/opto_3-oct/"
```

### 2. Configuration Schema ([src/fbpipe/config.py](src/fbpipe/config.py#L115))

Added `video_source_dirs: Tuple[str, ...]` field to `YoloCurationSettings` dataclass.

### 3. Video Search Logic ([src/fbpipe/steps/curate_yolo_dataset.py](src/fbpipe/steps/curate_yolo_dataset.py#L377-L388))

Enhanced video search to:
1. First check same directory as CSV (existing behavior)
2. Then search each configured `video_source_dirs` with mapped fly directory names
3. Use first video found

### 4. Documentation ([docs/YOLO_DATASET_CURATION.md](docs/YOLO_DATASET_CURATION.md#L173-L195))

Added section explaining the video source directories feature and how path mapping works.

## How It Works

### Example Scenario

**CSV Location:**
```
/path/to/Data/flys/opto_EB/september_16_fly_1/
  └── september_16_fly_1_testing_3/
      ├── september_16_fly_1_testing_3_fly1_distances.csv  ← YOLO output
      └── september_16_fly_1_testing_3_distances_merged.csv
```

**Video Location:**
```
/path/to/secure/storage/opto_EB/september_16_fly_1/
  └── output_september_16_fly_1_testing_3_20250916_143551.mp4  ← Original video
```

### Video Naming Pattern

Videos in secure storage follow this pattern:
- **Format**: `output_{fly_name}_{trial_type}_{N}_{timestamp}.mp4`
- **Example**: `output_september_16_fly_1_testing_3_20250916_143551.mp4`

The module automatically:
1. Extracts folder name from CSV path: `september_16_fly_1_testing_3`
2. Builds glob pattern: `output_september_16_fly_1_testing_3_*.mp4`
3. Matches video regardless of timestamp

### Search Process

When processing `september_16_fly_1_testing_3_fly1_distances.csv`:

**Step 1: Try same directory as CSV**
- `/home/.../opto_EB/september_16_fly_1/output_september_16_fly_1_testing_3_*.mp4` (not found)

**Step 2: Try secure storage (using fly directory name)**
- `/securedstorage/.../opto_EB/september_16_fly_1/output_september_16_fly_1_testing_3_*.mp4` ✓ **Found!**

The first matching video is used for frame extraction.

## Benefits

1. **No Video Duplication**: Videos stay in secure storage, no need to copy to working directory
2. **Flexible Storage**: Supports different storage locations for CSVs vs videos
3. **Backward Compatible**: If `video_source_dirs` is empty, works exactly as before
4. **Multiple Source Support**: Can search across multiple storage locations

## Testing

All secure storage directories verified as accessible:

```bash
$ python test_video_search.py
======================================================================
✓ Configuration loaded successfully
  - Curation enabled: False
  - Video source dirs: 10

  Configured video source directories:
    1. ✓ /path/to/secure/storage/opto_EB/
    2. ✓ /path/to/secure/storage/opto_EB(6-training)/
    ...
    10. ✓ /path/to/secure/storage/opto_3-oct/

✓ TEST PASSED
======================================================================
```

## Usage

No changes to usage! Simply enable curation and run:

```bash
# Enable in config/config.yaml
yolo_curation:
  enabled: true  # ← Set to true

# Run pipeline
python -m src.fbpipe.pipeline --config config/config.yaml curate_yolo_dataset
```

The module will automatically search secure storage for videos.

## Logging

New log messages show when secure storage search is active:

```
[CURATION] Starting YOLO dataset curation
[CURATION] Quality thresholds: jitter<=50.0px, missing<=10.0%
[CURATION] Searching for videos in 10 additional source directories
[CURATION] Processing root directory: /path/to/Data/flys/opto_EB/
...
```

If a video isn't found, the warning now shows how many locations were checked:

```
[CURATION] Cannot find video for trial_2024_01_15_A_fly1_distances.csv (tried 30 locations)
```

## Files Modified

1. [config/config.yaml](../config/config.yaml) - Added `video_source_dirs` configuration
2. [src/fbpipe/config.py](src/fbpipe/config.py) - Added config field and loader
3. [src/fbpipe/steps/curate_yolo_dataset.py](src/fbpipe/steps/curate_yolo_dataset.py) - Enhanced video search logic
4. [docs/YOLO_DATASET_CURATION.md](docs/YOLO_DATASET_CURATION.md) - Added documentation

## Files Created

1. [test_video_search.py](test_video_search.py) - Test for video source directory configuration

---

**Ready to use!** The curation module will now find your videos in secure storage when you enable it. 🚀
