# ✅ Video Search Verification Complete

## Summary

The YOLO curation module has been successfully updated to find videos in your secure storage using the correct naming pattern.

## Video Naming Pattern

Your videos follow this consistent pattern:
```
output_{fly_name}_{trial_type}_{N}_{timestamp}.mp4
```

**Examples from your data:**
- `output_september_16_fly_1_testing_3_20250916_143551.mp4`
- `output_december_10_batch_1_testing_9_20251210_155612.mp4`
- `output_september_18_fly_1_testing_5_20250918_124726.mp4`

## How It Works

### 1. CSV Structure
```
/home/.../Data/flys/opto_EB/september_16_fly_1/
└── september_16_fly_1_testing_3/
    └── september_16_fly_1_testing_3_fly1_distances.csv
```

### 2. Video Location
```
/securedstorage/.../Data-secured/opto_EB/september_16_fly_1/
└── output_september_16_fly_1_testing_3_20250916_143551.mp4
```

### 3. Search Logic

The module:
1. **Reads CSV path**: `/home/.../september_16_fly_1_testing_3/...csv`
2. **Extracts folder name**: `september_16_fly_1_testing_3`
3. **Builds glob pattern**: `output_september_16_fly_1_testing_3_*.mp4`
4. **Searches in**:
   - Same directory as CSV (usually not found)
   - Secure storage: `/securedstorage/.../opto_EB/september_16_fly_1/`
5. **Matches video** regardless of timestamp

## Test Results

### ✅ Single Video Test
```
Fly name: september_16_fly_1
Folder name: september_16_fly_1_testing_3
Pattern: output_september_16_fly_1_testing_3_*.mp4

✓ Pattern matched!
  Found: output_september_16_fly_1_testing_3_20250916_143551.mp4
```

### ✅ Multiple Flies Test
```
Tested 4 fly directories:
  ✓ december_10_batch_1_rig_2 - Pattern matched
  ✓ december_10_batch_2 - Pattern matched
  ✓ september_18_fly_1 - Pattern matched
  ✓ september_16_fly_2 - Pattern matched

Results: 4/4 successful pattern matches (100%)
```

## Configuration

Your [config/config.yaml](../config/config.yaml#L95-L106) is already configured:

```yaml
yolo_curation:
  enabled: true  # ← Ready to use
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

## Ready to Use!

The curation module will now:
1. ✅ Find your YOLO CSVs in `/home/.../Data/flys/`
2. ✅ Find corresponding videos in `/securedstorage/.../Data-secured/`
3. ✅ Match videos using glob patterns (handles timestamps automatically)
4. ✅ Extract frames for labeling
5. ✅ Support all 10 secure storage directories

## Run the Pipeline

```bash
# The module is already enabled in config/config.yaml
python -m src.fbpipe.pipeline --config config/config.yaml curate_yolo_dataset
```

Expected output:
```
[CURATION] Starting YOLO dataset curation
[CURATION] Quality thresholds: jitter<=50.0px, missing<=10.0%
[CURATION] Searching for videos in 10 additional source directories
[CURATION] Processing root directory: /home/.../opto_EB/
[CURATION] Inspecting fly directory: september_16_fly_1
[CURATION] Found video: /securedstorage/.../september_16_fly_1/output_..._testing_3_....mp4
[CURATION] Flagged: output_september_16_fly_1_testing_3_20250916_143551.mp4 (...)
[CURATION] Extracted 10 frames from video
```

## Files Updated

1. [src/fbpipe/steps/curate_yolo_dataset.py](src/fbpipe/steps/curate_yolo_dataset.py#L375-L419) - Enhanced video search with glob patterns
2. [config/config.yaml](../config/config.yaml#L95-L106) - Added video_source_dirs configuration
3. [CURATION_UPDATE.md](CURATION_UPDATE.md#L47-L84) - Updated documentation with real examples

## Test Scripts

- [test_video_search.py](test_video_search.py) - Verifies config loads correctly
- [test_video_search_complete.py](test_video_search_complete.py) - Tests pattern matching with real files

---

**Status: ✅ VERIFIED AND READY**

The video search is working correctly with your secure storage setup. You can now run the curation module!
