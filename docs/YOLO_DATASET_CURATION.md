# YOLO Dataset Curation Module

## Overview

The YOLO Dataset Curation module systematically identifies problematic tracking videos from YOLO inference results and extracts frames for manual labeling to improve your training dataset. This transforms reactive quality control into systematic dataset improvement.

## Features

- **Automated Quality Analysis**: Computes tracking quality metrics (jitter, missing frames)
- **Intelligent Frame Selection**: Stratified sampling extracts representative and challenging cases
- **Labeling Workflow Support**: Auto-detects and organizes labeled frames
- **Data Augmentation**: Automatically augments labeled data with horizontal flip, brightness/contrast jitter, and minor rotations
- **Resumable**: State tracking allows pausing and resuming the curation process

## Quick Start

### 1. Enable Curation in Config

Edit `config/config.yaml`:

```yaml
yolo_curation:
  enabled: true  # Set to true to enable curation
  quality_thresholds:
    max_jitter_px: 50.0  # Flag videos with median jitter > 50 pixels
    max_missing_pct: 0.10  # Flag videos with >10% frames missing
  target_frames:
    per_video: 10  # Extract ~10 frames per flagged video
    stratification:
      high_quality: 0.30  # 30% seed data (low jitter, valid tracking)
      low_quality: 0.50   # 50% problem cases (high jitter or missing)
      boundary: 0.20      # 20% edge cases (moderate quality)
  augmentation:
    enabled: true
    strategies:
      - horizontal_flip
      - brightness_contrast_jitter
      - minor_rotation
    multiplier: 2
```

### 2. Run Curation

```bash
# Run full pipeline (includes curation after YOLO)
python -m src.fbpipe.pipeline --config config/config.yaml

# Or run just the curation step
python -m src.fbpipe.pipeline --config config/config.yaml curate_yolo_dataset
```

### 3. Output Structure

For each fly directory, the curation module creates:

```
{FLY_DIR}/yolo_curation/
├── quality_metrics.json          # Computed quality metrics for all trials
├── flagged_videos.json           # List of videos needing re-labeling
├── curation_manifest.json        # State tracking for resumability
├── to_label/                     # PNG frames awaiting annotation
│   ├── video1_frame_000042_high_quality.png
│   ├── video1_frame_000153_low_quality.png
│   └── video2_frame_000089_boundary.png
├── labeled/                      # Frames you've annotated
│   ├── video1_frame_000042_high_quality.png
│   ├── video1_frame_000042_high_quality.txt  # YOLO format
│   └── ...
└── augmented/                    # Auto-generated augmentations
    ├── video1_frame_000042_high_quality_horizontal_flip.png
    ├── video1_frame_000042_high_quality_horizontal_flip.txt
    └── ...
```

## Workflow

### Phase 1: Identification & Extraction

1. **Run pipeline**: Curation module analyzes YOLO outputs
2. **Review flagged videos**: Check `yolo_curation/flagged_videos.json`
3. **Frames extracted**: See frames in `yolo_curation/to_label/`

### Phase 2: Labeling

1. **Choose annotation tool**: Roboflow, Label Studio, CVAT, or similar
2. **Annotate frames**: Label bounding boxes for eyes and proboscis
3. **Export in YOLO format**: Each PNG should have corresponding TXT file
4. **Place in to_label/**: Keep PNG+TXT pairs together

Example annotation (YOLO format TXT):
```
0 0.5234 0.4512 0.0823 0.1045  # class 0 (eye)
1 0.5123 0.6234 0.0512 0.0734  # class 1 (proboscis)
```

### Phase 3: Auto-Processing

1. **Re-run pipeline**: Module detects labeled frames
2. **Auto-move**: PNG+TXT pairs move to `labeled/`
3. **Augmentation**: Creates variations in `augmented/`

### Phase 4: Integration with YOLO Training

Once you have 200+ curated frames:

```bash
# Copy to YOLO training dataset
cp yolo_curation/labeled/*.png {YOLO_DATASET}/images/train/curated/
cp yolo_curation/labeled/*.txt {YOLO_DATASET}/labels/train/curated/

cp yolo_curation/augmented/*.png {YOLO_DATASET}/images/train/curated/
cp yolo_curation/augmented/*.txt {YOLO_DATASET}/labels/train/curated/

# Retrain YOLO
yolo train data={YOLO_DATASET}/data.yaml model=yolov8m-obb.pt epochs=100
```

## Configuration Reference

### Quality Thresholds

```yaml
quality_thresholds:
  max_jitter_px: 50.0
```
- **Purpose**: Flag videos where median frame-to-frame movement exceeds this threshold
- **Typical values**: 30-70px depending on your video resolution
- **Why it matters**: High jitter indicates identity swaps or optical flow failures

```yaml
quality_thresholds:
  max_missing_pct: 0.10
```
- **Purpose**: Flag videos where proboscis detection is missing in >10% of frames
- **Typical values**: 0.05-0.15 (5%-15%)
- **Why it matters**: Missing frames indicate model uncertainty or difficult poses

### Frame Extraction

```yaml
target_frames:
  per_video: 10
```
- **Purpose**: Number of frames to extract per flagged video
- **Trade-off**: More frames = more labeling work but better coverage

```yaml
stratification:
  high_quality: 0.30
  low_quality: 0.50
  boundary: 0.20
```
- **high_quality**: Frames with valid tracking and low jitter (seed data for validation)
- **low_quality**: Frames with missing/poor tracking (most valuable for training)
- **boundary**: Frames with moderate quality (edge cases for model uncertainty)

### Augmentation

```yaml
augmentation:
  enabled: true
  strategies:
    - horizontal_flip
    - brightness_contrast_jitter
    - minor_rotation
  multiplier: 2
```
- **horizontal_flip**: Exploits left-right symmetry in Drosophila
- **brightness_contrast_jitter**: Simulates lighting variations (±10% brightness, ±10% contrast)
- **minor_rotation**: Simulates camera tilt (±5°)
- **multiplier**: Target dataset expansion (2× means double the size)

### Video Source Directories

```yaml
video_source_dirs:
  - "/path/to/secure/storage/opto_EB/"
  - "/path/to/secure/storage/opto_hex/"
```

**Purpose**: Specify additional directories to search for video files when they're not in the same location as YOLO output CSVs.

**Use case**: If your YOLO outputs are in `/home/user/Data/flys/` but videos are in `/securedstorage/Data-secured/`, the module will:
1. First search in the same directory as the CSV
2. Then search in each configured `video_source_dirs` with matching fly directory names
3. Use the first video found

**How matching works**:
- CSV location: `/home/user/Data/flys/opto_EB/trial1/trial1_fly1_distances.csv`
- Module searches for `trial1.mp4` in:
  - `/home/user/Data/flys/opto_EB/trial1.mp4` (same location as CSV)
  - `/securedstorage/Data-secured/opto_EB/trial1.mp4` (mapped to source dir)
  - `/securedstorage/Data-secured/opto_EB/trial1_preprocessed.mp4` (with suffix)

This allows flexible video storage without duplicating files.

## Quality Metrics Explained

### Jitter (Spatial Consistency)

```python
jitter = sqrt((x[t] - x[t-1])² + (y[t] - y[t-1])²)
```

- **Low jitter (<20px)**: Smooth, consistent tracking
- **Moderate jitter (20-50px)**: Acceptable with occasional jumps
- **High jitter (>50px)**: Likely identity swaps or tracking failures

### Missing Frame Percentage

```python
pct_missing = (frames_with_NaN / total_frames)
```

- **Low (<5%)**: Excellent tracking coverage
- **Moderate (5-10%)**: Acceptable for most analyses
- **High (>10%)**: Significant data gaps, model uncertainty

## Troubleshooting

### No Videos Flagged

**Symptom**: `flagged_videos.json` is empty or has 0 entries.

**Possible causes**:
- Thresholds too strict: Increase `max_jitter_px` or `max_missing_pct`
- Tracking quality is actually good: Review quality metrics manually
- Wrong CSV format: Verify `proboscis_x` and `proboscis_y` columns exist

**Solution**:
```yaml
quality_thresholds:
  max_jitter_px: 30.0  # Lower threshold to flag more videos
  max_missing_pct: 0.05
```

### Too Many Videos Flagged

**Symptom**: Hundreds of flagged videos, overwhelming labeling workload.

**Possible causes**:
- Thresholds too lenient
- Systematic tracking failures across dataset

**Solution**:
```yaml
quality_thresholds:
  max_jitter_px: 70.0  # Raise threshold to be more selective
  max_missing_pct: 0.15

target_frames:
  per_video: 5  # Extract fewer frames per video
```

### Augmentation Not Running

**Symptom**: `augmented/` directory is empty.

**Possible causes**:
- No labeled frames yet: Augmentation only runs on labeled data
- Augmentation disabled in config

**Solution**:
1. Verify `augmentation.enabled: true`
2. Ensure labeled frames exist in `labeled/` with both PNG+TXT
3. Re-run pipeline: `python -m src.fbpipe.pipeline curate_yolo_dataset`

### Frame Extraction Seems Wrong

**Symptom**: Extracted frames don't look problematic or are all similar.

**Possible causes**:
- Stratification ratios need adjustment
- Jitter threshold doesn't match your video characteristics

**Solution**: Manually inspect `quality_metrics.json` and adjust thresholds based on actual values in your data.

## Advanced Usage

### Manual Frame Selection

If automated selection isn't working well:

1. Disable automated extraction: Set `target_frames.per_video: 0`
2. Manually extract frames from flagged videos using:
   ```bash
   ffmpeg -i video.mp4 -vf "select='eq(n,100)'" -vsync 0 frame_100.png
   ```
3. Place in `to_label/` and annotate

### Confidence-Based Metrics (Future Enhancement)

Currently, the module uses spatial metrics (jitter, missing frames). To add confidence-based metrics:

1. Modify `yolo_infer.py` to save confidence scores in CSV
2. Update `compute_quality_metrics()` to include:
   ```python
   avg_confidence = df['proboscis_confidence'].mean()
   ```
3. Add to flagging logic in `is_bad_tracking()`

### Custom Augmentation Strategies

Add new augmentation in `src/fbpipe/utils/augmentation.py`:

```python
def augment_gaussian_blur(image, annotations, kernel_size=5):
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred, annotations
```

Register in `augment_labeled_frames()`:
```python
strategy_functions = {
    "horizontal_flip": augment_horizontal_flip,
    "gaussian_blur": lambda img, ann: augment_gaussian_blur(img, ann, 5),
}
```

Update config:
```yaml
augmentation:
  strategies:
    - horizontal_flip
    - gaussian_blur
```

## API Reference

### Core Functions

#### `compute_quality_metrics(df, video_path, quality_thresholds)`

Computes per-video tracking quality metrics.

**Parameters**:
- `df` (DataFrame): Tracking data with `proboscis_x`, `proboscis_y` columns
- `video_path` (Path): Path to video file
- `quality_thresholds` (dict): Configuration thresholds

**Returns**: Dict with keys:
- `total_frames` (int)
- `missing_frames` (int)
- `pct_missing` (float)
- `median_jitter_px` (float)
- `max_jitter_px` (float)
- `mean_jitter_px` (float)

#### `is_bad_tracking(metrics, quality_thresholds)`

Determines if video should be flagged for curation.

**Parameters**:
- `metrics` (dict): Quality metrics from `compute_quality_metrics()`
- `quality_thresholds` (dict): Configuration thresholds

**Returns**: `bool` - True if video should be flagged

#### `extract_frames_stratified(video_path, df, output_dir, target_count, stratification)`

Extracts frames using stratified sampling.

**Parameters**:
- `video_path` (Path): Path to video
- `df` (DataFrame): Tracking data
- `output_dir` (Path): Output directory
- `target_count` (int): Number of frames to extract
- `stratification` (dict): Allocation ratios

**Returns**: List of Path objects for extracted frames

#### `augment_labeled_frames(labeled_dir, augmented_dir, strategies, multiplier)`

Augments all labeled frames in a directory.

**Parameters**:
- `labeled_dir` (Path): Directory with PNG+TXT pairs
- `augmented_dir` (Path): Output directory
- `strategies` (list): Augmentation strategy names
- `multiplier` (int): Target dataset expansion

**Returns**: `int` - Number of augmented images created

## Best Practices

1. **Start conservative**: Begin with strict thresholds, then relax if needed
2. **Label in batches**: Annotate 50 frames, retrain, evaluate, repeat
3. **Balance dataset**: Ensure mix of high/low quality frames
4. **Validate augmentations**: Manually inspect augmented frames for correctness
5. **Track metrics**: Compare YOLO performance before/after retraining
6. **Version control**: Keep dated snapshots of curated datasets

## Performance Metrics

After implementing this system and labeling ~200 curated frames, you should see:

- **Reduced missing frames**: 15-20% fewer missing detections
- **Improved confidence**: +5-10% average confidence scores
- **Better edge cases**: Reduced failures in challenging poses/lighting
- **Smoother tracking**: Lower jitter from fewer identity swaps

## Citation

If you use this curation system in your research, please cite:

```
Ramanlab Auto Data Analysis Pipeline - YOLO Dataset Curation Module
https://github.com/your-repo/ramanlab-auto-data-analysis
```

## Support

For issues or questions:
1. Check troubleshooting section above
2. Inspect `quality_metrics.json` and `flagged_videos.json` for insights
3. Open an issue on GitHub with sample metrics and config
