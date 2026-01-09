# Pseudo-label Export (Top-confidence Frame Mining)

This tool scans many videos, runs the trained YOLO model as a teacher, keeps the highest-confidence frames, and exports an Ultralytics-compatible dataset on disk.

## What it exports

```
dataset_out/
  images/train, images/val
  labels/train, labels/val
  data.yaml
  manifest.csv
```

- Labels are YOLO TXT files (per-image `.txt`).
- Default label format is axis-aligned boxes (`cls cx cy w h`, normalized to `[0,1]`).
- Optional `--label-format obb` writes OBB labels as 4 corner points (`cls x1 y1 ... x4 y4`, normalized).

## Quality controls

Frames are only kept when:

- Both `EYE_CLASS=2` and `PROBOSCIS_CLASS=8` are detected (unless `require_both: false`).
- Both detections exceed `min_conf_keep`.
- Optional sanity checks can reject boxes by geometry/size.

The selection score is `min(eye_conf, prob_conf)` by default.

## CLI

```bash
PYTHONPATH=src \
python -m fbpipe.steps.pseudolabel_export --config config/config.yaml \
  --dataset-out /path/to/dataset_out \
  --target-total 40000 \
  --stride 10 \
  --min-conf 0.85 \
  --val-frac 0.1 \
  --seed 1337
```

By default, the tool searches for videos under `main_directory`. If it finds **zero**
videos there (common when videos live only in secure storage), it automatically
retries using `yolo_curation.video_source_dirs`. You can always override with
`--roots ...`.

Optional scaling knobs:

- `--per-video-cap 400` limits how many frames each video can contribute.
- `--random-sample-per-video N` samples `N` random frames per video (useful when videos are extremely long).
- `--diversity-bins X Y SIZE CAP` caps selections per coarse bin of eye position and proboscis size.
- `--dry-run` writes `manifest.csv` + `data.yaml` only (no images/labels).
- `--overwrite` replaces an existing `dataset_out`.
- `--export-coco-json` also writes `annotations_train.json` and `annotations_val.json`.
- `--scan-only` only discovers videos and exits (prints the first 10).

## Config (`config/config.yaml`)

See the `pseudolabel:` section in `config/config.yaml`. CLI flags override config values.

## Manifest

`manifest.csv` includes, per exported frame:

- `source_video`, `frame_idx`
- `out_image_path`, `out_label_path`
- `eye_conf`, `prob_conf`
- `eye_bbox_xyxy`, `prob_bbox_xyxy`
- `width_px`, `height_px`
- `selection_score`, `rejection_reason`

Pseudo-labels require spot-checking before training; treat them as a starting point.
