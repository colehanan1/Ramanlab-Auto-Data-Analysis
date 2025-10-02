# Pipeline Overview

This document explains how the Fly Behavior Pipeline organizes its computation and where to look in the source tree when you need to extend a stage.

## Configuration loading

`fbpipe.config.load_settings` merges values from `config.yaml`, optional `.env`, and process environment variables into a single `Settings` dataclass. Paths such as `model_path` and `main_directory`, CUDA tuning flags, and detection thresholds are all centralized there.【F:src/fbpipe/config.py†L1-L73】

## Execution graph

`fbpipe.pipeline` defines the ordered list of processing stages and wires them into a simple CLI. Calling `python -m fbpipe.pipeline --config config.yaml all` loads the consolidated settings and then runs each step in `STEPS_IN_ORDER`. Selecting specific step names on the command line runs only those functions with the same shared configuration object.【F:src/fbpipe/pipeline.py†L1-L59】

## Core stages

The `fbpipe.steps` package contains modular functions that operate on one fly directory at a time under the configured `main_directory` root. Each stage both consumes and produces files that the downstream stages pick up.

- **YOLO inference** (`yolo_infer.main`): loads the Ultralytics YOLO model, applies optional optical-flow nudging, writes per-frame CSV measurements, and saves annotated videos per trial.【F:src/fbpipe/steps/yolo_infer.py†L1-L132】【F:src/fbpipe/steps/yolo_infer.py†L158-L213】
- **Distance statistics** (`distance_stats.main`): scans merged CSVs to compute global min/max distances for class-2 detections within configured limits, persisting `global_distance_stats_class_2.json` per fly.【F:src/fbpipe/steps/distance_stats.py†L1-L26】
- **Distance normalization** (`distance_normalize.main`): reads the stored min/max values to convert raw distances to percentage scores and stores them alongside the raw measurements for later visualization.【F:src/fbpipe/steps/distance_normalize.py†L1-L28】
- **Data hygiene** (`detect_dropped_frames.main` & `rms_copy_filter.main`): records missing/NaN frames and stages curated CSV columns in `RMS_calculations/` for downstream exports.【F:src/fbpipe/steps/detect_dropped_frames.py†L1-L30】【F:src/fbpipe/steps/rms_copy_filter.py†L1-L24】
- **OFM annotation** (`update_ofm_state.main`): folds `ActiveOFM` metadata into the staged CSVs so behavioural phases are available to later consumers.【F:src/fbpipe/steps/update_ofm_state.py†L1-L31】
- **Video delivery** (`move_videos.main` & `compose_videos_rms.main`): moves annotated trial videos and optionally renders RMS overlays once all metadata is baked in.【F:src/fbpipe/steps/move_videos.py†L1-L140】【F:src/fbpipe/steps/compose_videos_rms.py†L1-L170】

All visualisations built from Hilbert envelopes now flow through the float16 matrix emitted by `scripts/envelope_exports.py`; the follow-up scripts (`scripts/envelope_visuals.py` and `scripts/envelope_training.py`) generate reaction matrices, per-fly envelopes, and latency summaries directly from that matrix without touching raw CSVs.【F:scripts/envelope_exports.py†L190-L312】【F:scripts/envelope_visuals.py†L541-L720】【F:scripts/envelope_training.py†L1-L275】

## Extending the pipeline

Additions typically involve either creating a new step module in `fbpipe/steps/` or adjusting configuration defaults. New steps should accept the `Settings` object, resolve paths relative to `cfg.main_directory`, and emit artifacts under the relevant fly directory so that downstream stages can pick them up.
