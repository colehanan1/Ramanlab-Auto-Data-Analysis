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
- **Subsequent analysis and rendering**: plotting, histogramming, RMS/envelope calculations, dropped-frame detection, state updates, heatmaps, video moves, and RMS overlays are organized as independent modules (`plot_distance_time`, `histograms`, `rms_copy_filter`, `collect_envelope_plots`, `angle_compute_and_plots`, `angle_heatmaps`, `move_videos`, `compose_videos_rms`, etc.). Each module follows the same signature—`main(cfg: Settings)`—so you can invoke it directly or rely on the orchestrator to run everything in sequence.【F:src/fbpipe/pipeline.py†L7-L27】

## Extending the pipeline

Additions typically involve either creating a new step module in `fbpipe/steps/` or adjusting configuration defaults. New steps should accept the `Settings` object, resolve paths relative to `cfg.main_directory`, and emit artifacts under the relevant fly directory so that downstream stages can pick them up.
