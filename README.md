
# Fly Behavior Pipeline (YOLO → Distances/Angles → RMS/Envelope → Videos)

End-to-end, reproducible pipeline that:
1) Runs YOLO (OBB or axis-aligned) on videos with a Kalman tracker + optional optical flow.
2) Writes per-frame CSVs (coordinates, distances, angles).
3) Computes global min/max for distances; normalizes to percent.
4) Generates plots (time-series, histograms, envelopes).
5) Computes angles and centered-angle-% heatmaps.
6) Organizes trial videos and renders line-panel videos with RMS overlay.
7) Produces consistent, versioned outputs under each *fly* folder.

> **Minimum**: Linux + CUDA GPU. Set `model_path` and `main_directory` in `config.yaml` or `.env`.

---

## Quickstart

```bash
# 0) Prepare venv + deps
make setup

# 1) Configure paths
cp .env.example .env
# edit MODEL_PATH and MAIN_DIRECTORY, or edit config.yaml directly

# 2) Run everything
make run   # or: python -m fbpipe.pipeline --config config.yaml all
```

## Structure

```
fbpipe/
  config.py                  # load config (YAML + .env + env vars)
  pipeline.py                # simple CLI orchestrator
  utils/                     # helpers (timestamps, tracking, geometry)
  steps/                     # individual steps (you can run separately)
    yolo_infer.py
    distance_stats.py
    distance_normalize.py
    plot_distance_time.py
    detect_dropped_frames.py
    rms_copy_filter.py
    update_ofm_state.py
    histograms.py
    envelope_over_time.py
    collect_envelope_plots.py
    angle_compute_and_plots.py
    angle_heatmaps.py
    move_videos.py
    compose_videos_rms.py
```

Each step can also be run independently:
```bash
python -m fbpipe.steps.yolo_infer --config config.yaml
python -m fbpipe.steps.distance_stats --config config.yaml
...
```

## GPU requirements

The pipeline expects a CUDA-capable GPU. Set `allow_cpu: true` (or `ALLOW_CPU=true` in `.env`) **only** for smoke tests—performance will be poor.

## Notes

- The pipeline is resilient to missing files/columns; steps skip gracefully when inputs are absent.
- Angle centering uses the first frame where `distance_percentage == 0`, or otherwise the minimal absolute value across a fly.
- RMS/Envelope calculations ignore values outside `[0, 100]` and NaNs.
- Video overlay deletes source trial videos after composing by default; toggle in `config.yaml`.
