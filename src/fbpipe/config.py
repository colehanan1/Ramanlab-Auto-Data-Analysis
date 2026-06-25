
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_NAME = "config.yaml"
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / DEFAULT_CONFIG_NAME
DEFAULT_ENV_PATH = REPO_ROOT / "config" / ".env"


def _as_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


_flagged_cache: dict[tuple[str, float], set[tuple[str, str, str]]] = {}


def load_flagged_fly_exclusions(csv_path: str | Path) -> set[tuple[str, str, str]]:
    """Load the flagged-flies truth CSV and return (dataset, fly, fly_number) tuples to EXCLUDE.

    Flies with ``FLY-State`` == 1 are alive and kept; flies with 0 or -1 are
    excluded.  Flies not present in the CSV at all are *not* excluded.

    Results are cached by (resolved path, mtime) so repeated calls within a
    single pipeline run don't re-parse the CSV.
    """
    import pandas as pd

    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        print(f"[WARNING] Flagged flies CSV not found: {path}")
        return set()

    mtime = path.stat().st_mtime
    cache_key = (str(path), mtime)
    if cache_key in _flagged_cache:
        return _flagged_cache[cache_key]

    df = pd.read_csv(path)

    # Find the FLY-State column (header may contain extra text)
    state_col = None
    for col in df.columns:
        if "fly-state" in col.lower() or "fly_state" in col.lower():
            state_col = col
            break
    if state_col is None:
        print(f"[WARNING] No FLY-State column found in {path}")
        return set()

    df[state_col] = pd.to_numeric(df[state_col], errors="coerce")
    exclude_mask = df[state_col] != 1

    excluded: set[tuple[str, str, str]] = set()
    for _, row in df[exclude_mask].iterrows():
        dataset = str(row.get("dataset", "")).strip()
        fly = str(row.get("fly", "")).strip()
        fly_number = str(row.get("fly_number", "")).strip()
        excluded.add((dataset, fly, fly_number))

    if excluded:
        print(f"[INFO] Flagged flies CSV: {len(excluded)} flies marked for exclusion (state != 1)")
    _flagged_cache[cache_key] = excluded
    return excluded


def resolve_config_path(config_path: str | Path) -> Path:
    candidate = Path(config_path)
    if candidate.is_absolute():
        return candidate

    search_paths = [
        Path.cwd() / candidate,
        REPO_ROOT / candidate,
    ]
    if candidate.name == DEFAULT_CONFIG_NAME:
        search_paths.append(DEFAULT_CONFIG_PATH)

    for path in search_paths:
        if path.exists():
            return path.resolve()

    # Fall back to repo-relative path if nothing exists yet.
    return (REPO_ROOT / candidate).resolve()


def resolve_env_path(config_path: str | Path | None = None) -> Path:
    candidates = []
    if config_path is not None:
        resolved = resolve_config_path(config_path)
        candidates.append(resolved.parent / ".env")
    candidates.append(DEFAULT_ENV_PATH)
    candidates.append(REPO_ROOT / ".env")

    for path in candidates:
        if path.exists():
            return path

    return candidates[0]


def _expand_datasets(data: dict) -> dict:
    """Expand a top-level ``datasets`` list into all ``roots:``/``sources:`` lists.

    If the config contains::

        dataset_bases:
          data: /home/ramanlab/Documents/cole/Data/flys_New
          secured: /securedstorage/DATAsec/cole/Data-secured-New
        datasets:
          - 3Oct-Training
          - Hex-Control-24-0.005
          ...

    Then every ``roots:`` / ``sources:`` list that currently contains paths
    under one of those bases will be rebuilt from the ``datasets`` list,
    and ``main_directories`` will be rebuilt as well.

    If ``datasets`` is absent the config is returned unchanged.
    """
    datasets = data.get("datasets")
    if not datasets:
        return data

    bases = data.get("dataset_bases", {})
    data_base = bases.get("data", "/home/ramanlab/Documents/cole/Data/flys_New")
    secured_base = bases.get("secured", "/securedstorage/DATAsec/cole/Data-secured-New")

    # Only include directories that actually exist on disk
    data_roots = [f"{data_base.rstrip('/')}/{ds}/"
                  for ds in datasets if Path(data_base, ds).is_dir()]
    secured_roots = [f"{secured_base.rstrip('/')}/{ds}/"
                     for ds in datasets if Path(secured_base, ds).is_dir()]
    skipped = [ds for ds in datasets
               if not Path(data_base, ds).is_dir() and not Path(secured_base, ds).is_dir()]
    if skipped:
        print(f"[config] Skipping datasets not yet on disk: {', '.join(skipped)}")

    # main_directories
    data["main_directories"] = list(data_roots)

    # analysis.combined.combine.roots
    analysis = data.get("analysis", {})
    combined = analysis.get("combined", {})
    combine = combined.get("combine", {})
    if "roots" in combine:
        combine["roots"] = list(data_roots)

    # analysis.combined.wide.roots
    wide = combined.get("wide", {})
    if "roots" in wide:
        wide["roots"] = list(data_roots)

    # analysis.combined.combined_base.wide.roots — read LOCAL (data) roots, the
    # same place ``combine`` writes angle_distance_rms_envelope, so each run's
    # freshly processed data is included. (Previously read secured, which only
    # held prior runs' data → a one-run lag that dropped new datasets.)
    combined_base = combined.get("combined_base", {})
    cb_wide = combined_base.get("wide", {})
    if "roots" in cb_wide:
        cb_wide["roots"] = list(data_roots)

    # analysis.combined.distance_base.wide.roots — distance-only variant of
    # combined_base; reads the same per-trial files, so also LOCAL.
    distance_base = combined.get("distance_base", {})
    db_wide = distance_base.get("wide", {})
    if "roots" in db_wide:
        db_wide["roots"] = list(data_roots)

    # analysis.combined.secure_cleanup.sources
    cleanup = combined.get("secure_cleanup", {})
    if "sources" in cleanup:
        cleanup["sources"] = list(data_roots)

    # tools.collect_eye_prob_coords.sources (secured)
    tools = data.get("tools", {})
    eye_coords = tools.get("collect_eye_prob_coords", {})
    if "sources" in eye_coords:
        eye_coords["sources"] = list(secured_roots)

    # yolo_curation.video_source_dirs (secured)
    yolo_cur = data.get("yolo_curation", {})
    if "video_source_dirs" in yolo_cur:
        yolo_cur["video_source_dirs"] = list(secured_roots)

    return data


def load_raw_config(config_path: str | Path) -> Dict[str, Any]:
    path = resolve_config_path(config_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return _expand_datasets(data)
    return {}


def discover_flagged_directories(flagged_root: str | Path) -> list[Path]:
    """Return all immediate subdirectories of the flagged experiments root.

    Any new folder added under *flagged_root* will be picked up automatically.
    """
    root = Path(flagged_root).expanduser().resolve()
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def get_main_directories(cfg: Settings) -> list[Path]:
    """
    Normalize main_directories to always return a list of Path objects.

    Handles both single string and list of strings from config.
    Automatically appends any subdirectories found under ``flagged_root``
    unless auto_discover_flagged is set to False.

    Args:
        cfg: Settings object

    Returns:
        List of Path objects for all main directories
    """
    if isinstance(cfg.main_directories, list):
        dirs = [Path(d).expanduser().resolve() for d in cfg.main_directories]
    else:
        dirs = [Path(cfg.main_directories).expanduser().resolve()]

    if cfg.auto_discover_flagged and cfg.flagged_root:
        dirs.extend(discover_flagged_directories(cfg.flagged_root))

    return dirs


def infer_dataset_for_path(cfg: "Settings", path: str | Path) -> str:
    """Return the dataset name that ``path`` lives under.

    Matches path components against ``cfg.datasets``. Empty string when nothing
    matches (older configs without a top-level ``datasets:`` list).
    """
    parts = set(Path(path).resolve().parts)
    for ds in cfg.datasets:
        if ds in parts:
            return ds
    lower = {p.lower(): p for p in parts}
    for ds in cfg.datasets:
        if ds.lower() in lower:
            return ds
    return ""


def get_dataset_override(cfg: "Settings", path: str | Path) -> "DatasetOverride":
    """Return the ``DatasetOverride`` for the dataset containing ``path``.

    Returns an empty ``DatasetOverride`` (all fields ``None``) if no override
    is configured for that dataset, so callers can use it unconditionally.
    """
    ds = infer_dataset_for_path(cfg, path)
    if ds and ds in cfg.dataset_overrides:
        return cfg.dataset_overrides[ds]
    return DatasetOverride()


@dataclass
class ReactionMatrixSettings:
    out_dir: str = ""
    out_dir_smb: str = ""  # SMB path for matrix figures export
    latency_sec: float = 2.15
    after_window_sec: float = 30.0
    row_gap: float = 0.6
    height_per_gap_in: float = 3.0
    bottom_shift_in: float = 0.5
    trial_orders: Tuple[str, ...] = ("observed", "trained-first")
    include_hexanol: bool = True
    overwrite: bool = False


@dataclass
class ReactionPredictionSettings:
    data_csv: str = ""
    model_path: str = ""
    output_csv: str = ""
    output_csv_smb: str = ""  # SMB path for CSV export
    python: str = ""
    threshold: float | None = None
    run_prediction: bool = True
    model_type: str = "binary"  # "binary" (logistic regression) or "ordinal" (XGBoost -1..5)
    binary_threshold: int = 2   # ordinal score >= this → reaction (only used when model_type="ordinal")
    matrix: ReactionMatrixSettings = field(default_factory=ReactionMatrixSettings)


@dataclass
class TrackingConfig:
    """Tracking quality thresholds for filtering flies with poor proboscis detection."""
    max_missing_frames_per_trial: int = 5000
    max_missing_frames_pct_per_trial: float = 50.0  # Percent
    apply_missing_frame_check: bool = True  # Allow disabling if not needed


@dataclass
class YoloCurationQualityThresholds:
    """Quality thresholds for identifying problematic tracking videos."""
    max_jitter_px: float = 50.0  # Flag if median jitter exceeds this (pixels)
    max_missing_pct: float = 0.10  # Flag if >10% frames missing


@dataclass
class YoloCurationTargetFrames:
    """Configuration for frame extraction targets."""
    per_video: int = 10  # Extract ~10 frames per bad video
    total: int = 200  # Stop extraction at 200 total frames (not yet implemented)
    stratification: Dict[str, float] = field(default_factory=lambda: {
        "high_quality": 0.30,  # 30% from low jitter, valid tracking
        "low_quality": 0.50,   # 50% from high jitter or missing
        "boundary": 0.20,      # 20% from moderate quality
    })


@dataclass
class YoloCurationAugmentation:
    """Data augmentation configuration."""
    enabled: bool = True
    strategies: Tuple[str, ...] = (
        "horizontal_flip",
        "brightness_contrast_jitter",
        "minor_rotation",
    )
    multiplier: int = 2  # Aim for 2× dataset size via augmentation


@dataclass
class YoloCurationSettings:
    """Configuration for YOLO dataset curation module."""
    enabled: bool = True
    quality_thresholds: YoloCurationQualityThresholds = field(
        default_factory=YoloCurationQualityThresholds
    )
    target_frames: YoloCurationTargetFrames = field(
        default_factory=YoloCurationTargetFrames
    )
    augmentation: YoloCurationAugmentation = field(
        default_factory=YoloCurationAugmentation
    )
    output_dir: str = "yolo_curation"  # Relative to FLY_DIR
    video_source_dirs: Tuple[str, ...] = ()  # Additional directories to search for videos


@dataclass
class ProboscisFilterSettings:
    """Downstream rejection of physically-impossible proboscis detections.

    Runs *after* tracking on the per-fly distance CSVs. It only blanks (NaNs)
    suspect points — it never fabricates or moves a detection, so any gap it
    leaves can be interpolated later if desired. Two independent gates:

    * geometry: anisotropic — the proboscis can extend the full
      ``max_eye_prob_distance_px`` left, right, and down of its eye, but only
      ``max_eye_prob_distance_px / up_divisor`` upward. This is a flat-topped blob
      around the frozen eye, applied for every fly count. (``up_divisor`` of 1.0
      collapses it back to a plain circle.)
    * velocity: drop any proboscis that jumps more than ``max_jump_px`` from the
      previous accepted position between consecutive detections. Catches eye→
      wrong-proboscis switches and model hallucinations that land inside the
      radius.
    """

    enabled: bool = True
    max_eye_prob_distance_px: float = 150.0
    up_divisor: float = 4.0
    max_jump_px: float = 80.0


@dataclass
class PseudolabelDiversityBins:
    enabled: bool = False
    x_bins: int = 8
    y_bins: int = 8
    size_bins: int = 6
    per_bin_cap: int = 200


@dataclass
class PseudolabelSettings:
    enabled: bool = False
    target_total: int = 40000
    stride: int = 10
    random_sample_per_video: int = 0
    batch_size: int = 16

    # Confidence controls
    min_conf_keep: float = 0.85
    min_conf_export: float = 0.85

    # Dataset split
    val_frac: float = 0.1
    seed: int = 1337

    # Diversity / scaling controls
    per_video_cap: int = 400
    diversity_bins: PseudolabelDiversityBins = field(default_factory=PseudolabelDiversityBins)

    # Label quality controls
    require_both: bool = True
    export_classes: Tuple[int, ...] = (0, 1)
    max_eye_prob_center_dist_px: float = 0.0  # 0 disables the sanity check
    min_box_area_px: float = 0.0  # 0 disables the sanity check
    max_box_area_frac: float = 1.0  # 1 disables the sanity check
    reject_multi_eye_first_n_frames: int = 5
    reject_multi_eye_zero_iou_eps: float = 1e-9

    # Output controls
    dataset_out: str = ""
    image_ext: str = "jpg"
    jpeg_quality: int = 95
    label_format: str = "bbox"  # "bbox" (default) or "obb"
    export_coco_json: bool = False


@dataclass
class ForceSettings:
    pipeline: bool = True
    yolo: bool = True
    combined: bool = True
    reaction_prediction: bool = True
    reaction_matrix: bool = True
    envelope_visuals: bool = True
    training: bool = True
    dataset_means: bool = True


@dataclass
class ParallelSettings:
    """Opt-in CPU parallelism for the per-fly/per-CSV pipeline stages.

    Disabled by default so behaviour is identical to the historical serial
    pipeline unless explicitly enabled. ``n_jobs <= 0`` resolves to
    ``max(1, cpu_count - 2)`` at runtime.
    """

    enabled: bool = False
    n_jobs: int = 0  # 0 -> auto (cpu_count - 2)


@dataclass
class DatasetOverride:
    """Per-dataset overrides resolved at trial-loading time.

    ``trial_type_override`` forces every trial in the dataset into the chosen
    pipeline branch (``"training"`` or ``"testing"``) regardless of what the
    folder or sidecar cycle name says. ``odor_on_s`` / ``odor_off_s`` are used
    only as a last-resort fallback when the rig did not write an
    ``ActiveOFM`` sensor CSV. ``figure_output_subdir`` (if set) is appended to
    every figure output path so the dataset lands in its own folder.
    """

    trial_type_override: Optional[str] = None
    figure_output_subdir: Optional[str] = None
    odor_on_s: Optional[float] = None
    odor_off_s: Optional[float] = None
    # Light-only datasets (no odor delivery): when ``light_only`` is True,
    # plotting code should suppress the odor span and draw a light annotation
    # at ``[light_start_s, light_start_s + light_duration_s]`` instead.
    light_only: bool = False
    light_start_s: Optional[float] = None
    light_duration_s: Optional[float] = None
    # Per-dataset display-label remap. Keys are the display odor names
    # produced by the v2 trial-label resolver (e.g. ``"Citral"``) and values
    # are the substitute labels to render in figures (e.g.
    # ``"Sour Dough Yeast (25%)"``). Used when a dataset's rig configuration
    # delivered a different odor than the trial-label suffix claims, so the
    # figures need a per-dataset override.
    odor_remap: Dict[str, str] = field(default_factory=dict)


@dataclass
class Settings:
    model_path: str
    main_directories: str | list[str]  # Can be single path or list of paths
    cache_dir: str = ""
    flagged_root: str = ""  # Parent dir for flagged/bad experiments; subdirs auto-discovered
    flagged_secured_root: str = ""  # Secured storage mirror for flagged experiments
    auto_discover_flagged: bool = True  # If False, only use directories explicitly listed in main_directories
    allow_cpu: bool = False
    cuda_allow_tf32: bool = True
    non_reactive_span_px: float = 5.0
    flagged_flies_csv: str = ""
    anchor_x: float = 1079.0
    anchor_y: float = 540.0
    fps_default: float = 40.0
    window_sec: float = 0.25
    odor_on_s: float = 30.0
    odor_off_s: float = 60.0
    odor_latency_s: float = 0.0
    delete_source_after_render: bool = True

    # detector/tracker
    conf_thres: float = 0.40
    iou_match_thres: float = 0.25
    max_age: int = 15
    ema_alpha: float = 0.20
    use_optical_flow: bool = True
    flow_skip_edge: int = 10
    max_flies: int = 4
    max_proboscis_tracks: int = 4  # Backward-compatible config field; YOLO now mirrors the initial eye count.
    proboscis_match_max_dist_px: float = 80.0  # Center-distance gate for proboscis track association (small/fast object).
    pair_rebind_ratio: float = 0.20
    zero_iou_epsilon: float = 1e-8
    inference_batch_size: int = 32  # YOLO predict() chunk size; clamped to 1 for non-batch-capable .engine models
    engine_supports_batch: bool = False  # set True only for a dynamic-batch .engine (see scripts/convert/export_tensorrt.py --dynamic)

    # distance limits
    class2_min: float = 70.0
    class2_max: float = 250.0
    three_fly_max_eye_prob_distance_px: float = 180.0

    # reaction prediction
    reaction_prediction: ReactionPredictionSettings = field(default_factory=ReactionPredictionSettings)
    force: ForceSettings = field(default_factory=ForceSettings)

    # opt-in CPU parallelism for per-fly/per-CSV stages
    parallel: ParallelSettings = field(default_factory=ParallelSettings)

    # tracking quality
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    # YOLO dataset curation
    yolo_curation: YoloCurationSettings = field(default_factory=YoloCurationSettings)

    # Downstream proboscis rejection (geometry + velocity gates)
    proboscis_filter: ProboscisFilterSettings = field(default_factory=ProboscisFilterSettings)

    # Pseudolabel mining + export
    pseudolabel: PseudolabelSettings = field(default_factory=PseudolabelSettings)

    # Protocol version: "legacy" (old flys/) or "v2" (new flys_New/)
    protocol: str = "legacy"

    # Datasets declared under the top-level `datasets:` block (used by helpers
    # to infer the dataset name for an arbitrary trial path).
    datasets: Tuple[str, ...] = field(default_factory=tuple)

    # Per-dataset overrides keyed by the literal dataset name. Empty by default
    # so existing configs are untouched.
    dataset_overrides: Dict[str, DatasetOverride] = field(default_factory=dict)

def _get(d: Dict[str, Any], key: str, default: Any):
    return d.get(key, default)

def _parse_env_paths(raw_value: str) -> str | list[str]:
    value = raw_value.strip()
    if not value:
        return ""
    if "," in value:
        return [p.strip() for p in value.split(",") if p.strip()]
    if os.pathsep in value:
        return [p.strip() for p in value.split(os.pathsep) if p.strip()]
    return value


def load_settings(config_path: str | Path) -> Settings:
    env_path = resolve_env_path(config_path)
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    p = resolve_config_path(config_path)
    data: Dict[str, Any] = {}
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        data = _expand_datasets(data)

    # env overrides
    model_path = os.getenv("MODEL_PATH", _get(data, "model_path", ""))

    main_dirs_cfg = data.get("main_directories", None)
    if main_dirs_cfg is None:
        main_dirs_cfg = _get(data, "main_directory", "")

    main_dirs_env = os.getenv("MAIN_DIRECTORIES")
    if main_dirs_env is not None:
        main_directories = _parse_env_paths(main_dirs_env)
    else:
        main_dir_env = os.getenv("MAIN_DIRECTORY")
        if main_dir_env is not None:
            main_directories = _parse_env_paths(main_dir_env)
        else:
            main_directories = main_dirs_cfg

    allow_cpu = os.getenv("ALLOW_CPU", str(_get(data, "allow_cpu", False))).lower() == "true"
    cuda_allow_tf32 = os.getenv("CUDA_ALLOW_TF32", str(_get(data, "cuda_allow_tf32", True))).lower() == "true"

    cache_dir_cfg = _get(data, "cache_dir", "")
    cache_dir_env = os.getenv("CACHE_DIR")
    if cache_dir_env:
        cache_dir = cache_dir_env
    elif cache_dir_cfg:
        cache_dir = str(cache_dir_cfg)
    else:
        cache_dir = str(Path.home() / ".cache" / "ramanlab_auto_data_analysis")

    # nested
    yolo = data.get("yolo", {})
    dist_limits = data.get("distance_limits", {})
    reaction_cfg = data.get("reaction_prediction", {})
    reaction_matrix_cfg = reaction_cfg.get("matrix", {})

    include_hex_cfg = reaction_matrix_cfg.get("include_hexanol", True)
    if isinstance(include_hex_cfg, str):
        include_hex_cfg = include_hex_cfg.lower() == "true"

    overwrite_cfg = reaction_matrix_cfg.get("overwrite", False)
    if isinstance(overwrite_cfg, str):
        overwrite_cfg = overwrite_cfg.lower() == "true"

    matrix_defaults = ReactionMatrixSettings(
        out_dir=str(reaction_matrix_cfg.get("out_dir", "")),
        latency_sec=float(reaction_matrix_cfg.get("latency_sec", 2.15)),
        after_window_sec=float(reaction_matrix_cfg.get("after_window_sec", 30.0)),
        row_gap=float(reaction_matrix_cfg.get("row_gap", 0.6)),
        height_per_gap_in=float(reaction_matrix_cfg.get("height_per_gap_in", 3.0)),
        bottom_shift_in=float(reaction_matrix_cfg.get("bottom_shift_in", 0.5)),
        trial_orders=tuple(reaction_matrix_cfg.get("trial_orders", ("observed", "trained-first"))),
        include_hexanol=bool(include_hex_cfg),
        overwrite=bool(overwrite_cfg),
    )

    trial_orders_env = os.getenv("REACTION_MATRIX_TRIAL_ORDERS")
    if trial_orders_env:
        env_trials = [item.strip() for item in trial_orders_env.split(",") if item.strip()]
        if env_trials:
            matrix_defaults.trial_orders = tuple(env_trials)

    include_hex_env = os.getenv("REACTION_MATRIX_INCLUDE_HEXANOL")
    if include_hex_env is not None:
        matrix_defaults.include_hexanol = include_hex_env.lower() == "true"

    overwrite_env = os.getenv("REACTION_MATRIX_OVERWRITE")
    if overwrite_env is not None:
        matrix_defaults.overwrite = overwrite_env.lower() == "true"

    matrix = ReactionMatrixSettings(
        out_dir=str(os.getenv("REACTION_MATRIX_OUT_DIR", matrix_defaults.out_dir)),
        latency_sec=float(os.getenv("REACTION_MATRIX_LATENCY_SEC", matrix_defaults.latency_sec)),
        after_window_sec=float(
            os.getenv("REACTION_MATRIX_AFTER_WINDOW_SEC", matrix_defaults.after_window_sec)
        ),
        row_gap=float(os.getenv("REACTION_MATRIX_ROW_GAP", matrix_defaults.row_gap)),
        height_per_gap_in=float(
            os.getenv("REACTION_MATRIX_HEIGHT_PER_GAP_IN", matrix_defaults.height_per_gap_in)
        ),
        bottom_shift_in=float(
            os.getenv("REACTION_MATRIX_BOTTOM_SHIFT_IN", matrix_defaults.bottom_shift_in)
        ),
        trial_orders=matrix_defaults.trial_orders,
        include_hexanol=matrix_defaults.include_hexanol,
        overwrite=matrix_defaults.overwrite,
    )

    reaction_python = str(os.getenv("REACTION_PREDICTION_PYTHON", reaction_cfg.get("python", "")))

    run_prediction_cfg = reaction_cfg.get("run_prediction", True)
    if isinstance(run_prediction_cfg, str):
        run_prediction_cfg = run_prediction_cfg.lower() == "true"
    run_prediction_env = os.getenv("REACTION_RUN_PREDICTION")
    if run_prediction_env is not None:
        run_prediction_cfg = run_prediction_env.lower() == "true"

    threshold_value = reaction_cfg.get("threshold")
    threshold_env = os.getenv("REACTION_THRESHOLD")
    if threshold_env is not None:
        threshold_value = float(threshold_env)
    elif threshold_value is not None:
        threshold_value = float(threshold_value)

    reaction_prediction = ReactionPredictionSettings(
        data_csv=str(os.getenv("REACTION_DATA_CSV", reaction_cfg.get("data_csv", ""))),
        model_path=str(os.getenv("REACTION_MODEL_PATH", reaction_cfg.get("model_path", ""))),
        output_csv=str(os.getenv("REACTION_OUTPUT_CSV", reaction_cfg.get("output_csv", ""))),
        python=reaction_python,
        threshold=threshold_value,
        run_prediction=bool(run_prediction_cfg),
        model_type=str(reaction_cfg.get("model_type", "binary")),
        binary_threshold=int(reaction_cfg.get("binary_threshold", 2)),
        matrix=matrix,
    )

    force_cfg_raw = data.get("force") if isinstance(data.get("force"), dict) else {}
    force = ForceSettings(
        pipeline=_as_bool(force_cfg_raw.get("pipeline"), True),
        yolo=_as_bool(force_cfg_raw.get("yolo"), True),
        combined=_as_bool(force_cfg_raw.get("combined"), True),
        reaction_prediction=_as_bool(force_cfg_raw.get("reaction_prediction"), True),
        reaction_matrix=_as_bool(force_cfg_raw.get("reaction_matrix"), True),
        envelope_visuals=_as_bool(force_cfg_raw.get("envelope_visuals"), True),
        training=_as_bool(force_cfg_raw.get("training"), True),
        dataset_means=_as_bool(force_cfg_raw.get("dataset_means"), True),
    )

    force.pipeline = _as_bool(os.getenv("FORCE_PIPELINE"), force.pipeline)
    force.yolo = _as_bool(os.getenv("FORCE_YOLO"), force.yolo)
    force.combined = _as_bool(os.getenv("FORCE_COMBINED"), force.combined)
    force.reaction_prediction = _as_bool(
        os.getenv("FORCE_REACTION_PREDICTION"), force.reaction_prediction
    )
    force.reaction_matrix = _as_bool(
        os.getenv("FORCE_REACTION_MATRIX"), force.reaction_matrix
    )
    force.envelope_visuals = _as_bool(
        os.getenv("FORCE_ENVELOPE_VISUALS"), force.envelope_visuals
    )
    force.training = _as_bool(os.getenv("FORCE_TRAINING"), force.training)
    force.dataset_means = _as_bool(os.getenv("FORCE_DATASET_MEANS"), force.dataset_means)

    non_reactive_span_px = float(
        os.getenv("NON_REACTIVE_SPAN_PX", _get(data, "non_reactive_span_px", 5.0))
    )

    flagged_flies_csv = str(
        os.getenv("FLAGGED_FLIES_CSV", _get(data, "flagged_flies_csv", ""))
    )

    # tracking quality config
    tracking_cfg = data.get("tracking", {})
    tracking = TrackingConfig(
        max_missing_frames_per_trial=int(
            os.getenv("TRACKING_MAX_MISSING_FRAMES", tracking_cfg.get("max_missing_frames_per_trial", 5000))
        ),
        max_missing_frames_pct_per_trial=float(
            os.getenv("TRACKING_MAX_MISSING_PCT", tracking_cfg.get("max_missing_frames_pct_per_trial", 50.0))
        ),
        apply_missing_frame_check=_as_bool(
            os.getenv("TRACKING_APPLY_CHECK", tracking_cfg.get("apply_missing_frame_check", True)), True
        ),
    )

    # YOLO curation config
    curation_cfg = data.get("yolo_curation", {})
    quality_thresh_cfg = curation_cfg.get("quality_thresholds", {})
    target_frames_cfg = curation_cfg.get("target_frames", {})
    augmentation_cfg = curation_cfg.get("augmentation", {})

    yolo_curation_quality = YoloCurationQualityThresholds(
        max_jitter_px=float(quality_thresh_cfg.get("max_jitter_px", 50.0)),
        max_missing_pct=float(quality_thresh_cfg.get("max_missing_pct", 0.10)),
    )

    stratification_cfg = target_frames_cfg.get("stratification", {})
    yolo_curation_target_frames = YoloCurationTargetFrames(
        per_video=int(target_frames_cfg.get("per_video", 10)),
        total=int(target_frames_cfg.get("total", 200)),
        stratification={
            "high_quality": float(stratification_cfg.get("high_quality", 0.30)),
            "low_quality": float(stratification_cfg.get("low_quality", 0.50)),
            "boundary": float(stratification_cfg.get("boundary", 0.20)),
        },
    )

    augmentation_strategies = augmentation_cfg.get("strategies", [
        "horizontal_flip",
        "brightness_contrast_jitter",
        "minor_rotation",
    ])
    yolo_curation_augmentation = YoloCurationAugmentation(
        enabled=_as_bool(augmentation_cfg.get("enabled", True), True),
        strategies=tuple(augmentation_strategies),
        multiplier=int(augmentation_cfg.get("multiplier", 2)),
    )

    video_source_dirs = curation_cfg.get("video_source_dirs", [])
    if not isinstance(video_source_dirs, list):
        video_source_dirs = []

    yolo_curation = YoloCurationSettings(
        enabled=_as_bool(curation_cfg.get("enabled", False), False),
        quality_thresholds=yolo_curation_quality,
        target_frames=yolo_curation_target_frames,
        augmentation=yolo_curation_augmentation,
        output_dir=str(curation_cfg.get("output_dir", "yolo_curation")),
        video_source_dirs=tuple(video_source_dirs),
    )

    # proboscis filter config (downstream geometry + velocity rejection)
    prob_filter_cfg = data.get("proboscis_filter", {})
    if not isinstance(prob_filter_cfg, dict):
        prob_filter_cfg = {}
    proboscis_filter = ProboscisFilterSettings(
        enabled=_as_bool(
            os.getenv("PROBOSCIS_FILTER_ENABLED", prob_filter_cfg.get("enabled", True)), True
        ),
        max_eye_prob_distance_px=float(
            os.getenv(
                "PROBOSCIS_MAX_EYE_DIST_PX",
                prob_filter_cfg.get("max_eye_prob_distance_px", 150.0),
            )
        ),
        up_divisor=float(
            os.getenv(
                "PROBOSCIS_UP_DIVISOR",
                prob_filter_cfg.get("up_divisor", 4.0),
            )
        ),
        max_jump_px=float(
            os.getenv("PROBOSCIS_MAX_JUMP_PX", prob_filter_cfg.get("max_jump_px", 80.0))
        ),
    )

    # pseudolabel config
    pseudolabel_cfg = data.get("pseudolabel", {})
    if not isinstance(pseudolabel_cfg, dict):
        pseudolabel_cfg = {}

    min_conf = pseudolabel_cfg.get("min_conf", 0.85)
    min_conf_keep = float(pseudolabel_cfg.get("min_conf_keep", min_conf))
    min_conf_export = float(pseudolabel_cfg.get("min_conf_export", min_conf))

    export_classes_cfg = pseudolabel_cfg.get("export_classes", [2, 8])
    if not isinstance(export_classes_cfg, list):
        export_classes_cfg = [2, 8]
    export_classes_tuple = tuple(int(x) for x in export_classes_cfg)

    diversity_cfg = pseudolabel_cfg.get("diversity_bins", {})
    if not isinstance(diversity_cfg, dict):
        diversity_cfg = {}
    pseudolabel_diversity = PseudolabelDiversityBins(
        enabled=_as_bool(diversity_cfg.get("enabled", False), False),
        x_bins=int(diversity_cfg.get("x_bins", 8)),
        y_bins=int(diversity_cfg.get("y_bins", 8)),
        size_bins=int(diversity_cfg.get("size_bins", 6)),
        per_bin_cap=int(diversity_cfg.get("per_bin_cap", 200)),
    )

    pseudolabel = PseudolabelSettings(
        enabled=_as_bool(pseudolabel_cfg.get("enabled", False), False),
        target_total=int(pseudolabel_cfg.get("target_total", 40000)),
        stride=int(pseudolabel_cfg.get("stride", 10)),
        random_sample_per_video=int(pseudolabel_cfg.get("random_sample_per_video", 0)),
        batch_size=int(pseudolabel_cfg.get("batch_size", 16)),
        min_conf_keep=min_conf_keep,
        min_conf_export=min_conf_export,
        val_frac=float(pseudolabel_cfg.get("val_frac", 0.1)),
        seed=int(pseudolabel_cfg.get("seed", 1337)),
        per_video_cap=int(pseudolabel_cfg.get("per_video_cap", 400)),
        diversity_bins=pseudolabel_diversity,
        require_both=_as_bool(pseudolabel_cfg.get("require_both", True), True),
        export_classes=export_classes_tuple,
        max_eye_prob_center_dist_px=float(pseudolabel_cfg.get("max_eye_prob_center_dist_px", 0.0)),
        min_box_area_px=float(pseudolabel_cfg.get("min_box_area_px", 0.0)),
        max_box_area_frac=float(pseudolabel_cfg.get("max_box_area_frac", 1.0)),
        reject_multi_eye_first_n_frames=int(
            pseudolabel_cfg.get("reject_multi_eye_first_n_frames", 5)
        ),
        reject_multi_eye_zero_iou_eps=float(
            pseudolabel_cfg.get("reject_multi_eye_zero_iou_eps", 1e-9)
        ),
        dataset_out=str(pseudolabel_cfg.get("dataset_out", "")),
        image_ext=str(pseudolabel_cfg.get("image_ext", "jpg")),
        jpeg_quality=int(pseudolabel_cfg.get("jpeg_quality", 95)),
        label_format=str(pseudolabel_cfg.get("label_format", "bbox")),
        export_coco_json=_as_bool(pseudolabel_cfg.get("export_coco_json", False), False),
    )

    # Datasets declared at the top level (post-_expand_datasets the original
    # literal list is preserved on `data["datasets"]`).
    datasets_cfg = data.get("datasets") or []
    if not isinstance(datasets_cfg, list):
        datasets_cfg = []
    datasets_tuple = tuple(str(d) for d in datasets_cfg if d)

    # Per-dataset overrides
    raw_overrides = data.get("dataset_overrides") or {}
    if not isinstance(raw_overrides, dict):
        raw_overrides = {}
    dataset_overrides: Dict[str, DatasetOverride] = {}
    for ds_name, block in raw_overrides.items():
        if not isinstance(block, dict):
            continue
        dataset_overrides[str(ds_name)] = DatasetOverride(
            trial_type_override=(
                str(block["trial_type_override"]).strip().lower()
                if block.get("trial_type_override") is not None
                else None
            ),
            figure_output_subdir=(
                str(block["figure_output_subdir"]).strip()
                if block.get("figure_output_subdir")
                else None
            ),
            odor_on_s=(
                float(block["odor_on_s"]) if block.get("odor_on_s") is not None else None
            ),
            odor_off_s=(
                float(block["odor_off_s"]) if block.get("odor_off_s") is not None else None
            ),
            light_only=bool(block.get("light_only", False)),
            light_start_s=(
                float(block["light_start_s"])
                if block.get("light_start_s") is not None
                else None
            ),
            light_duration_s=(
                float(block["light_duration_s"])
                if block.get("light_duration_s") is not None
                else None
            ),
            odor_remap={
                str(k): str(v)
                for k, v in (block.get("odor_remap") or {}).items()
                if k is not None and v is not None
            },
        )

    flagged_root = str(
        os.getenv("FLAGGED_ROOT", _get(data, "flagged_root", ""))
    )
    flagged_secured_root = str(
        os.getenv("FLAGGED_SECURED_ROOT", _get(data, "flagged_secured_root", ""))
    )
    auto_discover_flagged = _as_bool(
        os.getenv("AUTO_DISCOVER_FLAGGED", _get(data, "auto_discover_flagged", True)),
        True
    )

    parallel_cfg_raw = data.get("parallel") if isinstance(data.get("parallel"), dict) else {}
    parallel = ParallelSettings(
        enabled=_as_bool(os.getenv("PARALLEL_ENABLED", parallel_cfg_raw.get("enabled")), False),
        n_jobs=int(os.getenv("PARALLEL_N_JOBS", parallel_cfg_raw.get("n_jobs", 0))),
    )

    return Settings(
        model_path=model_path,
        main_directories=main_directories,
        cache_dir=str(Path(cache_dir).expanduser()),
        flagged_root=flagged_root,
        flagged_secured_root=flagged_secured_root,
        auto_discover_flagged=auto_discover_flagged,
        allow_cpu=allow_cpu,
        cuda_allow_tf32=cuda_allow_tf32,
        non_reactive_span_px=non_reactive_span_px,
        flagged_flies_csv=flagged_flies_csv,
        anchor_x=float(os.getenv("ANCHOR_X", _get(data, "anchor_x", 1079.0))),
        anchor_y=float(os.getenv("ANCHOR_Y", _get(data, "anchor_y", 540.0))),
        fps_default=float(os.getenv("FPS_DEFAULT", _get(data, "fps_default", 40.0))),
        window_sec=float(os.getenv("WINDOW_SEC", _get(data, "window_sec", 0.25))),
        odor_on_s=float(os.getenv("ODOR_ON_S", _get(data, "odor_on_s", 30.0))),
        odor_off_s=float(os.getenv("ODOR_OFF_S", _get(data, "odor_off_s", 60.0))),
        odor_latency_s=float(os.getenv("ODOR_LATENCY_S", _get(data, "odor_latency_s", 0.0))),
        delete_source_after_render=(os.getenv("DELETE_SOURCE_AFTER_RENDER", str(_get(data, "delete_source_after_render", True))).lower()=="true"),

        conf_thres=float(os.getenv("CONF_THRES", yolo.get("conf_thres", 0.40))),
        iou_match_thres=float(os.getenv("IOU_MATCH_THRES", yolo.get("iou_match_thres", 0.25))),
        max_age=int(os.getenv("MAX_AGE", yolo.get("max_age", 15))),
        ema_alpha=float(os.getenv("EMA_ALPHA", yolo.get("ema_alpha", 0.20))),
        use_optical_flow=(os.getenv("USE_OPTICAL_FLOW", str(yolo.get("use_optical_flow", True))).lower()=="true"),
        flow_skip_edge=int(os.getenv("FLOW_SKIP_EDGE", yolo.get("flow_skip_edge", 10))),
        max_flies=int(os.getenv("MAX_FLIES", yolo.get("max_flies", 4))),
        max_proboscis_tracks=int(os.getenv("MAX_PROBOSCIS_TRACKS", yolo.get("max_proboscis_tracks", 4))),
        proboscis_match_max_dist_px=float(
            os.getenv("PROBOSCIS_MATCH_MAX_DIST_PX", yolo.get("proboscis_match_max_dist_px", 80.0))
        ),
        pair_rebind_ratio=float(os.getenv("PAIR_REBIND_RATIO", yolo.get("pair_rebind_ratio", 0.20))),
        zero_iou_epsilon=float(os.getenv("ZERO_IOU_EPSILON", yolo.get("zero_iou_epsilon", 1e-8))),
        inference_batch_size=int(os.getenv("INFERENCE_BATCH_SIZE", yolo.get("inference_batch_size", 32))),
        engine_supports_batch=(os.getenv("ENGINE_SUPPORTS_BATCH", str(yolo.get("engine_supports_batch", False))).lower()=="true"),

        class2_min=float(os.getenv("CLASS2_MIN", dist_limits.get("class2_min", 70.0))),
        class2_max=float(os.getenv("CLASS2_MAX", dist_limits.get("class2_max", 250.0))),
        three_fly_max_eye_prob_distance_px=float(
            os.getenv(
                "THREE_FLY_MAX_EYE_PROB_DISTANCE_PX",
                dist_limits.get("three_fly_max_eye_prob_distance_px", 180.0),
            )
        ),
        reaction_prediction=reaction_prediction,
        force=force,
        parallel=parallel,
        tracking=tracking,
        yolo_curation=yolo_curation,
        proboscis_filter=proboscis_filter,
        pseudolabel=pseudolabel,
        protocol=str(_get(data, "protocol", "legacy")),
        datasets=datasets_tuple,
        dataset_overrides=dataset_overrides,
    )
