
from __future__ import annotations
import os, yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from dotenv import load_dotenv


def _as_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def get_main_directories(cfg: Settings) -> list[Path]:
    """
    Normalize main_directory to always return a list of Path objects.

    Handles both single string and list of strings from config.

    Args:
        cfg: Settings object

    Returns:
        List of Path objects for all main directories
    """
    if isinstance(cfg.main_directory, list):
        return [Path(d).expanduser().resolve() for d in cfg.main_directory]
    else:
        return [Path(cfg.main_directory).expanduser().resolve()]


@dataclass
class ReactionMatrixSettings:
    out_dir: str = ""
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
    python: str = ""
    threshold: float | None = None
    run_prediction: bool = True
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
    export_classes: Tuple[int, ...] = (2, 8)
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


@dataclass
class Settings:
    model_path: str
    main_directory: str | list[str]  # Can be single path or list of paths
    cache_dir: str = ""
    allow_cpu: bool = False
    cuda_allow_tf32: bool = True
    non_reactive_span_px: float = 5.0
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
    max_proboscis_tracks: int = 4
    pair_rebind_ratio: float = 0.20
    zero_iou_epsilon: float = 1e-8

    # distance limits
    class2_min: float = 70.0
    class2_max: float = 250.0

    # reaction prediction
    reaction_prediction: ReactionPredictionSettings = field(default_factory=ReactionPredictionSettings)
    force: ForceSettings = field(default_factory=ForceSettings)

    # tracking quality
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    # YOLO dataset curation
    yolo_curation: YoloCurationSettings = field(default_factory=YoloCurationSettings)

    # Pseudolabel mining + export
    pseudolabel: PseudolabelSettings = field(default_factory=PseudolabelSettings)

def _get(d: Dict[str, Any], key: str, default: Any):
    return d.get(key, default)

def load_settings(config_path: str | Path) -> Settings:
    load_dotenv(dotenv_path=Path(".env"))  # optional
    p = Path(config_path)
    data: Dict[str, Any] = {}
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    # env overrides
    model_path = os.getenv("MODEL_PATH", _get(data, "model_path", ""))
    main_directory = os.getenv("MAIN_DIRECTORY", _get(data, "main_directory", ""))

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
        matrix=matrix,
    )

    force_cfg_raw = data.get("force") if isinstance(data.get("force"), dict) else {}
    force = ForceSettings(
        pipeline=_as_bool(force_cfg_raw.get("pipeline"), True),
        yolo=_as_bool(force_cfg_raw.get("yolo"), True),
        combined=_as_bool(force_cfg_raw.get("combined"), True),
        reaction_prediction=_as_bool(force_cfg_raw.get("reaction_prediction"), True),
        reaction_matrix=_as_bool(force_cfg_raw.get("reaction_matrix"), True),
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

    non_reactive_span_px = float(
        os.getenv("NON_REACTIVE_SPAN_PX", _get(data, "non_reactive_span_px", 5.0))
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

    return Settings(
        model_path=model_path,
        main_directory=main_directory,
        cache_dir=str(Path(cache_dir).expanduser()),
        allow_cpu=allow_cpu,
        cuda_allow_tf32=cuda_allow_tf32,
        non_reactive_span_px=non_reactive_span_px,
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
        pair_rebind_ratio=float(os.getenv("PAIR_REBIND_RATIO", yolo.get("pair_rebind_ratio", 0.20))),
        zero_iou_epsilon=float(os.getenv("ZERO_IOU_EPSILON", yolo.get("zero_iou_epsilon", 1e-8))),

        class2_min=float(os.getenv("CLASS2_MIN", dist_limits.get("class2_min", 70.0))),
        class2_max=float(os.getenv("CLASS2_MAX", dist_limits.get("class2_max", 250.0))),
        reaction_prediction=reaction_prediction,
        force=force,
        tracking=tracking,
        yolo_curation=yolo_curation,
        pseudolabel=pseudolabel,
    )
