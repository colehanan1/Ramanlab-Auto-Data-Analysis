
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
    )
