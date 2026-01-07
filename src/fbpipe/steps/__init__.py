"""Step modules used by the Fly Behavior pipeline."""

from importlib import import_module
from typing import Any

__all__ = [
    "yolo_infer",
    "curate_yolo_dataset",
    "pseudolabel_export",
    "distance_stats",
    "distance_normalize",
    "distance_normalize_gpu",  # GPU-accelerated version (6.2x faster)
    "distance_normalize_ultra",  # Batch GPU version (12-15x faster)
    "detect_dropped_frames",
    "rms_copy_filter",
    "update_ofm_state",
    "move_videos",
    "compose_videos_rms",
    "calculate_acceleration",
    "calculate_acceleration_gpu",  # GPU-accelerated version (5.3x faster)
    "predict_reactions",
    "reaction_matrix",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
