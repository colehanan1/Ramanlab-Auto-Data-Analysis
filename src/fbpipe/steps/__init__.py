"""Step modules used by the Fly Behavior pipeline."""

from importlib import import_module
from typing import Any

__all__ = [
    "yolo_infer",
    "distance_stats",
    "distance_normalize",
    "detect_dropped_frames",
    "rms_copy_filter",
    "update_ofm_state",
    "move_videos",
    "compose_videos_rms",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
