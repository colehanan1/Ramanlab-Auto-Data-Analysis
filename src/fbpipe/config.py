
from __future__ import annotations
import os, yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

@dataclass
class Settings:
    model_path: str
    main_directory: str
    allow_cpu: bool = False
    cuda_allow_tf32: bool = True
    anchor_x: float = 1079.0
    anchor_y: float = 540.0
    fps_default: float = 40.0
    window_sec: float = 0.25
    odor_on_s: float = 30.0
    odor_off_s: float = 60.0
    delete_source_after_render: bool = True

    # detector/tracker
    conf_thres: float = 0.40
    iou_match_thres: float = 0.25
    max_age: int = 15
    ema_alpha: float = 0.20
    use_optical_flow: bool = True
    flow_skip_edge: int = 10

    # distance limits
    class2_min: float = 70.0
    class2_max: float = 250.0

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

    # nested
    yolo = data.get("yolo", {})
    dist_limits = data.get("distance_limits", {})

    return Settings(
        model_path=model_path,
        main_directory=main_directory,
        allow_cpu=allow_cpu,
        cuda_allow_tf32=cuda_allow_tf32,
        anchor_x=float(os.getenv("ANCHOR_X", _get(data, "anchor_x", 1079.0))),
        anchor_y=float(os.getenv("ANCHOR_Y", _get(data, "anchor_y", 540.0))),
        fps_default=float(os.getenv("FPS_DEFAULT", _get(data, "fps_default", 40.0))),
        window_sec=float(os.getenv("WINDOW_SEC", _get(data, "window_sec", 0.25))),
        odor_on_s=float(os.getenv("ODOR_ON_S", _get(data, "odor_on_s", 30.0))),
        odor_off_s=float(os.getenv("ODOR_OFF_S", _get(data, "odor_off_s", 60.0))),
        delete_source_after_render=(os.getenv("DELETE_SOURCE_AFTER_RENDER", str(_get(data, "delete_source_after_render", True))).lower()=="true"),

        conf_thres=float(os.getenv("CONF_THRES", yolo.get("conf_thres", 0.40))),
        iou_match_thres=float(os.getenv("IOU_MATCH_THRES", yolo.get("iou_match_thres", 0.25))),
        max_age=int(os.getenv("MAX_AGE", yolo.get("max_age", 15))),
        ema_alpha=float(os.getenv("EMA_ALPHA", yolo.get("ema_alpha", 0.20))),
        use_optical_flow=(os.getenv("USE_OPTICAL_FLOW", str(yolo.get("use_optical_flow", True))).lower()=="true"),
        flow_skip_edge=int(os.getenv("FLOW_SKIP_EDGE", yolo.get("flow_skip_edge", 10))),

        class2_min=float(os.getenv("CLASS2_MIN", dist_limits.get("class2_min", 70.0))),
        class2_max=float(os.getenv("CLASS2_MAX", dist_limits.get("class2_max", 250.0))),
    )
