from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np

from .vision import order_corners


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def collect_detections(result, class_ids: Iterable[int]) -> Dict[int, Dict[str, object]]:
    """Collect YOLO detections per class from an Ultralytics result.

    Supports both axis-aligned boxes (``result.boxes``) and oriented boxes
    (``result.obb``). For OBB results, this returns an axis-aligned ``xyxy`` box
    derived from the four corner points and also preserves the ordered corners.
    """

    class_ids = tuple(int(c) for c in class_ids)
    out: Dict[int, Dict[str, List[object]]] = {
        cls: {"boxes": [], "scores": [], "corners": []} for cls in class_ids
    }

    if hasattr(result, "obb") and result.obb is not None:
        xyxyxyxy = _to_numpy(result.obb.xyxyxyxy)
        cls_arr = _to_numpy(result.obb.cls).astype(int)
        conf_arr = _to_numpy(result.obb.conf) if hasattr(result.obb, "conf") else None
        if conf_arr is None:
            conf_arr = np.ones_like(cls_arr, dtype=np.float32)
        else:
            conf_arr = conf_arr.astype(np.float32)

        for idx, (cls_id, conf) in enumerate(zip(cls_arr, conf_arr)):
            if int(cls_id) not in out:
                continue
            corners = np.asarray(xyxyxyxy[idx], dtype=np.float32).reshape(4, 2)
            x1, y1 = float(corners[:, 0].min()), float(corners[:, 1].min())
            x2, y2 = float(corners[:, 0].max()), float(corners[:, 1].max())
            out[int(cls_id)]["boxes"].append(np.array([x1, y1, x2, y2], dtype=np.float32))
            out[int(cls_id)]["scores"].append(float(conf))
            out[int(cls_id)]["corners"].append(order_corners(corners))
    else:
        if getattr(result, "boxes", None) is not None and len(result.boxes) > 0:
            xyxy = _to_numpy(result.boxes.xyxy).astype(np.float32)
            cls_arr = _to_numpy(result.boxes.cls).astype(int)
            conf_arr = _to_numpy(result.boxes.conf).astype(np.float32)
            for box, cls_id, conf in zip(xyxy, cls_arr, conf_arr):
                if int(cls_id) not in out:
                    continue
                out[int(cls_id)]["boxes"].append(np.asarray(box, dtype=np.float32))
                out[int(cls_id)]["scores"].append(float(conf))
                out[int(cls_id)]["corners"].append(None)

    parsed: Dict[int, Dict[str, object]] = {}
    for cls_id, data in out.items():
        boxes = (
            np.stack(data["boxes"], axis=0).astype(np.float32)
            if data["boxes"]
            else np.zeros((0, 4), np.float32)
        )
        scores = (
            np.asarray(data["scores"], dtype=np.float32)
            if data["scores"]
            else np.zeros((0,), np.float32)
        )
        parsed[cls_id] = {"boxes": boxes, "scores": scores, "corners": data["corners"]}
    return parsed


__all__ = ["collect_detections"]

