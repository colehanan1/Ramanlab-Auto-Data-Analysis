"""YOLO adapter for the 4-class OBB model.

The new model's class ids (0=abdomen, 1=abdomenTip, 2=eye, 3=proboscis) differ
from production's (0=eye, 1=proboscis), so everything here resolves classes by
NAME. ``ultralytics``/``torch`` are imported lazily: the pure helpers work in any
interpreter, only ``load_model``/``make_predict_fn`` need the yolo-env.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Union

import numpy as np

from ..utils.fly_geometry import ABDOMEN, ABDOMEN_TIP, EYE, PROBOSCIS, Part
from ..utils.yolo_results import _to_numpy

REQUIRED_CLASSES = (EYE, PROBOSCIS, ABDOMEN, ABDOMEN_TIP)


def _names_items(names: Union[Mapping[int, str], Sequence[str]]) -> List[tuple]:
    if isinstance(names, Mapping):
        return [(int(k), str(v)) for k, v in names.items()]
    return [(i, str(v)) for i, v in enumerate(names)]


def class_map_by_name(names: Union[Mapping[int, str], Sequence[str]]) -> Dict[str, int]:
    """``{canonical label: model class id}`` for the four required classes (case-insensitive)."""
    lower = {name.strip().lower(): cid for cid, name in _names_items(names)}
    out: Dict[str, int] = {}
    missing = []
    for label in REQUIRED_CLASSES:
        cid = lower.get(label.lower())
        if cid is None:
            missing.append(label)
        else:
            out[label] = cid
    if missing:
        raise ValueError(
            f"model lacks class(es) {missing}; available names: {[n for _, n in _names_items(names)]}. "
            "This pipeline needs the 4-class abdomen model (runs/obb/train), not the production model.")
    return out


def id_to_label_map(names: Union[Mapping[int, str], Sequence[str]]) -> Dict[int, str]:
    return {cid: label for label, cid in class_map_by_name(names).items()}


def model_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(model_path: Path):
    """Load the checkpoint, translating the ultralytics version mismatch into a clear error."""
    import ultralytics
    from ultralytics import YOLO
    try:
        return YOLO(str(model_path))
    except AttributeError as exc:   # e.g. "Can't get attribute 'OBB26'" on ultralytics < 8.4
        raise RuntimeError(
            f"cannot load {model_path} with ultralytics {ultralytics.__version__}: {exc}. "
            "YOLO26 checkpoints need ultralytics >= 8.4 -- run under `conda run -n yolo-env python`."
        ) from exc


class PredictProxy:
    """Wraps a model so every ``predict`` call carries ``imgsz`` and ``iou``.
    ``yolo_infer._make_batched_predict_fn`` passes neither."""

    def __init__(self, model: Any, imgsz: int, iou: float) -> None:
        self.model, self.imgsz, self.iou = model, int(imgsz), float(iou)

    def predict(self, images, **kw):
        kw.setdefault("imgsz", self.imgsz)
        kw.setdefault("iou", self.iou)
        return self.model.predict(images, **kw)


def make_predict_fn(model: Any, *, imgsz: int, iou: float, allow_cpu: bool) -> Callable[[list, float], list]:
    """``fn(images, conf) -> results`` with production's CUDA OOM sub-batch backoff."""
    import logging
    import torch
    from ..steps.yolo_infer import _make_batched_predict_fn

    # ultralytics >= 8.4 prints "'half' is deprecated" on EVERY predict call the
    # production backoff makes; that is thousands of lines per batch. Errors still show.
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    cuda = torch.cuda.is_available()
    if not cuda and not allow_cpu:
        raise RuntimeError("CUDA is not available. Set allow_cpu: true only for smoke tests.")
    device = {"cur": "cuda" if cuda and not allow_cpu else "cpu"}
    if hasattr(model, "to"):
        model.to(device["cur"])
    proxy = PredictProxy(model, imgsz, iou)
    return _make_batched_predict_fn(
        proxy,
        get_device=lambda: device["cur"],
        set_device=lambda target: device.__setitem__("cur", target),
        allow_cpu=allow_cpu,
    )


def parts_from_result(result: Any, id_to_label: Mapping[int, str]) -> List[Part]:
    """OBB detections of one frame as ``fly_geometry.Part`` (corners in the model's cyclic order)."""
    obb = getattr(result, "obb", None)
    if obb is None:
        return []
    try:
        if len(obb) == 0:
            return []
    except TypeError:
        return []
    corners = _to_numpy(obb.xyxyxyxy)
    if corners is None or corners.size == 0:
        return []
    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 4, 2)
    cls = _to_numpy(obb.cls).astype(int)
    conf = _to_numpy(obb.conf) if getattr(obb, "conf", None) is not None else np.ones(len(cls), np.float32)
    parts: List[Part] = []
    for pts, cid, score in zip(corners, cls, conf):
        label = id_to_label.get(int(cid))
        if label is None:
            continue
        parts.append(Part(label, pts, float(score)))
    return parts
