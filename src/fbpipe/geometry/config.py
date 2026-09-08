"""Configuration of the abdomen geometry pipeline.

The shared keys (``model_path``, ``dataset_bases``/``datasets`` ->
``main_directories``, ``yolo.*``, ``cache_dir``, odor timing, freeze rules) are
parsed by :func:`fbpipe.config.load_settings` exactly as for production. The
``geometry:`` block is ours. ``load_geometry_config`` refuses configurations
that would otherwise "succeed" by processing nothing.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import yaml

from ..config import Settings, get_main_directories, load_settings, resolve_config_path
from ..utils.fly_geometry import DEFAULT_LIMITS, FlyGeometryLimits
from .columns import GEOMETRY_SUBDIR_DEFAULT, validate_table_format

MEASURE_IN = ("native", "model")


@dataclass(frozen=True)
class GeometrySettings:
    imgsz: int = 1088
    iou: float = 0.7
    table_format: str = "csv"
    out_subdir: str = GEOMETRY_SUBDIR_DEFAULT
    measure_in: str = "native"
    warmup_frames: int = 25
    warmup_presence_min: float = 0.5
    proboscis_max_jump_px: float = 80.0
    per_min_px: float = 10.0
    per_effective_max_floor_px: float = 95.0
    angle_pct_min_range_deg: float = 0.0
    rest_min_samples: int = 200
    limits: FlyGeometryLimits = field(default_factory=lambda: DEFAULT_LIMITS)
    annotated_video: bool = True
    wide_out_dir: str = ""
    figures_out_dir: str = ""
    trace_max_s: float = 90.0


def _num(raw: Mapping[str, Any], key: str, default: float, *, minimum: Optional[float] = None,
         strict: bool = False, integer: bool = False) -> float:
    val = raw.get(key, default)
    try:
        val = int(val) if integer else float(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"geometry.{key} must be a number, got {val!r}") from exc
    if minimum is not None and (val < minimum or (strict and val <= minimum)):
        raise ValueError(f"geometry.{key} must be {'>' if strict else '>='} {minimum}, got {val}")
    return val


def parse_geometry_block(raw: Optional[Mapping[str, Any]]) -> GeometrySettings:
    raw = dict(raw or {})
    limits_raw = raw.get("limits") or {}
    if not isinstance(limits_raw, Mapping):
        raise ValueError("geometry.limits must be a mapping of FlyGeometryLimits fields")
    known = {f.name for f in dataclasses.fields(FlyGeometryLimits)}
    unknown = sorted(set(limits_raw) - known)
    if unknown:
        raise ValueError(f"geometry.limits has unknown field(s): {unknown}")
    limits = dataclasses.replace(DEFAULT_LIMITS, **{k: float(v) for k, v in limits_raw.items()})
    measure_in = str(raw.get("measure_in", "native")).strip().lower()
    if measure_in not in MEASURE_IN:
        raise ValueError(f"geometry.measure_in must be one of {MEASURE_IN}, got {measure_in!r}")
    return GeometrySettings(
        imgsz=int(_num(raw, "imgsz", 1088, minimum=32, integer=True)),
        iou=_num(raw, "iou", 0.7, minimum=0.0),
        table_format=validate_table_format(raw.get("table_format", "csv")),
        out_subdir=str(raw.get("out_subdir", GEOMETRY_SUBDIR_DEFAULT)).strip() or GEOMETRY_SUBDIR_DEFAULT,
        measure_in=measure_in,
        warmup_frames=int(_num(raw, "warmup_frames", 25, minimum=1, integer=True)),
        warmup_presence_min=_num(raw, "warmup_presence_min", 0.5, minimum=0.0),
        proboscis_max_jump_px=_num(raw, "proboscis_max_jump_px", 80.0, minimum=0.0),
        per_min_px=_num(raw, "per_min_px", 10.0, minimum=0.0),
        per_effective_max_floor_px=_num(raw, "per_effective_max_floor_px", 95.0, minimum=0.0),
        angle_pct_min_range_deg=_num(raw, "angle_pct_min_range_deg", 0.0, minimum=0.0),
        rest_min_samples=int(_num(raw, "rest_min_samples", 200, minimum=1, integer=True)),
        limits=limits,
        annotated_video=bool(raw.get("annotated_video", True)),
        wide_out_dir=str(raw.get("wide_out_dir", "") or ""),
        figures_out_dir=str(raw.get("figures_out_dir", "") or ""),
        trace_max_s=_num(raw, "trace_max_s", 90.0, minimum=0.0, strict=True),
    )


def load_geometry_config(config_path: str | Path) -> Tuple[Settings, GeometrySettings]:
    """Shared ``Settings`` plus the ``geometry:`` block, with hard validation:
    the data base must exist, at least one dataset root must resolve, the model
    file must exist, and both output dirs must be configured."""
    path = resolve_config_path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"[geometry] config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    bases = raw.get("dataset_bases") or {}
    data_base = bases.get("data")
    if data_base and not Path(str(data_base)).expanduser().is_dir():
        raise FileNotFoundError(
            f"[geometry] dataset_bases.data does not exist: {data_base} "
            f"(create the demo copy first: scripts/pipeline/make_geometry_demo_copy.py)")
    settings = load_settings(path)
    roots = get_main_directories(settings)
    if not roots:
        raise RuntimeError(
            "[geometry] no main_directories resolved: none of `datasets` exists under "
            f"dataset_bases.data={data_base!r}. Nothing would be processed.")
    if not settings.model_path or not Path(settings.model_path).expanduser().is_file():
        raise FileNotFoundError(f"[geometry] model_path is not a file: {settings.model_path!r}")
    gcfg = parse_geometry_block(raw.get("geometry"))
    if not gcfg.wide_out_dir or not gcfg.figures_out_dir:
        raise ValueError("[geometry] geometry.wide_out_dir and geometry.figures_out_dir are required")
    return settings, gcfg
