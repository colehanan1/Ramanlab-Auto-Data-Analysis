"""Configuration of the abdomen geometry pipeline.

Legacy (``config/config_abdomen*.yaml``): the shared keys (``model_path``,
``dataset_bases``/``datasets`` -> ``main_directories``, ``yolo.*``, ``cache_dir``, odor
timing, freeze rules) are parsed by :func:`fbpipe.config.load_settings` exactly as for
production. The ``geometry:`` block is ours.

Integrated (an ``abdomen:`` block in an ``extends:`` overlay of ``config_new.yaml``): the
production roots, odor timing and freeze rules still come from the base config, but everything
abdomen-specific -- model, detector settings, output tree, video sources, ``geometry:`` -- comes
from the ``abdomen:`` block (:func:`parse_abdomen_block`).

``load_geometry_config`` refuses configurations that would otherwise "succeed" by processing
nothing.
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import yaml

from ..config import Settings, get_main_directories, load_settings, resolve_config_path
from ..config_layers import ConfigLayers, resolve_layers
from ..utils.fly_geometry import DEFAULT_LIMITS, FlyGeometryLimits
from .columns import GEOMETRY_SUBDIR_DEFAULT, validate_table_format

MEASURE_IN = ("native", "model")
SLOT_SEED_MODES = ("warmup", "production")

# The abdomen model's own detector settings: nothing is inherited from config_new's PER ``yolo:``.
ABDOMEN_YOLO_DEFAULTS = {"conf_thres": 0.25, "inference_batch_size": 8, "engine_supports_batch": True,
                         "max_flies": 5, "num_workers": 3, "zero_iou_epsilon": 1e-8}
# ``enabled``, ``python`` and ``max_hours`` belong to the driver (run_workflows reads them from the
# overlay); geometry accepts them and ignores them.
ABDOMEN_BLOCK_KEYS = frozenset({"enabled", "python", "model_path", "out_base", "video_bases", "batch_date_from",
                                "datasets", "slot_seeds", "seed_radius_frac", "max_hours", "yolo", "geometry"})
# Never writable by the abdomen pipeline, whatever the config says.
SECURED_MOUNT = "/securedstorage"


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
    # Percentile trimmed off the RESTING end of each scale so one mis-tracked frame
    # cannot define a fly's range. The SIGNAL end always stays at the literal extreme:
    # above the 99th percentile of PER, frames are ~2.6x enriched for odor-on, i.e. real.
    per_trim_pct: float = 5.0      # off the low (resting) end of the proboscis distance
    angle_trim_pct: float = 5.0    # off the high (straight, resting) end of the abdomen angle
    rest_min_samples: int = 200
    limits: FlyGeometryLimits = field(default_factory=lambda: DEFAULT_LIMITS)
    annotated_video: bool = True
    prefer_engine: bool = True
    wide_out_dir: str = ""
    figures_out_dir: str = ""
    trace_max_s: float = 90.0
    # Integrated mode (the ``abdomen:`` block of an extends-overlay). ``out_base == ""`` is legacy:
    # the outputs sit beside the production batch, in ``<batch>/<out_subdir>``.
    out_base: str = ""                       # own output tree: <out_base>/<dataset>/<batch>/<out_subdir>
    video_bases: Tuple[str, ...] = ()        # per trial the first existing video wins, then the local batch dir
    batch_date_from: Optional[date] = None   # only batches recorded on/after this day; None = no cut-off
    datasets: Tuple[str, ...] = ()           # subset of the base config's datasets; () = all of them
    slot_seeds: str = "warmup"               # "production": abdomen slot N == production fly N
    seed_radius_frac: float = 0.5            # a production seed claims a warm-up eye within this x eye scale


def _num(raw: Mapping[str, Any], key: str, default: float, *, minimum: Optional[float] = None,
         maximum: Optional[float] = None, strict: bool = False, integer: bool = False,
         scope: str = "geometry") -> float:
    val = raw.get(key, default)
    try:
        val = int(val) if integer else float(val)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{scope}.{key} must be a number, got {val!r}") from exc
    # Written positively, so a NaN fails the bound instead of slipping past it.
    if minimum is not None and not (val > minimum if strict else val >= minimum):
        raise ValueError(f"{scope}.{key} must be {'>' if strict else '>='} {minimum}, got {val}")
    if maximum is not None and not val <= maximum:
        raise ValueError(f"{scope}.{key} must be <= {maximum}, got {val}")
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
        per_trim_pct=_num(raw, "per_trim_pct", 5.0, minimum=0.0, maximum=49.0),
        angle_trim_pct=_num(raw, "angle_trim_pct", 5.0, minimum=0.0, maximum=49.0),
        rest_min_samples=int(_num(raw, "rest_min_samples", 200, minimum=1, integer=True)),
        limits=limits,
        annotated_video=bool(raw.get("annotated_video", True)),
        prefer_engine=bool(raw.get("prefer_engine", True)),
        wide_out_dir=str(raw.get("wide_out_dir", "") or ""),
        figures_out_dir=str(raw.get("figures_out_dir", "") or ""),
        trace_max_s=_num(raw, "trace_max_s", 90.0, minimum=0.0, strict=True),
    )


def _path(block: Mapping[str, Any], key: str) -> str:
    """A path from the block, ``~`` expanded; "" when it is absent or blank (the loader insists on the
    required ones -- and an empty model_path is how the PER model is kept out of the abdomen settings)."""
    val = block.get(key)
    if val is None:
        return ""
    if not isinstance(val, str):
        raise ValueError(f"abdomen.{key} must be a path, got {val!r}")
    return str(Path(val.strip()).expanduser()) if val.strip() else ""


def _string_list(block: Mapping[str, Any], key: str) -> Tuple[str, ...]:
    """A YAML list of non-empty strings. A bare string is refused rather than iterated: it would
    split into one entry per character and quietly match nothing (and an empty path is the cwd)."""
    raw = block.get(key)
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)) or not all(isinstance(v, str) and v.strip() for v in raw):
        raise ValueError(f"abdomen.{key} must be a list of non-empty strings, got {raw!r}")
    return tuple(v.strip() for v in raw)


def _batch_date(raw: Any) -> Optional[date]:
    """``abdomen.batch_date_from``: an (unquoted) YAML date or an ISO ``YYYY-MM-DD`` string; None = no cut-off."""
    if raw is None:
        return None
    if isinstance(raw, datetime):          # a datetime is a date subclass, but the two do not compare
        return raw.date()
    if isinstance(raw, date):
        return raw
    if isinstance(raw, str) and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", raw.strip()):
        try:
            return date.fromisoformat(raw.strip())
        except ValueError:
            pass                           # right shape, not a calendar day (2026-13-40)
    raise ValueError(f"abdomen.batch_date_from must be a date (YYYY-MM-DD), got {raw!r}")


def _abdomen_yolo(raw: Any) -> dict:
    """The six abdomen detector values; keys the block leaves out take ABDOMEN_YOLO_DEFAULTS."""
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("abdomen.yolo must be a mapping")
    unknown = sorted(str(k) for k in raw if k not in ABDOMEN_YOLO_DEFAULTS)
    if unknown:
        raise ValueError(f"abdomen.yolo has unknown key(s): {unknown}; expected a subset of "
                         f"{sorted(ABDOMEN_YOLO_DEFAULTS)}")
    dflt = ABDOMEN_YOLO_DEFAULTS
    supports = raw.get("engine_supports_batch", dflt["engine_supports_batch"])
    if not isinstance(supports, bool):
        raise ValueError(f"abdomen.yolo.engine_supports_batch must be true or false, got {supports!r}")
    scope = "abdomen.yolo"
    return {
        "conf_thres": _num(raw, "conf_thres", dflt["conf_thres"], minimum=0.0, maximum=1.0, scope=scope),
        "inference_batch_size": _num(raw, "inference_batch_size", dflt["inference_batch_size"], minimum=1,
                                     integer=True, scope=scope),
        "engine_supports_batch": supports,
        "max_flies": _num(raw, "max_flies", dflt["max_flies"], minimum=1, integer=True, scope=scope),
        "num_workers": _num(raw, "num_workers", dflt["num_workers"], minimum=1, integer=True, scope=scope),
        "zero_iou_epsilon": _num(raw, "zero_iou_epsilon", dflt["zero_iou_epsilon"], minimum=0.0, scope=scope),
    }


def parse_abdomen_block(block: Mapping[str, Any], base: Settings) -> Tuple[Settings, GeometrySettings]:
    """The ``abdomen:`` block of an overlay, laid over the base config's ``Settings``.

    Self-contained on purpose: the model and every detector value come from the block (``yolo:`` keys
    it leaves out take ABDOMEN_YOLO_DEFAULTS) and are applied with ``dataclasses.replace`` on the
    already env-resolved ``base``. Neither config_new's PER ``yolo:``/``model_path`` nor the
    MODEL_PATH / INFERENCE_BATCH_SIZE / MAX_FLIES / CONF_THRES (...) env overrides can leak in.
    Everything else -- production roots, odor timing, freeze rules -- stays the base's. Pure: no
    filesystem access, and a key the block leaves out stays empty (never the base's value), so what
    must be present and what must exist is load_geometry_config's business. The one default that
    differs from a legacy config: ``geometry.annotated_video`` is False unless the block sets it.
    """
    if not isinstance(block, Mapping):
        raise ValueError("abdomen block must be a mapping")
    unknown = sorted(str(k) for k in block if k not in ABDOMEN_BLOCK_KEYS)
    if unknown:
        raise ValueError(f"abdomen has unknown key(s): {unknown}; expected a subset of {sorted(ABDOMEN_BLOCK_KEYS)}")
    model_path = _path(block, "model_path")
    out_base = _path(block, "out_base")
    video_bases = tuple(str(Path(v).expanduser()) for v in _string_list(block, "video_bases"))
    datasets = _string_list(block, "datasets")
    missing = [d for d in datasets if d not in base.datasets]
    if missing:
        raise ValueError(f"abdomen.datasets names dataset(s) {missing} that the base config does not list "
                         f"(its datasets: {list(base.datasets)})")
    slot_seeds = str(block.get("slot_seeds", "warmup")).strip().lower()
    if slot_seeds not in SLOT_SEED_MODES:
        raise ValueError(f"abdomen.slot_seeds must be one of {SLOT_SEED_MODES}, got {block.get('slot_seeds')!r}")
    geometry_raw = block.get("geometry")
    if geometry_raw is not None and not isinstance(geometry_raw, Mapping):
        raise ValueError("abdomen.geometry must be a mapping")
    yolo = _abdomen_yolo(block.get("yolo"))
    settings = dataclasses.replace(
        base, model_path=model_path, conf_thres=yolo["conf_thres"], inference_batch_size=yolo["inference_batch_size"],
        engine_supports_batch=yolo["engine_supports_batch"], max_flies=yolo["max_flies"],
        yolo_num_workers=yolo["num_workers"], zero_iou_epsilon=yolo["zero_iou_epsilon"])
    # Annotated videos are opt-in here: off unless the block sets annotated_video or the run passes
    # --annotated-video. (A legacy config keeps GeometrySettings' default of on.)
    geometry = parse_geometry_block({"annotated_video": False, **(geometry_raw or {})})
    gcfg = dataclasses.replace(
        geometry, out_base=out_base, video_bases=video_bases,
        batch_date_from=_batch_date(block.get("batch_date_from")), datasets=datasets, slot_seeds=slot_seeds,
        seed_radius_frac=_num(block, "seed_radius_frac", 0.5, minimum=0.0, strict=True, scope="abdomen"))
    return settings, gcfg


def assert_out_tree_safe(out_base: Path, forbidden: Sequence[Path]) -> None:
    """Refuse an output tree that equals, or lies inside, a read-only root (the production batches,
    the secured mirror). Compared after resolving symlinks and ``..``, component by component, so
    ``flys_New2`` is not "inside" ``flys_New``."""
    out = Path(out_base).expanduser().resolve()
    for root in forbidden:
        resolved = Path(root).expanduser().resolve()
        if out.is_relative_to(resolved):
            raise ValueError(
                f"[geometry] abdomen.out_base {out} is, or lies inside, {resolved}, which the abdomen pipeline only "
                "reads. Give the abdomen outputs their own tree (e.g. .../Data/flys_Abdomen).")


def _load_integrated(layers: ConfigLayers) -> Tuple[Settings, GeometrySettings]:
    """Integrated mode: the production roots, timing and freeze rules from the base config, everything
    abdomen-specific from ``layers.overlay['abdomen']``. Same hard checks as the legacy loader, plus the
    output tree must not be production."""
    bases = (yaml.safe_load(layers.base_path.read_text(encoding="utf-8")) or {}).get("dataset_bases") or {}
    data_base = bases.get("data")
    if data_base and not Path(str(data_base)).expanduser().is_dir():
        raise FileNotFoundError(
            f"[geometry] dataset_bases.data does not exist: {data_base} "
            f"(the production tree the abdomen jobs are enumerated from)")
    settings, gcfg = parse_abdomen_block(layers.overlay["abdomen"], load_settings(layers.base_path))
    if not settings.model_path:
        raise ValueError("[geometry] abdomen.model_path is required (the base config's PER model_path is never used)")
    if not gcfg.out_base:
        raise ValueError("[geometry] abdomen.out_base is required: without it the outputs would land beside "
                         "the production batches")
    assert_out_tree_safe(Path(gcfg.out_base),
                         [Path(str(p)) for p in (data_base, bases.get("secured"), SECURED_MOUNT) if p])
    if not get_main_directories(settings):
        raise RuntimeError(
            "[geometry] no main_directories resolved: none of the base config's `datasets` exists under "
            f"dataset_bases.data={data_base!r}. Nothing would be processed.")
    if not Path(settings.model_path).is_file():
        raise FileNotFoundError(f"[geometry] model_path is not a file: {settings.model_path!r}")
    if not any(Path(b).is_dir() for b in gcfg.video_bases):
        raise FileNotFoundError(
            f"[geometry] none of abdomen.video_bases is an existing directory: {list(gcfg.video_bases)}")
    if not gcfg.wide_out_dir or not gcfg.figures_out_dir:
        raise ValueError("[geometry] abdomen.geometry.wide_out_dir and abdomen.geometry.figures_out_dir are required")
    return settings, gcfg


def load_geometry_config(config_path: str | Path) -> Tuple[Settings, GeometrySettings]:
    """Shared ``Settings`` plus the ``geometry:`` block, with hard validation:
    the data base must exist, at least one dataset root must resolve, the model
    file must exist, and both output dirs must be configured.

    A config whose ``abdomen:`` block is present (an ``extends:`` overlay of config_new.yaml) is
    loaded in integrated mode instead -- see :func:`parse_abdomen_block`."""
    path = resolve_config_path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"[geometry] config not found: {path}")
    layers = resolve_layers(path)
    if "abdomen" in layers.overlay:
        return _load_integrated(layers)
    if layers.is_layered:
        raise ValueError(f"[geometry] {path} extends {layers.base_path} but adds no abdomen: block")
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
