"""Layered configs: ``config_new_abdomen.yaml`` = ``extends: config_new.yaml`` + new blocks.

PER parity by construction: the driver resolves the layers once and hands the BASE
path to every PER loader, raw reader and child process -- they see exactly the file
they see today (binarized_per_rasters even writes the --config string into its JSON).
Only the abdomen/mounting stages get the overlay. An overlay may only add
OVERLAY_BLOCKS; any other key would change PER behaviour and is refused.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping

import yaml

EXTENDS_KEY = "extends"
OVERLAY_BLOCKS = ("abdomen", "mounting")
OVERLAY_KEYS = frozenset((EXTENDS_KEY, *OVERLAY_BLOCKS))


class LayeredConfigError(ValueError):
    """An overlay that would change its base config, or cannot be resolved."""


def _read(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise LayeredConfigError(f"{path}: the top level must be a mapping")
    return data


@dataclass(frozen=True)
class ConfigLayers:
    path: Path                     # the file the user passed
    base_path: Path                # what every PER reader gets
    overlay: Mapping[str, Any]     # only the abdomen/mounting blocks (may be empty)

    @property
    def is_layered(self) -> bool:
        return self.path != self.base_path


def resolve_layers(path: str | Path) -> ConfigLayers:
    from .config import resolve_config_path        # local import: config.py imports this module

    path = Path(path)
    data = _read(path)
    blocks = MappingProxyType({k: data[k] for k in OVERLAY_BLOCKS if k in data})
    if EXTENDS_KEY not in data:
        return ConfigLayers(path, path, blocks)    # identical path object: zero behaviour change
    extra = sorted(set(data) - OVERLAY_KEYS)
    if extra:
        raise LayeredConfigError(
            f"{path}: an extends-overlay may only add {list(OVERLAY_BLOCKS)}; it also sets {extra}, "
            "which would change the base config's behaviour")
    base_path = resolve_config_path(path.resolve().parent / str(data[EXTENDS_KEY]))
    if base_path == path.resolve():
        raise LayeredConfigError(f"{path}: extends itself")
    if not base_path.is_file():
        raise LayeredConfigError(f"{path}: base config not found: {base_path}")
    base = _read(base_path)
    if EXTENDS_KEY in base:
        raise LayeredConfigError(f"{base_path}: chained extends is not supported")
    clash = sorted(set(OVERLAY_BLOCKS) & set(base))
    if clash:
        raise LayeredConfigError(f"{base_path} already defines {clash}; refusing to shadow it")
    return ConfigLayers(path, base_path, blocks)


def load_layered(path: str | Path) -> Dict[str, Any]:
    layers = resolve_layers(path)
    merged = _read(layers.base_path)
    merged.update(layers.overlay)
    return merged
