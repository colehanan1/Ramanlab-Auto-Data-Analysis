"""Save-time hook that emits an editable SVG next to every raster figure.

Many plotting call sites across the analysis scripts hard-code a ``.png``
output path. Rather than touch every ``savefig`` site (and risk missing one or
breaking the PNG-dependent plumbing — SMB sync globs, ``Results/Figures``
symlinks, preview tooling), this module monkeypatches
``matplotlib.figure.Figure.savefig`` so that whenever a figure is written to a
raster path it also writes a vector ``.svg`` sibling.

Patching ``Figure.savefig`` covers both ``fig.savefig(...)`` and
``plt.savefig(...)`` (the latter delegates to the current figure's method).

Usage
-----
Explicit::

    from fbpipe.figure_export import install_svg_sidecar
    install_svg_sidecar()

Subprocess-friendly (reads the ``FBPIPE_FIGURE_SVG`` env var)::

    from fbpipe.figure_export import maybe_install_from_env
    maybe_install_from_env()
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.figure as _mfig

_ENV_VAR = "FBPIPE_FIGURE_SVG"
_RASTER_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

_ORIG_SAVEFIG = _mfig.Figure.savefig
_INSTALLED = False


def _savefig_with_svg_sidecar(self, fname, *args, **kwargs):  # noqa: ANN001
    """Wrapped ``Figure.savefig`` that also writes a ``.svg`` next to rasters."""
    result = _ORIG_SAVEFIG(self, fname, *args, **kwargs)
    try:
        if isinstance(fname, (str, os.PathLike)):
            path = Path(os.fspath(fname))
            if path.suffix.lower() in _RASTER_EXTS:
                svg_path = path.with_suffix(".svg")
                # dpi is irrelevant for vector output; format is forced to svg.
                svg_kwargs = {
                    k: v for k, v in kwargs.items() if k not in ("format", "dpi")
                }
                _ORIG_SAVEFIG(self, os.fspath(svg_path), *args, format="svg", **svg_kwargs)
    except Exception:  # noqa: BLE001 — never let the sidecar break a real save
        pass
    return result


def install_svg_sidecar() -> None:
    """Patch ``Figure.savefig`` so every raster save also emits an SVG. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return
    _mfig.Figure.savefig = _savefig_with_svg_sidecar  # type: ignore[assignment]
    _INSTALLED = True


def uninstall_svg_sidecar() -> None:
    """Restore the original ``Figure.savefig`` (mainly for tests)."""
    global _INSTALLED
    if not _INSTALLED:
        return
    _mfig.Figure.savefig = _ORIG_SAVEFIG  # type: ignore[assignment]
    _INSTALLED = False


def maybe_install_from_env() -> bool:
    """Install the sidecar when ``FBPIPE_FIGURE_SVG`` is set to a truthy value.

    Returns True if the sidecar was (or already is) active.
    """
    value = str(os.environ.get(_ENV_VAR, "")).strip().lower()
    if value in {"1", "true", "yes", "on", "svg"}:
        install_svg_sidecar()
        return True
    return _INSTALLED
