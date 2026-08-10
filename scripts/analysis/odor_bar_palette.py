"""One odor palette for every bar figure that draws training bars.

The trace figures shade their odor windows green for Hexanol and orange for
Apple Cider Vinegar (``make_pubfig_october_02_fly_1_testing_1_to_5.py``). The
bar figures used to paint all training bars one dark blue, so the same odor
looked like two different things depending on the panel. These helpers give the
bars the trace colours; odors with no palette entry keep the original dark blue
(trained) / light blue (untrained) split.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerTuple

HEX_COLOR = "#6cc070"       # Hexanol: the trace figure's green
ACV_COLOR = "#ffc685"       # Apple Cider Vinegar: the trace figure's orange
DARK_GREEN = "#2e7d32"      # 3-Octanol
DARKER_GREEN = "#14532d"    # Linalool
PINK = "#e78ac3"            # Ethyl Butyrate
CITRAL_YELLOW = "#f2d43c"   # Citral
ISOAMYL_PURPLE = "#8e6bbf"  # Isoamyl Acetate

TRAIN_COLOR = "#1a3a6b"      # dark blue: the trained odor, no palette entry
NON_TRAIN_COLOR = "#4a7fbf"  # light blue: every other odor
CTRL_COLOR = "#b0b0b0"       # gray: control cohort

ODOR_BAR_COLORS = {
    "hexanol": HEX_COLOR,
    "apple cider vinegar": ACV_COLOR,
    "3-octanol": DARK_GREEN,
    "3-octonol": DARK_GREEN,  # older spelling still in some exports
    "linalool": DARKER_GREEN,
    "ethyl butyrate": PINK,
    "citral": CITRAL_YELLOW,
    "isoamyl acetate": ISOAMYL_PURPLE,
}


_LABEL_NOISE = re.compile(r"\s*\([^)]*\)\s*|\s+\d+\s*$")


def normalise_odor(odor: str) -> str:
    """Strip the concentration and presentation index off a bar label.

    Figures label bars ``"3-Octanol (0.1%) 2"`` or ``"Isoamyl Acetate (1%)"``;
    the palette is keyed on the odor alone.
    """
    name = str(odor).strip()
    while True:
        stripped = _LABEL_NOISE.sub(" ", name).strip()
        if stripped == name:
            return name.casefold()
        name = stripped


def odor_color(odor: str) -> str | None:
    """Palette colour for an odor, or ``None`` if it has no entry."""
    return ODOR_BAR_COLORS.get(normalise_odor(odor))


def training_bar_colors(
    odors: Iterable[str], is_trained: Iterable[bool]
) -> list[str]:
    """Bar colours for a training cohort, in plotted order.

    The odor identity wins over trained/untrained: Hexanol is the same green
    whether or not it was the trained odor, which is the point of sharing the
    palette with the traces.
    """
    return [
        odor_color(odor) or (TRAIN_COLOR if bool(trained) else NON_TRAIN_COLOR)
        for odor, trained in zip(odors, is_trained)
    ]


def trained_tick_color(odor: str) -> str:
    """Colour for the trained odor's x tick label.

    The palette greens and oranges are too pale to read as text, so those fall
    back to black and rely on bold for emphasis.
    """
    return "black" if odor_color(odor) else TRAIN_COLOR


def training_legend_swatches(train_colors: Sequence[str]) -> tuple[mpatches.Patch, ...]:
    """One swatch per distinct training colour, in plotted order."""
    seen: list[str] = []
    for color in train_colors:
        key = str(color)
        if key not in seen:
            seen.append(key)
    return tuple(
        mpatches.Patch(facecolor=c, edgecolor="black", linewidth=0.75) for c in seen
    )


def add_training_legend(
    ax,
    train_colors: Sequence[str],
    *,
    ctrl_color: str | None = None,
    train_label: str = "Training",
    ctrl_label: str = "Control",
    loc: str = "upper right",
    bbox_to_anchor=None,
    fontsize: float = 9,
    framealpha: float = 0.8,
) -> None:
    """Legend whose "Training" entry shows each odor colour side by side.

    A single swatch can no longer stand for a training cohort once its bars are
    coloured per odor, so the entry holds one swatch per distinct colour.
    """
    swatches = training_legend_swatches(train_colors)
    # An empty cohort has nothing to key: a tuple handle with no artists makes
    # matplotlib's HandlerTuple raise while drawing the legend.
    handles: list = [swatches] if swatches else []
    labels = [train_label] if swatches else []
    if ctrl_color is not None:
        handles.append(
            mpatches.Patch(facecolor=ctrl_color, edgecolor="black", linewidth=0.75)
        )
        labels.append(ctrl_label)

    # A tuple handle is squeezed into one handle's width, so a six-odor figure
    # would render the Training key as a barcode. Widen it with the count.
    if not handles:
        return
    ax.legend(
        handles,
        labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.35)},
        handlelength=max(2.0, 0.9 * len(swatches)),
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        fontsize=fontsize,
        framealpha=framealpha,
    )
