"""Shared matplotlib style for all analysis scripts.

Call ``apply_lab_style()`` once at module import instead of copy-pasting
``plt.rcParams.update(...)`` blocks in every script.
"""

from __future__ import annotations


def apply_lab_style(*, dpi: int = 300) -> None:
    """Apply the lab's standard matplotlib style globally."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
        }
    )
