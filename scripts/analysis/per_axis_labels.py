"""The two PER y-axis labels, in one place.

Every figure that plots a PER response rate or a PER score used to spell its own
y axis: ``"PER %"``, ``"PER%"``, ``"% of Flies Responding"``, ``"Responding
flies (%)"``, ``"Average PER%"``, ``"Mean Score"``. Panels from different
scripts then sat side by side in a figure claiming to measure the same thing
under six different names. Import from here instead.
"""
from __future__ import annotations

#: Fraction of trials (or flies) that produced a PER, as a percentage.
PERCENT_Y_LABEL = "Mean PER response %"

#: The ordinal PER score, -1..5.
SCORE_Y_LABEL = "Mean PER Score"

__all__ = ["PERCENT_Y_LABEL", "SCORE_Y_LABEL"]
