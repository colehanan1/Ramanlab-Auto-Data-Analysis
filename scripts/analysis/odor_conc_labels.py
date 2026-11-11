"""Odor names with their concentration written as ``<c>% (v/v)``.

Odor labels reach the figures in three spellings: raw ``~/odor_pins.txt``
tokens (``3-octonol-0.1``, ``EthylButyrate-0.1``), display names with the
percent in parentheses (``3-Octanol (0.1%)``, from a dataset's odor_remap),
and display name plus a bare number (``3-Octanol 0.1``). Every odor is a v/v
dilution, so the figures write all of them one way: ``3-Octanol 0.1% (v/v)``.

Display only. Grouping, odor matching and the ``(x%)`` parsing in
naive_vs_trained run on the original labels, so call these where the text is
drawn, never on a key. A label that carries no concentration gets none: the
pre-pins batches did not record one, and it is not guessed here.
"""
from __future__ import annotations

import math
import re

_PREFIX_RE = re.compile(r"^(?:pretest|testing|training)_\d+_", re.IGNORECASE)
# A raw pins token: no spaces, concentration after the last hyphen.
_RAW_RE = re.compile(r"^(?P<odor>[A-Za-z0-9][A-Za-z0-9-]*?)-(?P<conc>\d+(?:\.\d+)?)$")
_PAREN_RE = re.compile(r"^(?P<odor>.+?)\s*\(\s*(?P<conc>\d+(?:\.\d+)?)\s*%\s*\)$")
# Pins spellings the display table does not know, mapped to the rig spelling
# it does (as in starvation_pre_post_comparison._SPELLING).
_SPELLING = {"3octanol": "3-octonol", "3octonol": "3-octonol"}

VV = "% (v/v)"


def conc_text(conc: object) -> str:
    """``0.1`` / ``"0.10"`` / ``1.0`` -> ``"0.1% (v/v)"`` / ``"1% (v/v)"``."""
    return f"{float(conc):g}{VV}"


def _display(name: object) -> str:
    # Imported here: envelope_visuals is heavy, and every caller that draws
    # odor text has already loaded it.
    from scripts.analysis.envelope_visuals import _display_label_ci

    text = str(name).strip()
    return _display_label_ci(_SPELLING.get(text.lower(), text))


def _missing(conc: object) -> bool:
    if conc is None:
        return True
    if isinstance(conc, float) and math.isnan(conc):
        return True
    return not str(conc).strip()


def with_conc(name: object, conc: object = None) -> str:
    """``("3-octonol", "0.1")`` -> ``"3-Octanol 0.1% (v/v)"``; no conc -> the name alone."""
    display = _display(name)
    return display if _missing(conc) else f"{display} {conc_text(conc)}"


def odor_conc_label(label: object) -> str:
    """Display text for one odor label, concentration as ``<c>% (v/v)``.

    Takes a raw pins token (optionally with its ``testing_3_`` trial prefix,
    which is dropped) or a display name with a parenthesised percent. A label
    already in ``% (v/v)`` form, or one with no concentration, comes back as
    its display name. A bare trailing number is NOT read as a concentration:
    repeated presentations are suffixed " 1" / " 2" in some figures.
    """
    text = str(label).strip()
    if text.endswith(VV):
        return text
    body = _PREFIX_RE.sub("", text)
    m = _PAREN_RE.match(body) or _RAW_RE.match(body)
    if m:
        return with_conc(m["odor"], m["conc"])
    return _display(body)


__all__ = ["VV", "conc_text", "odor_conc_label", "with_conc"]
