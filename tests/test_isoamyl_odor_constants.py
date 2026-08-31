"""The rigs now write isoamyl acetate for the OFM_A valve.

Trial files carry ``IsoamylAcetate`` and cohort folders ``IAA-*``, so the
canonicalizer has to resolve both. Historical ``ACV-*`` data keeps resolving to
Apple Cider Vinegar untouched.
"""

from fbpipe.odor_constants import ODOR_CANON, DISPLAY_LABEL, ODOR_ORDER


def _canon(raw):
    return ODOR_CANON[raw.strip().lower()]


def test_isoamylacetate_label_canonicalizes():
    assert _canon("IsoamylAcetate") == "IAA"


def test_spaced_and_hyphenated_forms_canonicalize():
    assert _canon("isoamyl acetate") == "IAA"
    assert _canon("isoamyl-acetate") == "IAA"
    assert _canon("iaa") == "IAA"


def test_cohort_folder_names_canonicalize():
    assert _canon("IAA-Control") == "IAA-Control"
    assert _canon("IAA-Training") == "IAA-Training"


def test_display_label():
    assert DISPLAY_LABEL["IAA"] == "Isoamyl Acetate"
    assert DISPLAY_LABEL["IAA-Control"] == "Isoamyl Acetate"
    assert DISPLAY_LABEL["IAA-Training"] == "Isoamyl Acetate"


def test_historical_acv_untouched():
    """Past ACV-* datasets are real apple cider vinegar and must not be renamed."""
    assert _canon("acv") == "ACV"
    assert DISPLAY_LABEL["ACV"] == "Apple Cider Vinegar"


def test_iaa_has_a_figure_ordering_slot():
    assert "IAA" in ODOR_ORDER
