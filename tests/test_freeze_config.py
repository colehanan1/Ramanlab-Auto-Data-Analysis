"""Per-dataset freeze flags parse off the dataset_overrides block."""

import textwrap

from fbpipe.config import load_settings


def _write_cfg(tmp_path, overrides_yaml: str):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        textwrap.dedent(
            f"""
            protocol: v2
            dataset_bases:
              data: {tmp_path}/data
              secured: {tmp_path}/secured
            dataset_overrides:
            {overrides_yaml}
            """
        )
    )
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "secured").mkdir(exist_ok=True)
    return cfg


def test_freeze_flags_parse_both_true(tmp_path):
    cfg = _write_cfg(
        tmp_path,
        """
              EB-Control-24-1:
                freeze:
                  data: true
                  figures: true
        """,
    )
    s = load_settings(str(cfg))
    ov = s.dataset_overrides["EB-Control-24-1"]
    assert ov.freeze_data is True
    assert ov.freeze_figures is True


def test_freeze_flags_are_independent(tmp_path):
    """data:true + figures:false must be representable -- the styling-iteration mode."""
    cfg = _write_cfg(
        tmp_path,
        """
              EB-Control-24-1:
                freeze:
                  data: true
                  figures: false
              EB-Training-24-1:
                freeze:
                  figures: true
        """,
    )
    s = load_settings(str(cfg))
    a = s.dataset_overrides["EB-Control-24-1"]
    b = s.dataset_overrides["EB-Training-24-1"]
    assert (a.freeze_data, a.freeze_figures) == (True, False)
    assert (b.freeze_data, b.freeze_figures) == (False, True)


def test_absent_freeze_block_defaults_false(tmp_path):
    """A dataset with other overrides but no freeze: block is not frozen."""
    cfg = _write_cfg(
        tmp_path,
        """
              EB-Control-24-1:
                trial_type_override: testing
        """,
    )
    s = load_settings(str(cfg))
    ov = s.dataset_overrides["EB-Control-24-1"]
    assert ov.freeze_data is False
    assert ov.freeze_figures is False
    # The sibling override must still parse -- freeze must not disturb it.
    assert ov.trial_type_override == "testing"


def test_empty_freeze_block_defaults_false(tmp_path):
    cfg = _write_cfg(
        tmp_path,
        """
              EB-Control-24-1:
                freeze: {}
        """,
    )
    s = load_settings(str(cfg))
    ov = s.dataset_overrides["EB-Control-24-1"]
    assert ov.freeze_data is False
    assert ov.freeze_figures is False
