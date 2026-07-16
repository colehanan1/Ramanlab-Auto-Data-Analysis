"""A frozen dataset that is not on disk must fail loudly, not vanish."""

import textwrap

import pytest

from fbpipe.config import load_raw_config


def _cfg(tmp_path, body):
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "secured").mkdir(exist_ok=True)
    p = tmp_path / "cfg.yaml"
    p.write_text(
        textwrap.dedent(
            f"""
            protocol: v2
            dataset_bases:
              data: {tmp_path}/data
              secured: {tmp_path}/secured
            {body}
            """
        )
    )
    return p


def test_frozen_dataset_missing_from_disk_raises(tmp_path):
    cfg = _cfg(
        tmp_path,
        """
            datasets:
              - GONE-24-1
            dataset_overrides:
              GONE-24-1:
                freeze:
                  data: true
        """,
    )
    with pytest.raises(RuntimeError, match="GONE-24-1"):
        load_raw_config(str(cfg))


def test_unfrozen_dataset_missing_from_disk_still_skips_silently(tmp_path):
    """Pre-existing behavior must not change for unfrozen datasets."""
    cfg = _cfg(
        tmp_path,
        """
            datasets:
              - GONE-24-1
        """,
    )
    data = load_raw_config(str(cfg))
    assert data["main_directories"] == []


def test_frozen_dataset_but_present_on_disk_does_not_raise(tmp_path):
    """Presence, not the flag, is what avoids the error: confirm the folder is
    genuinely absent for the missing-case tests above, and that a frozen
    dataset which *is* on disk never raises."""
    (tmp_path / "data" / "HERE-24-2").mkdir(parents=True)
    assert not (tmp_path / "data" / "GONE-24-1").exists()
    assert not (tmp_path / "secured" / "GONE-24-1").exists()
    cfg = _cfg(
        tmp_path,
        """
            datasets:
              - HERE-24-2
            dataset_overrides:
              HERE-24-2:
                freeze:
                  data: true
        """,
    )
    data = load_raw_config(str(cfg))
    assert any("HERE-24-2" in d for d in data["main_directories"])


def test_frozen_dataset_present_on_disk_is_fine(tmp_path):
    (tmp_path / "data" / "HERE-24-1").mkdir(parents=True)
    cfg = _cfg(
        tmp_path,
        """
            datasets:
              - HERE-24-1
            dataset_overrides:
              HERE-24-1:
                freeze:
                  data: true
        """,
    )
    data = load_raw_config(str(cfg))
    assert any("HERE-24-1" in d for d in data["main_directories"])
