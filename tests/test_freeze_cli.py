"""--thaw / --thaw-all parsing and validation."""

import pytest

import scripts.pipeline.run_workflows as rw


def test_thaw_is_repeatable():
    p = rw._build_arg_parser()
    a = p.parse_args(["--thaw", "EB-Control-24-1", "--thaw", "Hex-Control-24-0.1"])
    assert a.thaw == ["EB-Control-24-1", "Hex-Control-24-0.1"]


def test_thaw_defaults_empty():
    a = rw._build_arg_parser().parse_args([])
    assert not a.thaw
    assert a.thaw_all is False


def test_thaw_all_is_a_flag():
    a = rw._build_arg_parser().parse_args(["--thaw-all"])
    assert a.thaw_all is True


def test_existing_flags_survive():
    """The parser extraction must not drop any existing flag."""
    a = rw._build_arg_parser().parse_args(
        ["--config", "c.yaml", "--folder", "F", "--figures-only", "--svg"]
    )
    assert a.config == "c.yaml"
    assert a.folder == "F"
    assert a.figures_only is True
    assert a.svg is True


def test_unknown_thaw_name_raises_listing_valid_names():
    """A silently ignored typo looks exactly like a successful thaw."""
    with pytest.raises(SystemExit) as exc_info:
        rw._validate_thaw(["Nope-24-1"], datasets=("EB-Control-24-1",))
    error_msg = str(exc_info.value)
    assert "Nope-24-1" in error_msg
    assert "EB-Control-24-1" in error_msg


def test_known_thaw_name_passes():
    rw._validate_thaw(["EB-Control-24-1"], datasets=("EB-Control-24-1",))
