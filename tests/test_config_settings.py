from __future__ import annotations

from pathlib import Path

from fbpipe.config import _expand_datasets, load_settings


def test_expand_datasets_rewrites_distance_base_roots(tmp_path: Path) -> None:
    """distance_base.wide.roots auto-expands to the secured roots, like
    combined_base.wide.roots, for datasets that exist on disk."""
    data_base = tmp_path / "data"
    secured_base = tmp_path / "secured"
    (data_base / "Hex-Training").mkdir(parents=True)
    (secured_base / "Hex-Training").mkdir(parents=True)

    data = {
        "dataset_bases": {"data": str(data_base), "secured": str(secured_base)},
        "datasets": ["Hex-Training"],
        "analysis": {
            "combined": {
                "combined_base": {"wide": {"roots": ["PLACEHOLDER"]}},
                "distance_base": {"wide": {"roots": ["PLACEHOLDER"]}},
            }
        },
    }

    expanded = _expand_datasets(data)
    combined = expanded["analysis"]["combined"]
    expected = [f"{secured_base}/Hex-Training/"]
    assert combined["distance_base"]["wide"]["roots"] == expected
    # combined_base behaviour must be unchanged.
    assert combined["combined_base"]["wide"]["roots"] == expected


def test_load_settings_reads_non_reactive_span(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
model_path: /tmp/model.pt
main_directory: /data/main
non_reactive_span_px: 12.5
""".strip()
    )

    settings = load_settings(cfg_path)

    assert settings.non_reactive_span_px == 12.5
