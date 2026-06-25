from __future__ import annotations

from pathlib import Path

from fbpipe.config import _expand_datasets, load_settings


def test_expand_datasets_combined_base_reads_local_roots(tmp_path: Path) -> None:
    """combined_base + distance_base wide builds must read LOCAL (data) roots —
    the same place ``combine`` writes its angle_distance_rms_envelope output — so
    each run's freshly processed data lands in the combined_base CSV/figures.
    Reading secured would only ever see prior runs' data (one-run lag)."""
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
    expected_local = [f"{data_base}/Hex-Training/"]
    assert combined["combined_base"]["wide"]["roots"] == expected_local
    assert combined["distance_base"]["wide"]["roots"] == expected_local


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
