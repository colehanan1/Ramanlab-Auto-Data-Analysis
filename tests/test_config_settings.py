from __future__ import annotations

from pathlib import Path

from fbpipe.config import load_settings


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
