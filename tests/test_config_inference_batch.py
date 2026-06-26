import os
from pathlib import Path
from fbpipe.config import Settings, load_settings

def test_default_inference_batch_size():
    assert Settings(model_path="m.pt", main_directories="/tmp").inference_batch_size == 32

def test_default_engine_supports_batch_is_false():
    assert Settings(model_path="m.pt", main_directories="/tmp").engine_supports_batch is False

def test_env_override(tmp_path, monkeypatch):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("yolo:\n  inference_batch_size: 4\n  engine_supports_batch: true\n")
    monkeypatch.setenv("INFERENCE_BATCH_SIZE", "16")
    s = load_settings(str(cfg))
    assert s.inference_batch_size == 16
    assert s.engine_supports_batch is True

def test_yaml_value_when_no_env(tmp_path, monkeypatch):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("yolo:\n  inference_batch_size: 8\n")
    monkeypatch.delenv("INFERENCE_BATCH_SIZE", raising=False)
    assert load_settings(str(cfg)).inference_batch_size == 8
