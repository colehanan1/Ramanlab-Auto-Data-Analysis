#!/usr/bin/env python3
"""
Export YOLO model to TensorRT engine for faster inference.

Usage:
    python scripts/convert/export_tensorrt.py

This will export best.pt → best.engine in the same directory.
"""
import argparse
import logging
import sys
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from fbpipe.config import load_settings

def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to TensorRT engine.")
    parser.add_argument(
        "--config",
        default=str(Path("config") / "config.yaml"),
        help="Path to pipeline configuration YAML.",
    )
    parser.add_argument("--model-path", default=None, help="Override the YOLO model path.")
    args = parser.parse_args()

    model_path_value = args.model_path or load_settings(args.config).model_path
    if not model_path_value:
        log.error("Model path not configured. Provide --model-path or set model_path in the config.")
        return

    model_path = Path(model_path_value).expanduser()
    if not model_path.exists():
        log.error(f"Model not found: {model_path}")
        return

    engine_path = model_path.with_suffix('.engine')
    if engine_path.exists():
        log.warning(f"TensorRT engine already exists: {engine_path}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            log.info("Export cancelled")
            return

    log.info(f"Loading PyTorch model: {model_path}")
    model = YOLO(str(model_path))

    log.info("Exporting to TensorRT engine (this may take 5-15 minutes)...")
    log.info("Using FP16 precision for RTX 3090 Tensor Cores")

    model.export(format="engine", half=True, device="cuda:0")

    log.info(f"✓ TensorRT engine created: {engine_path}")
    log.info("YOLO inference will now use this optimized engine automatically")

if __name__ == "__main__":
    main()
