#!/usr/bin/env python3
"""
Export YOLO model to TensorRT engine for faster inference.

Usage:
    python scripts/export_tensorrt.py

This will export best.pt → best.engine in the same directory.
"""
import logging
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Path from config.yaml
MODEL_PATH = "/home/ramanlab/Documents/cole/sam2/notebooks/YOLOProjectProboscisLegs/runs/obb/train5/weights/best.pt"

def main():
    model_path = Path(MODEL_PATH)
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
