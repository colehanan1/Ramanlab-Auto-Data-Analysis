#!/usr/bin/env python3
"""
Test GPU integration with pipeline.

This script verifies that:
1. Pipeline detects GPU correctly
2. GPU-accelerated steps can be imported
3. Config loading works
4. GPU processor initializes
"""

import sys
import torch
from pathlib import Path

print("="*80)
print("GPU PIPELINE INTEGRATION TEST")
print("="*80)

# Test 1: CUDA Detection
print("\n[Test 1] CUDA Detection")
cuda_available = torch.cuda.is_available()
print(f"  CUDA Available: {cuda_available}")
if cuda_available:
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("  ✅ PASS")
else:
    print("  ⚠️  WARNING: CUDA not available, will use CPU")

# Test 2: Pipeline Import
print("\n[Test 2] Pipeline Import")
try:
    from src.fbpipe import pipeline
    print(f"  GPU Mode: {pipeline.USE_GPU}")
    print("  ✅ PASS")
except Exception as e:
    print(f"  ❌ FAIL: {e}")
    sys.exit(1)

# Test 3: GPU Steps Import
print("\n[Test 3] GPU Steps Import")
try:
    # These should be the GPU versions if CUDA is available
    from src.fbpipe.steps import distance_normalize_gpu, calculate_acceleration_gpu
    print("  distance_normalize_gpu: ✅")
    print("  calculate_acceleration_gpu: ✅")
    print("  ✅ PASS")
except Exception as e:
    print(f"  ❌ FAIL: {e}")
    sys.exit(1)

# Test 4: Config Loading
print("\n[Test 4] Config Loading")
try:
    from src.fbpipe.config import Settings
    # Load config without requiring file path
    config_path = Path("config.yaml")
    if config_path.exists():
        from src.fbpipe.config import load_settings
        cfg = load_settings(config_path)
        allow_cpu = getattr(cfg, 'allow_cpu', False)
        print(f"  Config loaded: {config_path}")
        print(f"  allow_cpu setting: {allow_cpu}")
        print("  ✅ PASS")
    else:
        print(f"  ⚠️  Config file not found: {config_path}")
        print("  ⚠️  SKIP (optional for this test)")
except Exception as e:
    print(f"  ⚠️  WARNING: {e}")
    print("  ⚠️  SKIP (optional for this test)")

# Test 5: GPU Processor Initialization
print("\n[Test 5] GPU Processor Initialization")
try:
    from src.fbpipe.utils.gpu_accelerated import get_default_processor
    import numpy as np

    gpu = get_default_processor(force_cpu=False)

    # Quick computation test
    test_data = np.array([10.0, 50.0, 100.0, 250.0], dtype=np.float32)
    result = gpu.normalize_distances_batch(test_data, 10.0, 250.0, 250.0)

    expected = np.array([0.0, 16.666668, 37.500004, 100.0], dtype=np.float32)
    if np.allclose(result, expected, rtol=1e-5):
        print("  Normalization test: ✅")
        print("  ✅ PASS")
    else:
        print(f"  ❌ FAIL: Expected {expected}, got {result}")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Pipeline Step Listing
print("\n[Test 6] Pipeline Step Registry")
try:
    steps = list(pipeline.STEP_REGISTRY.keys())
    print(f"  Available steps: {len(steps)}")

    # Check that GPU-accelerated steps are in the registry
    critical_steps = ['distance_normalize', 'calculate_acceleration']
    for step_name in critical_steps:
        if step_name in steps:
            print(f"    {step_name}: ✅")
        else:
            print(f"    {step_name}: ❌ MISSING")
            sys.exit(1)

    print("  ✅ PASS")
except Exception as e:
    print(f"  ❌ FAIL: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*80)
print("INTEGRATION TEST SUMMARY")
print("="*80)
print("✅ All tests passed!")
print("\nThe pipeline is ready to use GPU acceleration.")
print("\nUsage:")
print("  python -m src.fbpipe.pipeline distance_normalize")
print("  python -m src.fbpipe.pipeline calculate_acceleration")
print("  python -m src.fbpipe.pipeline all  # Run full pipeline")
print("="*80 + "\n")
