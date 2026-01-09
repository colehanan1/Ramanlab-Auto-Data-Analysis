#!/usr/bin/env python3
"""
GPU Speed Benchmark: Compare CPU (pandas/numpy) vs GPU (PyTorch) for common operations.

This script tests the speedup potential for:
1. Distance normalization (division, clipping, boolean masking)
2. RMS calculation (rolling window operations)
3. Angle multiplier computation (vectorized math)
4. Acceleration calculation (diff operations)
"""

import time
import numpy as np
import pandas as pd
import torch

# ============================================================================
# Test Configuration
# ============================================================================

# Simulate typical dataset sizes
SMALL_SIZE = 10_000      # ~250 seconds at 40fps (single trial)
MEDIUM_SIZE = 100_000    # ~42 minutes at 40fps (typical experiment)
LARGE_SIZE = 1_000_000   # ~7 hours at 40fps (full "Matrix" dataset)

REPEAT_COUNT = 10  # Iterations for timing accuracy


def format_speedup(cpu_time, gpu_time):
    """Format speedup ratio with color coding."""
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    if speedup > 10:
        return f"\033[92m{speedup:.1f}x FASTER\033[0m"  # Green
    elif speedup > 5:
        return f"\033[93m{speedup:.1f}x faster\033[0m"  # Yellow
    else:
        return f"\033[91m{speedup:.1f}x\033[0m"  # Red


# ============================================================================
# Benchmark 1: Distance Normalization
# ============================================================================

def bench_distance_normalize_cpu(distances, gmin, gmax):
    """CPU version using pandas/numpy (current implementation)."""
    d = distances
    perc = np.empty_like(d, dtype=float)
    over = d > gmax
    under = d < gmin
    inr = ~(over | under)
    perc[over] = 101.0
    perc[under] = -1.0
    if gmax != gmin:
        perc[inr] = 100.0 * (d[inr] - gmin) / (gmax - gmin)
    else:
        perc[inr] = 0.0
    return perc


def bench_distance_normalize_gpu(distances_tensor, gmin, gmax):
    """GPU version using PyTorch tensors."""
    d = distances_tensor
    perc = torch.empty_like(d, dtype=torch.float32)
    over = d > gmax
    under = d < gmin
    inr = ~(over | under)
    perc[over] = 101.0
    perc[under] = -1.0
    if gmax != gmin:
        perc[inr] = 100.0 * (d[inr] - gmin) / (gmax - gmin)
    else:
        perc[inr] = 0.0
    return perc


# ============================================================================
# Benchmark 2: RMS Calculation
# ============================================================================

def bench_rms_cpu(values, window_size):
    """CPU version using pandas rolling window."""
    series = pd.Series(values)
    rms = series.rolling(window_size, min_periods=max(1, window_size // 2), center=True).apply(
        lambda x: float(np.sqrt(np.nanmean(np.square(x)))), raw=False
    ).to_numpy()
    return rms


def bench_rms_gpu(values_tensor, window_size):
    """GPU version using PyTorch unfold + vectorized operations."""
    # Pad the tensor to handle center=True
    pad_size = window_size // 2
    # For 1D tensor, add batch and channel dimensions before padding
    padded = torch.nn.functional.pad(values_tensor.unsqueeze(0).unsqueeze(0),
                                      (pad_size, pad_size), mode='replicate')

    # Remove batch dimension, keep as 2D [1, length]
    padded = padded.squeeze(0)

    # Unfold creates sliding windows: [1, num_windows, window_size]
    unfolded = padded.unfold(1, window_size, 1)

    # Compute RMS: sqrt(mean(x^2)) along the window dimension
    rms = torch.sqrt(torch.nanmean(unfolded ** 2, dim=-1))

    return rms.squeeze()


# ============================================================================
# Benchmark 3: Angle Multiplier (Vectorized Math)
# ============================================================================

def bench_angle_mult_cpu(angles):
    """CPU version using numpy (current implementation)."""
    clamped = np.clip(angles, -100.0, 100.0)
    multipliers = np.where(
        clamped < 0,
        0.5 + 0.5 * (1.0 + clamped / 100.0),
        1.0 + clamped / 100.0
    )
    return multipliers


def bench_angle_mult_gpu(angles_tensor):
    """GPU version using PyTorch."""
    clamped = torch.clamp(angles_tensor, -100.0, 100.0)
    multipliers = torch.where(
        clamped < 0,
        0.5 + 0.5 * (1.0 + clamped / 100.0),
        1.0 + clamped / 100.0
    )
    return multipliers


# ============================================================================
# Benchmark 4: Acceleration Calculation
# ============================================================================

def bench_acceleration_cpu(combined):
    """CPU version using numpy diff."""
    acceleration = np.full(len(combined), np.nan)
    acceleration[1:] = np.diff(combined)
    return acceleration


def bench_acceleration_gpu(combined_tensor):
    """GPU version using PyTorch diff."""
    acceleration = torch.full((combined_tensor.shape[0],), float('nan'), device=combined_tensor.device)
    acceleration[1:] = torch.diff(combined_tensor)
    return acceleration


# ============================================================================
# Main Benchmark Runner
# ============================================================================

def run_benchmark(size_name, n_rows):
    print(f"\n{'='*80}")
    print(f"Dataset Size: {size_name} ({n_rows:,} rows)")
    print(f"{'='*80}")

    # Generate synthetic data
    np.random.seed(42)
    distances_np = np.random.uniform(10, 250, n_rows).astype(np.float32)
    angles_np = np.random.uniform(-100, 100, n_rows).astype(np.float32)
    combined_np = np.random.uniform(0, 100, n_rows).astype(np.float32)

    # Transfer to GPU
    distances_gpu = torch.from_numpy(distances_np).cuda()
    angles_gpu = torch.from_numpy(angles_np).cuda()
    combined_gpu = torch.from_numpy(combined_np).cuda()

    gmin, gmax = 10.0, 250.0
    window_size = 40  # 1 second at 40fps

    # ========================================================================
    # Test 1: Distance Normalization
    # ========================================================================
    print(f"\n[Test 1] Distance Normalization")

    # CPU timing
    start = time.time()
    for _ in range(REPEAT_COUNT):
        result_cpu = bench_distance_normalize_cpu(distances_np, gmin, gmax)
    cpu_time = (time.time() - start) / REPEAT_COUNT

    # GPU timing
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(REPEAT_COUNT):
        result_gpu = bench_distance_normalize_gpu(distances_gpu, gmin, gmax)
    torch.cuda.synchronize()
    gpu_time = (time.time() - start) / REPEAT_COUNT

    print(f"  CPU: {cpu_time*1000:.2f} ms")
    print(f"  GPU: {gpu_time*1000:.2f} ms")
    print(f"  Speedup: {format_speedup(cpu_time, gpu_time)}")

    # ========================================================================
    # Test 2: RMS Calculation
    # ========================================================================
    print(f"\n[Test 2] RMS Calculation (rolling window)")

    # CPU timing
    start = time.time()
    for _ in range(REPEAT_COUNT):
        result_cpu = bench_rms_cpu(distances_np, window_size)
    cpu_time = (time.time() - start) / REPEAT_COUNT

    # GPU timing
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(REPEAT_COUNT):
        result_gpu = bench_rms_gpu(distances_gpu, window_size)
    torch.cuda.synchronize()
    gpu_time = (time.time() - start) / REPEAT_COUNT

    print(f"  CPU: {cpu_time*1000:.2f} ms")
    print(f"  GPU: {gpu_time*1000:.2f} ms")
    print(f"  Speedup: {format_speedup(cpu_time, gpu_time)}")

    # ========================================================================
    # Test 3: Angle Multiplier
    # ========================================================================
    print(f"\n[Test 3] Angle Multiplier (vectorized math)")

    # CPU timing
    start = time.time()
    for _ in range(REPEAT_COUNT):
        result_cpu = bench_angle_mult_cpu(angles_np)
    cpu_time = (time.time() - start) / REPEAT_COUNT

    # GPU timing
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(REPEAT_COUNT):
        result_gpu = bench_angle_mult_gpu(angles_gpu)
    torch.cuda.synchronize()
    gpu_time = (time.time() - start) / REPEAT_COUNT

    print(f"  CPU: {cpu_time*1000:.2f} ms")
    print(f"  GPU: {gpu_time*1000:.2f} ms")
    print(f"  Speedup: {format_speedup(cpu_time, gpu_time)}")

    # ========================================================================
    # Test 4: Acceleration Calculation
    # ========================================================================
    print(f"\n[Test 4] Acceleration Calculation (frame-to-frame diff)")

    # CPU timing
    start = time.time()
    for _ in range(REPEAT_COUNT):
        result_cpu = bench_acceleration_cpu(combined_np)
    cpu_time = (time.time() - start) / REPEAT_COUNT

    # GPU timing
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(REPEAT_COUNT):
        result_gpu = bench_acceleration_gpu(combined_gpu)
    torch.cuda.synchronize()
    gpu_time = (time.time() - start) / REPEAT_COUNT

    print(f"  CPU: {cpu_time*1000:.2f} ms")
    print(f"  GPU: {gpu_time*1000:.2f} ms")
    print(f"  Speedup: {format_speedup(cpu_time, gpu_time)}")


def main():
    print("\n" + "="*80)
    print("GPU ACCELERATION BENCHMARK")
    print("="*80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print("="*80)

    # Test different dataset sizes
    run_benchmark("SMALL (single trial)", SMALL_SIZE)
    run_benchmark("MEDIUM (typical experiment)", MEDIUM_SIZE)
    run_benchmark("LARGE (full matrix dataset)", LARGE_SIZE)

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("If most operations show >10x speedup on LARGE datasets,")
    print("GPU acceleration is strongly recommended for your pipeline.")
    print("="*80 + "\n")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Cannot run GPU benchmark.")
        exit(1)

    main()
