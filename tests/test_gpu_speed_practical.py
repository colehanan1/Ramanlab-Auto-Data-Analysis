#!/usr/bin/env python3
"""
Practical GPU Benchmark: Real-world batch processing scenario.

Instead of microbenchmarks, this tests the actual use case:
Processing multiple CSV files in parallel on GPU vs sequentially on CPU.
"""

import time
import numpy as np
import pandas as pd
import torch

print("\n" + "="*80)
print("PRACTICAL GPU ACCELERATION BENCHMARK")
print("="*80)
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("="*80)

# ============================================================================
# Scenario: Process 100 CSV files (typical experiment)
# ============================================================================

NUM_FILES = 100
ROWS_PER_FILE = 10_000  # ~250 seconds at 40fps (typical trial)
REPEAT_COUNT = 3

print(f"\n[Scenario] Processing {NUM_FILES} CSV files")
print(f"           Each file: {ROWS_PER_FILE:,} rows")
print(f"           Total data: {NUM_FILES * ROWS_PER_FILE:,} rows\n")

# Generate synthetic data for multiple files
np.random.seed(42)
files_data = []
for _ in range(NUM_FILES):
    distances = np.random.uniform(10, 250, ROWS_PER_FILE).astype(np.float32)
    angles = np.random.uniform(-100, 100, ROWS_PER_FILE).astype(np.float32)
    files_data.append((distances, angles))

gmin, gmax = 10.0, 250.0


# ============================================================================
# CPU Approach: Process each file sequentially
# ============================================================================

def process_file_cpu(distances, angles, gmin, gmax):
    """Process a single file on CPU (current implementation)."""
    # Step 1: Normalize distances
    d = distances
    perc = np.empty_like(d, dtype=float)
    over = d > gmax
    under = d < gmin
    inr = ~(over | under)
    perc[over] = 101.0
    perc[under] = -1.0
    perc[inr] = 100.0 * (d[inr] - gmin) / (gmax - gmin)

    # Step 2: Compute angle multiplier
    clamped = np.clip(angles, -100.0, 100.0)
    multipliers = np.where(
        clamped < 0,
        0.5 + 0.5 * (1.0 + clamped / 100.0),
        1.0 + clamped / 100.0
    )

    # Step 3: Compute combined metric
    combined = perc * multipliers

    # Step 4: Compute acceleration
    acceleration = np.full(len(combined), np.nan)
    acceleration[1:] = np.diff(combined)

    return perc, multipliers, combined, acceleration


print("[CPU] Sequential processing (one file at a time)...")
start = time.time()
for _ in range(REPEAT_COUNT):
    results_cpu = []
    for distances, angles in files_data:
        result = process_file_cpu(distances, angles, gmin, gmax)
        results_cpu.append(result)
cpu_time = (time.time() - start) / REPEAT_COUNT
print(f"      Time: {cpu_time:.3f} seconds")
print(f"      Throughput: {NUM_FILES / cpu_time:.1f} files/sec")


# ============================================================================
# GPU Approach: Batch process all files at once
# ============================================================================

def process_batch_gpu(distances_list, angles_list, gmin, gmax):
    """Process multiple files in a single GPU batch."""
    # Concatenate all files into a single batch
    all_distances = torch.from_numpy(np.concatenate(distances_list)).cuda()
    all_angles = torch.from_numpy(np.concatenate(angles_list)).cuda()

    # Step 1: Normalize distances (vectorized across all files)
    d = all_distances
    perc = torch.empty_like(d, dtype=torch.float32)
    over = d > gmax
    under = d < gmin
    inr = ~(over | under)
    perc[over] = 101.0
    perc[under] = -1.0
    perc[inr] = 100.0 * (d[inr] - gmin) / (gmax - gmin)

    # Step 2: Compute angle multiplier (vectorized)
    clamped = torch.clamp(all_angles, -100.0, 100.0)
    multipliers = torch.where(
        clamped < 0,
        0.5 + 0.5 * (1.0 + clamped / 100.0),
        1.0 + clamped / 100.0
    )

    # Step 3: Compute combined metric
    combined = perc * multipliers

    # Step 4: Compute acceleration (need to handle file boundaries)
    # For simplicity, treating as one large batch (in practice, would handle per-file)
    acceleration = torch.full((combined.shape[0],), float('nan'), device=combined.device)
    acceleration[1:] = torch.diff(combined)

    return perc, multipliers, combined, acceleration


print("\n[GPU] Batch processing (all files at once)...")
distances_list = [d for d, _ in files_data]
angles_list = [a for _, a in files_data]

# Warmup
_ = process_batch_gpu(distances_list[:10], angles_list[:10], gmin, gmax)
torch.cuda.synchronize()

# Actual timing
start = time.time()
for _ in range(REPEAT_COUNT):
    result_gpu = process_batch_gpu(distances_list, angles_list, gmin, gmax)
torch.cuda.synchronize()
gpu_time = (time.time() - start) / REPEAT_COUNT
print(f"      Time: {gpu_time:.3f} seconds")
print(f"      Throughput: {NUM_FILES / gpu_time:.1f} files/sec")


# ============================================================================
# Results Summary
# ============================================================================

speedup = cpu_time / gpu_time
print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"CPU Time:     {cpu_time:.3f} sec")
print(f"GPU Time:     {gpu_time:.3f} sec")
print(f"Speedup:      {speedup:.1f}x")
print(f"Time Saved:   {cpu_time - gpu_time:.3f} sec ({100*(cpu_time-gpu_time)/cpu_time:.0f}%)")

if speedup > 10:
    print("\n\033[92m✓ STRONG RECOMMENDATION: Implement GPU acceleration\033[0m")
    print("  The {:.1f}x speedup will significantly reduce pipeline runtime.".format(speedup))
elif speedup > 5:
    print("\n\033[93m✓ MODERATE RECOMMENDATION: GPU acceleration beneficial\033[0m")
    print("  The {:.1f}x speedup is worthwhile for large datasets.".format(speedup))
else:
    print("\n\033[91m⚠ LIMITED BENEFIT: GPU overhead dominates\033[0m")
    print("  The {:.1f}x speedup may not justify implementation complexity.".format(speedup))

print("="*80 + "\n")


# ============================================================================
# Memory Analysis
# ============================================================================

print("\n" + "="*80)
print("VRAM USAGE ANALYSIS")
print("="*80)

# Calculate memory requirements
bytes_per_file = ROWS_PER_FILE * 4  # float32
total_mb = (NUM_FILES * bytes_per_file * 4) / 1024**2  # 4 arrays per file
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

print(f"Memory per file:     {bytes_per_file / 1024:.1f} KB")
print(f"Total batch memory:  {total_mb:.1f} MB (for {NUM_FILES} files)")
print(f"Available VRAM:      {vram_gb:.1f} GB")
print(f"VRAM utilization:    {100 * total_mb / (vram_gb * 1024):.1f}%")

max_files = int((vram_gb * 1024 * 0.8) / (bytes_per_file * 4 / 1024))  # 80% VRAM limit
print(f"\nMax batch size:      {max_files} files (at 80% VRAM)")
print(f"Recommended chunks:  Process {max_files} files at a time")

print("="*80 + "\n")
