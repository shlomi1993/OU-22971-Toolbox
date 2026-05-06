#!/usr/bin/env python3
"""
Analyze how batch size scaling affects GPU utilization and throughput.

This script compares baseline vs larger batch configurations to answer the Unit 4 exercise questions:
1. What got denser?
2. What idle gaps shrank?
3. Did throughput improve?
4. Did synchronization become a smaller fraction of the full step?
"""

import json

from pathlib import Path
from typing import Any


def extract_trace_metrics(trace_path: Path) -> dict[str, Any]:
    """
    Extract key timing metrics from a profiler trace file.
    """
    with open(trace_path) as f:
        data = json.load(f)

    events = data if isinstance(data, list) else data.get('traceEvents', [])
    regions = ['train_step', 'next_batch', 'forward', 'backward', 'optimizer_step']

    # Collect all durations for each region
    region_durations: dict[str, list[float]] = {}
    for ev in events:
        if ev.get('ph') == 'X' and ev.get('name') in regions:
            name = ev['name']
            dur_ms = ev['dur'] / 1000.0  # Convert microseconds to milliseconds
            region_durations.setdefault(name, []).append(dur_ms)

    # Calculate averages
    averages = {}
    for region in regions:
        durations = region_durations.get(region, [0])
        averages[region] = sum(durations) / len(durations) if durations else 0.0

    return averages


def parse_log_file(log_path: Path) -> dict[str, Any]:
    """
    Extract throughput metrics from the stdout log file.
    """
    if not log_path.exists():
        return {}

    with open(log_path) as f:
        content = f.read()

    metrics = {}

    # Extract throughput
    for line in content.split('\n'):
        if 'estimated_global_images_per_second=' in line:
            parts = line.split('estimated_global_images_per_second=')
            if len(parts) > 1:
                throughput_str = parts[1].split()[0]
                if 'images/s' in throughput_str:
                    metrics['throughput'] = float(throughput_str.replace('images/s', ''))

        # Extract profiled window info
        if 'profiled_window_seconds=' in line:
            parts = line.split('profiled_window_seconds=')
            if len(parts) > 1:
                metrics['profiled_seconds'] = float(parts[1].split()[0])

        if 'profiled_window_images=' in line:
            parts = line.split('profiled_window_images=')
            if len(parts) > 1:
                metrics['profiled_images'] = int(parts[1].split()[0])

    return metrics


def analyze_configuration(trace_dir: Path, config_name: str, rank: int = 0) -> dict[str, Any]:
    """
    Analyze a single configuration's trace and log.
    """
    trace_path = trace_dir / f"{config_name}_rank{rank}.json"
    if not trace_path.exists():
        print(f"Warning: Trace file not found: {trace_path}")
        return {}

    metrics = extract_trace_metrics(trace_path)

    # Try to find corresponding log
    log_path = trace_dir.parent / "run_logs" / f"{config_name.replace('runpod_gpu_', '')}_stdout.log"
    if not log_path.exists():
        log_path = trace_dir / f"{config_name}_stdout.log"  # Try alternate location

    if log_path.exists():
        log_metrics = parse_log_file(log_path)
        metrics.update(log_metrics)

    return metrics


def print_comparison_table(baseline: dict[str, Any], experiment: dict[str, Any], exp_name: str) -> None:
    """
    Print a comparison table between baseline and experiment.
    """
    print(f"\nComparing: baseline vs {exp_name}")

    regions = ['train_step', 'next_batch', 'forward', 'backward', 'optimizer_step']

    print(f"\n{'Region':<20} {'Baseline (ms)':<15} {exp_name + ' (ms)':<15} {'Ratio':<10}")
    print('-' * 80)

    for region in regions:
        base_val = baseline.get(region, 0.0)
        exp_val = experiment.get(region, 0.0)
        ratio = exp_val / base_val if base_val > 0 else float('inf')

        print(f"{region:<20} {base_val:>12.2f}    {exp_val:>12.2f}    {ratio:>8.2f}x")

    # Throughput comparison
    if 'throughput' in baseline and 'throughput' in experiment:
        base_tp = baseline['throughput']
        exp_tp = experiment['throughput']
        tp_ratio = exp_tp / base_tp if base_tp > 0 else 0.0

        print('\n' + '-' * 80)
        print(f"{'Throughput (img/s)':<20} {base_tp:>12.1f}    {exp_tp:>12.1f}    {tp_ratio:>8.2f}x")

    # Compute vs sync analysis
    print('\n' + '-' * 80)
    print("Analysis:")
    print('-' * 80)

    # Calculate compute time (forward + backward)
    base_compute = baseline.get('forward', 0) + baseline.get('backward', 0)
    exp_compute = experiment.get('forward', 0) + experiment.get('backward', 0)

    # For DDP, gradient sync happens during backward, but we can estimate overhead by comparing step time increment compared to compute
    base_step = baseline.get('train_step', 0)
    exp_step = experiment.get('train_step', 0)

    if base_step > 0:
        base_compute_pct = (base_compute / base_step) * 100
        exp_compute_pct = (exp_compute / exp_step) * 100

        print(f"  Compute % of step:  {base_compute_pct:.1f}% -> {exp_compute_pct:.1f}%")
        print(f"  (Higher % = sync is smaller fraction of total time)")


def main() -> None:
    """
    Main analysis routine.
    """
    # Determine which traces are available (prefer runpod_output for actual GPU traces)
    trace_dir = Path("runpod_output")
    if not trace_dir.exists():
        trace_dir = Path("4_ddp_on_cloud_gpus/runpod_output")
    if not trace_dir.exists():
        trace_dir = Path(".")  # Fall back to current directory traces

    print("\nUnit 4 Exercise: Batch Size Scaling Analysis")

    # Define configurations to compare
    configs = [
        ("runpod_gpu_baseline", "Baseline (batch=64)"),
        ("runpod_gpu_batch256", "Larger Batch (batch=256)"),
    ]

    # Analyze each configuration
    results = {}
    for config_name, display_name in configs:
        print(f"\nAnalyzing {display_name}...")
        metrics = analyze_configuration(trace_dir, config_name)
        if metrics:
            results[config_name] = metrics
            print(f"  [OK] Found trace data")
        else:
            print(f"  [X]  No data found for {config_name}")

    # Print comparisons
    if "runpod_gpu_baseline" in results and "runpod_gpu_batch256" in results:
        print_comparison_table(results["runpod_gpu_baseline"], results["runpod_gpu_batch256"], "batch256")

        # Answer the exercise questions
        print("\nExercise Questions - Answers:")

        baseline = results["runpod_gpu_baseline"]
        batch256 = results["runpod_gpu_batch256"]

        print("\n1. What got denser?")
        fwd_ratio = batch256.get('forward', 0) / baseline.get('forward', 1)
        bwd_ratio = batch256.get('backward', 0) / baseline.get('backward', 1)
        print(f"   - Forward pass: {fwd_ratio:.2f}x longer (more work per kernel)")
        print(f"   - Backward pass: {bwd_ratio:.2f}x longer (more work per kernel)")
        print(f"   -> GPU kernels process more data per launch = denser execution")

        print("\n2. What idle gaps shrank?")
        next_batch_ratio = batch256.get('next_batch', 0) / baseline.get('next_batch', 1)
        print(f"   - next_batch overhead: {next_batch_ratio:.2f}x change")
        if next_batch_ratio < 1.5:
            print(f"   -> Data loading kept pace; gaps between training steps likely smaller")
        else:
            print(f"   -> Data loading became a bottleneck; need more num_workers")

        print("\n3. Did throughput improve?")
        if 'throughput' in baseline and 'throughput' in batch256:
            tp_ratio = batch256['throughput'] / baseline['throughput']
            improvement = (tp_ratio - 1) * 100
            print(f"   - Throughput: {baseline['throughput']:.1f} -> {batch256['throughput']:.1f} images/s")
            print(f"   - Improvement: {improvement:+.1f}%")
            print(f"   -> {'Yes' if tp_ratio > 1.05 else 'Marginal'}, processing more images per unit time")

        print("\n4. Did synchronization become a smaller fraction of the full step?")
        base_compute = baseline.get('forward', 0) + baseline.get('backward', 0)
        exp_compute = batch256.get('forward', 0) + batch256.get('backward', 0)
        base_step = baseline.get('train_step', 0)
        exp_step = batch256.get('train_step', 0)

        if base_step > 0 and exp_step > 0:
            base_compute_pct = (base_compute / base_step) * 100
            exp_compute_pct = (exp_compute / exp_step) * 100

            print(f"   - Compute as % of step: {base_compute_pct:.1f}% -> {exp_compute_pct:.1f}%")
            print(f"   -> {'Yes' if exp_compute_pct > base_compute_pct else 'No'}, gradient sync amortized over more compute")
            print(f"   -> Communication cost stayed roughly constant while compute scaled")
    else:
        print("\nWarning: Could not find both baseline and batch256 traces.")
        print("   Available traces:")
        for name in results:
            print(f"     - {name}")

    print("\nKey Takeaway:")
    print("Larger batches improve compute-to-communication ratio because:")
    print("  - Gradient sync cost ~ constant (same model -> same #parameters)")
    print("  - Compute cost scales linearly with batch size")
    print("  - Result: sync becomes smaller % of total step time")
    print("  - Limit: eventually hit GPU memory or diminishing returns")


if __name__ == "__main__":
    main()
