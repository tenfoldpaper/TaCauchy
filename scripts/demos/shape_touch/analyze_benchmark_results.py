#!/usr/bin/env python3
"""
Analyze and compare benchmark results from GUI vs Headless tests.

Usage:
    python scripts/demos/shape_touch/analyze_benchmark_results.py
    python scripts/demos/shape_touch/analyze_benchmark_results.py --plot
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import numpy as np


def load_latest_benchmarks(benchmark_dir: Path, sensor: str = None, num_envs: int = None):
    """Load the latest GUI and Headless benchmark results."""
    
    json_files = sorted(benchmark_dir.glob("benchmark_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    gui_result = None
    headless_result = None
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Filter by sensor and num_envs if specified
        if sensor and data['config']['sensor'] != sensor:
            continue
        if num_envs and data['config']['num_envs'] != num_envs:
            continue
        
        if not data['config']['headless'] and gui_result is None:
            gui_result = data
        elif data['config']['headless'] and headless_result is None:
            headless_result = data
        
        if gui_result and headless_result:
            break
    
    return gui_result, headless_result


def print_comparison(gui_result: Dict, headless_result: Dict):
    """Print detailed comparison between GUI and Headless results."""
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON: GUI vs HEADLESS")
    print("=" * 80)
    
    # Configuration
    print(f"\nConfiguration:")
    print(f"  Sensor Type:      {gui_result['config']['sensor']}")
    print(f"  Num Environments: {gui_result['config']['num_envs']}")
    print(f"  Steps:            {gui_result['config']['steps']}")
    
    # Overall Performance
    gui_stats = gui_result['statistics']
    headless_stats = headless_result['statistics']
    
    print(f"\n{'Metric':<25} {'GUI':<20} {'Headless':<20} {'Speedup':<15}")
    print("-" * 80)
    
    # FPS
    gui_fps = gui_stats['total']['fps']
    headless_fps = headless_stats['total']['fps']
    speedup = headless_fps / gui_fps
    print(f"{'Average FPS':<25} {gui_fps:>8.2f} fps       {headless_fps:>8.2f} fps       {speedup:>6.2f}x")
    
    # Step Time
    gui_step = gui_stats['total']['mean']
    headless_step = headless_stats['total']['mean']
    speedup = gui_step / headless_step
    print(f"{'Step Time (mean)':<25} {gui_step:>8.2f} ms        {headless_step:>8.2f} ms        {speedup:>6.2f}x")
    
    # Step Time Std
    gui_std = gui_stats['total']['std']
    headless_std = headless_stats['total']['std']
    print(f"{'Step Time (std)':<25} {gui_std:>8.2f} ms        {headless_std:>8.2f} ms")
    
    print("\n" + "-" * 80)
    print("Detailed Breakdown:")
    print("-" * 80)
    
    # Physics
    gui_physics = gui_stats['physics']['mean']
    headless_physics = headless_stats['physics']['mean']
    speedup = gui_physics / headless_physics if headless_physics > 0 else 1.0
    print(f"{'Physics Step':<25} {gui_physics:>8.2f} ms        {headless_physics:>8.2f} ms        {speedup:>6.2f}x")
    
    # Render
    gui_render = gui_stats['render']['mean']
    headless_render = headless_stats['render']['mean']
    speedup = gui_render / headless_render if headless_render > 0 else 1.0
    print(f"{'Render':<25} {gui_render:>8.2f} ms        {headless_render:>8.2f} ms        {speedup:>6.2f}x")
    
    # Sensor
    gui_sensor = gui_stats['sensor']['mean']
    headless_sensor = headless_stats['sensor']['mean']
    speedup = gui_sensor / headless_sensor if headless_sensor > 0 else 1.0
    print(f"{'Sensor Update':<25} {gui_sensor:>8.2f} ms        {headless_sensor:>8.2f} ms        {speedup:>6.2f}x")
    
    # Stress
    gui_stress = gui_stats['stress']['mean']
    headless_stress = headless_stats['stress']['mean']
    speedup = gui_stress / headless_stress if headless_stress > 0 else 1.0
    print(f"{'Stress Extract':<25} {gui_stress:>8.2f} ms        {headless_stress:>8.2f} ms        {speedup:>6.2f}x")
    
    print("\n" + "=" * 80)
    
    # Summary
    overall_speedup = headless_fps / gui_fps
    print(f"\nSUMMARY:")
    print(f"  Headless mode is {overall_speedup:.2f}x faster than GUI mode")
    print(f"  GUI:      {gui_fps:.2f} FPS ({gui_step:.2f} ms/step)")
    print(f"  Headless: {headless_fps:.2f} FPS ({headless_step:.2f} ms/step)")
    
    # Time savings
    total_time_gui = gui_step * gui_result['config']['steps'] / 1000  # seconds
    total_time_headless = headless_step * headless_result['config']['steps'] / 1000
    time_saved = total_time_gui - total_time_headless
    print(f"\n  Time saved: {time_saved:.2f} seconds ({time_saved/60:.2f} minutes)")
    print("=" * 80 + "\n")


def plot_comparison(gui_result: Dict, headless_result: Dict):
    """Plot comparison charts."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("[WARNING] matplotlib not available, skipping plots")
        return
    
    gui_stats = gui_result['statistics']
    headless_stats = headless_result['statistics']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. FPS Comparison
    ax = axes[0, 0]
    modes = ['GUI', 'Headless']
    fps_values = [gui_stats['total']['fps'], headless_stats['total']['fps']]
    bars = ax.bar(modes, fps_values, color=['#3498db', '#2ecc71'], alpha=0.8, edgecolor='black')
    ax.set_ylabel('FPS', fontsize=14, weight='bold')
    ax.set_title('Average FPS Comparison', fontsize=16, weight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, fps in zip(bars, fps_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{fps:.2f}',
                ha='center', va='bottom', fontsize=12, weight='bold')
    
    # 2. Step Time Breakdown
    ax = axes[0, 1]
    categories = ['Physics', 'Render', 'Sensor', 'Stress']
    gui_times = [
        gui_stats['physics']['mean'],
        gui_stats['render']['mean'],
        gui_stats['sensor']['mean'],
        gui_stats['stress']['mean']
    ]
    headless_times = [
        headless_stats['physics']['mean'],
        headless_stats['render']['mean'],
        headless_stats['sensor']['mean'],
        headless_stats['stress']['mean']
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, gui_times, width, label='GUI', color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, headless_times, width, label='Headless', color='#2ecc71', alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Time (ms)', fontsize=14, weight='bold')
    ax.set_title('Step Time Breakdown', fontsize=16, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Speedup Factors
    ax = axes[1, 0]
    speedups = []
    for gui_t, headless_t in zip(gui_times, headless_times):
        if headless_t > 0:
            speedups.append(gui_t / headless_t)
        else:
            speedups.append(1.0)
    
    bars = ax.bar(categories, speedups, color='#e74c3c', alpha=0.8, edgecolor='black')
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='No speedup')
    ax.set_ylabel('Speedup Factor', fontsize=14, weight='bold')
    ax.set_title('Headless Speedup (higher is better)', fontsize=16, weight='bold')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x',
                ha='center', va='bottom', fontsize=11, weight='bold')
    
    # 4. Time Distribution (Pie charts)
    ax = axes[1, 1]
    ax.axis('off')
    
    # GUI pie chart
    ax1 = plt.subplot(2, 4, 7)
    ax1.pie(gui_times, labels=categories, autopct='%1.1f%%', startangle=90, colors=['#3498db', '#e74c3c', '#f39c12', '#9b59b6'])
    ax1.set_title('GUI Time Distribution', fontsize=12, weight='bold')
    
    # Headless pie chart
    ax2 = plt.subplot(2, 4, 8)
    ax2.pie(headless_times, labels=categories, autopct='%1.1f%%', startangle=90, colors=['#3498db', '#e74c3c', '#f39c12', '#9b59b6'])
    ax2.set_title('Headless Time Distribution', fontsize=12, weight='bold')
    
    plt.suptitle(f"Performance Comparison: {gui_result['config']['sensor'].upper()} "
                f"({gui_result['config']['num_envs']} environments)", 
                fontsize=18, weight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("outputs/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"comparison_{gui_result['config']['sensor']}_env{gui_result['config']['num_envs']}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[INFO] Comparison plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--sensor", type=str, help="Filter by sensor type")
    parser.add_argument("--num_envs", type=int, help="Filter by number of environments")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots")
    args = parser.parse_args()
    
    # Load results
    benchmark_dir = Path("outputs/benchmarks")
    if not benchmark_dir.exists():
        print(f"[ERROR] Benchmark directory not found: {benchmark_dir}")
        return
    
    print("\n[INFO] Loading benchmark results...")
    gui_result, headless_result = load_latest_benchmarks(benchmark_dir, args.sensor, args.num_envs)
    
    if not gui_result or not headless_result:
        print("[ERROR] Could not find matching GUI and Headless benchmark results")
        print(f"  GUI result found: {gui_result is not None}")
        print(f"  Headless result found: {headless_result is not None}")
        return
    
    # Print comparison
    print_comparison(gui_result, headless_result)
    
    # Plot if requested
    if args.plot:
        print("\n[INFO] Generating comparison plots...")
        plot_comparison(gui_result, headless_result)


if __name__ == "__main__":
    main()
