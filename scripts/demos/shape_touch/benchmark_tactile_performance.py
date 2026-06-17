"""
Tactile Simulation Performance Benchmark

Benchmarks the simulation performance in both GUI and headless modes.
No OpenCV visualization windows - pure performance testing.

Usage:
    # Test in GUI mode
    python scripts/demos/shape_touch/benchmark_tactile_performance.py --sensor gelsight
    
    # Test in headless mode
    python scripts/demos/shape_touch/benchmark_tactile_performance.py --sensor gelsight --headless
    
    # Test with multiple environments
    python scripts/demos/shape_touch/benchmark_tactile_performance.py --sensor digit --num_envs 4
    
    # Custom benchmark duration
    python scripts/demos/shape_touch/benchmark_tactile_performance.py --sensor 9dtact --steps 1000
"""

from __future__ import annotations

import argparse
import time
import numpy as np
from pathlib import Path

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Tactile Simulation Performance Benchmark")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--sensor", type=str, default="gelsight", 
                   choices=["gelsight", "digit", "9dtact"], help="Sensor type")
parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps")
parser.add_argument("--warmup_steps", type=int, default=50, help="Warmup steps before timing")
parser.add_argument("--save_results", action="store_true", help="Save results to file")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaacsim.core.prims import XFormPrim

from tacex_uipc.sim import StressExtractor

# Import environment configuration helper
from simple_tactile_env import create_tactile_env


class PerformanceBenchmark:
    """Performance benchmark tracker."""
    
    def __init__(self, num_steps: int, warmup_steps: int):
        self.num_steps = num_steps
        self.warmup_steps = warmup_steps
        self.step_times = []
        self.physics_times = []
        self.render_times = []
        self.sensor_times = []
        self.stress_times = []
        
    def record_step(self, physics_time: float, render_time: float, 
                   sensor_time: float, stress_time: float):
        """Record timing for a single step."""
        total_time = physics_time + render_time + sensor_time + stress_time
        self.step_times.append(total_time)
        self.physics_times.append(physics_time)
        self.render_times.append(render_time)
        self.sensor_times.append(sensor_time)
        self.stress_times.append(stress_time)
    
    def get_statistics(self):
        """Calculate statistics."""
        if not self.step_times:
            return None
        
        return {
            'total': {
                'mean': np.mean(self.step_times) * 1000,  # ms
                'std': np.std(self.step_times) * 1000,
                'min': np.min(self.step_times) * 1000,
                'max': np.max(self.step_times) * 1000,
                'fps': 1.0 / np.mean(self.step_times),
            },
            'physics': {
                'mean': np.mean(self.physics_times) * 1000,
                'std': np.std(self.physics_times) * 1000,
            },
            'render': {
                'mean': np.mean(self.render_times) * 1000,
                'std': np.std(self.render_times) * 1000,
            },
            'sensor': {
                'mean': np.mean(self.sensor_times) * 1000,
                'std': np.std(self.sensor_times) * 1000,
            },
            'stress': {
                'mean': np.mean(self.stress_times) * 1000,
                'std': np.std(self.stress_times) * 1000,
            },
        }
    
    def print_results(self, config: dict):
        """Print benchmark results."""
        stats = self.get_statistics()
        if stats is None:
            print("[ERROR] No timing data collected")
            return
        
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Sensor Type:      {config['sensor']}")
        print(f"  Num Environments: {config['num_envs']}")
        print(f"  Render Mode:      {'GUI' if not config['headless'] else 'Headless'}")
        print(f"  Total Steps:      {config['steps']}")
        print(f"  Warmup Steps:     {config['warmup_steps']}")
        
        print(f"\nOverall Performance:")
        print(f"  Average FPS:      {stats['total']['fps']:.2f}")
        print(f"  Step Time:        {stats['total']['mean']:.2f} ± {stats['total']['std']:.2f} ms")
        print(f"  Min/Max:          {stats['total']['min']:.2f} / {stats['total']['max']:.2f} ms")
        
        print(f"\nDetailed Breakdown:")
        print(f"  Physics Step:     {stats['physics']['mean']:.2f} ± {stats['physics']['std']:.2f} ms")
        print(f"  Render:           {stats['render']['mean']:.2f} ± {stats['render']['std']:.2f} ms")
        print(f"  Sensor Update:    {stats['sensor']['mean']:.2f} ± {stats['sensor']['std']:.2f} ms")
        print(f"  Stress Extract:   {stats['stress']['mean']:.2f} ± {stats['stress']['std']:.2f} ms")
        
        print("\n" + "=" * 80)
    
    def save_results(self, config: dict, output_path: Path):
        """Save results to JSON file."""
        import json
        
        stats = self.get_statistics()
        if stats is None:
            print("[ERROR] No timing data to save")
            return
        
        results = {
            'config': config,
            'statistics': stats,
            'raw_times': {
                'step_times': [float(t) for t in self.step_times],
                'physics_times': [float(t) for t in self.physics_times],
                'render_times': [float(t) for t in self.render_times],
                'sensor_times': [float(t) for t in self.sensor_times],
                'stress_times': [float(t) for t in self.stress_times],
            }
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[INFO] Results saved to: {output_path}")


def main():
    """Main benchmark loop."""
    
    # Configuration
    config = {
        'sensor': args_cli.sensor,
        'num_envs': args_cli.num_envs,
        'headless': args_cli.headless,
        'steps': args_cli.steps,
        'warmup_steps': args_cli.warmup_steps,
    }
    
    # Create environment
    print(f"\n[INFO] Creating environment with {args_cli.sensor} sensor...")
    print(f"[INFO] Mode: {'Headless' if args_cli.headless else 'GUI'}")
    print(f"[INFO] Environments: {args_cli.num_envs}")
    
    env = create_tactile_env(
        sensor_type=args_cli.sensor,
        num_envs=args_cli.num_envs,
        debug_vis=False  # No debug visualization for benchmarking
    )
    
    # Reset environment
    env.reset()
    
    # Initialize goal control
    env.goal_prim_view = XFormPrim(prim_paths_expr="/World/envs/env_.*/goal", name="goals", usd=True)
    
    # Initialize stress extractor (no visualizers)
    stress_extractor = StressExtractor()
    
    # Initialize benchmark tracker
    benchmark = PerformanceBenchmark(args_cli.steps, args_cli.warmup_steps)
    
    print(f"\n[INFO] Starting benchmark...")
    print(f"[INFO] Warmup: {args_cli.warmup_steps} steps")
    print(f"[INFO] Benchmark: {args_cli.steps} steps")
    print(f"[INFO] No visualization windows (pure performance test)")
    print()
    
    step_count = 0
    total_steps = args_cli.warmup_steps + args_cli.steps
    
    try:
        while simulation_app.is_running() and step_count < total_steps:
            # Physics step
            t0 = time.perf_counter()
            env._pre_physics_step(None)
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.uipc_sim.update_render_meshes()
            t1 = time.perf_counter()
            physics_time = t1 - t0
            
            # Render
            t0 = time.perf_counter()
            env.sim.render()
            t1 = time.perf_counter()
            render_time = t1 - t0
            
            # Update goal-based IK commands
            positions, orientations = env.goal_prim_view.get_world_poses()
            env.ik_commands[:, :3] = positions - env.scene.env_origins
            env.ik_commands[:, 3:] = orientations
            
            # Update sensor
            t0 = time.perf_counter()
            env.gsmini.update(dt=env.physics_dt)
            env.scene.update(dt=env.physics_dt)
            t1 = time.perf_counter()
            sensor_time = t1 - t0
            
            # Extract stress data (no visualization)
            t0 = time.perf_counter()
            env._uipc_gelpad.update(0.0)
            stress_data = stress_extractor.get_object_surface_stress(env._uipc_gelpad)
            t1 = time.perf_counter()
            stress_time = t1 - t0
            
            # Record timing (skip warmup)
            if step_count >= args_cli.warmup_steps:
                benchmark.record_step(physics_time, render_time, sensor_time, stress_time)
            
            step_count += 1
            
            # Progress indicator
            if step_count % 50 == 0:
                if step_count < args_cli.warmup_steps:
                    print(f"[WARMUP] Step {step_count}/{args_cli.warmup_steps}")
                else:
                    actual_step = step_count - args_cli.warmup_steps
                    print(f"[BENCHMARK] Step {actual_step}/{args_cli.steps}")
            
    except KeyboardInterrupt:
        print("\n[INFO] Benchmark stopped by user")
    
    # Print results
    benchmark.print_results(config)
    
    # Save results if requested
    if args_cli.save_results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        mode = "headless" if args_cli.headless else "gui"
        filename = f"benchmark_{args_cli.sensor}_{mode}_env{args_cli.num_envs}_{timestamp}.json"
        output_path = Path("outputs") / "benchmarks" / filename
        benchmark.save_results(config, output_path)
    
    # Cleanup
    simulation_app.close()


if __name__ == "__main__":
    main()
