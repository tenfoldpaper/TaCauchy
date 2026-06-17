#!/usr/bin/env python3
"""
UIPC Performance Benchmark Script

This script measures the computational performance of UIPC finite element simulation
to estimate the maximum achievable frame rate for pure mechanics calculations.

Usage:
    python scripts/benchmarking/uipc_performance_benchmark.py --sensor gelsight --steps 100
"""

import argparse
import time
import numpy as np
from typing import Dict, List
import sys
import os

# Add the source directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "source"))

# Isaac Sim imports
from isaacsim import SimulationApp

# Configuration for headless mode (no GUI for better performance measurement)
simulation_app = SimulationApp({"headless": True, "width": 512, "height": 512})

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core import World

# TacEx imports
from tacex_uipc import UipcSim, UipcSimCfg
from tacex_uipc.envs import DirectUipcRlEnv, DirectUipcRlEnvCfg
from tacex_uipc.sim import ContactForceExtractor, StressExtractor
from tacex_uipc.utils import TetMeshCfg

# UIPC Timer
from uipc import Timer


class UipcPerformanceBenchmark:
    """Benchmark UIPC simulation performance."""
    
    def __init__(self, sensor_type: str = "gelsight"):
        self.sensor_type = sensor_type
        self.env = None
        self.uipc_sim = None
        self.contact_extractor = None
        self.stress_extractor = None
        
        # Performance metrics
        self.step_times = []
        self.contact_extraction_times = []
        self.stress_extraction_times = []
        self.mesh_info = {}
        
    def setup_environment(self, mesh_density: str = "standard"):
        """Setup the simulation environment with specified mesh density."""
        print(f"[Benchmark] Setting up {self.sensor_type} environment with {mesh_density} mesh...")
        
        # Configure mesh density
        mesh_configs = {
            "coarse": TetMeshCfg(
                edge_length_r=0.3,
                enable_center_refinement=False,
                log_level=6
            ),
            "standard": TetMeshCfg(
                edge_length_r=0.2,
                enable_center_refinement=True,
                center_region_ratio=0.3,
                center_edge_length_r=0.1,
                log_level=6
            ),
            "dense": TetMeshCfg(
                edge_length_r=0.15,
                enable_center_refinement=True,
                center_region_ratio=0.4,
                center_edge_length_r=0.06,
                log_level=6
            ),
            "ultra_dense": TetMeshCfg(
                edge_length_r=0.1,
                enable_center_refinement=True,
                center_region_ratio=0.5,
                center_edge_length_r=0.04,
                log_level=6
            )
        }
        
        mesh_cfg = mesh_configs.get(mesh_density, mesh_configs["standard"])
        
        # Environment configuration
        env_cfg = DirectUipcRlEnvCfg()
        env_cfg.scene.num_envs = 1  # Single environment for benchmarking
        env_cfg.scene.env_spacing = 2.0
        
        # UIPC simulation configuration
        uipc_cfg = UipcSimCfg()
        uipc_cfg.dt = 1/60  # 60 FPS target
        uipc_cfg.device = "cuda"
        uipc_cfg.logger_level = "Error"  # Minimize logging overhead
        
        # Contact parameters for realistic simulation
        uipc_cfg.contact.enable = True
        uipc_cfg.contact.enable_friction = True
        uipc_cfg.contact.default_friction_ratio = 0.5
        uipc_cfg.contact.d_hat = 0.0005  # 0.5mm contact threshold
        
        # Newton solver parameters
        uipc_cfg.newton.max_iter = 512
        uipc_cfg.newton.velocity_tol = 0.05
        
        # Create environment
        self.env = DirectUipcRlEnv(cfg=env_cfg)
        
        # Get UIPC simulation instance
        self.uipc_sim = self.env.uipc_sim
        self.uipc_sim.cfg = uipc_cfg
        
        # Initialize extractors
        self.contact_extractor = ContactForceExtractor(self.uipc_sim)
        self.stress_extractor = StressExtractor()
        
        # Reset environment to initialize everything
        self.env.reset()
        
        # Get mesh information
        self._collect_mesh_info()
        
        print(f"[Benchmark] Environment setup complete")
        print(f"  Mesh vertices: {self.mesh_info.get('num_vertices', 'Unknown')}")
        print(f"  Mesh tetrahedra: {self.mesh_info.get('num_tetrahedra', 'Unknown')}")
        print(f"  Surface triangles: {self.mesh_info.get('num_surface_triangles', 'Unknown')}")
        
    def _collect_mesh_info(self):
        """Collect mesh statistics for reporting."""
        try:
            if hasattr(self.env, '_uipc_gelpad') and self.env._uipc_gelpad:
                geo = self.env._uipc_gelpad.geo_slot_list[0].geometry()
                
                # Get vertex count
                positions = geo.positions().view()
                self.mesh_info['num_vertices'] = len(positions) // 3
                
                # Get tetrahedra count
                tets = geo.tetrahedra().topo()
                if tets is not None:
                    self.mesh_info['num_tetrahedra'] = tets.size() // 4
                
                # Get surface triangle count
                triangles = geo.triangles().topo()
                if triangles is not None:
                    self.mesh_info['num_surface_triangles'] = triangles.size() // 3
                    
        except Exception as e:
            print(f"[Warning] Could not collect mesh info: {e}")
    
    def run_benchmark(self, num_steps: int = 100, warmup_steps: int = 10):
        """Run the performance benchmark."""
        print(f"[Benchmark] Starting benchmark: {warmup_steps} warmup + {num_steps} measured steps")
        
        # Enable UIPC timers
        Timer.enable_all()
        
        # Warmup phase
        print("[Benchmark] Warmup phase...")
        for i in range(warmup_steps):
            self._single_step(measure=False)
            if i % 5 == 0:
                print(f"  Warmup step {i+1}/{warmup_steps}")
        
        # Clear any accumulated timer data
        Timer.clear_all()
        
        # Measurement phase
        print("[Benchmark] Measurement phase...")
        for i in range(num_steps):
            self._single_step(measure=True)
            if i % 20 == 0:
                print(f"  Benchmark step {i+1}/{num_steps}")
        
        # Analyze results
        self._analyze_results()
    
    def _single_step(self, measure: bool = True):
        """Execute a single simulation step with optional timing measurement."""
        
        if measure:
            # Measure UIPC step time
            step_start = time.perf_counter()
        
        # Execute UIPC simulation step
        self.uipc_sim.step()
        
        if measure:
            step_end = time.perf_counter()
            step_time = step_end - step_start
            self.step_times.append(step_time)
            
            # Measure contact force extraction time
            contact_start = time.perf_counter()
            contact_forces = self.contact_extractor.extract_contact_forces()
            contact_end = time.perf_counter()
            contact_time = contact_end - contact_start
            self.contact_extraction_times.append(contact_time)
            
            # Measure stress extraction time
            stress_start = time.perf_counter()
            if hasattr(self.env, '_uipc_gelpad') and self.env._uipc_gelpad:
                stress_data = self.stress_extractor.get_object_surface_stress(self.env._uipc_gelpad)
            stress_end = time.perf_counter()
            stress_time = stress_end - stress_start
            self.stress_extraction_times.append(stress_time)
    
    def _analyze_results(self):
        """Analyze and report benchmark results."""
        print("\n" + "="*70)
        print("UIPC PERFORMANCE BENCHMARK RESULTS")
        print("="*70)
        
        # Mesh information
        print(f"\nMesh Information:")
        print(f"  Sensor type: {self.sensor_type}")
        print(f"  Vertices: {self.mesh_info.get('num_vertices', 'Unknown')}")
        print(f"  Tetrahedra: {self.mesh_info.get('num_tetrahedra', 'Unknown')}")
        print(f"  Surface triangles: {self.mesh_info.get('num_surface_triangles', 'Unknown')}")
        
        # UIPC step timing
        step_times_ms = np.array(self.step_times) * 1000
        print(f"\nUIPC Simulation Step Timing:")
        print(f"  Mean: {np.mean(step_times_ms):.2f} ms")
        print(f"  Std:  {np.std(step_times_ms):.2f} ms")
        print(f"  Min:  {np.min(step_times_ms):.2f} ms")
        print(f"  Max:  {np.max(step_times_ms):.2f} ms")
        print(f"  P95:  {np.percentile(step_times_ms, 95):.2f} ms")
        print(f"  P99:  {np.percentile(step_times_ms, 99):.2f} ms")
        
        # Frame rate estimation
        mean_step_time = np.mean(self.step_times)
        max_fps = 1.0 / mean_step_time
        stable_fps = 1.0 / np.percentile(self.step_times, 95)  # 95th percentile for stability
        
        print(f"\nFrame Rate Estimation (Pure Mechanics):")
        print(f"  Theoretical max FPS: {max_fps:.1f}")
        print(f"  Stable FPS (P95):    {stable_fps:.1f}")
        print(f"  Real-time targets:")
        print(f"    60 FPS: {'✓ ACHIEVABLE' if stable_fps >= 60 else '✗ NOT ACHIEVABLE'}")
        print(f"    30 FPS: {'✓ ACHIEVABLE' if stable_fps >= 30 else '✗ NOT ACHIEVABLE'}")
        print(f"    15 FPS: {'✓ ACHIEVABLE' if stable_fps >= 15 else '✗ NOT ACHIEVABLE'}")
        
        # Contact force extraction timing
        contact_times_ms = np.array(self.contact_extraction_times) * 1000
        print(f"\nContact Force Extraction Timing:")
        print(f"  Mean: {np.mean(contact_times_ms):.3f} ms")
        print(f"  Max:  {np.max(contact_times_ms):.3f} ms")
        
        # Stress extraction timing
        stress_times_ms = np.array(self.stress_extraction_times) * 1000
        print(f"\nStress Extraction Timing:")
        print(f"  Mean: {np.mean(stress_times_ms):.3f} ms")
        print(f"  Max:  {np.max(stress_times_ms):.3f} ms")
        
        # Total processing time
        total_times = np.array(self.step_times) + np.array(self.contact_extraction_times) + np.array(self.stress_extraction_times)
        total_times_ms = total_times * 1000
        total_fps = 1.0 / np.mean(total_times)
        
        print(f"\nTotal Processing Time (UIPC + Force Extraction):")
        print(f"  Mean: {np.mean(total_times_ms):.2f} ms")
        print(f"  Max FPS with force extraction: {total_fps:.1f}")
        
        # UIPC internal timers (if available)
        try:
            uipc_step_info = Timer.get_timer_info("uipc_step")
            if uipc_step_info > 0:
                print(f"\nUIPC Internal Timer:")
                print(f"  UIPC step time: {uipc_step_info:.2f} ms")
        except:
            pass
        
        # Performance recommendations
        print(f"\nPerformance Recommendations:")
        if stable_fps < 30:
            print("  - Consider using coarser mesh (increase edge_length_r)")
            print("  - Reduce center refinement region")
            print("  - Increase Newton solver tolerances")
            print("  - Reduce contact detection accuracy (increase d_hat)")
        elif stable_fps < 60:
            print("  - Current settings suitable for 30 FPS real-time simulation")
            print("  - For 60 FPS, consider slight mesh coarsening")
        else:
            print("  - Excellent performance! Can achieve 60+ FPS")
            print("  - Consider increasing mesh density for higher accuracy")
        
        print("\n" + "="*70)


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="UIPC Performance Benchmark")
    parser.add_argument("--sensor", type=str, default="gelsight", 
                       choices=["gelsight", "digit", "9dtact"],
                       help="Sensor type to benchmark")
    parser.add_argument("--steps", type=int, default=100,
                       help="Number of benchmark steps")
    parser.add_argument("--warmup", type=int, default=10,
                       help="Number of warmup steps")
    parser.add_argument("--mesh", type=str, default="standard",
                       choices=["coarse", "standard", "dense", "ultra_dense"],
                       help="Mesh density level")
    
    args = parser.parse_args()
    
    try:
        # Create benchmark instance
        benchmark = UipcPerformanceBenchmark(sensor_type=args.sensor)
        
        # Setup environment
        benchmark.setup_environment(mesh_density=args.mesh)
        
        # Run benchmark
        benchmark.run_benchmark(num_steps=args.steps, warmup_steps=args.warmup)
        
    except KeyboardInterrupt:
        print("\n[Benchmark] Interrupted by user")
    except Exception as e:
        print(f"\n[Benchmark] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'benchmark' in locals() and benchmark.env:
            benchmark.env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()