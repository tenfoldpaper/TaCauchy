#!/usr/bin/env python3
"""
Quick UIPC Performance Test

A simplified script to quickly measure UIPC simulation performance.
"""

import time
import numpy as np
import sys
import os

# Add source to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "source"))

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from tacex_uipc.envs import UipcRLEnv
from tacex_uipc.envs.direct_uipc_rl_env import UipcRLEnvCfg
from tacex_uipc import UipcSimCfg
from tacex_uipc.utils import TetMeshCfg

def quick_benchmark(num_steps=50):
    """Run a quick performance benchmark."""
    print("Setting up environment...")
    
    # Simple configuration
    env_cfg = DirectUipcRlEnvCfg()
    env_cfg.scene.num_envs = 1
    
    env = DirectUipcRlEnv(cfg=env_cfg)
    env.reset()
    
    print("Running benchmark...")
    
    # Warmup
    for _ in range(5):
        env.uipc_sim.step()
    
    # Measure
    times = []
    for i in range(num_steps):
        start = time.perf_counter()
        env.uipc_sim.step()
        end = time.perf_counter()
        times.append(end - start)
        
        if i % 10 == 0:
            print(f"Step {i+1}/{num_steps}")
    
    # Results
    times_ms = np.array(times) * 1000
    mean_time = np.mean(times_ms)
    fps = 1000 / mean_time
    
    print(f"\nResults:")
    print(f"Mean step time: {mean_time:.2f} ms")
    print(f"Max FPS: {fps:.1f}")
    print(f"60 FPS achievable: {'Yes' if fps >= 60 else 'No'}")
    print(f"30 FPS achievable: {'Yes' if fps >= 30 else 'No'}")
    
    env.close()

if __name__ == "__main__":
    try:
        quick_benchmark()
    finally:
        simulation_app.close()