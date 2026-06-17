"""
Simple Tactile Sensor Demo

A clean, minimal script for tactile sensor simulation with:
- Multiple UIPC objects for interaction
- Goal-based robot control
- Normal/Tangential force visualization (point-based)
- Interpolated force visualization (20x20 grid)
- Tactile RGB image display

Usage:
    python scripts/demos/shape_touch/simple_tactile_demo.py --sensor gelsight
    python scripts/demos/shape_touch/simple_tactile_demo.py --sensor digit
    python scripts/demos/shape_touch/simple_tactile_demo.py --sensor 9dtact
"""

from __future__ import annotations

import argparse
import numpy as np
import cv2

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Simple Tactile Sensor Demo")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--sensor", type=str, default="gelsight", 
                   choices=["gelsight", "digit", "9dtact"], help="Sensor type")
parser.add_argument("--debug_vis", action="store_true", help="Enable debug visualization")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaacsim.core.prims import XFormPrim

from tacex_uipc.sim import StressExtractor
from tacex_uipc.utils.visualization import (
    TactileForceVisualizer,
    TactileImageViewer,
    InterpolatedForceVisualizer,
)

# Import environment configuration helper
from simple_tactile_env import create_tactile_env


def main():
    """Main simulation loop."""
    
    # Create environment
    print(f"\n[INFO] Creating environment with {args_cli.sensor} sensor...")
    env = create_tactile_env(
        sensor_type=args_cli.sensor,
        num_envs=args_cli.num_envs,
        debug_vis=args_cli.debug_vis
    )
    
    # Reset environment
    env.reset()
    
    # Initialize goal control
    env.goal_prim_view = XFormPrim(prim_paths_expr="/World/envs/env_.*/goal", name="goals", usd=True)
    print(f"[INFO] Goal control initialized - drag the red cube to move the robot")
    
    # Initialize visualizers
    print("\n[INFO] Initializing visualizers...")
    force_viz = TactileForceVisualizer()
    tactile_viewer = TactileImageViewer(
        window_name="Tactile RGB (Click to Save)",
    )
    stress_extractor = StressExtractor()
    
    # Initialize interpolated force visualizer (always enabled, like reference)
    interp_viz = InterpolatedForceVisualizer(
        grid_size=(25, 25),  # 30x30 regular grid (可以调整，如 (20,20), (40,40) 等)
        image_size=(512, 512),
        enable_window=True,  # Always enable window like reference
        noise_threshold_percentile=10.0  # Filter noise below 10th percentile
    )
    print("[INFO] InterpolatedForceVisualizer initialized (30x30 grid)")
    
    print("\n[INFO] Starting simulation...")
    print("[INFO] Windows:")
    print("  - Normal Pressure: Green circles (size = pressure)")
    print("  - Tangential Traction: Red circles (size = traction)")
    print("  - Normal Force (Interpolated): 30x30 grid with color-coded pressure")
    print("  - Tangential Force (Interpolated): 30x30 grid with direction arrows")
    print("  - Tactile RGB: Click to save images")
    print("\n[INFO] Press Ctrl+C to stop\n")
    
    step_count = 0
    
    try:
        while simulation_app.is_running():
            # Physics step
            env._pre_physics_step(None)
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.uipc_sim.update_render_meshes()
            env.sim.render()
            
            # Update goal-based IK commands
            positions, orientations = env.goal_prim_view.get_world_poses()
            env.ik_commands[:, :3] = positions - env.scene.env_origins
            env.ik_commands[:, 3:] = orientations
            
            # Update sensor
            env.gsmini.update(dt=env.physics_dt)
            env.scene.update(dt=env.physics_dt)
            
            # Visualization (every frame)
            if step_count % 1 == 0:
                # Extract stress data
                env._uipc_gelpad.update(0.0)
                stress_data = stress_extractor.get_object_surface_stress(env._uipc_gelpad)
                
                # Get gelpad positions
                gelpad_pos = env._uipc_gelpad.data.nodal_pos_w.cpu().numpy()
                
                # Visualize forces (point-based)
                force_viz.visualize(stress_data, gelpad_pos)
                
                # Interpolated force visualization (using sensor-local coordinates)
                if stress_data.triangle_indices is not None and len(stress_data.triangle_indices) > 0:
                    # Get triangle centers for interpolation
                    tri_indices = stress_data.triangle_indices
                    v0 = gelpad_pos[tri_indices[:, 0]]
                    v1 = gelpad_pos[tri_indices[:, 1]]
                    v2 = gelpad_pos[tri_indices[:, 2]]
                    tri_centers = (v0 + v1 + v2) / 3.0
                    
                    # Transform to sensor-local coordinates (fixed frame)
                    # Get sensor case transform
                    sensor_body_idx = env._robot.find_bodies(env.cfg.gelpad_attachment_cfg.body_name)[0][0]
                    sensor_pos_w = env._robot.data.body_link_pos_w[0, sensor_body_idx].cpu().numpy()
                    sensor_quat_w = env._robot.data.body_link_quat_w[0, sensor_body_idx].cpu().numpy()
                    
                    # Convert quaternion to rotation matrix (wxyz format from Isaac)
                    w, x, y, z = sensor_quat_w[0], sensor_quat_w[1], sensor_quat_w[2], sensor_quat_w[3]
                    # Rotation matrix from quaternion
                    R = np.array([
                        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
                        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
                        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
                    ])
                    
                    # Transform triangle centers to sensor-local frame
                    tri_centers_local = []
                    for pos in tri_centers:
                        pos_local = R.T @ (pos - sensor_pos_w)
                        tri_centers_local.append(pos_local)
                    tri_centers_local = np.array(tri_centers_local)
                    
                    # Project to 2D using local X-Y coordinates (Z is normal direction)
                    # Pass physical coordinates (in meters) to visualizer
                    # The visualizer will handle aspect ratio and scaling
                    x_min, x_max = tri_centers_local[:, 0].min(), tri_centers_local[:, 0].max()
                    y_min, y_max = tri_centers_local[:, 1].min(), tri_centers_local[:, 1].max()
                    
                    # Calculate physical dimensions for proper aspect ratio
                    x_range = max(x_max - x_min, 0.001)
                    y_range = max(y_max - y_min, 0.001)
                    
                    # Scale to pixel coordinates while preserving aspect ratio
                    # Use a base size and scale both dimensions proportionally
                    base_size = 512
                    margin = 50
                    available_size = base_size - 2 * margin
                    
                    # Determine scaling factor based on the larger dimension
                    scale = available_size / max(x_range, y_range)
                    
                    # Calculate actual image dimensions based on aspect ratio
                    img_w = int(margin * 2 + x_range * scale)
                    img_h = int(margin * 2 + y_range * scale)
                    
                    tri_positions_2d = []
                    for pos in tri_centers_local:
                        # Scale physical coordinates to pixels
                        px = margin + (pos[0] - x_min) * scale
                        py = margin + (y_max - pos[1]) * scale  # Flip Y for image coordinates
                        tri_positions_2d.append([px, py])
                    tri_positions_2d = np.array(tri_positions_2d)
                    
                    # Transform tangential traction to sensor-local frame
                    tangential_traction_local = []
                    for traction in stress_data.tangential_traction:
                        traction_local = R.T @ traction
                        # Only keep X-Y components (in-plane)
                        tangential_traction_local.append([traction_local[0], traction_local[1]])
                    tangential_traction_local = np.array(tangential_traction_local)
                    
                    # Get pressure magnitudes
                    pressure_magnitudes = np.abs(stress_data.normal_pressure)
                    
                    # Visualize interpolated forces (camera-independent)
                    interp_viz.visualize_from_values(
                        pressure_magnitudes,
                        tangential_traction_local,
                        tri_positions_2d,
                    )
                
                # Display tactile image
                if hasattr(env.gsmini, 'data') and hasattr(env.gsmini.data, 'output'):
                    tactile_data = env.gsmini.data.output.get("tactile_rgb")
                    if tactile_data is not None:
                        tactile_img = tactile_data[0].cpu().numpy()
                        tactile_viewer.show(tactile_img)
            
            step_count += 1
            
    except KeyboardInterrupt:
        print("\n[INFO] Simulation stopped by user")
    finally:
        force_viz.close()
        tactile_viewer.close()
        interp_viz.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
