"""
Contact Force Visualization Demo

This script extends single_uipc_attachment.py to demonstrate contact force
extraction and visualization. It shows:
1. Real-time contact force extraction from UIPC simulation
2. 3D force visualization in Isaac Sim scene (all force components)
3. 2D force visualization on tactile image window:
   - Tangential forces: Red arrows (direction and magnitude)
   - Normal forces: Green circles (size represents magnitude)
4. Interpolated force visualization on regular 20×20 grids:
   - Normal forces: Green circles on regular grid
   - Tangential forces: Red arrows on regular grid
   - Smooth interpolation from sparse contact points
5. Tet mesh structure visualization (cyan wireframe)
6. Deformation visualization with color-coded heatmap

Based on libuipc's contact system feature:
- Force from Newton's Law: f = -gradient / dt^2

Visualization Features:
- 3D Scene: All force arrows (yellow/green/red) + cyan tet mesh
- 2D Contact Forces: Red arrows (tangent) + Green circles (normal)
- 2D Interpolated: 20×20 regular grids with smooth force distribution
- Deformation heatmap: Blue (no deformation) -> Green -> Yellow -> Red (max)
"""

from __future__ import annotations

import argparse
import cv2
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Demo: Contact Force Visualization on GelSight Sensor"
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--sys", type=bool, default=True, help="Whether to track system utilization.")
parser.add_argument(
    "--debug_vis",
    default=True,
    action="store_true",
    help="Whether to render tactile images in the GUI",
)
parser.add_argument(
    "--force_scale",
    type=float,
    default=0.1,
    help="Scale factor for force arrow visualization (arrow_length = force_magnitude * scale)",
)
# Debug output control
parser.add_argument("--debug_ball", action="store_true", help="Enable ball velocity/position debug output")
parser.add_argument("--debug_attachment", action="store_true", help="Enable attachment debug output")
parser.add_argument("--debug_position", action="store_true", help="Enable position monitor output")
parser.add_argument("--debug_deformation", action="store_true", help="Enable deformation stats output")
parser.add_argument("--debug_force", action="store_true", help="Enable force extraction debug output")
parser.add_argument("--debug_tet", action="store_true", help="Enable tet mesh visualization debug output")
parser.add_argument("--debug_all", action="store_true", help="Enable all debug output")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import traceback

import carb
import pynvml
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.prims import XFormPrim

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

# Import Isaac Sim debug draw for 3D force visualization
# Note: debug_draw interface should be acquired after Isaac Sim is initialized
debug_draw = None


def get_debug_draw():
    """Lazy initialization of debug draw interface."""
    global debug_draw
    if debug_draw is None:
        try:
            from isaacsim.util.debug_draw import _debug_draw
            debug_draw = _debug_draw.acquire_debug_draw_interface()
            print("[INFO] debug_draw interface acquired successfully")
        except Exception as e:
            print(f"[Warning] debug_draw not available: {e}")
    return debug_draw

from tacex import GelSightSensor

from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_single_uipc import (
    FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_UIPC_CFG,
)
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg

from tacex_uipc import (
    UipcIsaacAttachments,
    UipcIsaacAttachmentsCfg,
    UipcObject,
    UipcObjectCfg,
    UipcRLEnv,
    UipcSimCfg,
)
from tacex_uipc.sim import ContactForceExtractor, ContactForceData
from tacex_uipc.utils import TetMeshCfg


def draw_3d_forces(force_data: ContactForceData, 
                   vertex_positions: np.ndarray, 
                   force_scale: float = 0.01):
    """
    Draw force arrows in 3D using Isaac Sim's debug draw.
    
    Args:
        force_data: Contact force data
        vertex_positions: 3D positions of vertices. Shape: (num_vertices, 3)
        force_scale: Scale factor for arrow length
    """
    dd = get_debug_draw()
    if dd is None:
        return
    
    # Clear previous drawings
    dd.clear_lines()
    dd.clear_points()
    
    if force_data.vertex_indices is None or len(force_data.vertex_indices) == 0:
        return
    
    # Print debug info only if debug_force is enabled
    if len(force_data.vertex_indices) > 0:
        # Debug output is controlled by caller
        pass
    
    # Colors (RGBA)
    total_color = (1.0, 1.0, 0.0, 1.0)    # Yellow
    normal_color = (0.0, 1.0, 0.0, 1.0)   # Green
    tangent_color = (1.0, 0.0, 0.0, 1.0)  # Red
    point_color = (1.0, 1.0, 1.0, 1.0)    # White
    
    # Draw contact points
    contact_positions = []
    for idx in force_data.vertex_indices:
        if idx < len(vertex_positions):
            contact_positions.append(vertex_positions[idx])
    
    if len(contact_positions) > 0:
        contact_positions = np.array(contact_positions)
        dd.draw_points(
            contact_positions.tolist(),
            [point_color] * len(contact_positions),
            [5] * len(contact_positions)
        )
    
    # Draw force arrows
    for i, (idx, force) in enumerate(zip(force_data.vertex_indices, force_data.contact_forces)):
        if idx >= len(vertex_positions):
            continue
        
        start_pos = vertex_positions[idx]
        
        # Total force arrow (yellow)
        end_pos = start_pos + force * force_scale
        dd.draw_lines(
            [start_pos.tolist()],
            [end_pos.tolist()],
            [total_color],
            [2]
        )
        
        # Normal force arrow (green)
        if force_data.normal_forces is not None and i < len(force_data.normal_forces):
            normal = force_data.normal_forces[i]
            normal_end = start_pos + normal * force_scale
            dd.draw_lines(
                [(start_pos + np.array([0.001, 0, 0])).tolist()],
                [normal_end.tolist()],
                [normal_color],
                [2]
            )
        
        # Tangential force arrow (red)
        if force_data.tangential_forces is not None and i < len(force_data.tangential_forces):
            tangent = force_data.tangential_forces[i]
            tangent_end = start_pos + tangent * force_scale
            dd.draw_lines(
                [(start_pos - np.array([0.001, 0, 0])).tolist()],
                [tangent_end.tolist()],
                [tangent_color],
                [2]
            )


def draw_tet_mesh(env: BallRollingEnv, debug: bool = False):
    """
    Draw tet mesh structure for debugging.
    
    Args:
        env: The environment containing gelpad data
        debug: Whether to print debug output
    """
    dd = get_debug_draw()
    if dd is None:
        return
    
    # Color for tet mesh (cyan)
    tet_mesh_color = (0.0, 1.0, 1.0, 1.0)  # Cyan
    
    try:
        # Get current gelpad vertex positions
        env._uipc_gelpad.update(0.0)
        gelpad_positions = env._uipc_gelpad.data.nodal_pos_w.cpu().numpy()
        
        # Get tet mesh topology from UIPC simulation
        tet_indices = _get_tet_indices_from_uipc(env, debug=debug)
        
        if tet_indices is None or len(tet_indices) == 0:
            if debug:
                print("[TET_VIS] No tet indices available")
            return
        
        # Draw tetrahedra edges
        if len(tet_indices) % 4 == 0:
            num_tets = len(tet_indices) // 4
            if debug:
                print(f"[TET_VIS] Drawing {num_tets} tetrahedra")
            
            # Each tetrahedron has 6 edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
            edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            
            drawn_count = 0
            for i in range(0, len(tet_indices), 4):
                v0, v1, v2, v3 = tet_indices[i:i+4]
                
                # Check if all indices are valid
                if max(v0, v1, v2, v3) >= len(gelpad_positions):
                    continue
                
                # Draw all 6 edges of the tetrahedron
                tet_verts = [v0, v1, v2, v3]
                for edge in edge_pairs:
                    idx1, idx2 = tet_verts[edge[0]], tet_verts[edge[1]]
                    
                    try:
                        dd.draw_lines(
                            [gelpad_positions[idx1].tolist()],
                            [gelpad_positions[idx2].tolist()],
                            [tet_mesh_color],
                            [3]  # Line width
                        )
                        drawn_count += 1
                    except Exception as e:
                        pass
            
            if debug:
                print(f"[TET_VIS] Drew {drawn_count} edges")
        
    except Exception as e:
        if debug:
            print(f"[TET_VIS] Error drawing tet mesh: {e}")
            import traceback
            traceback.print_exc()


def draw_ball_mesh(env: BallRollingEnv, debug: bool = False):
    """
    Draw object mesh structure for debugging (ball/sphere2/etc).
    
    Args:
        env: The environment containing object data
        debug: Whether to print debug output
    """
    dd = get_debug_draw()
    if dd is None:
        return
    
    # Color for object mesh (yellow/orange)
    ball_mesh_color = (1.0, 0.8, 0.0, 1.0)  # Yellow/Orange
    ball_point_color = (1.0, 0.5, 0.0, 1.0)  # Orange
    
    try:
        # Get current object vertex positions
        env.ball.update(0.0)
        ball_positions = env.ball.data.nodal_pos_w.cpu().numpy()
        
        if ball_positions is None or len(ball_positions) == 0:
            if debug:
                print("[OBJECT_VIS] No object positions available")
            return
        
        if debug:
            print(f"[OBJECT_VIS] Drawing object with {len(ball_positions)} vertices")
        
        # Draw ball vertices as points
        dd.draw_points(
            ball_positions.tolist(),
            [ball_point_color] * len(ball_positions),
            [8] * len(ball_positions)  # Point size
        )
        
        # Get tet mesh topology from UIPC simulation
        tet_indices = _get_tet_indices_from_uipc(env, obj=env.ball, debug=debug)
        
        if tet_indices is not None and len(tet_indices) > 0:
            # Draw tetrahedra edges
            if len(tet_indices) % 4 == 0:
                num_tets = len(tet_indices) // 4
                if debug:
                    print(f"[BALL_VIS] Drawing {num_tets} tetrahedra")
                
                # Each tetrahedron has 6 edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
                edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
                
                drawn_count = 0
                for i in range(0, len(tet_indices), 4):
                    v0, v1, v2, v3 = tet_indices[i:i+4]
                    
                    # Check if all indices are valid
                    if max(v0, v1, v2, v3) >= len(ball_positions):
                        continue
                    
                    # Draw all 6 edges of the tetrahedron
                    tet_verts = [v0, v1, v2, v3]
                    for edge in edge_pairs:
                        idx1, idx2 = tet_verts[edge[0]], tet_verts[edge[1]]
                        
                        try:
                            dd.draw_lines(
                                [ball_positions[idx1].tolist()],
                                [ball_positions[idx2].tolist()],
                                [ball_mesh_color],
                                [2]  # Line width
                            )
                            drawn_count += 1
                        except Exception as e:
                            pass
                
                if debug:
                    print(f"[BALL_VIS] Drew {drawn_count} edges")
        else:
            if debug:
                print("[BALL_VIS] No tet indices available, only drawing points")
        
    except Exception as e:
        if debug:
            print(f"[BALL_VIS] Error drawing ball mesh: {e}")
            import traceback
            traceback.print_exc()


def _get_tet_indices_from_uipc(env: BallRollingEnv, obj=None, debug: bool = False) -> np.ndarray:
    """
    Get tet mesh topology from UIPC simulation.
    
    Args:
        env: The environment containing UIPC simulation
        obj: The UIPC object to get tet indices from (default: gelpad)
        debug: Whether to print debug output
        
    Returns:
        Flattened array of tet indices, or None if not available
    """
    try:
        # Use gelpad by default
        if obj is None:
            obj = env._uipc_gelpad
        
        # Get geometry from UIPC simulation
        geom = env.uipc_sim.scene.geometries()
        geo_slot, geo_slot_rest = geom.find(obj.obj_id)
        
        if geo_slot is None:
            if debug:
                print(f"[TET_VIS] Cannot find geometry for obj_id={obj.obj_id}")
            return None
        
        geometry = geo_slot.geometry()
        
        # Try to get tetrahedra topology
        if hasattr(geometry, 'tetrahedra'):
            tets = geometry.tetrahedra()
            if tets is not None and hasattr(tets, 'topo'):
                tet_topo = tets.topo()
                
                # Convert to numpy array
                if hasattr(tet_topo, 'view'):
                    tet_view = tet_topo.view()
                    tet_array = np.array(tet_view)
                elif hasattr(tet_topo, 'to_numpy'):
                    tet_array = tet_topo.to_numpy()
                else:
                    tet_array = np.array(tet_topo)
                
                # Handle shape (n, 4, 1) -> (n, 4)
                if len(tet_array.shape) == 3 and tet_array.shape[2] == 1:
                    tet_array = tet_array.squeeze(axis=2)
                
                # Ensure correct shape (n_tets, 4)
                if len(tet_array.shape) == 2 and tet_array.shape[1] == 4:
                    tet_indices = tet_array.flatten()
                    if debug:
                        print(f"[TET_VIS] Got {len(tet_indices)} indices ({len(tet_indices)//4} tetrahedra)")
                    return tet_indices
                elif len(tet_array.shape) == 1 and len(tet_array) % 4 == 0:
                    if debug:
                        print(f"[TET_VIS] Got {len(tet_array)} indices ({len(tet_array)//4} tetrahedra)")
                    return tet_array
        
        if debug:
            print("[TET_VIS] No tetrahedra topology found")
        return None
        
    except Exception as e:
        if debug:
            print(f"[TET_VIS] Error getting tet indices: {e}")
            import traceback
            traceback.print_exc()
        return None


def draw_attachment_debug(env: BallRollingEnv):
    """
    Draw attachment points and constraints for debugging.
    
    Args:
        env: The environment containing attachment data
    """
    dd = get_debug_draw()
    if dd is None:
        return
    
    # Colors (RGBA)
    attachment_color = (1.0, 0.0, 0.0, 1.0)  # Red for attachment points
    target_color = (0.0, 0.0, 1.0, 1.0)      # Blue for target positions
    constraint_color = (0.0, 1.0, 1.0, 1.0)  # Cyan for constraint lines
    
    try:
        # Get current gelpad vertex positions
        env._uipc_gelpad.update(0.0)
        gelpad_positions = env._uipc_gelpad.data.nodal_pos_w.cpu().numpy()
        
        # Get attachment indices
        attachment_indices = env.attachment.attachment_points_idx
        if attachment_indices is None or len(attachment_indices) == 0:
            print("[DEBUG] No attachment indices available")
            return
        
        # Get attachment point positions on gelpad
        attachment_positions = []
        for idx in attachment_indices:
            if idx < len(gelpad_positions):
                attachment_positions.append(gelpad_positions[idx])
        
        if len(attachment_positions) == 0:
            print("[DEBUG] No valid attachment positions")
            return
        
        attachment_positions = np.array(attachment_positions)
        
        # Draw attachment points on gelpad (red spheres)
        dd.draw_points(
            attachment_positions.tolist(),
            [attachment_color] * len(attachment_positions),
            [8] * len(attachment_positions)  # Larger size for visibility
        )
        
        # Get target positions (aim positions) if available
        if hasattr(env.attachment, 'aim_positions') and env.attachment.aim_positions is not None:
            aim_positions = env.attachment.aim_positions
            if len(aim_positions) > 0:
                # Draw target positions (blue spheres)
                dd.draw_points(
                    aim_positions.tolist(),
                    [target_color] * len(aim_positions),
                    [6] * len(aim_positions)
                )
                
                # Draw constraint lines between attachment points and targets
                if len(attachment_positions) == len(aim_positions):
                    for att_pos, aim_pos in zip(attachment_positions, aim_positions):
                        dd.draw_lines(
                            [att_pos.tolist()],
                            [aim_pos.tolist()],
                            [constraint_color],
                            [1]
                        )
        
        print(f"[DEBUG] Drew {len(attachment_positions)} attachment points")
        
    except Exception as e:
        print(f"[DEBUG] Error drawing attachment debug: {e}")


def filter_contact_surface_vertices(all_positions: np.ndarray, 
                                     attachment_indices: list = None,
                                     z_threshold_percentile: float = 30.0) -> tuple:
    """
    Filter vertices to only include the bottom contact surface (opposite to attachment surface).
    
    Args:
        all_positions: All vertex positions (N, 3)
        attachment_indices: Indices of attachment points (top surface)
        z_threshold_percentile: Percentile threshold for Z coordinate to define bottom surface
        
    Returns:
        Tuple of (filtered_positions, filtered_indices, mask)
    """
    if all_positions is None or len(all_positions) == 0:
        return None, None, None
    
    # Method 1: Use Z coordinate threshold
    # The bottom surface (contact surface) has lower Z values
    z_coords = all_positions[:, 2]
    z_threshold = np.percentile(z_coords, z_threshold_percentile)
    
    # Create mask for bottom surface vertices
    bottom_surface_mask = z_coords <= z_threshold
    
    # Method 2: If attachment indices are provided, exclude them
    if attachment_indices is not None and len(attachment_indices) > 0:
        # Create a mask that excludes attachment points
        attachment_mask = np.ones(len(all_positions), dtype=bool)
        attachment_mask[attachment_indices] = False
        
        # Combine both masks: bottom surface AND not attachment points
        combined_mask = bottom_surface_mask & attachment_mask
    else:
        combined_mask = bottom_surface_mask
    
    # Get filtered positions and indices
    filtered_positions = all_positions[combined_mask]
    filtered_indices = np.where(combined_mask)[0]
    
    return filtered_positions, filtered_indices, combined_mask


def print_detailed_attachment_debug(env: BallRollingEnv, step: int):
    """Print detailed attachment debugging information."""
    print(f"\n[ATTACHMENT DEBUG - Step {step}]")
    print("=" * 50)
    
    try:
        # 1. Case information
        case_body_idx = env._robot.find_bodies("gelsight_mini_case")[0][0]
        case_pos = env._robot.data.body_link_pos_w[:, case_body_idx].cpu().numpy()
        case_quat = env._robot.data.body_link_quat_w[:, case_body_idx].cpu().numpy()
        
        print(f"1. CASE INFORMATION:")
        print(f"   Position: [{case_pos[0, 0]:.6f}, {case_pos[0, 1]:.6f}, {case_pos[0, 2]:.6f}]")
        print(f"   Quaternion: [{case_quat[0, 0]:.6f}, {case_quat[0, 1]:.6f}, {case_quat[0, 2]:.6f}, {case_quat[0, 3]:.6f}]")
        
        # 2. Gelpad information
        env._uipc_gelpad.update(0.0)
        gelpad_positions = env._uipc_gelpad.data.nodal_pos_w.cpu().numpy()
        gelpad_center = np.mean(gelpad_positions, axis=0)
        
        print(f"\n2. GELPAD INFORMATION:")
        print(f"   Total vertices: {len(gelpad_positions)}")
        print(f"   Center: [{gelpad_center[0]:.6f}, {gelpad_center[1]:.6f}, {gelpad_center[2]:.6f}]")
        print(f"   X range: [{gelpad_positions[:, 0].min():.6f}, {gelpad_positions[:, 0].max():.6f}]")
        print(f"   Y range: [{gelpad_positions[:, 1].min():.6f}, {gelpad_positions[:, 1].max():.6f}]")
        print(f"   Z range: [{gelpad_positions[:, 2].min():.6f}, {gelpad_positions[:, 2].max():.6f}]")
        
        # Calculate gelpad dimensions
        x_size = gelpad_positions[:, 0].max() - gelpad_positions[:, 0].min()
        y_size = gelpad_positions[:, 1].max() - gelpad_positions[:, 1].min()
        z_size = gelpad_positions[:, 2].max() - gelpad_positions[:, 2].min()
        print(f"   Dimensions: X={x_size*1000:.2f}mm, Y={y_size*1000:.2f}mm, Z={z_size*1000:.2f}mm")
        
        # 3. Attachment information
        attachment_indices = env.attachment.attachment_points_idx
        print(f"\n3. ATTACHMENT INFORMATION:")
        print(f"   Number of attachment points: {len(attachment_indices) if attachment_indices else 0}")
        
        if attachment_indices and len(attachment_indices) > 0:
            # Get attachment point positions
            attachment_positions = []
            for idx in attachment_indices:
                if idx < len(gelpad_positions):
                    attachment_positions.append(gelpad_positions[idx])
            
            if len(attachment_positions) > 0:
                attachment_positions = np.array(attachment_positions)
                att_center = np.mean(attachment_positions, axis=0)
                
                print(f"   Attachment indices: {attachment_indices[:5]}{'...' if len(attachment_indices) > 5 else ''}")
                print(f"   Attachment center: [{att_center[0]:.6f}, {att_center[1]:.6f}, {att_center[2]:.6f}]")
                print(f"   Attachment Z range: [{attachment_positions[:, 2].min():.6f}, {attachment_positions[:, 2].max():.6f}]")
                
                # 4. Aim positions (target positions)
                if hasattr(env.attachment, 'aim_positions') and env.attachment.aim_positions is not None:
                    aim_positions = env.attachment.aim_positions
                    if len(aim_positions) > 0:
                        aim_center = np.mean(aim_positions, axis=0)
                        print(f"\n4. AIM POSITIONS (TARGETS):")
                        print(f"   Number of aim positions: {len(aim_positions)}")
                        print(f"   Aim center: [{aim_center[0]:.6f}, {aim_center[1]:.6f}, {aim_center[2]:.6f}]")
                        print(f"   Aim Z range: [{aim_positions[:, 2].min():.6f}, {aim_positions[:, 2].max():.6f}]")
                        
                        # 5. Constraint analysis
                        print(f"\n5. CONSTRAINT ANALYSIS:")
                        if len(attachment_positions) == len(aim_positions):
                            distances = np.linalg.norm(attachment_positions - aim_positions, axis=1)
                            print(f"   Constraint distances (mm):")
                            print(f"     Mean: {np.mean(distances)*1000:.3f}")
                            print(f"     Max: {np.max(distances)*1000:.3f}")
                            print(f"     Min: {np.min(distances)*1000:.3f}")
                            print(f"     Std: {np.std(distances)*1000:.3f}")
                            
                            # Check if constraints are working
                            if np.max(distances) > 0.005:  # > 5mm
                                print(f"   ⚠️  WARNING: Large constraint violations! Max distance: {np.max(distances)*1000:.3f}mm")
                            elif np.max(distances) > 0.001:  # > 1mm
                                print(f"   ⚠️  CAUTION: Moderate constraint violations. Max distance: {np.max(distances)*1000:.3f}mm")
                            else:
                                print(f"   ✓ Constraints working well. Max distance: {np.max(distances)*1000:.3f}mm")
                        else:
                            print(f"   ❌ Mismatch: {len(attachment_positions)} attachment points vs {len(aim_positions)} aim positions")
                
                # 6. Relative positioning
                print(f"\n6. RELATIVE POSITIONING:")
                case_to_gelpad = gelpad_center - case_pos[0]
                case_to_attachment = att_center - case_pos[0]
                print(f"   Case to gelpad center: [{case_to_gelpad[0]*1000:.2f}, {case_to_gelpad[1]*1000:.2f}, {case_to_gelpad[2]*1000:.2f}] mm")
                print(f"   Case to attachment center: [{case_to_attachment[0]*1000:.2f}, {case_to_attachment[1]*1000:.2f}, {case_to_attachment[2]*1000:.2f}] mm")
                
                # Expected relative position (gelpad should be close to case)
                expected_z_offset = -0.005  # Gelpad should be ~5mm below case
                actual_z_offset = case_to_gelpad[2]
                print(f"   Expected Z offset: {expected_z_offset*1000:.2f}mm")
                print(f"   Actual Z offset: {actual_z_offset*1000:.2f}mm")
                print(f"   Z offset error: {(actual_z_offset - expected_z_offset)*1000:.2f}mm")
                
                if abs(actual_z_offset - expected_z_offset) > 0.01:  # > 10mm
                    print(f"   ❌ MAJOR ISSUE: Gelpad is not positioned correctly relative to case!")
                elif abs(actual_z_offset - expected_z_offset) > 0.005:  # > 5mm
                    print(f"   ⚠️  WARNING: Gelpad position may be incorrect")
                else:
                    print(f"   ✓ Gelpad position looks reasonable")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"[ERROR] Failed to print attachment debug: {e}")
        import traceback
        traceback.print_exc()


class DeformationVisualizer:
    """Visualizes tet mesh deformation."""
    
    def __init__(self,
                 image_size: tuple = (512, 512),
                 save_path: str = "/tmp/deformation_visualization.png"):
        """
        Initialize deformation visualizer.
        
        Args:
            image_size: Size of the visualization image (width, height)
            save_path: Path to save visualization images
        """
        self.image_size = image_size
        self.save_path = save_path
        
        # Try to create OpenCV window
        self.use_cv_window = False
        try:
            cv2.namedWindow("Contact Surface Deformation", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Contact Surface Deformation", 512, 512)
            self.use_cv_window = True
            print("[DeformationVisualizer] OpenCV window created")
        except cv2.error:
            print("[DeformationVisualizer] OpenCV window not available, saving to file instead.")
            print(f"[DeformationVisualizer] Images will be saved to: {save_path}")
        
        self.frame_count = 0
        self.max_deformation = 0.0
        
    def visualize_deformation(self,
                             current_positions: np.ndarray,
                             initial_positions: np.ndarray,
                             attachment_indices: list = None) -> np.ndarray:
        """
        Visualize tet mesh deformation as a heatmap.
        
        Visualization strategy:
        - X,Y displacement: Shown by point position on screen (points move)
        - Z displacement: Shown by color depth (darker = more Z displacement)
        
        Args:
            current_positions: Current vertex positions (N, 3)
            initial_positions: Initial vertex positions (N, 3)
            attachment_indices: Indices of attachment points
            
        Returns:
            Visualization image
        """
        # Calculate Z-direction deformation for each vertex (for color)
        z_deformation = current_positions[:, 2] - initial_positions[:, 2]
        
        # Update max Z deformation for normalization
        current_max_z = np.max(np.abs(z_deformation))
        if current_max_z > self.max_deformation:
            self.max_deformation = current_max_z
        
        # Create visualization image
        img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        
        # Project 3D positions to 2D using CURRENT positions (shows X,Y displacement)
        # Use current X and Y coordinates to show displacement in those directions
        x_coords = current_positions[:, 0]
        y_coords = current_positions[:, 1]
        
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        margin = 50
        w = self.image_size[0] - 2 * margin
        h = self.image_size[1] - 2 * margin
        
        # Normalize and map to image coordinates
        x_norm = (x_coords - x_min) / max(x_max - x_min, 0.001)
        y_norm = (y_coords - y_min) / max(y_max - y_min, 0.001)
        
        x_img = (margin + x_norm * w).astype(int)
        y_img = (margin + (1 - y_norm) * h).astype(int)  # Flip Y
        
        # Draw vertices with color based on Z deformation
        for i, (x, y, z_def) in enumerate(zip(x_img, y_img, z_deformation)):
            # Normalize Z deformation to [0, 1]
            if self.max_deformation > 0:
                norm_z_def = abs(z_def) / self.max_deformation
            else:
                norm_z_def = 0
            
            # Color map based on Z displacement:
            # Blue (no Z deformation) -> Green -> Yellow -> Red (max Z deformation)
            if norm_z_def < 0.5:
                # Blue to Green
                r = 0
                g = int(255 * norm_z_def * 2)
                b = int(255 * (1 - norm_z_def * 2))
            else:
                # Green to Red
                r = int(255 * (norm_z_def - 0.5) * 2)
                g = int(255 * (1 - (norm_z_def - 0.5) * 2))
                b = 0
            
            color = (b, g, r)  # BGR for OpenCV
            
            # Draw larger circles for attachment points
            if attachment_indices is not None and i in attachment_indices:
                cv2.circle(img, (x, y), 8, (255, 255, 255), 2)  # White outline
                cv2.circle(img, (x, y), 6, color, -1)
            else:
                cv2.circle(img, (x, y), 4, color, -1)
        
        # Add statistics overlay
        self._add_statistics(img, z_deformation, attachment_indices)
        
        # Add color bar legend
        self._add_colorbar(img)
        
        return img
    
    def _add_statistics(self, img: np.ndarray, z_deformation: np.ndarray, attachment_indices: list):
        """Add statistics overlay to the image."""
        # Stats box
        cv2.rectangle(img, (5, 5), (300, 140), (50, 50, 50), -1)
        cv2.rectangle(img, (5, 5), (300, 140), (255, 255, 255), 1)
        
        # Statistics
        mean_z_def = np.mean(z_deformation)
        max_z_def = np.max(z_deformation)
        min_z_def = np.min(z_deformation)
        
        y_offset = 25
        cv2.putText(img, "Contact Surface Deformation", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        cv2.putText(img, f"Vertices: {len(z_deformation)}", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.putText(img, f"Mean Z Def: {mean_z_def*1000:.3f} mm", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.putText(img, f"Max Z Def: {max_z_def*1000:.3f} mm", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.putText(img, f"Min Z Def: {min_z_def*1000:.3f} mm", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 20
        if attachment_indices is not None:
            cv2.putText(img, f"Attachment Points: {len(attachment_indices)}", (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _add_colorbar(self, img: np.ndarray):
        """Add color bar legend for Z displacement."""
        h, w = img.shape[:2]
        
        # Colorbar dimensions
        bar_width = 30
        bar_height = 200
        bar_x = w - bar_width - 20
        bar_y = h - bar_height - 20
        
        # Draw colorbar
        for i in range(bar_height):
            norm_val = i / bar_height
            
            # Same color mapping as vertices
            if norm_val < 0.5:
                r = 0
                g = int(255 * norm_val * 2)
                b = int(255 * (1 - norm_val * 2))
            else:
                r = int(255 * (norm_val - 0.5) * 2)
                g = int(255 * (1 - (norm_val - 0.5) * 2))
                b = 0
            
            cv2.rectangle(img, 
                         (bar_x, bar_y + bar_height - i),
                         (bar_x + bar_width, bar_y + bar_height - i + 1),
                         (b, g, r), -1)
        
        # Colorbar outline
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (255, 255, 255), 1)
        
        # Labels
        cv2.putText(img, "Z=0", (bar_x + bar_width + 5, bar_y + bar_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, f"{self.max_deformation*1000:.1f}mm", 
                   (bar_x + bar_width + 5, bar_y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def show(self, image: np.ndarray):
        """Display the visualization."""
        self.frame_count += 1
        if self.use_cv_window:
            cv2.imshow("Contact Surface Deformation", image)
            cv2.waitKey(1)
        else:
            # Save to file every 10 frames
            if self.frame_count % 10 == 0:
                cv2.imwrite(self.save_path, image)
    
    def close(self):
        """Close visualization window."""
        if self.use_cv_window:
            try:
                cv2.destroyWindow("Contact Surface Deformation")
            except:
                pass


class InterpolatedForceVisualizer:
    """Visualizes interpolated forces on a regular grid."""
    
    def __init__(self, 
                 grid_size: tuple = (100, 100),
                 image_size: tuple = (512, 512),
                 save_path_normal: str = "/tmp/force_normal_interpolated.png",
                 save_path_tangent: str = "/tmp/force_tangent_interpolated.png"):
        """
        Initialize interpolated force visualizer.
        
        Args:
            grid_size: Size of the regular grid (rows, cols)
            image_size: Size of the visualization image (width, height)
            save_path_normal: Path to save normal force visualization
            save_path_tangent: Path to save tangential force visualization
        """
        self.grid_size = grid_size
        self.image_size = image_size
        self.save_path_normal = save_path_normal
        self.save_path_tangent = save_path_tangent
        
        # Try to create OpenCV windows
        self.use_cv_window = False
        try:
            cv2.namedWindow("Normal Force (Interpolated)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Normal Force (Interpolated)", 512, 512)
            cv2.namedWindow("Tangential Force (Interpolated)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Tangential Force (Interpolated)", 512, 512)
            self.use_cv_window = True
            print("[InterpolatedForceVis] OpenCV windows created")
        except cv2.error:
            print("[InterpolatedForceVis] OpenCV windows not available, saving to file instead.")
        
        self.frame_count = 0
        
        # Create regular grid coordinates
        self.grid_x, self.grid_y = np.meshgrid(
            np.linspace(0, 1, grid_size[1]),
            np.linspace(0, 1, grid_size[0])
        )
    
    def interpolate_forces(self, 
                          force_data: ContactForceData,
                          vertex_positions_2d: np.ndarray = None) -> tuple:
        """
        Interpolate sparse force data to regular grid.
        
        Args:
            force_data: Contact force data
            vertex_positions_2d: 2D positions of vertices in normalized [0,1] space
            
        Returns:
            (normal_grid, tangent_x_grid, tangent_y_grid): Interpolated force grids
        """
        if force_data.vertex_indices is None or len(force_data.vertex_indices) == 0:
            # No contact - return zero grids
            return (np.zeros(self.grid_size), 
                    np.zeros(self.grid_size), 
                    np.zeros(self.grid_size))
        
        # Need at least 4 points for cubic interpolation
        if len(force_data.vertex_indices) < 4:
            # Silently return zero grids (no warning needed in normal operation)
            return (np.zeros(self.grid_size), 
                    np.zeros(self.grid_size), 
                    np.zeros(self.grid_size))
        
        # Get contact positions (normalized to [0, 1])
        if vertex_positions_2d is None:
            # Generate evenly distributed positions
            num_vertices = len(force_data.vertex_indices)
            vertex_positions_2d = self._generate_normalized_positions(num_vertices)
        else:
            # Normalize to [0, 1]
            vertex_positions_2d = vertex_positions_2d.copy()
            vertex_positions_2d[:, 0] /= self.image_size[0]
            vertex_positions_2d[:, 1] /= self.image_size[1]
        
        # Extract force magnitudes
        normal_magnitudes = []
        tangent_x = []
        tangent_y = []
        
        for i in range(len(force_data.vertex_indices)):
            # Normal force magnitude
            if force_data.normal_force_magnitudes is not None and i < len(force_data.normal_force_magnitudes):
                normal_magnitudes.append(force_data.normal_force_magnitudes[i])
            else:
                normal_magnitudes.append(0.0)
            
            # Tangential force components
            if force_data.tangential_forces is not None and i < len(force_data.tangential_forces):
                tangent = force_data.tangential_forces[i]
                # Use X and Y components (ignore Z for 2D visualization)
                tangent_x.append(tangent[0])
                tangent_y.append(tangent[1])
            else:
                tangent_x.append(0.0)
                tangent_y.append(0.0)
        
        normal_magnitudes = np.array(normal_magnitudes)
        tangent_x = np.array(tangent_x)
        tangent_y = np.array(tangent_y)
        
        # Interpolate to regular grid using scipy
        from scipy.interpolate import griddata
        
        points = vertex_positions_2d[:, :2]  # (N, 2)
        
        try:
            # Interpolate normal forces
            normal_grid = griddata(
                points, normal_magnitudes,
                (self.grid_x, self.grid_y),
                method='cubic',
                fill_value=0.0
            )
            
            # Interpolate tangential force components
            tangent_x_grid = griddata(
                points, tangent_x,
                (self.grid_x, self.grid_y),
                method='cubic',
                fill_value=0.0
            )
            
            tangent_y_grid = griddata(
                points, tangent_y,
                (self.grid_x, self.grid_y),
                method='cubic',
                fill_value=0.0
            )
        except Exception as e:
            print(f"[InterpolatedForceVis] Interpolation failed: {e}")
            print(f"[InterpolatedForceVis] Falling back to linear interpolation")
            try:
                # Try linear interpolation as fallback
                normal_grid = griddata(
                    points, normal_magnitudes,
                    (self.grid_x, self.grid_y),
                    method='linear',
                    fill_value=0.0
                )
                
                tangent_x_grid = griddata(
                    points, tangent_x,
                    (self.grid_x, self.grid_y),
                    method='linear',
                    fill_value=0.0
                )
                
                tangent_y_grid = griddata(
                    points, tangent_y,
                    (self.grid_x, self.grid_y),
                    method='linear',
                    fill_value=0.0
                )
            except Exception as e2:
                print(f"[InterpolatedForceVis] Linear interpolation also failed: {e2}")
                return (np.zeros(self.grid_size), 
                        np.zeros(self.grid_size), 
                        np.zeros(self.grid_size))
        
        # Clamp negative values to zero (forces can't be negative in magnitude)
        normal_grid = np.maximum(normal_grid, 0.0)
        
        return normal_grid, tangent_x_grid, tangent_y_grid
    
    def visualize(self, 
                  force_data: ContactForceData,
                  vertex_positions_2d: np.ndarray = None):
        """
        Create and display interpolated force visualizations.
        
        Args:
            force_data: Contact force data
            vertex_positions_2d: 2D positions of vertices
        """
        # Interpolate forces to grid
        normal_grid, tangent_x_grid, tangent_y_grid = self.interpolate_forces(
            force_data, vertex_positions_2d
        )
        
        # Create visualizations
        normal_img = self._visualize_normal_forces(normal_grid)
        tangent_img = self._visualize_tangent_forces(tangent_x_grid, tangent_y_grid)
        
        # Display or save
        self.frame_count += 1
        if self.use_cv_window:
            cv2.imshow("Normal Force (Interpolated)", normal_img)
            cv2.imshow("Tangential Force (Interpolated)", tangent_img)
            cv2.waitKey(1)
        else:
            # Save to file every 10 frames
            if self.frame_count % 10 == 0:
                cv2.imwrite(self.save_path_normal, normal_img)
                cv2.imwrite(self.save_path_tangent, tangent_img)
    
    def _visualize_normal_forces(self, normal_grid: np.ndarray) -> np.ndarray:
        """
        Visualize normal forces as circles on a grid.
        
        Args:
            normal_grid: Grid of normal force magnitudes (grid_size)
            
        Returns:
            Visualization image
        """
        img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        
        # Calculate cell size
        cell_w = self.image_size[0] / self.grid_size[1]
        cell_h = self.image_size[1] / self.grid_size[0]
        
        # Find max force for normalization
        max_force = np.max(normal_grid)
        if max_force < 0.0001:
            max_force = 1.0
        
        # Draw circles for each grid cell
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                force_mag = normal_grid[i, j]
                
                if force_mag < 0.0001:  # Skip very small forces
                    continue
                
                # Calculate circle center
                cx = int((j + 0.5) * cell_w)
                cy = int((i + 0.5) * cell_h)
                
                # Calculate circle radius based on force magnitude
                normalized_force = force_mag / max_force
                max_radius = min(cell_w, cell_h) * 0.4
                radius = int(normalized_force * max_radius)
                radius = max(2, radius)
                
                # Color based on force magnitude (green scale)
                intensity = int(255 * normalized_force)
                color = (0, intensity, 0)  # Green
                
                # Draw filled circle
                cv2.circle(img, (cx, cy), radius, color, -1)
                
                # Draw white outline for visibility
                cv2.circle(img, (cx, cy), radius, (255, 255, 255), 1)
        
        # Add statistics
        self._add_stats_normal(img, normal_grid, max_force)
        
        return img
    
    def _visualize_tangent_forces(self, tangent_x_grid: np.ndarray, 
                                   tangent_y_grid: np.ndarray) -> np.ndarray:
        """
        Visualize tangential forces as arrows on a grid.
        
        Args:
            tangent_x_grid: Grid of tangential force X components
            tangent_y_grid: Grid of tangential force Y components
            
        Returns:
            Visualization image
        """
        img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        
        # Calculate cell size
        cell_w = self.image_size[0] / self.grid_size[1]
        cell_h = self.image_size[1] / self.grid_size[0]
        
        # Calculate tangent force magnitudes
        tangent_mag = np.sqrt(tangent_x_grid**2 + tangent_y_grid**2)
        max_force = np.max(tangent_mag)
        if max_force < 0.0001:
            max_force = 1.0
        
        # Draw arrows for each grid cell
        arrow_scale = min(cell_w, cell_h) * 0.8
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                fx = tangent_x_grid[i, j]
                fy = tangent_y_grid[i, j]
                force_mag = tangent_mag[i, j]
                
                if force_mag < 0.0001:  # Skip very small forces
                    continue
                
                # Calculate arrow start position (cell center)
                cx = int((j + 0.5) * cell_w)
                cy = int((i + 0.5) * cell_h)
                
                # Calculate arrow end position
                normalized_force = force_mag / max_force
                dx = fx / force_mag * normalized_force * arrow_scale
                dy = -fy / force_mag * normalized_force * arrow_scale  # Flip Y
                
                ex = int(cx + dx)
                ey = int(cy + dy)
                
                # Color based on force magnitude (red scale)
                intensity = int(255 * normalized_force)
                color = (0, 0, intensity)  # Red
                
                # Draw arrow with thicker line
                if force_mag > 0.001:  # Only draw significant forces
                    cv2.arrowedLine(img, (cx, cy), (ex, ey), color, 2, tipLength=0.3)  # Thickness 2 -> 3
        
        # Add statistics
        self._add_stats_tangent(img, tangent_mag, max_force)
        
        return img
    
    def _add_stats_normal(self, img: np.ndarray, normal_grid: np.ndarray, max_force: float):
        """Add statistics overlay for normal forces."""
        cv2.rectangle(img, (5, 5), (250, 80), (50, 50, 50), -1)
        cv2.rectangle(img, (5, 5), (250, 80), (255, 255, 255), 1)
        
        mean_force = np.mean(normal_grid)
        nonzero_count = np.count_nonzero(normal_grid > 0.0001)
        
        cv2.putText(img, "Normal Force (Interpolated)", (15, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"Grid: {self.grid_size[0]}x{self.grid_size[1]}", (15, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, f"Max: {max_force:.3f} N", (15, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, f"Active cells: {nonzero_count}", (15, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _add_stats_tangent(self, img: np.ndarray, tangent_mag: np.ndarray, max_force: float):
        """Add statistics overlay for tangential forces."""
        cv2.rectangle(img, (5, 5), (250, 80), (50, 50, 50), -1)
        cv2.rectangle(img, (5, 5), (250, 80), (255, 255, 255), 1)
        
        mean_force = np.mean(tangent_mag)
        nonzero_count = np.count_nonzero(tangent_mag > 0.0001)
        
        cv2.putText(img, "Tangential Force (Interpolated)", (15, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"Grid: {self.grid_size[0]}x{self.grid_size[1]}", (15, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, f"Max: {max_force:.3f} N", (15, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, f"Active cells: {nonzero_count}", (15, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _generate_normalized_positions(self, num_vertices: int) -> np.ndarray:
        """Generate evenly distributed normalized positions."""
        positions = []
        cols = int(np.ceil(np.sqrt(num_vertices)))
        rows = int(np.ceil(num_vertices / cols))
        
        for i in range(num_vertices):
            row = i // cols
            col = i % cols
            x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            positions.append([x, y])
        
        return np.array(positions)
    
    def close(self):
        """Close visualization windows."""
        if self.use_cv_window:
            try:
                cv2.destroyWindow("Normal Force (Interpolated)")
                cv2.destroyWindow("Tangential Force (Interpolated)")
            except:
                pass


class ForceVisualizer:
    """Visualizes contact forces on tactile sensor images."""
    
    def __init__(self, 
                 image_size: tuple = (256, 256),
                 force_scale: float = 0.1,
                 normal_color: tuple = (0, 255, 0),    # Green for normal forces
                 tangent_color: tuple = (255, 0, 0),   # Red for tangential forces
                 total_color: tuple = (255, 255, 0),   # Yellow for total forces
                 save_path: str = "/tmp/force_visualization.png"):
        """
        Initialize force visualizer.
        
        Args:
            image_size: Size of the visualization image (width, height)
            force_scale: Scale factor for converting force to arrow length
            normal_color: BGR color for normal force arrows
            tangent_color: BGR color for tangential force arrows
            total_color: BGR color for total force arrows
            save_path: Path to save visualization images
        """
        self.image_size = image_size
        self.force_scale = force_scale
        self.normal_color = normal_color
        self.tangent_color = tangent_color
        self.total_color = total_color
        self.save_path = save_path
        
        # Try to create OpenCV window, fall back to file saving
        self.use_cv_window = False
        try:
            cv2.namedWindow("Contact Forces", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Contact Forces", 512, 512)
            self.use_cv_window = True
        except cv2.error:
            print("[ForceVisualizer] OpenCV window not available, saving to file instead.")
            print(f"[ForceVisualizer] Images will be saved to: {save_path}")
        
        # For statistics display
        self.force_history = []
        self.max_history = 100
        self.frame_count = 0
        
    def draw_force_arrows(self, 
                          base_image: np.ndarray,
                          force_data: ContactForceData,
                          vertex_positions_2d: np.ndarray = None) -> np.ndarray:
        """
        Draw force arrows on the tactile image.
        
        Visualization:
        - Tangential forces: Red arrows (direction and magnitude)
        - Normal forces: Green circles (size = magnitude)
        
        Args:
            base_image: Base tactile image (RGB or grayscale)
            force_data: Contact force data from extractor
            vertex_positions_2d: 2D positions of vertices in image space
            
        Returns:
            Image with force visualization drawn
        """
        # Convert to BGR for OpenCV if needed
        if len(base_image.shape) == 2:
            vis_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
        elif base_image.shape[2] == 3:
            vis_image = base_image.copy()
        else:
            vis_image = cv2.cvtColor(base_image, cv2.COLOR_RGBA2BGR)
        
        # Resize to visualization size
        vis_image = cv2.resize(vis_image, self.image_size)
        
        if force_data.vertex_indices is None or len(force_data.vertex_indices) == 0:
            # No contact - just show the image with "No Contact" text
            cv2.putText(vis_image, "No Contact", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return vis_image
        
        # If no 2D positions provided, distribute vertices evenly
        if vertex_positions_2d is None:
            num_vertices = len(force_data.vertex_indices)
            vertex_positions_2d = self._generate_vertex_positions(num_vertices)
        
        # Draw visualization for each contact point
        for i, (idx, force) in enumerate(zip(force_data.vertex_indices, force_data.contact_forces)):
            if i >= len(vertex_positions_2d):
                break
                
            pos = vertex_positions_2d[i]
            x, y = int(pos[0]), int(pos[1])
            
            # Skip if outside image
            if x < 0 or x >= self.image_size[0] or y < 0 or y >= self.image_size[1]:
                continue
            
            # Draw tangential force arrow (red) if available
            if force_data.tangential_forces is not None and i < len(force_data.tangential_forces):
                tangent = force_data.tangential_forces[i]
                self._draw_tangent_arrow(vis_image, x, y, tangent)
            
            # Draw normal force as circle (green) if available
            if force_data.normal_force_magnitudes is not None and i < len(force_data.normal_force_magnitudes):
                normal_mag = force_data.normal_force_magnitudes[i]
                self._draw_normal_circle(vis_image, x, y, normal_mag)
        
        # Add legend and statistics
        self._add_legend(vis_image, force_data)
        
        return vis_image
    
    def _draw_tangent_arrow(self, image: np.ndarray, x: int, y: int, tangent: np.ndarray):
        """Draw tangential force arrow (red)."""
        # Use x and y components for 2D visualization
        fx, fy, fz = tangent[0], tangent[1], tangent[2]
        
        # Force magnitude
        force_mag = np.sqrt(fx*fx + fy*fy + fz*fz)
        
        if force_mag < 0.0001:  # Skip very small forces
            return
        
        # Project 3D force to 2D
        scale = self.force_scale * self.image_size[0] * 1000
        dx = fx * scale
        dy = -fy * scale  # Negative because image Y is down
        
        # Add Z component to arrow length if significant
        if abs(fz) > max(abs(fx), abs(fy)):
            dy = -fz * scale * 0.5
        
        # Clamp arrow length
        max_length = 80
        min_length = 5
        length = np.sqrt(dx*dx + dy*dy)
        
        if length < min_length and force_mag > 0.0001:
            if length > 0:
                dx = dx * min_length / length
                dy = dy * min_length / length
            else:
                dy = -min_length
            length = min_length
        
        if length > max_length:
            dx = dx * max_length / length
            dy = dy * max_length / length
        
        # Draw arrow
        end_x = int(x + dx)
        end_y = int(y + dy)
        
        if length >= min_length:
            cv2.arrowedLine(image, (x, y), (end_x, end_y), self.tangent_color, 2, tipLength=0.3)
    
    def _draw_normal_circle(self, image: np.ndarray, x: int, y: int, normal_mag: float):
        """Draw normal force as a circle (green), size represents magnitude."""
        if normal_mag < 0.0001:  # Skip very small forces
            return
        
        # Calculate circle radius based on force magnitude
        # Scale factor to convert force to pixel radius
        radius_scale = 50.0  # Adjust this to change circle sizes
        radius = int(normal_mag * radius_scale)
        
        # Clamp radius
        min_radius = 3
        max_radius = 30
        radius = max(min_radius, min(max_radius, radius))
        
        # Draw filled circle
        cv2.circle(image, (x, y), radius, self.normal_color, -1)
        
        # Draw outline for better visibility
        cv2.circle(image, (x, y), radius, (255, 255, 255), 1)
    
    def _draw_arrow(self, image: np.ndarray, x: int, y: int, 
                    force: np.ndarray, color: tuple, label: str):
        """Draw a single force arrow."""
        # Use x and y components for 2D visualization
        # z component will be shown in arrow length
        fx, fy, fz = force[0], force[1], force[2]
        
        # Force magnitude
        force_mag = np.sqrt(fx*fx + fy*fy + fz*fz)
        
        # Project 3D force to 2D:
        # - Use X and Y for direction in image plane
        # - Use total magnitude for arrow length
        scale = self.force_scale * self.image_size[0] * 1000  # Increased scale for visibility
        dx = fx * scale
        dy = -fy * scale  # Negative because image Y is down
        
        # Add Z component to arrow length (Z is typically the pressing direction)
        # If force is mainly in Z, show as a vertical arrow
        if abs(fz) > max(abs(fx), abs(fy)):
            dy = -fz * scale * 0.5  # Show Z as Y direction
        
        # Clamp arrow length
        max_length = 80
        min_length = 5
        length = np.sqrt(dx*dx + dy*dy)
        
        # Ensure minimum visible length for any contact
        if length < min_length and force_mag > 0.0001:
            if length > 0:
                dx = dx * min_length / length
                dy = dy * min_length / length
            else:
                dy = -min_length  # Default downward arrow
            length = min_length
        
        if length > max_length:
            dx = dx * max_length / length
            dy = dy * max_length / length
        
        # Draw arrow
        end_x = int(x + dx)
        end_y = int(y + dy)
        
        # Always draw contact point, and arrow if force is significant
        cv2.circle(image, (x, y), 5, color, -1)  # Larger contact point marker
        if length >= min_length:
            cv2.arrowedLine(image, (x, y), (end_x, end_y), color, 3, tipLength=0.3)  # Thicker arrow
    
    def _generate_vertex_positions(self, num_vertices: int) -> np.ndarray:
        """Generate evenly distributed 2D positions for vertices."""
        positions = []
        cols = int(np.ceil(np.sqrt(num_vertices)))
        rows = int(np.ceil(num_vertices / cols))
        
        margin = 30
        w = self.image_size[0] - 2 * margin
        h = self.image_size[1] - 2 * margin
        
        for i in range(num_vertices):
            row = i // cols
            col = i % cols
            x = margin + col * w / max(cols - 1, 1)
            y = margin + row * h / max(rows - 1, 1)
            positions.append([x, y])
        
        return np.array(positions)
    
    def _add_legend(self, image: np.ndarray, force_data: ContactForceData):
        """Add legend and statistics to the image."""
        h, w = image.shape[:2]
        
        # Legend box
        cv2.rectangle(image, (5, h - 80), (200, h - 5), (50, 50, 50), -1)
        cv2.rectangle(image, (5, h - 80), (200, h - 5), (255, 255, 255), 1)
        
        # Legend items
        y_offset = h - 65
        
        # Tangential force arrow
        cv2.arrowedLine(image, (15, y_offset), (40, y_offset), self.tangent_color, 2)
        cv2.putText(image, "Tangent Force", (50, y_offset + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_offset += 20
        
        # Normal force circle
        cv2.circle(image, (27, y_offset), 8, self.normal_color, -1)
        cv2.circle(image, (27, y_offset), 8, (255, 255, 255), 1)
        cv2.putText(image, "Normal Force", (50, y_offset + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.putText(image, "(circle size = magnitude)", (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        # Statistics
        y_offset += 20
        num_contacts = len(force_data.vertex_indices) if force_data.vertex_indices is not None else 0
        cv2.putText(image, f"Contacts: {num_contacts}", (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Force magnitude stats
        if force_data.contact_forces is not None and len(force_data.contact_forces) > 0:
            max_force = np.max(np.linalg.norm(force_data.contact_forces, axis=1))
            total_force = np.sum(force_data.contact_forces, axis=0)
            
            # Stats box at top
            cv2.rectangle(image, (5, 5), (250, 75), (50, 50, 50), -1)
            cv2.rectangle(image, (5, 5), (250, 75), (255, 255, 255), 1)
            
            cv2.putText(image, f"Max Force: {max_force:.3f} N", (15, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, f"Total: [{total_force[0]:.2f}, {total_force[1]:.2f}, {total_force[2]:.2f}] N", 
                       (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            if force_data.normal_force_magnitudes is not None:
                max_normal = np.max(force_data.normal_force_magnitudes)
                max_tangent = np.max(force_data.tangential_force_magnitudes)
                cv2.putText(image, f"Max Normal: {max_normal:.2f}N, Tangent: {max_tangent:.2f}N", 
                           (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    def show(self, image: np.ndarray):
        """Display the visualization."""
        self.frame_count += 1
        if self.use_cv_window:
            cv2.imshow("Contact Forces", image)
            cv2.waitKey(1)
        else:
            # Save to file every 10 frames
            if self.frame_count % 10 == 0:
                cv2.imwrite(self.save_path, image)
    
    def close(self):
        """Close visualization window."""
        if self.use_cv_window:
            try:
                cv2.destroyWindow("Contact Forces")
            except:
                pass


def print_force_summary(force_data: ContactForceData, step: int, debug: bool = False):
    """Print force summary to console.
    
    Args:
        force_data: Contact force data
        step: Current step number
        debug: Whether to print (controlled by debug_force flag)
    """
    if not debug:
        return
    
    if force_data.vertex_indices is None or len(force_data.vertex_indices) == 0:
        return
    
    total_force = np.sum(force_data.contact_forces, axis=0)
    max_force = np.max(np.linalg.norm(force_data.contact_forces, axis=1))
    
    print(f"\n[Step {step}] Contact Force Summary:")
    print(f"  Vertices in contact: {len(force_data.vertex_indices)}")
    print(f"  Total Force: [{total_force[0]:.4f}, {total_force[1]:.4f}, {total_force[2]:.4f}] N")
    print(f"  Max Force Magnitude: {max_force:.4f} N")
    
    if force_data.normal_force_magnitudes is not None:
        print(f"  Max Normal Force: {np.max(force_data.normal_force_magnitudes):.4f} N")
        print(f"  Max Tangential Force: {np.max(force_data.tangential_force_magnitudes):.4f} N")


class CustomEnvWindow(BaseEnvWindow):
    """Window manager for the RL environment."""

    def __init__(self, env: DirectRLEnvCfg, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class BallRollingEnvCfg(DirectRLEnvCfg):
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1.9, 1.4, 0.3)
    viewer.lookat = (-1.5, -1.9, -1.1)

    debug_vis = True
    ui_window_class_type = CustomEnvWindow
    decimation = 1
    
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            enable_ccd=True,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=5.0,
            dynamic_friction=5.0,
            restitution=0.0,
        ),
    )

    uipc_sim = UipcSimCfg(
        ground_height=0.0025,
        contact=UipcSimCfg.Contact(
            d_hat=0.00005,  # Reduced from 0.0001 to 0.00005 for more precise contact
            default_contact_resistance=20.0,  # Increased from 10.0 to 20.0 GPa to limit deformation spread
            enable_friction=True,
            default_friction_ratio=0.5,
        ),
        gravity=(0.0, 0.0, -9.8),  # Explicitly set gravity
        dt=1/60,  # Explicit timestep (matches sim.dt)
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=1.5,
        replicate_physics=True,
        lazy_sensor_update=True,
    )

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    plate = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ground_plate",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/plate.usd",
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                kinematic_enabled=True,
            ),
        ),
    )

    # PhysX sphere2 for comparison (kinematic, fixed in place)
    physx_sphere2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/physx_sphere2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.03]),  # Same position as UIPC sphere2
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/tactile_test_shapes/sphere2.usd",
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                kinematic_enabled=True,  # Fixed in place like UIPC sphere2
                disable_gravity=True,
            ),
        ),
    )

    # sphere2 as UIPC object - uses precomputed tet mesh from USD file
    # Using AffineBody with kinematic=True to keep it fixed in place
    ball = UipcObjectCfg(
        prim_path="/World/envs/env_.*/sphere2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0.03]),  # 3cm above ground
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/tactile_test_shapes/sphere2.usd",
            # Disable PhysX collision for UIPC objects
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,  # Disable PhysX collision
            ),
        ),
        mesh_cfg=None,  # Use precomputed tet mesh from USD file (already generated)
        constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(
            m_kappa=100.0,      # Stiffness (hard rubber-like)
            kinematic=True,     # Fix the object in place (no movement)
        ),
        mass_density=1e3,  # kg/m^3 (wood density)
    )

    robot: ArticulationCfg = FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_UIPC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )
    
    # Gelpad uses precomputed tet mesh from USD file
    # Set mesh_cfg=None to load precomputed data instead of generating
    gelpad_cfg = UipcObjectCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_gelpad",
        mesh_cfg=None,  # Use precomputed tet mesh from USD
        constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
            youngs_modulus=0.01,  # Soft material (0.01 MPa) - rely on low Poisson ratio for localized deformation
            poisson_rate=0.30,    # Reduced to 0.30 to minimize deformation propagation (KEY PARAMETER!)
        ),
    )
    gelpad_attachment_cfg = UipcIsaacAttachmentsCfg(
        constraint_strength_ratio=50000.0,  # 🔥 INCREASED: 6000 -> 50000 for near-rigid top surface
        body_name="gelsight_mini_case", 
        debug_vis=False,
        compute_attachment_data=False,  # 使用预计算数据（稳定）
        constrain_entire_surface=True,  # 🔥 Constrain entire top surface
        surface_constraint_z_percentile=75.0,  # 🔥 INCREASED: 70 -> 75 (top 25% of vertices for larger rigid area)
    )
    
    # 🔥 NEW: Glass layer configuration (rigid body on top of gelpad)
    # This creates a truly rigid top surface that constrains the gelpad from above
    glass_layer_cfg = UipcObjectCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_gelpad/glass",
        mesh_cfg=None,  # Use precomputed tet mesh from USD
        constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(
            m_kappa=1e6,  # Very high stiffness (1000 MPa = 1 GPa) for rigid glass
            kinematic=False,  # Allow it to move with case (not fixed in space)
        ),
        mass_density=2500.0,  # Glass density (kg/m³)
    )
    
    # Glass layer attachment to case (very strong to keep it fixed)
    glass_attachment_cfg = UipcIsaacAttachmentsCfg(
        constraint_strength_ratio=100000.0,  # Very strong attachment to case
        body_name="gelsight_mini_case",
        debug_vis=False,
        compute_attachment_data=False,  # Use precomputed attachment data from USD
    )

    gsmini = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case",
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            update_period=0,
            resolution=(64, 64),  # Increased resolution for better visualization
            data_types=["depth"],
            clipping_range=(0.024, 0.034),
        ),
        device="cuda",
        debug_vis=True,
        marker_motion_sim_cfg=None,
        data_types=["tactile_rgb"],
    )
    gsmini.optical_sim_cfg = gsmini.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(64, 64),  # Increased for visualization
    )

    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    ball_radius = 0.005
    obj_pos_randomization_range = [-0.15, 0.15]
    episode_length_s = 0
    action_space = 0
    observation_space = 0
    state_space = 0


class BallRollingEnv(UipcRLEnv):
    cfg: BallRollingEnvCfg

    def __init__(self, cfg: BallRollingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )
        body_ids, body_names = self._robot.find_bodies("panda_hand")
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        self._jacobi_body_idx = self._body_idx - 1

        self._offset_pos = torch.tensor([0.0, 0.0, 0.131], device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        self.ik_commands = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        self.step_count = 0
        self.goal_prim_view = None
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        
        print("\n[DEBUG] Before clone_environments:")
        print(f"  Robot prim path: {self.cfg.robot.prim_path}")
        
        self.scene.clone_environments(copy_from_source=False)

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        ee_frame_cfg = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.131)),
                ),
            ],
        )

        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        self.gsmini = GelSightSensor(self.cfg.gsmini)
        self.scene.sensors["gsmini"] = self.gsmini

        # Create PhysX sphere2 for comparison
        self.physx_sphere2 = RigidObject(self.cfg.physx_sphere2)

        RigidObject(self.cfg.plate)

        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path, ground.spawn, translation=ground.init_state.pos, orientation=ground.init_state.rot
        )

        VisualCuboid(
            prim_path="/World/envs/env_0/goal",
            size=0.01,
            position=np.array([0.5, 0.0, 0.15]),
            orientation=np.array([0, 1, 0, 0]),
            color=np.array([255.0, 0.0, 0.0]),
            visible=False,  # Hide goal cube
        )

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # --- UIPC simulation setup ---
        # IMPORTANT: UIPC objects must be created AFTER scene.clone_environments()
        print("\n[DEBUG] Setting up UIPC objects:")
        self._uipc_gelpad = UipcObject(self.cfg.gelpad_cfg, self.uipc_sim)
        print(f"  Gelpad created: {self.cfg.gelpad_cfg.prim_path}")
        print(f"  Gelpad mesh_cfg: {self.cfg.gelpad_cfg.mesh_cfg}")
        print(f"  Fabric meshes count after gelpad: {len(self.uipc_sim._fabric_meshes)}")
        
        # 🔥 NEW: Create glass layer UIPC object (rigid body on top of gelpad)
        print(f"\n[DEBUG] Creating glass layer UIPC object:")
        print(f"  Glass prim path: {self.cfg.glass_layer_cfg.prim_path}")
        print(f"  Glass constitution: AffineBodyConstitution (rigid)")
        print(f"  Glass kappa: {self.cfg.glass_layer_cfg.constitution_cfg.m_kappa} MPa")
        
        self._uipc_glass = UipcObject(self.cfg.glass_layer_cfg, self.uipc_sim)
        print(f"  Glass layer created as UIPC rigid object")
        print(f"  Fabric meshes count after glass: {len(self.uipc_sim._fabric_meshes)}")

        # Ball as UIPC object (can interact with UIPC gelpad)
        print(f"\n[DEBUG] Creating sphere2 UIPC object:")
        print(f"  sphere2 prim path: {self.cfg.ball.prim_path}")
        print(f"  sphere2 mesh_cfg: {self.cfg.ball.mesh_cfg}")
        print(f"  sphere2 constitution: {type(self.cfg.ball.constitution_cfg).__name__}")
        
        self.ball = UipcObject(self.cfg.ball, self.uipc_sim)
        print(f"  sphere2 created as UIPC object")
        print(f"  sphere2 num_instances: {self.ball.num_instances}")
        print(f"  sphere2 vertex_count: {self.ball._vertex_count if hasattr(self.ball, '_vertex_count') else 'N/A'}")
        print(f"  sphere2 system_name: {self.ball._system_name if hasattr(self.ball, '_system_name') else 'N/A'}")
        print(f"  Fabric meshes count after sphere2: {len(self.uipc_sim._fabric_meshes)}")
        print(f"  Surface vertex offsets: {self.uipc_sim._surf_vertex_offsets}")
        
        # Note: obj_id will be set after _initialize_impl() is called during sim.reset()
        print(f"  Note: sphere2 will be initialized during sim.reset()")
        
        # Print fabric mesh info
        print(f"\n[DEBUG] Fabric mesh information:")
        for i, fabric_mesh in enumerate(self.uipc_sim._fabric_meshes):
            print(f"  Fabric mesh {i}: {fabric_mesh.GetPath()}")
            # Check visibility
            if fabric_mesh.HasAttribute("visibility"):
                vis = fabric_mesh.GetAttribute("visibility").Get()
                print(f"    Visibility: {vis}")
            else:
                print(f"    Visibility: No attribute")
            
            # Check if it has points
            if fabric_mesh.HasAttribute("points"):
                points = fabric_mesh.GetAttribute("points").Get()
                print(f"    Points count: {len(points) if points else 0}")
            else:
                print(f"    Points: No attribute")

        # create attachment
        print("\n[DEBUG] Creating attachment:")
        print(f"  Gelpad prim: {self.cfg.gelpad_cfg.prim_path}")
        print(f"  Robot prim: {self.scene.articulations['robot'].cfg.prim_path}")
        print(f"  Body name: {self.cfg.gelpad_attachment_cfg.body_name}")
        
        self.attachment = UipcIsaacAttachments(
            self.cfg.gelpad_attachment_cfg, self._uipc_gelpad, self.scene.articulations["robot"]
        )
        
        # 🔥 NEW: Create glass layer attachment to case
        print("\n[DEBUG] Creating glass layer attachment:")
        print(f"  Glass prim: {self.cfg.glass_layer_cfg.prim_path}")
        print(f"  Body name: {self.cfg.glass_attachment_cfg.body_name}")
        print(f"  Constraint strength: {self.cfg.glass_attachment_cfg.constraint_strength_ratio}")
        
        self.glass_attachment = UipcIsaacAttachments(
            self.cfg.glass_attachment_cfg, self._uipc_glass, self.scene.articulations["robot"]
        )
        print(f"  Glass attachment created successfully")
        print(f"  Attachment created with {len(self.attachment.attachment_points_idx)} attachment points")

    def _pre_physics_step(self, actions: torch.Tensor):
        self._ik_controller.set_command(self.ik_commands)

    def _apply_action(self):
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]

        if ee_pos_curr_b.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()
        self._robot.set_joint_position_target(joint_pos_des)
        self.step_count += 1

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def _get_rewards(self) -> torch.Tensor:
        pass

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    def _get_observations(self) -> dict:
        pass

    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :]

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w
        base_rot = self._robot.data.root_link_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
        )
        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self):
        jacobian = self.jacobian_b
        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])
        return jacobian


def run_simulator(env: BallRollingEnv, force_scale: float, debug_flags: dict = None):
    """Runs the simulation loop with force visualization.
    
    Args:
        env: The environment
        force_scale: Scale factor for force visualization
        debug_flags: Dictionary of debug flags (debug_ball, debug_attachment, etc.)
    """
    # Default debug flags
    if debug_flags is None:
        debug_flags = {
            'debug_ball': False,
            'debug_attachment': False,
            'debug_position': False,
            'debug_deformation': False,
            'debug_force': False,
            'debug_tet': False,
        }

    print(f"Starting simulation with {env.num_envs} envs")
    print("=" * 60)
    print("Contact Force & Deformation Visualization Demo")
    print("=" * 60)
    
    # Show enabled debug flags
    enabled_flags = [k for k, v in debug_flags.items() if v]
    if enabled_flags:
        print(f"\n[DEBUG] Enabled debug output: {', '.join(enabled_flags)}")
    else:
        print("\n[DEBUG] All debug output disabled (use --debug_all or specific --debug_* flags)")
    
    # Check if ball is properly initialized in UIPC scene
    print("\n[UIPC INITIALIZATION CHECK]")
    print(f"  Gelpad obj_id: {env._uipc_gelpad.obj_id if hasattr(env._uipc_gelpad, 'obj_id') else 'N/A'}")
    print(f"  sphere2 obj_id: {env.ball.obj_id if hasattr(env.ball, 'obj_id') else 'N/A'}")
    
    if hasattr(env.ball, 'obj_id'):
        try:
            geom = env.uipc_sim.scene.geometries()
            geo_slot, geo_slot_rest = geom.find(env.ball.obj_id)
            if geo_slot is not None:
                print(f"  ✓ sphere2 found in UIPC scene with obj_id={env.ball.obj_id}")
                geometry = geo_slot.geometry()
                print(f"  ✓ sphere2 geometry type: {type(geometry)}")
            else:
                print(f"  ❌ sphere2 NOT found in UIPC scene!")
        except Exception as e:
            print(f"  ❌ Error checking sphere2: {e}")
    else:
        print(f"  ❌ sphere2 obj_id not set - sphere2 not initialized!")
    
    print("")
    print("  TET MESH VISUALIZATION (3D Scene):")
    print("  - Cyan wireframe: Gelpad tetrahedra structure")
    print("  - Yellow/Orange wireframe: sphere2 tetrahedra structure")
    print("  - Orange points: sphere2 vertices")
    print("")
    print("  FORCE VISUALIZATION (3D Scene):")
    print("  - Yellow arrows: Total forces")
    print("  - Green arrows: Normal forces")
    print("  - Red arrows: Tangential forces")
    print("  - White points: Contact points")
    print("")
    print("  FORCE VISUALIZATION (2D Window - 'Contact Forces'):")
    print("  - Red arrows: Tangential forces (direction & magnitude)")
    print("  - Green circles: Normal forces (size = magnitude)")
    print("")
    print("  INTERPOLATED FORCE VISUALIZATION (2D Windows):")
    print("  - 'Normal Force (Interpolated)': 20x20 grid of green circles")
    print("  - 'Tangential Force (Interpolated)': 20x20 grid of red arrows")
    print("  - Smooth interpolation from sparse contact points")
    print("")
    print("  DEFORMATION VISUALIZATION (2D Window):")
    print("  - Point Position (X,Y): Shows X,Y displacement (points move on screen)")
    print("  - Color Depth: Shows Z displacement")
    print("    * Blue: No Z deformation")
    print("    * Green: Medium Z deformation")
    print("    * Yellow/Red: Maximum Z deformation")
    print("=" * 60)
    
    env.reset()

    env.goal_prim_view = XFormPrim(prim_paths_expr="/World/envs/env_.*/goal", name="goals", usd=True)

    # Enable debug visualization for tactile_rgb
    for prim in env.gsmini._prim_view.prims:
        if prim.HasAttribute("debug_tactile_rgb"):
            prim.GetAttribute("debug_tactile_rgb").Set(True)

    # Initialize force visualizer
    force_visualizer = ForceVisualizer(
        image_size=(512, 512),
        force_scale=force_scale
    )
    
    # Initialize deformation visualizer
    deformation_visualizer = DeformationVisualizer(
        image_size=(512, 512),
        save_path="/tmp/deformation_visualization.png"
    )
    
    # Initialize interpolated force visualizer
    interpolated_force_visualizer = InterpolatedForceVisualizer(
        grid_size=(20, 20),  # 20×20 grid for clearer arrows
        image_size=(512, 512),
        save_path_normal="/tmp/force_normal_interpolated.png",
        save_path_tangent="/tmp/force_tangent_interpolated.png"
    )
    
    # Get contact force extractor from UIPC simulation
    force_extractor = env.uipc_sim.contact_force_extractor
    if force_extractor is None:
        print("[ERROR] Force extractor not available!")
    else:
        print("[INFO] Force extractor ready")
    
    # Get initial gelpad positions for deformation calculation
    # We'll capture this after the first simulation step to ensure correct coordinates
    # Store initial positions and compute rigid transformation at each frame
    initial_gelpad_pos_w = None  # Initial positions in world frame
    initial_gelpad_center = None  # Initial center of mass
    initial_pos_captured = False
    
    # Get attachment indices
    attachment_indices = None
    if hasattr(env.attachment, 'attachment_points_idx'):
        attachment_indices = env.attachment.attachment_points_idx
        print(f"[INFO] Attachment indices loaded: {len(attachment_indices)} points")
    
    step_count = 0
    print_interval = 30  # Print force summary every N steps
    
    # Flag to check fabric mesh visibility once
    fabric_visibility_checked = False
    
    # Simulation loop
    try:
        while simulation_app.is_running():
            # Check and fix fabric mesh visibility on first step
            if not fabric_visibility_checked and step_count == 0:
                print("\n[FABRIC MESH VISIBILITY CHECK]")
                for i, fabric_mesh in enumerate(env.uipc_sim._fabric_meshes):
                    mesh_path = fabric_mesh.GetPath()
                    print(f"  Fabric mesh {i}: {mesh_path}")
                    
                    # Check visibility
                    if fabric_mesh.HasAttribute("visibility"):
                        vis = fabric_mesh.GetAttribute("visibility").Get()
                        print(f"    Current visibility: {vis}")
                        
                        # Force set to inherited if not visible
                        if vis != "inherited":
                            print(f"    ⚠️  Setting visibility to 'inherited'")
                            fabric_mesh.GetAttribute("visibility").Set("inherited")
                            # Force update to ensure change takes effect
                            fabric_mesh.GetAttribute("visibility").Set("inherited")
                    else:
                        print(f"    ⚠️  No visibility attribute, creating one")
                        vis_attr = fabric_mesh.CreateAttribute("visibility", "token")
                        vis_attr.Set("inherited")
                    
                    # Verify the change
                    if fabric_mesh.HasAttribute("visibility"):
                        final_vis = fabric_mesh.GetAttribute("visibility").Get()
                        print(f"    Final visibility: {final_vis}")
                    
                    # Check if it has points
                    if fabric_mesh.HasAttribute("points"):
                        points = fabric_mesh.GetAttribute("points").Get()
                        if points:
                            print(f"    ✓ Has {len(points)} points")
                        else:
                            print(f"    ⚠️  Points attribute exists but is empty")
                    else:
                        print(f"    ❌ No points attribute!")
                
                fabric_visibility_checked = True
                print("")
            
            # Perform physics step
            env._pre_physics_step(None)
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)

            env.uipc_sim.update_render_meshes()

            # Render scene
            env.sim.render()

            positions, orientations = env.goal_prim_view.get_world_poses()
            env.ik_commands[:, :3] = positions - env.scene.env_origins
            env.ik_commands[:, 3:] = orientations

            # Update sensor
            env.gsmini.update(dt=env.physics_dt)
            env.scene.update(dt=env.physics_dt)

            # === POSITION MONITORING ===
            if step_count % 30 == 0:
                # Force update gelpad data
                env._uipc_gelpad.update(0.0)
                
                # Print detailed attachment debug information
                if debug_flags.get('debug_attachment', False):
                    print_detailed_attachment_debug(env, step_count)
                
                # Monitor gelpad position
                if debug_flags.get('debug_position', False):
                    if hasattr(env._uipc_gelpad, 'data') and hasattr(env._uipc_gelpad.data, 'surf_nodal_pos_w'):
                        gelpad_pos = env._uipc_gelpad.data.surf_nodal_pos_w.cpu().numpy()
                        gelpad_center = np.mean(gelpad_pos, axis=0)
                        print(f"\n[Position Monitor - Step {step_count}]")
                        print(f"  Gelpad center: [{gelpad_center[0]:.4f}, {gelpad_center[1]:.4f}, {gelpad_center[2]:.4f}]")
                        print(f"  Gelpad Z range: [{gelpad_pos[:, 2].min():.4f}, {gelpad_pos[:, 2].max():.4f}]")
                    
                    # Monitor case position
                    case_body_idx = env._robot.find_bodies("gelsight_mini_case")[0][0]
                    case_pos = env._robot.data.body_link_pos_w[:, case_body_idx].cpu().numpy()
                    print(f"  Case position: [{case_pos[0, 0]:.4f}, {case_pos[0, 1]:.4f}, {case_pos[0, 2]:.4f}]")
                    
                    # Monitor sphere2 position (UIPC object)
                    if hasattr(env.ball, 'data') and hasattr(env.ball.data, 'nodal_pos_w'):
                        env.ball.update(0.0)  # Force update
                        ball_pos = env.ball.data.nodal_pos_w.cpu().numpy()
                        ball_center = np.mean(ball_pos, axis=0)
                        print(f"  sphere2 center: [{ball_center[0]:.4f}, {ball_center[1]:.4f}, {ball_center[2]:.4f}]")
                    
                    # Monitor goal position
                    if env.goal_prim_view is not None:
                        goal_positions, _ = env.goal_prim_view.get_world_poses()
                        print(f"  Goal position: [{goal_positions[0, 0]:.4f}, {goal_positions[0, 1]:.4f}, {goal_positions[0, 2]:.4f}]")
            
            # === DEFORMATION VISUALIZATION ===
            current_gelpad_pos = None
            try:
                # Force update gelpad data before accessing
                env._uipc_gelpad.update(0.0)
                
                # Get current gelpad vertex positions (world coordinates)
                current_gelpad_pos = env._uipc_gelpad.data.nodal_pos_w.cpu().numpy()
                
                # Capture initial position after gelpad has moved to correct position (step 60)
                if not initial_pos_captured and step_count == 60:
                    # Store initial positions in WORLD coordinates
                    initial_gelpad_pos_w = current_gelpad_pos.copy()
                    initial_gelpad_center = np.mean(initial_gelpad_pos_w, axis=0)
                    initial_pos_captured = True
                    print(f"[INFO] Initial gelpad positions captured at step {step_count}")
                    print(f"[INFO] Initial gelpad vertices: {initial_gelpad_pos_w.shape}")
                    print(f"[INFO] Initial gelpad center: {initial_gelpad_center}")
                    print(f"[INFO] Position range: X[{initial_gelpad_pos_w[:, 0].min():.4f}, {initial_gelpad_pos_w[:, 0].max():.4f}], "
                          f"Y[{initial_gelpad_pos_w[:, 1].min():.4f}, {initial_gelpad_pos_w[:, 1].max():.4f}], "
                          f"Z[{initial_gelpad_pos_w[:, 2].min():.4f}, {initial_gelpad_pos_w[:, 2].max():.4f}]")
                    
                    # Show contact surface filtering info
                    contact_surf_pos, contact_surf_idx, contact_surf_mask = filter_contact_surface_vertices(
                        initial_gelpad_pos_w,
                        attachment_indices,
                        z_threshold_percentile=30.0,
                        reference_positions=initial_gelpad_pos_w
                    )
                    if contact_surf_pos is not None:
                        print(f"[INFO] Contact surface vertices: {len(contact_surf_pos)} / {len(initial_gelpad_pos_w)} ({len(contact_surf_pos)/len(initial_gelpad_pos_w)*100:.1f}%)")
                        print(f"[INFO] Contact surface Z range: [{contact_surf_pos[:, 2].min():.4f}, {contact_surf_pos[:, 2].max():.4f}]")
                        if attachment_indices is not None:
                            print(f"[INFO] Attachment points excluded: {len(attachment_indices)}")
                
                if initial_gelpad_pos_w is not None and current_gelpad_pos is not None:
                    if current_gelpad_pos.shape == initial_gelpad_pos_w.shape:
                        # Compute current center of mass
                        current_gelpad_center = np.mean(current_gelpad_pos, axis=0)
                        
                        # Compute rigid translation (center of mass movement)
                        translation = current_gelpad_center - initial_gelpad_center
                        
                        # Remove rigid translation to get deformation-only positions
                        # This gives us positions as if the gelpad center stayed at the initial location
                        current_gelpad_pos_deform_only = current_gelpad_pos - translation
                        
                        # Now we could also remove rotation, but for small deformations, translation removal is usually sufficient
                        # If needed, we can add rotation removal using SVD/Procrustes alignment
                        
                        # Debug: Print transformation info every 60 steps
                        if step_count % 60 == 0 and debug_flags.get('debug_deformation', False):
                            print(f"\n[DEBUG] Rigid transformation removal at step {step_count}:")
                            print(f"  Initial center: {initial_gelpad_center}")
                            print(f"  Current center: {current_gelpad_center}")
                            print(f"  Translation: {translation}")
                            deformation_test = np.mean(np.linalg.norm(current_gelpad_pos_deform_only - initial_gelpad_pos_w, axis=1))
                            print(f"  Mean deformation (after removing translation): {deformation_test*1000:.3f} mm")
                        
                        # Filter to only show contact surface (bottom surface)
                        # Use initial positions as reference to keep tracking the same vertices
                        current_contact_surface, contact_indices, contact_mask = filter_contact_surface_vertices(
                            current_gelpad_pos_deform_only,  # Current positions with rigid motion removed
                            attachment_indices,
                            z_threshold_percentile=30.0,  # Bottom 30% of vertices by Z coordinate
                            reference_positions=initial_gelpad_pos_w  # Use initial positions to determine which vertices
                        )
                        
                        if current_contact_surface is not None and len(current_contact_surface) > 0:
                            # Get corresponding initial positions
                            initial_contact_surface = initial_gelpad_pos_w[contact_mask]
                            
                            # Visualize deformation (only contact surface, with rigid motion removed)
                            # Now the deformation is relative to the mesh's initial shape, not affected by rigid motion!
                            deformation_img = deformation_visualizer.visualize_deformation(
                                current_contact_surface,
                                initial_contact_surface,
                                attachment_indices=None  # No attachment points in filtered data
                            )
                            deformation_visualizer.show(deformation_img)
                            
                            # Print deformation stats every N steps
                            if debug_flags.get('debug_deformation', False) and step_count % print_interval == 0:
                                deformation = np.linalg.norm(current_contact_surface - initial_contact_surface, axis=1)
                                print(f"\n[Deformation Stats - Step {step_count}] (Contact Surface Only, Rigid Motion Removed)")
                                print(f"  Contact surface vertices: {len(current_contact_surface)} / {len(current_gelpad_pos)} (FIXED)")
                                print(f"  Mean deformation: {np.mean(deformation)*1000:.3f} mm")
                                print(f"  Max deformation: {np.max(deformation)*1000:.3f} mm")
                                print(f"  Min deformation: {np.min(deformation)*1000:.3f} mm")
                            
                            # Check attachment point deformation
                            if attachment_indices is not None and len(attachment_indices) > 0:
                                # Filter valid indices
                                valid_indices = [idx for idx in attachment_indices if idx < len(deformation)]
                                if len(valid_indices) > 0:
                                    attachment_def = deformation[valid_indices]
                                    print(f"  Attachment points mean def: {np.mean(attachment_def)*1000:.3f} mm")
                                    print(f"  Attachment points max def: {np.max(attachment_def)*1000:.3f} mm")
                                    if np.max(attachment_def) > 0.001:  # > 1mm
                                        print(f"  ⚠️  WARNING: Large attachment point deformation!")
                    else:
                        if step_count == 0:
                            print(f"[Warning] Shape still mismatch after retry: current={current_gelpad_pos.shape}, initial={initial_gelpad_pos.shape}")
            except Exception as e:
                if step_count == 0:
                    print(f"[Warning] Could not visualize deformation: {e}")
            
            # === BALL VELOCITY MONITORING (DEBUG) ===
            if debug_flags.get('debug_ball', False) and step_count % print_interval == 0:
                try:
                    env.ball.update(0.0)
                    ball_pos = env.ball.data.nodal_pos_w.cpu().numpy()
                    ball_center = np.mean(ball_pos, axis=0)
                    
                    # Check if sphere2 has velocity data
                    if hasattr(env.ball.data, 'nodal_vel_w'):
                        ball_vel = env.ball.data.nodal_vel_w.cpu().numpy()
                        ball_vel_mean = np.mean(np.linalg.norm(ball_vel, axis=1))
                        ball_vel_max = np.max(np.linalg.norm(ball_vel, axis=1))
                        print(f"\n[sphere2 Debug - Step {step_count}]")
                        print(f"  sphere2 center: [{ball_center[0]:.4f}, {ball_center[1]:.4f}, {ball_center[2]:.4f}]")
                        print(f"  sphere2 mean velocity: {ball_vel_mean:.6f} m/s")
                        print(f"  sphere2 max velocity: {ball_vel_max:.6f} m/s")
                        
                        # Check if sphere2 is stuck
                        if step_count > 60 and ball_vel_max < 0.001:
                            print(f"  ⚠️  WARNING: sphere2 appears to be stuck (no velocity)")
                    else:
                        print(f"\n[sphere2 Debug - Step {step_count}]")
                        print(f"  sphere2 center: [{ball_center[0]:.4f}, {ball_center[1]:.4f}, {ball_center[2]:.4f}]")
                        print(f"  ⚠️  WARNING: sphere2 has no velocity data")
                except Exception as e:
                    if step_count == 0:
                        print(f"[sphere2 Debug] Error: {e}")
            
            # === SENSOR DEPTH CHECK (Always on for first 5 steps to verify visibility fix) ===
            if step_count < 5 or (step_count % 100 == 0):
                try:
                    # Access depth data from sensor output dictionary
                    if "camera_depth" in env.gsmini.data.output and env.gsmini.data.output["camera_depth"] is not None:
                        depth_data = env.gsmini.data.output["camera_depth"]
                        depth_np = depth_data.cpu().numpy() if hasattr(depth_data, 'cpu') else depth_data
                        non_zero = (depth_np > 0).sum()
                        total = depth_np.size
                        
                        if step_count < 5:
                            print(f"\n[Sensor Depth Check - Step {step_count}]")
                            print(f"  Depth shape: {depth_np.shape}")
                            print(f"  Depth range: [{depth_np.min():.4f}, {depth_np.max():.4f}]")
                            print(f"  Non-zero pixels: {non_zero} / {total} ({100*non_zero/total:.1f}%)")
                            
                            if non_zero == 0:
                                print(f"  ⚠️  WARNING: Sensor cannot see any objects!")
                                print(f"  This may be normal if sphere2 hasn't reached sensor yet.")
                            else:
                                print(f"  ✓ Sensor is detecting objects")
                except Exception as e:
                    if step_count == 0:
                        print(f"[Sensor Depth Check] Error: {e}")
            
            # === FORCE EXTRACTION AND VISUALIZATION ===
            
            # Extract contact forces for the gelpad
            gelpad_forces = force_extractor.get_object_forces(env._uipc_gelpad)
            
            # Print force info every N steps
            if debug_flags.get('debug_force', False) and step_count % print_interval == 0:
                if gelpad_forces.vertex_indices is not None and len(gelpad_forces.vertex_indices) > 0:
                    print(f"\n[Step {step_count}] === Contact Force Info ===")
                    print(f"  Number of contact points: {len(gelpad_forces.vertex_indices)}")
                    if gelpad_forces.contact_forces is not None:
                        total_force = np.sum(gelpad_forces.contact_forces, axis=0)
                        max_force_mag = np.max(np.linalg.norm(gelpad_forces.contact_forces, axis=1))
                        print(f"  Total force: [{total_force[0]:.4f}, {total_force[1]:.4f}, {total_force[2]:.4f}] N")
                        print(f"  Max force magnitude: {max_force_mag:.4f} N")
                    if gelpad_forces.normal_force_magnitudes is not None:
                        print(f"  Max normal force: {np.max(gelpad_forces.normal_force_magnitudes):.4f} N")
                    if gelpad_forces.tangential_force_magnitudes is not None:
                        print(f"  Max tangential force: {np.max(gelpad_forces.tangential_force_magnitudes):.4f} N")
                else:
                    print(f"[Step {step_count}] No contact detected")
            
            # Get gelpad vertex positions for 3D visualization
            gelpad_vertex_pos = current_gelpad_pos if current_gelpad_pos is not None else None
            if gelpad_vertex_pos is None:
                try:
                    env._uipc_gelpad.update(0.0)  # Force update
                    gelpad_vertex_pos = env._uipc_gelpad.data.surf_nodal_pos_w.cpu().numpy()
                except:
                    pass
            
            if gelpad_vertex_pos is not None:
                # Draw 3D force arrows in the scene
                draw_3d_forces(gelpad_forces, gelpad_vertex_pos, force_scale=force_scale * 0.1)
                
                # Draw tet mesh structure (cyan wireframe)
                draw_tet_mesh(env, debug=debug_flags.get('debug_tet', False))
                
                # Draw ball mesh structure (yellow/orange wireframe)
                draw_ball_mesh(env, debug=debug_flags.get('debug_tet', False))
                
                # Draw attachment debug visualization (optional - comment out if not needed)
                # draw_attachment_debug(env)
            
            # Get tactile image for visualization base
            tactile_output = env.gsmini.data.output
            if tactile_output is not None and "tactile_rgb" in tactile_output:
                tactile_image = tactile_output["tactile_rgb"]
                if tactile_image is not None:
                    # Convert to numpy if tensor
                    if hasattr(tactile_image, 'cpu'):
                        tactile_image = tactile_image.cpu().numpy()
                    
                    # Handle batch dimension
                    if len(tactile_image.shape) == 4:
                        tactile_image = tactile_image[0]  # Take first env
                    
                    # Ensure uint8
                    if tactile_image.dtype != np.uint8:
                        tactile_image = (tactile_image * 255).astype(np.uint8)
                    
                    # Project 3D vertex positions to 2D image space
                    vertex_positions_2d = None
                    if gelpad_forces.vertex_indices is not None and len(gelpad_forces.vertex_indices) > 0 and gelpad_vertex_pos is not None:
                        try:
                            # Filter gelpad vertices to only show contact surface (bottom surface)
                            # Use initial positions as reference to keep tracking the same vertices
                            contact_surface_pos, contact_surface_indices, contact_surface_mask = filter_contact_surface_vertices(
                                gelpad_vertex_pos,  # Current positions
                                attachment_indices,
                                z_threshold_percentile=30.0,
                                reference_positions=initial_gelpad_pos  # Use initial positions as reference
                            )
                            
                            if contact_surface_pos is not None and len(contact_surface_pos) > 0:
                                # Use contact surface for projection reference
                                x_min, x_max = contact_surface_pos[:, 0].min(), contact_surface_pos[:, 0].max()
                                y_min, y_max = contact_surface_pos[:, 1].min(), contact_surface_pos[:, 1].max()
                                
                                # Get contact vertex 3D positions
                                contact_3d = gelpad_vertex_pos[gelpad_forces.vertex_indices]
                                
                                img_h, img_w = 512, 512  # visualizer image size
                                margin = 50
                                
                                vertex_positions_2d = []
                                for pos in contact_3d:
                                    # Normalize to [0, 1] based on contact surface range
                                    nx = (pos[0] - x_min) / max(x_max - x_min, 0.001)
                                    ny = (pos[1] - y_min) / max(y_max - y_min, 0.001)
                                    
                                    # Map to image coordinates
                                    px = margin + nx * (img_w - 2 * margin)
                                    py = margin + (1 - ny) * (img_h - 2 * margin)  # Flip Y
                                    vertex_positions_2d.append([px, py])
                                
                                vertex_positions_2d = np.array(vertex_positions_2d)
                                
                                if step_count % print_interval == 0 and len(vertex_positions_2d) > 0:
                                    print(f"  [2D Vis] Projected {len(vertex_positions_2d)} contact points to image")
                                    print(f"  [2D Vis] Using {len(contact_surface_pos)} contact surface vertices for reference")
                        except Exception as e:
                            if step_count == 0:
                                print(f"[Warning] Could not project 3D to 2D: {e}")
                    
                    # Draw force arrows on tactile image
                    vis_image = force_visualizer.draw_force_arrows(tactile_image, gelpad_forces, vertex_positions_2d)
                    
                    # Display
                    force_visualizer.show(vis_image)
                    
                    # Also create interpolated force visualization
                    interpolated_force_visualizer.visualize(gelpad_forces, vertex_positions_2d)
            else:
                # Create blank image for force visualization
                blank_image = np.zeros((64, 64, 3), dtype=np.uint8) + 50
                vis_image = force_visualizer.draw_force_arrows(blank_image, gelpad_forces)
                force_visualizer.show(vis_image)
            
            # Print force summary periodically (only if debug_force is enabled)
            if step_count % print_interval == 0:
                print_force_summary(gelpad_forces, step_count, debug=debug_flags.get('debug_force', False))
            
            step_count += 1

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        force_visualizer.close()
        deformation_visualizer.close()
        interpolated_force_visualizer.close()
        env.close()
        try:
            pynvml.nvmlShutdown()
        except:
            pass  # Ignore if already shutdown or not initialized


def main():
    """Main function."""
    env_cfg = BallRollingEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.gsmini.debug_vis = args_cli.debug_vis

    # Prepare debug flags
    debug_flags = {
        'debug_ball': args_cli.debug_ball or args_cli.debug_all,
        'debug_attachment': args_cli.debug_attachment or args_cli.debug_all,
        'debug_position': args_cli.debug_position or args_cli.debug_all,
        'debug_deformation': args_cli.debug_deformation or args_cli.debug_all,
        'debug_force': args_cli.debug_force or args_cli.debug_all,
        'debug_tet': args_cli.debug_tet or args_cli.debug_all,
    }

    experiment = BallRollingEnv(env_cfg)

    print("[INFO]: Setup complete...")
    run_simulator(env=experiment, force_scale=args_cli.force_scale, debug_flags=debug_flags)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        simulation_app.close()
