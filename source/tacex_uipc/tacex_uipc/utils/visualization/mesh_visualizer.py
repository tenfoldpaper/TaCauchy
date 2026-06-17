"""
3D mesh visualization utilities for UIPC simulation.

This module provides functions to visualize:
- Contact forces in 3D scene
- Tet mesh structure
- Object meshes
- Attachment constraints
"""

import numpy as np
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tacex_uipc.sim import ContactForceData

# Global debug draw interface (lazy initialization)
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


def draw_3d_forces(
    force_data: "ContactForceData", 
    vertex_positions: np.ndarray, 
    force_scale: float = 0.01
):
    """
    Draw force arrows in 3D using Isaac Sim's debug draw.
    
    Args:
        force_data: Contact force data from ContactForceExtractor
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
                [5]  # Increased from 2 to 5 for better visibility
            )


def draw_tet_mesh(env, debug: bool = False):
    """
    Draw tet mesh structure for debugging.
    
    Args:
        env: The environment containing gelpad data (must have _uipc_gelpad and uipc_sim)
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


def draw_ball_mesh(env, debug: bool = False):
    """
    Draw object mesh structure for debugging (ball/sphere/cone/etc).
    
    Args:
        env: The environment containing object data (must have .ball attribute)
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
                
                # Each tetrahedron has 6 edges
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


def draw_glass_mesh(env, debug: bool = False):
    """
    Draw glass layer tet mesh structure for debugging.
    
    Args:
        env: The environment containing glass layer data (must have .glass_layer)
        debug: Whether to print debug output
    """
    if not hasattr(env, "glass_layer") or env.glass_layer is None:
        return
    
    dd = get_debug_draw()
    if dd is None:
        return
    
    glass_mesh_color = (0.4, 0.8, 1.0, 1.0)  # Light blue
    
    try:
        env.glass_layer.update(0.0)
        glass_positions = env.glass_layer.data.nodal_pos_w.cpu().numpy()
        
        if glass_positions is None or len(glass_positions) == 0:
            if debug:
                print("[GLASS_VIS] No glass positions available")
            return
        
        tet_indices = _get_tet_indices_from_uipc(env, obj=env.glass_layer, debug=debug)
        if tet_indices is None or len(tet_indices) == 0:
            if debug:
                print("[GLASS_VIS] No tet indices available")
            return
        
        if len(tet_indices) % 4 == 0:
            num_tets = len(tet_indices) // 4
            if debug:
                print(f"[GLASS_VIS] Drawing {num_tets} tetrahedra")
            
            edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            drawn_count = 0
            for i in range(0, len(tet_indices), 4):
                v0, v1, v2, v3 = tet_indices[i:i+4]
                if max(v0, v1, v2, v3) >= len(glass_positions):
                    continue
                tet_verts = [v0, v1, v2, v3]
                for edge in edge_pairs:
                    idx1, idx2 = tet_verts[edge[0]], tet_verts[edge[1]]
                    dd.draw_lines(
                        [glass_positions[idx1].tolist()],
                        [glass_positions[idx2].tolist()],
                        [glass_mesh_color],
                        [2]
                    )
                    drawn_count += 1
            
            if debug:
                print(f"[GLASS_VIS] Drew {drawn_count} edges")
        
    except Exception as e:
        if debug:
            print(f"[GLASS_VIS] Error drawing glass mesh: {e}")
            import traceback
            traceback.print_exc()


def draw_attachment_debug(env):
    """
    Draw attachment points and constraints for debugging.
    
    Args:
        env: The environment containing attachment data (must have .attachment and ._uipc_gelpad)
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


def _get_tet_indices_from_uipc(env, obj=None, debug: bool = False) -> Optional[np.ndarray]:
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
