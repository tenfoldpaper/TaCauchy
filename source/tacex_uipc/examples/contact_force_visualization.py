"""
Contact Force Visualization Demo

This script extends single_uipc_attachment.py to demonstrate contact force
extraction and visualization. It shows:
1. Real-time contact force extraction from UIPC simulation
2. Force visualization with arrows on the tactile sensor image
3. Normal and tangential force decomposition display

Based on libuipc's contact system feature:
- Force from Newton's Law: f = -gradient / dt^2
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
    
    # Print debug info
    print(f"[DEBUG] Drawing 3D forces: {len(force_data.vertex_indices)} contact points")
    
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
            cv2.namedWindow("Tet Mesh Deformation", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Tet Mesh Deformation", 512, 512)
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
        
        Args:
            current_positions: Current vertex positions (N, 3)
            initial_positions: Initial vertex positions (N, 3)
            attachment_indices: Indices of attachment points
            
        Returns:
            Visualization image
        """
        # Calculate deformation magnitude for each vertex
        deformation = np.linalg.norm(current_positions - initial_positions, axis=1)
        
        # Update max deformation for normalization
        current_max = np.max(deformation)
        if current_max > self.max_deformation:
            self.max_deformation = current_max
        
        # Create visualization image
        img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        
        # Project 3D positions to 2D
        # Use X and Y coordinates, normalize to image space
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
        
        # Draw vertices with color based on deformation
        for i, (x, y, def_mag) in enumerate(zip(x_img, y_img, deformation)):
            # Normalize deformation to [0, 1]
            if self.max_deformation > 0:
                norm_def = def_mag / self.max_deformation
            else:
                norm_def = 0
            
            # Color map: Blue (no deformation) -> Green -> Yellow -> Red (max deformation)
            if norm_def < 0.5:
                # Blue to Green
                r = 0
                g = int(255 * norm_def * 2)
                b = int(255 * (1 - norm_def * 2))
            else:
                # Green to Red
                r = int(255 * (norm_def - 0.5) * 2)
                g = int(255 * (1 - (norm_def - 0.5) * 2))
                b = 0
            
            color = (b, g, r)  # BGR for OpenCV
            
            # Draw larger circles for attachment points
            if attachment_indices is not None and i in attachment_indices:
                cv2.circle(img, (x, y), 8, (255, 255, 255), 2)  # White outline
                cv2.circle(img, (x, y), 6, color, -1)
            else:
                cv2.circle(img, (x, y), 4, color, -1)
        
        # Add statistics overlay
        self._add_statistics(img, deformation, attachment_indices)
        
        # Add color bar legend
        self._add_colorbar(img)
        
        return img
    
    def _add_statistics(self, img: np.ndarray, deformation: np.ndarray, attachment_indices: list):
        """Add statistics overlay to the image."""
        # Stats box
        cv2.rectangle(img, (5, 5), (300, 120), (50, 50, 50), -1)
        cv2.rectangle(img, (5, 5), (300, 120), (255, 255, 255), 1)
        
        # Statistics
        mean_def = np.mean(deformation)
        max_def = np.max(deformation)
        min_def = np.min(deformation)
        
        y_offset = 25
        cv2.putText(img, "Tet Mesh Deformation", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        cv2.putText(img, f"Vertices: {len(deformation)}", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.putText(img, f"Mean Def: {mean_def*1000:.3f} mm", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.putText(img, f"Max Def: {max_def*1000:.3f} mm", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 20
        if attachment_indices is not None:
            cv2.putText(img, f"Attachment Points: {len(attachment_indices)}", (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _add_colorbar(self, img: np.ndarray):
        """Add color bar legend."""
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
        cv2.putText(img, "0", (bar_x + bar_width + 5, bar_y + bar_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, f"{self.max_deformation*1000:.1f}mm", 
                   (bar_x + bar_width + 5, bar_y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def show(self, image: np.ndarray):
        """Display the visualization."""
        self.frame_count += 1
        if self.use_cv_window:
            cv2.imshow("Tet Mesh Deformation", image)
            cv2.waitKey(1)
        else:
            # Save to file every 10 frames
            if self.frame_count % 10 == 0:
                cv2.imwrite(self.save_path, image)
    
    def close(self):
        """Close visualization window."""
        if self.use_cv_window:
            try:
                cv2.destroyWindow("Tet Mesh Deformation")
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
        
        Args:
            base_image: Base tactile image (RGB or grayscale)
            force_data: Contact force data from extractor
            vertex_positions_2d: 2D positions of vertices in image space
            
        Returns:
            Image with force arrows drawn
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
        
        # Draw arrows for each contact point
        for i, (idx, force) in enumerate(zip(force_data.vertex_indices, force_data.contact_forces)):
            if i >= len(vertex_positions_2d):
                break
                
            pos = vertex_positions_2d[i]
            x, y = int(pos[0]), int(pos[1])
            
            # Skip if outside image
            if x < 0 or x >= self.image_size[0] or y < 0 or y >= self.image_size[1]:
                continue
            
            # Draw total force arrow (yellow)
            self._draw_arrow(vis_image, x, y, force, self.total_color, "Total")
            
            # Draw normal force arrow (green) if available
            if force_data.normal_forces is not None and i < len(force_data.normal_forces):
                normal = force_data.normal_forces[i]
                self._draw_arrow(vis_image, x + 5, y, normal, self.normal_color, "Normal")
            
            # Draw tangential force arrow (red) if available
            if force_data.tangential_forces is not None and i < len(force_data.tangential_forces):
                tangent = force_data.tangential_forces[i]
                self._draw_arrow(vis_image, x - 5, y, tangent, self.tangent_color, "Tangent")
        
        # Add legend and statistics
        self._add_legend(vis_image, force_data)
        
        return vis_image
    
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
        cv2.rectangle(image, (5, h - 100), (180, h - 5), (50, 50, 50), -1)
        cv2.rectangle(image, (5, h - 100), (180, h - 5), (255, 255, 255), 1)
        
        # Legend items
        y_offset = h - 85
        cv2.arrowedLine(image, (15, y_offset), (40, y_offset), self.total_color, 2)
        cv2.putText(image, "Total Force", (50, y_offset + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.arrowedLine(image, (15, y_offset), (40, y_offset), self.normal_color, 2)
        cv2.putText(image, "Normal Force", (50, y_offset + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.arrowedLine(image, (15, y_offset), (40, y_offset), self.tangent_color, 2)
        cv2.putText(image, "Tangent Force", (50, y_offset + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Statistics
        y_offset += 25
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


def print_force_summary(force_data: ContactForceData, step: int):
    """Print force summary to console."""
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
        contact=UipcSimCfg.Contact(d_hat=0.0001),
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

    # Ball as PhysX RigidObject (not UIPC)
    ball = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.15]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/ball_wood.usd",
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
            ),
        ),
    )

    robot: ArticulationCfg = FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_UIPC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )
    
    # Gelpad uses precomputed tet mesh from USD file
    # Set mesh_cfg=None to load precomputed data instead of generating
    gelpad_cfg = UipcObjectCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_gelpad",
        mesh_cfg=None,  # Use precomputed tet mesh from USD
        constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(),
    )
    gelpad_attachment_cfg = UipcIsaacAttachmentsCfg(
        constraint_strength_ratio=100.0, 
        body_name="gelsight_mini_case", 
        debug_vis=False,
        compute_attachment_data=False  # 使用预计算数据（稳定）
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
        )

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # --- UIPC simulation setup ---
        print("\n[DEBUG] Setting up UIPC objects:")
        self._uipc_gelpad = UipcObject(self.cfg.gelpad_cfg, self.uipc_sim)
        print(f"  Gelpad created: {self.cfg.gelpad_cfg.prim_path}")

        # Ball is now a PhysX RigidObject, not UIPC
        self.ball = RigidObject(self.cfg.ball)
        print(f"  Ball created as PhysX RigidObject: {self.cfg.ball.prim_path}")

        # create attachment
        print("\n[DEBUG] Creating attachment:")
        print(f"  Gelpad prim: {self.cfg.gelpad_cfg.prim_path}")
        print(f"  Robot prim: {self.scene.articulations['robot'].cfg.prim_path}")
        print(f"  Body name: {self.cfg.gelpad_attachment_cfg.body_name}")
        
        self.attachment = UipcIsaacAttachments(
            self.cfg.gelpad_attachment_cfg, self._uipc_gelpad, self.scene.articulations["robot"]
        )
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


def run_simulator(env: BallRollingEnv, force_scale: float):
    """Runs the simulation loop with force visualization."""

    print(f"Starting simulation with {env.num_envs} envs")
    print("=" * 60)
    print("Contact Force & Deformation Visualization Demo")
    print("=" * 60)
    print("  FORCE VISUALIZATION:")
    print("  - Green arrows: Normal forces")
    print("  - Red arrows: Tangential forces")
    print("  - Yellow arrows: Total forces")
    print("  - Arrow length indicates force magnitude")
    print("")
    print("  DEFORMATION VISUALIZATION:")
    print("  - Blue: No deformation")
    print("  - Green: Medium deformation")
    print("  - Yellow/Red: Maximum deformation")
    print("  - White circles: Attachment points")
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
    
    # Get contact force extractor from UIPC simulation
    force_extractor = env.uipc_sim.contact_force_extractor
    if force_extractor is None:
        print("[ERROR] Force extractor not available!")
    else:
        print("[INFO] Force extractor ready")
    
    # Get initial gelpad positions for deformation calculation
    # We'll capture this after the first simulation step to ensure correct coordinates
    initial_gelpad_pos = None
    initial_pos_captured = False
    
    # Get attachment indices
    attachment_indices = None
    if hasattr(env.attachment, 'attachment_points_idx'):
        attachment_indices = env.attachment.attachment_points_idx
        print(f"[INFO] Attachment indices loaded: {len(attachment_indices)} points")
    
    step_count = 0
    print_interval = 30  # Print force summary every N steps
    
    # Simulation loop
    try:
        while simulation_app.is_running():
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
                
                # Monitor gelpad position
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
                
                # Monitor ball position (PhysX RigidObject)
                if hasattr(env.ball, 'data') and hasattr(env.ball.data, 'root_pos_w'):
                    ball_pos = env.ball.data.root_pos_w.cpu().numpy()
                    print(f"  Ball center: [{ball_pos[0, 0]:.4f}, {ball_pos[0, 1]:.4f}, {ball_pos[0, 2]:.4f}]")
                
                # Monitor goal position
                if env.goal_prim_view is not None:
                    goal_positions, _ = env.goal_prim_view.get_world_poses()
                    print(f"  Goal position: [{goal_positions[0, 0]:.4f}, {goal_positions[0, 1]:.4f}, {goal_positions[0, 2]:.4f}]")
            
            # === DEFORMATION VISUALIZATION ===
            current_gelpad_pos = None
            try:
                # Force update gelpad data before accessing
                env._uipc_gelpad.update(0.0)
                
                # Get current gelpad vertex positions
                current_gelpad_pos = env._uipc_gelpad.data.nodal_pos_w.cpu().numpy()
                
                # Capture initial position after gelpad has moved to correct position (step 60)
                if not initial_pos_captured and step_count == 60:
                    initial_gelpad_pos = current_gelpad_pos.copy()
                    initial_pos_captured = True
                    print(f"[INFO] Initial gelpad positions captured at step {step_count}: {initial_gelpad_pos.shape}")
                    print(f"[INFO] Initial gelpad center: {np.mean(initial_gelpad_pos, axis=0)}")
                    print(f"[INFO] Position range: X[{initial_gelpad_pos[:, 0].min():.4f}, {initial_gelpad_pos[:, 0].max():.4f}], "
                          f"Y[{initial_gelpad_pos[:, 1].min():.4f}, {initial_gelpad_pos[:, 1].max():.4f}], "
                          f"Z[{initial_gelpad_pos[:, 2].min():.4f}, {initial_gelpad_pos[:, 2].max():.4f}]")
                
                if initial_gelpad_pos is not None and current_gelpad_pos is not None:
                    if current_gelpad_pos.shape == initial_gelpad_pos.shape:
                        # Visualize deformation
                        deformation_img = deformation_visualizer.visualize_deformation(
                            current_gelpad_pos,
                            initial_gelpad_pos,
                            attachment_indices
                        )
                        deformation_visualizer.show(deformation_img)
                        
                        # Print deformation stats every N steps
                        if step_count % print_interval == 0:
                            deformation = np.linalg.norm(current_gelpad_pos - initial_gelpad_pos, axis=1)
                            print(f"\n[Deformation Stats - Step {step_count}]")
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
            
            # === FORCE EXTRACTION AND VISUALIZATION ===
            
            # Extract contact forces for the gelpad
            gelpad_forces = force_extractor.get_object_forces(env._uipc_gelpad)
            
            # Print force info every N steps
            if step_count % print_interval == 0:
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
                            # Get contact vertex 3D positions
                            contact_3d = gelpad_vertex_pos[gelpad_forces.vertex_indices]
                            
                            # Get gelpad center and orientation for projection
                            gelpad_center = np.mean(gelpad_vertex_pos, axis=0)
                            
                            # Project to image coordinates
                            # Assuming gelpad surface is roughly in local XY plane, looking down Z
                            # Map X,Y range to image coordinates
                            x_min, x_max = gelpad_vertex_pos[:, 0].min(), gelpad_vertex_pos[:, 0].max()
                            y_min, y_max = gelpad_vertex_pos[:, 1].min(), gelpad_vertex_pos[:, 1].max()
                            
                            img_h, img_w = 512, 512  # visualizer image size
                            margin = 50
                            
                            vertex_positions_2d = []
                            for pos in contact_3d:
                                # Normalize to [0, 1]
                                nx = (pos[0] - x_min) / max(x_max - x_min, 0.001)
                                ny = (pos[1] - y_min) / max(y_max - y_min, 0.001)
                                
                                # Map to image coordinates
                                px = margin + nx * (img_w - 2 * margin)
                                py = margin + (1 - ny) * (img_h - 2 * margin)  # Flip Y
                                vertex_positions_2d.append([px, py])
                            
                            vertex_positions_2d = np.array(vertex_positions_2d)
                            
                            if step_count % print_interval == 0 and len(vertex_positions_2d) > 0:
                                print(f"  [2D Vis] Projected {len(vertex_positions_2d)} contact points to image")
                        except Exception as e:
                            if step_count == 0:
                                print(f"[Warning] Could not project 3D to 2D: {e}")
                    
                    # Draw force arrows on tactile image
                    vis_image = force_visualizer.draw_force_arrows(tactile_image, gelpad_forces, vertex_positions_2d)
                    
                    # Display
                    force_visualizer.show(vis_image)
            else:
                # Create blank image for force visualization
                blank_image = np.zeros((64, 64, 3), dtype=np.uint8) + 50
                vis_image = force_visualizer.draw_force_arrows(blank_image, gelpad_forces)
                force_visualizer.show(vis_image)
            
            # Print force summary periodically
            if step_count % print_interval == 0:
                print_force_summary(gelpad_forces, step_count)
            
            step_count += 1

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        force_visualizer.close()
        deformation_visualizer.close()
        env.close()
        pynvml.nvmlShutdown()


def main():
    """Main function."""
    env_cfg = BallRollingEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.gsmini.debug_vis = args_cli.debug_vis

    experiment = BallRollingEnv(env_cfg)

    print("[INFO]: Setup complete...")
    run_simulator(env=experiment, force_scale=args_cli.force_scale)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        simulation_app.close()
