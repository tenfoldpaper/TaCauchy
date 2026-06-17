"""
Contact Force Visualization Demo (Using Modular Utilities)

This script provides the SAME functionality as contact_force_visualization_demo.py,
but uses the modular visualization utilities from tacex_uipc.utils.visualization.

This demonstrates how the refactored code is cleaner and more maintainable while
providing identical functionality.

Features:
1. Real-time contact force extraction from UIPC simulation
2. 3D force visualization in Isaac Sim scene (all force components)
3. 2D force visualization on tactile image window
4. Interpolated force visualization on regular 20×20 grids
5. Tet mesh structure visualization (cyan wireframe)
6. Deformation visualization with color-coded heatmap
7. Contact surface filtering (only show bottom surface)
"""

from __future__ import annotations

import argparse
import cv2
import numpy as np
import time
from typing import Dict, List
from dataclasses import dataclass

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(
    description="Demo: Contact Force Visualization (Modular Version)"
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--sys", type=bool, default=True, help="Whether to track system utilization.")
parser.add_argument(
    "--debug_vis",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Whether to render tactile images in the GUI",
)
parser.add_argument(
    "--force_scale",
    type=float,
    default=0.1,
    help="Scale factor for force arrow visualization",
)
parser.add_argument(
    "--sensor",
    type=str,
    default="gelsight",
    choices=["gelsight", "digit", "9dtact"],
    help="Tactile sensor type: 'gelsight' for GelSight Mini, 'digit' for DIGIT sensor, '9dtact' for 9DTact sensor",
)
parser.add_argument(
    "--enable_glass",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable glass layer UIPC object to stiffen top surface",
)
parser.add_argument(
    "--glass_vis",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to visualize the glass layer tet mesh",
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

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import traceback

import carb
import pynvml
from scipy.ndimage import gaussian_filter
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

from tacex import GelSightSensor
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_single_uipc import (
    FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_UIPC_CFG,
)
from tacex_assets.robots.franka.franka_digit_single_uipc import (
    FRANKA_PANDA_ARM_SINGLE_DIGIT_HIGH_PD_UIPC_CFG,
)
from tacex_assets.robots.franka.franka_9dtact_single_uipc import (
    FRANKA_PANDA_ARM_SINGLE_9DTACT_HIGH_PD_UIPC_CFG,
)
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg
from tacex_assets.sensors.ninedtact import NineDTactCfg

# Import for DIGIT sensor support
import os

from tacex_uipc import (
    UipcIsaacAttachments,
    UipcIsaacAttachmentsCfg,
    UipcObject,
    UipcObjectCfg,
    UipcRLEnv,
    UipcSimCfg,
)
from tacex_uipc.sim import ContactForceExtractor, ContactForceData, StressExtractor
from tacex_uipc.utils import TetMeshCfg

# ============================================================================
# Import modular visualization utilities
# ============================================================================
from tacex_uipc.utils.visualization import (
    ForceVisualizer,
    InterpolatedForceVisualizer,
    DeformationVisualizer,
    draw_3d_forces,
    draw_tet_mesh,
    draw_ball_mesh,
    draw_glass_mesh,
    draw_attachment_debug,
    filter_contact_surface_vertices,
)

# ============================================================================
# Performance Monitoring
# ============================================================================
@dataclass
class PerformanceMetrics:
    """Container for performance timing data."""
    uipc_step_times: List[float]
    contact_extraction_times: List[float]
    stress_extraction_times: List[float]
    total_step_times: List[float]
    
    def __init__(self):
        self.uipc_step_times = []
        self.contact_extraction_times = []
        self.stress_extraction_times = []
        self.total_step_times = []
    
    def add_measurements(self, uipc_time: float, contact_time: float, stress_time: float, total_time: float):
        """Add timing measurements for one step."""
        self.uipc_step_times.append(uipc_time)
        self.contact_extraction_times.append(contact_time)
        self.stress_extraction_times.append(stress_time)
        self.total_step_times.append(total_time)
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics."""
        if not self.uipc_step_times:
            return {}
        
        def compute_stats(times: List[float]) -> Dict[str, float]:
            times_ms = np.array(times) * 1000  # Convert to milliseconds
            return {
                'mean': np.mean(times_ms),
                'std': np.std(times_ms),
                'min': np.min(times_ms),
                'max': np.max(times_ms),
                'p95': np.percentile(times_ms, 95),
                'p99': np.percentile(times_ms, 99),
                'fps': 1000 / np.mean(times_ms) if np.mean(times_ms) > 0 else 0
            }
        
        return {
            'uipc_step': compute_stats(self.uipc_step_times),
            'contact_extraction': compute_stats(self.contact_extraction_times),
            'stress_extraction': compute_stats(self.stress_extraction_times),
            'total_step': compute_stats(self.total_step_times)
        }
    
    def print_summary(self, step_count: int):
        """Print performance summary."""
        if step_count < 20:  # Wait for some data
            return
            
        stats = self.get_statistics()
        if not stats:
            return
        
        print(f"\n{'='*70}")
        print(f"UIPC PERFORMANCE SUMMARY (Steps: {step_count})")
        print(f"{'='*70}")
        
        # UIPC Step Performance (most important)
        uipc_stats = stats['uipc_step']
        print(f"\n🔥 UIPC Finite Element Simulation:")
        print(f"  Mean: {uipc_stats['mean']:.2f} ms  |  Std: {uipc_stats['std']:.2f} ms")
        print(f"  Min:  {uipc_stats['min']:.2f} ms   |  Max: {uipc_stats['max']:.2f} ms")
        print(f"  P95:  {uipc_stats['p95']:.2f} ms   |  P99: {uipc_stats['p99']:.2f} ms")
        print(f"  Max FPS (Pure Mechanics): {uipc_stats['fps']:.1f}")
        
        # Contact Force Extraction
        contact_stats = stats['contact_extraction']
        print(f"\n📊 Contact Force Extraction:")
        print(f"  Mean: {contact_stats['mean']:.3f} ms  |  Max: {contact_stats['max']:.3f} ms")
        
        # Stress Extraction
        stress_stats = stats['stress_extraction']
        print(f"\n🔬 Stress Extraction:")
        print(f"  Mean: {stress_stats['mean']:.3f} ms  |  Max: {stress_stats['max']:.3f} ms")
        
        # Total Performance
        total_stats = stats['total_step']
        print(f"\n⚡ Total Step Performance:")
        print(f"  Mean: {total_stats['mean']:.2f} ms  |  Max FPS: {total_stats['fps']:.1f}")
        
        # Real-time capability assessment
        print(f"\n🎯 Real-time Capability Assessment:")
        stable_fps = 1000 / uipc_stats['p95']  # Use P95 for stability
        print(f"  Stable FPS (P95): {stable_fps:.1f}")
        print(f"  60 FPS: {'✅ ACHIEVABLE' if stable_fps >= 60 else '❌ NOT ACHIEVABLE'}")
        print(f"  30 FPS: {'✅ ACHIEVABLE' if stable_fps >= 30 else '❌ NOT ACHIEVABLE'}")
        print(f"  15 FPS: {'✅ ACHIEVABLE' if stable_fps >= 15 else '❌ NOT ACHIEVABLE'}")
        
        # Performance recommendations
        print(f"\n💡 Performance Analysis:")
        if stable_fps >= 60:
            print("  🚀 Excellent! Can achieve 60+ FPS real-time simulation")
            print("  💎 Consider increasing mesh density for higher accuracy")
        elif stable_fps >= 30:
            print("  ✅ Good performance for 30 FPS real-time applications")
            print("  🔧 For 60 FPS, consider mesh coarsening or parameter tuning")
        elif stable_fps >= 15:
            print("  ⚠️  Suitable for 15 FPS applications or offline processing")
            print("  🔧 Consider coarser mesh or relaxed solver tolerances")
        else:
            print("  🐌 Performance may be too slow for real-time applications")
            print("  🔧 Strongly recommend mesh optimization and parameter tuning")
        
        print(f"{'='*70}")

# Global performance metrics
performance_metrics = PerformanceMetrics()




# ============================================================================
# Top Surface Deformation Visualizer (for constrained surface)
# ============================================================================
class TopSurfaceDeformationVisualizer:
    """可视化顶部被约束表面的变形情况（移除刚体运动后的局部变形）"""
    
    def __init__(self, image_size: tuple = (512, 512), enable_window: bool = False):
        """
        初始化顶部表面位移可视化器
        
        Args:
            image_size: 可视化图像大小 (width, height)
        """
        self.image_size = image_size
        
        # 创建OpenCV窗口（可选）
        self.use_cv_window = False
        if enable_window:
            try:
                print("[TopSurfaceDeformationVisualizer] 尝试创建OpenCV窗口...")
                cv2.namedWindow("Top Surface Displacement (Constrained)", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Top Surface Displacement (Constrained)", 512, 512)
                self.use_cv_window = True
                print("[TopSurfaceDeformationVisualizer] ✓ OpenCV窗口已创建")
            except cv2.error as e:
                print(f"[TopSurfaceDeformationVisualizer] ✗ OpenCV窗口创建失败: {e}")
            except Exception as e:
                print(f"[TopSurfaceDeformationVisualizer] ✗ 未知错误: {e}")
        
        self.frame_count = 0
        self.max_deformation = 0.0
        print(f"[TopSurfaceDeformationVisualizer] 初始化完成 (use_cv_window={self.use_cv_window})")
    
    def visualize_displacement(self,
                              current_positions: np.ndarray,
                              initial_positions: np.ndarray,
                              attachment_indices: np.ndarray) -> np.ndarray:
        """
        可视化顶部表面的位移
        
        Args:
            current_positions: 当前顶点位置 (N, 3)
            initial_positions: 初始顶点位置 (N, 3)
            attachment_indices: 约束点索引
            
        Returns:
            可视化图像 (BGR格式)
        """
        # 提取顶部表面（被约束的点）
        if attachment_indices is None or len(attachment_indices) == 0:
            if self.frame_count == 0:
                print("[TopSurfaceDeformationVisualizer] attachment_indices为空，返回空图像")
            return np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        
        if self.frame_count == 0:
            print(f"[TopSurfaceDeformationVisualizer] 开始可视化，约束点数量: {len(attachment_indices)}")
        
        top_current = current_positions[attachment_indices]
        top_initial = initial_positions[attachment_indices]
        
        # 计算变形（移除刚体运动，只看局部变形）
        # 使用 Kabsch 算法对齐，移除整体平移和旋转
        if len(top_current) >= 3:
            # Center both point sets
            init_centroid = np.mean(top_initial, axis=0)
            curr_centroid = np.mean(top_current, axis=0)
            init_centered = top_initial - init_centroid
            curr_centered = top_current - curr_centroid
            
            # Compute rotation via SVD (Kabsch algorithm)
            h = init_centered.T @ curr_centered
            u, _, vt = np.linalg.svd(h)
            r = vt.T @ u.T
            if np.linalg.det(r) < 0:
                vt[-1, :] *= -1
                r = vt.T @ u.T
            
            # Apply rotation + translation to initial positions
            aligned_initial = (init_centered @ r) + curr_centroid
            
            # Calculate deformation (local shape change only)
            displacement = np.linalg.norm(top_current - aligned_initial, axis=1)
        else:
            # Fallback for too few points
            displacement = np.linalg.norm(top_current - top_initial, axis=1)
        
        # 更新最大位移用于归一化
        current_max = np.max(displacement)
        if current_max > self.max_deformation:
            self.max_deformation = current_max
            if self.frame_count % 50 == 0:
                print(f"[TopSurfaceDeformationVisualizer] 最大变形更新: {self.max_deformation*1000:.3f} mm")
        
        # 创建可视化图像
        img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        
        # 将3D位置投影到2D
        x_coords = top_current[:, 0]
        y_coords = top_current[:, 1]
        
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        margin = 50
        w = self.image_size[0] - 2 * margin
        h = self.image_size[1] - 2 * margin
        
        # 归一化并映射到图像坐标
        x_norm = (x_coords - x_min) / max(x_max - x_min, 0.001)
        y_norm = (y_coords - y_min) / max(y_max - y_min, 0.001)
        
        x_img = (margin + x_norm * w).astype(int)
        y_img = (margin + (1 - y_norm) * h).astype(int)  # 翻转Y轴
        
        # 绘制顶点，颜色基于位移大小
        for i, (x, y, disp_mag) in enumerate(zip(x_img, y_img, displacement)):
            # 归一化位移到 [0, 1]
            if self.max_deformation > 0:
                norm_disp = disp_mag / self.max_deformation
            else:
                norm_disp = 0
            
            # 颜色映射：蓝色（无位移）-> 绿色 -> 黄色 -> 红色（最大位移）
            if norm_disp < 0.5:
                # 蓝色到绿色
                r = 0
                g = int(255 * norm_disp * 2)
                b = int(255 * (1 - norm_disp * 2))
            else:
                # 绿色到红色
                r = int(255 * (norm_disp - 0.5) * 2)
                g = int(255 * (1 - (norm_disp - 0.5) * 2))
                b = 0
            
            color = (b, g, r)  # BGR for OpenCV
            
            # 绘制约束点（用方形标记）
            cv2.rectangle(img, (x-3, y-3), (x+3, y+3), color, -1)
            cv2.rectangle(img, (x-4, y-4), (x+4, y+4), (255, 255, 255), 1)  # 白色边框
        
        # 添加统计信息
        self._add_statistics(img, displacement, len(attachment_indices))
        
        # 添加颜色条
        self._add_colorbar(img)
        
        return img
    
    def _add_statistics(self, img: np.ndarray, displacement: np.ndarray, num_points: int):
        """添加统计信息覆盖层"""
        # 统计框
        cv2.rectangle(img, (5, 5), (320, 120), (50, 50, 50), -1)
        cv2.rectangle(img, (5, 5), (320, 120), (255, 255, 255), 1)
        
        # 统计数据
        mean_disp = np.mean(displacement)
        max_disp = np.max(displacement)
        
        y_offset = 25
        cv2.putText(img, "Top Surface Deformation", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        cv2.putText(img, f"Constrained Points: {num_points}", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.putText(img, f"Mean Deform: {mean_disp*1000:.3f} mm", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.putText(img, f"Max Deform: {max_disp*1000:.3f} mm", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _add_colorbar(self, img: np.ndarray):
        """添加颜色条图例"""
        h, w = img.shape[:2]
        
        # 颜色条尺寸
        bar_width = 30
        bar_height = 200
        bar_x = w - bar_width - 20
        bar_y = h - bar_height - 20
        
        # 绘制颜色条
        for i in range(bar_height):
            norm_val = i / bar_height
            
            # 与顶点相同的颜色映射
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
        
        # 颜色条边框
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (255, 255, 255), 1)
        
        # 标签
        cv2.putText(img, "0", (bar_x + bar_width + 5, bar_y + bar_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, f"{self.max_deformation*1000:.1f}mm", 
                   (bar_x + bar_width + 5, bar_y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def show(self, image: np.ndarray):
        """显示可视化"""
        self.frame_count += 1
        if self.use_cv_window:
            try:
                cv2.imshow("Top Surface Displacement (Constrained)", image)
                cv2.waitKey(1)
                if self.frame_count == 1:
                    print("[TopSurfaceDeformationVisualizer] 第一帧已显示")
            except Exception as e:
                if self.frame_count == 1:
                    print(f"[TopSurfaceDeformationVisualizer] 显示错误: {e}")
        else:
            if self.frame_count == 1:
                print("[TopSurfaceDeformationVisualizer] 窗口不可用，跳过显示")
    
    def close(self):
        """关闭可视化窗口"""
        if self.use_cv_window:
            try:
                cv2.destroyWindow("Top Surface Displacement (Constrained)")
            except:
                pass


# ============================================================================
# Surface Triangle Force Visualizer (stress × area = force)
# ============================================================================
class SurfaceTriangleForceVisualizer:
    """
    可视化表面三角形的受力（应力 × 面积 = 力）
    显示每个三角形的合力大小
    """
    
    def __init__(self, 
                 image_size: tuple = (512, 512),
                 window_name: str = "Surface Triangles - Total Force"):
        """
        初始化表面三角形受力可视化器
        
        Args:
            image_size: 可视化图像大小 (width, height)
            window_name: 窗口名称
        """
        self.image_size = image_size
        self.window_name = window_name
        
        # 创建OpenCV窗口
        self.use_cv_window = False
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 512, 512)
            self.use_cv_window = True
            print("[SurfaceTriForceVis] ✓ OpenCV窗口已创建")
        except cv2.error as e:
            print(f"[SurfaceTriForceVis] ✗ OpenCV窗口创建失败: {e}")
        
        self.frame_count = 0
        self.max_force = 0.0
    
    def visualize(self,
                  stress_data,
                  gelpad_object):
        """
        可视化表面三角形的受力
        
        Args:
            stress_data: 应力数据（包含 normal_pressure 和 tangential_traction）
            gelpad_object: UIPC gelpad 对象，用于获取三角形几何信息
        """
        if stress_data is None or stress_data.normal_pressure is None:
            # 无数据，显示空图像
            empty_img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
            cv2.putText(empty_img, "No Force Data", (self.image_size[0]//2 - 100, self.image_size[1]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            if self.use_cv_window:
                cv2.imshow(self.window_name, empty_img)
                cv2.waitKey(1)
            return
        
        # 计算每个三角形的受力
        try:
            # 获取三角形索引和顶点位置
            tri = stress_data.triangle_indices
            geo = gelpad_object.geo_slot_list[0].geometry()
            positions = geo.positions().view().reshape(-1, 3)
            
            # 计算每个三角形的面积
            v0 = positions[tri[:, 0]]
            v1 = positions[tri[:, 1]]
            v2 = positions[tri[:, 2]]
            
            # 使用叉积计算面积
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross = np.cross(edge1, edge2)
            areas = 0.5 * np.linalg.norm(cross, axis=1)
            
            # 计算法向力：法向压力 × 面积
            normal_forces = np.abs(stress_data.normal_pressure) * areas
            
            # 计算切向力：切向牵引力大小 × 面积
            if stress_data.tangential_traction is not None:
                if len(stress_data.tangential_traction.shape) == 2 and stress_data.tangential_traction.shape[1] >= 2:
                    tangent_traction_mag = np.linalg.norm(stress_data.tangential_traction[:, :2], axis=1)
                else:
                    tangent_traction_mag = np.abs(stress_data.tangential_traction)
                tangent_forces = tangent_traction_mag * areas
            else:
                tangent_forces = np.zeros_like(normal_forces)
            
            # 计算合力：法向力 + 切向力（标量和）
            total_forces = normal_forces + tangent_forces
            
            # 创建可视化（绘制三角形）
            img = self._visualize_triangles(positions, tri, total_forces, areas)
            
            # 显示
            self.frame_count += 1
            if self.use_cv_window:
                cv2.imshow(self.window_name, img)
                cv2.waitKey(1)
                
        except Exception as e:
            if self.frame_count == 0:
                print(f"[SurfaceTriForceVis] 计算受力失败: {e}")
    
    def _visualize_triangles(self,
                            positions: np.ndarray,
                            triangles: np.ndarray,
                            forces: np.ndarray,
                            areas: np.ndarray) -> np.ndarray:
        """绘制三角形并用颜色填充表示受力"""
        img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        
        if len(positions) == 0 or forces is None:
            return img
        
        # 更新最大力
        max_force = np.max(forces) if len(forces) > 0 else 0.0
        if max_force > self.max_force:
            self.max_force = max_force
        
        # 计算2D投影的边界
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # 添加边距
        margin = 50
        w = self.image_size[0] - 2 * margin
        h = self.image_size[1] - 2 * margin
        
        # 投影到2D图像坐标
        x_norm = (x_coords - x_min) / max(x_max - x_min, 0.001)
        y_norm = (y_coords - y_min) / max(y_max - y_min, 0.001)
        
        x_img = (margin + x_norm * w).astype(int)
        y_img = (margin + (1 - y_norm) * h).astype(int)
        
        # 绘制每个三角形
        for i, (tri_idx, force) in enumerate(zip(triangles, forces)):
            # 获取三角形的三个顶点在图像中的坐标
            p0 = (x_img[tri_idx[0]], y_img[tri_idx[0]])
            p1 = (x_img[tri_idx[1]], y_img[tri_idx[1]])
            p2 = (x_img[tri_idx[2]], y_img[tri_idx[2]])
            
            # 根据力的大小计算颜色
            if force < 0.00001:
                # 无力的三角形用深灰色
                color = (50, 50, 50)
            else:
                # 有力的三角形用紫色渐变
                norm_force = force / max(self.max_force, 0.001)
                intensity = int(100 + norm_force * 155)  # 100-255
                color = (intensity, 0, intensity)  # 紫色 (B, G, R)
            
            # 绘制填充三角形
            pts = np.array([p0, p1, p2], dtype=np.int32)
            cv2.fillPoly(img, [pts], color)
            
            # 绘制三角形边框（白色细线）
            cv2.polylines(img, [pts], True, (255, 255, 255), 1)
        
        # 添加统计信息
        self._add_stats(img, forces, areas)
        
        return img
    
    def _add_stats(self, img: np.ndarray, forces: np.ndarray, areas: np.ndarray):
        """添加统计信息"""
        # 统计框
        cv2.rectangle(img, (5, 5), (380, 160), (50, 50, 50), -1)
        cv2.rectangle(img, (5, 5), (380, 160), (255, 255, 255), 1)
        
        # 标题
        y_offset = 25
        cv2.putText(img, "Triangle Forces (Stress x Area)", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        
        # 统计数据
        if len(forces) > 0:
            num_active = np.sum(forces > 0.00001)
            avg_force = np.mean(forces[forces > 0.00001]) if num_active > 0 else 0.0
            total_force = np.sum(forces)
            avg_area = np.mean(areas) if len(areas) > 0 else 0.0
            
            y_offset += 25
            cv2.putText(img, f"Active Triangles: {num_active}/{len(forces)}", (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            y_offset += 22
            cv2.putText(img, f"Max Force: {self.max_force:.5f} N", (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            y_offset += 22
            cv2.putText(img, f"Avg Force: {avg_force:.5f} N", (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            y_offset += 22
            cv2.putText(img, f"Total Force: {total_force:.5f} N", (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            y_offset += 22
            cv2.putText(img, f"Avg Triangle Area: {avg_area*1e6:.3f} mm^2", (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # 颜色图例
        legend_y = self.image_size[1] - 40
        legend_x_start = 15
        legend_width = 200
        
        # 绘制渐变条（紫色）
        for i in range(legend_width):
            intensity = int(100 + (i / legend_width) * 155)
            color = (intensity, 0, intensity)  # 紫色
            cv2.line(img, (legend_x_start + i, legend_y), 
                    (legend_x_start + i, legend_y + 15), color, 1)
        
        # 标签
        cv2.putText(img, "0", (legend_x_start - 5, legend_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img, f"{self.max_force:.4f}N", (legend_x_start + legend_width - 50, legend_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def close(self):
        """关闭可视化窗口"""
        if self.use_cv_window:
            try:
                cv2.destroyWindow(self.window_name)
            except:
                pass


# ============================================================================
# Surface Triangle Stress Visualizers (from stress extractor)
# ============================================================================
class SurfaceTriangleStressVisualizer:
    """
    可视化表面三角形网格的应力数据（来自 StressExtractor）
    显示每个三角形的法向压力和切向牵引力
    """
    
    def __init__(self, 
                 image_size: tuple = (512, 512),
                 window_name_normal: str = "Surface Triangles - Normal Pressure",
                 window_name_tangent: str = "Surface Triangles - Tangential Traction"):
        """
        初始化表面三角形应力可视化器
        
        Args:
            image_size: 可视化图像大小 (width, height)
            window_name_normal: 法向压力窗口名称
            window_name_tangent: 切向牵引力窗口名称
        """
        self.image_size = image_size
        self.window_name_normal = window_name_normal
        self.window_name_tangent = window_name_tangent
        
        # 创建两个OpenCV窗口
        self.use_cv_window = False
        try:
            cv2.namedWindow(self.window_name_normal, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name_normal, 512, 512)
            cv2.namedWindow(self.window_name_tangent, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name_tangent, 512, 512)
            self.use_cv_window = True
            print("[SurfaceTriStressVis] ✓ OpenCV窗口已创建")
        except cv2.error as e:
            print(f"[SurfaceTriStressVis] ✗ OpenCV窗口创建失败: {e}")
        
        self.frame_count = 0
        self.max_normal_pressure = 0.0
        self.max_tangent_traction = 0.0
    
    def visualize(self,
                  stress_data,
                  triangle_positions: np.ndarray):
        """
        可视化表面三角形的应力数据
        
        Args:
            stress_data: 应力数据（包含 normal_pressure 和 tangential_traction）
            triangle_positions: 三角形中心位置的2D投影 (N, 2)
        """
        if stress_data is None or stress_data.normal_pressure is None:
            # 无数据，显示空图像
            empty_img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
            cv2.putText(empty_img, "No Stress Data", (self.image_size[0]//2 - 100, self.image_size[1]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            if self.use_cv_window:
                cv2.imshow(self.window_name_normal, empty_img)
                cv2.imshow(self.window_name_tangent, empty_img)
                cv2.waitKey(1)
            return
        
        # 创建法向压力和切向牵引力可视化
        normal_img = self._visualize_normal_pressure(
            triangle_positions, stress_data.normal_pressure
        )
        tangent_img = self._visualize_tangent_traction(
            triangle_positions, stress_data.tangential_traction
        )
        
        # 显示
        self.frame_count += 1
        if self.use_cv_window:
            cv2.imshow(self.window_name_normal, normal_img)
            cv2.imshow(self.window_name_tangent, tangent_img)
            cv2.waitKey(1)
    
    def _visualize_normal_pressure(self,
                                   positions: np.ndarray,
                                   pressures: np.ndarray) -> np.ndarray:
        """可视化法向压力（用圆圈大小和颜色表示）"""
        img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        
        if len(positions) == 0 or pressures is None:
            return img
        
        # 使用绝对值（压力可能为负）
        pressure_values = np.abs(pressures)
        max_pressure = np.max(pressure_values) if len(pressure_values) > 0 else 0.0
        
        if max_pressure > self.max_normal_pressure:
            self.max_normal_pressure = max_pressure
        
        # 绘制每个三角形中心点
        for i, (pos, pressure) in enumerate(zip(positions, pressure_values)):
            x, y = int(pos[0]), int(pos[1])
            
            # 确保在图像范围内
            if x < 0 or x >= self.image_size[0] or y < 0 or y >= self.image_size[1]:
                continue
            
            if pressure < 0.0001:
                # 无压力的点用小灰点表示
                cv2.circle(img, (x, y), 2, (100, 100, 100), -1)
            else:
                # 有压力的点
                norm_pressure = pressure / max(self.max_normal_pressure, 0.001)
                
                # 圆圈大小
                radius = int(3 + norm_pressure * 12)  # 3-15 像素
                
                # 颜色：绿色渐变（浅绿到深绿）
                intensity = int(100 + norm_pressure * 155)  # 100-255
                color = (0, intensity, 0)  # 绿色
                
                # 绘制填充圆
                cv2.circle(img, (x, y), radius, color, -1)
                # 白色边框
                cv2.circle(img, (x, y), radius, (255, 255, 255), 1)
        
        # 添加统计信息
        self._add_stats(img, "Normal Pressure (Surface Triangles)", 
                       pressure_values, self.max_normal_pressure, "Pa")
        
        return img
    
    def _visualize_tangent_traction(self,
                                    positions: np.ndarray,
                                    tractions: np.ndarray) -> np.ndarray:
        """可视化切向牵引力（用箭头表示方向和大小）"""
        img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        
        if len(positions) == 0 or tractions is None:
            return img
        
        # 确保 tractions 是 2D 数组
        if len(tractions.shape) == 1:
            # 如果是 1D，假设只有大小，没有方向
            traction_magnitudes = np.abs(tractions)
            traction_directions = np.zeros((len(tractions), 2))
        elif len(tractions.shape) == 2:
            if tractions.shape[1] >= 2:
                # 提取 x, y 分量
                traction_xy = tractions[:, :2]
                traction_magnitudes = np.linalg.norm(traction_xy, axis=1)
                # 计算方向（归一化）
                traction_directions = np.zeros_like(traction_xy)
                nonzero_mask = traction_magnitudes > 0.0001
                traction_directions[nonzero_mask] = traction_xy[nonzero_mask] / traction_magnitudes[nonzero_mask, np.newaxis]
            else:
                traction_magnitudes = np.abs(tractions[:, 0])
                traction_directions = np.zeros((len(tractions), 2))
        else:
            return img
        
        max_traction = np.max(traction_magnitudes) if len(traction_magnitudes) > 0 else 0.0
        
        if max_traction > self.max_tangent_traction:
            self.max_tangent_traction = max_traction
        
        # 绘制每个三角形的切向力箭头
        for i, (pos, magnitude, direction) in enumerate(zip(positions, traction_magnitudes, traction_directions)):
            x, y = int(pos[0]), int(pos[1])
            
            # 确保在图像范围内
            if x < 0 or x >= self.image_size[0] or y < 0 or y >= self.image_size[1]:
                continue
            
            if magnitude < 0.0001:
                # 无牵引力的点用小灰点表示
                cv2.circle(img, (x, y), 2, (100, 100, 100), -1)
            else:
                # 有牵引力的点，绘制箭头
                norm_traction = magnitude / max(self.max_tangent_traction, 0.001)
                
                # 箭头长度（根据力的大小）
                arrow_length = int(5 + norm_traction * 30)  # 5-35 像素
                
                # 颜色：红色渐变（浅红到深红）
                intensity = int(100 + norm_traction * 155)  # 100-255
                color = (0, 0, intensity)  # 红色 (BGR)
                
                # 计算箭头终点
                end_x = int(x + direction[0] * arrow_length)
                end_y = int(y + direction[1] * arrow_length)
                
                # 确保终点在图像范围内
                end_x = max(0, min(self.image_size[0] - 1, end_x))
                end_y = max(0, min(self.image_size[1] - 1, end_y))
                
                # 绘制箭头
                thickness = max(1, int(1 + norm_traction * 2))  # 1-3 像素
                cv2.arrowedLine(img, (x, y), (end_x, end_y), color, thickness, tipLength=0.3)
                
                # 在起点绘制小圆点以标记位置
                cv2.circle(img, (x, y), 2, (255, 255, 255), -1)
        
        # 添加统计信息
        self._add_stats(img, "Tangential Traction (Surface Triangles)", 
                       traction_magnitudes, self.max_tangent_traction, "Pa")
        
        return img
    
    def _add_stats(self, img: np.ndarray, title: str, values: np.ndarray, max_value: float, unit: str):
        """添加统计信息"""
        # 统计框
        cv2.rectangle(img, (5, 5), (350, 140), (50, 50, 50), -1)
        cv2.rectangle(img, (5, 5), (350, 140), (255, 255, 255), 1)
        
        # 标题
        y_offset = 25
        cv2.putText(img, title, (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 统计数据
        if len(values) > 0:
            num_active = np.sum(values > 0.0001)
            avg_value = np.mean(values[values > 0.0001]) if num_active > 0 else 0.0
            total_value = np.sum(values)
            
            y_offset += 25
            cv2.putText(img, f"Active Triangles: {num_active}/{len(values)}", (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            y_offset += 20
            cv2.putText(img, f"Max: {max_value:.2f} {unit}", (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            y_offset += 20
            cv2.putText(img, f"Avg: {avg_value:.2f} {unit}", (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            y_offset += 20
            cv2.putText(img, f"Total: {total_value:.2f} {unit}", (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # 颜色图例
        legend_y = self.image_size[1] - 40
        legend_x_start = 15
        legend_width = 200
        
        # 绘制渐变条
        for i in range(legend_width):
            intensity = int(100 + (i / legend_width) * 155)
            if "Normal" in title:
                color = (0, intensity, 0)  # 绿色
            else:
                color = (0, 0, intensity)  # 红色
            cv2.line(img, (legend_x_start + i, legend_y), 
                    (legend_x_start + i, legend_y + 15), color, 1)
        
        # 标签
        cv2.putText(img, "0", (legend_x_start - 5, legend_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img, f"{max_value:.1f}", (legend_x_start + legend_width - 30, legend_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def close(self):
        """关闭可视化窗口"""
        if self.use_cv_window:
            try:
                cv2.destroyWindow(self.window_name_normal)
                cv2.destroyWindow(self.window_name_tangent)
            except:
                pass


# ============================================================================
# Contact Force Grid Visualizer (contact surface)
# ============================================================================
class ContactForceGridVisualizer:
    """在接触面规则网格上显示合力大小（颜色表示）。"""

    def __init__(self,
                 grid_size: tuple = (20, 20),
                 image_size: tuple = (512, 512),
                 window_name: str = "Contact Force (Grid)"):
        self.grid_size = grid_size
        self.image_size = image_size
        self.window_name = window_name
        self.use_cv_window = False
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 512, 512)
            self.use_cv_window = True
            print("[ContactForceGridVis] OpenCV窗口已创建")
        except cv2.error:
            print("[ContactForceGridVis] OpenCV窗口不可用")

    def visualize(self, force_data: ContactForceData, vertex_positions_2d: np.ndarray):
        """将接触点压力/力大小投影到规则网格并显示。"""
        if force_data is None:
            return
        # Prefer betterforce pressure magnitudes, fallback to pressure vectors
        if force_data.pressure_magnitudes is not None:
            magnitudes = force_data.pressure_magnitudes
        elif force_data.contact_pressures is not None:
            magnitudes = np.linalg.norm(force_data.contact_pressures, axis=1)
        else:
            return
        self.visualize_from_values(magnitudes, vertex_positions_2d)

    def visualize_from_values(self, magnitudes: np.ndarray, vertex_positions_2d: np.ndarray):
        """将任意标量场投影到规则网格并显示。"""
        if magnitudes is None or len(magnitudes) == 0:
            return

        if vertex_positions_2d is None or len(vertex_positions_2d) == 0:
            vertex_positions_2d = self._generate_vertex_positions(len(magnitudes))

        grid_h, grid_w = self.grid_size
        grid = np.zeros((grid_h, grid_w), dtype=np.float32)
        counts = np.zeros((grid_h, grid_w), dtype=np.float32)

        if len(vertex_positions_2d) != len(magnitudes):
            min_len = min(len(vertex_positions_2d), len(magnitudes))
            vertex_positions_2d = vertex_positions_2d[:min_len]
            magnitudes = magnitudes[:min_len]

        img_w, img_h = self.image_size
        for pos, mag in zip(vertex_positions_2d, magnitudes):
            x, y = pos[0], pos[1]
            col = int(np.clip(x / img_w * grid_w, 0, grid_w - 1))
            row = int(np.clip(y / img_h * grid_h, 0, grid_h - 1))
            grid[row, col] += mag
            counts[row, col] += 1.0

        with np.errstate(divide="ignore", invalid="ignore"):
            grid = np.divide(grid, counts, out=np.zeros_like(grid), where=counts > 0)

        max_val = np.max(grid)
        if max_val < 1e-12:
            return

        grid_norm = (grid / max_val * 255.0).astype(np.uint8)
        heatmap = cv2.applyColorMap(grid_norm, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, self.image_size, interpolation=cv2.INTER_NEAREST)

        if self.use_cv_window:
            cv2.imshow(self.window_name, heatmap)
            cv2.waitKey(1)

    def close(self):
        if self.use_cv_window:
            try:
                cv2.destroyWindow(self.window_name)
            except:
                pass

    def _generate_vertex_positions(self, num_vertices: int) -> np.ndarray:
        """Generate evenly distributed 2D positions for fallback visualization."""
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


# ============================================================================
# Helper functions
# ============================================================================

def create_robot_cfg(sensor_type: str) -> ArticulationCfg:
    """
    Create robot configuration based on sensor type.
    
    Args:
        sensor_type: "gelsight", "digit", or "9dtact"
        
    Returns:
        Robot articulation configuration
    """
    if sensor_type == "digit":
        # Use DIGIT robot configuration
        robot_cfg = FRANKA_PANDA_ARM_SINGLE_DIGIT_HIGH_PD_UIPC_CFG.copy()
    elif sensor_type == "9dtact":
        # Use 9DTact robot configuration
        robot_cfg = FRANKA_PANDA_ARM_SINGLE_9DTACT_HIGH_PD_UIPC_CFG.copy()
    else:  # gelsight
        # Use GelSight Mini robot configuration
        robot_cfg = FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_UIPC_CFG.copy()
    
    return robot_cfg


def create_sensor_cfg(sensor_type: str, debug_vis: bool) -> GelSightMiniCfg:
    """
    Create sensor configuration based on sensor type.
    
    Args:
        sensor_type: "gelsight", "digit", or "9dtact"
        
    Returns:
        Sensor configuration
    """
    if sensor_type == "digit":
        # DIGIT sensor configuration
        sensor_cfg = GelSightMiniCfg(
            prim_path="/World/envs/env_.*/Robot/panda/digit_case",
            sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
                prim_path_appendix="/Camera",
                update_period=0,
                resolution=(64, 64),
                data_types=["depth"],
                clipping_range=(0.024, 0.034),
            ),
            device="cuda",
            debug_vis=debug_vis,
            marker_motion_sim_cfg=None,
            data_types=["tactile_rgb"],
        )
        sensor_cfg.optical_sim_cfg = sensor_cfg.optical_sim_cfg.replace(
            calib_folder_path=f"{TACEX_ASSETS_DATA_DIR}/Sensors/digit/calibs/640x480",
            with_shadow=False,
            device="cuda",
            tactile_img_res=(64, 64),
        )
    elif sensor_type == "9dtact":
        # 9DTact sensor configuration
        sensor_cfg = NineDTactCfg(
            prim_path="/World/envs/env_.*/Robot/case_9dtact",
            sensor_camera_cfg=NineDTactCfg.SensorCameraCfg(
                prim_path_appendix="/Camera",
                update_period=0,
                resolution=(64, 64),
                data_types=["depth"],
                clipping_range=(0.024, 0.034),
            ),
            device="cuda",
            debug_vis=debug_vis,
            marker_motion_sim_cfg=None,
            data_types=["tactile_rgb"],
        )
        sensor_cfg.optical_sim_cfg = sensor_cfg.optical_sim_cfg.replace(
            calib_folder_path=f"{TACEX_ASSETS_DATA_DIR}/Sensors/9dtact/calibs/640x480",
            with_shadow=False,
            device="cuda",
            tactile_img_res=(64, 64),
            convert_to_grayscale=True,  # 9DTact is a monochrome sensor
            apply_color_correction=True,  # Match calibration image tone
        )
    else:  # gelsight
        sensor_cfg = GelSightMiniCfg(
            prim_path="/World/envs/env_.*/Robot/gelsight_mini_case",
            sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
                prim_path_appendix="/Camera",
                update_period=0,
                resolution=(64, 64),
                data_types=["depth"],
                clipping_range=(0.024, 0.034),
            ),
            device="cuda",
            debug_vis=debug_vis,
            marker_motion_sim_cfg=None,
            data_types=["tactile_rgb"],
        )
        sensor_cfg.optical_sim_cfg = sensor_cfg.optical_sim_cfg.replace(
            with_shadow=False,
            device="cuda",
            tactile_img_res=(64, 64),
        )
    
    return sensor_cfg


def create_gelpad_cfg(sensor_type: str) -> UipcObjectCfg:
    """
    Create gelpad configuration based on sensor type.
    
    Args:
        sensor_type: "gelsight", "digit", or "9dtact"
        
    Returns:
        Gelpad UIPC object configuration
    """
    if sensor_type == "digit":
        gelpad_cfg = UipcObjectCfg(
            prim_path="/World/envs/env_.*/Robot/panda/digit_gelpad",
            mesh_cfg=None,
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
                youngs_modulus=0.1,
                poisson_rate=0.40,
            ),
        )
    elif sensor_type == "9dtact":
        gelpad_cfg = UipcObjectCfg(
            prim_path="/World/envs/env_.*/Robot/Gelpad9DTact",
            mesh_prim_name="mesh",
            mesh_cfg=None,
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
                youngs_modulus=0.01,  # 更软的材料，容易变形
                poisson_rate=0.40,
            ),
        )
    else:  # gelsight
        gelpad_cfg = UipcObjectCfg(
            prim_path="/World/envs/env_.*/Robot/gelsight_mini_gelpad",
            mesh_cfg=None,
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
                youngs_modulus=0.1,
                poisson_rate=0.40,
            ),
        )
    
    return gelpad_cfg


def create_gelpad_attachment_cfg(sensor_type: str) -> UipcIsaacAttachmentsCfg:
    """
    Create gelpad attachment configuration based on sensor type.
    
    Args:
        sensor_type: "gelsight", "digit", or "9dtact"
        
    Returns:
        Gelpad attachment configuration
    """
    if sensor_type == "digit":
        attachment_cfg = UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=2000.0,
            body_name="digit_case",
            debug_vis=False,
            compute_attachment_data=True,  # 使用运行时计算的约束数据
            constrain_entire_surface=False,  # 恢复为只约束附着点，顶部可变形
            surface_constraint_z_percentile=70.0  # 顶部30%的点（仅在开启整体约束时生效）
        )
    elif sensor_type == "9dtact":
        attachment_cfg = UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=2000.0,
            body_name="case_9dtact",
            debug_vis=False,
            compute_attachment_data=True,  # 使用运行时计算的约束数据
            constrain_entire_surface=False,  # 恢复为只约束附着点，顶部可变形
            surface_constraint_z_percentile=70.0,  # 顶部30%的点（仅在开启整体约束时生效）
            attachment_points_radius=4e-4  # ⭐ 0.4mm - 减小半径以减少附着点（9dtact gelpad较薄）
        )
    else:  # gelsight
        attachment_cfg = UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=2000.0,
            body_name="gelsight_mini_case",
            debug_vis=False,
            compute_attachment_data=True,  # 使用运行时计算的约束数据
            constrain_entire_surface=False,  # 恢复为只约束附着点，顶部可变形
            surface_constraint_z_percentile=70.0  # 顶部30%的点（仅在开启整体约束时生效）
        )
    
    return attachment_cfg


def create_glass_layer_cfg(sensor_type: str) -> UipcObjectCfg | None:
    """
    Create glass layer configuration (rigid UIPC object) for GelSight Mini and 9DTact.
    
    Args:
        sensor_type: "gelsight", "digit", or "9dtact"
        
    Returns:
        Glass layer UIPC object configuration or None if not available
    """
    if sensor_type == "gelsight":
        return UipcObjectCfg(
            prim_path="/World/envs/env_.*/Robot/gelsight_mini_gelpad",
            mesh_prim_name="glass",
            mesh_cfg=None,  # use precomputed tet data in USD
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
                youngs_modulus=50.0,  # high stiffness to approximate rigid glass
                poisson_rate=0.45,
            ),
            mass_density=2500.0,
        )
    elif sensor_type == "9dtact":
        return UipcObjectCfg(
            prim_path="/World/envs/env_.*/Robot/Gelpad9DTact",
            mesh_prim_name="glass",
            mesh_cfg=None,  # use precomputed tet data in USD
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
                youngs_modulus=50.0,  # high stiffness to approximate rigid glass
                poisson_rate=0.45,
            ),
            mass_density=2500.0,
        )
    else:
        return None


def create_glass_attachment_cfg(sensor_type: str) -> UipcIsaacAttachmentsCfg | None:
    """
    Create glass layer attachment configuration.
    
    Args:
        sensor_type: "gelsight", "digit", or "9dtact"
        
    Returns:
        Glass layer attachment configuration or None if not available
    """
    if sensor_type == "gelsight" or sensor_type == "9dtact":
        return UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=10000.0,
            body_name="case_9dtact" if sensor_type == "9dtact" else "gelsight_mini_case",
            debug_vis=False,
            compute_attachment_data=True,
        )
    else:
        return None


def print_force_summary(force_data: ContactForceData, step: int, debug: bool = False):
    """Print contact force summary."""
    if not debug:
        return
    
    if force_data.vertex_indices is None or len(force_data.vertex_indices) == 0:
        return
    
    total_force = np.sum(force_data.contact_forces, axis=0) if force_data.contact_forces is not None else np.zeros(3)
    max_force = np.max(np.linalg.norm(force_data.contact_forces, axis=1)) if force_data.contact_forces is not None else 0.0
    
    print(f"\n[Step {step}] Contact Force Summary:")
    print(f"  Vertices in contact: {len(force_data.vertex_indices)}")
    print(f"  Total Force: [{total_force[0]:.4f}, {total_force[1]:.4f}, {total_force[2]:.4f}] N")
    print(f"  Max Force Magnitude: {max_force:.4f} N")


def build_smoothed_pressure_map(
    force_data: ContactForceData,
    vertex_positions_2d: np.ndarray,
    image_size: tuple = (512, 512),
    grid_size: tuple = (20, 20),
    sigma: float = 1.0,
) -> np.ndarray | None:
    """Build a smoothed pressure map (binning + Gaussian filter) for visualization."""
    if force_data is None or vertex_positions_2d is None:
        return None
    if force_data.vertex_indices is None or len(force_data.vertex_indices) == 0:
        return None

    # Prefer precomputed pressure magnitudes from the extractor
    if force_data.pressure_magnitudes is not None and len(force_data.pressure_magnitudes) > 0:
        pressures = force_data.pressure_magnitudes
    elif force_data.contact_forces is not None and force_data.nodal_areas is not None:
        pressures = np.linalg.norm(force_data.contact_forces, axis=1) / (force_data.nodal_areas + 1e-8)
    else:
        return None

    if len(pressures) == 0:
        return None

    grid_h, grid_w = grid_size
    grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    counts = np.zeros((grid_h, grid_w), dtype=np.float32)

    img_w, img_h = image_size
    for pos, pressure in zip(vertex_positions_2d, pressures):
        x, y = pos[0], pos[1]
        col = int(np.clip(x / img_w * grid_w, 0, grid_w - 1))
        row = int(np.clip(y / img_h * grid_h, 0, grid_h - 1))
        grid[row, col] += pressure
        counts[row, col] += 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        grid = np.divide(grid, counts, out=np.zeros_like(grid), where=counts > 0)

    # Gaussian smoothing to emulate sensor low-pass characteristics
    grid_smoothed = gaussian_filter(grid, sigma=sigma)

    max_val = np.max(grid_smoothed)
    if max_val < 1e-12:
        return None

    grid_norm = (grid_smoothed / max_val * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(grid_norm, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, image_size, interpolation=cv2.INTER_CUBIC)
    return heatmap


def print_detailed_attachment_debug(env, step: int):
    """Print detailed attachment debugging information."""
    print(f"\n[ATTACHMENT DEBUG - Step {step}]")
    print("=" * 50)
    
    try:
        # Determine case name based on sensor type
        if env.cfg.sensor_type == "digit":
            case_name = "digit_case"
        elif env.cfg.sensor_type == "9dtact":
            case_name = "case_9dtact"
        else:
            case_name = "gelsight_mini_case"
        
        # Case information
        case_body_idx = env._robot.find_bodies(case_name)[0][0]
        case_pos = env._robot.data.body_link_pos_w[:, case_body_idx].cpu().numpy()
        
        # Gelpad information
        env._uipc_gelpad.update(0.0)
        gelpad_positions = env._uipc_gelpad.data.nodal_pos_w.cpu().numpy()
        gelpad_center = np.mean(gelpad_positions, axis=0)
        
        print(f"1. CASE POSITION ({case_name}): [{case_pos[0, 0]:.6f}, {case_pos[0, 1]:.6f}, {case_pos[0, 2]:.6f}]")
        print(f"2. GELPAD CENTER: [{gelpad_center[0]:.6f}, {gelpad_center[1]:.6f}, {gelpad_center[2]:.6f}]")
        print(f"3. GELPAD VERTICES: {len(gelpad_positions)}")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"[ERROR] Failed to print attachment debug: {e}")


# ============================================================================
# Custom UI Window (from original file)
# ============================================================================

class CustomEnvWindow(BaseEnvWindow):
    """Custom environment window for visualization controls."""
    
    def __init__(self, env, window_name: str = "Isaac Lab"):
        super().__init__(env, window_name)
    
    def _create_ui_elements(self):
        """Create custom UI elements."""
        super()._create_ui_elements()
        # Add custom UI elements here if needed


# ============================================================================
# Environment Configuration (from original file)
# ============================================================================

@configclass
class BallRollingEnvCfg(DirectRLEnvCfg):
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1.9, 1.4, 0.3)
    viewer.lookat = (-1.5, -1.9, -1.1)

    debug_vis = True
    ui_window_class_type = CustomEnvWindow
    decimation = 1
    
    # Sensor type selection
    sensor_type: str = "gelsight"  # "gelsight" or "digit"
    
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
        gravity=(0.0, 0.0, -9.8),
        dt=1/60,
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

    physx_sphere2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/physx_sphere2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.03]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/tactile_test_shapes/sphere2.usd",
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
    )

    ball = UipcObjectCfg(
        prim_path="/World/envs/env_.*/sphere2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0.03]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/tactile_test_shapes/sphere2.usd",
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
        ),
        mesh_cfg=None,
        constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(
            m_kappa=100.0,
            kinematic=True,
        ),
        mass_density=1e3,
    )

    ball_radius = 0.005
    obj_pos_randomization_range = [-0.15, 0.15]
    episode_length_s = 0
    action_space = 0
    observation_space = 0
    state_space = 0
    
    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    
    # Note: robot, gelpad_cfg, gelpad_attachment_cfg, and gsmini will be set dynamically
    # based on sensor_type in the main() function before creating the environment
    robot: ArticulationCfg = None
    gelpad_cfg: UipcObjectCfg = None
    gelpad_attachment_cfg: UipcIsaacAttachmentsCfg = None
    glass_layer_cfg: UipcObjectCfg | None = None
    glass_attachment_cfg: UipcIsaacAttachmentsCfg | None = None
    gsmini: GelSightMiniCfg = None


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

        # Offset depends on sensor type
        if self.cfg.sensor_type == "digit":
            # DIGIT sensor has slightly different geometry
            self._offset_pos = torch.tensor([0.0, 0.0, 0.132], device=self.device).repeat(self.num_envs, 1)
        elif self.cfg.sensor_type == "9dtact":
            # 9DTact sensor geometry (similar to GelSight Mini)
            self._offset_pos = torch.tensor([0.0, 0.0, 0.131], device=self.device).repeat(self.num_envs, 1)
        else:  # gelsight
            self._offset_pos = torch.tensor([0.0, 0.0, 0.131], device=self.device).repeat(self.num_envs, 1)
        
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        self.ik_commands = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        self.step_count = 0
        self.goal_prim_view = None
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.scene.clone_environments(copy_from_source=False)

        # Determine offset based on sensor type
        if self.cfg.sensor_type == "digit":
            ee_offset_z = 0.132
        elif self.cfg.sensor_type == "9dtact":
            ee_offset_z = 0.131
        else:  # gelsight
            ee_offset_z = 0.131

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
                    offset=OffsetCfg(pos=(0.0, 0.0, ee_offset_z)),
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
            position=np.array([0.5, 0.0, 0.027]),
            orientation=np.array([0, 1, 0, 0]),
            color=np.array([255.0, 0.0, 0.0]),
            visible=False,
        )

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # --- UIPC simulation setup ---
        self._uipc_gelpad = UipcObject(self.cfg.gelpad_cfg, self.uipc_sim)
        self.ball = UipcObject(self.cfg.ball, self.uipc_sim)
        self.glass_layer = None
        
        # Create attachment
        self.attachment = UipcIsaacAttachments(
            self.cfg.gelpad_attachment_cfg, self._uipc_gelpad, self.scene.articulations["robot"]
        )
        
        # Create glass layer if available
        if self.cfg.glass_layer_cfg is not None:
            self.glass_layer = UipcObject(self.cfg.glass_layer_cfg, self.uipc_sim)
            if self.cfg.glass_attachment_cfg is not None:
                self.glass_attachment = UipcIsaacAttachments(
                    self.cfg.glass_attachment_cfg, self.glass_layer, self.scene.articulations["robot"]
                )

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

    def _get_observations(self) -> dict:
        return {}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool), torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

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

    def _compute_frame_pose(self):
        """Compute end-effector pose."""
        ee_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        ee_pos_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pos_b, ee_quat_b, self._offset_pos, self._offset_rot
        )
        return ee_pos_b, ee_quat_b

    def _compute_frame_jacobian(self):
        """Compute end-effector Jacobian."""
        jacobian = self.jacobian_b
        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])
        return jacobian


# ============================================================================
# Main simulation loop
# ============================================================================

def run_simulation(env, debug_flags: dict):
    """
    Run the simulation with modular visualization tools.
    
    This function uses the refactored visualization modules to provide
    the same functionality as the original script.
    """
    # ========== Print visualization info ==========
    print("\n" + "=" * 60)
    print("VISUALIZATION WINDOWS:")
    print("=" * 60)
    print("  3D SCENE (Isaac Sim):")
    print("  - Cyan wireframe: Tet mesh structure")
    print("  - Red arrows: Contact forces (all components)")
    print("  - Yellow wireframe: Ball mesh")
    if args_cli.glass_vis:
        print("  - Light blue wireframe: Glass mesh")
    print("")
    print("  FORCE VISUALIZATION (2D Windows):")
    print("  - Tactile Image + Force Arrows")
    print("  - Interpolated Normal Forces (20×20 grid)")
    print("  - Interpolated Tangential Forces (20×20 grid)")
    print("")
    print("  DEFORMATION VISUALIZATION (2D Windows):")
    print("  - Contact Surface Deformation (底部接触面):")
    print("    * Blue: No deformation")
    print("    * Green: Medium deformation")
    print("    * Yellow/Red: Maximum deformation")
    print("  - Top Surface Deformation (顶部约束面 - 局部变形):")
    print("    * Blue: No deformation")
    print("    * Green: Medium deformation")
    print("    * Yellow/Red: Maximum deformation")
    print("=" * 60)
    
    # Reset environment
    env.reset()
    
    # Initialize goal control
    env.goal_prim_view = XFormPrim(prim_paths_expr="/World/envs/env_.*/goal", name="goals", usd=True)
    
    print(f"\n[INFO] Goal control initialized")
    print(f"[INFO] Goal controls panda_hand position (with 0.131m Z offset)")
    print(f"[INFO] You can drag the goal in Isaac Sim UI to move the robot")
    print(f"[INFO] Initial goal position: [0.5, 0.0, 0.027] (27mm above ground)\n")
    
    # Enable debug visualization for tactile_rgb
    if args_cli.debug_vis:
        for prim in env.gsmini._prim_view.prims:
            if prim.HasAttribute("debug_tactile_rgb"):
                prim.GetAttribute("debug_tactile_rgb").Set(True)
    
    # ========== Initialize visualizers ==========
    print("\n[INFO] Initializing visualization tools...")
    
    force_visualizer = None
    
    interpolated_force_visualizer = InterpolatedForceVisualizer(
        grid_size=(30, 30),
        image_size=(512, 512)
    )
    print("[INFO] InterpolatedForceVisualizer initialized")
    
    deformation_visualizer = DeformationVisualizer(
        image_size=(512, 512),
        enable_window=args_cli.debug_deformation or args_cli.debug_all,
    )
    print("[INFO] DeformationVisualizer initialized (底部接触面)")
    
    # 初始化顶部表面位移可视化器
    top_surface_visualizer = TopSurfaceDeformationVisualizer(
        image_size=(512, 512),
        enable_window=args_cli.debug_deformation or args_cli.debug_all,
    )
    print("[INFO] TopSurfaceDeformationVisualizer initialized (顶部约束面)")

    contact_force_grid_visualizer = ContactForceGridVisualizer(
        grid_size=(30, 30),
        image_size=(512, 512),
        window_name="Contact Force (Grid)",
    )
    print("[INFO] ContactForceGridVisualizer initialized (接触面真实压力)")
    
    # ⭐ 初始化表面三角形应力可视化器
    surface_triangle_stress_visualizer = SurfaceTriangleStressVisualizer(
        image_size=(512, 512),
        window_name_normal="Surface Triangles - Normal Pressure",
        window_name_tangent="Surface Triangles - Tangential Traction"
    )
    print("[INFO] SurfaceTriangleStressVisualizer initialized (表面三角形应力)")
    
    # ⭐ 初始化表面三角形受力可视化器
    surface_triangle_force_visualizer = SurfaceTriangleForceVisualizer(
        image_size=(512, 512),
        window_name="Surface Triangles - Total Force"
    )
    print("[INFO] SurfaceTriangleForceVisualizer initialized (表面三角形受力)")
    
    stress_extractor = StressExtractor()
    
    # 测试：显示一个测试图像
    test_img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.putText(test_img, "Top Surface - Waiting for data...", (50, 256),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if args_cli.debug_deformation or args_cli.debug_all:
        top_surface_visualizer.show(test_img)
    print("[INFO] 测试图像已显示到顶部表面窗口")
    
    # ========== Get force extractor ==========
    force_extractor = env.uipc_sim.contact_force_extractor
    if force_extractor is None:
        print("[ERROR] Force extractor not available!")
        return
    print("[INFO] Force extractor ready")
    
    # ========== Get attachment indices ==========
    attachment_indices = env.attachment.attachment_points_idx if hasattr(env, 'attachment') else None
    
    # ========== Simulation variables ==========
    step_count = 0
    print_interval = 50
    initial_gelpad_pos = None
    initial_pos_captured = False
    # ⭐ 固定的归一化范围（在第一次捕获时设置，之后保持不变）
    fixed_normalization_range = None
    force_scale = args_cli.force_scale
    fabric_visibility_checked = False
    
    print("\n[INFO] Starting simulation loop...")
    print("[INFO] Press Ctrl+C to stop\n")
    
    try:
        while simulation_app.is_running():
            # Start total step timing
            total_step_start = time.perf_counter()
            
            # Check fabric mesh visibility (once)
            if not fabric_visibility_checked and step_count == 10:
                print("\n[DEBUG] Checking fabric mesh visibility:")
                for i, fabric_mesh in enumerate(env.uipc_sim._fabric_meshes):
                    if fabric_mesh.HasAttribute("visibility"):
                        vis = fabric_mesh.GetAttribute("visibility").Get()
                        print(f"  Fabric mesh {i} ({fabric_mesh.GetPath()}): visibility = {vis}")
                fabric_visibility_checked = True
                print("")
            
            # ========== UIPC PHYSICS STEP (TIMED) ==========
            uipc_step_start = time.perf_counter()
            
            # Perform physics step
            env._pre_physics_step(None)
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            
            env.uipc_sim.update_render_meshes()
            
            uipc_step_end = time.perf_counter()
            uipc_step_time = uipc_step_end - uipc_step_start
            
            # Render scene
            env.sim.render()
            
            # Update IK commands from goal position
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
                    if env.cfg.sensor_type == "digit":
                        case_name = "digit_case"
                    elif env.cfg.sensor_type == "9dtact":
                        case_name = "case_9dtact"
                    else:
                        case_name = "gelsight_mini_case"
                    
                    case_body_idx = env._robot.find_bodies(case_name)[0][0]
                    case_pos = env._robot.data.body_link_pos_w[:, case_body_idx].cpu().numpy()
                    print(f"  Case position ({case_name}): [{case_pos[0, 0]:.4f}, {case_pos[0, 1]:.4f}, {case_pos[0, 2]:.4f}]")
                    
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
                
                # Get current gelpad vertex positions
                current_gelpad_pos = env._uipc_gelpad.data.nodal_pos_w.cpu().numpy()
                
                # Capture initial position after gelpad has moved to correct position (step 60)
                # ⭐ 每60帧更新一次归一化范围，保持画面居中同时能跟随 gelpad 移动
                if not initial_pos_captured and step_count == 60:
                    initial_gelpad_pos = current_gelpad_pos.copy()
                    initial_pos_captured = True
                    
                    # 固定归一化范围（基于初始位置）
                    fixed_normalization_range = {
                        'x_min': initial_gelpad_pos[:, 0].min(),
                        'x_max': initial_gelpad_pos[:, 0].max(),
                        'y_min': initial_gelpad_pos[:, 1].min(),
                        'y_max': initial_gelpad_pos[:, 1].max(),
                    }
                    
                    print(f"[INFO] Initial gelpad positions captured at step {step_count}: {initial_gelpad_pos.shape}")
                    print(f"[INFO] Normalization range: X=[{fixed_normalization_range['x_min']:.4f}, {fixed_normalization_range['x_max']:.4f}], Y=[{fixed_normalization_range['y_min']:.4f}, {fixed_normalization_range['y_max']:.4f}]")
                
                # ⭐ 每60帧更新归一化范围
                elif initial_pos_captured and step_count % 60 == 0:
                    fixed_normalization_range = {
                        'x_min': current_gelpad_pos[:, 0].min(),
                        'x_max': current_gelpad_pos[:, 0].max(),
                        'y_min': current_gelpad_pos[:, 1].min(),
                        'y_max': current_gelpad_pos[:, 1].max(),
                    }
                    if step_count % 300 == 0:  # 每300帧打印一次，避免刷屏
                        print(f"[INFO] Normalization range updated at step {step_count}: X=[{fixed_normalization_range['x_min']:.4f}, {fixed_normalization_range['x_max']:.4f}], Y=[{fixed_normalization_range['y_min']:.4f}, {fixed_normalization_range['y_max']:.4f}]")
                    
                    # Show contact surface filtering info
                    contact_surf_pos, _, _ = filter_contact_surface_vertices(
                        initial_gelpad_pos,
                        attachment_indices,
                        z_threshold_percentile=30.0,
                        reference_positions=initial_gelpad_pos,
                        use_robust_method=True,  # 使用鲁棒方法
                        use_relative_tolerance=True,  # 使用相对容差
                        relative_tolerance_ratio=0.5  # 容差为厚度的50%
                    )
                    if contact_surf_pos is not None:
                        print(f"[INFO] Contact surface vertices: {len(contact_surf_pos)} / {len(initial_gelpad_pos)} ({len(contact_surf_pos)/len(initial_gelpad_pos)*100:.1f}%)")
                        print(f"[INFO] Contact surface Z range: [{contact_surf_pos[:, 2].min():.4f}, {contact_surf_pos[:, 2].max():.4f}]")
            except Exception as e:
                if step_count == 0:
                    print(f"[Warning] Could not get gelpad positions: {e}")
            
            # Update UIPC gelpad (redundant but kept for consistency)
            env._uipc_gelpad.update(0.0)
            current_gelpad_pos = env._uipc_gelpad.data.nodal_pos_w.cpu().numpy()
            
            # Debug output
            if debug_flags.get('debug_attachment', False) and step_count % print_interval == 0:
                print_detailed_attachment_debug(env, step_count)
            
            # ========== CONTACT FORCE EXTRACTION (TIMED) ==========
            contact_extraction_start = time.perf_counter()
            
            # Extract contact forces
            gelpad_forces = force_extractor.get_object_forces(env._uipc_gelpad)
            
            contact_extraction_end = time.perf_counter()
            contact_extraction_time = contact_extraction_end - contact_extraction_start
            
            # ========== STRESS EXTRACTION (TIMED) ==========
            stress_extraction_start = time.perf_counter()
            
            # Extract stress data (for comparison)
            stress_data = None
            try:
                stress_data = stress_extractor.get_object_surface_stress(env._uipc_gelpad)
            except Exception as e:
                if step_count == 0:
                    print(f"[Warning] StressExtractor failed: {e}")
            
            stress_extraction_end = time.perf_counter()
            stress_extraction_time = stress_extraction_end - stress_extraction_start
            
            # Get current gelpad vertex positions (needed for both raw mesh and 3D visualization)
            gelpad_vertex_pos = current_gelpad_pos
            
            # ⭐ ========== Raw Mesh Node Force Visualization (Before Interpolation) ==========
            # 可视化插值前的原始网格节点受力情况
            if gelpad_vertex_pos is not None and gelpad_forces is not None:
                # 获取接触表面掩码（用于过滤显示）
                if initial_gelpad_pos is not None:
                    _, _, contact_surface_mask = filter_contact_surface_vertices(
                        gelpad_vertex_pos,
                        attachment_indices,
                        z_threshold_percentile=30.0,
                        reference_positions=initial_gelpad_pos,
                        use_robust_method=True,  # 使用鲁棒方法
                        use_relative_tolerance=True,  # 使用相对容差
                        relative_tolerance_ratio=0.5  # 容差为厚度的50%
                    )
                else:
                    contact_surface_mask = None
                
                # 调用原始网格受力可视化器（已禁用）
            
            # ========== 3D Visualization ==========
            if gelpad_vertex_pos is not None:
                # Draw 3D force arrows
                draw_3d_forces(gelpad_forces, gelpad_vertex_pos, force_scale=force_scale * 0.1)
                
                # Draw tet mesh structure
                draw_tet_mesh(env, debug=debug_flags.get('debug_tet', False))
                
                # Draw ball mesh structure
                draw_ball_mesh(env, debug=debug_flags.get('debug_tet', False))
                
                # Draw glass layer mesh structure (if present)
                if args_cli.glass_vis:
                    draw_glass_mesh(env, debug=debug_flags.get('debug_tet', False))
            
            # ========== Deformation Visualization ==========
            if initial_gelpad_pos is not None and current_gelpad_pos is not None:
                if current_gelpad_pos.shape == initial_gelpad_pos.shape:
                    # Filter to contact surface only
                    # IMPORTANT: Use initial_gelpad_pos as reference to determine which vertices
                    # belong to the contact surface. This ensures we track the same vertices
                    # even when they deform and move inward during contact.
                    current_contact_surface, _, contact_mask = filter_contact_surface_vertices(
                        current_gelpad_pos,  # Current positions for output
                        attachment_indices,
                        z_threshold_percentile=30.0,
                        reference_positions=initial_gelpad_pos,  # Use initial positions for filtering
                        use_robust_method=True,  # 使用鲁棒方法
                        use_relative_tolerance=True,  # 使用相对容差
                        relative_tolerance_ratio=0.5  # 容差为厚度的50%
                    )
                    
                    if current_contact_surface is not None and len(current_contact_surface) > 0:
                        initial_contact_surface = initial_gelpad_pos[contact_mask]
                        deformation_img = deformation_visualizer.visualize_deformation(
                            current_contact_surface,
                            initial_contact_surface,
                            attachment_indices=None,
                            remove_rigid_motion=True,
                        )
                        if debug_flags.get('debug_deformation', False):
                            deformation_visualizer.show(deformation_img)
                        
                        # Print deformation stats
                        if debug_flags.get('debug_deformation', False) and step_count % print_interval == 0:
                            deformation, _ = deformation_visualizer.compute_deformation(
                                current_positions=current_contact_surface,
                                initial_positions=initial_contact_surface,
                                remove_rigid_motion=True,
                            )
                            print(f"\n[Deformation Stats - Step {step_count}] (Contact Surface Only)")
                            print(f"  Contact surface vertices: {len(current_contact_surface)} / {len(current_gelpad_pos)}")
                            print(f"  Mean deformation: {np.mean(deformation)*1000:.3f} mm")
                            print(f"  Max deformation: {np.max(deformation)*1000:.3f} mm")
            
                    
                    # === 顶部约束面位移可视化 ===
                    if attachment_indices is not None and len(attachment_indices) > 0:
                        try:
                            top_surface_img = top_surface_visualizer.visualize_displacement(
                                current_gelpad_pos,
                                initial_gelpad_pos,
                                attachment_indices
                            )
                            if debug_flags.get('debug_deformation', False):
                                top_surface_visualizer.show(top_surface_img)
                            
                            # Print top surface displacement stats
                            if debug_flags.get('debug_deformation', False) and step_count % print_interval == 0:
                                top_current = current_gelpad_pos[attachment_indices]
                                top_initial = initial_gelpad_pos[attachment_indices]
                                displacement = np.linalg.norm(top_current - top_initial, axis=1)
                                print(f"\n[Top Surface Deformation Stats - Step {step_count}]")
                                print(f"  Constrained points: {len(attachment_indices)}")
                                print(f"  Mean deformation: {np.mean(displacement)*1000:.3f} mm")
                                print(f"  Max deformation: {np.max(displacement)*1000:.3f} mm")
                        except Exception as e:
                            if step_count % 100 == 0:  # Print error occasionally
                                print(f"[WARNING] Top surface visualization error: {e}")
                    elif step_count == 61:  # Print once after initial capture
                        print("[WARNING] Cannot visualize top surface: attachment_indices is None or empty")
            
            # ========== 2D Force Visualization ==========
            tactile_output = env.gsmini.data.output
            if tactile_output is not None and "tactile_rgb" in tactile_output:
                tactile_image = tactile_output["tactile_rgb"]
                if tactile_image is not None:
                    # Convert to numpy
                    if hasattr(tactile_image, 'cpu'):
                        tactile_image = tactile_image.cpu().numpy()
                    if len(tactile_image.shape) == 4:
                        tactile_image = tactile_image[0]
                    if tactile_image.dtype != np.uint8:
                        tactile_image = (tactile_image * 255).astype(np.uint8)
                    
                    # Project 3D to 2D for force visualization
                    vertex_positions_2d = None
                    if gelpad_forces.vertex_indices is not None and len(gelpad_forces.vertex_indices) > 0 and gelpad_vertex_pos is not None:
                        try:
                            # Filter to contact surface for projection
                            # Use initial positions as reference to ensure consistent surface definition
                            if initial_gelpad_pos is not None:
                                contact_surface_pos, _, _ = filter_contact_surface_vertices(
                                    gelpad_vertex_pos,  # Current positions for bounds calculation
                                    attachment_indices,
                                    z_threshold_percentile=30.0,
                                    reference_positions=initial_gelpad_pos,  # Use initial positions for filtering
                                    use_robust_method=True,  # 使用鲁棒方法
                                    use_relative_tolerance=True,  # 使用相对容差
                                    relative_tolerance_ratio=0.5  # 容差为厚度的50%
                                )
                            else:
                                # Fallback if initial positions not yet captured
                                contact_surface_pos, _, _ = filter_contact_surface_vertices(
                                    gelpad_vertex_pos,
                                    attachment_indices,
                                    z_threshold_percentile=30.0,
                                    use_robust_method=True,  # 使用鲁棒方法
                                    use_relative_tolerance=True,  # 使用相对容差
                                    relative_tolerance_ratio=0.5  # 容差为厚度的50%
                                )
                            
                            if contact_surface_pos is not None and len(contact_surface_pos) > 0:
                                x_min, x_max = contact_surface_pos[:, 0].min(), contact_surface_pos[:, 0].max()
                                y_min, y_max = contact_surface_pos[:, 1].min(), contact_surface_pos[:, 1].max()
                                
                                contact_3d = gelpad_vertex_pos[gelpad_forces.vertex_indices]
                                img_h, img_w = 512, 512
                                margin = 50
                                
                                vertex_positions_2d = []
                                for pos in contact_3d:
                                    nx = (pos[0] - x_min) / max(x_max - x_min, 0.001)
                                    ny = (pos[1] - y_min) / max(y_max - y_min, 0.001)
                                    px = margin + nx * (img_w - 2 * margin)
                                    py = margin + (1 - ny) * (img_h - 2 * margin)
                                    vertex_positions_2d.append([px, py])
                                
                                vertex_positions_2d = np.array(vertex_positions_2d)
                        except Exception as e:
                            if step_count == 0:
                                print(f"[Warning] Could not project 3D to 2D: {e}")
                    
                    # Build smoothed pressure map (binning + Gaussian filter) for tactile visualization
                    pressure_map_img = build_smoothed_pressure_map(
                        gelpad_forces,
                        vertex_positions_2d,
                        image_size=(512, 512),
                        grid_size=(20, 20),
                        sigma=1.0,
                    )

                    base_image = pressure_map_img if pressure_map_img is not None else tactile_image

                    # Interpolated force visualization (handled with stress-based data below)

                    # Stress-based pressure (cauchy stress -> surface pressure)
                    try:
                        stress_data = stress_extractor.get_object_surface_stress(env._uipc_gelpad)
                        if stress_data.normal_pressure is not None and stress_data.triangle_indices is not None:
                            tri = stress_data.triangle_indices
                            geo = env._uipc_gelpad.geo_slot_list[0].geometry()
                            positions = geo.positions().view().reshape(-1, 3)
                            tri_centers = (positions[tri[:, 0]] + positions[tri[:, 1]] + positions[tri[:, 2]]) / 3.0

                            # ⭐ 使用固定的归一化范围（第一帧捕获后保持不变），避免画面随 gelpad 移动而偏移
                            if fixed_normalization_range is not None:
                                x_min = fixed_normalization_range['x_min']
                                x_max = fixed_normalization_range['x_max']
                                y_min = fixed_normalization_range['y_min']
                                y_max = fixed_normalization_range['y_max']
                            elif initial_gelpad_pos is not None and len(initial_gelpad_pos) > 0:
                                # 回退：使用初始位置（但这会随 gelpad 移动而变化）
                                x_min, x_max = initial_gelpad_pos[:, 0].min(), initial_gelpad_pos[:, 0].max()
                                y_min, y_max = initial_gelpad_pos[:, 1].min(), initial_gelpad_pos[:, 1].max()
                            else:
                                # 最后回退：使用三角形中心的范围
                                x_min, x_max = tri_centers[:, 0].min(), tri_centers[:, 0].max()
                                y_min, y_max = tri_centers[:, 1].min(), tri_centers[:, 1].max()

                            img_h, img_w = 512, 512
                            margin = 50
                            tri_positions_2d = []
                            for pos in tri_centers:
                                nx = (pos[0] - x_min) / max(x_max - x_min, 0.001)
                                ny = (pos[1] - y_min) / max(y_max - y_min, 0.001)
                                px = margin + nx * (img_w - 2 * margin)
                                py = margin + (1 - ny) * (img_h - 2 * margin)
                                tri_positions_2d.append([px, py])
                            tri_positions_2d = np.array(tri_positions_2d)

                            pressure_magnitudes = np.abs(stress_data.normal_pressure)
                            contact_force_grid_visualizer.visualize_from_values(
                                pressure_magnitudes,
                                tri_positions_2d,
                            )
                            interpolated_force_visualizer.visualize_from_values(
                                pressure_magnitudes,
                                stress_data.tangential_traction,
                                tri_positions_2d,
                            )
                            
                            # ⭐ 表面三角形应力可视化
                            surface_triangle_stress_visualizer.visualize(
                                stress_data,
                                tri_positions_2d
                            )
                            
                            # ⭐ 表面三角形受力可视化（应力 × 面积）
                            surface_triangle_force_visualizer.visualize(
                                stress_data,
                                env._uipc_gelpad
                            )
                    except Exception as e:
                        if step_count == 0:
                            print(f"[Warning] StressExtractor failed: {e}")
            
            # Print force summary
            if step_count % print_interval == 0:
                print_force_summary(gelpad_forces, step_count, debug=debug_flags.get('debug_force', False))
            
            # ========== PERFORMANCE MONITORING ==========
            # Calculate total step time
            total_step_end = time.perf_counter()
            total_step_time = total_step_end - total_step_start
            
            # Record performance metrics
            performance_metrics.add_measurements(
                uipc_time=uipc_step_time,
                contact_time=contact_extraction_time,
                stress_time=stress_extraction_time,
                total_time=total_step_time
            )
            
            # Print performance summary periodically
            if step_count > 0 and step_count % 50 == 0:  # Every 50 steps
                performance_metrics.print_summary(step_count)
            
            # Print real-time performance info occasionally
            if step_count % 20 == 0 and step_count > 10:
                recent_uipc_times = performance_metrics.uipc_step_times[-10:]  # Last 10 steps
                if recent_uipc_times:
                    avg_uipc_time = np.mean(recent_uipc_times) * 1000  # Convert to ms
                    fps = 1000 / avg_uipc_time if avg_uipc_time > 0 else 0
                    print(f"[Performance] Step {step_count}: UIPC={avg_uipc_time:.1f}ms, FPS={fps:.1f}")
            
            step_count += 1
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        # Print final performance summary
        if step_count > 20:
            print(f"\n{'='*70}")
            print("FINAL PERFORMANCE SUMMARY")
            print(f"{'='*70}")
            performance_metrics.print_summary(step_count)
        
        print("\n[INFO] Cleaning up...")
        deformation_visualizer.close()
        interpolated_force_visualizer.close()
        contact_force_grid_visualizer.close()
        surface_triangle_stress_visualizer.close()  # ⭐ 关闭表面三角形应力可视化器
        surface_triangle_force_visualizer.close()  # ⭐ 关闭表面三角形受力可视化器
        top_surface_visualizer.close()
        env.close()
        try:
            pynvml.nvmlShutdown()
        except:
            pass


def main():
    """Main function."""
    print("=" * 70)
    print("Contact Force Visualization Demo (Modular Version)")
    print("=" * 70)
    print("\nThis script uses the modular visualization utilities from:")
    print("  tacex_uipc.utils.visualization")
    print("\nProvides the SAME functionality as the original demo, but with")
    print("cleaner, more maintainable code.\n")
    
    # Print sensor selection
    print(f"[INFO] Selected sensor: {args_cli.sensor.upper()}")
    if args_cli.sensor == "digit":
        print("  - Using DIGIT sensor configuration")
        print("  - Case prim: digit_case")
        print("  - Gelpad prim: digit_gelpad")
    elif args_cli.sensor == "9dtact":
        print("  - Using 9DTact sensor configuration")
        print("  - Case prim: case_9dtact")
        print("  - Gelpad prim: Gelpad9DTact/mesh")
        print("  - Glass prim: Gelpad9DTact/glass")
    else:
        print("  - Using GelSight Mini sensor configuration")
        print("  - Case prim: gelsight_mini_case")
        print("  - Gelpad prim: gelsight_mini_gelpad")
    print()
    
    # Create environment
    env_cfg = BallRollingEnvCfg()
    env_cfg.sensor_type = args_cli.sensor  # Set sensor type from command line

    # Resolve debug_vis default based on sensor type / headless mode
    if args_cli.debug_vis is None:
        args_cli.debug_vis = args_cli.sensor != "digit"
    if getattr(args_cli, "headless", False):
        args_cli.debug_vis = False
    
    # Initialize sensor-specific configurations
    print(f"[DEBUG] Initializing configurations for sensor: {env_cfg.sensor_type}")
    env_cfg.robot = create_robot_cfg(env_cfg.sensor_type).replace(prim_path="/World/envs/env_.*/Robot")
    env_cfg.gelpad_cfg = create_gelpad_cfg(env_cfg.sensor_type)
    env_cfg.gelpad_attachment_cfg = create_gelpad_attachment_cfg(env_cfg.sensor_type)
    if args_cli.enable_glass:
        env_cfg.glass_layer_cfg = create_glass_layer_cfg(env_cfg.sensor_type)
        env_cfg.glass_attachment_cfg = create_glass_attachment_cfg(env_cfg.sensor_type)
    else:
        env_cfg.glass_layer_cfg = None
        env_cfg.glass_attachment_cfg = None
    env_cfg.gsmini = create_sensor_cfg(env_cfg.sensor_type, args_cli.debug_vis)
    
    # Verify configurations
    print(f"[DEBUG] Robot USD path: {env_cfg.robot.spawn.usd_path}")
    print(f"[DEBUG] Gelpad prim path: {env_cfg.gelpad_cfg.prim_path}")
    print(f"[DEBUG] Attachment body name: {env_cfg.gelpad_attachment_cfg.body_name}")
    print(f"[DEBUG] Sensor prim path: {env_cfg.gsmini.prim_path}")
    
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.debug_vis = args_cli.debug_vis
    
    # Prepare debug flags
    debug_flags = {
        'debug_ball': args_cli.debug_ball or args_cli.debug_all,
        'debug_attachment': args_cli.debug_attachment or args_cli.debug_all,
        'debug_position': args_cli.debug_position or args_cli.debug_all,
        'debug_deformation': args_cli.debug_deformation or args_cli.debug_all,
        'debug_force': args_cli.debug_force or args_cli.debug_all,
        'debug_tet': args_cli.debug_tet or args_cli.debug_all,
    }
    
    print("[INFO] Creating environment...")
    experiment = BallRollingEnv(env_cfg)
    
    # Run simulation
    run_simulation(experiment, debug_flags)
    
    simulation_app.close()


if __name__ == "__main__":
    main()
