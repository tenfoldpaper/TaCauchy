"""
Force visualization utilities for contact forces.

This module provides visualizers for displaying contact forces in 2D:
- ForceVisualizer: Direct force visualization on tactile images
- InterpolatedForceVisualizer: Interpolated forces on regular grids
"""

import cv2
import numpy as np
from typing import TYPE_CHECKING, Optional, Dict
from scipy.interpolate import griddata

if TYPE_CHECKING:
    from tacex_uipc.sim import ContactForceData


class InterpolatedForceVisualizer:
    """Visualizes interpolated forces on a regular grid."""
    
    def __init__(self, 
                 grid_size: tuple = (100, 100),
                 image_size: tuple = (512, 512),
                 save_path_normal: str = "/tmp/force_normal_interpolated.png",
                 save_path_tangent: str = "/tmp/force_tangent_interpolated.png",
                 enable_window: bool = False,
                 noise_threshold_percentile: float = 10.0,
                 save_dir: str = "/home/ly/data/hengfei/getdata/force_field"):
        """
        Initialize interpolated force visualizer.
        
        Args:
            grid_size: Size of the regular grid (rows, cols)
            image_size: Size of the visualization image (width, height)
            save_path_normal: Path to save normal force visualization
            save_path_tangent: Path to save tangential force visualization
            enable_window: Whether to create OpenCV windows (default: False)
            noise_threshold_percentile: Percentile threshold for noise filtering (default: 10.0)
            save_dir: Directory to save images when clicking (default: /home/ly/data/hengfei/getdata/force_field)
        """
        self.grid_size = grid_size
        self.image_size = image_size
        self.save_path_normal = save_path_normal
        self.save_path_tangent = save_path_tangent
        self.noise_threshold_percentile = noise_threshold_percentile
        
        # Setup save directory for click-to-save
        from pathlib import Path
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_counter_normal_interp = 0
        self.save_counter_tangent_interp = 0
        self.current_normal_interp_img = None
        self.current_tangent_interp_img = None
        
        # Try to create OpenCV windows only if enabled
        self.use_cv_window = False
        if enable_window:
            try:
                cv2.namedWindow("Normal Force (Interpolated)", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Normal Force (Interpolated)", 512, 512)
                cv2.setMouseCallback("Normal Force (Interpolated)", self._mouse_callback_normal)
                
                cv2.namedWindow("Tangential Force (Interpolated)", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Tangential Force (Interpolated)", 512, 512)
                cv2.setMouseCallback("Tangential Force (Interpolated)", self._mouse_callback_tangent)
                
                self.use_cv_window = True
                print("[InterpolatedForceVis] OpenCV windows created")
                print(f"[InterpolatedForceVis] Click to save to: {save_dir}")
            except cv2.error:
                print("[InterpolatedForceVis] OpenCV windows not available, saving to file instead.")
        
        self.frame_count = 0
        
        # Create regular grid coordinates
        self.grid_x, self.grid_y = np.meshgrid(
            np.linspace(0, 1, grid_size[1]),
            np.linspace(0, 1, grid_size[0])
        )
    
    def _mouse_callback_normal(self, event, x, y, flags, param):
        """Save normal force (interpolated) image on click."""
        if event == cv2.EVENT_LBUTTONDOWN and self.current_normal_interp_img is not None:
            filename = f"normal_force_interpolated_{self.save_counter_normal_interp:04d}.png"
            filepath = self.save_dir / filename
            if cv2.imwrite(str(filepath), self.current_normal_interp_img):
                self.save_counter_normal_interp += 1
                print(f"[InterpolatedForceVis] ✓ Saved: {filepath}")
    
    def _mouse_callback_tangent(self, event, x, y, flags, param):
        """Save tangential force (interpolated) image on click."""
        if event == cv2.EVENT_LBUTTONDOWN and self.current_tangent_interp_img is not None:
            filename = f"tangential_force_interpolated_{self.save_counter_tangent_interp:04d}.png"
            filepath = self.save_dir / filename
            if cv2.imwrite(str(filepath), self.current_tangent_interp_img):
                self.save_counter_tangent_interp += 1
                print(f"[InterpolatedForceVis] ✓ Saved: {filepath}")
    
    def _interpolate_values(self,
                            normal_magnitudes: np.ndarray,
                            tangent_x: np.ndarray,
                            tangent_y: np.ndarray,
                            vertex_positions_2d: np.ndarray,
                            noise_threshold_percentile: float = 10.0) -> tuple:
        """
        Interpolate provided normal/tangential values to regular grid with noise filtering.
        
        Args:
            normal_magnitudes: Normal force/pressure magnitudes
            tangent_x: Tangential force X components
            tangent_y: Tangential force Y components
            vertex_positions_2d: 2D positions of vertices (in pixels)
            noise_threshold_percentile: Percentile threshold for filtering noise (default: 10.0)
                                       Values below this percentile are considered noise and removed
        
        Returns:
            Tuple of (normal_grid, tangent_x_grid, tangent_y_grid)
        """
        if normal_magnitudes is None or len(normal_magnitudes) == 0:
            return (np.zeros(self.grid_size),
                    np.zeros(self.grid_size),
                    np.zeros(self.grid_size))

        if vertex_positions_2d is None or len(vertex_positions_2d) == 0:
            vertex_positions_2d = self._generate_normalized_positions(len(normal_magnitudes))
            self._data_aspect_ratio = 1.0
        else:
            vertex_positions_2d = vertex_positions_2d.copy()
            
            # Calculate aspect ratio from actual data bounds (in pixel coordinates)
            x_min, x_max = vertex_positions_2d[:, 0].min(), vertex_positions_2d[:, 0].max()
            y_min, y_max = vertex_positions_2d[:, 1].min(), vertex_positions_2d[:, 1].max()
            x_range = max(x_max - x_min, 1.0)
            y_range = max(y_max - y_min, 1.0)
            self._data_aspect_ratio = x_range / y_range
            
            # Normalize to [0, 1] for interpolation
            vertex_positions_2d[:, 0] = (vertex_positions_2d[:, 0] - x_min) / x_range
            vertex_positions_2d[:, 1] = (vertex_positions_2d[:, 1] - y_min) / y_range

        # Ensure points and values have matching lengths
        num_points = len(vertex_positions_2d)
        num_values = len(normal_magnitudes)
        if num_points != num_values:
            min_len = min(num_points, num_values)
            vertex_positions_2d = vertex_positions_2d[:min_len]
            normal_magnitudes = normal_magnitudes[:min_len]
            tangent_x = tangent_x[:min_len]
            tangent_y = tangent_y[:min_len]

        # ========== 噪音过滤 ==========
        # 计算法向力的阈值（使用百分位数）
        if len(normal_magnitudes) > 0 and noise_threshold_percentile > 0:
            threshold = np.percentile(normal_magnitudes, noise_threshold_percentile)
            
            # 过滤掉低于阈值的点
            valid_mask = normal_magnitudes >= threshold
            
            # 如果过滤后点太少，保留所有非零点
            if np.sum(valid_mask) < 4:
                valid_mask = normal_magnitudes > 0.0001
            
            # 如果还是太少，保留所有点
            if np.sum(valid_mask) < 4:
                valid_mask = np.ones(len(normal_magnitudes), dtype=bool)
            
            # 应用过滤
            vertex_positions_2d = vertex_positions_2d[valid_mask]
            normal_magnitudes = normal_magnitudes[valid_mask]
            tangent_x = tangent_x[valid_mask]
            tangent_y = tangent_y[valid_mask]
            
            if len(normal_magnitudes) > 0:
                filtered_count = np.sum(~valid_mask)
                if filtered_count > 0 and self.frame_count % 30 == 0:  # 每30帧打印一次
                    print(f"[InterpolatedForceVis] Filtered {filtered_count} noise points (threshold: {threshold:.4f})")

        points = vertex_positions_2d[:, :2]  # (N, 2)

        try:
            normal_grid = griddata(
                points, normal_magnitudes,
                (self.grid_x, self.grid_y),
                method='cubic',
                fill_value=0.0
            )
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

        normal_grid = np.maximum(normal_grid, 0.0)
        return normal_grid, tangent_x_grid, tangent_y_grid

    def interpolate_forces(self, 
                          force_data: "ContactForceData",
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
        
        # Extract pressure magnitudes (betterforce only)
        normal_magnitudes = []
        tangent_x = []
        tangent_y = []
        normal_pressure_magnitudes = force_data.normal_pressure_magnitudes
        tangential_pressures = force_data.tangential_pressures
        if normal_pressure_magnitudes is None or tangential_pressures is None:
            return (np.zeros(self.grid_size),
                    np.zeros(self.grid_size),
                    np.zeros(self.grid_size))
        use_pressure = True
        
        for i in range(len(force_data.vertex_indices)):
            # Normal force magnitude
            if i < len(normal_pressure_magnitudes):
                normal_magnitudes.append(normal_pressure_magnitudes[i])
            else:
                normal_magnitudes.append(0.0)
            
            # Tangential force components
            if i < len(tangential_pressures):
                tangent = tangential_pressures[i]
                tangent_x.append(tangent[0])
                tangent_y.append(tangent[1])
            else:
                tangent_x.append(0.0)
                tangent_y.append(0.0)
        
        normal_magnitudes = np.array(normal_magnitudes)
        tangent_x = np.array(tangent_x)
        tangent_y = np.array(tangent_y)

        normal_grid, tangent_x_grid, tangent_y_grid = self._interpolate_values(
            normal_magnitudes, tangent_x, tangent_y, vertex_positions_2d,
            noise_threshold_percentile=self.noise_threshold_percentile
        )
        self._use_pressure = use_pressure
        return normal_grid, tangent_x_grid, tangent_y_grid

    def visualize_from_values(self,
                              normal_magnitudes: np.ndarray,
                              tangential_vectors: np.ndarray,
                              vertex_positions_2d: np.ndarray = None):
        """Visualize interpolated fields directly from values."""
        if normal_magnitudes is None or tangential_vectors is None:
            if self.frame_count == 0:
                print("[InterpolatedForceVis] No data to visualize (normal_magnitudes or tangential_vectors is None)")
            return
        
        if len(normal_magnitudes) == 0 or len(tangential_vectors) == 0:
            if self.frame_count == 0:
                print("[InterpolatedForceVis] Empty data arrays")
            return
        
        if self.frame_count == 0:
            print(f"[InterpolatedForceVis] Starting visualization with {len(normal_magnitudes)} data points")
            print(f"[InterpolatedForceVis] Window enabled: {self.use_cv_window}")
        
        tangent_x = tangential_vectors[:, 0] if len(tangential_vectors) > 0 else np.array([])
        tangent_y = tangential_vectors[:, 1] if len(tangential_vectors) > 0 else np.array([])
        normal_grid, tangent_x_grid, tangent_y_grid = self._interpolate_values(
            normal_magnitudes, tangent_x, tangent_y, vertex_positions_2d,
            noise_threshold_percentile=self.noise_threshold_percentile
        )
        normal_img = self._visualize_normal_forces(normal_grid)
        tangent_img = self._visualize_tangent_forces(tangent_x_grid, tangent_y_grid)
        
        # Store current images for saving
        self.current_normal_interp_img = normal_img.copy()
        self.current_tangent_interp_img = tangent_img.copy()
        
        self.frame_count += 1
        if self.use_cv_window:
            # 上下镜像翻转
            normal_img_flipped = cv2.flip(normal_img, 0)
            tangent_img_flipped = cv2.flip(tangent_img, 0)
            cv2.imshow("Normal Force (Interpolated)", normal_img_flipped)
            cv2.imshow("Tangential Force (Interpolated)", tangent_img_flipped)
            cv2.waitKey(1)
            if self.frame_count == 1:
                print("[InterpolatedForceVis] First frame displayed in windows (vertically flipped)")
        else:
            if self.frame_count % 10 == 0:
                # 保存时也进行上下镜像
                cv2.imwrite(self.save_path_normal, cv2.flip(normal_img, 0))
                cv2.imwrite(self.save_path_tangent, cv2.flip(tangent_img, 0))
                if self.frame_count == 10:
                    print(f"[InterpolatedForceVis] Saving to files (vertically flipped): {self.save_path_normal}, {self.save_path_tangent}")
    
    def visualize(self, 
                  force_data: "ContactForceData",
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
        
        # Store current images for saving
        self.current_normal_interp_img = normal_img.copy()
        self.current_tangent_interp_img = tangent_img.copy()
        
        # Display or save
        self.frame_count += 1
        if self.use_cv_window:
            # 上下镜像翻转
            normal_img_flipped = cv2.flip(normal_img, 0)
            tangent_img_flipped = cv2.flip(tangent_img, 0)
            cv2.imshow("Normal Force (Interpolated)", normal_img_flipped)
            cv2.imshow("Tangential Force (Interpolated)", tangent_img_flipped)
            cv2.waitKey(1)
        else:
            # Save to file every 10 frames (with vertical flip)
            if self.frame_count % 10 == 0:
                cv2.imwrite(self.save_path_normal, cv2.flip(normal_img, 0))
                cv2.imwrite(self.save_path_tangent, cv2.flip(tangent_img, 0))
    
    def _visualize_normal_forces(self, normal_grid: np.ndarray) -> np.ndarray:
        """
        Visualize normal forces as circles on a grid.
        
        Args:
            normal_grid: Grid of normal force magnitudes (grid_size)
            
        Returns:
            Visualization image
        """
        # Calculate aspect ratio from grid data bounds (stored during interpolation)
        aspect_ratio = getattr(self, '_data_aspect_ratio', 1.0)
        
        # Adjust image size to match data aspect ratio
        base_size = 512
        if aspect_ratio > 1.0:
            # Wider than tall
            img_w = base_size
            img_h = int(base_size / aspect_ratio)
        else:
            # Taller than wide
            img_h = base_size
            img_w = int(base_size * aspect_ratio)
        
        # Black background
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        
        # Calculate cell size
        cell_w = img_w / self.grid_size[1]
        cell_h = img_h / self.grid_size[0]
        
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
                
                # Draw white outline for visibility on black background
                cv2.circle(img, (cx, cy), radius, (255, 255, 255), 1)
        
        # No statistics overlay
        
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
        # Calculate aspect ratio from grid data bounds (stored during interpolation)
        aspect_ratio = getattr(self, '_data_aspect_ratio', 1.0)
        
        # Adjust image size to match data aspect ratio
        base_size = 512
        if aspect_ratio > 1.0:
            # Wider than tall
            img_w = base_size
            img_h = int(base_size / aspect_ratio)
        else:
            # Taller than wide
            img_h = base_size
            img_w = int(base_size * aspect_ratio)
        
        # Black background
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        
        # Calculate cell size
        cell_w = img_w / self.grid_size[1]
        cell_h = img_h / self.grid_size[0]
        
        # Calculate tangent force magnitudes
        tangent_mag = np.sqrt(tangent_x_grid**2 + tangent_y_grid**2)
        max_force = np.max(tangent_mag)
        if max_force < 0.0001:
            max_force = 1.0
        
        # Draw arrows for each grid cell
        arrow_scale = min(cell_w, cell_h) * 0.8 * 1.5  # 1.5x longer arrows for better visibility
        
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
                
                # Color based on force magnitude (red scale) - bright on black background
                intensity = int(255 * normalized_force)
                color = (0, 0, intensity)  # Red
                
                # Draw arrow with thicker line for better visibility
                if force_mag > 0.001:  # Only draw significant forces
                    cv2.arrowedLine(img, (cx, cy), (ex, ey), color, 3, tipLength=0.3)
        
        # No statistics overlay
        
        return img
    
    def _add_stats_normal(self, img: np.ndarray, normal_grid: np.ndarray, max_force: float):
        """Add statistics overlay for normal forces/pressures."""
        # Semi-transparent dark background for better visibility on black background
        cv2.rectangle(img, (5, 5), (250, 80), (50, 50, 50), -1)
        cv2.rectangle(img, (5, 5), (250, 80), (255, 255, 255), 2)
        
        mean_force = np.mean(normal_grid)
        nonzero_count = np.count_nonzero(normal_grid > 0.0001)
        
        title = "Normal Pressure (Interpolated)" if getattr(self, "_use_pressure", False) else "Normal Force (Interpolated)"
        cv2.putText(img, title, (15, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, f"Grid: {self.grid_size[0]}x{self.grid_size[1]}", (15, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        unit = "Pa" if getattr(self, "_use_pressure", False) else "N"
        cv2.putText(img, f"Max: {max_force:.3f} {unit}", (15, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, f"Active cells: {nonzero_count}", (15, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _add_stats_tangent(self, img: np.ndarray, tangent_mag: np.ndarray, max_force: float):
        """Add statistics overlay for tangential forces/pressures."""
        # Semi-transparent dark background for better visibility on black background
        cv2.rectangle(img, (5, 5), (250, 80), (50, 50, 50), -1)
        cv2.rectangle(img, (5, 5), (250, 80), (255, 255, 255), 2)
        
        mean_force = np.mean(tangent_mag)
        nonzero_count = np.count_nonzero(tangent_mag > 0.0001)
        
        title = "Tangential Pressure (Interpolated)" if getattr(self, "_use_pressure", False) else "Tangential Force (Interpolated)"
        cv2.putText(img, title, (15, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, f"Grid: {self.grid_size[0]}x{self.grid_size[1]}", (15, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        unit = "Pa" if getattr(self, "_use_pressure", False) else "N"
        cv2.putText(img, f"Max: {max_force:.3f} {unit}", (15, 60),
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
                          force_data: "ContactForceData",
                          vertex_positions_2d: np.ndarray = None) -> np.ndarray:
        """
        Draw force arrows on the tactile image.
        
        Visualization:
        - Tangential forces: Red arrows (direction and magnitude)
        - Normal magnitude: Colored dots (color = magnitude)
        
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
        if (
            vertex_positions_2d is None
            or len(vertex_positions_2d) == 0
            or len(vertex_positions_2d) != len(force_data.vertex_indices)
        ):
            num_vertices = len(force_data.vertex_indices)
            vertex_positions_2d = self._generate_vertex_positions(num_vertices)
        
        # Draw visualization for each contact point
        normal_magnitudes = force_data.normal_pressure_magnitudes
        tangent_magnitudes = force_data.tangential_pressure_magnitudes
        tangential_pressures = force_data.tangential_pressures
        if normal_magnitudes is None or tangent_magnitudes is None or tangential_pressures is None:
            cv2.putText(vis_image, "No Pressure Data", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return vis_image
        use_pressure = True
        max_normal = np.max(normal_magnitudes) if len(normal_magnitudes) > 0 else 0.0
        max_tangent = np.max(tangent_magnitudes) if len(tangent_magnitudes) > 0 else 0.0
        num_points = len(force_data.vertex_indices)
        if len(vertex_positions_2d) != num_points:
            vertex_positions_2d = self._generate_vertex_positions(num_points)

        for i in range(num_points):
            if i >= len(vertex_positions_2d):
                break
                
            pos = vertex_positions_2d[i]
            x, y = int(pos[0]), int(pos[1])
            
            # Skip if outside image
            if x < 0 or x >= self.image_size[0] or y < 0 or y >= self.image_size[1]:
                continue
            
            # Draw tangential force arrow (red) if available
            if i < len(tangential_pressures):
                tangent = tangential_pressures[i]
                tangent_mag = tangent_magnitudes[i] if i < len(tangent_magnitudes) else np.linalg.norm(tangent)
                normal_mag = normal_magnitudes[i] if i < len(normal_magnitudes) else 0.0
                color = self._map_scalar_to_bgr(normal_mag, max_normal)
                self._draw_tangent_arrow(vis_image, x, y, tangent, color=color, thickness=3)
            
            # Draw normal magnitude as colored dot (color = magnitude)
            if i < len(normal_magnitudes):
                normal_mag = normal_magnitudes[i]
                color = self._map_scalar_to_bgr(normal_mag, max_normal)
                cv2.circle(vis_image, (x, y), 4, color, -1)
        
        # Add legend and statistics
        self._add_legend(vis_image, force_data, use_pressure=use_pressure)
        
        return vis_image
    
    def _draw_tangent_arrow(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        tangent: np.ndarray,
        color: tuple | None = None,
        thickness: int = 2,
    ):
        """Draw tangential pressure arrow (color = normal magnitude)."""
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
            arrow_color = color if color is not None else self.tangent_color
            cv2.arrowedLine(image, (x, y), (end_x, end_y), arrow_color, thickness, tipLength=0.3)
    
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

    @staticmethod
    def _map_scalar_to_bgr(value: float, max_value: float) -> tuple:
        """Map a scalar to a blue->green->red color."""
        if max_value <= 1e-12:
            return (255, 255, 255)
        ratio = np.clip(value / max_value, 0.0, 1.0)
        if ratio < 0.5:
            r = 0
            g = int(255 * ratio * 2.0)
            b = int(255 * (1.0 - ratio * 2.0))
        else:
            r = int(255 * (ratio - 0.5) * 2.0)
            g = int(255 * (1.0 - (ratio - 0.5) * 2.0))
            b = 0
        return (b, g, r)
    
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
    
    def _add_legend(self, image: np.ndarray, force_data: "ContactForceData", use_pressure: bool = False):
        """Add legend and statistics to the image."""
        h, w = image.shape[:2]
        
        # Legend box
        cv2.rectangle(image, (5, h - 80), (200, h - 5), (50, 50, 50), -1)
        cv2.rectangle(image, (5, h - 80), (200, h - 5), (255, 255, 255), 1)
        
        # Legend items
        y_offset = h - 65
        
        # Tangential force arrow (color = magnitude)
        cv2.arrowedLine(image, (15, y_offset), (40, y_offset), (0, 0, 255), 2)
        tangent_label = "Arrow Color = Normal Pressure"
        cv2.putText(image, tangent_label, (50, y_offset + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_offset += 20
        
        # Normal magnitude color legend (blue -> red)
        cv2.rectangle(image, (15, y_offset - 6), (35, y_offset + 6), (255, 0, 0), -1)
        cv2.rectangle(image, (35, y_offset - 6), (55, y_offset + 6), (0, 0, 255), -1)
        normal_label = "Normal Mag (Color)" if not use_pressure else "Normal Pressure (Color)"
        cv2.putText(image, normal_label, (65, y_offset + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.putText(image, "(color = magnitude)", (15, y_offset), 
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

            if use_pressure and force_data.pressure_magnitudes is not None:
                max_pressure = np.max(force_data.pressure_magnitudes)
                cv2.putText(image, f"Max Pressure: {max_pressure:.3f} Pa", (15, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(image, f"Max Force: {max_force:.3f} N", (15, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(image, f"Total: [{total_force[0]:.2f}, {total_force[1]:.2f}, {total_force[2]:.2f}] N", 
                           (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            if use_pressure and force_data.normal_pressure_magnitudes is not None:
                max_normal = np.max(force_data.normal_pressure_magnitudes)
                max_tangent = np.max(force_data.tangential_pressure_magnitudes)
                cv2.putText(image, f"Max Normal: {max_normal:.2f}Pa, Tangent: {max_tangent:.2f}Pa", 
                           (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            elif force_data.normal_force_magnitudes is not None:
                max_normal = np.max(force_data.normal_force_magnitudes)
                max_tangent = np.max(force_data.tangential_force_magnitudes)
                cv2.putText(image, f"Max Normal: {max_normal:.2f}N, Tangent: {max_tangent:.2f}N", 
                           (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    def show(self, image: np.ndarray):
        """Display the visualization."""
        self.frame_count += 1
        if self.use_cv_window:
            # 顺时针旋转90度
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            cv2.imshow("Contact Forces", rotated_image)
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


