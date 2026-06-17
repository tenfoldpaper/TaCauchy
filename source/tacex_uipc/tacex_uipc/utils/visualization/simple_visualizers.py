"""
Simple visualization utilities for tactile sensor demos.

Provides clean, minimal visualizers for:
- Normal and tangential forces
- Tactile RGB images
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional


class TactileForceVisualizer:
    """
    Simple visualizer for normal and tangential forces on tactile sensor.
    Creates two windows showing pressure and traction distributions.
    """
    
    def __init__(self, image_size: tuple = (512, 512), show_legend: bool = True, save_dir: str = "/home/ly/data/hengfei/getdata/force_field"):
        self.image_size = image_size
        self.show_legend = show_legend
        self.use_cv_window = False
        
        # Setup save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_counter_normal = 0
        self.save_counter_tangent = 0
        self.current_normal_img = None
        self.current_tangent_img = None
        
        try:
            cv2.namedWindow("Normal Pressure", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Normal Pressure", 512, 512)
            cv2.setMouseCallback("Normal Pressure", self._mouse_callback_normal)
            
            cv2.namedWindow("Tangential Traction", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Tangential Traction", 512, 512)
            cv2.setMouseCallback("Tangential Traction", self._mouse_callback_tangent)
            
            self.use_cv_window = True
            print("[TactileForceVisualizer] ✓ Windows created")
            print(f"[TactileForceVisualizer] Click to save to: {save_dir}")
        except cv2.error as e:
            print(f"[TactileForceVisualizer] ✗ Could not create windows: {e}")
        
        self.max_normal = 0.0
        self.max_tangent = 0.0
    
    def _mouse_callback_normal(self, event, x, y, flags, param):
        """Save normal pressure image on click."""
        if event == cv2.EVENT_LBUTTONDOWN and self.current_normal_img is not None:
            filename = f"normal_pressure_{self.save_counter_normal:04d}.png"
            filepath = self.save_dir / filename
            if cv2.imwrite(str(filepath), self.current_normal_img):
                self.save_counter_normal += 1
                print(f"[TactileForceVisualizer] ✓ Saved: {filepath}")
    
    def _mouse_callback_tangent(self, event, x, y, flags, param):
        """Save tangential traction image on click."""
        if event == cv2.EVENT_LBUTTONDOWN and self.current_tangent_img is not None:
            filename = f"tangential_traction_{self.save_counter_tangent:04d}.png"
            filepath = self.save_dir / filename
            if cv2.imwrite(str(filepath), self.current_tangent_img):
                self.save_counter_tangent += 1
                print(f"[TactileForceVisualizer] ✓ Saved: {filepath}")
    
    def visualize(self, stress_data, gelpad_positions: np.ndarray):
        """Visualize forces from stress data."""
        if not self.use_cv_window or stress_data is None:
            return
        
        normal_pressure = np.abs(stress_data.normal_pressure) if stress_data.normal_pressure is not None else None
        tangential_traction = stress_data.tangential_traction if stress_data.tangential_traction is not None else None
        
        if normal_pressure is None or tangential_traction is None:
            return
        
        tri_indices = stress_data.triangle_indices
        if tri_indices is None or len(tri_indices) == 0:
            return
        
        # Triangle centers
        v0 = gelpad_positions[tri_indices[:, 0]]
        v1 = gelpad_positions[tri_indices[:, 1]]
        v2 = gelpad_positions[tri_indices[:, 2]]
        tri_centers = (v0 + v1 + v2) / 3.0
        
        # Project to 2D (now returns image size too)
        positions_2d, img_size = self._project_to_2d(tri_centers)
        
        # Create normal pressure image (green circles)
        normal_img = self._draw_scalar_field(positions_2d, normal_pressure, "Normal Pressure", "Pa", "green", img_size)
        
        # Create tangential traction image (red arrows)
        tangent_img = self._draw_tangent_arrows(positions_2d, tangential_traction, "Tangential Traction", "Pa", img_size)
        
        # Store current images for saving
        self.current_normal_img = cv2.rotate(normal_img, cv2.ROTATE_90_CLOCKWISE)
        self.current_tangent_img = cv2.rotate(tangent_img, cv2.ROTATE_90_CLOCKWISE)
        
        # Display (rotate 90 degrees clockwise)
        cv2.imshow("Normal Pressure", self.current_normal_img)
        cv2.imshow("Tangential Traction", self.current_tangent_img)
        cv2.waitKey(1)
    
    def _project_to_2d(self, positions_3d: np.ndarray) -> tuple:
        """
        Project 3D to 2D image coordinates.
        Returns (positions_2d, image_size) where image_size preserves aspect ratio.
        """
        x, y = positions_3d[:, 0], positions_3d[:, 1]
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        # Calculate physical dimensions
        x_range = max(x_max - x_min, 0.001)
        y_range = max(y_max - y_min, 0.001)
        aspect_ratio = x_range / y_range
        
        # Determine image size based on aspect ratio
        base_size = 512
        margin = 50
        
        if aspect_ratio > 1.0:
            # Wider than tall
            img_w = base_size
            img_h = int(base_size / aspect_ratio)
        else:
            # Taller than wide
            img_h = base_size
            img_w = int(base_size * aspect_ratio)
        
        # Calculate available space for drawing
        w = img_w - 2 * margin
        h = img_h - 2 * margin
        
        # Normalize and scale to image coordinates
        x_norm = (x - x_min) / x_range
        y_norm = (y - y_min) / y_range
        
        x_img = (margin + x_norm * w).astype(int)
        y_img = (margin + (1 - y_norm) * h).astype(int)
        
        return np.column_stack([x_img, y_img]), (img_w, img_h)
    
    def _draw_tangent_arrows(self, positions_2d, tangential_traction, title, unit, img_size):
        """Draw tangential traction as red arrows showing direction."""
        # White background
        img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255
        
        if len(tangential_traction) == 0:
            return img
        
        # Extract tangent components and magnitude
        if len(tangential_traction.shape) == 2 and tangential_traction.shape[1] >= 2:
            tangent_xy = tangential_traction[:, :2]
            tangent_mag = np.linalg.norm(tangent_xy, axis=1)
            # Normalize directions
            tangent_directions = np.zeros_like(tangent_xy)
            nonzero_mask = tangent_mag > 0.0001
            tangent_directions[nonzero_mask] = tangent_xy[nonzero_mask] / tangent_mag[nonzero_mask, np.newaxis]
        else:
            tangent_mag = np.abs(tangential_traction)
            tangent_directions = np.zeros((len(tangential_traction), 2))
        
        # Update max value
        max_val = np.max(tangent_mag)
        if max_val > self.max_tangent:
            self.max_tangent = max_val
        max_ref = self.max_tangent
        
        # Sort by magnitude (ascending) so stronger forces are drawn last (on top)
        sorted_indices = np.argsort(tangent_mag)
        
        # Draw arrows in order from weakest to strongest
        for idx in sorted_indices:
            pos = positions_2d[idx]
            mag = tangent_mag[idx]
            direction = tangent_directions[idx]
            
            x, y = int(pos[0]), int(pos[1])
            if x < 0 or x >= img_size[0] or y < 0 or y >= img_size[1]:
                continue
            
            # Normalize magnitude
            norm_mag = mag / max(max_ref, 0.001)
            
            if mag < 0.0001:
                # Very weak traction - draw very light gray dot (visible but subtle)
                cv2.circle(img, (x, y), 2, (220, 220, 220), -1)
            else:
                # Arrow length based on magnitude (larger range for better visibility)
                arrow_length = int(20 + norm_mag * 60)  # 20-80 pixels (更长)
                
                # Arrow thickness based on magnitude (狠狠加粗！)
                thickness = max(3, int(3 + norm_mag * 10))  # 3-13 pixels (原来是 1-6)
                
                # Color intensity based on magnitude (red on white background)
                # Use non-linear mapping to make medium forces more visible
                # Apply power function to make colors darker earlier
                color_intensity = norm_mag ** 0.5  # Square root makes medium values darker
                
                if color_intensity < 0.3:
                    # Very weak: light red/pink
                    intensity = int(255 - color_intensity * 600)  # 75-255
                    color = (intensity, intensity, 255)  # Light red in BGR
                    outline_color = (max(0, intensity - 80), max(0, intensity - 80), 200)  # Darker outline
                else:
                    # Medium to strong: darker red
                    # Map 0.3-1.0 to darker colors
                    intensity = int(255 * (1 - color_intensity * 0.9))  # 255 -> 26
                    color = (0, 0, 255 - intensity)  # Red in BGR, darker as magnitude increases
                    outline_color = (0, 0, max(0, 255 - intensity - 80))  # Even darker outline
                
                # Calculate arrow end point
                # Note: In image coordinates, y increases downward, so we need to flip y direction
                end_x = int(x + direction[0] * arrow_length)
                end_y = int(y - direction[1] * arrow_length)  # Flip y for image coordinates
                
                # Ensure end point is within image bounds
                end_x = max(0, min(img_size[0] - 1, end_x))
                end_y = max(0, min(img_size[1] - 1, end_y))
                
                # Draw arrow FIRST (so circle can cover the arrow's starting point)
                cv2.arrowedLine(img, (x, y), (end_x, end_y), color, thickness, tipLength=0.25)
                
                # Draw circle AFTER arrow (so it covers the arrow's starting point)
                # Use same radius as Normal Pressure for consistency
                circle_radius = int(4 + norm_mag * 18)  # 4-22 pixels (和 Normal 一样)
                # Draw circle outline (darker)
                cv2.circle(img, (x, y), circle_radius + 1, outline_color, -1)
                # Draw circle fill (lighter)
                circle_color = (max(0, 255 - int(norm_mag * 255)), 
                               max(0, 255 - int(norm_mag * 255)), 
                               max(0, 255 - int(norm_mag * 255)))
                cv2.circle(img, (x, y), circle_radius, circle_color, -1)
        
        # No legend
        
        return img
    
    def _add_arrow_legend(self, img, title, values, max_val, unit):
        """Add statistics legend for arrow visualization."""
        cv2.rectangle(img, (5, 5), (350, 140), (50, 50, 50), -1)
        cv2.rectangle(img, (5, 5), (350, 140), (255, 255, 255), 1)
        
        y = 25
        cv2.putText(img, title, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if len(values) > 0:
            num_active = np.sum(values > 0.0001)
            avg_val = np.mean(values[values > 0.0001]) if num_active > 0 else 0.0
            
            y += 25
            cv2.putText(img, f"Active: {num_active}/{len(values)}", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y += 20
            cv2.putText(img, f"Max: {max_val:.2f} {unit}", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y += 20
            cv2.putText(img, f"Avg: {avg_val:.2f} {unit}", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y += 20
            cv2.putText(img, "Arrow: Direction & Magnitude", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def _draw_scalar_field(self, positions_2d, values, title, unit, color_map, img_size):
        """Draw scalar field visualization."""
        # White background
        img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255
        
        if len(values) == 0:
            return img
        
        max_val = np.max(values)
        if color_map == "green":
            if max_val > self.max_normal:
                self.max_normal = max_val
            max_ref = self.max_normal
        else:
            if max_val > self.max_tangent:
                self.max_tangent = max_val
            max_ref = self.max_tangent
        
        # Sort by magnitude (ascending) so stronger forces are drawn last (on top)
        sorted_indices = np.argsort(values)
        
        # Draw circles in order from weakest to strongest
        for idx in sorted_indices:
            pos = positions_2d[idx]
            val = values[idx]
            
            x, y = int(pos[0]), int(pos[1])
            if x < 0 or x >= img_size[0] or y < 0 or y >= img_size[1]:
                continue
            
            # Normalize value
            norm_val = val / max(max_ref, 0.001)
            
            if val < 0.0001:
                # Very weak pressure - draw very light gray dot (visible but subtle)
                cv2.circle(img, (x, y), 2, (220, 220, 220), -1)
            else:
                # Radius based on magnitude (larger range for better visibility)
                radius = int(4 + norm_val * 18)  # 4-22 pixels
                
                # Color intensity based on magnitude (green on white background)
                # Use non-linear mapping to make medium forces more visible
                # Apply power function to make colors darker earlier
                color_intensity = norm_val ** 0.5  # Square root makes medium values darker
                
                if color_intensity < 0.3:
                    # Very weak: light green
                    intensity = int(255 - color_intensity * 600)  # 75-255
                    color = (intensity, 255, intensity)  # Light green in BGR
                else:
                    # Medium to strong: darker green
                    # Map 0.3-1.0 to darker colors
                    intensity = int(255 * (1 - color_intensity * 0.9))  # 255 -> 26
                    color = (0, 255 - intensity, 0)  # Green in BGR, darker as magnitude increases
                
                # Draw filled circle
                cv2.circle(img, (x, y), radius, color, -1)
                
                # Draw darker outline for better visibility
                outline_color = (max(0, intensity - 50), max(0, 255 - intensity - 50), max(0, intensity - 50))
                cv2.circle(img, (x, y), radius, outline_color, 1)
        
        # No legend
        
        return img
    
    def _add_legend(self, img, title, values, max_val, unit):
        """Add statistics legend."""
        cv2.rectangle(img, (5, 5), (350, 120), (50, 50, 50), -1)
        cv2.rectangle(img, (5, 5), (350, 120), (255, 255, 255), 1)
        
        y = 25
        cv2.putText(img, title, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if len(values) > 0:
            num_active = np.sum(values > 0.0001)
            avg_val = np.mean(values[values > 0.0001]) if num_active > 0 else 0.0
            
            y += 25
            cv2.putText(img, f"Active: {num_active}/{len(values)}", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y += 20
            cv2.putText(img, f"Max: {max_val:.2f} {unit}", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y += 20
            cv2.putText(img, f"Avg: {avg_val:.2f} {unit}", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    def close(self):
        """Close windows."""
        if self.use_cv_window:
            try:
                cv2.destroyWindow("Normal Pressure")
                cv2.destroyWindow("Tangential Traction")
            except:
                pass


class TactileImageViewer:
    """Simple viewer for tactile RGB images with click-to-save."""
    
    def __init__(self, window_name: str = "Tactile RGB", save_dir: str = "tactile_images"):
        self.window_name = window_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_cv_window = False
        self.current_image = None
        self.save_counter = 0
        
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 640, 480)
            cv2.setMouseCallback(window_name, self._mouse_callback)
            self.use_cv_window = True
            print(f"[TactileImageViewer] ✓ Window created: '{window_name}'")
            print(f"[TactileImageViewer] Click to save to: {save_dir}")
        except cv2.error as e:
            print(f"[TactileImageViewer] ✗ Could not create window: {e}")
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Save image on click."""
        if event == cv2.EVENT_LBUTTONDOWN and self.current_image is not None:
            filename = f"tactile_{self.save_counter:04d}.png"
            filepath = self.save_dir / filename
            
            img_resized = cv2.resize(self.current_image, (640, 480)) if self.current_image.shape[:2] != (480, 640) else self.current_image
            
            if cv2.imwrite(str(filepath), img_resized):
                self.save_counter += 1
                print(f"[TactileImageViewer] ✓ Saved: {filepath}")
    
    def show(self, image: np.ndarray):
        """Display tactile image."""
        if not self.use_cv_window:
            return
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            display_image = image
        
        if display_image.dtype != np.uint8:
            display_image = (display_image * 255).astype(np.uint8)
        
        self.current_image = display_image.copy()
        
        display_with_text = display_image.copy()
        cv2.putText(display_with_text, "Click to Save", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(self.window_name, display_with_text)
        cv2.waitKey(1)
    
    def close(self):
        """Close window."""
        if self.use_cv_window:
            try:
                cv2.destroyWindow(self.window_name)
            except:
                pass
