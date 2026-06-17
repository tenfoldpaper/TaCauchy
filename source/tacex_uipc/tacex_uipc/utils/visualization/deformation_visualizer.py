"""
Deformation visualization for UIPC mesh objects.

This module provides a visualizer for displaying mesh deformation as a heatmap.
"""

import cv2
import numpy as np
from typing import Optional


class DeformationVisualizer:
    """Visualizes mesh deformation with color-coded heatmap."""
    
    def __init__(self,
                 image_size: tuple = (512, 512),
                 save_path: str = "/tmp/deformation_visualization.png",
                 enable_window: bool = False):
        """
        Initialize deformation visualizer.
        
        Args:
            image_size: Size of the visualization image (width, height)
            save_path: Path to save visualization images when window is not available
        """
        self.image_size = image_size
        self.save_path = save_path
        
        # Try to create OpenCV window (optional)
        self.use_cv_window = False
        if enable_window:
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
                             attachment_indices: Optional[list] = None,
                             remove_rigid_motion: bool = True) -> np.ndarray:
        """
        Visualize mesh deformation as a heatmap.
        
        Args:
            current_positions: Current vertex positions (N, 3)
            initial_positions: Initial vertex positions (N, 3)
            attachment_indices: Indices of attachment points (optional)
            remove_rigid_motion: Remove global rigid motion before computing deformation
            
        Returns:
            Visualization image (BGR format for OpenCV)
        """
        # Calculate deformation magnitude for each vertex
        deformation, _ = self.compute_deformation(
            current_positions=current_positions,
            initial_positions=initial_positions,
            remove_rigid_motion=remove_rigid_motion,
        )
        
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

    def compute_deformation(self,
                            current_positions: np.ndarray,
                            initial_positions: np.ndarray,
                            remove_rigid_motion: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute per-vertex deformation magnitudes.

        Args:
            current_positions: Current vertex positions (N, 3)
            initial_positions: Initial vertex positions (N, 3)
            remove_rigid_motion: Remove global rigid motion before computing deformation

        Returns:
            (deformation, aligned_initial_positions)
        """
        if remove_rigid_motion and current_positions.shape[0] >= 3:
            aligned_initial = self._align_initial_to_current(initial_positions, current_positions)
        else:
            aligned_initial = initial_positions

        deformation = np.linalg.norm(current_positions - aligned_initial, axis=1)
        return deformation, aligned_initial

    @staticmethod
    def _align_initial_to_current(initial_positions: np.ndarray,
                                  current_positions: np.ndarray) -> np.ndarray:
        """
        Align initial positions to current positions using rigid Kabsch alignment.
        This removes global translation/rotation while preserving shape.
        """
        # Center both point sets
        init_centroid = np.mean(initial_positions, axis=0)
        curr_centroid = np.mean(current_positions, axis=0)
        init_centered = initial_positions - init_centroid
        curr_centered = current_positions - curr_centroid

        # Compute rotation via SVD (Kabsch)
        h = init_centered.T @ curr_centered
        u, _, vt = np.linalg.svd(h)
        r = vt.T @ u.T
        if np.linalg.det(r) < 0:
            vt[-1, :] *= -1
            r = vt.T @ u.T

        # Apply rotation + translation to initial positions
        aligned_initial = (init_centered @ r) + curr_centroid
        return aligned_initial
    
    def _add_statistics(self, img: np.ndarray, deformation: np.ndarray, attachment_indices: Optional[list]):
        """Add statistics overlay to the image."""
        # Stats box
        cv2.rectangle(img, (5, 5), (300, 120), (50, 50, 50), -1)
        cv2.rectangle(img, (5, 5), (300, 120), (255, 255, 255), 1)
        
        # Statistics
        mean_def = np.mean(deformation)
        max_def = np.max(deformation)
        
        y_offset = 25
        cv2.putText(img, "Contact Surface Deformation", (15, y_offset),
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
            # 顺时针旋转90度
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            cv2.imshow("Contact Surface Deformation", rotated_image)
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
