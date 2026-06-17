"""
Visualization utilities for UIPC simulation.

This package provides visualization tools for:
- Contact forces (2D and 3D)
- Mesh deformation
- Tet mesh structure
- Debug visualization
- Simple tactile visualizers (for demos)
"""

from .force_visualizer import ForceVisualizer, InterpolatedForceVisualizer
from .deformation_visualizer import DeformationVisualizer
from .mesh_visualizer import (
    draw_3d_forces,
    draw_tet_mesh,
    draw_ball_mesh,
    draw_glass_mesh,
    draw_attachment_debug,
)
from .surface_filter import filter_contact_surface_vertices
from .simple_visualizers import TactileForceVisualizer, TactileImageViewer

__all__ = [
    "ForceVisualizer",
    "InterpolatedForceVisualizer",
    "DeformationVisualizer",
    "draw_3d_forces",
    "draw_tet_mesh",
    "draw_ball_mesh",
    "draw_glass_mesh",
    "draw_attachment_debug",
    "filter_contact_surface_vertices",
    "TactileForceVisualizer",
    "TactileImageViewer",
]
