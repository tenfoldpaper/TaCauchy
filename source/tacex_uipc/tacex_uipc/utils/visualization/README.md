# UIPC Visualization Utilities

This package provides modular visualization tools for UIPC simulation, including contact forces, mesh deformation, and 3D mesh structure visualization.

## Module Structure

```
tacex_uipc/utils/visualization/
├── __init__.py                    # Package exports
├── force_visualizer.py            # 2D force visualization
├── deformation_visualizer.py      # Mesh deformation heatmap
├── mesh_visualizer.py             # 3D mesh and force visualization
├── surface_filter.py              # Surface filtering utilities
└── README.md                      # This file
```

## Modules

### 1. force_visualizer.py

Provides 2D force visualization on tactile sensor images.

**Classes:**
- `ForceVisualizer`: Direct force visualization with arrows and circles
- `InterpolatedForceVisualizer`: Interpolated forces on regular grids

**Usage:**
```python
from tacex_uipc.utils.visualization import ForceVisualizer

visualizer = ForceVisualizer(image_size=(512, 512), force_scale=0.1)
vis_image = visualizer.draw_force_arrows(tactile_image, force_data)
visualizer.show(vis_image)
visualizer.close()
```

**Features:**
- Red arrows for tangential forces
- Green circles for normal forces
- Automatic scaling and normalization
- Statistics overlay
- OpenCV window or file saving

### 2. deformation_visualizer.py

Visualizes mesh deformation with color-coded heatmap.

**Classes:**
- `DeformationVisualizer`: Heatmap visualization of vertex displacement

**Usage:**
```python
from tacex_uipc.utils.visualization import DeformationVisualizer

visualizer = DeformationVisualizer(image_size=(512, 512))
deformation_img = visualizer.visualize_deformation(
    current_positions,
    initial_positions,
    attachment_indices
)
visualizer.show(deformation_img)
visualizer.close()
```

**Features:**
- Color map: Blue (no deformation) → Green → Yellow → Red (max)
- Statistics overlay (mean, max deformation)
- Color bar legend
- Attachment point highlighting

### 3. mesh_visualizer.py

Provides 3D visualization in Isaac Sim scene using debug draw.

**Functions:**
- `draw_3d_forces()`: Draw force arrows in 3D
- `draw_tet_mesh()`: Draw tetrahedral mesh structure
- `draw_ball_mesh()`: Draw object mesh structure
- `draw_attachment_debug()`: Draw attachment constraints

**Usage:**
```python
from tacex_uipc.utils.visualization import (
    draw_3d_forces,
    draw_tet_mesh,
    draw_ball_mesh
)

# Draw 3D force arrows
draw_3d_forces(force_data, vertex_positions, force_scale=0.01)

# Draw tet mesh structure (cyan wireframe)
draw_tet_mesh(env, debug=True)

# Draw object mesh (yellow/orange)
draw_ball_mesh(env, debug=True)
```

**Features:**
- Yellow arrows for total forces
- Green arrows for normal forces
- Red arrows for tangential forces
- Cyan wireframe for tet mesh
- Orange wireframe for object mesh

### 4. surface_filter.py

Utilities for filtering vertices based on position.

**Functions:**
- `filter_contact_surface_vertices()`: Filter to contact surface only
- `get_surface_statistics()`: Get filtering statistics

**Usage:**
```python
from tacex_uipc.utils.visualization import filter_contact_surface_vertices

# Filter to bottom 30% of vertices (contact surface)
contact_pos, contact_idx, mask = filter_contact_surface_vertices(
    all_positions,
    attachment_indices,
    z_threshold_percentile=30.0
)

print(f"Contact surface: {len(contact_pos)} / {len(all_positions)} vertices")
```

**Features:**
- Z-coordinate based filtering
- Attachment point exclusion
- Configurable threshold
- Returns positions, indices, and mask

## Complete Example

```python
from tacex_uipc.utils.visualization import (
    ForceVisualizer,
    DeformationVisualizer,
    InterpolatedForceVisualizer,
    draw_3d_forces,
    draw_tet_mesh,
    filter_contact_surface_vertices,
)

# Initialize visualizers
force_vis = ForceVisualizer(image_size=(512, 512))
deform_vis = DeformationVisualizer(image_size=(512, 512))
interp_vis = InterpolatedForceVisualizer(grid_size=(20, 20))

# Get force data
force_data = force_extractor.get_object_forces(gelpad)

# 3D visualization
draw_3d_forces(force_data, vertex_positions, force_scale=0.01)
draw_tet_mesh(env, debug=True)

# 2D force visualization
vis_image = force_vis.draw_force_arrows(tactile_image, force_data)
force_vis.show(vis_image)

# Interpolated forces
interp_vis.visualize(force_data)

# Deformation visualization (contact surface only)
contact_pos, _, mask = filter_contact_surface_vertices(
    current_positions,
    attachment_indices
)
initial_contact = initial_positions[mask]
deform_img = deform_vis.visualize_deformation(contact_pos, initial_contact)
deform_vis.show(deform_img)

# Cleanup
force_vis.close()
deform_vis.close()
interp_vis.close()
```

## Benefits of Modular Structure

1. **Reusability**: Use visualization components in different scripts
2. **Maintainability**: Each module has a single responsibility
3. **Testability**: Easy to test individual components
4. **Extensibility**: Add new visualizers without modifying existing code
5. **Clean Code**: Main scripts focus on logic, not visualization details

## Integration with Existing Code

To use these utilities in your existing scripts:

```python
# Old way (everything in one file)
# - 2500+ lines in single file
# - Hard to maintain
# - Difficult to reuse

# New way (modular)
from tacex_uipc.utils.visualization import (
    ForceVisualizer,
    DeformationVisualizer,
    draw_3d_forces,
    draw_tet_mesh,
)

# Your main script is now much cleaner!
```

## Dependencies

- `numpy`: Array operations
- `cv2` (OpenCV): Image visualization
- `scipy`: Interpolation (for InterpolatedForceVisualizer)
- Isaac Sim debug draw interface (for 3D visualization)

## Notes

- All visualizers handle missing OpenCV windows gracefully (fall back to file saving)
- 3D visualization requires Isaac Sim debug draw interface
- Surface filtering assumes Z-axis points upward
- Force scale parameters may need adjustment based on your simulation

## Future Enhancements

Potential additions:
- Real-time plotting with matplotlib
- Video recording capabilities
- Custom color maps
- Interactive visualization controls
- Performance profiling overlays
