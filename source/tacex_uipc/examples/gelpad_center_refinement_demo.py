"""
Gelpad Center Refinement Demo

This script demonstrates how to use the center region mesh refinement feature
for gelpad simulation. The center region will have denser mesh for better
deformation capture.

Usage:
    python gelpad_center_refinement_demo.py
"""

import argparse
import numpy as np
from omni.isaac.lab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Gelpad center refinement demo")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument(
    "--refinement_level",
    type=str,
    default="medium",
    choices=["low", "medium", "high", "custom"],
    help="Mesh refinement level preset",
)
parser.add_argument("--center_ratio", type=float, default=0.3, help="Center region ratio (0.0-1.0)")
parser.add_argument("--center_edge_length", type=float, default=0.25, help="Center edge length ratio")
parser.add_argument("--visualize_mesh", action="store_true", help="Visualize the mesh structure")

args_cli = parser.parse_args()

# Launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after app launch
from omni.isaac.lab.sim import SimulationContext
from pxr import UsdGeom
import omni.usd

from tacex_uipc.utils.mesh_gen import TetMeshCfg, MeshGenerator


def get_refinement_preset(level: str) -> dict:
    """Get mesh refinement configuration preset."""
    presets = {
        "low": {
            "enable_center_refinement": True,
            "center_region_ratio": 0.2,
            "center_edge_length_r": 1 / 2.5,
            "edge_length_r": 1 / 1.5,
            "refinement_transition_ratio": 0.1,
        },
        "medium": {
            "enable_center_refinement": True,
            "center_region_ratio": 0.3,
            "center_edge_length_r": 1 / 4,
            "edge_length_r": 1 / 2,
            "refinement_transition_ratio": 0.1,
        },
        "high": {
            "enable_center_refinement": True,
            "center_region_ratio": 0.4,
            "center_edge_length_r": 1 / 5,
            "edge_length_r": 1 / 3,
            "refinement_transition_ratio": 0.15,
        },
    }
    return presets.get(level, presets["medium"])


def create_simple_gelpad(stage, prim_path: str, size: float = 0.02):
    """Create a simple gelpad mesh for testing."""
    # Create a cylinder as gelpad
    from pxr import UsdGeom, Gf

    geom_prim = UsdGeom.Cylinder.Define(stage, prim_path)
    geom_prim.GetRadiusAttr().Set(size)
    geom_prim.GetHeightAttr().Set(size * 0.3)
    geom_prim.GetAxisAttr().Set("Z")

    # Convert to mesh
    from omni.physx.scripts import deformableUtils

    mesh_prim = UsdGeom.Mesh.Define(stage, prim_path + "_mesh")

    # Get cylinder mesh data
    cylinder = geom_prim.GetPrim()
    # Note: In production, you would use proper mesh conversion
    # For this demo, we'll create a simple mesh manually

    return mesh_prim


def visualize_tet_mesh(tet_points, tet_indices, center_region_ratio=0.3):
    """Visualize the tetrahedral mesh structure."""
    from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()

    # Calculate center
    bbox_min = np.min(tet_points, axis=0)
    bbox_max = np.max(tet_points, axis=0)
    center = (bbox_min + bbox_max) / 2
    bbox_size = bbox_max - bbox_min
    max_distance = np.linalg.norm(bbox_size) / 2
    center_threshold = max_distance * center_region_ratio

    # Draw tetrahedra with different colors for center and outer regions
    for i in range(0, len(tet_indices), 4):
        tet_idx = tet_indices[i : i + 4]
        tet_verts = [tet_points[idx] for idx in tet_idx]

        # Calculate tet center
        tet_center = np.mean(tet_verts, axis=0)
        dist_to_center = np.linalg.norm(tet_center - center)

        # Choose color based on region
        if dist_to_center < center_threshold:
            color = [(255, 0, 0, 0.8)]  # Red for center region
            width = 3
        else:
            color = [(0, 0, 255, 0.5)]  # Blue for outer region
            width = 2

        # Draw tetrahedron edges
        draw.draw_lines([tet_verts[0]] * 3, tet_verts[1:], color * 3, [width] * 3)
        draw.draw_lines([tet_verts[1]] * 2, tet_verts[2:], color * 2, [width] * 2)
        draw.draw_lines([tet_verts[2]], [tet_verts[3]], color, [width])

    print(f"Visualization: Red = center region, Blue = outer region")


def main():
    """Main function."""
    print("\n" + "=" * 80)
    print("Gelpad Center Refinement Demo")
    print("=" * 80)

    # Create simulation context
    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        backend="numpy",
    )

    # Get stage
    stage = omni.usd.get_context().get_stage()

    # Create a simple test mesh (in production, you would load your gelpad model)
    print("\n[1/4] Creating test gelpad mesh...")
    gelpad_path = "/World/GelPad"

    # For this demo, we'll use a simple approach
    # In production, you would load your actual gelpad USD file
    from pxr import UsdGeom, Gf

    # Create a simple mesh for demonstration
    mesh_prim = UsdGeom.Mesh.Define(stage, gelpad_path)

    # Create a simple grid mesh
    resolution = 10
    size = 0.02
    vertices = []
    for i in range(resolution):
        for j in range(resolution):
            x = (i / (resolution - 1) - 0.5) * size
            y = (j / (resolution - 1) - 0.5) * size
            z = 0.0
            vertices.append(Gf.Vec3f(x, y, z))

    # Create triangles
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx = i * resolution + j
            # Triangle 1
            faces.extend([idx, idx + 1, idx + resolution])
            # Triangle 2
            faces.extend([idx + 1, idx + resolution + 1, idx + resolution])

    mesh_prim.GetPointsAttr().Set(vertices)
    mesh_prim.GetFaceVertexCountsAttr().Set([3] * (len(faces) // 3))
    mesh_prim.GetFaceVertexIndicesAttr().Set(faces)

    print(f"Created test mesh with {len(vertices)} vertices and {len(faces)//3} triangles")

    # Configure mesh generation
    print("\n[2/4] Configuring mesh generation...")

    if args_cli.refinement_level == "custom":
        print(f"Using custom parameters:")
        print(f"  - Center region ratio: {args_cli.center_ratio}")
        print(f"  - Center edge length: {args_cli.center_edge_length}")
        cfg = TetMeshCfg(
            enable_center_refinement=True,
            center_region_ratio=args_cli.center_ratio,
            center_edge_length_r=args_cli.center_edge_length,
            edge_length_r=0.5,
            log_level=0,  # Verbose logging
        )
    else:
        print(f"Using preset: {args_cli.refinement_level}")
        preset = get_refinement_preset(args_cli.refinement_level)
        print(f"  - Center region ratio: {preset['center_region_ratio']}")
        print(f"  - Center edge length: {preset['center_edge_length_r']}")
        print(f"  - Edge length: {preset['edge_length_r']}")
        cfg = TetMeshCfg(**preset, log_level=0)

    # Generate tetrahedral mesh
    print("\n[3/4] Generating tetrahedral mesh...")
    mesh_gen = MeshGenerator(config=cfg)

    try:
        tet_points, tet_indices, surf_points, surf_indices = mesh_gen.generate_tet_mesh_for_prim(mesh_prim)

        print(f"\nMesh generation successful!")
        print(f"  - Tetrahedral vertices: {len(tet_points)}")
        print(f"  - Tetrahedra: {len(tet_indices) // 4}")
        print(f"  - Surface vertices: {len(surf_points)}")
        print(f"  - Surface triangles: {len(surf_indices) // 3}")

        # Calculate mesh density statistics
        center = np.mean(tet_points, axis=0)
        distances = [np.linalg.norm(p - center) for p in tet_points]
        bbox_size = np.max(tet_points, axis=0) - np.min(tet_points, axis=0)
        max_dist = np.linalg.norm(bbox_size) / 2
        center_threshold = max_dist * cfg.center_region_ratio

        center_verts = sum(1 for d in distances if d < center_threshold)
        outer_verts = len(tet_points) - center_verts

        print(f"\nMesh density distribution:")
        print(f"  - Center region vertices: {center_verts} ({center_verts/len(tet_points)*100:.1f}%)")
        print(f"  - Outer region vertices: {outer_verts} ({outer_verts/len(tet_points)*100:.1f}%)")

        # Visualize if requested
        if args_cli.visualize_mesh:
            print("\n[4/4] Visualizing mesh structure...")
            print("Red edges = center region (denser)")
            print("Blue edges = outer region (coarser)")
            visualize_tet_mesh(tet_points, tet_indices, cfg.center_region_ratio)

            # Keep simulation running for visualization
            print("\nPress Ctrl+C to exit...")
            while simulation_app.is_running():
                sim.step()

    except Exception as e:
        print(f"\nError during mesh generation: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        simulation_app.close()
