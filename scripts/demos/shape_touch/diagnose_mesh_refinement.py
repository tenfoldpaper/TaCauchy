#!/usr/bin/env python3
"""
Diagnostic script to analyze mesh refinement behavior.

This script helps understand why center refinement may not be creating
more surface triangles as expected.
"""

import numpy as np


def analyze_refinement_config(bbox_size, config):
    """
    Analyze the refinement configuration and predict expected behavior.
    
    Args:
        bbox_size: (3,) array of bounding box dimensions [x, y, z]
        config: dict with refinement parameters
    """
    bbox_diag = np.linalg.norm(bbox_size)
    
    # Find thinnest axis
    thin_axis = int(np.argmin(bbox_size))
    thickness = bbox_size[thin_axis]
    planar_dims = [i for i in range(3) if i != thin_axis]
    planar_half_extent = max(bbox_size[d] for d in planar_dims) / 2.0
    
    # Calculate absolute edge lengths
    base_edge_abs = config.get("edge_length_r", 0.1) * bbox_diag
    center_edge_abs = config.get("center_edge_length_r", 0.035) * bbox_diag
    
    # Calculate refinement radii
    center_ratio = config.get("center_region_ratio", 0.35)
    transition_ratio = config.get("refinement_transition_ratio", 0.2)
    r_inner = planar_half_extent * center_ratio
    r_outer = planar_half_extent * (center_ratio + transition_ratio)
    
    print("=" * 70)
    print("MESH REFINEMENT CONFIGURATION ANALYSIS")
    print("=" * 70)
    print(f"\nBounding Box:")
    print(f"  Dimensions: {bbox_size[0]*1e3:.2f} × {bbox_size[1]*1e3:.2f} × {bbox_size[2]*1e3:.2f} mm")
    print(f"  Diagonal: {bbox_diag*1e3:.2f} mm")
    print(f"  Thinnest axis: {'XYZ'[thin_axis]} (thickness: {thickness*1e3:.2f} mm)")
    print(f"  Planar half-extent: {planar_half_extent*1e3:.2f} mm")
    
    print(f"\nEdge Lengths:")
    print(f"  Base edge length:   {base_edge_abs*1e3:.2f} mm (ratio: {config.get('edge_length_r', 0.1):.3f})")
    print(f"  Center edge length: {center_edge_abs*1e3:.2f} mm (ratio: {config.get('center_edge_length_r', 0.035):.3f})")
    print(f"  Refinement factor:  {base_edge_abs/center_edge_abs:.2f}x")
    
    print(f"\nRefinement Zones:")
    print(f"  Center region ratio: {center_ratio:.2f}")
    print(f"  Inner radius (dense): {r_inner*1e3:.2f} mm")
    print(f"  Outer radius (transition end): {r_outer*1e3:.2f} mm")
    print(f"  Transition width: {(r_outer - r_inner)*1e3:.2f} mm")
    
    # Calculate expected triangle counts (rough estimate)
    # For a circular region, area ≈ π * r²
    # Number of triangles ≈ area / (edge_length² * sqrt(3)/4)
    center_area = np.pi * r_inner**2
    transition_area = np.pi * (r_outer**2 - r_inner**2)
    
    # Estimate triangles in each zone
    tri_area_center = center_edge_abs**2 * np.sqrt(3) / 4
    tri_area_base = base_edge_abs**2 * np.sqrt(3) / 4
    
    est_tris_center = center_area / tri_area_center
    est_tris_transition = transition_area / tri_area_base  # Simplified
    
    print(f"\nExpected Surface Triangle Density:")
    print(f"  Center zone area: {center_area*1e6:.2f} mm²")
    print(f"  Estimated triangles in center: ~{est_tris_center:.0f}")
    print(f"  Triangle area (center): {tri_area_center*1e6:.4f} mm²")
    print(f"  Triangle area (base): {tri_area_base*1e6:.4f} mm²")
    
    print(f"\nDiagnostic Recommendations:")
    
    # Check if refinement is aggressive enough
    if base_edge_abs / center_edge_abs < 2.0:
        print("  ⚠ WARNING: Refinement factor < 2x may not create noticeable difference")
        print("    → Try decreasing center_edge_length_r to 0.02 or lower")
    
    if center_ratio < 0.3:
        print("  ⚠ WARNING: Center region ratio < 0.3 may be too small")
        print("    → Try increasing center_region_ratio to 0.4-0.5")
    
    if center_edge_abs > thickness / 2:
        print("  ⚠ WARNING: Center edge length > half thickness")
        print("    → May not create enough resolution through thickness")
    
    # Check if the sizing field will actually affect surface
    if center_edge_abs > base_edge_abs * 0.5:
        print("  ⚠ INFO: Center edge is > 50% of base edge")
        print("    → Surface refinement may be subtle")
    
    print("\n" + "=" * 70)


def suggest_aggressive_config(bbox_size):
    """Suggest a more aggressive refinement configuration."""
    bbox_diag = np.linalg.norm(bbox_size)
    
    print("\nSUGGESTED AGGRESSIVE CONFIGURATION:")
    print("-" * 70)
    
    # More aggressive settings
    configs = {
        "Conservative": {
            "edge_length_r": 0.1,
            "center_edge_length_r": 0.04,
            "center_region_ratio": 0.4,
            "refinement_factor": 2.5
        },
        "Moderate": {
            "edge_length_r": 0.1,
            "center_edge_length_r": 0.025,
            "center_region_ratio": 0.45,
            "refinement_factor": 4.0
        },
        "Aggressive": {
            "edge_length_r": 0.1,
            "center_edge_length_r": 0.015,
            "center_region_ratio": 0.5,
            "refinement_factor": 6.67
        }
    }
    
    for name, cfg in configs.items():
        base_edge = cfg["edge_length_r"] * bbox_diag * 1e3
        center_edge = cfg["center_edge_length_r"] * bbox_diag * 1e3
        print(f"\n{name}:")
        print(f"  edge_length_r: {cfg['edge_length_r']:.3f} ({base_edge:.2f} mm)")
        print(f"  center_edge_length_r: {cfg['center_edge_length_r']:.3f} ({center_edge:.2f} mm)")
        print(f"  center_region_ratio: {cfg['center_region_ratio']:.2f}")
        print(f"  Refinement factor: {cfg['refinement_factor']:.1f}x")


if __name__ == "__main__":
    # Example: GelSight Mini gelpad dimensions
    print("\nExample: GelSight Mini Gelpad")
    print("=" * 70)
    
    # Approximate dimensions from USD analysis
    bbox_size = np.array([0.0254, 0.0208, 0.0039])  # ~25.4 × 20.8 × 3.9 mm
    
    # Current configuration (from user's test)
    current_config = {
        "edge_length_r": 0.1,
        "center_edge_length_r": 0.035,
        "center_region_ratio": 0.35,
        "refinement_transition_ratio": 0.2,
        "enable_center_refinement": True
    }
    
    print("\nCURRENT CONFIGURATION:")
    analyze_refinement_config(bbox_size, current_config)
    
    # Suggest more aggressive configs
    suggest_aggressive_config(bbox_size)
    
    print("\n" + "=" * 70)
    print("INTERPRETATION OF USER'S RESULTS:")
    print("=" * 70)
    print("\nObserved: 356 → 378 surface faces (+22, +6.2%)")
    print("\nPossible reasons for small increase:")
    print("  1. Wildmeshing may prioritize internal tet refinement over surface")
    print("  2. The sizing field affects vertex placement, not directly surface triangulation")
    print("  3. Surface mesh is constrained by the input triangle mesh topology")
    print("  4. The refinement factor (2.86x) may not be aggressive enough")
    print("\nTo see more dramatic surface refinement:")
    print("  • Increase center_region_ratio to 0.5-0.6 (larger refined area)")
    print("  • Decrease center_edge_length_r to 0.015-0.02 (denser mesh)")
    print("  • This should give 4-6x refinement factor")
    print("  • Expected result: 2-4x more surface triangles in center")
    print("=" * 70)
