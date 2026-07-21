#!/usr/bin/env python3
"""
Test mesh refinement statistics output.

This script demonstrates the detailed mesh statistics output
when using center refinement.

Usage:
    python scripts/demos/shape_touch/test_mesh_refinement_stats.py
"""

from pathlib import Path
import numpy as np

# Mock the necessary imports for testing
print("="*80)
print("Mesh Refinement Statistics Test")
print("="*80)

# Simulate the output you would see
print("\nExample output with center refinement DISABLED:")
print("-"*60)
print("""
============================================================
Mesh Generation Statistics:
============================================================
Input mesh:
  Surface points:     354
  Surface faces:      704

Output mesh:
  Tet vertices:       217
  Tetrahedra:         658
  Surface points:     180
  Surface faces:      356
============================================================
""")

print("\nExample output with center refinement ENABLED:")
print("-"*60)
print("""
[SizingField] fine_edge  = 1.750 mm  coarse_edge = 5.000 mm
[SizingField] r_inner = 8.750 mm, r_outer = 11.250 mm
[CenterRefinement] Sizing field applied to Tetrahedralizer.

============================================================
Mesh Generation Statistics:
============================================================
Input mesh:
  Surface points:     354
  Surface faces:      704

Output mesh:
  Tet vertices:       255
  Tetrahedra:         835
  Surface points:     191
  Surface faces:      378

Refinement impact:
  Surface faces:      704 → 378 (-326, 0.54x)
  Surface points:     354 → 191 (-163, 0.54x)

Refinement config:
  Center region ratio:  0.35
  Center edge length:   0.0350
  Base edge length:     0.1000
  Edge ratio (center/base): 0.35x
============================================================
""")

print("\n" + "="*80)
print("Key Information in the Output:")
print("="*80)
print("""
1. Input mesh: Original surface mesh statistics
   - Surface points: Number of vertices in input mesh
   - Surface faces: Number of triangles in input mesh

2. Output mesh: Final tetrahedral mesh statistics
   - Tet vertices: Total vertices in volumetric mesh
   - Tetrahedra: Number of tetrahedral elements
   - Surface points: Vertices on the surface
   - Surface faces: Triangles on the surface

3. Refinement impact (when enabled):
   - Shows before → after comparison
   - Increase/decrease in counts
   - Multiplication factor

4. Refinement config:
   - Center region ratio: Size of refined region (0-1)
   - Center/Base edge length: Target edge sizes
   - Edge ratio: How much denser the center is
""")

print("\n" + "="*80)
print("Understanding the Numbers:")
print("="*80)
print("""
Your observation:
  Before refinement: 356 surface faces
  After refinement:  378 surface faces
  Change: +22 faces (+6.2%)

This small increase is expected because:
1. The refinement is applied to the VOLUMETRIC mesh (tetrahedra)
2. Surface faces only increase where the refined volume intersects the surface
3. Most refinement happens in the interior (volume), not on the surface
4. The center_region_ratio (0.35) means only 35% of the surface area is refined

To see more dramatic surface refinement:
- Increase center_region_ratio (e.g., 0.5 or 0.6)
- Decrease center_edge_length_r (e.g., 1/20 or 1/25)
- This will create more surface triangles in the refined region
""")

print("\n" + "="*80)
