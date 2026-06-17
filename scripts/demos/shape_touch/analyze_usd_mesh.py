#!/usr/bin/env python3
"""
Analyze USD mesh information for GelSight Mini gelpad.

Usage:
    python scripts/demos/shape_touch/analyze_usd_mesh.py
"""

from pxr import Usd, UsdGeom
import numpy as np
from pathlib import Path


def analyze_mesh(usd_path: Path):
    """Analyze mesh information from USD file."""
    
    print(f"\n{'='*80}")
    print(f"Analyzing USD Mesh: {usd_path.name}")
    print(f"{'='*80}\n")
    
    # Open USD stage
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        print(f"[ERROR] Failed to open USD file: {usd_path}")
        return
    
    # Find all mesh prims
    mesh_prims = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh_prims.append(prim)
    
    print(f"Found {len(mesh_prims)} mesh(es) in USD file\n")
    
    total_vertices = 0
    total_faces = 0
    total_tets = 0
    
    for i, prim in enumerate(mesh_prims, 1):
        mesh = UsdGeom.Mesh(prim)
        prim_path = prim.GetPath()
        
        print(f"{'='*80}")
        print(f"Mesh {i}: {prim_path}")
        print(f"{'='*80}")
        
        # Get points (vertices)
        points_attr = mesh.GetPointsAttr()
        if points_attr:
            points = points_attr.Get()
            num_vertices = len(points) if points else 0
            total_vertices += num_vertices
            print(f"  Vertices (Points):        {num_vertices:,}")
            
            if points and num_vertices > 0:
                points_array = np.array(points)
                print(f"  Vertex bounds:")
                print(f"    X: [{points_array[:, 0].min():.6f}, {points_array[:, 0].max():.6f}]")
                print(f"    Y: [{points_array[:, 1].min():.6f}, {points_array[:, 1].max():.6f}]")
                print(f"    Z: [{points_array[:, 2].min():.6f}, {points_array[:, 2].max():.6f}]")
        
        # Get face vertex counts
        face_vertex_counts_attr = mesh.GetFaceVertexCountsAttr()
        if face_vertex_counts_attr:
            face_vertex_counts = face_vertex_counts_attr.Get()
            num_faces = len(face_vertex_counts) if face_vertex_counts else 0
            total_faces += num_faces
            print(f"  Surface Faces:            {num_faces:,}")
            
            if face_vertex_counts and num_faces > 0:
                counts_array = np.array(face_vertex_counts)
                unique_counts = np.unique(counts_array)
                print(f"  Face types: {dict(zip(unique_counts, [np.sum(counts_array == c) for c in unique_counts]))}")
        
        # Get face vertex indices
        face_vertex_indices_attr = mesh.GetFaceVertexIndicesAttr()
        if face_vertex_indices_attr:
            face_vertex_indices = face_vertex_indices_attr.Get()
            if face_vertex_indices:
                print(f"  Face Vertex Indices:      {len(face_vertex_indices):,}")
        
        # Check for tetrahedral mesh data (custom attributes)
        # Look for common tetrahedral mesh attribute names
        tet_attrs = []
        for attr_name in ['tetrahedra', 'tets', 'tetIndices', 'volumeIndices']:
            attr = prim.GetAttribute(attr_name)
            if attr and attr.Get():
                tet_attrs.append((attr_name, attr))
        
        if tet_attrs:
            print(f"\n  Tetrahedral Mesh Data:")
            for attr_name, attr in tet_attrs:
                tet_data = attr.Get()
                if tet_data:
                    tet_array = np.array(tet_data)
                    if len(tet_array.shape) == 1:
                        # Flat array, assume groups of 4
                        num_tets = len(tet_array) // 4
                    else:
                        num_tets = len(tet_array)
                    total_tets += num_tets
                    print(f"    {attr_name}: {num_tets:,} tetrahedra")
        
        # Check all custom attributes
        custom_attrs = [attr for attr in prim.GetAttributes() 
                       if not attr.GetName().startswith('primvars:')]
        
        if custom_attrs:
            print(f"\n  Custom Attributes ({len(custom_attrs)}):")
            for attr in custom_attrs[:10]:  # Show first 10
                attr_name = attr.GetName()
                attr_value = attr.Get()
                if attr_value is not None:
                    if hasattr(attr_value, '__len__'):
                        print(f"    {attr_name}: {type(attr_value).__name__} (length: {len(attr_value)})")
                    else:
                        print(f"    {attr_name}: {attr_value}")
        
        print()
    
    # Summary
    print(f"{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"  Total Meshes:             {len(mesh_prims)}")
    print(f"  Total Vertices:           {total_vertices:,}")
    print(f"  Total Surface Faces:      {total_faces:,}")
    if total_tets > 0:
        print(f"  Total Tetrahedra:         {total_tets:,}")
    print(f"{'='*80}\n")


def main():
    # Path to GelSight Mini gelpad USD
    workspace_root = Path(__file__).parent.parent.parent.parent
    usd_path = workspace_root / "source/tacex_assets/tacex_assets/data/Robots/Franka/GelSight_Mini/Single_Adapter/uipc_gelpad.usd"
    
    if not usd_path.exists():
        print(f"[ERROR] USD file not found: {usd_path}")
        return
    
    analyze_mesh(usd_path)


if __name__ == "__main__":
    main()
