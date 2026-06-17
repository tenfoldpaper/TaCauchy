from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from uipc import builtin


@dataclass
class SurfaceStressData:
    """Surface stress projection data for a tetrahedral mesh surface."""

    triangle_indices: Optional[np.ndarray] = None
    """Surface triangle vertex indices. Shape: (num_triangles, 3)."""

    normals: Optional[np.ndarray] = None
    """Surface triangle normals (unit). Shape: (num_triangles, 3)."""

    traction: Optional[np.ndarray] = None
    """Surface traction vectors t = sigma * n. Shape: (num_triangles, 3)."""

    normal_pressure: Optional[np.ndarray] = None
    """Normal pressure p = n^T sigma n. Shape: (num_triangles,)."""

    tangential_traction: Optional[np.ndarray] = None
    """Tangential traction tau = t - p n. Shape: (num_triangles, 3)."""


class StressExtractor:
    """Extract Cauchy stress and project to surface pressure on tetrahedral meshes."""

    def get_object_surface_stress(self, uipc_object) -> SurfaceStressData:
        """Compute surface traction/pressure from Cauchy stress on a UIPC object."""
        if not uipc_object.geo_slot_list:
            return SurfaceStressData()

        geo = uipc_object.geo_slot_list[0].geometry()

        if geo.dim() != 3:
            return SurfaceStressData()

        stress_attr = geo.tetrahedra().find("cauchy_stress")
        if stress_attr is None:
            return SurfaceStressData()

        triangles_attr = geo.triangles().topo()
        if triangles_attr is None or triangles_attr.size() == 0:
            return SurfaceStressData()

        triangles = triangles_attr.view().reshape(-1, 3)
        positions = geo.positions().view().reshape(-1, 3)

        is_surf_attr = geo.triangles().find(builtin.is_surf)
        if is_surf_attr is not None:
            is_surf = is_surf_attr.view().astype(bool)
        else:
            is_surf = np.ones(len(triangles), dtype=bool)

        parent_attr = geo.triangles().find(builtin.parent_id)
        if parent_attr is None:
            return SurfaceStressData()

        parent_ids = parent_attr.view().astype(np.int64)
        surface_triangles = triangles[is_surf]
        surface_parent_ids = parent_ids[is_surf]

        if len(surface_triangles) == 0:
            return SurfaceStressData()

        stress = stress_attr.view().reshape(-1, 3, 3)
        if np.max(surface_parent_ids) >= len(stress):
            return SurfaceStressData()

        v0 = positions[surface_triangles[:, 0]]
        v1 = positions[surface_triangles[:, 1]]
        v2 = positions[surface_triangles[:, 2]]

        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normals = normals / norms

        sigma = stress[surface_parent_ids]
        traction = np.einsum("nij,nj->ni", sigma, normals)
        normal_pressure = np.einsum("ni,ni->n", traction, normals)
        tangential = traction - normal_pressure[:, None] * normals

        return SurfaceStressData(
            triangle_indices=surface_triangles,
            normals=normals,
            traction=traction,
            normal_pressure=normal_pressure,
            tangential_traction=tangential,
        )

