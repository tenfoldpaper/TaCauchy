# import pymesh
# from meshpy.tet import MeshInfo, build, Options
# Note: isaacsim and omni imports must be done after SimulationApp is initialized
# These will be imported lazily when needed
# from isaacsim.util.debug_draw import _debug_draw
# from omni.physx.scripts import deformableUtils
from pxr import Sdf, Usd, UsdGeom

# Lazy initialization of debug draw
_draw = None
def _get_debug_draw():
    """Get debug draw interface (lazy import)."""
    global _draw
    if _draw is None:
        from isaacsim.util.debug_draw import _debug_draw
        _draw = _debug_draw.acquire_debug_draw_interface()
    return _draw

draw = None  # Will be initialized when needed

import numpy as np
import random

import wildmeshing as wm

from isaaclab.utils import configclass


@configclass
class TetMeshCfg:
    """

    References:

    """

    stop_quality: int = 10
    """
    Max AMIPS energy for stopping mesh optimization.

    Larger means less optimization and sooner stopping.
    """

    max_its: int = 80
    """
    Max number of mesh optimization iterations.
    """

    epsilon_r: float = 1e-2
    """ Relative envelope epsilon_r (definies the envelope size).

    -> Absolute epsilon = epsilon_r * diagonal_of_bbox.

    Smaller envelope preserves features better.
    Large Envelope + large edge_length = tetmesh with low res
    """

    edge_length_r: float = 0.1
    """ Relative target edge length l_r.

    -> Absolute l = l_r * diagonal_of_bbox.

    Smaller edge length gives denser mesh.
    For GelSight Mini gelpad (~33mm diagonal), 0.1 → ~3.3mm edge length.
    """

    skip_simplify: bool = False

    coarsen: bool = True

    log_level: int = 6
    """
    Log level, ranges from 0 to 6  (0 = most verbose, 6 = off).
    """

    # Gelpad center refinement options
    enable_center_refinement: bool = False
    """
    Enable adaptive mesh refinement for the gelpad's thin-surface center region.
    When enabled, uses wildmeshing's sizing field to make the center region denser
    while keeping the periphery coarse.
    """

    center_region_ratio: float = 0.35
    """
    Ratio of the center region relative to the total planar extent (0.0 to 1.0).
    For example, 0.35 means the inner 35% (by radius) of the target surface is refined.
    For GelSight Mini, this gives a ~4.4mm radius dense zone on the ~12.6mm half-extent.
    """

    center_edge_length_r: float = 0.035
    """
    Relative target edge length for the center region.
    Should be smaller than edge_length_r for denser mesh in the center.
    This is an *absolute* ratio to bbox diagonal, same semantics as edge_length_r.
    For GelSight Mini (~33mm diagonal), 0.035 → ~1.16mm edge length in center.
    """

    refinement_transition_ratio: float = 0.2
    """
    Width of the smooth transition band between center and outer regions,
    expressed as a fraction of the max planar half-extent.
    A wider band (0.2) gives smoother gradation between dense center and sparse edges.
    """

    refine_surface: str = "auto"
    """
    Which surface to refine.
      - "auto" : automatically pick the surface on the thinnest axis
                 (i.e. the bottom face for a thin gelpad).
      - "bottom": the face at bbox_min along the thinnest axis.
      - "top"   : the face at bbox_max along the thinnest axis.
      - "both"  : refine both top and bottom faces.
    """


@configclass
class TriMeshCfg:
    """

    References:

    """

    stop_quality: int = 10
    """
    Max AMIPS energy for stopping mesh optimization.

    Larger means less optimization and sooner stopping.
    """

    max_its: int = 80
    """
    Max number of mesh optimization iterations.
    """

    epsilon_r: float = 1e-2
    """ Relative envelope epsilon_r (definies the envelope size).
    -> Absolute epsilon = epsilon_r * diagonal_of_bbox.

    Smaller envelope pereserves features better.
    Larger Envelope + larger edge_length = tetmesh with low res
    """

    edge_length_r: float = 1 / 2
    """ Relative target edge length l_r.
    -> Absolute l = l_r * diagonal_of_bbox.

    Smaller edge length gives denser mesh.
    """

    skip_simplify: bool = False

    coarsen: bool = True

    log_level: int = 6
    """
    Log level, ranges from 0 to 6  (0 = most verbose, 6 = off).
    """


class MeshGenerator:
    def __init__(self, config: dict | TetMeshCfg = None) -> None:
        if config is None:
            config = TetMeshCfg().to_dict()
            print("No config for tet computation provided, using default settings.")
        elif type(config) is TetMeshCfg:
            config = config.to_dict()
        self.cfg = config

        # tetgen = pymesh.tetgen()
        # tetgen.points = mesh.vertices # Input points.
        # tetgen.triangles = mesh.faces # Input triangles

        # # set tetgen settings
        # for k, v in config.items():
        #     try:
        #         getattr(tetgen, k)
        #     except AttributeError:
        #         raise ValueError("invalid option for tet generation: %s" % k)
        #     else:
        #         if v is not None:
        #             setattr(tetgen, k, v)

        # tetgen.run() # Execute tetgen
        # mesh = tetgen.mesh # Extract output tetrahedral mesh.
        # print("tet mesh information ")
        # print(mesh.num_vertices, mesh.num_faces, mesh.num_voxels)
        # tet_points = mesh.vertices
        # tet_indices = mesh.voxels

        # use Wildmeshing
        # reference: https://github.com/wildmeshing/wildmeshing-python/blob/master/wildmeshing/runners.py
        self.tetrahedralizer = wm.Tetrahedralizer(
            stop_quality=self.cfg["stop_quality"],  # Specify max AMIPS energy for stopping mesh optimization"
            max_its=self.cfg["max_its"],  # Max number of mesh optimization iterations
            epsilon=self.cfg[
                "epsilon_r"
            ],  # relative envelope epsilon_r. Absolute epsilon = epsilon_r * diagonal_of_bbox
            edge_length_r=self.cfg[
                "edge_length_r"
            ],  # Relative target edge length l_r. Absolute l = l_r * diagonal_of_bbox
            skip_simplify=self.cfg["skip_simplify"],
            coarsen=self.cfg["coarsen"],
        )
        self.tetrahedralizer.set_log_level(
            self.cfg["log_level"]
        )  # Log level (0 = most verbose, 6 = off). 6 also means, that no tet data files are being generated by wildmeshing

    # def fix_mesh(self, mesh, detail="normal"):
    #     """
    #     Remesh the input mesh to remove degeneracies and improve triangle quality.

    #     Source: https://github.com/PyMesh/PyMesh/blob/main/scripts/fix_mesh.py
    #     Args:
    #         mesh (_type_): _description_
    #     """
    #     bbox_min, bbox_max = mesh.bbox
    #     diag_len = norm(bbox_max - bbox_min)
    #     print(f"diag_len = {diag_len}")
    #     if detail == "high":
    #         target_len = diag_len * 2.5e-3
    #     elif detail == "normal":
    #          target_len = diag_len * 5e-3 # adjusted values, cause we deal with very small objects, orig: 5e-3
    #     elif detail == "low":
    #         target_len = diag_len * 1e-2
    #     elif detail == "very low":
    #         target_len = diag_len * 5e-2

    #     print("Target resolution: {} mm".format(target_len))

    #     count = 0
    #     mesh, __ = pymesh.remove_duplicated_vertices(mesh, tol=target_len*1e-2)
    #     mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100)
    #     #mesh, __ = pymesh.split_long_edges(mesh, target_len*2)
    #     num_vertices = mesh.num_vertices
    #     while True:
    #         mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6, preserve_feature=True)
    #         mesh, __ = pymesh.collapse_short_edges(mesh, target_len,
    #                                                preserve_feature=True)
    #         mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100)
    #         if mesh.num_vertices == num_vertices:
    #             break

    #         num_vertices = mesh.num_vertices
    #         print(f"#v: {num_vertices}")
    #         count += 1
    #         if count > 10: break

    #     mesh = pymesh.resolve_self_intersection(mesh)
    #     mesh, __ = pymesh.remove_duplicated_faces(mesh)
    #     mesh = pymesh.compute_outer_hull(mesh)
    #     mesh, __ = pymesh.remove_duplicated_faces(mesh)
    #     mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5)
    #     mesh, __ = pymesh.remove_isolated_vertices(mesh)

    #     return mesh
    def generate_tet_mesh_for_prim(self, prim: UsdGeom.Mesh):
        # compute mesh
        points = np.array(prim.GetPointsAttr().Get())
        # triangles is a list of indices: every 3 consecutive indices form a triangle
        # Import deformableUtils lazily (after SimulationApp is initialized)
        from omni.physx.scripts import deformableUtils
        triangles = deformableUtils.triangulate_mesh(prim)

        tet_mesh_points, tet_indices, surf_points, surf_indices = self.compute_tet_mesh(points, triangles)

        # draw the tet mesh
        # color = [(0,0,0,1)]
        # for i in range(0, len(tet_indices), 4):
        #     tet_points_idx = tet_indices[i:i+4]
        #     tet_points = [tet_mesh_points[i] for i in tet_points_idx]
        #     #draw.draw_points(tet_points, [(255,0,0,1)]*len(all_vertices), [10]*len(all_vertices))
        #     draw.draw_lines([tet_points[0]]*3, tet_points[1:], color*3, [10]*3) # draw from point 0 to every other point (3 times 0, cause line from 0 to the other 3 points)
        #     draw.draw_lines([tet_points[1]]*2, tet_points[2:], color*2, [10]*2)
        #     draw.draw_lines([tet_points[2]], [tet_points[3]], color, [10]) # draw line between the other 2 points

        return tet_mesh_points, tet_indices, surf_points, surf_indices

    def _compute_center_region(self, points):
        """
        Analyse the mesh geometry to identify:
          1. The thinnest axis (thickness direction for a gelpad).
          2. The target surface(s) to refine (top / bottom / both along that axis).
          3. The planar center + half-extent used by the sizing field.

        Returns
        -------
        thin_axis : int
            Index (0/1/2) of the thinnest bbox dimension.
        bbox_diag : float
            Length of the bbox diagonal (used to convert relative → absolute lengths).
        planar_center : np.ndarray, shape (3,)
            Center of the target surface(s) in world coords.
        planar_half_extent : float
            Half of the largest planar dimension (used as reference for center_region_ratio).
        surface_z_range : tuple (z_lo, z_hi)
            Min / max coordinate along thin_axis for the surface(s) we want to refine.
        """
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        bbox_size = bbox_max - bbox_min
        bbox_diag = np.linalg.norm(bbox_size)

        # ── 1. thinnest axis ──
        thin_axis = int(np.argmin(bbox_size))
        thickness = bbox_size[thin_axis]

        # ── 2. which surface to refine ──
        refine_surface = self.cfg.get("refine_surface", "auto")
        tol = thickness * 0.15            # 15 % of thickness
        if refine_surface in ("auto", "bottom"):
            z_lo = bbox_min[thin_axis]
            z_hi = bbox_min[thin_axis] + tol
        elif refine_surface == "top":
            z_lo = bbox_max[thin_axis] - tol
            z_hi = bbox_max[thin_axis]
        elif refine_surface == "both":
            z_lo = bbox_min[thin_axis]
            z_hi = bbox_max[thin_axis]      # whole thickness
        else:
            z_lo = bbox_min[thin_axis]
            z_hi = bbox_min[thin_axis] + tol

        # ── 3. planar center & half-extent ──
        planar_dims = [i for i in range(3) if i != thin_axis]
        planar_center = (bbox_min + bbox_max) / 2.0       # 3-D center; thin_axis component doesn't matter
        planar_half_extent = max(bbox_size[d] for d in planar_dims) / 2.0

        print(f"[CenterRefinement] Thin axis = {'XYZ'[thin_axis]}  (thickness {thickness*1e3:.2f} mm)")
        print(f"[CenterRefinement] Refine surface = '{refine_surface}', z ∈ [{z_lo:.5f}, {z_hi:.5f}]")
        print(f"[CenterRefinement] Planar center  = {planar_center},  half-extent = {planar_half_extent:.5f}")

        return thin_axis, bbox_diag, planar_center, planar_half_extent, (z_lo, z_hi)

    def _build_sizing_field_func(self, points):
        """
        Build a callable sizing field for wildmeshing's ``set_sizing_field_from_func``.

        The function returns *absolute* desired edge-length at any 3-D query point.
        - Inside the center region of the target surface → small edge length (dense).
        - Outside / on other surfaces → normal edge length (coarse).
        - A smooth cosine transition band connects the two.

        Returns
        -------
        sizing_fn : Callable[[np.ndarray shape (3,1)], float]
        """
        thin_axis, bbox_diag, planar_center, planar_half_ext, (z_lo, z_hi) = \
            self._compute_center_region(points)

        center_ratio   = self.cfg.get("center_region_ratio", 0.35)
        transition_r   = self.cfg.get("refinement_transition_ratio", 0.2)
        fine_edge_abs  = self.cfg.get("center_edge_length_r", 0.035) * bbox_diag
        coarse_edge_abs = self.cfg.get("edge_length_r", 0.1) * bbox_diag

        # radii expressed in world units
        r_inner = planar_half_ext * center_ratio
        r_outer = planar_half_ext * (center_ratio + transition_r)

        planar_dims = [i for i in range(3) if i != thin_axis]

        print(f"[SizingField] fine_edge  = {fine_edge_abs*1e3:.3f} mm  "
              f"coarse_edge = {coarse_edge_abs*1e3:.3f} mm")
        print(f"[SizingField] r_inner = {r_inner*1e3:.3f} mm, "
              f"r_outer = {r_outer*1e3:.3f} mm")

        # ---- closure that wildmeshing will call per-vertex ----
        def sizing_fn(pos):
            """pos: np.ndarray shape (3,) or (3,1)."""
            p = np.asarray(pos).ravel()

            # 1) Is the point near the target surface (along thin axis)?
            z_val = p[thin_axis]
            if z_val < z_lo or z_val > z_hi:
                return coarse_edge_abs          # not on the target surface → coarse

            # 2) Planar distance to center
            d = np.sqrt(sum((p[d_i] - planar_center[d_i]) ** 2 for d_i in planar_dims))

            if d <= r_inner:
                return fine_edge_abs            # centre → dense
            elif d >= r_outer:
                return coarse_edge_abs          # periphery → coarse
            else:
                # smooth cosine transition
                t = (d - r_inner) / (r_outer - r_inner)   # 0→1
                alpha = 0.5 * (1.0 - np.cos(np.pi * t))   # 0→1 smoothly
                return fine_edge_abs + alpha * (coarse_edge_abs - fine_edge_abs)

        return sizing_fn

    def compute_tet_mesh(self, points, triangles):
        triangles = np.array(triangles, dtype=np.uint32).reshape(-1, 3)
        points = np.array(points, dtype=np.float64)

        # Store initial mesh statistics
        initial_surf_faces = len(triangles)
        initial_surf_points = len(points)
        
        # Calculate bbox for parameter display
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        bbox_size = bbox_max - bbox_min
        bbox_diag = np.linalg.norm(bbox_size)
        
        # Print configuration parameters at the start
        print(f"\n{'='*60}")
        print(f"Tetrahedral Mesh Generation Configuration")
        print(f"{'='*60}")
        
        print(f"\nInput geometry:")
        print(f"  Bounding box size:  [{bbox_size[0]*1e3:.2f}, {bbox_size[1]*1e3:.2f}, {bbox_size[2]*1e3:.2f}] mm")
        print(f"  Diagonal length:    {bbox_diag*1e3:.2f} mm")
        print(f"  Input faces:        {initial_surf_faces}")
        print(f"  Input points:       {initial_surf_points}")
        
        print(f"\nBasic parameters:")
        print(f"  stop_quality:       {self.cfg['stop_quality']}")
        print(f"  max_its:            {self.cfg['max_its']}")
        print(f"  epsilon_r:          {self.cfg['epsilon_r']:.4f} (abs: {self.cfg['epsilon_r']*bbox_diag*1e3:.3f} mm)")
        print(f"  edge_length_r:      {self.cfg['edge_length_r']:.4f} (abs: {self.cfg['edge_length_r']*bbox_diag*1e3:.3f} mm)")
        print(f"  skip_simplify:      {self.cfg['skip_simplify']}")
        print(f"  coarsen:            {self.cfg['coarsen']}")
        print(f"  log_level:          {self.cfg['log_level']}")
        
        enable_refinement = self.cfg.get("enable_center_refinement", False)
        print(f"\nCenter refinement:")
        print(f"  enabled:            {enable_refinement}")
        
        if enable_refinement:
            center_ratio = self.cfg.get("center_region_ratio", 0.35)
            center_edge_r = self.cfg.get("center_edge_length_r", 0.035)
            transition_r = self.cfg.get("refinement_transition_ratio", 0.2)
            refine_surface = self.cfg.get("refine_surface", "auto")
            
            # Calculate absolute values
            center_edge_abs = center_edge_r * bbox_diag
            base_edge_abs = self.cfg['edge_length_r'] * bbox_diag
            
            # Estimate refinement region size
            thin_axis = int(np.argmin(bbox_size))
            planar_dims = [i for i in range(3) if i != thin_axis]
            planar_half_ext = max(bbox_size[d] for d in planar_dims) / 2.0
            r_inner = planar_half_ext * center_ratio
            r_outer = planar_half_ext * (center_ratio + transition_r)
            
            print(f"  center_region_ratio:      {center_ratio:.2f} ({center_ratio*100:.0f}% radius)")
            print(f"  center_edge_length_r:     {center_edge_r:.4f} (abs: {center_edge_abs*1e3:.3f} mm)")
            print(f"  base_edge_length_r:       {self.cfg['edge_length_r']:.4f} (abs: {base_edge_abs*1e3:.3f} mm)")
            print(f"  edge_ratio (center/base): {center_edge_r/self.cfg['edge_length_r']:.2f}x")
            print(f"  transition_ratio:         {transition_r:.2f}")
            print(f"  refine_surface:           {refine_surface}")
            print(f"  center_radius:            {r_inner*1e3:.3f} mm")
            print(f"  transition_radius:        {r_outer*1e3:.3f} mm")
            print(f"  thin_axis:                {'XYZ'[thin_axis]} (thickness: {bbox_size[thin_axis]*1e3:.2f} mm)")
        
        print(f"{'='*60}\n")

        # ── Use the default tetrahedralizer (coarse edge_length_r) ──
        tetrahedralizer = self.tetrahedralizer

        tetrahedralizer.set_mesh(points, triangles)

        # ── Apply per-vertex sizing field when center refinement is enabled ──
        enable_refinement = self.cfg.get("enable_center_refinement", False)
        if enable_refinement:
            sizing_fn = self._build_sizing_field_func(points)
            tetrahedralizer.set_sizing_field_from_func(sizing_fn)
            print("[CenterRefinement] Sizing field applied to Tetrahedralizer.")

        tetrahedralizer.tetrahedralize()
        tet_points, tet_indices, _ = tetrahedralizer.get_tet_mesh(
            correct_surface_orientation=True,
            manifold_surface=True,
            use_input_for_wn=False,
        )
        tet_indices = np.array(tet_indices).flatten().tolist()

        # surface
        surf_points, surf_indices = tetrahedralizer.get_tracked_surfaces()
        surf_points = surf_points[0]  # surf_points is a list -> in case object consists of multiple surfaces
        surf_indices = np.array(surf_indices).flatten().tolist()
        
        # Calculate final statistics
        final_surf_faces = len(surf_indices) // 3
        final_surf_points = len(surf_points)
        num_tets = len(tet_indices) // 4
        
        # Analyze surface mesh quality
        surf_triangles = np.array(surf_indices).reshape(-1, 3)
        surf_points_arr = np.array(surf_points)
        
        # Calculate edge lengths for surface triangles
        edge_lengths = []
        for tri in surf_triangles:
            v0, v1, v2 = surf_points_arr[tri[0]], surf_points_arr[tri[1]], surf_points_arr[tri[2]]
            edge_lengths.extend([
                np.linalg.norm(v1 - v0),
                np.linalg.norm(v2 - v1),
                np.linalg.norm(v0 - v2)
            ])
        edge_lengths = np.array(edge_lengths)
        
        # Calculate triangle areas
        tri_areas = []
        for tri in surf_triangles:
            v0, v1, v2 = surf_points_arr[tri[0]], surf_points_arr[tri[1]], surf_points_arr[tri[2]]
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            tri_areas.append(area)
        tri_areas = np.array(tri_areas)
        
        # Print detailed statistics
        print(f"\n{'='*60}")
        print(f"Mesh Generation Results")
        print(f"{'='*60}")
        print(f"\nMesh statistics:")
        print(f"  Tet vertices:       {len(tet_points):>6}")
        print(f"  Tetrahedra:         {num_tets:>6}")
        print(f"  Surface points:     {initial_surf_points:>6} → {final_surf_points:>6} ({final_surf_points/initial_surf_points:.2f}x)")
        print(f"  Surface faces:      {initial_surf_faces:>6} → {final_surf_faces:>6} ({final_surf_faces/initial_surf_faces:.2f}x)")
        
        if enable_refinement:
            # Calculate statistics for each region
            def calc_region_stats(face_indices):
                if len(face_indices) == 0:
                    return None
                region_edges = []
                region_areas = []
                for idx in face_indices:
                    tri = surf_triangles[idx]
                    v0, v1, v2 = surf_points_arr[tri[0]], surf_points_arr[tri[1]], surf_points_arr[tri[2]]
                    region_edges.extend([
                        np.linalg.norm(v1 - v0),
                        np.linalg.norm(v2 - v1),
                        np.linalg.norm(v0 - v2)
                    ])
                    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                    region_areas.append(area)
                return np.array(region_edges), np.array(region_areas)
            
            # Analyze center region statistics
            thin_axis, bbox_diag, planar_center, planar_half_ext, (z_lo, z_hi) = \
                self._compute_center_region(points)
            
            center_ratio = self.cfg.get("center_region_ratio", 0.35)
            transition_r = self.cfg.get("refinement_transition_ratio", 0.2)
            
            r_inner = planar_half_ext * center_ratio
            r_outer = planar_half_ext * (center_ratio + transition_r)
            
            planar_dims = [i for i in range(3) if i != thin_axis]
            
            # Classify surface triangles by region
            center_faces = []
            transition_faces = []
            outer_faces = []
            
            for i, tri in enumerate(surf_triangles):
                # Get triangle center
                v0, v1, v2 = surf_points_arr[tri[0]], surf_points_arr[tri[1]], surf_points_arr[tri[2]]
                tri_center = (v0 + v1 + v2) / 3.0
                
                # Check if on target surface
                z_val = tri_center[thin_axis]
                if z_lo <= z_val <= z_hi:
                    # Calculate planar distance to center
                    d = np.sqrt(sum((tri_center[d_i] - planar_center[d_i]) ** 2 for d_i in planar_dims))
                    
                    if d <= r_inner:
                        center_faces.append(i)
                    elif d <= r_outer:
                        transition_faces.append(i)
                    else:
                        outer_faces.append(i)
                else:
                    outer_faces.append(i)
            
            print(f"\nRefinement impact:")
            print(f"  Region distribution:")
            print(f"    Center:           {len(center_faces):>4} faces ({len(center_faces)/final_surf_faces*100:>5.1f}%)")
            print(f"    Transition:       {len(transition_faces):>4} faces ({len(transition_faces)/final_surf_faces*100:>5.1f}%)")
            print(f"    Outer:            {len(outer_faces):>4} faces ({len(outer_faces)/final_surf_faces*100:>5.1f}%)")
            
            # Print statistics for each region
            center_stats = calc_region_stats(center_faces)
            trans_stats = calc_region_stats(transition_faces)
            outer_stats = calc_region_stats(outer_faces)
            
            if center_stats is not None:
                c_edges, c_areas = center_stats
                print(f"\n  Center region quality:")
                print(f"    Edge length (mm):     {c_edges.min()*1e3:>5.2f} / {c_edges.mean()*1e3:>5.2f} / {c_edges.max()*1e3:>5.2f} (min/mean/max)")
                print(f"    Triangle area (mm²):  {c_areas.min()*1e6:>5.3f} / {c_areas.mean()*1e6:>5.3f} / {c_areas.max()*1e6:>5.3f}")
            
            if trans_stats is not None:
                t_edges, t_areas = trans_stats
                print(f"\n  Transition region quality:")
                print(f"    Edge length (mm):     {t_edges.min()*1e3:>5.2f} / {t_edges.mean()*1e3:>5.2f} / {t_edges.max()*1e3:>5.2f}")
                print(f"    Triangle area (mm²):  {t_areas.min()*1e6:>5.3f} / {t_areas.mean()*1e6:>5.3f} / {t_areas.max()*1e6:>5.3f}")
            
            if outer_stats is not None:
                o_edges, o_areas = outer_stats
                print(f"\n  Outer region quality:")
                print(f"    Edge length (mm):     {o_edges.min()*1e3:>5.2f} / {o_edges.mean()*1e3:>5.2f} / {o_edges.max()*1e3:>5.2f}")
                print(f"    Triangle area (mm²):  {o_areas.min()*1e6:>5.3f} / {o_areas.mean()*1e6:>5.3f} / {o_areas.max()*1e6:>5.3f}")
        else:
            print(f"\nOverall surface quality:")
            print(f"  Edge length (mm):     {edge_lengths.min()*1e3:>5.2f} / {edge_lengths.mean()*1e3:>5.2f} / {edge_lengths.max()*1e3:>5.2f} (min/mean/max)")
            print(f"  Triangle area (mm²):  {tri_areas.min()*1e6:>5.3f} / {tri_areas.mean()*1e6:>5.3f} / {tri_areas.max()*1e6:>5.3f}")
        
        print(f"{'='*60}\n")
        
        return tet_points, tet_indices, surf_points, surf_indices

    @staticmethod
    def update_usd_mesh(prim: UsdGeom.Mesh, surf_points, triangles: list[int]):
        triangles = np.array(triangles).reshape(-1, 3)

        prim.GetPointsAttr().Set(surf_points)
        prim.GetFaceVertexCountsAttr().Set(
            [3] * triangles.shape[0]
        )  # how many vertices each face has (3, cause triangles)
        prim.GetFaceVertexIndicesAttr().Set(triangles)
        prim.GetNormalsAttr().Set([])  # set to be empty, cause we use catmullClark and this gives us normals
        prim.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        # prim.GetSubdivisionSchemeAttr().Set("catmullClark") #none

        # set color with per face interpolation
        colors = [
            (random.uniform(0.0, 0.0), random.uniform(0.0, 0.75), random.uniform(0.0, 0.75))
            for _ in range(triangles.shape[0] * 3)
        ]
        prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.faceVarying).Set(colors)  # num_surf_tri * 3

        # texture map
        uv_coor = np.indices((int(triangles.size), 2)).transpose((1, 2, 0)).reshape((-1, 2))

        pv_api = UsdGeom.PrimvarsAPI(prim)
        if pv_api.HasPrimvar("primvars:st"):
            pv = pv_api.GetPrimvar("primvars:st")
            pv.SetInterpolation(UsdGeom.Tokens.faceVarying)
            # if uv_coor.size != pv.Get():
            #     print("primvars:st array has the wrong size - updating it")
            #     pv.Set(uv_coor)
        else:
            # set some values for the uv_coor variable, if no values exist
            pv = pv_api.CreatePrimvar(
                "primvars:st",
                Sdf.ValueTypeNames.TexCoord2fArray,
                UsdGeom.Tokens.faceVarying,
                # UsdGeom.Tokens.uniform,
                uv_coor.size,
            )
            pv.Set(uv_coor)

    @staticmethod
    def update_usd_mesh_with_uipc_surface(prim: Usd.Prim):
        """Method to update render mesh topology based on the surface of the mesh in UIPC.

        Used for workaround so that textures can be applied to the meshes.
        """
        from uipc.geometry import extract_surface, flip_inward_triangles, label_surface, label_triangle_orient, tetmesh

        tet_points = prim.GetAttribute("tet_points").Get()
        tet_points = np.array(tet_points)

        tet_indices = prim.GetAttribute("tet_indices").Get()
        tet_indices = np.array(tet_indices).reshape(-1, 4)

        # compute mesh
        mesh = tetmesh(tet_points.copy(), tet_indices.copy())
        # enable the contact by labeling the surface
        label_surface(mesh)
        label_triangle_orient(mesh)  # -> only needed when we want to export the mesh with uipc
        # flip the triangles inward for better rendering
        mesh = flip_inward_triangles(mesh)
        surf = extract_surface(mesh)

        surf_points = surf.positions().view().reshape(-1, 3)
        triangles = surf.triangles().topo().view().reshape(-1).tolist()

        # update usd attributes
        attr_tet_surf_points = prim.CreateAttribute("tet_surf_points", Sdf.ValueTypeNames.Vector3fArray)
        attr_tet_surf_points.Set(surf_points)

        attr_tet_surf_indices = prim.CreateAttribute("tet_surf_indices", Sdf.ValueTypeNames.UIntArray)
        attr_tet_surf_indices.Set(triangles)

        # update render mesh
        prim = UsdGeom.Mesh(prim)

        triangles = np.array(triangles).reshape(-1, 3)
        prim.GetPointsAttr().Set(surf_points)
        prim.GetFaceVertexCountsAttr().Set(
            [3] * triangles.shape[0]
        )  # how many vertices each face has (3, cause triangles)
        prim.GetFaceVertexIndicesAttr().Set(triangles)
        prim.GetNormalsAttr().Set([])  # set to be empty, cause we use catmullClark and this gives us normals
        prim.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        # prim.GetSubdivisionSchemeAttr().Set("catmullClark") #none

        # set color with per face interpolation
        colors = [
            (random.uniform(0.0, 0.0), random.uniform(0.0, 0.75), random.uniform(0.0, 0.75))
            for _ in range(triangles.shape[0] * 3)
        ]
        prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.faceVarying).Set(colors)  # num_surf_tri * 3

        # set uv_coor variable
        uv_coor = np.indices((int(triangles.shape[0] * 1.5), 2)).transpose((1, 2, 0)).reshape((-1, 2))

        pv_api = UsdGeom.PrimvarsAPI(prim)
        if pv_api.HasPrimvar("primvars:st"):
            pv = pv_api.GetPrimvar("primvars:st")
            pv.SetInterpolation(UsdGeom.Tokens.faceVarying)
        else:
            pv = pv_api.CreatePrimvar(
                "primvars:st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying, uv_coor.size
            )
        pv.Set(uv_coor)
