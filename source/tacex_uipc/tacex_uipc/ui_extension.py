# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import omni.ext
import omni.ui
import omni.usd
from omni.physx import get_physx_interface

# Lazy initialization of debug draw for Isaac Sim 5.x compatibility
_draw = None
def _get_debug_draw():
    """Get debug draw interface (lazy import)."""
    global _draw
    if _draw is None:
        try:
            # Try Isaac Sim 5.x path first
            from isaacsim.util.debug_draw import _debug_draw
            _draw = _debug_draw.acquire_debug_draw_interface()
        except ImportError:
            try:
                # Fallback to Isaac Sim 4.x path
                import omni.isaac.debug_draw._debug_draw as _debug_draw_module
                _draw = _debug_draw_module.acquire_debug_draw_interface()
            except ImportError:
                print("[tacex_uipc] Warning: Could not import debug_draw. Visualization features will be limited.")
                # Create a dummy object that does nothing
                class DummyDraw:
                    def clear_points(self): pass
                    def clear_lines(self): pass
                    def draw_points(self, *args, **kwargs): pass
                    def draw_lines(self, *args, **kwargs): pass
                _draw = DummyDraw()
    return _draw

# Create a proxy object that will lazily initialize draw
class _DrawProxy:
    def __getattr__(self, name):
        return getattr(_get_debug_draw(), name)

draw = _DrawProxy()

import numpy as np

import pxr
from pxr import Sdf, UsdGeom

from tacex_uipc.sim import UipcIsaacAttachments
from tacex_uipc.utils import MeshGenerator, TetMeshCfg


def get_selected_prim_path():
    """Return the path of the first selected prim"""
    context = omni.usd.get_context()
    selection = context.get_selection()
    paths = selection.get_selected_prim_paths()

    return None if not paths else paths[0]


def get_selected_prim_paths():
    """Return the paths of the selected prims"""
    context = omni.usd.get_context()
    selection = context.get_selection()
    paths = selection.get_selected_prim_paths()

    return paths


def get_stage_id():
    """Return the stage Id of the current stage"""
    context = omni.usd.get_context()
    return context.get_stage_id()


def _generate_tet_mesh(path, tet_cfg=None):
    """Generates a tetrahedra mesh for a USD trimesh.

    You need to make sure that the USD path belongs to the geom_mesh and not just the Xform of the prim.
    """
    if tet_cfg is None:
        tet_cfg = TetMeshCfg(edge_length_r=0.25)

    # # high res
    # if tet_cfg is None:
    #     tet_cfg = TetMeshCfg(
    #         max_its=250,
    #         epsilon_r=0.001,
    #         edge_length_r=0.05
    #     )

    mesh_gen = MeshGenerator(tet_cfg)

    stage = omni.usd.get_context().get_stage()
    # prim = stage.GetPrimAtPath(Sdf.Path(path))
    geom_mesh = UsdGeom.Mesh.Get(stage, path)
    tet_points, tet_indices, surf_points, tet_surf_indices = mesh_gen.generate_tet_mesh_for_prim(geom_mesh)

    tf_world = np.array(omni.usd.get_world_transform_matrix(geom_mesh))
    world_tet_points = tf_world.T @ np.vstack((tet_points.T, np.ones(tet_points.shape[0])))
    world_tet_points = world_tet_points[:-1].T

    world_tet_surf_points = tf_world.T @ np.vstack((surf_points.T, np.ones(surf_points.shape[0])))
    world_tet_surf_points = world_tet_surf_points[:-1].T

    draw.clear_points()
    draw.clear_lines()
    _draw_tets(world_tet_points, tet_indices)
    _draw_surface_trimesh(world_tet_surf_points, tet_surf_indices)

    # Dont save the transformed points ->  we want to save the local points. Transformations happens during scene creation
    _create_tet_data_attributes(
        path,
        tet_points=tet_points,
        tet_indices=tet_indices,
        tet_surf_points=surf_points,
        tet_surf_indices=tet_surf_indices,
    )
    return (
        f"Amount of tet points {len(tet_points)},\nAmount of tetrahedra: {int(len(tet_indices) / 4)},\nAmount of"
        f" surface points: {int(len(tet_surf_indices) / 3)}"
    )


def _draw_tets(all_vertices, tet_indices):
    draw.clear_lines()

    # first draw the tet mesh nodes
    # draw.draw_points(all_vertices, [(255,0,0,1)]*len(all_vertices), [10]*len(all_vertices))

    # connect nodes according to tet_indices
    color = [(125, 0, 0, 0.5)]
    for i in range(0, len(tet_indices), 4):
        tet_points_idx = tet_indices[i : i + 4]
        tet_points = [all_vertices[i] for i in tet_points_idx]
        # draw.draw_points(tet_points, [(255,0,0,1)]*len(all_vertices), [10]*len(all_vertices))
        draw.draw_lines(
            [tet_points[0]] * 3, tet_points[1:], color * 3, [10] * 3
        )  # draw from point 0 to every other point (3 times 0, cause line from 0 to the other 3 points)
        draw.draw_lines([tet_points[1]] * 2, tet_points[2:], color * 2, [10] * 2)
        draw.draw_lines([tet_points[2]], [tet_points[3]], color, [10])  # draw line between the other 2 points


def _draw_surface_trimesh(all_vertices, tet_surf_indices):
    color = [(0, 0, 125, 0.5)]
    # draw surface mesh
    for i in range(0, len(tet_surf_indices), 3):
        tri_points_idx = tet_surf_indices[i : i + 3]
        tri_points = [all_vertices[j] for j in tri_points_idx]
        draw.draw_points(tri_points, [(255, 255, 255, 1)] * len(tri_points), [40] * len(tri_points))
        draw.draw_lines(
            [tri_points[0]] * 2, tri_points[1:], color * 2, [10] * 2
        )  # draw from point 0 to every other point (3 times 0, cause line from 0 to the other 3 points)
        draw.draw_lines([tri_points[1]] * 1, tri_points[2:], color * 1, [10] * 1)


def _create_tet_data_attributes(path, tet_points, tet_indices, tet_surf_points, tet_surf_indices):
    """
    Creates an attribute for a prim that holds a boolean.
    See: https://graphics.pixar.com/usd/release/api/class_usd_prim.html.
    The attribute can then be found in the GUI under "Raw USD Properties" of the prim.
    Args:
        prim: A prim that should be holding the attribute.
        attribute_name: The name of the attribute to create.
    Returns:
    """
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)

    attr_tet_points = prim.CreateAttribute("tet_points", pxr.Sdf.ValueTypeNames.Vector3fArray)
    attr_tet_points.Set(tet_points)
    attr_tet_points.SetCustom(True)

    attr_tet_indices = prim.CreateAttribute("tet_indices", pxr.Sdf.ValueTypeNames.UIntArray)
    attr_tet_indices.Set(tet_indices)
    attr_tet_indices.SetCustom(True)

    attr_tet_surf_points = prim.CreateAttribute("tet_surf_points", pxr.Sdf.ValueTypeNames.Vector3fArray)
    attr_tet_surf_points.Set(tet_surf_points)

    attr_tet_surf_indices = prim.CreateAttribute("tet_surf_indices", pxr.Sdf.ValueTypeNames.UIntArray)
    attr_tet_surf_indices.Set(tet_surf_indices)

    print("*" * 40)
    print("Created tet data: ")
    print(f"tet_points (num {tet_points.shape[0]})")
    print(f"tet_indices (num {len(tet_indices)})")
    print(f"tet_surf_points (num {tet_surf_points.shape[0]})")
    print(f"tet_surf_indices (num {len(tet_surf_indices)})")
    # Calculate number of surface triangular faces
    num_surf_faces = len(tet_surf_indices) // 3
    print(f"tet_surf_faces (num {num_surf_faces})")
    print("*" * 40)


def _update_surf_mesh(path):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(pxr.Sdf.Path(path))
    print("prim ", prim)
    # extract surface data of tet mesh
    # surf_points = prim.GetAttribute("tet_surf_points").Get()
    # tet_surf_indices = prim.GetAttribute("tet_surf_indices").Get()

    # surf_points = np.array(surf_points)
    # triangles = tet_surf_indices
    # MeshGenerator.update_usd_mesh(UsdGeom.Mesh(prim), surf_points=surf_points, triangles=triangles)
    # print("Updated Surface Mesh of ", path)

    # update surface based on uipc_mesh surface
    MeshGenerator.update_usd_mesh_with_uipc_surface(prim)
    print("Update of surface Mesh via UIPC: ", path)


def _create_attachment(paths):
    print("paths are ", paths)

    isaac_mesh_path = paths[0]
    tet_mesh_path = paths[1]

    # extract data of tet mesh
    stage = omni.usd.get_context().get_stage()
    tet_prim = stage.GetPrimAtPath(pxr.Sdf.Path(tet_mesh_path))
    tet_points = np.array(tet_prim.GetAttribute("tet_points").Get())
    tet_indices = tet_prim.GetAttribute("tet_indices").Get()

    # convert to world coordinates
    tf_world = np.array(omni.usd.get_world_transform_matrix(tet_prim))
    print("tf world ", tf_world)
    world_tet_points = tf_world.T @ np.vstack((tet_points.T, np.ones(tet_points.shape[0])))
    world_tet_points = world_tet_points[:-1].T

    # disable collision of the mesh that should be simulated by uipc -> otherwise raycasts are only detecting the tet mesh
    try:
        collision_enabled = tet_prim.GetAttribute("physics:collisionEnabled")
        collision_enabled.Set(False)
    except RuntimeError:
        pass

    attachment_offsets, idx, rigid_prims, attachment_points_positions, obj_pos = (
        UipcIsaacAttachments.compute_attachment_data(isaac_mesh_path, world_tet_points, tet_indices)
    )
    _create_attachment_data_attributes(tet_mesh_path, attachment_offsets, idx)

    # draw attachment data
    draw.draw_points(
        attachment_points_positions,
        [(255, 0, 0, 0.5)] * attachment_points_positions.shape[0],
        [30] * attachment_points_positions.shape[0],
    )  # the new positions
    obj_center = obj_pos

    for j in range(0, attachment_points_positions.shape[0]):
        draw.draw_lines([obj_center], [attachment_points_positions[j, :]], [(255, 255, 0, 0.5)], [10])

    get_physx_interface().release_physics_objects()


def _create_attachment_data_attributes(path, attachment_offsets, attachment_indices):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)

    attr_tet_points = prim.CreateAttribute("attachment_offsets", pxr.Sdf.ValueTypeNames.Vector3fArray)
    attr_tet_points.Set(attachment_offsets)

    attr_attachment_indices = prim.CreateAttribute("attachment_indices", pxr.Sdf.ValueTypeNames.UIntArray)
    attr_attachment_indices.Set(attachment_indices)

    print("*" * 40)
    print("Created tet data: ")
    print(f"attachment_offsets (num {attachment_offsets.shape[0]})")
    print(f"attachment_indices (num {len(attachment_indices)})")
    print("*" * 40)


# --- for some funky UV texture stuff (just experimental) ---
def _extract_primvar_st(path):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)

    pv_api = UsdGeom.PrimvarsAPI(UsdGeom.Mesh(prim))
    if not pv_api.HasPrimvar("primvars:st"):
        print("No primvars:st")
        return

    primvars_st = np.array(pv_api.GetPrimvar("primvars:st").Get())
    print("primvars:st has shape ", primvars_st.shape)
    np.save("./primvars_st.npy", primvars_st)


def _set_primvar_st(path):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)

    # load uv coor from array
    uv_coor = np.load("./primvars_st.npy")

    pv_api = UsdGeom.PrimvarsAPI(UsdGeom.Mesh(prim))
    if not pv_api.HasPrimvar("primvars:st"):
        pv = pv_api.CreatePrimvar(
            "primvars:st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying, uv_coor.size
        )
    else:
        pv = pv_api.GetPrimvar("primvars:st")
    pv.Set(uv_coor)
    print("Set uv values for primvars:st")


# ---


# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class TacexIPCExtension(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        print("[tacex_uipc] startup")

        self._window = omni.ui.Window(
            "Generate Tet Meshes for the IPC simulation:",
            width=400,
            height=600,
            dockPreference=omni.ui.DockPreference.RIGHT_BOTTOM,
        )
        self._t = 0
        self._dt = 0.01
        self.sub = None
        self.playing = False
        
        # Initialize mesh generation parameters (tuned for GelSight Mini ~25x21x4mm gelpad)
        self._edge_length_r = 0.1
        self._enable_center_refinement = True
        self._center_region_ratio = 0.35
        self._center_edge_length_r = 0.035
        self._refinement_transition_ratio = 0.2
        self._refine_surface = "auto"
        self._epsilon_r = 0.01
        self._stop_quality = 10
        self._max_its = 80

        with self._window.frame:
            with omni.ui.VStack(spacing=5):
                # --- Scrollable parameter area (takes remaining space) ---
                with omni.ui.ScrollingFrame(
                    height=omni.ui.Fraction(1),
                    horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                ):
                    with omni.ui.VStack(spacing=8):
                        label = omni.ui.Label("Select a prim and configure parameters", alignment=omni.ui.Alignment.LEFT_TOP)
                        
                        # Separator
                        omni.ui.Separator(height=2)
                        
                        # Basic Parameters Section
                        omni.ui.Label("Basic Parameters:", height=20)
                        
                        with omni.ui.HStack(height=25):
                            omni.ui.Label("Edge Length Ratio:", width=150)
                            edge_length_field = omni.ui.FloatDrag(min=0.01, max=1.0, step=0.01)
                            edge_length_field.model.set_value(self._edge_length_r)
                            edge_length_field.model.add_value_changed_fn(
                                lambda m: setattr(self, '_edge_length_r', m.get_value_as_float())
                            )
                        
                        with omni.ui.HStack(height=25):
                            omni.ui.Label("Epsilon Ratio:", width=150)
                            epsilon_field = omni.ui.FloatDrag(min=0.001, max=0.1, step=0.001)
                            epsilon_field.model.set_value(self._epsilon_r)
                            epsilon_field.model.add_value_changed_fn(
                                lambda m: setattr(self, '_epsilon_r', m.get_value_as_float())
                            )
                        
                        with omni.ui.HStack(height=25):
                            omni.ui.Label("Stop Quality:", width=150)
                            quality_field = omni.ui.IntDrag(min=1, max=50, step=1)
                            quality_field.model.set_value(self._stop_quality)
                            quality_field.model.add_value_changed_fn(
                                lambda m: setattr(self, '_stop_quality', m.get_value_as_int())
                            )
                        
                        with omni.ui.HStack(height=25):
                            omni.ui.Label("Max Iterations:", width=150)
                            max_its_field = omni.ui.IntDrag(min=10, max=200, step=10)
                            max_its_field.model.set_value(self._max_its)
                            max_its_field.model.add_value_changed_fn(
                                lambda m: setattr(self, '_max_its', m.get_value_as_int())
                            )
                        
                        # Separator
                        omni.ui.Separator(height=2)
                        
                        # Center Refinement Section
                        omni.ui.Label("Center Refinement (Gelpad):", height=20)
                        
                        with omni.ui.HStack(height=25):
                            omni.ui.Label("Enable Center Refinement:", width=180)
                            refinement_checkbox = omni.ui.CheckBox()
                            refinement_checkbox.model.set_value(self._enable_center_refinement)
                            refinement_checkbox.model.add_value_changed_fn(
                                lambda m: setattr(self, '_enable_center_refinement', m.get_value_as_bool())
                            )
                        
                        with omni.ui.HStack(height=25):
                            omni.ui.Label("Center Region Ratio:", width=150)
                            center_ratio_field = omni.ui.FloatDrag(min=0.1, max=0.8, step=0.05)
                            center_ratio_field.model.set_value(self._center_region_ratio)
                            center_ratio_field.model.add_value_changed_fn(
                                lambda m: setattr(self, '_center_region_ratio', m.get_value_as_float())
                            )
                            omni.ui.Label("(0.1-0.8)", width=60)
                        
                        with omni.ui.HStack(height=25):
                            omni.ui.Label("Center Edge Length:", width=150)
                            center_edge_field = omni.ui.FloatDrag(min=0.05, max=0.5, step=0.01)
                            center_edge_field.model.set_value(self._center_edge_length_r)
                            center_edge_field.model.add_value_changed_fn(
                                lambda m: setattr(self, '_center_edge_length_r', m.get_value_as_float())
                            )
                            omni.ui.Label("(smaller=denser)", width=100)
                        
                        with omni.ui.HStack(height=25):
                            omni.ui.Label("Transition Ratio:", width=150)
                            transition_field = omni.ui.FloatDrag(min=0.05, max=0.3, step=0.01)
                            transition_field.model.set_value(self._refinement_transition_ratio)
                            transition_field.model.add_value_changed_fn(
                                lambda m: setattr(self, '_refinement_transition_ratio', m.get_value_as_float())
                            )
                        
                        with omni.ui.HStack(height=25):
                            omni.ui.Label("Refine Surface:", width=150)
                            surface_combo = omni.ui.ComboBox(0, "auto", "bottom", "top", "both")
                            _surface_options = ["auto", "bottom", "top", "both"]
                            def _on_surface_changed(model, item, opts=_surface_options):
                                idx = model.get_item_value_model().get_value_as_int()
                                self._refine_surface = opts[idx]
                            surface_combo.model.add_item_changed_fn(_on_surface_changed)
                        
                        # Separator
                        omni.ui.Separator(height=2)
                        
                        # Preset Buttons
                        omni.ui.Label("Quick Presets:", height=20)
                        
                        def apply_preset_balanced():
                            edge_length_field.model.set_value(0.1)
                            refinement_checkbox.model.set_value(True)
                            center_ratio_field.model.set_value(0.35)
                            center_edge_field.model.set_value(0.035)
                            transition_field.model.set_value(0.2)
                            label.text = "Applied: Balanced preset (~3.3mm outer, ~1.2mm center)"
                        
                        def apply_preset_high():
                            edge_length_field.model.set_value(0.08)
                            refinement_checkbox.model.set_value(True)
                            center_ratio_field.model.set_value(0.4)
                            center_edge_field.model.set_value(0.025)
                            transition_field.model.set_value(0.2)
                            label.text = "Applied: High Precision preset (~2.6mm outer, ~0.8mm center)"
                        
                        def apply_preset_fast():
                            edge_length_field.model.set_value(0.15)
                            refinement_checkbox.model.set_value(True)
                            center_ratio_field.model.set_value(0.25)
                            center_edge_field.model.set_value(0.06)
                            transition_field.model.set_value(0.15)
                            label.text = "Applied: Fast Prototype preset (~5mm outer, ~2mm center)"
                        
                        def apply_preset_standard():
                            edge_length_field.model.set_value(0.1)
                            refinement_checkbox.model.set_value(False)
                            label.text = "Applied: Standard (uniform ~3.3mm, no refinement)"
                        
                        with omni.ui.HStack(spacing=5):
                            omni.ui.Button("Standard", clicked_fn=apply_preset_standard, height=25)
                            omni.ui.Button("Fast", clicked_fn=apply_preset_fast, height=25)
                            omni.ui.Button("Balanced", clicked_fn=apply_preset_balanced, height=25)
                            omni.ui.Button("High", clicked_fn=apply_preset_high, height=25)

                # --- End of scrollable parameter area ---

                # --- Fixed action buttons at bottom (always visible) ---
                omni.ui.Separator(height=2)

                def compute_tet_mesh():
                    # Create config from UI parameters
                    tet_cfg = TetMeshCfg(
                        edge_length_r=self._edge_length_r,
                        epsilon_r=self._epsilon_r,
                        stop_quality=self._stop_quality,
                        max_its=self._max_its,
                        enable_center_refinement=self._enable_center_refinement,
                        center_region_ratio=self._center_region_ratio,
                        center_edge_length_r=self._center_edge_length_r,
                        refinement_transition_ratio=self._refinement_transition_ratio,
                        refine_surface=self._refine_surface,
                        log_level=0,  # Verbose for UI feedback
                    )
                    
                    result = _generate_tet_mesh(get_selected_prim_path(), tet_cfg=tet_cfg)
                    
                    # Add config info to result
                    config_info = f"\n\nConfiguration:\n"
                    config_info += f"- Edge length: {self._edge_length_r:.3f}\n"
                    config_info += f"- Center refinement: {'Enabled' if self._enable_center_refinement else 'Disabled'}\n"
                    if self._enable_center_refinement:
                        config_info += f"- Center ratio: {self._center_region_ratio:.2f}\n"
                        config_info += f"- Center edge: {self._center_edge_length_r:.3f}\n"
                    
                    label.text = result + config_info

                def update_surf_mesh():
                    _update_surf_mesh(get_selected_prim_path())

                def create_attachment():
                    _create_attachment(get_selected_prim_paths())

                omni.ui.Button("Compute Tet Mesh", clicked_fn=compute_tet_mesh, height=30)
                omni.ui.Button("Update Surface Mesh", clicked_fn=update_surf_mesh, height=25)
                omni.ui.Button(
                    "Create Attachment \n(Select rigid body, then tet mesh, then press button)",
                    clicked_fn=create_attachment,
                    height=40,
                )

                # experimental
                def extract_primvar_st():
                    _extract_primvar_st(get_selected_prim_path())

                def set_primvar_st():
                    _set_primvar_st(get_selected_prim_path())

                omni.ui.Button("Extract primvars:st values (uv map)", clicked_fn=extract_primvar_st, height=25)
                omni.ui.Button("Set primvars:st values (uv map)", clicked_fn=set_primvar_st, height=25)

    def on_shutdown(self):
        print("[tacex_uipc] shutdown")
