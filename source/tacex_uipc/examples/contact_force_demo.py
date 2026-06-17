"""
Contact Force Extraction Example

This example demonstrates how to extract contact force information from the 
UIPC simulation, including normal and tangential forces at marker/vertex points.

Based on libuipc's contact system feature:
- Force from Newton's Law: f = -gradient / dt^2
- Reference: https://github.com/spiriMirror/libuipc-samples/blob/main/python/20_contact_system_feature/main.py
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Demo: Extract contact forces from UIPC simulation"
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--num_steps", type=int, default=200, help="Number of simulation steps to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import omni.usd
from pxr import UsdGeom
from uipc import Animation, Vector3, builtin, view
from uipc.constitution import ElasticModuli, SoftPositionConstraint, StableNeoHookean
from uipc.geometry import (
    GeometrySlot,
    SimplicialComplex,
    flip_inward_triangles,
    label_surface,
    label_triangle_orient,
    tetmesh,
)
from uipc.unit import MPa

import isaaclab.sim as sim_utils
from isaaclab.utils.timer import Timer

from tacex_uipc import UipcSim, UipcSimCfg
from tacex_uipc.sim import ContactForceExtractor, ContactForceData


def setup_base_scene(sim: sim_utils.SimulationContext):
    """Setup basic Isaac Sim scene."""
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    # Ground plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func(
        prim_path="/World/defaultGroundPlane",
        cfg=cfg_ground,
        translation=[0, -0.5, 0],
        orientation=[0.7071068, -0.7071068, 0, 0],
    )

    # Light
    cfg_light_dome = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_dome.func("/World/lightDome", cfg_light_dome, translation=(1, 10, 0))


def create_falling_tetrahedron(scene):
    """Create a falling tetrahedron mesh for contact force demo."""
    snh = StableNeoHookean()
    tet_object = scene.objects().create("falling_tet")
    
    # Create tetrahedron vertices (starts above ground)
    Vs = np.array([
        [0, 1.5, 0],  # top vertex
        [0, 0.5, 1],  # front vertex
        [-np.sqrt(3) / 2, 0.5, -0.5],  # back-left vertex
        [np.sqrt(3) / 2, 0.5, -0.5]    # back-right vertex
    ])
    Ts = np.array([[0, 1, 2, 3]])
    
    tet = tetmesh(Vs, Ts)
    label_surface(tet)
    label_triangle_orient(tet)
    tet = flip_inward_triangles(tet)
    
    # Apply soft material
    moduli = ElasticModuli.youngs_poisson(0.1 * MPa, 0.49)
    snh.apply_to(tet, moduli, mass_density=1e3)
    
    tet_object.geometries().create(tet)
    
    return tet_object


def print_force_data(force_data: ContactForceData, step: int):
    """Print contact force information."""
    print(f"\n{'='*60}")
    print(f"Step {step}: Contact Force Information")
    print(f"{'='*60}")
    
    if force_data.vertex_indices is None or len(force_data.vertex_indices) == 0:
        print("  No contact detected")
        return
    
    print(f"  Contact Primitive Types: {force_data.contact_primitive_types}")
    print(f"  Number of vertices in contact: {len(force_data.vertex_indices)}")
    print(f"  Time step (dt): {force_data.dt:.6f} s")
    
    print(f"\n  Vertex Details:")
    for i, (idx, force) in enumerate(zip(force_data.vertex_indices, force_data.contact_forces)):
        print(f"    Vertex {idx}:")
        print(f"      Total Force: [{force[0]:.4f}, {force[1]:.4f}, {force[2]:.4f}] N")
        
        if force_data.normal_forces is not None:
            normal = force_data.normal_forces[i]
            tangent = force_data.tangential_forces[i]
            normal_mag = force_data.normal_force_magnitudes[i]
            tangent_mag = force_data.tangential_force_magnitudes[i]
            
            print(f"      Normal Force:     [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}] N (mag: {normal_mag:.4f} N)")
            print(f"      Tangential Force: [{tangent[0]:.4f}, {tangent[1]:.4f}, {tangent[2]:.4f}] N (mag: {tangent_mag:.4f} N)")
        
        if force_data.surface_normals is not None:
            normal_vec = force_data.surface_normals[i]
            print(f"      Surface Normal:   [{normal_vec[0]:.4f}, {normal_vec[1]:.4f}, {normal_vec[2]:.4f}]")
    
    # Summary statistics
    if len(force_data.contact_forces) > 0:
        total_force = np.sum(force_data.contact_forces, axis=0)
        max_force_mag = np.max(np.linalg.norm(force_data.contact_forces, axis=1))
        print(f"\n  Summary:")
        print(f"    Total Contact Force: [{total_force[0]:.4f}, {total_force[1]:.4f}, {total_force[2]:.4f}] N")
        print(f"    Max Force Magnitude: {max_force_mag:.4f} N")
        
        if force_data.normal_force_magnitudes is not None:
            max_normal = np.max(force_data.normal_force_magnitudes)
            max_tangent = np.max(force_data.tangential_force_magnitudes)
            print(f"    Max Normal Force: {max_normal:.4f} N")
            print(f"    Max Tangential Force: {max_tangent:.4f} N")


def main():
    """Main function."""
    
    # Create Isaac Sim simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.02)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    
    # Setup visual scene
    setup_base_scene(sim)
    
    # Create UIPC simulation
    uipc_cfg = UipcSimCfg(
        dt=0.02,
        gravity=(0.0, -9.8, 0.0),
        ground_height=-0.5,
        ground_normal=(0.0, 1.0, 0.0),
        logger_level="Info",
    )
    uipc_sim = UipcSim(uipc_cfg)
    
    # Create falling tetrahedron
    tet_object = create_falling_tetrahedron(uipc_sim.scene)
    
    # Initialize rendering
    uipc_sim.init_libuipc_scene_rendering()
    
    # Setup simulation (this initializes the contact force extractor)
    uipc_sim.world.init(uipc_sim.scene)
    uipc_sim.world.retrieve()
    
    # Initialize contact force extractor manually for demo
    force_extractor = ContactForceExtractor(uipc_sim)
    force_extractor.initialize()
    
    print("\n" + "="*60)
    print("Contact Force Extraction Demo")
    print("="*60)
    print("Simulating a falling tetrahedron...")
    print("Contact forces will be extracted when the object hits the ground.")
    print("="*60)
    
    # Play simulation
    sim.reset()
    
    contact_detected = False
    
    for step in range(args_cli.num_steps):
        # Step simulation
        uipc_sim.world.advance()
        uipc_sim.world.retrieve()
        
        # Extract contact forces
        force_data = force_extractor.extract_contact_forces()
        
        # Print force data when contact is detected
        has_contact = force_data.vertex_indices is not None and len(force_data.vertex_indices) > 0
        
        if has_contact and not contact_detected:
            print("\n>>> First contact detected! <<<")
            contact_detected = True
        
        # Print force info periodically when in contact
        if has_contact and step % 10 == 0:
            print_force_data(force_data, step)
        
        # Update rendering
        uipc_sim.update_render_meshes()
        sim.step()
        
        if sim.is_stopped():
            break
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


if __name__ == "__main__":
    main()
    simulation_app.close()
