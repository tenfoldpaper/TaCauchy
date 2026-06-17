#!/usr/bin/env python3
"""
Simple Contact Force Demo - Prints force information to terminal.

This script demonstrates contact force extraction from TacEx UIPC simulation
by printing force data to the terminal. No GUI visualization required.

Usage:
    python contact_force_print_demo.py --num_envs 1
"""

from __future__ import annotations

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Simple contact force demo with terminal output")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after app launcher
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.assets.rigid_object import RigidBodyPropertiesCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from isaaclab.utils.warp import XFormPrim
from omni.isaac.core.prims import VisualCuboid

from tacex import GelSightSensor
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_single_uipc import (
    FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_UIPC_CFG,
)
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg

from tacex_uipc import (
    UipcIsaacAttachments,
    UipcIsaacAttachmentsCfg,
    UipcObject,
    UipcObjectCfg,
    UipcRLEnv,
    UipcSimCfg,
)
from tacex_uipc.utils import TetMeshCfg


def print_force_summary(force_data, step: int):
    """Print force summary to terminal."""
    print(f"\n{'='*60}")
    print(f"Step {step}: Contact Force Summary")
    print(f"{'='*60}")
    
    if force_data.vertex_indices is None or len(force_data.vertex_indices) == 0:
        print("  No contact detected")
        return
    
    print(f"  Contact Points: {len(force_data.vertex_indices)}")
    
    if force_data.contact_forces is not None:
        total_force = np.sum(force_data.contact_forces, axis=0)
        force_mags = np.linalg.norm(force_data.contact_forces, axis=1)
        max_force = np.max(force_mags)
        avg_force = np.mean(force_mags)
        
        print(f"  Total Force: [{total_force[0]:.4f}, {total_force[1]:.4f}, {total_force[2]:.4f}] N")
        print(f"  Max Force Magnitude: {max_force:.4f} N")
        print(f"  Avg Force Magnitude: {avg_force:.4f} N")
        
    if force_data.normal_force_magnitudes is not None:
        print(f"  Max Normal Force: {np.max(force_data.normal_force_magnitudes):.4f} N")
        print(f"  Avg Normal Force: {np.mean(force_data.normal_force_magnitudes):.4f} N")
        
    if force_data.tangential_force_magnitudes is not None:
        print(f"  Max Tangential Force: {np.max(force_data.tangential_force_magnitudes):.4f} N")
        print(f"  Avg Tangential Force: {np.mean(force_data.tangential_force_magnitudes):.4f} N")


# =============================================================================
# Scene Configuration
# =============================================================================
@configclass
class BallRollingSceneCfg(InteractiveSceneCfg):
    """Scene configuration for ball rolling demo."""
    pass


@configclass
class BallRollingEnvCfg(DirectRLEnvCfg):
    """Environment configuration for ball rolling demo."""

    decimation = 1
    episode_length_s = 1e10
    num_actions = 7
    num_observations = 0
    num_states = 0

    gsmini = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case",
        update_period=0.0,
        history_length=0,
        debug=False,
        tactile_rgb=True,
    )

    scene: BallRollingSceneCfg = BallRollingSceneCfg(num_envs=1, env_spacing=1.5, replicate_physics=True)

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=1,
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            min_position_iteration_count=64,
            min_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            enable_stabilization=True,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=8,
        ),
    )

    uipc_sim_cfg = UipcSimCfg(
        dt=1 / 60,
        ground_height=0.0025,
        contact=UipcSimCfg.Contact(d_hat=0.0001),
    )

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),
        spawn=sim_utils.GroundPlaneCfg(size=(10, 10)),
    )

    plate = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ground_plate",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/plate.usd",
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    mesh_cfg = TetMeshCfg(stop_quality=8, max_its=100, edge_length_r=1/5)
    ball = UipcObjectCfg(
        prim_path="/World/envs/env_.*/ball",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0.12]),  # Lower to be closer to gelpad
        spawn=sim_utils.UsdFileCfg(usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/ball_wood.usd"),
        mesh_cfg=mesh_cfg,
        constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(),
    )

    robot: ArticulationCfg = FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_UIPC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    gelpad_mesh_cfg = TetMeshCfg(stop_quality=8, max_its=100, edge_length_r=1/15)
    gelpad_cfg = UipcObjectCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_gelpad",
        mesh_cfg=gelpad_mesh_cfg,
        constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(),
    )
    
    gelpad_attachment_cfg = UipcIsaacAttachmentsCfg(
        constraint_strength_ratio=100.0, 
        body_name="gelsight_mini_case", 
        debug_vis=True, 
        compute_attachment_data=True
    )


# =============================================================================
# Environment
# =============================================================================
class BallRollingEnv(UipcRLEnv):
    """Simple environment for force extraction demo."""

    cfg: BallRollingEnvCfg

    def __init__(self, cfg: BallRollingEnvCfg, render_mode: str = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self._robot = self.scene.articulations["robot"]
        self._body_idx = self._robot.find_bodies("panda_hand")[0][0]
        self._jacobi_body_idx = self._body_idx - 1

        self._ik_controller = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
            ),
            num_envs=self.num_envs,
            device=self.device,
        )

        self.ik_commands = torch.zeros(self.num_envs, 7, device=self.device)
        self._offset_pos = torch.tensor([[0.0, 0.0, 0.1]], device=self.device)
        self._offset_rot = torch.tensor([[1, 0, 0, 0]], device=self.device)
        self.step_count = 0

    def _setup_scene(self):
        self._robot = self.scene.articulations.get("robot")
        if self._robot is None:
            self._robot = self._scene.add_articulation(self.cfg.robot)
            self._scene.articulations["robot"] = self._robot

        ee_frame_cfg = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/panda_link0",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.1)),
                ),
            ],
        )
        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        self.gsmini = GelSightSensor(self.cfg.gsmini)
        self.scene.sensors["gsmini"] = self.gsmini

        RigidObject(self.cfg.plate)

        ground = self.cfg.ground
        ground.spawn.func(ground.prim_path, ground.spawn, 
                          translation=ground.init_state.pos, 
                          orientation=ground.init_state.rot)

        # Goal visual marker - positioned to make robot push into ball
        VisualCuboid(
            prim_path="/World/envs/env_0/goal",
            size=0.01,
            position=np.array([0.5, 0.0, 0.1]),  # Lower to push into ball
            color=np.array([255.0, 0.0, 0.0]),
        )

        sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)).func("/World/Light", None)

        # UIPC objects
        self._uipc_gelpad = UipcObject(self.cfg.gelpad_cfg, self.uipc_sim)
        self.ball = UipcObject(self.cfg.ball, self.uipc_sim)
        self.attachment = UipcIsaacAttachments(
            self.cfg.gelpad_attachment_cfg, self._uipc_gelpad, self.scene.articulations["robot"]
        )

    def _pre_physics_step(self, actions: torch.Tensor):
        self._ik_controller.set_command(self.ik_commands)

    def _apply_action(self):
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]

        if ee_pos_curr_b.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()
        self._robot.set_joint_position_target(joint_pos_des)
        self.step_count += 1

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def _get_rewards(self) -> torch.Tensor:
        pass

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    def _get_observations(self) -> dict:
        pass

    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :]

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w
        base_rot = self._robot.data.root_link_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
        )
        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self):
        jacobian = self.jacobian_b
        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])
        return jacobian


def run_simulator(env: BallRollingEnv):
    """Runs the simulation loop with force printing."""
    
    print("\n" + "=" * 60)
    print("Contact Force Print Demo")
    print("=" * 60)
    print("This demo prints contact force information to the terminal.")
    print("Forces are extracted from the UIPC simulation.")
    print("=" * 60 + "\n")
    
    env.reset()
    env.goal_prim_view = XFormPrim(prim_paths_expr="/World/envs/env_.*/goal", name="goals", usd=True)

    # Get force extractor
    force_extractor = env.uipc_sim.contact_force_extractor
    if force_extractor is None:
        print("[ERROR] Force extractor not available!")
        return
    print("[INFO] Force extractor initialized successfully")

    step_count = 0
    print_interval = 60  # Print every N steps
    
    try:
        while simulation_app.is_running():
            # Physics step
            env._pre_physics_step(None)
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.uipc_sim.update_render_meshes()
            env.sim.render()

            # Update goal position
            positions, orientations = env.goal_prim_view.get_world_poses()
            env.ik_commands[:, :3] = positions - env.scene.env_origins
            env.ik_commands[:, 3:] = orientations

            # Update sensors
            env.gsmini.update(dt=env.physics_dt)
            env.scene.update(dt=env.physics_dt)

            # Extract and print forces
            gelpad_forces = force_extractor.get_object_forces(env._uipc_gelpad)
            
            if step_count % print_interval == 0:
                print_force_summary(gelpad_forces, step_count)

            step_count += 1

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")


def main():
    """Main entry point."""
    env_cfg = BallRollingEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = BallRollingEnv(cfg=env_cfg)
    
    print("[INFO] Environment created successfully")
    run_simulator(env=env)
    
    simulation_app.close()


if __name__ == "__main__":
    main()
